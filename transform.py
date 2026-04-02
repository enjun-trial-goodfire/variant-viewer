"""Transform raw parquet + scores into the clean schema served to the frontend.

Single step: read → join → rename → compute aggregates → derive per-variant columns → write.

Outputs:
  - builds/clean.parquet  — every column the frontend needs, including z-scores and likelihood ratios
  - builds/statistics.json — distributions for histograms/heatmaps (stored in DuckDB global_config)
  - builds/heads.json     — merged head config (vocab + eval + stats) for the frontend

Usage:
    uv run vv transform --probe probe_v12 [--dev 1000]
"""

import json
from pathlib import Path

import polars as pl
import torch
import typer
import orjson
from loguru import logger

from constants import AA_SWAP_CLASSES, EVAL_KEYS, PROBE_NAME
from loaders import load_heads, load_variants
from paths import ARTIFACTS

LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
BINS = 20
HEATMAP_BINS = 10

# ── Column renames ────────────────────────────────────────────────────

RENAMES = {
    "disease_name": "disease",
    "clinical_significance": "significance",
    "vep_hgvsc": "hgvsc",
    "vep_hgvsp": "hgvsp",
    "vep_exon": "exon",
    "vep_loeuf": "loeuf",
    "vep_domains": "domains",
    "gnomad_af": "gnomad",
}

META_COLS = frozenset({
    "variant_id", "gene_name", "chrom", "pos", "ref", "alt", "vcf_pos", "gene_strand",
    "consequence", "substitution", "label", "significance", "stars", "disease",
    "pathogenicity", "rs_id", "allele_id", "gene_id",
    "hgvsc", "hgvsp", "vep_impact", "exon", "vep_transcript_id", "vep_protein_id",
    "domains", "loeuf", "gnomad",
    "gnomad_afr_af", "gnomad_amr_af", "gnomad_asj_af", "gnomad_eas_af",
    "gnomad_fin_af", "gnomad_nfe_af", "gnomad_sas_af", "gnomad_genomes_af",
    "variation_id", "cytogenetic", "review_status", "acmg",
    "n_submissions", "submitters", "last_evaluated", "clinical_features", "origin",
    "gnomad_af_c", "gnomad_exomes_c", "gnomad_genomes_c",
    "aa_swap", "pred_aa_swap",
})

EXCLUDED_DOMAIN_DBS = frozenset({"PDB-ENSP_mappings", "AFDB-ENSP_mappings", "ENSP_mappings", "Gene3D"})
EXCLUDE_CLINICAL = frozenset({"not provided", "not specified", ""})
QUALITY_PATH = Path("head_quality.json")

CONSEQUENCE_DISPLAY = {
    "missense_variant": "Missense", "synonymous_variant": "Synonymous",
    "frameshift_variant": "Frameshift", "nonsense": "Nonsense",
    "stop_gained": "Stop Gained", "splice_donor_variant": "Splice Donor",
    "splice_acceptor_variant": "Splice Acceptor", "splice_region_variant": "Splice Region",
    "intron_variant": "Intronic", "non-coding_transcript_variant": "Non-coding",
    "start_lost": "Start Lost", "stop_lost": "Stop Lost",
    "inframe_deletion": "In-frame Deletion", "inframe_insertion": "In-frame Insertion",
    "inframe_indel": "In-frame Indel", "5_prime_UTR_variant": "5' UTR",
    "3_prime_UTR_variant": "3' UTR", "genic_downstream_transcript_variant": "Downstream",
    "genic_upstream_transcript_variant": "Upstream", "initiator_codon_variant": "Initiator Codon",
    "no_sequence_alteration": "No Change",
}

LABEL_DISPLAY = {
    "pathogenic": "Pathogenic", "likely_pathogenic": "Likely Pathogenic",
    "benign": "Benign", "likely_benign": "Likely Benign", "VUS": "VUS",
}


# ── Helpers ───────────────────────────────────────────────────────────

def filter_heads(df: pl.DataFrame, included: set[str]) -> pl.DataFrame:
    """Drop head columns not in the included set."""
    head_prefixes = ("ref_score_", "var_score_", "ref_", "var_", "dist_", "spread_", "eff_")
    keep = []
    for c in df.columns:
        drop = False
        for prefix in head_prefixes:
            if c.startswith(prefix):
                head = c[len(prefix):]
                if head and head not in included:
                    drop = True
                break
        if not drop:
            keep.append(c)
    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.info(f"  Filtered heads: dropped {dropped} columns, kept {len(included)} heads")
    return df.select(keep)


def _semi_to_json(s: str | None, exclude: frozenset[str] = frozenset()) -> str:
    if not s:
        return "[]"
    return json.dumps([p.strip() for p in s.split(";") if p.strip() and p.strip().lower() not in exclude])


def _parse_domains(raw: str | None) -> str:
    if not raw:
        return "[]"
    result = []
    for entry in raw.split(","):
        parts = entry.strip().split(":", 1)
        if len(parts) == 2 and parts[0] not in EXCLUDED_DOMAIN_DBS:
            result.append({"db": parts[0], "id": parts[1]})
    return json.dumps(result)


def _decode_aa_swap(idx: int | None) -> str | None:
    if idx is None or idx < 0 or idx >= len(AA_SWAP_CLASSES):
        return None
    return AA_SWAP_CLASSES[idx]


def _hgvs_short(full: str | None) -> str | None:
    if not full or ":" not in full:
        return full
    return full.split(":")[-1]


def prebin(values: torch.Tensor, n_bins: int, lo: float, hi: float) -> list[int]:
    v = values[~values.isnan()]
    if v.numel() == 0:
        return [0] * n_bins
    mapped = ((v - lo) / (hi - lo) * n_bins).long()
    return torch.bincount(torch.clamp(mapped, 0, n_bins - 1), minlength=n_bins).tolist()


def _hist(values: torch.Tensor, ben_mask: torch.Tensor, path_mask: torch.Tensor,
          lo: float = 0.0, hi: float = 1.0) -> dict:
    """Histogram with class-normalized fractions (each sums to 1.0)."""
    ben_raw = prebin(values[ben_mask], BINS, lo, hi)
    path_raw = prebin(values[path_mask], BINS, lo, hi)
    b_total = max(sum(ben_raw), 1)
    p_total = max(sum(path_raw), 1)
    ben = [round(c / b_total, 4) for c in ben_raw]
    path = [round(c / p_total, 4) for c in path_raw]
    return {"benign": ben, "pathogenic": path, "bins": BINS,
            "range": [lo, hi], "_bTotal": b_total, "_pTotal": p_total}



# ── Heads config ──────────────────────────────────────────────────────

def build_heads_config(
    disruption_heads: tuple[str, ...], effect_heads: tuple[str, ...],
    head_stats: dict, heads_meta: dict, probe: str,
) -> dict:
    vocab_path = Path("heads.json")
    vocab = json.loads(vocab_path.read_text())
    vocab_meta = vocab.get("_meta", {})
    vocab_heads = vocab.get("heads", {})

    eval_metrics = {}
    eval_path = LABELED / probe / "eval.json"
    if eval_path.exists():
        for h, info in json.loads(eval_path.read_text()).items():
            for key, label in EVAL_KEYS:
                if key in info:
                    eval_metrics[h] = {"metric": label, "value": round(info[key], 2)}
                    break

    strip_prefixes = vocab_meta.get("display_name_strip_prefixes", [])
    group_tokens = vocab_meta.get("group_tokens", {})

    def _auto_display(h):
        for prefix in strip_prefixes:
            if h.startswith(prefix):
                return h[len(prefix):].replace("_", " ").title()
        return h.replace("_", " ").title()

    def _auto_group(h):
        for group, tokens in group_tokens.items():
            if any(h.startswith(t) for t in tokens):
                return group
        return "Other"

    disruption_set = set(disruption_heads)
    merged = {}
    for h in sorted(disruption_set | set(effect_heads)):
        entry = dict(vocab_heads.get(h, {}))
        entry["category"] = "disruption" if h in disruption_set else "effect"
        entry.setdefault("group", _auto_group(h))
        if "display" not in entry:
            meta = heads_meta.get(h, {})
            entry["display"] = meta.get("display_name", _auto_display(h)) if isinstance(meta, dict) else _auto_display(h)
        if h in eval_metrics:
            entry["eval"] = eval_metrics[h]
        if h in head_stats:
            entry["mean"] = head_stats[h]["mean"]
            entry["std"] = head_stats[h]["std"]
        entry.setdefault("quality", "pass")
        merged[h] = entry

    return {"_meta": vocab_meta, "heads": merged}


# ── Main ──────────────────────────────────────────────────────────────

def main(
    probe: str = typer.Option(PROBE_NAME, help="Probe version"),
    token_probe: str | None = typer.Option(None, help="Token probe version (replaces disruption scores)"),
    output: Path = typer.Option(Path("builds/clean.parquet"), help="Output parquet path"),
    dev: int | None = typer.Option(None, help="Dev mode: limit to N variants"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    logger.info("Loading variants")
    df_variants = load_variants()

    logger.info(f"Loading scores ({probe})")
    df_scores = pl.read_ipc(str(LABELED / probe / "scores.feather"))
    vus_path = VUS / probe / "scores.feather"
    if vus_path.exists():
        df_vus = pl.read_ipc(str(vus_path))
        logger.info(f"  + {df_vus.height:,} VUS rows")
        df_scores = pl.concat([df_scores, df_vus], how="diagonal")

    # ── Token probe: replace disruption scores with per-position data ──
    if token_probe:
        token_path = LABELED / token_probe / "token_scores.feather"
        logger.info(f"Loading token scores ({token_probe})")
        df_token = pl.read_ipc(str(token_path))
        logger.info(f"  {df_token.height:,} rows, {df_token.width} columns")

        # Token probe outputs ref_*, var_*, dist_*, spread_* (already canonical).
        # Drop overlapping sequence probe disruption columns (ref_score_*, var_score_*).
        token_heads = {c[4:] for c in df_token.columns if c.startswith("ref_")}
        seq_drop = [c for c in df_scores.columns
                    if (c.startswith("ref_score_") and c[10:] in token_heads)
                    or (c.startswith("var_score_") and c[10:] in token_heads)]
        if seq_drop:
            df_scores = df_scores.drop(seq_drop)
            logger.info(f"  Replaced {len(seq_drop)} sequence disruption columns with token scores")

        # Drop delta_ columns (redundant — recomputed from ref/var)
        df_token = df_token.drop([c for c in df_token.columns if c.startswith("delta_")])

        # Inner join: only keep variants that have token scores.
        # Variants without token data (48K VUS + ~186 edge cases) get no disruption columns.
        n_before = df_scores.height
        df_scores = df_scores.join(df_token, on="variant_id", how="inner")
        n_dropped = n_before - df_scores.height
        if n_dropped:
            logger.info(f"  Dropped {n_dropped:,} variants without token scores")

    # ── Rename to canonical convention BEFORE filtering ──────────────
    cfg = json.loads((LABELED / probe / "config.json").read_text())
    assert "effect_heads" in cfg, "config.json missing 'effect_heads' key"
    effect_set = set(cfg["effect_heads"])

    # Duplicate score_pathogenic → pathogenicity (metadata) before renaming score_ → eff_.
    if "score_pathogenic" in df_scores.columns:
        df_scores = df_scores.with_columns(pl.col("score_pathogenic").alias("pathogenicity"))

    # score_{h} → eff_{h} for effect heads
    early_renames = {c: f"eff_{c[6:]}" for c in df_scores.columns
                     if c.startswith("score_") and c[6:] in effect_set}
    if early_renames:
        df_scores = df_scores.rename(early_renames)

    # ── Filter to quality-passing heads + always keep predictor heads ──
    quality = json.loads(QUALITY_PATH.read_text())
    vocab = json.loads(Path("heads.json").read_text())
    predictor_heads = {h for h, info in vocab.get("heads", {}).items() if info.get("predictor")}
    included = set(quality["included"]) | predictor_heads
    df_scores = filter_heads(df_scores, included)

    # ── Join ──────────────────────────────────────────────────────────
    df = df_scores.join(df_variants, on="variant_id", how="left")
    logger.info(f"  {df.height:,} rows, {df.width} columns")

    if dev:
        df = df.head(dev)

    # ── gt_ prefix for annotation columns that clash with head names ──
    heads = load_heads()
    # Prefix annotation columns that share names with heads to avoid collisions.
    # By this point, head columns already have prefixes (ref_, var_, eff_).
    gt_renames = {h: f"gt_{h}" for h in heads if h in df.columns and h not in META_COLS}
    df = df.rename(gt_renames)

    # ── Rename to frontend field names ────────────────────────────────
    renames = {old: new for old, new in RENAMES.items() if old in df.columns}
    collisions = [new for new in renames.values() if new in df.columns and new not in renames]
    if collisions:
        df = df.drop(collisions)
    df = df.rename(renames)

    # ── Decode aa_swap ────────────────────────────────────────────────
    df = df.with_columns(
        pl.col("pred_aa_swap").map_elements(_decode_aa_swap, return_dtype=pl.String).alias("substitution")
    ).drop("pred_aa_swap")

    # ── Classify heads ──────────────────────────────────────────────────
    # Convention: ref_{h}, var_{h} = disruption; eff_{h} = effect; gt_{h} = ground truth
    # Effects already renamed to eff_ before filtering. Rename remaining
    # sequence probe disruption columns: ref_score_{h} → ref_{h}, var_score_{h} → var_{h}.
    assert "disruption_heads" in cfg, "config.json missing 'disruption_heads' key"
    disruption_set = set(cfg["disruption_heads"])

    col_renames = {}
    for c in df.columns:
        if c.startswith("ref_score_"):
            col_renames[c] = f"ref_{c[10:]}"
        elif c.startswith("var_score_"):
            col_renames[c] = f"var_{c[10:]}"
    if col_renames:
        df = df.rename(col_renames)

    ref_cols = sorted(c for c in df.columns if c.startswith("ref_") and c[4:] in disruption_set)
    var_cols = sorted(c for c in df.columns if c.startswith("var_") and c[4:] in disruption_set)
    eff_cols = sorted(c for c in df.columns if c.startswith("eff_") and c[4:] in effect_set)
    disruption_heads = tuple(c[4:] for c in ref_cols)
    effect_heads = tuple(c[4:] for c in eff_cols)
    logger.info(f"  {len(disruption_heads)} disruption + {len(effect_heads)} effect heads")

    # ── Round scores, fill nulls ──────────────────────────────────────
    float_cols = [c for c in df.columns
                  if (c.startswith("ref_") or c.startswith("var_") or
                      c.startswith("eff_") or c.startswith("gt_"))
                  and df[c].dtype in (pl.Float32, pl.Float64)]

    # Validate computed scores — crash on bad data, don't fill silently
    bad = df["pathogenicity"].is_null() | df["pathogenicity"].is_nan()
    assert bad.sum() == 0, f"pathogenicity has {bad.sum()} null/NaN values — fix upstream"

    df = df.with_columns(
        *(pl.col(c).round(4) for c in float_cols),
        pl.col("pathogenicity").round(4),
        (pl.col("pos") + 1).alias("vcf_pos"),
        pl.col("consequence").replace_strict(CONSEQUENCE_DISPLAY, default=None).fill_null(
            pl.col("consequence").str.replace_all("_", " ").str.to_titlecase()
        ).alias("consequence_display"),
        pl.col("label").replace_strict(LABEL_DISPLAY, default=pl.col("label").str.replace_all("_", " ")).alias("label_display"),
        pl.col("loeuf").map_elements(
            lambda v: "highly constrained" if v is not None and v < 0.35 else
                      "constrained" if v is not None and v < 0.6 else
                      "tolerant" if v is not None and v < 1.0 else
                      "unconstrained" if v is not None else None,
            return_dtype=pl.String
        ).alias("loeuf_label"),
        pl.col("gnomad").map_elements(
            lambda af: "Rare (supports PM2)" if af is not None and af > 0 and af < 0.0001 else
                       "Common (BA1)" if af is not None and af > 0.05 else
                       "Polymorphism (BS1)" if af is not None and af > 0.01 else None,
            return_dtype=pl.String
        ).alias("gnomad_label"),
        pl.col("gnomad").map_elements(
            lambda af: "Not observed" if af is None or af == 0 else
                       f"1 in {round(1/af):,}" if af < 0.01 else f"{af*100:.1f}%",
            return_dtype=pl.String
        ).alias("gnomad_display"),
        pl.col("hgvsc").map_elements(_hgvs_short, return_dtype=pl.String).alias("hgvsc_short"),
        pl.col("hgvsp").map_elements(_hgvs_short, return_dtype=pl.String).alias("hgvsp_short"),
    )

    # ── Semicolons → JSON arrays ──────────────────────────────────────
    df = df.with_columns(
        pl.col("acmg_codes").map_elements(_semi_to_json, return_dtype=pl.String).alias("acmg"),
        pl.col("clinical_features").map_elements(
            lambda s: _semi_to_json(s, EXCLUDE_CLINICAL), return_dtype=pl.String
        ).alias("clinical_features"),
        pl.col("submitters").map_elements(_semi_to_json, return_dtype=pl.String).alias("submitters"),
    ).drop("acmg_codes")

    # ── Domains → JSON ────────────────────────────────────────────────
    if "domains" in df.columns:
        df = df.with_columns(
            pl.col("domains").map_elements(_parse_domains, return_dtype=pl.String).alias("domains")
        )

    # ── Compute population statistics (torch) ─────────────────────────
    logger.info("Computing population statistics...")
    ben_mask = torch.from_numpy((df["label"] == "benign").to_numpy())
    path_mask = torch.from_numpy((df["label"] == "pathogenic").to_numpy())

    ref_matrix = torch.from_numpy(df.select(ref_cols).to_numpy(allow_copy=True).T).float()
    var_matrix = torch.from_numpy(df.select(var_cols).to_numpy(allow_copy=True).T).float()
    eff_matrix = torch.from_numpy(df.select(eff_cols).to_numpy(allow_copy=True).T).float()
    delta_matrix = var_matrix - ref_matrix

    # Head stats (mean/std of deltas)
    head_stats: dict[str, dict] = {}
    for j, h in enumerate(disruption_heads):
        valid = delta_matrix[j][~delta_matrix[j].isnan()]
        if len(valid) > 10:
            head_stats[h] = {"mean": round(valid.mean().item(), 5), "std": round(valid.std().item(), 5)}

    # Distributions (for histograms + heatmaps)
    scores_t = df["pathogenicity"].to_torch().float()
    distributions: dict = {"pathogenic": _hist(scores_t, ben_mask, path_mask)}

    path_bool = path_mask.float()
    n_cells = HEATMAP_BINS ** 2

    # Per-disruption-head: heatmap + ref distribution (for likelihood ratio)
    ref_dists: dict[str, dict] = {}
    for j, h in enumerate(disruption_heads):
        # 1D ref distribution (for likelihood ratio computation)
        ref_dist = _hist(ref_matrix[j], ben_mask, path_mask)
        ref_dists[h] = ref_dist

        # 2D heatmap
        delta_dist = _hist(delta_matrix[j], ben_mask, path_mask, lo=-1.0, hi=1.0)
        valid = ~ref_matrix[j].isnan() & ~var_matrix[j].isnan()
        if valid.sum() <= 10:
            distributions[h] = {"ref": ref_dist, "delta": delta_dist}
            continue
        rv, vv, pv = ref_matrix[j][valid], var_matrix[j][valid], path_bool[valid]
        rb = (rv * HEATMAP_BINS).long().clamp(0, HEATMAP_BINS - 1)
        vb = (vv * HEATMAP_BINS).long().clamp(0, HEATMAP_BINS - 1)
        idx = rb * HEATMAP_BINS + vb
        total = torch.bincount(idx, minlength=n_cells)
        path_c = torch.bincount(idx, weights=pv, minlength=n_cells)
        cells = []
        for ci in total.nonzero(as_tuple=True)[0]:
            c = int(total[ci])
            bi, bj = divmod(ci.item(), HEATMAP_BINS)
            cells.append([bi, bj, round(float(path_c[ci]) / c * 100, 1), c])
        distributions[h] = {
            "data": cells, "bins": HEATMAP_BINS,
            "ref": ref_dist,
            "delta": _hist(delta_matrix[j], ben_mask, path_mask, lo=-1.0, hi=1.0),
        }

    # Per-effect-head distribution
    eff_dists: dict[str, dict] = {}
    for j, h in enumerate(effect_heads):
        d = _hist(eff_matrix[j], ben_mask, path_mask)
        eff_dists[h] = d
        distributions[h] = d

    # 1D histogram for gt_ predictor heads that are disruption-type
    # (e.g., phylop_100way, gerp_c — shown as single-value predictors but have no effect distribution)
    vocab_heads = json.loads(Path("heads.json").read_text()).get("heads", {})
    for h, info in vocab_heads.items():
        if not info.get("predictor"):
            continue
        gt_col = f"gt_{h}"
        if gt_col not in df.columns:
            continue
        vals = torch.from_numpy(df[gt_col].to_numpy(allow_copy=True)).float()
        if (~vals.isnan()).sum() > 10:
            gt_hist = _hist(vals, ben_mask, path_mask)
            if h in distributions:
                distributions[h]["gt_hist"] = gt_hist
            else:
                distributions[h] = gt_hist

    # ── Per-variant z-scores ────────────────────────────────────────────
    logger.info("Computing per-variant z-scores...")
    new_cols = []

    for j, h in enumerate(disruption_heads):
        delta_col = delta_matrix[j]
        stats = head_stats.get(h)

        if stats and stats["std"] > 0:
            z = ((delta_col - stats["mean"]) / stats["std"]).abs()
        else:
            z = torch.zeros_like(delta_col)
        new_cols.append(pl.Series(f"z_{h}", z.round(decimals=2).numpy()).cast(pl.Float32))

    df = df.hstack(new_cols)
    logger.info(f"  Added {len(new_cols)} derived columns")

    # ── Validate: computed columns must not have null/NaN ──────────────
    # Only gt_ columns (database annotations) are allowed to be null.
    computed_prefixes = ("ref_", "var_", "z_", "eff_")
    for c in df.columns:
        if not any(c.startswith(p) for p in computed_prefixes):
            continue
        if df[c].dtype not in (pl.Float32, pl.Float64):
            continue
        n_null = df[c].is_null().sum()
        n_nan = df[c].is_nan().sum()
        assert n_null == 0 and n_nan == 0, (
            f"Column {c} has {n_null} nulls and {n_nan} NaNs — fix upstream data, don't fill silently"
        )

    # ── Write outputs ─────────────────────────────────────────────────
    logger.info(f"Writing {output} ({df.height:,} rows, {df.width} columns)")
    df.write_parquet(output)

    stats_path = output.parent / "statistics.json"
    logger.info(f"Writing {stats_path}")
    stats_path.write_bytes(orjson.dumps(distributions))

    heads_config = build_heads_config(disruption_heads, effect_heads, head_stats, load_heads(), probe)
    heads_path = output.parent / "heads.json"
    logger.info(f"Writing {heads_path}")
    heads_path.write_bytes(orjson.dumps(heads_config))

    logger.info("Done")


if __name__ == "__main__":
    typer.run(main)
