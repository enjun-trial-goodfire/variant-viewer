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
    "score_pathogenic", "rs_id", "allele_id", "gene_id",
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
    keep = [c for c in df.columns
            if not (c.startswith("ref_score_") and c[10:] not in included)
            and not (c.startswith("var_score_") and c[10:] not in included)
            and not (c.startswith("score_") and c != "score_pathogenic" and c[6:] not in included)]
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


def _hist(values: torch.Tensor, ben_mask: torch.Tensor, path_mask: torch.Tensor) -> dict:
    """Histogram with class-normalized fractions (each sums to 1.0)."""
    ben_raw = prebin(values[ben_mask], BINS, 0.0, 1.0)
    path_raw = prebin(values[path_mask], BINS, 0.0, 1.0)
    b_total = max(sum(ben_raw), 1)
    p_total = max(sum(path_raw), 1)
    ben = [round(c / b_total, 4) for c in ben_raw]
    path = [round(c / p_total, 4) for c in path_raw]
    return {"benign": ben, "pathogenic": path, "bins": BINS,
            "range": [0.0, 1.0], "_bTotal": b_total, "_pTotal": p_total}


def likelihood_ratio(values: torch.Tensor, ben_mask: torch.Tensor, path_mask: torch.Tensor,
                     n_bins: int = BINS) -> torch.Tensor:
    """Vectorized likelihood ratio from raw values + masks. Returns tensor in [0, 1]."""
    mapped = (values * n_bins).long().clamp(0, n_bins - 1)
    ben_bins = mapped[ben_mask]
    path_bins = mapped[path_mask]
    b = torch.bincount(ben_bins, minlength=n_bins).float()
    p = torch.bincount(path_bins, minlength=n_bins).float()
    b_rate = b / b.sum().clamp(min=1)
    p_rate = p / p.sum().clamp(min=1)
    denom = p_rate + b_rate
    lr_per_bin = torch.where(denom > 0, p_rate / denom, torch.tensor(0.5))
    lr_per_bin = torch.where((b + p) >= 5, lr_per_bin, torch.tensor(0.5))
    return lr_per_bin[mapped]


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
    gt_renames = {h: f"gt_{h}" for h in heads if h in df.columns and h not in META_COLS
                  and not h.startswith("ref_score_") and not h.startswith("var_score_") and not h.startswith("score_")}
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

    # ── Classify heads ────────────────────────────────────────────────
    cfg = json.loads((LABELED / probe / "config.json").read_text())
    disruption_set = set(cfg["disruption_heads"])
    effect_set = set(cfg["effect_heads"])

    ref_cols = sorted(c for c in df.columns if c.startswith("ref_score_") and c[10:] in disruption_set)
    var_cols = sorted(c for c in df.columns if c.startswith("var_score_") and c[10:] in disruption_set)
    eff_cols = sorted(c for c in df.columns if c.startswith("score_") and c[6:] in effect_set)
    disruption_heads = tuple(c[10:] for c in ref_cols)
    effect_heads = tuple(c[6:] for c in eff_cols)
    logger.info(f"  {len(disruption_heads)} disruption + {len(effect_heads)} effect heads")

    # ── Round scores, fill nulls ──────────────────────────────────────
    float_cols = [c for c in df.columns
                  if (c.startswith("ref_score_") or c.startswith("var_score_") or
                      c.startswith("score_") or c.startswith("gt_"))
                  and c != "score_pathogenic"
                  and df[c].dtype in (pl.Float32, pl.Float64)]

    df = df.with_columns(
        *(pl.col(c).round(4).fill_nan(None) for c in float_cols),
        pl.col("gene_name").fill_null("?"),
        pl.col("consequence").fill_null("unknown"),
        pl.col("label").fill_null("?"),
        pl.col("significance").fill_null(""),
        pl.col("stars").fill_null(0),
        pl.col("disease").fill_null(""),
        pl.col("score_pathogenic").fill_null(0.0).fill_nan(0.0).round(4),
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
    elif "vep_domains" in df.columns:
        df = df.with_columns(
            pl.col("vep_domains").map_elements(_parse_domains, return_dtype=pl.String).alias("domains")
        ).drop("vep_domains")

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
    scores_t = torch.tensor(df["score_pathogenic"].to_list(), dtype=torch.float32)
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
        valid = ~ref_matrix[j].isnan() & ~var_matrix[j].isnan()
        if valid.sum() <= 10:
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
        distributions[h] = {"data": cells, "bins": HEATMAP_BINS}

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

    # ── Per-variant z-scores and likelihood ratios ────────────────────
    logger.info("Computing per-variant z-scores and likelihood ratios...")
    new_cols = []

    for j, h in enumerate(disruption_heads):
        delta_col = delta_matrix[j]
        stats = head_stats.get(h)

        # Z-score
        if stats and stats["std"] > 0:
            z = ((delta_col - stats["mean"]) / stats["std"]).abs()
        else:
            z = torch.zeros_like(delta_col)
        new_cols.append(pl.Series(f"z_{h}", z.round(decimals=2).numpy()).cast(pl.Float32))

        # Ref and var likelihood ratios (computed from raw values + masks, not from histograms)
        ref_lr = likelihood_ratio(ref_matrix[j], ben_mask, path_mask)
        var_lr = likelihood_ratio(var_matrix[j], ben_mask, path_mask)
        new_cols.append(pl.Series(f"ref_lr_{h}", ref_lr.round(decimals=3).numpy()).cast(pl.Float32))
        new_cols.append(pl.Series(f"var_lr_{h}", var_lr.round(decimals=3).numpy()).cast(pl.Float32))

    for j, h in enumerate(effect_heads):
        lr = likelihood_ratio(eff_matrix[j], ben_mask, path_mask)
        new_cols.append(pl.Series(f"lr_{h}", lr.round(decimals=3).numpy()).cast(pl.Float32))

    df = df.hstack(new_cols)
    logger.info(f"  Added {len(new_cols)} derived columns")

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
