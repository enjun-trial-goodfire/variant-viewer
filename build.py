"""Build variant viewer static site from probe scores.

Writes per-variant JSONs + global.json + search.json to /tmp staging,
then optionally rsyncs to the output path. See SCHEMA.md for the data contract.

Two score types:
  - **disruption**: ref_score/var_score from the ref-view probe. Shows what changed
    between reference and variant allele. Stored as delta scalars per head.
  - **effect**: score from the diff-view probe. Predicts variant-level properties
    (clinical predictors, consequence, domain effects). Stored as scalars per head.

Timings (H200, 232K variants, probe_v11):
  Load 3s | GPU similarity 10s | Neighbors 3s | UMAP 40s | Write 70s | Total ~2min
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from loguru import logger
from rich.progress import track
from sklearn.decomposition import PCA
from umap import UMAP

from attribution import attribute
from constants import EVAL_KEYS, LABEL_TO_IDX, PROBE_NAME
from display import curated_effect_group, curated_group, display_name
from loaders import load_heads, load_variants
from paths import ARTIFACTS, HEADS, VARIANTS, sanitize_vid


def resolve_domains(raw: str | None, domain_cache: dict[str, str]) -> list[dict] | None:
    """Convert 'Pfam:PF00079;CDD:cd02056;...' to [{db, id, name?}, ...]."""
    if not raw:
        return None
    result = []
    for entry in raw.split(";"):
        parts = entry.split(":", 1)
        if len(parts) != 2:
            continue
        db, did = parts
        name = domain_cache.get(f"{db}:{did}")
        d: dict = {"db": db, "id": did}
        if name:
            d["name"] = name
        result.append(d)
    return result or None


LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
PROBE = PROBE_NAME
K_NEIGHBORS = 10
VEP_COLS = (
    "vep_hgvsc", "vep_hgvsp", "vep_impact",
    "vep_exon", "vep_transcript_id", "vep_protein_id", "vep_swissprot",
    "vep_domains", "vep_loeuf",
)
GNOMAD_COLS = (
    "gnomad_af", "gnomad_afr_af", "gnomad_amr_af", "gnomad_asj_af",
    "gnomad_eas_af", "gnomad_fin_af", "gnomad_nfe_af", "gnomad_sas_af",
)




def prebin(values: torch.Tensor, n_bins: int = 40, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    """Histogram of values into n_bins bins over [lo, hi)."""
    v = values[~values.isnan()]
    if v.numel() == 0:
        return [0] * n_bins
    mapped = ((v - lo) / (hi - lo) * n_bins).long()
    return torch.bincount(
        torch.clamp(mapped, 0, n_bins - 1), minlength=n_bins
    ).tolist()


# ── Head classification ─────────────────────────────────────────────────


def _classify_heads(
    df: pl.DataFrame, cfg: dict,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Classify score columns into disruption/effect head groups from probe config.

    Returns:
        (ref_cols, var_cols, eff_cols, gt_cols, disruption_heads, effect_heads)
    """
    disruption_set = set(cfg["disruption_heads"])
    effect_set = set(cfg["effect_heads"])

    ref_cols = tuple(sorted(c for c in df.columns if c.startswith("ref_score_") and c[10:] in disruption_set))
    var_cols = tuple(sorted(c for c in df.columns if c.startswith("var_score_") and c[10:] in disruption_set))
    eff_cols = tuple(sorted(c for c in df.columns if c.startswith("score_") and c != "score_pathogenic" and c[6:] in effect_set))
    gt_cols = tuple(sorted(c for c in df.columns if c.startswith("gt_") and df[c].dtype in (pl.Float32, pl.Float64)))

    disruption_heads = tuple(c[10:] for c in ref_cols)
    effect_heads = tuple(c[6:] for c in eff_cols)

    return ref_cols, var_cols, eff_cols, gt_cols, disruption_heads, effect_heads


# ── Data loading ────────────────────────────────────────────────────────


def load_data() -> tuple[pl.DataFrame, dict]:
    """Load variants.parquet + probe scores → unified DataFrame.

    All metadata, VEP, annotations, and submissions are pre-joined in variants.parquet.
    Only the probe scores need to be joined at build time (they're per-probe).

    Returns:
        (df, cfg) where df has all scores rounded/filled and cfg is the probe config dict.
    """
    variants = load_variants()
    heads = load_heads()

    # Join labeled scores
    scores_l = pl.read_ipc(str(LABELED / PROBE / "scores.feather"))
    labeled = variants.filter(pl.col("label").str.to_lowercase() != "vus")
    parts = [scores_l.join(labeled, on="variant_id", how="left")]

    # Join VUS scores (if available)
    vus_path = VUS / PROBE / "scores.feather"
    if vus_path.exists():
        scores_v = pl.read_ipc(str(vus_path))
        vus = variants.filter(pl.col("label").str.to_lowercase() == "vus")
        parts.append(scores_v.join(vus, on="variant_id", how="left"))
    else:
        logger.warning(f"No VUS scores at {vus_path}, building labeled-only")

    df = pl.concat(parts, how="diagonal")

    # 0-based → 1-based position for VCF display
    df = df.with_columns((pl.col("pos") + 1).alias("vcf_pos"))

    # Decode categorical head predictions to string labels
    from constants import AA_SWAP_CLASSES, CONSEQUENCE_CLASSES
    if "pred_aa_swap" in df.columns:
        df = df.with_columns(
            pl.col("pred_aa_swap").replace_strict(
                dict(enumerate(AA_SWAP_CLASSES)), default=None
            ).alias("substitution"),
        )
    if "substitution" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("substitution"))

    # Rename annotation columns to gt_* prefix for display
    meta_cols = {"variant_id", "chrom", "pos", "ref", "alt", "gene_name", "label",
                 "clinical_significance", "stars", "consequence", "disease_name",
                 "rs_id", "allele_id", "gene_id", "gene_strand"} | set(VEP_COLS) | set(GNOMAD_COLS)
    gt_renames = {h: f"gt_{h}" for h in heads if h in df.columns and h not in meta_cols}
    df = df.rename(gt_renames)

    # Probe config
    cfg = json.loads((LABELED / PROBE / "config.json").read_text())

    # Round scores and fill nulls
    ref_cols, var_cols, eff_cols, gt_cols, _, _ = _classify_heads(df, cfg)
    score_cols = ref_cols + var_cols + eff_cols + gt_cols
    float_score_cols = [c for c in score_cols if df[c].dtype in (pl.Float32, pl.Float64)]

    df = df.with_columns(
        *(pl.col(c).round(4).fill_nan(None) for c in float_score_cols),
        pl.col("gene_name").fill_null("?"),
        pl.col("consequence").fill_null("unknown"),
        pl.col("label").fill_null("?"),
        pl.col("clinical_significance").fill_null(""),
        pl.col("stars").fill_null(0),
        pl.col("disease_name").fill_null(""),
        pl.col("score_pathogenic").fill_null(0.0).round(4),
    )

    return df, cfg


# ── Embeddings ──────────────────────────────────────────────────────────


def _load_emb(path: Path, d_hidden: int) -> tuple[torch.Tensor, list[str]]:
    """Load embeddings from an ActivationDataset, extracting the diff-view slice."""
    storage = FilesystemStorage(path / PROBE)
    dataset = ActivationDataset(storage, "embeddings", batch_size=4096, include_provenance=True)
    embeddings, ids = [], []
    d_h2 = d_hidden ** 2
    for batch in dataset.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False):
        flat = batch.acts.flatten(1)
        embeddings.append(flat[:, :d_h2])
        ids.extend(batch.sequence_ids)
    return torch.cat(embeddings), ids


def _load_all_embeddings(cfg: dict) -> tuple[torch.Tensor, list[str]]:
    """Load and L2-normalize embeddings from labeled + VUS datasets."""
    emb_l, ids_l = _load_emb(LABELED, cfg["d_hidden"])
    emb_v, ids_v = _load_emb(VUS, cfg["d_hidden"])
    emb = torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1)
    return emb, ids_l + ids_v


# ── Neighbors ───────────────────────────────────────────────────────────


def compute_neighbors(
    emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, k: int = 10,
) -> dict[str, list]:
    """GPU cosine similarity + polars neighbor table.

    Args:
        emb: L2-normalized embeddings, shape (n, d).
        emb_ids: Variant IDs corresponding to rows of emb.
        df: Unified DataFrame with gene_name, consequence, label, score_pathogenic.
        k: Number of nearest neighbors per variant.

    Returns:
        Dict mapping variant_id to list of neighbor dicts.
    """
    n = len(emb_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_gpu = emb.to(device, non_blocking=True)

    topk_indices, topk_values = [], []
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        similarity = emb_gpu[start:end] @ emb_gpu.T
        similarity[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1
        topk = similarity.topk(k, dim=1)
        topk_indices.append(topk.indices.cpu())
        topk_values.append(topk.values.cpu())
    topk_i = torch.cat(topk_indices).numpy()
    topk_v = torch.cat(topk_values).numpy()
    del emb_gpu

    emb_df = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select(
            "variant_id", pl.col("gene_name").alias("gene"),
            "consequence", "label", pl.col("score_pathogenic").alias("score"),
        ), on="variant_id", how="left")
        .with_columns(
            pl.col("gene").fill_null("?"),
            pl.col("consequence").fill_null("?"),
            pl.col("label").fill_null("?"),
            pl.col("score").fill_null(0.0),
        )
    )

    edges = pl.DataFrame({
        "src_i": torch.arange(n).repeat_interleave(k).to(torch.int32).numpy(),
        "dst_i": topk_i.ravel().astype(np.int32),
        "similarity": topk_v.ravel().round(4).astype(np.float32),
    })
    nb = (edges
          .join(emb_df.select(
              pl.col("emb_i").alias("dst_i"), pl.col("variant_id").alias("id"),
              "gene", "consequence", "label", "score",
          ), on="dst_i", how="left")
          .join(emb_df.select(
              pl.col("emb_i").alias("src_i"), pl.col("variant_id").alias("src_vid"),
          ), on="src_i", how="left")
          .drop("src_i", "dst_i"))

    nb_grouped = nb.group_by("src_vid").agg(
        pl.struct("id", "gene", "consequence", "label", "score", "similarity").alias("neighbors"))
    result = dict(zip(nb_grouped["src_vid"].to_list(), nb_grouped["neighbors"].to_list(), strict=True))
    logger.info(f"Neighbors: {nb.height:,} edges -> {nb_grouped.height:,} variants")
    return result


# ── UMAP ────────────────────────────────────────────────────────────────


def compute_umap(
    emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, n_sample: int = 30_000,
) -> dict:
    """PCA + UMAP on a random subsample. Returns dict ready for global.json "umap" key.

    Args:
        emb: L2-normalized embeddings, shape (n, d).
        emb_ids: Variant IDs corresponding to rows of emb.
        df: Unified DataFrame with gene_name, label, score_pathogenic.
        n_sample: Number of points to subsample for UMAP.

    Returns:
        Dict with keys: x, y, score, ids, genes, labels, gene_list.
    """
    n = len(emb_ids)
    rng = np.random.RandomState(42)
    n_sample = min(n_sample, n)
    umap_idx = np.sort(rng.choice(n, n_sample, replace=False))

    pca = PCA(n_components=50, random_state=42).fit_transform(emb[umap_idx].numpy())
    coords = UMAP(
        n_components=2, n_neighbors=30, min_dist=0.05, spread=10.0,
        metric="correlation", random_state=42,
    ).fit_transform(pca)

    umap_sub = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select(
            "variant_id", pl.col("gene_name").alias("gene"),
            "label", pl.col("score_pathogenic").alias("score"),
        ), on="variant_id", how="left")
        .with_columns(
            pl.col("gene").fill_null("?"),
            pl.col("label").fill_null("?"),
            pl.col("score").fill_null(0.0).round(2),
        )
        .select("variant_id", "gene", "label", "score")
    )[umap_idx.tolist()]

    gene_list = sorted(umap_sub["gene"].unique().to_list())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    return {
        "x": np.round(coords[:, 0], 2).tolist(),
        "y": np.round(coords[:, 1], 2).tolist(),
        "score": umap_sub["score"].to_list(),
        "ids": umap_sub["variant_id"].to_list(),
        "genes": [gene_to_idx[g] for g in umap_sub["gene"].to_list()],
        "labels": [LABEL_TO_IDX.get(lab, 2) for lab in umap_sub["label"].to_list()],
        "gene_list": gene_list,
    }


# ── Variant serialization ──────────────────────────────────────────────


def _build_variant_dict(
    i: int,
    col_data: dict[str, list],
    disruption_heads: tuple[str, ...],
    effect_heads: tuple[str, ...],
    ref_d: dict[str, list],
    var_d: dict[str, list],
    eff_d: dict[str, list],
    gt_d: dict[str, list],
    vep_data: dict[str, list],
    gnomad_pops: dict[str, list],
    clinvar_data: dict[str, list],
    nb_map: dict[str, list],
    attr_by_vid: dict[str, list],
    heads_meta: dict,
) -> dict:
    """Build the JSON-serializable dict for a single variant at row index i."""
    vid = col_data["variant_id"][i]

    # All disruption heads with ref/var values
    disruption = {}
    for h in disruption_heads:
        r = ref_d[h][i]
        if r is None:
            continue
        va = var_d[h][i] if var_d[h][i] is not None else r
        disruption[h] = [round(r, 4), round(va, 4)]

    effect = {h: eff_d[h][i] for h in effect_heads if eff_d[h][i] is not None}
    gt = {h: v for h, col in gt_d.items() if (v := col[i]) is not None and v >= 0}

    nbs = nb_map.get(vid, [])
    n_p = sum(1 for nb in nbs if "pathogenic" in nb.get("label", ""))
    n_b = sum(1 for nb in nbs if "benign" in nb.get("label", ""))

    # Schema contract:
    #   Strings: always "" (never null). Empty string = absent.
    #   Lists/dicts: always []/{}. Empty = absent.
    #   Sparse dicts (disruption/effect/gt): missing key = 0.
    #   Genuinely nullable: loeuf, gnomad, allele_id, gene_id, n_submissions, last_evaluated.
    return {
        "id": vid,
        "gene": col_data["gene_name"][i] or "",
        "chrom": col_data["chrom"][i] or "",
        "pos": col_data["pos"][i],
        "ref": col_data["ref"][i] or "",
        "alt": col_data["alt"][i] or "",
        "vcf_pos": col_data["vcf_pos"][i],
        "gene_strand": col_data["gene_strand"][i] or "",
        "consequence": col_data["consequence"][i] or "",
        "substitution": col_data["substitution"][i] or "",
        "label": col_data["label"][i] or "",
        "significance": col_data["clinical_significance"][i] or "",
        "stars": col_data["stars"][i] or 0,
        "disease": col_data["disease_name"][i] or "",
        "score": col_data["score_pathogenic"][i],
        "rs_id": col_data["rs_id"][i] or "",
        "allele_id": col_data["allele_id"][i],         # nullable
        "gene_id": col_data["gene_id"][i] or "",
        "hgvsc": vep_data["hgvsc"][i] or "",
        "hgvsp": vep_data["hgvsp"][i] or "",
        "impact": vep_data["impact"][i] or "",
        "exon": vep_data["exon"][i] or "",
        "transcript": vep_data["transcript_id"][i] or "",
        "swissprot": vep_data["swissprot"][i] or "",
        "domains": resolve_domains(vep_data["domains"][i], heads_meta.get("_domain_cache", {})) or [],
        "loeuf": vep_data["loeuf"][i],                  # nullable
        "gnomad": vep_data["gnomad"][i],                   # nullable
        "gnomad_pop": {k: v for k, col in gnomad_pops.items() if (v := col[i]) is not None and v > 0},
        "variation_id": clinvar_data["variation_id"][i] or "",
        "cytogenetic": clinvar_data["cytogenetic"][i] or "",
        "review_status": clinvar_data["review_status"][i] or "",
        "acmg": [c for c in (clinvar_data["acmg_codes"][i] or "").split(";") if c],
        "n_submissions": clinvar_data["n_submissions"][i],  # nullable
        "submitters": [s for s in (clinvar_data["submitters"][i] or "").split(";") if s],
        "last_evaluated": clinvar_data["last_evaluated"][i],  # nullable
        "clinical_features": [f for f in (clinvar_data["clinical_features"][i] or "").split(";") if f and f != "not provided"],
        "origin": clinvar_data["origin"][i] or "",
        "disruption": disruption,
        "effect": effect,
        "gt": gt,
        "attribution": attr_by_vid.get(vid, []),
        "neighbors": nbs,
        "nP": n_p, "nB": n_b, "nV": len(nbs) - n_p - n_b,
    }


def write_variants(
    df: pl.DataFrame,
    cfg: dict,
    nb_map: dict[str, list],
    attr_by_vid: dict[str, list],
    heads_meta: dict,
    staging: Path,
) -> int:
    """Serialize per-variant JSONs to staging/variants/. Returns count written."""
    ref_cols, var_cols, eff_cols, gt_cols, disruption_heads, effect_heads = _classify_heads(df, cfg)

    staging_vdir = staging / "variants"
    staging_vdir.mkdir(exist_ok=True)

    meta_fields = (
        "variant_id", "gene_name", "chrom", "pos", "ref", "alt",
        "vcf_pos", "gene_strand",
        "consequence", "substitution", "label", "clinical_significance",
        "stars", "disease_name", "score_pathogenic", "rs_id", "allele_id", "gene_id",
    )
    vep_fields = [c for c in VEP_COLS if c != "variant_id" and c in df.columns]
    all_cols = [c for c in (*meta_fields, *vep_fields, *ref_cols, *var_cols, *eff_cols, *gt_cols) if c in df.columns]
    col_data = df.select(all_cols).to_dict(as_series=False)

    ref_d = {h: col_data[c] for c, h in zip(ref_cols, disruption_heads, strict=True)}
    var_d = {h: col_data[c] for c, h in zip(var_cols, disruption_heads, strict=True)}
    eff_d = {h: col_data[c] for c, h in zip(eff_cols, effect_heads, strict=True)}
    gt_d = {c[3:]: col_data[c] for c in gt_cols}

    # Pre-resolve optional column lookups (shared default avoids per-key allocation)
    n = df.height
    defaults = [None] * n
    vep_data = {
        k: col_data.get(f"vep_{k}", defaults)
        for k in ("hgvsc", "hgvsp", "impact", "exon",
                  "transcript_id", "swissprot", "domains", "loeuf")
    }
    vep_data["gnomad"] = col_data.get("gnomad_af", defaults)
    gnomad_pops = {
        k: col_data.get(f"gnomad_{k}_af", defaults)
        for k in ("afr", "amr", "asj", "eas", "fin", "nfe", "sas")
    }
    clinvar_data = {
        k: col_data.get(k, defaults)
        for k in ("variation_id", "cytogenetic", "review_status", "acmg_codes",
                  "submitters", "clinical_features", "n_submissions",
                  "last_evaluated", "origin")
    }

    for i in track(range(n), description="Writing variants..."):
        data = _build_variant_dict(
            i, col_data, disruption_heads, effect_heads,
            ref_d, var_d, eff_d, gt_d,
            vep_data, gnomad_pops, clinvar_data,
            nb_map, attr_by_vid, heads_meta,
        )
        vid = col_data["variant_id"][i]
        (staging_vdir / f"{sanitize_vid(vid)}.json").write_bytes(orjson.dumps(data))

    return n


# ── Global + search + static files ─────────────────────────────────────


def write_global(
    df: pl.DataFrame,
    cfg: dict,
    heads_meta: dict,
    umap_data: dict | None,
    staging: Path,
) -> None:
    """Write global.json, search.json, and copy static files to staging."""
    ref_cols, var_cols, eff_cols, _, disruption_heads, effect_heads = _classify_heads(df, cfg)
    all_head_names = disruption_heads + effect_heads

    scores_t = torch.tensor(df["score_pathogenic"].to_list(), dtype=torch.float32)
    ben_mask = torch.from_numpy((df["label"] == "benign").to_numpy())
    path_mask = torch.from_numpy((df["label"] == "pathogenic").to_numpy())

    # Disruption histograms use delta (var - ref), effect histograms use raw scores
    delta_exprs = [(pl.col(vc) - pl.col(rc)).alias(f"delta_{h}")
                   for rc, vc, h in zip(ref_cols, var_cols, disruption_heads)]
    delta_cols = [f"delta_{h}" for h in disruption_heads]
    df_hist = df.with_columns(delta_exprs)
    score_matrix = torch.from_numpy(
        df_hist.select(delta_cols + list(eff_cols)).to_numpy(allow_copy=True).T
    ).float()  # (n_heads, n_variants)

    n_disruption = len(disruption_heads)

    # Per-head delta stats for percentile normalization (before remapping)
    head_stats = {}
    for j, h in enumerate(disruption_heads):
        vals = score_matrix[j]
        valid = vals[~vals.isnan()]
        if len(valid) > 10:
            head_stats[h] = {
                "mean": round(valid.mean().item(), 5),
                "std": round(valid.std().item(), 5),
            }

    # Ref-view score histograms for disruption heads (for dual histogram display)
    ref_matrix = torch.from_numpy(
        df.select(list(ref_cols)).to_numpy(allow_copy=True).T
    ).float()  # (n_disruption, n_variants) — values in [0, 1]

    def _adaptive_range(vals: torch.Tensor) -> tuple[float, float]:
        """Compute adaptive bin range from 0.5th-99.5th percentile with 5% padding."""
        valid = vals[~vals.isnan()]
        if len(valid) < 10:
            return 0.0, 1.0
        lo = valid.quantile(0.005).item()
        hi = valid.quantile(0.995).item()
        pad = max((hi - lo) * 0.05, 1e-6)
        return lo - pad, hi + pad

    hh = {}
    for j, head_name in enumerate(all_head_names):
        if j < n_disruption:
            # Disruption head: adaptive ranges for both delta and ref
            raw_delta = score_matrix[j]
            d_lo, d_hi = _adaptive_range(raw_delta)
            raw_ref = ref_matrix[j]
            r_lo, r_hi = _adaptive_range(raw_ref)
            hh[head_name] = {
                "delta": {
                    "benign": prebin(raw_delta[ben_mask], 40, d_lo, d_hi),
                    "pathogenic": prebin(raw_delta[path_mask], 40, d_lo, d_hi),
                    "bins": 40, "range": [round(d_lo, 4), round(d_hi, 4)],
                },
                "ref": {
                    "benign": prebin(raw_ref[ben_mask], 40, r_lo, r_hi),
                    "pathogenic": prebin(raw_ref[path_mask], 40, r_lo, r_hi),
                    "bins": 40, "range": [round(r_lo, 4), round(r_hi, 4)],
                },
            }
        else:
            # Effect head: adaptive range
            raw_eff = score_matrix[j]
            e_lo, e_hi = _adaptive_range(raw_eff)
            hh[head_name] = {
                "benign": prebin(raw_eff[ben_mask], 40, e_lo, e_hi),
                "pathogenic": prebin(raw_eff[path_mask], 40, e_lo, e_hi),
                "bins": 40, "range": [round(e_lo, 4), round(e_hi, 4)],
            }

    eval_metrics = {}
    eval_path = LABELED / PROBE / "eval.json"
    if eval_path.exists():
        for h, info in json.loads(eval_path.read_text()).items():
            for key, label in EVAL_KEYS:
                if key in info:
                    eval_metrics[h] = {"metric": label, "value": round(info[key], 2)}
                    break

    decomp_path = LABELED / PROBE / "score_decomposition.json"
    if umap_data:
        (staging / "umap.json").write_bytes(orjson.dumps(umap_data))
    (staging / "global.json").write_bytes(orjson.dumps({
        "distributions": {
            "pathogenic": {
                "benign": prebin(scores_t[ben_mask], 80),
                "pathogenic": prebin(scores_t[path_mask], 80),
                "bins": 80,
            },
            **hh,
        },
        "eval": eval_metrics,
        "heads": {
            "disruption": curated_group(set(disruption_heads), quality_file=Path("head_quality.json")),
            "effect": curated_effect_group(set(effect_heads)),
        },
        "display": {h: heads_meta.get(h, {}).get("display_name", display_name(h)) for h in all_head_names},
        "head_stats": head_stats,
        "descriptions": json.loads(Path("head_descriptions.json").read_text()) if Path("head_descriptions.json").exists() else {},
        "decomposition": json.loads(decomp_path.read_text()) if decomp_path.exists() else None,
    }))

    # ── search.json ──────────────────────────────────────────────────────
    search_df = (df
        .select("variant_id", "gene_name", "label", "score_pathogenic", "consequence")
        .filter(pl.col("gene_name") != "?")
        .sort("score_pathogenic", descending=True)
        .with_columns(pl.col("gene_name").str.to_uppercase().alias("gene_upper")))

    search_agg = search_df.group_by("gene_upper").agg(
        pl.struct(
            pl.col("variant_id").alias("v"),
            pl.col("label").alias("l"),
            pl.col("score_pathogenic").alias("s"),
            pl.col("consequence").alias("c"),
        ).alias("variants"))

    search_idx = dict(zip(search_agg["gene_upper"].to_list(), search_agg["variants"].to_list(), strict=True))
    (staging / "search.json").write_bytes(orjson.dumps(search_idx))

    # ── Copy static files ────────────────────────────────────────────────
    shutil.copy2(Path(__file__).parent / "index.html", staging / "index.html")
    logos_dir = Path(__file__).parent / "logos"
    if logos_dir.exists():
        for f in logos_dir.iterdir():
            shutil.copy2(f, staging / f.name)
    interp_src = VUS / "interpretations"
    if interp_src.exists():
        interp_dst = staging / "interpretations"
        interp_dst.mkdir(parents=True, exist_ok=True)
        for f in interp_src.glob("*.json"):
            shutil.copy2(f, interp_dst / f.name)


# ── Main orchestrator ───────────────────────────────────────────────────


def main(
    output: Path | None = None,
    sync: bool = False,
    umap: bool = False,
    neighbors: bool = False,
    probe: str = PROBE,
    dev: int | None = None,
) -> Path:
    """Build the variant viewer static site.

    Args:
        output: Output directory for rsync (default: /tmp staging only).
        sync: If True, rsync staging to output directory.
        umap: Compute UMAP embedding (~40s).
        neighbors: Compute nearest neighbors (requires GPU).

    Returns:
        Path to the staging directory (or output if synced).
    """
    global PROBE
    PROBE = probe

    t0 = time.time()

    def _t(msg: str) -> None:
        logger.info(f"[{time.time() - t0:.1f}s] {msg}")

    # ── Validate inputs ──────────────────────────────────────────────────
    missing = []
    for path, desc in [
        (LABELED / PROBE / "scores.feather", f"Labeled scores ({PROBE})"),
        (VARIANTS, "Variants parquet"),
        (HEADS, "Heads JSON"),
    ]:
        if not path.exists():
            missing.append(f"  {desc}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required data files:\n" + "\n".join(missing) + "\n\nRun: uv run vv check"
        )

    # ── Load ─────────────────────────────────────────────────────────────
    _t("Loading...")
    df, cfg = load_data()
    df_full = df  # keep full df for neighbor/UMAP metadata
    if dev:
        df = df.head(dev)
        logger.info(f"Dev mode: {dev} variants")
    heads_meta = load_heads()
    # Build domain cache from Pfam heads for resolve_domains()
    domain_cache = {}
    for h, info in heads_meta.items():
        if h.startswith("pfam_"):
            pfam_id = h.replace("pfam_", "")
            name = info["display_name"].split(" (")[0]
            domain_cache[f"Pfam:{pfam_id}"] = name
    heads_meta["_domain_cache"] = domain_cache
    _t(f"{df.height:,} variants loaded, {len(heads_meta) - 1} heads")

    # ── Embeddings + neighbors + UMAP ────────────────────────────────────
    nb_map: dict[str, list] = {}
    umap_data: dict | None = None

    if neighbors or umap:
        _t("Loading embeddings...")
        emb, emb_ids = _load_all_embeddings(cfg)

        if neighbors:
            _t("GPU cosine similarity...")
            nb_map = compute_neighbors(emb, emb_ids, df_full, k=K_NEIGHBORS)

        if umap:
            _t("UMAP...")
            umap_data = compute_umap(emb, emb_ids, df_full)
    else:
        _t("Skipping embeddings (use --neighbors or --umap to enable)")

    # ── Attribution (z-scored disruption deltas) ───────────────────────

    disruption_heads = tuple(cfg.get("disruption_heads", cfg.get("ref_heads", ())))
    attr_by_vid = attribute(df, disruption_heads)
    _t(f"Attribution: {len(attr_by_vid):,} variants")

    # ── Write to staging ─────────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="variant_viewer_"))

    _t("Writing variant JSONs...")
    n = write_variants(df, cfg, nb_map, attr_by_vid, heads_meta, staging)
    _t(f"Wrote {n:,} variants to staging")

    _t("global.json + search.json...")
    write_global(df, cfg, heads_meta, umap_data, staging)

    # ── Sync or serve from staging ───────────────────────────────────────
    if sync and output:
        _t(f"Syncing to {output}...")
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{staging}/", f"{output}/"], check=True)
        shutil.rmtree(staging)
        _t(f"Done. {n:,} variants in {output}")
        return output
    else:
        _t(f"Done. {n:,} variants in {staging}")
        print(f"STAGING_DIR={staging}")
        return staging


if __name__ == "__main__":
    main()
