"""Build variant viewer DuckDB from probe scores.

Pipeline: load → join → bulk insert → compute aggregates → done.

The DuckDB stores flat columns. The server returns rows as-is.
The frontend derives structure from column naming conventions + heads config.
"""

import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import orjson
import polars as pl
import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from loguru import logger
from sklearn.decomposition import PCA
from umap import UMAP

from constants import EVAL_KEYS, LABEL_TO_IDX, PROBE_NAME
from loaders import load_heads, load_variants
from paths import ARTIFACTS, HEADS, VARIANTS

LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"
K_NEIGHBORS = 10

# Column prefixes that define the schema. Everything else is metadata.
META_COLS = frozenset({
    "variant_id", "chrom", "pos", "ref", "alt", "gene_name", "label",
    "clinical_significance", "stars", "consequence", "disease_name",
    "rs_id", "allele_id", "gene_id", "gene_strand",
    "vep_hgvsc", "vep_hgvsp", "vep_impact", "vep_exon",
    "vep_transcript_id", "vep_protein_id", "vep_swissprot", "vep_domains", "vep_loeuf",
    "gnomad_af", "gnomad_afr_af", "gnomad_amr_af", "gnomad_asj_af",
    "gnomad_eas_af", "gnomad_fin_af", "gnomad_nfe_af", "gnomad_sas_af",
    "gnomad_genomes_af",
})


class HeadClassification(NamedTuple):
    """Column classification computed once, used everywhere."""
    ref_cols: tuple[str, ...]
    var_cols: tuple[str, ...]
    eff_cols: tuple[str, ...]
    gt_cols: tuple[str, ...]
    disruption_heads: tuple[str, ...]
    effect_heads: tuple[str, ...]


def classify_heads(df: pl.DataFrame, cfg: dict) -> HeadClassification:
    """Classify score columns into head groups from probe config. Called once."""
    disruption_set = set(cfg["disruption_heads"])
    effect_set = set(cfg["effect_heads"])

    ref_cols = tuple(sorted(c for c in df.columns if c.startswith("ref_score_") and c[10:] in disruption_set))
    var_cols = tuple(sorted(c for c in df.columns if c.startswith("var_score_") and c[10:] in disruption_set))
    eff_cols = tuple(sorted(c for c in df.columns if c.startswith("score_") and c != "score_pathogenic" and c[6:] in effect_set))
    gt_cols = tuple(sorted(c for c in df.columns if c.startswith("gt_") and df[c].dtype in (pl.Float32, pl.Float64)))

    return HeadClassification(
        ref_cols=ref_cols, var_cols=var_cols, eff_cols=eff_cols, gt_cols=gt_cols,
        disruption_heads=tuple(c[10:] for c in ref_cols),
        effect_heads=tuple(c[6:] for c in eff_cols),
    )


# ── Data loading ────────────────────────────────────────────────────────


def load_scores(probe: str) -> pl.DataFrame:
    """Load and concatenate labeled + VUS score files."""
    scores_l = pl.read_ipc(str(LABELED / probe / "scores.feather"))
    parts = [scores_l]

    vus_path = VUS / probe / "scores.feather"
    if vus_path.exists():
        parts.append(pl.read_ipc(str(vus_path)))
    else:
        logger.warning(f"No VUS scores at {vus_path}, building labeled-only")

    return pl.concat(parts, how="diagonal") if len(parts) > 1 else parts[0]


def join_and_clean(scores: pl.DataFrame, probe: str) -> tuple[pl.DataFrame, dict]:
    """Join scores with variant metadata, rename gt columns, fill nulls."""
    variants = load_variants()
    heads = load_heads()
    cfg = json.loads((LABELED / probe / "config.json").read_text())

    df = scores.join(variants, on="variant_id", how="left")
    df = df.with_columns((pl.col("pos") + 1).alias("vcf_pos"))

    # Decode predicted amino acid swap to string
    from constants import AA_SWAP_CLASSES
    if "pred_aa_swap" in df.columns:
        df = df.with_columns(
            pl.col("pred_aa_swap").replace_strict(
                dict(enumerate(AA_SWAP_CLASSES)), default=None
            ).alias("substitution"),
        )
    if "substitution" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("substitution"))

    # Rename annotation columns that overlap with head names → gt_ prefix
    gt_renames = {h: f"gt_{h}" for h in heads if h in df.columns and h not in META_COLS}
    df = df.rename(gt_renames)

    # Round scores, fill nulls in metadata
    hc = classify_heads(df, cfg)
    float_cols = [c for c in hc.ref_cols + hc.var_cols + hc.eff_cols + hc.gt_cols
                  if df[c].dtype in (pl.Float32, pl.Float64)]

    df = df.with_columns(
        *(pl.col(c).round(4).fill_nan(None) for c in float_cols),
        pl.col("gene_name").fill_null("?"),
        pl.col("consequence").fill_null("unknown"),
        pl.col("label").fill_null("?"),
        pl.col("clinical_significance").fill_null(""),
        pl.col("stars").fill_null(0),
        pl.col("disease_name").fill_null(""),
        pl.col("score_pathogenic").fill_null(0.0).round(4),
    )

    return df, cfg


# ── Aggregates ──────────────────────────────────────────────────────────


def prebin(values: torch.Tensor, n_bins: int = 40, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    v = values[~values.isnan()]
    if v.numel() == 0:
        return [0] * n_bins
    mapped = ((v - lo) / (hi - lo) * n_bins).long()
    return torch.bincount(torch.clamp(mapped, 0, n_bins - 1), minlength=n_bins).tolist()


def _adaptive_range(vals: torch.Tensor) -> tuple[float, float]:
    valid = vals[~vals.isnan()]
    if len(valid) < 10:
        return 0.0, 1.0
    lo = valid.quantile(0.005).item()
    hi = valid.quantile(0.995).item()
    pad = max((hi - lo) * 0.05, 1e-6)
    return lo - pad, hi + pad


def _make_hist(values: torch.Tensor, ben_mask: torch.Tensor, path_mask: torch.Tensor, n_bins: int = 40) -> dict:
    """Compute a single histogram with precomputed totals."""
    lo, hi = _adaptive_range(values)
    ben = prebin(values[ben_mask], n_bins, lo, hi)
    path = prebin(values[path_mask], n_bins, lo, hi)
    return {
        "benign": ben, "pathogenic": path, "bins": n_bins,
        "range": [round(lo, 4), round(hi, 4)],
        "_bTotal": sum(ben), "_pTotal": sum(path),
    }


def compute_aggregates(df: pl.DataFrame, hc: HeadClassification) -> tuple[dict, dict]:
    """Compute distributions and head_stats from the data.

    Returns:
        (distributions, head_stats) where distributions is ready for JSON storage
        and head_stats maps head_name → {mean, std}.
    """
    ben_mask = torch.from_numpy((df["label"] == "benign").to_numpy())
    path_mask = torch.from_numpy((df["label"] == "pathogenic").to_numpy())

    # Delta matrix for disruption heads
    delta_exprs = [(pl.col(vc) - pl.col(rc)).alias(f"delta_{h}")
                   for rc, vc, h in zip(hc.ref_cols, hc.var_cols, hc.disruption_heads)]
    delta_cols = [f"delta_{h}" for h in hc.disruption_heads]
    df_d = df.with_columns(delta_exprs)

    delta_matrix = torch.from_numpy(df_d.select(delta_cols).to_numpy(allow_copy=True).T).float()
    ref_matrix = torch.from_numpy(df.select(list(hc.ref_cols)).to_numpy(allow_copy=True).T).float()
    eff_matrix = torch.from_numpy(df.select(list(hc.eff_cols)).to_numpy(allow_copy=True).T).float()

    # Head stats: mean/std of disruption deltas
    head_stats = {}
    for j, h in enumerate(hc.disruption_heads):
        valid = delta_matrix[j][~delta_matrix[j].isnan()]
        if len(valid) > 10:
            head_stats[h] = {"mean": round(valid.mean().item(), 5), "std": round(valid.std().item(), 5)}

    # Distributions
    distributions: dict = {}

    # Overall pathogenicity score distribution
    scores_t = torch.tensor(df["score_pathogenic"].to_list(), dtype=torch.float32)
    distributions["pathogenic"] = _make_hist(scores_t, ben_mask, path_mask, n_bins=80)

    # Per-head distributions
    for j, h in enumerate(hc.disruption_heads):
        distributions[h] = {
            "delta": _make_hist(delta_matrix[j], ben_mask, path_mask),
            "ref": _make_hist(ref_matrix[j], ben_mask, path_mask),
        }
    for j, h in enumerate(hc.effect_heads):
        distributions[h] = _make_hist(eff_matrix[j], ben_mask, path_mask)

    return distributions, head_stats


def build_heads_config(
    hc: HeadClassification,
    head_stats: dict[str, dict],
    heads_meta: dict,
    probe: str,
) -> dict:
    """Merge head_vocab.json + eval.json + computed stats → single heads config."""
    vocab_path = Path("head_vocab.json")
    vocab = json.loads(vocab_path.read_text()) if vocab_path.exists() else {"_meta": {}, "heads": {}}
    vocab_meta = vocab.get("_meta", {})
    vocab_heads = vocab.get("heads", {})

    # Eval metrics
    eval_metrics = {}
    eval_path = LABELED / probe / "eval.json"
    if eval_path.exists():
        for h, info in json.loads(eval_path.read_text()).items():
            for key, label in EVAL_KEYS:
                if key in info:
                    eval_metrics[h] = {"metric": label, "value": round(info[key], 2)}
                    break

    # Quality gates
    removed_cats = set(vocab_meta.get("removed_categories", []))
    removed_exact = set(vocab_meta.get("removed_exact", []))

    # Auto display name
    strip_prefixes = vocab_meta.get("display_name_strip_prefixes", [])

    def _auto_display(head: str) -> str:
        for prefix in strip_prefixes:
            if head.startswith(prefix):
                return head[len(prefix):].replace("_", " ").title()
        return head.replace("_", " ").title()

    # Group assignment fallback
    group_tokens = vocab_meta.get("group_tokens", {})

    def _auto_group(head: str) -> str:
        for group, tokens in group_tokens.items():
            if any(head.startswith(t) or f"_{t}" in head for t in tokens):
                return group
        return "Other"

    # Build merged entries
    disruption_set = set(hc.disruption_heads)
    merged = {}
    for h in sorted(set(hc.disruption_heads) | set(hc.effect_heads)):
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
        if h in removed_exact or any(h.startswith(cat) for cat in removed_cats):
            entry["quality"] = "removed"
        else:
            entry["quality"] = "pass"
        merged[h] = entry

    return {"_meta": vocab_meta, "heads": merged}


# ── Embeddings + neighbors + UMAP ──────────────────────────────────────


def _load_emb(path: Path, probe: str, d_hidden: int) -> tuple[torch.Tensor, list[str]]:
    storage = FilesystemStorage(path / probe)
    dataset = ActivationDataset(storage, "embeddings", batch_size=4096, include_provenance=True)
    embeddings, ids = [], []
    d_h2 = d_hidden ** 2
    for batch in dataset.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False):
        embeddings.append(batch.acts.flatten(1)[:, :d_h2])
        ids.extend(batch.sequence_ids)
    return torch.cat(embeddings), ids


def load_embeddings(cfg: dict, probe: str) -> tuple[torch.Tensor, list[str]]:
    emb_l, ids_l = _load_emb(LABELED, probe, cfg["d_hidden"])
    emb_v, ids_v = _load_emb(VUS, probe, cfg["d_hidden"])
    emb = torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1)
    return emb, ids_l + ids_v


def compute_neighbors(emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, k: int = 10) -> dict[str, list]:
    """GPU cosine similarity → per-variant neighbor lists."""
    n = len(emb_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_gpu = emb.to(device, non_blocking=True)

    topk_indices, topk_values = [], []
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        sim = emb_gpu[start:end] @ emb_gpu.T
        sim[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -1
        topk = sim.topk(k, dim=1)
        topk_indices.append(topk.indices.cpu())
        topk_values.append(topk.values.cpu())
    topk_i = torch.cat(topk_indices).numpy()
    topk_v = torch.cat(topk_values).numpy()
    del emb_gpu

    emb_df = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select("variant_id", pl.col("gene_name").alias("gene"),
                        "consequence", "label", pl.col("score_pathogenic").alias("score")),
              on="variant_id", how="left")
        .with_columns(pl.col("gene").fill_null("?"), pl.col("consequence").fill_null("?"),
                      pl.col("label").fill_null("?"), pl.col("score").fill_null(0.0))
    )

    edges = pl.DataFrame({
        "src_i": torch.arange(n).repeat_interleave(k).to(torch.int32).numpy(),
        "dst_i": topk_i.ravel().astype(np.int32),
        "similarity": topk_v.ravel().round(4).astype(np.float32),
    })
    nb = (edges
          .join(emb_df.select(pl.col("emb_i").alias("dst_i"), pl.col("variant_id").alias("id"),
                              "gene", "consequence", "label", "score"), on="dst_i", how="left")
          .join(emb_df.select(pl.col("emb_i").alias("src_i"), pl.col("variant_id").alias("src_vid")),
                on="src_i", how="left")
          .drop("src_i", "dst_i"))

    grouped = nb.group_by("src_vid").agg(
        pl.struct("id", "gene", "consequence", "label", "score", "similarity").alias("neighbors"))
    result = dict(zip(grouped["src_vid"].to_list(), grouped["neighbors"].to_list(), strict=True))
    logger.info(f"Neighbors: {nb.height:,} edges → {grouped.height:,} variants")
    return result


def compute_umap(emb: torch.Tensor, emb_ids: list[str], df: pl.DataFrame, n_sample: int = 30_000) -> dict:
    n = len(emb_ids)
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, min(n_sample, n), replace=False))

    pca = PCA(n_components=50, random_state=42).fit_transform(emb[idx].numpy())
    coords = UMAP(n_components=2, n_neighbors=30, min_dist=0.05, spread=10.0,
                   metric="correlation", random_state=42).fit_transform(pca)

    sub = (
        pl.DataFrame({"emb_i": range(n), "variant_id": emb_ids})
        .join(df.select("variant_id", pl.col("gene_name").alias("gene"),
                        "label", pl.col("score_pathogenic").alias("score")),
              on="variant_id", how="left")
        .with_columns(pl.col("gene").fill_null("?"), pl.col("label").fill_null("?"),
                      pl.col("score").fill_null(0.0).round(2))
        .select("variant_id", "gene", "label", "score")
    )[idx.tolist()]

    gene_list = sorted(sub["gene"].unique().to_list())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    return {
        "x": np.round(coords[:, 0], 2).tolist(),
        "y": np.round(coords[:, 1], 2).tolist(),
        "score": sub["score"].to_list(),
        "ids": sub["variant_id"].to_list(),
        "genes": [gene_to_idx[g] for g in sub["gene"].to_list()],
        "labels": [LABEL_TO_IDX.get(lab, 2) for lab in sub["label"].to_list()],
        "gene_list": gene_list,
    }


# ── Main ────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path("builds/variants.duckdb")


def main(
    db_path: Path = DEFAULT_DB_PATH,
    umap: bool = False,
    neighbors: bool = False,
    probe: str = PROBE_NAME,
    dev: int | None = None,
) -> Path:
    """Build the variant viewer DuckDB database.

    Steps:
      1. Load scores + variants → join → clean
      2. (Optional) Compute neighbors + UMAP from embeddings
      3. Bulk insert flat table into DuckDB
      4. Compute aggregates (distributions, head stats)
      5. Merge head_vocab + eval + stats → heads config
      6. Store aggregates + heads + umap in global_config
    """
    import duckdb
    from db import create_db

    t0 = time.time()
    def _t(msg: str) -> None:
        logger.info(f"[{time.time() - t0:.1f}s] {msg}")

    # Validate
    for path, desc in [
        (LABELED / probe / "scores.feather", f"Labeled scores ({probe})"),
        (VARIANTS, "Variants parquet"),
        (HEADS, "Heads JSON"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {desc} at {path}")

    # 1. Load + join
    _t("Loading...")
    scores = load_scores(probe)
    df, cfg = join_and_clean(scores, probe)
    df_full = df
    if dev:
        df = df.head(dev)
        logger.info(f"Dev mode: {dev} variants")

    hc = classify_heads(df, cfg)
    heads_meta = load_heads()
    _t(f"{df.height:,} variants, {len(hc.disruption_heads)} disruption + {len(hc.effect_heads)} effect heads")

    # 2. Optional GPU steps
    nb_map: dict[str, list] = {}
    umap_data: dict | None = None

    if neighbors or umap:
        _t("Loading embeddings...")
        emb, emb_ids = load_embeddings(cfg, probe)
        if neighbors:
            _t("Computing neighbors...")
            nb_map = compute_neighbors(emb, emb_ids, df_full, k=K_NEIGHBORS)
        if umap:
            _t("Computing UMAP...")
            umap_data = compute_umap(emb, emb_ids, df_full)
    else:
        _t("Skipping embeddings (use --neighbors or --umap)")

    # 3. Add neighbors column + bulk insert
    if nb_map:
        vids = df["variant_id"].to_list()
        df = df.with_columns(
            pl.Series("neighbors", [orjson.dumps(nb_map.get(v, [])).decode() for v in vids], dtype=pl.Utf8)
        )

    conn = create_db(db_path)
    _t(f"Inserting {df.height:,} variants ({len(df.columns)} cols)...")
    arrow_table = df.to_arrow()
    conn.execute("CREATE TABLE variants AS SELECT * FROM arrow_table")
    conn.execute("CREATE INDEX idx_variant_id ON variants(variant_id)")
    conn.execute("CREATE INDEX idx_gene ON variants(gene_name)")

    # 4. Aggregates
    _t("Computing aggregates...")
    distributions, head_stats = compute_aggregates(df, hc)

    # 5. Heads config
    heads_config = build_heads_config(hc, head_stats, heads_meta, probe)
    _t(f"Heads config: {len(heads_config['heads'])} heads")

    # 6. Store in global_config
    for key, value in [("heads", heads_config), ("distributions", distributions)]:
        conn.execute("INSERT INTO global_config VALUES (?, ?)", [key, orjson.dumps(value).decode()])
    if umap_data:
        conn.execute("INSERT INTO global_config VALUES (?, ?)", ["umap", orjson.dumps(umap_data).decode()])

    conn.close()
    _t(f"Done. {df.height:,} variants in {db_path}")
    return db_path


if __name__ == "__main__":
    main()
