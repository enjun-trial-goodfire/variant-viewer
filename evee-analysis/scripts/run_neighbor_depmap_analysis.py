#!/usr/bin/env python3
"""Neighbor-DepMap biology analysis: do Evo2 embedding neighbors reflect dependency biology?

Primary question: Do cross-gene nearest neighbors in the Evo2 second-order embedding space
have more similar DEMETER2 dependency profiles than matched random cross-gene controls?

Secondary question: Is this effect stronger for protein-altering coding variants than synonymous?

Data sources:
  - builds/variants.duckdb: 184,177 variants with precomputed neighbors (JSON column)
    join key: variant_id (chr:pos:ref:alt)
    gene key: gene_name (HUGO symbol)
    consequence: consequence_display (Missense, Nonsense, Frameshift, etc.)
  - evee-analysis/data/RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv
    707 cell lines x 16,838 genes (HUGO symbols), DEMETER2 effect scores
    cell line metadata: depmap_id, cell_line_display_name, lineage_1..6

Run from variant-viewer root:
    uv run python evee-analysis/scripts/run_neighbor_depmap_analysis.py
"""
from __future__ import annotations

import json
import logging
import random
import sqlite3
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from reproducibility import (
    enforce_seeds,
    save_checksums,
    save_environment,
    save_run_config,
    save_run_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"
DEMETER2_PATH = EVEE_ROOT / "data" / "RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv"
EMB_INDEX_PATH = EVEE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool" / "index.sqlite"

OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"

TOP_K = 10
MIN_OVERLAP = 50
BOOTSTRAP_N = 500
RANDOM_SEED = 42
DATASET_NAME = "demeter2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Consequence binning ────────────────────────────────────────────────

CONSEQUENCE_BIN_MAP = {
    "Missense": "missense",
    "Nonsense": "nonsense_like",
    "Frameshift": "frameshift",
    "Splice Donor": "splice",
    "Splice Acceptor": "splice",
    "Synonymous": "synonymous",
}
VALID_BINS = frozenset(CONSEQUENCE_BIN_MAP.values())


def bin_consequence(csq_display: str | None) -> str | None:
    if csq_display is None:
        return None
    return CONSEQUENCE_BIN_MAP.get(csq_display)


# ── Stage 2: Build analysis table ─────────────────────────────────────

def load_variant_data() -> pl.DataFrame:
    log.info("Loading variants from DuckDB...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    arrow = con.execute(
        "SELECT variant_id, gene_name, consequence_display, neighbors "
        "FROM variants "
        "WHERE variant_id IS NOT NULL AND gene_name IS NOT NULL AND gene_name != ''"
    ).arrow()
    con.close()
    df = pl.from_arrow(arrow)
    log.info(f"  Loaded {df.height:,} variants")
    return df


def get_embedding_ids() -> set[str]:
    conn = sqlite3.connect(str(EMB_INDEX_PATH))
    ids = set(r[0] for r in conn.execute("SELECT sequence_id FROM sequence_locations").fetchall())
    conn.close()
    log.info(f"  Embedding IDs: {len(ids):,}")
    return ids


def build_analysis_table(df: pl.DataFrame, emb_ids: set[str], demeter2_genes: set[str]) -> pl.DataFrame:
    log.info("Building analysis table...")

    df = df.with_columns(
        pl.col("consequence_display").map_elements(bin_consequence, return_dtype=pl.Utf8).alias("consequence_bin")
    )
    before = df.height
    df = df.filter(pl.col("consequence_bin").is_not_null())
    log.info(f"  After consequence filter: {df.height:,} (dropped {before - df.height:,})")

    df = df.with_columns(
        pl.col("variant_id").is_in(list(emb_ids)).alias("has_embedding")
    )
    before = df.height
    df = df.filter(pl.col("has_embedding"))
    log.info(f"  After embedding filter: {df.height:,} (dropped {before - df.height:,})")

    df = df.with_columns(
        pl.col("gene_name").is_in(list(demeter2_genes)).alias("has_demeter2_gene")
    )
    before = df.height
    df = df.filter(pl.col("has_demeter2_gene"))
    log.info(f"  After DEMETER2 gene filter: {df.height:,} (dropped {before - df.height:,})")

    df = df.filter(
        pl.col("neighbors").is_not_null()
        & (pl.col("neighbors") != "[]")
        & (pl.col("neighbors") != "")
    )
    log.info(f"  After neighbor non-null filter: {df.height:,}")

    gene_var_counts = df.group_by("gene_name").len().rename({"len": "gene_variant_count"})
    df = df.join(gene_var_counts, on="gene_name", how="left")

    log.info(f"  Final analysis table: {df.height:,} variants, {df['gene_name'].n_unique()} genes")
    for bin_name in sorted(VALID_BINS):
        cnt = df.filter(pl.col("consequence_bin") == bin_name).height
        log.info(f"    {bin_name:25s} {cnt:>8,}")

    return df


# ── Stage 3: DEMETER2 loading ─────────────────────────────────────────

def load_demeter2() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Returns (gene_to_vec, cell_line_ids_array, lineage_1_per_cell_line)."""
    log.info("Loading DEMETER2...")
    df = pl.read_csv(str(DEMETER2_PATH))
    meta_cols = ["depmap_id", "cell_line_display_name", "lineage_1", "lineage_2",
                 "lineage_3", "lineage_4", "lineage_6"]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    cell_ids = df["depmap_id"].to_list()
    lineage_1 = df["lineage_1"].to_list()

    gene_to_vec: dict[str, np.ndarray] = {}
    mat = df.select(gene_cols).to_numpy().astype(np.float64)
    for i, g in enumerate(gene_cols):
        gene_to_vec[g] = mat[:, i]

    log.info(f"  DEMETER2: {len(gene_cols)} genes x {len(cell_ids)} cell lines")
    log.info(f"  {len(set(lineage_1))} unique lineages")
    return gene_to_vec, np.array(cell_ids), lineage_1


def pairwise_profile_corr(vec_a: np.ndarray, vec_b: np.ndarray, min_overlap: int = MIN_OVERLAP) -> tuple[float | None, int]:
    mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
    n = mask.sum()
    if n < min_overlap:
        return None, int(n)
    a, b = vec_a[mask], vec_b[mask]
    if a.std() < 1e-10 or b.std() < 1e-10:
        return None, int(n)
    r = np.corrcoef(a, b)[0, 1]
    return float(r), int(n)


def build_lineage_vectors(gene_to_vec: dict[str, np.ndarray], lineage_list: list[str]) -> dict[str, np.ndarray]:
    """Build gene -> lineage mean dependency vector."""
    unique_lineages = sorted(set(lineage_list))
    lineage_to_idx = {l: i for i, l in enumerate(unique_lineages)}
    lineage_arr = np.array([lineage_to_idx[l] for l in lineage_list])

    lineage_counts = np.bincount(lineage_arr, minlength=len(unique_lineages))
    valid_lineages = lineage_counts >= 5
    n_valid = valid_lineages.sum()

    gene_lineage_vecs: dict[str, np.ndarray] = {}
    for gene, vec in gene_to_vec.items():
        lin_means = np.full(len(unique_lineages), np.nan)
        for li in range(len(unique_lineages)):
            if not valid_lineages[li]:
                continue
            mask = (lineage_arr == li) & ~np.isnan(vec)
            if mask.sum() >= 3:
                lin_means[li] = vec[mask].mean()
        lin_means_valid = lin_means[valid_lineages]
        gene_lineage_vecs[gene] = lin_means_valid

    log.info(f"  Built lineage vectors for {len(gene_lineage_vecs)} genes ({n_valid} valid lineages)")
    return gene_lineage_vecs


# ── Stage 4: Neighbor extraction + random controls ────────────────────

def extract_neighbor_pairs(analysis_df: pl.DataFrame) -> pl.DataFrame:
    """Parse top-K neighbors from JSON. Returns long table of pairs."""
    log.info(f"Extracting top-{TOP_K} neighbor pairs...")
    rows = []
    variant_ids = analysis_df["variant_id"].to_list()
    gene_names = analysis_df["gene_name"].to_list()
    csq_bins = analysis_df["consequence_bin"].to_list()
    nb_jsons = analysis_df["neighbors"].to_list()

    for i in range(len(variant_ids)):
        src_vid = variant_ids[i]
        src_gene = gene_names[i]
        src_csq = csq_bins[i]
        raw = nb_jsons[i]
        if raw is None or not isinstance(raw, str) or raw.strip() in ("", "[]"):
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for rank, nb in enumerate(parsed[:TOP_K], start=1):
            if not isinstance(nb, dict):
                continue
            nb_vid = nb.get("id")
            nb_gene = nb.get("gene")
            nb_sim = nb.get("similarity")
            if nb_vid is None or nb_gene is None:
                continue
            is_same_gene = (nb_gene == src_gene)
            rows.append({
                "source_variant_id": src_vid,
                "source_gene": src_gene,
                "source_consequence_bin": src_csq,
                "neighbor_variant_id": str(nb_vid),
                "neighbor_gene": str(nb_gene),
                "rank": rank,
                "similarity": float(nb_sim) if nb_sim is not None else None,
                "is_same_gene": is_same_gene,
            })

    pairs = pl.DataFrame(rows)
    n_same = pairs.filter(pl.col("is_same_gene")).height
    n_cross = pairs.filter(~pl.col("is_same_gene")).height
    log.info(f"  Neighbor pairs: {pairs.height:,} total ({n_same:,} same-gene, {n_cross:,} cross-gene)")
    return pairs


def sample_matched_random_pairs(
    pairs_cross: pl.DataFrame,
    analysis_df: pl.DataFrame,
    gene_to_vec: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """For each cross-gene neighbor pair, sample one matched random cross-gene pair."""
    log.info("Sampling matched random controls...")

    valid_genes = set(gene_to_vec.keys())
    gene_by_bin: dict[str, list[str]] = {}
    for row in analysis_df.select("gene_name", "consequence_bin").unique().iter_rows(named=True):
        gene_by_bin.setdefault(row["consequence_bin"], []).append(row["gene_name"])
    for k in gene_by_bin:
        gene_by_bin[k] = sorted(set(gene_by_bin[k]) & valid_genes)

    rows = []
    src_genes = pairs_cross["source_gene"].to_list()
    src_vids = pairs_cross["source_variant_id"].to_list()
    src_csqs = pairs_cross["source_consequence_bin"].to_list()

    for i in range(len(src_genes)):
        src_gene = src_genes[i]
        csq = src_csqs[i]
        candidates = gene_by_bin.get(csq, [])
        if len(candidates) < 2:
            continue
        for _ in range(20):
            rand_gene = candidates[rng.integers(len(candidates))]
            if rand_gene != src_gene:
                rows.append({
                    "source_variant_id": src_vids[i],
                    "source_gene": src_gene,
                    "source_consequence_bin": csq,
                    "neighbor_variant_id": None,
                    "neighbor_gene": rand_gene,
                    "rank": None,
                    "similarity": None,
                    "is_same_gene": False,
                })
                break

    random_df = pl.DataFrame(rows)
    log.info(f"  Random controls: {random_df.height:,}")
    return random_df


def compute_pair_correlations(
    pairs: pl.DataFrame,
    gene_to_vec: dict[str, np.ndarray],
    gene_lineage_vecs: dict[str, np.ndarray],
    pair_type_label: str,
) -> list[dict]:
    """Compute profile_corr and lineage_corr for each pair."""
    results = []
    src_genes = pairs["source_gene"].to_list()
    nb_genes = pairs["neighbor_gene"].to_list()
    src_vids = pairs["source_variant_id"].to_list()
    nb_vids = pairs["neighbor_variant_id"].to_list()
    src_csqs = pairs["source_consequence_bin"].to_list()
    ranks = pairs["rank"].to_list()
    sims = pairs["similarity"].to_list()
    is_same = pairs["is_same_gene"].to_list()

    for i in range(len(src_genes)):
        sg, ng = src_genes[i], nb_genes[i]
        va = gene_to_vec.get(sg)
        vb = gene_to_vec.get(ng)
        if va is None or vb is None:
            continue

        corr, n_ol = pairwise_profile_corr(va, vb)

        lin_corr = None
        top_lin_match = None
        lva = gene_lineage_vecs.get(sg)
        lvb = gene_lineage_vecs.get(ng)
        if lva is not None and lvb is not None:
            lc, ln = pairwise_profile_corr(lva, lvb, min_overlap=10)
            lin_corr = lc
            mask_a = ~np.isnan(lva)
            mask_b = ~np.isnan(lvb)
            if mask_a.any() and mask_b.any():
                top_a = np.nanargmin(lva)
                top_b = np.nanargmin(lvb)
                top_lin_match = bool(top_a == top_b)

        results.append({
            "source_variant_id": src_vids[i],
            "source_gene": sg,
            "source_consequence_bin": src_csqs[i],
            "target_gene": ng,
            "target_variant_id": nb_vids[i],
            "rank": ranks[i],
            "similarity": sims[i],
            "is_same_gene": is_same[i],
            "pair_type": pair_type_label,
            "profile_corr": corr,
            "n_overlap": n_ol,
            "lineage_corr": lin_corr,
            "top_lineage_match": top_lin_match,
        })
    return results


# ── Stage 4 continued: Bootstrap ──────────────────────────────────────

def bootstrap_delta_by_gene(
    result_df: pl.DataFrame,
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Bootstrap CI for delta(mean neighbor corr - mean random corr), resampling by source gene."""
    rng = np.random.default_rng(seed)

    nb_df = result_df.filter(pl.col("pair_type") == "neighbor_cross_gene").filter(pl.col("profile_corr").is_not_null())
    rd_df = result_df.filter(pl.col("pair_type") == "random_cross_gene").filter(pl.col("profile_corr").is_not_null())

    nb_genes = nb_df["source_gene"].unique().to_list()
    rd_genes = rd_df["source_gene"].unique().to_list()
    common_genes = sorted(set(nb_genes) & set(rd_genes))
    if len(common_genes) < 10:
        return 0.0, 0.0, 0.0

    nb_gene_means = nb_df.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean_corr"))
    rd_gene_means = rd_df.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean_corr"))
    nb_dict = dict(zip(nb_gene_means["source_gene"].to_list(), nb_gene_means["mean_corr"].to_list()))
    rd_dict = dict(zip(rd_gene_means["source_gene"].to_list(), rd_gene_means["mean_corr"].to_list()))

    genes_arr = np.array(common_genes)
    nb_vals = np.array([nb_dict[g] for g in common_genes])
    rd_vals = np.array([rd_dict[g] for g in common_genes])

    deltas = np.empty(n_boot)
    n = len(genes_arr)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = nb_vals[idx].mean() - rd_vals[idx].mean()

    return float(np.percentile(deltas, 2.5)), float(np.mean(deltas)), float(np.percentile(deltas, 97.5))


# ── Stage 5: Consequence-bin analysis ─────────────────────────────────

def consequence_summary(result_df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    bin_map = {
        "protein_altering_strong": ["nonsense_like", "frameshift", "splice"],
        "missense": ["missense"],
        "synonymous": ["synonymous"],
    }
    for reported_bin, csq_list in bin_map.items():
        sub = result_df.filter(pl.col("source_consequence_bin").is_in(csq_list))
        nb = sub.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null())
        rd = sub.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null())
        nb_mean = nb["profile_corr"].mean() if nb.height > 0 else None
        rd_mean = rd["profile_corr"].mean() if rd.height > 0 else None
        ci_lo, ci_mid, ci_hi = bootstrap_delta_by_gene(sub) if nb.height > 100 and rd.height > 100 else (None, None, None)
        delta = ci_mid if ci_mid is not None else ((nb_mean - rd_mean) if nb_mean is not None and rd_mean is not None else None)
        n_src = sub.filter(pl.col("pair_type") == "neighbor_cross_gene")["source_variant_id"].n_unique() if nb.height > 0 else 0
        rows.append({
            "reported_bin": reported_bin,
            "n_source_variants": n_src,
            "n_neighbor_pairs": nb.height,
            "neighbor_mean_corr": nb_mean,
            "random_mean_corr": rd_mean,
            "delta": delta,
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
        })
    return pl.DataFrame(rows)


# ── Stage 6: Same-gene sanity + gene-collapse ─────────────────────────

def gene_collapse_robustness(result_df: pl.DataFrame, n_repeats: int = 30, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    nb_df = result_df.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null())
    rd_df = result_df.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null())

    nb_by_gene = {}
    for row in nb_df.select("source_gene", "source_variant_id", "profile_corr").iter_rows(named=True):
        nb_by_gene.setdefault(row["source_gene"], []).append((row["source_variant_id"], row["profile_corr"]))

    rd_by_gene = {}
    for row in rd_df.select("source_gene", "source_variant_id", "profile_corr").iter_rows(named=True):
        rd_by_gene.setdefault(row["source_gene"], []).append((row["source_variant_id"], row["profile_corr"]))

    common = sorted(set(nb_by_gene.keys()) & set(rd_by_gene.keys()))
    if len(common) < 10:
        return 0.0, 0.0

    deltas = []
    for _ in range(n_repeats):
        nb_corrs, rd_corrs = [], []
        for g in common:
            nb_items = nb_by_gene[g]
            rd_items = rd_by_gene[g]
            chosen_nb = nb_items[rng.integers(len(nb_items))]
            chosen_rd = rd_items[rng.integers(len(rd_items))]
            nb_corrs.append(chosen_nb[1])
            rd_corrs.append(chosen_rd[1])
        deltas.append(np.mean(nb_corrs) - np.mean(rd_corrs))

    return float(np.mean(deltas)), float(np.std(deltas))


# ── Stage 7: Figures ──────────────────────────────────────────────────

def make_figures(result_df: pl.DataFrame, csq_summary: pl.DataFrame) -> None:
    log.info("Generating figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    nb_cross = result_df.filter(
        (pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()
    rd_cross = result_df.filter(
        (pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()

    # Figure 1: Profile similarity histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.5, 1.0, 80)
    ax.hist(nb_cross, bins=bins, alpha=0.6, density=True, label=f"Cross-gene neighbors (n={len(nb_cross):,})")
    ax.hist(rd_cross, bins=bins, alpha=0.6, density=True, label=f"Random controls (n={len(rd_cross):,})")
    ax.set_xlabel("DEMETER2 profile Pearson correlation")
    ax.set_ylabel("Density")
    ax.set_title("Gene dependency profile similarity:\nEmbedding neighbors vs. random controls")
    ax.legend()
    ax.axvline(np.mean(nb_cross), color="C0", linestyle="--", alpha=0.8, label=f"Neighbor mean={np.mean(nb_cross):.4f}")
    ax.axvline(np.mean(rd_cross), color="C1", linestyle="--", alpha=0.8, label=f"Random mean={np.mean(rd_cross):.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_profile_similarity_hist.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved fig_profile_similarity_hist.png")

    # Figure 2: Consequence delta bar plot
    csq_rows = csq_summary.to_dicts()
    labels = [r["reported_bin"] for r in csq_rows]
    deltas = [r["delta"] if r["delta"] is not None else 0 for r in csq_rows]
    ci_lo_vals = [r["bootstrap_ci_lo"] for r in csq_rows]
    ci_hi_vals = [r["bootstrap_ci_hi"] for r in csq_rows]
    yerr_lo = [max(0, d - lo) if lo is not None else 0 for d, lo in zip(deltas, ci_lo_vals)]
    yerr_hi = [max(0, hi - d) if hi is not None else 0 for d, hi in zip(deltas, ci_hi_vals)]

    bar_colors = ["#e74c3c", "#3498db", "#95a5a6"][:len(labels)]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(x, deltas, yerr=[yerr_lo, yerr_hi], capsize=5, color=bar_colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Δ mean profile correlation\n(neighbor − random)")
    ax.set_title("Effect size by consequence class")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_consequence_delta.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved fig_consequence_delta.png")

    # Figure 3: Same vs cross vs random
    nb_same = result_df.filter(
        (pl.col("pair_type") == "neighbor_same_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()

    means = [np.mean(nb_same) if len(nb_same) > 0 else 0, np.mean(nb_cross), np.mean(rd_cross)]
    bar_labels = ["Same-gene\nneighbors", "Cross-gene\nneighbors", "Cross-gene\nrandom"]
    colors = ["#2ecc71", "#3498db", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(range(3), means, color=colors, edgecolor="black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Mean DEMETER2 profile correlation")
    ax.set_title("Sanity check: same-gene vs. cross-gene vs. random")
    for i, v in enumerate(means):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_same_vs_cross_vs_random.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved fig_same_vs_cross_vs_random.png")

    # Figure 4: Lineage similarity histogram
    nb_lin = result_df.filter(
        (pl.col("pair_type") == "neighbor_cross_gene") & pl.col("lineage_corr").is_not_null()
    )["lineage_corr"].to_numpy()
    rd_lin = result_df.filter(
        (pl.col("pair_type") == "random_cross_gene") & pl.col("lineage_corr").is_not_null()
    )["lineage_corr"].to_numpy()

    if len(nb_lin) > 0 and len(rd_lin) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-1.0, 1.0, 80)
        ax.hist(nb_lin, bins=bins, alpha=0.6, density=True, label=f"Cross-gene neighbors (n={len(nb_lin):,})")
        ax.hist(rd_lin, bins=bins, alpha=0.6, density=True, label=f"Random controls (n={len(rd_lin):,})")
        ax.set_xlabel("Lineage-mean DEMETER2 profile Pearson correlation")
        ax.set_ylabel("Density")
        ax.set_title("Lineage dependency concordance:\nEmbedding neighbors vs. random controls")
        ax.axvline(np.mean(nb_lin), color="C0", linestyle="--", alpha=0.8)
        ax.axvline(np.mean(rd_lin), color="C1", linestyle="--", alpha=0.8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_lineage_similarity_hist.png", dpi=150)
        plt.close(fig)
        log.info(f"  Saved fig_lineage_similarity_hist.png")


# ── Stage 8: README ───────────────────────────────────────────────────

def write_readme(
    analysis_df: pl.DataFrame,
    result_df: pl.DataFrame,
    csq_summary: pl.DataFrame,
    gc_mean: float, gc_std: float,
    lineage_stats: dict,
) -> None:
    log.info("Writing README_results.md...")

    nb_cross = result_df.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null())
    rd_cross = result_df.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null())
    nb_same = result_df.filter((pl.col("pair_type") == "neighbor_same_gene") & pl.col("profile_corr").is_not_null())

    nb_mean = nb_cross["profile_corr"].mean()
    rd_mean = rd_cross["profile_corr"].mean()
    pair_delta = nb_mean - rd_mean if nb_mean is not None and rd_mean is not None else None

    ci_lo, ci_mid, ci_hi = bootstrap_delta_by_gene(result_df)
    delta = ci_mid  # gene-level mean matches the gene-level bootstrap CI

    all_pairs = result_df.filter(pl.col("pair_type").str.starts_with("neighbor"))
    n_same = all_pairs.filter(pl.col("is_same_gene")).height
    n_cross_nb = all_pairs.filter(~pl.col("is_same_gene")).height

    lines = [
        "# Evo2 Embedding Neighbors vs. DEMETER2 Dependency Biology",
        "",
        "## 1. Question",
        "",
        "**Primary:** Do cross-gene nearest neighbors in the Evo2 second-order embedding space "
        "have more similar DEMETER2 dependency profiles than matched random cross-gene controls?",
        "",
        "**Secondary:** Is this effect stronger for protein-altering coding variants than for synonymous variants?",
        "",
        "## 2. Cohort",
        "",
        f"- **Variants retained:** {analysis_df.height:,}",
        f"- **Unique genes:** {analysis_df['gene_name'].n_unique():,}",
        f"- **Same-gene neighbor pairs:** {n_same:,}",
        f"- **Cross-gene neighbor pairs:** {n_cross_nb:,}",
        f"- **Fraction cross-gene:** {n_cross_nb / (n_same + n_cross_nb):.3f}" if (n_same + n_cross_nb) > 0 else "",
        "",
        "### Consequence breakdown",
        "",
    ]
    for bin_name in sorted(VALID_BINS):
        cnt = analysis_df.filter(pl.col("consequence_bin") == bin_name).height
        lines.append(f"- {bin_name}: {cnt:,}")

    lines += [
        "",
        "## 3. Main Result (all coding, cross-gene)",
        "",
        f"- **Neighbor mean profile correlation:** {nb_mean:.4f}" if nb_mean else "- Neighbor mean: N/A",
        f"- **Random mean profile correlation:** {rd_mean:.4f}" if rd_mean else "- Random mean: N/A",
        f"- **Delta (neighbor − random):** {delta:.4f}" if delta else "- Delta: N/A",
        f"- **95% gene-bootstrap CI:** [{ci_lo:.4f}, {ci_hi:.4f}]",
        "",
        "## 4. Lineage Concordance",
        "",
    ]
    if lineage_stats:
        lines += [
            f"- **Neighbor mean lineage correlation:** {lineage_stats.get('nb_lineage_mean', 'N/A'):.4f}",
            f"- **Random mean lineage correlation:** {lineage_stats.get('rd_lineage_mean', 'N/A'):.4f}",
            f"- **Delta:** {lineage_stats.get('lineage_delta', 'N/A'):.4f}",
            f"- **Neighbor top-lineage match rate:** {lineage_stats.get('nb_top_match', 'N/A'):.4f}",
            f"- **Random top-lineage match rate:** {lineage_stats.get('rd_top_match', 'N/A'):.4f}",
        ]

    lines += [
        "",
        "## 5. Consequence Robustness",
        "",
        "| Bin | N src variants | N pairs | Neighbor mean | Random mean | Delta | 95% CI |",
        "|-----|---------------|---------|--------------|-------------|-------|--------|",
    ]
    for row in csq_summary.iter_rows(named=True):
        ci = f"[{row['bootstrap_ci_lo']:.4f}, {row['bootstrap_ci_hi']:.4f}]" if row["bootstrap_ci_lo"] is not None else "N/A"
        nb_str = f"{row['neighbor_mean_corr']:.4f}" if row["neighbor_mean_corr"] is not None else "N/A"
        rd_str = f"{row['random_mean_corr']:.4f}" if row["random_mean_corr"] is not None else "N/A"
        d_str = f"{row['delta']:.4f}" if row["delta"] is not None else "N/A"
        lines.append(
            f"| {row['reported_bin']} | {row['n_source_variants']:,} | {row['n_neighbor_pairs']:,} | "
            f"{nb_str} | {rd_str} | {d_str} | {ci} |"
        )

    same_mean = nb_same["profile_corr"].mean() if nb_same.height > 0 else None
    lines += [
        "",
        "## 6. Sanity Checks",
        "",
        "### Same-gene comparison",
        "",
        f"- Same-gene neighbor mean corr: {same_mean:.4f}" if same_mean else "- Same-gene: N/A",
        f"- Cross-gene neighbor mean corr: {nb_mean:.4f}" if nb_mean else "",
        f"- Random cross-gene mean corr: {rd_mean:.4f}" if rd_mean else "",
        "",
        "### Gene-collapse robustness (1 variant per gene, 30 repeats)",
        "",
        f"- Mean delta: {gc_mean:.4f}",
        f"- Std delta: {gc_std:.4f}",
        "",
        "## 7. Caveats",
        "",
        "- Embeddings are variant-level but DEMETER2 labels are gene-level; "
        "multiple variants in the same gene share identical dependency profiles.",
        "- Same-gene redundancy can inflate apparent signal; the primary analysis "
        "uses **cross-gene pairs only**.",
        "- This is association, not mechanism — embedding similarity may reflect "
        "shared sequence context rather than shared function.",
        "- Second-order pooled (covariance) embeddings lose layer and position "
        "information from the original Evo2 activations.",
        "- DEMETER2 measures RNAi-based gene dependency, which is not identical "
        "to all notions of gene function or cellular survival.",
        "",
    ]

    readme_path = OUT_DIR / "README_results.md"
    readme_path.write_text("\n".join(lines))
    log.info(f"  Saved {readme_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility: seed enforcement + metadata ──
    log.info("=" * 60)
    log.info("REPRODUCIBILITY: Enforcing seeds and saving metadata")
    log.info("=" * 60)
    enforce_seeds(RANDOM_SEED)
    save_run_config(OUT_DIR, {
        "dataset": DATASET_NAME,
        "filtering_thresholds": {
            "min_overlap": MIN_OVERLAP,
            "variance_threshold": 1e-10,
        },
        "neighbor_settings": {
            "top_k": TOP_K,
            "source": "precomputed_duckdb",
        },
        "consequence_bins": dict(sorted(CONSEQUENCE_BIN_MAP.items())),
        "valid_bins": sorted(VALID_BINS),
        "random_seeds": {
            "master_seed": RANDOM_SEED,
            "numpy_seed": RANDOM_SEED,
            "python_random_seed": RANDOM_SEED,
            "PYTHONHASHSEED": RANDOM_SEED,
        },
        "bootstrap_n": BOOTSTRAP_N,
        "gene_collapse_repeats": 30,
        "data_paths": {
            "duckdb": str(DB_PATH),
            "demeter2": str(DEMETER2_PATH),
            "embeddings_index": str(EMB_INDEX_PATH),
        },
    })
    save_environment(OUT_DIR)
    save_run_manifest(OUT_DIR)

    # ── Stage 2 ──
    log.info("=" * 60)
    log.info("STAGE 2: Build analysis table")
    log.info("=" * 60)
    raw_df = load_variant_data()
    emb_ids = get_embedding_ids()
    gene_to_vec, cell_ids, lineage_list = load_demeter2()
    demeter2_genes = set(gene_to_vec.keys())
    analysis_df = build_analysis_table(raw_df, emb_ids, demeter2_genes)
    analysis_df = analysis_df.sort("variant_id")
    analysis_df.write_parquet(OUT_DIR / "analysis_table.parquet")
    log.info(f"  Saved analysis_table.parquet ({analysis_df.height:,} rows)")

    gene_list = sorted(analysis_df["gene_name"].unique().to_list())
    (OUT_DIR / "gene_list.txt").write_text("\n".join(gene_list) + "\n")
    log.info(f"  Saved gene_list.txt ({len(gene_list)} genes)")

    # ── Stage 3 (already loaded above) ──
    log.info("=" * 60)
    log.info("STAGE 3: DEMETER2 gene profile lookup ready")
    log.info("=" * 60)
    gene_lineage_vecs = build_lineage_vectors(gene_to_vec, lineage_list)

    # ── Stage 4: Extract pairs + random controls ──
    log.info("=" * 60)
    log.info("STAGE 4: Cross-gene neighbor vs matched random")
    log.info("=" * 60)
    pairs = extract_neighbor_pairs(analysis_df)
    pairs_cross = pairs.filter(~pl.col("is_same_gene"))
    pairs_same = pairs.filter(pl.col("is_same_gene"))

    rng = np.random.default_rng(RANDOM_SEED)
    random_pairs = sample_matched_random_pairs(pairs_cross, analysis_df, gene_to_vec, rng)

    log.info("Computing profile correlations for cross-gene neighbor pairs...")
    nb_cross_results = compute_pair_correlations(pairs_cross, gene_to_vec, gene_lineage_vecs, "neighbor_cross_gene")
    log.info(f"  Cross-gene neighbor correlations: {len(nb_cross_results):,}")

    log.info("Computing profile correlations for same-gene neighbor pairs...")
    nb_same_results = compute_pair_correlations(pairs_same, gene_to_vec, gene_lineage_vecs, "neighbor_same_gene")
    log.info(f"  Same-gene neighbor correlations: {len(nb_same_results):,}")

    log.info("Computing profile correlations for random controls...")
    rd_results = compute_pair_correlations(random_pairs, gene_to_vec, gene_lineage_vecs, "random_cross_gene")
    log.info(f"  Random control correlations: {len(rd_results):,}")

    all_results = nb_cross_results + nb_same_results + rd_results
    result_df = pl.DataFrame(all_results)
    result_df = result_df.sort(["pair_type", "source_variant_id", "target_gene"])
    result_df.write_parquet(OUT_DIR / "neighbor_vs_random_profile_similarity.parquet")
    log.info(f"  Saved neighbor_vs_random_profile_similarity.parquet ({result_df.height:,} rows)")

    # ── Primary result ──
    nb_c = result_df.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null())
    rd_c = result_df.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null())
    nb_mean = nb_c["profile_corr"].mean()
    rd_mean = rd_c["profile_corr"].mean()
    log.info(f"  PRIMARY RESULT: neighbor cross-gene mean corr = {nb_mean:.4f}")
    log.info(f"  PRIMARY RESULT: random cross-gene mean corr   = {rd_mean:.4f}")
    log.info(f"  PRIMARY RESULT: delta                         = {nb_mean - rd_mean:.4f}")

    # Lineage stats
    nb_lin = result_df.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("lineage_corr").is_not_null())
    rd_lin = result_df.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("lineage_corr").is_not_null())
    lineage_stats = {}
    if nb_lin.height > 0 and rd_lin.height > 0:
        lineage_stats["nb_lineage_mean"] = nb_lin["lineage_corr"].mean()
        lineage_stats["rd_lineage_mean"] = rd_lin["lineage_corr"].mean()
        lineage_stats["lineage_delta"] = lineage_stats["nb_lineage_mean"] - lineage_stats["rd_lineage_mean"]
        nb_match = result_df.filter((pl.col("pair_type") == "neighbor_cross_gene") & pl.col("top_lineage_match").is_not_null())
        rd_match = result_df.filter((pl.col("pair_type") == "random_cross_gene") & pl.col("top_lineage_match").is_not_null())
        lineage_stats["nb_top_match"] = nb_match["top_lineage_match"].mean() if nb_match.height > 0 else None
        lineage_stats["rd_top_match"] = rd_match["top_lineage_match"].mean() if rd_match.height > 0 else None
        log.info(f"  LINEAGE: neighbor mean lineage corr = {lineage_stats['nb_lineage_mean']:.4f}")
        log.info(f"  LINEAGE: random mean lineage corr   = {lineage_stats['rd_lineage_mean']:.4f}")
        log.info(f"  LINEAGE: delta                      = {lineage_stats['lineage_delta']:.4f}")

    # ── Stage 5 ──
    log.info("=" * 60)
    log.info("STAGE 5: Consequence-bin robustness")
    log.info("=" * 60)
    csq_summary = consequence_summary(result_df)
    csq_summary = csq_summary.sort("reported_bin")
    csq_summary.write_parquet(OUT_DIR / "consequence_summary.parquet")
    log.info(f"  Saved consequence_summary.parquet")
    for row in csq_summary.iter_rows(named=True):
        d_str = f"{row['delta']:.4f}" if row["delta"] is not None else "N/A"
        log.info(f"  {row['reported_bin']:30s} delta={d_str:>8s}  n_pairs={row['n_neighbor_pairs']:,}")

    # ── Stage 6 ──
    log.info("=" * 60)
    log.info("STAGE 6: Sanity checks")
    log.info("=" * 60)
    same_mean = result_df.filter(
        (pl.col("pair_type") == "neighbor_same_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].mean()
    log.info(f"  Same-gene neighbor mean corr:  {same_mean:.4f}" if same_mean else "  Same-gene: N/A")
    log.info(f"  Cross-gene neighbor mean corr: {nb_mean:.4f}")
    log.info(f"  Random cross-gene mean corr:   {rd_mean:.4f}")

    log.info("  Running gene-collapse robustness (30 repeats)...")
    gc_mean, gc_std = gene_collapse_robustness(result_df)
    log.info(f"  Gene-collapse delta: mean={gc_mean:.4f}, std={gc_std:.4f}")

    # ── Stage 7 ──
    log.info("=" * 60)
    log.info("STAGE 7: Figures")
    log.info("=" * 60)
    make_figures(result_df, csq_summary)

    # ── Stage 8 ──
    log.info("=" * 60)
    log.info("STAGE 8: README")
    log.info("=" * 60)
    write_readme(analysis_df, result_df, csq_summary, gc_mean, gc_std, lineage_stats)

    # ── Checksums ──
    save_checksums(OUT_DIR, FIG_DIR)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"DONE in {elapsed / 60:.1f} minutes")
    log.info("=" * 60)

    # Final terminal summary
    all_nb = result_df.filter(pl.col("pair_type").str.starts_with("neighbor"))
    n_same_total = all_nb.filter(pl.col("is_same_gene")).height
    n_cross_total = all_nb.filter(~pl.col("is_same_gene")).height
    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY")
    print("=" * 60)
    print(f"  Filtered variants:        {analysis_df.height:,}")
    print(f"  Unique genes:             {analysis_df['gene_name'].n_unique():,}")
    print(f"  Same-gene neighbor pairs: {n_same_total:,}")
    print(f"  Cross-gene neighbor pairs:{n_cross_total:,}")
    print(f"  Fraction cross-gene:      {n_cross_total / (n_same_total + n_cross_total):.3f}" if (n_same_total + n_cross_total) > 0 else "")
    boot_lo, boot_mid, boot_hi = bootstrap_delta_by_gene(result_df)
    print(f"  All-coding effect size:   delta = {boot_mid:.4f} (gene-level mean)")
    print(f"  Bootstrap 95% CI:         [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(f"  Pair-level delta:         {nb_mean - rd_mean:.4f} (for reference; not used with gene-level CI)")
    print("\n  Consequence-bin effect sizes:")
    for row in csq_summary.iter_rows(named=True):
        d = f"{row['delta']:.4f}" if row["delta"] is not None else "N/A"
        print(f"    {row['reported_bin']:30s} delta = {d}")
    print(f"\n  Elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
