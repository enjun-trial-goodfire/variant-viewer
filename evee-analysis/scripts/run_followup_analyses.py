#!/usr/bin/env python3
"""Follow-up analyses for neighbor-DepMap biology.

Three analyses, run for both DEMETER2 and Chronos:
  1) Full distributions of neighbor vs random correlations (KDE + percentiles)
  2) Delta vs k curve (k=5, 10, 20, 50) — requires recomputing kNN from embeddings
  3) Fraction of pairs above correlation thresholds

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_followup_analyses.py
"""
from __future__ import annotations

import json
import logging
import os
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
import torch
from safetensors.torch import load_file as safetensors_load

from reproducibility import enforce_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"
EMB_DIR = EVEE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool"
DEMETER2_PATH = EVEE_ROOT / "data" / "RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv"
CHRONOS_PATH = EVEE_ROOT / "data" / "CRISPR_DepMap_Public_26Q1Score_Chronos_subsetted.csv"

OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"

RANDOM_SEED = 42
K_VALUES = [5, 10, 20, 50]
CORR_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
MIN_OVERLAP = 50


# ── Helpers ───────────────────────────────────────────────────────────

def pairwise_profile_corr(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
    n = mask.sum()
    if n < MIN_OVERLAP:
        return None
    a, b = vec_a[mask], vec_b[mask]
    if a.std() < 1e-10 or b.std() < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── Analysis 1: Full distributions ────────────────────────────────────

def analysis_1_distributions(result_df: pl.DataFrame, dataset: str) -> dict:
    """KDE/histogram + percentile table for neighbor vs random correlations."""
    log.info(f"  [{dataset}] Analysis 1: Full distributions")
    nb = result_df.filter(
        (pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()
    rd = result_df.filter(
        (pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()

    percentiles = [50, 75, 90, 95]
    stats = {"dataset": dataset, "n_neighbor": len(nb), "n_random": len(rd)}
    for p in percentiles:
        stats[f"neighbor_p{p}"] = float(np.percentile(nb, p))
        stats[f"random_p{p}"] = float(np.percentile(rd, p))
        stats[f"delta_p{p}"] = stats[f"neighbor_p{p}"] - stats[f"random_p{p}"]

    log.info(f"    Percentiles:")
    for p in percentiles:
        log.info(f"      p{p}: neighbor={stats[f'neighbor_p{p}']:.4f}  "
                 f"random={stats[f'random_p{p}']:.4f}  "
                 f"delta={stats[f'delta_p{p}']:.4f}")

    # KDE histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(-0.4, 0.5, 100)
    axes[0].hist(nb, bins=bins, alpha=0.6, density=True, label=f"Neighbors (n={len(nb):,})", color="C0")
    axes[0].hist(rd, bins=bins, alpha=0.6, density=True, label=f"Random (n={len(rd):,})", color="C1")
    axes[0].axvline(np.median(nb), color="C0", linestyle="--", alpha=0.8, label=f"Neighbor median={np.median(nb):.4f}")
    axes[0].axvline(np.median(rd), color="C1", linestyle="--", alpha=0.8, label=f"Random median={np.median(rd):.4f}")
    axes[0].set_xlabel("Profile Pearson correlation")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{dataset}: Full distribution")
    axes[0].legend(fontsize=8)

    # Right tail zoom
    bins_tail = np.linspace(0.0, 0.5, 80)
    axes[1].hist(nb, bins=bins_tail, alpha=0.6, density=True, label="Neighbors", color="C0")
    axes[1].hist(rd, bins=bins_tail, alpha=0.6, density=True, label="Random", color="C1")
    for p in [75, 90, 95]:
        axes[1].axvline(np.percentile(nb, p), color="C0", linestyle=":", alpha=0.5)
        axes[1].axvline(np.percentile(rd, p), color="C1", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Profile Pearson correlation")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"{dataset}: Right tail (r > 0)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Neighbor vs Random Correlation Distributions ({dataset})", fontsize=13, y=1.02)
    fig.tight_layout()
    prefix = "chronos_" if dataset == "Chronos" else ""
    fig.savefig(FIG_DIR / f"{prefix}fig_followup_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    Saved {prefix}fig_followup_distributions.png")

    return stats


# ── Analysis 2: Delta vs k ───────────────────────────────────────────

def load_embeddings_for_variants(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load and normalize embeddings for a set of variant IDs."""
    log.info(f"  Loading embeddings for {len(variant_ids):,} variants...")

    idx_conn = sqlite3.connect(str(EMB_DIR / "index.sqlite"))
    placeholders = ",".join(["?"] * min(999, len(variant_ids)))

    vid_to_loc: dict[str, tuple[int, int]] = {}
    batch_size = 999
    for i in range(0, len(variant_ids), batch_size):
        batch = variant_ids[i:i + batch_size]
        ph = ",".join(["?"] * len(batch))
        rows = idx_conn.execute(
            f"SELECT sequence_id, chunk_id, offset FROM sequence_locations WHERE sequence_id IN ({ph})",
            batch,
        ).fetchall()
        for seq_id, chunk_id, offset in rows:
            vid_to_loc[seq_id] = (chunk_id, offset)
    idx_conn.close()

    found = [v for v in variant_ids if v in vid_to_loc]
    log.info(f"    Found {len(found):,} / {len(variant_ids):,} in embedding index")

    by_chunk: dict[int, list[tuple[str, int]]] = {}
    for vid in found:
        cid, off = vid_to_loc[vid]
        by_chunk.setdefault(cid, []).append((vid, off))

    embeddings = np.zeros((len(found), 4096), dtype=np.float32)
    vid_order = []
    idx = 0

    for cid in sorted(by_chunk.keys()):
        chunk_path = EMB_DIR / "chunks" / f"chunk_{cid:06d}" / "activations.safetensors"
        items = by_chunk[cid]

        tensors = safetensors_load(str(chunk_path))
        tensor = tensors["activations"].float().numpy()

        for vid, off in items:
            mat = tensor[off].reshape(4096)
            embeddings[idx] = mat
            vid_order.append(vid)
            idx += 1
        log.info(f"    Loaded chunk {cid}: {len(items):,} variants")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings /= norms
    log.info(f"    Embeddings loaded and L2-normalized: shape={embeddings.shape}")

    return embeddings, vid_order


def compute_knn(embeddings: np.ndarray, max_k: int) -> np.ndarray:
    """Compute top-k nearest neighbors using brute-force cosine similarity."""
    from sklearn.neighbors import NearestNeighbors
    log.info(f"  Computing {max_k}-NN for {embeddings.shape[0]:,} variants...")
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    log.info(f"    kNN complete")
    return indices[:, 1:]  # exclude self


def analysis_2_delta_vs_k(
    analysis_parquet: Path,
    gene_to_vec: dict[str, np.ndarray],
    dataset: str,
) -> list[dict]:
    """Compute delta vs k curve by recomputing kNN from embeddings."""
    log.info(f"  [{dataset}] Analysis 2: Delta vs k")

    analysis_df = pl.read_parquet(analysis_parquet)
    variant_ids = analysis_df["variant_id"].to_list()
    gene_names = analysis_df["gene_name"].to_list()

    embeddings, vid_order = load_embeddings_for_variants(variant_ids)

    vid_to_idx = {v: i for i, v in enumerate(vid_order)}
    vid_to_gene = dict(zip(variant_ids, gene_names))
    valid_vids = [v for v in variant_ids if v in vid_to_idx]
    log.info(f"    {len(valid_vids):,} variants with embeddings")

    max_k = max(K_VALUES)
    knn_indices = compute_knn(embeddings, max_k)

    rng = np.random.default_rng(RANDOM_SEED)
    valid_genes = sorted(set(gene_to_vec.keys()))
    gene_idx_map = {g: i for i, g in enumerate(valid_genes)}

    results = []
    for k in K_VALUES:
        log.info(f"    Computing correlations for k={k}...")
        nb_corrs = []
        rd_corrs = []

        for q_vid in valid_vids:
            q_idx = vid_to_idx[q_vid]
            q_gene = vid_to_gene[q_vid]
            q_vec = gene_to_vec.get(q_gene)
            if q_vec is None:
                continue

            top_k_indices = knn_indices[q_idx, :k]
            cross_gene_corrs = []
            for nb_idx in top_k_indices:
                nb_vid = vid_order[nb_idx]
                nb_gene = vid_to_gene.get(nb_vid)
                if nb_gene is None or nb_gene == q_gene:
                    continue
                nb_vec = gene_to_vec.get(nb_gene)
                if nb_vec is None:
                    continue
                c = pairwise_profile_corr(q_vec, nb_vec)
                if c is not None:
                    cross_gene_corrs.append(c)

            n_needed = len(cross_gene_corrs)
            if n_needed == 0:
                continue

            rand_corrs_for_variant = []
            for _ in range(n_needed):
                for _ in range(20):
                    rg = valid_genes[rng.integers(len(valid_genes))]
                    if rg == q_gene:
                        continue
                    rv = gene_to_vec.get(rg)
                    if rv is not None:
                        c = pairwise_profile_corr(q_vec, rv)
                        if c is not None:
                            rand_corrs_for_variant.append(c)
                            break

            nb_corrs.extend(cross_gene_corrs)
            rd_corrs.extend(rand_corrs_for_variant)

        nb_mean = np.mean(nb_corrs) if nb_corrs else 0
        rd_mean = np.mean(rd_corrs) if rd_corrs else 0
        delta = nb_mean - rd_mean

        row = {
            "dataset": dataset,
            "k": k,
            "n_neighbor_pairs": len(nb_corrs),
            "n_random_pairs": len(rd_corrs),
            "neighbor_mean": float(nb_mean),
            "random_mean": float(rd_mean),
            "delta": float(delta),
        }
        results.append(row)
        log.info(f"      k={k}: nb_mean={nb_mean:.4f}, rd_mean={rd_mean:.4f}, "
                 f"delta={delta:.4f} (n_pairs={len(nb_corrs):,})")

    return results


# ── Analysis 3: Fraction above thresholds ─────────────────────────────

def analysis_3_threshold_fractions(result_df: pl.DataFrame, dataset: str) -> list[dict]:
    """Fraction of pairs above correlation thresholds."""
    log.info(f"  [{dataset}] Analysis 3: Fraction above thresholds")

    nb = result_df.filter(
        (pl.col("pair_type") == "neighbor_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()
    rd = result_df.filter(
        (pl.col("pair_type") == "random_cross_gene") & pl.col("profile_corr").is_not_null()
    )["profile_corr"].to_numpy()

    rows = []
    for t in CORR_THRESHOLDS:
        nb_frac = float((nb > t).mean())
        rd_frac = float((rd > t).mean())
        enrichment = nb_frac / rd_frac if rd_frac > 0 else float("inf")
        row = {
            "dataset": dataset,
            "threshold": t,
            "neighbor_frac": nb_frac,
            "random_frac": rd_frac,
            "enrichment": enrichment,
        }
        rows.append(row)
        log.info(f"    r>{t:.2f}: neighbor={nb_frac:.4f}, random={rd_frac:.4f}, "
                 f"enrichment={enrichment:.2f}x")
    return rows


# ── Plotting ──────────────────────────────────────────────────────────

def plot_delta_vs_k(delta_k_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in ["DEMETER2", "Chronos"]:
        rows = [r for r in delta_k_results if r["dataset"] == dataset]
        if not rows:
            continue
        ks = [r["k"] for r in rows]
        deltas = [r["delta"] for r in rows]
        ax.plot(ks, deltas, "o-", label=dataset, markersize=8, linewidth=2)
        for k, d in zip(ks, deltas):
            ax.annotate(f"{d:.4f}", (k, d), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Δ mean profile correlation\n(neighbor − random)")
    ax.set_title("Effect size vs. number of neighbors")
    ax.set_xticks(K_VALUES)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_followup_delta_vs_k.png", dpi=150)
    plt.close(fig)
    log.info("  Saved fig_followup_delta_vs_k.png")


def plot_threshold_fractions(threshold_results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, dataset in enumerate(["DEMETER2", "Chronos"]):
        rows = [r for r in threshold_results if r["dataset"] == dataset]
        if not rows:
            continue
        ax = axes[ax_idx]
        thresholds = [r["threshold"] for r in rows]
        nb_fracs = [r["neighbor_frac"] for r in rows]
        rd_fracs = [r["random_frac"] for r in rows]
        enrichments = [r["enrichment"] for r in rows]

        x = np.arange(len(thresholds))
        w = 0.35
        bars1 = ax.bar(x - w / 2, nb_fracs, w, label="Neighbors", color="C0", edgecolor="black")
        bars2 = ax.bar(x + w / 2, rd_fracs, w, label="Random", color="C1", edgecolor="black")

        ax2 = ax.twinx()
        ax2.plot(x, enrichments, "D-", color="C2", markersize=7, linewidth=1.5, label="Enrichment")
        ax2.set_ylabel("Enrichment (fold)", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")

        ax.set_xticks(x)
        ax.set_xticklabels([f"r > {t}" for t in thresholds])
        ax.set_ylabel("Fraction of pairs")
        ax.set_title(f"{dataset}")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Fraction of gene pairs above correlation thresholds", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_followup_threshold_fractions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_followup_threshold_fractions.png")


# ── Data loading ──────────────────────────────────────────────────────

def load_demeter2_profiles() -> dict[str, np.ndarray]:
    df = pl.read_csv(str(DEMETER2_PATH))
    meta_cols = ["depmap_id", "cell_line_display_name", "lineage_1", "lineage_2",
                 "lineage_3", "lineage_4", "lineage_6"]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    mat = df.select(gene_cols).to_numpy().astype(np.float64)
    return {g: mat[:, i] for i, g in enumerate(gene_cols)}


def load_chronos_profiles() -> dict[str, np.ndarray]:
    chron_df = pl.read_csv(str(CHRONOS_PATH))
    chron_df = chron_df.rename({chron_df.columns[0]: "depmap_id"})
    dem_df = pl.read_csv(str(DEMETER2_PATH))
    dem_ids = set(dem_df["depmap_id"].to_list())
    chron_df = chron_df.filter(pl.col("depmap_id").is_in(list(dem_ids)))
    gene_cols = [c for c in chron_df.columns if c != "depmap_id"]
    mat = chron_df.select(gene_cols).to_numpy().astype(np.float64)
    return {g: mat[:, i] for i, g in enumerate(gene_cols)}


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = {
        "random_seed": RANDOM_SEED,
        "k_values": K_VALUES,
        "corr_thresholds": CORR_THRESHOLDS,
        "min_overlap": MIN_OVERLAP,
        "command": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cfg_path = OUT_DIR / "followup_run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    log.info(f"  Saved {cfg_path.name}")

    datasets = {
        "DEMETER2": {
            "pair_parquet": OUT_DIR / "neighbor_vs_random_profile_similarity.parquet",
            "analysis_parquet": OUT_DIR / "analysis_table.parquet",
            "load_profiles": load_demeter2_profiles,
        },
        "Chronos": {
            "pair_parquet": OUT_DIR / "chronos_neighbor_vs_random_profile_similarity.parquet",
            "analysis_parquet": OUT_DIR / "chronos_analysis_table.parquet",
            "load_profiles": load_chronos_profiles,
        },
    }

    all_dist_stats = []
    all_delta_k = []
    all_threshold = []

    for dataset, paths in datasets.items():
        log.info("=" * 60)
        log.info(f"Processing {dataset}")
        log.info("=" * 60)

        result_df = pl.read_parquet(paths["pair_parquet"])
        log.info(f"  Loaded {result_df.height:,} pair results")

        # Analysis 1
        dist_stats = analysis_1_distributions(result_df, dataset)
        all_dist_stats.append(dist_stats)

        # Analysis 3 (before 2, since 2 is expensive)
        threshold_rows = analysis_3_threshold_fractions(result_df, dataset)
        all_threshold.extend(threshold_rows)

        # Analysis 2: delta vs k
        log.info(f"  Loading {dataset} gene profiles for delta-vs-k...")
        gene_to_vec = paths["load_profiles"]()
        delta_k_rows = analysis_2_delta_vs_k(paths["analysis_parquet"], gene_to_vec, dataset)
        all_delta_k.extend(delta_k_rows)

    # Save tables
    log.info("=" * 60)
    log.info("Saving results and figures")
    log.info("=" * 60)

    pl.DataFrame(all_dist_stats).write_parquet(OUT_DIR / "followup_distribution_stats.parquet")
    pl.DataFrame(all_delta_k).write_parquet(OUT_DIR / "followup_delta_vs_k.parquet")
    pl.DataFrame(all_threshold).write_parquet(OUT_DIR / "followup_threshold_fractions.parquet")
    log.info("  Saved followup parquet files")

    # Plots
    plot_delta_vs_k(all_delta_k)
    plot_threshold_fractions(all_threshold)

    # Print summary tables
    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed / 60:.1f} minutes")

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Percentile Table")
    print("=" * 70)
    print(f"{'Dataset':>10} {'Pctl':>5} {'Neighbor':>10} {'Random':>10} {'Delta':>10}")
    for s in all_dist_stats:
        for p in [50, 75, 90, 95]:
            print(f"{s['dataset']:>10} p{p:>3d}  {s[f'neighbor_p{p}']:>10.4f} "
                  f"{s[f'random_p{p}']:>10.4f} {s[f'delta_p{p}']:>10.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Delta vs k")
    print("=" * 70)
    print(f"{'Dataset':>10} {'k':>4} {'NB mean':>10} {'RD mean':>10} {'Delta':>10} {'N pairs':>10}")
    for r in all_delta_k:
        print(f"{r['dataset']:>10} {r['k']:>4d} {r['neighbor_mean']:>10.4f} "
              f"{r['random_mean']:>10.4f} {r['delta']:>10.4f} {r['n_neighbor_pairs']:>10,}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Fraction above thresholds")
    print("=" * 70)
    print(f"{'Dataset':>10} {'Threshold':>10} {'NB frac':>10} {'RD frac':>10} {'Enrichment':>12}")
    for r in all_threshold:
        print(f"{r['dataset']:>10} r>{r['threshold']:.2f}     {r['neighbor_frac']:>10.4f} "
              f"{r['random_frac']:>10.4f} {r['enrichment']:>10.2f}x")

    print(f"\nElapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
