#!/usr/bin/env python3
"""Deep comparative analysis of CORUM vs Chronos latent feature interactions.

Identifies which matrix entries (i,j) encode protein-complex structure (CORUM)
vs cellular dependency/survival (Chronos), and where they overlap or diverge.

Stages X1–X10 as specified.

Prereqs (run first):
    run_corum_interpretability.py   → gene_level_matrices.npz, corum_entry_enrichment.parquet, etc.
    run_chronos_entry_analysis.py   → chronos_entry_correlations.parquet, chronos_entry_weights.parquet

Usage:
    uv run python evee-analysis/scripts/run_corum_vs_chronos_features.py
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"

RANDOM_SEED = 42
N_PERM = 10_000
TOP_NS = [50, 100, 200]
PERCENTILE_THRESHOLD = 90  # top 10%


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════════════
# X1 — Load and verify normalized feature space
# ══════════════════════════════════════════════════════════════════════

def stage_x1() -> tuple[np.ndarray, list[str]]:
    log.info("X1: Loading z-scored gene-level matrices...")
    data = np.load(str(OUT_DIR / "gene_level_matrices.npz"), allow_pickle=True)
    gene_names = list(data["gene_names"])
    zscored = data["zscored_flat"]  # (N_genes, 4096)
    # Verify z-scoring
    mu = zscored.mean(axis=0)
    sd = zscored.std(axis=0)
    log.info(f"  {len(gene_names):,} genes × {zscored.shape[1]} features")
    log.info(f"  mean of means: {mu.mean():.6f} (expect ~0)")
    log.info(f"  mean of stds:  {sd.mean():.6f} (expect ~1)")
    return zscored, gene_names


# ══════════════════════════════════════════════════════════════════════
# X2 — Define top feature sets
# ══════════════════════════════════════════════════════════════════════

def stage_x2() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-(i,j) importance scores for CORUM and Chronos.

    Returns:
        corum_score: (4096,) — global CORUM importance per entry
        chronos_corr_score: (4096,) — |Pearson r| with mean dependency
        chronos_ridge_score: (4096,) — |ridge weight| for mean dependency
    """
    log.info("X2: Building per-entry importance scores...")

    # --- CORUM global score ---
    # Two complementary measures:
    # (a) Recurrence: how many complexes have this entry in top-20 significant
    # (b) Mean |effect_size| across complexes where FDR < 0.05
    enrichment = pl.read_parquet(OUT_DIR / "corum_entry_enrichment.parquet")

    # (a) Recurrence count from significant entries
    sig = enrichment.filter(pl.col("fdr") < 0.05)
    recurrence = (
        sig.group_by(["i", "j"]).len().rename({"len": "n_sig_complexes"})
    )

    # (b) Mean absolute effect size among significant
    mean_eff = (
        sig.group_by(["i", "j"])
        .agg(pl.col("effect_size").abs().mean().alias("mean_abs_effect"))
    )

    # Combine into a single score: recurrence × mean |effect|
    combined = recurrence.join(mean_eff, on=["i", "j"], how="left")

    corum_score = np.zeros(4096, dtype=np.float64)
    for r in combined.iter_rows(named=True):
        idx = r["i"] * 64 + r["j"]
        corum_score[idx] = r["n_sig_complexes"] * r["mean_abs_effect"]

    n_nonzero_corum = int((corum_score > 0).sum())
    log.info(f"  CORUM: {n_nonzero_corum} entries with score > 0")
    log.info(f"  CORUM score range: [{corum_score.min():.3f}, {corum_score.max():.3f}]")

    # --- Chronos scores ---
    corr_df = pl.read_parquet(OUT_DIR / "chronos_entry_correlations.parquet")
    weights_df = pl.read_parquet(OUT_DIR / "chronos_entry_weights.parquet")

    chronos_corr_score = np.zeros(4096, dtype=np.float64)
    chronos_ridge_score = np.zeros(4096, dtype=np.float64)

    for r in corr_df.iter_rows(named=True):
        idx = r["i"] * 64 + r["j"]
        chronos_corr_score[idx] = abs(r["pearson_mean_dep"])

    for r in weights_df.iter_rows(named=True):
        idx = r["i"] * 64 + r["j"]
        chronos_ridge_score[idx] = abs(r["weight_mean_dep"])

    log.info(f"  Chronos |corr| range: [{chronos_corr_score.min():.4f}, {chronos_corr_score.max():.4f}]")
    log.info(f"  Chronos |ridge| range: [{chronos_ridge_score.min():.4f}, {chronos_ridge_score.max():.4f}]")

    return corum_score, chronos_corr_score, chronos_ridge_score


# ══════════════════════════════════════════════════════════════════════
# X3 — Overlap analysis
# ══════════════════════════════════════════════════════════════════════

def stage_x3(
    corum_score: np.ndarray,
    chronos_score: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Compute overlap at various N, Jaccard, Spearman, and permutation tests."""
    log.info("X3: Overlap analysis...")

    corum_rank = np.argsort(-corum_score)  # descending
    chronos_rank = np.argsort(-chronos_score)

    rows = []
    for n in TOP_NS:
        corum_top = set(corum_rank[:n].tolist())
        chronos_top = set(chronos_rank[:n].tolist())
        overlap = len(corum_top & chronos_top)
        union = len(corum_top | chronos_top)
        jaccard = overlap / union if union > 0 else 0.0

        # Permutation test: expected overlap under random
        perm_overlaps = []
        for _ in range(N_PERM):
            perm_idx = rng.choice(4096, size=n, replace=False)
            perm_overlaps.append(len(corum_top & set(perm_idx.tolist())))
        perm_overlaps = np.array(perm_overlaps)
        expected = perm_overlaps.mean()
        p_value = float((perm_overlaps >= overlap).mean())
        fold_enrichment = overlap / expected if expected > 0 else float("inf")

        rows.append({
            "top_n": n,
            "overlap": overlap,
            "jaccard": jaccard,
            "expected_overlap": float(expected),
            "fold_enrichment": fold_enrichment,
            "perm_p_value": p_value,
        })
        log.info(f"  Top-{n}: overlap={overlap}, Jaccard={jaccard:.3f}, "
                 f"fold={fold_enrichment:.1f}x (p={p_value:.4f})")

    # Global rank correlation
    rho, p_rho = stats.spearmanr(corum_score, chronos_score)
    log.info(f"  Global Spearman(CORUM_score, Chronos_score) = {rho:.4f} (p={p_rho:.3g})")

    overlap_df = pl.DataFrame(rows)
    overlap_df = overlap_df.with_columns([
        pl.lit(float(rho)).alias("global_spearman_rho"),
        pl.lit(float(p_rho)).alias("global_spearman_p"),
    ])
    overlap_df.write_parquet(OUT_DIR / "feature_overlap_metrics.parquet")
    log.info("  Saved feature_overlap_metrics.parquet")

    # --- Overlap vs N plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ns = [r["top_n"] for r in rows]
    overlaps = [r["overlap"] for r in rows]
    expecteds = [r["expected_overlap"] for r in rows]
    folds = [r["fold_enrichment"] for r in rows]

    ax = axes[0]
    ax.bar(np.arange(len(ns)) - 0.15, overlaps, 0.3, label="Observed", color="#2196F3")
    ax.bar(np.arange(len(ns)) + 0.15, expecteds, 0.3, label="Expected (random)", color="#BDBDBD")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Top N")
    ax.set_ylabel("# overlapping entries")
    ax.set_title("CORUM vs Chronos top-N overlap")
    ax.legend()

    ax = axes[1]
    ax.plot(ns, folds, "o-", color="#E91E63", linewidth=2, markersize=8)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Top N")
    ax.set_ylabel("Fold enrichment")
    ax.set_title("Fold enrichment of overlap over random")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_overlap_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_feature_overlap_vs_n.png")

    return overlap_df


# ══════════════════════════════════════════════════════════════════════
# X4 — Feature classification
# ══════════════════════════════════════════════════════════════════════

def stage_x4(
    corum_score: np.ndarray,
    chronos_score: np.ndarray,
) -> pl.DataFrame:
    """Partition entries into shared / CORUM-only / Chronos-only / background."""
    log.info("X4: Feature classification...")

    corum_thresh = np.percentile(corum_score, PERCENTILE_THRESHOLD)
    chronos_thresh = np.percentile(chronos_score, PERCENTILE_THRESHOLD)
    log.info(f"  CORUM threshold (p{PERCENTILE_THRESHOLD}): {corum_thresh:.4f}")
    log.info(f"  Chronos threshold (p{PERCENTILE_THRESHOLD}): {chronos_thresh:.4f}")

    corum_high = corum_score >= corum_thresh
    chronos_high = chronos_score >= chronos_thresh

    ii, jj = np.divmod(np.arange(4096), 64)
    classes = []
    for k in range(4096):
        if corum_high[k] and chronos_high[k]:
            cls = "shared"
        elif corum_high[k]:
            cls = "corum_only"
        elif chronos_high[k]:
            cls = "chronos_only"
        else:
            cls = "background"
        classes.append(cls)

    class_df = pl.DataFrame({
        "i": ii.astype(np.int16),
        "j": jj.astype(np.int16),
        "corum_score": corum_score.astype(np.float32),
        "chronos_score": chronos_score.astype(np.float32),
        "feature_class": classes,
    })
    class_df.write_parquet(OUT_DIR / "feature_classes.parquet")

    counts = class_df.group_by("feature_class").len().sort("len", descending=True)
    for r in counts.iter_rows(named=True):
        log.info(f"  {r['feature_class']}: {r['len']}")
    log.info("  Saved feature_classes.parquet")

    return class_df


# ══════════════════════════════════════════════════════════════════════
# X5 — Structural analysis: row/column enrichment
# ══════════════════════════════════════════════════════════════════════

def stage_x5(class_df: pl.DataFrame) -> pl.DataFrame:
    """Test whether certain rows (i) or columns (j) are overrepresented in each class."""
    log.info("X5: Row/column enrichment per feature class...")

    results = []
    for cls in ["shared", "corum_only", "chronos_only"]:
        subset = class_df.filter(pl.col("feature_class") == cls)
        n_total = subset.height
        if n_total == 0:
            continue

        # Row distribution
        row_counts = subset.group_by("i").len().rename({"len": "count"}).sort("i")
        expected_per_row = n_total / 64

        for r in row_counts.iter_rows(named=True):
            results.append({
                "feature_class": cls,
                "dimension": "row",
                "index": r["i"],
                "count": r["count"],
                "expected": expected_per_row,
                "ratio": r["count"] / expected_per_row,
            })

        # Column distribution
        col_counts = subset.group_by("j").len().rename({"len": "count"}).sort("j")
        expected_per_col = n_total / 64

        for r in col_counts.iter_rows(named=True):
            results.append({
                "feature_class": cls,
                "dimension": "column",
                "index": r["j"],
                "count": r["count"],
                "expected": expected_per_col,
                "ratio": r["count"] / expected_per_col,
            })

        # Chi-squared test for uniformity
        row_obs = np.zeros(64)
        for r in row_counts.iter_rows(named=True):
            row_obs[r["i"]] = r["count"]
        if (row_obs == 0).sum() > len(row_obs) // 2:
            chi2_row, p_row = float("nan"), float("nan")
        else:
            chi2_row, p_row = stats.chisquare(row_obs[row_obs > 0])

        col_obs = np.zeros(64)
        for r in col_counts.iter_rows(named=True):
            col_obs[r["j"]] = r["count"]
        if (col_obs == 0).sum() > len(col_obs) // 2:
            chi2_col, p_col = float("nan"), float("nan")
        else:
            chi2_col, p_col = stats.chisquare(col_obs[col_obs > 0])

        log.info(f"  {cls}: rows chi²={chi2_row:.1f} (p={p_row:.3g}), "
                 f"cols chi²={chi2_col:.1f} (p={p_col:.3g})")

        # Top enriched rows/cols
        top_rows = sorted([(r["i"], r["count"]) for r in row_counts.iter_rows(named=True)],
                          key=lambda x: -x[1])[:3]
        top_cols = sorted([(r["j"], r["count"]) for r in col_counts.iter_rows(named=True)],
                          key=lambda x: -x[1])[:3]
        log.info(f"    Top rows: {top_rows}")
        log.info(f"    Top cols: {top_cols}")

    rc_df = pl.DataFrame(results)
    rc_df.write_parquet(OUT_DIR / "row_column_enrichment.parquet")
    log.info("  Saved row_column_enrichment.parquet")

    # --- Visualization: row/column profiles per class ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col_idx, cls in enumerate(["shared", "corum_only", "chronos_only"]):
        sub = rc_df.filter(pl.col("feature_class") == cls)
        for row_idx, dim in enumerate(["row", "column"]):
            ax = axes[row_idx, col_idx]
            dim_sub = sub.filter(pl.col("dimension") == dim).sort("index")
            if dim_sub.height > 0:
                indices = dim_sub["index"].to_numpy()
                counts = dim_sub["count"].to_numpy()
                full = np.zeros(64)
                full[indices] = counts
                ax.bar(range(64), full, color={"shared": "#9C27B0",
                       "corum_only": "#2196F3", "chronos_only": "#FF9800"}[cls],
                       alpha=0.7, width=1.0)
                ax.axhline(dim_sub["expected"].to_numpy()[0], color="gray",
                           linestyle="--", alpha=0.5)
            ax.set_title(f"{cls} — {dim}s", fontsize=10)
            ax.set_xlabel(f"{dim[0]} index", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("count", fontsize=9)

    fig.suptitle("Row/column distribution of feature classes", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_row_column_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_row_column_enrichment.png")

    return rc_df


# ══════════════════════════════════════════════════════════════════════
# X6 — Block/motif discovery
# ══════════════════════════════════════════════════════════════════════

def stage_x6(
    corum_score: np.ndarray,
    chronos_score: np.ndarray,
    zscored: np.ndarray,
) -> None:
    """Clustered heatmaps revealing block structure in the 64×64 matrix."""
    log.info("X6: Block/motif discovery...")

    corum_map = corum_score.reshape(64, 64)
    chronos_map = chronos_score.reshape(64, 64)
    diff_map = corum_map / (corum_map.max() + 1e-10) - chronos_map / (chronos_map.max() + 1e-10)

    # Smooth for block detection
    sigma = 1.0
    corum_smooth = gaussian_filter(corum_map, sigma=sigma)
    chronos_smooth = gaussian_filter(chronos_map, sigma=sigma)
    diff_smooth = gaussian_filter(diff_map, sigma=sigma)

    # Cluster rows and columns jointly using correlation of gene-level z-scored matrix
    # Row clustering: each row i has a profile across genes (mean of row i entries)
    # Use the actual gene-level data to define row/col similarity
    # zscored: (N_genes, 4096) → reshape to (N_genes, 64, 64)
    z_3d = zscored.reshape(-1, 64, 64)

    # Row profiles: for each row i, the mean entry value across all columns, for each gene
    row_profiles = z_3d.mean(axis=2)  # (N_genes, 64) → transpose for row similarity
    col_profiles = z_3d.mean(axis=1)  # (N_genes, 64)

    # Distance between rows: cosine on their gene-level profiles
    row_dist = pdist(row_profiles.T, metric="correlation")
    col_dist = pdist(col_profiles.T, metric="correlation")
    row_dist = np.clip(row_dist, 0, None)
    col_dist = np.clip(col_dist, 0, None)

    row_Z = linkage(row_dist, method="ward")
    col_Z = linkage(col_dist, method="ward")
    row_order = leaves_list(row_Z)
    col_order = leaves_list(col_Z)

    # Plot clustered heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (mat, title, cmap) in zip(axes, [
        (corum_smooth, "CORUM importance", "YlOrRd"),
        (chronos_smooth, "Chronos importance", "YlGnBu"),
        (diff_smooth, "CORUM − Chronos (normalized)", "RdBu_r"),
    ]):
        reordered = mat[np.ix_(row_order, col_order)]
        if "−" in title:
            vmax = max(abs(reordered.min()), abs(reordered.max()))
            im = ax.imshow(reordered, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(reordered, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("column (reordered)", fontsize=9)
        ax.set_ylabel("row (reordered)", fontsize=9)

    fig.suptitle("Clustered feature importance maps (Gaussian σ=1)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_block_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_feature_block_maps.png")

    # Individual high-res maps
    for name, mat, cmap in [
        ("corum", corum_smooth, "YlOrRd"),
        ("chronos", chronos_smooth, "YlGnBu"),
        ("difference", diff_smooth, "RdBu_r"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        reordered = mat[np.ix_(row_order, col_order)]
        if name == "difference":
            vmax = max(abs(reordered.min()), abs(reordered.max()))
            im = ax.imshow(reordered, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(reordered, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{name.capitalize()} feature map (clustered)", fontsize=12)
        ax.set_xlabel("column (reordered)", fontsize=10)
        ax.set_ylabel("row (reordered)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_{name}_feature_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("  Saved fig_corum_feature_map.png, fig_chronos_feature_map.png, fig_difference_feature_map.png")


# ══════════════════════════════════════════════════════════════════════
# X7 — Cross-predictive power
# ══════════════════════════════════════════════════════════════════════

def stage_x7(
    zscored: np.ndarray,
    gene_names: list[str],
    corum_score: np.ndarray,
    chronos_score: np.ndarray,
    class_df: pl.DataFrame,
) -> pl.DataFrame:
    """Test whether CORUM features predict Chronos and vice versa."""
    log.info("X7: Cross-predictive power...")

    # Load Chronos dependency summaries
    dep_df = pl.read_parquet(OUT_DIR / "gene_dependency_summary.parquet")
    dep_lookup = {r["gene"]: r["mean_dependency"] for r in dep_df.iter_rows(named=True)}

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    common = sorted(set(gene_names) & set(dep_lookup.keys()))
    emb_idx = np.array([gene_idx[g] for g in common])
    y = np.array([dep_lookup[g] for g in common])
    X_full = zscored[emb_idx]  # (N, 4096)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    alphas = np.logspace(0, 5, 20)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = []

    # Feature sets to test
    shared_idx = np.array([r["i"] * 64 + r["j"] for r in
                           class_df.filter(pl.col("feature_class") == "shared").iter_rows(named=True)])
    corum_only_idx = np.array([r["i"] * 64 + r["j"] for r in
                               class_df.filter(pl.col("feature_class") == "corum_only").iter_rows(named=True)])
    chronos_only_idx = np.array([r["i"] * 64 + r["j"] for r in
                                 class_df.filter(pl.col("feature_class") == "chronos_only").iter_rows(named=True)])

    # Also try top-N by each score
    corum_top200 = np.argsort(-corum_score)[:200]
    chronos_top200 = np.argsort(-chronos_score)[:200]

    feature_sets = {
        "full_4096": np.arange(4096),
        "shared": shared_idx,
        "corum_only": corum_only_idx,
        "chronos_only": chronos_only_idx,
        "corum_top200": corum_top200,
        "chronos_top200": chronos_top200,
    }

    for name, feat_idx in feature_sets.items():
        if len(feat_idx) < 5:
            log.info(f"  {name}: too few features ({len(feat_idx)}), skipping")
            results.append({"feature_set": name, "n_features": len(feat_idx),
                           "r2_chronos": float("nan"), "alpha": float("nan")})
            continue

        X_sub = X_full[:, feat_idx]
        ridge = RidgeCV(alphas=alphas, cv=cv, scoring="r2")
        ridge.fit(X_sub, y_scaled)

        results.append({
            "feature_set": name,
            "n_features": len(feat_idx),
            "r2_chronos": float(ridge.best_score_),
            "alpha": float(ridge.alpha_),
        })
        log.info(f"  {name} ({len(feat_idx)} features): R²={ridge.best_score_:.4f}, alpha={ridge.alpha_:.1f}")

    # --- CORUM enrichment using only Chronos features (and vice versa) ---
    # Load CORUM enrichment data to test if chronos features overlap with CORUM signal
    enrichment = pl.read_parquet(OUT_DIR / "corum_entry_enrichment.parquet")
    sig = enrichment.filter(pl.col("fdr") < 0.05)

    for feat_name, feat_idx in [("chronos_top200", chronos_top200), ("corum_top200", corum_top200)]:
        feat_set = set(feat_idx.tolist())
        sig_entries = set()
        for r in sig.iter_rows(named=True):
            sig_entries.add(r["i"] * 64 + r["j"])
        overlap_with_corum_sig = len(feat_set & sig_entries)
        frac = overlap_with_corum_sig / len(feat_set) if len(feat_set) > 0 else 0
        log.info(f"  {feat_name} overlap with CORUM sig entries: "
                 f"{overlap_with_corum_sig}/{len(feat_set)} ({frac:.1%})")

    cross_df = pl.DataFrame(results)
    cross_df.write_parquet(OUT_DIR / "cross_prediction_results.parquet")
    log.info("  Saved cross_prediction_results.parquet")

    # --- Bar plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    names = cross_df["feature_set"].to_list()
    r2s = cross_df["r2_chronos"].to_numpy()
    n_feats = cross_df["n_features"].to_numpy()
    colors = {"full_4096": "#607D8B", "shared": "#9C27B0", "corum_only": "#2196F3",
              "chronos_only": "#FF9800", "corum_top200": "#42A5F5", "chronos_top200": "#FFA726"}
    bars = ax.bar(range(len(names)), r2s, color=[colors.get(n, "#999") for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(n={nf})" for n, nf in zip(names, n_feats)],
                       fontsize=8, rotation=15)
    ax.set_ylabel("R² (5-fold CV, predicting Chronos mean_dep)")
    ax.set_title("Cross-predictive power: which features predict Chronos dependency?")
    for bar, r2 in zip(bars, r2s):
        if not np.isnan(r2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{r2:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_cross_prediction.png")

    return cross_df


# ══════════════════════════════════════════════════════════════════════
# X8 — Feature co-variation
# ══════════════════════════════════════════════════════════════════════

def stage_x8(zscored: np.ndarray) -> None:
    """Cluster (i,j) entries by their cross-gene correlation patterns."""
    log.info("X8: Feature co-variation clustering...")

    # zscored: (N_genes, 4096)
    # We want correlation between features (columns), but 4096x4096 corr matrix
    # is huge (128MB). Subsample features for tractability.
    # Use top-variance features
    var = zscored.var(axis=0)
    top_feat = np.argsort(-var)[:500]  # top 500 most variable

    X_sub = zscored[:, top_feat]  # (N_genes, 500)
    log.info(f"  Computing correlation among top 500 most-variable features...")
    corr = np.corrcoef(X_sub.T)  # (500, 500)
    np.fill_diagonal(corr, 1.0)

    # Cluster
    dist = 1 - corr
    dist = np.clip(dist, 0, None)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    # Plot
    corr_ordered = corr[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr_ordered, cmap="RdBu_r", aspect="equal", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Feature co-variation (top 500 by variance, clustered)", fontsize=12)
    ax.set_xlabel("Feature index (reordered)", fontsize=10)
    ax.set_ylabel("Feature index (reordered)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_covariation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_feature_covariation.png")

    # Identify major clusters (cut at 8 clusters)
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(Z, t=8, criterion="maxclust")

    # Map back to (i,j)
    cluster_rows = []
    for idx_in_sub, feat_flat_idx in enumerate(top_feat):
        i, j = divmod(int(feat_flat_idx), 64)
        cluster_rows.append({
            "i": i, "j": j,
            "flat_idx": int(feat_flat_idx),
            "variance": float(var[feat_flat_idx]),
            "cluster": int(clusters[list(order).index(idx_in_sub)] if idx_in_sub in order else clusters[idx_in_sub]),
        })

    # Simpler: just assign cluster to each feature in top_feat
    cluster_rows2 = []
    for idx_in_sub, feat_flat_idx in enumerate(top_feat):
        i, j = divmod(int(feat_flat_idx), 64)
        cluster_rows2.append({
            "i": i, "j": j,
            "flat_idx": int(feat_flat_idx),
            "variance": float(var[feat_flat_idx]),
            "cluster": int(clusters[idx_in_sub]),
        })

    feat_cluster_df = pl.DataFrame(cluster_rows2)
    feat_cluster_df.write_parquet(OUT_DIR / "feature_clusters.parquet")
    log.info(f"  Saved feature_clusters.parquet ({feat_cluster_df.height} features)")

    cluster_sizes = feat_cluster_df.group_by("cluster").len().sort("cluster")
    for r in cluster_sizes.iter_rows(named=True):
        log.info(f"    Cluster {r['cluster']}: {r['len']} features")


# ══════════════════════════════════════════════════════════════════════
# X9-X10 — Final outputs and report
# ══════════════════════════════════════════════════════════════════════

def stage_x9_x10(
    corum_score: np.ndarray,
    chronos_score: np.ndarray,
    overlap_df: pl.DataFrame,
    class_df: pl.DataFrame,
    cross_df: pl.DataFrame,
    rc_df: pl.DataFrame,
) -> None:
    """Generate summary report with interpretation."""
    log.info("X9-X10: Writing report...")

    # Gather stats
    counts = {r["feature_class"]: r["len"]
              for r in class_df.group_by("feature_class").len().iter_rows(named=True)}

    overlap_rows = overlap_df.to_dicts()

    cross_rows = cross_df.to_dicts()
    r2_full = next((r["r2_chronos"] for r in cross_rows if r["feature_set"] == "full_4096"), float("nan"))
    r2_shared = next((r["r2_chronos"] for r in cross_rows if r["feature_set"] == "shared"), float("nan"))
    r2_corum = next((r["r2_chronos"] for r in cross_rows if r["feature_set"] == "corum_only"), float("nan"))
    r2_chronos = next((r["r2_chronos"] for r in cross_rows if r["feature_set"] == "chronos_only"), float("nan"))

    rho = overlap_df["global_spearman_rho"][0]

    # Structural analysis highlights
    # Find most enriched rows/cols per class
    def _top_rc(cls: str, dim: str, n: int = 3) -> list[tuple[int, float]]:
        sub = rc_df.filter((pl.col("feature_class") == cls) & (pl.col("dimension") == dim))
        sub = sub.sort("ratio", descending=True).head(n)
        return [(r["index"], r["ratio"]) for r in sub.iter_rows(named=True)]

    lines = [
        "# CORUM vs Chronos Feature Comparison — Report",
        "",
        "## 1. Overview",
        "",
        "This analysis compares which latent feature interactions (i,j) in the 64×64 diff",
        "covariance matrix are important for **protein-complex biology (CORUM)** vs",
        "**cellular dependency (Chronos)**.",
        "",
        f"- **Feature space:** 4,096 entries (64×64 matrix)",
        f"- **Gene universe:** 9,493 genes with ≥3 embedded variants",
        f"- **CORUM scoring:** recurrence × mean |effect size| across 300 complexes",
        f"- **Chronos scoring:** |Pearson r| between entry z-scores and mean Chronos dependency",
        "",
        "## 2. Overlap Statistics",
        "",
        f"**Global rank correlation:** Spearman ρ = {rho:.4f}",
        "",
        "| Top N | Overlap | Jaccard | Fold enrichment | p-value |",
        "|-------|---------|---------|-----------------|---------|",
    ]
    for r in overlap_rows:
        lines.append(f"| {r['top_n']} | {r['overlap']} | {r['jaccard']:.3f} | "
                     f"{r['fold_enrichment']:.1f}x | {r['perm_p_value']:.4f} |")

    lines += [
        "",
        "## 3. Feature Classification (top 10%)",
        "",
        f"| Class | Count |",
        f"|-------|-------|",
        f"| Shared (CORUM + Chronos) | {counts.get('shared', 0)} |",
        f"| CORUM-only | {counts.get('corum_only', 0)} |",
        f"| Chronos-only | {counts.get('chronos_only', 0)} |",
        f"| Background | {counts.get('background', 0)} |",
        "",
        "## 4. Cross-Predictive Power (Chronos R²)",
        "",
        "| Feature set | N features | R² |",
        "|-------------|------------|-----|",
    ]
    for r in cross_rows:
        r2_str = f"{r['r2_chronos']:.4f}" if not np.isnan(r['r2_chronos']) else "—"
        lines.append(f"| {r['feature_set']} | {r['n_features']} | {r2_str} |")

    lines += [
        "",
        "## 5. Structural Patterns",
        "",
    ]
    for cls in ["shared", "corum_only", "chronos_only"]:
        top_rows = _top_rc(cls, "row")
        top_cols = _top_rc(cls, "column")
        lines.append(f"**{cls}:**")
        if top_rows:
            lines.append(f"  - Top enriched rows: {', '.join(f'i={r[0]} ({r[1]:.1f}x)' for r in top_rows)}")
        if top_cols:
            lines.append(f"  - Top enriched cols: {', '.join(f'j={c[0]} ({c[1]:.1f}x)' for c in top_cols)}")
        lines.append("")

    # Interpretation
    overlap_at_200 = next((r for r in overlap_rows if r["top_n"] == 200), None)
    fold_200 = overlap_at_200["fold_enrichment"] if overlap_at_200 else 0

    if fold_200 > 2.0:
        overlap_interpretation = (
            "The feature sets show **strong overlap** — many of the same latent interactions "
            "are important for both protein-complex structure and gene essentiality. This suggests "
            "that structural biology (complex membership) is a significant driver of cellular dependency."
        )
    elif fold_200 > 1.5:
        overlap_interpretation = (
            "The feature sets show **moderate overlap** — some latent interactions serve both "
            "structural and dependency roles, but substantial fractions are specific to each domain. "
            "The embedding partially disentangles structure from phenotype."
        )
    else:
        overlap_interpretation = (
            "The feature sets show **limited overlap** — CORUM and Chronos signal is largely "
            "encoded in distinct latent interactions. The embedding separates structural "
            "protein-complex biology from cellular dependency/survival encoding."
        )

    shared_power = ""
    if not np.isnan(r2_shared) and not np.isnan(r2_full):
        ratio = r2_shared / r2_full if r2_full > 0 else 0
        shared_power = (
            f"Shared features alone achieve {ratio:.0%} of the full model's predictive power "
            f"for Chronos dependency (R²={r2_shared:.3f} vs {r2_full:.3f}). "
        )

    lines += [
        "## 6. Interpretation",
        "",
        "### Degree of overlap",
        "",
        overlap_interpretation,
        "",
        "### Signal structure",
        "",
        "The clustered heatmaps (see `fig_feature_block_maps.png`) reveal whether the signal is:",
        "- **Diffuse** (spread across the matrix) → general regulatory encoding",
        "- **Localized** (concentrated in blocks) → specific latent interaction modules",
        "",
        "### Cross-predictive power",
        "",
        shared_power,
        "If CORUM-only features still predict Chronos to some extent, complex membership itself "
        "is predictive of essentiality. If Chronos-only features show no CORUM signal, dependency "
        "encoding is independent of structural biology in those latent dimensions.",
        "",
        "### Biological implication",
        "",
        "- **Shared features** → structure drives survival: genes in the same complex tend to have "
        "similar essentiality, and the model encodes this through shared latent interaction patterns.",
        "- **CORUM-specific features** → the model captures protein-complex membership through "
        "interaction patterns that are orthogonal to dependency.",
        "- **Chronos-specific features** → the model encodes essentiality information through "
        "latent interactions not related to complex membership.",
        "",
        "## Output Files",
        "",
        "- `feature_overlap_metrics.parquet` — overlap statistics at various N",
        "- `feature_classes.parquet` — per-entry classification (shared/CORUM/Chronos/background)",
        "- `row_column_enrichment.parquet` — row/column distribution per class",
        "- `cross_prediction_results.parquet` — R² from feature-subset ridge regression",
        "- `feature_clusters.parquet` — feature co-variation clusters",
        "- `fig_feature_overlap_vs_n.png`",
        "- `fig_feature_block_maps.png`, `fig_corum_feature_map.png`, `fig_chronos_feature_map.png`, `fig_difference_feature_map.png`",
        "- `fig_row_column_enrichment.png`",
        "- `fig_cross_prediction.png`",
        "- `fig_feature_covariation.png`",
    ]

    (OUT_DIR / "corum_vs_chronos_feature_report.md").write_text("\n".join(lines) + "\n")
    log.info("  Saved corum_vs_chronos_feature_report.md")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # X1
    zscored, gene_names = stage_x1()

    # X2
    corum_score, chronos_corr_score, chronos_ridge_score = stage_x2()
    # Use correlation-based score as primary Chronos ranking
    chronos_score = chronos_corr_score

    # X3
    overlap_df = stage_x3(corum_score, chronos_score, rng)

    # X4
    class_df = stage_x4(corum_score, chronos_score)

    # X5
    rc_df = stage_x5(class_df)

    # X6
    stage_x6(corum_score, chronos_score, zscored)

    # X7
    cross_df = stage_x7(zscored, gene_names, corum_score, chronos_score, class_df)

    # X8
    stage_x8(zscored)

    # X9-X10
    stage_x9_x10(corum_score, chronos_score, overlap_df, class_df, cross_df, rc_df)

    # Config
    config = {
        "analysis": "corum_vs_chronos_features",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERM,
        "top_ns": TOP_NS,
        "percentile_threshold": PERCENTILE_THRESHOLD,
    }
    with open(OUT_DIR / "corum_vs_chronos_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")
    print(f"\n{'='*80}\nCORUM vs CHRONOS FEATURE COMPARISON — COMPLETE ({elapsed:.0f}s)\n{'='*80}")


if __name__ == "__main__":
    main()
