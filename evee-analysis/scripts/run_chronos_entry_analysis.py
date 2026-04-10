#!/usr/bin/env python3
"""Chronos dependency × matrix entry analysis.

Extends the CORUM feature-level interpretability to continuous Chronos
dependency labels. Identifies which latent feature interactions (i,j) are
associated with gene dependency patterns.

Stages:
  1. Gene-level dependency summaries from Chronos
  2. Entry-wise Pearson/Spearman correlation with dependency metrics + FDR
  3. Ridge regression (4096-d → dependency metric) + weight heatmap
  4. Compare top CORUM-enriched entries with top Chronos-associated entries
  5. Outputs — top-50 tables, heatmaps, report

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_chronos_entry_analysis.py
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
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
CHRONOS_PATH = EVEE_ROOT / "data" / "CRISPR_DepMap_Public_26Q1Score_Chronos_subsetted.csv"
DEMETER2_PATH = EVEE_ROOT / "data" / "RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv"
OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"

RANDOM_SEED = 42
DEP_THRESHOLD = -0.5  # Chronos score below this = "dependent"


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    fdr = p_values * n / ranks
    fdr_sorted = fdr[order]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[order] = fdr_sorted
    return np.clip(fdr, 0, 1)


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — Gene-level dependency summaries
# ══════════════════════════════════════════════════════════════════════

def stage1_dependency_summaries(
    gene_names: list[str],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Load Chronos, compute per-gene dependency statistics."""
    log.info("STAGE 1: Gene-level dependency summaries...")

    chron_df = pl.read_csv(str(CHRONOS_PATH))
    chron_df = chron_df.rename({chron_df.columns[0]: "depmap_id"})

    # Filter to DEMETER2 overlapping cell lines for consistency
    dem_df = pl.read_csv(str(DEMETER2_PATH))
    dem_ids = set(dem_df["depmap_id"].to_list())
    chron_df = chron_df.filter(pl.col("depmap_id").is_in(list(dem_ids)))
    log.info(f"  Chronos: {chron_df.height} cell lines (DEMETER2 overlap)")

    gene_cols = [c for c in chron_df.columns if c != "depmap_id"]
    chron_genes = {g.upper(): g for g in gene_cols}
    log.info(f"  Chronos genes: {len(gene_cols)}")

    gene_set = set(gene_names)
    overlap = sorted(gene_set & set(chron_genes.keys()))
    log.info(f"  Overlap with embedding genes: {len(overlap)}")

    mat = chron_df.select(gene_cols).to_numpy().astype(np.float64)
    gene_to_col = {g.upper(): i for i, g in enumerate(gene_cols)}

    rows = []
    gene_to_profile: dict[str, np.ndarray] = {}
    for gene in overlap:
        col_idx = gene_to_col[gene]
        vals = mat[:, col_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        rows.append({
            "gene": gene,
            "mean_dependency": float(np.mean(valid)),
            "std_dependency": float(np.std(valid)),
            "fraction_dependent": float((valid < DEP_THRESHOLD).mean()),
            "n_cell_lines": len(valid),
        })
        gene_to_profile[gene] = vals

    summary_df = pl.DataFrame(rows).sort("mean_dependency")
    summary_df.write_parquet(OUT_DIR / "gene_dependency_summary.parquet")
    log.info(f"  {summary_df.height} genes with valid Chronos profiles")
    log.info(f"  mean_dependency: median={summary_df['mean_dependency'].median():.3f}, "
             f"range=[{summary_df['mean_dependency'].min():.3f}, {summary_df['mean_dependency'].max():.3f}]")
    log.info(f"  fraction_dependent: median={summary_df['fraction_dependent'].median():.3f}, "
             f"max={summary_df['fraction_dependent'].max():.3f}")

    return summary_df, gene_to_profile


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — Entry-wise correlation
# ══════════════════════════════════════════════════════════════════════

def stage2_entry_correlations(
    zscored: np.ndarray,
    gene_names: list[str],
    dep_summary: pl.DataFrame,
) -> pl.DataFrame:
    """Correlate each matrix entry (i,j) with dependency metrics across genes."""
    log.info("STAGE 2: Entry-wise correlation with dependency metrics...")

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    dep_genes = set(dep_summary["gene"].to_list())
    common = sorted(set(gene_names) & dep_genes)
    log.info(f"  Common genes: {len(common)}")

    dep_lookup = {}
    for r in dep_summary.iter_rows(named=True):
        dep_lookup[r["gene"]] = r

    emb_idx = np.array([gene_idx[g] for g in common])
    z_sub = zscored[emb_idx]  # (N_common, 4096)

    mean_dep = np.array([dep_lookup[g]["mean_dependency"] for g in common])
    frac_dep = np.array([dep_lookup[g]["fraction_dependent"] for g in common])

    n_entries = 4096
    # Vectorized Pearson across all entries at once
    # z_sub: (N, 4096), mean_dep: (N,)
    n = len(common)

    def _vectorized_corr(X: np.ndarray, y: np.ndarray):
        """Pearson r and p for each column of X against y."""
        y_c = y - y.mean()
        X_c = X - X.mean(axis=0, keepdims=True)
        y_std = y_c.std()
        X_std = X_c.std(axis=0)
        X_std[X_std < 1e-10] = 1e-10
        r = (X_c * y_c[:, None]).mean(axis=0) / (X_std * y_std)
        # t-statistic for significance
        t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-15))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
        return r, p

    def _vectorized_spearman(X: np.ndarray, y: np.ndarray):
        """Spearman rho for each column of X against y."""
        y_ranks = stats.rankdata(y)
        X_ranks = np.apply_along_axis(stats.rankdata, 0, X)
        return _vectorized_corr(X_ranks, y_ranks)

    log.info("  Computing Pearson correlations with mean_dependency...")
    pearson_mean_r, pearson_mean_p = _vectorized_corr(z_sub, mean_dep)
    log.info("  Computing Spearman correlations with mean_dependency...")
    spearman_mean_r, spearman_mean_p = _vectorized_spearman(z_sub, mean_dep)
    log.info("  Computing Pearson correlations with fraction_dependent...")
    pearson_frac_r, pearson_frac_p = _vectorized_corr(z_sub, frac_dep)
    log.info("  Computing Spearman correlations with fraction_dependent...")
    spearman_frac_r, spearman_frac_p = _vectorized_spearman(z_sub, frac_dep)

    # FDR on all 4 p-value sets
    fdr_pearson_mean = benjamini_hochberg(pearson_mean_p)
    fdr_spearman_mean = benjamini_hochberg(spearman_mean_p)
    fdr_pearson_frac = benjamini_hochberg(pearson_frac_p)
    fdr_spearman_frac = benjamini_hochberg(spearman_frac_p)

    ii, jj = np.divmod(np.arange(n_entries), 64)

    result = pl.DataFrame({
        "i": ii.astype(np.int16),
        "j": jj.astype(np.int16),
        "pearson_mean_dep": pearson_mean_r.astype(np.float32),
        "pearson_mean_dep_p": pearson_mean_p.astype(np.float32),
        "pearson_mean_dep_fdr": fdr_pearson_mean.astype(np.float32),
        "spearman_mean_dep": spearman_mean_r.astype(np.float32),
        "spearman_mean_dep_p": spearman_mean_p.astype(np.float32),
        "spearman_mean_dep_fdr": fdr_spearman_mean.astype(np.float32),
        "pearson_frac_dep": pearson_frac_r.astype(np.float32),
        "pearson_frac_dep_p": pearson_frac_p.astype(np.float32),
        "pearson_frac_dep_fdr": fdr_pearson_frac.astype(np.float32),
        "spearman_frac_dep": spearman_frac_r.astype(np.float32),
        "spearman_frac_dep_p": spearman_frac_p.astype(np.float32),
        "spearman_frac_dep_fdr": fdr_spearman_frac.astype(np.float32),
    })
    result.write_parquet(OUT_DIR / "chronos_entry_correlations.parquet")
    log.info(f"  Saved chronos_entry_correlations.parquet ({result.height} rows)")

    n_sig_pearson = int((fdr_pearson_mean < 0.05).sum())
    n_sig_spearman = int((fdr_spearman_mean < 0.05).sum())
    n_sig_frac = int((fdr_pearson_frac < 0.05).sum())
    log.info(f"  Sig entries (FDR<0.05): Pearson(mean_dep)={n_sig_pearson}, "
             f"Spearman(mean_dep)={n_sig_spearman}, Pearson(frac_dep)={n_sig_frac}")

    top_abs = result.with_columns(
        pl.col("pearson_mean_dep").abs().alias("abs_r")
    ).sort("abs_r", descending=True).head(5)
    for r in top_abs.iter_rows(named=True):
        log.info(f"    ({r['i']},{r['j']}): pearson_mean_dep={r['pearson_mean_dep']:.4f}, "
                 f"fdr={r['pearson_mean_dep_fdr']:.4g}")

    return result


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — Ridge regression projection
# ══════════════════════════════════════════════════════════════════════

def stage3_ridge_regression(
    zscored: np.ndarray,
    gene_names: list[str],
    dep_summary: pl.DataFrame,
) -> np.ndarray:
    """Ridge regression: 4096-d gene embedding → dependency metric."""
    log.info("STAGE 3: Ridge regression (embedding → dependency)...")

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    dep_genes = set(dep_summary["gene"].to_list())
    common = sorted(set(gene_names) & dep_genes)

    dep_lookup = {}
    for r in dep_summary.iter_rows(named=True):
        dep_lookup[r["gene"]] = r

    emb_idx = np.array([gene_idx[g] for g in common])
    X = zscored[emb_idx]  # (N, 4096)
    y_mean = np.array([dep_lookup[g]["mean_dependency"] for g in common])
    y_frac = np.array([dep_lookup[g]["fraction_dependent"] for g in common])

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # RidgeCV with cross-validation
    alphas = np.logspace(-2, 6, 50)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    for target_name, y in [("mean_dependency", y_mean), ("fraction_dependent", y_frac)]:
        log.info(f"  Fitting ridge for {target_name}...")
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        ridge = RidgeCV(alphas=alphas, cv=cv, scoring="r2")
        ridge.fit(X_scaled, y_scaled)

        log.info(f"    Best alpha={ridge.alpha_:.2f}, R²={ridge.best_score_:.4f}")

        weights = ridge.coef_  # (4096,)
        results[target_name] = {
            "weights": weights,
            "alpha": float(ridge.alpha_),
            "r2": float(ridge.best_score_),
        }

    # Save weights table
    ii, jj = np.divmod(np.arange(4096), 64)
    w_mean = results["mean_dependency"]["weights"]
    w_frac = results["fraction_dependent"]["weights"]

    weights_df = pl.DataFrame({
        "i": ii.astype(np.int16),
        "j": jj.astype(np.int16),
        "weight_mean_dep": w_mean.astype(np.float32),
        "weight_frac_dep": w_frac.astype(np.float32),
    })
    weights_df.write_parquet(OUT_DIR / "chronos_entry_weights.parquet")
    log.info("  Saved chronos_entry_weights.parquet")

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, info) in zip(axes, results.items()):
        w_mat = info["weights"].reshape(64, 64)
        vmax = np.abs(w_mat).max()
        im = ax.imshow(w_mat, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Ridge weights: {name}\n(alpha={info['alpha']:.1f}, R²={info['r2']:.3f})",
                     fontsize=10)
        ax.set_xlabel("j (right feature)", fontsize=9)
        ax.set_ylabel("i (left feature)", fontsize=9)

    fig.suptitle("Chronos dependency → latent feature weights", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dependency_weight_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_dependency_weight_heatmap.png")

    return w_mean


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — Compare with CORUM
# ══════════════════════════════════════════════════════════════════════

def stage4_corum_comparison(
    corr_df: pl.DataFrame,
    ridge_weights: np.ndarray,
) -> pl.DataFrame:
    """Compare top CORUM-enriched entries with top Chronos-associated entries."""
    log.info("STAGE 4: CORUM vs Chronos entry comparison...")

    # Load CORUM recurrent entries
    rec_path = OUT_DIR / "corum_recurrent_entries.parquet"
    if not rec_path.exists():
        log.warning("  corum_recurrent_entries.parquet not found — skipping comparison")
        return pl.DataFrame()

    corum_rec = pl.read_parquet(rec_path)
    log.info(f"  CORUM recurrent entries: {corum_rec.height}")

    # Load CORUM top entries for a richer comparison
    top_path = OUT_DIR / "corum_complex_top_entries.parquet"
    corum_top = pl.read_parquet(top_path) if top_path.exists() else pl.DataFrame()

    # Build global CORUM significance: for each (i,j), the number of complexes
    # where it's in the top-20 and FDR<0.05
    corum_entry_score = {}
    if corum_rec.height > 0:
        for r in corum_rec.iter_rows(named=True):
            corum_entry_score[(r["i"], r["j"])] = r["n_complexes"]

    # Build Chronos correlation ranking
    abs_pearson = corr_df.with_columns(
        pl.col("pearson_mean_dep").abs().alias("abs_pearson_mean"),
        pl.col("spearman_mean_dep").abs().alias("abs_spearman_mean"),
    )

    # Top 200 by Chronos correlation
    top_chronos = abs_pearson.sort("abs_pearson_mean", descending=True).head(200)
    chronos_top_set = set()
    for r in top_chronos.iter_rows(named=True):
        chronos_top_set.add((r["i"], r["j"]))

    # Top 200 by CORUM recurrence
    corum_top_set = set()
    for r in corum_rec.sort("n_complexes", descending=True).head(200).iter_rows(named=True):
        corum_top_set.add((r["i"], r["j"]))

    overlap = chronos_top_set & corum_top_set
    log.info(f"  Top-200 overlap: {len(overlap)} entries")
    log.info(f"  Chronos-only: {len(chronos_top_set - corum_top_set)}")
    log.info(f"  CORUM-only: {len(corum_top_set - chronos_top_set)}")

    # Rank correlation across all 4096 entries
    chronos_ranks = np.zeros(4096)
    for r in corr_df.iter_rows(named=True):
        idx = r["i"] * 64 + r["j"]
        chronos_ranks[idx] = abs(r["pearson_mean_dep"])

    corum_ranks = np.zeros(4096)
    for (i, j), score in corum_entry_score.items():
        corum_ranks[i * 64 + j] = score

    # Ridge weight magnitude as a third ranking
    ridge_ranks = np.abs(ridge_weights)

    rho_corr_corum, p_corr_corum = stats.spearmanr(chronos_ranks, corum_ranks)
    rho_ridge_corum, p_ridge_corum = stats.spearmanr(ridge_ranks, corum_ranks)
    rho_corr_ridge, p_corr_ridge = stats.spearmanr(chronos_ranks, ridge_ranks)

    log.info(f"  Spearman(Chronos_corr, CORUM_recurrence) = {rho_corr_corum:.4f} (p={p_corr_corum:.3g})")
    log.info(f"  Spearman(Ridge_weight, CORUM_recurrence) = {rho_ridge_corum:.4f} (p={p_ridge_corum:.3g})")
    log.info(f"  Spearman(Chronos_corr, Ridge_weight) = {rho_corr_ridge:.4f} (p={p_corr_ridge:.3g})")

    # Build joined table for all 4096 entries
    ii, jj = np.divmod(np.arange(4096), 64)
    joined_rows = []
    for k in range(4096):
        i, j = int(ii[k]), int(jj[k])
        joined_rows.append({
            "i": i,
            "j": j,
            "corum_n_complexes": int(corum_entry_score.get((i, j), 0)),
            "chronos_abs_pearson": float(chronos_ranks[k]),
            "ridge_abs_weight": float(ridge_ranks[k]),
            "in_top200_corum": (i, j) in corum_top_set,
            "in_top200_chronos": (i, j) in chronos_top_set,
            "in_both_top200": (i, j) in overlap,
        })

    joined = pl.DataFrame(joined_rows)
    joined.write_parquet(OUT_DIR / "corum_vs_chronos_overlap.parquet")
    log.info("  Saved corum_vs_chronos_overlap.parquet")

    return joined


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — Outputs: top-50 tables, heatmaps, report
# ══════════════════════════════════════════════════════════════════════

def stage5_outputs(
    dep_summary: pl.DataFrame,
    corr_df: pl.DataFrame,
    ridge_weights: np.ndarray,
    overlap_df: pl.DataFrame,
) -> None:
    """Generate final summary tables, figures, and report."""
    log.info("STAGE 5: Final outputs...")

    # --- Correlation heatmaps ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (col, title) in zip(axes.flat, [
        ("pearson_mean_dep", "Pearson(mean dependency)"),
        ("spearman_mean_dep", "Spearman(mean dependency)"),
        ("pearson_frac_dep", "Pearson(fraction dependent)"),
        ("spearman_frac_dep", "Spearman(fraction dependent)"),
    ]):
        vals = corr_df[col].to_numpy().reshape(64, 64)
        vmax = max(abs(vals.min()), abs(vals.max()), 0.05)
        im = ax.imshow(vals, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("j", fontsize=9)
        ax.set_ylabel("i", fontsize=9)

    fig.suptitle("Entry-wise correlation with Chronos dependency", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_chronos_entry_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_chronos_entry_correlations.png")

    # --- Overlap visualization ---
    if overlap_df.height > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (col, title, cmap) in zip(axes, [
            ("corum_n_complexes", "CORUM recurrence", "YlOrRd"),
            ("chronos_abs_pearson", "Chronos |Pearson r|", "YlGnBu"),
            ("ridge_abs_weight", "Ridge |weight|", "YlGnBu"),
        ]):
            vals = overlap_df.sort(["i", "j"])[col].to_numpy().reshape(64, 64)
            im = ax.imshow(vals, cmap=cmap, aspect="equal")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("j", fontsize=9)
            ax.set_ylabel("i", fontsize=9)

        fig.suptitle("CORUM vs Chronos entry importance comparison", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_corum_vs_chronos_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  Saved fig_corum_vs_chronos_comparison.png")

    # --- Top-50 tables ---
    top50_corr = (
        corr_df
        .with_columns(pl.col("pearson_mean_dep").abs().alias("abs_r"))
        .sort("abs_r", descending=True)
        .head(50)
        .drop("abs_r")
    )
    top50_corr.write_parquet(OUT_DIR / "chronos_top50_entries.parquet")
    log.info("  Saved chronos_top50_entries.parquet")

    # --- Report ---
    n_sig_pearson = int((corr_df["pearson_mean_dep_fdr"].to_numpy() < 0.05).sum())
    n_sig_spearman = int((corr_df["spearman_mean_dep_fdr"].to_numpy() < 0.05).sum())
    n_sig_frac = int((corr_df["pearson_frac_dep_fdr"].to_numpy() < 0.05).sum())

    # Overlap stats
    if overlap_df.height > 0:
        n_both = int(overlap_df["in_both_top200"].sum())
        n_corum_only = int(overlap_df["in_top200_corum"].sum()) - n_both
        n_chronos_only = int(overlap_df["in_top200_chronos"].sum()) - n_both
        rho_vals = overlap_df.sort(["i", "j"])
        spearman_r, spearman_p = stats.spearmanr(
            rho_vals["corum_n_complexes"].to_numpy(),
            rho_vals["chronos_abs_pearson"].to_numpy(),
        )
    else:
        n_both = n_corum_only = n_chronos_only = 0
        spearman_r = spearman_p = float("nan")

    abs_pearson_arr = corr_df["pearson_mean_dep"].to_numpy()
    max_r_idx = np.argmax(np.abs(abs_pearson_arr))

    lines = [
        "# Chronos Dependency × Latent Feature Analysis — Report",
        "",
        "## 1. Gene-Level Dependency Summaries",
        "",
        f"- **Genes with valid Chronos profiles:** {dep_summary.height:,}",
        f"- **Dependency threshold:** Chronos < {DEP_THRESHOLD}",
        f"- **Median mean_dependency:** {dep_summary['mean_dependency'].median():.3f}",
        f"- **Median fraction_dependent:** {dep_summary['fraction_dependent'].median():.3f}",
        "",
        "## 2. Entry-wise Correlations",
        "",
        f"- **Significant entries (FDR<0.05):**",
        f"  - Pearson with mean_dependency: {n_sig_pearson:,} / 4,096",
        f"  - Spearman with mean_dependency: {n_sig_spearman:,} / 4,096",
        f"  - Pearson with fraction_dependent: {n_sig_frac:,} / 4,096",
        "",
        f"- **Strongest entry:** ({int(max_r_idx//64)},{int(max_r_idx%64)}) "
        f"with r={abs_pearson_arr[max_r_idx]:.4f}",
        "",
        "Individual matrix entries (i,j) show significant correlations with gene dependency, "
        "meaning specific latent feature interactions systematically track with how essential "
        "a gene is across cancer cell lines.",
        "",
        "## 3. Ridge Regression Weights",
        "",
        "A ridge regression model predicts gene dependency from the full 4096-d z-scored "
        "embedding vector. The weight map reveals which entries are jointly informative.",
        "",
        "See `fig_dependency_weight_heatmap.png`.",
        "",
        "## 4. CORUM vs Chronos Comparison",
        "",
        f"- **Top-200 overlap:** {n_both} entries appear in both CORUM and Chronos top-200",
        f"- **CORUM-only:** {n_corum_only}",
        f"- **Chronos-only:** {n_chronos_only}",
        f"- **Spearman(CORUM_recurrence, Chronos_|r|):** {spearman_r:.4f} (p={spearman_p:.3g})",
        "",
    ]

    if n_both > 0:
        lines += [
            "**Shared entries** suggest latent features that encode both protein-complex "
            "membership and gene essentiality — these are the most biologically interpretable.",
            "",
            "**CORUM-only entries** likely capture structural/interaction biology not directly "
            "tied to dependency.",
            "",
            "**Chronos-only entries** capture dependency-relevant information that is not "
            "specifically about protein-complex co-membership.",
        ]
    else:
        lines += [
            "Low overlap suggests CORUM complex biology and gene essentiality are encoded "
            "in largely distinct latent features.",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "The Evo2 latent features encode both structural protein-complex biology (CORUM) "
        "and gene essentiality (Chronos) to varying degrees. Where these signals overlap, "
        "the features likely capture fundamental aspects of gene function that are relevant "
        "to both complex membership and cellular fitness.",
        "",
        "## Output Files",
        "",
        "- `gene_dependency_summary.parquet`",
        "- `chronos_entry_correlations.parquet`",
        "- `chronos_entry_weights.parquet`",
        "- `corum_vs_chronos_overlap.parquet`",
        "- `chronos_top50_entries.parquet`",
        "- `fig_chronos_entry_correlations.png`",
        "- `fig_dependency_weight_heatmap.png`",
        "- `fig_corum_vs_chronos_comparison.png`",
    ]

    (OUT_DIR / "chronos_entry_analysis_report.md").write_text("\n".join(lines) + "\n")
    log.info("  Saved chronos_entry_analysis_report.md")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load gene-level matrices (from CORUM interpretability Stage 1)
    npz_path = OUT_DIR / "gene_level_matrices.npz"
    if not npz_path.exists():
        log.error(f"gene_level_matrices.npz not found at {npz_path}")
        log.error("Run run_corum_interpretability.py first (Stage 1)")
        sys.exit(1)

    log.info("Loading gene-level matrices...")
    data = np.load(str(npz_path), allow_pickle=True)
    gene_names = list(data["gene_names"])
    zscored = data["zscored_flat"]
    log.info(f"  {len(gene_names):,} genes, {zscored.shape[1]} features")

    # Stage 1
    dep_summary, gene_to_profile = stage1_dependency_summaries(gene_names)

    # Stage 2
    corr_df = stage2_entry_correlations(zscored, gene_names, dep_summary)

    # Stage 3
    ridge_weights = stage3_ridge_regression(zscored, gene_names, dep_summary)

    # Stage 4
    overlap_df = stage4_corum_comparison(corr_df, ridge_weights)

    # Stage 5
    stage5_outputs(dep_summary, corr_df, ridge_weights, overlap_df)

    # Run config
    config = {
        "analysis": "chronos_entry_analysis",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "dep_threshold": DEP_THRESHOLD,
        "n_genes_embedded": len(gene_names),
        "n_genes_with_chronos": dep_summary.height,
    }
    with open(OUT_DIR / "chronos_entry_analysis_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")
    print(f"\n{'='*80}\nCHRONOS ENTRY ANALYSIS — COMPLETE ({elapsed:.0f}s)\n{'='*80}")


if __name__ == "__main__":
    main()
