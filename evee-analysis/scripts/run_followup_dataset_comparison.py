#!/usr/bin/env python3
"""Cross-dataset latent feature comparison: extend Figure 7 to DEMETER2 and STRING.

Replicates the Figure 7 analysis pipeline (originally CORUM + Chronos) for two
additional datasets, then performs all pairwise cross-dataset comparisons.

Datasets:
  Structural (group-based):
    - CORUM: protein complex co-membership (original)
    - STRING: protein–protein interaction partners (new)
  Functional (continuous):
    - Chronos: CRISPR dependency (original)
    - DEMETER2: RNAi dependency (new)

Score definitions (consistent across dataset types):
  Structural score: n_significant_groups × mean_abs_effect
    For each gene group, Welch t-test per (i,j) entry comparing z-scored values
    of genes IN the group vs OUT. Score = recurrence × magnitude.
  Functional score: |Pearson r| between z-scored (i,j) entry and mean dependency
    across genes. Identical pipeline to run_chronos_entry_analysis.py Stage 2.

Does NOT modify original scripts.

Prereqs:
    run_corum_interpretability.py  → gene_level_matrices.npz, corum_entry_enrichment.parquet
    run_chronos_entry_analysis.py  → chronos_entry_correlations.parquet

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_followup_dataset_comparison.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproducibility import enforce_seeds, save_run_config, save_run_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
DATA_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"
OUT_DIR = DATA_DIR / "dataset_comparison"

DEMETER2_PATH = EVEE_ROOT / "data" / "RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv"
STRING_INFO = EVEE_ROOT / "data" / "9606.protein.info.v12.0.txt"
STRING_LINKS = EVEE_ROOT / "data" / "9606.protein.links.full.v12.0.txt"

RANDOM_SEED = 42
DATE_TAG = "20260409"
STRING_SCORE_THRESHOLD = 700
STRING_MIN_PARTNERS = 5
STRING_MAX_GROUPS = 500
FDR_THRESHOLD = 0.05
TOP_N_VALUES = [50, 100, 200, 410]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "CORUM": "#1b9e77",
    "STRING": "#d95f02",
    "Chronos": "#f28e2b",
    "DEMETER2": "#4e79a7",
}


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / f"{DATE_TAG}_{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved {path.name}")
    return path


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


# ═══════════════════════════════════════════════════════════════════════
# PART 1A — DEMETER2 functional score (analogous to Chronos)
# ═══════════════════════════════════════════════════════════════════════

def compute_demeter2_score(
    zscored: np.ndarray, gene_names: list[str],
) -> np.ndarray:
    """Compute per-(i,j) |Pearson r| with mean DEMETER2 dependency.

    Identical pipeline to run_chronos_entry_analysis.py Stage 2,
    but using DEMETER2 instead of Chronos.
    """
    log.info("Computing DEMETER2 importance score...")

    dem_df = pl.read_csv(str(DEMETER2_PATH))
    metadata_cols = {"depmap_id", "cell_line_display_name",
                     "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    gene_cols = [c for c in dem_df.columns if c not in metadata_cols]
    dem_genes = {g.upper(): g for g in gene_cols}
    log.info(f"  DEMETER2: {dem_df.height} cell lines, {len(gene_cols)} genes")

    gene_set = set(gene_names)
    overlap = sorted(gene_set & set(dem_genes.keys()))
    log.info(f"  Overlap with embedding genes: {len(overlap)}")

    mat = dem_df.select(gene_cols).to_numpy().astype(np.float64)
    gene_to_col = {g.upper(): i for i, g in enumerate(gene_cols)}

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    common_genes = []
    mean_deps = []
    emb_indices = []
    for gene in overlap:
        col_idx = gene_to_col[gene]
        vals = mat[:, col_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        common_genes.append(gene)
        mean_deps.append(float(np.mean(valid)))
        emb_indices.append(gene_idx[gene])

    log.info(f"  Genes with valid DEMETER2 profiles: {len(common_genes)}")

    z_sub = zscored[np.array(emb_indices)]
    mean_dep = np.array(mean_deps)
    n = len(common_genes)

    y_c = mean_dep - mean_dep.mean()
    X_c = z_sub - z_sub.mean(axis=0, keepdims=True)
    y_std = y_c.std()
    X_std = X_c.std(axis=0)
    X_std[X_std < 1e-10] = 1e-10
    pearson_r = (X_c * y_c[:, None]).mean(axis=0) / (X_std * y_std)

    t = pearson_r * np.sqrt((n - 2) / (1 - pearson_r**2 + 1e-15))
    p_vals = 2 * stats.t.sf(np.abs(t), n - 2)
    fdr = benjamini_hochberg(p_vals)

    n_sig = int((fdr < FDR_THRESHOLD).sum())
    log.info(f"  Significant entries (FDR<0.05): {n_sig}/4096")
    log.info(f"  |r| range: [{np.abs(pearson_r).min():.4f}, {np.abs(pearson_r).max():.4f}]")

    ii, jj = np.divmod(np.arange(4096), 64)
    corr_df = pl.DataFrame({
        "i": ii.astype(np.int16),
        "j": jj.astype(np.int16),
        "pearson_mean_dep": pearson_r.astype(np.float32),
        "pearson_mean_dep_p": p_vals.astype(np.float32),
        "pearson_mean_dep_fdr": fdr.astype(np.float32),
    })
    corr_df.write_parquet(OUT_DIR / "demeter2_entry_correlations.parquet")
    log.info(f"  Saved demeter2_entry_correlations.parquet")

    score = np.abs(pearson_r)
    return score


# ═══════════════════════════════════════════════════════════════════════
# PART 1B — STRING structural score (analogous to CORUM)
# ═══════════════════════════════════════════════════════════════════════

def _load_string_gene_groups(gene_names: list[str]) -> list[dict]:
    """Build STRING-based gene groups: each gene's high-confidence interaction partners.

    Analogous to CORUM complexes. For each gene with >= STRING_MIN_PARTNERS
    high-confidence (>=700) interaction partners in the embedding, create a group.
    """
    log.info("  Loading STRING data...")
    info = pl.read_csv(
        str(STRING_INFO), separator="\t", comment_prefix="#", has_header=False,
        new_columns=["string_protein_id", "preferred_name", "protein_size", "annotation"],
    ).select("string_protein_id", "preferred_name")
    prot_to_gene = {
        k: v.upper()
        for k, v in zip(info["string_protein_id"].to_list(), info["preferred_name"].to_list())
    }

    log.info("  Loading STRING links...")
    links = pl.read_csv(str(STRING_LINKS), separator=" ")
    log.info(f"    {links.height:,} raw edges")

    gene_set = set(gene_names)
    neighbors: dict[str, set[str]] = defaultdict(set)

    for row in links.filter(pl.col("combined_score") >= STRING_SCORE_THRESHOLD).iter_rows(named=True):
        g1 = prot_to_gene.get(row["protein1"])
        g2 = prot_to_gene.get(row["protein2"])
        if g1 and g2 and g1 != g2 and g1 in gene_set and g2 in gene_set:
            neighbors[g1].add(g2)
            neighbors[g2].add(g1)

    groups = []
    for gene, partners in sorted(neighbors.items()):
        if len(partners) >= STRING_MIN_PARTNERS:
            groups.append({
                "group_id": gene,
                "group_name": f"STRING partners of {gene}",
                "member_genes": sorted(partners | {gene}),
            })

    groups.sort(key=lambda g: len(g["member_genes"]), reverse=True)
    if len(groups) > STRING_MAX_GROUPS:
        groups = groups[:STRING_MAX_GROUPS]

    log.info(f"  STRING groups: {len(groups)} genes with >= {STRING_MIN_PARTNERS} partners")
    sizes = [len(g["member_genes"]) for g in groups]
    log.info(f"  Group size range: [{min(sizes)}, {max(sizes)}], median={np.median(sizes):.0f}")

    return groups


def compute_string_score(
    zscored: np.ndarray, gene_names: list[str],
) -> np.ndarray:
    """Compute per-(i,j) STRING importance: recurrence × mean |effect|.

    For each STRING gene group, run Welch t-test per (i,j) comparing genes IN
    the group vs OUT. Identical statistical procedure to CORUM entry enrichment.
    """
    log.info("Computing STRING importance score...")

    groups = _load_string_gene_groups(gene_names)
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    n_entries = 4096
    ii_template, jj_template = np.divmod(np.arange(n_entries), 64)

    sig_count = np.zeros(n_entries, dtype=np.int64)
    sig_effect_sum = np.zeros(n_entries, dtype=np.float64)
    sig_effect_count = np.zeros(n_entries, dtype=np.int64)

    for gi, group in enumerate(groups):
        members = group["member_genes"]
        in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])
        out_mask = np.ones(n_genes, dtype=bool)
        out_mask[in_idx] = False

        n_in = len(in_idx)
        n_out = int(out_mask.sum())
        if n_in < 3 or n_out < 3:
            continue

        z_in = zscored[in_idx]
        z_out = zscored[out_mask]

        mean_in = z_in.mean(axis=0)
        mean_out = z_out.mean(axis=0)
        delta = mean_in - mean_out

        var_in = z_in.var(axis=0, ddof=1)
        var_out = z_out.var(axis=0, ddof=1)

        se = np.sqrt(var_in / n_in + var_out / n_out)
        se[se < 1e-10] = 1e-10
        t_stat = delta / se

        num = (var_in / n_in + var_out / n_out) ** 2
        denom = (var_in / n_in) ** 2 / (n_in - 1) + (var_out / n_out) ** 2 / (n_out - 1)
        denom[denom < 1e-10] = 1e-10
        df_ws = np.clip(num / denom, 1.0, None)
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df_ws)

        pooled_var = ((n_in - 1) * var_in + (n_out - 1) * var_out) / (n_in + n_out - 2)
        pooled_sd = np.sqrt(np.maximum(pooled_var, 1e-20))
        cohens_d = delta / pooled_sd
        fdr = benjamini_hochberg(p_vals)

        sig_mask = fdr < FDR_THRESHOLD
        sig_count += sig_mask.astype(np.int64)
        sig_effect_sum += np.where(sig_mask, np.abs(cohens_d), 0.0)
        sig_effect_count += sig_mask.astype(np.int64)

        if (gi + 1) % 100 == 0:
            log.info(f"    [{gi+1}/{len(groups)}] groups processed")

    mean_abs_effect = np.divide(
        sig_effect_sum, sig_effect_count,
        out=np.zeros_like(sig_effect_sum),
        where=sig_effect_count > 0,
    )
    score = sig_count * mean_abs_effect

    n_nonzero = int((score > 0).sum())
    log.info(f"  STRING score: {n_nonzero} entries with score > 0")
    log.info(f"  Score range: [{score.min():.3f}, {score.max():.3f}]")

    # Save per-entry details
    entry_df = pl.DataFrame({
        "i": ii_template.astype(np.int16),
        "j": jj_template.astype(np.int16),
        "n_sig_groups": sig_count.astype(np.int32),
        "mean_abs_effect": mean_abs_effect.astype(np.float32),
        "score": score.astype(np.float32),
    })
    entry_df.write_parquet(OUT_DIR / "string_entry_enrichment.parquet")
    log.info(f"  Saved string_entry_enrichment.parquet")

    return score


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — Load existing scores (CORUM, Chronos)
# ═══════════════════════════════════════════════════════════════════════

def load_corum_score() -> np.ndarray:
    """Load precomputed CORUM score: n_sig_complexes × mean_abs_effect."""
    enrichment = pl.read_parquet(DATA_DIR / "corum_entry_enrichment.parquet")
    sig = enrichment.filter(pl.col("fdr") < FDR_THRESHOLD)
    recurrence = sig.group_by(["i", "j"]).len().rename({"len": "n_sig_complexes"})
    mean_eff = sig.group_by(["i", "j"]).agg(
        pl.col("effect_size").abs().mean().alias("mean_abs_effect")
    )
    combined = recurrence.join(mean_eff, on=["i", "j"], how="left")
    score = np.zeros(4096, dtype=np.float64)
    for r in combined.iter_rows(named=True):
        score[r["i"] * 64 + r["j"]] = r["n_sig_complexes"] * r["mean_abs_effect"]
    log.info(f"  CORUM score loaded: range [{score.min():.3f}, {score.max():.3f}]")
    return score


def load_chronos_score() -> np.ndarray:
    """Load precomputed Chronos score: |Pearson r| with mean dependency."""
    corr_df = pl.read_parquet(DATA_DIR / "chronos_entry_correlations.parquet")
    score = np.zeros(4096, dtype=np.float64)
    for r in corr_df.iter_rows(named=True):
        score[r["i"] * 64 + r["j"]] = abs(r["pearson_mean_dep"])
    log.info(f"  Chronos score loaded: range [{score.min():.4f}, {score.max():.4f}]")
    return score


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — Heatmap generation
# ═══════════════════════════════════════════════════════════════════════

def _make_feature_heatmap(
    score: np.ndarray,
    name: str,
    cmap: str,
    filename: str,
    subtitle: str,
) -> None:
    """Generate a 3-panel heatmap: original order, clustered, smoothed+clustered."""
    grid = score.reshape(64, 64)

    flat = grid.reshape(64, -1)
    dist = pdist(flat, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0)
    Z_row = linkage(dist, method="average")
    row_order = leaves_list(Z_row)

    dist_col = pdist(flat.T, metric="correlation")
    dist_col = np.nan_to_num(dist_col, nan=0.0)
    Z_col = linkage(dist_col, method="average")
    col_order = leaves_list(Z_col)

    grid_clust = grid[np.ix_(row_order, col_order)]
    grid_smooth = gaussian_filter(grid_clust, sigma=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(grid, cmap=cmap, aspect="equal", interpolation="nearest")
    fig.colorbar(im0, ax=axes[0], shrink=0.75, pad=0.02)
    axes[0].set_title("(a) Original order", fontsize=11, loc="left")
    axes[0].set_xlabel("j (column index)")
    axes[0].set_ylabel("i (row index)")

    im1 = axes[1].imshow(grid_clust, cmap=cmap, aspect="equal", interpolation="nearest")
    fig.colorbar(im1, ax=axes[1], shrink=0.75, pad=0.02)
    axes[1].set_title("(b) Hierarchically clustered", fontsize=11, loc="left")
    axes[1].set_xlabel("column (reordered)")
    axes[1].set_ylabel("row (reordered)")

    im2 = axes[2].imshow(grid_smooth, cmap=cmap, aspect="equal", interpolation="nearest")
    fig.colorbar(im2, ax=axes[2], shrink=0.75, pad=0.02)
    axes[2].set_title("(c) Clustered + smoothed (σ=1)", fontsize=11, loc="left")
    axes[2].set_xlabel("column (reordered)")
    axes[2].set_ylabel("row (reordered)")

    fig.suptitle(f"{name} latent feature importance map\n{subtitle}", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════
# PART 4 — Cross-dataset comparisons
# ═══════════════════════════════════════════════════════════════════════

def compute_pairwise_comparisons(
    scores: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Compute Spearman correlation and top-N overlap for all dataset pairs."""
    names = list(scores.keys())
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = scores[a_name], scores[b_name]

            rho, p = stats.spearmanr(a, b)

            for top_n in TOP_N_VALUES:
                top_a = set(np.argsort(-a)[:top_n].tolist())
                top_b = set(np.argsort(-b)[:top_n].tolist())
                overlap = len(top_a & top_b)
                jaccard = overlap / len(top_a | top_b) if len(top_a | top_b) > 0 else 0.0
                expected = top_n * top_n / 4096
                fold = overlap / expected if expected > 0 else 0.0

                rows.append({
                    "dataset_a": a_name,
                    "dataset_b": b_name,
                    "comparison_type": _classify_comparison(a_name, b_name),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "top_n": top_n,
                    "overlap": overlap,
                    "jaccard": float(jaccard),
                    "fold_enrichment": float(fold),
                    "expected_overlap": float(expected),
                })

    df = pl.DataFrame(rows)
    return df


def _classify_comparison(a: str, b: str) -> str:
    structural = {"CORUM", "STRING"}
    functional = {"Chronos", "DEMETER2"}
    if {a, b} <= structural:
        return "within-structural"
    elif {a, b} <= functional:
        return "within-functional"
    else:
        return "cross-type"


def plot_comparison_summary(
    scores: dict[str, np.ndarray],
    similarity_df: pl.DataFrame,
) -> None:
    """Generate the main cross-dataset comparison figure."""
    names = list(scores.keys())
    n = len(names)

    # Spearman correlation matrix
    rho_mat = np.eye(n)
    for row in similarity_df.filter(pl.col("top_n") == TOP_N_VALUES[0]).iter_rows(named=True):
        i = names.index(row["dataset_a"])
        j = names.index(row["dataset_b"])
        rho_mat[i, j] = row["spearman_rho"]
        rho_mat[j, i] = row["spearman_rho"]

    # Overlap matrix at top-200
    top_n_ref = 200
    overlap_mat = np.eye(n) * top_n_ref
    for row in similarity_df.filter(pl.col("top_n") == top_n_ref).iter_rows(named=True):
        i = names.index(row["dataset_a"])
        j = names.index(row["dataset_b"])
        overlap_mat[i, j] = row["overlap"]
        overlap_mat[j, i] = row["overlap"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Cross-dataset latent feature similarity\n"
        "Per-entry (i,j) importance scores across 4 biological datasets",
        fontsize=12, y=1.03,
    )

    colors_list = [PALETTE[n] for n in names]

    im1 = ax1.imshow(rho_mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="equal")
    fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(names, fontsize=10, rotation=30, ha="right")
    ax1.set_yticklabels(names, fontsize=10)
    for ii in range(n):
        for jj in range(n):
            txt = f"{rho_mat[ii, jj]:.3f}"
            ax1.text(jj, ii, txt, ha="center", va="center", fontsize=9,
                     fontweight="bold" if ii != jj else "normal",
                     color="white" if rho_mat[ii, jj] > 0.6 else "black")
    ax1.set_title("(a) Spearman ρ (all 4096 entries)")

    im2 = ax2.imshow(overlap_mat, cmap="YlOrRd", vmin=0, aspect="equal")
    fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(names, fontsize=10, rotation=30, ha="right")
    ax2.set_yticklabels(names, fontsize=10)
    for ii in range(n):
        for jj in range(n):
            val = int(overlap_mat[ii, jj])
            ax2.text(jj, ii, str(val), ha="center", va="center", fontsize=9,
                     fontweight="bold" if ii != jj else "normal",
                     color="white" if overlap_mat[ii, jj] > top_n_ref * 0.4 else "black")
    ax2.set_title(f"(b) Top-{top_n_ref} overlap (of {top_n_ref} each)")

    fig.tight_layout()
    _save(fig, "feature_map_comparison_heatmap")


def plot_difference_maps(scores: dict[str, np.ndarray]) -> None:
    """Generate difference heatmaps for key dataset pairs."""
    pairs = [
        ("CORUM", "STRING", "within-structural"),
        ("Chronos", "DEMETER2", "within-functional"),
        ("CORUM", "Chronos", "cross-type (original)"),
        ("STRING", "DEMETER2", "cross-type (new)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Normalized difference maps between dataset pairs\n"
        "Per-entry (i,j): (A_norm − B_norm) after min-max normalization",
        fontsize=12, y=1.02,
    )

    for ax, (a_name, b_name, label) in zip(axes.flat, pairs):
        a = scores[a_name].reshape(64, 64)
        b = scores[b_name].reshape(64, 64)
        a_norm = a / (a.max() or 1)
        b_norm = b / (b.max() or 1)
        diff = gaussian_filter(a_norm - b_norm, sigma=1.0)
        vmax = max(abs(diff.min()), abs(diff.max())) or 1
        im = ax.imshow(diff, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(f"{a_name} − {b_name}\n({label})", fontsize=10)
        ax.set_xlabel("j")
        ax.set_ylabel("i")

    fig.tight_layout()
    _save(fig, "feature_map_difference_maps")


def plot_all_four_heatmaps(scores: dict[str, np.ndarray]) -> None:
    """Side-by-side smoothed+clustered heatmaps for all 4 datasets."""
    # Joint clustering using all 4 score maps
    grids = {name: score.reshape(64, 64) for name, score in scores.items()}
    combined_flat = np.hstack([g.reshape(64, -1) for g in grids.values()])

    dist_row = pdist(combined_flat, metric="correlation")
    dist_row = np.nan_to_num(dist_row, nan=0.0)
    Z_row = linkage(dist_row, method="average")
    row_order = leaves_list(Z_row)

    dist_col = pdist(combined_flat.T[:64], metric="correlation")
    dist_col = np.nan_to_num(dist_col, nan=0.0)
    Z_col = linkage(dist_col, method="average")
    col_order = leaves_list(Z_col)

    cmaps = {"CORUM": "YlOrRd", "STRING": "YlOrRd", "Chronos": "YlGnBu", "DEMETER2": "YlGnBu"}

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "Latent feature importance across 4 datasets (joint clustering)\n"
        "Per-entry (i,j) | smoothed (σ=1) | rows and columns hierarchically clustered together",
        fontsize=12, y=1.02,
    )

    for ax, (name, grid) in zip(axes, grids.items()):
        reordered = grid[np.ix_(row_order, col_order)]
        smoothed = gaussian_filter(reordered, sigma=1.0)
        im = ax.imshow(smoothed, cmap=cmaps[name], aspect="equal", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(name, fontsize=12, fontweight="bold", color=PALETTE[name])
        ax.set_xlabel("column (reordered)")
        ax.set_ylabel("row (reordered)")

    fig.tight_layout()
    _save(fig, "feature_map_all_four_datasets")


# ═══════════════════════════════════════════════════════════════════════
# PART 5 — Summary report
# ═══════════════════════════════════════════════════════════════════════

def write_report(
    scores: dict[str, np.ndarray],
    similarity_df: pl.DataFrame,
    gene_universe_info: dict,
) -> None:
    """Write a concise markdown summary."""
    lines = [
        "# Cross-Dataset Latent Feature Comparison — Summary Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Datasets and Score Definitions",
        "",
        "| Dataset | Type | Score definition | Gene universe |",
        "|---------|------|-----------------|---------------|",
    ]

    score_defs = {
        "CORUM": "n_sig_complexes × mean |effect size| (Welch t-test, FDR<0.05)",
        "STRING": "n_sig_groups × mean |effect size| (Welch t-test, FDR<0.05)",
        "Chronos": "|Pearson r| with mean CRISPR dependency across genes",
        "DEMETER2": "|Pearson r| with mean RNAi dependency across genes",
    }
    types = {"CORUM": "Structural", "STRING": "Structural", "Chronos": "Functional", "DEMETER2": "Functional"}

    for name in scores:
        gu = gene_universe_info.get(name, "N/A")
        lines.append(f"| {name} | {types[name]} | {score_defs[name]} | {gu} |")

    lines.extend([
        "",
        "## 2. Score Distributions",
        "",
        "| Dataset | Min | Median | Max | Non-zero |",
        "|---------|-----|--------|-----|----------|",
    ])
    for name, s in scores.items():
        lines.append(f"| {name} | {s.min():.4f} | {np.median(s):.4f} | {s.max():.4f} | {int((s > 0).sum())} |")

    lines.extend([
        "",
        "## 3. Global Similarity (Spearman ρ)",
        "",
        "| Dataset A | Dataset B | Type | Spearman ρ | p-value |",
        "|-----------|-----------|------|------------|---------|",
    ])
    seen = set()
    for row in similarity_df.filter(pl.col("top_n") == TOP_N_VALUES[0]).sort("spearman_rho", descending=True).iter_rows(named=True):
        pair = (row["dataset_a"], row["dataset_b"])
        if pair not in seen:
            seen.add(pair)
            lines.append(f"| {row['dataset_a']} | {row['dataset_b']} | {row['comparison_type']} | "
                         f"{row['spearman_rho']:.4f} | {row['spearman_p']:.2e} |")

    lines.extend([
        "",
        "## 4. Top-N Overlap",
        "",
        "| Dataset A | Dataset B | Top-N | Overlap | Expected | Fold |",
        "|-----------|-----------|-------|---------|----------|------|",
    ])
    for row in similarity_df.filter(pl.col("top_n") == 200).sort("fold_enrichment", descending=True).iter_rows(named=True):
        lines.append(f"| {row['dataset_a']} | {row['dataset_b']} | {row['top_n']} | "
                     f"{row['overlap']} | {row['expected_overlap']:.1f} | {row['fold_enrichment']:.2f}× |")

    lines.extend([
        "",
        "## 5. Key Findings",
        "",
    ])

    # Auto-generate findings from data
    spearman_rows = similarity_df.filter(pl.col("top_n") == TOP_N_VALUES[0]).sort("spearman_rho", descending=True)
    within_struct = spearman_rows.filter(pl.col("comparison_type") == "within-structural")
    within_func = spearman_rows.filter(pl.col("comparison_type") == "within-functional")
    cross_type = spearman_rows.filter(pl.col("comparison_type") == "cross-type")

    if within_struct.height > 0:
        r = within_struct.row(0, named=True)
        qual = "similar" if r["spearman_rho"] > 0.5 else "partially overlapping" if r["spearman_rho"] > 0.2 else "largely distinct"
        lines.append(f"1. **Within-structural ({r['dataset_a']} vs {r['dataset_b']}):** "
                     f"ρ = {r['spearman_rho']:.4f}. "
                     f"CORUM and STRING capture {qual} latent feature patterns. "
                     f"**Caveat:** STRING groups are ego-networks (a gene plus its high-confidence "
                     f"interaction partners), structurally different from CORUM's curated multi-gene "
                     f"complexes. This comparison measures whether the same matrix entries are enriched "
                     f"in both contexts, not whether the group definitions are equivalent.")

    if within_func.height > 0:
        r = within_func.row(0, named=True)
        qual = "high" if r["spearman_rho"] > 0.5 else "moderate" if r["spearman_rho"] > 0.2 else "low"
        lines.append(f"2. **Within-functional ({r['dataset_a']} vs {r['dataset_b']}):** "
                     f"ρ = {r['spearman_rho']:.4f}. "
                     f"Chronos and DEMETER2 show {qual} agreement in which latent features track with dependency.")

    if cross_type.height > 0:
        rho_vals = [r["spearman_rho"] for r in cross_type.iter_rows(named=True)]
        lines.append(f"3. **Cross-type (structural vs functional):** "
                     f"ρ range = [{min(rho_vals):.4f}, {max(rho_vals):.4f}]. "
                     "Structural and functional signals are encoded in "
                     f"{'overlapping' if max(rho_vals) > 0.3 else 'partially distinct'} "
                     "regions of the embedding matrix.")

    lines.extend([
        "",
        "## 6. Output Files",
        "",
        "### Data",
        "- `dataset_comparison/feature_map_similarity.parquet`",
        "- `dataset_comparison/demeter2_entry_correlations.parquet`",
        "- `dataset_comparison/string_entry_enrichment.parquet`",
        "- `dataset_comparison/all_scores.parquet`",
        "",
        "### Figures",
        f"- `{DATE_TAG}_demeter2_feature_map.png`",
        f"- `{DATE_TAG}_stringdb_feature_map.png`",
        f"- `{DATE_TAG}_feature_map_comparison_heatmap.png`",
        f"- `{DATE_TAG}_feature_map_difference_maps.png`",
        f"- `{DATE_TAG}_feature_map_all_four_datasets.png`",
    ])

    report_path = OUT_DIR / "dataset_comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {report_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    log.info("=" * 70)
    log.info("Cross-Dataset Latent Feature Comparison")
    log.info("=" * 70)

    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    save_run_config(OUT_DIR, {
        "random_seed": RANDOM_SEED,
        "string_score_threshold": STRING_SCORE_THRESHOLD,
        "string_min_partners": STRING_MIN_PARTNERS,
        "string_max_groups": STRING_MAX_GROUPS,
        "fdr_threshold": FDR_THRESHOLD,
        "top_n_values": TOP_N_VALUES,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
    })
    save_run_manifest(OUT_DIR)

    # Load shared gene-level z-scored matrices
    log.info("Loading gene-level matrices...")
    npz = np.load(str(DATA_DIR / "gene_level_matrices.npz"), allow_pickle=True)
    gene_names = list(npz["gene_names"])
    zscored = npz["zscored_flat"]
    log.info(f"  {len(gene_names):,} genes × {zscored.shape[1]} features")

    gene_universe_info = {}

    # ── Load existing scores ──
    log.info("\n── Loading existing scores ──")
    corum_score = load_corum_score()
    chronos_score = load_chronos_score()
    gene_universe_info["CORUM"] = f"{len(gene_names)} embedded genes"
    gene_universe_info["Chronos"] = f"{len(gene_names)} embedded genes (subset with Chronos data)"

    # ── Compute new scores ──
    log.info("\n── Computing DEMETER2 score ──")
    demeter2_score = compute_demeter2_score(zscored, gene_names)
    gene_universe_info["DEMETER2"] = f"{len(gene_names)} embedded genes (subset with DEMETER2 data)"

    log.info("\n── Computing STRING score ──")
    string_score = compute_string_score(zscored, gene_names)
    gene_universe_info["STRING"] = f"{len(gene_names)} embedded genes (STRING ≥ {STRING_SCORE_THRESHOLD})"

    scores = {
        "CORUM": corum_score,
        "STRING": string_score,
        "Chronos": chronos_score,
        "DEMETER2": demeter2_score,
    }

    # Save all scores together
    ii, jj = np.divmod(np.arange(4096), 64)
    all_scores_df = pl.DataFrame({
        "i": ii.astype(np.int16),
        "j": jj.astype(np.int16),
        "corum_score": corum_score.astype(np.float32),
        "string_score": string_score.astype(np.float32),
        "chronos_score": chronos_score.astype(np.float32),
        "demeter2_score": demeter2_score.astype(np.float32),
    })
    all_scores_df.write_parquet(OUT_DIR / "all_scores.parquet")
    log.info("Saved all_scores.parquet")

    # ── Generate individual heatmaps ──
    log.info("\n── Generating heatmaps ──")
    _make_feature_heatmap(
        demeter2_score, "DEMETER2", "YlGnBu", "demeter2_feature_map",
        "Per-entry |Pearson r| with mean RNAi dependency across genes",
    )
    _make_feature_heatmap(
        string_score, "STRING", "YlOrRd", "stringdb_feature_map",
        f"Per-entry structural enrichment (Welch t-test, FDR<0.05, score threshold ≥ {STRING_SCORE_THRESHOLD})",
    )

    # ── Cross-dataset comparisons ──
    log.info("\n── Cross-dataset comparisons ──")
    similarity_df = compute_pairwise_comparisons(scores)
    similarity_df.write_parquet(OUT_DIR / "feature_map_similarity.parquet")
    log.info("Saved feature_map_similarity.parquet")

    log.info("\nPairwise Spearman correlations:")
    for row in similarity_df.filter(pl.col("top_n") == TOP_N_VALUES[0]).iter_rows(named=True):
        log.info(f"  {row['dataset_a']:>8s} vs {row['dataset_b']:<8s}: "
                 f"ρ = {row['spearman_rho']:.4f} ({row['comparison_type']})")

    log.info("\nTop-200 overlaps:")
    for row in similarity_df.filter(pl.col("top_n") == 200).iter_rows(named=True):
        log.info(f"  {row['dataset_a']:>8s} vs {row['dataset_b']:<8s}: "
                 f"{row['overlap']} / 200 ({row['fold_enrichment']:.2f}× expected)")

    # ── Figures ──
    log.info("\n── Generating comparison figures ──")
    plot_comparison_summary(scores, similarity_df)
    plot_difference_maps(scores)
    plot_all_four_heatmaps(scores)

    # ── Report ──
    log.info("\n── Writing report ──")
    write_report(scores, similarity_df, gene_universe_info)

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
