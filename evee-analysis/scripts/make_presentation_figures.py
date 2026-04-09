#!/usr/bin/env python3
"""Generate polished presentation figures from existing analysis outputs.

All data is read from evee-analysis/data/intermediate/ — no recomputation.
Figures are saved to evee-analysis/outputs/figures/ with 20260409_ prefix.

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/make_presentation_figures.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "evee-analysis" / "data" / "intermediate"
FIG_DIR = REPO_ROOT / "evee-analysis" / "outputs" / "figures"
PREFIX = "20260409_"

# ── Global palette (consistent across all figures for <10 classes) ────

PALETTE = {
    # Structural ground truths (Figures 1, 6)
    "CORUM": "#1b9e77",
    "STRING": "#d95f02",
    "Gene families": "#7570b3",
    # Dependency datasets (Figures 2, 3)
    "DEMETER2": "#4e79a7",
    "Chronos": "#f28e2b",
    # Feature classes (Figures 5, 7)
    "shared": "#8c564b",
    "corum_only": "#1b9e77",
    "chronos_only": "#f28e2b",
    "background": "#bdbdbd",
    "full_4096": "#636363",
}

MARKERS = {"CORUM": "o", "STRING": "s", "Gene families": "D"}
DEP_MARKERS = {"DEMETER2": "o", "Chronos": "s"}

# ── Global rcParams ───────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{PREFIX}{name}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {path.name}")


# ── Figure 1: Structural Enrichment Summary ───────────────────────────

def fig1_structural_enrichment() -> None:
    log.info("Figure 1: Structural enrichment summary")

    corum = pl.read_parquet(DATA_DIR / "corum_full_enrichment_vs_k.parquet")
    string = pl.read_parquet(DATA_DIR / "string_enrichment_vs_k.parquet")
    gf = pl.read_parquet(DATA_DIR / "gene_family_enrichment_vs_k.parquet")

    fig, (ax_raw, ax_fold) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle(
        "Embedding neighbors are enriched for known biological relationships\n"
        "kNN on full 64×64 embedding (cosine distance, cross-gene only)",
        fontsize=12, y=1.02,
    )

    datasets_fold = [
        ("CORUM",        corum, "fold_enrichment", "fold_ci_lo", "fold_ci_hi", 3519),
        ("STRING",       string, "fold_enrichment", "fold_ci_lo", "fold_ci_hi", 12759),
        ("Gene families", gf,   "fold_enrichment", "fold_ci_lo", "fold_ci_hi", 8854),
    ]

    # nb_frac / rd_frac columns differ slightly between datasets
    datasets_raw = [
        ("CORUM",         corum, "nb_frac",            "rd_frac",            3519),
        ("STRING",        string, "nb_frac_in_string",  "rd_frac_in_string",  12759),
        ("Gene families", gf,    "nb_frac",            "rd_frac",            8854),
    ]

    # ── Left panel: actual fractions ──
    for label, df, nb_col, rd_col, n_genes in datasets_raw:
        k = df["k"].to_numpy()
        nb_frac = df[nb_col].to_numpy()
        rd_frac = df[rd_col].to_numpy()
        ax_raw.plot(
            k, nb_frac, f"{MARKERS[label]}-",
            color=PALETTE[label], linewidth=2.0, markersize=7,
            label=f"{label} neighbor (n={n_genes:,})",
        )
        ax_raw.plot(
            k, rd_frac, f"{MARKERS[label]}--",
            color=PALETTE[label], linewidth=1.2, markersize=5, alpha=0.5,
            label=f"{label} random",
        )
        ax_raw.annotate(
            f"{nb_frac[0]:.1%}", (k[0], nb_frac[0]),
            textcoords="offset points", xytext=(-8, 10),
            fontsize=8, fontweight="bold", color=PALETTE[label],
        )

    ax_raw.set_xlabel("k (number of neighbors)")
    ax_raw.set_ylabel("Fraction of pairs sharing\na known relationship")
    ax_raw.set_xticks([5, 10, 20, 50])
    ax_raw.set_title("Absolute rates (neighbor vs random)")
    ax_raw.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_raw.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.28),
                  framealpha=0.9, ncol=3)

    # ── Right panel: fold enrichment ──
    for label, df, col, lo, hi, n_genes in datasets_fold:
        k = df["k"].to_numpy()
        y = df[col].to_numpy()
        ci_lo = df[lo].to_numpy()
        ci_hi = df[hi].to_numpy()
        yerr = np.array([y - ci_lo, ci_hi - y])
        ax_fold.errorbar(
            k, y, yerr=yerr, fmt=f"{MARKERS[label]}-",
            color=PALETTE[label], linewidth=2.0, markersize=7,
            capsize=5, capthick=1.5,
            label=f"{label} (n={n_genes:,})",
        )
        ax_fold.annotate(
            f"{y[0]:.2f}×", (k[0], y[0]),
            textcoords="offset points", xytext=(-8, 12),
            fontsize=9, fontweight="bold", color=PALETTE[label],
        )

    ax_fold.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax_fold.text(52, 1.03, "No enrichment", fontsize=8, color="gray", va="bottom")

    ax_fold.set_xlabel("k (number of neighbors)")
    ax_fold.set_ylabel("Fold enrichment\n(neighbor / random)")
    ax_fold.set_xticks([5, 10, 20, 50])
    ax_fold.set_title("Fold enrichment over random baseline")
    ax_fold.legend(loc="upper right", framealpha=0.9)
    ax_fold.set_ylim(bottom=0.8)

    fig.tight_layout()
    _save(fig, "fig1_structural_enrichment.png")


# ── Figure 2: Dependency Profile Similarity ───────────────────────────

def fig2_dependency_delta() -> None:
    log.info("Figure 2: Dependency delta (DEMETER2 vs Chronos)")

    data = {
        "DEMETER2": {
            "delta": 0.0016, "ci_lo": 0.0013, "ci_hi": 0.0020,
            "nb_mean": 0.0109, "rd_mean": 0.0088,
            "n_genes": 10963, "cells": 707,
        },
        "Chronos": {
            "delta": 0.0053, "ci_lo": 0.0049, "ci_hi": 0.0057,
            "nb_mean": 0.0083, "rd_mean": 0.0020,
            "n_genes": 12434, "cells": 551,
        },
    }

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle(
        "Embedding neighbors share cancer dependency profiles\n"
        "kNN on full 64×64 embedding | cross-gene pairs vs matched random",
        fontsize=11, y=1.02,
    )

    # ── Left panel: absolute mean correlations ──
    bar_width = 0.3
    x_abs = np.array([0, 1])
    for i, (name, d) in enumerate(data.items()):
        ax_abs.bar(
            x_abs[i] - bar_width / 2, d["nb_mean"], width=bar_width,
            color=PALETTE[name], edgecolor="black", linewidth=0.6,
            label="Neighbor" if i == 0 else None,
        )
        ax_abs.bar(
            x_abs[i] + bar_width / 2, d["rd_mean"], width=bar_width,
            color=PALETTE[name], edgecolor="black", linewidth=0.6,
            alpha=0.35, label="Random" if i == 0 else None,
        )
        ax_abs.text(
            x_abs[i] - bar_width / 2, d["nb_mean"] + 0.0003,
            f"{d['nb_mean']:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
        ax_abs.text(
            x_abs[i] + bar_width / 2, d["rd_mean"] + 0.0003,
            f"{d['rd_mean']:.4f}", ha="center", va="bottom", fontsize=8, color="#555555",
        )

    ax_abs.set_xticks(x_abs)
    ax_abs.set_xticklabels([
        f"DEMETER2\n({data['DEMETER2']['cells']} cell lines)",
        f"Chronos\n({data['Chronos']['cells']} cell lines)",
    ], fontsize=9)
    ax_abs.set_ylabel("Mean profile correlation")
    ax_abs.set_title("Absolute values")
    ax_abs.legend(frameon=False, fontsize=9)
    ax_abs.axhline(0, color="black", linewidth=0.5)
    ax_abs.set_ylim(bottom=-0.001, top=0.016)

    # ── Right panel: delta with CI ──
    x_delta = np.arange(len(data))
    for i, (name, d) in enumerate(data.items()):
        yerr = np.array([[d["delta"] - d["ci_lo"]], [d["ci_hi"] - d["delta"]]])
        ax_delta.bar(
            i, d["delta"], width=0.55, color=PALETTE[name],
            edgecolor="black", linewidth=0.6, yerr=yerr,
            capsize=8, error_kw={"linewidth": 1.5},
        )
        ax_delta.text(
            i, d["ci_hi"] + 0.0002,
            f"Δ = {d['delta']:.4f}\n[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]",
            ha="center", va="bottom", fontsize=9,
        )

    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(x_delta)
    ax_delta.set_xticklabels([
        f"DEMETER2\n(n={data['DEMETER2']['n_genes']:,} genes)",
        f"Chronos\n(n={data['Chronos']['n_genes']:,} genes)",
    ], fontsize=9)
    ax_delta.set_ylabel("Δ mean profile correlation\n(neighbor − random)")
    ax_delta.set_title("Effect size (delta)")
    ax_delta.set_ylim(bottom=-0.0005, top=0.0075)

    fig.text(
        0.5, -0.03,
        "Gene-level mean; 95% CI from 5,000 gene-level bootstrap iterations. "
        "Correlations are small in absolute terms — the signal is a consistent shift, not a strong per-pair effect.",
        ha="center", fontsize=8, color="gray",
    )

    fig.tight_layout()
    _save(fig, "fig2_dependency_delta.png")


# ── Figure 3: Delta vs k Decay ────────────────────────────────────────

def fig3_delta_vs_k() -> None:
    log.info("Figure 3: Delta vs k decay")

    df = pl.read_parquet(DATA_DIR / "followup_delta_vs_k.parquet")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-point annotation offsets: (x_offset, y_offset) in points
    annot_offsets = {
        "DEMETER2": {5: (12, 10), 10: (12, 8), 20: (12, 8), 50: (-60, 10)},
        "Chronos":  {5: (12, -24), 10: (12, 8), 20: (12, -24), 50: (-60, 8)},
    }

    for dataset in ["DEMETER2", "Chronos"]:
        sub = df.filter(pl.col("dataset") == dataset)
        k = sub["k"].to_numpy()
        delta = sub["delta"].to_numpy()
        n_pairs = sub["n_neighbor_pairs"].to_numpy()

        ax.plot(
            k, delta, f"{DEP_MARKERS[dataset]}-",
            color=PALETTE[dataset], linewidth=2.0, markersize=7,
            label=dataset,
        )
        offsets = annot_offsets[dataset]
        for ki, di, ni in zip(k, delta, n_pairs):
            xo, yo = offsets[int(ki)]
            ax.annotate(
                f"{di:.4f}\n({ni/1e6:.1f}M pairs)",
                (ki, di), textcoords="offset points",
                xytext=(xo, yo),
                fontsize=7.5, color=PALETTE[dataset],
            )

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Δ mean profile correlation\n(neighbor − random)")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_ylim(bottom=0.0015, top=0.0068)
    ax.set_title(
        "Signal decays with distance — closest neighbors are most biologically similar\n"
        "kNN on full 64×64 embedding | dependency profile correlation delta vs k",
        fontsize=11,
    )
    ax.legend(loc="center right", framealpha=0.9)

    _save(fig, "fig3_delta_vs_k.png")


# ── Figure 4: Feature Interpretability ─────────────────────────────────

def fig4_feature_interpretability() -> None:
    log.info("Figure 4: Feature interpretability (recurrent entries + example complex)")

    recurrent = pl.read_parquet(DATA_DIR / "corum_recurrent_entries.parquet")
    top_entries = pl.read_parquet(DATA_DIR / "corum_complex_top_entries.parquet")

    grid = np.zeros((64, 64))
    for row in recurrent.iter_rows(named=True):
        grid[row["i"], row["j"]] = row["n_complexes"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel (a): Recurrent entry heatmap
    ax = axes[0]
    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("# complexes (FDR < 0.05)", fontsize=10)
    ax.set_xlabel("j (right feature)", fontsize=11)
    ax.set_ylabel("i (left feature)", fontsize=11)
    ax.set_title("(a)  Recurrent features across 271 complexes", fontsize=11, loc="left")

    # Panel (b): Example complex — BAF complex delta heatmap
    baf = top_entries.filter(pl.col("complex_name") == "BAF complex")
    delta_grid = np.zeros((64, 64))
    for row in baf.iter_rows(named=True):
        delta_grid[row["i"], row["j"]] = row["delta"]

    ax2 = axes[1]
    ax2.set_facecolor("#f5f5f5")
    vmax = max(abs(delta_grid.min()), abs(delta_grid.max())) or 1.0
    im2 = ax2.imshow(delta_grid, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax, interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label("Δ z-score (IN − OUT)", fontsize=10)
    ax2.set_xlabel("j (right feature)", fontsize=11)
    ax2.set_ylabel("i (left feature)", fontsize=11)
    ax2.set_title("(b)  BAF complex — top significant entries", fontsize=11, loc="left")

    fig.suptitle(
        "Individual matrix features distinguish specific protein complexes\n"
        "Per-entry (i,j) analysis | Welch t-test, FDR < 0.05",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig4_feature_interpretability.png")


# ── Figure 5: Feature Overlap + Cross-Prediction ──────────────────────

def fig5_feature_overlap() -> None:
    log.info("Figure 5: Feature overlap (CORUM vs Chronos)")

    fc = pl.read_parquet(DATA_DIR / "feature_classes.parquet")
    cp = pl.read_parquet(DATA_DIR / "cross_prediction_results.parquet")

    counts = fc["feature_class"].value_counts().sort("count", descending=True)
    class_order = ["shared", "corum_only", "chronos_only", "background"]
    class_labels = {
        "shared": f"Shared\n(n=225)",
        "corum_only": f"CORUM-only\n(n=185)",
        "chronos_only": f"Chronos-only\n(n=185)",
        "background": f"Background\n(n=3,501)",
    }
    count_map = dict(zip(counts["feature_class"].to_list(), counts["count"].to_list()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel (a): Feature class counts
    for i, cls in enumerate(class_order):
        c = count_map.get(cls, 0)
        ax1.barh(i, c, color=PALETTE.get(cls, "#999"), edgecolor="black", linewidth=0.5, height=0.6)
        ax1.text(c + 30, i, f"{c:,}", va="center", fontsize=10)

    ax1.set_yticks(range(len(class_order)))
    ax1.set_yticklabels([class_labels[c] for c in class_order], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of (i,j) features")
    ax1.set_title("(a)  Feature classification", fontsize=11, loc="left")
    ax1.set_xlim(right=4200)

    # Panel (b): Cross-prediction R²
    show_sets = ["full_4096", "shared", "corum_only", "chronos_only"]
    r2_labels = {
        "full_4096": "All 4,096\nfeatures",
        "shared": "225 shared",
        "corum_only": "185 CORUM-\nonly",
        "chronos_only": "185 Chronos-\nonly",
    }
    cp_dict = dict(zip(cp["feature_set"].to_list(), cp["r2_chronos"].to_list()))
    nf_dict = dict(zip(cp["feature_set"].to_list(), cp["n_features"].to_list()))

    x = np.arange(len(show_sets))
    for i, fs in enumerate(show_sets):
        r2 = cp_dict[fs]
        ax2.bar(
            i, r2, width=0.6,
            color=PALETTE.get(fs, "#999"),
            edgecolor="black", linewidth=0.5,
        )
        ax2.text(i, r2 + 0.003, f"R² = {r2:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([r2_labels[s] for s in show_sets], fontsize=9)
    ax2.set_ylabel("R² (5-fold CV, predicting Chronos mean dependency)")
    ax2.set_title("(b)  Cross-predictive power", fontsize=11, loc="left")
    ax2.set_ylim(top=0.21)

    fig.suptitle(
        "Structural and functional signals share latent features\n"
        "Per-entry (i,j) analysis | CORUM enrichment score vs Chronos correlation",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig5_feature_overlap.png")


# ── Figure 6: UMAP ────────────────────────────────────────────────────

def fig6_umap() -> None:
    log.info("Figure 6: UMAP gene families (polished)")

    df = pl.read_parquet(DATA_DIR / "umap_gene_family_labels.parquet")

    top_families = [
        "Zinc fingers C2H2-type",
        "KRAB domain containing",
        "CD molecules",
        "Flavoproteins",
        "WD repeat domain containing",
        "Ring finger proteins",
        "EF-hand domain containing",
    ]

    family_colors = {
        "Zinc fingers C2H2-type": "#e41a1c",
        "KRAB domain containing": "#377eb8",
        "CD molecules": "#4daf4a",
        "Flavoproteins": "#984ea3",
        "WD repeat domain containing": "#ff7f00",
        "Ring finger proteins": "#a65628",
        "EF-hand domain containing": "#f781bf",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    bg = df.filter(~pl.col("gene_class").is_in(top_families))
    ax.scatter(
        bg["umap_1"].to_numpy(), bg["umap_2"].to_numpy(),
        s=0.3, c="#cccccc", alpha=0.06, rasterized=True,
    )

    for fam in top_families:
        sub = df.filter(pl.col("gene_class") == fam)
        if sub.height == 0:
            continue
        ax.scatter(
            sub["umap_1"].to_numpy(), sub["umap_2"].to_numpy(),
            s=3, c=family_colors[fam], alpha=0.6, label=f"{fam} ({sub.height:,})",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        "Embedding space clusters genes by family\n"
        "UMAP of gene-level full 64×64 embeddings",
        fontsize=12,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9,
        markerscale=4, framealpha=0.9,
    )
    ax.grid(False)
    ax.set_aspect("equal")

    _save(fig, "fig6_umap.png")


# ── Figure 7: Feature Block Maps ──────────────────────────────────────

def fig7_feature_maps() -> None:
    log.info("Figure 7: Feature block maps (CORUM vs Chronos)")

    fc = pl.read_parquet(DATA_DIR / "feature_classes.parquet")

    corum_grid = np.zeros((64, 64))
    chronos_grid = np.zeros((64, 64))
    for row in fc.iter_rows(named=True):
        corum_grid[row["i"], row["j"]] = row["corum_score"]
        chronos_grid[row["i"], row["j"]] = row["chronos_score"]

    from scipy.ndimage import gaussian_filter
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    flat_c = corum_grid.reshape(64, -1)
    flat_ch = chronos_grid.reshape(64, -1)
    combined = np.hstack([flat_c, flat_ch])

    dist = pdist(combined, metric="correlation")
    Z = linkage(dist, method="average")
    row_order = leaves_list(Z)

    dist_col = pdist(combined.T[:64], metric="correlation")
    Z_col = linkage(dist_col, method="average")
    col_order = leaves_list(Z_col)[:64]

    corum_r = corum_grid[np.ix_(row_order, col_order)]
    chronos_r = chronos_grid[np.ix_(row_order, col_order)]

    sigma = 1.0
    corum_s = gaussian_filter(corum_r, sigma=sigma)
    chronos_s = gaussian_filter(chronos_r, sigma=sigma)

    c_norm = corum_s / (corum_s.max() or 1)
    ch_norm = chronos_s / (chronos_s.max() or 1)
    diff = c_norm - ch_norm

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    panels = [
        ("(a)  CORUM importance", corum_s, "YlOrRd"),
        ("(b)  Chronos importance", chronos_s, "YlGnBu"),
        ("(c)  Difference (CORUM − Chronos, normalized)", diff, "RdBu_r"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        kwargs = {}
        if cmap == "RdBu_r":
            vmax = max(abs(data.min()), abs(data.max())) or 1
            kwargs = {"vmin": -vmax, "vmax": vmax}
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", **kwargs)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_xlabel("column (reordered)", fontsize=10)
        ax.set_ylabel("row (reordered)", fontsize=10)
        ax.set_title(title, fontsize=11, loc="left")

    fig.suptitle(
        "Evo2 learns distinct feature blocks for complex membership and gene essentiality\n"
        "Per-entry (i,j) importance | hierarchically clustered rows and columns",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig7_feature_maps.png")


def fig7b_feature_maps_original() -> None:
    """Same as Figure 7 but with original matrix row/column ordering (no clustering)."""
    log.info("Figure 7b: Feature maps — original matrix order")

    fc = pl.read_parquet(DATA_DIR / "feature_classes.parquet")

    corum_grid = np.zeros((64, 64))
    chronos_grid = np.zeros((64, 64))
    for row in fc.iter_rows(named=True):
        corum_grid[row["i"], row["j"]] = row["corum_score"]
        chronos_grid[row["i"], row["j"]] = row["chronos_score"]

    from scipy.ndimage import gaussian_filter

    sigma = 1.0
    corum_s = gaussian_filter(corum_grid, sigma=sigma)
    chronos_s = gaussian_filter(chronos_grid, sigma=sigma)

    c_norm = corum_s / (corum_s.max() or 1)
    ch_norm = chronos_s / (chronos_s.max() or 1)
    diff = c_norm - ch_norm

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    panels = [
        ("(a)  CORUM importance", corum_s, "YlOrRd"),
        ("(b)  Chronos importance", chronos_s, "YlGnBu"),
        ("(c)  Difference (CORUM − Chronos, normalized)", diff, "RdBu_r"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels):
        kwargs = {}
        if cmap == "RdBu_r":
            vmax = max(abs(data.min()), abs(data.max())) or 1
            kwargs = {"vmin": -vmax, "vmax": vmax}
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", **kwargs)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_xlabel("j (column index)", fontsize=10)
        ax.set_ylabel("i (row index)", fontsize=10)
        ax.set_title(title, fontsize=11, loc="left")

    fig.suptitle(
        "Evo2 learns distinct feature blocks for complex membership and gene essentiality\n"
        "Per-entry (i,j) importance | original matrix order (no clustering)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig7b_feature_maps_original.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Generating presentation figures...")

    fig1_structural_enrichment()
    fig2_dependency_delta()
    fig3_delta_vs_k()
    fig4_feature_interpretability()
    fig5_feature_overlap()
    fig6_umap()
    fig7_feature_maps()
    fig7b_feature_maps_original()

    log.info("Done — all figures saved to %s", FIG_DIR)


if __name__ == "__main__":
    main()
