#!/usr/bin/env python3
"""Figure 7b variant: CORUM absolute recurrence counts vs Chronos importance.

Left panel shows raw count of significant CORUM complexes per (i,j) entry
(FDR < 0.05), with NO Gaussian blur and NO composite scoring.

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/plot_fig7b_absolute_counts.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "evee-analysis" / "data" / "intermediate"
FIG_DIR = REPO_ROOT / "evee-analysis" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = "20260409_"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main() -> None:
    log.info("Building Figure 7b (absolute counts, no blur)...")

    # ── CORUM: raw recurrence counts ──────────────────────────────────
    enrichment = pl.read_parquet(DATA_DIR / "corum_entry_enrichment.parquet")
    sig = enrichment.filter(pl.col("fdr") < 0.05)
    recurrence = sig.group_by(["i", "j"]).len().rename({"len": "n_sig_complexes"})

    corum_counts = np.zeros((64, 64), dtype=np.float64)
    for row in recurrence.iter_rows(named=True):
        corum_counts[row["i"], row["j"]] = row["n_sig_complexes"]

    log.info(
        f"  CORUM counts: nonzero={np.count_nonzero(corum_counts)}, "
        f"max={corum_counts.max():.0f}, mean(nonzero)={corum_counts[corum_counts > 0].mean():.1f}"
    )

    # ── Chronos: |Pearson r| (same as original fig7b) ────────────────
    fc = pl.read_parquet(DATA_DIR / "feature_classes.parquet")
    chronos_grid = np.zeros((64, 64), dtype=np.float64)
    for row in fc.iter_rows(named=True):
        chronos_grid[row["i"], row["j"]] = row["chronos_score"]

    # ── Difference: normalize each to [0, 1] then subtract ───────────
    c_norm = corum_counts / (corum_counts.max() or 1)
    ch_norm = chronos_grid / (chronos_grid.max() or 1)
    diff = c_norm - ch_norm

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Left: CORUM absolute recurrence count
    im0 = axes[0].imshow(
        corum_counts, cmap="YlOrRd", aspect="equal", interpolation="nearest",
    )
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.75, pad=0.02)
    cb0.set_label("# significant complexes (FDR < 0.05)", fontsize=9)
    axes[0].set_title("(a)  CORUM recurrence count", fontsize=11, loc="left")
    axes[0].set_xlabel("j (column index)", fontsize=10)
    axes[0].set_ylabel("i (row index)", fontsize=10)

    # Middle: Chronos |Pearson r| (no blur)
    im1 = axes[1].imshow(
        chronos_grid, cmap="YlGnBu", aspect="equal", interpolation="nearest",
    )
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.75, pad=0.02)
    cb1.set_label("|Pearson r| with mean dependency", fontsize=9)
    axes[1].set_title("(b)  Chronos importance", fontsize=11, loc="left")
    axes[1].set_xlabel("j (column index)", fontsize=10)
    axes[1].set_ylabel("i (row index)", fontsize=10)

    # Right: Difference (CORUM − Chronos, each normalized to [0, 1])
    vmax = max(abs(diff.min()), abs(diff.max())) or 1
    im2 = axes[2].imshow(
        diff, cmap="RdBu_r", aspect="equal", interpolation="nearest",
        vmin=-vmax, vmax=vmax,
    )
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.75, pad=0.02)
    cb2.set_label("normalized difference", fontsize=9)
    axes[2].set_title(
        "(c)  Difference (CORUM − Chronos, each normalized to [0,1])",
        fontsize=11, loc="left",
    )
    axes[2].set_xlabel("j (column index)", fontsize=10)
    axes[2].set_ylabel("i (row index)", fontsize=10)

    fig.suptitle(
        "CORUM absolute recurrence counts vs Chronos importance\n"
        "Per-entry (i,j) | original matrix order | no smoothing",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    out_path = FIG_DIR / f"{PREFIX}fig7b_absolute_counts.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out_path.name}")


if __name__ == "__main__":
    main()
