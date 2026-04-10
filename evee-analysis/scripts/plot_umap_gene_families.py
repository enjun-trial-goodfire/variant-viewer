#!/usr/bin/env python3
"""UMAP of Evo2 second-order embeddings colored by HGNC gene family.

Pipeline:
  1. Load 124K embeddings (4096-d, from safetensors chunks)
  2. PCA → 50 dims
  3. UMAP → 2 dims
  4. Color by top-N gene families, rest as "Other", unannotated as "Unannotated"
  5. Save UMAP coordinates as intermediate, generate figure

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/plot_umap_gene_families.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"
KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"
HGNC_PATH = EVEE_ROOT / "data" / "hgnc_complete_set.txt"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"
EMB_DIR = EVEE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool"

RANDOM_SEED = 42
N_TOP_CLASSES = 14
PCA_DIMS = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_embeddings(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load embeddings for given variant_ids from safetensors chunks."""
    import sqlite3
    from safetensors.torch import load_file as safetensors_load

    idx_conn = sqlite3.connect(str(EMB_DIR / "index.sqlite"))
    vid_to_loc: dict[str, tuple[int, int]] = {}
    for i in range(0, len(variant_ids), 999):
        batch = variant_ids[i : i + 999]
        ph = ",".join(["?"] * len(batch))
        rows = idx_conn.execute(
            f"SELECT sequence_id, chunk_id, offset FROM sequence_locations WHERE sequence_id IN ({ph})",
            batch,
        ).fetchall()
        for seq_id, cid, off in rows:
            vid_to_loc[seq_id] = (cid, off)
    idx_conn.close()

    found = [v for v in variant_ids if v in vid_to_loc]
    log.info(f"  Found {len(found):,} / {len(variant_ids):,} in embedding index")

    by_chunk: dict[int, list[tuple[str, int]]] = {}
    for vid in found:
        cid, off = vid_to_loc[vid]
        by_chunk.setdefault(cid, []).append((vid, off))

    embeddings = np.zeros((len(found), 4096), dtype=np.float32)
    vid_order: list[str] = []
    idx = 0

    for cid in sorted(by_chunk.keys()):
        chunk_path = EMB_DIR / "chunks" / f"chunk_{cid:06d}" / "activations.safetensors"
        tensor = safetensors_load(str(chunk_path))["activations"].float().numpy()
        for vid, off in by_chunk[cid]:
            embeddings[idx] = tensor[off].reshape(4096)
            vid_order.append(vid)
            idx += 1
        if cid % 20 == 0:
            log.info(f"    Chunk {cid}: running total {idx:,}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)
    log.info(f"  Shape: {embeddings.shape}, L2-normalized")
    return embeddings, vid_order


def assign_gene_class(
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    n_top: int,
) -> tuple[list[str], list[str]]:
    """Assign each variant a single gene class label.

    For genes with multiple groups, pick the rarest group (most specific).
    Returns (labels, class_order) where class_order lists classes from most
    to least common, with 'Other' and 'Unannotated' last.
    """
    hgnc = pl.read_csv(str(HGNC_PATH), separator="\t", infer_schema_length=0)
    gg_df = hgnc.filter(pl.col("gene_group").is_not_null() & (pl.col("gene_group") != ""))

    # Build gene → groups mapping
    raw_gene_groups: dict[str, list[str]] = {}
    group_counts: Counter = Counter()
    for sym, groups_str in zip(gg_df["symbol"].to_list(), gg_df["gene_group"].to_list()):
        groups = [g.strip() for g in groups_str.split("|") if g.strip()]
        if groups:
            raw_gene_groups[sym] = groups
            for g in groups:
                group_counts[g] += 1

    # Count variants per group to pick top-N by variant count (not gene count)
    variant_group_counts: Counter = Counter()
    for vid in vid_order:
        gene = vid_to_gene.get(vid)
        if gene and gene in raw_gene_groups:
            for g in raw_gene_groups[gene]:
                variant_group_counts[g] += 1

    top_groups = [g for g, _ in variant_group_counts.most_common(n_top)]
    top_set = set(top_groups)
    log.info(f"  Top {n_top} groups (by variant count): {top_groups}")

    # Assign each variant exactly one label
    labels: list[str] = []
    label_counts: Counter = Counter()
    for vid in vid_order:
        gene = vid_to_gene.get(vid)
        if gene is None:
            labels.append("Unannotated")
            label_counts["Unannotated"] += 1
            continue

        if gene not in raw_gene_groups:
            labels.append("Unannotated")
            label_counts["Unannotated"] += 1
            continue

        groups = raw_gene_groups[gene]
        # Pick the first matching top group; if none, "Other"
        matched = [g for g in groups if g in top_set]
        if matched:
            # Pick the rarest one for specificity
            best = min(matched, key=lambda g: group_counts[g])
            labels.append(best)
            label_counts[best] += 1
        else:
            labels.append("Other")
            label_counts["Other"] += 1

    # Order: top groups sorted by variant count descending, then Other, Unannotated
    class_order = [g for g in top_groups if label_counts.get(g, 0) > 0]
    class_order.append("Other")
    class_order.append("Unannotated")

    for c in class_order:
        log.info(f"    {label_counts.get(c, 0):>7,}  {c}")

    return labels, class_order


def make_colormap(class_order: list[str]) -> dict[str, str]:
    """Generate a distinct color for each class."""
    # Use a qualitative palette for named classes, gray shades for Other/Unannotated
    n_named = len(class_order) - 2
    # Tab20 gives 20 distinct colors
    tab20 = plt.cm.get_cmap("tab20")
    colors = {}
    for i, cls in enumerate(class_order):
        if cls == "Unannotated":
            colors[cls] = "#d9d9d9"  # light gray
        elif cls == "Other":
            colors[cls] = "#969696"  # medium gray
        else:
            colors[cls] = mcolors.to_hex(tab20(i / max(n_named - 1, 1)))
    return colors


def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check for cached UMAP coords ─────────────────────────────────
    umap_cache = OUT_DIR / "umap_coords.npz"
    if umap_cache.exists():
        log.info("Loading cached UMAP coordinates...")
        cached = np.load(str(umap_cache), allow_pickle=True)
        umap_xy = cached["umap_xy"]
        vid_order = cached["vid_order"].tolist()
        log.info(f"  Loaded {len(vid_order):,} points")
    else:
        # ── Load kNN vid_order for consistent variant set ─────────────
        log.info("Loading variant list from kNN cache...")
        knn_data = np.load(str(KNN_CACHE), allow_pickle=True)
        vid_order_knn = knn_data["vid_order"].tolist()
        log.info(f"  {len(vid_order_knn):,} variants")

        # ── Load embeddings ───────────────────────────────────────────
        log.info("Loading embeddings...")
        embeddings, vid_order = load_embeddings(vid_order_knn)

        # ── PCA ───────────────────────────────────────────────────────
        log.info(f"PCA → {PCA_DIMS} dimensions...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=PCA_DIMS, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(embeddings)
        var_explained = pca.explained_variance_ratio_.sum()
        log.info(f"  Variance explained: {var_explained:.2%}")
        del embeddings

        # ── UMAP ──────────────────────────────────────────────────────
        log.info(f"UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=RANDOM_SEED,
            verbose=True,
        )
        umap_xy = reducer.fit_transform(X_pca)
        del X_pca

        # ── Save coordinates ──────────────────────────────────────────
        np.savez_compressed(
            str(umap_cache),
            umap_xy=umap_xy,
            vid_order=np.array(vid_order, dtype=object),
        )
        log.info(f"  Saved umap_coords.npz ({umap_cache.stat().st_size / 1e6:.1f} MB)")

    # ── Gene mapping ──────────────────────────────────────────────────
    log.info("Loading gene annotations...")
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT variant_id, gene_name FROM variants WHERE gene_name IS NOT NULL").fetchall()
    con.close()
    vid_to_gene = {v: g.upper() for v, g in rows}

    # ── Assign classes ────────────────────────────────────────────────
    log.info("Assigning gene family labels...")
    labels, class_order = assign_gene_class(vid_order, vid_to_gene, N_TOP_CLASSES)
    color_map = make_colormap(class_order)

    # ── Plot ──────────────────────────────────────────────────────────
    log.info("Plotting UMAP...")
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot background classes first (Unannotated, Other), then named classes on top
    plot_order = ["Unannotated", "Other"] + [c for c in class_order if c not in ("Unannotated", "Other")]
    label_arr = np.array(labels)

    for cls in plot_order:
        mask = label_arr == cls
        n = mask.sum()
        if n == 0:
            continue
        is_bg = cls in ("Unannotated", "Other")
        ax.scatter(
            umap_xy[mask, 0], umap_xy[mask, 1],
            c=color_map[cls],
            s=0.3 if is_bg else 1.2,
            alpha=0.15 if is_bg else 0.5,
            label=f"{cls} ({n:,})",
            rasterized=True,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title("Evo2 Second-Order Embeddings — Colored by Gene Family", fontsize=15, pad=15)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)

    # Legend outside the plot
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        markerscale=8,
        frameon=True,
        framealpha=0.9,
        handletextpad=0.3,
        borderpad=0.6,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(0.9)

    fig.savefig(
        FIG_DIR / "fig_umap_gene_families.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    log.info("  Saved fig_umap_gene_families.png")

    # ── Save labelled coordinates for reuse ───────────────────────────
    label_df = pl.DataFrame({
        "variant_id": vid_order,
        "gene": [vid_to_gene.get(v, "") for v in vid_order],
        "gene_class": labels,
        "umap_1": umap_xy[:, 0].tolist(),
        "umap_2": umap_xy[:, 1].tolist(),
    })
    label_df.write_parquet(OUT_DIR / "umap_gene_family_labels.parquet")
    log.info(f"  Saved umap_gene_family_labels.parquet ({label_df.height:,} rows)")

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.0f}s")

    print(f"\n{'=' * 60}")
    print("UMAP GENE FAMILY PLOT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Variants plotted: {len(vid_order):,}")
    print(f"Classes: {len(class_order)} ({N_TOP_CLASSES} named + Other + Unannotated)")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
