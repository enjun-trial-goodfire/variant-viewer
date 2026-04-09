#!/usr/bin/env python3
"""Left vs Right probe view comparison — two-stage analysis.

Stage 1 (primary): CORUM co-complex enrichment for each embedding view
Stage 2 (secondary): DepMap dependency + lineage correlation delta

Three representations per variant (from the 64x64 covariance matrix M):
  left_64d  = M.mean(dim=1)   -> row mean -> 64-d, L2-norm
  right_64d = M.mean(dim=0)   -> col mean -> 64-d, L2-norm
  full_4096d = M.flatten()    -> 4096-d,   L2-norm

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_left_right_analysis.py
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from safetensors.torch import load_file as safetensors_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EEVE_ROOT = REPO_ROOT / "eeve-analysis"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"
EMB_DIR = EEVE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool"
CORUM_PATH = EEVE_ROOT / "data" / "corum_humanComplexes.json"
DEMETER2_PATH = EEVE_ROOT / "data" / "RNAi_AchillesDRIVEMarcotte,_DEMETER2_subsetted-2.csv"
CHRONOS_PATH = EEVE_ROOT / "data" / "CRISPR_DepMap_Public_26Q1Score_Chronos_subsetted.csv"

OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"

FULL_KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"

# ── Parameters ────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
MIN_COMPLEX_SIZE = 3
DEPMAP_K = 10
MIN_OVERLAP = 50
VIEWS = ["left_64d", "right_64d", "full_4096d"]


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── CORUM loading ─────────────────────────────────────────────────────

def load_corum() -> tuple[dict[str, set[int]], set[frozenset[str]], int, int]:
    raw = json.loads(CORUM_PATH.read_text())
    gene_to_cx: dict[str, set[int]] = defaultdict(set)
    co_complex_pairs: set[frozenset[str]] = set()
    n_total = len(raw)
    n_kept = 0
    for cx in raw:
        cid = cx["complex_id"]
        genes: set[str] = set()
        for su in cx["subunits"]:
            gn = su.get("swissprot", {}).get("gene_name")
            if gn:
                genes.add(gn.upper())
        if len(genes) < MIN_COMPLEX_SIZE:
            continue
        n_kept += 1
        for g in genes:
            gene_to_cx[g].add(cid)
        gl = sorted(genes)
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                co_complex_pairs.add(frozenset((gl[i], gl[j])))
    log.info(f"  CORUM: {n_total} complexes, {n_kept} with >={MIN_COMPLEX_SIZE} genes, "
             f"{len(gene_to_cx)} genes, {len(co_complex_pairs):,} pairs")
    return dict(gene_to_cx), co_complex_pairs, n_total, n_kept


# ── Variant loading ───────────────────────────────────────────────────

def load_variants() -> pl.DataFrame:
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT variant_id, gene_name, consequence_display
        FROM variants
        WHERE neighbors IS NOT NULL AND neighbors != '[]'
          AND gene_name IS NOT NULL
    """).pl()
    con.close()
    log.info(f"  Loaded {df.height:,} variants ({df['gene_name'].n_unique():,} genes)")
    return df


# ── Embedding loading — three views ──────────────────────────────────

def load_three_views(variant_ids: list[str]) -> tuple[
    dict[str, np.ndarray],  # view_name -> (N, d) array
    list[str],              # vid_order
]:
    """Load embeddings as [64,64], derive left/right/full views, L2-normalize."""
    log.info("  Loading embeddings and deriving 3 views...")

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
    n = len(found)
    log.info(f"    Found {n:,} / {len(variant_ids):,} in embedding index")

    by_chunk: dict[int, list[tuple[str, int]]] = {}
    for vid in found:
        cid, off = vid_to_loc[vid]
        by_chunk.setdefault(cid, []).append((vid, off))

    left = np.zeros((n, 64), dtype=np.float32)
    right = np.zeros((n, 64), dtype=np.float32)
    full = np.zeros((n, 4096), dtype=np.float32)
    vid_order: list[str] = []
    idx = 0

    for cid in sorted(by_chunk.keys()):
        chunk_path = EMB_DIR / "chunks" / f"chunk_{cid:06d}" / "activations.safetensors"
        tensor = safetensors_load(str(chunk_path))["activations"].float().numpy()
        for vid, off in by_chunk[cid]:
            mat = tensor[off]  # shape [64, 64]
            left[idx] = mat.mean(axis=1)   # row mean
            right[idx] = mat.mean(axis=0)  # col mean
            full[idx] = mat.reshape(4096)
            vid_order.append(vid)
            idx += 1
        if cid % 20 == 0:
            log.info(f"      Chunk {cid}: running total {idx:,}")

    def _l2_norm(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    views = {
        "left_64d": _l2_norm(left),
        "right_64d": _l2_norm(right),
        "full_4096d": _l2_norm(full),
    }
    log.info(f"    3 views ready: left {left.shape}, right {right.shape}, full {full.shape}")
    return views, vid_order


# ── kNN ───────────────────────────────────────────────────────────────

def compute_knn(embeddings: np.ndarray, max_k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    log.info(f"    Computing {max_k}-NN for {embeddings.shape[0]:,} x {embeddings.shape[1]}-d ...")
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    return indices[:, 1:]


# ── CORUM evaluation (same as run_corum_full.py) ─────────────────────

def evaluate_corum_at_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    gene_to_cx: dict[str, set[int]],
    co_complex_pairs: set[frozenset[str]],
    rng: np.random.Generator,
) -> dict:
    n_variants = len(vid_order)
    corum_genes_set = set(gene_to_cx.keys())
    corum_genes_list = sorted(corum_genes_set)
    n_corum = len(corum_genes_list)

    gene_nb_cor: dict[str, int] = defaultdict(int)
    gene_nb_sha: dict[str, int] = defaultdict(int)
    gene_rd_cor: dict[str, int] = defaultdict(int)
    gene_rd_sha: dict[str, int] = defaultdict(int)

    for qi in range(n_variants):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in corum_genes_set:
            continue
        nb_idx = knn_indices[qi, :k]
        n_valid = 0
        for ni in nb_idx:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in corum_genes_set:
                continue
            gene_nb_cor[src] += 1
            n_valid += 1
            if frozenset((src, tgt)) in co_complex_pairs:
                gene_nb_sha[src] += 1
        for _ in range(n_valid):
            for _a in range(50):
                rg = corum_genes_list[rng.integers(n_corum)]
                if rg != src:
                    break
            gene_rd_cor[src] += 1
            if frozenset((src, rg)) in co_complex_pairs:
                gene_rd_sha[src] += 1

    common = sorted(set(gene_nb_cor.keys()) & set(gene_rd_cor.keys()))
    n_genes = len(common)
    nb_t = sum(gene_nb_cor[g] for g in common)
    nb_s = sum(gene_nb_sha.get(g, 0) for g in common)
    rd_t = sum(gene_rd_cor[g] for g in common)
    rd_s = sum(gene_rd_sha.get(g, 0) for g in common)

    nb_frac = nb_s / nb_t if nb_t > 0 else 0
    rd_frac = rd_s / rd_t if rd_t > 0 else 0
    fold = nb_frac / rd_frac if rd_frac > 0 else float("inf")
    a, b, c, d = nb_s, nb_t - nb_s, rd_s, rd_t - rd_s
    or_val = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")

    nb_ca = np.array([gene_nb_cor[g] for g in common])
    nb_sa = np.array([gene_nb_sha.get(g, 0) for g in common])
    rd_ca = np.array([gene_rd_cor[g] for g in common])
    rd_sa = np.array([gene_rd_sha.get(g, 0) for g in common])

    boot_fold = np.empty(N_BOOTSTRAP)
    boot_or = np.empty(N_BOOTSTRAP)
    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        snc = nb_ca[idx].sum(); sns = nb_sa[idx].sum()
        src_val = rd_ca[idx].sum(); srs = rd_sa[idx].sum()
        nf = sns / snc if snc > 0 else 0
        rf = srs / src_val if src_val > 0 else 0
        boot_fold[bi] = nf / rf if rf > 0 else np.nan
        ba, bb, bc, bd = sns, snc - sns, srs, src_val - srs
        boot_or[bi] = (ba * bd) / (bb * bc) if bb > 0 and bc > 0 else np.nan

    fv = boot_fold[~np.isnan(boot_fold)]
    ov = boot_or[~np.isnan(boot_or)]

    return {
        "k": k,
        "n_genes": n_genes,
        "nb_pairs": int(nb_t), "nb_shared": int(nb_s),
        "rd_pairs": int(rd_t), "rd_shared": int(rd_s),
        "nb_frac": float(nb_frac), "rd_frac": float(rd_frac),
        "fold": float(fold),
        "fold_ci_lo": float(np.percentile(fv, 2.5)) if len(fv) > 0 else None,
        "fold_ci_hi": float(np.percentile(fv, 97.5)) if len(fv) > 0 else None,
        "or": float(or_val),
        "or_ci_lo": float(np.percentile(ov, 2.5)) if len(ov) > 0 else None,
        "or_ci_hi": float(np.percentile(ov, 97.5)) if len(ov) > 0 else None,
    }


# ── DepMap helpers ────────────────────────────────────────────────────

def load_demeter2() -> tuple[dict[str, np.ndarray], list[str]]:
    df = pl.read_csv(str(DEMETER2_PATH))
    meta_cols = {"depmap_id", "cell_line_display_name",
                 "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    gene_cols = [c for c in df.columns if c not in meta_cols]
    lineage_1 = df["lineage_1"].to_list()
    gene_to_vec: dict[str, np.ndarray] = {}
    mat = df.select(gene_cols).to_numpy().astype(np.float64)
    for i, g in enumerate(gene_cols):
        gene_to_vec[g] = mat[:, i]
    log.info(f"  DEMETER2: {len(gene_cols)} genes x {len(lineage_1)} cell lines")
    return gene_to_vec, lineage_1


def load_chronos_with_lineage() -> tuple[dict[str, np.ndarray], list[str]]:
    chron_df = pl.read_csv(str(CHRONOS_PATH))
    id_col = chron_df.columns[0]
    chron_df = chron_df.rename({id_col: "depmap_id"})
    gene_cols = [c for c in chron_df.columns if c != "depmap_id"]

    dem_df = pl.read_csv(str(DEMETER2_PATH))
    dem_meta = dem_df.select(["depmap_id", "lineage_1"])
    dem_ids = set(dem_meta["depmap_id"].to_list())
    chron_df = chron_df.filter(pl.col("depmap_id").is_in(list(dem_ids)))
    chron_df = chron_df.join(dem_meta, on="depmap_id", how="left")
    lineage_1 = chron_df["lineage_1"].to_list()

    gene_to_vec: dict[str, np.ndarray] = {}
    mat = chron_df.select(gene_cols).to_numpy().astype(np.float64)
    for i, g in enumerate(gene_cols):
        gene_to_vec[g] = mat[:, i]
    log.info(f"  Chronos: {len(gene_cols)} genes x {chron_df.height} cell lines")
    return gene_to_vec, lineage_1


def build_lineage_vectors(
    gene_to_vec: dict[str, np.ndarray], lineage_list: list[str],
) -> dict[str, np.ndarray]:
    unique = sorted(set(lineage_list))
    lin_to_idx = {l: i for i, l in enumerate(unique)}
    lin_arr = np.array([lin_to_idx[l] for l in lineage_list])
    lin_counts = np.bincount(lin_arr, minlength=len(unique))
    valid = lin_counts >= 5
    gene_lin_vecs: dict[str, np.ndarray] = {}
    for gene, vec in gene_to_vec.items():
        means = np.full(len(unique), np.nan)
        for li in range(len(unique)):
            if not valid[li]:
                continue
            mask = (lin_arr == li) & ~np.isnan(vec)
            if mask.sum() >= 3:
                means[li] = vec[mask].mean()
        gene_lin_vecs[gene] = means[valid]
    return gene_lin_vecs


def pairwise_corr(a: np.ndarray, b: np.ndarray, min_ol: int = MIN_OVERLAP) -> float | None:
    mask = ~(np.isnan(a) | np.isnan(b))
    n = mask.sum()
    if n < min_ol:
        return None
    va, vb = a[mask], b[mask]
    if va.std() < 1e-10 or vb.std() < 1e-10:
        return None
    return float(np.corrcoef(va, vb)[0, 1])


def depmap_view_comparison(
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    gene_to_vec: dict[str, np.ndarray],
    gene_lin_vecs: dict[str, np.ndarray],
    dataset_name: str,
    rng: np.random.Generator,
) -> dict:
    """Compute neighbor-vs-random profile and lineage correlation delta at k=DEPMAP_K."""
    k = DEPMAP_K
    n_variants = len(vid_order)
    dep_genes = set(gene_to_vec.keys())
    dep_genes_list = sorted(dep_genes)
    n_dep = len(dep_genes_list)

    nb_prof: list[float] = []
    rd_prof: list[float] = []
    nb_lin: list[float] = []
    rd_lin: list[float] = []

    # Per-gene accumulators for bootstrap
    gene_nb_prof: dict[str, list[float]] = defaultdict(list)
    gene_rd_prof: dict[str, list[float]] = defaultdict(list)
    gene_nb_lin: dict[str, list[float]] = defaultdict(list)
    gene_rd_lin: dict[str, list[float]] = defaultdict(list)

    for qi in range(n_variants):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in dep_genes:
            continue
        nb_idx = knn_indices[qi, :k]
        cross_targets = []
        for ni in nb_idx:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in dep_genes:
                continue
            cross_targets.append(tgt)

            pc = pairwise_corr(gene_to_vec[src], gene_to_vec[tgt])
            if pc is not None:
                nb_prof.append(pc)
                gene_nb_prof[src].append(pc)

            lc = pairwise_corr(gene_lin_vecs.get(src, np.array([])),
                               gene_lin_vecs.get(tgt, np.array([])),
                               min_ol=10)
            if lc is not None:
                nb_lin.append(lc)
                gene_nb_lin[src].append(lc)

        for _ in range(len(cross_targets)):
            for _a in range(50):
                rg = dep_genes_list[rng.integers(n_dep)]
                if rg != src:
                    break
            pc = pairwise_corr(gene_to_vec[src], gene_to_vec[rg])
            if pc is not None:
                rd_prof.append(pc)
                gene_rd_prof[src].append(pc)
            lc = pairwise_corr(gene_lin_vecs.get(src, np.array([])),
                               gene_lin_vecs.get(rg, np.array([])),
                               min_ol=10)
            if lc is not None:
                rd_lin.append(lc)
                gene_rd_lin[src].append(lc)

    nb_prof_mean = np.mean(nb_prof) if nb_prof else 0
    rd_prof_mean = np.mean(rd_prof) if rd_prof else 0
    nb_lin_mean = np.mean(nb_lin) if nb_lin else 0
    rd_lin_mean = np.mean(rd_lin) if rd_lin else 0

    # Bootstrap delta by gene
    def _bootstrap_delta(nb_dict, rd_dict):
        common = sorted(set(nb_dict.keys()) & set(rd_dict.keys()))
        if len(common) < 10:
            return 0.0, 0.0, 0.0
        nb_means = np.array([np.mean(nb_dict[g]) for g in common])
        rd_means = np.array([np.mean(rd_dict[g]) for g in common])
        n = len(common)
        deltas = np.empty(500)
        for b in range(500):
            idx = rng.integers(0, n, size=n)
            deltas[b] = nb_means[idx].mean() - rd_means[idx].mean()
        return float(np.percentile(deltas, 2.5)), float(np.mean(deltas)), float(np.percentile(deltas, 97.5))

    prof_ci = _bootstrap_delta(gene_nb_prof, gene_rd_prof)
    lin_ci = _bootstrap_delta(gene_nb_lin, gene_rd_lin)

    return {
        "dataset": dataset_name,
        "k": k,
        "n_nb_prof": len(nb_prof),
        "n_rd_prof": len(rd_prof),
        "nb_prof_mean": float(nb_prof_mean),
        "rd_prof_mean": float(rd_prof_mean),
        "prof_delta": float(nb_prof_mean - rd_prof_mean),
        "prof_ci_lo": prof_ci[0], "prof_ci_mid": prof_ci[1], "prof_ci_hi": prof_ci[2],
        "nb_lin_mean": float(nb_lin_mean),
        "rd_lin_mean": float(rd_lin_mean),
        "lin_delta": float(nb_lin_mean - rd_lin_mean),
        "lin_ci_lo": lin_ci[0], "lin_ci_mid": lin_ci[1], "lin_ci_hi": lin_ci[2],
    }


# ── Plotting ──────────────────────────────────────────────────────────

def plot_corum_by_view(all_results: dict[str, list[dict]]) -> None:
    """Overlaid fold enrichment vs k for all 3 views."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"left_64d": "C0", "right_64d": "C1", "full_4096d": "C2"}
    markers = {"left_64d": "o", "right_64d": "s", "full_4096d": "D"}
    offsets = {"left_64d": -0.8, "right_64d": 0, "full_4096d": 0.8}
    labels = {"left_64d": "Left (row mean, 64d)",
              "right_64d": "Right (col mean, 64d)",
              "full_4096d": "Full (flattened, 4096d)"}

    for view in VIEWS:
        rows = all_results[view]
        ks = [r["k"] + offsets[view] for r in rows]
        folds = [r["fold"] for r in rows]
        ci_lo = [r["fold_ci_lo"] for r in rows]
        ci_hi = [r["fold_ci_hi"] for r in rows]
        yerr = [[f - lo for f, lo in zip(folds, ci_lo)],
                [hi - f for f, hi in zip(folds, ci_hi)]]
        ax.errorbar(ks, folds, yerr=yerr, fmt=f"{markers[view]}-",
                    capsize=4, markersize=8, linewidth=2.2,
                    color=colors[view], label=labels[view])

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Fold enrichment (neighbor / random)", fontsize=12)
    ax.set_title("CORUM co-complex enrichment by embedding view", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_enrichment_by_view.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_enrichment_by_view.png")


def plot_depmap_by_view(depmap_results: list[dict]) -> None:
    """Grouped bar chart: delta per view, faceted by dataset and metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in [
        (axes[0], "prof", "Dependency profile correlation delta"),
        (axes[1], "lin", "Lineage correlation delta"),
    ]:
        datasets = sorted(set(r["dataset"] for r in depmap_results))
        n_ds = len(datasets)
        n_views = len(VIEWS)
        width = 0.22
        x = np.arange(n_ds)

        colors = {"left_64d": "C0", "right_64d": "C1", "full_4096d": "C2"}
        labels_map = {"left_64d": "Left 64d", "right_64d": "Right 64d", "full_4096d": "Full 4096d"}

        for vi, view in enumerate(VIEWS):
            deltas = []
            ci_lo_list = []
            ci_hi_list = []
            for ds in datasets:
                match = [row for row in depmap_results if row["dataset"] == ds and row["view"] == view][0]
                mid = match[f"{metric}_ci_mid"]
                deltas.append(mid)
                ci_lo_list.append(match[f"{metric}_ci_lo"])
                ci_hi_list.append(match[f"{metric}_ci_hi"])
            yerr = [[max(0, d - lo) for d, lo in zip(deltas, ci_lo_list)],
                    [max(0, hi - d) for d, hi in zip(deltas, ci_hi_list)]]
            offset = (vi - (n_views - 1) / 2) * width
            ax.bar(x + offset, deltas, width, yerr=yerr, capsize=3,
                   color=colors[view], label=labels_map[view], alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
        ax.set_ylabel("Delta (neighbor - random)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_depmap_view_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_depmap_view_comparison.png")


# ── Report ────────────────────────────────────────────────────────────

def write_report(
    corum_results: dict[str, list[dict]],
    depmap_results: list[dict],
) -> None:
    md_path = OUT_DIR / "left_right_report.md"
    lines = [
        "# Left vs Right Probe View Comparison",
        "",
        "## Representations",
        "",
        "From each variant's 64x64 cross-covariance matrix M:",
        "- **Left (64d):** row mean = `M.mean(dim=1)`, L2-normalized",
        "- **Right (64d):** column mean = `M.mean(dim=0)`, L2-normalized",
        "- **Full (4096d):** `M.flatten()`, L2-normalized",
        "",
        "The matrix is asymmetric (proj_left x proj_right), so left and right marginals "
        "carry different information (cosine similarity ~0.3-0.6).",
        "",
        "---",
        "",
        "## Stage 1 — CORUM Co-Complex Enrichment (Primary)",
        "",
        "| View | k | NB shared/total | NB % | RD % | Fold | 95% CI | OR | 95% CI |",
        "|------|---|----------------|------|------|------|--------|-----|--------|",
    ]

    for view in VIEWS:
        for r in corum_results[view]:
            lines.append(
                f"| {view} | {r['k']} | {r['nb_shared']:,}/{r['nb_pairs']:,} "
                f"| {r['nb_frac']:.3%} | {r['rd_frac']:.3%} "
                f"| {r['fold']:.2f}x | [{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}] "
                f"| {r['or']:.2f} | [{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] |"
            )

    # Find best view at k=10
    k10 = {v: [r for r in corum_results[v] if r["k"] == 10][0] for v in VIEWS}
    best_view = max(VIEWS, key=lambda v: k10[v]["fold"])
    lines += [
        "",
        f"**Best CORUM view at k=10:** {best_view} (fold = {k10[best_view]['fold']:.2f}x)",
        "",
    ]

    # Interpretation
    folds_at_10 = {v: k10[v]["fold"] for v in VIEWS}
    full_f = folds_at_10["full_4096d"]
    left_f = folds_at_10["left_64d"]
    right_f = folds_at_10["right_64d"]

    lines.append("**Interpretation:**")
    if full_f > left_f * 1.1 and full_f > right_f * 1.1:
        lines.append("- Full > both marginals: interaction structure contributes beyond either marginal alone.")
    elif abs(left_f - full_f) / full_f < 0.1:
        lines.append("- Left ~ Full: the left (row) marginal captures most of the biological signal.")
    elif abs(right_f - full_f) / full_f < 0.1:
        lines.append("- Right ~ Full: the right (column) marginal captures most of the biological signal.")
    else:
        lines.append(f"- Left={left_f:.2f}x, Right={right_f:.2f}x, Full={full_f:.2f}x")

    if left_f > right_f * 1.15:
        lines.append("- Left marginal is substantially stronger than right marginal.")
    elif right_f > left_f * 1.15:
        lines.append("- Right marginal is substantially stronger than left marginal.")
    else:
        lines.append("- Left and right marginals show similar enrichment.")

    lines += [
        "",
        "---",
        "",
        "## Stage 2 — DepMap Validation (Secondary)",
        "",
        "| View | Dataset | Prof delta | 95% CI | Lin delta | 95% CI |",
        "|------|---------|-----------|--------|----------|--------|",
    ]
    for r in depmap_results:
        lines.append(
            f"| {r['view']} | {r['dataset']} "
            f"| {r['prof_delta']:.4f} | [{r['prof_ci_lo']:.4f}, {r['prof_ci_hi']:.4f}] "
            f"| {r['lin_delta']:.4f} | [{r['lin_ci_lo']:.4f}, {r['lin_ci_hi']:.4f}] |"
        )

    # Find best depmap view
    dem_rows = [r for r in depmap_results if r["dataset"] == "demeter2"]
    if dem_rows:
        best_dep = max(dem_rows, key=lambda r: r["prof_delta"])
        lines += [
            "",
            f"**Best DepMap view (DEMETER2 profile delta):** {best_dep['view']} "
            f"(delta = {best_dep['prof_delta']:.4f})",
        ]

    # Cross-stage interpretation
    lines += [
        "",
        "---",
        "",
        "## Cross-Stage Summary",
        "",
    ]
    corum_best = best_view
    depmap_best = best_dep["view"] if dem_rows else "N/A"
    if corum_best == depmap_best:
        lines.append(
            f"CORUM and DepMap agree: **{corum_best}** is the strongest view across both readouts. "
            "This suggests a consistent biological signal across protein-complex structure "
            "and cellular dependency phenotypes."
        )
    else:
        lines.append(
            f"CORUM prefers **{corum_best}** while DepMap prefers **{depmap_best}**. "
            "This suggests protein-complex structure and cellular-context biology "
            "are encoded differently in the left vs right projection spaces."
        )

    lines += [
        "",
        "## Figures",
        "",
        "- `fig_corum_enrichment_by_view.png`",
        "- `fig_depmap_view_comparison.png`",
    ]

    md_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {md_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CORUM ────────────────────────────────────────────────────
    log.info("Loading CORUM...")
    gene_to_cx, co_complex_pairs, n_total, n_kept = load_corum()

    # ── Load variants ─────────────────────────────────────────────────
    log.info("Loading variants...")
    var_df = load_variants()
    variant_ids = var_df["variant_id"].to_list()
    vid_to_gene = {v: g.upper() for v, g in zip(
        var_df["variant_id"].to_list(), var_df["gene_name"].to_list()
    )}

    # ── Load embeddings — 3 views ─────────────────────────────────────
    log.info("Loading embeddings and building views...")
    views, vid_order = load_three_views(variant_ids)

    # ── Compute / load kNN per view ───────────────────────────────────
    max_k = max(K_VALUES)
    knn_per_view: dict[str, np.ndarray] = {}

    # Full: try to reuse cache
    if FULL_KNN_CACHE.exists():
        log.info("  Loading cached full kNN...")
        cached = np.load(str(FULL_KNN_CACHE), allow_pickle=True)
        cached_vid = cached["vid_order"].tolist()
        if cached_vid == vid_order:
            knn_per_view["full_4096d"] = cached["knn_indices"]
            log.info(f"    Cache hit: {knn_per_view['full_4096d'].shape}")
        else:
            log.info("    Cache vid_order mismatch, recomputing full kNN...")

    for view in VIEWS:
        if view in knn_per_view:
            continue
        log.info(f"  Computing kNN for {view}...")
        knn_per_view[view] = compute_knn(views[view], max_k)

    # Save left/right kNN
    for view in ["left_64d", "right_64d"]:
        path = OUT_DIR / f"{view}_knn_indices.npz"
        np.savez_compressed(str(path), knn_indices=knn_per_view[view],
                            vid_order=np.array(vid_order, dtype=object))
        log.info(f"    Saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    # ════════════════════════════════════════════════════════════════
    # STAGE 1 — CORUM
    # ════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STAGE 1: CORUM co-complex enrichment per view")
    log.info("=" * 60)

    corum_results: dict[str, list[dict]] = {}
    for view in VIEWS:
        log.info(f"\n  --- {view} ---")
        results_v = []
        for k in K_VALUES:
            rng = np.random.default_rng(RANDOM_SEED + k)
            r = evaluate_corum_at_k(
                k, knn_per_view[view], vid_order, vid_to_gene,
                gene_to_cx, co_complex_pairs, rng,
            )
            r["view"] = view
            results_v.append(r)
        corum_results[view] = results_v

    # Save
    all_corum_rows = []
    for view in VIEWS:
        all_corum_rows.extend(corum_results[view])
    pl.DataFrame(all_corum_rows).write_parquet(OUT_DIR / "corum_enrichment_by_view.parquet")
    log.info("  Saved corum_enrichment_by_view.parquet")

    # ════════════════════════════════════════════════════════════════
    # STAGE 2 — DepMap
    # ════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STAGE 2: DepMap validation per view")
    log.info("=" * 60)

    depmap_results: list[dict] = []

    for ds_name, loader in [("demeter2", load_demeter2), ("chronos", load_chronos_with_lineage)]:
        log.info(f"\n  Loading {ds_name}...")
        gene_to_vec, lineage_1 = loader()
        gene_lin_vecs = build_lineage_vectors(gene_to_vec, lineage_1)

        for view in VIEWS:
            log.info(f"    {ds_name} / {view} (k={DEPMAP_K})...")
            rng = np.random.default_rng(RANDOM_SEED + hash(ds_name) % 10000)
            r = depmap_view_comparison(
                knn_per_view[view], vid_order, vid_to_gene,
                gene_to_vec, gene_lin_vecs, ds_name, rng,
            )
            r["view"] = view
            depmap_results.append(r)
            log.info(f"      prof_delta={r['prof_delta']:.4f} [{r['prof_ci_lo']:.4f}, {r['prof_ci_hi']:.4f}]  "
                     f"lin_delta={r['lin_delta']:.4f} [{r['lin_ci_lo']:.4f}, {r['lin_ci_hi']:.4f}]")

    pl.DataFrame(depmap_results).write_parquet(OUT_DIR / "depmap_view_comparison.parquet")
    log.info("  Saved depmap_view_comparison.parquet")

    # ── Plots + report ────────────────────────────────────────────────
    log.info("\nGenerating figures and report...")
    plot_corum_by_view(corum_results)
    plot_depmap_by_view(depmap_results)
    write_report(corum_results, depmap_results)

    # ── Run config ────────────────────────────────────────────────────
    config = {
        "analysis": "left_right_view_comparison",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap_corum": N_BOOTSTRAP,
        "n_bootstrap_depmap": 500,
        "k_values_corum": K_VALUES,
        "k_depmap": DEPMAP_K,
        "min_complex_size": MIN_COMPLEX_SIZE,
        "min_overlap_depmap": MIN_OVERLAP,
        "views": VIEWS,
        "n_variants": len(vid_order),
        "n_corum_genes": len(gene_to_cx),
    }
    cfg_path = OUT_DIR / "left_right_run_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    log.info(f"  Saved {cfg_path.name}")

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")

    # ── Terminal summary ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("LEFT vs RIGHT PROBE VIEW COMPARISON")
    print("=" * 100)

    print("\n  STAGE 1 — CORUM co-complex enrichment")
    print(f"  {'View':<15s} {'k':>3s} {'Fold':>8s} {'95% CI':>18s} {'OR':>8s}")
    print("  " + "-" * 55)
    for view in VIEWS:
        for r in corum_results[view]:
            print(f"  {view:<15s} {r['k']:>3d} {r['fold']:>7.2f}x "
                  f"[{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}]  {r['or']:>7.2f}")

    print("\n  STAGE 2 — DepMap validation (k=10)")
    print(f"  {'View':<15s} {'Dataset':<10s} {'Prof delta':>12s} {'95% CI':>22s} {'Lin delta':>12s} {'95% CI':>22s}")
    print("  " + "-" * 95)
    for r in depmap_results:
        print(f"  {r['view']:<15s} {r['dataset']:<10s} "
              f"{r['prof_delta']:>11.4f} [{r['prof_ci_lo']:.4f}, {r['prof_ci_hi']:.4f}]  "
              f"{r['lin_delta']:>11.4f} [{r['lin_ci_lo']:.4f}, {r['lin_ci_hi']:.4f}]")

    print(f"\n  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
