#!/usr/bin/env python3
"""CORUM complex co-membership enrichment vs number of neighbors (k).

Recomputes top-50 kNN from embeddings, then for each k in {5,10,20,50}
evaluates the fraction of cross-gene neighbor pairs that share a CORUM
complex vs matched random pairs.  Reports fold enrichment + odds ratio
with gene-level bootstrap CIs.

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_corum_vs_k.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file as safetensors_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EEVE_ROOT = REPO_ROOT / "eeve-analysis"
EMB_DIR = EEVE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool"
OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"
CORUM_PATH = EEVE_ROOT / "data" / "corum_humanComplexes.json"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]


# ── CORUM ─────────────────────────────────────────────────────────────

def load_corum() -> tuple[dict[str, set[int]], set[frozenset[str]]]:
    """Returns gene→complex-ids map and a set of co-complex gene pairs."""
    raw = json.loads(CORUM_PATH.read_text())
    gene_to_cx: dict[str, set[int]] = defaultdict(set)
    co_complex_pairs: set[frozenset[str]] = set()

    for cx in raw:
        cid = cx["complex_id"]
        genes = set()
        for su in cx["subunits"]:
            gn = su.get("swissprot", {}).get("gene_name")
            if gn:
                gene_to_cx[gn].add(cid)
                genes.add(gn)
        genes_list = sorted(genes)
        for i in range(len(genes_list)):
            for j in range(i + 1, len(genes_list)):
                co_complex_pairs.add(frozenset((genes_list[i], genes_list[j])))

    log.info(f"  CORUM: {len(raw)} complexes, {len(gene_to_cx)} genes, "
             f"{len(co_complex_pairs):,} co-complex gene pairs")
    return dict(gene_to_cx), co_complex_pairs


# ── Embedding loading + kNN ───────────────────────────────────────────

def load_embeddings(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    idx_conn = sqlite3.connect(str(EMB_DIR / "index.sqlite"))
    vid_to_loc: dict[str, tuple[int, int]] = {}
    for i in range(0, len(variant_ids), 999):
        batch = variant_ids[i:i + 999]
        ph = ",".join(["?"] * len(batch))
        rows = idx_conn.execute(
            f"SELECT sequence_id, chunk_id, offset FROM sequence_locations WHERE sequence_id IN ({ph})",
            batch,
        ).fetchall()
        for seq_id, cid, off in rows:
            vid_to_loc[seq_id] = (cid, off)
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
        tensor = safetensors_load(str(chunk_path))["activations"].float().numpy()
        for vid, off in by_chunk[cid]:
            embeddings[idx] = tensor[off].reshape(4096)
            vid_order.append(vid)
            idx += 1
        log.info(f"    Chunk {cid}: {len(by_chunk[cid]):,} variants")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)
    return embeddings, vid_order


def compute_knn(embeddings: np.ndarray, max_k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    log.info(f"  Computing {max_k}-NN for {embeddings.shape[0]:,} variants...")
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    log.info(f"    kNN complete")
    return indices[:, 1:]


# ── Per-k CORUM enrichment ────────────────────────────────────────────

def evaluate_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    gene_to_cx: dict[str, set[int]],
    co_complex_pairs: set[frozenset[str]],
    rng: np.random.Generator,
) -> dict:
    """Compute CORUM sharing stats for top-k neighbors + matched random."""
    n_variants = len(vid_order)
    corum_genes = set(gene_to_cx.keys())
    all_genes = sorted({g for g in vid_to_gene.values() if g in corum_genes})
    gene_set = set(all_genes)

    # Per source-gene accumulators for bootstrap
    gene_nb_corum: dict[str, int] = defaultdict(int)
    gene_nb_shared: dict[str, int] = defaultdict(int)
    gene_rd_corum: dict[str, int] = defaultdict(int)
    gene_rd_shared: dict[str, int] = defaultdict(int)

    for qi in range(n_variants):
        src_gene = vid_to_gene.get(vid_order[qi])
        if src_gene is None:
            continue
        src_in_corum = src_gene in corum_genes

        nb_indices = knn_indices[qi, :k]
        cross_gene_targets = []
        for ni in nb_indices:
            tgt_gene = vid_to_gene.get(vid_order[ni])
            if tgt_gene is None or tgt_gene == src_gene:
                continue
            cross_gene_targets.append(tgt_gene)
            if src_in_corum and tgt_gene in corum_genes:
                gene_nb_corum[src_gene] += 1
                if frozenset((src_gene, tgt_gene)) in co_complex_pairs:
                    gene_nb_shared[src_gene] += 1

        n_random = len(cross_gene_targets)
        if n_random == 0 or not src_in_corum:
            continue

        for _ in range(n_random):
            for _attempt in range(20):
                rg = all_genes[rng.integers(len(all_genes))]
                if rg != src_gene:
                    break
            gene_rd_corum[src_gene] += 1
            if frozenset((src_gene, rg)) in co_complex_pairs:
                gene_rd_shared[src_gene] += 1

    # Aggregate point estimates
    common_genes = sorted(set(gene_nb_corum.keys()) & set(gene_rd_corum.keys()))
    nb_corum_total = sum(gene_nb_corum[g] for g in common_genes)
    nb_shared_total = sum(gene_nb_shared[g] for g in common_genes)
    rd_corum_total = sum(gene_rd_corum[g] for g in common_genes)
    rd_shared_total = sum(gene_rd_shared[g] for g in common_genes)

    nb_frac = nb_shared_total / nb_corum_total if nb_corum_total > 0 else 0
    rd_frac = rd_shared_total / rd_corum_total if rd_corum_total > 0 else 0
    fold = nb_frac / rd_frac if rd_frac > 0 else float("inf")
    a, b = nb_shared_total, nb_corum_total - nb_shared_total
    c, d = rd_shared_total, rd_corum_total - rd_shared_total
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")

    # Bootstrap
    n_genes = len(common_genes)
    nb_cor_arr = np.array([gene_nb_corum[g] for g in common_genes])
    nb_sha_arr = np.array([gene_nb_shared[g] for g in common_genes])
    rd_cor_arr = np.array([gene_rd_corum[g] for g in common_genes])
    rd_sha_arr = np.array([gene_rd_shared[g] for g in common_genes])

    boot_fold = np.empty(N_BOOTSTRAP)
    boot_or = np.empty(N_BOOTSTRAP)
    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        s_nb_cor = nb_cor_arr[idx].sum()
        s_nb_sha = nb_sha_arr[idx].sum()
        s_rd_cor = rd_cor_arr[idx].sum()
        s_rd_sha = rd_sha_arr[idx].sum()
        nf = s_nb_sha / s_nb_cor if s_nb_cor > 0 else 0
        rf = s_rd_sha / s_rd_cor if s_rd_cor > 0 else 0
        boot_fold[bi] = nf / rf if rf > 0 else np.nan
        ba, bb = s_nb_sha, s_nb_cor - s_nb_sha
        bc, bd = s_rd_sha, s_rd_cor - s_rd_sha
        boot_or[bi] = (ba * bd) / (bb * bc) if bb > 0 and bc > 0 else np.nan

    fold_valid = boot_fold[~np.isnan(boot_fold)]
    or_valid = boot_or[~np.isnan(boot_or)]

    result = {
        "k": k,
        "n_source_genes": n_genes,
        "nb_corum_pairs": int(nb_corum_total),
        "nb_shared": int(nb_shared_total),
        "rd_corum_pairs": int(rd_corum_total),
        "rd_shared": int(rd_shared_total),
        "nb_frac": float(nb_frac),
        "rd_frac": float(rd_frac),
        "fold_enrichment": float(fold),
        "odds_ratio": float(odds_ratio),
        "fold_ci_lo": float(np.percentile(fold_valid, 2.5)),
        "fold_ci_hi": float(np.percentile(fold_valid, 97.5)),
        "or_ci_lo": float(np.percentile(or_valid, 2.5)),
        "or_ci_hi": float(np.percentile(or_valid, 97.5)),
    }

    log.info(f"    k={k:>2d}: nb={nb_shared_total}/{nb_corum_total} ({nb_frac:.3%})  "
             f"rd={rd_shared_total}/{rd_corum_total} ({rd_frac:.3%})  "
             f"fold={fold:.2f}x [{result['fold_ci_lo']:.2f}, {result['fold_ci_hi']:.2f}]  "
             f"OR={odds_ratio:.2f} [{result['or_ci_lo']:.2f}, {result['or_ci_hi']:.2f}]")
    return result


# ── Plotting ──────────────────────────────────────────────────────────

def plot_enrichment_vs_k(all_results: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Fold enrichment vs k
    ax = axes[0]
    for dataset, rows in all_results.items():
        ks = [r["k"] for r in rows]
        folds = [r["fold_enrichment"] for r in rows]
        ci_lo = [r["fold_ci_lo"] for r in rows]
        ci_hi = [r["fold_ci_hi"] for r in rows]
        yerr_lo = [f - lo for f, lo in zip(folds, ci_lo)]
        yerr_hi = [hi - f for f, hi in zip(folds, ci_hi)]
        ax.errorbar(ks, folds, yerr=[yerr_lo, yerr_hi], fmt="o-",
                    capsize=5, markersize=8, linewidth=2, label=dataset)
        for k_val, f_val in zip(ks, folds):
            ax.annotate(f"{f_val:.2f}×", (k_val, f_val), textcoords="offset points",
                        xytext=(8, 8), fontsize=9)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Fold enrichment\n(neighbor / random)", fontsize=12)
    ax.set_title("CORUM co-complex fold enrichment vs k")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Panel 2: Odds ratio vs k
    ax = axes[1]
    for dataset, rows in all_results.items():
        ks = [r["k"] for r in rows]
        ors = [r["odds_ratio"] for r in rows]
        ci_lo = [r["or_ci_lo"] for r in rows]
        ci_hi = [r["or_ci_hi"] for r in rows]
        yerr_lo = [o - lo for o, lo in zip(ors, ci_lo)]
        yerr_hi = [hi - o for o, hi in zip(ors, ci_hi)]
        ax.errorbar(ks, ors, yerr=[yerr_lo, yerr_hi], fmt="s-",
                    capsize=5, markersize=8, linewidth=2, label=dataset)
        for k_val, o_val in zip(ks, ors):
            ax.annotate(f"{o_val:.2f}", (k_val, o_val), textcoords="offset points",
                        xytext=(8, 8), fontsize=9)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Odds ratio", fontsize=12)
    ax.set_title("CORUM co-complex odds ratio vs k")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_enrichment_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_enrichment_vs_k.png")


def plot_sharing_rates(all_results: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dataset, rows in all_results.items():
        ks = [r["k"] for r in rows]
        nb_pct = [r["nb_frac"] * 100 for r in rows]
        rd_pct = [r["rd_frac"] * 100 for r in rows]
        ax.plot(ks, nb_pct, "o-", linewidth=2, markersize=8, label=f"{dataset} neighbors")
        ax.plot(ks, rd_pct, "s--", linewidth=1.5, markersize=6, alpha=0.7, label=f"{dataset} random")

    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("% pairs sharing CORUM complex\n(among CORUM-annotated pairs)", fontsize=11)
    ax.set_title("Co-complex sharing rate vs k")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_sharing_rate_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_sharing_rate_vs_k.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading CORUM complexes...")
    gene_to_cx, co_complex_pairs = load_corum()

    datasets = {
        "DEMETER2": OUT_DIR / "analysis_table.parquet",
        "Chronos": OUT_DIR / "chronos_analysis_table.parquet",
    }

    all_results: dict[str, list[dict]] = {}

    for dataset, analysis_path in datasets.items():
        log.info("=" * 60)
        log.info(f"Processing {dataset}")
        log.info("=" * 60)

        analysis_df = pl.read_parquet(analysis_path)
        variant_ids = analysis_df["variant_id"].to_list()
        gene_names = analysis_df["gene_name"].to_list()
        vid_to_gene = dict(zip(variant_ids, gene_names))

        log.info(f"  Loading embeddings for {len(variant_ids):,} variants...")
        embeddings, vid_order = load_embeddings(variant_ids)

        max_k = max(K_VALUES)
        knn_indices = compute_knn(embeddings, max_k)

        results_for_dataset = []
        for k in K_VALUES:
            rng = np.random.default_rng(RANDOM_SEED + k)
            res = evaluate_k(k, knn_indices, vid_order, vid_to_gene,
                             gene_to_cx, co_complex_pairs, rng)
            res["dataset"] = dataset
            results_for_dataset.append(res)

        all_results[dataset] = results_for_dataset

    # Save
    flat_rows = [r for rows in all_results.values() for r in rows]
    pl.DataFrame(flat_rows).write_parquet(OUT_DIR / "corum_enrichment_vs_k.parquet")
    log.info("  Saved corum_enrichment_vs_k.parquet")

    # Plots
    plot_enrichment_vs_k(all_results)
    plot_sharing_rates(all_results)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed / 60:.1f} minutes")

    # Terminal summary
    print("\n" + "=" * 70)
    print("CORUM ENRICHMENT vs k")
    print("=" * 70)
    print(f"{'Dataset':>10} {'k':>4} {'NB shared/corum':>18} {'NB %':>8} "
          f"{'RD %':>8} {'Fold':>8} {'95% CI':>16} {'OR':>8} {'95% CI':>16}")
    for dataset, rows in all_results.items():
        for r in rows:
            print(f"{dataset:>10} {r['k']:>4d} "
                  f"{r['nb_shared']:>6d}/{r['nb_corum_pairs']:<10d} "
                  f"{r['nb_frac']:>7.3%} {r['rd_frac']:>7.3%} "
                  f"{r['fold_enrichment']:>7.2f}x "
                  f"[{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}] "
                  f"{r['odds_ratio']:>7.2f} "
                  f"[{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}]")
    print(f"\nElapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
