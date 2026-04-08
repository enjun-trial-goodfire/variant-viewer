#!/usr/bin/env python3
"""Binary retrieval evaluation: embedding neighbors as predictor of CORUM co-complex membership.

Evaluation universe: CORUM genes (complexes ≥ 3) present in the kNN graph.
Pairs are undirected, deduplicated at gene level. Self-pairs excluded.

True positives:  gene pairs sharing ≥ 1 CORUM complex.
Predicted positives: gene pairs connected by ≥ 1 top-k neighbor edge.

For k = 5, 10, 20, 50:
  Global (micro):  TP, FP, FN, precision, recall, F1
  Per-gene (macro): precision@k, recall@k (averaged over query genes)

Gene-level bootstrap (5000 iter) for 95% CIs on precision, recall, F1.

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_corum_retrieval.py
"""
from __future__ import annotations

import datetime
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
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EEVE_ROOT = REPO_ROOT / "eeve-analysis"
OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"
CORUM_PATH = EEVE_ROOT / "data" / "corum_humanComplexes.json"
KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
MIN_COMPLEX_SIZE = 3


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── CORUM ─────────────────────────────────────────────────────────────

def load_corum() -> tuple[set[tuple[str, str]], dict[str, set[str]], int]:
    """Load CORUM, return (positive_pairs, gene→set-of-co-complex-partners, n_complexes_kept)."""
    raw = json.loads(CORUM_PATH.read_text())
    gene_to_partners: dict[str, set[str]] = defaultdict(set)
    positive_pairs: set[tuple[str, str]] = set()
    n_kept = 0

    for cx in raw:
        genes: set[str] = set()
        for su in cx["subunits"]:
            gn = su.get("swissprot", {}).get("gene_name")
            if gn:
                genes.add(gn.upper())
        if len(genes) < MIN_COMPLEX_SIZE:
            continue
        n_kept += 1
        gl = sorted(genes)
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                positive_pairs.add((gl[i], gl[j]))
                gene_to_partners[gl[i]].add(gl[j])
                gene_to_partners[gl[j]].add(gl[i])

    return positive_pairs, dict(gene_to_partners), n_kept


# ── Build gene-pair graph from kNN at given k ─────────────────────────

def build_predicted_pairs(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    corum_genes: set[str],
) -> set[tuple[str, str]]:
    """Extract undirected, deduplicated, cross-gene pairs from top-k neighbors.

    Only includes pairs where BOTH genes are in corum_genes.
    """
    predicted: set[tuple[str, str]] = set()
    for qi in range(len(vid_order)):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in corum_genes:
            continue
        for ni in knn_indices[qi, :k]:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in corum_genes:
                continue
            pair = (src, tgt) if src < tgt else (tgt, src)
            predicted.add(pair)
    return predicted


# ── Per-gene retrieval stats ──────────────────────────────────────────

def per_gene_stats(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    corum_genes: set[str],
    gene_to_partners: dict[str, set[str]],
) -> dict[str, dict]:
    """For each CORUM gene in the kNN graph, compute its retrieved set and
    per-gene TP/FP/FN, precision, recall.

    A gene's "retrieved set" is the union of cross-gene CORUM neighbors
    across all its variants' top-k lists (deduplicated).
    """
    gene_retrieved: dict[str, set[str]] = defaultdict(set)

    for qi in range(len(vid_order)):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in corum_genes:
            continue
        for ni in knn_indices[qi, :k]:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in corum_genes:
                continue
            gene_retrieved[src].add(tgt)

    stats: dict[str, dict] = {}
    for g in sorted(corum_genes):
        if g not in gene_retrieved:
            continue
        retrieved = gene_retrieved[g]
        true_partners = gene_to_partners.get(g, set()) & corum_genes
        tp = len(retrieved & true_partners)
        fp = len(retrieved - true_partners)
        fn = len(true_partners - retrieved)
        n_retrieved = len(retrieved)
        n_true = len(true_partners)
        prec = tp / n_retrieved if n_retrieved > 0 else 0
        rec = tp / n_true if n_true > 0 else 0
        stats[g] = {
            "tp": tp, "fp": fp, "fn": fn,
            "n_retrieved": n_retrieved, "n_true": n_true,
            "precision": prec, "recall": rec,
        }
    return stats


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_at_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    corum_genes: set[str],
    positive_pairs: set[tuple[str, str]],
    gene_to_partners: dict[str, set[str]],
    rng: np.random.Generator,
) -> dict:
    """Full evaluation at a given k."""
    log.info(f"  Evaluating k={k}...")

    # Global (micro) metrics
    predicted = build_predicted_pairs(k, knn_indices, vid_order, vid_to_gene, corum_genes)

    # Restrict positives to pairs where both genes appear in the kNN universe
    genes_in_knn = {vid_to_gene[v] for v in vid_order if vid_to_gene.get(v) in corum_genes}
    positives_in_universe = {(a, b) for a, b in positive_pairs
                             if a in genes_in_knn and b in genes_in_knn}

    tp_set = predicted & positives_in_universe
    fp_set = predicted - positives_in_universe
    fn_set = positives_in_universe - predicted

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    log.info(f"    Global: TP={tp:,} FP={fp:,} FN={fn:,} "
             f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    log.info(f"    Predicted pairs: {len(predicted):,}  "
             f"True positives in universe: {len(positives_in_universe):,}")

    # Per-gene (macro) metrics
    pg = per_gene_stats(k, knn_indices, vid_order, vid_to_gene, corum_genes, gene_to_partners)
    precisions = [s["precision"] for s in pg.values()]
    recalls = [s["recall"] for s in pg.values()]
    macro_prec = float(np.mean(precisions)) if precisions else 0
    macro_rec = float(np.mean(recalls)) if recalls else 0
    macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                if (macro_prec + macro_rec) > 0 else 0)

    log.info(f"    Macro (per-gene avg over {len(pg)} genes): "
             f"P@{k}={macro_prec:.4f} R@{k}={macro_rec:.4f} F1@{k}={macro_f1:.4f}")

    # Gene-level bootstrap for micro precision, recall, F1
    # Precompute per-gene TP/FP/FN contributions to the GLOBAL counts.
    # For undirected pairs, attribute each pair to the lexicographically
    # smaller gene to avoid double-counting.
    gene_global_tp: dict[str, int] = defaultdict(int)
    gene_global_fp: dict[str, int] = defaultdict(int)
    gene_global_fn: dict[str, int] = defaultdict(int)

    for a, b in tp_set:
        gene_global_tp[a] += 1  # a < b by construction
    for a, b in fp_set:
        gene_global_fp[a] += 1
    for a, b in fn_set:
        gene_global_fn[a] += 1

    boot_genes = sorted(genes_in_knn)
    n_genes = len(boot_genes)
    tp_arr = np.array([gene_global_tp.get(g, 0) for g in boot_genes])
    fp_arr = np.array([gene_global_fp.get(g, 0) for g in boot_genes])
    fn_arr = np.array([gene_global_fn.get(g, 0) for g in boot_genes])

    # Also per-gene macro stats
    macro_prec_arr = np.array([pg[g]["precision"] if g in pg else np.nan for g in boot_genes])
    macro_rec_arr = np.array([pg[g]["recall"] if g in pg else np.nan for g in boot_genes])

    boot_micro_p = np.empty(N_BOOTSTRAP)
    boot_micro_r = np.empty(N_BOOTSTRAP)
    boot_micro_f1 = np.empty(N_BOOTSTRAP)
    boot_macro_p = np.empty(N_BOOTSTRAP)
    boot_macro_r = np.empty(N_BOOTSTRAP)

    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        s_tp = tp_arr[idx].sum()
        s_fp = fp_arr[idx].sum()
        s_fn = fn_arr[idx].sum()
        p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0
        r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0
        boot_micro_p[bi] = p
        boot_micro_r[bi] = r
        boot_micro_f1[bi] = 2 * p * r / (p + r) if (p + r) > 0 else 0

        mp = macro_prec_arr[idx]
        mr = macro_rec_arr[idx]
        boot_macro_p[bi] = np.nanmean(mp)
        boot_macro_r[bi] = np.nanmean(mr)

    def _ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    result = {
        "k": k,
        "n_predicted_pairs": len(predicted),
        "n_positive_pairs_in_universe": len(positives_in_universe),
        "n_genes_in_universe": len(genes_in_knn),
        "TP": tp, "FP": fp, "FN": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "micro_precision_ci_lo": _ci(boot_micro_p)[0],
        "micro_precision_ci_hi": _ci(boot_micro_p)[1],
        "micro_recall_ci_lo": _ci(boot_micro_r)[0],
        "micro_recall_ci_hi": _ci(boot_micro_r)[1],
        "micro_f1_ci_lo": _ci(boot_micro_f1)[0],
        "micro_f1_ci_hi": _ci(boot_micro_f1)[1],
        "macro_precision_at_k": macro_prec,
        "macro_recall_at_k": macro_rec,
        "macro_f1_at_k": macro_f1,
        "macro_precision_ci_lo": _ci(boot_macro_p)[0],
        "macro_precision_ci_hi": _ci(boot_macro_p)[1],
        "macro_recall_ci_lo": _ci(boot_macro_r)[0],
        "macro_recall_ci_hi": _ci(boot_macro_r)[1],
    }

    log.info(f"    Bootstrap CIs: micro P=[{result['micro_precision_ci_lo']:.4f}, "
             f"{result['micro_precision_ci_hi']:.4f}]  "
             f"R=[{result['micro_recall_ci_lo']:.4f}, {result['micro_recall_ci_hi']:.4f}]  "
             f"F1=[{result['micro_f1_ci_lo']:.4f}, {result['micro_f1_ci_hi']:.4f}]")
    return result


# ── Plotting ──────────────────────────────────────────────────────────

def plot_retrieval_curves(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ks = [r["k"] for r in results]

    # Panel 1: Micro precision / recall / F1 vs k
    ax = axes[0]
    for metric, label, color, marker in [
        ("micro_precision", "Precision", "C0", "o"),
        ("micro_recall", "Recall", "C1", "s"),
        ("micro_f1", "F1", "C2", "D"),
    ]:
        vals = [r[metric] for r in results]
        ci_lo = [r[f"{metric}_ci_lo"] for r in results]
        ci_hi = [r[f"{metric}_ci_hi"] for r in results]
        yerr = [[v - lo for v, lo in zip(vals, ci_lo)],
                [hi - v for v, hi in zip(vals, ci_hi)]]
        ax.errorbar(ks, vals, yerr=yerr, fmt=f"{marker}-", capsize=4,
                    markersize=8, linewidth=2, color=color, label=label)
        for k_val, v in zip(ks, vals):
            ax.annotate(f"{v:.4f}", (k_val, v), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color=color)
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Score")
    ax.set_title("Micro (global) retrieval metrics")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Macro precision@k / recall@k
    ax = axes[1]
    for metric, label, color, marker in [
        ("macro_precision_at_k", "Precision@k", "C0", "o"),
        ("macro_recall_at_k", "Recall@k", "C1", "s"),
        ("macro_f1_at_k", "F1@k", "C2", "D"),
    ]:
        vals = [r[metric] for r in results]
        ci_key_lo = metric.replace("_at_k", "_ci_lo").replace("macro_f1_at_k", "")
        if f"macro_{metric.split('_')[1]}_ci_lo" in results[0]:
            ci_lo = [r[f"macro_{metric.split('macro_')[1].split('_at_k')[0]}_ci_lo"] for r in results]
            ci_hi = [r[f"macro_{metric.split('macro_')[1].split('_at_k')[0]}_ci_hi"] for r in results]
            yerr = [[v - lo for v, lo in zip(vals, ci_lo)],
                    [hi - v for v, hi in zip(vals, ci_hi)]]
            ax.errorbar(ks, vals, yerr=yerr, fmt=f"{marker}-", capsize=4,
                        markersize=8, linewidth=2, color=color, label=label)
        else:
            ax.plot(ks, vals, f"{marker}-", markersize=8, linewidth=2, color=color, label=label)
        for k_val, v in zip(ks, vals):
            ax.annotate(f"{v:.4f}", (k_val, v), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color=color)
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Score")
    ax.set_title("Macro (per-gene avg) retrieval metrics")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: TP / FP / FN counts
    ax = axes[2]
    tps = [r["TP"] for r in results]
    fps = [r["FP"] for r in results]
    fns = [r["FN"] for r in results]
    x = np.arange(len(ks))
    w = 0.25
    ax.bar(x - w, tps, w, label="TP", color="C2", edgecolor="black")
    ax.bar(x, fps, w, label="FP", color="C3", edgecolor="black")
    ax.bar(x + w, fns, w, label="FN", color="C7", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Count")
    ax.set_title("TP / FP / FN")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, (t, f, n) in enumerate(zip(tps, fps, fns)):
        ax.text(i - w, t + 200, f"{t:,}", ha="center", fontsize=7)
        ax.text(i, f + 200, f"{f:,}", ha="center", fontsize=7)
        ax.text(i + w, n + 200, f"{n:,}", ha="center", fontsize=7)

    fig.suptitle("CORUM Co-complex Retrieval via Embedding Neighbors", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_retrieval_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_retrieval_vs_k.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load CORUM
    log.info("Loading CORUM (complexes ≥ %d genes)...", MIN_COMPLEX_SIZE)
    positive_pairs, gene_to_partners, n_cx = load_corum()
    corum_genes = set(gene_to_partners.keys())
    log.info(f"  {n_cx} complexes, {len(corum_genes)} genes, "
             f"{len(positive_pairs):,} positive pairs")

    # Load kNN
    log.info("Loading kNN indices...")
    data = np.load(str(KNN_CACHE), allow_pickle=True)
    vid_order = data["vid_order"].tolist()
    knn_indices = data["knn_indices"]
    log.info(f"  {knn_indices.shape[0]:,} variants, max_k={knn_indices.shape[1]}")

    # Load variant→gene mapping
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT variant_id, gene_name FROM variants WHERE gene_name IS NOT NULL").fetchall()
    con.close()
    vid_to_gene = {v: g.upper() for v, g in rows}

    # Evaluate
    results = []
    for k in K_VALUES:
        rng = np.random.default_rng(RANDOM_SEED + k)
        res = evaluate_at_k(
            k, knn_indices, vid_order, vid_to_gene, corum_genes,
            positive_pairs, gene_to_partners, rng,
        )
        results.append(res)

    # Save
    pl.DataFrame(results).write_parquet(OUT_DIR / "corum_retrieval_vs_k.parquet")
    log.info("  Saved corum_retrieval_vs_k.parquet")

    config = {
        "analysis": "corum_retrieval",
        "description": "Binary retrieval: embedding neighbors as CORUM co-complex predictor",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "min_complex_size": MIN_COMPLEX_SIZE,
        "n_complexes": n_cx,
        "n_corum_genes": len(corum_genes),
        "n_positive_pairs_total": len(positive_pairs),
        "pairs_undirected_deduplicated": True,
        "self_pairs_excluded": True,
    }
    config_path = OUT_DIR / "corum_retrieval_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    log.info(f"  Saved {config_path.name}")

    # Plot
    plot_retrieval_curves(results)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.0f}s")

    # Terminal summary
    print("\n" + "=" * 110)
    print("CORUM CO-COMPLEX BINARY RETRIEVAL — EMBEDDING NEIGHBORS")
    print("=" * 110)
    print(f"Universe: {results[0]['n_genes_in_universe']} CORUM genes in kNN graph  |  "
          f"{results[0]['n_positive_pairs_in_universe']:,} positive pairs")
    print()

    print(f"{'k':>4}  {'Predicted':>10} {'TP':>8} {'FP':>8} {'FN':>8}  "
          f"{'Precision':>10} {'95% CI':>18}  "
          f"{'Recall':>10} {'95% CI':>18}  "
          f"{'F1':>10} {'95% CI':>18}")
    print("-" * 135)
    for r in results:
        print(
            f"{r['k']:>4d}  {r['n_predicted_pairs']:>10,} "
            f"{r['TP']:>8,} {r['FP']:>8,} {r['FN']:>8,}  "
            f"{r['micro_precision']:>10.4f} "
            f"[{r['micro_precision_ci_lo']:.4f}, {r['micro_precision_ci_hi']:.4f}]  "
            f"{r['micro_recall']:>10.4f} "
            f"[{r['micro_recall_ci_lo']:.4f}, {r['micro_recall_ci_hi']:.4f}]  "
            f"{r['micro_f1']:>10.4f} "
            f"[{r['micro_f1_ci_lo']:.4f}, {r['micro_f1_ci_hi']:.4f}]"
        )

    print()
    print(f"{'k':>4}  {'P@k':>10} {'95% CI':>18}  {'R@k':>10} {'95% CI':>18}  {'F1@k':>10}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['k']:>4d}  {r['macro_precision_at_k']:>10.4f} "
            f"[{r['macro_precision_ci_lo']:.4f}, {r['macro_precision_ci_hi']:.4f}]  "
            f"{r['macro_recall_at_k']:>10.4f} "
            f"[{r['macro_recall_ci_lo']:.4f}, {r['macro_recall_ci_hi']:.4f}]  "
            f"{r['macro_f1_at_k']:>10.4f}"
        )

    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
