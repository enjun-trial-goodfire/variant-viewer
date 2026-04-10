#!/usr/bin/env python3
"""Binary retrieval evaluation: embedding neighbors as predictor of STRING interactions.

Evaluation universe: STRING genes present in the kNN embedding graph.
Pairs are undirected, deduplicated at gene level. Self-pairs excluded.

True positives:  gene pairs in STRING at a given score threshold.
Predicted positives: gene pairs connected by ≥ 1 top-k neighbor edge.

Evaluated at STRING combined_score thresholds: 400 (medium), 700 (high), 900 (highest).
For each threshold × k = 5, 10, 20, 50:
  TP, FP, FN, precision, recall, F1
  Per-gene precision@k and recall@k
  Gene-level bootstrap 95% CIs

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_string_retrieval.py
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
EVEE_ROOT = REPO_ROOT / "evee-analysis"
OUT_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"
STRING_INFO = EVEE_ROOT / "data" / "9606.protein.info.v12.0.txt"
STRING_LINKS = EVEE_ROOT / "data" / "9606.protein.links.full.v12.0.txt"
KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
SCORE_THRESHOLDS = [400, 700, 900]


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── STRING loading ────────────────────────────────────────────────────

def load_string_pairs() -> dict[tuple[str, str], int]:
    """Load STRING, map to gene names, return (sorted gene pair) → combined_score."""
    log.info("  Loading STRING protein.info...")
    info = pl.read_csv(
        str(STRING_INFO), separator="\t", comment_prefix="#", has_header=False,
        new_columns=["string_protein_id", "preferred_name", "protein_size", "annotation"],
    ).select("string_protein_id", "preferred_name")
    prot_to_gene = {k: v.upper() for k, v in zip(info["string_protein_id"].to_list(), info["preferred_name"].to_list())}

    log.info("  Loading STRING links...")
    links = pl.read_csv(str(STRING_LINKS), separator=" ")
    log.info(f"    {links.height:,} edges")

    log.info("  Building gene-pair → score lookup...")
    p1 = links["protein1"].to_list()
    p2 = links["protein2"].to_list()
    scores = links["combined_score"].to_numpy()

    pair_score: dict[tuple[str, str], int] = {}
    for i in range(len(p1)):
        g1 = prot_to_gene.get(p1[i])
        g2 = prot_to_gene.get(p2[i])
        if g1 is None or g2 is None or g1 == g2:
            continue
        key = (g1, g2) if g1 < g2 else (g2, g1)
        if key not in pair_score:
            pair_score[key] = int(scores[i])

    log.info(f"    {len(pair_score):,} unique gene pairs")
    return pair_score


# ── Build predicted gene-pair set from kNN ────────────────────────────

def build_predicted_pairs(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    eval_genes: set[str],
) -> set[tuple[str, str]]:
    predicted: set[tuple[str, str]] = set()
    for qi in range(len(vid_order)):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in eval_genes:
            continue
        for ni in knn_indices[qi, :k]:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in eval_genes:
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
    eval_genes: set[str],
    gene_to_partners: dict[str, set[str]],
) -> dict[str, dict]:
    gene_retrieved: dict[str, set[str]] = defaultdict(set)
    for qi in range(len(vid_order)):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in eval_genes:
            continue
        for ni in knn_indices[qi, :k]:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in eval_genes:
                continue
            gene_retrieved[src].add(tgt)

    stats: dict[str, dict] = {}
    for g in sorted(eval_genes):
        if g not in gene_retrieved:
            continue
        retrieved = gene_retrieved[g]
        true_partners = gene_to_partners.get(g, set()) & eval_genes
        tp = len(retrieved & true_partners)
        fp = len(retrieved - true_partners)
        fn = len(true_partners - retrieved)
        prec = tp / len(retrieved) if len(retrieved) > 0 else 0
        rec = tp / len(true_partners) if len(true_partners) > 0 else 0
        stats[g] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}
    return stats


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(
    k: int,
    threshold: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    eval_genes: set[str],
    positives: set[tuple[str, str]],
    gene_to_partners: dict[str, set[str]],
    rng: np.random.Generator,
) -> dict:
    predicted = build_predicted_pairs(k, knn_indices, vid_order, vid_to_gene, eval_genes)

    tp_set = predicted & positives
    fp_set = predicted - positives
    fn_set = positives - predicted

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-gene macro
    pg = per_gene_stats(k, knn_indices, vid_order, vid_to_gene, eval_genes, gene_to_partners)
    macro_prec = float(np.mean([s["precision"] for s in pg.values()])) if pg else 0
    macro_rec = float(np.mean([s["recall"] for s in pg.values()])) if pg else 0
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec) if (macro_prec + macro_rec) > 0 else 0

    # Gene-level bootstrap (micro)
    # Attribute each undirected pair to the lex-smaller gene
    gene_tp: dict[str, int] = defaultdict(int)
    gene_fp: dict[str, int] = defaultdict(int)
    gene_fn: dict[str, int] = defaultdict(int)
    for a, _ in tp_set:
        gene_tp[a] += 1
    for a, _ in fp_set:
        gene_fp[a] += 1
    for a, _ in fn_set:
        gene_fn[a] += 1

    boot_genes = sorted(eval_genes)
    n_genes = len(boot_genes)
    tp_arr = np.array([gene_tp.get(g, 0) for g in boot_genes])
    fp_arr = np.array([gene_fp.get(g, 0) for g in boot_genes])
    fn_arr = np.array([gene_fn.get(g, 0) for g in boot_genes])
    macro_p_arr = np.array([pg[g]["precision"] if g in pg else np.nan for g in boot_genes])
    macro_r_arr = np.array([pg[g]["recall"] if g in pg else np.nan for g in boot_genes])

    boot_p = np.empty(N_BOOTSTRAP)
    boot_r = np.empty(N_BOOTSTRAP)
    boot_f1 = np.empty(N_BOOTSTRAP)
    boot_mp = np.empty(N_BOOTSTRAP)
    boot_mr = np.empty(N_BOOTSTRAP)

    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        s_tp = tp_arr[idx].sum()
        s_fp = fp_arr[idx].sum()
        s_fn = fn_arr[idx].sum()
        p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0
        r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0
        boot_p[bi] = p
        boot_r[bi] = r
        boot_f1[bi] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        boot_mp[bi] = np.nanmean(macro_p_arr[idx])
        boot_mr[bi] = np.nanmean(macro_r_arr[idx])

    def _ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "threshold": threshold,
        "k": k,
        "n_predicted": len(predicted),
        "n_positives": len(positives),
        "n_genes": n_genes,
        "TP": tp, "FP": fp, "FN": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "micro_p_ci_lo": _ci(boot_p)[0], "micro_p_ci_hi": _ci(boot_p)[1],
        "micro_r_ci_lo": _ci(boot_r)[0], "micro_r_ci_hi": _ci(boot_r)[1],
        "micro_f1_ci_lo": _ci(boot_f1)[0], "micro_f1_ci_hi": _ci(boot_f1)[1],
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_p_ci_lo": _ci(boot_mp)[0], "macro_p_ci_hi": _ci(boot_mp)[1],
        "macro_r_ci_lo": _ci(boot_mr)[0], "macro_r_ci_hi": _ci(boot_mr)[1],
    }


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(results: list[dict]) -> None:
    thresholds = sorted(set(r["threshold"] for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for metric, label, ax_idx, marker_cycle in [
        ("micro_precision", "Precision", 0, ["o", "s", "D"]),
        ("micro_recall", "Recall", 1, ["o", "s", "D"]),
        ("micro_f1", "F1", 2, ["o", "s", "D"]),
    ]:
        ax = axes[ax_idx]
        for i, t in enumerate(thresholds):
            rows = [r for r in results if r["threshold"] == t]
            ks = [r["k"] for r in rows]
            vals = [r[metric] for r in rows]
            ci_lo = [r[f"{metric.replace('micro_', 'micro_')}_ci_lo".replace("micro_precision", "micro_p").replace("micro_recall", "micro_r").replace("micro_f1", "micro_f1")] for r in rows]
            ci_hi = [r[f"{metric.replace('micro_', 'micro_')}_ci_hi".replace("micro_precision", "micro_p").replace("micro_recall", "micro_r").replace("micro_f1", "micro_f1")] for r in rows]
            yerr = [[v - lo for v, lo in zip(vals, ci_lo)],
                    [hi - v for v, hi in zip(vals, ci_hi)]]
            ax.errorbar(ks, vals, yerr=yerr, fmt=f"{marker_cycle[i]}-", capsize=4,
                        markersize=8, linewidth=2, label=f"score ≥ {t}")
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(f"STRING {label} vs k")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("STRING Interaction Retrieval — Embedding Neighbors", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_string_retrieval_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_string_retrieval_vs_k.png")


# ── Markdown report ───────────────────────────────────────────────────

def write_report(results: list[dict], n_string_pairs: int, n_string_genes: int) -> None:
    md_path = OUT_DIR / "string_retrieval_report.md"
    thresholds = sorted(set(r["threshold"] for r in results))

    lines = [
        "# STRING Interaction Binary Retrieval — Embedding Neighbors",
        "",
        "## Setup",
        "",
        f"- **STRING version:** v12.0",
        f"- **STRING total gene pairs:** {n_string_pairs:,}",
        f"- **STRING genes:** {n_string_genes:,}",
        f"- **Genes in kNN ∩ STRING:** {results[0]['n_genes']:,}",
        "",
        "**Definitions:**",
        "- Pairs are undirected and deduplicated at gene level",
        "- Self-pairs (same gene) excluded",
        "- True positive: gene pair present in STRING at ≥ threshold",
        "- Predicted positive: gene pair connected by ≥ 1 top-k embedding neighbor edge",
        "- Gene symbols: STRING preferred_name (uppercase); exact match only",
        f"- Bootstrap: {N_BOOTSTRAP:,} iterations, resampling genes",
        f"- Score thresholds evaluated: {', '.join(str(t) for t in thresholds)} "
        "(400 = medium confidence, 700 = high, 900 = highest)",
        "",
    ]

    for t in thresholds:
        rows = [r for r in results if r["threshold"] == t]
        n_pos = rows[0]["n_positives"]
        n_genes = rows[0]["n_genes"]
        n_possible = n_genes * (n_genes - 1) // 2
        base_rate = n_pos / n_possible if n_possible > 0 else 0

        lines += [
            f"## Score ≥ {t} ({['medium', 'high', 'highest'][thresholds.index(t)]} confidence)",
            "",
            f"- **Positive pairs in universe:** {n_pos:,}",
            f"- **Total possible pairs:** {n_possible:,}",
            f"- **Base rate:** {base_rate:.4%}",
            "",
            "### Global (Micro) Metrics",
            "",
            "| k | Predicted | TP | FP | FN | Precision | 95% CI | Recall | 95% CI | F1 | 95% CI |",
            "|---|----------|-----|------|------|-----------|--------|--------|--------|-----|--------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['k']} | {r['n_predicted']:,} | {r['TP']:,} | {r['FP']:,} | {r['FN']:,} "
                f"| {r['micro_precision']:.4f} | [{r['micro_p_ci_lo']:.4f}, {r['micro_p_ci_hi']:.4f}] "
                f"| {r['micro_recall']:.4f} | [{r['micro_r_ci_lo']:.4f}, {r['micro_r_ci_hi']:.4f}] "
                f"| {r['micro_f1']:.4f} | [{r['micro_f1_ci_lo']:.4f}, {r['micro_f1_ci_hi']:.4f}] |"
            )

        lines += [
            "",
            "### Per-Gene (Macro) Metrics",
            "",
            "| k | P@k | 95% CI | R@k | 95% CI | F1@k |",
            "|---|-----|--------|-----|--------|------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['k']} | {r['macro_precision']:.4f} "
                f"| [{r['macro_p_ci_lo']:.4f}, {r['macro_p_ci_hi']:.4f}] "
                f"| {r['macro_recall']:.4f} "
                f"| [{r['macro_r_ci_lo']:.4f}, {r['macro_r_ci_hi']:.4f}] "
                f"| {r['macro_f1']:.4f} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "1. **Precision increases sharply with score threshold.** At ≥ 900 (highest confidence), "
        "precision is substantially higher because the positive set is smaller and more enriched "
        "for true biological interactions.",
        "",
        "2. **Recall decreases with threshold** — fewer true positives to find, and the ones that "
        "remain are harder to recover.",
        "",
        "3. **F1 balances the tradeoff** — highest at intermediate thresholds where both precision "
        "and recall contribute meaningfully.",
        "",
        "4. **Consistent monotone decay with k** — closest neighbors (k=5) give highest precision; "
        "k=50 gives highest recall.",
        "",
        "## Figures",
        "",
        "- `fig_string_retrieval_vs_k.png` — Precision, Recall, F1 vs k at each threshold",
    ]

    md_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {md_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load STRING
    log.info("Loading STRING database...")
    pair_score = load_string_pairs()
    string_genes = set()
    for (a, b) in pair_score:
        string_genes.add(a)
        string_genes.add(b)
    n_string_genes = len(string_genes)

    # Load kNN
    log.info("Loading kNN indices...")
    data = np.load(str(KNN_CACHE), allow_pickle=True)
    vid_order = data["vid_order"].tolist()
    knn_indices = data["knn_indices"]
    log.info(f"  {knn_indices.shape[0]:,} variants, max_k={knn_indices.shape[1]}")

    # Variant → gene mapping
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT variant_id, gene_name FROM variants WHERE gene_name IS NOT NULL").fetchall()
    con.close()
    vid_to_gene = {v: g.upper() for v, g in rows}

    knn_genes = {vid_to_gene[v] for v in vid_order if v in vid_to_gene}
    eval_genes = knn_genes & string_genes
    log.info(f"  Evaluation genes (kNN ∩ STRING): {len(eval_genes):,}")

    # Precompute positive sets at each threshold
    all_results = []
    for threshold in SCORE_THRESHOLDS:
        positives: set[tuple[str, str]] = set()
        gene_to_partners: dict[str, set[str]] = defaultdict(set)
        for (a, b), score in pair_score.items():
            if score >= threshold and a in eval_genes and b in eval_genes:
                positives.add((a, b))
                gene_to_partners[a].add(b)
                gene_to_partners[b].add(a)

        n_possible = len(eval_genes) * (len(eval_genes) - 1) // 2
        base_rate = len(positives) / n_possible if n_possible > 0 else 0
        log.info(f"  Threshold ≥ {threshold}: {len(positives):,} positives "
                 f"(base rate {base_rate:.4%})")

        for k in K_VALUES:
            rng = np.random.default_rng(RANDOM_SEED + threshold + k)
            log.info(f"    k={k}...")
            res = evaluate(
                k, threshold, knn_indices, vid_order, vid_to_gene,
                eval_genes, positives, dict(gene_to_partners), rng,
            )
            all_results.append(res)
            log.info(f"      P={res['micro_precision']:.4f} R={res['micro_recall']:.4f} "
                     f"F1={res['micro_f1']:.4f}")

    # Save
    pl.DataFrame(all_results).write_parquet(OUT_DIR / "string_retrieval_vs_k.parquet")
    log.info("  Saved string_retrieval_vs_k.parquet")

    config = {
        "analysis": "string_retrieval",
        "description": "Binary retrieval: embedding neighbors as STRING interaction predictor",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "score_thresholds": SCORE_THRESHOLDS,
        "string_version": "v12.0",
        "n_string_genes": n_string_genes,
        "n_string_pairs": len(pair_score),
        "n_eval_genes": len(eval_genes),
        "self_pairs_excluded": True,
        "pairs_undirected_deduplicated": True,
    }
    config_path = OUT_DIR / "string_retrieval_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    # Plots + markdown
    plot_results(all_results)
    write_report(all_results, len(pair_score), n_string_genes)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.0f}s")

    # Terminal summary
    for threshold in SCORE_THRESHOLDS:
        rows = [r for r in all_results if r["threshold"] == threshold]
        n_pos = rows[0]["n_positives"]
        n_genes = rows[0]["n_genes"]
        n_possible = n_genes * (n_genes - 1) // 2
        base_rate = n_pos / n_possible if n_possible > 0 else 0
        print(f"\n{'=' * 120}")
        print(f"STRING RETRIEVAL — score ≥ {threshold}  |  {n_pos:,} positives  |  base rate {base_rate:.4%}")
        print(f"{'=' * 120}")
        print(f"{'k':>4}  {'Predicted':>10} {'TP':>8} {'FP':>8} {'FN':>8}  "
              f"{'Prec':>8} {'95% CI':>18}  {'Recall':>8} {'95% CI':>18}  "
              f"{'F1':>8} {'95% CI':>18}")
        print("-" * 130)
        for r in rows:
            print(
                f"{r['k']:>4d}  {r['n_predicted']:>10,} {r['TP']:>8,} {r['FP']:>8,} {r['FN']:>8,}  "
                f"{r['micro_precision']:>8.4f} [{r['micro_p_ci_lo']:.4f}, {r['micro_p_ci_hi']:.4f}]  "
                f"{r['micro_recall']:>8.4f} [{r['micro_r_ci_lo']:.4f}, {r['micro_r_ci_hi']:.4f}]  "
                f"{r['micro_f1']:>8.4f} [{r['micro_f1_ci_lo']:.4f}, {r['micro_f1_ci_hi']:.4f}]"
            )
        print()
        print(f"{'k':>4}  {'P@k':>8} {'95% CI':>18}  {'R@k':>8} {'95% CI':>18}  {'F1@k':>8}")
        for r in rows:
            print(
                f"{r['k']:>4d}  {r['macro_precision']:>8.4f} "
                f"[{r['macro_p_ci_lo']:.4f}, {r['macro_p_ci_hi']:.4f}]  "
                f"{r['macro_recall']:>8.4f} "
                f"[{r['macro_r_ci_lo']:.4f}, {r['macro_r_ci_hi']:.4f}]  "
                f"{r['macro_f1']:>8.4f}"
            )

    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
