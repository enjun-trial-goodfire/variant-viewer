#!/usr/bin/env python3
"""Gene family/class enrichment for embedding neighbors.

Loads HGNC gene_group annotations, maps to the embedding gene universe,
and evaluates whether top-k neighbors are enriched for shared gene-group
membership relative to matched random controls.

For each k = 5, 10, 20, 50:
  - Global: fraction of cross-gene neighbor pairs sharing ≥ 1 group vs random
  - Fold enrichment + odds ratio with gene-level bootstrap 95% CIs
  - Per-class precision@k: which groups drive the strongest signal

Outputs:
  gene_family_annotations.parquet   — gene → groups mapping (intermediate)
  gene_family_enrichment_vs_k.parquet
  gene_family_per_class.parquet
  gene_family_run_config.json
  gene_family_report.md
  fig_gene_family_enrichment_vs_k.png
  fig_gene_family_top_classes.png

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_gene_family_analysis.py
"""
from __future__ import annotations

import datetime
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
HGNC_PATH = EVEE_ROOT / "data" / "hgnc_complete_set.txt"
KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
MIN_GROUP_SIZE = 5  # minimum genes in kNN to include a group


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── HGNC loading ──────────────────────────────────────────────────────

def load_hgnc_groups(eval_genes: set[str]) -> tuple[
    dict[str, list[str]],       # gene → list of groups
    dict[str, set[str]],        # group → set of genes (in eval_genes)
    set[frozenset[str]],        # co-group gene pairs
    pl.DataFrame,               # annotation table for saving
]:
    """Load HGNC, build gene→groups mapping restricted to eval_genes."""
    hgnc = pl.read_csv(str(HGNC_PATH), separator="\t", infer_schema_length=0)
    gg_df = hgnc.filter(pl.col("gene_group").is_not_null() & (pl.col("gene_group") != ""))

    # Build raw gene→groups
    raw_gene_groups: dict[str, list[str]] = {}
    for sym, groups_str in zip(gg_df["symbol"].to_list(), gg_df["gene_group"].to_list()):
        groups = [g.strip() for g in groups_str.split("|") if g.strip()]
        if groups:
            raw_gene_groups[sym] = groups

    # Restrict to eval_genes
    gene_to_groups: dict[str, list[str]] = {}
    group_to_genes: dict[str, set[str]] = defaultdict(set)
    for g in eval_genes:
        if g in raw_gene_groups:
            gene_to_groups[g] = raw_gene_groups[g]
            for grp in raw_gene_groups[g]:
                group_to_genes[grp].add(g)

    # Filter groups to minimum size
    usable_groups = {grp for grp, genes in group_to_genes.items() if len(genes) >= MIN_GROUP_SIZE}
    group_to_genes = {grp: genes for grp, genes in group_to_genes.items() if grp in usable_groups}

    # Rebuild gene_to_groups with only usable groups
    gene_to_groups_filtered: dict[str, list[str]] = {}
    for g, groups in gene_to_groups.items():
        filtered = [grp for grp in groups if grp in usable_groups]
        if filtered:
            gene_to_groups_filtered[g] = filtered
    gene_to_groups = gene_to_groups_filtered

    # Build co-group pair set
    co_group_pairs: set[frozenset[str]] = set()
    for grp, genes in group_to_genes.items():
        gl = sorted(genes)
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                co_group_pairs.add(frozenset((gl[i], gl[j])))

    log.info(f"  HGNC groups (≥{MIN_GROUP_SIZE} in kNN): {len(group_to_genes)}")
    log.info(f"  Genes with usable groups: {len(gene_to_groups):,}")
    log.info(f"  Co-group gene pairs: {len(co_group_pairs):,}")

    # Annotation table
    ann_rows = []
    for g in sorted(gene_to_groups.keys()):
        for grp in gene_to_groups[g]:
            ann_rows.append({"gene": g, "gene_group": grp, "group_size": len(group_to_genes[grp])})
    ann_df = pl.DataFrame(ann_rows)

    return gene_to_groups, group_to_genes, co_group_pairs, ann_df


# ── Per-k evaluation ──────────────────────────────────────────────────

def evaluate_at_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    annotated_genes: set[str],
    co_group_pairs: set[frozenset[str]],
    rng: np.random.Generator,
) -> dict:
    """Evaluate shared gene-group enrichment at given k."""
    n_variants = len(vid_order)
    ann_list = sorted(annotated_genes)
    n_ann = len(ann_list)

    gene_nb_total: dict[str, int] = defaultdict(int)
    gene_nb_shared: dict[str, int] = defaultdict(int)
    gene_rd_total: dict[str, int] = defaultdict(int)
    gene_rd_shared: dict[str, int] = defaultdict(int)

    for qi in range(n_variants):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in annotated_genes:
            continue

        nb_indices = knn_indices[qi, :k]
        n_valid = 0

        for ni in nb_indices:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in annotated_genes:
                continue
            gene_nb_total[src] += 1
            n_valid += 1
            if frozenset((src, tgt)) in co_group_pairs:
                gene_nb_shared[src] += 1

        for _ in range(n_valid):
            for _a in range(50):
                rg = ann_list[rng.integers(n_ann)]
                if rg != src:
                    break
            gene_rd_total[src] += 1
            if frozenset((src, rg)) in co_group_pairs:
                gene_rd_shared[src] += 1

    common = sorted(set(gene_nb_total.keys()) & set(gene_rd_total.keys()))
    n_genes = len(common)
    nb_total = sum(gene_nb_total[g] for g in common)
    nb_shared = sum(gene_nb_shared.get(g, 0) for g in common)
    rd_total = sum(gene_rd_total[g] for g in common)
    rd_shared = sum(gene_rd_shared.get(g, 0) for g in common)

    nb_frac = nb_shared / nb_total if nb_total > 0 else 0
    rd_frac = rd_shared / rd_total if rd_total > 0 else 0
    fold = nb_frac / rd_frac if rd_frac > 0 else float("inf")
    a, b = nb_shared, nb_total - nb_shared
    c, d = rd_shared, rd_total - rd_shared
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")

    # Bootstrap
    nb_tot_arr = np.array([gene_nb_total[g] for g in common])
    nb_sha_arr = np.array([gene_nb_shared.get(g, 0) for g in common])
    rd_tot_arr = np.array([gene_rd_total[g] for g in common])
    rd_sha_arr = np.array([gene_rd_shared.get(g, 0) for g in common])

    boot_fold = np.empty(N_BOOTSTRAP)
    boot_or = np.empty(N_BOOTSTRAP)
    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        s_nb_t = nb_tot_arr[idx].sum()
        s_nb_s = nb_sha_arr[idx].sum()
        s_rd_t = rd_tot_arr[idx].sum()
        s_rd_s = rd_sha_arr[idx].sum()
        nf = s_nb_s / s_nb_t if s_nb_t > 0 else 0
        rf = s_rd_s / s_rd_t if s_rd_t > 0 else 0
        boot_fold[bi] = nf / rf if rf > 0 else np.nan
        ba, bb = s_nb_s, s_nb_t - s_nb_s
        bc, bd = s_rd_s, s_rd_t - s_rd_s
        boot_or[bi] = (ba * bd) / (bb * bc) if bb > 0 and bc > 0 else np.nan

    fv = boot_fold[~np.isnan(boot_fold)]
    ov = boot_or[~np.isnan(boot_or)]

    def _ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) > 0 else (None, None)

    result = {
        "k": k,
        "n_genes": n_genes,
        "nb_pairs": int(nb_total),
        "nb_shared": int(nb_shared),
        "rd_pairs": int(rd_total),
        "rd_shared": int(rd_shared),
        "nb_frac": float(nb_frac),
        "rd_frac": float(rd_frac),
        "base_rate": float(rd_frac),
        "fold_enrichment": float(fold),
        "fold_ci_lo": _ci(fv)[0],
        "fold_ci_hi": _ci(fv)[1],
        "odds_ratio": float(odds_ratio),
        "or_ci_lo": _ci(ov)[0],
        "or_ci_hi": _ci(ov)[1],
    }

    log.info(f"    k={k:>2d}: nb={nb_shared:,}/{nb_total:,} ({nb_frac:.3%})  "
             f"rd={rd_shared:,}/{rd_total:,} ({rd_frac:.3%})  "
             f"fold={fold:.2f}x [{result['fold_ci_lo']:.2f}, {result['fold_ci_hi']:.2f}]  "
             f"OR={odds_ratio:.2f} [{result['or_ci_lo']:.2f}, {result['or_ci_hi']:.2f}]")
    return result


# ── Per-class precision@k ─────────────────────────────────────────────

def per_class_precision(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    gene_to_groups: dict[str, list[str]],
    group_to_genes: dict[str, set[str]],
    annotated_genes: set[str],
    rng: np.random.Generator,
) -> list[dict]:
    """For each gene group, compute the fraction of neighbor pairs that share that group
    vs random baseline."""
    # Build per-gene retrieved set
    gene_retrieved: dict[str, set[str]] = defaultdict(set)
    for qi in range(len(vid_order)):
        src = vid_to_gene.get(vid_order[qi])
        if src is None or src not in annotated_genes:
            continue
        for ni in knn_indices[qi, :k]:
            tgt = vid_to_gene.get(vid_order[ni])
            if tgt is None or tgt == src or tgt not in annotated_genes:
                continue
            gene_retrieved[src].add(tgt)

    rows = []
    for grp, grp_genes in sorted(group_to_genes.items()):
        grp_set = grp_genes & annotated_genes
        if len(grp_set) < MIN_GROUP_SIZE:
            continue

        # For each gene in this group, how many of its neighbors are also in the group?
        tp_total = 0
        retrieved_total = 0
        n_query_genes = 0

        for g in grp_set:
            if g not in gene_retrieved:
                continue
            retrieved = gene_retrieved[g]
            tp = len(retrieved & grp_set - {g})
            tp_total += tp
            retrieved_total += len(retrieved)
            n_query_genes += 1

        if retrieved_total == 0:
            continue

        precision = tp_total / retrieved_total
        # Random baseline: group has N genes out of M total annotated genes
        # P(random neighbor in group) ≈ (N-1) / (M-1) for a member of the group
        n_ann = len(annotated_genes)
        base_rate = (len(grp_set) - 1) / (n_ann - 1) if n_ann > 1 else 0
        lift = precision / base_rate if base_rate > 0 else float("inf")

        rows.append({
            "gene_group": grp,
            "group_size": len(grp_set),
            "k": k,
            "n_query_genes": n_query_genes,
            "tp": tp_total,
            "retrieved": retrieved_total,
            "precision": precision,
            "base_rate": base_rate,
            "lift": lift,
        })

    return rows


# ── Plotting ──────────────────────────────────────────────────────────

def plot_enrichment_vs_k(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ks = [r["k"] for r in results]

    ax = axes[0]
    folds = [r["fold_enrichment"] for r in results]
    ci_lo = [r["fold_ci_lo"] for r in results]
    ci_hi = [r["fold_ci_hi"] for r in results]
    yerr = [[f - lo for f, lo in zip(folds, ci_lo)],
            [hi - f for f, hi in zip(folds, ci_hi)]]
    ax.errorbar(ks, folds, yerr=yerr, fmt="o-", capsize=5, markersize=9, linewidth=2.5, color="C0")
    for k_val, f_val in zip(ks, folds):
        ax.annotate(f"{f_val:.2f}×", (k_val, f_val), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Fold enrichment\n(neighbor / random)", fontsize=12)
    ax.set_title("Gene family co-membership enrichment")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)

    ax = axes[1]
    nb_pct = [r["nb_frac"] * 100 for r in results]
    rd_pct = [r["rd_frac"] * 100 for r in results]
    ax.plot(ks, nb_pct, "o-", linewidth=2.5, markersize=9, label="Neighbors", color="C0")
    ax.plot(ks, rd_pct, "s--", linewidth=2, markersize=7, label="Random", color="C1")
    for k_val, n, r in zip(ks, nb_pct, rd_pct):
        ax.annotate(f"{n:.1f}%", (k_val, n), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="C0")
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("% pairs sharing gene group", fontsize=12)
    ax.set_title("Co-family sharing rate vs k")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gene_family_enrichment_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_gene_family_enrichment_vs_k.png")


def plot_top_classes(per_class_df: pl.DataFrame, k_plot: int = 10) -> None:
    """Bar chart of top classes by lift at a given k."""
    df = per_class_df.filter(
        (pl.col("k") == k_plot) & (pl.col("group_size") >= 10)
    ).sort("lift", descending=True).head(25)

    if df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    groups = df["gene_group"].to_list()
    lifts = df["lift"].to_list()
    precisions = df["precision"].to_list()
    sizes = df["group_size"].to_list()

    y = np.arange(len(groups))
    bars = ax.barh(y, lifts, color="C0", edgecolor="black", alpha=0.8)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)

    labels = [f"{g} (n={s})" for g, s in zip(groups, sizes)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Lift (precision@k / base rate)", fontsize=11)
    ax.set_title(f"Top 25 gene families by lift at k={k_plot}\n(families with ≥ 10 genes)", fontsize=12)

    for i, (l, p) in enumerate(zip(lifts, precisions)):
        ax.text(l + 0.1, i, f"P={p:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gene_family_top_classes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_gene_family_top_classes.png")


# ── Markdown report ───────────────────────────────────────────────────

def write_report(
    results: list[dict],
    per_class_df: pl.DataFrame,
    n_groups: int,
    n_annotated: int,
    n_co_pairs: int,
    n_knn: int,
) -> None:
    md_path = OUT_DIR / "gene_family_report.md"
    lines = [
        "# Gene Family Co-Membership Enrichment — Embedding Neighbors",
        "",
        "## Setup",
        "",
        f"- **Source:** HGNC gene_group annotations",
        f"- **Gene groups (≥ {MIN_GROUP_SIZE} genes in kNN):** {n_groups}",
        f"- **Annotated genes in kNN:** {n_annotated:,} / {n_knn:,}",
        f"- **Co-group gene pairs:** {n_co_pairs:,}",
        "",
        "**Rules:**",
        "- Self-pairs excluded",
        "- Pairs canonicalized as undirected (sorted gene pair)",
        "- Random controls: matched count per source gene, sampled from annotated genes",
        f"- Bootstrap: {N_BOOTSTRAP:,} iterations over source genes",
        "",
        "## Global Enrichment vs k",
        "",
        "| k | NB pairs | NB shared | NB % | RD % (base rate) | Fold | 95% CI | OR | 95% CI |",
        "|---|---------|-----------|------|------------------|------|--------|-----|--------|",
    ]

    for r in results:
        lines.append(
            f"| {r['k']} | {r['nb_pairs']:,} | {r['nb_shared']:,} | {r['nb_frac']:.3%} "
            f"| {r['rd_frac']:.3%} | {r['fold_enrichment']:.2f}× "
            f"| [{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}] "
            f"| {r['odds_ratio']:.2f} | [{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] |"
        )

    # Top classes at k=10
    top_k = 10
    top_df = per_class_df.filter(
        (pl.col("k") == top_k) & (pl.col("group_size") >= 10)
    ).sort("lift", descending=True).head(20)

    lines += [
        "",
        f"## Top Gene Families by Lift (k={top_k}, groups ≥ 10 genes)",
        "",
        "| Gene Group | Group Size | Precision@k | Base Rate | Lift | TP | Retrieved |",
        "|-----------|-----------|-------------|-----------|------|-----|----------|",
    ]
    for row in top_df.iter_rows(named=True):
        lines.append(
            f"| {row['gene_group']} | {row['group_size']} "
            f"| {row['precision']:.4f} | {row['base_rate']:.4f} "
            f"| {row['lift']:.2f}× | {row['tp']} | {row['retrieved']} |"
        )

    # Bottom classes (weakest signal)
    bot_df = per_class_df.filter(
        (pl.col("k") == top_k) & (pl.col("group_size") >= 10)
    ).sort("lift").head(10)

    lines += [
        "",
        f"## Weakest Gene Families (k={top_k}, groups ≥ 10 genes)",
        "",
        "| Gene Group | Group Size | Precision@k | Base Rate | Lift |",
        "|-----------|-----------|-------------|-----------|------|",
    ]
    for row in bot_df.iter_rows(named=True):
        lines.append(
            f"| {row['gene_group']} | {row['group_size']} "
            f"| {row['precision']:.4f} | {row['base_rate']:.4f} "
            f"| {row['lift']:.2f}× |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "- `fig_gene_family_enrichment_vs_k.png` — Fold enrichment + sharing rate vs k",
        "- `fig_gene_family_top_classes.png` — Top 25 families by lift at k=10",
    ]

    md_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {md_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    log.info(f"  kNN genes: {len(knn_genes):,}")

    # Load HGNC groups
    log.info("Loading HGNC gene_group annotations...")
    gene_to_groups, group_to_genes, co_group_pairs, ann_df = load_hgnc_groups(knn_genes)
    annotated_genes = set(gene_to_groups.keys())

    # Save annotation table
    ann_df.write_parquet(OUT_DIR / "gene_family_annotations.parquet")
    log.info(f"  Saved gene_family_annotations.parquet ({ann_df.height:,} rows)")

    # Evaluate enrichment at each k
    log.info("Evaluating gene-group enrichment per k...")
    results = []
    all_per_class = []
    for k in K_VALUES:
        rng = np.random.default_rng(RANDOM_SEED + k)
        res = evaluate_at_k(
            k, knn_indices, vid_order, vid_to_gene,
            annotated_genes, co_group_pairs, rng,
        )
        results.append(res)

        log.info(f"    Computing per-class precision@{k}...")
        pc_rng = np.random.default_rng(RANDOM_SEED + k + 1000)
        pc = per_class_precision(
            k, knn_indices, vid_order, vid_to_gene,
            gene_to_groups, group_to_genes, annotated_genes, pc_rng,
        )
        all_per_class.extend(pc)

    # Save results
    pl.DataFrame(results).write_parquet(OUT_DIR / "gene_family_enrichment_vs_k.parquet")
    per_class_df = pl.DataFrame(all_per_class).sort(["k", "gene_group"])
    per_class_df.write_parquet(OUT_DIR / "gene_family_per_class.parquet")
    log.info("  Saved gene_family_enrichment_vs_k.parquet + gene_family_per_class.parquet")

    # Run config
    config = {
        "analysis": "gene_family_enrichment",
        "description": "HGNC gene_group co-membership enrichment for embedding neighbors",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "min_group_size": MIN_GROUP_SIZE,
        "hgnc_source": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        "n_groups": len(group_to_genes),
        "n_annotated_genes": len(annotated_genes),
        "n_knn_genes": len(knn_genes),
        "n_co_group_pairs": len(co_group_pairs),
        "self_pairs_excluded": True,
        "pairs_canonicalized_undirected": True,
    }
    config_path = OUT_DIR / "gene_family_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    # Plots + report
    log.info("Generating figures and report...")
    plot_enrichment_vs_k(results)
    plot_top_classes(per_class_df, k_plot=10)
    write_report(results, per_class_df, len(group_to_genes), len(annotated_genes),
                 len(co_group_pairs), len(knn_genes))

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.0f}s")

    # Terminal summary
    print("\n" + "=" * 100)
    print("GENE FAMILY CO-MEMBERSHIP ENRICHMENT — EMBEDDING NEIGHBORS")
    print("=" * 100)
    print(f"HGNC groups (≥{MIN_GROUP_SIZE}): {len(group_to_genes)}  |  "
          f"Annotated genes: {len(annotated_genes):,} / {len(knn_genes):,}  |  "
          f"Co-group pairs: {len(co_group_pairs):,}")
    print()

    print(f"{'k':>4}  {'NB pairs':>10} {'NB shared':>10} {'NB %':>8} "
          f"{'RD % (base)':>12} {'Fold':>8} {'95% CI':>18} {'OR':>8} {'95% CI':>18}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['k']:>4d}  {r['nb_pairs']:>10,} {r['nb_shared']:>10,} "
            f"{r['nb_frac']:>7.3%} {r['rd_frac']:>11.3%} "
            f"{r['fold_enrichment']:>7.2f}x [{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}]  "
            f"{r['odds_ratio']:>7.2f} [{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}]"
        )

    print(f"\nTop 10 gene families by lift (k=10, size≥10):")
    top = per_class_df.filter(
        (pl.col("k") == 10) & (pl.col("group_size") >= 10)
    ).sort("lift", descending=True).head(10)
    print(f"  {'Gene Group':<45s} {'Size':>5} {'P@k':>8} {'Base':>8} {'Lift':>8}")
    for row in top.iter_rows(named=True):
        print(f"  {row['gene_group']:<45s} {row['group_size']:>5d} "
              f"{row['precision']:>8.4f} {row['base_rate']:>8.4f} {row['lift']:>7.2f}x")

    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
