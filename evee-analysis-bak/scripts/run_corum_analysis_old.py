#!/usr/bin/env python3
"""CORUM complex co-membership analysis for embedding neighbors.

For each cross-gene neighbor pair and matched random pair, check whether
both genes belong to at least one shared CORUM complex.  Report fold
enrichment and odds ratio with gene-level bootstrap CIs.

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_corum_analysis.py
"""
from __future__ import annotations

import json
import logging
import os
import random
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

RANDOM_SEED = 42
N_BOOTSTRAP = 5000


# ── CORUM loading ─────────────────────────────────────────────────────

def load_corum() -> tuple[dict[str, set[int]], dict[int, set[str]], int]:
    """Returns (gene→set-of-complex-ids, complex-id→set-of-genes, n_complexes)."""
    raw = json.loads(CORUM_PATH.read_text())
    gene_to_cx: dict[str, set[int]] = defaultdict(set)
    cx_to_genes: dict[int, set[str]] = defaultdict(set)

    for cx in raw:
        cid = cx["complex_id"]
        for su in cx["subunits"]:
            gn = su.get("swissprot", {}).get("gene_name")
            if gn:
                gene_to_cx[gn].add(cid)
                cx_to_genes[cid].add(gn)

    log.info(f"  CORUM: {len(raw)} complexes, {len(gene_to_cx)} unique genes")
    return dict(gene_to_cx), dict(cx_to_genes), len(raw)


def genes_share_complex(gene_a: str, gene_b: str, gene_to_cx: dict[str, set[int]]) -> bool:
    cx_a = gene_to_cx.get(gene_a)
    cx_b = gene_to_cx.get(gene_b)
    if cx_a is None or cx_b is None:
        return False
    return bool(cx_a & cx_b)


# ── Core analysis ─────────────────────────────────────────────────────

def compute_sharing_stats(
    pair_df: pl.DataFrame,
    gene_to_cx: dict[str, set[int]],
    pair_type: str,
) -> dict:
    """Compute fraction of pairs sharing a CORUM complex."""
    rows = pair_df.filter(pl.col("pair_type") == pair_type)
    src_genes = rows["source_gene"].to_list()
    tgt_genes = rows["target_gene"].to_list()

    n_total = 0
    n_shared = 0
    n_both_in_corum = 0

    for sg, tg in zip(src_genes, tgt_genes):
        n_total += 1
        in_corum_a = sg in gene_to_cx
        in_corum_b = tg in gene_to_cx
        if in_corum_a and in_corum_b:
            n_both_in_corum += 1
            if gene_to_cx[sg] & gene_to_cx[tg]:
                n_shared += 1

    frac_all = n_shared / n_total if n_total > 0 else 0
    frac_corum = n_shared / n_both_in_corum if n_both_in_corum > 0 else 0

    return {
        "pair_type": pair_type,
        "n_total": n_total,
        "n_both_in_corum": n_both_in_corum,
        "n_shared_complex": n_shared,
        "frac_of_all": frac_all,
        "frac_of_corum_pairs": frac_corum,
    }


def bootstrap_by_source_gene(
    pair_df: pl.DataFrame,
    gene_to_cx: dict[str, set[int]],
    n_boot: int,
) -> dict:
    """Bootstrap enrichment and odds ratio by resampling source genes.

    Precomputes per-gene (n_corum, n_shared) counts for both neighbor and
    random pair types, then each bootstrap iteration is just vectorized
    array indexing + summation.
    """
    nb_df = pair_df.filter(pl.col("pair_type") == "neighbor_cross_gene")
    rd_df = pair_df.filter(pl.col("pair_type") == "random_cross_gene")

    def _per_gene_counts(df: pl.DataFrame) -> dict[str, tuple[int, int]]:
        counts: dict[str, tuple[int, int]] = {}
        by_gene: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for sg, tg in zip(df["source_gene"].to_list(), df["target_gene"].to_list()):
            by_gene[sg].append((sg, tg))
        for g, pairs in by_gene.items():
            n_corum = 0
            n_shared = 0
            for sg, tg in pairs:
                if sg in gene_to_cx and tg in gene_to_cx:
                    n_corum += 1
                    if gene_to_cx[sg] & gene_to_cx[tg]:
                        n_shared += 1
            counts[g] = (n_corum, n_shared)
        return counts

    log.info(f"    Precomputing per-gene CORUM counts...")
    nb_counts = _per_gene_counts(nb_df)
    rd_counts = _per_gene_counts(rd_df)

    common_genes = sorted(set(nb_counts.keys()) & set(rd_counts.keys()))
    n_genes = len(common_genes)
    log.info(f"    Bootstrap: {n_genes} source genes, {n_boot} iterations")

    nb_corum_arr = np.array([nb_counts[g][0] for g in common_genes])
    nb_shared_arr = np.array([nb_counts[g][1] for g in common_genes])
    rd_corum_arr = np.array([rd_counts[g][0] for g in common_genes])
    rd_shared_arr = np.array([rd_counts[g][1] for g in common_genes])

    rng = np.random.default_rng(RANDOM_SEED)
    fold_enrichments = np.empty(n_boot)
    odds_ratios = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n_genes, size=n_genes)

        nb_corum = nb_corum_arr[idx].sum()
        nb_shared = nb_shared_arr[idx].sum()
        rd_corum = rd_corum_arr[idx].sum()
        rd_shared = rd_shared_arr[idx].sum()

        nb_frac = nb_shared / nb_corum if nb_corum > 0 else 0
        rd_frac = rd_shared / rd_corum if rd_corum > 0 else 0

        fold_enrichments[i] = nb_frac / rd_frac if rd_frac > 0 else np.nan

        a, b = nb_shared, nb_corum - nb_shared
        c, d = rd_shared, rd_corum - rd_shared
        odds_ratios[i] = (a * d) / (b * c) if b > 0 and c > 0 else np.nan

    fold_valid = fold_enrichments[~np.isnan(fold_enrichments)]
    or_valid = odds_ratios[~np.isnan(odds_ratios)]

    return {
        "fold_enrichment_mean": float(np.mean(fold_valid)) if len(fold_valid) > 0 else None,
        "fold_enrichment_ci_lo": float(np.percentile(fold_valid, 2.5)) if len(fold_valid) > 0 else None,
        "fold_enrichment_ci_hi": float(np.percentile(fold_valid, 97.5)) if len(fold_valid) > 0 else None,
        "odds_ratio_mean": float(np.mean(or_valid)) if len(or_valid) > 0 else None,
        "odds_ratio_ci_lo": float(np.percentile(or_valid, 2.5)) if len(or_valid) > 0 else None,
        "odds_ratio_ci_hi": float(np.percentile(or_valid, 97.5)) if len(or_valid) > 0 else None,
        "n_valid_bootstrap": len(fold_valid),
        "fold_arr": fold_valid,
        "or_arr": or_valid,
    }


def run_one_dataset(
    dataset: str,
    pair_path: Path,
    gene_to_cx: dict[str, set[int]],
) -> dict:
    """Run full CORUM analysis for one dataset."""
    log.info(f"  Loading pair results: {pair_path.name}")
    pair_df = pl.read_parquet(pair_path)
    log.info(f"    {pair_df.height:,} rows")

    nb_stats = compute_sharing_stats(pair_df, gene_to_cx, "neighbor_cross_gene")
    rd_stats = compute_sharing_stats(pair_df, gene_to_cx, "random_cross_gene")

    log.info(f"    Neighbor: {nb_stats['n_shared_complex']}/{nb_stats['n_both_in_corum']} "
             f"share complex ({nb_stats['frac_of_corum_pairs']:.4%})")
    log.info(f"    Random:   {rd_stats['n_shared_complex']}/{rd_stats['n_both_in_corum']} "
             f"share complex ({rd_stats['frac_of_corum_pairs']:.4%})")

    # Point estimates
    fold_enrich = (nb_stats["frac_of_corum_pairs"] / rd_stats["frac_of_corum_pairs"]
                   if rd_stats["frac_of_corum_pairs"] > 0 else float("inf"))

    a = nb_stats["n_shared_complex"]
    b = nb_stats["n_both_in_corum"] - a
    c = rd_stats["n_shared_complex"]
    d = rd_stats["n_both_in_corum"] - c
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")

    log.info(f"    Fold enrichment: {fold_enrich:.3f}")
    log.info(f"    Odds ratio: {odds_ratio:.3f}")

    # Bootstrap
    log.info(f"    Running gene-level bootstrap ({N_BOOTSTRAP} iterations)...")
    boot = bootstrap_by_source_gene(pair_df, gene_to_cx, N_BOOTSTRAP)

    log.info(f"    Fold enrichment: {boot['fold_enrichment_mean']:.3f} "
             f"[{boot['fold_enrichment_ci_lo']:.3f}, {boot['fold_enrichment_ci_hi']:.3f}]")
    log.info(f"    Odds ratio: {boot['odds_ratio_mean']:.3f} "
             f"[{boot['odds_ratio_ci_lo']:.3f}, {boot['odds_ratio_ci_hi']:.3f}]")

    return {
        "dataset": dataset,
        "neighbor": nb_stats,
        "random": rd_stats,
        "fold_enrichment": fold_enrich,
        "odds_ratio": odds_ratio,
        "bootstrap": boot,
    }


# ── Plotting ──────────────────────────────────────────────────────────

def plot_enrichment_summary(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Sharing fractions (bar chart)
    ax = axes[0]
    datasets = [r["dataset"] for r in results]
    nb_fracs = [r["neighbor"]["frac_of_corum_pairs"] * 100 for r in results]
    rd_fracs = [r["random"]["frac_of_corum_pairs"] * 100 for r in results]
    x = np.arange(len(datasets))
    w = 0.35
    ax.bar(x - w / 2, nb_fracs, w, label="Neighbors", color="C0", edgecolor="black")
    ax.bar(x + w / 2, rd_fracs, w, label="Random", color="C1", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("% pairs sharing CORUM complex\n(among pairs where both genes in CORUM)")
    ax.set_title("Co-complex rate")
    ax.legend()
    for i, (n, r) in enumerate(zip(nb_fracs, rd_fracs)):
        ax.text(i - w / 2, n + 0.02, f"{n:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, r + 0.02, f"{r:.2f}%", ha="center", va="bottom", fontsize=9)

    # Panel 2: Bootstrap fold enrichment distributions
    ax = axes[1]
    for res in results:
        fold_arr = res["bootstrap"]["fold_arr"]
        ax.hist(fold_arr, bins=60, alpha=0.5, density=True, label=res["dataset"])
        ci_lo = res["bootstrap"]["fold_enrichment_ci_lo"]
        ci_hi = res["bootstrap"]["fold_enrichment_ci_hi"]
        ax.axvline(np.median(fold_arr), linestyle="--", alpha=0.7)
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="No enrichment")
    ax.set_xlabel("Fold enrichment (neighbor / random)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap fold enrichment")
    ax.legend(fontsize=9)

    # Panel 3: Bootstrap odds ratio distributions
    ax = axes[2]
    for res in results:
        or_arr = res["bootstrap"]["or_arr"]
        ax.hist(or_arr, bins=60, alpha=0.5, density=True, label=res["dataset"])
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="OR = 1")
    ax.set_xlabel("Odds ratio")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap odds ratio")
    ax.legend(fontsize=9)

    fig.suptitle("CORUM Complex Co-membership: Embedding Neighbors vs Random", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_enrichment.png")


def plot_consequence_breakdown(results: list[dict], gene_to_cx: dict[str, set[int]]) -> None:
    """Per-consequence-bin co-complex fractions."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5), squeeze=False)

    for col_idx, res in enumerate(results):
        ax = axes[0, col_idx]
        dataset = res["dataset"]
        prefix = "chronos_" if dataset == "Chronos" else ""
        pair_path = OUT_DIR / f"{prefix}neighbor_vs_random_profile_similarity.parquet"
        pair_df = pl.read_parquet(pair_path)

        csq_bins = sorted(pair_df.filter(
            pl.col("source_consequence_bin").is_not_null()
        )["source_consequence_bin"].unique().to_list())

        nb_fracs_by_csq = []
        rd_fracs_by_csq = []
        enrichments = []

        for csq in csq_bins:
            csq_df = pair_df.filter(pl.col("source_consequence_bin") == csq)
            nb = compute_sharing_stats(csq_df, gene_to_cx, "neighbor_cross_gene")
            rd = compute_sharing_stats(csq_df, gene_to_cx, "random_cross_gene")
            nb_fracs_by_csq.append(nb["frac_of_corum_pairs"] * 100)
            rd_fracs_by_csq.append(rd["frac_of_corum_pairs"] * 100)
            enr = nb["frac_of_corum_pairs"] / rd["frac_of_corum_pairs"] if rd["frac_of_corum_pairs"] > 0 else 0
            enrichments.append(enr)

        x = np.arange(len(csq_bins))
        w = 0.35
        ax.bar(x - w / 2, nb_fracs_by_csq, w, label="Neighbors", color="C0", edgecolor="black")
        ax.bar(x + w / 2, rd_fracs_by_csq, w, label="Random", color="C1", edgecolor="black")

        ax2 = ax.twinx()
        ax2.plot(x, enrichments, "D-", color="C2", markersize=6, label="Fold enrichment")
        ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.set_ylabel("Fold enrichment", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")

        short_labels = [c[:12] for c in csq_bins]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% sharing complex (CORUM pairs)")
        ax.set_title(f"{dataset}")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("CORUM Co-membership by Consequence Class", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_by_consequence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_by_consequence.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading CORUM complexes...")
    gene_to_cx, cx_to_genes, n_complexes = load_corum()

    datasets = {
        "DEMETER2": OUT_DIR / "neighbor_vs_random_profile_similarity.parquet",
        "Chronos": OUT_DIR / "chronos_neighbor_vs_random_profile_similarity.parquet",
    }

    results = []
    for dataset, pair_path in datasets.items():
        log.info("=" * 60)
        log.info(f"Processing {dataset}")
        log.info("=" * 60)
        res = run_one_dataset(dataset, pair_path, gene_to_cx)
        results.append(res)

    # Save results table
    summary_rows = []
    for res in results:
        summary_rows.append({
            "dataset": res["dataset"],
            "n_neighbor_pairs": res["neighbor"]["n_total"],
            "n_random_pairs": res["random"]["n_total"],
            "n_neighbor_corum": res["neighbor"]["n_both_in_corum"],
            "n_random_corum": res["random"]["n_both_in_corum"],
            "n_neighbor_shared": res["neighbor"]["n_shared_complex"],
            "n_random_shared": res["random"]["n_shared_complex"],
            "neighbor_frac": res["neighbor"]["frac_of_corum_pairs"],
            "random_frac": res["random"]["frac_of_corum_pairs"],
            "fold_enrichment": res["fold_enrichment"],
            "odds_ratio": res["odds_ratio"],
            "fold_enrichment_boot_mean": res["bootstrap"]["fold_enrichment_mean"],
            "fold_enrichment_ci_lo": res["bootstrap"]["fold_enrichment_ci_lo"],
            "fold_enrichment_ci_hi": res["bootstrap"]["fold_enrichment_ci_hi"],
            "odds_ratio_boot_mean": res["bootstrap"]["odds_ratio_mean"],
            "odds_ratio_ci_lo": res["bootstrap"]["odds_ratio_ci_lo"],
            "odds_ratio_ci_hi": res["bootstrap"]["odds_ratio_ci_hi"],
        })
    pl.DataFrame(summary_rows).write_parquet(OUT_DIR / "corum_enrichment_summary.parquet")
    log.info("  Saved corum_enrichment_summary.parquet")

    # Plots
    log.info("Generating figures...")
    plot_enrichment_summary(results)
    plot_consequence_breakdown(results, gene_to_cx)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.0f}s")

    # Terminal summary
    print("\n" + "=" * 70)
    print("CORUM COMPLEX CO-MEMBERSHIP ANALYSIS")
    print("=" * 70)
    print(f"CORUM: {n_complexes} complexes, {len(gene_to_cx)} unique genes")
    print()
    print(f"{'':>12} {'NB pairs':>10} {'NB CORUM':>10} {'NB shared':>10} "
          f"{'NB %':>8} {'RD %':>8} {'Fold':>8} {'OR':>8} {'95% CI (fold)':>20}")
    for res in results:
        nb = res["neighbor"]
        rd = res["random"]
        b = res["bootstrap"]
        print(f"{res['dataset']:>12} {nb['n_total']:>10,} {nb['n_both_in_corum']:>10,} "
              f"{nb['n_shared_complex']:>10,} {nb['frac_of_corum_pairs']:>7.3%} "
              f"{rd['frac_of_corum_pairs']:>7.3%} {res['fold_enrichment']:>7.3f}x "
              f"{res['odds_ratio']:>7.3f} [{b['fold_enrichment_ci_lo']:.3f}, {b['fold_enrichment_ci_hi']:.3f}]")
    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
