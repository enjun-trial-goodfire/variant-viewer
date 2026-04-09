#!/usr/bin/env python3
"""Quick follow-up analyses Q1–Q4 from the presentation plan.

Q1: Normalized effect size (Cohen's d) for DEMETER2 vs Chronos
Q2: Per-complex CORUM enrichment ranking
Q3: ClinVar pathogenicity stratification of dependency signal
Q4: Permutation null distribution for structural enrichment

Outputs go to evee-analysis/data/intermediate/quick_analyses/
Figures go to evee-analysis/outputs/figures/

Run from variant-viewer root:
    uv run python evee-analysis/scripts/run_quick_analyses.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproducibility import enforce_seeds, save_run_config, save_run_manifest

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVEE_ROOT = REPO_ROOT / "evee-analysis"
DATA_DIR = EVEE_ROOT / "data" / "intermediate"
FIG_DIR = EVEE_ROOT / "outputs" / "figures"
OUT_DIR = DATA_DIR / "quick_analyses"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"

RANDOM_SEED = 42
BOOTSTRAP_N = 2000
DATE_TAG = time.strftime("%Y%m%d")

PALETTE = {
    "DEMETER2": "#2271B5",
    "Chronos": "#D32F2F",
    "Pathogenic": "#D32F2F",
    "Benign": "#2271B5",
    "Observed": "#2271B5",
    "Null (permuted)": "#BDBDBD",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / f"{DATE_TAG}_{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved {path.name}")
    return path


# ── Q1: Cohen's d for DEMETER2 vs Chronos ──────────────────────────────


def _gene_level_stats(pair_df: pl.DataFrame) -> dict:
    """Compute gene-level means for neighbor and random, return stats dict."""
    nb = (
        pair_df.filter(pl.col("pair_type") == "neighbor_cross_gene")
        .filter(pl.col("profile_corr").is_not_null())
    )
    rd = (
        pair_df.filter(pl.col("pair_type") == "random_cross_gene")
        .filter(pl.col("profile_corr").is_not_null())
    )

    nb_gene = nb.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))
    rd_gene = rd.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))

    common = sorted(set(nb_gene["source_gene"].to_list()) & set(rd_gene["source_gene"].to_list()))
    nb_dict = dict(zip(nb_gene["source_gene"].to_list(), nb_gene["mean"].to_list()))
    rd_dict = dict(zip(rd_gene["source_gene"].to_list(), rd_gene["mean"].to_list()))

    nb_vals = np.array([nb_dict[g] for g in common])
    rd_vals = np.array([rd_dict[g] for g in common])
    deltas = nb_vals - rd_vals

    n = len(common)
    mean_delta = deltas.mean()
    pooled_std = np.sqrt((nb_vals.var(ddof=1) + rd_vals.var(ddof=1)) / 2)
    cohens_d = mean_delta / pooled_std if pooled_std > 0 else 0.0

    threshold = 0.10
    frac_nb_above = (nb_vals > threshold).mean()
    frac_rd_above = (rd_vals > threshold).mean()
    fold_above = frac_nb_above / frac_rd_above if frac_rd_above > 0 else float("inf")

    return {
        "n_genes": n,
        "mean_nb": float(nb_vals.mean()),
        "mean_rd": float(rd_vals.mean()),
        "mean_delta": float(mean_delta),
        "pooled_std": float(pooled_std),
        "cohens_d": float(cohens_d),
        "frac_nb_above_0.10": float(frac_nb_above),
        "frac_rd_above_0.10": float(frac_rd_above),
        "fold_above_0.10": float(fold_above),
        "nb_pair_count": nb.height,
        "rd_pair_count": rd.height,
    }


def _bootstrap_cohens_d(pair_df: pl.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap Cohen's d by resampling genes."""
    rng = np.random.default_rng(seed)

    nb = pair_df.filter(pl.col("pair_type") == "neighbor_cross_gene").filter(pl.col("profile_corr").is_not_null())
    rd = pair_df.filter(pl.col("pair_type") == "random_cross_gene").filter(pl.col("profile_corr").is_not_null())

    nb_gene = nb.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))
    rd_gene = rd.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))

    common = sorted(set(nb_gene["source_gene"].to_list()) & set(rd_gene["source_gene"].to_list()))
    nb_dict = dict(zip(nb_gene["source_gene"].to_list(), nb_gene["mean"].to_list()))
    rd_dict = dict(zip(rd_gene["source_gene"].to_list(), rd_gene["mean"].to_list()))

    nb_vals = np.array([nb_dict[g] for g in common])
    rd_vals = np.array([rd_dict[g] for g in common])
    n = len(common)

    ds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        nb_b, rd_b = nb_vals[idx], rd_vals[idx]
        delta = nb_b.mean() - rd_b.mean()
        pooled = np.sqrt((nb_b.var(ddof=1) + rd_b.var(ddof=1)) / 2)
        ds[b] = delta / pooled if pooled > 0 else 0.0
    return ds


def q1_normalized_effect_size() -> dict:
    """Cohen's d for both datasets, with bootstrap CIs, for fair cross-dataset comparison."""
    log.info("Q1: Normalized effect size (Cohen's d) ────────────────────")

    dem_df = pl.read_parquet(DATA_DIR / "neighbor_vs_random_profile_similarity.parquet")
    chr_df = pl.read_parquet(DATA_DIR / "chronos_neighbor_vs_random_profile_similarity.parquet")

    dem_stats = _gene_level_stats(dem_df)
    chr_stats = _gene_level_stats(chr_df)

    dem_boot = _bootstrap_cohens_d(dem_df, BOOTSTRAP_N, RANDOM_SEED)
    chr_boot = _bootstrap_cohens_d(chr_df, BOOTSTRAP_N, RANDOM_SEED + 1)

    dem_stats["cohens_d_ci_lo"] = float(np.percentile(dem_boot, 2.5))
    dem_stats["cohens_d_ci_hi"] = float(np.percentile(dem_boot, 97.5))
    chr_stats["cohens_d_ci_lo"] = float(np.percentile(chr_boot, 2.5))
    chr_stats["cohens_d_ci_hi"] = float(np.percentile(chr_boot, 97.5))

    results = {"DEMETER2": dem_stats, "Chronos": chr_stats}

    log.info(f"  DEMETER2: d={dem_stats['cohens_d']:.4f} [{dem_stats['cohens_d_ci_lo']:.4f}, {dem_stats['cohens_d_ci_hi']:.4f}]")
    log.info(f"  Chronos:  d={chr_stats['cohens_d']:.4f} [{chr_stats['cohens_d_ci_lo']:.4f}, {chr_stats['cohens_d_ci_hi']:.4f}]")
    log.info(f"  DEMETER2: fold above r>0.10 = {dem_stats['fold_above_0.10']:.2f}")
    log.info(f"  Chronos:  fold above r>0.10 = {chr_stats['fold_above_0.10']:.2f}")

    # ── Figure: side-by-side Cohen's d comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("DEMETER2 vs Chronos: Normalized effect size\nkNN on full 64×64 covariance matrix · gene-level bootstrap", fontsize=11, y=1.02)

    datasets = ["DEMETER2", "Chronos"]
    ds = [dem_stats["cohens_d"], chr_stats["cohens_d"]]
    ci_lo = [dem_stats["cohens_d_ci_lo"], chr_stats["cohens_d_ci_lo"]]
    ci_hi = [dem_stats["cohens_d_ci_hi"], chr_stats["cohens_d_ci_hi"]]
    colors = [PALETTE["DEMETER2"], PALETTE["Chronos"]]
    yerr_lo = [d - lo for d, lo in zip(ds, ci_lo)]
    yerr_hi = [hi - d for d, hi in zip(ds, ci_hi)]

    bars = ax1.bar(datasets, ds, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    ax1.errorbar(datasets, ds, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax1.set_ylabel("Cohen's d")
    ax1.set_title("Effect size comparison")
    for bar, d_val in zip(bars, ds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(yerr_hi) * 0.15,
                 f"d = {d_val:.4f}", ha="center", va="bottom", fontsize=9)

    # Bootstrap distribution
    for boot, name, color in [(dem_boot, "DEMETER2", PALETTE["DEMETER2"]),
                               (chr_boot, "Chronos", PALETTE["Chronos"])]:
        ax2.hist(boot, bins=50, alpha=0.5, color=color, label=name, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Cohen's d (bootstrap)")
    ax2.set_ylabel("Count")
    ax2.set_title("Bootstrap distributions (n=2000)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    _save(fig, "q1_cohens_d_comparison")

    return results


# ── Q2: Per-complex CORUM enrichment ranking ────────────────────────────


def q2_per_complex_ranking() -> dict:
    """Rank CORUM complexes by embedding enrichment strength."""
    log.info("Q2: Per-complex CORUM enrichment ranking ──────────────────")

    ee = pl.read_parquet(DATA_DIR / "corum_entry_enrichment.parquet")
    gs = pl.read_parquet(DATA_DIR / "corum_complex_gene_sets.parquet")

    per_complex = ee.group_by("complex_id", "complex_name").agg([
        (pl.col("fdr") < 0.05).sum().alias("n_sig_entries"),
        pl.len().alias("n_total_entries"),
        pl.col("effect_size").mean().alias("mean_effect"),
        pl.col("effect_size").filter(pl.col("fdr") < 0.05).mean().alias("mean_sig_effect"),
        pl.col("effect_size").filter(pl.col("fdr") < 0.05).std().alias("std_sig_effect"),
        pl.col("effect_size").abs().mean().alias("mean_abs_effect"),
        pl.col("p_value").filter(pl.col("fdr") < 0.05).median().alias("median_sig_pval"),
    ])

    sizes = gs.select(["complex_id", "n_genes_embedded", "fcg_category"]).rename({"fcg_category": "category"})
    ranked = per_complex.join(sizes, on="complex_id", how="left")
    ranked = ranked.with_columns([
        (pl.col("n_sig_entries") / pl.col("n_total_entries")).alias("frac_sig"),
    ]).sort("n_sig_entries", descending=True)

    out_path = OUT_DIR / "q2_complex_ranking.parquet"
    ranked.write_parquet(out_path)
    log.info(f"  Saved {out_path.name} ({ranked.height} complexes)")

    top20 = ranked.head(20)
    log.info(f"  Top 5 complexes by # significant entries:")
    for row in top20.head(5).iter_rows(named=True):
        log.info(f"    {row['complex_name']}: {row['n_sig_entries']}/{row['n_total_entries']} sig "
                 f"({row['frac_sig']:.1%}), n_genes={row['n_genes_embedded']}")

    # ── Figure: top-20 bar chart ──
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("CORUM complexes ranked by embedding signature strength\n"
                 "Per-entry (i,j) latent feature analysis · FDR < 0.05", fontsize=11, y=1.02)

    names = top20["complex_name"].to_list()
    names = [n[:40] + "…" if len(n) > 40 else n for n in names]
    fracs = top20["frac_sig"].to_list()
    n_genes = top20["n_genes_embedded"].to_list()

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, fracs, color="#2271B5", edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of matrix entries with FDR < 0.05")
    ax.set_title("Top 20 complexes")

    for i, (bar, ng) in enumerate(zip(bars, n_genes)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={ng}", va="center", fontsize=7, color="#666666")

    fig.tight_layout()
    _save(fig, "q2_complex_ranking")

    # ── Figure: enrichment vs complex size ──
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.suptitle("Complex size vs embedding signature coverage\n"
                  "Per-entry (i,j) latent feature analysis · FDR < 0.05", fontsize=11, y=1.02)

    all_fracs = ranked["frac_sig"].to_numpy()
    all_ngenes = ranked["n_genes_embedded"].to_numpy()
    ax2.scatter(all_ngenes, all_fracs, s=20, alpha=0.5, c="#2271B5", edgecolors="white", linewidths=0.3)

    for row in ranked.head(5).iter_rows(named=True):
        label = row["complex_name"][:25]
        ax2.annotate(label, (row["n_genes_embedded"], row["frac_sig"]),
                     fontsize=6, alpha=0.8, xytext=(5, 3), textcoords="offset points")

    ax2.set_xlabel("Number of genes in complex (with embeddings)")
    ax2.set_ylabel("Fraction of matrix entries FDR < 0.05")
    ax2.set_title("Enrichment strength vs complex size")

    fig2.tight_layout()
    _save(fig2, "q2_complex_size_vs_enrichment")

    return {
        "n_complexes": ranked.height,
        "top5": ranked.head(5).select(["complex_name", "n_sig_entries", "frac_sig", "n_genes_embedded"]).to_dicts(),
    }


# ── Q3: ClinVar pathogenicity stratification ────────────────────────────


def _load_significance_map() -> dict[str, str]:
    """Map variant_id → pathogenicity class (pathogenic/benign) from DuckDB."""
    db = duckdb.connect(str(DB_PATH), read_only=True)
    df = db.execute("""
        SELECT variant_id, significance FROM variants
    """).pl()
    db.close()

    sig_map = {}
    for vid, sig in zip(df["variant_id"].to_list(), df["significance"].to_list()):
        if sig is None:
            continue
        s = sig.lower()
        if "pathogenic" in s and "benign" not in s:
            sig_map[vid] = "pathogenic"
        elif "benign" in s and "pathogenic" not in s:
            sig_map[vid] = "benign"
    return sig_map


def _stratified_bootstrap(pair_df: pl.DataFrame, sig_col: str, n_boot: int, seed: int) -> dict:
    """Bootstrap delta by gene for each pathogenicity stratum."""
    rng = np.random.default_rng(seed)
    results = {}

    for label in ["pathogenic", "benign"]:
        sub = pair_df.filter(pl.col(sig_col) == label)
        nb = sub.filter(pl.col("pair_type") == "neighbor_cross_gene").filter(pl.col("profile_corr").is_not_null())
        rd = sub.filter(pl.col("pair_type") == "random_cross_gene").filter(pl.col("profile_corr").is_not_null())

        nb_gene = nb.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))
        rd_gene = rd.group_by("source_gene").agg(pl.col("profile_corr").mean().alias("mean"))

        common = sorted(set(nb_gene["source_gene"].to_list()) & set(rd_gene["source_gene"].to_list()))
        if len(common) < 10:
            results[label] = {"n_genes": len(common), "skip": True}
            continue

        nb_dict = dict(zip(nb_gene["source_gene"].to_list(), nb_gene["mean"].to_list()))
        rd_dict = dict(zip(rd_gene["source_gene"].to_list(), rd_gene["mean"].to_list()))
        nb_vals = np.array([nb_dict[g] for g in common])
        rd_vals = np.array([rd_dict[g] for g in common])
        n = len(common)

        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            deltas[b] = nb_vals[idx].mean() - rd_vals[idx].mean()

        results[label] = {
            "n_genes": n,
            "mean_delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(deltas, 2.5)),
            "ci_hi": float(np.percentile(deltas, 97.5)),
            "nb_pairs": nb.height,
            "rd_pairs": rd.height,
            "mean_nb": float(nb_vals.mean()),
            "mean_rd": float(rd_vals.mean()),
            "skip": False,
        }

    return results


def q3_pathogenicity_stratification() -> dict:
    """Stratify the dependency signal by ClinVar pathogenicity class."""
    log.info("Q3: ClinVar pathogenicity stratification ──────────────────")

    sig_map = _load_significance_map()
    log.info(f"  Loaded {len(sig_map)} variant→significance mappings "
             f"(pathogenic: {sum(1 for v in sig_map.values() if v == 'pathogenic')}, "
             f"benign: {sum(1 for v in sig_map.values() if v == 'benign')})")

    all_results = {}

    for dataset_name, parquet_name in [
        ("DEMETER2", "neighbor_vs_random_profile_similarity.parquet"),
        ("Chronos", "chronos_neighbor_vs_random_profile_similarity.parquet"),
    ]:
        pair_df = pl.read_parquet(DATA_DIR / parquet_name)
        pair_df = pair_df.with_columns(
            pl.col("source_variant_id").replace_strict(sig_map, default=None).alias("clinvar_class")
        )

        n_mapped = pair_df.filter(pl.col("clinvar_class").is_not_null()).height
        log.info(f"  {dataset_name}: {n_mapped}/{pair_df.height} pairs mapped to ClinVar class")

        strat = _stratified_bootstrap(pair_df, "clinvar_class", BOOTSTRAP_N, RANDOM_SEED)
        all_results[dataset_name] = strat

        for cls, stats in strat.items():
            if stats.get("skip"):
                log.info(f"    {cls}: only {stats['n_genes']} genes, skipped")
            else:
                log.info(f"    {cls}: delta={stats['mean_delta']:.4f} [{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}] "
                         f"({stats['n_genes']} genes, {stats['nb_pairs']} nb pairs)")

    # ── Figure: grouped bar chart ──
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Dependency signal by ClinVar pathogenicity class\n"
                 "kNN on full 64×64 covariance matrix · gene-level bootstrap", fontsize=11, y=1.02)

    group_labels = []
    deltas = []
    ci_los = []
    ci_his = []
    colors = []
    x_positions = []
    x = 0
    bar_width = 0.35

    for dataset_name in ["DEMETER2", "Chronos"]:
        strat = all_results[dataset_name]
        for cls in ["pathogenic", "benign"]:
            if strat[cls].get("skip"):
                continue
            group_labels.append(f"{dataset_name}\n{cls.capitalize()}")
            deltas.append(strat[cls]["mean_delta"])
            ci_los.append(strat[cls]["mean_delta"] - strat[cls]["ci_lo"])
            ci_his.append(strat[cls]["ci_hi"] - strat[cls]["mean_delta"])
            colors.append(PALETTE[cls.capitalize()])
            x_positions.append(x)
            x += bar_width + 0.05
        x += 0.3  # gap between datasets

    bars = ax.bar(x_positions, deltas, width=bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(x_positions, deltas, yerr=[ci_los, ci_his], fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Δ correlation (neighbor − random)")
    ax.set_title("Effect stratified by variant pathogenicity")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE["Pathogenic"], label="Pathogenic"),
                       Patch(facecolor=PALETTE["Benign"], label="Benign")]
    ax.legend(handles=legend_elements, frameon=False, loc="upper right")

    fig.tight_layout()
    _save(fig, "q3_pathogenicity_stratification")

    return all_results


# ── Q4: Permutation null for structural enrichment ──────────────────────


def q4_permutation_null() -> dict:
    """Compute a permutation null distribution for CORUM co-complex enrichment."""
    log.info("Q4: Permutation null distribution for enrichment ──────────")

    knn_data = np.load(DATA_DIR / "corum_full_knn_indices.npz", allow_pickle=True)
    knn_indices = knn_data["knn_indices"]  # (n_variants, 50)
    vid_order = knn_data["vid_order"]  # variant IDs

    gs = pl.read_parquet(DATA_DIR / "corum_complex_gene_sets.parquet")

    db = duckdb.connect(str(DB_PATH), read_only=True)
    vid_gene = db.execute("SELECT variant_id, gene_name FROM variants").pl()
    db.close()

    vid_to_gene = dict(zip(vid_gene["variant_id"].to_list(), vid_gene["gene_name"].to_list()))

    gene_to_complexes: dict[str, set[int]] = {}
    for row in gs.iter_rows(named=True):
        members = json.loads(row["member_genes"]) if isinstance(row["member_genes"], str) else row["member_genes"]
        for g in members:
            gene_to_complexes.setdefault(g, set()).add(row["complex_id"])

    vid_list = vid_order.tolist()
    gene_list = [vid_to_gene.get(v) for v in vid_list]

    def _compute_enrichment(knn_idx: np.ndarray, k: int) -> float:
        """Fraction of kNN pairs that share at least one CORUM complex."""
        shared = 0
        total = 0
        for i in range(len(knn_idx)):
            g_i = gene_list[i]
            if g_i is None:
                continue
            cx_i = gene_to_complexes.get(g_i, set())
            if not cx_i:
                continue
            for j_idx in knn_idx[i, :k]:
                g_j = gene_list[j_idx]
                if g_j is None or g_j == g_i:
                    continue
                cx_j = gene_to_complexes.get(g_j, set())
                total += 1
                if cx_i & cx_j:
                    shared += 1
        return shared / total if total > 0 else 0.0

    k = 10
    log.info(f"  Computing observed enrichment (k={k})...")
    observed = _compute_enrichment(knn_indices, k)
    log.info(f"  Observed co-complex fraction: {observed:.6f}")

    n_perms = 200
    rng = np.random.default_rng(RANDOM_SEED)
    perm_fracs = np.empty(n_perms)

    log.info(f"  Running {n_perms} permutations...")
    for p in range(n_perms):
        perm_knn = knn_indices.copy()
        for i in range(perm_knn.shape[0]):
            rng.shuffle(perm_knn[i])
        shuffled_mapping = list(range(len(gene_list)))
        rng.shuffle(shuffled_mapping)
        old_gene_list = gene_list.copy()
        for idx, new_idx in enumerate(shuffled_mapping):
            gene_list[idx] = old_gene_list[new_idx]

        perm_fracs[p] = _compute_enrichment(knn_indices, k)

        gene_list[:] = old_gene_list

        if (p + 1) % 50 == 0:
            log.info(f"    {p+1}/{n_perms} permutations done")

    p_value = (np.sum(perm_fracs >= observed) + 1) / (n_perms + 1)
    z_score = (observed - perm_fracs.mean()) / perm_fracs.std() if perm_fracs.std() > 0 else float("inf")

    log.info(f"  Permutation null mean: {perm_fracs.mean():.6f} ± {perm_fracs.std():.6f}")
    log.info(f"  z-score: {z_score:.2f}, empirical p-value: {p_value:.4f}")

    results = {
        "k": k,
        "observed_fraction": float(observed),
        "null_mean": float(perm_fracs.mean()),
        "null_std": float(perm_fracs.std()),
        "null_percentiles": {
            "2.5": float(np.percentile(perm_fracs, 2.5)),
            "97.5": float(np.percentile(perm_fracs, 97.5)),
        },
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_permutations": n_perms,
    }

    # ── Figure: histogram with observed line ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle("Permutation null for CORUM co-complex enrichment\n"
                 "kNN on full 64×64 covariance matrix · gene-label permutation test", fontsize=11, y=1.02)

    ax.hist(perm_fracs, bins=30, color=PALETTE["Null (permuted)"], edgecolor="white",
            linewidth=0.5, label=f"Null (n={n_perms} permutations)", zorder=2)
    ax.axvline(observed, color=PALETTE["Observed"], linewidth=2, linestyle="-",
               label=f"Observed = {observed:.5f}", zorder=3)
    ax.axvline(perm_fracs.mean(), color="#666666", linewidth=1, linestyle="--",
               label=f"Null mean = {perm_fracs.mean():.5f}", zorder=3)

    ax.set_xlabel("Fraction of kNN pairs sharing a CORUM complex")
    ax.set_ylabel("Count")
    ax.set_title(f"k={k}, z={z_score:.1f}, p={p_value:.4f}")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save(fig, "q4_permutation_null")

    return results


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=" * 60)
    log.info("Quick Analyses Q1–Q4")
    log.info("=" * 60)

    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    save_run_config(OUT_DIR, {
        "random_seed": RANDOM_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
    })
    save_run_manifest(OUT_DIR)

    all_results = {}

    all_results["Q1_cohens_d"] = q1_normalized_effect_size()
    all_results["Q2_complex_ranking"] = q2_per_complex_ranking()
    all_results["Q3_pathogenicity"] = q3_pathogenicity_stratification()
    all_results["Q4_permutation_null"] = q4_permutation_null()

    summary_path = OUT_DIR / "quick_analyses_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSaved summary → {summary_path}")

    # ── Write markdown report ──
    _write_report(all_results)

    log.info("\nDone.")


def _write_report(results: dict) -> None:
    """Write a human-readable summary of all quick analyses."""
    q1 = results["Q1_cohens_d"]
    q2 = results["Q2_complex_ranking"]
    q3 = results["Q3_pathogenicity"]
    q4 = results["Q4_permutation_null"]

    lines = [
        "# Quick Analyses Q1–Q4: Summary Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Q1: Normalized Effect Size (Cohen's d)",
        "",
        "**Question:** Is the dependency signal genuinely stronger in Chronos, or is it an artifact of fewer cell lines?",
        "",
        "| Dataset | Cohen's d | 95% CI | n genes | Fold above r>0.10 |",
        "|---------|-----------|--------|---------|-------------------|",
    ]

    for name in ["DEMETER2", "Chronos"]:
        s = q1[name]
        lines.append(f"| {name} | {s['cohens_d']:.4f} | [{s['cohens_d_ci_lo']:.4f}, {s['cohens_d_ci_hi']:.4f}] "
                     f"| {s['n_genes']} | {s['fold_above_0.10']:.2f}× |")

    lines.extend([
        "",
        "**Interpretation:** Cohen's d normalizes the raw delta by pooled standard deviation, enabling ",
        "fair comparison across datasets with different numbers of cell lines. If Chronos d > DEMETER2 d, ",
        "the stronger signal is genuine rather than a variance artifact.",
        "",
        "---",
        "",
        "## Q2: Per-Complex CORUM Enrichment Ranking",
        "",
        "**Question:** Which protein complexes have the strongest embedding signatures?",
        "",
        f"- **{q2['n_complexes']}** complexes analyzed",
        "",
        "| Rank | Complex | Sig. entries (%) | n genes |",
        "|------|---------|------------------|---------|",
    ])

    for i, row in enumerate(q2["top5"]):
        lines.append(f"| {i+1} | {row['complex_name'][:45]} | {row['frac_sig']:.1%} | {row['n_genes_embedded']} |")

    lines.extend([
        "",
        "**Interpretation:** Complexes with many significant matrix entries have distinctive ",
        "embedding patterns. Larger complexes tend to have more statistical power, but the fraction ",
        "of significant entries controls for this.",
        "",
        "---",
        "",
        "## Q3: ClinVar Pathogenicity Stratification",
        "",
        "**Question:** Is the dependency signal driven by pathogenic or benign variants?",
        "",
    ])

    for dataset_name in ["DEMETER2", "Chronos"]:
        strat = q3[dataset_name]
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append("| Class | Δ correlation | 95% CI | n genes |")
        lines.append("|-------|---------------|--------|---------|")
        for cls in ["pathogenic", "benign"]:
            s = strat[cls]
            if s.get("skip"):
                lines.append(f"| {cls.capitalize()} | — | — | {s['n_genes']} (skipped) |")
            else:
                lines.append(f"| {cls.capitalize()} | {s['mean_delta']:.4f} | "
                             f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] | {s['n_genes']} |")
        lines.append("")

    lines.extend([
        "**Interpretation:** If pathogenic variants show a stronger neighbor-dependency signal than benign, ",
        "this suggests the embedding captures variant-level functional information beyond just gene identity.",
        "",
        "---",
        "",
        "## Q4: Permutation Null for Structural Enrichment",
        "",
        "**Question:** Is the CORUM co-complex enrichment statistically significant against a permutation null?",
        "",
        f"- **Observed co-complex fraction:** {q4['observed_fraction']:.6f}",
        f"- **Null mean ± std:** {q4['null_mean']:.6f} ± {q4['null_std']:.6f}",
        f"- **z-score:** {q4['z_score']:.1f}",
        f"- **Empirical p-value:** {q4['p_value']:.4f} (n={q4['n_permutations']} permutations)",
        "",
        "**Interpretation:** A large z-score and small p-value confirm that the CORUM enrichment is not ",
        "an artifact of the kNN graph structure or gene frequency distribution. The gene-label permutation ",
        "preserves graph topology while destroying biological signal.",
    ])

    report_path = OUT_DIR / "quick_analyses_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {report_path.name}")


if __name__ == "__main__":
    main()
