#!/usr/bin/env python3
"""STRING interaction enrichment for embedding neighbors.

For each embedding neighbor pair (k = 5, 10, 20, 50), compute:
  - Fraction present in STRING
  - Mean combined_score, experimental score, database score
  - Comparison to matched random gene pairs
  - Fold enrichment + gene-level bootstrap 95% CIs

Uses ALL variants with embeddings (no DepMap / CORUM filter).
Reuses precomputed kNN from corum_full_knn_indices.npz if available,
otherwise recomputes from embeddings.

Gene mapping: STRING protein.info preferred_name (exact match, already
uppercase).  DuckDB gene_name uppercased for matching.  Self-pairs excluded.

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_string_analysis.py
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

STRING_INFO = EEVE_ROOT / "data" / "9606.protein.info.v12.0.txt"
STRING_LINKS = EEVE_ROOT / "data" / "9606.protein.links.full.v12.0.txt"

OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"
KNN_CACHE = OUT_DIR / "corum_full_knn_indices.npz"

# ── Parameters ────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]


# ── Reproducibility ───────────────────────────────────────────────────

def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"  Seeds enforced: {seed}")


# ── STRING loading ────────────────────────────────────────────────────

def load_string() -> tuple[dict[tuple[str, str], tuple[int, int, int]], set[str]]:
    """Load STRING, map to gene symbols, return pair→scores lookup.

    Returns:
      pair_scores: (geneA, geneB) sorted tuple → (combined, experiments, database)
      string_genes: set of all genes present in STRING
    """
    log.info("  Loading STRING protein.info...")
    info = pl.read_csv(
        str(STRING_INFO), separator="\t", comment_prefix="#", has_header=False,
        new_columns=["string_protein_id", "preferred_name", "protein_size", "annotation"],
    ).select("string_protein_id", "preferred_name")
    prot_to_gene: dict[str, str] = dict(
        zip(info["string_protein_id"].to_list(), info["preferred_name"].to_list())
    )
    string_genes = set(prot_to_gene.values())
    log.info(f"    {len(prot_to_gene):,} proteins → {len(string_genes):,} unique genes")

    log.info("  Loading STRING protein.links.full...")
    links = pl.read_csv(str(STRING_LINKS), separator=" ")
    log.info(f"    {links.height:,} edges")

    log.info("  Mapping protein IDs → gene names and building lookup...")
    p1 = links["protein1"].to_list()
    p2 = links["protein2"].to_list()
    combined = links["combined_score"].to_numpy()
    experiments = links["experiments"].to_numpy()
    database = links["database"].to_numpy()

    pair_scores: dict[tuple[str, str], tuple[int, int, int]] = {}
    n_mapped = 0
    for i in range(len(p1)):
        g1 = prot_to_gene.get(p1[i])
        g2 = prot_to_gene.get(p2[i])
        if g1 is None or g2 is None or g1 == g2:
            continue
        key = (g1, g2) if g1 < g2 else (g2, g1)
        if key not in pair_scores:
            pair_scores[key] = (int(combined[i]), int(experiments[i]), int(database[i]))
            n_mapped += 1

    log.info(f"    {n_mapped:,} unique gene pairs in STRING lookup")
    return pair_scores, string_genes


# ── Variant loading ───────────────────────────────────────────────────

def load_variants() -> pl.DataFrame:
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT variant_id, gene_name
        FROM variants
        WHERE neighbors IS NOT NULL AND neighbors != '[]'
          AND gene_name IS NOT NULL
    """).pl()
    con.close()
    df = df.with_columns(pl.col("gene_name").str.to_uppercase().alias("gene_upper"))
    log.info(f"  Loaded {df.height:,} variants ({df['gene_upper'].n_unique():,} genes)")
    return df


# ── kNN loading / computation ─────────────────────────────────────────

def load_or_compute_knn(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load cached kNN if available and variant order matches, else recompute."""
    if KNN_CACHE.exists():
        log.info(f"  Loading cached kNN from {KNN_CACHE.name}...")
        data = np.load(str(KNN_CACHE), allow_pickle=True)
        cached_vid = data["vid_order"].tolist()
        cached_knn = data["knn_indices"]

        if cached_vid == variant_ids[:len(cached_vid)] and len(cached_vid) == cached_knn.shape[0]:
            log.info(f"    Loaded: {cached_knn.shape[0]:,} variants, k_max={cached_knn.shape[1]}")
            return cached_knn, cached_vid
        else:
            log.info(f"    Cache mismatch (vid_order differs), recomputing...")

    return _compute_knn_from_embeddings(variant_ids)


def _compute_knn_from_embeddings(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and compute brute-force cosine kNN."""
    import torch
    from safetensors.torch import load_file as safetensors_load

    log.info(f"  Loading embeddings for {len(variant_ids):,} variants...")
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
    log.info(f"    Found {len(found):,} / {len(variant_ids):,}")

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
        log.info(f"    Chunk {cid}: {len(by_chunk[cid]):,} variants")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)

    from sklearn.neighbors import NearestNeighbors
    max_k = max(K_VALUES)
    log.info(f"  Computing {max_k}-NN for {embeddings.shape[0]:,} variants...")
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    knn_indices = indices[:, 1:]
    log.info(f"    kNN complete")

    np.savez_compressed(
        str(OUT_DIR / "string_knn_indices.npz"),
        knn_indices=knn_indices,
        vid_order=np.array(vid_order, dtype=object),
    )
    return knn_indices, vid_order


# ── Per-k evaluation ──────────────────────────────────────────────────

def evaluate_at_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene: dict[str, str],
    pair_scores: dict[tuple[str, str], tuple[int, int, int]],
    string_genes: set[str],
    rng: np.random.Generator,
) -> dict:
    """Evaluate STRING metrics for top-k neighbors vs matched random."""
    n_variants = len(vid_order)
    string_genes_list = sorted(string_genes)
    n_string_genes = len(string_genes_list)

    # Per-source-gene accumulators
    gene_nb_total: dict[str, int] = defaultdict(int)       # cross-gene pairs evaluated
    gene_nb_in_string: dict[str, int] = defaultdict(int)   # found in STRING
    gene_nb_combined: dict[str, list[int]] = defaultdict(list)
    gene_nb_experiments: dict[str, list[int]] = defaultdict(list)
    gene_nb_database: dict[str, list[int]] = defaultdict(list)

    gene_rd_total: dict[str, int] = defaultdict(int)
    gene_rd_in_string: dict[str, int] = defaultdict(int)
    gene_rd_combined: dict[str, list[int]] = defaultdict(list)
    gene_rd_experiments: dict[str, list[int]] = defaultdict(list)
    gene_rd_database: dict[str, list[int]] = defaultdict(list)

    for qi in range(n_variants):
        src_gene = vid_to_gene.get(vid_order[qi])
        if src_gene is None or src_gene not in string_genes:
            continue

        nb_indices = knn_indices[qi, :k]
        n_valid = 0

        for ni in nb_indices:
            tgt_gene = vid_to_gene.get(vid_order[ni])
            if tgt_gene is None or tgt_gene == src_gene or tgt_gene not in string_genes:
                continue

            n_valid += 1
            gene_nb_total[src_gene] += 1
            key = (src_gene, tgt_gene) if src_gene < tgt_gene else (tgt_gene, src_gene)
            scores = pair_scores.get(key)
            if scores is not None:
                gene_nb_in_string[src_gene] += 1
                gene_nb_combined[src_gene].append(scores[0])
                gene_nb_experiments[src_gene].append(scores[1])
                gene_nb_database[src_gene].append(scores[2])

        # Matched random
        for _ in range(n_valid):
            for _attempt in range(50):
                rg = string_genes_list[rng.integers(n_string_genes)]
                if rg != src_gene:
                    break
            gene_rd_total[src_gene] += 1
            key = (src_gene, rg) if src_gene < rg else (rg, src_gene)
            scores = pair_scores.get(key)
            if scores is not None:
                gene_rd_in_string[src_gene] += 1
                gene_rd_combined[src_gene].append(scores[0])
                gene_rd_experiments[src_gene].append(scores[1])
                gene_rd_database[src_gene].append(scores[2])

    # Point estimates
    common_genes = sorted(set(gene_nb_total.keys()) & set(gene_rd_total.keys()))
    n_genes = len(common_genes)

    nb_total = sum(gene_nb_total[g] for g in common_genes)
    nb_in_str = sum(gene_nb_in_string.get(g, 0) for g in common_genes)
    rd_total = sum(gene_rd_total[g] for g in common_genes)
    rd_in_str = sum(gene_rd_in_string.get(g, 0) for g in common_genes)

    nb_frac = nb_in_str / nb_total if nb_total > 0 else 0
    rd_frac = rd_in_str / rd_total if rd_total > 0 else 0
    fold = nb_frac / rd_frac if rd_frac > 0 else float("inf")

    nb_all_combined = [s for g in common_genes for s in gene_nb_combined.get(g, [])]
    nb_all_experiments = [s for g in common_genes for s in gene_nb_experiments.get(g, [])]
    nb_all_database = [s for g in common_genes for s in gene_nb_database.get(g, [])]
    rd_all_combined = [s for g in common_genes for s in gene_rd_combined.get(g, [])]
    rd_all_experiments = [s for g in common_genes for s in gene_rd_experiments.get(g, [])]
    rd_all_database = [s for g in common_genes for s in gene_rd_database.get(g, [])]

    def _safe_mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    # Bootstrap by source gene
    nb_total_arr = np.array([gene_nb_total[g] for g in common_genes])
    nb_in_str_arr = np.array([gene_nb_in_string.get(g, 0) for g in common_genes])
    rd_total_arr = np.array([gene_rd_total[g] for g in common_genes])
    rd_in_str_arr = np.array([gene_rd_in_string.get(g, 0) for g in common_genes])
    nb_sum_combined_arr = np.array([sum(gene_nb_combined.get(g, [])) for g in common_genes], dtype=np.float64)
    nb_sum_exper_arr = np.array([sum(gene_nb_experiments.get(g, [])) for g in common_genes], dtype=np.float64)
    nb_sum_db_arr = np.array([sum(gene_nb_database.get(g, [])) for g in common_genes], dtype=np.float64)
    rd_sum_combined_arr = np.array([sum(gene_rd_combined.get(g, [])) for g in common_genes], dtype=np.float64)
    rd_sum_exper_arr = np.array([sum(gene_rd_experiments.get(g, [])) for g in common_genes], dtype=np.float64)
    rd_sum_db_arr = np.array([sum(gene_rd_database.get(g, [])) for g in common_genes], dtype=np.float64)

    boot_fold = np.empty(N_BOOTSTRAP)
    boot_nb_combined = np.empty(N_BOOTSTRAP)
    boot_rd_combined = np.empty(N_BOOTSTRAP)
    boot_nb_exper = np.empty(N_BOOTSTRAP)
    boot_rd_exper = np.empty(N_BOOTSTRAP)
    boot_nb_db = np.empty(N_BOOTSTRAP)
    boot_rd_db = np.empty(N_BOOTSTRAP)

    for bi in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_genes, size=n_genes)
        s_nb_tot = nb_total_arr[idx].sum()
        s_nb_in = nb_in_str_arr[idx].sum()
        s_rd_tot = rd_total_arr[idx].sum()
        s_rd_in = rd_in_str_arr[idx].sum()
        nf = s_nb_in / s_nb_tot if s_nb_tot > 0 else 0
        rf = s_rd_in / s_rd_tot if s_rd_tot > 0 else 0
        boot_fold[bi] = nf / rf if rf > 0 else np.nan

        s_nb_in_f = float(s_nb_in)
        s_rd_in_f = float(s_rd_in)
        boot_nb_combined[bi] = nb_sum_combined_arr[idx].sum() / s_nb_in_f if s_nb_in_f > 0 else np.nan
        boot_rd_combined[bi] = rd_sum_combined_arr[idx].sum() / s_rd_in_f if s_rd_in_f > 0 else np.nan
        boot_nb_exper[bi] = nb_sum_exper_arr[idx].sum() / s_nb_in_f if s_nb_in_f > 0 else np.nan
        boot_rd_exper[bi] = rd_sum_exper_arr[idx].sum() / s_rd_in_f if s_rd_in_f > 0 else np.nan
        boot_nb_db[bi] = nb_sum_db_arr[idx].sum() / s_nb_in_f if s_nb_in_f > 0 else np.nan
        boot_rd_db[bi] = rd_sum_db_arr[idx].sum() / s_rd_in_f if s_rd_in_f > 0 else np.nan

    fold_valid = boot_fold[~np.isnan(boot_fold)]
    delta_combined = boot_nb_combined - boot_rd_combined
    delta_combined_valid = delta_combined[~np.isnan(delta_combined)]
    delta_exper = boot_nb_exper - boot_rd_exper
    delta_exper_valid = delta_exper[~np.isnan(delta_exper)]
    delta_db = boot_nb_db - boot_rd_db
    delta_db_valid = delta_db[~np.isnan(delta_db)]

    def _ci(arr):
        if len(arr) == 0:
            return (None, None)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    result = {
        "k": k,
        "n_source_genes": n_genes,
        "nb_pairs": int(nb_total),
        "nb_in_string": int(nb_in_str),
        "rd_pairs": int(rd_total),
        "rd_in_string": int(rd_in_str),
        "nb_frac_in_string": float(nb_frac),
        "rd_frac_in_string": float(rd_frac),
        "fold_enrichment": float(fold),
        "fold_ci_lo": _ci(fold_valid)[0],
        "fold_ci_hi": _ci(fold_valid)[1],
        "nb_mean_combined": _safe_mean(nb_all_combined),
        "rd_mean_combined": _safe_mean(rd_all_combined),
        "delta_combined_ci_lo": _ci(delta_combined_valid)[0],
        "delta_combined_ci_hi": _ci(delta_combined_valid)[1],
        "nb_mean_experiments": _safe_mean(nb_all_experiments),
        "rd_mean_experiments": _safe_mean(rd_all_experiments),
        "delta_experiments_ci_lo": _ci(delta_exper_valid)[0],
        "delta_experiments_ci_hi": _ci(delta_exper_valid)[1],
        "nb_mean_database": _safe_mean(nb_all_database),
        "rd_mean_database": _safe_mean(rd_all_database),
        "delta_database_ci_lo": _ci(delta_db_valid)[0],
        "delta_database_ci_hi": _ci(delta_db_valid)[1],
    }

    log.info(
        f"    k={k:>2d}: in_string nb={nb_in_str:,}/{nb_total:,} ({nb_frac:.3%})  "
        f"rd={rd_in_str:,}/{rd_total:,} ({rd_frac:.3%})  "
        f"fold={fold:.2f}x [{result['fold_ci_lo']:.2f}, {result['fold_ci_hi']:.2f}]"
    )
    log.info(
        f"          combined: nb={result['nb_mean_combined']:.0f} rd={result['rd_mean_combined']:.0f}  "
        f"exper: nb={result['nb_mean_experiments']:.0f} rd={result['rd_mean_experiments']:.0f}  "
        f"database: nb={result['nb_mean_database']:.0f} rd={result['rd_mean_database']:.0f}"
    )
    return result


# ── Plotting ──────────────────────────────────────────────────────────

def plot_enrichment_vs_k(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ks = [r["k"] for r in results]

    # Fold enrichment of STRING presence
    ax = axes[0]
    folds = [r["fold_enrichment"] for r in results]
    ci_lo = [r["fold_ci_lo"] for r in results]
    ci_hi = [r["fold_ci_hi"] for r in results]
    yerr = [[f - lo for f, lo in zip(folds, ci_lo)],
            [hi - f for f, hi in zip(folds, ci_hi)]]
    ax.errorbar(ks, folds, yerr=yerr, fmt="o-", capsize=5, markersize=9,
                linewidth=2.5, color="C0")
    for k_val, f_val in zip(ks, folds):
        ax.annotate(f"{f_val:.2f}×", (k_val, f_val), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Fold enrichment\n(neighbor / random)", fontsize=12)
    ax.set_title("STRING interaction presence:\nembedding neighbors vs random")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)

    # STRING presence rate
    ax = axes[1]
    nb_pct = [r["nb_frac_in_string"] * 100 for r in results]
    rd_pct = [r["rd_frac_in_string"] * 100 for r in results]
    ax.plot(ks, nb_pct, "o-", linewidth=2.5, markersize=9, label="Neighbors", color="C0")
    ax.plot(ks, rd_pct, "s--", linewidth=2, markersize=7, label="Random", color="C1")
    for k_val, n, r in zip(ks, nb_pct, rd_pct):
        ax.annotate(f"{n:.1f}%", (k_val, n), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="C0")
        ax.annotate(f"{r:.1f}%", (k_val, r), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color="C1")
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("% pairs in STRING", fontsize=12)
    ax.set_title("STRING interaction rate vs k")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_string_enrichment_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_string_enrichment_vs_k.png")


def plot_scores_vs_k(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ks = [r["k"] for r in results]

    for ax_idx, (score_key, title) in enumerate([
        ("combined", "Combined score"),
        ("experiments", "Experimental score"),
        ("database", "Database score"),
    ]):
        ax = axes[ax_idx]
        nb_vals = [r[f"nb_mean_{score_key}"] for r in results]
        rd_vals = [r[f"rd_mean_{score_key}"] for r in results]

        ax.plot(ks, nb_vals, "o-", linewidth=2.5, markersize=9, label="Neighbors", color="C0")
        ax.plot(ks, rd_vals, "s--", linewidth=2, markersize=7, label="Random", color="C1")

        for k_val, n, r in zip(ks, nb_vals, rd_vals):
            ax.annotate(f"{n:.0f}", (k_val, n), textcoords="offset points",
                        xytext=(8, 8), fontsize=9, color="C0")
            ax.annotate(f"{r:.0f}", (k_val, r), textcoords="offset points",
                        xytext=(8, -12), fontsize=9, color="C1")

        ax.set_xlabel("k (number of neighbors)", fontsize=12)
        ax.set_ylabel(f"Mean {title}\n(among STRING pairs)", fontsize=11)
        ax.set_title(f"STRING {title} vs k")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_string_scores_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_string_scores_vs_k.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load STRING ───────────────────────────────────────────────────
    log.info("Loading STRING database...")
    pair_scores, string_genes = load_string()

    # ── Load variants ─────────────────────────────────────────────────
    log.info("Loading variants from DuckDB...")
    var_df = load_variants()
    variant_ids = var_df["variant_id"].to_list()
    vid_to_gene = dict(zip(variant_ids, var_df["gene_upper"].to_list()))

    # Gene overlap
    db_genes = set(vid_to_gene.values())
    overlap = db_genes & string_genes
    log.info(f"  Gene overlap: {len(overlap):,} / {len(db_genes):,} DB genes in STRING")

    # ── kNN ───────────────────────────────────────────────────────────
    knn_indices, vid_order = load_or_compute_knn(variant_ids)

    # Update vid_to_gene for the kNN vid_order (may be subset if embeddings missing)
    vid_to_gene_knn = {v: vid_to_gene[v] for v in vid_order if v in vid_to_gene}

    # ── Evaluate each k ───────────────────────────────────────────────
    log.info("Evaluating STRING enrichment per k...")
    results = []
    for k in K_VALUES:
        rng = np.random.default_rng(RANDOM_SEED + k)
        res = evaluate_at_k(
            k, knn_indices, vid_order, vid_to_gene_knn,
            pair_scores, string_genes, rng,
        )
        results.append(res)

    # ── Save ──────────────────────────────────────────────────────────
    pl.DataFrame(results).write_parquet(OUT_DIR / "string_enrichment_vs_k.parquet")
    log.info("  Saved string_enrichment_vs_k.parquet")

    # Run config
    config = {
        "analysis": "string_enrichment",
        "description": "STRING interaction enrichment for embedding neighbors, full variant set",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "gene_normalization": "uppercase exact match via STRING preferred_name",
        "string_version": "v12.0",
        "string_unique_gene_pairs": len(pair_scores),
        "string_genes": len(string_genes),
        "db_genes": len(db_genes),
        "gene_overlap": len(overlap),
        "variants_total": var_df.height,
        "embeddings_loaded": len(vid_order),
        "knn_max_k": max(K_VALUES),
        "self_pairs_excluded": True,
        "random_control": "matched count per source gene, sampled from STRING genes only",
    }
    config_path = OUT_DIR / "string_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    log.info(f"  Saved {config_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────
    log.info("Generating figures...")
    plot_enrichment_vs_k(results)
    plot_scores_vs_k(results)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed / 60:.1f} minutes")

    # ── Terminal summary ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STRING INTERACTION ENRICHMENT — EMBEDDING NEIGHBORS vs RANDOM")
    print("=" * 100)
    print(f"STRING v12.0: {len(pair_scores):,} gene pairs, {len(string_genes):,} genes")
    print(f"Variants: {len(vid_order):,} with embeddings, {len(overlap):,} genes in STRING")
    print()

    # Presence / fold enrichment
    print(f"{'k':>4}  {'NB pairs':>10} {'NB in STR':>10} {'NB %':>8} "
          f"{'RD %':>8} {'Fold':>8} {'95% CI':>18}")
    print("-" * 75)
    for r in results:
        print(f"{r['k']:>4d}  {r['nb_pairs']:>10,} {r['nb_in_string']:>10,} "
              f"{r['nb_frac_in_string']:>7.2%} {r['rd_frac_in_string']:>7.2%} "
              f"{r['fold_enrichment']:>7.2f}x "
              f"[{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}]")

    # Scores
    print()
    print(f"{'k':>4}  {'NB combined':>12} {'RD combined':>12} {'Δ CI':>20}  "
          f"{'NB exper':>10} {'RD exper':>10} {'Δ CI':>20}  "
          f"{'NB db':>8} {'RD db':>8} {'Δ CI':>20}")
    print("-" * 140)
    for r in results:
        print(
            f"{r['k']:>4d}  {r['nb_mean_combined']:>12.1f} {r['rd_mean_combined']:>12.1f} "
            f"[{r['delta_combined_ci_lo']:+.1f}, {r['delta_combined_ci_hi']:+.1f}]  "
            f"{r['nb_mean_experiments']:>10.1f} {r['rd_mean_experiments']:>10.1f} "
            f"[{r['delta_experiments_ci_lo']:+.1f}, {r['delta_experiments_ci_hi']:+.1f}]  "
            f"{r['nb_mean_database']:>8.1f} {r['rd_mean_database']:>8.1f} "
            f"[{r['delta_database_ci_lo']:+.1f}, {r['delta_database_ci_hi']:+.1f}]"
        )

    print(f"\nElapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
