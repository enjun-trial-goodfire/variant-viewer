#!/usr/bin/env python3
"""CORUM complex co-membership enrichment — full variant set (no DepMap filter).

Uses ALL variants with embeddings from DuckDB.  CORUM complexes filtered to
size ≥ 3 genes.  Gene symbols upper-cased for matching.  Self-pairs excluded.
Bootstrap CIs computed by resampling source *genes*.  kNN recomputed from
embeddings at k = 5, 10, 20, 50.

Outputs (in eeve-analysis/data/intermediate/):
  corum_full_enrichment_vs_k.parquet   — per-k enrichment table
  corum_full_run_config.json           — all parameters for reproducibility
  corum_full_knn_indices.npz           — reusable kNN index + variant order

Figures (in eeve-analysis/outputs/figures/):
  fig_corum_full_enrichment_vs_k.png
  fig_corum_full_sharing_rate_vs_k.png

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_corum_full.py
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
import torch
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

OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"

# ── Parameters ────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
MIN_COMPLEX_SIZE = 3  # genes per complex


# ── Reproducibility ───────────────────────────────────────────────────

def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"  Seeds enforced: {seed}")


def save_run_config(out_dir: Path, config: dict) -> None:
    path = out_dir / "corum_full_run_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    log.info(f"  Saved {path.name}")


# ── CORUM loading ─────────────────────────────────────────────────────

def load_corum(min_size: int) -> tuple[dict[str, set[int]], set[frozenset[str]], int, int]:
    """Load CORUM, normalize gene symbols to uppercase, filter to size ≥ min_size.

    Returns:
      gene_to_cx:       upper-cased gene → set of complex IDs
      co_complex_pairs: set of frozenset({geneA, geneB}) sharing ≥1 complex
      n_total:          total complexes in file
      n_kept:           complexes kept after size filter
    """
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

        if len(genes) < min_size:
            continue
        n_kept += 1

        for g in genes:
            gene_to_cx[g].add(cid)

        genes_list = sorted(genes)
        for i in range(len(genes_list)):
            for j in range(i + 1, len(genes_list)):
                co_complex_pairs.add(frozenset((genes_list[i], genes_list[j])))

    log.info(f"  CORUM: {n_total} total complexes, {n_kept} with ≥{min_size} genes")
    log.info(f"    {len(gene_to_cx)} unique genes, {len(co_complex_pairs):,} co-complex gene pairs")
    return dict(gene_to_cx), co_complex_pairs, n_total, n_kept


# ── Variant loading from DuckDB ───────────────────────────────────────

def load_variants_from_db() -> pl.DataFrame:
    """Load all variants with neighbors and a gene name."""
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT variant_id, gene_name, consequence_display
        FROM variants
        WHERE neighbors IS NOT NULL
          AND neighbors != '[]'
          AND gene_name IS NOT NULL
    """).pl()
    con.close()
    log.info(f"  Loaded {df.height:,} variants from DuckDB ({df['gene_name'].n_unique():,} genes)")
    return df


# ── Embedding loading + kNN ───────────────────────────────────────────

def load_embeddings(variant_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load, flatten, and L2-normalize embeddings for given variant IDs."""
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
    log.info(f"    Found {len(found):,} / {len(variant_ids):,} in embedding index")

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
    log.info(f"    Shape: {embeddings.shape}, L2-normalized")
    return embeddings, vid_order


def compute_knn(embeddings: np.ndarray, max_k: int) -> np.ndarray:
    """Brute-force cosine kNN. Returns indices array (N, max_k), self excluded."""
    from sklearn.neighbors import NearestNeighbors

    log.info(f"  Computing {max_k}-NN for {embeddings.shape[0]:,} variants (brute cosine)...")
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    log.info(f"    kNN complete")
    return indices[:, 1:]  # drop self


# ── Per-k evaluation ──────────────────────────────────────────────────

def evaluate_at_k(
    k: int,
    knn_indices: np.ndarray,
    vid_order: list[str],
    vid_to_gene_upper: dict[str, str],
    gene_to_cx: dict[str, set[int]],
    co_complex_pairs: set[frozenset[str]],
    rng: np.random.Generator,
) -> dict:
    """For top-k neighbors, compute CORUM co-membership enrichment.

    Only considers pairs where both genes are in CORUM (after size filter).
    Self-pairs (same gene) are excluded.  Random controls match the number
    of valid cross-gene CORUM pairs per source gene.
    """
    n_variants = len(vid_order)
    corum_genes_set = set(gene_to_cx.keys())
    corum_genes_list = sorted(corum_genes_set)
    n_corum_genes = len(corum_genes_list)

    # Per-source-gene accumulators (for bootstrap)
    gene_nb_corum: dict[str, int] = defaultdict(int)
    gene_nb_shared: dict[str, int] = defaultdict(int)
    gene_rd_corum: dict[str, int] = defaultdict(int)
    gene_rd_shared: dict[str, int] = defaultdict(int)

    for qi in range(n_variants):
        src_gene = vid_to_gene_upper.get(vid_order[qi])
        if src_gene is None or src_gene not in corum_genes_set:
            continue

        nb_indices = knn_indices[qi, :k]
        n_valid_nb = 0

        for ni in nb_indices:
            tgt_gene = vid_to_gene_upper.get(vid_order[ni])
            if tgt_gene is None or tgt_gene == src_gene:
                continue
            if tgt_gene not in corum_genes_set:
                continue
            gene_nb_corum[src_gene] += 1
            n_valid_nb += 1
            if frozenset((src_gene, tgt_gene)) in co_complex_pairs:
                gene_nb_shared[src_gene] += 1

        # Matched random: same number of cross-gene CORUM pairs
        for _ in range(n_valid_nb):
            for _attempt in range(50):
                rg = corum_genes_list[rng.integers(n_corum_genes)]
                if rg != src_gene:
                    break
            gene_rd_corum[src_gene] += 1
            if frozenset((src_gene, rg)) in co_complex_pairs:
                gene_rd_shared[src_gene] += 1

    # Point estimates
    common_genes = sorted(set(gene_nb_corum.keys()) & set(gene_rd_corum.keys()))
    n_genes = len(common_genes)
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

    # Gene-level bootstrap
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
        "fold_ci_lo": float(np.percentile(fold_valid, 2.5)) if len(fold_valid) > 0 else None,
        "fold_ci_hi": float(np.percentile(fold_valid, 97.5)) if len(fold_valid) > 0 else None,
        "or_ci_lo": float(np.percentile(or_valid, 2.5)) if len(or_valid) > 0 else None,
        "or_ci_hi": float(np.percentile(or_valid, 97.5)) if len(or_valid) > 0 else None,
        "n_bootstrap_valid": int(len(fold_valid)),
    }

    log.info(
        f"    k={k:>2d}: nb={nb_shared_total:,}/{nb_corum_total:,} ({nb_frac:.3%})  "
        f"rd={rd_shared_total:,}/{rd_corum_total:,} ({rd_frac:.3%})  "
        f"fold={fold:.2f}x [{result['fold_ci_lo']:.2f}, {result['fold_ci_hi']:.2f}]  "
        f"OR={odds_ratio:.2f} [{result['or_ci_lo']:.2f}, {result['or_ci_hi']:.2f}]  "
        f"({n_genes} genes)"
    )
    return result


# ── Plotting ──────────────────────────────────────────────────────────

def plot_enrichment_vs_k(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ks = [r["k"] for r in rows]

    # Fold enrichment
    ax = axes[0]
    folds = [r["fold_enrichment"] for r in rows]
    ci_lo = [r["fold_ci_lo"] for r in rows]
    ci_hi = [r["fold_ci_hi"] for r in rows]
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
    ax.set_title("CORUM co-complex fold enrichment vs k\n(all variants, complexes ≥ 3 genes)")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)

    # Odds ratio
    ax = axes[1]
    ors = [r["odds_ratio"] for r in rows]
    or_lo = [r["or_ci_lo"] for r in rows]
    or_hi = [r["or_ci_hi"] for r in rows]
    yerr = [[o - lo for o, lo in zip(ors, or_lo)],
            [hi - o for o, hi in zip(ors, or_hi)]]
    ax.errorbar(ks, ors, yerr=yerr, fmt="s-", capsize=5, markersize=9,
                linewidth=2.5, color="C1")
    for k_val, o_val in zip(ks, ors):
        ax.annotate(f"{o_val:.2f}", (k_val, o_val), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Odds ratio", fontsize=12)
    ax.set_title("CORUM co-complex odds ratio vs k\n(all variants, complexes ≥ 3 genes)")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_full_enrichment_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_full_enrichment_vs_k.png")


def plot_sharing_rates(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks = [r["k"] for r in rows]
    nb_pct = [r["nb_frac"] * 100 for r in rows]
    rd_pct = [r["rd_frac"] * 100 for r in rows]

    ax.plot(ks, nb_pct, "o-", linewidth=2.5, markersize=9, label="Embedding neighbors", color="C0")
    ax.plot(ks, rd_pct, "s--", linewidth=2, markersize=7, label="Random (matched)", color="C1")

    for k_val, n, r in zip(ks, nb_pct, rd_pct):
        ax.annotate(f"{n:.2f}%", (k_val, n), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="C0")
        ax.annotate(f"{r:.2f}%", (k_val, r), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color="C1")

    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("% pairs sharing CORUM complex\n(among CORUM-annotated pairs)", fontsize=11)
    ax.set_title("Co-complex sharing rate vs k\n(all variants, complexes ≥ 3 genes)")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corum_full_sharing_rate_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_corum_full_sharing_rate_vs_k.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CORUM ────────────────────────────────────────────────────
    log.info("Loading CORUM complexes (size ≥ %d)...", MIN_COMPLEX_SIZE)
    gene_to_cx, co_complex_pairs, n_total_cx, n_kept_cx = load_corum(MIN_COMPLEX_SIZE)
    corum_genes = set(gene_to_cx.keys())

    # ── Load all variants ─────────────────────────────────────────────
    log.info("Loading variants from DuckDB...")
    var_df = load_variants_from_db()
    var_df = var_df.with_columns(pl.col("gene_name").str.to_uppercase().alias("gene_upper"))

    # Restrict to variants whose gene is in CORUM
    var_in_corum = var_df.filter(pl.col("gene_upper").is_in(list(corum_genes)))
    n_all = var_df.height
    n_corum = var_in_corum.height
    n_genes_corum = var_in_corum["gene_upper"].n_unique()
    log.info(f"  Variants with gene in CORUM (≥{MIN_COMPLEX_SIZE}): "
             f"{n_corum:,} / {n_all:,} ({n_genes_corum:,} genes)")

    # We compute kNN over ALL variants (not just CORUM), because neighbors
    # of a non-CORUM variant could still land on a CORUM gene
    variant_ids = var_df["variant_id"].to_list()
    gene_names = var_df["gene_upper"].to_list()
    vid_to_gene_upper = dict(zip(variant_ids, gene_names))

    # ── Embeddings + kNN ──────────────────────────────────────────────
    embeddings, vid_order = load_embeddings(variant_ids)
    max_k = max(K_VALUES)
    knn_indices = compute_knn(embeddings, max_k)

    # Save kNN indices for reuse
    knn_path = OUT_DIR / "corum_full_knn_indices.npz"
    np.savez_compressed(
        str(knn_path),
        knn_indices=knn_indices,
        vid_order=np.array(vid_order, dtype=object),
    )
    log.info(f"  Saved knn_indices: {knn_path.name} ({knn_path.stat().st_size / 1e6:.1f} MB)")

    # ── Evaluate each k ───────────────────────────────────────────────
    results = []
    for k in K_VALUES:
        rng = np.random.default_rng(RANDOM_SEED + k)
        res = evaluate_at_k(
            k, knn_indices, vid_order, vid_to_gene_upper,
            gene_to_cx, co_complex_pairs, rng,
        )
        results.append(res)

    # ── Save results ──────────────────────────────────────────────────
    pl.DataFrame(results).write_parquet(OUT_DIR / "corum_full_enrichment_vs_k.parquet")
    log.info("  Saved corum_full_enrichment_vs_k.parquet")

    # ── Save run config ───────────────────────────────────────────────
    config = {
        "analysis": "corum_full",
        "description": "CORUM co-membership enrichment, full variant set, no DepMap filter",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "min_complex_size": MIN_COMPLEX_SIZE,
        "gene_normalization": "uppercase (no alias resolution)",
        "corum_complexes_total": n_total_cx,
        "corum_complexes_kept": n_kept_cx,
        "corum_genes": len(corum_genes),
        "corum_co_complex_pairs": len(co_complex_pairs),
        "variants_total": n_all,
        "variants_in_corum": n_corum,
        "genes_in_corum_and_db": n_genes_corum,
        "embeddings_loaded": len(vid_order),
        "knn_max_k": max_k,
        "knn_metric": "cosine",
        "knn_algorithm": "brute",
        "self_pairs_excluded": True,
        "random_control": "matched count per source gene, sampled from CORUM genes only",
    }
    save_run_config(OUT_DIR, config)

    # ── Plots ─────────────────────────────────────────────────────────
    log.info("Generating figures...")
    plot_enrichment_vs_k(results)
    plot_sharing_rates(results)

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed / 60:.1f} minutes")

    # ── Terminal summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CORUM ENRICHMENT vs k — FULL VARIANT SET (no DepMap filter)")
    print("=" * 80)
    print(f"CORUM: {n_kept_cx} complexes (≥{MIN_COMPLEX_SIZE} genes), "
          f"{len(corum_genes)} genes, {len(co_complex_pairs):,} co-complex pairs")
    print(f"Variants: {len(vid_order):,} with embeddings, "
          f"{n_corum:,} in CORUM genes ({n_genes_corum:,} unique genes)")
    print()
    print(f"{'k':>4}  {'NB shared':>10} {'NB CORUM':>10} {'NB %':>8} "
          f"{'RD %':>8} {'Fold':>8} {'95% CI':>18} {'OR':>8} {'95% CI':>18}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['k']:>4d}  {r['nb_shared']:>10,} {r['nb_corum_pairs']:>10,} "
            f"{r['nb_frac']:>7.3%} {r['rd_frac']:>7.3%} "
            f"{r['fold_enrichment']:>7.2f}x "
            f"[{r['fold_ci_lo']:.2f}, {r['fold_ci_hi']:.2f}]  "
            f"{r['odds_ratio']:>7.2f} "
            f"[{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}]"
        )
    print(f"\nElapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
