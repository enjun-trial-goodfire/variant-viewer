#!/usr/bin/env python3
"""Forward-in-time CORUM prediction: derive features from v3, predict v5 novelties.

Methodology:
  The key insight is that predictive features must be derived ONLY from the older
  (v3) dataset. Using v5-derived features to predict v5-novel complexes is circular.
  This script corrects that by re-running the interpretability pipeline (Stages 2-4)
  on v3 complexes to produce v3-derived recurrent entries, then uses those features
  in the prediction tests.

  Part 1 — Database comparison (v3 vs current)
  Part 2 — Forward-in-time feature derivation (Stages 2-4 on v3 complexes)
  Part 3 — Prediction tests:
      A: kNN neighbor enrichment for novel co-complex pairs
      B: Latent-space cosine similarity (novel vs old vs random)
      C: v3-derived feature-score prediction for novel pairs (CORRECTED)
      D: Complex-level enrichment for current-only complexes
  Part 4 — Feature stability (v3-derived vs v5-derived features)

  v3-specific handling:
    - Compressed: zipfile containing allComplexes.json
    - Multi-organism: filter on Organism == "Human"
    - Different schema: gene names in 'subunits(Gene name)' (semicolon-separated)

Reuses existing intermediate artifacts:
    gene_level_matrices.npz, corum_full_knn_indices.npz

Usage (from variant-viewer root):
    uv run python evee-analysis/scripts/run_corum_forward_prediction.py
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn2
import numpy as np
import polars as pl
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EEVE_ROOT = REPO_ROOT / "evee-analysis"
DATA_DIR = EEVE_ROOT / "data"
INT_DIR = DATA_DIR / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"

CORUM_V3_PATH = DATA_DIR / "corumv3_humanComplexes.json.zip"
CORUM_CURRENT_PATH = DATA_DIR / "corum_humanComplexes.json"

PREFIX = "corum_forward_"
FIG_PREFIX = "20260409_corum_forward_"

# ── Constants ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_BOOTSTRAP = 5000
K_VALUES = [5, 10, 20, 50]
MIN_COMPLEX_GENES = 3
N_RANDOM_PAIRS = 5000
FDR_THRESHOLD = 0.05
MIN_EMBEDDED_GENES_PER_COMPLEX = 5
N_TOP_ENTRIES = 20
STANDARD_EMBED_THRESHOLD = 5
REDUCED_EMBED_THRESHOLD = 3


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    fdr = p_values * n / ranks
    fdr_sorted = fdr[order]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[order] = fdr_sorted
    return np.clip(fdr, 0, 1)


# ══════════════════════════════════════════════════════════════════════
# CORUM loading helpers
# ══════════════════════════════════════════════════════════════════════

def load_corum_v3(path: Path) -> list[dict]:
    """Load CORUM v3 from zip, filter to Human, parse semicolon gene names."""
    zf = zipfile.ZipFile(str(path))
    raw = json.loads(zf.read("allComplexes.json"))
    zf.close()

    complexes = []
    for cx in raw:
        if cx.get("Organism") != "Human":
            continue
        gn_str = cx.get("subunits(Gene name)", "")
        if not gn_str:
            continue
        genes = {g.strip().upper() for g in gn_str.split(";") if g.strip()}
        if len(genes) < MIN_COMPLEX_GENES:
            continue
        complexes.append({
            "complex_id": cx["ComplexID"],
            "complex_name": cx.get("ComplexName", ""),
            "genes": genes,
        })
    return complexes


def load_corum_current(path: Path) -> list[dict]:
    """Load current CORUM (v4.1+ format with nested subunits)."""
    raw = json.loads(path.read_text())
    complexes = []
    for cx in raw:
        genes: set[str] = set()
        for su in cx.get("subunits", []):
            gn = su.get("swissprot", {}).get("gene_name", "")
            if gn:
                genes.add(gn.upper())
        if len(genes) < MIN_COMPLEX_GENES:
            continue
        complexes.append({
            "complex_id": cx["complex_id"],
            "complex_name": cx.get("complex_name", ""),
            "genes": genes,
        })
    return complexes


def build_cocomplex_pairs(
    complexes: list[dict], gene_filter: set[str] | None = None,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for cx in complexes:
        genes = cx["genes"] & gene_filter if gene_filter else cx["genes"]
        for a, b in combinations(sorted(genes), 2):
            pairs.add((a, b))
    return pairs


# ══════════════════════════════════════════════════════════════════════
# PART 1 — Formal database comparison
# ══════════════════════════════════════════════════════════════════════

def part1_database_comparison(
    old_cxs: list[dict],
    new_cxs: list[dict],
    emb_genes: set[str],
) -> dict:
    log.info("=" * 70)
    log.info("PART 1: Formal database comparison (v3 vs current)")
    log.info("=" * 70)

    old_ids = {c["complex_id"] for c in old_cxs}
    new_ids = {c["complex_id"] for c in new_cxs}
    shared_ids = old_ids & new_ids
    old_only_ids = old_ids - new_ids
    new_only_ids = new_ids - old_ids

    old_map = {c["complex_id"]: c for c in old_cxs}
    new_map = {c["complex_id"]: c for c in new_cxs}

    log.info(f"  Old (v3 Human): {len(old_cxs)} complexes (>= {MIN_COMPLEX_GENES} genes)")
    log.info(f"  Current: {len(new_cxs)} complexes")
    log.info(f"  Shared: {len(shared_ids)} | v3-only: {len(old_only_ids)} | Current-only: {len(new_only_ids)}")

    changed = []
    for cid in sorted(shared_ids):
        og = old_map[cid]["genes"]
        ng = new_map[cid]["genes"]
        if og != ng:
            changed.append({
                "complex_id": cid,
                "name": new_map[cid]["complex_name"],
                "added": sorted(ng - og),
                "removed": sorted(og - ng),
                "n_added": len(ng - og),
                "n_removed": len(og - ng),
            })
    log.info(f"  Shared complexes with changed membership: {len(changed)}")

    old_genes = set().union(*(c["genes"] for c in old_cxs))
    new_genes = set().union(*(c["genes"] for c in new_cxs))
    jaccard_genes = len(old_genes & new_genes) / len(old_genes | new_genes)
    log.info(f"  Gene universe: old={len(old_genes)}, new={len(new_genes)}, "
             f"shared={len(old_genes & new_genes)}, Jaccard={jaccard_genes:.4f}")

    old_pairs = build_cocomplex_pairs(old_cxs, emb_genes)
    new_pairs = build_cocomplex_pairs(new_cxs, emb_genes)
    novel_pairs = new_pairs - old_pairs
    lost_pairs = old_pairs - new_pairs
    shared_pairs = old_pairs & new_pairs

    log.info(f"  Co-complex pairs (embedded genes): old={len(old_pairs)}, new={len(new_pairs)}")
    log.info(f"    Shared: {len(shared_pairs)} | Novel: {len(novel_pairs)} | Lost: {len(lost_pairs)}")

    new_only_cxs = [new_map[cid] for cid in new_only_ids]
    new_only_emb_sizes = [len(c["genes"] & emb_genes) for c in new_only_cxs]
    ge3 = sum(1 for s in new_only_emb_sizes if s >= 3)
    ge5 = sum(1 for s in new_only_emb_sizes if s >= 5)
    log.info(f"  Current-only complexes: {len(new_only_cxs)}")
    log.info(f"    Embedded genes >= 3: {ge3}, >= 5: {ge5}")

    _fig_venn(old_ids, new_ids)
    _fig_size_distributions(old_cxs, new_cxs, new_only_cxs, emb_genes)

    return {
        "n_old": len(old_cxs),
        "n_new": len(new_cxs),
        "n_shared_ids": len(shared_ids),
        "n_old_only": len(old_only_ids),
        "n_new_only": len(new_only_ids),
        "n_changed_membership": len(changed),
        "changed_details": changed[:20],
        "gene_jaccard": jaccard_genes,
        "n_old_genes": len(old_genes),
        "n_new_genes": len(new_genes),
        "n_shared_genes": len(old_genes & new_genes),
        "n_old_pairs": len(old_pairs),
        "n_new_pairs": len(new_pairs),
        "n_novel_pairs": len(novel_pairs),
        "n_lost_pairs": len(lost_pairs),
        "n_shared_pairs": len(shared_pairs),
        "new_only_ge3_embedded": ge3,
        "new_only_ge5_embedded": ge5,
    }


def _fig_venn(old_ids: set[int], new_ids: set[int]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    venn2(
        [old_ids, new_ids],
        set_labels=("CORUM v3 (Human)", "CORUM current (v5)"),
        ax=ax,
    )
    ax.set_title("Complex ID overlap: CORUM v3 vs current", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}venn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}venn.png")


def _fig_size_distributions(
    old_cxs: list[dict],
    new_cxs: list[dict],
    new_only_cxs: list[dict],
    emb_genes: set[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    old_sizes = [len(c["genes"]) for c in old_cxs]
    new_sizes = [len(c["genes"]) for c in new_cxs]
    new_only_sizes = [len(c["genes"]) for c in new_only_cxs]
    upper = min(max(max(old_sizes), max(new_sizes)) + 2, 80)
    bins = np.arange(3, upper) - 0.5
    ax.hist(old_sizes, bins=bins, alpha=0.5, label=f"v3 Human (n={len(old_cxs)})", color="#4C72B0")
    ax.hist(new_sizes, bins=bins, alpha=0.5, label=f"Current (n={len(new_cxs)})", color="#DD8452")
    if new_only_sizes:
        ax.hist(new_only_sizes, bins=bins, alpha=0.7, label=f"Current-only (n={len(new_only_cxs)})",
                color="#C44E52", histtype="step", linewidth=2)
    ax.set_xlabel("Total genes per complex", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Complex size distribution (total genes)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(2, 60)

    ax = axes[1]
    old_emb = [len(c["genes"] & emb_genes) for c in old_cxs]
    new_emb = [len(c["genes"] & emb_genes) for c in new_cxs]
    new_only_emb = [len(c["genes"] & emb_genes) for c in new_only_cxs]
    upper_emb = min(max(max(old_emb), max(new_emb)) + 2, 80)
    bins_emb = np.arange(0, upper_emb) - 0.5
    ax.hist(old_emb, bins=bins_emb, alpha=0.5, label=f"v3 Human (n={len(old_cxs)})", color="#4C72B0")
    ax.hist(new_emb, bins=bins_emb, alpha=0.5, label=f"Current (n={len(new_cxs)})", color="#DD8452")
    if new_only_emb:
        ax.hist(new_only_emb, bins=bins_emb, alpha=0.7, label=f"Current-only (n={len(new_only_cxs)})",
                color="#C44E52", histtype="step", linewidth=2)
    ax.axvline(5, color="gray", ls="--", lw=1, label="Standard threshold (5)")
    ax.axvline(3, color="gray", ls=":", lw=1, label="Reduced threshold (3)")
    ax.set_xlabel("Embedded genes per complex", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Complex size distribution (embedded genes only)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 60)

    fig.suptitle(
        "CORUM v3 vs current: complex size distributions\n"
        f"v3 Human = {len(old_cxs)} | Current = {len(new_cxs)} | "
        f"Current-only = {len(new_only_cxs)}",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}size_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}size_distributions.png")


# ══════════════════════════════════════════════════════════════════════
# PART 2 — Forward-in-time feature derivation (Stages 2-4 on v3)
# ══════════════════════════════════════════════════════════════════════

def part2_derive_v3_features(
    old_cxs: list[dict],
    zscored: np.ndarray,
    gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Re-run interpretability Stages 2-4 using ONLY v3 complexes.

    Returns (feature_indices, feature_weights, v3_recurrent_df).
    """
    log.info("=" * 70)
    log.info("PART 2: Forward-in-time feature derivation (v3 complexes only)")
    log.info("=" * 70)

    gene_set = set(gene_names)
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    n_entries = 4096

    # ── Stage 2: Build v3 gene sets ──
    log.info("  Stage 2: Filtering v3 complexes to embedded gene sets...")
    v3_complexes = []
    for cx in old_cxs:
        embedded = sorted(cx["genes"] & gene_set)
        if len(embedded) >= MIN_EMBEDDED_GENES_PER_COMPLEX:
            v3_complexes.append({
                "complex_id": cx["complex_id"],
                "complex_name": cx["complex_name"],
                "member_genes": embedded,
                "n_genes_embedded": len(embedded),
            })
    log.info(f"  Retained {len(v3_complexes)} v3 complexes (>= {MIN_EMBEDDED_GENES_PER_COMPLEX} embedded genes)")

    if not v3_complexes:
        log.warning("  No v3 complexes pass threshold. Cannot derive features.")
        return np.array([]), np.array([]), pl.DataFrame()

    sizes = [c["n_genes_embedded"] for c in v3_complexes]
    log.info(f"  Embedded genes/complex: min={min(sizes)}, median={np.median(sizes):.0f}, max={max(sizes)}")

    # ── Stage 3: Entry-wise enrichment (Welch t-test per v3 complex) ──
    log.info("  Stage 3: Entry-wise enrichment for v3 complexes...")
    ii_template, jj_template = np.divmod(np.arange(n_entries), 64)
    enrichment_dfs = []

    for ci, cx in enumerate(v3_complexes):
        members = cx["member_genes"]
        in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])
        out_mask = np.ones(n_genes, dtype=bool)
        out_mask[in_idx] = False

        n_in = len(in_idx)
        n_out = int(out_mask.sum())
        if n_in < 3 or n_out < 3:
            continue

        z_in = zscored[in_idx]
        z_out = zscored[out_mask]

        mean_in = z_in.mean(axis=0)
        mean_out = z_out.mean(axis=0)
        delta = mean_in - mean_out

        var_in = z_in.var(axis=0, ddof=1)
        var_out = z_out.var(axis=0, ddof=1)

        se = np.sqrt(var_in / n_in + var_out / n_out)
        se[se < 1e-10] = 1e-10
        t_stat = delta / se

        num = (var_in / n_in + var_out / n_out) ** 2
        denom = (var_in / n_in) ** 2 / (n_in - 1) + (var_out / n_out) ** 2 / (n_out - 1)
        denom[denom < 1e-10] = 1e-10
        df_ws = np.clip(num / denom, 1.0, None)
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df_ws)

        pooled_var = ((n_in - 1) * var_in + (n_out - 1) * var_out) / (n_in + n_out - 2)
        pooled_sd = np.sqrt(np.maximum(pooled_var, 1e-20))
        cohens_d = delta / pooled_sd

        fdr = benjamini_hochberg(p_vals)

        cx_df = pl.DataFrame({
            "complex_id": np.full(n_entries, cx["complex_id"], dtype=np.int64),
            "complex_name": [cx["complex_name"]] * n_entries,
            "i": ii_template.astype(np.int16),
            "j": jj_template.astype(np.int16),
            "effect_size": cohens_d.astype(np.float32),
            "fdr": fdr.astype(np.float32),
        })
        enrichment_dfs.append(cx_df)

        if (ci + 1) % 50 == 0 or ci < 5:
            n_sig = int((fdr < FDR_THRESHOLD).sum())
            log.info(f"    [{ci+1}/{len(v3_complexes)}] {cx['complex_name'][:40]}: "
                     f"{n_in} genes, {n_sig} sig entries")

    enrichment_df = pl.concat(enrichment_dfs)
    total_sig = enrichment_df.filter(pl.col("fdr") < FDR_THRESHOLD).height
    log.info(f"  Total enrichment rows: {enrichment_df.height:,}, significant: {total_sig:,}")

    # ── Stage 4: Top entries + recurrence ──
    log.info("  Stage 4: Top entries per complex + recurrence...")
    top_entries = (
        enrichment_df
        .sort(["complex_id", "fdr", pl.col("effect_size").abs()], descending=[False, False, True])
        .group_by("complex_id")
        .head(N_TOP_ENTRIES)
    )

    sig_top = top_entries.filter(pl.col("fdr") < FDR_THRESHOLD)
    rec_dict: dict[tuple[int, int], list] = defaultdict(list)
    for row in sig_top.iter_rows(named=True):
        rec_dict[(row["i"], row["j"])].append({
            "complex_id": row["complex_id"],
            "complex_name": row["complex_name"],
            "effect_size": row["effect_size"],
        })

    recurrence_rows = []
    for (i, j), entries in sorted(rec_dict.items()):
        recurrence_rows.append({
            "i": i,
            "j": j,
            "n_complexes": len(entries),
            "mean_effect": float(np.mean([e["effect_size"] for e in entries])),
            "complexes": json.dumps([e["complex_name"] for e in entries]),
        })

    v3_rec_df = pl.DataFrame(recurrence_rows).sort("n_complexes", descending=True)
    v3_rec_df.write_parquet(INT_DIR / f"{PREFIX}v3_derived_recurrent_entries.parquet")
    log.info(f"  Saved {PREFIX}v3_derived_recurrent_entries.parquet ({v3_rec_df.height} entries)")

    top_features = v3_rec_df.filter(pl.col("n_complexes") >= 3)
    n_predictive = top_features.height
    log.info(f"  v3-derived predictive features (recurrence >= 3): {n_predictive}")

    if n_predictive > 0:
        for r in top_features.head(5).iter_rows(named=True):
            log.info(f"    ({r['i']},{r['j']}): {r['n_complexes']} complexes, "
                     f"mean_d={r['mean_effect']:.3f}")

    feature_indices = []
    feature_weights = []
    for row in top_features.iter_rows(named=True):
        flat_idx = row["i"] * 64 + row["j"]
        feature_indices.append(flat_idx)
        feature_weights.append(abs(row["mean_effect"]))

    feature_indices = np.array(feature_indices)
    feature_weights = np.array(feature_weights)
    if len(feature_weights) > 0:
        feature_weights = feature_weights / feature_weights.sum()

    return feature_indices, feature_weights, v3_rec_df


# ══════════════════════════════════════════════════════════════════════
# PART 3A — kNN neighbor enrichment for novel pairs
# ══════════════════════════════════════════════════════════════════════

def test_a_knn_enrichment(
    novel_pairs: set[tuple[str, str]],
    old_pairs: set[tuple[str, str]],
    emb_genes: set[str],
    rng: np.random.Generator,
) -> dict:
    log.info("=" * 70)
    log.info("TEST A: kNN neighbor enrichment for novel co-complex pairs")
    log.info("=" * 70)

    knn_data = np.load(str(INT_DIR / "corum_full_knn_indices.npz"), allow_pickle=True)
    knn_indices = knn_data["knn_indices"]
    vid_order = list(knn_data["vid_order"])

    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT variant_id, gene_name FROM variants WHERE gene_name IS NOT NULL"
    ).fetchall()
    con.close()
    vid_to_gene = {v: g.upper() for v, g in rows}
    genes_in_knn = {vid_to_gene[v] for v in vid_order if v in vid_to_gene}

    novel_in_knn = {(a, b) for a, b in novel_pairs if a in genes_in_knn and b in genes_in_knn}
    old_in_knn = {(a, b) for a, b in old_pairs if a in genes_in_knn and b in genes_in_knn}

    log.info(f"  Novel pairs in kNN universe: {len(novel_in_knn)} / {len(novel_pairs)}")
    log.info(f"  Old (v3) pairs in kNN universe: {len(old_in_knn)} / {len(old_pairs)}")

    all_positive = old_in_knn | novel_in_knn
    knn_gene_list = sorted(genes_in_knn & emb_genes)
    random_pairs: set[tuple[str, str]] = set()
    while len(random_pairs) < N_RANDOM_PAIRS:
        i, j = rng.integers(0, len(knn_gene_list), size=2)
        if i == j:
            continue
        p = _ordered_pair(knn_gene_list[i], knn_gene_list[j])
        if p not in all_positive and p not in random_pairs:
            random_pairs.add(p)

    results_per_k = []
    for k in K_VALUES:
        predicted: set[tuple[str, str]] = set()
        for qi in range(len(vid_order)):
            src = vid_to_gene.get(vid_order[qi])
            if src is None or src not in emb_genes:
                continue
            for ni in knn_indices[qi, :k]:
                tgt = vid_to_gene.get(vid_order[ni])
                if tgt is None or tgt == src or tgt not in emb_genes:
                    continue
                predicted.add(_ordered_pair(src, tgt))

        novel_hit = len(novel_in_knn & predicted)
        novel_total = len(novel_in_knn)
        old_hit = len(old_in_knn & predicted)
        old_total = len(old_in_knn)
        random_hit = len(random_pairs & predicted)
        random_total = len(random_pairs)

        novel_rate = novel_hit / novel_total if novel_total > 0 else 0
        old_rate = old_hit / old_total if old_total > 0 else 0
        random_rate = random_hit / random_total if random_total > 0 else 0

        novel_miss = novel_total - novel_hit
        random_miss = random_total - random_hit
        if novel_total > 0 and random_total > 0:
            table = np.array([[novel_hit, novel_miss], [random_hit, random_miss]])
            odds_ratio, fisher_p = stats.fisher_exact(table, alternative="greater")
        else:
            odds_ratio, fisher_p = float("nan"), float("nan")

        novel_genes_involved = set()
        for a, b in novel_in_knn:
            novel_genes_involved.add(a)
            novel_genes_involved.add(b)
        novel_gene_list = sorted(novel_genes_involved)
        gene_hit = defaultdict(int)
        gene_total_pairs = defaultdict(int)
        for a, b in novel_in_knn:
            gene_total_pairs[a] += 1
            gene_total_pairs[b] += 1
            if (a, b) in predicted:
                gene_hit[a] += 1
                gene_hit[b] += 1

        hit_arr = np.array([gene_hit.get(g, 0) for g in novel_gene_list])
        total_arr = np.array([gene_total_pairs.get(g, 0) for g in novel_gene_list])
        boot_rates = np.empty(N_BOOTSTRAP)
        n_genes_boot = len(novel_gene_list)
        if n_genes_boot > 0:
            for bi in range(N_BOOTSTRAP):
                idx = rng.integers(0, n_genes_boot, size=n_genes_boot)
                s_hit = hit_arr[idx].sum()
                s_total = total_arr[idx].sum()
                boot_rates[bi] = s_hit / s_total if s_total > 0 else 0
            ci_lo = float(np.percentile(boot_rates, 2.5))
            ci_hi = float(np.percentile(boot_rates, 97.5))
        else:
            ci_lo, ci_hi = 0.0, 0.0

        log.info(f"  k={k}: novel={novel_rate:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], "
                 f"old={old_rate:.4f}, random={random_rate:.4f}, "
                 f"OR={odds_ratio:.2f}, Fisher p={fisher_p:.3g}")

        results_per_k.append({
            "k": k,
            "novel_hit": novel_hit, "novel_total": novel_total, "novel_rate": novel_rate,
            "novel_ci_lo": ci_lo, "novel_ci_hi": ci_hi,
            "old_hit": old_hit, "old_total": old_total, "old_rate": old_rate,
            "random_hit": random_hit, "random_total": random_total, "random_rate": random_rate,
            "fisher_odds_ratio": odds_ratio, "fisher_p": fisher_p,
        })

    _fig_knn_enrichment(results_per_k)
    return {"results_per_k": results_per_k}


def _fig_knn_enrichment(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ks = [r["k"] for r in results]
    x = np.arange(len(ks))
    width = 0.25

    novel_rates = [r["novel_rate"] for r in results]
    old_rates = [r["old_rate"] for r in results]
    random_rates = [r["random_rate"] for r in results]
    novel_ci_lo = [r["novel_ci_lo"] for r in results]
    novel_ci_hi = [r["novel_ci_hi"] for r in results]
    novel_err = [
        [nr - lo for nr, lo in zip(novel_rates, novel_ci_lo)],
        [hi - nr for nr, hi in zip(novel_rates, novel_ci_hi)],
    ]

    ax.bar(x - width, novel_rates, width, label="Novel pairs (v5-only)", color="#C44E52",
           yerr=novel_err, capsize=3, error_kw={"lw": 1})
    ax.bar(x, old_rates, width, label="Known pairs (v3)", color="#4C72B0")
    ax.bar(x + width, random_rates, width, label="Random pairs", color="#CCCCCC")

    for i, r in enumerate(results):
        p = r["fisher_p"]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_pos = max(novel_rates[i] + (novel_err[1][i] if novel_err[1][i] else 0), old_rates[i]) + 0.005
        ax.text(x[i] - width / 2, y_pos, star, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=11)
    ax.set_ylabel("Fraction of pairs found as kNN neighbors", fontsize=11)
    ax.set_title(
        "Test A: kNN retrieval of novel co-complex pairs\n"
        "Novel = in current CORUM but not v3 | kNN on full Evo2 embeddings",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.text(
        0.5, -0.03,
        "Fisher's exact (one-sided): novel vs. random. "
        f"Novel n={results[0]['novel_total']:,}, Random n={results[0]['random_total']:,}. "
        "95% CI: gene-level bootstrap (5,000 iter).",
        ha="center", fontsize=8, color="gray",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}knn_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}knn_enrichment.png")


# ══════════════════════════════════════════════════════════════════════
# PART 3B — Latent space cosine similarity
# ══════════════════════════════════════════════════════════════════════

def test_b_latent_distance(
    novel_pairs: set[tuple[str, str]],
    old_pairs: set[tuple[str, str]],
    zscored: np.ndarray,
    gene_names: list[str],
    rng: np.random.Generator,
) -> dict:
    log.info("=" * 70)
    log.info("TEST B: Latent space cosine similarity")
    log.info("=" * 70)

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    gene_set = set(gene_names)

    novel_emb = [(a, b) for a, b in novel_pairs if a in gene_set and b in gene_set]
    old_sample = [(a, b) for a, b in old_pairs if a in gene_set and b in gene_set]
    if len(old_sample) > N_RANDOM_PAIRS:
        idx = rng.choice(len(old_sample), size=N_RANDOM_PAIRS, replace=False)
        old_sample = [old_sample[i] for i in idx]

    novel_subsample = novel_emb
    if len(novel_emb) > N_RANDOM_PAIRS:
        idx = rng.choice(len(novel_emb), size=N_RANDOM_PAIRS, replace=False)
        novel_subsample = [novel_emb[i] for i in idx]

    gene_list = sorted(gene_set)
    all_positive = old_pairs | novel_pairs
    random_sample: list[tuple[str, str]] = []
    while len(random_sample) < N_RANDOM_PAIRS:
        i, j = rng.integers(0, len(gene_list), size=2)
        if i == j:
            continue
        p = _ordered_pair(gene_list[i], gene_list[j])
        if p not in all_positive:
            random_sample.append(p)

    def _cosine_sims(pairs: list[tuple[str, str]]) -> np.ndarray:
        sims = []
        for a, b in pairs:
            if a not in gene_idx or b not in gene_idx:
                continue
            va = zscored[gene_idx[a]]
            vb = zscored[gene_idx[b]]
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a < 1e-10 or norm_b < 1e-10:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(va, vb) / (norm_a * norm_b)))
        return np.array(sims)

    novel_sims = _cosine_sims(novel_subsample)
    old_sims = _cosine_sims(old_sample)
    random_sims = _cosine_sims(random_sample)

    log.info(f"  Novel: n={len(novel_sims)}, mean={novel_sims.mean():.4f}, median={np.median(novel_sims):.4f}")
    log.info(f"  Old:   n={len(old_sims)}, mean={old_sims.mean():.4f}, median={np.median(old_sims):.4f}")
    log.info(f"  Random: n={len(random_sims)}, mean={random_sims.mean():.4f}, median={np.median(random_sims):.4f}")

    if len(novel_sims) > 0 and len(random_sims) > 0:
        u_stat, mw_p = stats.mannwhitneyu(novel_sims, random_sims, alternative="greater")
        rank_biserial = 1 - (2 * u_stat) / (len(novel_sims) * len(random_sims))
    else:
        u_stat, mw_p, rank_biserial = float("nan"), float("nan"), float("nan")

    if len(novel_sims) > 0 and len(old_sims) > 0:
        u_old, mw_p_old = stats.mannwhitneyu(novel_sims, old_sims, alternative="two-sided")
    else:
        u_old, mw_p_old = float("nan"), float("nan")

    log.info(f"  Novel vs Random: U={u_stat:.0f}, p={mw_p:.3g}, rank-biserial r={rank_biserial:.4f}")
    log.info(f"  Novel vs Old: U={u_old:.0f}, p={mw_p_old:.3g}")

    _fig_latent_distance(novel_sims, old_sims, random_sims, mw_p)

    return {
        "n_novel": len(novel_sims),
        "n_old_sample": len(old_sims),
        "n_random": len(random_sims),
        "novel_mean": float(novel_sims.mean()) if len(novel_sims) > 0 else None,
        "novel_median": float(np.median(novel_sims)) if len(novel_sims) > 0 else None,
        "old_mean": float(old_sims.mean()),
        "old_median": float(np.median(old_sims)),
        "random_mean": float(random_sims.mean()),
        "random_median": float(np.median(random_sims)),
        "novel_vs_random_U": float(u_stat),
        "novel_vs_random_p": float(mw_p),
        "novel_vs_random_rank_biserial": float(rank_biserial),
        "novel_vs_old_p": float(mw_p_old),
    }


def _fig_latent_distance(
    novel_sims: np.ndarray,
    old_sims: np.ndarray,
    random_sims: np.ndarray,
    mw_p: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(-0.3, 0.8, 60)
    ax.hist(random_sims, bins=bins, alpha=0.5, label=f"Random (n={len(random_sims):,})",
            color="#CCCCCC", density=True)
    ax.hist(old_sims, bins=bins, alpha=0.5, label=f"v3 co-complex (n={len(old_sims):,})",
            color="#4C72B0", density=True)
    ax.hist(novel_sims, bins=bins, alpha=0.7, label=f"Novel co-complex (n={len(novel_sims):,})",
            color="#C44E52", density=True)
    ax.set_xlabel("Cosine similarity (z-scored 4096-dim space)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of pairwise cosine similarity", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [random_sims, old_sims, novel_sims]
    labels = ["Random", "v3 co-complex", "Novel (v5-only)"]
    colors = ["#CCCCCC", "#4C72B0", "#C44E52"]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cosine similarity", fontsize=10)
    ax.set_title("Pairwise cosine similarity by pair class", fontsize=11)

    p_str = f"p={mw_p:.2e}" if mw_p < 0.001 else f"p={mw_p:.4f}"
    ax.text(0.5, 0.95, f"Novel vs. Random (Mann-Whitney U): {p_str}",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    fig.suptitle(
        "Test B: Latent space proximity of novel co-complex pairs\n"
        "Cosine similarity of gene-level z-scored 64x64 embedding vectors",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}latent_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}latent_distance.png")


# ══════════════════════════════════════════════════════════════════════
# PART 3C — Feature score prediction (using v3-derived features)
# ══════════════════════════════════════════════════════════════════════

def test_c_feature_scores(
    novel_pairs: set[tuple[str, str]],
    old_pairs: set[tuple[str, str]],
    zscored: np.ndarray,
    gene_names: list[str],
    v3_feature_indices: np.ndarray,
    v3_feature_weights: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    log.info("=" * 70)
    log.info("TEST C: v3-derived feature score prediction (FORWARD-IN-TIME)")
    log.info("=" * 70)

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    gene_set = set(gene_names)

    if len(v3_feature_indices) == 0:
        log.warning("  No v3-derived features available. Skipping Test C.")
        return {"n_predictive_features": 0}

    log.info(f"  Using {len(v3_feature_indices)} v3-derived predictive features")

    def _pair_score(a: str, b: str) -> float | None:
        if a not in gene_idx or b not in gene_idx:
            return None
        va = zscored[gene_idx[a], v3_feature_indices]
        vb = zscored[gene_idx[b], v3_feature_indices]
        combined = np.abs(va + vb)
        return float(np.average(combined, weights=v3_feature_weights))

    novel_emb = [(a, b) for a, b in novel_pairs if a in gene_set and b in gene_set]
    old_sample = [(a, b) for a, b in old_pairs if a in gene_set and b in gene_set]
    if len(old_sample) > N_RANDOM_PAIRS:
        idx = rng.choice(len(old_sample), size=N_RANDOM_PAIRS, replace=False)
        old_sample = [old_sample[i] for i in idx]

    all_positive = old_pairs | novel_pairs
    random_sample: list[tuple[str, str]] = []
    gene_list = sorted(gene_set)
    while len(random_sample) < N_RANDOM_PAIRS:
        i, j = rng.integers(0, len(gene_list), size=2)
        if i == j:
            continue
        p = _ordered_pair(gene_list[i], gene_list[j])
        if p not in all_positive:
            random_sample.append(p)

    novel_scores = np.array([s for s in (_pair_score(a, b) for a, b in novel_emb) if s is not None])
    old_scores = np.array([s for s in (_pair_score(a, b) for a, b in old_sample) if s is not None])
    random_scores = np.array([s for s in (_pair_score(a, b) for a, b in random_sample) if s is not None])

    log.info(f"  Novel: n={len(novel_scores):,}, mean={novel_scores.mean():.4f}")
    log.info(f"  Old:   n={len(old_scores):,}, mean={old_scores.mean():.4f}")
    log.info(f"  Random: n={len(random_scores):,}, mean={random_scores.mean():.4f}")

    if len(novel_scores) > 0 and len(random_scores) > 0:
        u_stat, mw_p = stats.mannwhitneyu(novel_scores, random_scores, alternative="greater")
    else:
        u_stat, mw_p = float("nan"), float("nan")

    if len(novel_scores) > 1 and len(random_scores) > 1:
        pooled_std = np.sqrt(
            ((len(novel_scores) - 1) * novel_scores.var(ddof=1) +
             (len(random_scores) - 1) * random_scores.var(ddof=1)) /
            (len(novel_scores) + len(random_scores) - 2)
        )
        cohens_d = (novel_scores.mean() - random_scores.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = float("nan")

    log.info(f"  Novel vs Random: U={u_stat:.0f}, p={mw_p:.3g}, Cohen's d={cohens_d:.3f}")

    _fig_feature_scores(novel_scores, old_scores, random_scores, mw_p, cohens_d)

    return {
        "n_predictive_features": len(v3_feature_indices),
        "n_novel": len(novel_scores),
        "n_old": len(old_scores),
        "n_random": len(random_scores),
        "novel_mean": float(novel_scores.mean()),
        "old_mean": float(old_scores.mean()),
        "random_mean": float(random_scores.mean()),
        "novel_vs_random_p": float(mw_p),
        "novel_vs_random_cohens_d": float(cohens_d),
    }


def _fig_feature_scores(
    novel_scores: np.ndarray,
    old_scores: np.ndarray,
    random_scores: np.ndarray,
    mw_p: float,
    cohens_d: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    all_vals = np.concatenate([novel_scores, old_scores, random_scores])
    bins = np.linspace(all_vals.min() - 0.1, np.percentile(all_vals, 99), 50)
    ax.hist(random_scores, bins=bins, alpha=0.5, label=f"Random (n={len(random_scores):,})",
            color="#CCCCCC", density=True)
    ax.hist(old_scores, bins=bins, alpha=0.5, label=f"v3 co-complex (n={len(old_scores):,})",
            color="#4C72B0", density=True)
    ax.hist(novel_scores, bins=bins, alpha=0.7, label=f"Novel (v5-only, n={len(novel_scores):,})",
            color="#C44E52", density=True)
    ax.set_xlabel("Feature score (weighted |z_a + z_b| at v3-derived entries)", fontsize=9)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of v3-derived feature scores", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [random_scores, old_scores, novel_scores]
    labels = ["Random", "v3 co-complex", "Novel (v5-only)"]
    colors = ["#CCCCCC", "#4C72B0", "#C44E52"]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Feature score", fontsize=10)
    ax.set_title("v3-derived feature scores by pair class", fontsize=11)

    p_str = f"p={mw_p:.2e}" if mw_p < 0.001 else f"p={mw_p:.4f}"
    ax.text(0.5, 0.95, f"Novel vs. Random: {p_str}, d={cohens_d:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    fig.suptitle(
        "Test C (Forward-in-Time): Features derived from v3 complexes\n"
        "predict elevated scores for novel (v5-only) co-complex pairs",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}feature_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}feature_scores.png")


# ══════════════════════════════════════════════════════════════════════
# PART 3D — Complex-level enrichment for current-only complexes
# ══════════════════════════════════════════════════════════════════════

def test_d_complex_enrichment(
    new_cxs: list[dict],
    old_ids: set[int],
    zscored: np.ndarray,
    gene_names: list[str],
    emb_genes: set[str],
) -> dict:
    log.info("=" * 70)
    log.info("TEST D: Complex-level enrichment for current-only complexes")
    log.info("=" * 70)

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    n_entries = 4096

    new_only_cxs = [c for c in new_cxs if c["complex_id"] not in old_ids]

    testable_t5 = []
    testable_t3 = []
    for cx in new_only_cxs:
        embedded = sorted(cx["genes"] & emb_genes)
        n_emb = len(embedded)
        if n_emb >= STANDARD_EMBED_THRESHOLD:
            testable_t5.append({**cx, "embedded_genes": embedded})
        if n_emb >= REDUCED_EMBED_THRESHOLD:
            testable_t3.append({**cx, "embedded_genes": embedded})

    log.info(f"  Current-only complexes with >= {STANDARD_EMBED_THRESHOLD} embedded genes: {len(testable_t5)}")
    log.info(f"  Current-only complexes with >= {REDUCED_EMBED_THRESHOLD} embedded genes: {len(testable_t3)}")

    # Baseline from existing v5 enrichment
    existing_enrich = pl.read_parquet(INT_DIR / "corum_entry_enrichment.parquet")
    existing_nsig = (
        existing_enrich.filter(pl.col("fdr") < FDR_THRESHOLD)
        .group_by("complex_id").len().rename({"len": "n_sig"})
    )
    baseline_median = float(existing_nsig["n_sig"].median()) if existing_nsig.height > 0 else 0
    baseline_mean = float(existing_nsig["n_sig"].mean()) if existing_nsig.height > 0 else 0
    log.info(f"  Baseline (300 existing): median n_sig={baseline_median:.0f}, mean={baseline_mean:.0f}")

    def _run_enrichment(testable: list[dict]) -> list[dict]:
        results_list = []
        for cx in testable:
            members = cx["embedded_genes"]
            in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])
            out_mask = np.ones(n_genes, dtype=bool)
            out_mask[in_idx] = False

            n_in = len(in_idx)
            n_out = int(out_mask.sum())
            if n_in < 2 or n_out < 2:
                continue

            z_in = zscored[in_idx]
            z_out = zscored[out_mask]
            mean_in = z_in.mean(axis=0)
            mean_out = z_out.mean(axis=0)
            delta = mean_in - mean_out

            var_in = z_in.var(axis=0, ddof=1)
            var_out = z_out.var(axis=0, ddof=1)
            se = np.sqrt(var_in / n_in + var_out / n_out)
            se[se < 1e-10] = 1e-10
            t_stat = delta / se

            num = (var_in / n_in + var_out / n_out) ** 2
            denom = (var_in / n_in) ** 2 / max(n_in - 1, 1) + (var_out / n_out) ** 2 / (n_out - 1)
            denom[denom < 1e-10] = 1e-10
            df_ws = np.clip(num / denom, 1.0, None)
            p_vals = 2 * stats.t.sf(np.abs(t_stat), df_ws)

            fdr = benjamini_hochberg(p_vals)
            n_sig = int((fdr < FDR_THRESHOLD).sum())
            mean_abs_effect = float(np.abs(delta[fdr < FDR_THRESHOLD]).mean()) if n_sig > 0 else 0

            results_list.append({
                "complex_id": cx["complex_id"],
                "complex_name": cx["complex_name"],
                "n_genes_embedded": n_in,
                "n_sig_entries": n_sig,
                "mean_abs_delta_sig": mean_abs_effect,
            })
        return results_list

    results_t5 = _run_enrichment(testable_t5)
    results_t3 = _run_enrichment(testable_t3)

    for r in sorted(results_t5, key=lambda x: -x["n_sig_entries"])[:10]:
        log.info(f"    [t5] {r['complex_name'][:45]}: {r['n_genes_embedded']} genes, "
                 f"{r['n_sig_entries']} sig entries")

    t5_with_sig = sum(1 for r in results_t5 if r["n_sig_entries"] > 0)
    t3_with_sig = sum(1 for r in results_t3 if r["n_sig_entries"] > 0)
    log.info(f"  Threshold 5: {t5_with_sig}/{len(results_t5)} have >= 1 sig entry")
    log.info(f"  Threshold 3: {t3_with_sig}/{len(results_t3)} have >= 1 sig entry")

    _fig_complex_enrichment(results_t5, results_t3, existing_nsig)

    return {
        "n_testable_t5": len(results_t5),
        "n_testable_t3": len(results_t3),
        "n_with_sig_t5": t5_with_sig,
        "n_with_sig_t3": t3_with_sig,
        "baseline_median_nsig": baseline_median,
        "results_t5": results_t5,
        "results_t3": results_t3,
    }


def _fig_complex_enrichment(
    results_t5: list[dict],
    results_t3: list[dict],
    existing_nsig: pl.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    existing_cx = pl.read_parquet(INT_DIR / "corum_complex_gene_sets.parquet")
    existing_sizes = []
    existing_sigs = []
    for row in existing_cx.iter_rows(named=True):
        cid = row["complex_id"]
        nsig_row = existing_nsig.filter(pl.col("complex_id") == cid)
        nsig = nsig_row["n_sig"][0] if nsig_row.height > 0 else 0
        existing_sizes.append(row["n_genes_embedded"])
        existing_sigs.append(nsig)

    ax.scatter(existing_sizes, existing_sigs, alpha=0.3, s=15, color="#4C72B0",
               label="Existing (v5 baseline)")
    for r in results_t5:
        ax.scatter(r["n_genes_embedded"], r["n_sig_entries"], s=60, color="#C44E52",
                   marker="*", zorder=5, alpha=0.7)

    ax.set_xlabel("Number of embedded genes", fontsize=10)
    ax.set_ylabel("Significant entries (FDR < 0.05)", fontsize=10)
    ax.set_title(f"Novel complexes vs existing: enrichment signal\n"
                 f"(n={len(results_t5)} novel at threshold >= 5)", fontsize=11)
    new_patch = mpatches.Patch(color="#C44E52", label=f"Current-only (n={len(results_t5)})")
    old_patch = mpatches.Patch(color="#4C72B0", label="Existing complexes (n=300)")
    ax.legend(handles=[old_patch, new_patch], fontsize=9)

    ax = axes[1]
    new_sizes = [r["n_genes_embedded"] for r in results_t5]
    new_sigs = [r["n_sig_entries"] for r in results_t5]
    if new_sizes:
        size_lo, size_hi = min(new_sizes), max(new_sizes)
        matched_existing = [(s, n) for s, n in zip(existing_sizes, existing_sigs)
                            if size_lo <= s <= size_hi]
        matched_sigs = [n for _, n in matched_existing]

        data_to_plot = []
        labels_plot = []
        if matched_sigs:
            data_to_plot.append(matched_sigs)
            labels_plot.append(f"Existing\n(size {size_lo}-{size_hi},\nn={len(matched_sigs)})")
        data_to_plot.append(new_sigs)
        labels_plot.append(f"Current-only\n(n={len(new_sigs)})")

        bp = ax.boxplot(data_to_plot, tick_labels=labels_plot, patch_artist=True)
        box_colors = ["#4C72B0", "#C44E52"] if len(data_to_plot) == 2 else ["#C44E52"]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        if matched_sigs and new_sigs:
            u, p = stats.mannwhitneyu(new_sigs, matched_sigs, alternative="two-sided")
            p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.4f}"
            ax.text(0.5, 0.95, f"Mann-Whitney U: {p_str}",
                    transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    ax.set_ylabel("Significant entries (FDR < 0.05)", fontsize=10)
    ax.set_title("Size-matched comparison of enrichment signal", fontsize=11)

    fig.suptitle(
        "Test D: Entry-level enrichment for current-only complexes\n"
        f"n={len(results_t5)} at threshold >= 5, n={len(results_t3)} at threshold >= 3",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}complex_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}complex_enrichment.png")


# ══════════════════════════════════════════════════════════════════════
# PART 4 — Feature stability (v3-derived vs v5-derived)
# ══════════════════════════════════════════════════════════════════════

def part4_feature_stability(v3_rec_df: pl.DataFrame) -> dict:
    log.info("=" * 70)
    log.info("PART 4: Feature stability — v3-derived vs v5-derived features")
    log.info("=" * 70)

    v5_rec_path = INT_DIR / "corum_recurrent_entries.parquet"
    if not v5_rec_path.exists():
        log.warning("  v5-derived recurrent entries not found. Skipping Part 4.")
        return {}

    v5_rec_df = pl.read_parquet(v5_rec_path)

    v3_all = set(zip(v3_rec_df["i"].to_list(), v3_rec_df["j"].to_list()))
    v5_all = set(zip(v5_rec_df["i"].to_list(), v5_rec_df["j"].to_list()))

    v3_top = v3_rec_df.filter(pl.col("n_complexes") >= 3)
    v5_top = v5_rec_df.filter(pl.col("n_complexes") >= 3)
    v3_top_set = set(zip(v3_top["i"].to_list(), v3_top["j"].to_list()))
    v5_top_set = set(zip(v5_top["i"].to_list(), v5_top["j"].to_list()))

    overlap_all = v3_all & v5_all
    overlap_top = v3_top_set & v5_top_set
    jaccard_all = len(overlap_all) / len(v3_all | v5_all) if (v3_all | v5_all) else 0
    jaccard_top = len(overlap_top) / len(v3_top_set | v5_top_set) if (v3_top_set | v5_top_set) else 0

    log.info(f"  v3 recurrent entries: {len(v3_all)} total, {len(v3_top_set)} with >= 3 complexes")
    log.info(f"  v5 recurrent entries: {len(v5_all)} total, {len(v5_top_set)} with >= 3 complexes")
    log.info(f"  Overlap (all): {len(overlap_all)}, Jaccard={jaccard_all:.4f}")
    log.info(f"  Overlap (top, >= 3): {len(overlap_top)}, Jaccard={jaccard_top:.4f}")

    # Spearman on shared entries (effect sizes)
    v3_effects = {(r["i"], r["j"]): r["mean_effect"] for r in v3_rec_df.iter_rows(named=True)}
    v5_effects = {(r["i"], r["j"]): r["mean_effect"] for r in v5_rec_df.iter_rows(named=True)}

    shared_keys = sorted(overlap_all)
    if len(shared_keys) >= 3:
        v3_vals = np.array([v3_effects[k] for k in shared_keys])
        v5_vals = np.array([v5_effects[k] for k in shared_keys])
        spearman_r, spearman_p = stats.spearmanr(v3_vals, v5_vals)
        pearson_r, pearson_p = stats.pearsonr(v3_vals, v5_vals)
        log.info(f"  Effect-size correlation (shared entries): Spearman r={spearman_r:.4f} (p={spearman_p:.3g}), "
                 f"Pearson r={pearson_r:.4f}")
    else:
        spearman_r, spearman_p = float("nan"), float("nan")
        pearson_r, pearson_p = float("nan"), float("nan")

    # Heatmaps
    _fig_feature_stability(v3_rec_df, v5_rec_df, v3_top_set, v5_top_set, overlap_top)

    return {
        "n_v3_all": len(v3_all),
        "n_v5_all": len(v5_all),
        "n_v3_top": len(v3_top_set),
        "n_v5_top": len(v5_top_set),
        "overlap_all": len(overlap_all),
        "overlap_top": len(overlap_top),
        "jaccard_all": jaccard_all,
        "jaccard_top": jaccard_top,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    }


def _fig_feature_stability(
    v3_rec_df: pl.DataFrame,
    v5_rec_df: pl.DataFrame,
    v3_top_set: set[tuple[int, int]],
    v5_top_set: set[tuple[int, int]],
    overlap_top: set[tuple[int, int]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: v3 recurrence heatmap
    hmap_v3 = np.zeros((64, 64), dtype=int)
    for r in v3_rec_df.iter_rows(named=True):
        hmap_v3[r["i"], r["j"]] = r["n_complexes"]
    ax = axes[0]
    im = ax.imshow(hmap_v3, cmap="YlOrRd", aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="# complexes")
    ax.set_title(f"v3-derived features\n({len(v3_top_set)} with >= 3 cx)", fontsize=11)
    ax.set_xlabel("j", fontsize=10)
    ax.set_ylabel("i", fontsize=10)

    # Panel B: v5 recurrence heatmap
    hmap_v5 = np.zeros((64, 64), dtype=int)
    for r in v5_rec_df.iter_rows(named=True):
        hmap_v5[r["i"], r["j"]] = r["n_complexes"]
    ax = axes[1]
    im = ax.imshow(hmap_v5, cmap="YlOrRd", aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="# complexes")
    ax.set_title(f"v5-derived features\n({len(v5_top_set)} with >= 3 cx)", fontsize=11)
    ax.set_xlabel("j", fontsize=10)
    ax.set_ylabel("i", fontsize=10)

    # Panel C: overlap (binary: v3-only, v5-only, shared)
    overlap_map = np.zeros((64, 64), dtype=float)
    for i, j in v3_top_set:
        overlap_map[i, j] = 1.0  # v3-only
    for i, j in v5_top_set:
        if (i, j) in overlap_top:
            overlap_map[i, j] = 3.0  # shared
        else:
            overlap_map[i, j] = 2.0  # v5-only
    ax = axes[2]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["white", "#4C72B0", "#DD8452", "#55A868"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(overlap_map, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    patches = [
        mpatches.Patch(color="#4C72B0", label=f"v3-only ({len(v3_top_set - overlap_top)})"),
        mpatches.Patch(color="#DD8452", label=f"v5-only ({len(v5_top_set - overlap_top)})"),
        mpatches.Patch(color="#55A868", label=f"Shared ({len(overlap_top)})"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.set_title(f"Feature overlap (top entries, >= 3 cx)\nJaccard = {len(overlap_top)/len(v3_top_set | v5_top_set):.3f}" if (v3_top_set | v5_top_set) else "Feature overlap", fontsize=11)
    ax.set_xlabel("j", fontsize=10)
    ax.set_ylabel("i", fontsize=10)

    fig.suptitle(
        "Part 4: Feature stability — v3-derived vs v5-derived recurrent entries\n"
        "Are the same latent features predictive across CORUM versions?",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}feature_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PREFIX}feature_stability.png")


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

def write_report(
    part1: dict,
    n_v3_complexes_used: int,
    n_v3_features: int,
    test_a: dict,
    test_b: dict,
    test_c: dict,
    test_d: dict,
    stability: dict,
) -> None:
    log.info("Writing report...")

    pct_shared = 100 * part1["n_shared_ids"] / part1["n_old"] if part1["n_old"] > 0 else 0

    lines = [
        "# Forward-in-Time CORUM Prediction: v3 → v5",
        "",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Methodology",
        "",
        "This analysis corrects a circularity in the previous CORUM version comparison.",
        "Previously, predictive features (recurrent matrix entries) were derived from the",
        "current (v5) CORUM dataset and used to score v5-novel pairs — a partially circular test.",
        "",
        "**Corrected approach:**",
        "",
        "1. Derive predictive features solely from v3 CORUM complexes (Stages 2-4 of the",
        "   interpretability pipeline, re-run on v3 data only)",
        "2. Use those v3-derived features to predict novel (v5-only) co-complex pairs",
        "3. Compare v3-derived features to v5-derived features (meta-analysis of stability)",
        "",
        "Tests A (kNN) and B (cosine similarity) use the raw embedding space and are unaffected.",
        "Test D (complex-level enrichment) re-computes enrichment from scratch per complex.",
        "**Only Test C (feature scores) required correction.**",
        "",
        "## Part 1: Database Comparison (v3 vs Current)",
        "",
        f"CORUM v3 (Human, >= 3 genes): **{part1['n_old']}** complexes. "
        f"Current (v5): **{part1['n_new']}** complexes. "
        f"{part1['n_shared_ids']} IDs ({pct_shared:.1f}%) shared.",
        "",
        f"The current version adds **{part1['n_new_only']}** new complexes, of which "
        f"{part1['new_only_ge5_embedded']} have >= 5 and "
        f"{part1['new_only_ge3_embedded']} have >= 3 embedded genes.",
        "",
        f"**{part1['n_novel_pairs']:,}** novel co-complex pairs (embedded genes) "
        f"appear in v5 but not v3.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| v3 Human complexes | {part1['n_old']} |",
        f"| Current complexes | {part1['n_new']} |",
        f"| Shared IDs | {part1['n_shared_ids']} ({pct_shared:.1f}%) |",
        f"| v3-only | {part1['n_old_only']} |",
        f"| Current-only | {part1['n_new_only']} |",
        f"| Gene Jaccard | {part1['gene_jaccard']:.4f} |",
        f"| Co-complex pairs (v3, embedded) | {part1['n_old_pairs']:,} |",
        f"| Co-complex pairs (v5, embedded) | {part1['n_new_pairs']:,} |",
        f"| Novel pairs | {part1['n_novel_pairs']:,} |",
        f"| Lost pairs | {part1['n_lost_pairs']:,} |",
        "",
        "## Part 2: Forward-in-Time Feature Derivation",
        "",
        f"Re-ran Stages 2-4 of the interpretability pipeline on **v3 complexes only**.",
        f"",
        f"- v3 complexes passing threshold (>= {MIN_EMBEDDED_GENES_PER_COMPLEX} embedded genes): "
        f"**{n_v3_complexes_used}**",
        f"- v3-derived predictive features (recurrence >= 3 complexes): **{n_v3_features}**",
        "",
        "These features have **zero knowledge** of v5 additions.",
        "",
    ]

    # Test A
    lines += [
        "## Test A: kNN Neighbor Enrichment",
        "",
        "**Question:** Are novel co-complex pairs enriched among kNN neighbors in Evo2 embedding space?",
        "",
        "| k | Novel rate [95% CI] | v3 rate | Random rate | Fisher p | OR |",
        "|---|---------------------|---------|-------------|----------|-----|",
    ]
    for r in test_a["results_per_k"]:
        lines.append(
            f"| {r['k']} | {r['novel_rate']:.4f} [{r['novel_ci_lo']:.4f}, {r['novel_ci_hi']:.4f}] "
            f"| {r['old_rate']:.4f} | {r['random_rate']:.4f} "
            f"| {r['fisher_p']:.3g} | {r['fisher_odds_ratio']:.2f} |"
        )
    lines.append("")

    # Test B
    lines += [
        "## Test B: Latent Space Cosine Similarity",
        "",
        f"- Novel: mean = {test_b['novel_mean']:.4f} (n={test_b['n_novel']:,})",
        f"- v3 co-complex: mean = {test_b['old_mean']:.4f} (n={test_b['n_old_sample']:,})",
        f"- Random: mean = {test_b['random_mean']:.4f} (n={test_b['n_random']:,})",
        f"- Mann-Whitney U (novel > random): p = {test_b['novel_vs_random_p']:.3g}",
        f"- Rank-biserial r = {test_b['novel_vs_random_rank_biserial']:.4f}",
        "",
    ]

    # Test C
    lines += [
        "## Test C: v3-Derived Feature Score Prediction (CORRECTED)",
        "",
        "**This is the key corrected test.** Features derived from v3 complexes are used to score",
        "novel (v5-only) pairs, eliminating the circularity of using v5-derived features.",
        "",
    ]
    if test_c.get("n_predictive_features", 0) > 0:
        lines += [
            f"- v3-derived predictive features: {test_c['n_predictive_features']}",
            f"- Novel pairs: mean score = {test_c['novel_mean']:.4f} (n={test_c['n_novel']:,})",
            f"- v3 co-complex: mean score = {test_c['old_mean']:.4f} (n={test_c['n_old']:,})",
            f"- Random: mean score = {test_c['random_mean']:.4f} (n={test_c['n_random']:,})",
            f"- Mann-Whitney U (novel > random): p = {test_c['novel_vs_random_p']:.3g}",
            f"- Cohen's d = {test_c['novel_vs_random_cohens_d']:.3f}",
        ]
    else:
        lines.append("No v3-derived features available (insufficient v3 complexes).")
    lines.append("")

    # Test D
    lines += [
        "## Test D: Complex-Level Enrichment",
        "",
        f"- Threshold >= 5: **{test_d['n_with_sig_t5']}/{test_d['n_testable_t5']}** "
        f"({100*test_d['n_with_sig_t5']/test_d['n_testable_t5']:.0f}%) "
        f"have >= 1 significant entry" if test_d["n_testable_t5"] > 0 else "",
        f"- Threshold >= 3: **{test_d['n_with_sig_t3']}/{test_d['n_testable_t3']}** "
        f"({100*test_d['n_with_sig_t3']/test_d['n_testable_t3']:.0f}%) "
        f"have >= 1 significant entry" if test_d["n_testable_t3"] > 0 else "",
        f"- Baseline (existing complexes): median sig entries = {test_d['baseline_median_nsig']:.0f}",
        "",
    ]

    if test_d.get("results_t5"):
        top10 = sorted(test_d["results_t5"], key=lambda x: -x["n_sig_entries"])[:15]
        lines.append("### Top current-only complexes (threshold >= 5)")
        lines.append("")
        lines.append("| Complex | Genes | Sig entries | Mean |delta| |")
        lines.append("|---------|-------|-------------|------------|")
        for r in top10:
            lines.append(
                f"| {r['complex_name'][:40]} | {r['n_genes_embedded']} "
                f"| {r['n_sig_entries']} | {r['mean_abs_delta_sig']:.3f} |"
            )
        lines.append("")

    # Part 4
    if stability:
        lines += [
            "## Part 4: Feature Stability (v3-derived vs v5-derived)",
            "",
            "How stable are the predictive features across CORUM database versions?",
            "",
            f"- v3-derived top features (>= 3 complexes): {stability['n_v3_top']}",
            f"- v5-derived top features (>= 3 complexes): {stability['n_v5_top']}",
            f"- Overlap: {stability['overlap_top']}",
            f"- Jaccard (top features): {stability['jaccard_top']:.4f}",
            f"- Spearman correlation (shared entry effect sizes): "
            f"r = {stability['spearman_r']:.4f} (p = {stability['spearman_p']:.3g})",
            f"- Pearson correlation: r = {stability['pearson_r']:.4f}",
            "",
        ]

    # Conclusions
    novel_detected_k10 = next((r for r in test_a["results_per_k"] if r["k"] == 10), None)
    test_a_sig = novel_detected_k10 and novel_detected_k10["fisher_p"] < 0.05
    test_b_sig = test_b["novel_vs_random_p"] < 0.05
    test_c_sig = test_c.get("novel_vs_random_p", 1.0) < 0.05

    lines += [
        "## Conclusions",
        "",
        "### Can v3-derived features predict v5-novel complexes?",
        "",
    ]

    if test_a_sig:
        lines.append(f"- **Test A (kNN): YES** — Novel pairs enriched among kNN neighbors "
                      f"(p={novel_detected_k10['fisher_p']:.3g}, OR={novel_detected_k10['fisher_odds_ratio']:.2f}).")
    else:
        qual = "Marginal" if novel_detected_k10 and novel_detected_k10["fisher_p"] < 0.1 else "No"
        p_val = novel_detected_k10['fisher_p'] if novel_detected_k10 else float("nan")
        lines.append(f"- **Test A (kNN): {qual}** — p={p_val:.3g}.")

    if test_b_sig:
        lines.append(f"- **Test B (Cosine): YES** — Novel pairs closer than random "
                      f"(p={test_b['novel_vs_random_p']:.3g}).")
    else:
        lines.append(f"- **Test B (Cosine): No** — p={test_b['novel_vs_random_p']:.3g}.")

    if test_c_sig:
        lines.append(f"- **Test C (v3-derived features): YES** — v3-derived features fire for novel pairs "
                      f"(p={test_c['novel_vs_random_p']:.3g}, d={test_c['novel_vs_random_cohens_d']:.2f}). "
                      f"**This is the corrected, non-circular test.**")
    elif test_c.get("n_predictive_features", 0) > 0:
        lines.append(f"- **Test C (v3-derived features): No** — p={test_c['novel_vs_random_p']:.3g}.")
    else:
        lines.append("- **Test C:** Could not evaluate (no v3 features).")

    if test_d["n_testable_t5"] > 0:
        frac_sig = test_d["n_with_sig_t5"] / test_d["n_testable_t5"]
        lines.append(f"- **Test D (Complex enrichment):** {100*frac_sig:.0f}% of {test_d['n_testable_t5']} "
                      f"novel complexes show significant signal.")

    if stability and stability.get("jaccard_top", 0) > 0:
        lines += [
            "",
            "### Feature stability",
            "",
            f"The v3-derived and v5-derived feature sets share {stability['overlap_top']} of "
            f"{len(set()) or stability['n_v3_top'] + stability['n_v5_top'] - stability['overlap_top']} "
            f"total features (Jaccard = {stability['jaccard_top']:.3f}). "
            f"Effect sizes on shared entries correlate at Spearman r = {stability['spearman_r']:.3f}. ",
        ]
        if stability["jaccard_top"] > 0.3:
            lines.append(
                "This indicates **substantial stability** in the features that the model deems "
                "predictive of protein complex membership, despite the ~decade gap between "
                "v3 and the current CORUM."
            )
        elif stability["jaccard_top"] > 0.1:
            lines.append(
                "This indicates **moderate stability** — some features are consistently "
                "predictive, while others are database-version-specific."
            )
        else:
            lines.append(
                "The low overlap suggests feature sets are **largely version-specific**. "
                "However, positive Test C results (if significant) indicate that v3 features "
                "still carry predictive power despite low overlap with v5 features."
            )

    lines += [
        "",
        "### Limitations",
        "",
        "- Gene name mappings may have evolved between CORUM versions (symbol updates, aliases)",
        "- v3 uses a different schema (semicolon gene names, different field names)",
        "- The embedding itself was trained on v5-era ClinVar data; this is not a limitation",
        "  for this test since we are testing CORUM complex membership, not variant pathogenicity",
        "",
        "## Reproducibility",
        "",
        f"- Random seed: {RANDOM_SEED}",
        f"- Bootstrap iterations: {N_BOOTSTRAP}",
        f"- k values: {K_VALUES}",
        f"- Random pairs: {N_RANDOM_PAIRS}",
        f"- v3 source: `{CORUM_V3_PATH.name}` (filtered to Organism=Human)",
        f"- v5 source: `{CORUM_CURRENT_PATH.name}`",
        "",
        "## Output Files",
        "",
        f"- `{FIG_PREFIX}venn.png`",
        f"- `{FIG_PREFIX}size_distributions.png`",
        f"- `{FIG_PREFIX}knn_enrichment.png`",
        f"- `{FIG_PREFIX}latent_distance.png`",
        f"- `{FIG_PREFIX}feature_scores.png`",
        f"- `{FIG_PREFIX}complex_enrichment.png`",
        f"- `{FIG_PREFIX}feature_stability.png`",
        f"- `{PREFIX}v3_derived_recurrent_entries.parquet`",
        f"- `{PREFIX}comparison.parquet`",
        f"- `{PREFIX}novel_pair_predictions.parquet`",
        f"- `{PREFIX}prediction_report.md`",
        f"- `{PREFIX}config.json`",
    ]

    report_path = INT_DIR / f"{PREFIX}prediction_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log.info(f"  Saved {report_path.name}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    INT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load shared artifacts ──
    log.info("Loading shared artifacts...")
    npz_data = np.load(str(INT_DIR / "gene_level_matrices.npz"), allow_pickle=True)
    gene_names = list(npz_data["gene_names"])
    zscored = npz_data["zscored_flat"]
    emb_genes = set(gene_names)
    log.info(f"  {len(gene_names):,} genes, z-scored shape: {zscored.shape}")

    # ── Load both CORUM versions ──
    log.info("Loading CORUM databases...")
    old_cxs = load_corum_v3(CORUM_V3_PATH)
    new_cxs = load_corum_current(CORUM_CURRENT_PATH)
    log.info(f"  v3 Human: {len(old_cxs)} complexes")
    log.info(f"  Current: {len(new_cxs)} complexes")

    # ── Part 1: Database comparison ──
    part1 = part1_database_comparison(old_cxs, new_cxs, emb_genes)

    # ── Part 2: Forward-in-time feature derivation ──
    v3_feature_indices, v3_feature_weights, v3_rec_df = part2_derive_v3_features(
        old_cxs, zscored, gene_names
    )
    n_v3_complexes_used = v3_rec_df.filter(pl.col("n_complexes") >= 1).height if v3_rec_df.height > 0 else 0
    # More accurately: count complexes that passed the filter
    v3_filtered = [cx for cx in old_cxs if len(cx["genes"] & emb_genes) >= MIN_EMBEDDED_GENES_PER_COMPLEX]
    n_v3_complexes_used = len(v3_filtered)
    n_v3_features = len(v3_feature_indices)

    # Build pair sets
    old_pairs = build_cocomplex_pairs(old_cxs, emb_genes)
    new_pairs = build_cocomplex_pairs(new_cxs, emb_genes)
    novel_pairs = new_pairs - old_pairs
    old_ids = {c["complex_id"] for c in old_cxs}

    # ── Part 3: Prediction tests ──
    test_a = test_a_knn_enrichment(novel_pairs, old_pairs, emb_genes, rng)
    test_b = test_b_latent_distance(novel_pairs, old_pairs, zscored, gene_names, rng)
    test_c = test_c_feature_scores(
        novel_pairs, old_pairs, zscored, gene_names,
        v3_feature_indices, v3_feature_weights, rng,
    )
    test_d = test_d_complex_enrichment(new_cxs, old_ids, zscored, gene_names, emb_genes)

    # ── Part 4: Feature stability ──
    stability = part4_feature_stability(v3_rec_df)

    # ── Save comparison data ──
    comparison_rows = []
    for cx in new_cxs:
        cid = cx["complex_id"]
        comparison_rows.append({
            "complex_id": cid,
            "complex_name": cx["complex_name"],
            "n_genes": len(cx["genes"]),
            "n_genes_embedded": len(cx["genes"] & emb_genes),
            "in_v3": cid in old_ids,
            "in_current": True,
        })
    for cx in old_cxs:
        if cx["complex_id"] not in {c["complex_id"] for c in new_cxs}:
            comparison_rows.append({
                "complex_id": cx["complex_id"],
                "complex_name": cx["complex_name"],
                "n_genes": len(cx["genes"]),
                "n_genes_embedded": len(cx["genes"] & emb_genes),
                "in_v3": True,
                "in_current": False,
            })
    pl.DataFrame(comparison_rows).write_parquet(INT_DIR / f"{PREFIX}comparison.parquet")
    log.info(f"Saved {PREFIX}comparison.parquet")

    # Save per-novel-pair predictions using v3 features
    pair_rows = []
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    for a, b in sorted(novel_pairs):
        row: dict = {"gene_a": a, "gene_b": b, "pair_class": "novel"}
        if a in gene_idx and b in gene_idx:
            va = zscored[gene_idx[a]]
            vb = zscored[gene_idx[b]]
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            row["cosine_similarity"] = float(np.dot(va, vb) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0
            if len(v3_feature_indices) > 0:
                combined = np.abs(va[v3_feature_indices] + vb[v3_feature_indices])
                row["v3_feature_score"] = float(np.average(combined, weights=v3_feature_weights))
            else:
                row["v3_feature_score"] = None
        else:
            row["cosine_similarity"] = None
            row["v3_feature_score"] = None
        pair_rows.append(row)
    pl.DataFrame(pair_rows).write_parquet(INT_DIR / f"{PREFIX}novel_pair_predictions.parquet")
    log.info(f"Saved {PREFIX}novel_pair_predictions.parquet ({len(pair_rows):,} rows)")

    # ── Report ──
    write_report(part1, n_v3_complexes_used, n_v3_features, test_a, test_b, test_c, test_d, stability)

    # ── Config ──
    config = {
        "analysis": "corum_forward_prediction",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "methodology": "forward-in-time: features derived from v3 only, tested on v5-novel pairs",
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_values": K_VALUES,
        "n_random_pairs": N_RANDOM_PAIRS,
        "min_complex_genes": MIN_COMPLEX_GENES,
        "min_embedded_genes_per_complex": MIN_EMBEDDED_GENES_PER_COMPLEX,
        "n_top_entries_per_complex": N_TOP_ENTRIES,
        "fdr_threshold": FDR_THRESHOLD,
        "corum_v3_path": str(CORUM_V3_PATH),
        "corum_current_path": str(CORUM_CURRENT_PATH),
        "v3_organism_filter": "Human",
        "n_genes": len(gene_names),
        "n_v3_complexes": len(old_cxs),
        "n_v3_complexes_for_features": n_v3_complexes_used,
        "n_current_complexes": len(new_cxs),
        "n_novel_pairs": len(novel_pairs),
        "n_v3_derived_features": n_v3_features,
    }
    with open(INT_DIR / f"{PREFIX}config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")
    print(f"\n{'='*80}")
    print(f"FORWARD-IN-TIME CORUM PREDICTION — COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
