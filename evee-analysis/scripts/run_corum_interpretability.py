#!/usr/bin/env python3
"""Feature-level CORUM interpretability on diff covariance matrices.

Stages:
  1. Gene-level 64x64 matrices (averaged across variants)
  2. CORUM gene sets (filtered, annotated)
  3. Entry-wise enrichment (Welch t, FDR per complex)
  4. Top entries per complex + global recurrence
  5. Delta heatmaps for selected complexes
  6. Complex signature clustering
  7. Broad class enrichment
  8. Robustness checks
  9-10. Report + interpretation

Usage (from variant-viewer root):
    uv run python eeve-analysis/scripts/run_corum_interpretability.py
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from safetensors.torch import load_file as safetensors_load
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    fdr = p_values * n / ranks
    # Enforce monotonicity (step-up)
    fdr_sorted = fdr[order]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[order] = fdr_sorted
    return np.clip(fdr, 0, 1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EEVE_ROOT = REPO_ROOT / "eeve-analysis"
DB_PATH = REPO_ROOT / "builds" / "variants.duckdb"
EMB_DIR = EEVE_ROOT / "data" / "clinvar-deconfounded-covariance64_pool"
CORUM_PATH = EEVE_ROOT / "data" / "corum_humanComplexes.json"
OUT_DIR = EEVE_ROOT / "data" / "intermediate"
FIG_DIR = EEVE_ROOT / "outputs" / "figures"

RANDOM_SEED = 42
MIN_VARIANTS_PER_GENE = 3
MIN_EMBEDDED_GENES_PER_COMPLEX = 5
N_TOP_ENTRIES = 20
FDR_THRESHOLD = 0.05
N_SELECTED_COMPLEXES = 8
N_SPLIT_HALF_ITERS = 100
MIN_CLASS_COMPLEXES = 3

KEYWORD_CLASSES = {
    "Ribosome": "ribosom",
    "Spliceosome": "spliceosom",
    "Proteasome": "proteasom",
    "Mitochondrial": "mitochond",
    "Chromatin/remodeling": "chromatin|remodel|nucleosome",
    "Transcription/polymerase": "transcript|polymerase",
    "Kinase": "kinase",
    "Histone": "histone",
    "Cohesin/condensin": "cohesin|condensin",
    "Exosome": "exosome",
    "Mediator": "mediator",
    "Integrin": "integrin",
    "Chaperonin": "chaperonin",
}


def enforce_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()[:60]


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — Gene-level matrices
# ══════════════════════════════════════════════════════════════════════

def stage1_gene_matrices() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load embeddings as [64,64], average per gene, z-score."""
    log.info("STAGE 1: Building gene-level matrices...")

    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT variant_id, gene_name FROM variants WHERE gene_name IS NOT NULL"
    ).fetchall()
    con.close()
    vid_to_gene = {v: g.upper() for v, g in rows}

    idx_conn = sqlite3.connect(str(EMB_DIR / "index.sqlite"))
    all_locs = idx_conn.execute("SELECT sequence_id, chunk_id, offset FROM sequence_locations").fetchall()
    idx_conn.close()

    vid_to_loc = {seq_id: (cid, off) for seq_id, cid, off in all_locs if seq_id in vid_to_gene}
    log.info(f"  {len(vid_to_loc):,} variants with both gene and embedding")

    gene_accum: dict[str, list[np.ndarray]] = defaultdict(list)
    by_chunk: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for vid, (cid, off) in vid_to_loc.items():
        by_chunk[cid].append((vid, off))

    for cid in sorted(by_chunk.keys()):
        chunk_path = EMB_DIR / "chunks" / f"chunk_{cid:06d}" / "activations.safetensors"
        tensor = safetensors_load(str(chunk_path))["activations"].float().numpy()
        for vid, off in by_chunk[cid]:
            mat = tensor[off]  # [64, 64]
            gene = vid_to_gene[vid]
            gene_accum[gene].append(mat)

    # Average per gene, filter by min variants
    gene_names = []
    matrices_list = []
    n_variants_list = []
    for gene in sorted(gene_accum.keys()):
        mats = gene_accum[gene]
        if len(mats) < MIN_VARIANTS_PER_GENE:
            continue
        gene_names.append(gene)
        mean_mat = np.mean(mats, axis=0).astype(np.float32)
        matrices_list.append(mean_mat)
        n_variants_list.append(len(mats))

    matrices = np.array(matrices_list)  # (N_genes, 64, 64)
    n_variants = np.array(n_variants_list)

    log.info(f"  Retained {len(gene_names):,} genes (>= {MIN_VARIANTS_PER_GENE} variants)")
    log.info(f"  Variants/gene: min={n_variants.min()}, median={np.median(n_variants):.0f}, "
             f"mean={n_variants.mean():.1f}, max={n_variants.max()}")

    # Z-score per entry across genes
    flat = matrices.reshape(len(gene_names), -1)  # (N_genes, 4096)
    mu = flat.mean(axis=0, keepdims=True)
    sigma = flat.std(axis=0, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    zscored = ((flat - mu) / sigma).astype(np.float32)

    # Save
    npz_path = OUT_DIR / "gene_level_matrices.npz"
    np.savez_compressed(
        str(npz_path),
        gene_names=np.array(gene_names, dtype=object),
        matrices=matrices,
        n_variants=n_variants,
        zscored_flat=zscored,
    )
    log.info(f"  Saved {npz_path.name} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    return matrices, zscored, gene_names, n_variants


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — CORUM gene sets
# ══════════════════════════════════════════════════════════════════════

def stage2_corum_gene_sets(gene_names: list[str]) -> pl.DataFrame:
    log.info("STAGE 2: Building CORUM gene sets...")
    gene_set = set(gene_names)
    raw = json.loads(CORUM_PATH.read_text())

    rows = []
    for cx in raw:
        cid = cx["complex_id"]
        cname = cx.get("complex_name", "")
        genes_corum: set[str] = set()
        for su in cx["subunits"]:
            gn = su.get("swissprot", {}).get("gene_name")
            if gn:
                genes_corum.add(gn.upper())
        if len(genes_corum) < 3:
            continue
        embedded = sorted(genes_corum & gene_set)
        if len(embedded) < MIN_EMBEDDED_GENES_PER_COMPLEX:
            continue

        fcg_cat = ""
        fcg_name = ""
        if cx.get("fcgs") and isinstance(cx["fcgs"], list) and len(cx["fcgs"]) > 0:
            fcg = cx["fcgs"][0].get("fcg", {})
            fcg_name = fcg.get("name", "")
            cat = fcg.get("fcg_category", {})
            fcg_cat = cat.get("name", "") if isinstance(cat, dict) else ""

        rows.append({
            "complex_id": cid,
            "complex_name": cname,
            "n_genes_corum": len(genes_corum),
            "n_genes_embedded": len(embedded),
            "member_genes": json.dumps(embedded),
            "fcg_category": fcg_cat,
            "fcg_name": fcg_name,
        })

    df = pl.DataFrame(rows).sort("n_genes_embedded", descending=True)
    df.write_parquet(OUT_DIR / "corum_complex_gene_sets.parquet")
    log.info(f"  Retained {df.height} complexes (>= {MIN_EMBEDDED_GENES_PER_COMPLEX} embedded genes)")
    sizes = df["n_genes_embedded"].to_numpy()
    log.info(f"  Embedded gene count: min={sizes.min()}, median={np.median(sizes):.0f}, "
             f"mean={sizes.mean():.1f}, max={sizes.max()}")
    return df


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — Entry-wise enrichment
# ══════════════════════════════════════════════════════════════════════

def stage3_entry_enrichment(
    zscored: np.ndarray,
    gene_names: list[str],
    complexes_df: pl.DataFrame,
) -> pl.DataFrame:
    log.info("STAGE 3: Entry-wise enrichment per complex...")
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    n_entries = 4096
    all_idx = np.arange(n_genes)

    ii_template, jj_template = np.divmod(np.arange(n_entries), 64)
    dfs = []

    for row_i, cx_row in enumerate(complexes_df.iter_rows(named=True)):
        cid = cx_row["complex_id"]
        cname = cx_row["complex_name"]
        members = json.loads(cx_row["member_genes"])
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
            "complex_id": np.full(n_entries, cid, dtype=np.int64),
            "complex_name": [cname] * n_entries,
            "i": ii_template.astype(np.int16),
            "j": jj_template.astype(np.int16),
            "mean_in": mean_in.astype(np.float32),
            "mean_out": mean_out.astype(np.float32),
            "delta": delta.astype(np.float32),
            "effect_size": cohens_d.astype(np.float32),
            "p_value": p_vals.astype(np.float32),
            "fdr": fdr.astype(np.float32),
        })
        dfs.append(cx_df)

        if (row_i + 1) % 50 == 0 or row_i < 10:
            n_sig = int((fdr < FDR_THRESHOLD).sum())
            log.info(f"    [{row_i+1}/{complexes_df.height}] {cname[:40]}: "
                     f"{n_in} genes, {n_sig} sig entries (FDR<{FDR_THRESHOLD})")

    result = pl.concat(dfs)
    result.write_parquet(OUT_DIR / "corum_entry_enrichment.parquet")
    log.info(f"  Saved corum_entry_enrichment.parquet ({result.height:,} rows)")
    return result


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — Top entries + recurrence
# ══════════════════════════════════════════════════════════════════════

def stage4_top_and_recurrence(enrichment_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    log.info("STAGE 4: Top entries per complex + recurrence...")

    # Top N entries per complex
    top_entries = (
        enrichment_df
        .sort(["complex_id", "fdr", pl.col("effect_size").abs()], descending=[False, False, True])
        .group_by("complex_id")
        .head(N_TOP_ENTRIES)
    )
    top_entries.write_parquet(OUT_DIR / "corum_complex_top_entries.parquet")
    log.info(f"  Saved corum_complex_top_entries.parquet ({top_entries.height:,} rows)")

    # Recurrence: entries that appear significant across many complexes
    sig_entries = enrichment_df.filter(pl.col("fdr") < FDR_THRESHOLD)
    log.info(f"  Total significant entries (FDR<{FDR_THRESHOLD}): {sig_entries.height:,}")

    # Also use the top-N per complex for recurrence
    sig_top = top_entries.filter(pl.col("fdr") < FDR_THRESHOLD)

    recurrence_rows = []
    rec_dict: dict[tuple[int, int], list] = defaultdict(list)
    for row in sig_top.iter_rows(named=True):
        rec_dict[(row["i"], row["j"])].append({
            "complex_id": row["complex_id"],
            "complex_name": row["complex_name"],
            "effect_size": row["effect_size"],
        })

    for (i, j), entries in sorted(rec_dict.items()):
        recurrence_rows.append({
            "i": i,
            "j": j,
            "n_complexes": len(entries),
            "mean_effect": float(np.mean([e["effect_size"] for e in entries])),
            "complexes": json.dumps([e["complex_name"] for e in entries]),
        })

    rec_df = pl.DataFrame(recurrence_rows).sort("n_complexes", descending=True)
    rec_df.write_parquet(OUT_DIR / "corum_recurrent_entries.parquet")
    log.info(f"  Saved corum_recurrent_entries.parquet ({rec_df.height:,} entries)")

    if rec_df.height > 0:
        top5 = rec_df.head(5)
        for r in top5.iter_rows(named=True):
            log.info(f"    ({r['i']},{r['j']}): {r['n_complexes']} complexes, "
                     f"mean_d={r['mean_effect']:.3f}")

    # Recurrence heatmap
    heatmap = np.zeros((64, 64), dtype=int)
    for r in rec_df.iter_rows(named=True):
        heatmap[r["i"], r["j"]] = r["n_complexes"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="# complexes with entry in top-20 (FDR<0.05)")
    ax.set_xlabel("j (right feature)", fontsize=11)
    ax.set_ylabel("i (left feature)", fontsize=11)
    ax.set_title("Recurrent matrix entries across CORUM complexes", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_recurrent_entry_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_recurrent_entry_heatmap.png")

    return top_entries, rec_df


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — Delta heatmaps for selected complexes
# ══════════════════════════════════════════════════════════════════════

def stage5_heatmaps(
    enrichment_df: pl.DataFrame,
    complexes_df: pl.DataFrame,
    zscored: np.ndarray,
    gene_names: list[str],
) -> list[dict]:
    log.info("STAGE 5: Delta heatmaps for selected complexes...")
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Select complexes: by number of significant entries, then by size
    sig_counts = (
        enrichment_df
        .filter(pl.col("fdr") < FDR_THRESHOLD)
        .group_by("complex_id")
        .len()
        .rename({"len": "n_sig"})
        .sort("n_sig", descending=True)
    )

    # Deduplicate by complex name to avoid near-identical entries
    seen_names: set[str] = set()
    top_ids: list[int] = []
    for r in sig_counts.iter_rows(named=True):
        cid = r["complex_id"]
        cx = complexes_df.filter(pl.col("complex_id") == cid)
        if cx.height == 0:
            continue
        name = cx.row(0, named=True)["complex_name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        top_ids.append(cid)
        if len(top_ids) >= N_SELECTED_COMPLEXES:
            break
    if len(top_ids) < N_SELECTED_COMPLEXES:
        for cx_row in complexes_df.iter_rows(named=True):
            if cx_row["complex_id"] not in top_ids and cx_row["complex_name"] not in seen_names:
                top_ids.append(cx_row["complex_id"])
                seen_names.add(cx_row["complex_name"])
                if len(top_ids) >= N_SELECTED_COMPLEXES:
                    break

    selected = []
    for cid in top_ids:
        cx = complexes_df.filter(pl.col("complex_id") == cid)
        if cx.height == 0:
            continue
        cx_row = cx.row(0, named=True)
        members = json.loads(cx_row["member_genes"])
        in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])
        out_mask = np.ones(len(gene_names), dtype=bool)
        out_mask[in_idx] = False

        z_in = zscored[in_idx].reshape(-1, 64, 64)
        z_out = zscored[out_mask].reshape(-1, 64, 64)
        mean_in = z_in.mean(axis=0)
        mean_out = z_out.mean(axis=0)
        delta = mean_in - mean_out

        sname = sanitize_name(cx_row["complex_name"])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        vmax = max(abs(delta.min()), abs(delta.max()), 0.5)

        for ax, mat, title in [
            (axes[0], mean_in, f"Mean IN ({len(in_idx)} genes)"),
            (axes[1], mean_out, f"Mean OUT ({out_mask.sum()} genes)"),
            (axes[2], delta, "Delta (IN - OUT)"),
        ]:
            if "Delta" in title:
                im = ax.imshow(mat, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(mat, cmap="viridis", aspect="equal")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("j", fontsize=9)
            ax.set_ylabel("i", fontsize=9)

        fig.suptitle(f"{cx_row['complex_name']} (id={cid})", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_complex_{sname}_delta_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        selected.append({
            "complex_id": cid,
            "complex_name": cx_row["complex_name"],
            "n_genes": len(in_idx),
            "max_abs_delta": float(np.abs(delta).max()),
        })
        log.info(f"    {cx_row['complex_name']}: {len(in_idx)} genes, max|delta|={np.abs(delta).max():.3f}")

    log.info(f"  Saved {len(selected)} delta heatmaps")
    return selected


# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — Complex signature clustering
# ══════════════════════════════════════════════════════════════════════

def stage6_clustering(
    zscored: np.ndarray,
    gene_names: list[str],
    complexes_df: pl.DataFrame,
) -> None:
    log.info("STAGE 6: Complex signature clustering...")
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    signatures = []
    cx_labels = []
    cx_ids = []
    for cx_row in complexes_df.iter_rows(named=True):
        members = json.loads(cx_row["member_genes"])
        in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])
        if len(in_idx) < 3:
            continue
        sig = zscored[in_idx].mean(axis=0)  # (4096,)
        signatures.append(sig)
        cx_labels.append(cx_row["complex_name"][:35])
        cx_ids.append(cx_row["complex_id"])

    sig_matrix = np.array(signatures)  # (N_cx, 4096)
    log.info(f"  {sig_matrix.shape[0]} complex signatures")

    # Pairwise cosine similarity
    norms = np.linalg.norm(sig_matrix, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    sig_normed = sig_matrix / norms
    cos_sim = sig_normed @ sig_normed.T
    np.fill_diagonal(cos_sim, 1.0)

    # Save similarity
    sim_rows = []
    for i in range(len(cx_ids)):
        for j in range(i + 1, len(cx_ids)):
            sim_rows.append({
                "complex_id_a": cx_ids[i],
                "complex_id_b": cx_ids[j],
                "name_a": cx_labels[i],
                "name_b": cx_labels[j],
                "cosine_similarity": float(cos_sim[i, j]),
            })
    pl.DataFrame(sim_rows).write_parquet(OUT_DIR / "complex_signature_similarity.parquet")
    log.info("  Saved complex_signature_similarity.parquet")

    # Hierarchical clustering
    dist = 1 - cos_sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Plot
    n = len(cx_labels)
    fig_h = max(8, n * 0.06)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_h),
                             gridspec_kw={"width_ratios": [1, 4]})

    # Dendrogram
    ax_dend = axes[0]
    dend = dendrogram(Z, orientation="left", labels=cx_labels, ax=ax_dend,
                      leaf_font_size=4, no_labels=True)
    ax_dend.set_title("Dendrogram", fontsize=10)

    # Reorder similarity matrix
    order = dend["leaves"]
    cos_ordered = cos_sim[np.ix_(order, order)]

    ax_heat = axes[1]
    im = ax_heat.imshow(cos_ordered, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax_heat, shrink=0.5, label="Cosine similarity")
    ax_heat.set_title(f"Complex signature similarity ({n} complexes)", fontsize=11)
    ax_heat.set_xticks([])
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels([cx_labels[i] for i in order], fontsize=3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_complex_signature_clustering.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved fig_complex_signature_clustering.png")


# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — Broad class enrichment
# ══════════════════════════════════════════════════════════════════════

def stage7_class_enrichment(
    zscored: np.ndarray,
    gene_names: list[str],
    complexes_df: pl.DataFrame,
) -> pl.DataFrame | None:
    log.info("STAGE 7: Broad class enrichment...")
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Assign classes from fcg_category, then keyword fallback
    cx_to_class: dict[int, str] = {}
    for cx_row in complexes_df.iter_rows(named=True):
        cid = cx_row["complex_id"]
        fcg_cat = cx_row["fcg_category"]
        if fcg_cat:
            cx_to_class[cid] = fcg_cat
            continue
        name_lower = cx_row["complex_name"].lower()
        for cls_name, pattern in KEYWORD_CLASSES.items():
            if re.search(pattern, name_lower):
                cx_to_class[cid] = cls_name
                break

    # Count classes
    class_counts = Counter(cx_to_class.values())
    valid_classes = {c for c, n in class_counts.items() if n >= MIN_CLASS_COMPLEXES}
    log.info(f"  Classes with >= {MIN_CLASS_COMPLEXES} complexes: {len(valid_classes)}")
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        marker = "*" if cls in valid_classes else " "
        log.info(f"    {marker} {cls}: {n}")

    if not valid_classes:
        log.info("  No valid classes. Skipping class enrichment.")
        return None

    # For each valid class, pool member genes and run enrichment
    all_genes_set = set(gene_names)
    n_genes = len(gene_names)
    all_idx = np.arange(n_genes)
    all_rows = []

    for cls in sorted(valid_classes):
        cids = [cid for cid, c in cx_to_class.items() if c == cls]
        # Pool all member genes from these complexes
        class_genes: set[str] = set()
        for cx_row in complexes_df.filter(pl.col("complex_id").is_in(cids)).iter_rows(named=True):
            class_genes.update(json.loads(cx_row["member_genes"]))
        class_genes = class_genes & all_genes_set

        in_idx = np.array([gene_idx[g] for g in class_genes])
        out_mask = np.ones(n_genes, dtype=bool)
        out_mask[in_idx] = False
        n_in = len(in_idx)
        n_out = out_mask.sum()

        if n_in < 5 or n_out < 5:
            continue

        z_in = zscored[in_idx]
        z_out = zscored[all_idx[out_mask]]
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
        pooled_sd = np.sqrt(pooled_var)
        pooled_sd[pooled_sd < 1e-10] = 1e-10
        cohens_d = delta / pooled_sd

        fdr = benjamini_hochberg(p_vals)

        ii, jj = np.divmod(np.arange(4096), 64)
        for k in range(4096):
            all_rows.append({
                "class_name": cls,
                "n_genes_in": n_in,
                "n_complexes": len(cids),
                "i": int(ii[k]), "j": int(jj[k]),
                "mean_in": float(mean_in[k]),
                "mean_out": float(mean_out[k]),
                "delta": float(delta[k]),
                "effect_size": float(cohens_d[k]),
                "p_value": float(p_vals[k]),
                "fdr": float(fdr[k]),
            })

        n_sig = (fdr < FDR_THRESHOLD).sum()
        log.info(f"    {cls}: {n_in} genes, {len(cids)} complexes, {n_sig} sig entries")

    if not all_rows:
        return None

    cls_df = pl.DataFrame(all_rows)
    cls_df.write_parquet(OUT_DIR / "corum_class_entry_enrichment.parquet")
    log.info(f"  Saved corum_class_entry_enrichment.parquet ({cls_df.height:,} rows)")
    return cls_df


# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — Robustness checks
# ══════════════════════════════════════════════════════════════════════

def stage8_robustness(
    zscored: np.ndarray,
    gene_names: list[str],
    n_variants: np.ndarray,
    enrichment_df: pl.DataFrame,
    top_entries: pl.DataFrame,
    complexes_df: pl.DataFrame,
    selected_complexes: list[dict],
) -> dict:
    log.info("STAGE 8: Robustness checks...")
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    results = {}

    # 8A: Variant-count bias
    log.info("  8A: Variant-count bias check...")
    top_gene_set = set()
    for cx_row in complexes_df.iter_rows(named=True):
        members = json.loads(cx_row["member_genes"])
        top_gene_set.update(members)
    in_top = np.array([1 if g in top_gene_set else 0 for g in gene_names])
    rho, p = stats.spearmanr(n_variants, in_top)
    log.info(f"    Spearman(n_variants, in_any_complex) = {rho:.3f}, p={p:.3g}")
    results["variant_count_bias"] = {"spearman_r": float(rho), "p_value": float(p)}

    # 8B: Min-variant threshold sensitivity
    log.info("  8B: Threshold sensitivity (>=2 vs >=3)...")
    # Would need to rebuild matrices with >=2. Instead, report how many genes
    # we'd gain and estimate impact.
    all_gene_counts = Counter()
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT gene_name FROM variants WHERE gene_name IS NOT NULL").fetchall()
    con.close()
    for (g,) in rows:
        all_gene_counts[g.upper()] += 1
    n_ge2 = sum(1 for c in all_gene_counts.values() if c >= 2)
    n_ge3 = sum(1 for c in all_gene_counts.values() if c >= 3)
    log.info(f"    Genes >=2: {n_ge2}, >=3: {n_ge3}, diff: {n_ge2 - n_ge3}")
    results["threshold_sensitivity"] = {"n_ge2": n_ge2, "n_ge3": n_ge3, "diff": n_ge2 - n_ge3}

    # 8C: Split-half stability for selected complexes
    log.info("  8C: Split-half stability...")
    rng = np.random.default_rng(RANDOM_SEED)
    split_results = []

    for cx_info in selected_complexes:
        cid = cx_info["complex_id"]
        cx = complexes_df.filter(pl.col("complex_id") == cid)
        if cx.height == 0:
            continue
        members = json.loads(cx.row(0, named=True)["member_genes"])
        in_idx = np.array([gene_idx[g] for g in members if g in gene_idx])

        if len(in_idx) < 6:
            continue

        correlations = []
        for _ in range(N_SPLIT_HALF_ITERS):
            perm = rng.permutation(len(in_idx))
            half = len(in_idx) // 2
            a_idx = in_idx[perm[:half]]
            b_idx = in_idx[perm[half:]]

            delta_a = zscored[a_idx].mean(axis=0) - zscored.mean(axis=0)
            delta_b = zscored[b_idx].mean(axis=0) - zscored.mean(axis=0)
            r, _ = stats.spearmanr(delta_a, delta_b)
            correlations.append(r)

        mean_r = float(np.mean(correlations))
        split_results.append({
            "complex_id": cid,
            "complex_name": cx_info["complex_name"],
            "n_genes": len(in_idx),
            "mean_split_half_r": mean_r,
            "std_split_half_r": float(np.std(correlations)),
        })
        log.info(f"    {cx_info['complex_name'][:35]}: n={len(in_idx)}, "
                 f"mean_r={mean_r:.3f} +/- {np.std(correlations):.3f}")

    results["split_half"] = split_results
    return results


# ══════════════════════════════════════════════════════════════════════
# STAGE 9-10 — Report
# ══════════════════════════════════════════════════════════════════════

def write_report(
    gene_names: list[str],
    n_variants: np.ndarray,
    complexes_df: pl.DataFrame,
    enrichment_df: pl.DataFrame,
    top_entries: pl.DataFrame,
    rec_df: pl.DataFrame,
    selected_complexes: list[dict],
    robustness: dict,
    class_df: pl.DataFrame | None,
) -> None:
    log.info("STAGE 9-10: Writing report...")

    # Compute summary stats
    sig_per_cx = (
        enrichment_df.filter(pl.col("fdr") < FDR_THRESHOLD)
        .group_by("complex_id").len().rename({"len": "n_sig"})
    )
    cx_with_sig = sig_per_cx.filter(pl.col("n_sig") > 0).height
    total_sig = enrichment_df.filter(pl.col("fdr") < FDR_THRESHOLD).height
    n_recurrent = rec_df.filter(pl.col("n_complexes") >= 3).height if rec_df.height > 0 else 0

    lines = [
        "# Feature-Level CORUM Interpretability — Report",
        "",
        "## 1. Gene-Level Setup",
        "",
        f"- **Genes retained:** {len(gene_names):,} (>= {MIN_VARIANTS_PER_GENE} variants each)",
        f"- **Variants/gene:** median={int(np.median(n_variants))}, mean={n_variants.mean():.1f}, max={n_variants.max()}",
        f"- **Matrix representation:** 64x64 diff covariance, averaged across variants per gene",
        f"- **Z-scoring:** per entry (i,j), standardized across all genes before testing",
        "",
        "## 2. CORUM Complexes",
        "",
        f"- **Retained complexes:** {complexes_df.height} (>= {MIN_EMBEDDED_GENES_PER_COMPLEX} embedded genes)",
        f"- **Embedded genes/complex:** median={int(np.median(complexes_df['n_genes_embedded'].to_numpy()))}, "
        f"max={complexes_df['n_genes_embedded'].max()}",
        "",
        "## 3. Entry-Level Results",
        "",
        f"- **Total tests:** {enrichment_df.height:,} (4096 entries x {complexes_df.height} complexes)",
        f"- **Significant entries (FDR < {FDR_THRESHOLD}):** {total_sig:,}",
        f"- **Complexes with >= 1 significant entry:** {cx_with_sig} / {complexes_df.height}",
        "",
        "Some (i,j) entries are systematically elevated or depleted in specific complexes, "
        "indicating that particular latent feature interactions are characteristic of those complexes.",
        "",
        "## 4. Recurrent Entries",
        "",
        f"- **Entries appearing in top-{N_TOP_ENTRIES} of >= 3 complexes:** {n_recurrent}",
        "",
    ]

    if rec_df.height > 0:
        lines.append("**Top 10 most recurrent entries:**")
        lines.append("")
        lines.append("| (i,j) | # Complexes | Mean effect |")
        lines.append("|-------|------------|------------|")
        for r in rec_df.head(10).iter_rows(named=True):
            lines.append(f"| ({r['i']},{r['j']}) | {r['n_complexes']} | {r['mean_effect']:.3f} |")
        lines.append("")
        lines.append("These recurrent entries suggest shared latent interaction motifs for complex biology.")

    lines += [
        "",
        "## 5. Complex Specificity",
        "",
        "Selected complexes with strongest signatures:",
        "",
        "| Complex | Genes | Max |delta| |",
        "|---------|-------|------------|",
    ]
    for sc in selected_complexes:
        lines.append(f"| {sc['complex_name']} | {sc['n_genes']} | {sc['max_abs_delta']:.3f} |")

    lines += [
        "",
        "Some entries are unique to particular complexes, suggesting specialized interaction signatures. "
        "See `fig_complex_*_delta_heatmap.png` for visual examples.",
        "",
        "## 6. Robustness",
        "",
        f"- **Variant-count bias:** Spearman r(n_variants, in_complex) = "
        f"{robustness['variant_count_bias']['spearman_r']:.3f} "
        f"(p={robustness['variant_count_bias']['p_value']:.3g})",
        f"- **Threshold sensitivity:** {robustness['threshold_sensitivity']['n_ge3']} genes at >=3 vs "
        f"{robustness['threshold_sensitivity']['n_ge2']} at >=2 (diff = {robustness['threshold_sensitivity']['diff']})",
    ]

    if robustness.get("split_half"):
        lines.append("- **Split-half stability:**")
        lines.append("")
        lines.append("| Complex | Genes | Mean Spearman r |")
        lines.append("|---------|-------|----------------|")
        for sh in robustness["split_half"]:
            lines.append(f"| {sh['complex_name'][:35]} | {sh['n_genes']} | {sh['mean_split_half_r']:.3f} |")

    if class_df is not None:
        cls_sig = (
            class_df.filter(pl.col("fdr") < FDR_THRESHOLD)
            .group_by("class_name").len().rename({"len": "n_sig"})
            .sort("n_sig", descending=True)
        )
        lines += [
            "",
            "## 7. Broad Class Enrichment",
            "",
            "| Class | Sig entries |",
            "|-------|-----------|",
        ]
        for r in cls_sig.iter_rows(named=True):
            lines.append(f"| {r['class_name']} | {r['n_sig']:,} |")

    lines += [
        "",
        "## Interpretation Caveat",
        "",
        "Latent features (i,j) are not yet semantically labeled. Entries should be interpreted as "
        "\"latent interaction motifs\" — systematic patterns in the model's learned covariance structure "
        "that distinguish specific complexes from the background. They do not directly correspond to "
        "known biochemical mechanisms without further analysis (e.g., probing individual features against "
        "known annotations).",
        "",
        "## Output Files",
        "",
        "- `gene_level_matrices.npz` — gene-level averaged 64x64 matrices + z-scored",
        "- `corum_complex_gene_sets.parquet` — retained CORUM complexes with member genes",
        "- `corum_entry_enrichment.parquet` — per-complex, per-entry enrichment results",
        "- `corum_complex_top_entries.parquet` — top-20 entries per complex",
        "- `corum_recurrent_entries.parquet` — globally recurrent entries",
        "- `complex_signature_similarity.parquet` — pairwise complex similarity",
        "- `corum_class_entry_enrichment.parquet` — broad class enrichment",
        "- `fig_recurrent_entry_heatmap.png`",
        "- `fig_complex_signature_clustering.png`",
        "- `fig_complex_*_delta_heatmap.png` — per-complex delta heatmaps",
    ]

    (OUT_DIR / "corum_interpretability_report.md").write_text("\n".join(lines) + "\n")
    log.info("  Saved corum_interpretability_report.md")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    enforce_seeds(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 (cache-aware)
    npz_path = OUT_DIR / "gene_level_matrices.npz"
    if npz_path.exists():
        log.info("STAGE 1: Loading cached gene_level_matrices.npz...")
        data = np.load(str(npz_path), allow_pickle=True)
        gene_names = list(data["gene_names"])
        matrices = data["matrices"]
        n_variants = data["n_variants"]
        zscored = data["zscored_flat"]
        log.info(f"  {len(gene_names):,} genes loaded from cache")
    else:
        matrices, zscored, gene_names, n_variants = stage1_gene_matrices()

    # Stage 2 (cache-aware)
    gs_path = OUT_DIR / "corum_complex_gene_sets.parquet"
    if gs_path.exists():
        log.info("STAGE 2: Loading cached corum_complex_gene_sets.parquet...")
        complexes_df = pl.read_parquet(gs_path)
        log.info(f"  {complexes_df.height} complexes loaded from cache")
    else:
        complexes_df = stage2_corum_gene_sets(gene_names)

    # Stage 3 (cache-aware)
    enrich_path = OUT_DIR / "corum_entry_enrichment.parquet"
    if enrich_path.exists():
        log.info("STAGE 3: Loading cached corum_entry_enrichment.parquet...")
        enrichment_df = pl.read_parquet(enrich_path)
        log.info(f"  {enrichment_df.height:,} rows loaded from cache")
    else:
        enrichment_df = stage3_entry_enrichment(zscored, gene_names, complexes_df)

    # Stage 4
    top_entries, rec_df = stage4_top_and_recurrence(enrichment_df)

    # Stage 5
    selected_complexes = stage5_heatmaps(enrichment_df, complexes_df, zscored, gene_names)

    # Stage 6
    stage6_clustering(zscored, gene_names, complexes_df)

    # Stage 7
    class_df = stage7_class_enrichment(zscored, gene_names, complexes_df)

    # Stage 8
    robustness = stage8_robustness(
        zscored, gene_names, n_variants, enrichment_df,
        top_entries, complexes_df, selected_complexes,
    )

    # Stage 9-10
    write_report(gene_names, n_variants, complexes_df, enrichment_df,
                 top_entries, rec_df, selected_complexes, robustness, class_df)

    # Run config
    config = {
        "analysis": "corum_feature_interpretability",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "random_seed": RANDOM_SEED,
        "min_variants_per_gene": MIN_VARIANTS_PER_GENE,
        "min_embedded_genes_per_complex": MIN_EMBEDDED_GENES_PER_COMPLEX,
        "n_top_entries": N_TOP_ENTRIES,
        "fdr_threshold": FDR_THRESHOLD,
        "n_selected_complexes": N_SELECTED_COMPLEXES,
        "n_split_half_iters": N_SPLIT_HALF_ITERS,
        "n_genes": len(gene_names),
        "n_complexes": complexes_df.height,
    }
    with open(OUT_DIR / "corum_interpretability_run_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed:.0f}s")
    print(f"\n{'='*80}\nCORUM FEATURE INTERPRETABILITY — COMPLETE ({elapsed:.0f}s)\n{'='*80}")


if __name__ == "__main__":
    main()
