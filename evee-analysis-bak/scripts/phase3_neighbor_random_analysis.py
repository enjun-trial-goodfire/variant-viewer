#!/usr/bin/env python3
"""Phase 3: neighbor vs matched random controls in interpretable feature space.

Reads Phase 2 exports under ``eeve-analysis/data/intermediate/`` (see ``export_analysis_datasets.py``).

Run from variant-viewer repo root::

  uv run python eeve-analysis/scripts/phase3_neighbor_random_analysis.py

Options::

  --seed 0
  --random-multiplier 1.0
  --match-target-path-bin
  --write-z-deltas-long
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent.parent
for p in (_REPO_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from export_analysis_datasets import _coding_mask_polars  # noqa: E402
from phase3_feature_groups import discover_feature_groups, summary_group_order  # noqa: E402

_INTERMEDIATE = _SCRIPTS.parent / "data" / "intermediate"
_OUTPUT_FIGURES = _SCRIPTS.parent / "outputs" / "figures"


def _cons_bucket_expr() -> pl.Expr:
    return (
        pl.coalesce(
            pl.col("consequence_display").cast(pl.Utf8),
            pl.col("consequence").cast(pl.Utf8),
        )
        .str.strip_chars()
        .str.to_lowercase()
        .fill_null("unknown")
    )


def _path_bin_expr(col: str = "pathogenicity") -> pl.Expr:
    p = pl.col(col)
    return (
        pl.when(p.is_null() | p.is_nan())
        .then(pl.lit("p_nan"))
        .when(p < 0.25)
        .then(pl.lit("p_lo"))
        .when(p < 0.5)
        .then(pl.lit("p_midlo"))
        .when(p < 0.75)
        .then(pl.lit("p_midhi"))
        .otherwise(pl.lit("p_hi"))
    )


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity; NaN if either vector norm is 0 or <2 finite dims."""
    m = np.isfinite(a) & np.isfinite(b)
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    for i in range(a.shape[0]):
        mm = m[i]
        if mm.sum() < 2:
            continue
        va, vb = a[i, mm], b[i, mm]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            continue
        out[i] = float(np.dot(va, vb) / (na * nb))
    return out


def _group_metrics(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row mean |diff|, median |diff|, L2(|diff|), cosine(src, tgt)."""
    d = np.abs(src - tgt)
    mad = np.nanmean(d, axis=1)
    med = np.nanmedian(d, axis=1)
    l2 = np.sqrt(np.nansum(d**2, axis=1))
    cos = _cosine_rows(src, tgt)
    return mad, med, l2, cos


def _feature_block_matrix(
    subset: pl.DataFrame,
    ordered_ids: list[str],
    cols: list[str],
) -> np.ndarray:
    if not cols:
        return np.zeros((len(ordered_ids), 0), dtype=np.float64)
    use = ["variant_id"] + [c for c in cols if c in subset.columns]
    if len(use) < 2:
        return np.full((len(ordered_ids), len(cols)), np.nan, dtype=np.float64)
    sub = subset.select(use).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in use[1:]]
    )
    ids_df = pl.DataFrame({"variant_id": ordered_ids}).with_columns(pl.col("variant_id").cast(pl.Utf8))
    j = ids_df.join(sub, on="variant_id", how="left")
    return j.select(use[1:]).to_numpy()


def _compute_pair_table(
    pair_label: str,
    src_ids: list[str],
    tgt_ids: list[str],
    pair_keys: list[str],
    neighbor_rank: list[int | None],
    meta_src: dict[str, dict],
    meta_tgt: dict[str, dict],
    id2idx: dict[str, int],
    ordered_ids: list[str],
    groups: dict[str, list[str]],
    subset: pl.DataFrame,
    group_order: list[str],
) -> pl.DataFrame:
    si = np.array([id2idx[s] for s in src_ids], dtype=np.int64)
    ti = np.array([id2idx[t] for t in tgt_ids], dtype=np.int64)

    rows: dict[str, object] = {
        "pair_id": pair_keys,
        "pair_label": [pair_label] * len(src_ids),
        "source_variant_id": src_ids,
        "target_variant_id": tgt_ids,
        "neighbor_rank": neighbor_rank,
    }

    sg = [meta_src[s]["gene_name"] for s in src_ids]
    tg = [meta_tgt[t]["gene_name"] for t in tgt_ids]
    sc = [meta_src[s]["cons_bucket"] for s in src_ids]
    tc = [meta_tgt[t]["cons_bucket"] for t in tgt_ids]
    rows["source_gene_name"] = sg
    rows["target_gene_name"] = tg
    rows["source_cons_bucket"] = sc
    rows["target_cons_bucket"] = tc
    rows["same_gene"] = [a == b and a is not None and a != "" for a, b in zip(sg, tg, strict=True)]
    rows["same_cons_bucket"] = [a == b for a, b in zip(sc, tc, strict=True)]

    for gname in group_order:
        cols = groups.get(gname, [])
        if not cols:
            continue
        F = _feature_block_matrix(subset, ordered_ids, cols)
        if F.size == 0 or F.shape[1] == 0:
            continue
        Fs, Ft = F[si], F[ti]
        mad, med, l2, cos = _group_metrics(Fs, Ft)
        rows[f"mad__{gname}"] = mad
        rows[f"med_ad__{gname}"] = med
        rows[f"l2__{gname}"] = l2
        rows[f"cos__{gname}"] = cos

    return pl.DataFrame(rows)


def _sample_random_controls(
    neighbor_src: list[str],
    neighbor_tgt: list[str],
    id2bucket: dict[str, str],
    bucket_arrays: dict[str, np.ndarray],
    all_ids_array: np.ndarray,
    nb_map: dict[str, set[str]],
    stratum_path_bin: list[str] | None,
    id2pathbin: dict[str, str],
    rng: np.random.Generator,
    multiplier: float,
) -> tuple[list[str], list[str], list[int | None], list[str], list[str]]:
    """Draw ``max(1, round(multiplier))`` random control targets per neighbor row.

    Each control target is uniform among subset variants in the **same consequence bucket
    as the neighbor** (fallback: any variant). Excludes self and true neighbors of the
    source. Optional path-bin filter narrows candidates to the neighbor's pathogenicity bin.
    """
    n = len(neighbor_src)
    k_per_row = max(1, int(round(multiplier)))

    out_s: list[str] = []
    out_t: list[str] = []
    out_rnk: list[int | None] = []
    out_key: list[str] = []
    out_anchor_nbr: list[str] = []
    key_i = 0

    for i in range(n):
        s, nbr = neighbor_src[i], neighbor_tgt[i]
        B = id2bucket.get(nbr, "unknown")
        arr = bucket_arrays.get(B)
        if arr is None or len(arr) == 0:
            arr = all_ids_array

        nb_s = nb_map.get(s, set())
        nb_list = np.array(list(nb_s), dtype=object) if nb_s else np.array([], dtype=object)

        for _rep in range(k_per_row):
            mask = arr.astype(object) != s
            if nb_list.size:
                mask &= ~np.isin(arr.astype(object), nb_list)
            cand = arr[mask]
            if stratum_path_bin is not None:
                pb_n = stratum_path_bin[i]
                if len(cand) > 0:
                    pb_ok = np.array([id2pathbin.get(str(x), "p_nan") == pb_n for x in cand], dtype=bool)
                    if pb_ok.any():
                        cand = cand[pb_ok]
            if len(cand) == 0:
                mask2 = all_ids_array.astype(object) != s
                if nb_list.size:
                    mask2 &= ~np.isin(all_ids_array.astype(object), nb_list)
                cand = all_ids_array[mask2]
            if len(cand) == 0:
                continue
            r = str(rng.choice(cand))
            out_s.append(s)
            out_t.append(r)
            out_rnk.append(None)
            out_key.append(f"rnd::{key_i}::{s}::{r}")
            out_anchor_nbr.append(nbr)
            key_i += 1

    return out_s, out_t, out_rnk, out_key, out_anchor_nbr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide", type=Path, default=_INTERMEDIATE / "variants_wide.parquet")
    ap.add_argument(
        "--neighbor-joined",
        type=Path,
        default=_INTERMEDIATE / "neighbor_pairs_joined_coding_sources.parquet",
        help="Joined neighbor pairs (coding sources); falls back to long pairs + join",
    )
    ap.add_argument(
        "--neighbor-pairs-fallback",
        type=Path,
        default=_INTERMEDIATE / "neighbor_pairs_coding_sources.parquet",
    )
    ap.add_argument("--out-intermediate", type=Path, default=_INTERMEDIATE)
    ap.add_argument("--out-figures", type=Path, default=_OUTPUT_FIGURES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--random-multiplier",
        type=float,
        default=1.0,
        help="Independent random controls drawn **per neighbor row** (rounded); 1 ≈ 1:1 size",
    )
    ap.add_argument(
        "--match-target-path-bin",
        action="store_true",
        help="Also match control target pathogenicity bin to neighbor's bin (stricter)",
    )
    ap.add_argument(
        "--write-z-deltas-long",
        action="store_true",
        help="Write long-format z_* abs/sq deltas (can be large)",
    )
    args = ap.parse_args()

    out_i = args.out_intermediate.resolve()
    out_f = args.out_figures.resolve()
    out_i.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    if not args.wide.is_file():
        logging.error("Missing wide table: %s", args.wide)
        raise SystemExit(1)

    logging.info("Loading %s", args.wide)
    wide = pl.read_parquet(args.wide)

    subset = wide.filter(_coding_mask_polars()).with_columns(
        cons_bucket=_cons_bucket_expr(),
        path_bin=_path_bin_expr("pathogenicity"),
    )
    subset_path = out_i / "variants_analysis_subset.parquet"
    logging.info("Analysis subset: %s rows -> %s", subset.height, subset_path)
    subset.write_parquet(subset_path)

    subset_ids = set(subset["variant_id"].cast(pl.Utf8).to_list())
    ordered_ids = sorted(subset_ids)
    id2idx = {v: i for i, v in enumerate(ordered_ids)}

    meta_cols = ["variant_id", "gene_name", "cons_bucket", "path_bin", "pathogenicity"]
    meta_df = subset.select([c for c in meta_cols if c in subset.columns])
    meta_rows = meta_df.to_dicts()
    meta_by_id: dict[str, dict] = {}
    for row in meta_rows:
        vid = str(row["variant_id"])
        meta_by_id[vid] = {
            "gene_name": row.get("gene_name"),
            "cons_bucket": row.get("cons_bucket") or "unknown",
            "path_bin": row.get("path_bin") or "p_nan",
        }

    if args.neighbor_joined.is_file():
        nbr = pl.read_parquet(args.neighbor_joined)
        logging.info("Neighbor joined: %s rows", nbr.height)
    else:
        logging.warning("Missing %s — using %s + subset joins", args.neighbor_joined, args.neighbor_pairs_fallback)
        long_p = pl.read_parquet(args.neighbor_pairs_fallback)
        src_meta = subset.select(["variant_id", "gene_name", "cons_bucket", "path_bin"]).rename(
            {
                "variant_id": "source_variant_id",
                "gene_name": "source_gene_name",
                "cons_bucket": "source_cons_bucket",
                "path_bin": "source_path_bin",
            }
        )
        nb_meta = subset.select(
            ["variant_id", "gene_name", "cons_bucket", "path_bin", "pathogenicity"]
        ).rename(
            {
                "variant_id": "neighbor_variant_id",
                "gene_name": "neighbor_gene_name",
                "cons_bucket": "neighbor_cons_bucket",
                "path_bin": "neighbor_path_bin",
                "pathogenicity": "neighbor_pathogenicity",
            }
        )
        nbr = long_p.join(src_meta, on="source_variant_id", how="left").join(
            nb_meta, on="neighbor_variant_id", how="left"
        )

    nbr = nbr.filter(
        pl.col("source_variant_id").cast(pl.Utf8).is_in(list(subset_ids))
        & pl.col("neighbor_variant_id").cast(pl.Utf8).is_in(list(subset_ids))
        & (pl.col("source_variant_id") != pl.col("neighbor_variant_id"))
    )

    if nbr.height == 0:
        logging.error("No neighbor pairs with both ends in analysis subset.")
        raise SystemExit(1)

    src_n = nbr["source_variant_id"].cast(pl.Utf8).to_list()
    tgt_n = nbr["neighbor_variant_id"].cast(pl.Utf8).to_list()
    if "neighbor_rank" in nbr.columns:
        raw_r = nbr["neighbor_rank"].to_list()
        rnk_n = [int(raw_r[i]) if raw_r[i] is not None else i + 1 for i in range(len(raw_r))]
    else:
        rnk_n = list(range(1, len(src_n) + 1))
    pair_keys_n = [f"nbr::{s}::{t}::r{r}" for s, t, r in zip(src_n, tgt_n, rnk_n, strict=True)]

    nb_map: dict[str, set[str]] = defaultdict(set)
    for s, t in zip(src_n, tgt_n, strict=True):
        nb_map[s].add(t)

    id2bucket = {str(r["variant_id"]): str(r["cons_bucket"]) for r in subset.select(["variant_id", "cons_bucket"]).to_dicts()}
    id2pathbin = {str(r["variant_id"]): str(r["path_bin"]) for r in subset.select(["variant_id", "path_bin"]).to_dicts()}

    gbuck = (
        subset.group_by("cons_bucket")
        .agg(pl.col("variant_id").cast(pl.Utf8))
        .sort("cons_bucket")
    )
    bucket_arrays: dict[str, np.ndarray] = {}
    for row in gbuck.iter_rows(named=True):
        bucket_arrays[str(row["cons_bucket"])] = np.array(row["variant_id"], dtype=object)
    all_ids_array = np.array(ordered_ids, dtype=object)

    rng = np.random.default_rng(args.seed)
    tgt_pb: list[str] | None = None
    if args.match_target_path_bin:
        tgt_pb = [id2pathbin.get(t, "p_nan") for t in tgt_n]

    r_src, r_tgt, r_rnk, r_keys, r_anchor_nbr = _sample_random_controls(
        src_n,
        tgt_n,
        id2bucket,
        bucket_arrays,
        all_ids_array,
        nb_map,
        tgt_pb,
        id2pathbin,
        rng,
        args.random_multiplier,
    )

    matched_path = out_i / "random_pairs_matched.parquet"
    pl.DataFrame(
        {
            "pair_id": r_keys,
            "source_variant_id": r_src,
            "target_variant_id": r_tgt,
            "anchor_neighbor_variant_id": r_anchor_nbr,
            "anchor_neighbor_cons_bucket": [id2bucket.get(n, "unknown") for n in r_anchor_nbr],
            "source_cons_bucket": [id2bucket.get(s, "unknown") for s in r_src],
            "target_cons_bucket": [id2bucket.get(t, "unknown") for t in r_tgt],
        }
    ).write_parquet(matched_path)
    logging.info("Wrote %s (%s rows)", matched_path, len(r_src))

    all_cols = subset.columns
    groups = discover_feature_groups(all_cols)
    group_order = summary_group_order(groups)
    logging.info("Feature groups with ≥1 column: %s", len(groups))

    meta_src = meta_by_id
    meta_tgt = meta_by_id

    tbl_n = _compute_pair_table(
        "neighbor",
        src_n,
        tgt_n,
        pair_keys_n,
        [int(x) for x in rnk_n],
        meta_src,
        meta_tgt,
        id2idx,
        ordered_ids,
        groups,
        subset,
        group_order,
    )
    tbl_r = _compute_pair_table(
        "random",
        r_src,
        r_tgt,
        r_keys,
        r_rnk,
        meta_src,
        meta_tgt,
        id2idx,
        ordered_ids,
        groups,
        subset,
        group_order,
    )

    tbl_n = tbl_n.with_columns(pl.col("neighbor_rank").cast(pl.Int64))
    tbl_r = tbl_r.with_columns(pl.col("neighbor_rank").cast(pl.Int64))
    summary = pl.concat([tbl_n, tbl_r], how="vertical_relaxed")
    summary_path = out_i / "pair_distance_summary.parquet"
    logging.info("Writing %s (%s rows)", summary_path, summary.height)
    summary.write_parquet(summary_path)

    n_path = out_i / "neighbor_pairs_features.parquet"
    r_path = out_i / "random_pairs_features.parquet"
    tbl_n.write_parquet(n_path)
    tbl_r.write_parquet(r_path)
    logging.info("Wrote %s and %s", n_path, r_path)

    # --- Validation counts & JSON ---
    n_same_gene = int(tbl_n["same_gene"].sum())
    n_cross_gene = int(tbl_n.height - n_same_gene)
    r_same_gene = int(tbl_r["same_gene"].sum())
    r_cross_gene = int(tbl_r.height - r_same_gene)
    counts = {
        "neighbor_pairs": int(tbl_n.height),
        "random_pairs": int(tbl_r.height),
        "random_controls_per_neighbor_row": max(1, int(round(args.random_multiplier))),
        "random_multiplier_arg": args.random_multiplier,
        "analysis_subset_variants": int(subset.height),
        "neighbor_same_gene": n_same_gene,
        "neighbor_cross_gene": n_cross_gene,
        "random_same_gene": r_same_gene,
        "random_cross_gene": r_cross_gene,
        "seed": args.seed,
        "match_target_path_bin": bool(args.match_target_path_bin),
    }
    if "l2__disruption_z_all" in summary.columns:
        zn = tbl_n["l2__disruption_z_all"].drop_nulls()
        zr = tbl_r["l2__disruption_z_all"].drop_nulls()
        counts["mean_l2_disruption_z_neighbors"] = float(zn.mean()) if zn.len() > 0 else None
        counts["mean_l2_disruption_z_random"] = float(zr.mean()) if zr.len() > 0 else None
        for key, g in [
            ("structure", "l2__disruption_z_structure_context"),
            ("domain", "l2__disruption_z_interpro_domain"),
            ("splice", "l2__disruption_z_splice"),
        ]:
            if g in summary.columns:
                counts[f"mean_{key}_l2_neighbors"] = float(tbl_n[g].drop_nulls().mean()) if tbl_n[g].drop_nulls().len() > 0 else None
                counts[f"mean_{key}_l2_random"] = float(tbl_r[g].drop_nulls().mean()) if tbl_r[g].drop_nulls().len() > 0 else None

    counts_path = out_i / "phase3_validation_counts.json"
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    logging.info("Wrote %s", counts_path)

    # --- Comparison table (aggregate) ---
    agg_rows = []
    metric_cols = [c for c in summary.columns if c.startswith(("mad__", "l2__", "med_ad__", "cos__"))]
    for col in metric_cols:
        stat, gname = col.split("__", 1)
        subn = tbl_n.select(col).drop_nulls()
        subr = tbl_r.select(col).drop_nulls()
        agg_rows.append(
            {
                "metric": stat,
                "group": gname,
                "neighbor_mean": float(subn[col].mean()) if subn.height > 0 else None,
                "random_mean": float(subr[col].mean()) if subr.height > 0 else None,
                "neighbor_median": float(subn[col].median()) if subn.height > 0 else None,
                "random_median": float(subr[col].median()) if subr.height > 0 else None,
            }
        )
    if agg_rows:
        agg_df = pl.DataFrame(agg_rows)
        agg_path = out_i / "phase3_group_metric_comparison.parquet"
        agg_df.write_parquet(agg_path)
        logging.info("Wrote %s", agg_path)

    seg_cols = ["pair_label", "same_gene", "same_cons_bucket"]
    if all(c in summary.columns for c in seg_cols) and "l2__disruption_z_all" in summary.columns:
        seg_exprs = [
            pl.col("l2__disruption_z_all").mean().alias("mean_l2__disruption_z_all"),
        ]
        for extra in (
            "l2__disruption_z_structure_context",
            "l2__disruption_z_interpro_domain",
            "l2__disruption_z_splice",
        ):
            if extra in summary.columns:
                seg_exprs.append(pl.col(extra).mean().alias(f"mean_{extra}"))
        seg_df = summary.group_by(seg_cols).agg(seg_exprs)
        seg_path = out_i / "phase3_means_by_pairtype_gene_cons.parquet"
        seg_df.write_parquet(seg_path)
        logging.info("Wrote %s", seg_path)

    # --- Sanity examples ---
    examples = []
    if "l2__disruption_z_all" in tbl_n.columns:
        thresh = float(tbl_n["l2__disruption_z_all"].drop_nulls().quantile(0.01))
        low = tbl_n.filter(pl.col("l2__disruption_z_all") <= thresh).head(15)
        examples.append(low.with_columns(pl.lit("low_neighbor_z_l2").alias("sanity_tag")))
    if "l2__disruption_z_interpro_domain" in tbl_n.columns:
        cg = tbl_n.filter(~pl.col("same_gene")).sort("l2__disruption_z_interpro_domain").head(15)
        examples.append(cg.with_columns(pl.lit("cross_gene_low_interpro_l2").alias("sanity_tag")))
    med_n = float(tbl_n["l2__disruption_z_all"].drop_nulls().median()) if "l2__disruption_z_all" in tbl_n.columns else None
    if med_n is not None and "l2__disruption_z_all" in tbl_r.columns:
        weird = tbl_r.filter(pl.col("l2__disruption_z_all") < med_n).head(15)
        examples.append(weird.with_columns(pl.lit("random_below_neighbor_median_z_l2").alias("sanity_tag")))
    if examples:
        ex = pl.concat(examples, how="vertical_relaxed")
        ex_path = out_i / "phase3_sanity_examples.parquet"
        ex.write_parquet(ex_path)
        logging.info("Wrote %s", ex_path)

    # --- Optional long z deltas ---
    if args.write_z_deltas_long and groups.get("disruption_z_all"):
        zcols = groups["disruption_z_all"]
        if zcols:
            F = _feature_block_matrix(subset, ordered_ids, zcols)
            long_rows: list[dict] = []

            def add_pairs(label: str, si_l: list[str], ti_l: list[str], pid: list[str]):
                for s, t, p in zip(si_l, ti_l, pid, strict=True):
                    i, j = id2idx[s], id2idx[t]
                    for k, cname in enumerate(zcols):
                        a, b = F[i, k], F[j, k]
                        if not (np.isfinite(a) and np.isfinite(b)):
                            continue
                        long_rows.append(
                            {
                                "pair_id": p,
                                "pair_label": label,
                                "feature": cname,
                                "abs_diff": float(abs(a - b)),
                                "sq_diff": float((a - b) ** 2),
                            }
                        )

            add_pairs("neighbor", src_n, tgt_n, pair_keys_n)
            add_pairs("random", r_src, r_tgt, r_keys)
            if long_rows:
                pl.DataFrame(long_rows).write_parquet(out_i / "pair_z_deltas_long.parquet")
                logging.info("Wrote pair_z_deltas_long.parquet (%s rows)", len(long_rows))

    # --- Simple figures ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skip figures")
        return

    if "l2__disruption_z_all" in tbl_n.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        nn = tbl_n["l2__disruption_z_all"].drop_nulls().to_numpy()
        rr = tbl_r["l2__disruption_z_all"].drop_nulls().to_numpy()
        ax.hist(nn, bins=60, alpha=0.55, label="neighbor", density=True)
        ax.hist(rr, bins=60, alpha=0.55, label="random", density=True)
        ax.set_xlabel("L2 |Δz| (all z_*)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title("Phase 3: disruption profile distance — neighbor vs random")
        fig.tight_layout()
        fp = out_f / "phase3_hist_l2_disruption_z_all.png"
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        logging.info("Wrote %s", fp)

    logging.info("Phase 3 done.")


if __name__ == "__main__":
    main()
