#!/usr/bin/env python3
"""Export wide variant + neighbor tables from variants.duckdb using repo-standard access.

Reuses:
  * ``db.open_db`` (variant-viewer/db.py) — same connection pattern as the app / build.
  * Flat ``SELECT * FROM variants`` — same row shape as ``serve.variant_endpoint``.
  * Neighbor parsing — same rule as ``serve._flat_to_prompt_dict`` / frontend ``JSON_FIELDS``:
    ``json.loads(neighbors)`` when the column is a JSON string.
  * Neighbor JSON keys match ``build.py`` ``compute_neighbors`` struct:
    ``id``, ``gene``, ``consequence_display``, ``label``, ``label_display``, ``score``, ``similarity``.

New logic (not in app — bulk export only):
  * Polars filters mirroring ``eeve-analysis/sql/phase2_neighbor_pipeline.sql`` coding cohort.
  * Row-wise explosion of ``neighbors`` into a long pair table (app loads one variant at a time).
  * Optional column catalog (e.g. ``~/duckdb_cols.txt`` from ``DESCRIBE variants``) to order wide
    output and to choose join keys for neighbor tables (exclude dense probe/annotation blocks).

Run from **variant-viewer** repo root (needs project deps: duckdb, polars, orjson):

  cd /path/to/variant-viewer
  uv run python eeve-analysis/scripts/export_analysis_datasets.py

Optional:

  uv run python eeve-analysis/scripts/export_analysis_datasets.py --limit 5000
  EVEE_DUCKDB_PATH=/path/to/variants.duckdb uv run python ...
  uv run python eeve-analysis/scripts/export_analysis_datasets.py --cols-file ~/duckdb_cols.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # variant-viewer/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "intermediate"
_DEFAULT_DB = _REPO_ROOT / "builds" / "variants.duckdb"
_DEFAULT_COLS_FILE = Path.home() / "duckdb_cols.txt"

# Dense feature / head columns: keep out of neighbor join to avoid huge duplicated blocks.
# Matches ``duckdb_cols.txt`` layout: ref/var/dist/spread annotation projections, multi-scale w*,
# z_* disruption, eff_* effect heads, pred_* classifier heads (incl. pred_pfam_*, pred_impact, …).
_FEATURE_COL_PREFIXES = (
    "ref_",
    "var_",
    "dist_",
    "spread_",
    "w0_",
    "w2_",
    "w64_",
    "z_",
    "eff_",
    "pred_",
)

# Fallback join keys if no catalog file (order preserved; still filtered by presence in DF)
_JOIN_META_COLS_FALLBACK = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene_name",
    "gene_id",
    "gene_strand",
    "consequence",
    "consequence_display",
    "vep_impact",
    "hgvsc",
    "hgvsp",
    "hgvsc_short",
    "hgvsp_short",
    "vep_transcript_id",
    "vep_protein_id",
    "domains",
    "label",
    "label_display",
    "significance",
    "review_status",
    "stars",
    "disease",
    "clinical_features",
    "pathogenicity",
    "score_pathogenic",
    "loeuf",
    "loeuf_label",
    "gnomad",
    "gnomad_display",
    "gnomad_label",
    "gnomad_af",
    "gnomad_afr_af",
    "gnomad_amr_af",
    "gnomad_asj_af",
    "gnomad_eas_af",
    "gnomad_fin_af",
    "gnomad_nfe_af",
    "gnomad_sas_af",
    "gnomad_genomes_af",
    "gnomad_af_c",
    "gnomad_exomes_c",
    "gnomad_genomes_c",
    "rs_id",
    "allele_id",
    "variation_id",
    "vcf_pos",
    "exon",
    "aa_swap",
    "acmg",
    "substitution",
    "cytogenetic",
    "submitters",
    "n_submissions",
    "last_evaluated",
    "origin",
]


def _load_column_catalog(path: Path | None) -> tuple[list[str], dict[str, str]]:
    """Parse DuckDB DESCRIBE-style CSV: column_name,column_type,...

    Returns (ordered names, name -> sql type string). Empty if path missing/unreadable.
    """
    names: list[str] = []
    types: dict[str, str] = {}
    if path is None or not path.is_file():
        return names, types
    try:
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "column_name" not in r.fieldnames:
                logging.warning("Catalog %s: missing column_name header", path)
                return names, types
            tkey = "column_type" if "column_type" in r.fieldnames else None
            for row in r:
                n = (row.get("column_name") or "").strip()
                if n:
                    names.append(n)
                    if tkey:
                        types[n] = (row.get(tkey) or "").strip()
    except OSError as e:
        logging.warning("Could not read column catalog %s: %s", path, e)
    return names, types


def _is_dense_feature_column(name: str) -> bool:
    if name == "neighbors":
        return True
    return name.startswith(_FEATURE_COL_PREFIXES)


def _join_meta_columns(
    df_columns: set[str],
    catalog_order: list[str],
) -> list[str]:
    """Context columns for neighbor joins: catalog order, minus dense feature blocks."""
    if catalog_order:
        out = [c for c in catalog_order if c in df_columns and not _is_dense_feature_column(c)]
        if out:
            return out
    return [c for c in _JOIN_META_COLS_FALLBACK if c in df_columns]


def _order_wide_columns(df_columns: set[str], catalog_order: list[str]) -> list[str]:
    """Stable order: catalog sequence first, then any extra columns alphabetically."""
    seen: set[str] = set()
    ordered: list[str] = []
    for c in catalog_order:
        if c in df_columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    rest = sorted(df_columns - seen)
    ordered.extend(rest)
    return ordered


def _neighbor_score_snapshot(el: dict) -> float | None:
    """``build.py`` stores pathogenicity-like score under ``score``."""
    v = el.get("score")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v2 = el.get("pathogenicity")
    if v2 is not None:
        try:
            return float(v2)
        except (TypeError, ValueError):
            pass
    return None


def _parse_neighbors_cell(raw: object) -> list[dict]:
    """Match serve.py / api.ts: JSON string -> list of neighbor dicts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s == "[]":
            return []
        try:
            import orjson as _orjson

            out = _orjson.loads(s)
        except Exception:
            out = json.loads(s)
        return out if isinstance(out, list) else []
    return []


def _coding_mask_polars():
    """Polars filter expression (mirrors phase2_neighbor_pipeline.sql WHERE)."""
    import polars as pl

    c = pl.col("consequence").cast(pl.Utf8).str.to_lowercase().fill_null("")
    cd = pl.col("consequence_display").cast(pl.Utf8).str.to_lowercase().fill_null("")
    vid = pl.col("variant_id").cast(pl.Utf8).str.strip_chars()
    g = pl.col("gene_name").cast(pl.Utf8).str.strip_chars()
    gid = pl.col("gene_id").cast(pl.Utf8).str.strip_chars()

    tokens_c = (
        c.str.contains("missense")
        | c.str.contains("frameshift")
        | c.str.contains("stop")
        | c.str.contains("splice")
        | c.str.contains("inframe")
        | c.str.contains("start_lost")
        | c.str.contains("stop_lost")
    )
    tokens_cd = (
        cd.str.contains("missense")
        | cd.str.contains("frameshift")
        | cd.str.contains("stop")
        | cd.str.contains("splice")
        | cd.str.contains("inframe")
        | cd.str.contains("start lost")
        | cd.str.contains("stop lost")
    )
    impact = pl.col("vep_impact").is_in(["HIGH", "MODERATE"])
    return (
        vid.is_not_null()
        & (vid != "")
        & g.is_not_null()
        & (g != "")
        & gid.is_not_null()
        & (gid != "")
        & (impact | tokens_c | tokens_cd)
    )


def _explode_neighbor_pairs(df):
    """Build list[dict] for neighbor long table."""
    import polars as pl

    rows: list[dict] = []
    n = df.height
    vids = df["variant_id"].to_list()
    nbs = df["neighbors"].to_list()
    for i in range(n):
        src = vids[i]
        if src is None:
            continue
        src = str(src).strip()
        if not src:
            continue
        parsed = _parse_neighbors_cell(nbs[i])
        for rank, el in enumerate(parsed, start=1):
            if not isinstance(el, dict):
                continue
            nid = el.get("id")
            if nid is None or str(nid).strip() == "":
                continue
            rows.append(
                {
                    "source_variant_id": src,
                    "neighbor_variant_id": str(nid).strip(),
                    "neighbor_rank": rank,
                    "neighbor_similarity": el.get("similarity"),
                    "neighbor_gene_json": el.get("gene"),
                    "neighbor_consequence_display_json": el.get("consequence_display"),
                    "neighbor_label_json": el.get("label"),
                    "neighbor_label_display_json": el.get("label_display"),
                    "neighbor_score_snapshot": _neighbor_score_snapshot(el),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "source_variant_id": pl.Utf8,
                "neighbor_variant_id": pl.Utf8,
                "neighbor_rank": pl.Int64,
                "neighbor_similarity": pl.Float64,
                "neighbor_gene_json": pl.Utf8,
                "neighbor_consequence_display_json": pl.Utf8,
                "neighbor_label_json": pl.Utf8,
                "neighbor_label_display_json": pl.Utf8,
                "neighbor_score_snapshot": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("EVEE_DUCKDB_PATH", str(_DEFAULT_DB))).expanduser(),
    )
    ap.add_argument("--out-dir", type=Path, default=_OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="max rows for debugging")
    ap.add_argument(
        "--cols-file",
        type=Path,
        default=None,
        help="CSV (DESCRIBE) with column_name column; default ~/duckdb_cols.txt or EVEE_DUCKDB_COLS",
    )
    ap.add_argument(
        "--ignore-cols-catalog",
        action="store_true",
        help="Skip catalog: keep DuckDB column order; use fallback join column list only",
    )
    args = ap.parse_args()

    db_path = args.db.resolve()
    if not db_path.is_file():
        logging.error("Database not found: %s", db_path)
        raise SystemExit(1)

    try:
        import polars as pl
    except ImportError:
        logging.error("polars required. Run from variant-viewer: uv run python ...")
        raise SystemExit(1)

    from db import open_db

    logging.info("Opening %s (read-only)", db_path)
    con = open_db(db_path, read_only=True)
    try:
        arrow = con.execute("SELECT * FROM variants").fetch_arrow_table()
        df = pl.from_arrow(arrow)
    finally:
        con.close()

    if args.limit is not None:
        df = df.head(args.limit)
        logging.info("Limited to %s rows", args.limit)

    logging.info("Loaded variants: %s rows × %s cols", df.height, df.width)

    cols_path: Path | None = None
    explicit_catalog = args.cols_file is not None
    if not args.ignore_cols_catalog:
        env_cols = os.environ.get("EVEE_DUCKDB_COLS")
        if args.cols_file is not None:
            cols_path = args.cols_file.expanduser()
        elif env_cols:
            cols_path = Path(env_cols).expanduser()
        else:
            cols_path = _DEFAULT_COLS_FILE

    if cols_path is not None and not cols_path.is_file():
        if explicit_catalog:
            logging.error("Column catalog not found: %s", cols_path)
            raise SystemExit(1)
        logging.info(
            "Column catalog not found (%s); keeping DuckDB column order; join keys from fallback list",
            cols_path,
        )
        cols_path = None

    cat_names, _cat_types = _load_column_catalog(cols_path)
    df_cols = set(df.columns)
    if cat_names:
        missing = [c for c in cat_names if c not in df_cols]
        extra = df_cols - set(cat_names)
        if missing:
            logging.warning(
                "Column catalog: %s listed names missing from table (catalog may be stale)",
                len(missing),
            )
        if extra:
            logging.info(
                "Column catalog: %s table columns not in catalog (appended after catalog order)",
                len(extra),
            )
        order = _order_wide_columns(df_cols, cat_names)
        df = df.select(order)
        logging.info("Wide column order: catalog (%s cols) + extras", len(cat_names))
    elif cols_path and cols_path.is_file():
        logging.warning("Catalog file unreadable or empty: %s", cols_path)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_path = out_dir / "variants_wide.parquet"
    logging.info("Writing %s", wide_path)
    df.write_parquet(wide_path)

    coding = df.filter(_coding_mask_polars())
    coding_path = out_dir / "coding_variants_base.parquet"
    logging.info("Coding subset: %s rows -> %s", coding.height, coding_path)
    coding.write_parquet(coding_path)

    pairs = _explode_neighbor_pairs(df)
    pairs_path = out_dir / "neighbor_pairs.parquet"
    logging.info("Neighbor pairs (all sources): %s rows -> %s", pairs.height, pairs_path)
    pairs.write_parquet(pairs_path)

    coding_ids = set(coding["variant_id"].cast(pl.Utf8).to_list())
    pairs_coding = pairs.filter(pl.col("source_variant_id").is_in(list(coding_ids)))
    pairs_coding_path = out_dir / "neighbor_pairs_coding_sources.parquet"
    logging.info(
        "Neighbor pairs (coding sources only): %s rows -> %s",
        pairs_coding.height,
        pairs_coding_path,
    )
    pairs_coding.write_parquet(pairs_coding_path)

    meta_cols = _join_meta_columns(set(df.columns), cat_names)
    logging.info("Neighbor join context columns: %s", len(meta_cols))
    ren_src = {"variant_id": "source_variant_id"}
    ren_src.update({c: f"source_{c}" for c in meta_cols if c != "variant_id"})
    ren_nb = {"variant_id": "neighbor_variant_id"}
    ren_nb.update({c: f"neighbor_{c}" for c in meta_cols if c != "variant_id"})
    src_meta = df.select(meta_cols).rename(ren_src)
    nb_meta = df.select(meta_cols).rename(ren_nb)
    joined = pairs.join(src_meta, on="source_variant_id", how="left").join(
        nb_meta, on="neighbor_variant_id", how="left"
    )
    joined_path = out_dir / "neighbor_pairs_joined.parquet"
    logging.info("Neighbor pairs joined (all sources): %s rows -> %s", joined.height, joined_path)
    joined.write_parquet(joined_path)

    joined_coding = pairs_coding.join(src_meta, on="source_variant_id", how="left").join(
        nb_meta, on="neighbor_variant_id", how="left"
    )
    joined_coding_path = out_dir / "neighbor_pairs_joined_coding_sources.parquet"
    logging.info(
        "Neighbor pairs joined (coding sources): %s rows -> %s",
        joined_coding.height,
        joined_coding_path,
    )
    joined_coding.write_parquet(joined_coding_path)

    logging.info("Done.")


if __name__ == "__main__":
    main()
