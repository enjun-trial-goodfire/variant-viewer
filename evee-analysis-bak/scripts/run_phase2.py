#!/usr/bin/env python3
"""Run Phase 2 SQL against variants.duckdb, materialize tables, export Parquet/CSV.

Usage (from eeve-analysis/):

  uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
  python scripts/run_phase2.py

Or one-shot:

  cd eeve-analysis && uv run --with duckdb python scripts/run_phase2.py

Override DB path:

  EVEE_DUCKDB_PATH=/path/to/variants.duckdb python scripts/run_phase2.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SQL_FILE = _REPO_ROOT / "sql" / "phase2_neighbor_pipeline.sql"
_DEFAULT_DB = Path(
    "/mnt/polished-lake/home/enjunyang/variant-viewer/builds/variants.duckdb"
)

_MATERIALIZED = (
    ("phase2_coding_variants_base", "coding_variants_base"),
    ("phase2_neighbor_pairs_raw", "neighbor_pairs_raw"),
    ("phase2_neighbor_pairs_joined", "neighbor_pairs_joined"),
)

_EXPORTS = (
    "coding_variants_base",
    "neighbor_pairs_raw",
    "neighbor_pairs_joined",
)


def _require_sql_file() -> None:
    if not _SQL_FILE.is_file():
        logging.error("Missing SQL file: %s", _SQL_FILE)
        raise SystemExit(1)


def _require_variants_table(con) -> None:
    row = con.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'variants'
        """
    ).fetchone()
    if not row or row[0] != 1:
        logging.error("Table main.variants not found in this database.")
        raise SystemExit(1)


def _execute_sql_file(con, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        con.execute(text)
    except Exception as e:
        logging.error("Failed executing %s: %s", path, e)
        raise SystemExit(1) from e


def _require_views(con) -> None:
    for _, view in _MATERIALIZED:
        q = """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
        """
        n = con.execute(q, [view]).fetchone()[0]
        if n != 1:
            logging.error("Expected view main.%s to exist after SQL run.", view)
            raise SystemExit(1)


def _materialize(con) -> None:
    for table_name, view_name in _MATERIALIZED:
        logging.info("Materializing %s <- %s", table_name, view_name)
        con.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM "{view_name}"'
        )
        n = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        logging.info("  rows: %s", f"{n:,}")


def _export(con, out_dir: Path, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_u = fmt.lower().strip()
    if fmt_u not in ("parquet", "csv"):
        logging.error("format must be parquet or csv, got %s", fmt)
        raise SystemExit(1)

    for logical in _EXPORTS:
        table = f"phase2_{logical}"
        # Ensure table exists
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if exists != 1:
            logging.error("Missing materialized table %s — materialization step failed?", table)
            raise SystemExit(1)

        out_path = out_dir / f"{logical}.{fmt_u}"
        logging.info("Exporting %s -> %s", table, out_path)
        path_sql = str(out_path).replace("'", "''")
        if fmt_u == "parquet":
            con.execute(f"COPY (SELECT * FROM \"{table}\") TO '{path_sql}' (FORMAT PARQUET)")
        else:
            con.execute(f"COPY (SELECT * FROM \"{table}\") TO '{path_sql}' (HEADER, DELIMITER ',')")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("EVEE_DUCKDB_PATH", str(_DEFAULT_DB))).expanduser(),
        help="Path to variants.duckdb (default: EVEE_DUCKDB_PATH or builds path)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "intermediate",
        help="Directory for exported files",
    )
    ap.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Export format",
    )
    ap.add_argument(
        "--read-only",
        action="store_true",
        help="Open DB read-only (will fail: materialization requires write access)",
    )
    args = ap.parse_args()

    db_path = args.db.resolve()
    if not db_path.is_file():
        logging.error("Database file not found: %s", db_path)
        raise SystemExit(1)
    if args.read_only:
        logging.error("--read-only is incompatible with creating phase2_* tables.")
        raise SystemExit(1)

    _require_sql_file()
    logging.info("Database: %s", db_path)
    logging.info("SQL file: %s", _SQL_FILE)

    try:
        import duckdb
    except ImportError:
        logging.error("duckdb is not installed. Run: uv pip install -r requirements.txt")
        raise SystemExit(1) from None

    con = duckdb.connect(str(db_path), read_only=False)
    try:
        _require_variants_table(con)
        logging.info("Executing Phase 2 SQL…")
        _execute_sql_file(con, _SQL_FILE)
        _require_views(con)
        _materialize(con)
        _export(con, args.out_dir.resolve(), args.format)
        logging.info("Done. Outputs under %s", args.out_dir.resolve())
    finally:
        con.close()


if __name__ == "__main__":
    main()
