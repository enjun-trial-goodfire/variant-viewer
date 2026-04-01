"""DuckDB schema and helpers for the variant viewer.

The build step writes columns with names matching the frontend Variant type.
The server returns rows as-is. No transformation at serve time.

Tables:
  - variants: columns named to match frontend Variant type, indexed on id + gene
  - global_config: key-value store for heads, distributions, umap
  - interpretations: cached Claude interpretations
"""

from pathlib import Path

import duckdb
from loguru import logger

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS global_config (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS interpretations (
    variant_id    VARCHAR PRIMARY KEY,
    summary       VARCHAR NOT NULL,
    mechanism     VARCHAR NOT NULL,
    confidence    VARCHAR NOT NULL,
    key_evidence  VARCHAR NOT NULL,
    model         VARCHAR NOT NULL,
    generated_at  DOUBLE NOT NULL
);
"""


def create_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a new DuckDB database (deletes existing)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_SQL)
    logger.info(f"Created database: {db_path}")
    return conn


def open_db(db_path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open an existing DuckDB database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return duckdb.connect(str(db_path), read_only=read_only)
