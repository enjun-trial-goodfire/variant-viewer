"""DuckDB schema and helpers for the variant viewer.

Replaces the 232K static JSON files with a single DuckDB database.
Schema mirrors the per-variant JSON contract in SCHEMA.md.
"""

import json
from pathlib import Path

import duckdb
import orjson
from loguru import logger

# ── Schema ────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS variants (
    variant_id       VARCHAR PRIMARY KEY,
    gene             VARCHAR NOT NULL DEFAULT '',
    chrom            VARCHAR NOT NULL DEFAULT '',
    pos              INTEGER NOT NULL,
    ref              VARCHAR NOT NULL DEFAULT '',
    alt              VARCHAR NOT NULL DEFAULT '',
    vcf_pos          INTEGER NOT NULL,
    gene_strand      VARCHAR NOT NULL DEFAULT '',
    consequence      VARCHAR NOT NULL DEFAULT '',
    substitution     VARCHAR NOT NULL DEFAULT '',
    label            VARCHAR NOT NULL DEFAULT '',
    significance     VARCHAR NOT NULL DEFAULT '',
    stars            TINYINT NOT NULL DEFAULT 0,
    disease          VARCHAR NOT NULL DEFAULT '',
    score            DOUBLE NOT NULL,
    rs_id            VARCHAR NOT NULL DEFAULT '',
    allele_id        INTEGER,
    gene_id          VARCHAR NOT NULL DEFAULT '',
    hgvsc            VARCHAR NOT NULL DEFAULT '',
    hgvsp            VARCHAR NOT NULL DEFAULT '',
    impact           VARCHAR NOT NULL DEFAULT '',
    exon             VARCHAR NOT NULL DEFAULT '',
    transcript       VARCHAR NOT NULL DEFAULT '',
    swissprot        VARCHAR NOT NULL DEFAULT '',
    domains          VARCHAR NOT NULL DEFAULT '[]',
    loeuf            DOUBLE,
    gnomad           DOUBLE,
    gnomad_pop       VARCHAR NOT NULL DEFAULT '{}',
    variation_id     VARCHAR NOT NULL DEFAULT '',
    cytogenetic      VARCHAR NOT NULL DEFAULT '',
    review_status    VARCHAR NOT NULL DEFAULT '',
    acmg             VARCHAR NOT NULL DEFAULT '[]',
    n_submissions    INTEGER,
    submitters       VARCHAR NOT NULL DEFAULT '[]',
    last_evaluated   VARCHAR,
    clinical_features VARCHAR NOT NULL DEFAULT '[]',
    origin           VARCHAR NOT NULL DEFAULT '',
    disruption       VARCHAR NOT NULL DEFAULT '{}',
    effect           VARCHAR NOT NULL DEFAULT '{}',
    gt               VARCHAR NOT NULL DEFAULT '{}',
    attribution      VARCHAR NOT NULL DEFAULT '[]',
    neighbors        VARCHAR NOT NULL DEFAULT '[]',
    n_pathogenic     SMALLINT NOT NULL DEFAULT 0,
    n_benign         SMALLINT NOT NULL DEFAULT 0,
    n_vus            SMALLINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_gene_score ON variants(gene, score DESC);

CREATE TABLE IF NOT EXISTS global_config (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS umap (
    idx        INTEGER PRIMARY KEY,
    x          DOUBLE NOT NULL,
    y          DOUBLE NOT NULL,
    score      DOUBLE NOT NULL,
    variant_id VARCHAR NOT NULL,
    gene_idx   SMALLINT NOT NULL,
    label_idx  TINYINT NOT NULL
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
    """Create a new DuckDB database with the variant viewer schema."""
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


# ── Column mapping ────────────────────────────────────────────────────

# Maps DB column names to variant JSON keys (only where they differ)
_DB_TO_JSON = {
    "variant_id": "id",
    "n_pathogenic": "nP",
    "n_benign": "nB",
    "n_vus": "nV",
}

# JSON blob columns that need parsing
_JSON_COLUMNS = frozenset({
    "domains", "gnomad_pop", "acmg", "submitters", "clinical_features",
    "disruption", "effect", "gt", "attribution", "neighbors",
})


def row_to_variant(row: tuple, columns: list[str]) -> dict:
    """Convert a DuckDB row to the per-variant JSON dict expected by the frontend."""
    result = {}
    for col_name, value in zip(columns, row, strict=True):
        json_key = _DB_TO_JSON.get(col_name, col_name)
        if col_name in _JSON_COLUMNS:
            result[json_key] = json.loads(value) if value else ([] if col_name != "gnomad_pop" and col_name != "disruption" and col_name != "effect" and col_name != "gt" else {})
        else:
            result[json_key] = value
    return result


def variant_to_row(variant: dict) -> tuple:
    """Convert a variant dict (from _build_variant_dict) to a DB row tuple.

    Returns values in the same order as the variants table columns.
    """
    return (
        variant["id"],
        variant["gene"],
        variant["chrom"],
        variant["pos"],
        variant["ref"],
        variant["alt"],
        variant["vcf_pos"],
        variant["gene_strand"],
        variant["consequence"],
        variant["substitution"],
        variant["label"],
        variant["significance"],
        variant["stars"],
        variant["disease"],
        variant["score"],
        variant["rs_id"],
        variant.get("allele_id"),
        variant["gene_id"],
        variant["hgvsc"],
        variant["hgvsp"],
        variant["impact"],
        variant["exon"],
        variant["transcript"],
        variant["swissprot"],
        orjson.dumps(variant["domains"]).decode(),
        variant.get("loeuf"),
        variant.get("gnomad"),
        orjson.dumps(variant["gnomad_pop"]).decode(),
        variant["variation_id"],
        variant["cytogenetic"],
        variant["review_status"],
        orjson.dumps(variant["acmg"]).decode(),
        variant.get("n_submissions"),
        orjson.dumps(variant["submitters"]).decode(),
        variant.get("last_evaluated"),
        orjson.dumps(variant["clinical_features"]).decode(),
        variant["origin"],
        orjson.dumps(variant["disruption"]).decode(),
        orjson.dumps(variant["effect"]).decode(),
        orjson.dumps(variant["gt"]).decode(),
        orjson.dumps(variant["attribution"]).decode(),
        orjson.dumps(variant["neighbors"]).decode(),
        variant["nP"],
        variant["nB"],
        variant["nV"],
    )


VARIANT_COLUMNS = (
    "variant_id", "gene", "chrom", "pos", "ref", "alt", "vcf_pos",
    "gene_strand", "consequence", "substitution", "label", "significance",
    "stars", "disease", "score", "rs_id", "allele_id", "gene_id",
    "hgvsc", "hgvsp", "impact", "exon", "transcript", "swissprot",
    "domains", "loeuf", "gnomad", "gnomad_pop",
    "variation_id", "cytogenetic", "review_status", "acmg", "n_submissions",
    "submitters", "last_evaluated", "clinical_features", "origin",
    "disruption", "effect", "gt", "attribution", "neighbors",
    "n_pathogenic", "n_benign", "n_vus",
)

INSERT_VARIANTS_SQL = (
    f"INSERT INTO variants ({', '.join(VARIANT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in VARIANT_COLUMNS)})"
)


def interpretation_to_row(interp: dict) -> tuple:
    """Convert an interpretation dict to a DB row tuple."""
    return (
        interp["variant_id"],
        interp["summary"],
        interp["mechanism"],
        interp["confidence"],
        orjson.dumps(interp["key_evidence"]).decode(),
        interp["model"],
        interp["generated_at"],
    )


def row_to_interpretation(row: tuple, columns: list[str]) -> dict:
    """Convert a DuckDB interpretation row to the JSON dict expected by the frontend."""
    result = {"status": "ok"}
    for col_name, value in zip(columns, row, strict=True):
        if col_name == "key_evidence":
            result[col_name] = json.loads(value) if value else []
        else:
            result[col_name] = value
    return result
