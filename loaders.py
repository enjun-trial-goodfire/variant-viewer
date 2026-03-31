"""Data loading utilities for the variant viewer.

Single data source: data/variants.parquet (pre-joined by annotator export).
Head metadata: data/heads.json (specs + display names + categories).
"""

import json

import polars as pl

from paths import HEADS, VARIANTS


def load_variants(columns: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Load the pre-joined variants parquet.

    Args:
        columns: Optional subset of columns to load (for memory efficiency).
    """
    return pl.read_parquet(VARIANTS, columns=list(columns) if columns else None)


def load_heads() -> dict[str, dict]:
    """Load head metadata: {name: {n_classes, kind, category, display_name, group}}."""
    return json.loads(HEADS.read_text())
