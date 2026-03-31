"""Training utilities for probe development."""

from __future__ import annotations

import polars as pl

from loaders import load_heads
from probe.covariance import HeadSpec


def load_head_specs() -> tuple[dict[str, HeadSpec], dict[str, HeadSpec]]:
    """Load heads split into (disruption, effect) from heads.json."""
    heads = load_heads()
    disruption, effect = {}, {}
    for name, info in heads.items():
        spec = HeadSpec(n_classes=info["n_classes"], kind=info["kind"])
        if info["category"] == "effect":
            effect[name] = spec
        else:
            disruption[name] = spec
    return disruption, effect


def gene_split(
    metadata: pl.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_name",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split metadata by genes — no gene appears in both sets.

    Gene-level splitting prevents data leakage from shared gene effects.
    Deterministic: same (metadata, test_size, seed) → same split.
    """
    genes = metadata.select(gene_col).unique().sort(gene_col)
    n_test = int(len(genes) * test_size)
    shuffled = genes.sample(fraction=1.0, seed=seed, shuffle=True)
    test_genes = set(shuffled.head(n_test)[gene_col].to_list())

    test_mask = metadata[gene_col].is_in(list(test_genes))
    return metadata.filter(~test_mask), metadata.filter(test_mask)
