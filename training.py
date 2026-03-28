"""Training utilities for probe development.

Small, focused functions used by the probe training pipeline.
Extracted from gfm_gen/src/ to make variant-viewer self-contained.
"""

from __future__ import annotations

import polars as pl

N_BINS = 16  # Number of bins for continuous head soft-binning

# Effect-probe heads: variant-level predictions, clinical scores, cross-features
_DIFF_PATTERNS = frozenset({
    "pathogenic", "consequence", "csq_x_path", "impact", "aa_swap",
    "splice_disrupting", "charge_altering", "region_x_path",
    "cadd_c", "alphamissense_c", "sift_c", "polyphen_c", "eve_c", "vest4_c",
    "mpc_c", "gnomad_af_c", "revel_c", "primateai_c", "mutpred_c", "metalr_c",
    "mcap_c", "mvp_c", "clinpred_c", "deogen2_c", "bayesdel_c",
    "spliceai_max_c", "spliceai_ag_c", "spliceai_al_c", "spliceai_dg_c", "spliceai_dl_c",
    "blosum62_c", "grantham_c", "hydrophobicity_c", "mw_c", "volume_c", "cadd_wg_c",
})
_DIFF_PREFIXES = ("pfam_",)


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


def discover_heads(labels: pl.DataFrame) -> dict[str, tuple[int, str]]:
    """Auto-discover head definitions from an annotated DataFrame.

    Returns {name: (n_classes, kind)} where kind is "binary", "continuous", or "categorical".
    """
    heads: dict[str, tuple[int, str]] = {}
    for col in labels.columns:
        if col == "variant_id":
            continue
        dtype = labels[col].dtype
        non_null = labels[col].drop_nulls()
        if non_null.len() == 0:
            continue

        if dtype == pl.Boolean:
            heads[col] = (2, "binary")
        elif dtype in (pl.Float32, pl.Float64):
            heads[col] = (N_BINS, "continuous")
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            unique = set(non_null.unique().to_list())
            if unique <= {0, 1}:
                heads[col] = (2, "binary")
            else:
                heads[col] = (int(non_null.max()) + 1, "categorical")
    return heads


def split_heads(
    heads: dict[str, tuple[int, str]],
    pathogenic_weight: float = 2.0,
) -> tuple[dict[str, tuple[int, str, float]], dict[str, tuple[int, str, float]]]:
    """Split auto-discovered heads into disruption (ref) and effect (diff) sets.

    Returns (disruption_heads, effect_heads) — each {name: (n_classes, kind, weight)}.
    """
    ref, diff = {}, {}
    for name, (n_classes, kind) in heads.items():
        weight = pathogenic_weight if name == "pathogenic" else 1.0
        is_diff = name in _DIFF_PATTERNS or any(name.startswith(p) for p in _DIFF_PREFIXES)
        if name.endswith("_x_path"):
            is_diff = True
        target = diff if is_diff else ref
        target[name] = (n_classes, kind, weight)
    return ref, diff
