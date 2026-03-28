"""Data loading utilities for the variant viewer build pipeline."""

import json
from pathlib import Path

import polars as pl

from paths import MAYO_DATA, VEP_DOMAIN_CACHE


def load_vep(vep_dir: Path, columns: tuple[str, ...]) -> pl.DataFrame:
    """Load per-chromosome VEP parquets with automatic schema normalization.

    Handles the common issue where some chromosomes have String columns
    where others have Float64 (VEP annotation inconsistency).
    """
    string_cols = {"variant_id", "vep_hgvsc", "vep_hgvsp", "vep_impact",
                   "vep_exon", "vep_transcript_id", "vep_protein_id",
                   "vep_swissprot", "vep_domains"}
    dfs = []
    for f in sorted(vep_dir.glob("variant_annotations_chr*.parquet")):
        available = [c for c in columns if c in pl.read_parquet_schema(f)]
        chunk = pl.read_parquet(f, columns=available)
        float_casts = [c for c in chunk.columns if c not in string_cols and chunk[c].dtype in (pl.Utf8, pl.String)]
        if float_casts:
            chunk = chunk.with_columns(*(pl.col(c).cast(pl.Float64, strict=False) for c in float_casts))
        dfs.append(chunk)
    return pl.concat(dfs, how="diagonal") if dfs else pl.DataFrame()


def load_domain_names() -> dict[str, str]:
    """Load VEP domain ID->name cache (e.g., 'Pfam:PF00079' -> 'Serpin')."""
    if VEP_DOMAIN_CACHE.exists():
        with open(VEP_DOMAIN_CACHE) as f:
            return json.load(f)
    return {}


def resolve_domains(raw: str | None, cache: dict[str, str]) -> list[dict] | None:
    """Convert 'Pfam:PF00079;CDD:cd02056;...' to [{db, id, name?}, ...]."""
    if not raw:
        return None
    result = []
    for entry in raw.split(";"):
        parts = entry.split(":", 1)
        if len(parts) != 2:
            continue
        db, did = parts
        name = cache.get(f"{db}:{did}")
        d: dict = {"db": db, "id": did}
        if name:
            d["name"] = name
        result.append(d)
    return result or None


def load_metadata(preset: str) -> pl.DataFrame:
    """Load ClinVar metadata and enrich with gene_id/gene_strand from GENCODE."""
    meta = pl.read_ipc(MAYO_DATA / "clinvar" / preset / "metadata.feather")
    genes = (
        pl.read_ipc(MAYO_DATA / "gencode" / "genes.feather")
        .select("gene_name", "gene_id", "strand")
        .unique(subset=["gene_name"])
    )
    return (
        meta.join(genes, on="gene_name", how="inner")
        .unique(subset=["variant_id"])
        .rename({"strand": "gene_strand"})
    )
