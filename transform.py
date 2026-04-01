"""Transform raw parquet + scores into the clean schema served to the frontend.

Single step: read → join → rename → convert → write.
The output parquet has column names matching the frontend Variant type.
build.py reads this output — no further column renaming downstream.

Usage:
    uv run vv transform --probe probe_v12 [--dev 1000] [--output builds/clean.parquet]
"""

import json
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from constants import AA_SWAP_CLASSES, PROBE_NAME
from loaders import load_heads, load_variants
from paths import ARTIFACTS

LABELED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS = ARTIFACTS / "clinvar_evo2_vus"

# ── Column renames ────────────────────────────────────────────────────
# Raw parquet/scores name → frontend Variant field name.
# Score columns (ref_score_*, var_score_*, score_*) pass through unchanged.

RENAMES = {
    "variant_id": "id",
    "gene_name": "gene",
    "disease_name": "disease",
    "clinical_significance": "significance",
    "score_pathogenic": "score",
    "vep_hgvsc": "hgvsc",
    "vep_hgvsp": "hgvsp",
    "vep_impact": "impact",
    "vep_exon": "exon",
    "vep_transcript_id": "transcript",
    "vep_protein_id": "protein_id",
    "vep_loeuf": "loeuf",
    "gnomad_af": "gnomad",
}

# Columns that are metadata, not probe heads. Everything else that
# isn't a score column and matches a head name gets a gt_ prefix.
META_COLS = frozenset({
    "id", "gene", "chrom", "pos", "ref", "alt", "vcf_pos", "gene_strand",
    "consequence", "substitution", "label", "significance", "stars", "disease",
    "score", "rs_id", "allele_id", "gene_id",
    "hgvsc", "hgvsp", "impact", "exon", "transcript", "protein_id",
    "domains", "loeuf", "gnomad",
    "gnomad_afr_af", "gnomad_amr_af", "gnomad_asj_af", "gnomad_eas_af",
    "gnomad_fin_af", "gnomad_nfe_af", "gnomad_sas_af", "gnomad_genomes_af",
    "variation_id", "cytogenetic", "review_status", "acmg",
    "n_submissions", "submitters", "last_evaluated", "clinical_features", "origin",
    # Raw columns that survive through but aren't heads
    "gnomad_af_c", "gnomad_exomes_c", "gnomad_genomes_c",
    "aa_swap", "pred_aa_swap",
})

# Domains to exclude from the parsed list
EXCLUDED_DOMAIN_DBS = frozenset({"PDB-ENSP_mappings", "AFDB-ENSP_mappings", "ENSP_mappings", "Gene3D"})
EXCLUDE_CLINICAL = frozenset({"not provided", "not specified", ""})


def _semi_to_json(s: str | None, exclude: frozenset[str] = frozenset()) -> str:
    if not s:
        return "[]"
    parts = [p.strip() for p in s.split(";") if p.strip() and p.strip().lower() not in exclude]
    return json.dumps(parts)


def _parse_domains(raw: str | None) -> str:
    if not raw:
        return "[]"
    result = []
    for entry in raw.split(","):
        parts = entry.strip().split(":", 1)
        if len(parts) == 2 and parts[0] not in EXCLUDED_DOMAIN_DBS:
            result.append({"db": parts[0], "id": parts[1]})
    return json.dumps(result)


def _decode_aa_swap(idx: int | None) -> str | None:
    if idx is None or idx < 0 or idx >= len(AA_SWAP_CLASSES):
        return None
    return AA_SWAP_CLASSES[idx]


def main(
    probe: str = typer.Option(PROBE_NAME, help="Probe version"),
    output: Path = typer.Option(Path("builds/clean.parquet"), help="Output parquet path"),
    dev: int | None = typer.Option(None, help="Dev mode: limit to N variants"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    logger.info("Loading variants")
    df_variants = load_variants()
    logger.info(f"  {df_variants.height:,} rows, {df_variants.width} columns")

    logger.info(f"Loading scores ({probe})")
    df_scores = pl.read_ipc(str(LABELED / probe / "scores.feather"))
    vus_path = VUS / probe / "scores.feather"
    if vus_path.exists():
        df_vus = pl.read_ipc(str(vus_path))
        logger.info(f"  + {df_vus.height:,} VUS rows")
        df_scores = pl.concat([df_scores, df_vus], how="diagonal")
    logger.info(f"  {df_scores.height:,} score rows, {df_scores.width} columns")

    # ── Join ──────────────────────────────────────────────────────────
    df = df_scores.join(df_variants, on="variant_id", how="left")
    logger.info(f"  Joined: {df.height:,} rows, {df.width} columns")

    if dev:
        df = df.head(dev)
        logger.info(f"  Dev mode: {dev} variants")

    # ── Rename annotation columns that clash with head names → gt_ prefix
    heads = load_heads()
    gt_renames = {h: f"gt_{h}" for h in heads if h in df.columns and h not in META_COLS
                  and not h.startswith("ref_score_") and not h.startswith("var_score_") and not h.startswith("score_")}
    df = df.rename(gt_renames)

    # ── Rename to frontend field names ────────────────────────────────
    # Drop raw columns that would collide with renamed versions
    renames = {old: new for old, new in RENAMES.items() if old in df.columns}
    collisions = [new for new in renames.values() if new in df.columns and new not in renames]
    if collisions:
        logger.info(f"Dropping colliding columns before rename: {collisions}")
        df = df.drop(collisions)
    df = df.rename(renames)

    # ── Decode aa_swap → substitution string ──────────────────────────
    if "pred_aa_swap" in df.columns:
        df = df.with_columns(
            pl.col("pred_aa_swap").map_elements(_decode_aa_swap, return_dtype=pl.String).alias("substitution")
        ).drop("pred_aa_swap")
    elif "substitution" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.String).alias("substitution"))

    # ── Compute vcf_pos ───────────────────────────────────────────────
    if "vcf_pos" not in df.columns:
        df = df.with_columns((pl.col("pos") + 1).alias("vcf_pos"))

    # ── Semicolons → JSON arrays ──────────────────────────────────────
    if "acmg_codes" in df.columns:
        df = df.with_columns(
            pl.col("acmg_codes").map_elements(_semi_to_json, return_dtype=pl.String).alias("acmg")
        ).drop("acmg_codes")

    if "clinical_features" in df.columns:
        df = df.with_columns(
            pl.col("clinical_features").map_elements(
                lambda s: _semi_to_json(s, EXCLUDE_CLINICAL), return_dtype=pl.String
            ).alias("clinical_features")
        )

    if "submitters" in df.columns:
        df = df.with_columns(
            pl.col("submitters").map_elements(_semi_to_json, return_dtype=pl.String).alias("submitters")
        )

    # ── Domains → JSON array of {db, id} ─────────────────────────────
    if "domains" in df.columns:
        df = df.with_columns(
            pl.col("domains").map_elements(_parse_domains, return_dtype=pl.String).alias("domains")
        )
    elif "vep_domains" in df.columns:
        df = df.with_columns(
            pl.col("vep_domains").map_elements(_parse_domains, return_dtype=pl.String).alias("domains")
        ).drop("vep_domains")

    # ── Fill nulls + round floats ─────────────────────────────────────
    cfg = json.loads((LABELED / probe / "config.json").read_text())
    disruption_set = set(cfg.get("disruption_heads", []))
    effect_set = set(cfg.get("effect_heads", []))

    float_cols = [c for c in df.columns
                  if (c.startswith("ref_score_") or c.startswith("var_score_") or
                      c.startswith("score_") or c.startswith("gt_"))
                  and df[c].dtype in (pl.Float32, pl.Float64)]
    df = df.with_columns(
        *(pl.col(c).round(4).fill_nan(None) for c in float_cols),
        pl.col("gene").fill_null("?"),
        pl.col("consequence").fill_null("unknown"),
        pl.col("label").fill_null("?"),
        pl.col("significance").fill_null(""),
        pl.col("stars").fill_null(0),
        pl.col("disease").fill_null(""),
        pl.col("score").fill_null(0.0).fill_nan(0.0).round(4),
    )

    # ── Write ─────────────────────────────────────────────────────────
    logger.info(f"Writing {output} ({df.height:,} rows, {df.width} columns)")
    df.write_parquet(output)
    logger.info("Done")


if __name__ == "__main__":
    typer.run(main)
