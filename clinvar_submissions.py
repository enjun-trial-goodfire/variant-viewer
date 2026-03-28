"""Download and parse ClinVar submission-level data.

Produces a clean feather file with per-variant clinical metadata:
ACMG criteria, submitters, dates, phenotypes, inheritance, cytogenetic band.

Downloads ~800MB from ClinVar FTP (cached). Output: data/clinvar/submissions.feather

Usage:
    uv run python clinvar_submissions.py
"""

import gzip
import re
import urllib.request
from pathlib import Path

import polars as pl
from loguru import logger

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited"
DOWNLOAD_DIR = Path("data/clinvar/_downloads")
OUTPUT = Path("data/clinvar/submissions.feather")

VARIANT_SUMMARY_URL = f"{FTP_BASE}/variant_summary.txt.gz"
SUBMISSION_SUMMARY_URL = f"{FTP_BASE}/submission_summary.txt.gz"


def _download(url: str, dest: Path) -> Path:
    """Download if not cached."""
    if dest.exists():
        logger.info(f"Cached: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Saved: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _read_tsv_gz(path: Path) -> pl.DataFrame:
    """Read ClinVar gzipped TSV via pandas (handles ragged/malformed rows).

    Uses pandas instead of polars because ClinVar's ragged TSV format causes
    polars' CSV parser to fail. pandas' on_bad_lines="skip" handles this.

    ClinVar files have # comment lines. The LAST # line is the header.
    variant_summary uses #AlleleID (single #), submission_summary uses ## + #.
    """
    import pandas as pd

    # Find the header: last line starting with #
    header = None
    skip_rows = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                header = line.lstrip("#").strip().split("\t")
                skip_rows += 1
            else:
                break

    df = pd.read_csv(path, sep="\t", skiprows=skip_rows, header=None,
                     dtype=str, on_bad_lines="skip", na_values=["-"])
    df.columns = header[:len(df.columns)]
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns from {path.name}")
    return pl.from_pandas(df)


def parse_variant_summary(path: Path) -> pl.DataFrame:
    """Parse variant_summary.txt.gz → per-allele variant metadata (GRCh38 only)."""
    logger.info("Parsing variant_summary...")
    df = _read_tsv_gz(path)
    # Filter to GRCh38
    df = df.filter(pl.col("Assembly") == "GRCh38")
    return df.select(
        pl.col("AlleleID").cast(pl.Int64).alias("allele_id"),
        pl.col("VariationID").cast(pl.Int64).alias("variation_id"),
        pl.col("Cytogenetic").alias("cytogenetic"),
        pl.col("ReviewStatus").alias("review_status"),
        pl.col("NumberSubmitters").cast(pl.Int32).alias("n_submitters"),
        pl.col("LastEvaluated").alias("last_evaluated"),
        pl.col("Origin").alias("origin"),
        pl.col("PhenotypeList").alias("phenotypes"),
        pl.col("PhenotypeIDS").alias("phenotype_ids"),
        pl.col("OtherIDs").alias("other_ids"),
    ).unique(subset=["allele_id"])


def parse_submission_summary(path: Path) -> pl.DataFrame:
    """Parse submission_summary.txt.gz → per-submission details, aggregated by VariationID."""
    logger.info("Parsing submission_summary...")
    df = _read_tsv_gz(path)

    # Extract ACMG codes from ExplanationOfInterpretation
    # Format varies: "PS2_MOD, PS3, PM1, PM2, PP2, PP3" or embedded in prose
    # The Description field sometimes has them too
    # Most reliable: ExplanationOfInterpretation column
    return df.select(
        pl.col("VariationID").cast(pl.Int64).alias("variation_id"),
        pl.col("Submitter").alias("submitter"),
        pl.col("DateLastEvaluated").alias("date_evaluated"),
        pl.col("ReviewStatus").alias("submission_review_status"),
        pl.col("CollectionMethod").alias("collection_method"),
        pl.col("SubmittedPhenotypeInfo").alias("submitted_phenotype"),
        pl.col("Description").alias("description"),
        pl.col("ClinicalSignificance").alias("submission_significance"),
        pl.col("SCV").alias("scv"),
    )


def _extract_acmg_codes(text: str) -> str:
    """Extract ACMG/AMP criteria codes from free text."""
    pattern = r"\b(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])(?:_(?:Strong|Moderate|Supporting|VeryStrong|MOD|SUP|STR))?\b"
    codes = re.findall(pattern, text, re.IGNORECASE)
    # Also check for underscore-modified codes like PS2_MOD
    pattern_mod = r"\b(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])_(Strong|Moderate|Supporting|VeryStrong|MOD|SUP|STR)\b"
    mods = re.findall(pattern_mod, text, re.IGNORECASE)
    full_codes = [f"{c}_{m}" for c, m in mods]
    # Merge: use modified versions where available, base otherwise
    base_with_mods = {c for c, _ in mods}
    result = [c for c in codes if c.upper() not in {b.upper() for b in base_with_mods}] + full_codes
    return ";".join(sorted(set(result))) if result else ""


def aggregate_submissions(submissions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-submission data to per-variation_id."""
    logger.info("Aggregating submissions...")

    # Extract ACMG codes from Description field (where submitters put "Criteria applied: PS2, PM1...")
    descriptions = submissions.select("variation_id", "description").to_dicts()
    acmg_by_var: dict[int, set[str]] = {}
    for row in descriptions:
        vid = row["variation_id"]
        text = row["description"] or ""
        codes = _extract_acmg_codes(text)
        if codes:
            acmg_by_var.setdefault(vid, set()).update(codes.split(";"))

    acmg_df = pl.DataFrame([
        {"variation_id": vid, "acmg_codes": ";".join(sorted(codes))}
        for vid, codes in acmg_by_var.items()
    ]) if acmg_by_var else pl.DataFrame({"variation_id": pl.Series([], dtype=pl.Int64), "acmg_codes": pl.Series([], dtype=pl.Utf8)})

    agg = submissions.group_by("variation_id").agg(
        pl.col("submitter").unique().sort().str.concat(";").alias("submitters"),
        pl.col("date_evaluated").max().alias("latest_evaluation"),
        pl.col("collection_method").unique().sort().str.concat(";").alias("collection_methods"),
        pl.col("submitted_phenotype").unique().sort().str.concat(";").alias("clinical_features"),
        pl.len().alias("n_submissions"),
    )

    return agg.join(acmg_df, on="variation_id", how="left").with_columns(
        pl.col("acmg_codes").fill_null(""),
    )


def main() -> None:
    # Download
    vs_path = _download(VARIANT_SUMMARY_URL, DOWNLOAD_DIR / "variant_summary.txt.gz")
    ss_path = _download(SUBMISSION_SUMMARY_URL, DOWNLOAD_DIR / "submission_summary.txt.gz")

    # Parse
    variants = parse_variant_summary(vs_path)
    submissions = parse_submission_summary(ss_path)
    logger.info(f"Variant summary: {variants.height:,} variants (GRCh38)")
    logger.info(f"Submission summary: {submissions.height:,} submissions")

    # Aggregate submissions by variation_id
    sub_agg = aggregate_submissions(submissions)

    # Join: variant_summary has allele_id + variation_id, sub_agg has variation_id
    result = variants.join(sub_agg, on="variation_id", how="left").with_columns(
        pl.col("acmg_codes").fill_null(""),
        pl.col("submitters").fill_null(""),
        pl.col("clinical_features").fill_null(""),
        pl.col("n_submissions").fill_null(0),
    )

    # Clean up
    result = result.with_columns(
        pl.col("cytogenetic").replace("-", None),
        pl.col("last_evaluated").replace("-", None),
        pl.col("origin").replace("-", None),
        pl.col("phenotypes").replace("-", None),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.write_ipc(OUTPUT)
    logger.info(f"Saved: {OUTPUT} ({result.height:,} variants, {len(result.columns)} columns)")

    # Summary stats
    has_acmg = result.filter(pl.col("acmg_codes") != "").height
    has_submitters = result.filter(pl.col("submitters") != "").height
    has_cyto = result.filter(pl.col("cytogenetic").is_not_null()).height
    logger.info(f"ACMG codes: {has_acmg:,} variants ({has_acmg/result.height:.1%})")
    logger.info(f"Submitters: {has_submitters:,} variants ({has_submitters/result.height:.1%})")
    logger.info(f"Cytogenetic: {has_cyto:,} variants ({has_cyto/result.height:.1%})")


if __name__ == "__main__":
    main()
