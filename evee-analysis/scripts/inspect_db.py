#!/usr/bin/env python3
"""Inspect the variants DuckDB to understand its shape and contents.

Uses the repo's own ``db.open_db`` to connect (same path as the app server).

Usage (from repo root):
    uv run python eeve-analysis/scripts/inspect_db.py
    uv run python eeve-analysis/scripts/inspect_db.py --db builds/variants.duckdb
    uv run python eeve-analysis/scripts/inspect_db.py --sample 5
    uv run python eeve-analysis/scripts/inspect_db.py --output  # also saves to eeve-analysis/outputs/tables/
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_EEVE_ROOT = _SCRIPTS_DIR.parent                        # eeve-analysis/
_REPO_ROOT = _EEVE_ROOT.parent                          # variant-viewer/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from db import open_db  # noqa: E402

_DEFAULT_DB = _REPO_ROOT / "builds" / "variants.duckdb"
_OUTPUT_DIR = _EEVE_ROOT / "outputs" / "tables"

# Column prefixes that define logical groups (from the app's column naming convention).
# serve.py returns flat rows; api.ts uses these prefixes to reconstruct nested structures.
_PREFIX_GROUPS = {
    "ref_": "disruption ref score (annotation projection, ref allele)",
    "var_": "disruption var score (annotation projection, var allele)",
    "z_": "disruption z-score (normalized ref-var delta)",
    "dist_": "disruption distance (peak proximity)",
    "spread_": "disruption spread",
    "eff_": "effect head score",
    "gt_": "ground-truth (database) values",
    "pred_": "predictor / classifier head",
    "w0_": "multi-scale window (w=0, single-position)",
    "w2_": "multi-scale window (w=2)",
    "w64_": "multi-scale window (w=64)",
    "ref_pred_": "ref-allele structural predictor (overrides ref_)",
    "var_pred_": "var-allele structural predictor (overrides var_)",
}

# JSON-string columns (stored as VARCHAR, parsed client-side)
_JSON_COLUMNS = {"acmg", "clinical_features", "submitters", "domains", "neighbors", "attribution"}


def _classify_column(name: str) -> str:
    for prefix in sorted(_PREFIX_GROUPS, key=len, reverse=True):
        if name.startswith(prefix):
            return prefix
    return "metadata"


def _fmt_json_preview(raw: str | None, maxlen: int = 120) -> str:
    if raw is None:
        return "NULL"
    s = str(raw).strip()
    if len(s) <= maxlen:
        return s
    return s[:maxlen] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Path to variants.duckdb")
    ap.add_argument("--sample", type=int, default=3, help="Number of sample rows to display")
    ap.add_argument("--output", action="store_true", help="Also write report to eeve-analysis/outputs/tables/")
    args = ap.parse_args()

    db_path = args.db.resolve()
    if not db_path.is_file():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        raise SystemExit(1)

    con = open_db(db_path, read_only=True)

    # ── 1. Tables in this database ────────────────────────────────────
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    table_names = [r[0] for r in tables]
    print("=" * 72)
    print("TABLES IN DATABASE")
    print("=" * 72)
    for t in table_names:
        cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t:30s}  {cnt:>10,} rows")
    print()

    # ── 2. Variants table schema ──────────────────────────────────────
    if "variants" not in table_names:
        print("No 'variants' table found — nothing more to inspect.")
        con.close()
        return

    schema = con.execute("DESCRIBE variants").fetchall()
    col_names = [r[0] for r in schema]
    col_types = {r[0]: r[1] for r in schema}

    print("=" * 72)
    print(f"VARIANTS TABLE: {len(col_names)} columns")
    print("=" * 72)

    group_cols: dict[str, list[str]] = defaultdict(list)
    for c in col_names:
        group_cols[_classify_column(c)].append(c)

    for grp in sorted(group_cols, key=lambda g: (g != "metadata", g)):
        members = group_cols[grp]
        label = _PREFIX_GROUPS.get(grp, "core metadata / identifiers / clinical fields")
        print(f"\n  [{grp}] — {label} ({len(members)} cols)")
        types_in_group = Counter(col_types[c] for c in members)
        type_str = ", ".join(f"{t}×{n}" for t, n in types_in_group.most_common())
        print(f"    types: {type_str}")
        if len(members) <= 20:
            for c in members:
                print(f"      {c:50s}  {col_types[c]}")
        else:
            for c in members[:5]:
                print(f"      {c:50s}  {col_types[c]}")
            print(f"      ... ({len(members) - 10} more) ...")
            for c in members[-5:]:
                print(f"      {c:50s}  {col_types[c]}")

    # ── 3. Row count breakdown ────────────────────────────────────────
    total = con.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
    print(f"\n{'=' * 72}")
    print(f"ROW COUNTS")
    print(f"{'=' * 72}")
    print(f"  Total variants:  {total:>10,}")

    with_neighbors = con.execute(
        "SELECT COUNT(*) FROM variants WHERE neighbors IS NOT NULL AND neighbors != '[]' AND neighbors != ''"
    ).fetchone()[0]
    print(f"  With neighbors:  {with_neighbors:>10,}")

    with_pathogenicity = con.execute(
        "SELECT COUNT(*) FROM variants WHERE pathogenicity IS NOT NULL"
    ).fetchone()[0]
    print(f"  With pathogenicity: {with_pathogenicity:>10,}")

    # Consequence distribution (top 10)
    csq = con.execute(
        "SELECT consequence_display, COUNT(*) AS n FROM variants "
        "GROUP BY consequence_display ORDER BY n DESC LIMIT 10"
    ).fetchall()
    print(f"\n  Top consequences:")
    for label, n in csq:
        print(f"    {str(label):40s}  {n:>8,}")

    # Label distribution
    labels = con.execute(
        "SELECT label, COUNT(*) AS n FROM variants GROUP BY label ORDER BY n DESC"
    ).fetchall()
    print(f"\n  Labels:")
    for label, n in labels:
        print(f"    {str(label):40s}  {n:>8,}")

    # VEP impact distribution
    impacts = con.execute(
        "SELECT vep_impact, COUNT(*) AS n FROM variants GROUP BY vep_impact ORDER BY n DESC"
    ).fetchall()
    print(f"\n  VEP impact:")
    for label, n in impacts:
        print(f"    {str(label):40s}  {n:>8,}")

    # Pathogenicity stats
    pstats = con.execute(
        "SELECT MIN(pathogenicity), MAX(pathogenicity), AVG(pathogenicity), "
        "APPROX_QUANTILE(pathogenicity, 0.5) FROM variants WHERE pathogenicity IS NOT NULL"
    ).fetchone()
    if pstats and pstats[0] is not None:
        print(f"\n  Pathogenicity (where not null):")
        print(f"    min={pstats[0]:.4f}  max={pstats[1]:.4f}  mean={pstats[2]:.4f}  median≈{pstats[3]:.4f}")

    # Chromosome distribution
    chroms = con.execute(
        "SELECT chrom, COUNT(*) AS n FROM variants GROUP BY chrom ORDER BY n DESC LIMIT 10"
    ).fetchall()
    print(f"\n  Top chromosomes:")
    for ch, n in chroms:
        print(f"    {str(ch):10s}  {n:>8,}")

    # ── 4. Neighbor structure ─────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("NEIGHBOR STRUCTURE")
    print("=" * 72)
    print(f"  Column type:  {col_types.get('neighbors', 'NOT FOUND')}")
    print(f"  Storage:      JSON string (parsed client-side via JSON.parse / orjson.loads)")
    print(f"  Schema per neighbor object (from build.py):")
    print(f"    id: str, gene: str, consequence_display: str, label: str,")
    print(f"    label_display: str, score: float, similarity: float")

    sample_nb = con.execute(
        "SELECT variant_id, neighbors FROM variants "
        "WHERE neighbors IS NOT NULL AND neighbors != '[]' AND neighbors != '' LIMIT 1"
    ).fetchone()
    if sample_nb:
        vid, raw = sample_nb
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        count = len(parsed) if isinstance(parsed, list) else 0
        print(f"\n  Example (variant_id={vid}): {count} neighbors")
        if isinstance(parsed, list) and parsed:
            print(f"    Keys in first neighbor: {sorted(parsed[0].keys())}")
            for i, nb in enumerate(parsed[:3]):
                print(f"    [{i}] id={nb.get('id')}, gene={nb.get('gene')}, "
                      f"score={nb.get('score')}, similarity={nb.get('similarity'):.4f}")

    nb_counts = con.execute(
        "SELECT len(json_extract(neighbors, '$')) AS k, COUNT(*) AS n "
        "FROM variants WHERE neighbors IS NOT NULL AND neighbors != '[]' AND neighbors != '' "
        "GROUP BY k ORDER BY k"
    ).fetchall()
    if nb_counts:
        print(f"\n  Neighbor count distribution:")
        for k, n in nb_counts:
            print(f"    {k} neighbors: {n:>8,} variants")

    # ── 5. Sample rows ────────────────────────────────────────────────
    key_cols = [
        "variant_id", "chrom", "pos", "vcf_pos", "ref", "alt",
        "gene_name", "gene_id", "consequence_display", "vep_impact",
        "label", "label_display", "pathogenicity", "score_pathogenic",
        "hgvsc_short", "hgvsp_short", "rs_id",
        "loeuf", "gnomad", "stars",
    ]
    existing_key = [c for c in key_cols if c in col_types]
    select_str = ", ".join(existing_key)

    print(f"\n{'=' * 72}")
    print(f"SAMPLE ROWS ({args.sample} variants with neighbors + pathogenicity)")
    print("=" * 72)

    samples = con.execute(
        f"SELECT {select_str} FROM variants "
        "WHERE pathogenicity IS NOT NULL AND neighbors IS NOT NULL AND neighbors != '[]' "
        f"ORDER BY pathogenicity DESC LIMIT {args.sample}"
    ).fetchall()
    for row in samples:
        d = dict(zip(existing_key, row))
        print()
        for k, v in d.items():
            print(f"    {k:30s}  {v}")

    # ── 6. Global config ──────────────────────────────────────────────
    if "global_config" in table_names:
        print(f"\n{'=' * 72}")
        print("GLOBAL_CONFIG TABLE")
        print("=" * 72)
        gkeys = con.execute("SELECT key, LENGTH(value) FROM global_config").fetchall()
        for k, sz in gkeys:
            print(f"  key={k:20s}  value_length={sz:>10,} chars")

        heads_row = con.execute("SELECT value FROM global_config WHERE key = 'heads'").fetchone()
        if heads_row:
            hd = json.loads(heads_row[0])
            heads = hd.get("heads", hd)
            cats = Counter(v.get("category", "?") for v in heads.values())
            print(f"\n  heads.json: {len(heads)} heads")
            for cat, n in cats.most_common():
                print(f"    {cat}: {n}")

    # ── 7. JSON-string columns preview ────────────────────────────────
    json_in_db = [c for c in _JSON_COLUMNS if c in col_types]
    if json_in_db:
        print(f"\n{'=' * 72}")
        print("JSON-STRING COLUMNS (stored as VARCHAR, parsed client-side)")
        print("=" * 72)
        for c in json_in_db:
            sample = con.execute(
                f'SELECT "{c}" FROM variants WHERE "{c}" IS NOT NULL AND "{c}" != \'[]\' LIMIT 1'
            ).fetchone()
            preview = _fmt_json_preview(sample[0] if sample else None, maxlen=120)
            print(f"  {c:25s}  type={col_types[c]:10s}  sample: {preview}")

    # ── 8. Indexes ────────────────────────────────────────────────────
    indexes = con.execute(
        "SELECT index_name, table_name, is_unique FROM duckdb_indexes()"
    ).fetchall()
    if indexes:
        print(f"\n{'=' * 72}")
        print("INDEXES")
        print("=" * 72)
        for name, tbl, uniq in indexes:
            print(f"  {name:30s}  on {tbl:15s}  unique={uniq}")

    con.close()
    print(f"\n{'=' * 72}")
    print("Done.")

    if args.output:
        print(f"\nTip: to save the full report, re-run with tee:")
        print(f"  uv run python eeve-analysis/scripts/inspect_db.py --sample {args.sample} "
              f"| tee eeve-analysis/outputs/tables/db_inspection_report.txt")


if __name__ == "__main__":
    main()
