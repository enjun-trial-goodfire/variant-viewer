#!/bin/bash
# Validate all inputs before running build.py or the pipeline.
# Usage: bash preflight.sh [probe_name]
#
# Exits 0 if everything is ready, 1 if something is missing.

set -euo pipefail

PROBE="${1:-probe_v9}"
ARTIFACTS="/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
LABELED="$ARTIFACTS/clinvar_evo2_deconfounded_full"
VUS="$ARTIFACTS/clinvar_evo2_vus"
VEP="$ARTIFACTS/clinvar_evo2_labeled/variant_annotations"
DATA="$(dirname "$0")/data"

ok=0
fail=0

check() {
    if [ -e "$1" ]; then
        echo "  OK  $2"
        ok=$((ok + 1))
    else
        echo "  MISSING  $2  ($1)"
        fail=$((fail + 1))
    fi
}

echo "=== Preflight Check: $PROBE ==="
echo ""

echo "Local data (data/):"
check "$DATA/clinvar/deconfounded-full/metadata.feather" "Labeled metadata"
check "$DATA/clinvar/vus/metadata.feather" "VUS metadata"
check "$DATA/gencode/genes.feather" "GENCODE genes"
check "$DATA/clinvar/deconfounded-full/annotations_v8.feather" "Annotations v8"

echo ""
echo "Shared artifacts (probe scores + embeddings):"
check "$LABELED/$PROBE/scores.feather" "Labeled scores"
check "$LABELED/$PROBE/config.json" "Probe config"
check "$LABELED/$PROBE/embeddings" "Labeled embeddings"
check "$VUS/$PROBE/scores.feather" "VUS scores"
check "$VUS/$PROBE/embeddings" "VUS embeddings"

echo ""
echo "VEP annotations:"
n_vep=$(ls "$VEP"/variant_annotations_chr*.parquet 2>/dev/null | wc -l)
if [ "$n_vep" -ge 22 ]; then
    echo "  OK  $n_vep chromosome parquets"
    ok=$((ok + 1))
else
    echo "  MISSING  Only $n_vep/24 chromosome parquets at $VEP"
    fail=$((fail + 1))
fi

echo ""
echo "Optional:"
[ -f "$LABELED/$PROBE/eval.json" ] && echo "  OK  eval.json" || echo "  --  eval.json (no eval badges)"
[ -f "$LABELED/$PROBE/attribution.json" ] && echo "  OK  attribution.json" || echo "  --  attribution.json (no attribution)"
[ -f "$DATA/clinvar/submissions.feather" ] && echo "  OK  submissions.feather" || echo "  --  submissions.feather (no ACMG/submitters)"

echo ""
echo "Dependencies:"
python3 -c "import goodfire_core; print(f'  OK  goodfire-core {goodfire_core.__version__}')" 2>/dev/null \
    || echo "  MISSING  goodfire-core (run: uv sync)"
python3 -c "import polars; print(f'  OK  polars {polars.__version__}')" 2>/dev/null \
    || echo "  MISSING  polars"

echo ""
if [ "$fail" -eq 0 ]; then
    echo "All $ok checks passed. Ready to build."
    exit 0
else
    echo "$fail checks FAILED ($ok passed). Fix missing items before building."
    exit 1
fi
