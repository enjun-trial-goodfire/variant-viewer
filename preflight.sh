#!/bin/bash
# Validate all inputs before running build.py or the pipeline.
# Usage: bash preflight.sh [probe_name]
#
# Exits 0 if everything is ready, 1 if something is missing.

set -euo pipefail

PROBE="${1:-probe_v11}"
ARTIFACTS="${VV_ARTIFACTS:-/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian}"
LABELED="$ARTIFACTS/clinvar_evo2_deconfounded_full"
VUS="$ARTIFACTS/clinvar_evo2_vus"
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
check "$DATA/variants.parquet" "Variants parquet"
check "$DATA/heads.json" "Heads JSON"

echo ""
echo "Shared artifacts (probe scores + embeddings):"
check "$LABELED/$PROBE/scores.feather" "Labeled scores"
check "$LABELED/$PROBE/config.json" "Probe config"
check "$LABELED/$PROBE/embeddings" "Labeled embeddings"
check "$VUS/$PROBE/scores.feather" "VUS scores"
check "$VUS/$PROBE/embeddings" "VUS embeddings"

echo ""
echo "Optional:"
[ -f "$LABELED/$PROBE/eval.json" ] && echo "  OK  eval.json" || echo "  --  eval.json (no eval badges)"
[ -f "$LABELED/$PROBE/attribution.json" ] && echo "  OK  attribution.json" || echo "  --  attribution.json (no attribution)"

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
