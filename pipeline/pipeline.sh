#!/bin/bash
# End-to-end pipeline: extract → finalize → eval → log-eval → build.
#
# Usage:
#   bash pipeline/pipeline.sh /path/to/probe [--labeled-only]
#
# Example:
#   ACTS=/mnt/polished-lake/artifacts/.../clinvar_evo2_deconfounded_full
#   bash pipeline/pipeline.sh $ACTS/probe_v11
#
# Chains SLURM jobs with dependencies so the full pipeline runs unattended.

set -euo pipefail

PROBE="${1:?Usage: pipeline.sh <probe-dir> [--labeled-only]}"
LABELED_ONLY="${2:-}"

ACTS=$(dirname "$PROBE")
PROBE_NAME=$(basename "$PROBE")
VUS=$(dirname "$ACTS")/clinvar_evo2_vus
cd "$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Pipeline: ${PROBE_NAME} ==="
echo "Probe:   ${PROBE}"
echo "Labeled: ${ACTS}"
echo "VUS:     ${VUS}"
echo ""

# Validate probe artifacts exist
for f in weights.pt config.json; do
    if [ ! -f "${PROBE}/${f}" ]; then
        echo "ERROR: ${PROBE}/${f} not found. Run training first."
        exit 1
    fi
done

# ── Clean stale SQLite locks (NFS + WAL = deadlock) ──────────────────
rm -f "${ACTS}/activations/index.sqlite-shm" "${ACTS}/activations/index.sqlite-wal"
rm -f "${VUS}/activations/index.sqlite-shm" "${VUS}/activations/index.sqlite-wal"

# ── Step 1: Extract (3-view embeddings + scores) ─────────────────────
echo "Step 1: Extract..."
EXT_L=$(sbatch --parsable --array=0-7 pipeline/extract.sh --probe "$PROBE" --activations "$ACTS")
echo "  Labeled extract: ${EXT_L}"

if [ -z "$LABELED_ONLY" ]; then
    EXT_V=$(sbatch --parsable --array=0-7 pipeline/extract.sh --probe "$PROBE" --activations "$VUS")
    echo "  VUS extract: ${EXT_V}"
fi

# ── Step 2: Finalize (merge shards + index embeddings) ───────────────
echo "Step 2: Finalize..."
FIN_L=$(sbatch --parsable --dependency=afterok:${EXT_L} pipeline/finalize.sh "$ACTS/${PROBE_NAME}")
echo "  Labeled finalize: ${FIN_L} (after ${EXT_L})"

if [ -z "$LABELED_ONLY" ]; then
    FIN_V=$(sbatch --parsable --dependency=afterok:${EXT_V} pipeline/finalize.sh "$VUS/${PROBE_NAME}")
    echo "  VUS finalize: ${FIN_V} (after ${EXT_V})"
    WAIT_FOR="${FIN_L}:${FIN_V}"
else
    WAIT_FOR="${FIN_L}"
fi

# ── Step 3: Eval + log-eval + build ──────────────────────────────────
echo "Step 3: Eval + build..."
EVAL_BUILD=$(sbatch --parsable --dependency=afterok:${WAIT_FOR} \
    --job-name=eval-build --gpus=1 --time=02:00:00 \
    --output=outputs/eval_build_%j.out \
    --wrap="cd \${SLURM_SUBMIT_DIR} && \
        uv run --frozen python pipeline/eval.py --probe-dir '${ACTS}/${PROBE_NAME}' && \
        uv run --frozen --extra train vv log-eval '${ACTS}/${PROBE_NAME}' && \
        uv run --frozen vv build --probe ${PROBE_NAME}")

echo "  Eval + build: ${EVAL_BUILD} (after ${WAIT_FOR})"
echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$(whoami) --format='%.10i %.20j %.8T %.10M'"
