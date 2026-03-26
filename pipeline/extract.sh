#!/bin/bash
# SLURM wrapper for scripts/extract.py (3-view embeddings + scores).
#
# Usage:
#   EXTRACT=$(sbatch --parsable --array=0-7 scripts/extract.sh \
#       --probe /path/to/probe \
#       --activations /path/to/storage)
#   sbatch --dependency=afterok:${EXTRACT} scripts/finalize_embed.sh /path/to/storage/probe_name

#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract/%x_%A_%a.out
#SBATCH --error=logs/extract/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/extract

N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=== Extract ==="
echo "Shard: ${SLURM_ARRAY_TASK_ID} / ${N_SHARDS}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Args:  $@"
echo ""

uv run python scripts/extract.py \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    "$@"

echo ""
echo "End: $(date)"
