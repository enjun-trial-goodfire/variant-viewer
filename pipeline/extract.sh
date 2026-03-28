#!/bin/bash
# SLURM wrapper for pipeline/extract.py (3-view embeddings + scores).
#
# Usage:
#   sbatch --parsable --array=0-7 pipeline/extract.sh \
#       --probe /path/to/probe --activations /path/to/storage

#SBATCH --job-name=extract
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract/%x_%A_%a.out
#SBATCH --error=logs/extract/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/extract

echo "=== Extract shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-1} on $(hostname) ==="

uv run python pipeline/extract.py \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    "$@"
