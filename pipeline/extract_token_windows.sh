#!/bin/bash
# Extract window-mean disruption scores via SLURM array.
#
# Usage:
#   sbatch --array=0-31 pipeline/extract_token_windows.sh          # 32 GPUs
#   sbatch --array=0-63 pipeline/extract_token_windows.sh          # 64 GPUs
#   uv run python pipeline/extract_token_windows.py --n-shards 1   # single GPU

#SBATCH --job-name=extract_token_win
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/extract_token_win_%A_%a.out
#SBATCH --error=outputs/extract_token_win_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
# SLURM_ARRAY_TASK_COUNT counts from TASK_MIN, so --array=1-31 gives 31 not 32.
# Compute from the actual range: max_id + 1.
N_SHARDS=$(( SLURM_ARRAY_TASK_MAX + 1 ))

echo "Shard ${SHARD_ID}/${N_SHARDS} on $(hostname)"

uv run --frozen python pipeline/extract_token_windows.py \
    --shard-id "${SHARD_ID}" \
    --n-shards "${N_SHARDS}" \
    "$@"
