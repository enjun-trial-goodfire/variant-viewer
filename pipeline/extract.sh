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

# Validate probe artifacts before using GPU time
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--probe" ]]; then
        for f in weights.pt config.json; do
            if [ ! -f "${arg}/${f}" ]; then
                echo "ERROR: ${arg}/${f} not found. Run training first."
                exit 1
            fi
        done
    fi
    prev="$arg"
done

# Clean stale SQLite WAL locks (NFS + WAL = deadlock).
# Only shard 0 cleans to avoid races between shards.
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    for arg in "$@"; do
        if [[ -d "$arg/activations" ]]; then
            rm -f "$arg/activations/index.sqlite-wal" "$arg/activations/index.sqlite-shm"
        fi
    done
fi

echo "=== Extract shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-1} on $(hostname) ==="

uv run --frozen python pipeline/extract.py \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    "$@"
