#!/bin/bash
# Train per-token reference probe via torchrun (DDP).
#
# Usage:
#   sbatch --gpus=4 pipeline/train_token.sh [--name probe_token_v1 ...]

#SBATCH --job-name=train_token
#SBATCH --gpus=4
#SBATCH --time=08:00:00
#SBATCH --output=outputs/train_token_%j.out
#SBATCH --error=outputs/train_token_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

NGPUS="${SLURM_GPUS_ON_NODE:-1}"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR" outputs

echo "Job ${SLURM_JOB_ID}: ${NGPUS} GPUs, MASTER_PORT=${MASTER_PORT}"

uv run --frozen torchrun \
    --nproc-per-node="${NGPUS}" \
    pipeline/train_token.py "$@"
