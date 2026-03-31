#!/bin/bash
# Train multihead covariance probe via torchrun (DDP).
#
# Usage:
#   sbatch --gpus=4 pipeline/train.sh pretrain-cmd [--d-probe 256 ...]
#   sbatch --gpus=4 pipeline/train.sh finetune-cmd --checkpoint $ACTS/probe_v11/pretrain.pt

#SBATCH --job-name=train_probe
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=outputs/train_%j.out
#SBATCH --error=outputs/train_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

NGPUS="${SLURM_GPUS_ON_NODE:-1}"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR" outputs

echo "Job ${SLURM_JOB_ID}: ${NGPUS} GPUs, MASTER_PORT=${MASTER_PORT}"

uv run --frozen torchrun \
    --nproc-per-node="${NGPUS}" \
    pipeline/train.py "$@"
