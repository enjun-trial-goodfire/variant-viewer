#!/bin/bash
# Finalize partitioned embedding output + merge shard scores.
#
# Usage:
#   EXT=$(sbatch --parsable --array=0-7 pipeline/extract.sh --probe $P --activations $A)
#   sbatch --dependency=afterok:${EXT} pipeline/finalize.sh /path/to/probe-output-dir

#SBATCH --job-name=finalize_embed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/embed/%x_%j.out
#SBATCH --error=logs/embed/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/embed

PROBE_DIR="${1:?Usage: finalize.sh <probe-dir>}"
export PROBE_DIR

echo "=== Finalize Embeddings ==="
echo "Output: ${PROBE_DIR}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

uv run --frozen python -c "
import os
from pathlib import Path
import polars as pl
from goodfire_core.storage import FilesystemStorage
from goodfire_core.storage.finalizer import finalize_partitions

probe_dir = Path(os.environ['PROBE_DIR'])
storage = FilesystemStorage(probe_dir)

# Merge scores FIRST (fast, most important output)
# Merge per-shard score files into one
shard_files = sorted(probe_dir.glob('scores_shard_*.feather'))
if shard_files:
    # Validate all expected shards exist (shard IDs should be contiguous from 0)
    shard_ids = sorted(int(f.stem.split('_')[-1]) for f in shard_files)
    n_expected = max(shard_ids) + 1
    missing = set(range(n_expected)) - set(shard_ids)
    if missing:
        raise RuntimeError(
            f'Missing score shards: {sorted(missing)} (have {len(shard_files)}/{n_expected}). '
            f'Re-run failed embed array jobs before finalizing.'
        )

    print(f'Merging {len(shard_files)} shard score files...')
    scores = pl.concat([pl.read_ipc(f) for f in shard_files], how='diagonal')

    # Validate: no duplicates
    n_unique = scores['variant_id'].n_unique()
    assert n_unique == scores.height, (
        f'Duplicate variant IDs in scores: {scores.height} rows, {n_unique} unique'
    )

    scores_path = probe_dir / 'scores.feather'
    scores.write_ipc(scores_path)
    print(f'  Merged {scores.height:,} scores → {scores_path}')

    # Clean up shard files
    for f in shard_files:
        f.unlink()
    print(f'  Removed {len(shard_files)} shard files')
else:
    print('No shard files found (single-GPU run?)')

# Merge token window score shards (from extract_token_windows.py)
def merge_feather_shards(probe_dir, prefix):
    shard_files = sorted(probe_dir.glob(f'{prefix}_shard_*.feather'))
    if not shard_files:
        return
    shard_ids = sorted(int(f.stem.split('_')[-1]) for f in shard_files)
    n_expected = max(shard_ids) + 1
    missing = set(range(n_expected)) - set(shard_ids)
    if missing:
        raise RuntimeError(
            f'Missing {prefix} shards: {sorted(missing)} (have {len(shard_files)}/{n_expected}). '
            f'Re-run failed array jobs before finalizing.'
        )
    print(f'Merging {len(shard_files)} {prefix} shard files...')
    df = pl.concat([pl.read_ipc(f) for f in shard_files], how='diagonal')
    n_unique = df['variant_id'].n_unique()
    assert n_unique == df.height, (
        f'Duplicate variant IDs in {prefix}: {df.height} rows, {n_unique} unique'
    )
    out_path = probe_dir / f'{prefix}.feather'
    df.write_ipc(out_path)
    print(f'  Merged {df.height:,} rows → {out_path}')
    for f in shard_files:
        f.unlink()
    print(f'  Removed {len(shard_files)} shard files')

merge_feather_shards(probe_dir, 'token_scores')
merge_feather_shards(probe_dir, 'token_window_scores')

# Validate embedding partitions exist for all expected shards
if shard_files:
    emb_dir = probe_dir / 'embeddings'
    missing_parts = [
        i for i in range(n_expected)
        if not (emb_dir / f'partition_{i}').exists()
    ]
    if missing_parts:
        raise RuntimeError(
            f'Missing embedding partitions: {sorted(missing_parts)} '
            f'(expected {n_expected}). Re-run failed embed array jobs before finalizing.'
        )
    print(f'  All {n_expected} embedding partitions present')

# Finalize embedding dataset (slow — merges partitioned binary chunks)
# This is for UMAP/neighbors, not for scores. Scores are already merged above.
if (probe_dir / 'embeddings').exists():
    print('Finalizing embeddings...')
    meta = finalize_partitions(storage, 'embeddings', cleanup=True, build_index=True)
    print(f'  {meta[\"total_items\"]:,} items, {meta[\"num_chunks\"]} chunks')
else:
    print('No embeddings directory (score-only run)')

print('\nDone.')
"

echo ""
echo "End: $(date)"
