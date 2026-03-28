#!/bin/bash
# End-to-end pipeline: extract → finalize → eval → build webapp.
#
# Usage:
#   bash scripts/pipeline.sh /path/to/probe [--labeled-only]
#
# Example:
#   bash scripts/pipeline.sh $ACTS/probe_v9
#
# This chains SLURM jobs with dependencies so the full pipeline runs unattended.
# The final output is a webapp build in /tmp ready to serve.

set -euo pipefail

PROBE="${1:?Usage: pipeline.sh <probe-dir> [--labeled-only]}"
LABELED_ONLY="${2:-}"

ACTS=$(dirname "$PROBE")
VUS="/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian/clinvar_evo2_vus"
PROBE_NAME=$(basename "$PROBE")
cd "$(dirname "$0")/.."

echo "=== Pipeline: ${PROBE_NAME} ==="
echo "Probe:   ${PROBE}"
echo "Labeled: ${ACTS}"
echo "VUS:     ${VUS}"
echo ""

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

# ── Step 3: Eval + webapp build (after all finalize jobs) ─────────────
echo "Step 3: Eval + build..."
EVAL_BUILD=$(sbatch --parsable --dependency=afterok:${WAIT_FOR} \
    --job-name=eval-build --gpus=1 --time=02:00:00 \
    --output=outputs/eval_build_%j.out \
    --wrap="cd \${SLURM_SUBMIT_DIR} && uv run python -c \"
import json, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from pathlib import Path

probe_dir = Path('${ACTS}/${PROBE_NAME}')
scores = pl.read_ipc(str(probe_dir / 'scores.feather'))
split = pl.read_ipc(str(probe_dir / 'split.feather'))
test_ids = set(split.filter(pl.col('split') == 'test')['variant_id'].to_list())

gt = pl.read_ipc('data/clinvar/deconfounded-full/annotations.feather')
meta = pl.read_ipc('data/clinvar/deconfounded-full/metadata.feather').select('variant_id', 'label')
df = scores.join(gt, on='variant_id', how='left').join(meta, on='variant_id', how='left').filter(pl.col('variant_id').is_in(list(test_ids)))

def eval_head(pred, truth):
    valid = ~(np.isnan(pred) | np.isnan(truth))
    if valid.sum() < 20: return None
    p, t = pred[valid], truth[valid]
    if len(np.unique(t)) == 2:
        return {'kind': 'binary', 'auc': round(float(roc_auc_score(t, p)), 4), 'n': int(valid.sum())}
    r, _ = pearsonr(p, t)
    if np.isnan(r): return None
    return {'kind': 'continuous', 'correlation': round(float(r), 4), 'n': int(valid.sum())}

eval_results = {}
for col in [c for c in df.columns if c.startswith('score_') and c != 'score_pathogenic']:
    head = col[6:]
    if head not in gt.columns: continue
    r = eval_head(df[col].to_numpy().astype(np.float64), df[head].to_numpy().astype(np.float64))
    if r: eval_results[head] = r
for col in [c for c in df.columns if c.startswith('ref_score_')]:
    head = col[10:]
    if head not in gt.columns or head in eval_results: continue
    r = eval_head(df[col].to_numpy().astype(np.float64), df[head].to_numpy().astype(np.float64))
    if r: eval_results[head] = r

df_path = df.filter(pl.col('label').is_in(['benign', 'pathogenic']))
y_true = (df_path['label'] == 'pathogenic').to_numpy().astype(int)
y_score = df_path['score_pathogenic'].to_numpy()
valid = ~np.isnan(y_score)
eval_results['pathogenic'] = {'kind': 'binary', 'auc': round(float(roc_auc_score(y_true[valid], y_score[valid])), 4), 'n': int(valid.sum())}

(probe_dir / 'eval.json').write_text(json.dumps(eval_results, indent=2))
print(f'eval.json: {len(eval_results)} heads, pathogenicity AUC={eval_results[\"pathogenic\"][\"auc\"]}')
\" && echo 'Eval done, building webapp...' && uv run python build.py")

echo "  Eval + build: ${EVAL_BUILD} (after ${WAIT_FOR})"
echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$(whoami) --format='%.10i %.20j %.8T %.10M'"
