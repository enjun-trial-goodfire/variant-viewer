#!/bin/bash
# End-to-end pipeline: extract → finalize → eval → build webapp.
#
# Usage:
#   bash scripts/pipeline.sh /path/to/probe [--labeled-only]
#
# Example:
#   bash scripts/pipeline.sh $ACTS/probe_v8
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

# ── Step 1: Extract (3-view embeddings + scores) ─────────────────────
echo "Step 1: Extract..."
EXT_L=$(sbatch --parsable --array=0-7 scripts/extract.sh --probe "$PROBE" --activations "$ACTS")
echo "  Labeled extract: ${EXT_L}"

if [ -z "$LABELED_ONLY" ]; then
    EXT_V=$(sbatch --parsable --array=0-7 scripts/extract.sh --probe "$PROBE" --activations "$VUS")
    echo "  VUS extract: ${EXT_V}"
fi

# ── Step 2: Finalize (merge shards + index embeddings) ───────────────
echo "Step 2: Finalize..."
FIN_L=$(sbatch --parsable --dependency=afterok:${EXT_L} scripts/finalize_embed.sh "$ACTS/${PROBE_NAME}")
echo "  Labeled finalize: ${FIN_L} (after ${EXT_L})"

if [ -z "$LABELED_ONLY" ]; then
    FIN_V=$(sbatch --parsable --dependency=afterok:${EXT_V} scripts/finalize_embed.sh "$VUS/${PROBE_NAME}")
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
import json, sys, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from pathlib import Path
sys.path.insert(0, '.')
from src.datasets import clinvar
from src.datasets.clinvar.main import CONSEQUENCE_CLASSES

ARTS = Path('${ACTS}')
probe_dir = ARTS / '${PROBE_NAME}'
scores = pl.read_ipc(str(probe_dir / 'scores.feather'))
split = pl.read_ipc(str(probe_dir / 'split.feather'))
test_ids = set(split.filter(pl.col('split') == 'test')['variant_id'].to_list())
gt = clinvar.annotations('deconfounded-full')
meta = clinvar.metadata('deconfounded-full').select('variant_id', 'label')
df = scores.join(gt, on='variant_id', how='left').join(meta, on='variant_id', how='left').filter(pl.col('variant_id').is_in(list(test_ids)))

eval_results = {}
for col in [c for c in df.columns if c.startswith('score_') and c != 'score_pathogenic']:
    head = col[6:]
    if head not in gt.columns: continue
    s = df[col].to_numpy().astype(np.float64)
    g = df[head].to_numpy().astype(np.float64)
    valid = ~(np.isnan(s) | np.isnan(g))
    if valid.sum() < 20: continue
    unique = np.unique(g[valid])
    if len(unique) == 2:
        eval_results[head] = {'kind': 'binary', 'auc': round(float(roc_auc_score(g[valid], s[valid])), 4), 'n': int(valid.sum())}
    else:
        r, _ = pearsonr(s[valid], g[valid])
        eval_results[head] = {'kind': 'continuous', 'correlation': round(float(r), 4), 'n': int(valid.sum())}

df_path = df.filter(pl.col('label').is_in(['benign', 'pathogenic']))
y_true = (df_path['label'] == 'pathogenic').to_numpy().astype(int)
y_score = df_path['score_pathogenic'].to_numpy()
valid = ~np.isnan(y_score)
eval_results['pathogenic'] = {'kind': 'binary', 'auc': round(float(roc_auc_score(y_true[valid], y_score[valid])), 4), 'n': int(valid.sum())}

(probe_dir / 'eval.json').write_text(json.dumps(eval_results, indent=2))
print(f'eval.json: {len(eval_results)} heads, pathogenicity AUC={eval_results[\"pathogenic\"][\"auc\"]}')
\" && echo 'Eval done, building webapp...' && uv run python webapp/build.py --no-sync")

echo "  Eval + build: ${EVAL_BUILD} (after ${WAIT_FOR})"
echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$(whoami) --format='%.10i %.20j %.8T %.10M'"
