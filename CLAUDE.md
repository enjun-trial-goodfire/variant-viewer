# variant-viewer

Full pipeline for Evo2 ClinVar variant effect prediction: train, extract, eval, build, serve.

## CLI

```bash
uv run vv train pretrain-cmd --gpus 4        # probe training (SLURM + torchrun)
uv run vv extract --probe $P --activations $A  # extract scores (8 GPU shards)
uv run vv eval $ACTS/probe_v10               # per-head eval → eval.json
uv run vv log-eval $ACTS/probe_v10           # upload to wandb
uv run vv build --probe probe_v10            # build static site
uv run vv serve /tmp/variant_viewer_*        # serve with AI interpretation
uv run vv pipeline $ACTS/probe_v10           # full chain
```

## Pipeline

```
train → extract → finalize → eval → log-eval → build → serve
```

All steps are standalone scripts chainable via SLURM `--dependency=afterok`.
`pipeline/pipeline.sh` chains them automatically.

## Probe heads

Two types, defined in `config.json`:
- **disruption_heads** (~500): scored on ref and var views separately (delta = disruption)
- **effect_heads** (~200): scored on diff view (variant-level predictions)

Older probes use `ref_heads`/`diff_heads` — code handles both.

## Key files

```
cli.py                   Typer CLI: uv run vv <cmd>
build.py                 Static site generator (--probe flag selects version)
serve.py                 Dev server + Claude interpretation (loads .env for API key)
index.html               Single-page frontend (vanilla JS, no build step)
attribution.py           Z-score attribution with sigmoid-gated delta filtering
training.py              Gene split, head discovery, head classification
prompts.py               Claude interpretation prompt builder
probe/covariance.py      MultiHeadCovProbeV2 + focal multihead loss
pipeline/train.py        Probe training (pretrain + finetune phases, DDP)
pipeline/train.sh        SLURM wrapper (MASTER_PORT, WANDB_DIR, --frozen)
pipeline/extract.py      3-view scoring with artifact validation
pipeline/eval.py         Per-head metrics: AUC, correlation, accuracy
pipeline/pipeline.sh     End-to-end SLURM chain
pipeline/interpret.py    Batch Claude interpretation (cost-guarded >20 variants)
pipeline/ref_labels.py   Token label lookup for SAE pretraining
```

## Data

**Local** (`data/` → `~/projects/data/`, symlinked, not in git):
- `clinvar/deconfounded-full/{metadata,annotations}.feather`
- `clinvar/vus/metadata.feather`
- `clinvar/submissions.feather`
- `gencode/genes.feather`

**Shared artifacts** (`paths.py:ARTIFACTS`):
- `clinvar_evo2_deconfounded_full/{probe}/scores.feather`
- `clinvar_evo2_deconfounded_full/{probe}/embeddings/`
- `clinvar_evo2_vus/{probe}/scores.feather` (optional, build works without)

## Code conventions

- **polars** not pandas, **torch** not numpy, **uv** not pip
- **orjson** for JSON (speed matters: 232K variant files)
- **typer** for all CLIs, **wandb** for experiment tracking
- Probe version is a CLI flag (`--probe probe_v10`), not hardcoded
- `.env` file for ANTHROPIC_API_KEY (gitignored)
- SLURM scripts: `set -euo pipefail`, `cd ${SLURM_SUBMIT_DIR}`, `uv run --frozen`
- Current probe: `probe_v10` (`d_hidden=64, d_probe=128, focal_gamma=3.0`)

## wandb

Project: `gfm-probes`. Each probe version has eval metrics:
```bash
uv run vv log-eval $ACTS/probe_v10   # per-head table + summary scalars
```
Training logs: `loss`, `loss_diff`, `loss_ref`, `grad_norm` per step.
