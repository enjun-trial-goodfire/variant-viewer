# Variant Effect Viewer

Interactive browser for Evo2 ClinVar probe predictions. Visualizes pathogenicity
scores, disruption/effect profiles, UMAP embeddings, nearest neighbors, AI
interpretations, and VEP annotations for ~230K ClinVar variants.

## Quick start

```bash
uv sync
uv run vv check                              # validate all data is present
uv run vv build --probe probe_v10            # build static site (~90s)
uv run vv serve /tmp/variant_viewer_*        # serve with on-demand interpretation
```

For public access on the cluster:
```bash
source /mnt/polished-lake/scripts/config.sh
$SCRIPTS_DIR/bin/tunnel-url 8501
```

## CLI

```bash
uv run vv train pretrain-cmd --gpus 4        # submit probe training to SLURM
uv run vv extract --probe $P --activations $A  # submit extraction (8 GPU shards)
uv run vv eval $ACTS/probe_v10               # compute per-head metrics → eval.json
uv run vv log-eval $ACTS/probe_v10           # upload eval.json to wandb
uv run vv build --probe probe_v10            # build static site to /tmp
uv run vv build --probe probe_v10 --umap     # build with UMAP embedding
uv run vv serve /tmp/variant_viewer_*        # serve with AI interpretation
uv run vv pipeline $ACTS/probe_v10           # full chain: extract → eval → build
```

## Data setup

The build requires **local data files** (symlinked) and **shared probe artifacts**.

**Local data** (`data/` → `~/projects/data/`, not in git):
```
data/
├── clinvar/
│   ├── deconfounded-full/
│   │   ├── metadata.feather         # ClinVar labeled variant metadata
│   │   └── annotations.feather      # Ground truth annotation labels (700 heads)
│   ├── vus/
│   │   └── metadata.feather         # VUS variant metadata
│   └── submissions.feather          # [optional] ClinVar submission details
└── gencode/
    └── genes.feather                # GENCODE gene coordinates
```

**Shared artifacts** (on cluster filesystem):
```
$ARTIFACTS/clinvar_evo2_deconfounded_full/
├── activations/                     # Raw Evo2 activations (2.2TB)
└── probe_v10/
    ├── weights.pt                   # Trained probe checkpoint
    ├── config.json                  # Head definitions (disruption/effect heads)
    ├── scores.feather               # 3-view scores (ref_score_*, var_score_*, score_*)
    ├── embeddings/                  # Covariance embeddings (for UMAP + neighbors)
    ├── split.feather                # Train/test gene-level split
    └── eval.json                    # Per-head eval metrics (AUC, correlation)
```

## Pipeline

```
train → extract → finalize → eval → log-eval → build → serve
```

Run the full chain:
```bash
bash pipeline/pipeline.sh $ACTS/probe_v10
```

Or step by step:
```bash
# 1. Train (4 GPU DDP)
sbatch --gpus=4 pipeline/train.sh pretrain-cmd
sbatch --gpus=4 pipeline/train.sh finetune-cmd --checkpoint $ACTS/probe_v10/pretrain.pt

# 2. Extract scores (8 parallel GPU shards)
uv run vv extract --probe $ACTS/probe_v10 --activations $ACTS

# 3. Eval + upload to wandb
uv run vv eval $ACTS/probe_v10
uv run vv log-eval $ACTS/probe_v10

# 4. Build + serve
uv run vv build --probe probe_v10 --umap
uv run vv serve /tmp/variant_viewer_*
```

## Dependency graph

```
variant-viewer
├── goodfire-core          (library: ActivationDataset, storage, optimizers)
├── ~/projects/data/       (annotations + metadata, symlinked into data/)
│   ├── clinvar/deconfounded-full/{metadata,annotations}.feather
│   ├── clinvar/vus/metadata.feather
│   ├── clinvar/submissions.feather
│   └── gencode/genes.feather
└── ARTIFACTS/             (activations + probe checkpoints, read-only)
    └── clinvar_evo2_deconfounded_full/
        ├── activations/   (2.2TB raw Evo2 activations)
        └── probe_v*/      (trained probe weights + config)
```

Two external dependencies, no others:
1. **Annotations** (`~/projects/data/`): Static feather files. Updated by the annotator repo.
2. **Activations** (`ARTIFACTS/`): Raw Evo2 activations + trained probe checkpoints.

## Architecture

| File | Role |
|------|------|
| `cli.py` | Typer CLI entry point (`uv run vv <cmd>`) |
| `build.py` | Static site generator: scores + metadata → per-variant JSONs |
| `serve.py` | Dev server with on-demand Claude interpretation |
| `index.html` | Single-page frontend (vanilla JS + CSS, no build step) |
| `attribution.py` | Z-score attribution with sigmoid-gated delta filtering |
| `training.py` | Training utilities: gene_split, head discovery |
| `prompts.py` | Claude interpretation prompt builder |
| `display.py` | Head display names and category grouping |
| `constants.py` | Consequence classes, calibration |
| `loaders.py` | Metadata and VEP loaders |
| `paths.py` | Artifact path constants, FNV-1a variant ID hashing |
| `probe/covariance.py` | MultiHeadCovProbeV2 + focal multihead loss |
| `probe/binning.py` | Soft-binning for continuous probe heads |
| `pipeline/train.py` | Probe training (pretrain + finetune, DDP) |
| `pipeline/train.sh` | SLURM wrapper for torchrun |
| `pipeline/extract.py` | 3-view scoring: activations → embeddings + scores |
| `pipeline/extract.sh` | SLURM wrapper with artifact validation |
| `pipeline/eval.py` | Per-head eval: AUC, correlation, accuracy → eval.json |
| `pipeline/finalize.sh` | Merge shard scores + index embeddings |
| `pipeline/pipeline.sh` | End-to-end SLURM chain with dependencies |
| `pipeline/interpret.py` | Batch Claude interpretation (cost-guarded) |
| `pipeline/ref_labels.py` | Token label lookup for SAE pretraining |
| `pipeline/serve.sh` | SLURM: build + serve + tunnel in one job |

## Probe heads

Two types of heads (defined in `config.json`):

- **Disruption heads** (~500): Scored on reference and variant views separately.
  Delta (var - ref) shows what the mutation disrupted. Examples: helix, domain, ChIP-seq, conservation.

- **Effect heads** (~200): Scored on diff view (var - ref activations).
  Predict variant-level properties. Examples: pathogenicity, CADD, AlphaMissense, SpliceAI.

## wandb integration

Every probe version has eval metrics in wandb (`gfm-probes` project):
```bash
uv run vv log-eval $ACTS/probe_v10    # uploads per-head table + summary scalars
```

Logged per run: `eval/heads` table (AUC/correlation per head), `pathogenic_auc`, `mean_auc`, `mean_correlation`. Use wandb's run comparer to scatter v9 vs v10 per-head metrics.

Training logs per-step: `loss`, `loss_diff`, `loss_ref`, `grad_norm`, `batch_size`.
