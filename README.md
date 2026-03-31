# Variant Effect Viewer

Interactive browser for Evo2 ClinVar probe predictions. Visualizes pathogenicity
scores, disruption/effect profiles, UMAP embeddings, nearest neighbors, AI
interpretations, and VEP annotations for ~4.2M ClinVar variants.

## Quick start

```bash
uv sync
uv run vv check                              # validate all data is present
uv run vv build                              # build static site (~90s)
uv run vv serve /tmp/variant_viewer_*        # serve with on-demand interpretation
```

## Data

All variant data lives in two files (produced by the annotator's `export_for_viewer.py`):

```
data/
├── variants.parquet     # 4.2M ClinVar variants, 687 cols, ~925 MB
├── labeled.parquet      # 1.53M pathogenic/benign (stars ≥ 1), subset of variants.parquet
├── unlabeled.parquet    # 2.71M VUS/conflicting/other, subset of variants.parquet
└── heads.json           # 646 probe head definitions (specs + display names + categories)
```

Each parquet contains ALL data pre-joined: variant metadata, gene info, VEP annotations
(HGVS, domains, impact), gnomAD frequencies, ClinVar submissions (ACMG, submitters),
LOEUF constraint, and 646 training label columns. No symlinks, no external data deps.

**To regenerate** (from the annotator repo):
```bash
cd ~/projects/annotator
uv run python scripts/export_for_viewer.py \
    --output ~/projects/variant-viewer/data \
    --variant-ann-dir data/annotations_output/genomics_variants/datasources \
    --vep-cli-dir data
```

**Shared artifacts** (activations + probe outputs, on cluster):
```bash
export VV_ARTIFACTS=/path/to/artifacts  # where clinvar_evo2_deconfounded_full/ lives
```

```
$VV_ARTIFACTS/
├── clinvar_evo2_deconfounded_full/
│   ├── activations/           # Raw Evo2 activations (2.2TB)
│   └── probe_v12/             # Trained probe outputs
│       ├── weights.pt
│       ├── config.json
│       ├── scores.feather     # Per-variant scores (from extract)
│       ├── embeddings/        # Covariance embeddings (for UMAP + neighbors)
│       ├── split.feather
│       └── eval.json
└── clinvar_evo2_vus/          # [optional] VUS activations
```

## CLI

```bash
uv run vv build                              # build static site to /tmp
uv run vv build --umap --neighbors           # build with UMAP + neighbors (needs GPU)
uv run vv serve /tmp/variant_viewer_*        # serve with AI interpretation
uv run vv eval $ACTS/probe_v12               # compute per-head metrics → eval.json
uv run vv log-eval $ACTS/probe_v12           # upload eval.json to wandb
uv run vv pipeline $ACTS/probe_v12           # full chain: extract → eval → build
```

## Pipeline

```
train → extract → finalize → eval → log-eval → build → serve
```

Single command (submits SLURM jobs with dependency chains):
```bash
uv run vv pipeline $VV_ARTIFACTS/clinvar_evo2_deconfounded_full/probe_v12
```

Train a new probe:
```bash
sbatch --gpus=4 pipeline/train.sh --name probe_v12 --focal-gamma 0.5
```

## Probe heads

646 heads in two categories (defined in `data/heads.json`):

- **Disruption** (443): Scored on ref view. Delta (var - ref) shows what the mutation
  disrupted. Examples: helix structure, domain membership, ChIP-seq peaks, conservation.

- **Effect** (203): Scored on diff view (var - ref activations). Predict variant-level
  properties. Examples: pathogenicity, CADD, AlphaMissense, SpliceAI, Pfam domains.

## Architecture

| File | Role |
|------|------|
| `cli.py` | Typer CLI entry point (`uv run vv <cmd>`) |
| `build.py` | Static site generator: variants.parquet + scores → per-variant JSONs |
| `serve.py` | Dev server with on-demand Claude interpretation |
| `index.html` | Single-page frontend (vanilla JS + CSS) |
| `attribution.py` | Z-score disruption attribution |
| `prompts.py` | Claude interpretation prompt builder |
| `display.py` | Head display names, curation, quality filtering |
| `constants.py` | Probe name, model config, consequence classes, calibration |
| `loaders.py` | `load_variants()` + `load_heads()` |
| `paths.py` | `VV_ARTIFACTS` env var, path constants |
| `training.py` | `gene_split()`, `load_head_specs()` |
| `probe/covariance.py` | MultiHeadCovProbeV2 + focal multihead loss |
| `pipeline/train.py` | Dual-pass probe training (DDP, focal loss) |
| `pipeline/extract.py` | 3-view scoring: activations → embeddings + scores |
| `pipeline/eval.py` | Per-head eval: AUC, correlation → eval.json |
| `pipeline/interpret.py` | Batch Claude interpretation |

## Transfer to new server

```bash
# Copy data files (only ~1GB)
scp data/{labeled,unlabeled,variants}.parquet data/heads.json newserver:variant-viewer/data/

# Clone goodfire-core alongside
git clone goodfire-core ../goodfire-core

# Set artifacts path and run
export VV_ARTIFACTS=/path/to/activations
uv sync
uv run vv pipeline $VV_ARTIFACTS/probe_v12
```

## wandb

Every probe version has eval metrics in wandb (`gfm-probes` project):
```bash
uv run vv log-eval $VV_ARTIFACTS/clinvar_evo2_deconfounded_full/probe_v12
```
