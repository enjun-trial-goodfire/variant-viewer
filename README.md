# EVEE — Evo Variant Effect Explorer

Interactive browser for Evo2 ClinVar probe predictions. Visualizes pathogenicity
scores, disruption/effect profiles, UMAP embeddings, nearest neighbors, AI
interpretations, and VEP annotations for ~232K ClinVar variants.

## Quick start

```bash
# Backend (Python)
uv sync

# Frontend (Svelte + Vite)
curl -fsSL https://bun.sh/install | bash     # one-time: install bun
cd frontend && bun install && cd ..           # one-time: install JS deps

# Build data
uv run vv transform --probe probe_v12        # raw → clean parquet (~2min)
uv run vv build --probe probe_v12 --neighbors # parquet → DuckDB (~50s)

# Serve
bash dev.sh                                   # build frontend + serve + tunnel
```

## Development

```bash
# Quick iteration (5K variant subset)
bash dev.sh --data --dev 5000 --probe probe_v12 --neighbors

# Frontend only (data already built)
bash dev.sh

# Two terminals for hot reload
uv run vv serve --port 8501                   # terminal 1: API
cd frontend && bun run dev                    # terminal 2: Vite HMR on :5173
```

## Pipeline

```
scores.feather + variants.parquet
  → transform (rename, filter, z-scores, stats)
  → builds/clean.parquet
  → build (DuckDB insert, neighbors, UMAP)
  → builds/variants.duckdb
  → serve (flat JSON API + static frontend)
```

### CLI

```bash
uv run vv transform --probe probe_v12 [--dev N]          # step 1
uv run vv build --probe probe_v12 [--neighbors] [--umap] # step 2
uv run vv serve [--port 8501]                             # step 3
uv run vv eval $ACTS/probe_v12                            # eval metrics
uv run vv log-eval $ACTS/probe_v12                        # upload to wandb
uv run vv pipeline $ACTS/probe_v12                        # full SLURM chain
```

## Architecture

| File | Role |
|------|------|
| `transform.py` | Raw data → clean parquet (the single transform step) |
| `build.py` | Clean parquet → DuckDB (flat insert + GPU neighbors/UMAP) |
| `serve.py` | DuckDB → flat JSON API (no reconstruction, no nesting) |
| `db.py` | DuckDB schema + connection helpers |
| `heads.json` | Head vocabulary: display names, groups, predictor config |
| `head_quality.json` | Quality-filtered head list |
| `prompts.py` | Claude interpretation prompt builder |
| `frontend/` | Svelte 5 + Vite + ECharts |

### Frontend components

```
frontend/src/
  App.svelte                  # Hash router (#/ landing, #/variant/{id})
  components/
    Header.svelte             # Search + branding
    LandingPage.svelte        # UMAP scatter
    VariantPage.svelte        # Orchestrates variant cards
    DisruptionRow.svelte      # Disruption bar + expandable heatmap
    EffectRow.svelte           # Effect bar + expandable histogram
    HeadHeatmap.svelte        # ECharts 2D ref×var heatmap
    HeadHistogram.svelte      # ECharts 1D distribution histogram
    cards/
      VerdictCard.svelte      # Gene, coords, score, ClinVar metadata
      InterpretationCard.svelte  # AI interpretation
      DisruptionCard.svelte   # Top disruptions + all disruptions
      PredictorsCard.svelte   # Computational predictors (10 tools)
      NeighborsCard.svelte    # Nearest neighbors table
      PopulationCard.svelte   # gnomAD population frequencies
  lib/
    api.ts                    # Fetch + normalizeVariant (flat → typed)
    types.ts                  # TypeScript interfaces
    colors.ts                 # Color utilities
    helpers.ts                # Display helpers
```

## Data

All variant data in two files (from the annotator repo):

```
data/
├── variants.parquet     # 4.2M ClinVar variants, 687 cols, ~925 MB
└── heads.json           # 646 probe head definitions
```

Shared artifacts (on cluster):
```
$VV_ARTIFACTS/
├── clinvar_evo2_deconfounded_full/
│   └── probe_v12/
│       ├── scores.feather     # 184K × 1201 cols
│       ├── embeddings/        # for neighbors + UMAP
│       ├── config.json
│       └── eval.json
└── clinvar_evo2_vus/          # optional VUS scores
```

## Deployment

### Local (DuckDB)
```bash
bash dev.sh --data --probe probe_v12 --neighbors
```

### AWS (DynamoDB + CloudFront)
See `terraform/` for infrastructure. Frontend detects `API_BASE` via `config.js`.

## Performance

| Metric | Value |
|--------|-------|
| Transform | ~2 min (232K variants) |
| Build + neighbors | ~50s (GPU) |
| DB size | 746 MB |
| Variant lookup | ~30ms |
| Gene search | ~8ms |
| Global config | ~7ms |

## wandb

```bash
uv run vv log-eval $VV_ARTIFACTS/clinvar_evo2_deconfounded_full/probe_v12
```
