# variant-viewer (EVEE)

Evo Variant Effect Explorer. Svelte 5 + DuckDB web app for visualizing Evo2 ClinVar probe predictions.

## Quick start

```bash
uv sync                                          # Python deps
cd frontend && bun install && cd ..              # JS deps (one-time)
uv run vv transform --probe probe_v12 --dev 5000 # raw data → clean parquet
uv run vv build --probe probe_v12 --neighbors    # parquet → DuckDB + neighbors
bash dev.sh                                      # frontend build + serve + tunnel
```

## Pipeline

```
transform → build → serve
```

- **transform.py**: scores.feather + variants.parquet → `builds/clean.parquet` (renames, quality filter, z-scores, LR, display strings, 2D heatmaps, 1D histograms)
- **build.py**: clean.parquet → `builds/variants.duckdb` (flat table + indexes + optional neighbors/UMAP)
- **serve.py**: DuckDB → flat JSON API. No reconstruction, no nesting. Frontend derives structure from column naming conventions + heads.json.

## CLI

```bash
uv run vv transform --probe probe_v12 [--dev N]     # raw → clean parquet
uv run vv build --probe probe_v12 [--neighbors] [--umap]  # parquet → DuckDB
uv run vv serve [--port 8501]                        # serve API + frontend
uv run vv eval $ACTS/probe_v12                       # per-head metrics → eval.json
uv run vv log-eval $ACTS/probe_v12                   # upload to wandb
uv run vv pipeline $ACTS/probe_v12                   # full SLURM chain
bash dev.sh --data --dev 5000 --probe probe_v12 --neighbors  # one-click dev rebuild
```

## Key files

```
cli.py                   Typer CLI entry point
transform.py             Raw data → clean parquet (the big transform)
build.py                 Clean parquet → DuckDB (insert + indexes + GPU steps)
serve.py                 Starlette API server (flat rows, no logic)
db.py                    DuckDB schema + helpers
heads.json               Head vocabulary: display names, groups, predictor config, quality thresholds
head_quality.json        Quality-passing head list (from eval pipeline)
prompts.py               Claude interpretation prompt builder
constants.py             Probe name, calibration, consequence/AA classes

frontend/                Svelte 5 + Vite + Bun + ECharts
  src/App.svelte         Router (hash-based: #/ and #/variant/{id})
  src/lib/api.ts         API calls + normalizeVariant (flat row → typed Variant)
  src/lib/types.ts       TypeScript interfaces
  src/components/        All UI components (cards, rows, charts)

pipeline/                SLURM training/extraction pipeline
  train.py               Probe training (DDP, focal loss)
  extract.py             3-view scoring → scores.feather
  eval.py                Per-head metrics → eval.json
```

## Architecture

**Server is a dumb pipe.** `SELECT * FROM variants WHERE variant_id = ?` → flat JSON dict. No nesting, no grouping, no reconstruction.

**Frontend derives everything.** `normalizeVariant()` in `api.ts` groups `ref_score_*`/`var_score_*` into `disruption`, `score_*` into `effect`, `gt_*` into `gt` using column naming conventions. Attribution is computed client-side (sigmoid-gated z-scores). Head metadata (display names, groups, predictor config, eval metrics) comes from `/api/global`.

**Column naming IS the schema:**
- `ref_score_{head}` / `var_score_{head}` → disruption heads
- `score_{head}` → effect heads
- `gt_{head}` → ground truth (database) values
- `z_{head}` / `ref_lr_{head}` / `var_lr_{head}` / `lr_{head}` → precomputed stats

## Data

**Local** (`data/` → `~/projects/data/`, symlinked):
- `variants.parquet` (4.2M × 687 cols, ~925MB)
- `heads.json` (646 head definitions)

**Artifacts** (`$VV_ARTIFACTS` or `paths.py:ARTIFACTS`):
- `clinvar_evo2_deconfounded_full/{probe}/scores.feather` (184K × 1201 cols)
- `clinvar_evo2_deconfounded_full/{probe}/embeddings/` (for neighbors/UMAP)
- `clinvar_evo2_vus/{probe}/scores.feather` (optional)

**Build outputs** (`builds/`):
- `clean.parquet` — transformed data ready for DuckDB
- `variants.duckdb` — serving database (~750MB for 232K variants)
- `heads.json` — merged head config (vocab + eval + stats)
- `statistics.json` — distributions + heatmaps

## Code conventions

- **polars** not pandas, **torch** not numpy, **uv** not pip, **bun** not npm
- **orjson** for fast JSON, **DuckDB** for serving
- **ECharts** for all charts (heatmaps, histograms), no chart.js
- **Svelte 5 runes**: `$state`, `$derived`, `$derived.by`, `$effect`, `$props`
- Probe version is a CLI flag, not hardcoded. Current: `probe_v12`
- `.env` for ANTHROPIC_API_KEY (gitignored)
- `heads.json` is the single source of truth for head display, grouping, predictor config, quality thresholds

## Current probe

`probe_v12` (d_hidden=64, d_probe=128, focal_gamma=0.0, 1 epoch)

## wandb

Project: `gfm-probes`. Each probe version has eval metrics:
```bash
uv run vv log-eval $ACTS/probe_v12
```
