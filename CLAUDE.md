# variant-viewer

Interactive variant effect viewer for Evo2 ClinVar probes. Extracted from
[gfm_gen](https://github.com/goodfire-ai/gfm_gen).

## Build & serve

```bash
uv sync
bash preflight.sh               # validate data is present
uv run python build.py          # builds to /tmp with --no-sync, or webapp/build/
python -m http.server -d webapp/build/ 8080
```

For SLURM: `sbatch pipeline/serve.sh` (builds + serves + tunnels).

## Pipeline

```
[gfm_gen]     harvest → train
                  ↓
[this repo]   extract → finalize → eval → build → serve
              pipeline/  pipeline/         build.py  serve.py
```

- `pipeline/extract.py` — 3-view scoring: diff/ref/var activations → embeddings + scores
- `pipeline/finalize.sh` — merge shard scores, index embeddings
- `pipeline/pipeline.sh` — end-to-end chain with SLURM dependencies
- `build.py` — static site generator: scores + metadata → JSON files
- `serve.py` — development server with live Claude interpretation

## Probe heads

Two types, defined in `config.json`:
- **disruption_heads** (~500): scored on ref and var views separately (delta = disruption)
- **effect_heads** (~200): scored on diff view (variant-level predictions)

Older probes use `ref_heads`/`diff_heads` — build.py handles both.

## File layout

```
build.py                 Static site generator
display.py               Head display names + category grouping
attribution.py           Ridge attribution model (per-variant feature importance)
serve.py                 Development server (static files + Claude API)
clinvar_submissions.py   ClinVar XML → submissions.feather (run once)
preflight.sh             Validate all inputs before building
index.html               Single-page frontend (vanilla JS)
SCHEMA.md                JSON data contract (global.json, variants/*.json, search.json)

probe/
  covariance.py          MultiHeadCovProbeV2 (architecture + checkpoint loading)
  binning.py             Soft-binning for continuous heads

pipeline/
  extract.py             3-view scoring (SLURM array, GPU)
  extract.sh             SLURM wrapper for extract.py
  finalize.sh            Merge shards + index embeddings
  interpret.py           Claude API variant interpretation
  pipeline.sh            End-to-end chain (extract → finalize → eval → build)
  serve.sh               SLURM: build + serve + tunnel
```

## Data

**Local** (`data/`, not in git — symlink to gfm_gen/data on cluster):
- `data/clinvar/deconfounded-full/metadata.feather` — labeled metadata
- `data/clinvar/deconfounded-full/annotations.feather` — ground truth
- `data/clinvar/vus/metadata.feather` — VUS metadata
- `data/gencode/genes.feather` — gene coordinates

**Shared artifacts** (hardcoded in build.py to `/mnt/polished-lake/artifacts/...`):
- `clinvar_evo2_deconfounded_full/{probe}/scores.feather`
- `clinvar_evo2_deconfounded_full/{probe}/embeddings/`
- `clinvar_evo2_vus/{probe}/scores.feather`
- `clinvar_evo2_labeled/variant_annotations/variant_annotations_chr*.parquet`

Run `bash preflight.sh probe_v9` to check everything exists.

## Code conventions

- **polars** not pandas (`pl.read_ipc`, `.filter()`, `.join()`)
- **torch** not numpy (convert at API boundaries only)
- **uv** not pip
- **orjson** for JSON serialization (speed matters: 232K variant files)
- See `SCHEMA.md` for the JSON data contract
- Hardcoded paths are in build.py constants — update when artifacts move
- Current probe: `probe_v9` (`d_hidden=32, d_probe=128, focal_gamma=3.0`)

## Known limitations

- All artifact paths are hardcoded (not yet configurable via CLI/env)
- `data/` must be symlinked or populated manually
- `goodfire-core` is an editable dependency via `../goodfire-core`
- Pipeline scripts assume SLURM cluster with GPUs
