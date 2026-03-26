# variant-viewer

Standalone variant effect viewer for Evo2 ClinVar probes. Extracted from [gfm_gen](https://github.com/tdooms-goodfire/gfm_gen).

## Build & serve

```bash
uv sync
uv run python build.py [output_dir]          # default: webapp/build/
python -m http.server -d webapp/build/ 8080   # open http://localhost:8080
```

## Pipeline

```
harvest (gfm_gen)  ->  extract  ->  finalize  ->  eval  ->  build
                       probe/       probe/        probe/     build.py
```

- `probe/extract.sh` — stream activations through trained probe, write embeddings + scores
- `probe/finalize.sh` — merge shard scores into `scores.feather`
- `build.py` — generate static site JSONs (global, search, per-variant)
- `streaming.py` — shared iteration utilities for activation datasets
- `display.py` — head display names and category grouping (no external deps)

## Code conventions

- **polars** not pandas (`pl.read_ipc`, `.filter()`, `.join()`)
- **torch** not numpy (convert at API boundaries only)
- **uv** not pip
- **plotly** not matplotlib
- Use `orjson` for JSON serialization
- See `SCHEMA.md` for the data contract between `build.py` and the frontend

## File structure

```
build.py          Static site generator (scores -> JSON)
display.py        Head display names and grouping constants
streaming.py      Activation dataset streaming utilities
index.html        Single-page frontend (vanilla JS)
SCHEMA.md         Data contract (global.json, variant JSON, search.json)
probe/            Probe weights, extract/finalize/eval scripts
pipeline/         Production pipeline scripts (FROZEN, do not modify)
```

## Artifacts

Expects artifacts at:
```
/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian/
  clinvar_evo2_deconfounded_full/   Labeled variants (benign + pathogenic)
  clinvar_evo2_vus/                 VUS variants
```
