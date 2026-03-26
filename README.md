# Variant Effect Viewer

Interactive browser for Evo2 ClinVar probe predictions. Visualizes pathogenicity scores, per-head disruption/effect profiles, UMAP embeddings, nearest neighbors, and VEP annotations for ~230K ClinVar variants.

## Quick start

```bash
uv sync
uv run python build.py           # builds to webapp/build/ (~2 min on H200)
python -m http.server -d webapp/build/ 8080
```

## Pipeline

The viewer consumes artifacts produced by the probe pipeline:

```
harvest          Extract Evo2 bidir activations (SLURM array)
    |
extract          Stream activations through probe -> embeddings + scores
finalize         Merge shard scores into scores.feather
    |
eval             AUROC/AUPRC with bootstrap CIs
    |
build.py         Generate static site (global.json, search.json, per-variant JSONs)
```

Run the full extract-finalize-eval chain: `bash probe/pipeline.sh /path/to/probe`

## Architecture

| File | Role |
|------|------|
| `build.py` | Static site generator: loads scores + metadata, computes UMAP + neighbors, writes JSON |
| `display.py` | Head display names and category grouping (standalone, no deps) |
| `streaming.py` | Streaming iteration over chunked activation datasets |
| `index.html` | Single-page frontend (vanilla JS, no build step) |
| `probe/` | Probe weights, extraction scripts, eval |
| `pipeline/` | Production pipeline (frozen) |
| `SCHEMA.md` | Data contract between build.py and the frontend |

## Data requirements

Expects probe artifacts under the shared artifact directory:
- `clinvar_evo2_deconfounded_full/{probe}/scores.feather` — labeled variant scores
- `clinvar_evo2_vus/{probe}/scores.feather` — VUS scores
- `clinvar_evo2_deconfounded_full/{probe}/embeddings/` — probe embeddings (chunked)
- `clinvar_evo2_deconfounded_full/{probe}/config.json` — probe head definitions
- `clinvar_evo2_labeled/variant_annotations/` — VEP annotations (per-chromosome parquet)

See [SCHEMA.md](SCHEMA.md) for the full data contract.
