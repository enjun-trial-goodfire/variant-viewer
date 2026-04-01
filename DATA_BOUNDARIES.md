# EVEE Data Boundaries

Spec for the DuckDB + Svelte migration. Defines what data exists, where it lives,
what crosses each boundary, and what gets computed where.

## Artifacts

Four serving artifacts, all produced offline by the build pipeline.

### 1. `variants.duckdb` (~50 GB)

One wide table, one row per variant. DuckDB native format with indexes.
The server does indexed lookups and returns flat rows. No computation.

```
CREATE TABLE variants AS SELECT * FROM read_parquet('joined.parquet');
CREATE INDEX idx_id ON variants(id);
CREATE INDEX idx_gene ON variants(gene);
```

**Column families:**

| Family | Prefix | ~Count | Example |
|--------|--------|--------|---------|
| Metadata | (none) | ~50 | `id`, `gene`, `chrom`, `pos`, `ref`, `alt`, `consequence`, `label`, `score`, `gnomad`, `loeuf`, `hgvsc`, `hgvsp`, `domains`, ... |
| Disruption ref | `ref_score_` | ~500 | `ref_score_helix`, `ref_score_phylop_100way` |
| Disruption var | `var_score_` | ~500 | `var_score_helix`, `var_score_phylop_100way` |
| Effect | `score_` | ~200 | `score_cadd_c`, `score_consequence` |
| Ground truth | `gt_` | ~300 | `gt_cadd_c`, `gt_phylop_100way` |
| Neighbors | (special) | 3 | `neighbor_ids` (JSON), `neighbor_meta` (JSON), `neighbor_similarities` (JSON) |

Neighbors are JSON columns because they are variable-length arrays of objects.
They are precomputed (GPU cosine similarity) and nullable (not all variants have them).

**What is NOT stored:**
- Deltas (frontend computes `var_score - ref_score`)
- Z-scores (frontend computes `(delta - mean) / std` using heads.json)
- Attribution rankings (frontend sorts by |z| and deduplicates)
- Interpretations (separate on-demand concern)
- Display names, groups, eval metrics (those live in heads.json)

### 2. `heads.json` (~200 KB)

The complete head vocabulary for the frontend. Produced by the build step
from `head_vocab.json` (config) + `eval.json` (metrics) + `head_stats.json` (population stats).

```json
{
  "helix": {
    "display": "Helix",
    "category": "disruption",
    "group": "Protein",
    "eval": {"metric": "AUC", "value": 0.89},
    "mean": -0.002,
    "std": 0.041,
    "description": "Helix secondary structure probability",
    "quality": "pass"
  },
  "cadd_c": {
    "display": "CADD",
    "category": "effect",
    "group": "Clinical",
    "eval": {"metric": "r", "value": 0.83},
    "description": "Combined score predicting variant deleteriousness",
    "predictor": {"order": 4, "threshold": 0.333},
    "quality": "pass"
  },
  "sift_c": {
    "display": "SIFT",
    "category": "effect",
    "group": "Clinical",
    "predictor": {"order": 7, "threshold": 0.05, "invert": true, "display": "1 − SIFT"},
    "quality": "pass"
  }
}
```

**Fields (per head):**
- `display`: Human-readable name. Auto-derived (strip prefix, title-case) if not in head_vocab.json.
- `category`: `"disruption"` or `"effect"`. From probe config.json.
- `group`: Display group (Protein, Structure, Regulatory, ...). From head_vocab.json `_meta.group_tokens`.
- `eval`: Metric name + value from eval.json. Nullable.
- `mean`, `std`: Population statistics for this head's delta (disruption) or score (effect). From head_stats.json. Null for effect heads without meaningful population reference.
- `description`: One-line description. From head_vocab.json.
- `quality`: `"pass"` or `"fail"` based on eval thresholds in head_vocab.json `_meta.quality_thresholds`.
- `predictor`: Only present for the ~10 heads shown in the Computational Predictors card. Contains `order` (display position), `threshold`, and optionally `invert` and `display` (predictor-specific display name).
- `exclude_from_attribution`: If true, skip when computing top disruptions.
- `exclude_from_effect_expansion`: If true, skip when showing the "All Effects" expansion.

**The frontend reads this once on load and uses it to interpret flat variant rows.**

### 3. `distributions.json` (~1.5 MB)

Pre-aggregated histograms across all variants. Used for bar coloring (likelihood ratio)
and per-head histogram popouts.

```json
{
  "pathogenic": {
    "benign": [120, 340, ...],
    "pathogenic": [5, 12, ...],
    "bins": 80,
    "range": [0.0, 1.0]
  },
  "helix": {
    "ref": {"benign": [...], "pathogenic": [...], "bins": 40, "range": [0.02, 0.98]},
    "delta": {"benign": [...], "pathogenic": [...], "bins": 40, "range": [-0.4, 0.4]}
  },
  "cadd_c": {
    "benign": [...],
    "pathogenic": [...],
    "bins": 40,
    "range": [0.0, 1.0]
  }
}
```

- `"pathogenic"` key: the overall score distribution (special, for the Score Distribution card).
- Disruption heads: `.ref` and `.delta` sub-distributions.
- Effect heads: flat distribution.
- Bin ranges are adaptive (0.5th-99.5th percentile), not fixed [0,1].

### 4. `umap.json` (~2 MB)

30K sampled variants for the landing scatter plot.

```json
{
  "x": [1.23, -0.45, ...],
  "y": [0.78, 2.11, ...],
  "score": [0.92, 0.03, ...],
  "ids": ["chr10:100042514:C:T", ...],
  "genes": [4, 12, ...],
  "labels": [1, 0, ...],
  "gene_list": ["BRCA1", "TP53", ...]
}
```

- `genes` are indexes into `gene_list` (compact encoding).
- `labels`: 0=benign, 1=pathogenic, 2=VUS.

## Server API

The server is a DuckDB query proxy. No computation, no transformation.

### `GET /variant/{id}`

```sql
SELECT * FROM variants WHERE id = ?
```

Returns one flat JSON row (~13-18 KB). All ~1900 columns.
The frontend knows column naming conventions from heads.json.

### `GET /search?q={query}`

```sql
SELECT id, gene, label, score, consequence
FROM variants
WHERE gene = upper(?)
ORDER BY score DESC
LIMIT 30
```

For prefix search: `WHERE gene LIKE upper(?) || '%'`.
No substring search (requires full-text indexing, fights the static design).
For fuzzy gene matching, serve a gene list and do client-side fuzzy match.

### `GET /heads.json`, `GET /distributions.json`, `GET /umap.json`

Static file serving. Cache forever (content-hash filename or etag).

### `POST /interpret/{id}` (only non-static endpoint)

On-demand Claude interpretation. Reads variant from DuckDB, builds prompt,
calls Claude API, caches result. Completely separate from the data pipeline.

## Frontend Computation

The frontend operates on **one variant at a time**. None of this grows with dataset size.

### From flat row + heads.json:

| Computation | Formula | Why frontend? |
|---|---|---|
| Delta | `var_score_{h} - ref_score_{h}` | Trivial subtraction. Storing doubles disruption columns. |
| Z-score | `(delta - heads[h].mean) / heads[h].std` | Trivial. heads.json has mean/std. |
| Attribution | Sort disruption heads by \|z\|, dedup by prefix, top N | ~500 comparisons. Dedup logic is in head_vocab.json. |
| Bar color | Bin lookup in distributions.json | O(1) per head. Distributions loaded once. |
| Predictor display | Read `heads[h].predictor` for order, threshold, inversion | 10 heads. Config-driven, not hardcoded. |
| LOEUF label | 4 threshold comparisons on `variant.loeuf` | Display logic. |
| gnomAD classification | 3 threshold comparisons on `variant.gnomad` | Display logic. |
| Neighbor counts | Count 10 items by label | Trivial. |

### Head grouping and filtering:

The frontend reads heads.json and builds its display structure:

```js
// Disruption groups (for "All Disruptions" expansion)
const disruptionGroups = {};
for (const [key, head] of Object.entries(heads)) {
  if (head.category !== 'disruption') continue;
  if (head.quality === 'fail') continue;
  const group = head.group;
  (disruptionGroups[group] ??= []).push(key);
}

// Predictors (for Computational Predictors card)
const predictors = Object.entries(heads)
  .filter(([_, h]) => h.predictor)
  .sort(([_, a], [__, b]) => a.predictor.order - b.predictor.order);
```

No hardcoded head lists in the frontend. Everything comes from heads.json.

## Pipeline: Artifact Chain

```
EXTERNAL (immutable)
  activations/           Raw Evo2 activations
  variants.parquet       Annotated variants (ClinVar + VEP + gnomAD)
  head_vocab.json        Head config (display names, groups, predictor config, curation rules)

STAGE 1: train                                          GPU, SLURM
  IN:  activations/, head specs
  OUT: weights.pt, config.json

STAGE 2: extract                                        GPU, SLURM (8 shards → finalize)
  IN:  weights.pt, config.json, activations/
  OUT: scores.feather      all variants in one file (labeled + VUS)
       embeddings/         per-variant [3, d_h, d_h] matrices

STAGE 3: eval                                           CPU
  IN:  scores.feather, variants.parquet
  OUT: eval.json            {head: {metric, value}}

STAGE 4: aggregate                                      GPU (UMAP, KNN) + CPU (stats)
  IN:  scores.feather, embeddings/, variants.parquet
  OUT: head_stats.json      {head: {mean, std}}
       distributions.json   per-head histograms
       neighbors.feather    variant_id → [10 neighbor IDs + similarities]
       umap.json            30K sampled points

STAGE 5: build                                          CPU only
  IN:  variants.parquet, scores.feather, neighbors.feather,
       eval.json, head_stats.json, head_vocab.json, config.json
  OUT: variants.duckdb      indexed, ready to serve
       heads.json           merged vocabulary (vocab + eval + stats + category)
       (distributions.json and umap.json copied from stage 4)
```

Every arrow is an explicit artifact. No hardcoded logic leaks between stages.
Build is a pure join + index step — no GPU, no aggregation, no curation logic.

## Column Naming Conventions

The frontend (and server) never needs a "column registry" — the naming convention
tells you everything:

| Prefix | Meaning | Category | Example |
|--------|---------|----------|---------|
| `ref_score_` | Probe output on reference allele | disruption | `ref_score_helix` |
| `var_score_` | Probe output on variant allele | disruption | `var_score_helix` |
| `score_` | Probe output on diff (var-ref) view | effect | `score_cadd_c` |
| `gt_` | Ground truth from database annotation | ground truth | `gt_cadd_c` |
| (none) | Variant metadata | metadata | `gene`, `pos`, `label` |

To get the head key from a column name: strip the prefix.
To look up the head's metadata: `heads[key]`.

## Key Design Decisions

1. **DuckDB native format, not feather, for serving.** Feather is the pipeline's
   intermediate format. DuckDB database is the serving artifact because it supports
   indexed point lookups (O(log n) vs O(n) scan).

2. **Flat rows over the wire.** ~13-18 KB per variant. Don't try to be sparse.
   Bandwidth is not the bottleneck at this payload size.

3. **All aggregate data precomputed offline.** Distributions, head_stats, UMAP —
   never computed at request time. At 50 GB these are minutes of batch compute.

4. **Neighbors are nullable.** Not every variant has precomputed embeddings.
   The frontend hides the neighbors card when null.

5. **Gene search is exact/prefix only.** No substring search. Serve a gene list
   for client-side fuzzy matching if needed.

6. **head_vocab.json is the single configuration artifact.** All head display names,
   groups, predictor config, quality thresholds, and curation rules live there.
   The build step merges it with eval.json + head_stats.json + config.json to
   produce the serving heads.json.

7. **The frontend never hardcodes head lists.** Predictor card, disruption groups,
   effect groups — all driven by heads.json. Adding a new predictor means editing
   head_vocab.json, not touching frontend code.
