# Webapp Data Schema

The variant viewer serves three JSON files. `global.json` holds all metadata
(loaded once). Per-variant JSONs hold only values (loaded on demand).
`search.json` holds the gene-keyed search index (loaded lazily on first search).

## global.json (~1.8 MB)

```
umap                  UMAP scatter data (30K points, compact encoding)
  x, y              float[] — 2dp coordinates
  score             float[] — pathogenicity score per point
  ids               string[] — variant IDs (for click-to-navigate)
  genes             int[] — index into gene_list
  labels            int[] — 0=benign, 1=pathogenic, 2=VUS
  gene_list         string[] — deduplicated gene names

distributions         Per-head histograms (unified, includes pathogenicity)
  pathogenic          Pathogenicity score histogram (80 bins)
  {head_key}          {benign: int[], pathogenic: int[], bins: int}
                      Effect heads: histogram of predicted value (score_*)
                      Disruption heads: histogram of var-ref delta

heads                 Head schema — defines groups and membership
  disruption          {group_name: [head_key, ...], ...}
  effect              {group_name: [head_key, ...], ...}

display               Head display names
  {head_key}          string — human-readable name (e.g., "PhyloP")

eval                  Per-head evaluation metrics
  {head_key}          {metric: "r"|"AUC"|"acc", value: float}

decomposition         Score decomposition (linear model coefficients)
```

## variant JSON (~18 KB with 700 heads)

```
id                    string — variant ID (e.g., "chr21:45989644:G:A")
gene                  string — gene name
chrom                 string — chromosome
pos                   int — 0-based position
ref, alt              string — reference and alternate alleles
consequence           string — VEP consequence type
substitution          string|null — amino acid change (e.g., "G>R")
label                 string — "benign", "pathogenic", or "VUS"
significance          string — ClinVar clinical significance
stars                 int — ClinVar review stars (0-4)
disease               string — associated disease
score                 float — pathogenicity score (0-1)

rs_id                 string|null — dbSNP ID
allele_id             int|null — ClinVar allele ID
gene_id               string|null — Ensembl gene ID
hgvsc                 string|null — HGVS coding notation
hgvsp                 string|null — HGVS protein notation
impact                string|null — VEP impact (HIGH/MODERATE/LOW/MODIFIER)
exon                  string|null — exon number (e.g., "14/35")
transcript            string|null — Ensembl transcript ID
swissprot             string|null — UniProt accession
domains               array|null — protein domains
  []                  {db, id, name?} — database, accession, human-readable name
loeuf                 float|null — loss-of-function observed/expected upper bound
gnomad                float|null — gnomAD exome allele frequency
gnomad_pop            Per-ancestry frequencies (sparse, only >0)
  {pop_code}          float — allele frequency (afr, amr, asj, eas, fin, nfe, sas)

disruption            Ref→var disruption scores (flat dict)
                      Each head is scored on ref and var activations separately.
                      The delta (var - ref) shows what the mutation disrupted.
  {head_key}          [ref, var] — 2-tuple of floats

effect                Variant effect scores (flat dict)
                      Each head is scored on diff (var-ref) activations.
                      Predicts clinical predictors, consequence, domain effects.
  {head_key}          float — predicted value

gt                    Ground truth annotations (sparse, only non-null positive)
  {head_key}          float — database value

attribution           Per-variant program attribution (from src.attribution)
  []                  array of programs, sorted by |contribution|
    id                int — program atom index
    contribution      float — signed contribution to pathogenicity logit
    direction         "pathogenic"|"protective"
    heads[]           contributing heads within this program
      name            string — head key
      kind            "effect"|"disruption"
      score           float — head score for this variant
      contribution    float — signed contribution (atom_weight * score * program_weight)

neighbors             10 nearest neighbors in embedding space
  []                  {id, gene, consequence, label, score, similarity}

nP, nB, nV            int — neighbor label counts (pathogenic, benign, VUS)
```

## search.json (~17 MB)

```
{GENE_NAME_UPPER}     array of {v: variant_id, l: label, s: score, c: consequence}
```

Sorted by score descending within each gene.

## Frontend contract

The JS joins variant values with global metadata at render time:

```js
for (const [group, keys] of Object.entries(globalData.heads.disruption)) {
  const items = keys
    .filter(k => v.disruption?.[k])
    .map(k => ({ key: k, ref: v.disruption[k][0], var: v.disruption[k][1], gt: v.gt?.[k] }))
    .sort((a, b) => Math.abs(b.var - b.ref) - Math.abs(a.var - a.ref));
  // render with globalData.display[key] and globalData.eval[key]
}
```

Display names: `globalData.display[key]`
Eval badges: `globalData.eval[key]`

## Probe config.json

```
preset              string — dataset preset used for training
d_model             int — input activation dimension (8192 for Evo2 20B)
d_hidden            int — covariance embedding dimension (32 → [32, 32] matrix)
d_probe             int — per-head factored bottleneck dimension
disruption_heads    string[] — heads scored on ref/var views (disruption profile)
effect_heads        string[] — heads scored on diff view (variant effect predictions)
n_disruption_heads  int — count of disruption heads
n_effect_heads      int — count of effect heads
```

Note: older probes use `ref_heads`/`diff_heads` — the pipeline supports both.

## Extract pipeline

`scripts/extract.py` produces both embeddings and scores in a single pass:

```
Raw activations [B, 2, 3, K, d]
    ├── unified_diff() → diff view → probe.embedding() → diff_emb [B, d_h, d_h]
    │                              → probe.forward_dict() → score_* (effect heads)
    ├── unified_ref()  → ref view  → probe.embedding() → ref_emb [B, d_h, d_h]
    │                              → probe.forward_dict() → ref_score_* (disruption heads)
    └── unified_var()  → var view  → probe.embedding() → var_emb [B, d_h, d_h]
                                   → probe.forward_dict() → var_score_* (disruption heads)

Embeddings stacked: [B, 3, d_h, d_h] (index 0=diff, 1=ref, 2=var)
```

End-to-end: `bash scripts/pipeline.sh /path/to/probe`
