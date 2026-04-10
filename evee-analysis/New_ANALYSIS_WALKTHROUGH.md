# EVEE Analysis Walkthrough

A plain-language guide to each analysis, what it tests, and what safeguards are in place. Written so a biologist can explain the full pipeline to a technical audience.

---

## Background: What Are We Testing?

Evo2 is a large genomic language model trained on raw DNA sequences. When we run ClinVar variants through Evo2 and a trained probe, we extract a 64-by-64 matrix for each variant — a compact "fingerprint" capturing how the model sees that variant's effect. Each matrix has 4,096 individual entries. We have approximately 184,000 such fingerprints across ~13,000 genes.

The central question is: **Do these fingerprints encode real biology?** Specifically, do variants in functionally related genes end up with similar fingerprints? We test this against five independent sources of biological ground truth spanning two broad categories:

**Structural biology** (who physically interacts with whom):
- CORUM protein complexes
- STRING protein-protein interactions
- HGNC gene families

**Functional biology** (who behaves like whom in cellular assays):
- DEMETER2 (RNAi) dependency profiles
- Chronos (CRISPR) dependency profiles

For each variant, we precompute its nearest neighbors — the other variants whose 4,096-dimensional fingerprints are most similar (cosine distance, brute-force exact search). Then we ask: are those neighbors biologically related to the original variant more often than you would expect by chance?

Beyond neighbor enrichment, we also investigate *how* the embedding encodes biology: which specific matrix entries drive the signal, whether structural and functional biology share latent features, and whether the model's learned representations generalize to data it has never seen.

---

## Analysis 1: CORUM Co-Complex Enrichment

**Script:** `run_corum_full.py`

### What it does

CORUM is a curated database of protein complexes — groups of proteins that physically bind together inside cells. If two genes encode proteins in the same complex, they are "co-complex partners."

This analysis asks: when we look at a variant's nearest neighbors, do those neighbors come from genes in the same protein complex more often than random? We measure this as "fold enrichment" — how many times more likely neighbors are to share a complex compared to a random baseline.

The script:
1. Loads all variants with embeddings from DuckDB (not just those with DepMap annotations).
2. Loads the 64×64 embeddings from safetensors chunk files via an SQLite index, flattens them to 4,096-d, and L2-normalizes.
3. Computes exact brute-force cosine kNN over all variants at k = 5, 10, 20, 50.
4. For each variant whose gene is in CORUM, examines its top-k neighbors. For every cross-gene neighbor pair where both genes are CORUM-annotated, checks whether they share at least one complex.
5. Generates matched random controls by sampling the same number of random CORUM genes per source gene.
6. Computes fold enrichment and odds ratios with gene-level bootstrap confidence intervals (5,000 iterations).

### Controls and safeguards

- **Cross-gene only:** Same-gene pairs are excluded, since variants in the same gene trivially share complexes. The code explicitly checks `tgt_gene != src_gene`.
- **Matched random controls:** For each variant, the script samples exactly the same number of random CORUM genes (with rejection sampling to avoid self-pairing), so the baseline accounts for the varying number of valid neighbor pairs per gene.
- **Gene-level bootstrap:** Confidence intervals are computed by resampling *genes*, not individual pairs. This avoids inflating significance from genes with many variants — a critical correction since some genes contribute hundreds of variants.
- **Fixed random seed (42):** Each k value uses a deterministic RNG (`default_rng(RANDOM_SEED + k)`), ensuring full reproducibility.
- **CORUM minimum size filter:** Complexes with fewer than 3 genes are excluded. Gene symbols are uppercased for consistent matching.
- **kNN indices are saved** (`corum_full_knn_indices.npz`) for reuse by downstream analyses, avoiding recomputation and ensuring consistency.

---

## Analysis 2: CORUM Binary Retrieval

**Script:** `run_corum_retrieval.py`

### What it does

A stricter evaluation of the CORUM signal, treating the neighbor graph as a binary classifier. Instead of asking "how enriched are neighbors for co-complex membership?", this asks: "if we treated every cross-gene neighbor pair as a predicted co-complex pair, how accurate would that prediction be?"

This is measured with standard retrieval metrics:
- **Precision:** What fraction of predicted pairs are true co-complex partners?
- **Recall:** What fraction of all true co-complex pairs are found among neighbors?
- **F1 score:** The harmonic mean of precision and recall.

### Controls and safeguards

- Reuses the saved kNN indices from Analysis 1, ensuring identical neighbor graphs.
- Pairs are undirected and deduplicated (if A→B and B→A both appear, they count once).
- The gene universe is restricted to genes present in both CORUM and the neighbor graph.
- Gene-level bootstrap for confidence intervals.

### Key consideration

This is an extremely imbalanced task — over 99% of possible gene pairs are *not* co-complex. Even small lifts in precision above the base rate are meaningful in this regime, but absolute precision values will look low by design.

---

## Analysis 3: STRING Protein Interaction Enrichment & Retrieval

**Scripts:** `run_string_analysis.py`, `run_string_retrieval.py`

### What it does

STRING is a database of protein-protein interactions — broader than CORUM because it includes indirect functional associations, not just physical complexes. This analysis repeats the CORUM enrichment and retrieval framework using STRING v12.0 as ground truth.

`run_string_analysis.py` computes:
1. **Fraction of neighbors sharing a STRING interaction** at various confidence thresholds.
2. **Average STRING combined score** among neighbor pairs that do interact, vs. random.
3. Both metrics computed at k = 5, 10, 20, 50, with gene-level bootstrap CIs.

`run_string_retrieval.py` evaluates binary retrieval (precision, recall, F1) at three STRING confidence thresholds: 400 (medium), 700 (high), and 900 (highest confidence).

### Controls and safeguards

- Same matched-random and gene-level bootstrap framework as CORUM.
- STRING gene names are normalized to uppercase to prevent missed matches due to capitalization.
- Multiple confidence thresholds tested (400, 700, 900) to show robustness across stringency levels.
- This is a completely independent data source from CORUM, providing a genuine replication.

---

## Analysis 4: HGNC Gene Family Enrichment

**Script:** `run_gene_family_analysis.py`

### What it does

HGNC (Human Gene Nomenclature Committee) classifies genes into families based on shared evolutionary origin and function — for example, all zinc finger proteins, all kinases, or all ion channels. This analysis asks: do embedding neighbors belong to the same gene family more often than expected?

The script provides two levels of analysis:
1. **Global enrichment:** Overall fold enrichment across all gene families at k = 5, 10, 20, 50.
2. **Per-class precision:** For each gene family above a size threshold, what fraction of a member gene's neighbors are also in that family? This identifies which families the embedding captures best.

### Controls and safeguards

- Gene families smaller than 5 genes are excluded (`MIN_GROUP_SIZE = 5`), avoiding noise from tiny groups where a single coincidence would produce extreme enrichment.
- Same matched-random and gene-level bootstrap framework.
- Per-class precision breakdown shows which families contribute most, preventing the global metric from being dominated by a few large families.

---

## Analysis 5: DEMETER2 Dependency Profile Similarity

**Script:** `run_neighbor_depmap_analysis.py`

### What it does

This analysis shifts from "do neighbors share annotations?" to "do neighbors behave similarly in experimental assays?" The DepMap project's DEMETER2 dataset uses RNA interference (RNAi) to knock down each gene across 707 cancer cell lines and measures the effect on cell survival. If two genes have similar "dependency profiles" (they are essential in the same cell types and dispensable in others), they likely participate in the same biological pathway.

We ask: are the Pearson dependency-profile correlations of embedding neighbor pairs higher than those of matched random pairs?

The script:
1. Loads variants and their embedding neighbors from DuckDB.
2. Loads DEMETER2 dependency matrix (cell lines × genes).
3. For each variant, identifies its cross-gene neighbors and computes the Pearson correlation between the source gene's and neighbor gene's dependency profiles.
4. For each neighbor pair, generates a matched random pair from the same source gene and same variant consequence category (missense, nonsense, splice, synonymous).
5. Computes the mean correlation "delta" (neighbor minus random) with gene-level bootstrap CIs.

### Controls and safeguards

- **Cross-gene only:** Same-gene pairs are excluded because a gene's correlation with itself is trivially 1.0.
- **Matched random controls:** Each random pair is matched on both source gene and variant consequence category, controlling for gene-specific and consequence-specific effects.
- **Gene-level bootstrap:** Confidence intervals resample genes, not individual pairs.
- **Gene-collapse robustness test:** The analysis is repeated 30 times using only 1 randomly chosen variant per gene. If the delta remains significant, the effect is not driven by multi-variant genes inflating the sample.
- **Minimum overlap:** At least 50 cell lines must have valid data for both genes in a pair.
- **Stratification by consequence type:** Separate deltas are reported for missense, nonsense, splice, and synonymous variants, checking that the signal is not driven by one consequence class.

---

## Analysis 6: Chronos (CRISPR) Dependency Profile Similarity

**Script:** `run_neighbor_chronos_analysis.py`

### What it does

The same analysis as DEMETER2, but using Chronos (CRISPR interference) data — a fundamentally different experimental technology measuring the same biological concept (gene essentiality). RNAi and CRISPR perturb genes through different mechanisms, so replication across both provides stronger evidence.

### Controls and safeguards

- Same controls as DEMETER2 (cross-gene, matched random, gene-level bootstrap, gene-collapse).
- **Important caveat:** The script filters Chronos cell lines to those that overlap with DEMETER2, so lineage annotations can be borrowed. This means only 551 of 1,208 Chronos cell lines are used — **55% of the Chronos data is discarded**. This is a conservative choice that ensures consistent lineage handling, but it substantially reduces the available data.
- Raw delta values between DEMETER2 and Chronos are **not directly comparable** because they use different numbers of cell lines, which affects the baseline correlation magnitude. Analysis 13 (Quick Analyses Q1) addresses this with Cohen's d.

---

## Analysis 7: Follow-up Analyses (Distributions, Delta vs. k, Thresholds)

**Script:** `run_followup_analyses.py`

### What it does

Three deeper dives into the DEMETER2 and Chronos results:

1. **Full distributions:** Shows the complete distribution of neighbor vs. random correlations (not just the mean delta), confirming the shift is across the whole distribution rather than driven by a handful of outlier pairs.

2. **Delta vs. k:** Tests whether the dependency signal changes as we look at more neighbors (k = 5, 10, 20, 50). Biologically, closer neighbors (smaller k) should be more similar, so we expect the delta to be largest at k = 5 and to decrease with increasing k. This pattern would confirm the signal is related to embedding proximity, not an artifact.

3. **Threshold fractions:** What fraction of neighbor pairs exceed various correlation thresholds (0.05, 0.10, 0.15, 0.20)? This addresses whether the mean delta comes from a general shift or from a subpopulation of highly correlated pairs.

### Controls and safeguards

- Reuses saved pair data from the primary analyses (Analyses 5 and 6) — no recomputation of the base results ensures consistency.
- For delta-vs-k, kNN must be recomputed from raw embeddings to evaluate different k values.

---

## Analysis 8: CORUM Feature Interpretability

**Script:** `run_corum_interpretability.py`

### What it does

Moves from "the embedding works" to "how does it work?" Each variant fingerprint is a 64×64 matrix with 4,096 individual entries. This analysis asks: do specific matrix entries "light up" for specific protein complexes?

The script has 10 stages:

1. **Gene-level matrices:** Averages the 64×64 matrix across all variants of the same gene (minimum 3 variants per gene, `MIN_VARIANTS_PER_GENE = 3`), producing a single representative matrix per gene. These are then z-scored across genes for each entry.

2. **CORUM gene sets:** Filters complexes to those with at least 5 embedded genes (`MIN_EMBEDDED_GENES_PER_COMPLEX = 5`). Annotates each complex with a broad functional class (Ribosome, Spliceosome, Proteasome, etc.) using keyword matching.

3. **Entry-wise enrichment:** For each complex and each of the 4,096 entries, compares the z-scored values of member genes vs. all other genes using a Welch t-test. Computes Cohen's d for effect size. Applies Benjamini-Hochberg FDR correction per complex (4,096 tests per complex).

4. **Top entries and recurrence:** Selects the top 20 entries per complex (by FDR, then by effect size). Identifies entries that are significant across 3 or more complexes — these "recurrently significant" entries likely encode broadly relevant biological features.

5. **Delta heatmaps:** For selected complexes, visualizes the 64×64 difference between member-gene means and background means, showing where the signal lives in the matrix.

6. **Complex signature clustering:** Uses hierarchical clustering on the complex-level signatures (top-entry vectors) to find groups of complexes with similar embedding patterns.

7. **Broad class enrichment:** Tests whether certain broad classes (ribosomal, spliceosomal, etc.) have more significant entries than others.

8. **Robustness checks:** Includes variant-count bias check (correlation between a gene's variant count and its mean matrix values) and split-half stability test (randomly splitting each complex in half and comparing enriched entries between halves, repeated 100 times).

### Controls and safeguards

- **FDR correction (Benjamini-Hochberg)** applied per complex — essential because each complex tests 4,096 entries.
- **Minimum gene thresholds:** At least 3 variants per gene for averaging, at least 5 embedded genes per complex for testing.
- **Z-scoring** per entry across all genes to normalize scale, preventing entries with high absolute variance from dominating.
- **Variant-count bias check:** Confirms results aren't driven by data quantity per gene.
- **Split-half stability (100 iterations):** The top entries identified from one half of a complex's genes replicate in the other half, confirming the signal is a property of the complex, not a sample-specific artifact.
- **Gene-level matrices are saved** (`gene_level_matrices.npz`) for reuse by downstream analyses (Analyses 9, 10, 14, 15).

---

## Analysis 9: Chronos Entry-Level Correlations

**Script:** `run_chronos_entry_analysis.py`

### What it does

Connects the interpretability analysis to gene essentiality. For each of the 4,096 matrix entries, it correlates the gene-level z-scored values with Chronos dependency metrics: mean dependency score (how essential is the gene on average?) and fraction of cell lines where the gene is essential. Entries that correlate strongly with dependency may encode essentiality-related features.

As a more holistic test, it trains a Ridge regression model using all 4,096 features to predict mean Chronos dependency, with cross-validation to prevent overfitting.

The script also identifies which rows and columns of the 64×64 matrix are most enriched for dependency-associated entries, and compares the top dependency-associated entries with the top CORUM-enriched entries from Analysis 8.

### Controls and safeguards

- **FDR correction** across all 4,096 entries × 4 dependency metrics (Pearson and Spearman correlations for both mean dependency and fraction dependent).
- **Ridge regression with cross-validation** to prevent overfitting — the penalty parameter is selected automatically.
- Requires `gene_level_matrices.npz` from Analysis 8 to ensure consistent gene-level representations.

---

## Analysis 10: CORUM vs. Chronos Feature Comparison

**Script:** `run_corum_vs_chronos_features.py`

### What it does

Asks: do the matrix entries that distinguish protein complexes (CORUM, structural biology) overlap with the entries that predict gene essentiality (Chronos, functional biology)? This tests whether the embedding uses shared or separate latent features for different types of biology.

The analysis includes:
1. **Feature overlap:** Counts how many of the top-N CORUM features and top-N Chronos features overlap, for N = 50, 100, 200, 410.
2. **Permutation test (2,000 iterations):** Assesses whether the observed overlap exceeds what you'd expect if the two feature sets were randomly drawn from the 4,096 entries.
3. **Feature classification:** Classifies each of the 4,096 entries as shared, CORUM-only, Chronos-only, or background, based on whether they appear in the top-N of each dataset.
4. **Cross-predictive power:** Can CORUM-only features predict Chronos dependency? Can Chronos-only features predict CORUM complex membership? This uses Ridge regression.
5. **Structural analysis:** Tests whether significant features cluster in specific rows or columns of the 64×64 matrix (row and column enrichment tests).
6. **Block/motif discovery:** Searches for rectangular blocks of the matrix where features are concentrated.

### Controls and safeguards

- **Permutation test** for overlap significance — the observed overlap is compared against 2,000 random draws of the same size from 4,096 entries.
- **Separate derivation of CORUM and Chronos scores** — the CORUM score (recurrence × mean effect) and Chronos score (|Pearson r| with mean dependency) are computed independently, ensuring no circularity.
- **Feature classes are mutually exclusive** (shared / CORUM-only / Chronos-only / background).

---

## Analysis 11: Left vs. Right Projection Comparison

**Script:** `run_left_right_analysis.py`

### What it does

The 64×64 matrix can be summarized in three ways:
- **Left (row mean):** Average across rows → 64-dimensional vector
- **Right (column mean):** Average across columns → 64-dimensional vector
- **Full (flatten):** All 4,096 entries

These correspond to different ways of reading the probe's output. The "left" projection captures one axis of the covariance structure, the "right" captures the other, and the "full" uses everything.

This analysis compares all three views for their ability to capture:
1. **CORUM co-complex enrichment** (Stage 1): Computes kNN and fold enrichment for each view separately.
2. **DepMap/Chronos dependency correlation** (Stage 2): Computes the neighbor-vs-random delta in dependency profile correlation for each view.

### Controls and safeguards

- Same enrichment framework as Analysis 1 (cross-gene, matched random, gene-level bootstrap).
- All three views use L2-normalized vectors for fair comparison.
- kNN is recomputed separately for each view, ensuring that the neighbor graph reflects that view's geometry.

---

## Analysis 12: UMAP Visualization

**Script:** `plot_umap_gene_families.py`

### What it does

Creates a 2D visualization of all gene-level embeddings using UMAP (Uniform Manifold Approximation and Projection). The pipeline:
1. Computes gene-level mean 64×64 matrices (averaged across variants per gene).
2. Flattens to 4,096-d and applies PCA for initial dimensionality reduction.
3. Runs UMAP to project into 2D.
4. Overlays HGNC gene family labels to show whether families cluster visually.

### Key consideration

UMAP is a qualitative visualization tool — it preserves local structure but can distort global distances. The quantitative enrichment results (Analyses 1–6) are the rigorous tests; UMAP provides visual intuition.

---

## Analysis 13: Quick Follow-up Analyses

**Script:** `run_quick_analyses.py`

### What it does

Four targeted follow-up analyses addressing specific questions raised during review:

**Q1 — Cohen's d for DEMETER2 vs. Chronos:**
The raw deltas from Analyses 5 and 6 are not directly comparable because the two datasets use different numbers of cell lines, which affects correlation magnitudes. Cohen's d normalizes the delta by pooled standard deviation, enabling a fair cross-dataset comparison. Bootstrap CIs (2,000 iterations, gene-level) are computed for both.

**Q2 — Per-complex CORUM ranking:**
Ranks all 300 CORUM complexes by the strength of their embedding signature (fraction of 4,096 entries significant at FDR < 0.05). This identifies which specific complexes the model captures best and whether enrichment is concentrated in a few complexes or spread broadly.

**Q3 — ClinVar pathogenicity stratification:**
Stratifies the dependency signal by ClinVar clinical significance (pathogenic vs. benign). If pathogenic variants show a stronger neighbor-dependency signal than benign ones, this would suggest the embedding captures variant-level functional information beyond just gene identity.

**Q4 — Permutation null for structural enrichment:**
Constructs a formal permutation null distribution for the CORUM enrichment. Gene labels are randomly shuffled (1,000 permutations) while preserving the kNN graph topology, and the co-complex fraction is recomputed each time. The observed value is compared to this null to compute a z-score and empirical p-value. This is a more stringent test than the matched-random approach because it directly controls for the kNN graph structure.

### Controls and safeguards

- Gene-level bootstrap for all CI computations.
- Permutation test preserves kNN graph topology while destroying biological signal, ruling out artifacts of graph structure.
- Uses the `reproducibility.py` module for seed enforcement, run configuration logging, and output manifesting.

---

## Analysis 14: Cross-Dataset Latent Feature Comparison

**Script:** `run_followup_dataset_comparison.py`

### What it does

Extends Analysis 10 (CORUM vs. Chronos) to all four biological datasets — CORUM, STRING, Chronos, DEMETER2 — performing all six pairwise comparisons of latent feature importance.

Two new "importance scores" are computed:
- **DEMETER2 functional score:** Identical pipeline to Chronos (Analysis 9), but using the DEMETER2 RNAi dependency matrix. For each of the 4,096 entries, computes |Pearson r| between the gene-level z-scored entry value and the gene's mean DEMETER2 dependency.
- **STRING structural score:** Analogous to CORUM (Analysis 8), but using STRING-derived gene groups. For each gene with ≥5 high-confidence (≥700) interaction partners in the embedding, creates a group and runs Welch t-tests per entry. Score = n_significant_groups × mean |effect size|.

The analysis then:
1. Computes Spearman rank correlations between all six pairs of 4,096-dimensional importance vectors.
2. Computes top-N overlap (N = 50, 100, 200, 410) for each pair, with fold enrichment over chance.
3. Classifies each comparison as within-structural, within-functional, or cross-type.
4. Generates heatmaps showing individual and joint-clustered feature importance maps.
5. Produces normalized difference maps between key dataset pairs.

### Controls and safeguards

- Consistent scoring methodology: structural datasets use the same Welch t-test framework; functional datasets use the same Pearson correlation framework.
- STRING groups use a minimum of 5 high-confidence partners and are capped at 500 groups, preventing a single hub gene from dominating.
- Hierarchical clustering of heatmaps uses a joint ordering (all four score vectors combined), ensuring visual comparability across panels.
- Uses `reproducibility.py` for seed enforcement and run configuration logging.

---

## Analysis 15: Forward-in-Time CORUM Prediction

**Script:** `run_corum_forward_prediction.py`

### What it does

This is the most methodologically rigorous analysis in the suite. It addresses a critical question: **Can the embedding predict protein complex membership that was discovered after the model was trained?**

The key insight is that our main interpretability analysis (Analysis 8) uses the current CORUM v5 database to both derive predictive features and evaluate them — a form of circularity. This script corrects that by using an older CORUM version (v3) as the sole source of predictive features, then testing those features on "novel" co-complex pairs that appear only in the current v5.

The script has four parts:

**Part 1 — Database comparison:** Formally compares CORUM v3 (Human-filtered) and current (v5), quantifying the number of shared complexes, new complexes, new co-complex pairs, changed membership, and gene universe overlap (Jaccard index). The v3 database uses a different file format (compressed JSON with semicolon-separated gene names in a different schema), which the script handles explicitly.

**Part 2 — Forward-in-time feature derivation:** Re-runs Stages 2–4 of the interpretability pipeline (Analysis 8) using ONLY v3 complexes. This produces "v3-derived" recurrent entries — matrix features that the model considers important for v3-era protein complexes. These features have zero knowledge of v5 additions.

**Part 3 — Four prediction tests:**

- **Test A (kNN enrichment):** Are novel co-complex pairs (v5-only) enriched among the kNN neighbors? This test uses raw embedding distances and is unaffected by the feature derivation circularity. Uses Fisher's exact test with gene-level bootstrap CIs.

- **Test B (Cosine similarity):** Are novel co-complex gene pairs closer in latent space than random gene pairs? Uses Mann-Whitney U test and rank-biserial correlation.

- **Test C (v3-derived feature scores):** The corrected, non-circular test. Computes a feature-based score for each gene pair using only the v3-derived features: weighted |z_a + z_b| at v3-derived entry positions. Compares novel pair scores to random pair scores via Mann-Whitney U and Cohen's d.

- **Test D (Complex-level enrichment):** For entirely new complexes (present in v5 but absent in v3), runs the full Welch t-test enrichment analysis from scratch. Compares the number of significant entries in novel complexes against the baseline of existing complexes.

**Part 4 — Feature stability:** Compares the v3-derived and v5-derived recurrent entries. Computes Jaccard overlap, Spearman correlation of effect sizes on shared entries, and generates side-by-side heatmaps showing which 64×64 positions are stable across database versions.

### Controls and safeguards

- **Strict temporal separation:** v3-derived features are computed exclusively from v3 complex definitions. The embedding itself was trained on v5-era ClinVar data, but this is acceptable because we are testing CORUM complex membership, not variant pathogenicity.
- **Multiple complementary tests:** Tests A and B use raw embedding geometry (no derived features). Test C uses v3-derived features (the corrected test). Test D uses ab-initio enrichment. If all four agree, the result is robust.
- **Fisher's exact test** (one-sided) for Test A, Mann-Whitney U for Tests B and C, Welch t-test with FDR correction for Test D.
- **Gene-level bootstrap CIs** for Test A.
- **Random pair controls** sample from the full gene universe, excluding both v3 and v5 co-complex pairs.
- **v3 organism filter:** The v3 database is multi-organism; only human complexes are retained.
- **Configuration saved** as JSON for reproducibility.

---

## Utility Scripts

### `reproducibility.py`
A shared module providing:
- `enforce_seeds()`: Sets random, numpy, and environment seeds for deterministic execution.
- `save_run_config()`: Saves all analysis parameters as JSON.
- `save_run_manifest()`: Captures environment details (Python version, package versions, OS, git commit).

Used by Analyses 13, 14, and the quick analyses.

### `inspect_db.py`
Inspects the `variants.duckdb` database — schema, row counts, consequence distributions, label distributions, neighbor structure, and global configuration. Useful for verifying the data pipeline before running analyses.

### `verify_determinism.py`
Runs a specified analysis script twice and compares file checksums (SHA-256) of all outputs. Any mismatch indicates non-determinism, which would undermine reproducibility claims.

### `make_presentation_figures.py`
Reads intermediate data files from previous analyses and generates polished, publication-quality figures with consistent styling.

---

## Summary of Controls Across All Analyses

| Control | Purpose | Applied in |
|---|---|---|
| Cross-gene restriction | Prevents trivial same-gene matches | All neighbor analyses |
| Matched random controls | Ensures fair baseline comparison | All enrichment and dependency analyses |
| Gene-level bootstrap | Avoids pseudoreplication from multi-variant genes | All confidence intervals |
| Gene-collapse robustness | Confirms effect with 1 variant per gene | DEMETER2, Chronos |
| FDR correction (Benjamini-Hochberg) | Controls false discoveries across 4,096 entries | Interpretability, entry-level, STRING groups |
| Permutation tests | Tests enrichment against graph-structure null | Quick analysis Q4, feature overlap |
| Split-half stability | Checks within-complex reproducibility | Interpretability (100 iterations) |
| Temporal separation (v3 vs. v5) | Eliminates circularity in forward prediction | Forward-in-time analysis |
| Fisher's exact test | Non-parametric test for enrichment significance | Forward prediction Test A |
| Cohen's d normalization | Enables fair cross-dataset effect size comparison | Quick analysis Q1 |
| ClinVar stratification | Tests whether signal varies by pathogenicity | Quick analysis Q3 |
| Fixed random seed (42) | Full reproducibility | All scripts |
| Config/environment logging | Enables exact replication | All scripts (via reproducibility.py) |
| Minimum size thresholds | Ensures statistical power | Complexes ≥ 3 genes, families ≥ 5, complex embedded genes ≥ 5, variants per gene ≥ 3 |
| Multiple independent databases | Cross-validates findings | 5 ground-truth databases across 2 biology types |

---

## Dependency Graph Between Analyses

```
Data preparation
─────────────────
DuckDB + safetensors embeddings
        │
        ▼
Analysis 1: CORUM full enrichment
 ├─ saves: corum_full_knn_indices.npz
 │   ├─ reused by: Analysis 2 (retrieval)
 │   ├─ reused by: Analysis 13 Q4 (permutation)
 │   └─ reused by: Analysis 15 Test A (forward prediction kNN)
 │
 ▼
Analysis 8: CORUM interpretability
 ├─ saves: gene_level_matrices.npz (gene-level z-scored embeddings)
 │   ├─ reused by: Analysis 9 (Chronos entry)
 │   ├─ reused by: Analysis 14 (cross-dataset comparison)
 │   └─ reused by: Analysis 15 Part 2 (forward prediction features)
 ├─ saves: corum_entry_enrichment.parquet
 │   ├─ reused by: Analysis 10 (CORUM score)
 │   └─ reused by: Analysis 14 (CORUM score)
 └─ saves: corum_complex_gene_sets.parquet
     └─ reused by: Analyses 13, 15

Analysis 9: Chronos entry correlations
 └─ saves: chronos_entry_correlations.parquet
     ├─ reused by: Analysis 10 (Chronos score)
     └─ reused by: Analysis 14 (Chronos score)

Analyses 5, 6: DEMETER2 / Chronos dependency
 └─ saves: *_profile_similarity.parquet
     ├─ reused by: Analysis 7 (follow-up)
     └─ reused by: Analysis 13 Q1, Q3

Independent analyses (no upstream dependencies beyond DuckDB):
  Analysis 3: STRING enrichment & retrieval
  Analysis 4: HGNC gene family enrichment
  Analysis 11: Left/right projection comparison
  Analysis 12: UMAP visualization
```

---

## Relationship Between Analyses

```
Structural biology                   Functional biology
─────────────────                    ──────────────────
CORUM complexes ──────┐         ┌──── DEMETER2 (RNAi, 707 cell lines)
STRING interactions ──┤         ├──── Chronos (CRISPR, 551 cell lines)
HGNC gene families ───┘         │
                                │
          ┌─────────────────────┘
          │
     Embedding neighbors
          │
          ├── Enriched for structural biology? (Analyses 1-4)
          │     → Yes (2-3× fold enrichment, permutation p ≈ 0)
          │
          ├── Share functional profiles? (Analyses 5-7)
          │     → Yes (small but significant deltas, consistent across technologies)
          │
          ├── Which matrix features drive this? (Analyses 8-10)
          │     → Specific entries distinguish complexes; features shared between
          │       structural and functional biology
          │
          ├── Do rows and columns capture different biology? (Analysis 11)
          │     → Partially — left and right projections have different strengths
          │
          ├── Deeper questions (Analysis 13):
          │     → Cohen's d for fair comparison, pathogenicity stratification,
          │       permutation null, per-complex ranking
          │
          ├── Cross-dataset feature landscape (Analysis 14):
          │     → All four datasets compared pairwise; within-type pairs
          │       more similar than cross-type
          │
          └── Forward prediction (Analysis 15):
                → v3-derived features predict v5-novel complexes — the
                  embedding generalizes to discoveries made after training
```
