# EVEE Analysis Walkthrough

A plain-language guide to each analysis, what it tests, and what safeguards are in place.

---

## Background: What Are We Testing?

Evo2 is a large genomic language model trained on DNA sequences. When we run ClinVar variants through Evo2 and a trained probe, we extract a 64-by-64 matrix for each variant — a compact "fingerprint" capturing how the model sees that variant's effect. We have ~184,000 such fingerprints.

The central question is: **Do these fingerprints encode real biology?** Specifically, do variants in functionally related genes end up with similar fingerprints? We test this against three independent sources of biological ground truth.

For each variant, we precompute its nearest neighbors — the other variants whose fingerprints are most similar. Then we ask: are those neighbors biologically related to the original variant more often than you'd expect by chance?

---

## Analysis 1: CORUM Co-Complex Enrichment

**Script:** `run_corum_full.py`

### What it does

CORUM is a curated database of protein complexes — groups of proteins that physically bind together inside cells. If two genes encode proteins in the same complex, they're "co-complex partners."

This analysis asks: when we look at a variant's nearest neighbors, do those neighbors come from genes in the same protein complex more often than random? We measure this as a "fold enrichment" — how many times more likely neighbors are to share a complex compared to a random baseline.

### Key result

At k=5 (looking at the 5 closest neighbors), neighbors share a complex **2.84 times** more often than random (95% confidence interval: 2.26 to 3.56). This enrichment holds across k=5, 10, 20, and 50.

### Controls and safeguards

- **Cross-gene only:** We exclude same-gene pairs because variants in the same gene trivially share complexes.
- **Matched random controls:** For each variant, we sample random genes from the same CORUM gene universe with the same count, so the comparison is apples-to-apples.
- **Gene-level bootstrap:** Confidence intervals are computed by resampling genes (not individual pairs), which avoids inflating significance from genes with many variants.
- **Fixed random seed (42):** Results are fully reproducible.
- **184,177 variants, 3,519 CORUM genes in the analysis.**

---

## Analysis 2: CORUM Binary Retrieval

**Script:** `run_corum_retrieval.py`

### What it does

A stricter version of the enrichment test. Instead of asking "how enriched are neighbors for co-complex membership?", this asks: "if we treated the neighbor graph as a predictor of co-complex pairs, how accurate would it be?" This is measured with precision (what fraction of predicted pairs are real) and recall (what fraction of real pairs are found).

### Key result

At k=5, precision is 1.03% — low in absolute terms, but **1.6 times the base rate** of 0.66%. This is an extremely imbalanced task (over 99% of possible gene pairs are not co-complex), so even small lifts above the base rate are meaningful. At k=50, the neighbor graph recovers 8.1% of all true co-complex pairs.

### Controls and safeguards

- Same gene-level bootstrap as above.
- Pairs are undirected and deduplicated.
- The gene universe is restricted to genes present in both CORUM and the neighbor graph.

---

## Analysis 3: STRING Protein Interaction Enrichment & Retrieval

**Scripts:** `run_string_analysis.py`, `run_string_retrieval.py`

### What it does

STRING is a database of protein-protein interactions (broader than CORUM — it includes indirect functional associations, not just physical complexes). This analysis repeats the CORUM enrichment and retrieval tests using STRING as the ground truth, at three confidence levels (400 = medium, 700 = high, 900 = highest).

### Key result

At k=5, neighbors share a STRING interaction **2.54 times** more often than random (CI: 2.47–2.62). This validates the CORUM finding with a completely independent data source and a much larger set of interactions (6.8 million gene pairs vs. 46,000 for CORUM).

### Controls and safeguards

- Same matched-random and gene-level bootstrap framework.
- STRING gene names are normalized to uppercase to prevent missed matches.
- Multiple confidence thresholds tested to show robustness.
- **12,759 STRING genes overlap with the neighbor graph.**

---

## Analysis 4: HGNC Gene Family Enrichment

**Script:** `run_gene_family_analysis.py`

### What it does

HGNC (Human Gene Nomenclature Committee) classifies genes into families based on shared evolutionary origin and function — for example, all zinc finger proteins, all kinases, or all ion channels. This analysis asks: do embedding neighbors belong to the same gene family more often than expected?

### Key result

At k=5, neighbors share a gene family **3.37 times** more often than random (CI: 3.08–3.72) — the strongest enrichment of the three ground truths. The top individual families show even stronger effects: KRAB domain proteins (8.7x lift) and zinc finger C2H2-type proteins (4.1x lift).

### Controls and safeguards

- Gene families smaller than 5 genes are excluded (avoids noise from tiny groups).
- Same matched-random and gene-level bootstrap framework.
- Per-class precision breakdown shows which families contribute most.

---

## Analysis 5: DEMETER2 Dependency Profile Similarity

**Script:** `run_neighbor_depmap_analysis.py`

### What it does

This analysis shifts from "do neighbors share annotations?" to "do neighbors behave similarly in cancer experiments?" The DepMap project (DEMETER2 dataset) uses RNA interference to knock down each gene across 707 cancer cell lines and measures the effect on cell survival. If two genes have similar "dependency profiles" (they're essential in the same cell types), they likely participate in the same biological pathway.

We ask: are the dependency profiles of embedding neighbors more correlated than those of random gene pairs?

### Key result

Yes. The gene-level mean correlation delta is **0.0016** (95% CI: 0.0013–0.0020). This is a small but statistically significant effect, and it is consistent across variant consequence types (missense, nonsense, splice, synonymous).

### Controls and safeguards

- **Cross-gene only:** Same-gene pairs excluded (same-gene correlation is trivially 1.0).
- **Matched random controls:** For each neighbor pair, a random pair is sampled with the same source gene and same variant consequence category, ensuring the comparison controls for gene-specific and consequence-specific effects.
- **Gene-level bootstrap:** Confidence intervals resample genes, not pairs, to avoid pseudoreplication from genes with many variants.
- **Gene-collapse robustness test:** Repeating the analysis with only 1 randomly chosen variant per gene (30 iterations) gives mean delta = 0.0014 (std = 0.0008), confirming the effect is not driven by multi-variant genes.
- **Minimum 50 cell lines overlap** required for any correlation to be computed.
- **86,746 variants, 10,963 genes, 779,526 cross-gene neighbor pairs.**

---

## Analysis 6: Chronos (CRISPR) Dependency Profile Similarity

**Script:** `run_neighbor_chronos_analysis.py`

### What it does

The same analysis as DEMETER2, but using Chronos CRISPR interference data — a different experimental technology measuring the same concept (gene essentiality). This provides an independent replication.

### Key result

Gene-level delta = **0.0053** (CI: 0.0049–0.0057). The signal replicates in this independent dataset.

### Controls and safeguards

- Same controls as DEMETER2 (cross-gene, matched random, gene-level bootstrap, gene-collapse).
- **Important caveat:** Only 551 out of 1,208 Chronos cell lines are used — the analysis restricts to cell lines that overlap with DEMETER2 so that lineage annotations can be borrowed. This means 55% of Chronos data is discarded.
- Raw delta values between DEMETER2 and Chronos are **not directly comparable** because they use different numbers of cell lines (707 vs. 551), which affects correlation magnitudes.
- **94,538 variants, 12,434 genes, 912,079 cross-gene neighbor pairs.**

---

## Analysis 7: Follow-up Analyses (Delta vs. k, Distributions, Thresholds)

**Script:** `run_followup_analyses.py`

### What it does

Three deeper dives into the DEMETER2 and Chronos results:

1. **Full distributions:** Shows the complete distribution of neighbor vs. random correlations (not just the mean), confirming the shift is across the whole distribution, not driven by outliers.
2. **Delta vs. k:** Tests whether the effect changes as we look at more neighbors (k=5, 10, 20, 50). The delta is strongest at k=5 and gradually decreases, which is expected — closer neighbors should be more biologically similar.
3. **Threshold fractions:** What fraction of neighbor pairs exceed various correlation thresholds (0.05, 0.10, 0.15, 0.20)? Neighbors consistently exceed thresholds more often than random.

### Controls and safeguards

- Uses the same saved pair data from the primary analyses — no recomputation of the base results.
- For delta-vs-k, kNN is recomputed from raw embeddings to test different k values.

---

## Analysis 8: CORUM Feature Interpretability

**Script:** `run_corum_interpretability.py`

### What it does

Moves from "the embedding works" to "how does it work?" Each variant fingerprint is a 64x64 matrix with 4,096 individual entries. This analysis asks: do specific matrix entries light up for specific protein complexes?

For each CORUM complex, it compares the gene-averaged matrix entries of complex members vs. all other genes using a Welch t-test, then applies false discovery rate (FDR) correction.

### Key result

- **105 of 300 complexes** have at least one significantly enriched matrix entry (FDR < 0.05).
- **122 matrix entries** are recurrently significant across 3 or more complexes, suggesting these entries encode broadly relevant protein interaction features.
- Split-half stability test: randomly splitting each complex in half and comparing the enriched entries between halves shows good reproducibility.

### Controls and safeguards

- **FDR correction (Benjamini-Hochberg)** applied per complex to control for testing 4,096 entries.
- **Minimum 5 embedded genes per complex** to ensure statistical power.
- **Z-scoring** per entry across all genes to normalize scale.
- **Variant-count bias check:** Correlation between a gene's variant count and its mean matrix values is low, confirming the results aren't driven by data quantity.
- **Split-half stability:** The top entries identified from one half of a complex's genes replicate in the other half.

---

## Analysis 9: Chronos Entry-Level Correlations

**Script:** `run_chronos_entry_analysis.py`

### What it does

Connects the interpretability analysis to gene essentiality. For each of the 4,096 matrix entries, it correlates the gene-level values with Chronos dependency metrics (mean dependency score and fraction of cell lines where the gene is essential). Entries that correlate strongly with dependency may encode essentiality-related features.

### Key result

- The strongest entry-level correlations are modest (|r| ~ 0.17), but highly significant after FDR correction.
- A Ridge regression model using all 4,096 features predicts mean Chronos dependency with R-squared = **0.17** — the embedding captures meaningful variance in gene essentiality.
- Row 31 of the 64x64 matrix is particularly enriched for dependency-associated entries.

### Controls and safeguards

- **FDR correction** across all 4,096 entries times 4 dependency metrics (Pearson and Spearman for mean and fraction).
- **Ridge regression with cross-validation** to prevent overfitting.
- **9,493 genes** with both embeddings and Chronos data.

---

## Analysis 10: CORUM vs. Chronos Feature Comparison

**Script:** `run_corum_vs_chronos_features.py`

### What it does

Asks: do the matrix entries that distinguish protein complexes (CORUM) overlap with the entries that predict gene essentiality (Chronos)? This tests whether the embedding uses shared or separate features for structural vs. functional biology.

### Key result

- Of the top 200 CORUM-associated and top 200 Chronos-associated features, **75 overlap** (Jaccard = 0.23). A permutation test shows this overlap is **8x more than expected by chance** (p < 0.001).
- Feature classification: **225 shared**, 185 CORUM-only, 185 Chronos-only, 3,501 background.
- Global rank correlation between CORUM and Chronos feature scores: Spearman rho = **0.78** (p ~ 0).
- CORUM-enriched features can predict Chronos dependency (R-squared = 0.08), and vice versa (R-squared = 0.09), demonstrating cross-predictive power.

### Controls and safeguards

- **Permutation test (2,000 iterations)** for overlap significance.
- **Separate training of CORUM and Chronos scores** — no circularity.
- **Feature classes are mutually exclusive** (shared/CORUM-only/Chronos-only/background).

---

## Analysis 11: Left vs. Right Projection Comparison

**Script:** `run_left_right_analysis.py`

### What it does

The 64x64 matrix can be summarized in three ways: average across rows ("left" 64-d vector), average across columns ("right" 64-d vector), or flatten the full matrix (4,096-d vector). This analysis compares these three "views" for their ability to capture CORUM and DepMap biology.

### Key result

- For CORUM: the left projection gives the strongest enrichment (3.07x at k=5), slightly above the full matrix (2.87x). The right projection is weaker (2.47x).
- For DepMap (at k=10): all three views show significant effects, with the right projection slightly stronger for DEMETER2 and the left projection slightly stronger for Chronos.

This suggests that the rows and columns of the matrix capture partially different biological signals.

### Controls and safeguards

- Same enrichment framework as Analysis 1.
- All three views use L2-normalized vectors for fair comparison.
- kNN recomputed separately for each view.

---

## Analysis 12: UMAP Visualization

**Script:** `plot_umap_gene_families.py`

### What it does

Creates a 2D visualization of all gene-level embeddings using UMAP (a dimensionality reduction technique). Gene family labels from HGNC are overlaid to show whether families cluster visually.

### Key result

A qualitative visualization showing that gene families form visible clusters in the embedding space, consistent with the quantitative enrichment results.

---

## Summary of Controls Across All Analyses

| Control | Purpose | Applied in |
|---|---|---|
| Cross-gene restriction | Prevents trivial same-gene matches | All neighbor analyses |
| Matched random controls | Ensures fair baseline comparison | All enrichment and DepMap analyses |
| Gene-level bootstrap | Avoids pseudoreplication from multi-variant genes | All CIs |
| Gene-collapse robustness | Confirms effect with 1 variant per gene | DepMap/Chronos |
| FDR correction (Benjamini-Hochberg) | Controls false discoveries in multiple testing | Interpretability, entry-level |
| Permutation tests | Tests overlap significance | Feature comparison |
| Split-half stability | Checks reproducibility within complexes | Interpretability |
| Fixed random seed (42) | Full reproducibility | All scripts |
| Config/environment logging | Enables exact replication | All scripts |
| Minimum size thresholds | Ensures statistical power | Complexes >= 3 genes, families >= 5, cell line overlap >= 50 |

---

## Relationship Between Analyses

```
Structural biology                   Functional biology
─────────────────                    ──────────────────
CORUM complexes ──────┐         ┌──── DEMETER2 (RNAi)
STRING interactions ──┤         ├──── Chronos (CRISPR)
HGNC gene families ───┘         │
                                │
          ┌─────────────────────┘
          │
     Embedding
     neighbors ──── Are they enriched for all of the above?
          │                   Answer: YES (2-3x for structure, significant for function)
          │
     Matrix entries ── Which specific features drive this?
          │                   Answer: 105/300 complexes have specific features; 225 features shared
          │                           between structure and function
          │
     Left/Right views ── Do rows and columns capture different biology?
                              Answer: Partially yes
```
