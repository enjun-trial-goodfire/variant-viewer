# Evo2 Variant Embeddings: Findings Summary

## Overview

We extracted 64×64 matrix fingerprints for ~184,000 ClinVar variants from Evo2 via trained probes and tested whether they capture known biology. Nearest-neighbor analyses were validated against three structural databases and two functional datasets. Interpretability analyses examined which specific matrix features drive the signal. A forward-in-time prediction test assessed generalization to protein complexes discovered after model training.

All results below are taken from actual script outputs. Where effect sizes are small, we say so.

---

## Finding 1: Embedding neighbors are enriched for known biological relationships

Variants whose fingerprints are similar come from genes that share protein complexes, protein interactions, and gene families more often than expected by chance.

| Ground truth | Enrichment at k=5 | 95% CI | Genes tested |
|---|---|---|---|
| **CORUM** protein complexes | **2.84×** | [2.26, 3.56] | 2,643 source genes |
| **STRING** protein interactions | **2.54×** | [2.47, 2.62] | 12,759 |
| **HGNC** gene families | **3.37×** | [3.08, 3.72] | 8,858 annotated |

All confidence intervals exclude 1.0 (no enrichment). All use gene-level bootstrap (5,000 iterations) to avoid inflation from genes with many variants. Enrichment decreases monotonically with k, as expected — closer neighbors carry stronger signal.

**Permutation null (Q4):** Gene-label permutation (1,000 iterations) at k=10 yields a null fraction of 0.0026 ± 0.0001, vs. the observed 0.0085. The z-score is 39.8 (p = 0.001), confirming the enrichment is not an artifact of kNN graph structure.

**Caveat:** Fold enrichment looks impressive, but absolute sharing rates are low. At k=5 for CORUM, only 1.64% of neighbor pairs share a complex vs. 0.58% for random. The signal is real but sparse — most neighbor pairs are not in the same complex, even at the closest distances.

---

## Finding 2: Embedding neighbors share cancer dependency profiles

Genes with similar embeddings tend to be essential in the same cancer cell types. This holds in two independent experimental technologies.

| Dataset | Technology | Cell lines | Gene-level delta | 95% CI | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| **DEMETER2** | RNAi | 707 | 0.0016 | [0.0013, 0.0020] | 0.089 | [0.068, 0.110] |
| **Chronos** | CRISPRi | 551* | 0.0053 | [0.0049, 0.0057] | 0.338 | [0.315, 0.361] |

\*Chronos restricted to DEMETER2-overlapping cell lines; 55% of Chronos data discarded.

**Interpreting the numbers:**
- Deltas represent the gene-level mean difference in Pearson correlation between neighbor pairs and matched random pairs. Both CIs exclude zero.
- **Cohen's d** normalizes the raw delta by pooled standard deviation, enabling fair cross-dataset comparison. The Chronos signal (d = 0.34) is genuinely ~4× stronger than DEMETER2 (d = 0.09) — this is not a variance artifact.
- However, even the Chronos Cohen's d of 0.34 is a "small" effect by conventional standards (d = 0.2 is small, 0.5 is medium). The signal is statistically significant but practically modest.
- The delta-vs-k analysis confirms the expected pattern: effect is strongest at k=5 (DEMETER2: 0.0026, Chronos: 0.0059) and decreases to k=50 (DEMETER2: 0.0024, Chronos: 0.0046).

**Robustness:**
- Consistent across consequence types (missense, nonsense/splice, synonymous) in both datasets.
- Gene-collapse test (1 variant per gene, 30 repeats): DEMETER2 delta = 0.0014 ± 0.0008, Chronos delta = 0.0054 ± 0.0008 — the effect survives deduplication.

**Pathogenicity stratification (Q3):** Pathogenic variants show slightly stronger deltas than benign variants in both datasets (DEMETER2: 0.0021 vs. 0.0015; Chronos: 0.0068 vs. 0.0050). Both classes are independently significant. The pathogenic-benign difference is suggestive but the confidence intervals overlap, so this should not be over-interpreted.

---

## Finding 3: Individual matrix features distinguish specific protein complexes

Of 4,096 matrix entries tested across 300 CORUM complexes (1.2 million tests total):
- **105 of 300 complexes** have at least one significantly enriched entry (FDR < 0.05).
- **49,388 significant entry-complex pairs** total.
- **122 entries** are recurrently significant across 3+ complexes.

The most recurrent entry, position (31,16), appears in 17 complexes with a mean effect size of 1.60. The top-ranked complex by signature strength is the Spliceosome E complex (73.5% of entries significant, 62 embedded genes), followed by the BAF complex (58.3%, 9 genes) and the LARC complex (55.5%, 10 genes).

**Split-half stability** is high: for the top complexes, Spearman correlations between half-complex signatures range from 0.81 (Multisynthetase) to 0.93 (Spliceosome E), indicating the signatures are properties of the complexes, not sample artifacts.

**Caveat:** There is a weak but statistically significant correlation between variant count per gene and complex membership (Spearman r = 0.097). While small, this means we cannot entirely rule out that well-studied genes (which tend to have more variants) drive some of the enrichment signal.

---

## Finding 4: Structural and functional biology share latent features

Among the top 200 CORUM-associated and top 200 Chronos-associated matrix features:
- **75 overlap** (Jaccard = 0.23, 7.6× more than expected by chance, permutation p < 0.001).

Feature classification (top 10% by each score):
- **225 shared** (important for both CORUM and Chronos)
- **185 CORUM-only**
- **185 Chronos-only**
- **3,501 background**

Global rank correlation between CORUM and Chronos feature importance scores: Spearman ρ = **0.78** (p ≈ 0).

**Cross-predictive power (Ridge regression on Chronos mean dependency):**
- Full 4,096 features: R² = **0.171**
- Shared features only (225): R² = 0.102
- CORUM-only features (185): R² = 0.084
- Chronos-only features (185): R² = 0.094

The shared features alone achieve ~60% of the full model's predictive power. CORUM-only features can still weakly predict Chronos dependency (R² = 0.08), consistent with the idea that protein complex membership is partially predictive of gene essentiality.

**Caveat:** The R² = 0.17 for the full model means the embedding explains ~17% of variance in mean gene dependency. This is nontrivial for a sequence-derived feature, but 83% of the variance comes from other sources. The embedding is not a strong dependency predictor in isolation.

---

## Finding 5: The four biological datasets converge on shared latent features

Extending the CORUM-vs-Chronos comparison to all four datasets (CORUM, STRING, Chronos, DEMETER2):

| Comparison | Type | Spearman ρ | Top-200 overlap | Fold over chance |
|---|---|---|---|---|
| Chronos vs. DEMETER2 | within-functional | **0.92** | 122 | 12.5× |
| CORUM vs. STRING | within-structural | **0.81** | 68 | 7.0× |
| STRING vs. DEMETER2 | cross-type | 0.79 | 82 | 8.4× |
| CORUM vs. Chronos | cross-type | 0.78 | 75 | 7.7× |
| CORUM vs. DEMETER2 | cross-type | 0.77 | 76 | 7.8× |
| STRING vs. Chronos | cross-type | 0.77 | 82 | 8.4× |

Within-type comparisons are strongest (ρ = 0.81–0.92), but cross-type comparisons still show substantial overlap (ρ = 0.77–0.79). All overlaps far exceed chance expectation (7–12× fold enrichment).

**Interpretation:** The same latent features tend to be important across all four databases, suggesting the embedding captures a coherent biological signal rather than dataset-specific artifacts. The within-functional agreement (Chronos–DEMETER2, ρ = 0.92) is particularly strong, as expected for two assays measuring the same concept (gene essentiality) via different technologies.

---

## Finding 6: Left and right matrix projections capture partially different biology

The 64×64 matrix can be summarized as row means (left, 64-d), column means (right, 64-d), or the full flatten (4,096-d):

| View | CORUM fold (k=10) | DEMETER2 delta | Chronos delta |
|---|---|---|---|
| Left (64-d) | 2.58× | 0.0015 | 0.0034 |
| Right (64-d) | 2.35× | 0.0016 | 0.0037 |
| Full (4,096-d) | **2.79×** | 0.0015 | **0.0042** |

The full matrix is strongest or near-strongest across all readouts, indicating that the left-right interaction structure carries information beyond what either marginal captures alone. Both marginals independently carry significant signal. The matrix is asymmetric (left ≠ right), confirming that rows and columns encode partially different biological information.

---

## Finding 7: Embedding features generalize to newly discovered protein complexes

The forward-in-time CORUM prediction analysis (v3 → v5) tests whether the embedding can predict complex membership that was discovered after model training.

**Database comparison:** CORUM v3 has 1,947 human complexes; current (v5) has 2,589. There are 822 current-only complexes and 3,758 novel co-complex gene pairs.

**Four prediction tests:**

| Test | Result | p-value | Effect size |
|---|---|---|---|
| A: kNN enrichment for novel pairs | Novel pairs found among neighbors 2.8× more than random (k=10) | 4.0 × 10⁻¹⁸ | OR = 2.83 |
| B: Cosine similarity | Novel pairs closer than random in embedding space | 1.3 × 10⁻⁴ | rank-biserial r = −0.046 |
| C: v3-derived feature scores (corrected) | v3 features score novel pairs higher than random | 9.1 × 10⁻⁶⁵ | Cohen's d = 0.48 |
| D: Complex-level enrichment | 32% of 63 novel complexes (≥5 genes) show significant signal | — | — |

Test C is the most important: it uses features derived *solely* from the older v3 database to score pairs that appear only in v5 — a non-circular forward prediction. The Cohen's d of 0.48 (approaching "medium" by convention) with p = 9 × 10⁻⁶⁵ is the strongest evidence that the embedding's learned features generalize beyond the training-era annotations.

**Feature stability:** The v3-derived and v5-derived recurrent features overlap substantially (Jaccard = 0.71, Spearman effect-size correlation = 0.99), indicating that the same latent positions are predictive across a decade of database updates.

**Caveats:**
- Test B's rank-biserial r is very small (−0.046), meaning the cosine similarity distributions overlap substantially.
- Test D shows only 32% of novel complexes (at the ≥5 gene threshold) have any significant entries, suggesting the model does not capture all new biology equally.
- Gene name mappings may have changed between CORUM versions.

---

## Per-family enrichment highlights

The gene family analysis identifies specific families the embedding captures particularly well:

| Family | Lift at k=10 | Family size |
|---|---|---|
| Toll-like receptors | 31.1× | 10 |
| Tubulin beta | 30.8× | 10 |
| Pregnancy-specific glycoproteins | 25.4× | 10 |
| Mucins | 20.9× | 14 |
| Keratins, type II | 12.8× | 26 |
| NLR family | 12.7× | 20 |
| Gap junction proteins | 10.7× | 20 |

However, many families show zero enrichment (e.g., BCL2 family, claudins, actin-related proteins), demonstrating that the embedding does not universally capture all gene families.

---

## Key Caveats (collected)

1. **Effect sizes for dependency similarity are small.** Cohen's d ranges from 0.09 (DEMETER2) to 0.34 (Chronos). The signal is statistically significant at these sample sizes but practically modest. Do not over-interpret.

2. **Association, not causation.** Embedding similarity may reflect shared sequence context, regulatory proximity, or evolutionary relatedness rather than shared molecular function per se.

3. **Chronos uses only 551 of 1,208 available cell lines** (the DEMETER2 overlap set). The full Chronos dataset might show different effect sizes.

4. **Entry-level correlations with dependency are modest** (|r| up to 0.18). Individually, no single matrix feature is a strong dependency predictor.

5. **No external test set for the embedding itself.** All evaluations use the same embedding; results reflect in-sample structure. The forward-prediction analysis (Finding 7) partially addresses this for CORUM, but the embedding model was not retrained.

6. **The R² = 0.17 for predicting gene dependency** from the full embedding means 83% of variance is unexplained. The embedding contributes, but is far from a complete picture.

7. **Variant-count bias** exists weakly (r = 0.097 between variant count and complex membership). Well-characterized genes have more variants and may be more likely to appear in databases like CORUM.

8. **The gene family signal is heterogeneous.** Some families show >30× enrichment; many show zero. The embedding captures certain types of biology (structurally similar proteins, immune receptors, cytoskeletal components) much better than others.

9. **Not all novel complexes are detected.** In the forward-prediction analysis, only 32% of novel complexes (≥5 embedded genes) show any significant enrichment signal. The embedding misses the majority of new biology at this stringency.

10. **Permutation test limitations.** The Q4 permutation null (z = 39.8) is extremely significant, but uses gene-label shuffling which may not capture all potential confounders (e.g., chromosomal proximity, gene length correlations).

---

## Bottom Line

Evo2 variant fingerprints capture genuine biological structure at multiple levels — from physical protein complexes to functional cancer dependencies — validated across five independent databases with rigorous controls. The signal is consistent, replicable across technologies, and generalizes to protein complexes discovered after training. However, effect sizes for functional similarity are small (d < 0.35), predictive power for gene dependency is moderate (R² ≈ 0.17), and the embedding does not capture all types of biology equally. These results are best interpreted as evidence that genomic language models learn meaningful biological representations, not as evidence of strong individual-level prediction.
