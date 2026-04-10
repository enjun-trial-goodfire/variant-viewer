# Evo2 Variant Embeddings Encode Real Biology

## Overview

We extracted 64x64 matrix fingerprints for 184,177 ClinVar variants from Evo2 via trained probes and tested whether they capture known biology. Nearest-neighbor analyses were validated against three independent structural databases and two independent functional datasets.

## Finding 1: Embedding neighbors are enriched for known biological relationships

Variants whose fingerprints are similar come from genes that share protein complexes, protein interactions, and gene families far more often than expected by chance.

| Ground truth | Enrichment at k=5 | 95% CI | Genes tested |
|---|---|---|---|
| **CORUM** protein complexes | **2.84x** | [2.26, 3.56] | 3,519 |
| **STRING** protein interactions | **2.54x** | [2.47, 2.62] | 12,759 |
| **HGNC** gene families | **3.37x** | [3.08, 3.72] | 9,410 |

All confidence intervals exclude 1.0 (no enrichment). All use gene-level bootstrap to avoid inflation from genes with many variants.

## Finding 2: Embedding neighbors share cancer dependency profiles

Genes with similar embeddings tend to be essential in the same cancer cell types. This holds in two independent experimental technologies:

| Dataset | Technology | Cell lines | Gene-level delta | 95% CI |
|---|---|---|---|---|
| **DEMETER2** | RNAi | 707 | 0.0016 | [0.0013, 0.0020] |
| **Chronos** | CRISPRi | 551* | 0.0053 | [0.0049, 0.0057] |

Deltas represent gene-level mean (neighbor correlation minus random correlation). Both CIs exclude zero. Effect is robust to: consequence type (missense, nonsense, splice, synonymous), number of neighbors (k=5 through 50), and collapsing to one variant per gene.

*Chronos restricted to DEMETER2-overlapping cell lines. Raw deltas are not directly comparable between datasets.

## Finding 3: Individual matrix features distinguish specific protein complexes

Of 4,096 matrix entries, specific entries are significantly enriched (FDR < 0.05) in 105 of 300 tested CORUM complexes. 122 entries are recurrently significant across 3+ complexes. These associations are stable in split-half validation.

## Finding 4: Structural and functional signals share latent features

Among the top 200 CORUM-associated and top 200 Chronos-associated matrix features, 75 overlap (8x more than chance, permutation p < 0.001). A Ridge model using all 4,096 features predicts mean Chronos gene dependency with R-squared = 0.17. The global correlation between CORUM and Chronos feature rankings is rho = 0.78.

## Key Caveats

- **Effect sizes for dependency similarity are small** (deltas of 0.002–0.005 on a correlation scale). The signal is real but subtle.
- **Association, not causation.** Embedding similarity may reflect shared sequence context rather than shared function.
- **Chronos uses only 551 of 1,208 available cell lines** (the DEMETER2 overlap). This discards 55% of Chronos data.
- **Entry-level correlations with dependency are modest** (|r| up to 0.18). Statistically significant at this sample size, but limited individual predictive power.
- **No external test set.** All evaluations use the same embedding; results reflect in-sample structure.

## Bottom Line

Evo2 variant fingerprints capture genuine biological structure at multiple levels — from physical protein complexes to functional cancer dependencies — validated across five independent databases with rigorous controls for gene-level pseudoreplication, matched random baselines, and multiple testing correction.
