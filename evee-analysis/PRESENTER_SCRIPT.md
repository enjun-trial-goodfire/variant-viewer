# Presenter Script: Evo2 Embedding Analysis Figures

*Suggested slide order and talking points for each figure. Times are approximate. Adapt tone and depth for your audience.*

---

## Slide 1 — Figure 1: Structural Enrichment

**File:** `fig1_structural_enrichment.png`

**Setup (30 sec):**
"We start with a basic question: do genes that are close together in the Evo2 embedding space actually share known biological relationships? We test this against three independent ground-truth databases — CORUM protein complexes, STRING protein-protein interactions, and HGNC gene families. For each gene, we find its k nearest neighbors in embedding space, then ask how often those neighbors share a relationship compared to random gene pairs."

**Left panel — Absolute rates (30 sec):**
"The left panel shows the raw numbers, which is important for honest interpretation. For STRING, about 12% of neighbor pairs share a known interaction at k=5, versus roughly 5% for random — so the absolute rate is modest. For gene families it's about 4% versus 1.3%. For CORUM co-complex membership, it's 1.6% versus 0.6%. These are real but small absolute differences."

**Right panel — Fold enrichment (30 sec):**
"The right panel normalizes these as fold enrichment over the random baseline. All three databases show 2.5–3.4× enrichment at k=5, decaying toward 2× as we include more distant neighbors. The key takeaway is that this enrichment is consistent across three completely independent biological databases, which makes it unlikely to be an artifact of any single annotation source."

**Transition:**
"So the embedding captures structural and family relationships. Next question: does it also capture functional biology?"

---

## Slide 2 — Figure 2: Dependency Profile Similarity

**File:** `fig2_dependency_delta.png`

**Setup (20 sec):**
"We now test a different kind of biology — cancer dependency. If two genes are neighbors in embedding space, do they tend to have similar essentiality profiles across hundreds of cancer cell lines? We test this with two independent datasets: DEMETER2, which uses RNAi across 707 cell lines, and Chronos, which uses CRISPR across 551 cell lines."

**Left panel — Absolute values (30 sec):**
"The left panel shows the actual mean profile correlations. For DEMETER2, neighbor pairs have a mean correlation of 0.0109 versus 0.0088 for random — both are very close to zero. For Chronos, it's 0.0083 versus 0.0020. I want to be upfront: these are tiny correlations in absolute terms. We are not claiming that individual neighbor pairs have strong functional similarity."

**Right panel — Effect size (30 sec):**
"The right panel shows the delta — the difference between neighbor and random means — with bootstrap confidence intervals. DEMETER2 shows a delta of 0.0016 and Chronos shows 0.0053, both significantly above zero. The signal here is a consistent population-level shift: across thousands of genes, neighbors are systematically slightly more similar than random. It's a weak but real and reproducible effect."

**Transition:**
"Chronos appears to show a stronger signal. But is that a real biological difference or just an artifact of using fewer cell lines?"

---

## Slide 3 — Supplementary: Cohen's d Comparison (Q1)

**File:** `q1_cohens_d_comparison.png`

**Talking points (45 sec):**
"To fairly compare the two datasets, we compute Cohen's d, which normalizes the delta by the pooled standard deviation — this controls for different variance scales caused by different numbers of cell lines. DEMETER2 gives d = 0.089, a small effect. Chronos gives d = 0.338, a small-to-medium effect. The right panel shows the bootstrap distributions are completely non-overlapping. So the stronger Chronos signal is genuine, not a variance artifact. This makes biological sense: CRISPR knockouts produce cleaner loss-of-function phenotypes than RNAi knockdowns, so we'd expect the dependency signal to be more specific."

---

## Slide 4 — Figure 3: Signal Decay with Distance

**File:** `fig3_delta_vs_k.png`

**Talking points (30 sec):**
"This figure shows how the dependency signal changes as we include more distant neighbors. For both datasets, the effect is strongest at k=5 — the very closest neighbors — and decays as we go to k=50. This is exactly what we'd expect if the embedding has a meaningful distance metric: closer neighbors are more biologically similar, and the signal dilutes as we pull in weaker matches. The decay is smooth and monotonic, which argues against it being noise."

---

## Slide 5 — Supplementary: Pathogenicity Stratification (Q3)

**File:** `q3_pathogenicity_stratification.png`

**Talking points (45 sec):**
"A natural question is whether this signal is driven by pathogenic or benign variants. We stratify the dependency analysis by ClinVar pathogenicity class. In both datasets, pathogenic variants show a stronger dependency signal than benign variants — about 40% stronger in DEMETER2 and 36% stronger in Chronos. The confidence intervals partially overlap within each dataset, so this is suggestive rather than definitive. But the consistent direction across both datasets is encouraging: it suggests the embedding captures something about variant-level functional impact, not just gene identity."

---

## Slide 6 — Supplementary: Permutation Null (Q4)

**File:** `q4_permutation_null.png`

**Talking points (30 sec):**
"To confirm the structural enrichment isn't an artifact of graph structure or gene frequency, we ran a permutation test. We shuffled gene labels across variants 1,000 times, preserving the kNN graph topology but destroying the gene-biology mapping. The observed co-complex fraction is 0.85%, while the permutation null averages 0.26% — a z-score of 40. None of the 1,000 permutations came anywhere close to the observed value. This confirms the enrichment is driven by real biological signal in the embeddings, not by structural properties of the kNN graph."

---

## Slide 7 — Figure 4: Feature Interpretability

**File:** `fig4_feature_interpretability.png`

**Setup (20 sec):**
"Now we shift from 'does it work' to 'can we understand what it learned.' The Evo2 embedding is a 64×64 covariance matrix — that's 4,096 individual features. We ask: for each protein complex, which specific matrix entries distinguish its member genes from non-members?"

**Panel (a) — Recurrence heatmap (20 sec):**
"The left panel shows recurrence: how many of the 300 tested complexes have a significant signal at each matrix entry. Most entries are significant for zero or one complex — the matrix is mostly 'quiet.' But a handful of entries light up for 10–17 complexes, meaning they carry broad structural information."

**Panel (b) — BAF complex example (20 sec):**
"The right panel zooms in on one complex — BAF, a chromatin remodeling complex. Red entries have higher values inside the complex; blue entries have lower values. The sparse pattern of significant entries shows that the embedding encodes this complex through a specific, interpretable subset of features, not through a diffuse global signal."

---

## Slide 8 — Figure 5: Feature Overlap and Cross-Prediction

**File:** `fig5_feature_overlap.png`

**Setup (20 sec):**
"We've shown the embedding captures both structural (CORUM) and functional (Chronos) biology. Are these the same features or different ones?"

**Panel (a) — Classification (20 sec):**
"We rank all 4,096 matrix entries by their CORUM importance score and separately by their Chronos correlation score, then take the top 10% of each. Of the 4,096 entries, 225 are in the top 10% for both — these are shared features. 185 are important only for CORUM, and 185 only for Chronos. The vast majority — 3,501 — are background."

**Panel (b) — Cross-prediction (30 sec):**
"The more interesting question is whether CORUM features can predict Chronos dependency. Using all 4,096 features gives R² = 0.171. The 225 shared features alone give R² = 0.102 — almost 60% of the full model's performance with only 5% of the features. Even the 185 CORUM-only features predict Chronos at R² = 0.084. This means structural features carry functional information and vice versa — the two types of biology are partially encoded in overlapping regions of the embedding."

---

## Slide 9 — Supplementary: Per-Complex Ranking (Q2)

**File:** `q2_complex_ranking.png`

**Talking points (30 sec):**
"Which complexes have the strongest embedding signatures? The Spliceosome E complex leads with 73% of matrix entries showing significant differences — but it's also the largest complex at 62 genes, so it has the most statistical power. More interesting are the BAF and cBAF chromatin remodeling complexes, which rank 2nd and 5th with only 9 genes each but 50–58% significant entries. These are biologically important complexes frequently mutated in cancer, and the embedding has learned a strong representation of them."

**Optional:** Show `q2_complex_size_vs_enrichment.png` as backup if asked about size confounds — it shows the scatter of enrichment vs complex size with no simple linear relationship.

---

## Slide 10 — Figure 6: UMAP

**File:** `fig6_umap.png`

**Talking points (30 sec):**
"This is a UMAP projection of the full 64×64 gene-level embeddings, colored by the seven largest HGNC gene families. You can see that genes from the same family tend to cluster together — zinc fingers in red form a distinct region, KRAB domain proteins in blue cluster nearby, and CD molecules in green form their own neighborhood. This is a qualitative visualization, but it reinforces the quantitative results: the embedding organizes genes by biological function and structural properties."

---

## Slide 11 — Figure 7a: Feature Maps (Clustered)

**File:** `fig7_feature_maps.png`

**Setup (20 sec):**
"Finally, we visualize the full 64×64 importance landscape. Each pixel represents one matrix entry. The rows and columns have been reordered by hierarchical clustering to reveal block structure."

**Panels (30 sec):**
"Panel (a) shows CORUM importance — you can see clear blocks of features that are important for complex membership. Panel (b) shows Chronos importance — a different spatial pattern, with some overlap. Panel (c) is the normalized difference: red regions are relatively more important for structural biology, blue for functional biology. The block structure confirms that the embedding organizes its 4,096-dimensional representation into functionally distinct modules."

---

## Slide 12 — Figure 7b: Feature Maps (Original Order)

**File:** `fig7b_feature_maps_original.png`

**Talking points (20 sec):**
"For comparison, this is the same data without the hierarchical clustering — the rows and columns are in their original index order. You can still see some structure, but it's much less coherent. This illustrates two things: first, the block structure is real, not an artifact of the clustering algorithm. Second, the embedding dimensions don't have an inherent spatial ordering — the clustering is revealing latent organization that's there but not aligned to the index numbers."

---

## Closing Summary (1 min)

"To summarize:

1. **The embedding captures known biology.** Nearest neighbors are 2–3× enriched for CORUM, STRING, and gene family relationships, confirmed by a permutation test with z = 40.

2. **It also captures functional biology.** Neighbors share cancer dependency profiles with a small but highly significant and reproducible effect, stronger in CRISPR (Chronos) than RNAi (DEMETER2), and stronger for pathogenic than benign variants.

3. **The signal is distance-sensitive.** It's strongest for the closest neighbors and decays smoothly — consistent with a meaningful distance metric.

4. **The representation is interpretable.** Individual matrix entries can be mapped to specific complexes. Structural and functional features partially overlap — CORUM features predict Chronos dependency, suggesting the embedding captures a shared biological signal.

5. **Important caveat:** The absolute effect sizes are small. We are not claiming strong per-pair predictions. The value is in the *consistency* of a weak signal across multiple independent biological validations."

---

## Suggested Slide Order

| Slide | Figure | Purpose |
|-------|--------|---------|
| 1 | Fig 1 | Structural enrichment (main result) |
| 2 | Fig 2 | Dependency signal (main result) |
| 3 | Q1 | Cohen's d — fair comparison (supplementary) |
| 4 | Fig 3 | Distance decay (supporting) |
| 5 | Q3 | Pathogenicity stratification (supplementary) |
| 6 | Q4 | Permutation null (supplementary) |
| 7 | Fig 4 | Feature interpretability (mechanistic) |
| 8 | Fig 5 | Feature overlap (mechanistic) |
| 9 | Q2 | Per-complex ranking (supplementary) |
| 10 | Fig 6 | UMAP visualization (qualitative) |
| 11 | Fig 7a | Feature maps — clustered (mechanistic) |
| 12 | Fig 7b | Feature maps — original (comparison) |

Slides 3, 5, 6, and 9 can be moved to backup/appendix for shorter presentations. The core story is Slides 1, 2, 4, 7, 8, 11 — about 6 slides.
