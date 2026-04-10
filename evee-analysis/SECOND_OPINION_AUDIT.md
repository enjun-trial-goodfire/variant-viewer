# Second-Opinion Technical Audit (Updated)

**Date:** 2026-04-09
**Scope:** All 21 scripts in `evee-analysis/scripts/` and directly referenced helpers.
**Purpose:** Pre-presentation review — skeptical, concrete, conservative.
**Supersedes:** Previous audit from earlier on 2026-04-09 (which covered 18 scripts; 3 were written after that audit).

---

## Executive Summary

The analysis pipeline is **well-engineered and largely correct**. Seed enforcement, config logging, cross-gene pair restrictions, and gene-level bootstrap are all properly implemented. The headline enrichment results (CORUM ~2.8×, STRING ~2.5×, gene families ~3.4× at k=5) are methodologically sound and reproducible.

This updated audit extends coverage to three previously unaudited scripts (`run_corum_forward_prediction.py`, `run_followup_dataset_comparison.py`, `run_quick_analyses.py`) and documents additional issues found across the codebase.

**Issues found and fixed in this round (all verified by rerun):**

1. **[Critical]** Random control under-sampling in `run_followup_analyses.py` delta-vs-k — Chronos random pairs were ~3.2% fewer than neighbor pairs due to a loop bug. Fixed; deltas change <3% and headline claims hold.
2. **[Important]** 15 of 21 scripts used the wrong directory name (`eeve-analysis` instead of `evee-analysis`). No symlink existed, so these scripts would fail on any clean clone. Fixed across all 15 files.
3. **[Important]** Dead permutation code and low permutation count in `run_quick_analyses.py` Q4. Fixed: removed dead code, increased from 200 to 1000 permutations.
4. **[Important]** Non-deterministic `DATE_TAG` in figure filenames in two scripts. Fixed with a fixed string.
5. **[Important]** STRING ego-network caveat missing from dataset comparison report. Added.

**Previously fixed issues (from prior audit round, verified in place):**

1. **[Critical]** Pair-level/CI mismatch in DEMETER2 and Chronos reports — delta now uses gene-level bootstrap mean.
2. **[Important]** Misleading "2-3× enrichment" cross-reference in CORUM retrieval report — corrected.
3. **[Important]** Chi-squared pseudocount in `run_corum_vs_chronos_features.py` — removed.
4. **[Minor]** Reproducibility infrastructure added to `run_followup_analyses.py`.
5. **[Minor]** STRING gene names uppercased for safe matching.
6. **[Minor]** Chronos cell-line caveat added to README.

---

## Issue List

### Critical

#### C1. Point estimate inconsistent with bootstrap CI in DEMETER2 and Chronos reports
**Status:** FIXED (prior round)
**Files:** `run_neighbor_depmap_analysis.py`, `run_neighbor_chronos_analysis.py`

The pair-level delta was reported as the headline, but the CI came from gene-level bootstrap. These are different estimators. The corrected deltas are DEMETER2: 0.0016 (was 0.0021), Chronos: 0.0053 (was 0.0063). Both CIs exclude zero.

#### C2. Random control under-sampling in `run_followup_analyses.py` delta-vs-k
**Status:** FIXED (this round)
**File:** `run_followup_analyses.py` lines 250–260

The inner retry loop for random gene sampling broke out on the first non-self gene, regardless of whether `pairwise_profile_corr` returned None. This caused systematic under-counting of random controls when profiles had insufficient overlap or zero variance.

**Impact before fix (Chronos):**

| k | n_neighbor | n_random (before) | shortfall | n_random (after) |
|---|---|---|---|---|
| 5 | 445,505 | 430,961 | −3.3% | 445,505 |
| 10 | 901,414 | 872,341 | −3.2% | 901,414 |
| 20 | 1,816,855 | 1,758,003 | −3.2% | 1,816,855 |
| 50 | 4,570,431 | 4,422,588 | −3.2% | 4,570,431 |

DEMETER2 was affected minimally (<0.03% shortfall). After fix, n_random == n_neighbor exactly at all k values.

**Impact on deltas:** Changes <3% (e.g., Chronos k=10: 0.005419→0.005550). No headline claims change.

### Important

#### I1. Misleading lift claim in `corum_retrieval_report.md`
**Status:** FIXED (prior round)
The retrieval report cited "2–3× enrichment" from the enrichment analysis; corrected to cite its own 1.6× precision lift.

#### I2. Chronos vs DEMETER2 comparison uses different cell line sets — not clearly flagged
**Status:** FIXED (prior round)
Caveat added to `chronos_README_results.md` about 551/1208 cell line overlap.

#### I3. Random controls use subtly different gene universe than neighbors in DepMap scripts
**Status:** NOT A BUG
Both neighbor and random pairs are effectively restricted to DEMETER2/Chronos genes via filtering. No leakage.

#### I4. Chi-squared pseudocount in `run_corum_vs_chronos_features.py`
**Status:** FIXED (prior round)
Removed `+ 0.5` pseudocount; added sparse-data handling.

#### I5. 15 scripts use wrong directory name `eeve-analysis` (directory is `evee-analysis`)
**Status:** FIXED (this round)
All 15 affected scripts updated: path string corrected and variable renamed from `EEVE_ROOT` to `EVEE_ROOT` for consistency with the 4 newer scripts. No symlink existed, so these scripts would crash immediately on any clean environment.

**Files fixed:** `inspect_db.py`, `plot_umap_gene_families.py`, `run_chronos_entry_analysis.py`, `run_corum_full.py`, `run_corum_interpretability.py`, `run_corum_retrieval.py`, `run_corum_vs_chronos_features.py`, `run_followup_analyses.py`, `run_gene_family_analysis.py`, `run_left_right_analysis.py`, `run_neighbor_chronos_analysis.py`, `run_neighbor_depmap_analysis.py`, `run_string_analysis.py`, `run_string_retrieval.py`, `verify_determinism.py`

#### I6. Dead permutation code and low permutation count in `run_quick_analyses.py` Q4
**Status:** FIXED (this round)

`q4_permutation_null()` created `perm_knn = knn_indices.copy()` and shuffled it each iteration but never used it — the actual permutation was via `gene_list` shuffling, which is correct. The dead code suggested an incomplete/confused design. Additionally, 200 permutations gave minimum p-value resolution of 1/201 ≈ 0.005.

**Fix:** Removed dead `perm_knn` code, simplified to direct `gene_list` shuffle, increased to 1000 permutations.

**Impact:** z-score changed from 38.07 to 39.75 (more precise null estimate). p-value resolution improved from 0.005 to 0.001. Conclusion unchanged (overwhelmingly significant).

#### I7. Non-deterministic `DATE_TAG` in figure filenames
**Status:** FIXED (this round)
Both `run_followup_dataset_comparison.py` and `run_quick_analyses.py` used `time.strftime("%Y%m%d")` for figure filenames. Running on different days produced different filenames, breaking checksum verification via `verify_determinism.py`. Fixed to `"20260409"`.

#### I8. STRING ego-network caveat missing from dataset comparison
**Status:** FIXED (this round)
STRING "gene groups" are ego-networks (a gene plus its high-confidence interaction partners), structurally very different from CORUM's curated multi-gene complexes. The report's auto-generated interpretation now explicitly notes this when reporting within-structural comparisons.

### Minor

#### M1. Bootstrap attribution asymmetry in `run_corum_retrieval.py`
**Status:** DOCUMENTED
Undirected pairs attributed to lex-smaller gene only. Valid but reduces effective N.

#### M2. Different metrics across CORUM retrieval vs enrichment
**Status:** DOCUMENTED
Retrieval (precision/recall/F1) vs enrichment (fold/odds) measure different things. Should not be conflated.

#### M3. `run_followup_analyses.py` reproducibility infrastructure
**Status:** FIXED (prior round)
Now imports `enforce_seeds` and saves `followup_run_config.json`.

#### M4. STRING gene names not uppercased
**Status:** FIXED (prior round)

#### M5. Multiple scripts duplicate `enforce_seeds()` instead of importing
**Status:** DOCUMENTED — optional improvement, not blocking.

#### M6. Split-half baseline in `run_corum_interpretability.py`
**Status:** DOCUMENTED — uses global mean instead of "out" group, minor inconsistency.

#### M7. `checksums.sha256` does not recurse into subdirectories
**Status:** DOCUMENTED — all current outputs are flat, so no practical impact.

#### M8. `run_corum_forward_prediction.py` Test C uses |z_a + z_b| as pair score
**Status:** DOCUMENTED — design choice, not a bug.
The metric `np.average(np.abs(va + vb), weights=v3_feature_weights)` favors same-sign co-activation. Cosine similarity at selected features would be more standard. Should be documented if cited.

#### M9. `run_quick_analyses.py` Q3 uses `replace_strict` with `default=None`
**Status:** NOT A BUG — `default=None` correctly handles missing keys.

---

## Cohort Consistency Table

| Analysis | Input cohort | Gene universe | Variant universe | Background/random | Normalization | Cell lines |
|---|---|---|---|---|---|---|
| `run_neighbor_depmap_analysis` | DuckDB variants with embeddings + neighbors + DEMETER2 gene | Genes in DEMETER2 ∩ embedding index | Variants with consequence bin + embedding + DEMETER2 gene + neighbors | Matched random: same consequence bin, different gene from analysis table genes | Raw Pearson correlation | DEMETER2: 707 |
| `run_neighbor_chronos_analysis` | Same as above but Chronos genes | Genes in Chronos ∩ embedding index | Same filters but Chronos gene match | Same as above | Raw Pearson correlation | Chronos overlap with DEMETER2: 551 |
| `run_followup_analyses` | Reads saved parquet from above two + recomputes kNN from embeddings | Same as above per dataset | Same as above | Same + delta-vs-k recomputes kNN from embeddings | Raw Pearson correlation | DEMETER2: 707, Chronos: 551 |
| `run_corum_full` | All DuckDB variants with embeddings + neighbors | CORUM genes (≥3 gene complexes) ∩ DB genes | All variants with embeddings | Matched random: same count per source gene, from CORUM genes | Fold enrichment of sharing rate | N/A |
| `run_corum_retrieval` | kNN cache from `run_corum_full` | CORUM genes ∩ kNN graph genes | kNN variant set | No random — binary retrieval TP/FP/FN | Precision/recall/F1 | N/A |
| `run_string_analysis` | Same kNN cache as corum_full | STRING genes ∩ kNN graph genes | kNN variant set | Matched random: same count, from STRING genes | Fold enrichment | N/A |
| `run_string_retrieval` | Same kNN cache | STRING genes ∩ kNN graph genes | kNN variant set | No random — binary retrieval | Precision/recall/F1 | N/A |
| `run_gene_family_analysis` | Same kNN cache | HGNC-annotated genes (≥5 per group) ∩ kNN genes | kNN variant set | Matched random: same count, from annotated genes | Fold enrichment | N/A |
| `run_corum_interpretability` | Deconfounded covariance embeddings | Genes with ≥3 variants and embedding | Gene-averaged 64×64 matrices | Welch t-test: complex members vs all other genes | Z-scored per entry across genes | N/A |
| `run_chronos_entry_analysis` | Gene-level matrices + Chronos | Genes with ≥3 variants and Chronos profile (551 cell lines) | Gene-averaged | Pearson/Spearman correlation; Ridge regression | Z-scored | Chronos overlap: 551 |
| `run_corum_vs_chronos_features` | Outputs from interpretability + chronos entry analysis | Same 9,493 genes | 4096 matrix entries | Permutation test for overlap | Normalized scores | N/A |
| `run_left_right_analysis` | Deconfounded covariance embeddings (3 views) | CORUM genes + DepMap genes | Variants with embeddings and neighbors | CORUM: matched random; DepMap: matched random | L2-normalized views | DEMETER2: 707, Chronos: 551 |
| `run_corum_forward_prediction` | gene_level_matrices.npz + CORUM v3 zip + CORUM v5 JSON + corum_full_knn_indices.npz + DuckDB | Genes with ≥3 embedded variants in kNN graph | Gene-level matrices | Novel pairs = v5 co-complex minus v3 co-complex, restricted to embedded genes. Random = 5000 non-co-complex pairs from kNN genes | Feature-weighted scoring (|z_a + z_b|) | N/A |
| `run_followup_dataset_comparison` | gene_level_matrices.npz + precomputed CORUM/Chronos scores + DEMETER2 CSV + STRING data | 9,493 genes with ≥3 variants | Gene-level matrices | Spearman correlation, top-N overlap, fold enrichment across 4 datasets | Scores differ per dataset (structural = Welch t-test enrichment, functional = |Pearson r|) | N/A |
| `run_quick_analyses` | Precomputed pair parquets (DEMETER2/Chronos), CORUM enrichment, DuckDB, kNN cache | Q1: same as DepMap/Chronos analyses; Q2: CORUM genes with ≥10 embedded genes; Q3: adds ClinVar significance; Q4: kNN graph + CORUM membership | Varies by question | Q1: gene-level bootstrap; Q2: per-complex ranking; Q3: stratified by pathogenicity; Q4: label-permutation test (1000 perms) | Q1: Cohen's d; Q2: fraction significant; Q3: gene-level delta; Q4: fraction co-complex | N/A |

---

## Chronos vs DEMETER2 Assessment

### Is comparing raw deltas fair?

**Partially.** DEMETER2 uses all 707 cell lines; Chronos is filtered to 551 (the DEMETER2 overlap). Fewer cell lines can inflate correlation magnitudes (smaller N → more variable correlations with fatter tails). A larger raw delta for Chronos could partly reflect this artifact.

### Best comparison metric

The Q1 analysis in `run_quick_analyses.py` computes **Cohen's d** for both datasets, which is the correct normalized effect size for cross-dataset comparison:
- **DEMETER2:** d = 0.089 [0.068, 0.110]
- **Chronos:** d = 0.338 [0.315, 0.361]

Chronos shows a ~3.8× larger normalized effect. This is a fair comparison because Cohen's d controls for variance differences. However, the 551 vs 707 cell line difference in the underlying profiles remains a confounder for the correlation distributions that feed into the d calculation.

### Is "Chronos stronger than DEMETER2" justified?

**With caveats.** The Cohen's d comparison is the fairest available metric and shows a substantial difference. But the cell line count difference remains a confounder. Safe wording: "The signal is present in both datasets and substantially larger in Chronos (Cohen's d = 0.34 vs 0.09), though this comparison is confounded by different cell line counts."

### Raw delta plots

The `run_followup_analyses.py` delta-vs-k plot shows raw (unnormalized) deltas side-by-side. These should carry a prominent caveat about differing cell line counts. Present the Cohen's d comparison (Q1) as the primary fair comparison.

---

## CORUM vs DepMap Assessment

### Are these compared as if they were the same target?

No — the analyses properly separate:
- **CORUM enrichment** (complex co-membership): a structural biology signal
- **DepMap profile correlation** (dependency co-essentiality): a phenotypic signal
- **CORUM vs Chronos feature comparison** (`run_corum_vs_chronos_features.py`): explicitly measures overlap and divergence

### Complex membership vs dependency phenotype

The distinction is properly maintained throughout. The feature comparison script classifies features as "shared", "CORUM-only", or "Chronos-only" and tests cross-predictive power.

### Shared-latent-feature claims

The overlap analysis uses permutation tests. The interpretation appropriately uses conditional language. **One sensitivity:** The CORUM "score" is `recurrence × mean|effect_size|` — a composite. Different compositions could change overlap statistics. Note as a sensitivity.

### STRING comparison caveat

STRING groups are ego-networks (gene + interaction partners), structurally different from CORUM's multi-gene complexes. The comparison measures whether the same matrix entries are enriched in both contexts, not whether group definitions are equivalent. This caveat is now included in the report.

---

## Statistical Details

### Bootstrap unit
- **CORUM / STRING / gene family enrichment:** Gene-level. ✓
- **CORUM / STRING retrieval:** Gene-level on micro metrics (pairs attributed to lex-smaller gene). ✓
- **DepMap / Chronos neighbor analysis:** Gene-level. ✓
- **Quick analyses Q1/Q3:** Gene-level. ✓

### Multiple testing correction
- **CORUM interpretability (stage 3):** BH FDR per complex. ✓
- **Chronos entry analysis (stage 2):** BH FDR across 4096 entries × 4 metrics. ✓
- **Forward prediction:** Per-test FDR within each comparison. ✓
- **Cross-script:** No correction across the entire suite. Acceptable for exploratory analysis but should be noted.

### Treatment of NaNs
- Profile correlations: pairs with < MIN_OVERLAP non-NaN values dropped. ✓
- Z-scoring: entries with std < 1e-10 get std set to 1.0. ✓
- Bootstrap: NaN samples filtered before CI computation. ✓

### Variance thresholds
- `pairwise_profile_corr`: std < 1e-10 → return None. ✓
- Z-scoring: sigma < 1e-10 → set to 1.0. ✓

### Minimum overlap thresholds
- DepMap/Chronos: MIN_OVERLAP = 50 cell lines. ✓
- CORUM complexes: ≥3 genes. ✓
- CORUM interpretability: ≥5 embedded genes per complex. ✓
- Gene families: ≥5 genes in kNN. ✓
- Forward prediction: ≥3 embedded variants per gene. ✓

### CI consistency with point estimates
All bootstrap CIs bracket their point estimates. No inconsistencies found after prior fix C1.

---

## Reproducibility Details

### Fixed seeds
All scripts use `RANDOM_SEED = 42`. ✓
Per-k evaluations use `np.random.default_rng(RANDOM_SEED + k)` for independent streams. ✓

### Environment logging
- `run_neighbor_depmap_analysis.py` and `run_neighbor_chronos_analysis.py`: Full logging via `reproducibility.py`. ✓
- All other scripts: Save `run_config.json` with timestamp and command. ✓
- `run_followup_analyses.py`: Now saves `followup_run_config.json`. ✓

### Config logging
Every script saves a JSON config with all parameters. ✓

### Deterministic output filenames
- Two scripts previously used `time.strftime` for figure filenames — now fixed to static string. ✓
- All other outputs use fixed filenames. ✓

### Directory paths
- All 21 scripts now use `evee-analysis` (the correct directory name). ✓
- Previously, 15 scripts used `eeve-analysis` which did not exist.

### Deterministic sorting / output ordering
- Parquet outputs sorted before writing in most scripts. ✓
- Gene names sorted as iteration keys. ✓
- `verify_determinism.py` exists for checksum comparison. ✓

---

## Verification: Expected Outputs

### Data outputs (`evee-analysis/data/intermediate/`)

| Expected output | Exists | Notes |
|---|---|---|
| `analysis_table.parquet` | ✓ | DEMETER2 analysis table |
| `chronos_analysis_table.parquet` | ✓ | Chronos analysis table |
| `neighbor_vs_random_profile_similarity.parquet` | ✓ | DEMETER2 pair results |
| `chronos_neighbor_vs_random_profile_similarity.parquet` | ✓ | Chronos pair results |
| `consequence_summary.parquet` | ✓ | |
| `chronos_consequence_summary.parquet` | ✓ | |
| `README_results.md` | ✓ | |
| `chronos_README_results.md` | ✓ | |
| `corum_full_enrichment_vs_k.parquet` | ✓ | |
| `corum_full_knn_indices.npz` | ✓ | Reused by multiple scripts |
| `corum_full_run_config.json` | ✓ | |
| `string_enrichment_vs_k.parquet` | ✓ | |
| `string_knn_indices.npz` | ✓ | |
| `string_run_config.json` | ✓ | |
| `corum_retrieval_vs_k.parquet` | ✓ | |
| `corum_retrieval_report.md` | ✓ | |
| `corum_retrieval_run_config.json` | ✓ | |
| `string_retrieval_vs_k.parquet` | ✓ | |
| `string_retrieval_report.md` | ✓ | |
| `string_retrieval_run_config.json` | ✓ | |
| `gene_family_enrichment_vs_k.parquet` | ✓ | |
| `gene_family_per_class.parquet` | ✓ | |
| `gene_family_annotations.parquet` | ✓ | |
| `gene_family_report.md` | ✓ | |
| `gene_family_run_config.json` | ✓ | |
| `gene_level_matrices.npz` | ✓ | Gene-averaged 64×64 matrices |
| `corum_complex_gene_sets.parquet` | ✓ | |
| `corum_entry_enrichment.parquet` | ✓ | |
| `corum_complex_top_entries.parquet` | ✓ | |
| `corum_recurrent_entries.parquet` | ✓ | |
| `complex_signature_similarity.parquet` | ✓ | |
| `corum_class_entry_enrichment.parquet` | ✓ | |
| `corum_interpretability_report.md` | ✓ | |
| `corum_interpretability_run_config.json` | ✓ | |
| `gene_dependency_summary.parquet` | ✓ | |
| `chronos_entry_correlations.parquet` | ✓ | |
| `chronos_entry_weights.parquet` | ✓ | |
| `chronos_top50_entries.parquet` | ✓ | |
| `chronos_entry_analysis_config.json` | ✓ | |
| `chronos_entry_analysis_report.md` | ✓ | |
| `corum_vs_chronos_overlap.parquet` | ✓ | |
| `feature_overlap_metrics.parquet` | ✓ | |
| `feature_classes.parquet` | ✓ | |
| `row_column_enrichment.parquet` | ✓ | |
| `cross_prediction_results.parquet` | ✓ | |
| `feature_clusters.parquet` | ✓ | |
| `corum_vs_chronos_config.json` | ✓ | |
| `corum_vs_chronos_feature_report.md` | ✓ | |
| `corum_enrichment_by_view.parquet` | ✓ | |
| `depmap_view_comparison.parquet` | ✓ | |
| `left_right_report.md` | ✓ | |
| `left_right_run_config.json` | ✓ | |
| `followup_delta_vs_k.parquet` | ✓ | **Updated this round** |
| `followup_distribution_stats.parquet` | ✓ | |
| `followup_threshold_fractions.parquet` | ✓ | |
| `followup_run_config.json` | ✓ | |
| `umap_coords.npz` | ✓ | |
| `umap_gene_family_labels.parquet` | ✓ | |
| `environment.txt` | ✓ | |
| `run_config.json` | ✓ | |
| `run_manifest.txt` | ✓ | |
| `corum_forward_comparison.parquet` | ✓ | Forward prediction |
| `corum_forward_config.json` | ✓ | |
| `corum_forward_novel_pair_predictions.parquet` | ✓ | |
| `corum_forward_prediction_report.md` | ✓ | |
| `corum_forward_v3_derived_recurrent_entries.parquet` | ✓ | |
| `dataset_comparison/` subdirectory | ✓ | 7 files |
| `quick_analyses/` subdirectory | ✓ | **Updated this round** |
| 70 figures in `outputs/figures/` | ✓ | |

**No missing outputs. All scripts' claimed outputs are present.**

---

## Final Judgment: What's Solid, What Needs Caveats, What Should Be Removed

### Solid claims (present as-is)

1. **Embedding neighbors are enriched for CORUM co-complex membership** (2.8× at k=5, gene-level bootstrap CI excludes 1.0). Robust across k values, validated independently via STRING and HGNC gene families.

2. **Embedding neighbors have more similar DEMETER2 dependency profiles than random controls** (gene-level delta = 0.0016, CI [0.0013, 0.0020] excludes 0). Cross-gene only, gene-collapse robust.

3. **The enrichment signal is present across multiple independent biological annotations** (CORUM, STRING, HGNC gene families). Consistent ~2-3× fold enrichment.

4. **Latent feature interactions (64×64 matrix entries) distinguish individual protein complexes** (Welch t-test with FDR correction, split-half stability).

5. **Left and right projections carry distinguishable biological signals** (different enrichment magnitudes per view).

6. **CORUM enrichment significantly exceeds permutation null** (z = 39.8, p < 0.001 from 1000 label permutations). The observed co-complex fraction (0.85%) is 3.3× the null mean (0.26%).

7. **Cohen's d effect size for Chronos** (d = 0.34) is substantially larger than for DEMETER2 (d = 0.09). Both CIs exclude zero.

8. **The signal is stronger for pathogenic variants than benign** (Chronos: 0.0068 vs 0.0050; DEMETER2: 0.0021 vs 0.0015). CIs overlap, so this is a directional finding, not a definitive separation.

9. **CORUM v3→v5 forward prediction shows features derived exclusively from older data can predict novel complex memberships**, avoiding circularity.

### Claims needing caveats

1. **"Chronos shows stronger signal than DEMETER2"** — caveat that Chronos uses only 551/1208 cell lines (DEMETER2 overlap), and the cell line count difference confounds correlation-based comparisons. Present Cohen's d as the fairest comparison, but note the denominator confounder.

2. **"2–3× enrichment"** — specify *which* metric. Enrichment fold is 2–3×. Retrieval precision lift is 1.0–1.6×. These measure different things.

3. **"Shared latent features encode both complex membership and gene essentiality"** — the overlap analysis is correlational. Could be driven by a small number of hub genes. Present with: "associated with", not "encode".

4. **Entry-level Chronos correlations** — the strongest Pearson r values are modest (~0.15–0.20). With 9,493 genes, even small correlations are significant. Present effect sizes, not just p-values.

5. **CORUM and STRING capture similar latent feature patterns** — STRING groups are ego-networks, structurally different from CORUM complexes. The comparison shows similar matrix entries are enriched in both contexts, but the group definitions are not equivalent.

6. **Forward prediction Test C pair scoring** — uses |z_a + z_b| which favors co-activation patterns. If cited specifically, note this design choice and that cosine similarity would be a more standard alternative.

7. **Pathogenic vs benign stratification** — the delta difference is directionally consistent but CIs overlap. Present as "suggestive" rather than "established".

### Claims to remove or heavily qualify

1. **Any direct comparison of DEMETER2 and Chronos absolute delta values** (e.g., from the delta-vs-k plot) should not be used for magnitude comparison. Use Cohen's d instead.

2. **The chi-squared uniformity tests in `run_corum_vs_chronos_features.py`** should not be cited (the pseudocount invalidated them; while fixed, the test was only logged not saved). The visual row/column profiles are fine.

3. **No cross-suite multiple testing correction** — the suite tests many hypotheses (CORUM, STRING, gene families, DepMap, Chronos, etc.) without family-wise correction. Any individual finding with p near 0.05 should be treated cautiously. The strong findings (z > 10) are fine regardless.
