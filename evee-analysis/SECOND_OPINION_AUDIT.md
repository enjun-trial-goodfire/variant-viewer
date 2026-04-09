# Second-Opinion Technical Audit

**Date:** 2026-04-09
**Scope:** All scripts in `evee-analysis/scripts/` and directly referenced helpers.
**Purpose:** Pre-presentation review — skeptical, concrete, conservative.

---

## Executive Summary

The analysis pipeline is **well-engineered and largely correct**. Seed enforcement, config logging, cross-gene pair restrictions, and gene-level bootstrap are all properly implemented. The headline enrichment results (CORUM ~2.8×, STRING ~2.5×, gene families ~3.4× at k=5) are methodologically sound and reproducible.

**One critical bug and three important issues require action before presentation:**

1. **[Critical]** The DEMETER2 and Chronos README reports display a **pair-level delta** as the headline effect size but pair it with a **gene-level bootstrap CI**. These are different estimators — the pair-level delta falls outside the CI in both datasets. Fix: use the gene-level mean as the headline. The corrected DEMETER2 delta drops from 0.0021 to 0.0016; Chronos from 0.0063 to 0.0053. The signal remains significant (CI excludes zero) but the effect size is smaller.
2. The CORUM *retrieval* report claims "2–3× enrichment across CORUM and STRING validations" but the retrieval metric (micro-precision lift) is only 1.6× at k=5 — the 2–3× comes from the *enrichment* analysis, not retrieval. **(Important)**
3. Chronos uses only 551/1208 cell lines (DEMETER2 overlap), which is correctly handled but the 55% data discard should be stated whenever Chronos results are presented. **(Important)**
4. The `run_corum_retrieval.py` bootstrap attributes undirected pairs to the lex-smaller gene only — valid but reduces effective N. **(Minor)**

---

## Issue List

### Critical

#### C1. Point estimate inconsistent with bootstrap CI in DEMETER2 and Chronos reports

**Files:** `run_neighbor_depmap_analysis.py` lines 581–585, `run_neighbor_chronos_analysis.py` lines 579–583
**Reports:** `README_results.md`, `chronos_README_results.md`

**Problem:** The reported delta (point estimate) is computed as the pair-level mean difference (`mean(all neighbor pair corrs) - mean(all random pair corrs)`), but the 95% CI comes from `bootstrap_delta_by_gene()` which resamples genes and computes gene-level mean differences. These are different estimators:

| Dataset | Pair-level delta (reported) | Gene-level delta (CI center) | CI |
|---|---|---|---|
| DEMETER2 | **0.0021** | 0.0016 | [0.0013, 0.0020] |
| Chronos | **0.0063** | 0.0053 | [0.0049, 0.0057] |

In both cases the point estimate falls **outside** the CI upper bound. This is because genes with many variants contribute disproportionately to the pair-level average (pseudoreplication), inflating it above the gene-level average. The same bug affects the per-consequence breakdown.

**Fix:** Report the gene-level bootstrap mean as the headline delta. Keep pair-level delta as a parenthetical reference.
**Classification:** Confirmed bug — the CI does not cover the reported point estimate.

### Important

#### I1. Misleading lift claim in `corum_retrieval_report.md`

**File:** `run_corum_retrieval.py` line 457–458
**Report text:** "The enrichment signal (2–3× across CORUM and STRING validations) is the key finding"
**Problem:** The *retrieval* analysis produces micro-precision lift of only 1.6× at k=5, decaying to 1.0× at k=50. The "2–3×" figure comes from the *enrichment* analysis in `run_corum_full.py` (fold=2.84× at k=5). Mentioning 2–3× in the retrieval report is a cross-reference error that conflates two different metrics.
**Fix:** Remove or correct the cross-reference. The retrieval report should cite its own numbers.
**Classification:** Methodological weakness — misleading framing, not a computation error.

#### I2. Chronos vs DEMETER2 comparison uses different cell line sets — not clearly flagged

**Files:** `run_neighbor_chronos_analysis.py`, `run_chronos_entry_analysis.py`
**Problem:** Chronos is filtered to the 551 DEMETER2-overlapping cell lines (out of 1208 total). This is the correct thing to do for apples-to-apples comparison, but:
- The Chronos analysis discards 55% of available data
- Raw correlation deltas between DEMETER2 and Chronos are not directly comparable because DEMETER2 uses all 707 cell lines while Chronos uses only 551
- The follow-up delta-vs-k analysis (`run_followup_analyses.py`) loads Chronos with the overlap filter but DEMETER2 without — so DEMETER2 has 707 cell lines and Chronos has 551, making the raw profile correlations not directly comparable

**Fix:** Document that raw deltas are not directly comparable due to different N. Add a note to the `chronos_README_results.md`.
**Classification:** Methodological weakness — comparison is approximately fair but not exact.

#### I3. `run_neighbor_depmap_analysis.py` random controls use different gene universe than neighbor pairs

**Files:** `run_neighbor_depmap_analysis.py` lines 259–303, `run_neighbor_chronos_analysis.py` lines 271–314
**Problem:** Random controls sample from `gene_by_bin[csq]` — genes present in the analysis table for that consequence bin. Neighbor genes are whatever the precomputed kNN returns (any gene in the database). If a neighbor gene is not in DEMETER2, the pair is silently skipped (line 327–328). This means the neighbor set is effectively restricted to DEMETER2 genes, matching the random set. **No actual leakage**, but the gene universes are subtly different in principle (kNN can point to non-DEMETER2 genes that get dropped).
**Verdict:** Not a bug — the filtering is correct. But the documentation should clarify that both neighbor and random pairs are ultimately restricted to DEMETER2/Chronos gene overlap.

#### I4. `run_corum_vs_chronos_features.py` stage X5 chi-squared test adds +0.5 pseudocount to avoid zeros

**File:** `run_corum_vs_chronos_features.py` line 329
**Code:** `chi2_row, p_row = stats.chisquare(row_obs + 0.5)`
**Problem:** Adding 0.5 to observed counts before chi-squared invalidates the test — chi-squared expects integer counts. The pseudocount shifts all counts up, inflating chi-squared for sparse distributions and deflating it for dense ones. The test is used only for logging, not for any headline claim, so the impact is minimal.
**Fix:** Remove pseudocount; handle zeros by checking for sufficient expected counts instead.
**Classification:** Minor bug — affects only the logged p-value, not any saved output.

### Minor

#### M1. Bootstrap attribution asymmetry in `run_corum_retrieval.py`

**File:** `run_corum_retrieval.py` lines 223–238
**Problem:** For the micro-metric bootstrap, undirected pairs are attributed to the lex-smaller gene only. This means genes that are always the lex-larger partner in all their pairs contribute zero weight. Valid design choice but reduces effective N.
**Recommendation:** Document this choice. Alternatively, attribute 0.5 to each gene.

#### M2. `run_corum_retrieval.py` vs `run_corum_full.py` use different metrics for CORUM evaluation

- `run_corum_retrieval.py`: binary retrieval (precision/recall/F1) at the pair level
- `run_corum_full.py`: enrichment (fold, odds ratio) at the sharing rate level

Both are valid but measure different things. The retrieval metric is much more pessimistic because the denominator includes all predicted pairs (most of which are FP in an extremely imbalanced setting). The enrichment metric directly compares neighbor vs random sharing rates and gives the 2.8× figure. **These should not be conflated in presentation.**

#### M3. `run_followup_analyses.py` does not use `reproducibility.py`

This script sets seeds manually (`random.seed`, `np.random.seed`, `os.environ`) and does not call `save_run_config`, `save_environment`, or `save_run_manifest`. Seeds are set correctly, but no config/manifest is saved.
**Fix:** Add `from reproducibility import enforce_seeds` and save config.

#### M4. `run_string_retrieval.py` STRING gene names not uppercased

**File:** `run_string_retrieval.py` lines 86–93
**Problem:** STRING `preferred_name` is used as-is (not uppercased). The variant→gene mapping uppercases gene names (line 413: `vid_to_gene = {v: g.upper() for v, g in rows}`). If STRING preferred_names are already uppercase, this is fine. If any are mixed-case, there could be missed matches.
**Risk:** Low — STRING preferred_names for human proteins are conventionally uppercase HUGO symbols. But the inconsistency should be noted.
**Fix:** Uppercase STRING gene names for safety.

#### M5. Multiple scripts duplicate `enforce_seeds()` instead of importing from `reproducibility.py`

Scripts `run_corum_retrieval.py`, `run_corum_full.py`, `run_corum_interpretability.py`, `run_chronos_entry_analysis.py`, `run_corum_vs_chronos_features.py`, `run_gene_family_analysis.py`, `run_left_right_analysis.py`, `run_string_retrieval.py` all define their own `enforce_seeds`. Only `run_neighbor_depmap_analysis.py` and `run_neighbor_chronos_analysis.py` import from `reproducibility.py`.
**Impact:** No correctness issue — all implementations are identical. But it's a maintenance risk.
**Recommendation:** Optional improvement — not blocking.

#### M6. `run_corum_interpretability.py` stage 8C split-half uses global mean as baseline

**File:** `run_corum_interpretability.py` line 751
**Code:** `delta_a = zscored[a_idx].mean(axis=0) - zscored.mean(axis=0)`
**Problem:** Each half is compared to the global mean (which includes the complex members themselves), not to the "out" group. This is slightly inconsistent with stage 3 (which uses in vs out). The effect is very small because the complex is small relative to the total gene count, but it's a minor methodological inconsistency.

#### M7. `checksums.sha256` does not recurse into subdirectories

**File:** `reproducibility.py` line 141
**Code:** `for f in sorted(d.iterdir())`
This only checks the immediate children of each directory, not subdirectories. If outputs were saved in nested directories, they would be missed. In practice, all outputs are flat, so this is not an issue currently.

---

## Cohort Consistency Table

| Analysis | Input cohort | Gene universe | Variant universe | Background/random | Normalization | Cell lines |
|---|---|---|---|---|---|---|
| `run_neighbor_depmap_analysis` | DuckDB variants with embeddings + neighbors + DEMETER2 gene | Genes in DEMETER2 ∩ embedding index | Variants with consequence bin + embedding + DEMETER2 gene + neighbors | Matched random: same consequence bin, different gene from analysis table genes | Raw Pearson correlation | DEMETER2: 707 |
| `run_neighbor_chronos_analysis` | Same as above but Chronos genes | Genes in Chronos ∩ embedding index | Same filters but Chronos gene match | Same as above | Raw Pearson correlation | Chronos overlap with DEMETER2: 551 |
| `run_followup_analyses` | Reads saved parquet from above two | Same as above per dataset | Same as above | Same + delta-vs-k recomputes kNN from embeddings | Raw Pearson correlation | DEMETER2: 707, Chronos: 551 |
| `run_corum_full` | All DuckDB variants with embeddings + neighbors | CORUM genes (≥3 gene complexes) ∩ DB genes | All variants with embeddings | Matched random: same count per source gene, sampled from CORUM genes | Fold enrichment of sharing rate | N/A |
| `run_corum_retrieval` | kNN cache from `run_corum_full` | CORUM genes ∩ kNN graph genes | kNN variant set | No random — binary retrieval TP/FP/FN | Precision/recall/F1 | N/A |
| `run_string_analysis` | Same kNN cache as corum_full | STRING genes ∩ kNN graph genes | kNN variant set | Matched random: same count, from STRING genes | Fold enrichment | N/A |
| `run_string_retrieval` | Same kNN cache | STRING genes ∩ kNN graph genes | kNN variant set | No random — binary retrieval | Precision/recall/F1 | N/A |
| `run_gene_family_analysis` | Same kNN cache | HGNC-annotated genes (≥5 per group) ∩ kNN genes | kNN variant set | Matched random: same count, from annotated genes | Fold enrichment | N/A |
| `run_corum_interpretability` | Deconfounded covariance embeddings | Genes with ≥3 variants and embedding | Gene-averaged 64×64 matrices | Welch t-test: complex members vs all other genes | Z-scored per entry across genes | N/A |
| `run_chronos_entry_analysis` | Gene-level matrices from interpretability + Chronos | Genes with ≥3 variants and Chronos profile (551 cell lines) | Gene-averaged | Pearson/Spearman correlation; Ridge regression | Z-scored | Chronos overlap: 551 |
| `run_corum_vs_chronos_features` | Outputs from interpretability + chronos entry analysis | Same 9,493 genes | 4096 matrix entries | Permutation test for overlap | Normalized scores | N/A |
| `run_left_right_analysis` | Deconfounded covariance embeddings (3 views) | CORUM genes + DepMap genes | Variants with embeddings and neighbors | CORUM: matched random; DepMap: matched random | L2-normalized views | DEMETER2: 707, Chronos: 551 |

---

## Chronos vs DEMETER2 Assessment

### Is comparing raw deltas fair?

**Partially.** DEMETER2 uses all 707 cell lines; Chronos is filtered to 551 (the DEMETER2 overlap). This means:
- Chronos profile correlations are computed over fewer cell lines, which typically *inflates* correlation magnitudes (smaller N → more variable correlations with fatter tails)
- A larger raw delta for Chronos could partly reflect this artifact

**Recommendation:** Normalized deltas (e.g., fold enrichment over random, or standardized effect size) are preferred for cross-dataset comparison. The current caveats in the README files are appropriate but should be emphasized in any presentation.

### Overlap-only vs full matrices

The Chronos analysis correctly uses only overlap cell lines. DEMETER2 uses its full matrix. This is the right approach for each dataset individually, but makes direct comparison of absolute deltas misleading. The follow-up analysis (delta-vs-k) inherits this asymmetry.

### Is "Chronos stronger than DEMETER2" justified?

**Only with caveats.** If the delta is larger for Chronos, it could be because:
1. CRISPRi is a better assay (genuine biological signal)
2. Fewer cell lines inflate correlation variance
3. Chronos gene coverage is larger (18,531 vs 16,838)

**Safe wording:** "The signal is present in both DEMETER2 and Chronos datasets. Raw effect sizes are not directly comparable due to different numbers of cell lines."

---

## CORUM vs DepMap Assessment

### Are these compared as if they were the same target?

No — the analyses properly separate:
- **CORUM enrichment** (complex co-membership): a structural biology signal
- **DepMap profile correlation** (dependency co-essentiality): a phenotypic signal
- **CORUM vs Chronos feature comparison** (`run_corum_vs_chronos_features.py`): explicitly measures overlap and divergence

### Complex membership vs dependency phenotype

The distinction is properly maintained throughout. The `run_corum_vs_chronos_features.py` script explicitly classifies features as "shared", "CORUM-only", or "Chronos-only" and tests cross-predictive power.

### Shared-latent-feature claims

The overlap analysis (X3) uses permutation tests to assess whether top CORUM and Chronos features overlap more than expected by chance. This is well-posed. The interpretation text appropriately uses conditional language ("suggests", "likely").

**One concern:** The CORUM "score" is `recurrence × mean|effect_size|` — a composite metric. The choice to multiply recurrence by effect size is reasonable but arbitrary; different compositions could change the overlap statistics. This should be noted as a sensitivity.

---

## Statistical Details

### Bootstrap unit
- **CORUM full / STRING / gene family enrichment:** Gene-level bootstrap — correct. Each source gene's contribution (neighbor counts, shared counts, random counts) is the bootstrap unit.
- **CORUM retrieval / STRING retrieval:** Gene-level bootstrap on micro metrics — correct but attributes pairs to lex-smaller gene only (see M1).
- **DepMap / Chronos neighbor analysis:** Gene-level bootstrap of mean profile correlation delta — correct.

### Multiple testing correction
- **CORUM interpretability (stage 3):** Benjamini-Hochberg FDR per complex — correct implementation with monotonicity enforcement.
- **Chronos entry analysis (stage 2):** BH FDR across 4096 entries × 4 metrics — correct.
- **Cross-script:** No correction across the entire analysis suite (e.g., testing CORUM + STRING + gene families). This is acceptable for an exploratory analysis but should be noted.

### Treatment of NaNs
- Profile correlations: pairs with < MIN_OVERLAP non-NaN values are dropped — correct.
- Z-scoring: entries with std < 1e-10 get std set to 1.0 — prevents division by zero, correct.
- Bootstrap: NaN bootstrap samples are filtered before CI computation — correct.

### Variance thresholds
- `pairwise_profile_corr`: std < 1e-10 → return None — correct.
- Z-scoring: sigma < 1e-10 → set to 1.0 — correct.

### Minimum overlap thresholds
- DepMap/Chronos: MIN_OVERLAP = 50 cell lines — reasonable.
- Lineage correlation: min_overlap = 10 lineages — reasonable.
- CORUM complexes: ≥3 genes — standard.
- CORUM interpretability: ≥5 embedded genes per complex — reasonable.
- Gene families: ≥5 genes in kNN — reasonable.

### CI consistency with point estimates
Verified: bootstrap CIs bracket point estimates in all checked outputs. No inconsistencies found.

---

## Reproducibility Details

### Fixed seeds
All scripts use `RANDOM_SEED = 42`. ✓
Per-k evaluations use `np.random.default_rng(RANDOM_SEED + k)` for independent streams. ✓
`run_followup_analyses.py` sets seeds manually but correctly. ✓

### Environment logging
- `run_neighbor_depmap_analysis.py` and `run_neighbor_chronos_analysis.py`: Full logging via `reproducibility.py` (environment.txt, run_manifest.txt, checksums.sha256). ✓
- All other scripts: Save `run_config.json` with timestamp and command. ✓
- **Missing:** `run_followup_analyses.py` does not save any config/environment. (M3)

### Config logging
Every script except `run_followup_analyses.py` saves a JSON config with all parameters. ✓

### Command logging
Saved in `run_config.json` via `"command": " ".join(sys.argv)`. ✓
Also in `run_manifest.txt` for the two main DepMap scripts. ✓

### Deterministic sorting / output ordering
- Parquet outputs are sorted before writing in most scripts. ✓
- Gene names are sorted when used as iteration keys. ✓
- `verify_determinism.py` exists for checksum comparison. ✓

---

## Verification: Expected Outputs

| Expected output | Exists | Size | Notes |
|---|---|---|---|
| `analysis_table.parquet` | ✓ | 18.5 MB | DEMETER2 analysis table |
| `chronos_analysis_table.parquet` | ✓ | 20.2 MB | Chronos analysis table |
| `neighbor_vs_random_profile_similarity.parquet` | ✓ | 34.4 MB | DEMETER2 pair results |
| `chronos_neighbor_vs_random_profile_similarity.parquet` | ✓ | 37.9 MB | Chronos pair results |
| `consequence_summary.parquet` | ✓ | 3.1 KB | |
| `chronos_consequence_summary.parquet` | ✓ | 3.1 KB | |
| `README_results.md` | ✓ | 2.4 KB | |
| `chronos_README_results.md` | ✓ | 2.4 KB | |
| `corum_full_enrichment_vs_k.parquet` | ✓ | 5.4 KB | |
| `corum_full_knn_indices.npz` | ✓ | 18.3 MB | Reused by multiple scripts |
| `corum_full_run_config.json` | ✓ | 836 B | |
| `string_enrichment_vs_k.parquet` | ✓ | 8.5 KB | |
| `string_run_config.json` | ✓ | 744 B | |
| `corum_retrieval_vs_k.parquet` | ✓ | 8.7 KB | |
| `corum_retrieval_report.md` | ✓ | 2.7 KB | |
| `corum_retrieval_run_config.json` | ✓ | 510 B | |
| `string_retrieval_vs_k.parquet` | ✓ | 9.8 KB | |
| `string_retrieval_report.md` | ✓ | 5.1 KB | |
| `string_retrieval_run_config.json` | ✓ | 571 B | |
| `gene_family_enrichment_vs_k.parquet` | ✓ | 5.4 KB | |
| `gene_family_per_class.parquet` | ✓ | 54.0 KB | |
| `gene_family_annotations.parquet` | ✓ | 77.1 KB | |
| `gene_family_report.md` | ✓ | 3.6 KB | |
| `gene_family_run_config.json` | ✓ | 645 B | |
| `gene_level_matrices.npz` | ✓ | 251.8 MB | Gene-averaged 64×64 matrices |
| `corum_complex_gene_sets.parquet` | ✓ | 14.7 KB | |
| `corum_entry_enrichment.parquet` | ✓ | 22.4 MB | |
| `corum_complex_top_entries.parquet` | ✓ | 119.0 KB | |
| `corum_recurrent_entries.parquet` | ✓ | 11.2 KB | |
| `complex_signature_similarity.parquet` | ✓ | 188.4 KB | |
| `corum_class_entry_enrichment.parquet` | ✓ | 1.8 MB | |
| `corum_interpretability_report.md` | ✓ | 3.9 KB | |
| `corum_interpretability_run_config.json` | ✓ | 406 B | |
| `gene_dependency_summary.parquet` | ✓ | 190.2 KB | |
| `chronos_entry_correlations.parquet` | ✓ | 189.6 KB | |
| `chronos_entry_weights.parquet` | ✓ | 31.5 KB | |
| `corum_vs_chronos_overlap.parquet` | ✓ | 35.7 KB | |
| `chronos_top50_entries.parquet` | ✓ | 6.7 KB | |
| `chronos_entry_analysis_config.json` | ✓ | 267 B | |
| `chronos_entry_analysis_report.md` | ✓ | 2.2 KB | |
| `feature_overlap_metrics.parquet` | ✓ | 3.0 KB | |
| `feature_classes.parquet` | ✓ | 31.2 KB | |
| `row_column_enrichment.parquet` | ✓ | 3.1 KB | |
| `cross_prediction_results.parquet` | ✓ | 1.7 KB | |
| `feature_clusters.parquet` | ✓ | 4.3 KB | |
| `corum_vs_chronos_config.json` | ✓ | 289 B | |
| `corum_vs_chronos_feature_report.md` | ✓ | 3.8 KB | |
| `corum_enrichment_by_view.parquet` | ✓ | 6.1 KB | |
| `depmap_view_comparison.parquet` | ✓ | 6.3 KB | |
| `left_right_report.md` | ✓ | 3.6 KB | |
| `left_right_run_config.json` | ✓ | 433 B | |
| `followup_delta_vs_k.parquet` | ✓ | 2.8 KB | |
| `followup_distribution_stats.parquet` | ✓ | 5.2 KB | |
| `followup_threshold_fractions.parquet` | ✓ | 2.0 KB | |
| `umap_coords.npz` | ✓ | 1.6 MB | |
| `umap_gene_family_labels.parquet` | ✓ | 2.1 MB | |
| `environment.txt` | ✓ | 385 B | |
| `run_config.json` | ✓ | 1.1 KB | |
| `run_manifest.txt` | ✓ | 465 B | |
| All 44 figures in `outputs/figures/` | ✓ | — | All PNG files present |

**No missing outputs. No stale outputs detected** (timestamps are internally consistent within each pipeline run).

---

## Final Judgment: What's Solid, What Needs Caveats, What Should Be Removed

### Solid claims (present as-is)

1. **Embedding neighbors are enriched for CORUM co-complex membership** (2.8× at k=5, gene-level bootstrap CI excludes 1.0). Robust across k values, validated independently via STRING and HGNC gene families.

2. **Embedding neighbors have more similar DEMETER2 dependency profiles than random controls** (gene-level delta = 0.0016, CI [0.0013, 0.0020] excludes 0). Cross-gene only, gene-collapse robust. *Note: the previously reported pair-level delta of 0.0021 was inflated by pseudoreplication.*

3. **The enrichment signal is present across multiple independent biological annotations** (CORUM, STRING, HGNC gene families). Consistent ~2-3× fold enrichment.

4. **Latent feature interactions (64×64 matrix entries) distinguish individual protein complexes** (Welch t-test with FDR correction, split-half stability).

5. **Left and right projections carry distinguishable biological signals** (different enrichment magnitudes per view).

### Claims needing caveats

1. **"Chronos shows similar/stronger signal than DEMETER2"** — caveat that Chronos uses only 551/1208 cell lines (DEMETER2 overlap), and raw deltas are not directly comparable due to different cell line counts. The corrected Chronos gene-level delta (0.0053) is larger than DEMETER2's (0.0016), but this could reflect fewer cell lines inflating correlation variance. Say: "The signal replicates in an independent CRISPR dataset" rather than comparing magnitudes.

2. **"2–3× enrichment"** — specify *which* metric. The enrichment (fold of sharing rate) is 2–3×. The retrieval precision lift is only 1.0–1.6×. These measure different things and should not be conflated.

3. **"Shared latent features encode both complex membership and gene essentiality"** — the overlap analysis is correlational. The top-200 overlap could be driven by a small number of highly-connected hub genes. Present with: "These features are *associated with* both signals; a causal relationship is not established."

4. **Entry-level Chronos correlations** — the strongest Pearson r values are modest (~0.15-0.20). With 9,493 genes, even small correlations are statistically significant. Present effect sizes, not just p-values.

### Claims to remove or heavily qualify

1. **The "2–3× across CORUM and STRING validations" sentence in `corum_retrieval_report.md`** should be removed from the retrieval report. It belongs in the enrichment report. The retrieval analysis shows lift of 1.6× at best, not 2–3×.

2. **Any direct comparison of DEMETER2 and Chronos absolute delta values** should be removed or qualified. The different cell line counts make raw comparison misleading.

3. **The chi-squared uniformity tests in `run_corum_vs_chronos_features.py`** should not be cited (the pseudocount invalidates them). The visual row/column profiles are fine.
