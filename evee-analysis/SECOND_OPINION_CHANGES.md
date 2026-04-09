# Second-Opinion Code Changes

**Date:** 2026-04-09

---

## Summary of Changes

Six targeted fixes were applied across eight files. No structural changes, no new analyses, no broadening of scope. All changes are conservative — fixing a point-estimate/CI mismatch, misleading text, an invalid statistical test, a potential gene-name case mismatch, and a reproducibility gap.

---

## Change 0 (Critical): Fix point-estimate / CI mismatch in DEMETER2 and Chronos reports

### Files modified
- `evee-analysis/scripts/run_neighbor_depmap_analysis.py` (lines ~581–585, ~416, ~868)
- `evee-analysis/scripts/run_neighbor_chronos_analysis.py` (lines ~579–583, ~420)
- `evee-analysis/data/intermediate/README_results.md` (section 3)
- `evee-analysis/data/intermediate/chronos_README_results.md` (section 3)

### What changed
The reported delta was computed as the **pair-level mean** of `(neighbor corr - random corr)` across all individual pairs, but the 95% CI was computed from `bootstrap_delta_by_gene()`, which resamples **genes** (not pairs) and averages within each gene first. These are different estimators because genes with many variant pairs contribute disproportionately to the pair-level mean.

| Dataset | Old (pair-level) delta | New (gene-level) delta | CI |
|---|---|---|---|
| DEMETER2 | 0.0021 (**outside CI**) | 0.0016 | [0.0013, 0.0020] ✓ |
| Chronos | 0.0063 (**outside CI**) | 0.0053 | [0.0049, 0.0057] ✓ |

**Before** (both scripts):
```python
delta = nb_mean - rd_mean  # pair-level
ci_lo, ci_mid, ci_hi = bootstrap_delta_by_gene(result_df)  # gene-level
# → delta outside [ci_lo, ci_hi]
```

**After:**
```python
pair_delta = nb_mean - rd_mean  # pair-level, kept for reference
ci_lo, ci_mid, ci_hi = bootstrap_delta_by_gene(result_df)
delta = ci_mid  # gene-level mean matches the gene-level bootstrap CI
```

The same fix was applied to the per-consequence breakdown in both scripts.

### Why necessary
This is a confirmed statistical error: the point estimate and confidence interval came from different estimators, making the CI meaningless as a measure of uncertainty around the reported value. A reviewer or auditor seeing the point estimate outside the 95% CI would (correctly) flag this as suspicious.

### Impact on headline results
The signal remains statistically significant — the CI excludes zero in both cases. But the effect sizes are ~20-25% smaller:
- DEMETER2: 0.0021 → 0.0016
- Chronos: 0.0063 → 0.0053

### Rerun required
**Yes** for the README reports to regenerate. The static README files have been corrected directly. The underlying pair-level data (parquet files) does not change — only the summary statistics in the report.

---

## Change 1: Fix misleading "2–3× enrichment" cross-reference in CORUM retrieval report

### Files modified
- `evee-analysis/scripts/run_corum_retrieval.py` (line ~470)
- `evee-analysis/data/intermediate/corum_retrieval_report.md` (line 47)

### What changed
The interpretation section said: *"The enrichment signal (2–3× across CORUM and STRING validations) is the key finding"*

This is misleading because:
- The retrieval analysis produces micro-precision lift of only **1.6× at k=5** (decaying to ~1.0× at k=50)
- The "2–3×" comes from the *enrichment* analysis (`run_corum_full.py`), which measures a different quantity (fold of sharing rate, not retrieval precision)

### New text
*"The precision lift over base rate (~1.6× at k=5) is the key finding; raw precision/recall numbers should be interpreted in context of the class imbalance. Note: the higher fold-enrichment figures (2–3×) come from the co-membership enrichment analysis (`run_corum_full.py`), which measures a different quantity."*

### Why necessary
Conflating two different metrics with different magnitudes would mislead a presentation audience. The retrieval analysis should cite its own numbers.

### Rerun required
No — this only affects the report text template, not computed values. The existing `corum_retrieval_report.md` was also updated directly.

---

## Change 2: Fix invalid chi-squared test with pseudocount

### File modified
- `evee-analysis/scripts/run_corum_vs_chronos_features.py` (lines ~329, ~334)

### What changed
**Before:**
```python
chi2_row, p_row = stats.chisquare(row_obs + 0.5)  # +0.5 to avoid zeros
...
chi2_col, p_col = stats.chisquare(col_obs + 0.5)
```

**After:**
```python
if (row_obs == 0).sum() > len(row_obs) // 2:
    chi2_row, p_row = float("nan"), float("nan")
else:
    chi2_row, p_row = stats.chisquare(row_obs[row_obs > 0])
...
if (col_obs == 0).sum() > len(col_obs) // 2:
    chi2_col, p_col = float("nan"), float("nan")
else:
    chi2_col, p_col = stats.chisquare(col_obs[col_obs > 0])
```

### Why necessary
Adding 0.5 to all observed counts before chi-squared invalidates the test: chi-squared expects integer counts, and the additive shift systematically biases the test statistic. The fix properly handles sparse distributions by skipping the test when more than half the bins are empty, and testing only non-zero bins otherwise.

### Impact
The chi-squared results are only logged (not saved to any output file), so no saved outputs change. The logged p-values will differ on rerun.

### Rerun required
Optional — affects only log output.

---

## Change 3: Add reproducibility infrastructure to `run_followup_analyses.py`

### File modified
- `evee-analysis/scripts/run_followup_analyses.py` (lines ~14, ~32, ~402–415)

### What changed
1. Added `import sys` and `from reproducibility import enforce_seeds`
2. Replaced manual `random.seed() / np.random.seed() / os.environ` with `enforce_seeds(RANDOM_SEED)`
3. Added config save: writes `followup_run_config.json` with seed, k-values, thresholds, command, and timestamp

### Why necessary
This was the only analysis script not saving any run configuration or using the shared `reproducibility.py` infrastructure. The seeds were correctly set but no config was logged, making it harder to reproduce.

### Rerun required
No — the computational behavior is identical. The only new output is `followup_run_config.json`.

---

## Change 4: Uppercase STRING gene names for case-safe matching

### Files modified
- `evee-analysis/scripts/run_string_retrieval.py` (line ~74)
- `evee-analysis/scripts/run_string_analysis.py` (line ~90)

### What changed
**Before:**
```python
prot_to_gene = dict(zip(info["string_protein_id"].to_list(), info["preferred_name"].to_list()))
```

**After:**
```python
prot_to_gene = {k: v.upper() for k, v in zip(info["string_protein_id"].to_list(), info["preferred_name"].to_list())}
```

### Why necessary
The variant→gene mapping uppercases gene names (`vid_to_gene = {v: g.upper() ...}`), but STRING preferred_names were used as-is. While STRING human protein names are conventionally uppercase HUGO symbols, this creates an inconsistency that could silently drop matches if any name were mixed-case.

### Impact
In practice, STRING human preferred_names are already uppercase, so this change is expected to produce **identical results**. It's a safety fix against future data updates.

### Rerun required
Optional — expected to produce identical results.

---

## Change 5: Add cell-line caveat to Chronos README

### File modified
- `evee-analysis/data/intermediate/chronos_README_results.md` (Caveats section)

### What changed
Added two new caveat bullets:
1. *"55% of the 1,208 total Chronos cell lines are discarded"* — making the data loss explicit
2. *"Raw correlation deltas are not directly comparable to DEMETER2 deltas because the two datasets use different numbers of cell lines"* — flagging the comparability issue

### Why necessary
The existing caveats mentioned the 551 cell line overlap but did not quantify how much data was discarded or explicitly warn against direct magnitude comparison with DEMETER2. This is critical for presentation.

### Rerun required
No — documentation change only.

---

## Scripts Requiring Rerun

| Script | Rerun needed? | Reason | Expected result change |
|---|---|---|---|
| `run_neighbor_depmap_analysis.py` | **Yes** | CI/delta mismatch fix | Delta: 0.0021 → 0.0016; CI unchanged; all pair data unchanged |
| `run_neighbor_chronos_analysis.py` | **Yes** | CI/delta mismatch fix | Delta: 0.0063 → 0.0053; CI unchanged; all pair data unchanged |
| `run_corum_retrieval.py` | **No** | Text-only fix in report template | None — report already updated directly |
| `run_corum_vs_chronos_features.py` | **Optional** | Chi-squared logged values will change | Only log output; no saved files affected |
| `run_followup_analyses.py` | **Optional** | New config file saved | Adds `followup_run_config.json`; all other outputs identical |
| `run_string_retrieval.py` | **Optional** | Case-safe gene names | Expected identical results |
| `run_string_analysis.py` | **Optional** | Case-safe gene names | Expected identical results |

**Headline result change:** The DEMETER2 and Chronos deltas are ~20-25% smaller when correctly computed at the gene level. The signal remains significant (CI excludes zero) but the corrected numbers should be used in presentation. Static README files have been updated directly.

---

## Verification Checklist

- [x] All 67 data output files present in `evee-analysis/data/intermediate/`
- [x] All 44 figure files present in `evee-analysis/outputs/figures/`
- [x] All modified scripts pass Python syntax check (`ast.parse`)
- [x] All modified scripts pass linter (no new errors)
- [x] Enrichment values verified: CORUM 2.84×, STRING 2.54×, Gene family 3.37× (all at k=5)
- [x] Bootstrap CIs verified: gene-level delta now brackets point estimates (DEMETER2: 0.0016 ∈ [0.0013, 0.0020] ✓; Chronos: 0.0053 ∈ [0.0049, 0.0057] ✓)
- [x] Chronos 551/1208 cell line caveat documented
- [x] CORUM retrieval report no longer conflates retrieval precision with enrichment fold
- [x] Point-estimate/CI mismatch fixed in both DEMETER2 and Chronos scripts and reports
