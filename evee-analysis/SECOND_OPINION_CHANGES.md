# Second-Opinion Code Changes (Updated)

**Date:** 2026-04-09
**Supersedes:** Previous changes document from earlier on 2026-04-09.
**This document covers both the original round of fixes and the second audit round.**

---

## Summary of All Changes

**Round 1 (prior audit):** Six targeted fixes across eight files — point-estimate/CI mismatch, misleading text, invalid chi-squared, STRING case matching, reproducibility infrastructure, Chronos caveat.

**Round 2 (this audit):** Five targeted fixes across 19 files — random control under-sampling bug, directory name inconsistency (15 files), dead permutation code, non-deterministic filenames, STRING ego-network caveat.

---

## Round 1 Changes (from prior audit — verified in place)

### Change R1-0 (Critical): Fix point-estimate / CI mismatch in DEMETER2 and Chronos reports

**Files:** `run_neighbor_depmap_analysis.py`, `run_neighbor_chronos_analysis.py`, `README_results.md`, `chronos_README_results.md`

The reported delta was pair-level but the CI was gene-level (different estimators). Fixed to use gene-level bootstrap mean as headline delta.

| Dataset | Old (pair-level) delta | New (gene-level) delta | CI |
|---|---|---|---|
| DEMETER2 | 0.0021 (outside CI) | 0.0016 | [0.0013, 0.0020] ✓ |
| Chronos | 0.0063 (outside CI) | 0.0053 | [0.0049, 0.0057] ✓ |

### Change R1-1: Fix misleading "2–3× enrichment" in CORUM retrieval report

**File:** `run_corum_retrieval.py` line ~470

Removed cross-reference to enrichment analysis (2–3×) from retrieval report. The retrieval metric is precision lift of ~1.6×.

### Change R1-2: Fix invalid chi-squared with pseudocount

**File:** `run_corum_vs_chronos_features.py` lines ~329, ~334

Removed `+ 0.5` pseudocount from `stats.chisquare` calls. Added sparse-data handling.

### Change R1-3: Add reproducibility to `run_followup_analyses.py`

**File:** `run_followup_analyses.py`

Added `from reproducibility import enforce_seeds`, replaced manual seed setting, added `followup_run_config.json` save.

### Change R1-4: Uppercase STRING gene names

**Files:** `run_string_retrieval.py`, `run_string_analysis.py`

Applied `.upper()` to STRING preferred_name values for case-safe matching.

### Change R1-5: Add cell-line caveat to Chronos README

**File:** `chronos_README_results.md`

Added explicit caveat about 551/1208 cell line overlap and non-comparability of raw deltas.

---

## Round 2 Changes (this audit)

### Change R2-0 (Critical): Fix random control under-sampling in delta-vs-k

**File:** `run_followup_analyses.py` lines 250–260

**Problem:** The inner retry loop for random gene sampling broke out on the first non-self gene, regardless of whether `pairwise_profile_corr` returned None. This caused systematic under-counting of random controls when profiles had insufficient overlap or zero variance.

**Before:**
```python
rand_corrs_for_variant = []
for _ in range(n_needed):
    for _ in range(20):
        rg = valid_genes[rng.integers(len(valid_genes))]
        if rg != q_gene:
            rv = gene_to_vec.get(rg)
            if rv is not None:
                c = pairwise_profile_corr(q_vec, rv)
                if c is not None:
                    rand_corrs_for_variant.append(c)
            break  # breaks regardless of success
```

**After:**
```python
rand_corrs_for_variant = []
for _ in range(n_needed):
    for _ in range(20):
        rg = valid_genes[rng.integers(len(valid_genes))]
        if rg == q_gene:
            continue
        rv = gene_to_vec.get(rg)
        if rv is not None:
            c = pairwise_profile_corr(q_vec, rv)
            if c is not None:
                rand_corrs_for_variant.append(c)
                break  # only breaks on success
```

**Impact on data (Chronos, which was most affected):**

| k | n_random BEFORE | n_random AFTER | delta BEFORE | delta AFTER |
|---|---|---|---|---|
| 5 | 430,961 (−3.3%) | 445,505 (=nb) | 0.005976 | 0.005893 |
| 10 | 872,341 (−3.2%) | 901,414 (=nb) | 0.005419 | 0.005550 |
| 20 | 1,758,003 (−3.2%) | 1,816,855 (=nb) | 0.005159 | 0.005170 |
| 50 | 4,422,588 (−3.2%) | 4,570,431 (=nb) | 0.004574 | 0.004612 |

DEMETER2 was minimally affected (<0.03% shortfall). All deltas change <3% relative. No headline claims change.

**Rerun:** Yes — `run_followup_analyses.py` rerun completed in 34.6 minutes.

### Change R2-1 (Important): Fix directory name in 15 scripts

**Files (15 total):** `inspect_db.py`, `plot_umap_gene_families.py`, `run_chronos_entry_analysis.py`, `run_corum_full.py`, `run_corum_interpretability.py`, `run_corum_retrieval.py`, `run_corum_vs_chronos_features.py`, `run_followup_analyses.py`, `run_gene_family_analysis.py`, `run_left_right_analysis.py`, `run_neighbor_chronos_analysis.py`, `run_neighbor_depmap_analysis.py`, `run_string_analysis.py`, `run_string_retrieval.py`, `verify_determinism.py`

**Problem:** All used `EEVE_ROOT = REPO_ROOT / "eeve-analysis"` but the actual directory is `evee-analysis`. No symlink existed. These scripts would crash immediately on any clean environment.

**Fix:** Replaced all occurrences of `eeve-analysis` with `evee-analysis` and renamed the variable from `EEVE_ROOT` to `EVEE_ROOT` for consistency with the 4 newer scripts. Also updated string references in comments, docstrings, and print statements.

**Rerun:** Not needed for correctness since existing outputs were generated when a symlink was presumably in place.

### Change R2-2 (Important): Remove dead permutation code and increase permutation count

**File:** `run_quick_analyses.py` lines 506–523

**Problem:** `perm_knn = knn_indices.copy()` was shuffled each iteration but never used — the actual permutation was via `gene_list` shuffling (which is correct). Additionally, 200 permutations gave minimum p-value resolution of only 1/201 ≈ 0.005.

**Before:**
```python
n_perms = 200
...
for p in range(n_perms):
    perm_knn = knn_indices.copy()           # dead code
    for i in range(perm_knn.shape[0]):      # dead code
        rng.shuffle(perm_knn[i])            # dead code
    shuffled_mapping = list(range(len(gene_list)))
    rng.shuffle(shuffled_mapping)
    old_gene_list = gene_list.copy()
    for idx, new_idx in enumerate(shuffled_mapping):
        gene_list[idx] = old_gene_list[new_idx]
    perm_fracs[p] = _compute_enrichment(knn_indices, k)
    gene_list[:] = old_gene_list
```

**After:**
```python
n_perms = 1000
...
for p in range(n_perms):
    old_gene_list = gene_list.copy()
    rng.shuffle(gene_list)
    perm_fracs[p] = _compute_enrichment(knn_indices, k)
    gene_list[:] = old_gene_list
```

**Result comparison:**

| Metric | Before (200 perms) | After (1000 perms) |
|---|---|---|
| Null mean | 0.002612 | 0.002591 |
| Null std | 0.000154 | 0.000148 |
| z-score | 38.07 | 39.75 |
| p-value | 0.00498 (= 1/201) | 0.00100 (= 1/1001) |

The conclusion is unchanged (overwhelmingly significant). The z-score increased slightly due to more precise null estimation.

**Rerun:** Yes — `run_quick_analyses.py` rerun completed in 77 seconds.

### Change R2-3 (Important): Fix non-deterministic DATE_TAG in figure filenames

**Files:** `run_followup_dataset_comparison.py` line 70, `run_quick_analyses.py` line 46

**Before:**
```python
DATE_TAG = time.strftime("%Y%m%d")
```

**After:**
```python
DATE_TAG = "20260409"
```

Running the script on different days previously produced different figure filenames, breaking `verify_determinism.py` checksum comparisons.

**Rerun:** `run_quick_analyses.py` rerun completed above. `run_followup_dataset_comparison.py` was not rerun (only the filename constant changed, no computational changes).

### Change R2-4 (Important): Add STRING ego-network caveat to dataset comparison report

**File:** `run_followup_dataset_comparison.py` lines 695–704

Added caveat text to the within-structural finding auto-generation:

> **Caveat:** STRING groups are ego-networks (a gene plus its high-confidence interaction partners), structurally different from CORUM's curated multi-gene complexes. This comparison measures whether the same matrix entries are enriched in both contexts, not whether the group definitions are equivalent.

**Rerun:** Not needed — the report is regenerated from the template on next run.

---

## Scripts Requiring Rerun (complete summary)

| Script | Rerun needed? | Rerun done? | Reason | Result change |
|---|---|---|---|---|
| `run_followup_analyses.py` | **Yes** | **Yes** (34.6 min) | C-NEW-1: random under-sampling fix | n_random now equals n_neighbor exactly; deltas change <3% |
| `run_quick_analyses.py` | **Yes** | **Yes** (77 sec) | I-NEW-1: dead code removal + 1000 perms | z-score 38.1→39.8; p-value 0.005→0.001 |
| `run_neighbor_depmap_analysis.py` | Done (R1) | Yes | R1-0: CI/delta fix | Delta 0.0021→0.0016 |
| `run_neighbor_chronos_analysis.py` | Done (R1) | Yes | R1-0: CI/delta fix | Delta 0.0063→0.0053 |
| `run_followup_dataset_comparison.py` | Optional | No | R2-3 + R2-4: filename + caveat text only | None expected |
| 15 scripts with dir name fix | Optional | No | R2-1: would fail on clean clone | None — fix is path-only |
| All other scripts | No | No | — | — |

**Headline result impact:** No headline claims change. The random under-sampling fix changes Chronos deltas by <3% and DEMETER2 by <0.1%. The permutation test z-score improves slightly. All signals remain statistically significant.

---

## Verification Checklist

- [x] All 75 data output files present in `evee-analysis/data/intermediate/` (including subdirectories)
- [x] All 70 figure files present in `evee-analysis/outputs/figures/`
- [x] `followup_delta_vs_k.parquet` updated with matched n_random == n_neighbor
- [x] `quick_analyses_summary.json` updated with 1000 permutations
- [x] No `eeve-analysis` string remaining in any script
- [x] All `EVEE_ROOT` variables point to existing `evee-analysis` directory
- [x] `DATE_TAG` is fixed string in both affected scripts
- [x] STRING caveat present in dataset comparison report template
