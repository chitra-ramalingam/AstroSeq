# Stage J Whiteness Patch Plan

Date: 2026-04-17

Scope: patch planning only. No code changes, no pipeline reruns, no policy changes.

## Objective

Apply the smallest safe implementation patch that makes runtime and saved whiteness outputs:

- explicit about meaning
- information-preserving for extreme p-values
- directly comparable across saved/runtime paths
- backward compatible with existing CSV consumers

Scientific thresholds and keep/reject policy remain unchanged.

## Recommended Minimal Strategy

Use an **additive runtime patch**:

- keep legacy `triage_whiteness_score`
- add explicit whiteness fields alongside it
- preserve `triage_whiteness_definition`
- leave threshold logic unchanged
- make downstream comparison tools prefer explicit fields when available

This avoids changing the scientific decision rule while preventing runtime `0.0` from being the only surviving representation of an extreme p-value.

## Exact Files / Functions To Patch

### 1. Runtime whiteness producer

File: `src/Classifiers/K2/Systematics/K2_NoiseHandler.py`

Functions / types:

- `K2NoiseMetrics` dataclass
- `K2_NoiseHandler._metrics_single`
- `K2_NoiseHandler.explain`

Planned patch:

- extend `K2NoiseMetrics` to carry explicit whiteness components
- compute and store both the primary configured output and the underlying comparable components
- preserve current `whiteness_score` behavior so gating logic stays unchanged

### 2. Runtime loader row serialization

File: `src/Classifiers/K2/Systematics/K2NoiseLoader.py`

Functions:

- `_error_row`
- `run_one`

Planned patch:

- initialize new whiteness fields on error/fetch-fail rows as `NaN`/empty
- emit explicit whiteness fields into the serializable row dict
- preserve existing `whiteness_score`, `whiteness_definition`, `score_global`, `why_not_usable`

### 3. Detector summary passthrough

File: `src/Classifiers/K2/K2_TimeDomainTransitPipeline.py`

Function:

- `run_one`

Planned patch:

- likely no logic change needed if summary continues to forward `noise_row`
- verify tests still pass once loader rows contain new fields

### 4. Batch CSV writer / retriage compatibility

File: `src/Classifiers/K2/Pipeline/K2_BatchRunner.py`

Functions:

- `_hard_fail_reasons`
- `_debug_whiteness_gate`
- `retriage_results_df`
- `_process_query`

Planned patch:

- copy explicit runtime whiteness fields into `triage_*` columns in batch rows
- keep `triage_whiteness_score` as the backward-compatible alias
- in retriage and hard-fail logic, prefer `triage_whiteness_pvalue` when mode is p-value and field exists; otherwise fall back to legacy `triage_whiteness_score`
- do not change thresholds or pass/fail conditions

### 5. Saved precompute output compatibility

File: `src/Classifiers/K2/Pipeline/K2WhitenessRunner.py`

Function:

- `_build_precompute_output_df`

Planned patch:

- preserve current `triage_whiteness_pvalue` output behavior
- pass through the new explicit whiteness fields if present
- do not rely on renaming `triage_whiteness_score` as the only way to expose p-value semantics

### 6. Downstream raw-whiteness consumer

File: `src/Classifiers/K2/Batch/K2ShortlistPeriodRunner.py`

Functions:

- `_choose_whiteness_value_column`
- raw table normalization around lines `1464-1496`

Planned patch:

- continue preferring `triage_whiteness_pvalue`
- tolerate presence of new fields without requiring them
- keep current fallback to `triage_whiteness_score`

### 7. Saved/runtime diagnosis consumer

File: `src/Classifiers/K2/Batch/K2StageHWhitenessPolicyDiagnosis.py`

Functions:

- `REQUIRED_ORIGINAL_COLUMNS`
- `REQUIRED_PATCHED_COLUMNS`
- `_normalize_original`
- `_normalize_patched`
- `_diagnosis_frame`

Planned patch:

- accept explicit runtime p-value fields when present
- compare saved p-value against runtime explicit p-value first
- retain legacy fallback to `runtime_triage_whiteness_score` for old CSVs

## New Output Fields To Add

At minimum, add and use:

- `triage_whiteness_pvalue`
- `triage_whiteness_log10_pvalue`
- `triage_whiteness_statistic_abs_rho`
- `triage_whiteness_z`
- `triage_whiteness_mode`

Recommended additional field:

- `triage_whiteness_underflowed`

Rationale:

- `triage_whiteness_pvalue` provides the explicit semantic quantity used for current policy
- `triage_whiteness_log10_pvalue` preserves information when raw p-value underflows to `0.0`
- `triage_whiteness_statistic_abs_rho` gives the underlying effect size and is comparable across numeric ranges
- `triage_whiteness_z` preserves the normal-approximation test statistic used to derive the p-value
- `triage_whiteness_mode` disambiguates p-value vs statistic storage
- `triage_whiteness_underflowed` cleanly distinguishes numeric underflow from a valid representable zero-like value and from missingness

## Field Meanings

### Runtime p-value mode

When `triage_whiteness_mode == "pvalue"`:

- `triage_whiteness_pvalue`: raw p-value as computed; may be exact `0.0` if underflow occurs
- `triage_whiteness_log10_pvalue`: finite log10 p-value computed in a way that survives underflow when possible
- `triage_whiteness_statistic_abs_rho`: `abs(rho_1)`
- `triage_whiteness_z`: `abs(rho_1) * sqrt(n - 1)`
- `triage_whiteness_score`: compatibility alias equal to `triage_whiteness_pvalue`
- `triage_whiteness_definition`: unchanged, still `lag1_autocorr_pvalue_normal_approx`

### Runtime statistic mode

When `triage_whiteness_mode == "statistic"`:

- `triage_whiteness_pvalue`: `NaN`
- `triage_whiteness_log10_pvalue`: `NaN`
- `triage_whiteness_statistic_abs_rho`: `abs(rho_1)`
- `triage_whiteness_z`: `NaN` or the derived z only if explicitly chosen to compute it for diagnostics
- `triage_whiteness_score`: compatibility alias equal to `triage_whiteness_statistic_abs_rho`
- `triage_whiteness_definition`: unchanged, still `lag1_abs_autocorr_statistic`

### Missing / non-computable

When whiteness cannot be computed:

- `triage_whiteness_mode`: still reflects configured mode, not missingness
- all numeric explicit whiteness fields: `NaN`
- `triage_whiteness_underflowed`: `False`
- legacy `triage_whiteness_score`: `NaN`

This keeps missing/non-computable distinct from underflowed extreme p-values.

## Log-pvalue vs Floor-Clamped Pvalue

### Recommendation

Store **both**:

- raw `triage_whiteness_pvalue`
- explicit `triage_whiteness_log10_pvalue`

Do **not** replace raw p-values with a floor-clamped p-value as the primary field.

### Why

Reasons:

- floor-clamping changes the numeric value and can be mistaken for a real p-value
- raw p-value preserves current semantics and backward compatibility
- log10 p-value is the information-preserving channel for extreme values
- the pair `(pvalue, log10_pvalue)` cleanly distinguishes:
  - representable p-value
  - underflowed p-value
  - missing/non-computable value

### Underflow handling recommendation

Recommended representation:

- `triage_whiteness_pvalue`: leave as actual computed value, including `0.0` if underflow occurs
- `triage_whiteness_log10_pvalue`: compute from a numerically stable path if possible
- `triage_whiteness_underflowed`: `True` when `pvalue == 0.0` but z/statistic is finite and indicates a valid computed extreme p-value

This is safer than inventing a floor such as `1e-300`.

## Compatibility Aliases / Renames

### Keep unchanged

Do not remove or redefine:

- `triage_whiteness_score`
- `triage_whiteness_definition`
- `triage_score_global`
- `triage_why_not_usable`

### Alias behavior

Recommended compatibility contract:

- in p-value mode:
  - `triage_whiteness_score == triage_whiteness_pvalue`
- in statistic mode:
  - `triage_whiteness_score == triage_whiteness_statistic_abs_rho`

### No hard rename now

Do not rename `triage_whiteness_score` in the runtime/batch CSV schema in the minimal patch.

Reason:

- too many existing readers and tests already reference it
- additive explicit fields solve the comparability problem without immediate schema churn

## Backward Compatibility Plan

### Existing CSV consumers

Preserve current consumers by keeping:

- `triage_whiteness_score`
- existing definition strings
- existing pass/fail thresholds
- existing `why_not_usable` text format

### Reader precedence

Downstream readers should be updated to prefer:

1. `triage_whiteness_pvalue` when mode is p-value
2. otherwise `triage_whiteness_score`

That applies especially to:

- `K2ShortlistPeriodRunner`
- `K2StageHWhitenessPolicyDiagnosis`

### Saved whiteness CSV

Saved `batch_results_whiteness.csv` is already explicit on the p-value side. Minimal patch should:

- keep `triage_whiteness_pvalue`
- optionally pass through the new diagnostic fields
- avoid removing any existing saved columns required by current consumers

## Minimal Safe Patch Set Now

### Patch set

1. Extend `K2NoiseMetrics` with explicit whiteness component fields.
2. In `_metrics_single`, compute:
   - `abs_rho`
   - `z`
   - raw `pvalue` in p-value mode
   - `log10_pvalue`
   - `underflowed`
3. In `K2NoiseLoader.run_one`, emit explicit whiteness fields in runtime rows.
4. In `K2_BatchRunner._process_query`, write batch columns:
   - `triage_whiteness_pvalue`
   - `triage_whiteness_log10_pvalue`
   - `triage_whiteness_statistic_abs_rho`
   - `triage_whiteness_z`
   - `triage_whiteness_mode`
   - optional `triage_whiteness_underflowed`
5. In `K2_BatchRunner.retriage_results_df` and `_hard_fail_reasons`, prefer explicit p-value if available, else fall back to legacy field.
6. In `K2WhitenessRunner`, pass through explicit whiteness fields into saved outputs.
7. In Stage H diagnosis, compare saved p-value to runtime explicit p-value when present.

### What this patch set intentionally does not do

- no threshold changes
- no alpha changes
- no changes to `score_global` formula
- no label policy changes
- no removal of legacy fields
- no science reruns inside the patch itself

## Distinguishing P-Value Mode, Statistic Mode, Missing, Underflow

Recommended interpretation table:

- p-value mode, normal representable:
  - `triage_whiteness_mode = "pvalue"`
  - `triage_whiteness_pvalue` finite and `> 0`
  - `triage_whiteness_log10_pvalue` finite
  - `triage_whiteness_underflowed = False`

- p-value mode, underflowed:
  - `triage_whiteness_mode = "pvalue"`
  - `triage_whiteness_pvalue == 0.0`
  - `triage_whiteness_log10_pvalue` finite
  - `triage_whiteness_z` finite
  - `triage_whiteness_underflowed = True`

- statistic mode:
  - `triage_whiteness_mode = "statistic"`
  - `triage_whiteness_statistic_abs_rho` finite
  - `triage_whiteness_pvalue = NaN`
  - `triage_whiteness_log10_pvalue = NaN`

- missing/non-computable:
  - `triage_whiteness_mode` still set from config
  - all numeric explicit whiteness fields `NaN`
  - `triage_whiteness_underflowed = False`

## Validation After Patching

### Existing tests to rerun

Recommended focused suite:

- `tests/test_k2_noise_pipeline.py`
- `tests/test_k2_whiteness_runner.py`
- `tests/test_k2_batch_runner.py`
- `tests/test_k2_shortlist_period_runner.py`
- `tests/test_k2_stage_h_whiteness_policy_diagnosis.py`

Recommended commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_k2_noise_pipeline.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_k2_whiteness_runner.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_k2_batch_runner.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_k2_shortlist_period_runner.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_k2_stage_h_whiteness_policy_diagnosis.py -q
```

### New targeted test coverage to add

Add tests for:

- p-value mode row where p-value is ordinary and finite
- p-value mode row where raw p-value underflows to `0.0` but `log10_pvalue` remains informative
- statistic mode row with explicit `abs_rho`
- non-computable row where all explicit whiteness numerics are `NaN`
- retriage and Stage H consumers preferring explicit p-value field when present

### Saved/runtime comparability checks

After patching, verify:

1. For p-value mode runtime rows:
   - `triage_whiteness_score == triage_whiteness_pvalue`
2. For statistic mode runtime rows:
   - `triage_whiteness_score == triage_whiteness_statistic_abs_rho`
3. Saved `batch_results_whiteness.csv` and runtime batch rows expose p-value via explicit `triage_whiteness_pvalue`
4. Stage H diagnosis uses explicit runtime p-value when available and no longer has to compare saved p-value against a semantically ambiguous generic score field

### Distinguishing NaN vs underflow vs valid tiny p-value

Confirm with assertions:

- missing/non-computable:
  - `pvalue`, `log10_pvalue`, `z`, `abs_rho` are `NaN`
  - `underflowed == False`
- underflow:
  - `pvalue == 0.0`
  - `log10_pvalue` finite
  - `z` finite
  - `underflowed == True`
- valid tiny but representable p-value:
  - `pvalue > 0.0`
  - `log10_pvalue` finite
  - `underflowed == False`

## Implementation Recommendation

### Minimal safe patch now

Implement the additive explicit-field patch:

- keep legacy `triage_whiteness_score`
- add explicit p-value/statistic/z/logp/mode fields
- add underflow indicator
- update only the small set of readers that compare or choose whiteness columns

This fixes representation and comparability without changing policy.

### Optional cleanup later

After Stage J lands and validation passes:

- migrate downstream consumers away from `triage_whiteness_score`
- de-emphasize the generic score field in reports
- consider a later schema cleanup where explicit fields become primary and `triage_whiteness_score` is treated as legacy-only

