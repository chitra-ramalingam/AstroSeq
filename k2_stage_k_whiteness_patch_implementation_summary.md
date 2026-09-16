# Stage K Whiteness Patch Implementation Summary

## Scope

Implemented the minimal safe Stage J additive patch to make runtime and saved whiteness outputs explicit, information-preserving, and directly comparable.

Not changed:

- scientific policy
- whiteness thresholds
- alpha
- ranking logic
- routing logic
- batching behavior

## Files Changed

Implementation files:

- `src/Classifiers/K2/Systematics/K2_NoiseHandler.py`
- `src/Classifiers/K2/Systematics/K2NoiseLoader.py`
- `src/Classifiers/K2/Pipeline/K2_BatchRunner.py`
- `src/Classifiers/K2/Pipeline/K2WhitenessRunner.py`
- `src/Classifiers/K2/Batch/K2ShortlistPeriodRunner.py`
- `src/Classifiers/K2/Batch/K2StageHWhitenessPolicyDiagnosis.py`

Focused tests:

- `tests/test_k2_noise_pipeline.py`
- `tests/test_k2_batch_runner.py`
- `tests/test_k2_whiteness_runner.py`
- `tests/test_k2_shortlist_period_runner.py`
- `tests/test_k2_stage_h_whiteness_policy_diagnosis.py`

Summary artifact:

- `k2_stage_k_whiteness_patch_implementation_summary.md`

## Fields Added

Added explicit runtime/saved whiteness fields while keeping legacy fields unchanged:

- `triage_whiteness_pvalue`
- `triage_whiteness_log10_pvalue`
- `triage_whiteness_statistic_abs_rho`
- `triage_whiteness_z`
- `triage_whiteness_mode`
- `triage_whiteness_underflowed`

Legacy compatibility fields preserved unchanged:

- `triage_whiteness_score`
- `triage_whiteness_definition`
- `triage_score_global`
- `triage_why_not_usable`

## Behavior Implemented

### 1. P-value mode

- `triage_whiteness_score` remains the legacy compatibility alias.
- `triage_whiteness_pvalue` is now the explicit p-value field.
- `triage_whiteness_log10_pvalue` is computed with a stable fallback so information survives even when `math.erfc(...)` underflows to `0.0`.
- `triage_whiteness_underflowed=True` when raw p-value collapses to `0.0` but a finite `z` and finite `log10_pvalue` still exist.

### 2. Statistic mode

- `triage_whiteness_score` remains the legacy compatibility alias.
- `triage_whiteness_statistic_abs_rho` is now the explicit statistic field.
- `triage_whiteness_pvalue` and `triage_whiteness_log10_pvalue` remain `NaN`.
- `triage_whiteness_mode="statistic"` makes the interpretation explicit.

### 3. Missing / non-computable whiteness

- Missing or non-computable whiteness remains represented as `NaN`.
- `triage_whiteness_underflowed=False` for missing/non-computable cases.
- This preserves the distinction between:
  - valid tiny p-value that underflowed
  - valid ordinary finite p-value
  - missing/non-computable whiteness

## Compatibility / Reader Fallbacks

Minimal reader/consumer updates were applied to prefer explicit fields when present and fall back to legacy fields otherwise.

### `K2_BatchRunner`

- Added mode coercion from explicit `triage_whiteness_mode` with fallback to `triage_whiteness_definition`.
- In p-value mode, gating/retriage now prefers `triage_whiteness_pvalue` and falls back to `triage_whiteness_score`.
- In statistic mode, gating/retriage now prefers `triage_whiteness_statistic_abs_rho` and falls back to `triage_whiteness_score`.
- Existing threshold policy text and legacy compatibility behavior were preserved.

### `K2WhitenessRunner`

- Output is now additive instead of destructive.
- When all rows are p-value mode, the runner keeps `triage_whiteness_score` and also exposes `triage_whiteness_pvalue`.
- `whiteness_value_column` still points to the appropriate preferred column for downstream reporting.

### `K2ShortlistPeriodRunner`

- Continues preferring `triage_whiteness_pvalue` when available, with fallback to `triage_whiteness_score`.
- Additional explicit whiteness fields are now passed through in normalized output tables.

### `K2StageHWhitenessPolicyDiagnosis`

- Runtime comparison now prefers `triage_whiteness_pvalue` when present, with fallback to `triage_whiteness_score`.
- Diagnosis output now carries explicit runtime fields:
  - `runtime_triage_whiteness_pvalue`
  - `runtime_triage_whiteness_log10_pvalue`
  - `runtime_triage_whiteness_statistic_abs_rho`
  - `runtime_triage_whiteness_z`
  - `runtime_triage_whiteness_mode`
  - `runtime_triage_whiteness_underflowed`

## Focused Validation

Executed focused unit tests only. No full science pipeline rerun. No new production batch run.

Command run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_k2_noise_pipeline tests.test_k2_batch_runner tests.test_k2_whiteness_runner tests.test_k2_shortlist_period_runner tests.test_k2_stage_h_whiteness_policy_diagnosis -v
```

Result:

- 50 tests run
- 50 passed
- 0 failed

Covered cases:

- ordinary finite p-value case
- underflowed p-value case
- statistic mode case
- missing/non-computable case
- reader fallback behavior

## Open Caveats

- `pytest` is not currently available in the project virtual environment (`No module named pytest`), so validation used `unittest`.
- A direct `python -m py_compile ...` attempt hit a Windows filesystem permission/rename issue in `__pycache__`, so compile-only validation was not used as the primary signal.
- Existing legacy consumers can continue reading `triage_whiteness_score`, but any consumer that needs exact comparability or underflow diagnosis should migrate to the explicit fields.
