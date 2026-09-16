# Stage I Whiteness Implementation Audit

Date: 2026-04-17

Scope: implementation and interpretation audit only. No science pipeline reruns were performed.

## Executive Conclusion

- `saved_triage_whiteness_pvalue` is not a separately computed whiteness measure. In the saved/precompute path it is a renamed copy of the upstream `triage_whiteness_score` when the definition string says the score is a p-value.
- Runtime `triage_whiteness_score` is a generic storage field. When `triage_whiteness_definition == "lag1_autocorr_pvalue_normal_approx"`, that field already contains a p-value, not a raw statistic.
- There is no code path that uses `0.0` as a hard-failure sentinel for whiteness. Unavailable or non-computable whiteness is represented as `NaN`, not `0.0`.
- A runtime whiteness value of exact `0.0` is therefore most consistent with numeric collapse of the implemented p-value conversion (`math.erfc(...)`) for an extremely small p-value, not with a sentinel assignment.
- `score_global == -1.0` is also not a sentinel. In p-value mode it follows directly from the score formula when whiteness is `0.0` and `alpha == 0.01`.
- Saved and runtime interpretation rules are consistent when the definition label contains `pvalue`: higher is better, and the row fails the whiteness gate when `pvalue < alpha`.

Primary recommendation: **A: fix a bug/inconsistency**.

The inconsistency is not that saved and runtime use different scientific rules. The inconsistency is that the same p-value is stored under two different semantic field names (`..._pvalue` vs `..._score`), and the runtime path preserves no non-underflow representation of the p-value. That makes cross-run comparison brittle and easy to misread.

## 1. Saved `saved_triage_whiteness_pvalue`: where it really comes from

### 1.1 There is no saved-only recomputation

The saved Stage H field is prepared in `src/Classifiers/K2/Batch/K2StageHWhitenessPolicyDiagnosis.py:133-152`. That code only renames columns from `batch_results_whiteness.csv`:

- `triage_whiteness_pvalue -> saved_triage_whiteness_pvalue`
- `triage_step_score -> saved_triage_step_score`
- `triage_score_global -> saved_triage_score_global`

It does not recompute whiteness.

### 1.2 `batch_results_whiteness.csv` is built by renaming a prior runtime field

`src/Classifiers/K2/Pipeline/K2WhitenessRunner.py:423-451` does this:

1. Reads `batch_results_retriaged.csv`.
2. Instantiates `K2BatchRunner(... whiteness_score_definition="pvalue")`.
3. Calls `runner.retriage_results_df(df)`.
4. Writes `batch_results_whiteness.csv`.

The key transformation is in `src/Classifiers/K2/Pipeline/K2WhitenessRunner.py:378-394`:

- If every non-empty `triage_whiteness_definition` contains `pvalue`, then:
  - `triage_whiteness_score` is copied into `triage_whiteness_pvalue`
  - `triage_whiteness_score` is dropped
  - `triage_whiteness_interpretation` is set to "Two-sided lag-1 autocorrelation p-value (normal approximation); higher means more white."
  - `triage_whiteness_one_minus_pvalue = 1.0 - triage_whiteness_pvalue`

So the saved field is an aliasing/storage change, not a new computation.

### 1.3 Saved retriage uses the stored runtime number and reapplies gates

`src/Classifiers/K2/Pipeline/K2_BatchRunner.py:740-805` (`retriage_results_df`) does not recompute the whiteness metric from flux. It reads stored columns:

- `triage_step_score`
- `triage_whiteness_score`
- `triage_whiteness_definition`

Then it rebuilds `triage_usable` and `triage_why_not_usable` by applying threshold logic again:

- if definition contains `pvalue`, fail when `white < noisy_whiteness_threshold`
- otherwise fail when `white > noisy_whiteness_threshold`

This means the saved path is downstream of the original runtime computation.

## 2. Runtime `triage_whiteness_score`: how it is computed

### 2.1 The runtime compute function

The actual whiteness calculation is in `src/Classifiers/K2/Systematics/K2_NoiseHandler.py:680-717`.

For a cleaned light curve:

- `step_score = median(abs(diff(flux))) / (robust_sigma + 1e-12)`
- whiteness uses lag-1 autocorrelation on median-centered flux

Relevant implementation:

- `fr = f - med`
- if all values are non-finite or `np.nanstd(fr) == 0`, whiteness is `NaN`
- otherwise:
  - `rho = mean(fr0 * fr1) / (std(fr0) * std(fr1) + 1e-12)`
  - if `whiteness_score_definition == "pvalue"`:
    - `z = abs(rho) * sqrt(n - 1)`
    - `w = math.erfc(z / sqrt(2))`
  - else:
    - `w = abs(rho)`

This is the only place in the audited source where the whiteness number itself is computed.

### 2.2 Definition strings

`src/Classifiers/K2/Systematics/K2_NoiseHandler.py:803-806` defines:

- p-value mode: `lag1_autocorr_pvalue_normal_approx`
- statistic mode: `lag1_abs_autocorr_statistic`

So when the runtime definition string says `lag1_autocorr_pvalue_normal_approx`, the runtime `whiteness_score` is already a p-value.

### 2.3 Threshold interpretation

`src/Classifiers/K2/Systematics/K2_NoiseHandler.py:808-815` defines the threshold:

- p-value mode: threshold is `whiteness_alpha` (default strict preset `0.01`)
- statistic mode: threshold is `max_whiteness_score`

The strict-mode default `whiteness_alpha = 0.01` is set in `src/Classifiers/K2/Systematics/K2_NoiseHandler.py:68-77`.

## 3. Runtime storage and propagation into batch rows

### 3.1 `K2NoiseLoader.run_one` stores the computed value directly

`src/Classifiers/K2/Systematics/K2NoiseLoader.py:262-304`:

- computes `score_global = handler.score(global_m)`
- computes `usable`
- stores:
  - `whiteness_definition = handler.whiteness_definition()`
  - `score_global`
  - `why_not_usable`
  - all metric fields from `global_m`, including `whiteness_score`

There is no runtime `whiteness_pvalue` field here. The field name is always `whiteness_score`, even in p-value mode.

### 3.2 The detector summary simply forwards that loader summary

`src/Classifiers/K2/K2_TimeDomainTransitPipeline.py:415-429` calls `self.loader.run_one(...)` and returns that row as `summary`.

### 3.3 Batch rows copy the runtime summary unchanged

`src/Classifiers/K2/Pipeline/K2_BatchRunner.py:1077-1115` copies summary fields into batch columns:

- `triage_score_global <- summary["score_global"]`
- `triage_step_score <- summary["step_score"]`
- `triage_whiteness_score <- summary["whiteness_score"]`
- `triage_whiteness_definition <- summary["whiteness_definition"]`
- `triage_why_not_usable <- summary["why_not_usable"]`

So runtime batch rows do not convert a statistic into a p-value later. In p-value mode, the p-value is already stored in `triage_whiteness_score`.

## 4. Conversion from raw statistic to p-value

The only raw-statistic to p-value conversion found in the audited code is:

- `src/Classifiers/K2/Systematics/K2_NoiseHandler.py:699-701`

```python
z = abs(rho) * np.sqrt(max(float(n - 1), 1.0))
w = float(math.erfc(float(z) / np.sqrt(2.0)))
```

Meaning:

- raw correlation proxy: `abs(rho)`
- normal approximation test statistic: `z = abs(rho) * sqrt(n-1)`
- two-sided p-value: `erfc(z / sqrt(2))`

No other p-value conversion logic was found for saved or runtime whiteness.

## 5. Clipping, flooring, rounding, underflow, fallback, sentinel behavior

### 5.1 What exists

- Denominator stabilizers only:
  - `mad + 1e-12` at `K2_NoiseHandler.py:668-669`
  - `std(fr0) * std(fr1) + 1e-12` at `K2_NoiseHandler.py:697`
- Non-computable whiteness returns `NaN`:
  - too few points: `K2_NoiseHandler.py:643-656`
  - all non-finite or zero variance: `K2_NoiseHandler.py:689-692`
- Upstream fetch/clean/metrics failures also write `whiteness_score = NaN`:
  - `K2NoiseLoader.py:79-103`
  - `K2NoiseLoader.py:196-220`

### 5.2 What does not exist

No code was found that:

- clips `rho` into `[-1, 1]`
- floors p-values to a minimum positive number
- rounds the stored numeric whiteness field before saving
- replaces failed whiteness with `0.0`
- uses `0.0` as a sentinel for missing whiteness

### 5.3 Why runtime `0.0` is most likely numeric collapse, not a sentinel

Because:

- missing/unavailable whiteness is stored as `NaN`, not `0.0`
- the p-value is computed via `math.erfc(...)`
- there is no lower bound such as `max(p, 1e-300)`

In local CPython on this machine, `math.erfc(x)` first underflows to exact `0.0` around `x = 27.23`. Under the implemented formula this corresponds to roughly:

- `z / sqrt(2) >= 27.23`
- `z >= 38.5`
- `abs(rho) * sqrt(n - 1) >= 38.5`

So exact runtime `0.0` is fully consistent with an extremely small p-value collapsing to zero in floating-point arithmetic.

## 6. Why `score_global` becomes exactly `-1.0`

### 6.1 There is no hard assignment to `-1.0`

A repository-wide search found no source code that explicitly sets `score_global = -1.0`.

### 6.2 It comes from the score formula

`src/Classifiers/K2/Systematics/K2_NoiseHandler.py:722-752` defines `score(m)` as the minimum normalized margin to all thresholds.

For p-value-mode whiteness, the whiteness margin is:

- `margin = (whiteness_score - whiteness_threshold) / whiteness_threshold`

from `K2_NoiseHandler.py:744-749`.

If:

- `whiteness_score = 0.0`
- `whiteness_threshold = whiteness_alpha = 0.01`

then:

- `(0.0 - 0.01) / 0.01 = -1.0`

Therefore exact `score_global == -1.0` is expected when the whiteness term is the limiting failure and the p-value has collapsed to zero.

## 7. How whiteness maps to `triage_usable` and final `label_reason`

### 7.1 Runtime `usable`

In `src/Classifiers/K2/Systematics/K2NoiseLoader.py:280-283`:

- per-segment mode: `usable = (score_best_seg > 0.0) and not catastrophic_outlier`
- non-segment mode: `usable = (score_global > 0.0)`

The transit pipeline uses per-segment mode (`src/Classifiers/K2/K2_TimeDomainTransitPipeline.py:416-423`), so runtime `triage_usable` comes from the per-segment loader path.

### 7.2 Runtime `why_not_usable`

`src/Classifiers/K2/Systematics/K2NoiseLoader.py:262-301` builds `why_not_usable` from `handler.explain(global_m)`.

In `src/Classifiers/K2/Systematics/K2_NoiseHandler.py:782-801`, whiteness failures are emitted as:

- p-value mode: `whiteness_pvalue={m.whiteness_score:.6g}<{wt}`
- statistic mode: `whiteness_score={m.whiteness_score:.6g}>{wt}`

### 7.3 Saved retriage `triage_usable`

`src/Classifiers/K2/Pipeline/K2_BatchRunner.py:772-789` strips prior managed reasons and rebuilds them from:

- `triage_status`
- `triage_step_score`
- `triage_whiteness_score`
- `triage_whiteness_definition`

Then:

- `triage_usable = (status == "ok") and (len(reasons) == 0)`

Important: saved retriage does not use `triage_score_global` to decide `triage_usable`.

### 7.4 Final `label_reason`

`src/Classifiers/K2/Pipeline/K2_BatchRunner.py:347-371` (`_hard_fail_reasons`) adds:

- `usable=False:{why_not_usable}` if `usable` is false
- a second explicit whiteness gate message:
  - p-value mode: `whiteness_pvalue<{threshold:.3f} ({white:.3f})`
  - statistic mode: `whiteness_score>{threshold:.3f} ({white:.3f})`

Then `src/Classifiers/K2/Pipeline/K2_BatchRunner.py:626-645` labels the row:

- any hard reason -> `Noisy_trash`
- label reason becomes `"; ".join(hard_reasons)`

This explains strings like:

- `usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)`

The first clause comes from runtime `why_not_usable`.
The second clause is the batch-layer hard gate re-added with rounded formatting.

## 8. Direct answers to the Stage I questions

### Are saved and runtime whiteness actually the same quantity?

Yes, when the definition is `lag1_autocorr_pvalue_normal_approx`.

In that mode:

- runtime `triage_whiteness_score` is already a p-value
- saved `triage_whiteness_pvalue` is a renamed copy of that same semantic quantity

They are not intended to be different scientific measures.

### Is one a p-value and the other a score/statistic?

Storage names differ, but semantics do not in p-value mode.

- runtime field name: `triage_whiteness_score`
- saved field name: `triage_whiteness_pvalue`

If the definition contains `pvalue`, both represent a p-value.

### Does runtime `0.0` reflect true extreme significance, numeric collapse, or a hard failure sentinel?

It does **not** look like a hard failure sentinel.

Implementation evidence points to:

- true extremely small p-value, represented as exact `0.0` after floating-point underflow in `math.erfc(...)`

Failure/unavailable whiteness is represented elsewhere as `NaN`.

### Are saved and runtime using different interpretation rules despite the same definition label?

No material difference was found.

For `pvalue` definitions, both paths use:

- higher whiteness value = better
- fail when `pvalue < alpha`

The mismatch is naming and numeric representation, not threshold direction.

## 9. Root Cause Framing

The strongest code-level explanation for the observed divergence is:

1. The repository uses one generic runtime field, `triage_whiteness_score`, for two different semantic modes.
2. In the audited runs the definition label says that field is a p-value.
3. The saved path later renames that field to `triage_whiteness_pvalue`, which makes the saved side explicit.
4. The runtime path never stores a non-underflow representation such as log-p, raw `rho`, or `z`.
5. Therefore a very small runtime p-value can collapse to exact `0.0`, while the saved side may still show a normal-looking numeric p-value from a different prior execution.

That is a comparability and representation problem, not evidence that saved uses a statistic while runtime uses a p-value.

## 10. Primary Recommendation

### Recommendation A: fix a bug/inconsistency

Recommended implementation change after Stage I:

1. In runtime outputs, stop using only `triage_whiteness_score` for p-value mode.
2. Emit explicit fields such as:
   - `triage_whiteness_pvalue`
   - `triage_whiteness_statistic_abs_rho`
   - optionally `triage_whiteness_z`
3. Preserve numeric comparability by flooring or logging the p-value before CSV export, for example storing a raw p-value plus `triage_whiteness_log10_pvalue`.
4. Keep `0.0` reserved for actual numeric results only, and continue using `NaN` for missing/non-computable whiteness.

Why A and not B/C/D:

- not B: rename-only is insufficient because the current runtime representation still loses information at underflow
- not C: recomputing one side alone does not solve the overloaded field semantics and numeric collapse
- not D: the scientific policy should not be revised until the implementation emits comparable, information-preserving whiteness fields

