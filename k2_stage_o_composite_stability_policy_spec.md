# Stage O Composite Stability Policy Specification

Date: 2026-04-19

Scope: policy-specification step only. No code was patched, no production batches were rerun, and the current official default policy was not replaced.

## Executive Outcome

Primary Stage O composite policy proposal: **O1: global alternate supported composite stability policy**.

Backup Stage O composite policy proposal: **O2: conditional composite stability policy for the current K2 validation lane**.

This Stage O policy is intended to operationalize the accepted Stage N redesign direction `N2` while preserving the current Stage A default policy during validation.

## Policy Intent

The purpose of the Stage O policy is to replace the current brittle single-metric whiteness hard gate with an interpretable composite stability rule for validation.

The policy is explicitly **not**:

- a threshold relaxation of the current lag-1 p-value gate
- a default-policy replacement
- a production rollout decision

The policy **is**:

- an **alternate supported policy** for validation
- a conservative, review-aware composite
- a way to test whether runtime stability can be assessed more scientifically than with a single lag-1 p-value veto

## Why A Composite Is Scientifically Better Than The Current Single Gate

The current official gate lets one quantity, `lag1_autocorr_pvalue_normal_approx`, veto the entire row.

That is scientifically weak for this K2 population because:

- **Stage H** showed that saved and runtime whiteness behaved as non-equivalent scientific quantities even when the definition label matched.
- **Stage I** showed that runtime `0.0` was real numeric underflow, not missingness.
- **Stage L** showed that the representation fix worked and the lane still failed `100/100`, so the brittleness is now scientific rather than bookkeeping-related.
- **Stage N** concluded that the better redesign is to combine a more robust whiteness component with step/stability evidence rather than keep a single hard whiteness veto.

The composite is scientifically better because it:

- uses the whiteness/autocorrelation signal as one stability component rather than the sole decision-maker
- explicitly includes step discontinuity evidence, which Stage H showed remained coherent across saved and runtime views
- preserves conservative hard-fail behavior for genuinely unusable rows
- routes ambiguous high-risk rows to review instead of forcing them into automatic rejection on one numerically extreme field

## Input Fields

The exact proposed policy inputs are:

Required inputs:

- `triage_status`
- `triage_n_points`
- `triage_step_score`
- `triage_whiteness_statistic_abs_rho`
- `triage_whiteness_mode`
- `triage_whiteness_definition`
- `triage_whiteness_underflowed`

Optional support input:

- `n_events`

Optional diagnostic inputs:

- `triage_whiteness_log10_pvalue`
- `triage_whiteness_z`
- `triage_why_not_usable`

Structured field metadata is in `k2_stage_o_composite_stability_policy_fields.csv`.

## Component Definitions

### 1. Autocorrelation / whiteness component

Primary field:

- `triage_whiteness_statistic_abs_rho`

Scientific role:

- bounded effect-size description of short-lag residual autocorrelation
- lower values are more consistent with whiteness

Reason this is the whiteness component:

- it preserves the underlying autocorrelation information without depending on the p-value transform that collapsed in Stage L
- it is more scientifically stable for composite use than `triage_whiteness_pvalue`

### 2. Step / discontinuity stability component

Primary field:

- `triage_step_score`

Scientific role:

- summarizes first-difference roughness relative to robust scatter
- lower values indicate smoother residual behavior

Why it is included:

- Stage H showed step-score agreement even while whiteness diverged
- this means step-score is an important independent stability signal that should no longer be secondary to one hard whiteness failure

### 3. Data sufficiency prerequisite

Primary fields:

- `triage_status`
- `triage_n_points`

Scientific role:

- prevent the composite from turning operational or insufficient-data rows into false rescues

### 4. Semantic guards

Primary fields:

- `triage_whiteness_mode`
- `triage_whiteness_definition`

Scientific role:

- ensure that the autocorrelation component is interpreted using an explicit, known runtime semantic contract
- prevent the new composite from quietly recreating the old brittle p-value gate in disguised form

Allowed semantic combinations for Stage O validation:

- `triage_whiteness_mode = "pvalue"` with `triage_whiteness_definition = "lag1_autocorr_pvalue_normal_approx"`
- `triage_whiteness_mode = "statistic"` with `triage_whiteness_definition = "lag1_abs_autocorr_statistic"`

Interpretation:

- the composite uses `triage_whiteness_statistic_abs_rho` as the scientific autocorrelation input in either allowed mode
- the mode/definition fields are guards, not score drivers
- if the semantic contract is missing, blank, or inconsistent, the row is not eligible for automatic `pass`

### 5. Diagnostic escalation fields

Primary field:

- `triage_whiteness_underflowed`

Optional audit context:

- `triage_whiteness_log10_pvalue`
- `triage_whiteness_z`

Scientific role:

- identify the exact regime that caused the current p-value policy to become brittle
- escalate review intensity without making p-value semantics the primary decision input

### 6. Optional support feature: `n_events`

Primary field:

- `n_events`

Scientific role:

- optional evidence that the detector found event structure worth carrying forward into review or tie-breaking

Stage O decision:

- `n_events` is **not** a primary composite component
- `n_events` is **not** allowed to override bad step stability, bad autocorrelation stability, insufficient points, or semantic-guard failure
- `n_events` may be used only as an optional support feature for within-tier ordering or for a small secondary score annotation during validation

Reasoning:

- `n_events` is detector output, not a pure stability/noise descriptor
- using it as a dominant term would risk turning Stage O into a disguised recovery policy rather than a stability policy
- this keeps the composite centered on `triage_step_score` and `triage_whiteness_statistic_abs_rho`, as required

## Component Interpretation

The composite uses simple interpretable bands rather than a black-box optimization.

### Autocorrelation band from `triage_whiteness_statistic_abs_rho`

- `A0 low`: `abs_rho <= 0.20`
- `A1 moderate`: `0.20 < abs_rho <= 0.60`
- `A2 severe`: `abs_rho > 0.60`

Interpretation:

- `0.60` is intentionally conservative because it matches the current strict statistic-mode boundary already present in `K2_NoiseHandler`
- `0.20` creates a clearly lower-risk zone rather than treating all non-severe rows as equally acceptable

### Step band from `triage_step_score`

- `S0 low`: `step_score <= 0.10`
- `S1 moderate`: `0.10 < step_score <= 0.30`
- `S2 elevated`: `0.30 < step_score <= 1.50`
- `S3 fail`: `step_score > 1.50` or non-finite

Interpretation:

- `1.50` preserves the current strict hard-fail threshold from `K2_NoiseHandler`
- `0.10` and `0.30` are conservative review bands intended for validation, not production claims
- this keeps the policy interpretable: low, moderate, elevated, or catastrophic step instability

### Sufficiency band from `triage_n_points`

- `N_fail`: `triage_n_points < 800`
- `N0 sufficient`: `800 <= triage_n_points < 1200`
- `N1 strong`: `triage_n_points >= 1200`

Interpretation:

- `800` preserves the current strict minimum-points requirement
- `1200` is not a failure boundary; it is only a positive support indicator for the alternate policy

### Optional event-support band from `n_events`

- `E0 none`: `n_events <= 0`
- `E1 present`: `1 <= n_events <= 2`
- `E2 strong`: `n_events >= 3`

Interpretation:

- this band is **support-only**
- it does not determine `fail`, `review_high`, `review_standard`, or `pass`
- it may be reported in validation output and used only for within-tier ordering or descriptive comparison

## Exact Policy Shape

The exact Stage O primary composite shape is:

1. Apply prerequisite gates:
   - `triage_status` must be `ok`
   - `triage_n_points >= 800`
   - `triage_step_score` must be finite
   - `triage_whiteness_statistic_abs_rho` must be finite
   - `triage_whiteness_mode` and `triage_whiteness_definition` must satisfy an allowed semantic combination
2. Evaluate two central scientific components:
   - autocorrelation stability from `triage_whiteness_statistic_abs_rho`
   - step/discontinuity stability from `triage_step_score`
3. Apply `triage_n_points` as a conservative support modifier.
4. Apply `triage_whiteness_underflowed` only as review escalation.
5. Carry `triage_whiteness_log10_pvalue`, `triage_whiteness_z`, and `triage_why_not_usable` as audit-only context.
6. Carry `n_events` only as optional support annotation or within-tier ordering support.

This is intentionally not a p-value rescue rule. The score and tier system are defined so the old p-value gate cannot reappear as the hidden dominant driver.

## Composite Output

The composite outputs all three of the following:

- a **continuous score**
- a **discrete risk class**
- a **pass/fail plus review tier**

Primary decision output:

- **pass/fail plus review tier**

Secondary transparency outputs:

- **discrete risk class**
- **continuous score**

This ordering keeps the policy interpretable. The score is there for ordering within tiers, not for opaque threshold hunting.

## Continuous Score Definition

The continuous score is a transparent secondary summary:

```text
autocorr_component = clip(1 - (triage_whiteness_statistic_abs_rho / 0.60), 0, 1)
step_component = clip(1 - (triage_step_score / 0.30), 0, 1)
sufficiency_component = clip((triage_n_points - 800) / 800, 0, 1)
underflow_penalty = 0.25 if triage_whiteness_underflowed else 0.0

composite_stability_score =
    100 * max(
        0,
        0.45 * autocorr_component
      + 0.35 * step_component
      + 0.20 * sufficiency_component
      - underflow_penalty
    )
```

Interpretation:

- the score is bounded to `[0, 100]`
- autocorrelation and step carry most of the weight
- sufficiency supports the score but does not dominate it
- underflow is treated as a review penalty, not an automatic hard fail
- `n_events` is intentionally excluded from the primary score so the policy stays centered on stability rather than event-count rescue

## Discrete Risk Class

The policy assigns:

- `low_risk`
- `moderate_risk`
- `high_risk`
- `extreme_risk`

Risk class logic:

- `extreme_risk` if any hard-fail prerequisite is met
- `high_risk` if `A2 severe`, `S2 elevated`, or `triage_whiteness_underflowed=True`
- `moderate_risk` if `A1 moderate` or `S1 moderate`
- `low_risk` only if `A0 low`, `S0 low`, and no underflow

## Decision Tier

The final Stage O decision tier is:

- `fail`
- `review_high`
- `review_standard`
- `pass`

### Hard-fail prerequisites

Automatic `fail` if any of the following hold:

- `triage_status != "ok"`
- `triage_n_points < 800`
- `triage_step_score` is non-finite
- `triage_whiteness_statistic_abs_rho` is non-finite
- `triage_whiteness_mode` / `triage_whiteness_definition` are blank, inconsistent, or unsupported
- `triage_step_score > 1.50`

### Review / pass matrix

If no hard-fail prerequisite is met:

- `pass`:
  - `A0 low`
  - `S0 low`
  - `triage_whiteness_underflowed = False`

- `review_standard`:
  - `(A1 and S0)` or `(A0 and S1)`
  - and `triage_whiteness_underflowed = False`

- `review_high`:
  - `triage_whiteness_underflowed = True`, or
  - `A2 severe`, or
  - `S2 elevated`, or
  - `(A1 and S1)`

Interpretation:

- the policy is conservative because ambiguous rows are routed to review, not auto-pass
- the current brittle p-value collapse becomes a review escalator rather than a universal veto
- rows with positive `n_events` may be ordered ahead of otherwise similar rows inside the same review tier, but `n_events` does not change the tier itself in the primary proposal

## Why This Stays Interpretable

The design avoids becoming an opaque rescue score because:

- only a small number of fields are used
- each field has a single scientific role
- the final decision is driven by explicit bands and a rule matrix
- the continuous score is secondary and fully documented
- semantic guards are explicit and non-scored
- `triage_score_global` is intentionally excluded as a policy input because it already embeds the current brittle whiteness gate
- `triage_whiteness_log10_pvalue` and `triage_whiteness_z` are kept only as audit context, not as hidden score proxies

## Why This Preserves A Conservative Posture

The policy remains conservative because:

- operationally bad or insufficient-data rows still hard-fail
- catastrophic step instability still hard-fails
- severe autocorrelation does not become automatic pass; it becomes at least `review_high`
- underflowed rows are escalated to review rather than silently treated as good
- semantic ambiguity hard-fails the alternate policy rather than being hand-waved away
- the policy is introduced only as an alternate supported policy during validation

## Relationship To Current Stage A Default Policy

During validation, Stage A default policy remains unchanged.

Coexistence rule:

- current official policy remains the default and reference comparator
- Stage O composite policy is evaluated in parallel as an alternate supported policy
- all validation reporting must show both outcomes side by side for the same rows
- no row should be considered operationally promoted under Stage O until the paired validation experiment is accepted

This preserves the repo's conservative-default framing while still allowing scientific redesign to be tested honestly.

## Minimum Validation Experiment

The smallest acceptable validation experiment is:

1. Freeze the Stage O primary proposal exactly as specified here.
2. Reuse the existing high-priority calibration slice already used in Stages F, H, and L.
3. Run a paired replay or rerun on that same slice:
   - current official policy
   - Stage O composite policy
4. Compare row-level outputs side by side.

Required outputs:

- current-policy label / reason
- Stage O risk class
- Stage O decision tier
- Stage O composite stability score
- row-level component bands (`A*`, `S*`, `N*`, optional `E*`)

### Success criteria

Minimum success:

- the composite must break the current `100/100` universal-whiteness-fail outcome
- at least some rows must move from automatic rejection into `review_standard`, `review_high`, or `pass`
- no row with hard-fail prerequisites may become `pass`
- the accepted or reviewable subset must not show obvious immediate instability collapse on `triage_step_score` or sufficiency fields
- any benefit attributable to optional `n_events` support must be limited to within-tier ordering or audit interpretation, not tier override

### Failure criteria

Failure if any of the following occur:

- the policy still behaves as universal rejection with no practical difference from current policy
- the policy turns nearly everything into `pass`, indicating indiscriminate rescue behavior
- hard-fail rows are being reclassified as `pass`
- results are only explainable by the continuous score while the documented band logic is not doing the real work

## Primary Stage O Proposal

Primary proposal: **O1: global alternate supported composite stability policy**.

Definition:

- apply the Stage O composite to the relevant K2 runtime rows during validation
- keep current policy as the default comparator
- use the decision matrix above with `pass`, `review_standard`, `review_high`, and `fail`
- use `n_events` only as optional support annotation / within-tier ordering, not as a tier-changing component

Why this is primary:

- it best matches the accepted Stage N direction
- it uses the runtime fields that are already explicit after Stage K and Stage L
- it fixes the current brittleness without immediately asserting that all severe-autocorrelation rows are acceptable
- it keeps the scientific center on `triage_step_score` and `triage_whiteness_statistic_abs_rho`

## Backup Stage O Proposal

Backup proposal: **O2: conditional composite stability policy for the current K2 validation lane**.

Definition:

- apply the same composite structure only within the current high-priority K2 validation lane
- keep all other populations entirely on the existing default policy
- keep `n_events` as optional support-only in this backup as well
- use this if the first paired validation suggests the redesign is useful for this lane but not yet general enough for broader alternate-policy use

Why this is backup:

- it is more conservative operationally
- it fits the accepted Stage N backup direction toward conditional policy if needed
- it avoids claiming broader population validity before evidence exists

## Immediate Outcome

- no code changes
- no production reruns
- no default-policy replacement
- Stage O now defines a concrete, interpretable alternate supported composite policy ready for a minimal paired validation test
