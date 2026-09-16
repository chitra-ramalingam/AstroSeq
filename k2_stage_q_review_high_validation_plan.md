# Stage Q Review-High Validation Plan

Date: 2026-04-22

Scope: operational validation planning only. The current official default policy is not replaced, full production execution is not continued, and the Stage O alternate policy is not widened beyond validation.

## Inputs

- Stage P paired validation CSV: `plots/k2_batch/k2_stage_p_stage_o_paired_validation.csv`
- Stage L post-patch results: `plots/k2_batch/k2_stage_l_batch_001b_postpatch_results.csv`
- Proposed Stage Q slice: `k2_stage_q_review_high_validation_slice.csv`

## Stage P Starting Point

Stage P converted the existing calibration slice from:

- current official policy: `Noisy_trash = 100/100`
- Stage O alternate policy: `review_high = 100/100`, `pass = 0/100`, `fail = 0/100`

This is useful but intentionally not decisive. Stage O did not rescue rows indiscriminately; it moved them into a conservative high-risk review lane. Stage Q therefore tests whether that review lane contains any cases worth deeper evaluation.

## Smallest Safe Review Slice

Primary slice size: **12 rows**.

Selection rule:

- select `4` rows with the highest Stage O composite stability scores
- select `4` rows closest to the median Stage O composite stability score
- select `4` rows with the lowest Stage O composite stability scores
- break ties by higher `n_events`, then lower `new_execution_order`
- keep only rows with `stage_o_decision_tier = review_high`

Reasoning:

- all 100 Stage P rows share the same categorical Stage O outcome: `A2_severe`, `S0_low`, `N1_strong`, `E2_strong`, and underflow escalation
- a full 100-row manual review is unnecessary as the next step
- a 12-row stratified slice is large enough to test whether the review lane has any signal-bearing cases while small enough to avoid turning validation into production

Why not use the full 100 immediately:

- the Stage O lane is still scientifically high risk
- all rows retain severe autocorrelation and underflow escalation
- the immediate question is not yield estimation; it is whether any `review_high` row merits deeper evaluation at all

## Required Manual Review

Each row in the Stage Q slice should be reviewed against the existing artifacts before any new execution.

Required checks:

- inspect `events_csv` for repeated event timing, depth consistency, and obvious single-cadence/systematic artifacts
- inspect available EPIC output directory artifacts listed in `epic_dir`
- compare `n_events`, `best_depth_snr`, and `best_shape_score` against the event table
- confirm that high autocorrelation risk is understood and not ignored
- record a manual outcome: `promote_to_deeper_eval`, `hold_review_high`, or `reject_as_noise_or_artifact`

No row should move directly from `review_high` to accepted science output in Stage Q.

## Downstream Follow-Up

Downstream follow-up is allowed only for rows manually promoted from the 12-row slice.

Allowed deeper evaluation:

- targeted period/event validation for promoted rows only
- artifact inspection and reviewer notes
- no full-lane batch continuation
- no change to the default policy

Not allowed:

- promoting Stage O to default
- running later production batches
- widening Stage O to the full high-priority lane before review evidence exists

## Promotion Evidence

A `review_high` row may be promoted to deeper evaluation only if manual review finds:

- repeated event morphology that is coherent across events
- event timing that is not obviously random, cadence-driven, or artifact-driven
- no obvious contamination, quality artifact, or processing failure explaining the event structure
- enough detector support to make deeper evaluation meaningful
- reviewer acceptance that severe autocorrelation remains a risk, not a solved issue

Promoted rows should remain tagged as Stage O review-lane cases, not default-policy passes.

## Falsification Evidence

The usefulness of Stage O as an alternate review policy is falsified for this lane if:

- the 12-row review slice yields `0` rows worth promoting to deeper evaluation
- review cannot distinguish any row from red-noise/autocorrelation structure
- event tables are dominated by obvious artifacts, non-repeatable dips, or cadence/systematic patterns
- rows that look strongest by Stage O score are not meaningfully better than rows with lower Stage O score
- reviewers conclude that `review_high` adds workload but no actionable cases

If falsified, Stage O should remain a diagnostic tool only, not an operational alternate lane.

## Metrics To Record

For the 12-row slice, record:

- number reviewed
- number promoted to deeper evaluation
- number held as unresolved `review_high`
- number rejected as noise/artifact
- promotion rate by selection bucket
- whether promoted rows cluster in high-score, median-score, or low-score buckets
- reviewer notes on whether `n_events` helped only as support, not as a tier override

## Primary Next Experiment

Primary experiment: **12-row stratified manual review of the Stage O `review_high` lane**.

Use `k2_stage_q_review_high_validation_slice.csv`.

Decision after review:

- if at least `2` of `12` rows are promoted to deeper evaluation and none violate hard-fail expectations, proceed to targeted deeper evaluation for promoted rows only
- if `0` of `12` are promoted, treat Stage O as not operationally useful for this lane without redesign
- if `1` of `12` is promoted, hold and review whether the signal is strong enough to justify a second small slice

## More Conservative Backup Experiment

Backup experiment: **6-row pilot review**.

Selection:

- `2` highest composite-score rows
- `2` median composite-score rows
- `2` lowest composite-score rows

Use this if manual-review capacity is tight or if reviewers want an initial calibration pass before committing to 12 rows.

Decision after backup:

- if `0/6` promote, stop and do not expand
- if `1-2/6` promote, expand to the full 12-row Stage Q slice
- do not proceed beyond manual review without an explicit follow-up decision

## Slice Summary

- candidate pool: `100` Stage O `review_high` rows
- selected rows: `12`
- selected score range: `27.037` to `29.895`
- full pool score range: `27.037` to `29.895`

## Immediate Outcome

- created `k2_stage_q_review_high_validation_slice.csv`
- no default-policy replacement
- no production continuation
- no Stage O widening beyond validation
