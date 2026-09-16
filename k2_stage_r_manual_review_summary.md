# Stage R Manual Review Package Summary

Date: 2026-04-22

Scope: manual-review packaging only. The current official default policy is unchanged, production execution remains paused, and Stage O is not widened beyond the 12 selected Stage Q rows.

## Inputs

- Stage Q review slice: `k2_stage_q_review_high_validation_slice.csv`
- Stage P paired validation output: `plots/k2_batch/k2_stage_p_stage_o_paired_validation.csv`
- Stage L post-patch results: `plots/k2_batch/k2_stage_l_batch_001b_postpatch_results.csv`
- Manual review sheet: `k2_stage_r_manual_review_sheet.csv`

## Review Sheet

The manual review sheet contains one row per selected EPIC and includes:

- Stage O score, risk class, decision tier, and component bands
- runtime stability fields: `triage_step_score`, `triage_whiteness_statistic_abs_rho`, `triage_whiteness_underflowed`, and `triage_n_points`
- detector support fields: `n_events`, `best_depth_snr`, and `best_shape_score`
- artifact references: `events_csv` and `epic_dir`
- blank reviewer fields for structured manual review

Allowed `reviewer_outcome` values are restricted to:

- `promote_to_deeper_eval`
- `hold_review_high`
- `reject_as_noise_or_artifact`

## Selected EPICs By Bucket

### High-score bucket

| epic_id | query | new_execution_order | Stage O score | n_events | best_depth_snr | best_shape_score |
|---|---:|---:|---:|---:|---:|---:|
| EPIC_211529255 | EPIC 211529255 | 89 | 29.895 | 18 | 38.721 | 0.769 |
| EPIC_212007631 | EPIC 212007631 | 19 | 29.817 | 44 | 37.616 | 0.753 |
| EPIC_211805106 | EPIC 211805106 | 18 | 29.797 | 44 | 37.832 | 0.713 |
| EPIC_211759736 | EPIC 211759736 | 1 | 29.669 | 29 | 194.977 | 0.720 |

### Median-score bucket

| epic_id | query | new_execution_order | Stage O score | n_events | best_depth_snr | best_shape_score |
|---|---:|---:|---:|---:|---:|---:|
| EPIC_211955365 | EPIC 211955365 | 41 | 28.477 | 24 | 105.420 | 0.797 |
| EPIC_211537297 | EPIC 211537297 | 31 | 28.468 | 35 | 79.723 | 0.798 |
| EPIC_211525754 | EPIC 211525754 | 52 | 28.461 | 23 | 111.398 | 0.807 |
| EPIC_211421186 | EPIC 211421186 | 39 | 28.457 | 25 | 98.431 | 0.765 |

### Low-score bucket

| epic_id | query | new_execution_order | Stage O score | n_events | best_depth_snr | best_shape_score |
|---|---:|---:|---:|---:|---:|---:|
| EPIC_211530033 | EPIC 211530033 | 100 | 27.399 | 26 | 181.628 | 0.809 |
| EPIC_212003686 | EPIC 212003686 | 99 | 27.310 | 24 | 359.525 | 0.843 |
| EPIC_211972767 | EPIC 211972767 | 68 | 27.037 | 33 | 385.129 | 0.817 |
| EPIC_211490307 | EPIC 211490307 | 98 | 27.037 | 31 | 173.332 | 0.819 |

## Review Questions

For each row, answer:

- Does `events_csv` show repeated event morphology that is coherent across events?
- Is event timing coherent, or does it look random, cadence-driven, or systematic?
- Are there obvious cadence artifacts, systematics, single-cadence dips, or processing artifacts?
- Are there contamination or quality-artifact concerns visible in the available EPIC directory artifacts?
- Is detector support strong enough to justify deeper evaluation despite `A2_severe` autocorrelation and underflow escalation?
- Does `n_events` support the review conclusion without overriding the stability concerns?

## Reviewer Checklist

- Open the row's `events_csv`.
- Inspect repeated event timing, depth consistency, and shape consistency.
- Inspect the row's `epic_dir` for available artifacts.
- Compare `n_events`, `best_depth_snr`, and `best_shape_score` to the event table.
- Record `reviewer_outcome` using only the allowed values.
- Record `reviewer_confidence`.
- Fill in `repeated_event_morphology`, `timing_coherence`, `obvious_artifact_or_contamination`, and `notes`.
- Set `promote_to_deeper_eval` only when the row has enough coherent evidence for targeted deeper evaluation.

## Stage Q Promotion Rule

After all 12 rows are reviewed:

- If at least `2` of `12` promote, proceed to targeted deeper evaluation for promoted rows only.
- If `0` of `12` promote, treat Stage O as not operationally useful for this lane without redesign.
- If `1` of `12` promotes, pause and reassess before expanding or running a second small slice.

Promoted rows remain Stage O review-lane cases. They are not default-policy passes.

## Artifact Availability

Available for all 12 rows:

- `events_csv`
- `epic_dir`

Missing or blank in the current Stage Q source for all 12 rows:

- `best_hits_csv`
- `best_misses_csv`
- `best_uncovered_csv`
- `best_hitmap_png`
- `best_phase_offset_png`

Review quality is therefore sufficient for an immediate first manual review using event tables and EPIC directories, but limited for phase/hitmap-specific inspection unless those artifacts are generated or located separately.

## Immediate Outcome

The Stage R package is ready for human review immediately. It does not replace the default policy, widen Stage O beyond validation, continue production batches, or change code.
