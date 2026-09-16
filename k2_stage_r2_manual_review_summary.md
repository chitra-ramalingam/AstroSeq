# Stage R2 Manual Review Package Summary

Date: 2026-04-23

Scope: manual-review packaging only. The current official default policy is unchanged, production execution remains paused, and Stage O is not widened beyond the 12 selected Q2 rows.

## Inputs

- Active Q2 review slice: `k2_stage_q2_top12_review_slice.csv`
- Manual review sheet: `k2_stage_r2_manual_review_sheet.csv`

## Review Sheet

The R2 manual review sheet contains one row per selected EPIC and includes:

- Stage O score, risk class, decision tier, and component bands
- runtime stability fields: `triage_step_score`, `triage_whiteness_statistic_abs_rho`, `triage_whiteness_underflowed`, and `triage_n_points`
- detector support fields: `n_events`, `best_depth_snr`, and `best_shape_score`
- artifact references: `events_csv` and `epic_dir`
- blank reviewer fields for structured manual review

Allowed `reviewer_outcome` values are restricted to:

- `promote_to_deeper_eval`
- `hold_review_high`
- `reject_as_noise_or_artifact`

## Selected EPICs In Review Order

| R2 rank | epic_id | query | new_execution_order | Stage O score | n_events | best_depth_snr | best_shape_score |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | EPIC_211529255 | EPIC 211529255 | 89 | 29.895 | 18 | 38.721 | 0.769 |
| 2 | EPIC_212007631 | EPIC 212007631 | 19 | 29.817 | 44 | 37.616 | 0.753 |
| 3 | EPIC_211805106 | EPIC 211805106 | 18 | 29.797 | 44 | 37.832 | 0.713 |
| 4 | EPIC_211759736 | EPIC 211759736 | 1 | 29.669 | 29 | 194.977 | 0.720 |
| 5 | EPIC_211698887 | EPIC 211698887 | 46 | 29.560 | 23 | 48.195 | 0.719 |
| 6 | EPIC_212023491 | EPIC 212023491 | 9 | 29.519 | 31 | 63.383 | 0.760 |
| 7 | EPIC_211324054 | EPIC 211324054 | 72 | 29.480 | 21 | 42.319 | 0.708 |
| 8 | EPIC_211394018 | EPIC 211394018 | 59 | 29.419 | 32 | 30.686 | 0.721 |
| 9 | EPIC_211633247 | EPIC 211633247 | 8 | 29.265 | 42 | 70.245 | 0.748 |
| 10 | EPIC_211791780 | EPIC 211791780 | 70 | 29.243 | 28 | 34.103 | 0.822 |
| 11 | EPIC_211945111 | EPIC 211945111 | 36 | 29.086 | 34 | 40.060 | 0.765 |
| 12 | EPIC_211836788 | EPIC 211836788 | 11 | 29.060 | 22 | 114.831 | 0.775 |

## Practical First-Review Order

Recommended first pass: start with the rows that have the strongest detector support inside the fixed Q2 top-12 set. This does not change the active review set or policy; it is only an efficient order for human inspection.

| first-pass order | epic_id | Q2 rank | n_events | best_shape_score | best_depth_snr | Stage O score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | EPIC_212007631 | Q2 rank 2 | 44 | 0.753 | 37.616 | 29.817 |
| 2 | EPIC_211805106 | Q2 rank 3 | 44 | 0.713 | 37.832 | 29.797 |
| 3 | EPIC_211633247 | Q2 rank 9 | 42 | 0.748 | 70.245 | 29.265 |
| 4 | EPIC_211945111 | Q2 rank 11 | 34 | 0.765 | 40.060 | 29.086 |
| 5 | EPIC_211394018 | Q2 rank 8 | 32 | 0.721 | 30.686 | 29.419 |
| 6 | EPIC_212023491 | Q2 rank 6 | 31 | 0.760 | 63.383 | 29.519 |
| 7 | EPIC_211759736 | Q2 rank 4 | 29 | 0.720 | 194.977 | 29.669 |
| 8 | EPIC_211791780 | Q2 rank 10 | 28 | 0.822 | 34.103 | 29.243 |
| 9 | EPIC_211698887 | Q2 rank 5 | 23 | 0.719 | 48.195 | 29.560 |
| 10 | EPIC_211836788 | Q2 rank 12 | 22 | 0.775 | 114.831 | 29.060 |
| 11 | EPIC_211324054 | Q2 rank 7 | 21 | 0.708 | 42.319 | 29.480 |
| 12 | EPIC_211529255 | Q2 rank 1 | 18 | 0.769 | 38.721 | 29.895 |

## Reviewer Checklist

For each row:

- Open the row's `events_csv`.
- Inspect repeated event timing, depth consistency, and shape consistency.
- Inspect the row's `epic_dir` for available artifacts.
- Compare `n_events`, `best_depth_snr`, and `best_shape_score` to the event table.
- Record `reviewer_outcome` using only the allowed values.
- Record `reviewer_confidence`.
- Fill in `repeated_event_morphology`, `timing_coherence`, `obvious_artifact_or_contamination`, and `notes`.
- Set `promote_to_deeper_eval` only when the row has enough coherent evidence for targeted deeper evaluation.

## Promotion Rule

After all 12 rows are reviewed:

- promoted >= 2: proceed to deeper evaluation for promoted rows only.
- promoted == 1: pause and reassess before expanding or running another slice.
- promoted == 0: treat Stage O as not operationally useful for this lane without redesign.

Promoted rows remain Stage O review-lane cases. They are not default-policy passes.

## Artifact Availability

Available in the R2 sheet:

- `events_csv`: 12/12 rows
- `epic_dir`: 12/12 rows

Missing or blank in the Q2 source for all 12 rows:

- `best_hits_csv`
- `best_misses_csv`
- `best_uncovered_csv`
- `best_hitmap_png`
- `best_phase_offset_png`

Review quality is sufficient for manual review using event tables and EPIC directories, but phase/hitmap-specific inspection remains limited unless those artifacts are generated or located separately.

## Immediate Outcome

Created `k2_stage_r2_manual_review_sheet.csv` and `k2_stage_r2_manual_review_summary.md` from the Q2 best-chance slice. No policy was changed, the slice was not widened beyond 12 rows, and production execution was not continued.
