# Stage Q2 Top-12 Review Slice Summary

Date: 2026-04-23

Scope: slice selection only. The current official policy is unchanged, production execution remains paused, and Stage O is not widened beyond the 12 selected rows.

## Inputs

- Stage P paired validation output: `plots/k2_batch/k2_stage_p_stage_o_paired_validation.csv`
- Stage L post-patch result/artifact fields: `plots/k2_batch/k2_stage_l_batch_001b_postpatch_results.csv`
- Earlier stratified Stage Q slice: `k2_stage_q_review_high_validation_slice.csv`
- New top-12 slice: `k2_stage_q2_top12_review_slice.csv`

## Selection Rule

Selected the top 12 rows with `stage_o_decision_tier = review_high`, ordered by:

1. descending `stage_o_composite_stability_score`
2. descending `n_events`
3. descending `best_shape_score`
4. ascending `new_execution_order`

No policy thresholds were changed, no rows outside `review_high` were included, and the slice remains capped at 12 rows.

## Selected EPICs

| rank | epic_id | query | new_execution_order | Stage O score | n_events | best_depth_snr | best_shape_score |
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

## Ranges

- Stage O composite stability score: `29.060` to `29.895`
- `n_events`: `18` to `44`
- `best_depth_snr`: `30.686` to `194.977`
- `best_shape_score`: `0.708` to `0.822`

## Difference From Earlier Stratified Stage Q Slice

The earlier Stage Q slice was stratified: 4 highest-score rows, 4 median-score rows, and 4 lowest-score rows from the 100-row `review_high` lane. This Q2 slice is not stratified; it is a best-chance slice containing the top 12 rows by Stage O composite stability score with the requested tie-breakers.

Rows retained from the earlier stratified slice (4): `EPIC_211529255, EPIC_212007631, EPIC_211805106, EPIC_211759736`.

Rows newly added by Q2 (8): `EPIC_211698887, EPIC_212023491, EPIC_211324054, EPIC_211394018, EPIC_211633247, EPIC_211791780, EPIC_211945111, EPIC_211836788`.

Rows removed from the earlier stratified slice (8): `EPIC_211530033, EPIC_212003686, EPIC_211972767, EPIC_211490307, EPIC_211955365, EPIC_211537297, EPIC_211525754, EPIC_211421186`.

Operationally, Q2 keeps the same 4 high-score rows from Stage Q and replaces all 8 median/low stratification rows with ranks 5-12 from the score-ordered `review_high` pool. This gives the Stage O alternate policy its fairest best-case validation chance while staying within the original 12-row manual-review limit.

## Packaging Fields Preserved

`k2_stage_q2_top12_review_slice.csv` preserves the Stage Q review-package schema, including Stage O decision fields, triage stability fields, detector support fields, period counters, saved/planned support fields, artifact paths, and Stage Q manual-review action/follow-up fields.

## Immediate Outcome

Created `k2_stage_q2_top12_review_slice.csv` and `k2_stage_q2_top12_review_summary.md` only. No policy was changed, the slice was not widened beyond 12 rows, and no production execution was continued.
