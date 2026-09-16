# Stage R Review Chart Summary

Date: 2026-04-23

Source data: `k2_stage_r_manual_review_sheet.csv`

Chart output: `k2_stage_r_review_chart.png`

Scope: visualization only. This artifact does not change policy, does not widen Stage O, and does not assign reviewer decisions.

## Chart Design

The chart visualizes the 12 selected EPICs from the Stage R manual-review sheet, grouped by `selection_bucket`:

- `highest_composite_score`
- `median_composite_score`
- `lowest_composite_score`

Each row is shown as a horizontal bar using `stage_o_composite_stability_score`. Rows are sorted descending by composite score within each bucket and labeled with their overall descending score rank. Each bar is annotated with:

- `n_events`
- `best_depth_snr`
- `best_shape_score`

## Bucket Summary

| selection_bucket | rows | min score | median score | max score |
|---|---:|---:|---:|---:|
| highest_composite_score | 4 | 29.669 | 29.807 | 29.895 |
| median_composite_score | 4 | 28.457 | 28.464 | 28.477 |
| lowest_composite_score | 4 | 27.037 | 27.174 | 27.399 |

## Overall Score Order

| rank | epic_id | selection_bucket | Stage O score |
|---:|---|---|---:|
| 1 | EPIC_211529255 | highest_composite_score | 29.895 |
| 2 | EPIC_212007631 | highest_composite_score | 29.817 |
| 3 | EPIC_211805106 | highest_composite_score | 29.797 |
| 4 | EPIC_211759736 | highest_composite_score | 29.669 |
| 5 | EPIC_211955365 | median_composite_score | 28.477 |
| 6 | EPIC_211537297 | median_composite_score | 28.468 |
| 7 | EPIC_211525754 | median_composite_score | 28.461 |
| 8 | EPIC_211421186 | median_composite_score | 28.457 |
| 9 | EPIC_211530033 | lowest_composite_score | 27.399 |
| 10 | EPIC_212003686 | lowest_composite_score | 27.310 |
| 11 | EPIC_211972767 | lowest_composite_score | 27.037 |
| 12 | EPIC_211490307 | lowest_composite_score | 27.037 |

## Review Use

Use the chart as a navigation aid for manual review. The bar lengths show relative Stage O composite score position, while the annotations provide detector-context fields that reviewers can compare against event morphology and artifact checks. The chart intentionally does not recommend promotion, hold, or rejection outcomes.
