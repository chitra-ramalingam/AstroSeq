# Stage P Paired Validation: Stage O Composite vs Current Policy

Date: 2026-04-22

Scope: paired replay validation only. No code was patched, no production batch was rerun, and the current official default policy was not replaced.

## Inputs

- Source rows: `plots\k2_batch\k2_stage_l_batch_001b_postpatch_results.csv`
- Stage O policy spec: `k2_stage_o_composite_stability_policy_spec.md`
- Rows evaluated: `100`

## Paired Outcome

Current official policy:

- `Noisy_trash`: `100/100`
- dominant reason: `usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)`

Stage O composite policy:

- `review_high`: `100/100`
- `review_standard`: `0/100`
- `pass`: `0/100`
- `fail`: `0/100`

Stage O breaks the current universal automatic rejection pattern, but it does **not** produce automatic passes. All rows are routed to `review_high` because every row has severe autocorrelation (`A2_severe`) and underflow review escalation, while step and sufficiency remain strong.

## Component Results

- Autocorrelation band: `{'A2_severe': 100}`
- Step band: `{'S0_low': 100}`
- N-points band: `{'N1_strong': 100}`
- Event support band: `{'E2_strong': 100}`
- Semantic guard ok: `100/100`
- Underflowed: `100/100`

Key distributions:

- `stage_o_composite_stability_score`: min `27.037`, median `28.473`, max `29.895`
- `triage_step_score`: min `0.000903`, median `0.013092`, max `0.025394`
- `triage_whiteness_statistic_abs_rho`: min `0.991142`, median `0.999716`, max `0.999997`
- `triage_n_points`: min `2004`, median `3172`, max `3360`
- `n_events`: min `15`, median `28`, max `66`
- `triage_whiteness_log10_pvalue`: min `-730.846`, median `-690.188`, max `-433.762`

## Validation Criteria

- `breaks_universal_whiteness_fail`: `True`
- `some_rows_move_to_review_or_pass`: `True`
- `no_hard_fail_rows_become_pass`: `True`
- `reviewable_subset_no_step_or_sufficiency_collapse`: `True`
- `n_events_not_used_as_tier_override`: `True`
- `not_indiscriminate_pass`: `True`

Minimum validation success: **`True`**.

## Scientific Interpretation

Stage O succeeds as a conservative paired replay policy because it replaces `100/100` automatic `Noisy_trash` outcomes with `100/100` `review_high` outcomes while preserving hard-fail protections and avoiding automatic pass inflation.

The result is intentionally cautious. The rows remain scientifically high risk under Stage O because the autocorrelation effect-size component is severe for every row and every row carries underflow escalation. However, the current p-value underflow no longer acts as an unconditional hard veto. Instead, the rows become reviewable high-risk cases with transparent component evidence.

## Decision

Stage P paired validation result: **Stage O passes the minimum replay validation criteria for a review-tier alternate policy**.

This does not justify production adoption or default replacement. It does justify implementing a Stage O validation runner/replayer so the composite can be evaluated reproducibly and then tested on the minimum paired validation slice before any broader policy decision.

## Artifacts

- Paired validation CSV: `plots\k2_batch\k2_stage_p_stage_o_paired_validation.csv`
- Summary CSV: `plots\k2_batch\k2_stage_p_stage_o_paired_validation_summary.csv`
- Summary markdown: `k2_stage_p_stage_o_paired_validation_summary.md`
