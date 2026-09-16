# Stage M Whiteness Policy Decision Analysis

Date: 2026-04-17

Scope: decision analysis only. No code was patched, no new science was run, and no later execution batches were continued in this step.

## Executive Decision

Primary recommendation: **C: redefine how whiteness is computed or interpreted scientifically**.

Backup recommendation: **D: split policy by population type and apply whiteness conditionally**.

Rejected as primary actions:

- **A** is a defensible conservative hold, but it answers the problem by abandoning the high-priority lane rather than resolving the scientific mismatch now exposed by Stages H, I, and L.
- **B** is not scientifically credible under the observed evidence because the runtime failures are not marginal threshold misses.

## Evidence Base Used

This analysis uses:

- Stage F batch 001
- Stage F batch 001b
- Stage G population-level whiteness audit
- Stage H whiteness-policy diagnosis
- Stage I implementation audit
- Stage L post-patch rerun

## Key Evidence

### 1. Stage F batch 001 established a real lane-level failure, not an execution failure

- `rows_attempted = 100`
- `rows_completed = 100`
- `rows_failed = 0`
- `final_label = Noisy_trash: 100`
- dominant reason: `usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)`

Interpretation:

- The lane failed scientifically under the official policy, not operationally.
- The executed rows had strong planned detector signal, so this was not a simple weak-candidate batch.

### 2. Stage F batch 001b showed the failure was not fixed by queue/routing changes

- `rows_attempted = 100`
- `rows_completed = 100`
- `rows_failed = 0`
- original `Noisy_trash = 100`
- patched `Noisy_trash = 100`
- original whiteness rejection count `= 100`
- patched whiteness rejection count `= 100`
- `whiteness_rejection_frequency_improved = no`
- `enough_evidence_to_proceed_to_later_batches = no`

Interpretation:

- Reordering the lane did not change the scientific outcome.
- This weakened the case for execution-order or queue-selection fixes as the next lever.

### 3. Stage G showed the problem generalizes beyond an isolated bad first slice

Population-level saved-proxy context from `plots/k2_batch/batch_results_whiteness.csv`:

- full pool rows: `25771`
- saved `triage_whiteness_pvalue` min `= 0.01729`
- saved `triage_whiteness_pvalue` q25 `= 0.96464`
- saved `triage_whiteness_pvalue` median `= 0.99301`
- saved `triage_whiteness_pvalue` q90 `= 0.99953`
- saved `triage_whiteness_pvalue < 0.01`: `0`

Stage G calibration interpretation:

- both calibration batches were consistent with the broader pool profile rather than an isolated pathological tail
- conservative low-yield estimate under the current policy was at least `85.8%`
- Stage G recommendation: revisit whiteness policy scientifically

Interpretation:

- The current saved proxy view does not identify the calibration rows as obvious whiteness failures.
- This is evidence against a narrow reranking fix and toward a policy-definition problem.

### 4. Stage H isolated a scientific non-equivalence, not just a threshold problem

Across both calibration batches combined:

- `final_label = Noisy_trash: 200`
- saved `triage_whiteness_pvalue` median `= 0.999529`
- runtime `triage_whiteness_score` median `= 0.0`
- runtime `triage_whiteness_zero_count = 200`
- saved/runtime whiteness definition same: `yes`
- runtime step score exactly matches saved step score: `200/200`
- runtime score_global median `= -1.0`
- Stage H recommendation: `C: redefine how whiteness is computed/interpreted`

Interpretation:

- The same named whiteness definition behaves as if it is not the same effective scientific quantity across saved and runtime contexts.
- Because step scores match exactly, the mismatch is specifically in whiteness behavior, not in the broader row reconstruction.

### 5. Stage I removed the sentinel/missingness explanation

Stage I established:

- runtime `0.0` is not a hard-failure sentinel
- missing or non-computable whiteness is represented as `NaN`, not `0.0`
- runtime `0.0` is consistent with p-value underflow from `math.erfc(...)`
- saved/runtime interpretation rules are consistent when a field is semantically a p-value

Interpretation:

- The representation bug hypothesis was narrowed to field semantics and information loss, not to policy behavior itself.
- This made a minimal additive representation patch the correct next step before any scientific decision.

### 6. Stage L confirmed the representation fix worked and the scientific failure remained

Stage L summary:

- `rows_attempted = 100`
- `rows_completed = 100`
- `rows_failed = 0`
- `triage_whiteness_underflowed = True: 100`
- `triage_whiteness_pvalue == 0.0: 100`
- `triage_whiteness_log10_pvalue finite: 100`
- `legacy_score_explicit_pvalue_agree_in_pvalue_mode = 100`
- `patch_successfully_resolved_saved_runtime_comparability_problem = yes`
- `batch_still_scientifically_rejected_after_representation_fix = yes`

Stage L audit numerics:

- runtime `log10_pvalue` median about `-690.19`
- runtime `log10_pvalue` range about `[-730.85, -433.76]`
- runtime `z` median about `56.30`
- runtime `z` range about `[44.60, 57.94]`

Interpretation:

- The failures are astronomically below the official threshold `0.01`, not near it.
- That makes a simple threshold relaxation scientifically implausible.
- Stage L therefore converts the problem from an implementation ambiguity into a policy decision.

## Policy Option Evaluation

The structured table is in `k2_stage_m_whiteness_policy_options.csv`. The decision logic is:

### A. Keep the current policy unchanged and stop using this high-priority lane

Strengths:

- maximally conservative
- fully compatible with the existing default-policy posture
- avoids spending more time on a lane that currently returns zero yield under the official rule

Weaknesses:

- does not explain the strong saved-versus-runtime mismatch now shown to be real after Stage L
- effectively concedes the lane without resolving whether the current whiteness construct is scientifically wrong for this population

Decision:

- acceptable operational hold
- not the best scientific action

### B. Relax the whiteness threshold

Strengths:

- lowest apparent process-change cost
- would be attractive only if the failures were modest near-threshold misses

Weaknesses:

- Stage L shows the failures are not modest
- accepting p-values with median log10 near `-690` would not be a tuning change; it would effectively abandon the meaning of the current test
- Stage H indicates the root problem is not just threshold placement

Decision:

- reject

### C. Redefine how whiteness is computed or interpreted scientifically

Strengths:

- directly addresses the root problem suggested by Stages H, I, and L
- consistent with the evidence that saved and runtime whiteness are not scientifically interchangeable even when the naming is aligned
- offers a path to preserve a strict gate while making it scientifically meaningful for this K2 lane

Weaknesses:

- highest scientific design cost
- requires new validation before operational use
- must be introduced carefully so the current official policy remains interpretable as the conservative baseline

Decision:

- primary recommendation

### D. Split policy by population type and apply whiteness conditionally

Strengths:

- more targeted than a blanket threshold relaxation
- fits well with the repo's existing pattern of keeping a conservative default while allowing explicitly supported alternate modes

Weaknesses:

- current evidence does not yet identify the correct population split
- Stage G argues against a naive split based only on current saved whiteness proxies
- without a real scientific taxonomy, this becomes an ad hoc carve-out

Decision:

- backup recommendation

## Compatibility With Existing Default-Policy Framing

There is no explicit Stage A source file in the current repo snapshot, so compatibility is interpreted from the established repo-wide policy framing already used elsewhere:

- keep the conservative default unchanged unless evidence clearly justifies an alternate policy
- if an alternate policy is supported, make it explicit, bounded, and guardrailed rather than silently replacing the default

Under that framing:

- **A** is maximally compatible
- **B** is only superficially compatible, because the required relaxation would be too extreme to still function as a conservative default
- **C** is compatible if the current rule stays as the default and the redefined whiteness rule becomes an explicitly justified supported policy
- **D** is compatible if the conditional split is explicit, scientifically justified, and guardrailed

## Final Recommendation

Primary recommendation: **C: redefine how whiteness is computed or interpreted scientifically**.

Why:

- Stage L closed the representation question.
- Stage F batch 001b showed that reranking or queue adjustment did not improve yield.
- Stage H and Stage L together show that the present runtime whiteness construct is not merely too strict by a small amount. It is behaving as a fundamentally different scientific discriminator than the saved proxy view.
- Because the runtime p-values are astronomically below threshold, **B** is not a serious option.
- Because the lane is explicitly high priority, **A** is too conservative as the main scientific action unless the project decides to retire the lane entirely.

Backup recommendation: **D: split policy by population type and apply whiteness conditionally**.

Why:

- If the next scientific review concludes that the current whiteness construct is valid only in some K2 subpopulations, a conditional policy is the least blunt way to preserve the conservative default while recovering lane utility.
- It should only be adopted after the population split is scientifically defined; the present evidence does not yet justify a proxy-only carve-out.

## Immediate Decision Outcome

- Do not continue later batches under the unchanged current whiteness policy.
- Do not spend more time on representation fixes.
- Treat the next phase as a scientific policy-design problem centered on option **C**, with **D** as the fallback framing if the science points to a population-conditional rule rather than a single replacement measure.
