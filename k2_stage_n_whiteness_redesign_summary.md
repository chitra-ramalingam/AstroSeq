# Stage N Whiteness Redesign Proposal

Date: 2026-04-19

Scope: scientific redesign proposal only. No code was patched, no production batches were rerun, and no policy was changed in this step.

## Executive Recommendation

Primary redesign recommendation: **N2: composite stability score using whiteness plus step/stability metrics**.

Backup redesign recommendation: **N4: population-conditional whiteness rule**.

Minimum validation experiment before adoption:

- define one pre-registered redesign candidate
- run one paired calibration rerun on the already-used high-priority calibration slice under:
  - current official policy
  - proposed redesign policy
- require that the redesign eliminates the universal whiteness-collapse behavior without obvious quality collapse in the resulting accepted rows

## Why Stage N Is Needed

Stage M accepted that the next step is not threshold tuning and not more representation cleanup. The remaining problem is scientific redesign.

The core evidence chain is now stable:

- **Stage G**: the failed calibration batches were not isolated in an obviously pathological saved-proxy tail
- **Stage H**: saved and runtime whiteness behaved as non-equivalent scientific quantities despite matching definition text
- **Stage I**: runtime `0.0` was shown to be underflow, not sentinel missingness
- **Stage L**: representation ambiguity was removed, but the scientific rejection remained universal
- **Stage M**: selected `C` as the main policy direction and `D` as backup

That means the redesign target is the current scientific role of `lag1_autocorr_pvalue_normal_approx`, especially its use as a single hard reject gate.

## Design Requirements Implied By The Evidence

Any defensible redesign should satisfy all of the following:

1. It must remove dependence on a quantity that can collapse to exact `0.0` for an entire calibration slice while still being reported as scientifically decisive.
2. It must preserve interpretability. The replacement cannot become an opaque rescue heuristic.
3. It must respect the repo's conservative-default framing: alternate behavior should be explicit and justified before any default change.
4. It must use information already shown to matter. Stage H showed that step-score remained stable while whiteness diverged, so redesign should not ignore other stability evidence.
5. It must be testable on the already-established calibration slice before any broader rollout.

## Evidence Summary

### Stage G

Population-level saved-proxy evidence argues against a simple explanation that batch 001 and 001b were just drawn from the worst saved-whiteness tail.

Relevant implications:

- the redesign should not assume the solution is merely better queue ordering
- the redesign should explain why saved proxies looked broadly healthy while runtime whiteness catastrophically failed

### Stage H

Stage H is the strongest direct evidence for redesign:

- saved whiteness remained strong
- runtime whiteness collapsed to `0.0`
- step-score matched exactly across saved and runtime views

Scientific implication:

- the failure is not a generic instability problem across all triage metrics
- the redesign should treat current whiteness handling as the mis-specified component, not the whole stability stack

### Stage I

Stage I showed:

- `0.0` is not missingness
- the current runtime p-value is a real numeric consequence of the implemented transform

Scientific implication:

- the redesign should avoid sample-size-amplified p-value behavior as the decisive scientific object

### Stage L

Stage L showed:

- explicit p-value, log10 p-value, z, and underflow fields all agree
- the calibration slice still fails 100/100 under current policy
- the runtime p-values are astronomically below threshold, not slightly below it

Scientific implication:

- threshold relaxation is not the right redesign path
- the redesign must change either the measured quantity, the policy role of the quantity, or both

### Stage M

Stage M concluded:

- primary path: redefine whiteness scientifically
- backup path: conditional policy by population type
- not acceptable as main path: threshold relaxation

Scientific implication:

- Stage N should focus on defensible replacements for the current whiteness construct and policy role

## Redesign Options

The structured table is in `k2_stage_n_whiteness_redesign_options.csv`.

### N1. Replace lag-1 p-value with a robust autocorrelation effect-size descriptor

This option keeps the intuition that local autocorrelation matters, but rejects the use of a significance p-value as the main scientific quantity.

Scientific idea:

- measure autocorrelation magnitude or a short-lag correlation bundle directly
- report a bounded effect-size descriptor rather than a p-value

Why it helps:

- removes the underflow-prone p-value transform
- prevents huge sample size from turning moderate structural correlation into an all-or-nothing reject signal

Why it is not the primary recommendation:

- it fixes the measurement scale, but not necessarily the policy shape
- a single replacement metric could still be too brittle

### N2. Composite stability score using whiteness plus step/stability metrics

This is the strongest redesign because it addresses both identified weaknesses:

- the current whiteness measurement is scientifically mismatched
- the current policy gives one metric too much veto power

Scientific idea:

- replace p-value whiteness with a robust effect-size-style whiteness component
- combine it with step-score and limited additional stability evidence into a calibrated stability score or risk class
- keep the design interpretable and explicitly documented

Why it best fits the evidence:

- Stage H already proved that step-score retains meaningful agreement while whiteness does not
- Stage L proved the current single-metric gate is too brittle
- Stage G suggests the failure is broader than a queue-specific artifact

Recommended policy posture:

- first introduce as an alternate supported policy
- do not make it the global default until calibration evidence is strong

### N3. Reinterpret whiteness as a ranking/calibration feature rather than a hard reject gate

This option changes the role of whiteness more than the measurement itself.

Scientific idea:

- keep a whiteness-like descriptor
- demote it from absolute gate to ranking, review, or calibration signal

Why it is useful:

- avoids universal hard rejection from one unstable quantity
- preserves possible information content in the metric

Why it is not primary:

- it is a larger policy-role change than N2
- if the underlying measurement remains scientifically weak, this can become a soft version of the same problem

### N4. Population-conditional whiteness rule

This is the best backup if later science shows the current mismatch is regime-specific rather than global.

Scientific idea:

- define subpopulations where whiteness should be measured or used differently
- preserve the current conservative rule where it is valid
- apply alternate handling only in scientifically justified regimes

Why it is only backup now:

- Stage G supports population-aware thinking but does not yet provide a defensible split variable
- without a pre-defined taxonomy, a conditional rule would look like post hoc rescue logic

## Primary Recommendation

Primary redesign recommendation: **N2: composite stability score using whiteness plus step/stability metrics**.

Reasoning:

- It directly addresses the Stage H observation that step-score remains coherent while whiteness does not.
- It solves the Stage L problem at the right level: not by loosening the threshold, but by stopping one underflow-prone p-value from acting as a universal veto.
- It still preserves a conservative scientific posture because the redesign can be introduced first as an explicit alternate supported policy rather than an immediate default replacement.
- It is more robust than N1 because it changes both the quantity and the decision structure.

## Backup Recommendation

Backup redesign recommendation: **N4: population-conditional whiteness rule**.

Reasoning:

- If subsequent scientific review shows that the current failure is concentrated in a real K2 subpopulation with distinct noise behavior, a conditional rule is the cleanest way to preserve a conservative default while recovering useful cases.
- It should only proceed after the split variable is defined independently of the rerun result.

## Minimum Validation Experiment

Before adopting any redesign, the minimum experiment should be:

1. Pre-register one candidate redesign.
2. Freeze the redesign formula and decision rule before rerun.
3. Reuse the existing high-priority calibration slice rather than expanding scope.
4. Run a paired comparison:
   - current official policy
   - proposed redesign policy
5. Compare:
   - whether universal whiteness rejection disappears
   - how many rows move from `Noisy_trash` to accepted or reviewable status
   - whether accepted rows show immediate evidence of quality collapse on already-available stability signals

Minimum success condition:

- the redesign must break the current `100/100` universal-whiteness-fail pattern on the calibration slice
- without replacing it with an obviously indiscriminate acceptance rule

## Proposed Adoption Sequence

1. Start with **N2** as an alternate supported policy, not an immediate default replacement.
2. Keep **N4** in reserve only if the validation evidence suggests regime-specific rather than global mis-specification.
3. Revisit default-policy status only after the paired calibration rerun and follow-up scientific review.

## Immediate Outcome

- no code changes
- no production reruns
- no policy changes
- Stage N now defines the scientifically preferred redesign direction and the minimum validation needed before any implementation step
