# Phase 2 Split and Evaluation Plan

## Leakage-safe split policy

Create a canonical `epic_id` group key before any row expansion. All candidate periods, detections, segments, planets in multi-planet systems, aliases, and P/2–P–2P views for one EPIC stay in one split. Deduplicate/resolve label evidence before splitting. Confirmed systems repeated across files become one host group with a provenance list; conflicting physical labels are quarantined.

Preferred campaign split after multi-campaign ingestion: train on C5 plus several nonadjacent campaigns (for example C1–C4 and C6–C8), validate on C9–C11, and keep C12–C18 as an untouched blind campaign group, subject to class coverage. Do not hard-code these assignments until campaign counts are audited. With the current C5-only processed data, use grouped repeated stratified cross-validation for development and reserve a deterministic 20% EPIC blind set, stratified by class/SNR/period—but recognize that this tests within-campaign generalization only. Never tune on the blind set.

Near-duplicate periods are not independent examples: assign by EPIC first. If external catalogues reveal blends/duplicate targets for the same physical source, add a sky-position/source-system group and keep the entire group together.

## Evaluation

Report per-class precision/recall and confusion matrix, candidate recall and precision, candidate one-vs-rest PR-AUC, EB/variable recall, noise/artifact recall, and false-promotion rate = negative EPICs predicted candidate_like / all negative EPICs. For the period head report exact P/2/P/2P/unresolved accuracy, macro-F1, and confusion matrix. Calibration: multiclass log loss, Brier score, expected calibration error, and reliability plots. Select a confidence threshold on validation only; report manual-review fraction and selective risk/coverage.

Slice every metric by SNR bins, log-period bins, campaign, event count/coverage, missing-feature pattern, and the predeclared high-CNN-score hard-negative cohort. Bootstrap confidence intervals by EPIC, not row.

## First-baseline success criterion

Before seeing the blind set, define operating points. A useful first baseline should improve candidate PR-AUC over CNN-score-only and majority/class-prior baselines; retain ≥90% candidate recall on validation (with confidence intervals reported); reduce false promotions among high-CNN hard negatives by at least 30% relative to the CNN-only operating point; achieve ≥80% recall for each negative superclass where sample size supports estimation; show no catastrophic campaign/SNR slice; and produce calibrated or post-calibratable probabilities. Any threshold must route low-confidence cases to manual review. Exact deployment thresholds require more reliable positives and are not claimed by this audit.
