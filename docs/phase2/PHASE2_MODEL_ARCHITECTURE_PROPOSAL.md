# Phase 2 Model Architecture Proposal

## Baseline A — learned tabular vetter (train first)

Use CatBoost first: native missing values/categoricals, strong small-data behaviour, class weighting, and SHAP diagnostics. Inputs are leakage-screened numerical diagnostics, missingness indicators, the frozen CNN probability, and optionally a compact PCA projection or regularized subset of the 128-D embedding. Advantages are low complexity, inspectability, fast ablations, and graceful missingness. Limitations are loss of detailed phase morphology and dependence on consistent diagnostics. Expect at least hundreds of reliable examples per major class for a credible four-class result; current effective positives are below that bar. First experiment: class-weighted CatBoost on diagnostics + CNN probability, with uncertain rows excluded from physical loss, EPIC grouping, campaign-aware validation, and feature-family ablations. Add embeddings only after the scalar baseline is stable.

## Baseline B — multi-input neural vetter

Freeze the 128-D CNN encoder; create shared-weight view encoders for P/2, P, and 2P global/local/odd/even/secondary/stack/baseline tensors; concatenate with a masked scalar branch. Heads: four-way physical class, four-way period hypothesis, and review confidence. Advantages are direct learned morphology comparisons and multi-task sharing. Limitations are much higher implementation/QA cost, calibration difficulty, opacity, sensitivity to missing views, and substantially larger balanced data needs—preferably thousands per physical class plus broad campaigns and SNR/period coverage. It is not suitable for the current explicit three-hypothesis volume. First experiment comes only after Phase 2B: frozen encoder, small shared view tower, masked diagnostics, no CNN fine-tuning.

## Required exclusions and uncertain policy

Never input GateVetter final prediction/decision/recommendation, human/master/final label, manual reason, training label/rule, decision authority, or fields derived from those targets. Diagnostic measurements are allowed; rule verdicts should be omitted where their construction encodes an outcome.

Do not force `uncertain_hold` positive or negative. Initially exclude it from the physical-class loss and use it only to develop/evaluate `requires_manual_review` (after enough examples exist). Preserve uncertain cases for later soft-label/semi-supervised work; do not use pseudo-labels in the first baseline.

## Recommendation

Train CatBoost first, after Phase 2A review and data-gap closure. Its first job is to establish whether measurements add scientific separation beyond the frozen morphology score, with calibrated probabilities and transparent error analysis—not to automate promotion.
