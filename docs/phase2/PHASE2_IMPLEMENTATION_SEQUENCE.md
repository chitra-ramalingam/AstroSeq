# Phase 2 Implementation Sequence

## Phase 2A — audit and dataset contract (current gate)

1. Review this package and approve the label-source precedence and normalization mapping.
2. Ingest/version authoritative positive, EB/variable, and false-positive catalogues; do not overwrite ledgers.
3. Build one EPIC-level label table with conflicts quarantined and uncertain labels preserved.
4. Build a leakage-screened feature table plus explicit missingness/provenance flags.
5. Export frozen 128-D embeddings from the official CNN without weight changes, retaining per-segment embeddings and a documented EPIC aggregation.
6. Audit campaign/class/SNR/period coverage, then materialize the immutable group split and hash its manifests.
7. Obtain Phase 2A approval. No training before this gate.

## Phase 2B — tabular baseline

Train CatBoost with diagnostics + CNN probability first; run scalar/embedding feature-family ablations; tune only on validation; calibrate; evaluate once on blind data; publish SHAP importance, confusion/error cohorts, and hard-negative analysis. Do not promote candidates automatically.

## Phase 2C — hypothesis views and neural model

Generate deterministic masked tensors for P/2, P, 2P global/local/odd/even/secondary/event-stack/baseline views. Validate tensor coverage and invariants. Implement the frozen-CNN, shared-view, scalar multi-task model and compare against Phase 2B under the identical split.

## Phase 2D — active learning and possible fine-tuning

Prioritize model disagreements, high-uncertainty cases, candidate-like predictions, and high-CNN negatives for blinded manual review. Add labels via append-only provenance. Consider CNN fine-tuning only after the learned-vetter benefit, dataset scale, and leakage audit justify it.

## Exact next implementation task (after approval)

`python scripts/build_phase2_feature_table.py --labels docs/phase2/phase2_label_inventory.csv --output data/phase2/phase2_feature_table.parquet --audit-only`

That script does not yet exist; creating it (with tests and no training) is the proposed next task.
