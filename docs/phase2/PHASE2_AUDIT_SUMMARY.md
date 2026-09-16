# Phase 2 Audit Summary

Phase 2 should freeze `models/k2_nocrop_flux_seed46_split303.best.keras` as a 128-D morphology encoder and move scientific classification into a learned downstream model. GateVetter remains a diagnostic/view generator; its decisions are excluded from inputs.

The union inventory contains **706 unique EPICs** with proposed normalized counts: **candidate_like=381**, **false_positive_eb_or_variable=93**, **reject_as_noise_or_artifact=191**, **uncertain_hold=41**. Only **268** meet the current conservative supervised flag; **9** are reliable internally reviewed positives excluding stripped ephemeris-only catalogue rows. There are **22** manually grounded negatives with CNN score ≥0.5. Availability is uneven: diagnostics **210**, explicit P/2–P–2P comparisons **15**, accessible C5 tensors/embedding-exportable **395**, CNN scores **45**.

The biggest gaps are verified positive/EB/variable/public-false-positive catalogues with dispositions and campaigns, population-wide consistent diagnostics, multi-campaign coverage, and P/2–P–2P view tensors. The 15-object deep review is valuable hard-negative/traceability evidence, not sufficient training data.

Recommendation: after Phase 2A review, build the audited EPIC-level feature table and train a class-weighted CatBoost baseline before attempting a multi-input neural model. `uncertain_hold` is excluded from the physical loss and reserved for review-head work. No model was trained, no CNN or labels were modified, and no candidate batch was run.
