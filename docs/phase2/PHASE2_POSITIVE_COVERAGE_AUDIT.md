# Phase 2 Positive Coverage Audit — Corrected Baseline

## Supersession

This generated summary supersedes the earlier 1-Gold/5-Silver/6-eligible-positive assumptions. Historical manual and repository evidence remains in `phase2_label_inventory.csv` and the append-only `phase2_positive_tier_corrections.csv`.

## Current outcome

- Confirmed Gold positives: **0**.
- Silver candidate positives: **4**, all physical-loss eligible and explicitly unconfirmed.
- Bronze candidates: **3**, retained for lineage and excluded from physical loss.
- External-confirmation-pending: **1**, quarantined.
- Eligible EB/variable negatives: **86**.
- Eligible noise/artifact negatives: **174**.
- Uncertain rows: **41**, all excluded from physical loss.
- High-CNN eligible hard negatives: **257**.

The retained 268-row feature artifact contains eight candidate-labelled rows (four eligible Silver, three excluded Bronze, one confirmation-pending), 86 EB/variable rows, and 174 noise/artifact rows. Nominal-diagnostic coverage is 8/8, 73/86, and 120/174 respectively. Frozen CNN embedding coverage is complete for all 268 retained rows.

## Corrected evidence distinctions

`EPIC_211915147` is an eligible `false_positive_eb_or_variable` hard negative supported by authoritative archive false-positive evidence; its earlier manual candidate-like judgment remains historical evidence. `EPIC_211889692` has no exact `k2pandc` mapping and remains external-confirmation-pending. The four Silver objects are candidate evidence, not confirmed planets.

## Leakage and non-actions

Catalogue dispositions, evidence tiers, training roles, correction bases, manual labels/reasons, GateVetter outputs, final recommendations, and target-derived verdicts are target/provenance metadata and are excluded from the scientific feature matrix. CNN probabilities, embeddings, numerical diagnostics, and missingness indicators remain unchanged.

No model was trained, no split was frozen, no CNN artifact was loaded or modified, and no candidate search was run.
