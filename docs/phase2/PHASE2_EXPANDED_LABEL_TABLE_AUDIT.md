# Phase 2 Expanded Label Table Audit

## Outcome

- Retained catalogue object/solution rows before aggregation: **4064**.
- Canonical one-row-per-EPIC expanded hosts: **1872**.
- Archive evidence hosts: confirmed **414**, candidate **920**, false-positive/refuted **250**.
- Archive-only cross-class conflicts: **1**; expanded archive/internal cross-class conflicts: **11**.
- Multi-object systems preserved: **168** hosts. Default and non-default solutions remain in the object-evidence parquet and per-host solution-provenance JSON.

| Corrected physical class | Hosts | Physical-loss eligible |
| --- | ---: | ---: |
| `candidate_like` | 898 | 895 |
| `confirmed_planet` | 414 | 414 |
| `cross_class_conflict` | 11 | 0 |
| `false_positive_eb_or_variable` | 328 | 321 |
| `reject_as_noise_or_artifact` | 184 | 167 |
| `uncertain_hold` | 37 | 0 |

## Adjudication and aggregation policy

The accepted append-only correction makes `EPIC_212024647` an eligible `false_positive_eb_or_variable` / `negative_archive_false_positive` while preserving its internal promotion and Bronze evidence. Unmatched internal candidates retain their internal state; catalogue absence is never negative evidence. Confirmed archive evidence refines compatible internal candidate evidence, while unresolved physical-class disagreements are assigned `cross_class_conflict` and excluded from physical-class loss.

All 4,064 source rows are preserved before host aggregation. Object IDs, candidate suffixes, multi-object systems, references, periods, campaigns, and default/non-default solution provenance remain explicit. No train/validation/test split is assigned.

## Leakage protection and non-actions

This label table is target/provenance/training-control metadata, not a scientific feature matrix. Archive disposition/class/reference, evidence tier, training role, manual labels/reasons, correction basis, GateVetter outputs, normalized target, and loss eligibility were not added to the accepted scientific feature matrix. No training, CNN operation, candidate search, embedding generation, diagnostic generation, or download occurred.
