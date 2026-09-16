# Phase 2 Conflict Review Packet Audit

## Outcome

- Unresolved conflict hosts retained: **11**.
- Underlying catalogue object/solution rows retained: **26**.
- Physical-loss-eligible conflict hosts: **0**.
- Reviewer decisions populated: **0**.

| Conflict category | Hosts |
| --- | ---: |
| `archive_object_mixed_candidate_false_positive` | 1 |
| `internal_eb_variable_vs_archive_candidate` | 3 |
| `internal_noise_artifact_vs_archive_candidate` | 1 |
| `internal_noise_artifact_vs_archive_false_positive` | 6 |

## Existing-only feature context

| Availability | Hosts |
| --- | ---: |
| Cached local light curve | 10 |
| 512-sample tensor | 10 |
| Existing CNN probability | 10 |
| Existing 128-D embedding | 10 |
| Nominal numerical diagnostics | 8 |
| P/2-P-2P diagnostics | 0 |
| Trusted period | 0 |

The packet includes existing numerical measurements only. Archive dispositions, manual labels/reasons, conflict categories, review fields, and provenance are target/review metadata and are not scientific features. GateVetter decisions, actions, recommendations, and verdict-derived fields are not included as feature inputs.

## Read-only guarantee

This build did not alter `phase2_expanded_label_table.parquet`, the internal label inventory, positive tiers, corrections, baseline supersessions, feature table, or any loss-eligibility field. Every conflict remains `cross_class_conflict`, unresolved, and excluded from physical-class loss. No training, split assignment, CNN operation, feature generation, or catalogue download occurred.

The feature-generation manifest is intentionally blocked until a completed adjudication file supplies one permitted decision, reason, reviewer, and UTC timestamp for each of the 11 EPICs.
