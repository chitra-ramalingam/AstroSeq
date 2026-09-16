# Phase 2A Positive Tier Audit - Corrected

## Corrected outcome

| Evidence state | Count | Hard physical loss | Meaning |
| --- | ---: | --- | --- |
| `positive_gold` | 0 | none | no authoritative confirmed EPIC mapping has yet been reproduced |
| `positive_silver` | 4 | eligible as provisional candidate positives | candidate evidence, not confirmed planets |
| `positive_bronze` | 2 | excluded | lineage/sensitivity only |
| `external_confirmation_pending` | 1 | quarantined | exact EPIC mapping and confirmed disposition reference required |
| corrected EB/variable negatives | 2 | eligible as negatives | published EB and archive false-positive hard negatives |

The four Silver candidate positives are `EPIC_211357782`, `EPIC_211497712`, `EPIC_211534076`, `EPIC_211953866`. Their evidence distinctions remain separate: direct manual candidate-like review and recovered-known-unconfirmed-candidate provenance are not collapsed into confirmation.

The Bronze rows `EPIC_211682657`, `EPIC_212001099` retain their normalized labels and diagnostics but remain excluded from the initial hard physical-class loss.

## Corrections and precedence

`EPIC_211915147` is now `false_positive_eb_or_variable`, with external disposition `published_eclipsing_binary`, training role `negative_eb_variable`, and fallback/untrusted event-spacing period provenance. The 2026-05-25 manual `candidate_like` judgment remains historical evidence. Later external physical-class evidence supersedes that judgment; the manual and CNN morphology support make the row a high-value hard negative.

`EPIC_211889692` remains normalized `candidate_like` for repository lineage but is now `external_confirmation_pending`, `unverified_mapping`, loss-ineligible, and quarantined. The recovered-known-confirmed-planet assertion remains historical evidence. Gold may be restored only through a new append-only correction after exact authoritative EPIC mapping and a confirmed disposition reference are ingested.

`EPIC_212024647` is now `false_positive_eb_or_variable`, with `archive_false_positive` disposition and `negative_archive_false_positive` training role. The authoritative `k2pandc` object `EPIC 212024647.01` (Yu et al. 2018; 3.696871 days; campaigns 5, 16, 18) supersedes the internal candidate target. The manual promotion and Bronze tier remain append-only historical evidence.

## Leakage and non-actions

Evidence tier, external disposition/status, training role, correction basis, manual labels/reasons, GateVetter outputs, final recommendations, catalogue-derived target fields, and target-derived verdicts are metadata only and are excluded from the scientific model feature matrix. CNN probabilities, embeddings, and numerical light-curve diagnostics were not changed.

No model was trained, no CNN was loaded or modified, no split was frozen, and no candidate-search batch was run.
