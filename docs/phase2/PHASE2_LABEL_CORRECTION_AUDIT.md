# Phase 2A Label Correction Audit

## Outcome

- Confirmed Gold positives: **0**.
- Silver candidate positives: **4**.
- Bronze excluded: **2**.
- External-confirmation-pending: **1**.
- Physical-loss-eligible candidate positives: **4**.
- Physical-loss-eligible EB/variable rows: **87**.
- Physical-loss-eligible noise/artifact rows: **174**.
- Uncertain rows: **41 total; 0 loss-eligible**.
- High-CNN hard negatives: **258** eligible corrected negative rows with accepted frozen CNN probability >= 0.5.
- Explicit append-only high-value hard-negative corrections: **2** (`EPIC_211915147`, `EPIC_212024647`).

## Corrected rows

| EPIC | Before | After | Eligibility | Training role |
| --- | --- | --- | --- | --- |
| `EPIC_211915147` | `candidate_like`; `positive_silver`; candidate positive | `false_positive_eb_or_variable`; no positive tier; `published_eclipsing_binary` | true -> true (now as a negative) | `negative_eb_variable` |
| `EPIC_211889692` | `candidate_like`; `positive_gold`; repository asserted confirmed mapping | `candidate_like`; `external_confirmation_pending`; `unverified_mapping` | true -> false | `quarantine_until_catalogue_verification` |
| `EPIC_212024647` | `candidate_like`; `positive_bronze`; internal promote/hold provenance | `false_positive_eb_or_variable`; no positive tier; `archive_false_positive` | false -> true (now as a negative) | `negative_archive_false_positive` |

The append-only CSV contains the full before/after fields and evidence basis. Historical manual/repository evidence was not deleted or overwritten.

## Corrected coverage

| Corrected class | Retained rows | Eligible | Nominal diagnostics | Eligible diagnostics | CNN embeddings | Eligible embeddings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `candidate_like` | 7 | 4 | 7 | 4 | 7 | 4 |
| `false_positive_eb_or_variable` | 87 | 87 | 74 | 74 | 87 | 87 |
| `reject_as_noise_or_artifact` | 174 | 174 | 120 | 120 | 174 | 174 |

The retained feature table has 268 rows. The three loss-ineligible candidate rows remain present for lineage. Scientific feature hash before and after the metadata-only patch: `e184d2b8b5b7e6ecf03fe18919af6544c2d55a5611916da9f76881e95c171bd0` (identical).

## Leakage audit and non-actions

The scientific matrix excludes positive evidence tier, external disposition/status, training role, correction basis, manual labels/reasons, GateVetter predictions/reasons/actions, final recommendations, catalogue-derived normalized target fields, and rule verdicts derived from target class. Only label/control metadata changed. CNN probabilities, all 128 embedding dimensions, numerical diagnostics, and missingness indicators are value-identical.

No model was trained; the CNN was not loaded, modified, or retrained; no split was frozen; no candidate search was run; and no catalogue data were downloaded. The already accepted local `k2pandc` ingestion was used only as correction evidence.

## Accepted archive evidence

The versioned local NASA Exoplanet Archive `k2pandc` snapshot and its accepted 4,064-row object-level ingestion were used offline. `EPIC 212024647.01` is retained as `FALSE POSITIVE`, reference `Yu et al. 2018`, period 3.696871 days, campaigns 5, 16, and 18. No additional catalogue was downloaded or ingested in this task.
