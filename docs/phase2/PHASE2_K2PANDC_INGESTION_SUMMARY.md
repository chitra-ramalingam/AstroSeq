# Phase 2A K2PANDC Ingestion Summary

## Immutable source

- Source: NASA Exoplanet Archive `k2pandc`.
- Snapshot: `data/phase2/catalogues/nea_k2pandc/2026-08-05/raw/k2pandc.csv`.
- Retrieval UTC: `2026-08-05T17:08:18.2375669Z`.
- SHA-256: `609b8c3eedbc02d8e43783be183ecea62e2074042c872ab10cbdd406d7b2676d`.
- Bytes: **488651**.
- Rows: **4064**; unique canonical EPIC hosts: **1559**.
- Schema drift: **none** across the ten requested fields.

## Object-level ingestion audit

| Measure | Count |
| --- | ---: |
| Confirmed hosts (any retained solution) | 414 |
| Candidate hosts (any retained solution) | 920 |
| False-positive/refuted hosts (any retained solution) | 250 |
| Unknown-disposition rows | 0 |
| Missing disposition-reference rows | 0 |
| Exact duplicate rows beyond the first | 1 |
| Cross-class conflict hosts | 1 |
| Object rows provisionally eligible after reference/default/conflict checks | 1802 |

All **4064** source rows are retained in `data/phase2/catalogues/phase2_k2_catalogue_rows.parquet`. Candidate suffixes remain in `object_id` and `candidate_suffix`; multiple planets and parameter solutions remain separate. Unknown, unreferenced, malformed/missing-EPIC, non-default, and cross-class-conflict rows are loss-ineligible proposals. No internal label was changed.

## Nine-EPIC focused mapping audit

| EPIC | Match | Archive object IDs | Dispositions | References | Archive cross-class conflict | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
| `EPIC_211357782` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_211497712` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_211534076` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_211915147` | true | EPIC 211915147.01 | FALSE POSITIVE | Yu et al. 2018 | false | `archive_negative_supports_current_negative_no_automatic_change` |
| `EPIC_211953866` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_211889692` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_211682657` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_212001099` | false | (none) | (none) | (none) | false | `no_archive_match_retain_current_status` |
| `EPIC_212024647` | true | EPIC 212024647.01 | FALSE POSITIVE | Yu et al. 2018 | false | `manual_adjudication_required_archive_negative_conflicts_with_internal_candidate` |

## Blockers before an expanded EPIC-level label table

1. Only two of the nine audited EPICs have exact `k2pandc` host matches; the seven unmatched objects retain their current internal status.
2. `EPIC_212024647` has authoritative archive false-positive evidence that conflicts with its retained internal Bronze candidate provenance and requires append-only adjudication.
3. Archive-wide cross-class hosts must remain quarantined when object evidence is aggregated to EPIC level.
4. This task ingested only `k2pandc`; the separately planned EB/variable catalogues and full campaign manifest remain absent and were not downloaded.
5. The expanded builder must join internal corrections without converting catalogue provenance fields into scientific model features.

No CatBoost training, split freeze, CNN load/change/retraining, candidate search, GateVetter classification, expanded-label build, or automatic label modification occurred.
