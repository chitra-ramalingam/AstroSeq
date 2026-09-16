# Phase 2 Expanded Feature Coverage Audit

## Existing-only coverage

| Corrected class | Hosts | Eligible | Local light curves | 512 tensors | CNN probabilities | 128-D embeddings | Nominal diagnostics | P/2-P-2P diagnostics | Trusted periods | Fallback/missing periods | Conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `candidate_like` | 898 | 895 | 199 | 116 | 7 | 7 | 7 | 7 | 4 | 894 | 0 |
| `confirmed_planet` | 414 | 414 | 88 | 61 | 0 | 0 | 0 | 0 | 0 | 414 | 0 |
| `cross_class_conflict` | 11 | 0 | 10 | 10 | 10 | 10 | 8 | 0 | 0 | 11 | 11 |
| `false_positive_eb_or_variable` | 328 | 321 | 211 | 135 | 84 | 84 | 72 | 2 | 1 | 327 | 0 |
| `reject_as_noise_or_artifact` | 184 | 167 | 184 | 184 | 167 | 167 | 114 | 0 | 0 | 184 | 0 |
| `uncertain_hold` | 37 | 0 | 37 | 37 | 0 | 0 | 0 | 0 | 0 | 37 | 0 |

The companion CSV reports the same fields by corrected class and campaign. A host appearing in multiple archive campaigns is counted once in each applicable campaign, so campaign totals are intentionally non-additive. `missing_feature_patterns_json` reports exact combinations of missing existing artifacts per group.

This is coverage auditing only. No light curve was downloaded; no tensor, CNN probability, embedding, numerical diagnostic, or period diagnostic was generated.

The exact blocker before feature generation is: **1143** hosts lack an accessible cached light curve, **1329** lack a 512-sample tensor, **1671** lack nominal diagnostics, **1863** lack P/2-P-2P diagnostics, and **11** unresolved cross-class conflicts are quarantined. A provenance-controlled multi-campaign light-curve/tensor generation plan and conflict adjudication are required before population feature generation or physical-class loss.
