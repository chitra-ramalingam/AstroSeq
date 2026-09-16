# K2 Stage R to G Run Summary

Generated: 2026-04-29 21:18:48 +01:00

Git commit: 1c33af8c706026eb31670f92340a6b7448060c51
Worktree dirty: True

## Policy Versions
- Stage R: Stage R v2 frozen promote tiering from `k2_stage_r_promote_tiers.csv`.
- Stage D: Tier_A deeper eval only, cached events/light curves, `min_cluster_count=2`, no downloads, period consistency and event-family metrics.
- Stage E: OC visual review labels from `k2_stage_e_visual_review_sheet_updated.csv`.
- Stage F: phase-0.5 secondary search, alias checks (`P/2`, `2P/3`, `P`, `3P/2`, `2P`), event-family depth/radius estimate, odd/even explicit check, OOT variability check.
- Stage G: final candidate dossiers and recommendation mapping.

## Counts
- Stage R promote tiers: 67 rows; Tier_A=52, Tier_B=15, Tier_C=0
- Stage D Tier_A input: 52
- Stage D results: pass=36, hold=16, fail=0
- Stage E follow-up targets: 3
- Stage E reject/binary notes: 5
- Stage F labels: planet_like=2, hold=1
- Stage G dossiers: 2

## Final Ranked Candidates
| Rank | EPIC | Stage F label | Final recommendation | Period days | Alias risk |
|---:|---|---|---|---:|---|
| 1 | EPIC_211534076 | stage_f_planet_like | top_followup_candidate | 15.2215800482552 | low |
| 2 | EPIC_211889692 | stage_f_planet_like | followup_candidate_with_variability_caution | 17.6734234999205 | low |
| 3 | EPIC_211555255 | stage_f_hold | secondary_queue_hold | 9.92978153091099 | low |

## Key Outputs
- `k2_stage_r_to_g_run_manifest.json`
- `k2_stage_g_candidate_dossiers.csv`
- `plots/k2_batch/stage_g_dossiers/`
- `k2_stage_f_final_ranked_candidates.csv`
- `k2_stage_f_planet_like_candidates.csv`
- `k2_stage_f_hold_candidates.csv`

