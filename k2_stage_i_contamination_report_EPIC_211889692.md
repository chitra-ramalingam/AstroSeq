# Stage I Contamination Report: EPIC_211889692

Generated: 2026-04-29 21:33:54 +0100

## Current Status
- Stage G recommendation: `followup_candidate_with_variability_caution`
- Stage F label: `stage_f_planet_like`
- Stage H status: `new_candidate_needs_external_check`
- Contamination risk: `high`
- Promotion rule: do not advance as a clean planet candidate until neighbor EPIC_211890226 is cleared.

## Target Metrics
- best_period_days: `17.673423499920546`
- primary_depth: `0.0009543328081398`
- radius_ratio_sqrt_depth: `0.03089227748385994`
- transit_duration_hours: `9.807273853279183`
- secondary_to_primary_depth_ratio: `0.0`
- odd_even_depth_delta_explicit: `0.5521356662214799`
- oot_variability_to_depth: `2.2516310215513196`
- alias_risk: `low`

## Neighbor of Concern
- Neighbor EPIC: `EPIC_211890226`
- Separation: `27.7 arcsec`
- Target Kp: `11.469`
- Neighbor Kp: `11.708`
- Delta mag: `0.239`
- Target coordinates: RA `124.317115`, Dec `18.57549`
- Neighbor coordinates: RA `124.317935`, Dec `18.583145`
- Both target and neighbor have cached EVEREST light curves locally.
- Neighbor has local detector events at `plots/k2_batch/epics/EPIC_211890226/events.csv`.

External source checked: Vanderburg K2 Campaign 5 target list, `https://lweb.cfa.harvard.edu/~avanderb/allk2c5obs.html`.

## Preliminary Same-Period Local Check
Using target period `17.673423499920546` d and target cluster phase:

- Target event support at target phase: `5`; median target-phase event depth `0.0009543328081398`
- Neighbor total detector events: `3`
- Neighbor best support at the target period: `2`, but at a different phase (`~0.8788`)
- Neighbor support at target phase: `0`

Interpretation: local event timing does not immediately show the same target-phase transit in EPIC_211890226, but the neighbor is bright and close enough that aperture/pixel-level contamination remains unresolved.

## Required Contamination Tasks
- [ ] Generate target-vs-neighbor folded light curve comparison at target period and target phase.
- [ ] Inspect EPIC_211890226 phase fold at its own apparent event phase.
- [ ] Retrieve/inspect target pixel files or aperture masks for both EPICs if available.
- [ ] Check whether EPIC_211890226 lies inside or near the EVEREST/K2 extraction aperture for EPIC_211889692.
- [ ] Estimate flux dilution / centroid shift risk from separation and delta magnitude.
- [ ] Query external Gaia/MAST/ExoFOP/SIMBAD for close companions and proper source IDs.
- [ ] Only promote if neighbor/aperture/centroid checks rule out contamination.

## Artifact Paths
- Stage F phase 0 plot: `plots/k2_batch/stage_f_followup/EPIC_211889692/phase_0_folded.png`
- Stage F secondary check: `plots/k2_batch/stage_f_followup/EPIC_211889692/phase_05_secondary_check.png`
- Stage F alias comparison: `plots/k2_batch/stage_f_followup/EPIC_211889692/alias_period_comparison.png`
- Stage F odd/even zoom: `plots/k2_batch/stage_f_followup/EPIC_211889692/odd_even_zoom.png`
- Stage F validation JSON: `plots/k2_batch/stage_f_followup/EPIC_211889692/validation_summary.json`
- Neighbor events CSV: `plots/k2_batch/epics/EPIC_211890226/events.csv`
