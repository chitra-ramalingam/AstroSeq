# Stage I Candidate Report: EPIC_211534076

Generated: 2026-04-29 21:33:54 +0100

## Current Status
- Stage G recommendation: `top_followup_candidate`
- Stage F label: `stage_f_planet_like`
- Stage H status: `new_candidate_needs_external_check`
- Local contamination risk: `low`

## Candidate Metrics
- best_period_days: `15.221580048255191`
- primary_depth: `0.0010519935860486`
- radius_ratio_sqrt_depth: `0.0324344506050064`
- transit_duration_hours: `10.787924320902674`
- secondary_to_primary_depth_ratio: `0.0`
- odd_even_depth_delta_explicit: `0.1429201407325919`
- oot_variability_to_depth: `0.32249088483269617`
- alias_risk: `low`

## Host Metadata Currently Available
- EPIC: `211534076`
- K2 Campaign 5 target-list coordinates: RA `135.598416`, Dec `13.618975`
- Kp magnitude: `9.94`
- Investigation IDs: `GO5020_LC|GO5104_LC` locally; external C5 page lists `GO5020_LC|GO4104_LC`
- Local Gaia/TIC/SIMBAD identifiers: not found
- Stellar radius: not found locally

External source checked: Vanderburg K2 Campaign 5 target list, `https://lweb.cfa.harvard.edu/~avanderb/allk2c5obs.html`.

## Known-Object Checks So Far
- Local NASA confirmed K2 planet reference: no match
- Local K2 ephemerides / candidate ephemerides: no match
- Local confirmed planet audit tables: no match
- Local binary-star classification file: no match
- Stage B manifest: unresolved and still needing triage/classification
- Lightweight external web search: no obvious known planet/EB hit found in this pass

## Planet Radius Work Item
Current depth gives `Rp/Rstar = sqrt(depth) = 0.03243445`.

Planet radius cannot be estimated yet because no stellar radius was found locally. Once `Rstar` is available:

`Rp_earth = 0.03243445 * Rstar_solar * 109.076`

## Required Final Vetting Checklist
- [ ] Resolve host IDs in MAST/K2 EPIC, Gaia DR3, TIC, SIMBAD, VizieR, ExoFOP if available.
- [ ] Retrieve stellar radius, Teff, logg, mass, RUWE/astrometric quality, and photometric colors.
- [ ] Compute planet radius and uncertainty range.
- [ ] Confirm no known planet/candidate/EB/variable disposition in external archives.
- [ ] Confirm no nearby contaminating source in imaging or aperture products.
- [ ] Preserve Stage D/F plots and JSON summaries in the final report package.
- [ ] Decide final external-vetted status.

## Artifact Paths
- Stage G dossier CSV: `k2_stage_g_candidate_dossiers.csv`
- Stage F validation CSV: `k2_stage_f_followup_validation.csv`
- Stage H external vetting CSV: `k2_stage_h_external_vetting.csv`
- Stage F phase 0 plot: `plots/k2_batch/stage_f_followup/EPIC_211534076/phase_0_folded.png`
- Stage F secondary check: `plots/k2_batch/stage_f_followup/EPIC_211534076/phase_05_secondary_check.png`
- Stage F alias comparison: `plots/k2_batch/stage_f_followup/EPIC_211534076/alias_period_comparison.png`
- Stage F odd/even zoom: `plots/k2_batch/stage_f_followup/EPIC_211534076/odd_even_zoom.png`
- Stage F validation JSON: `plots/k2_batch/stage_f_followup/EPIC_211534076/validation_summary.json`
