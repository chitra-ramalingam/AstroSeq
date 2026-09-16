# Phase 2 Data Gap Analysis

## Observed availability

- Unique labelled EPICs in the union inventory: **706**.
- Proposed normalized distribution: `candidate_like`=381, `false_positive_eb_or_variable`=93, `reject_as_noise_or_artifact`=191, `uncertain_hold`=41.
- Rows currently safe under the conservative supervised flag: **268**.
- With any GateVetter batch diagnostics: **210**; with the explicit P/2–P–2P comparison: **15**.
- With Campaign-5 inference tensors / accessible cached light-curve representation: **395**; current CNN scores: **45**; therefore embedding-exportable without new downloads: **395**.
- High-CNN-score (≥0.5), manually grounded negative EPICs: **22**.
- Reliable internal positive examples excluding unverified stripped ephemeris-only rows: **9**.

The 15-object deep-review shortlist is hard-negative evidence and traceability/validation material, not a training population.

## Sufficiency

**Tabular baseline:** not yet training-ready as a four-class model. There are many labels, but only a subset has consistently aligned diagnostics, and positives are sparse once catalogue-only entries requiring disposition verification are excluded. A binary/three-way feasibility experiment could become viable after deterministic feature joining and catalogue verification.

**Multi-input neural vetter:** insufficient. Only 15 EPICs have explicit three-hypothesis diagnostics, and equivalent view tensors are not stored at population scale. The label volume and class balance cannot support a new multi-branch neural model without public catalogue ingestion and view generation.

## Campaign and coverage gaps

The active processed universe and inference tensors are Campaign 5. `K2_ephemerides.csv` spans K2 but lacks campaign and disposition columns. There is no repository-wide, provenance-rich campaign mapping for every labelled EPIC, so campaign distribution is `C5 where evidenced; unknown otherwise`; metadata was not invented. A leakage-safe held-out campaign cannot be instantiated until other campaigns are ingested and processed consistently.

## Local catalogues and missing datasets

Available: `K2_ephemerides.csv` (NASA Exoplanet Archive-derived transit ephemerides; 512 rows, stripped disposition), `plots/k2_batch/confirmed_planet_audit/nasa_confirmed_k2_planets_reference.csv` if regenerated/located by the existing audit class, `k2_recovered_known_planets.csv`, `k2_recovered_positive_controls.csv`, `k2_real_nonplanet_systems.csv`, the manual ledger, and final ledger. The current checkout does not contain a population-scale, authoritative K2 EB catalogue, variable catalogue, or published false-positive table joined to EPIC.

Before training, ingest snapshots (do not download in this audit) of: NASA Exoplanet Archive K2 confirmed planets with disposition/provenance; ExoFOP-K2 or equivalent candidate/false-positive dispositions; the Villanova/K2 eclipsing-binary catalogue; and a published K2 variability catalogue with stable EPIC identifiers. Preserve catalogue version/date and object-level disposition.

Match by canonical digits (`EPIC 211...`, `EPIC_211...`, and integer → `EPIC_#########`). Collapse multiple planets to one host only after retaining planet rows. Precedence: confirmed planet evidence overrides candidate status but never silently overrides an EB/variable conflict; contradictory host labels go to adjudication and are excluded from training. Catalogue-only positives stay validation/traceability-only until disposition and campaign provenance are verified.

## Largest blockers

1. No normalized, provenance-rich public four-class label population.
2. Severe reliable-positive shortage in the internally reviewed set.
3. Sparse population-wide diagnostics and only 15 explicit P/2–P–2P cases.
4. No generated P/2–P–2P view tensor dataset.
5. Single-campaign processing prevents a genuine campaign holdout.
6. Candidate-period rows and host labels need a deterministic EPIC-level join/conflict policy.
