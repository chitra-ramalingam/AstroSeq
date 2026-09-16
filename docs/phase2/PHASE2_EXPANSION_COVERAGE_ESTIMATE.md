# Phase 2A Expansion Coverage Estimate

## Basis and interpretation

This is a planning estimate, not an executed catalogue join. No external catalogue data were downloaded. Counts labelled **measured** come from accepted repository artifacts; counts labelled **estimated** use published source sizes and the repository-local legacy extracts. Estimates must be replaced by exact host-level counts after versioned ingestion and conflict quarantine.

No astrophysical rule decisions were calculated.

## Measured repository baseline

| Capability | Measured EPIC count | Interpretation |
| --- | ---: | --- |
| Current union label inventory | 706 | all proposed labels, including validation/traceability-only rows |
| Accessible local K2 light-curve/tensor representation | 395 | Campaign 5 evidence only |
| Frozen CNN probability | 268 | accepted supervised freeze; model unchanged |
| Frozen 128-D embedding | 268 | accepted supervised freeze; model unchanged |
| Aligned nominal diagnostics retained in the feature artifact | 201 | 8 candidate-labelled rows plus 193 corrected negative rows; 197 are currently loss-eligible |
| Explicit P/2-P-2P diagnostics | 15 total | includes all 9 current positives; the remainder are legacy deep-review rows |
| Campaign metadata | 395 | all are Campaign 5; 311 union rows remain campaign-unknown |

The accepted 268-row CNN freeze is retained unchanged; four candidate-labelled rows are now loss-ineligible. It supersedes the older `has_current_cnn_score` flags in `phase2_label_inventory.csv`, which predate the completed export.

The local stripped `K2_ephemerides.csv` has 512 object rows and 372 unique EPIC hosts. Of those hosts, 61 intersect the current Campaign-5 accessible representation. The file has neither disposition nor campaign and therefore does not by itself make any row physical-loss eligible.

## Expected expanded label population

The following source sizes anchor the estimate:

- Existing NASA confirmed-K2 reference: 549 planet rows, 547 EPIC-mapped rows, **399 unique EPIC hosts**.
- Local unversioned K2 archive-style subset: about **919 candidate EPIC hosts** and **242 false-positive EPIC hosts**; these are estimates only until `k2pandc` is ingested.
- Kruse et al. 2019: **818 candidates** and **579 EBs** in Campaigns 0-8.
- K2VARCAT II: about **10,536 class-significant variable/EB examples** in Campaigns 0-4.
- LaCourse et al. 2015: **207 Campaign-0 EBs**, substantially overlapping later variable/EB sources.

After expected overlap across candidate, EB, variable, and false-positive catalogues, the planning union is **11,000-12,500 unique catalogue-labelled EPIC hosts**. The estimate is intentionally a range because the external files are not local and object-to-host/cross-source deduplication has not been run.

## Post-ingestion coverage estimate

| Capability | Estimated unique EPICs | Confidence and limiting factor |
| --- | ---: | --- |
| Accessible K2 light curves | 10,500-12,200 | medium; all proposed sources are K2-derived, but product availability, target/victim roles, corrupt files, and identifier mismatches will reduce coverage |
| Frozen CNN probability | 10,500-12,200 can receive one | medium; requires a later run of the accepted frozen exporter on accessible curves; currently only 268 are frozen |
| Frozen 128-D embedding | 10,500-12,200 can receive one | medium; same availability gate and frozen encoder as CNN probability |
| Nominal numerical diagnostics | 10,000-11,800 | low-medium; requires an accessible curve plus a catalogue period/frequency with retained provenance |
| P/2-P-2P numerical diagnostics | 10,000-11,800 | low-medium; generated from the same accepted period rows, with failed or untrusted periods explicit |
| Campaign metadata | 11,000-12,500 | high after full MAST join; campaign metadata is expected for nearly all K2 EPIC targets |

These are counts that *can receive* features after the prepared pipeline is implemented and run. They are not claims that those features already exist. A light-curve accessibility audit must report exact success/failure counts before inference or diagnostics.

## Expected class-scale result after ingestion

- **Verified positive EPICs:** planning range **1,100-1,300 unique hosts**, anchored by 399 archive-confirmed hosts and about 919 candidate EPIC hosts before cross-disposition deduplication. Candidate rows qualify only when the current disposition and reference are present; conflicts are excluded.
- **EB/variable examples:** planning range **10,500-11,100 unique hosts** after overlap between K2VARCAT, Kruse, and LaCourse.
- **Published false positives:** approximately **242 unique EPIC hosts**, based on the local archive-style K2 extract; exact current `k2pandc` count is required.

These categories can overlap before conflict adjudication. They must not be summed into a training total until the host-level builder has run.

## Campaign coverage

- Current accessible/frozen feature infrastructure: **Campaign 5 only**.
- Proposed confirmed/candidate/false-positive archive source: K2 campaigns represented in `k2pandc`.
- Kruse candidate/EB source: **C0-C8**.
- K2VARCAT II: **C0-C4**.
- LaCourse EB source: **C0**.
- Planned MAST metadata/light-curve availability join: **C0-C19**.

This broadens label provenance across the mission, but it does not justify freezing a campaign split. Exact per-class counts by campaign and repeated-campaign hosts must be audited first.

## Required Phase 2A status report

- Confirmed Gold positives: **0**.
- Silver candidate positives: **4** (unconfirmed candidate evidence).
- Bronze excluded rows: **3**.
- External-confirmation-pending rows: **1**; physical-loss-eligible candidate positives: **4**.
- Physical-loss-eligible candidate positives with trusted periods: **1**.
- Physical-loss-eligible candidate positives with fallback periods: **3**; corrected EB/variable hard negative with fallback period: **1**.
- Catalogue inventory rows with some source material already local: **4**.
- Complete/versioned source documents still missing: **5**.
- Estimated verified positives after ingestion: **1,100-1,300 unique EPIC hosts**.
- Estimated EB/variable examples: **10,500-11,100 unique EPIC hosts**.
- Estimated published false positives: **about 242 unique EPIC hosts**.
- Campaign coverage: current features **C5**; planned label/metadata coverage **C0-C19**, subject to exact ingestion audit.

## Exact blocker before the first CatBoost smoke test

The blocker is the absence of a versioned, hashed authoritative catalogue snapshot set—especially `k2pandc`—and therefore the absence of a conflict-resolved expanded EPIC label table joined to the accepted frozen CNN outputs and leakage-screened numerical features. Until that gate is completed and approved, the provenance-complete multi-campaign class population does not exist. The final train/validation/test split is intentionally not frozen in this task.

No training, candidate search, CNN modification/retraining, rule-based GateVetter classifier, or final split was performed.
