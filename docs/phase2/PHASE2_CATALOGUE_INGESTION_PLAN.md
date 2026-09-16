# Phase 2A Versioned K2 Catalogue-Ingestion Plan

## Scope and non-actions

This plan prepares authoritative K2 label expansion. It does not download a catalogue, execute ingestion, generate diagnostics, calculate astrophysical rule decisions, train a model, change the frozen CNN, freeze a final split, or run candidate search.

The source-by-source contract is in `docs/phase2/phase2_catalogue_source_inventory.csv`. The versioned `k2pandc` snapshot dated 2026-08-05 is now present and ingested at object/solution level. Four additional complete/versioned documents are still required for the broader expansion: Kruse 2019, K2VARCAT II, LaCourse 2015, and the full C0-C19 MAST campaign manifest. Legacy local extracts remain audit-only.

## Versioned snapshot layout

Every external retrieval is immutable and date/version scoped:

```text
data/phase2/catalogues/
  <source_slug>/<version-or-YYYY-MM-DD>/
    raw/<upstream filenames>
    source_manifest.json
    normalized_rows.parquet
    ingestion_audit.json
```

`source_manifest.json` records source name, authoritative URL, exact query or file URL, retrieval UTC timestamp, upstream version where available, HTTP metadata where available, SHA-256 and byte size for every raw file, required citation/DOI, schema version, and the ingesting git commit. Raw files are never overwritten. A new retrieval creates a new directory.

Network access is disabled by default in the ingestion script. Later retrieval requires an explicit `--allow-download`; local-snapshot ingestion uses `--offline` and fails if a declared raw file is absent. This task creates neither snapshots nor manifests.

## Canonical row contract

Ingestion first emits one row per catalogue object/solution, not one row per host. Required normalized columns are:

```text
catalogue_row_id
source_slug
source_version
source_retrieved_at_utc
source_sha256
source_record_locator
epic_id
object_id
host_id_raw
campaign
raw_disposition
raw_disposition_reference
raw_class
normalized_label_proposal
evidence_kind
period_days
period_source
period_trusted
source_citation
```

EPIC normalization accepts `EPIC 211...`, `EPIC_211...`, integer-like values, and both eight- and nine-digit EPIC identifiers. It emits `EPIC_<digits>` without changing the digits. A decimal candidate suffix such as `.01` belongs in `object_id`, not in host `epic_id`.

Planet systems remain object-level through normalization so multiple candidates/planets at one host are not lost. Host-level aggregation occurs only in the expanded-label builder, after all evidence rows and conflicts are retained.

## Class mapping and evidence strength

- Archive/published `CONFIRMED` maps to proposed `candidate_like` with confirmed evidence.
- Archive/published `CANDIDATE` maps to proposed `candidate_like` with vetted-candidate evidence only when a disposition reference is present.
- Published `FALSE POSITIVE [CANDIDATE]`, `REFUTED [PLANET]`, EB, EA, and stellar-variable classes map to proposed `false_positive_eb_or_variable`; raw subtype is mandatory.
- Noise, low-confidence automated classes, unclassified anomalies, and records without sufficient disposition provenance map to `uncertain_hold`, not to a hard class.
- Campaign/target metadata never maps to a class.
- `normalized_label` remains a physical class field. Evidence tier, catalogue disposition, provenance, eligibility, and period trust are separate fields.

## Conflict and precedence policy

No source silently overwrites another source.

1. Exact duplicate evidence rows are deduplicated by source, version, object ID, disposition, and reference; duplicate count remains in the audit.
2. A current confirmed disposition may strengthen a candidate-only record for the same object.
3. Confirmed/candidate evidence does not override EB, variable, refuted, or published false-positive evidence at the same EPIC. The host is marked `cross_class_conflict=true`, excluded from physical loss, and written to an adjudication table.
4. Source/victim relationships in EB catalogues are retained. A contaminating victim is not relabelled as the physical EB source.
5. Multiple campaigns are repeated metadata values, not conflicts.
6. Period disagreements are retained at object level. A trusted period is selected only by declared provenance precedence; other periods remain evidence and diagnostics are never used to select a class.
7. Internal labels and manual reviews are appended as a separate source family. GateVetter decisions/reasons and manual reasons are not features.

## Proposed scripts

### `scripts/ingest_phase2_k2_catalogues.py`

Purpose: validate declared local raw snapshots, hash them, normalize source-specific columns into the object-level contract, and emit source audits. It must not perform network access unless an explicit future `--allow-download` is supplied.

Proposed interface:

```text
python scripts/ingest_phase2_k2_catalogues.py \
  --source-inventory docs/phase2/phase2_catalogue_source_inventory.csv \
  --snapshot-root data/phase2/catalogues \
  --output data/phase2/catalogues/phase2_k2_catalogue_rows.parquet \
  --audit-output docs/phase2/phase2_catalogue_ingestion_audit.json \
  --offline
```

Required behaviours: strict schema validation; eight/nine-digit EPIC canonicalization; row-level preservation; raw disposition/reference preservation; campaign parsing including repeated campaigns; SHA-256 verification; deterministic ordering; structured schema-drift errors; no inference from filenames; and no physical rule decisions.

### `scripts/build_phase2_expanded_label_table.py`

Purpose: join normalized external evidence, the accepted internal inventory, and the positive-tier table into an EPIC-level expanded label table plus object-evidence and conflict tables.

Proposed outputs:

- `data/phase2/phase2_expanded_catalogue_evidence.parquet` — all object/source rows.
- `data/phase2/phase2_expanded_label_table.parquet` — one row per EPIC.
- `docs/phase2/phase2_expanded_label_conflicts.csv` — adjudication queue.
- `docs/phase2/phase2_expanded_label_summary.json` — class/source/campaign counts and hashes.

Required behaviours: preserve existing `normalized_label`; add catalogue proposals separately; gold/silver eligibility and bronze exclusion; null sample weights; current-disposition/reference checks; physical-loss exclusion for ambiguity/conflict; host aggregation only after object-row retention; deterministic provenance JSON; forbidden-feature audit; and no split assignment.

### `scripts/generate_phase2_period_diagnostics.py`

Purpose: generate only numerical nominal and P/2-P-2P measurements for rows with an accessible light curve and an accepted period provenance.

Proposed outputs:

- `data/phase2/phase2_expanded_period_diagnostics.parquet` — one row per EPIC/object/period role.
- `docs/phase2/phase2_expanded_period_diagnostic_coverage.csv` — counts by source/class/campaign/trust.
- `docs/phase2/phase2_expanded_period_diagnostic_failures.csv` — load/measurement failures without class decisions.

It should call the existing numerical `evaluate_period` implementation through a factored, label-blind helper. Labels, evidence tiers, GateVetter decisions, and manual reasons are not passed to measurement code. Output contains `period_role`, `period_days`, `period_source`, `period_feature_trust`, numerical diagnostics, code hash, light-curve provenance, and failure metadata. It emits no `odd_even_assessment`, `secondary_assessment`, `alias_risk`, recommendation, or class decision.

The frozen CNN probability and 128-D embedding are supplied by the already accepted export path. The proposed diagnostic script neither loads nor modifies CNN weights.

## Proposed tests

### `tests/test_ingest_phase2_k2_catalogues.py`

- Canonicalize eight- and nine-digit EPIC forms and keep candidate suffixes object-level.
- Reject malformed or missing identifier/disposition/reference schemas.
- Preserve multi-planet and multi-solution rows.
- Map every declared disposition/class exactly and send unknown values to `uncertain_hold`.
- Parse single/multiple/engineering campaigns without treating campaign repetition as conflict.
- Detect source-file hash mismatch and schema drift.
- Verify deterministic output and zero network calls in `--offline` mode.

### `tests/test_build_phase2_expanded_label_table.py`

- Assert the corrected current tier counts are Gold 0, Silver 4, Bronze excluded 3, external-confirmation-pending 1, and physical-loss-eligible candidate positives 4.
- Assert all nine current `normalized_label` values are unchanged and all weights are null.
- Assert every bronze row is excluded from hard physical loss.
- Retain object rows before correct EPIC-level aggregation.
- Strengthen candidate evidence with confirmed evidence without deleting provenance.
- Quarantine planet-versus-EB/variable/false-positive conflicts.
- Keep uncertain/missing-disposition rows out of hard physical loss.
- Reject leakage fields including GateVetter actions/reasons/recommendations and manual reasons.
- Produce stable hashes and row ordering without assigning train/validation/test.

### `tests/test_generate_phase2_period_diagnostics.py`

- Generate exactly `p_half`, `p`, and `2p` with periods 0.5P, P, and 2P.
- Preserve trusted versus fallback period provenance and never infer trust from class.
- Match the accepted nine-positive numerical fixture within declared tolerances.
- Keep missing or failed light curves as explicit failures, not zero-valued measurements.
- Verify deterministic results and code/source hashes.
- Assert no label is passed to numerical measurement functions.
- Assert no rule-decision or recommendation columns are emitted.

## Readiness gate and exact CatBoost blocker

The first CatBoost smoke test remains blocked until the remaining declared catalogue snapshots have been ingested, `k2pandc` and cross-source conflicts have been quarantined in an expanded EPIC table, and the resulting physical-loss-eligible labels have been joined to the accepted frozen CNN outputs and leakage-screened numerical feature schema. The versioned, hashed `k2pandc` object-level ingestion gate is complete.

The final train/validation/test split is deliberately not a prerequisite for this ingestion work and is not frozen here. After the ingestion gate is approved, a smoke test can use an explicitly provisional EPIC-grouped development resampling scheme. No training occurs in Phase 2A.
