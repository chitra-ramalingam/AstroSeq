from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "phase2"
DATA = ROOT / "data" / "phase2"
CORRECTIONS_PATH = DOCS / "phase2_positive_tier_corrections.csv"
CORRECTION_SOURCE = "docs/phase2/phase2_positive_tier_corrections.csv"

SILVER = {
    "EPIC_211357782": ("Direct manual candidate-like review.", "fallback_untrusted_event_spacing"),
    "EPIC_211497712": ("Direct manual candidate-like review.", "fallback_untrusted_event_spacing"),
    "EPIC_211534076": ("Recovered known unconfirmed candidate.", "trusted_saved_period"),
    "EPIC_211953866": ("Direct manual candidate-like review.", "fallback_untrusted_event_spacing"),
}
BRONZE = {"EPIC_211682657", "EPIC_212001099"}
PENDING = "EPIC_211889692"
EB_CORRECTION = "EPIC_211915147"
ARCHIVE_FP_CORRECTION = "EPIC_212024647"
LOSS_INELIGIBLE = BRONZE | {PENDING}

CORRECTION_COLUMNS = [
    "epic_id", "previous_normalized_label", "new_normalized_label",
    "previous_positive_evidence_tier", "new_positive_evidence_tier",
    "previous_external_disposition", "new_external_disposition",
    "previous_physical_loss_eligible", "new_physical_loss_eligible",
    "previous_training_role", "new_training_role", "correction_basis",
    "evidence_source", "correction_timestamp", "original_evidence_preserved",
]

CORRECTIONS = [
    {
        "epic_id": ARCHIVE_FP_CORRECTION,
        "previous_normalized_label": "candidate_like",
        "new_normalized_label": "false_positive_eb_or_variable",
        "previous_positive_evidence_tier": "positive_bronze",
        "new_positive_evidence_tier": "",
        "previous_external_disposition": "",
        "new_external_disposition": "archive_false_positive",
        "previous_physical_loss_eligible": "false",
        "new_physical_loss_eligible": "true",
        "previous_training_role": "candidate_bronze_traceability_only",
        "new_training_role": "negative_archive_false_positive",
        "correction_basis": (
            "Authoritative NASA Exoplanet Archive k2pandc object EPIC 212024647.01 "
            "is FALSE POSITIVE (Yu et al. 2018; period 3.696871 days; campaigns 5, 16, 18) "
            "and supersedes the internal candidate label as physical training truth."
        ),
        "evidence_source": (
            "data/phase2/catalogues/phase2_k2_catalogue_rows.parquet; "
            "NASA Exoplanet Archive k2pandc snapshot retrieved 2026-08-05"
        ),
        "original_evidence_preserved": "true",
    },
    {
        "epic_id": EB_CORRECTION,
        "previous_normalized_label": "candidate_like",
        "new_normalized_label": "false_positive_eb_or_variable",
        "previous_positive_evidence_tier": "positive_silver",
        "new_positive_evidence_tier": "",
        "previous_external_disposition": "",
        "new_external_disposition": "published_eclipsing_binary",
        "previous_physical_loss_eligible": "true",
        "new_physical_loss_eligible": "true",
        "previous_training_role": "candidate_positive_silver",
        "new_training_role": "negative_eb_variable",
        "correction_basis": (
            "Later external evidence identifies a published eclipsing binary and "
            "supersedes the earlier manual candidate-like physical-class target; "
            "the morphology-supported manual miss makes this a high-value hard negative."
        ),
        "evidence_source": (
            "User-supplied Phase 2A correction directive dated 2026-08-03; "
            "authoritative catalogue citation to be retained by the versioned ingestion record."
        ),
        "original_evidence_preserved": "true",
    },
    {
        "epic_id": PENDING,
        "previous_normalized_label": "candidate_like",
        "new_normalized_label": "candidate_like",
        "previous_positive_evidence_tier": "positive_gold",
        "new_positive_evidence_tier": "external_confirmation_pending",
        "previous_external_disposition": "repository_asserted_confirmed_planet",
        "new_external_disposition": "confirmation_pending_exact_epic_mapping",
        "previous_physical_loss_eligible": "true",
        "new_physical_loss_eligible": "false",
        "previous_training_role": "confirmed_positive_benchmark",
        "new_training_role": "quarantine_until_catalogue_verification",
        "correction_basis": (
            "The repository's recovered-known-confirmed-planet assertion does not reproduce "
            "an exact authoritative EPIC-to-object mapping and confirmed-disposition reference."
        ),
        "evidence_source": (
            "Repository final ledger and Phase 2A correction directive dated 2026-08-03; "
            "authoritative mapping remains pending."
        ),
        "original_evidence_preserved": "true",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def append_once(text: object, value: str, separator: str = "; ") -> str:
    current = "" if pd.isna(text) else str(text)
    if value in current:
        return current
    return f"{current}{separator if current else ''}{value}"


def ensure_correction_log() -> pd.DataFrame:
    if CORRECTIONS_PATH.exists():
        log = pd.read_csv(CORRECTIONS_PATH, dtype=str, keep_default_na=False)
        missing = [c for c in CORRECTION_COLUMNS if c not in log.columns]
        if missing:
            raise ValueError(f"Correction log is missing required columns: {missing}")
    else:
        log = pd.DataFrame(columns=CORRECTION_COLUMNS)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for correction in CORRECTIONS:
        exists = (
            log["epic_id"].eq(correction["epic_id"])
            & log["new_normalized_label"].eq(correction["new_normalized_label"])
            & log["new_positive_evidence_tier"].eq(correction["new_positive_evidence_tier"])
        ).any()
        if not exists:
            record = {**correction, "correction_timestamp": timestamp}
            log = pd.concat([log, pd.DataFrame([record], columns=CORRECTION_COLUMNS)], ignore_index=True)
    log.to_csv(CORRECTIONS_PATH, index=False)
    return log


def ensure_baseline_supersession() -> None:
    path = DOCS / "phase2_baseline_supersessions.csv"
    columns = ["supersession_id", "previous_baseline", "new_baseline", "effective_date", "basis", "historical_evidence_preserved"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    supersession_id = "phase2a_k2pandc_212024647_adjudication_2026-08-05"
    if not frame["supersession_id"].eq(supersession_id).any():
        frame = pd.concat([frame, pd.DataFrame([{
            "supersession_id": supersession_id,
            "previous_baseline": "Gold=0; Silver=4; Bronze_excluded=3; external_confirmation_pending=1; physical_loss_eligible_candidate_positives=4; eligible_eb_variable=86; eligible_noise_artifact=174",
            "new_baseline": "Gold=0; Silver=4; Bronze_excluded=2; external_confirmation_pending=1; physical_loss_eligible_candidate_positives=4; eligible_eb_variable=87; eligible_noise_artifact=174",
            "effective_date": "2026-08-05",
            "basis": "Authoritative k2pandc FALSE POSITIVE adjudication for EPIC_212024647; internal candidate evidence retained historically",
            "historical_evidence_preserved": "true",
        }], columns=columns)], ignore_index=True)
    frame.to_csv(path, index=False)


def patch_tiers() -> pd.DataFrame:
    path = DOCS / "phase2_positive_training_tiers.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    additions = [
        "external_disposition", "external_confirmation_status", "training_role",
        "historical_evidence_source", "original_evidence_preserved", "hard_negative_flag",
    ]
    for column in additions:
        if column not in frame:
            frame[column] = ""

    for epic, (basis, period_trust) in SILVER.items():
        mask = frame["epic_id"].eq(epic)
        frame.loc[mask, ["normalized_label", "positive_evidence_tier", "positive_evidence_basis",
                         "physical_loss_eligible", "period_feature_trust",
                         "external_confirmation_status", "training_role",
                         "original_evidence_preserved"]] = [
            "candidate_like", "positive_silver", basis, "true", period_trust,
            "unconfirmed_candidate", "candidate_positive_silver", "true",
        ]
        frame.loc[mask, "training_caveat"] = (
            "Eligible for the provisional candidate baseline as candidate evidence; "
            "not a confirmed planet and evidence distinctions remain preserved."
        )

    mask = frame["epic_id"].eq(EB_CORRECTION)
    frame.loc[mask, ["normalized_label", "positive_evidence_tier", "positive_evidence_basis",
                     "physical_loss_eligible", "period_feature_trust", "training_caveat",
                     "label_source", "external_disposition", "external_confirmation_status",
                     "training_role", "historical_evidence_source", "original_evidence_preserved"]] = [
        "false_positive_eb_or_variable", "",
        "Published eclipsing-binary evidence supersedes the earlier manual physical-class target.",
        "true", "fallback_untrusted_event_spacing",
        "High-value hard negative: manual review and CNN morphology support were candidate-like.",
        CORRECTION_SOURCE, "published_eclipsing_binary", "published_eclipsing_binary",
        "negative_eb_variable",
        "plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv=candidate_like",
        "true",
    ]

    mask = frame["epic_id"].eq(PENDING)
    frame.loc[mask, ["normalized_label", "positive_evidence_tier", "positive_evidence_basis",
                     "physical_loss_eligible", "training_caveat", "label_source",
                     "external_disposition", "external_confirmation_status", "training_role",
                     "historical_evidence_source", "original_evidence_preserved"]] = [
        "candidate_like", "external_confirmation_pending",
        "Recovered-known-confirmed-planet repository evidence retained, but exact authoritative EPIC mapping is not reproduced.",
        "false", "Quarantine until a versioned authoritative catalogue verifies the exact EPIC-to-object mapping and disposition reference.",
        CORRECTION_SOURCE, "confirmation_pending_exact_epic_mapping", "unverified_mapping",
        "quarantine_until_catalogue_verification",
        "plots/k2_batch/final_candidate_master_ledger.csv=recovered_known_confirmed_planet",
        "true",
    ]

    for epic in BRONZE:
        mask = frame["epic_id"].eq(epic)
        frame.loc[mask, ["normalized_label", "positive_evidence_tier", "physical_loss_eligible",
                         "external_confirmation_status", "training_role",
                         "original_evidence_preserved"]] = [
            "candidate_like", "positive_bronze", "false", "unconfirmed_candidate_excluded_bronze",
            "candidate_bronze_traceability_only", "true",
        ]
        frame.loc[mask, "training_caveat"] = (
            "Excluded from the initial hard physical-class loss; retained for lineage, "
            "traceability, sensitivity analysis, and later soft-label experiments."
        )

    mask = frame["epic_id"].eq(ARCHIVE_FP_CORRECTION)
    frame.loc[mask, ["normalized_label", "positive_evidence_tier", "positive_evidence_basis",
                     "physical_loss_eligible", "period_feature_trust", "training_caveat",
                     "label_source", "external_disposition", "external_confirmation_status",
                     "training_role", "historical_evidence_source", "original_evidence_preserved"]] = [
        "false_positive_eb_or_variable", "",
        "Authoritative k2pandc FALSE POSITIVE evidence supersedes the internal candidate physical target.",
        "true", "trusted_archive_period",
        "Eligible hard negative; the original promote_to_stage_g/candidate evidence remains historical provenance.",
        CORRECTION_SOURCE, "archive_false_positive", "archive_false_positive_adjudicated",
        "negative_archive_false_positive",
        "plots/k2_batch/final_candidate_master_ledger.csv=promote_to_stage_g; positive_bronze",
        "true",
    ]
    frame["hard_negative_flag"] = "false"
    frame.loc[frame["epic_id"].isin([EB_CORRECTION, ARCHIVE_FP_CORRECTION]), "hard_negative_flag"] = "true"
    frame.to_csv(path, index=False)
    return frame


def patch_verification(tiers: pd.DataFrame) -> pd.DataFrame:
    path = DOCS / "phase2_positive_label_verification.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ["positive_evidence_tier", "external_disposition", "external_confirmation_status",
                   "physical_loss_eligible", "superseding_external_evidence",
                   "original_evidence_preserved", "hard_negative_flag"]:
        if column not in frame:
            frame[column] = ""
    tier_map = tiers.set_index("epic_id")
    for idx, row in frame.iterrows():
        epic = row["epic_id"]
        tier = tier_map.loc[epic]
        frame.loc[idx, ["normalized_label", "positive_evidence_tier", "external_disposition",
                        "external_confirmation_status", "physical_loss_eligible",
                        "recommended_training_role", "original_evidence_preserved", "hard_negative_flag"]] = [
            tier["normalized_label"], tier["positive_evidence_tier"], tier["external_disposition"],
            tier["external_confirmation_status"], tier["physical_loss_eligible"],
            tier["training_role"], "true", tier["hard_negative_flag"],
        ]
        frame.loc[idx, "safe_for_supervised_training"] = tier["physical_loss_eligible"].title()
    mask = frame["epic_id"].eq(EB_CORRECTION)
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "candidate_evidence"] = "Historical manual candidate_like review preserved (2026-05-25)."
    frame.loc[mask, "superseding_external_evidence"] = "published_eclipsing_binary"
    frame.loc[mask, "conflicting_evidence"] = "candidate_like | published_eclipsing_binary"
    frame.loc[mask, "verification_notes"] = (
        "External EB evidence supersedes the manual target. The manual/CNN-supported miss is retained as hard-negative provenance."
    )
    mask = frame["epic_id"].eq(PENDING)
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "confirmed_planet_evidence"] = "Historical repository assertion retained; authoritative EPIC mapping not reproduced."
    frame.loc[mask, "verification_notes"] = "Quarantined pending exact authoritative mapping and confirmed disposition reference."
    mask = frame["epic_id"].eq(ARCHIVE_FP_CORRECTION)
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "candidate_evidence"] = "Historical manual promote_to_stage_g and positive_bronze evidence preserved."
    frame.loc[mask, "superseding_external_evidence"] = "k2pandc: EPIC 212024647.01; FALSE POSITIVE; Yu et al. 2018; P=3.696871 d; campaigns=5,16,18"
    frame.loc[mask, "conflicting_evidence"] = "historical_candidate_like | archive_false_positive"
    frame.loc[mask, "verification_notes"] = "Adjudicated archive false positive; authoritative archive evidence controls physical training truth."
    frame.to_csv(path, index=False)
    return frame


def patch_inventory(tiers: pd.DataFrame) -> pd.DataFrame:
    path = DOCS / "phase2_label_inventory.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    additions = [
        "positive_evidence_tier", "external_disposition", "external_confirmation_status",
        "physical_loss_eligible", "training_role", "hard_negative_status",
        "hard_negative_flag", "correction_record", "original_evidence_preserved",
    ]
    for column in additions:
        if column not in frame:
            frame[column] = ""
    safe = as_bool(frame["safe_for_supervised_training"])
    frame["physical_loss_eligible"] = safe.map({True: "true", False: "false"})
    # Candidate truth is controlled by the corrected tier table, not by a generic safe flag.
    frame.loc[frame["current_final_label"].eq("candidate_like"), "physical_loss_eligible"] = "false"
    tier_map = tiers.set_index("epic_id")
    for epic in tier_map.index:
        idx = frame["epic_id"].eq(epic)
        tier = tier_map.loc[epic]
        for column in ["positive_evidence_tier", "external_disposition", "external_confirmation_status",
                       "physical_loss_eligible", "training_role", "hard_negative_flag", "original_evidence_preserved"]:
            frame.loc[idx, column] = tier[column]
        frame.loc[idx, "safe_for_supervised_training"] = tier["physical_loss_eligible"]
        frame.loc[idx, "validation_or_traceability_only"] = str(tier["physical_loss_eligible"] != "true").lower()

    mask = frame["epic_id"].eq(EB_CORRECTION)
    frame.loc[mask, "current_final_label"] = "false_positive_eb_or_variable"
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "known_eb_variable_status"] = "published_eclipsing_binary"
    frame.loc[mask, "ambiguity_status"] = "resolved_external_evidence_supersedes_manual_target"
    frame.loc[mask, "duplicate_conflicting_labels"] = "candidate_like | published_eclipsing_binary"
    frame.loc[mask, "hard_negative_status"] = "high_value_manual_and_cnn_morphology_false_positive"
    frame.loc[mask, "correction_record"] = CORRECTION_SOURCE
    for idx in frame.index[mask]:
        frame.at[idx, "all_label_evidence"] = append_once(
            frame.at[idx, "all_label_evidence"],
            f"{CORRECTION_SOURCE}=published_eclipsing_binary->false_positive_eb_or_variable",
            " || ",
        )
        frame.at[idx, "possible_leakage_concerns"] = append_once(
            frame.at[idx, "possible_leakage_concerns"],
            "correction/provenance fields are target metadata and must be excluded from model features",
        )

    mask = frame["epic_id"].eq(PENDING)
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "confirmed_planet_status"] = "external_confirmation_pending_unverified_mapping"
    frame.loc[mask, "ambiguity_status"] = "external_confirmation_mapping_pending"
    frame.loc[mask, "correction_record"] = CORRECTION_SOURCE
    for idx in frame.index[mask]:
        frame.at[idx, "all_label_evidence"] = append_once(
            frame.at[idx, "all_label_evidence"],
            f"{CORRECTION_SOURCE}=external_confirmation_pending/unverified_mapping",
            " || ",
        )

    mask = frame["epic_id"].eq(ARCHIVE_FP_CORRECTION)
    frame.loc[mask, "current_final_label"] = "false_positive_eb_or_variable"
    frame.loc[mask, "label_source"] = CORRECTION_SOURCE
    frame.loc[mask, "known_eb_variable_status"] = "archive_false_positive"
    frame.loc[mask, "ambiguity_status"] = "resolved_archive_evidence_supersedes_internal_candidate_target"
    frame.loc[mask, "duplicate_conflicting_labels"] = "historical_candidate_like | archive_false_positive"
    frame.loc[mask, "hard_negative_status"] = "authoritative_archive_false_positive_with_candidate_like_morphology"
    frame.loc[mask, "correction_record"] = CORRECTION_SOURCE
    for idx in frame.index[mask]:
        frame.at[idx, "all_label_evidence"] = append_once(
            frame.at[idx, "all_label_evidence"],
            f"{CORRECTION_SOURCE}=k2pandc_archive_false_positive->false_positive_eb_or_variable",
            " || ",
        )
        frame.at[idx, "possible_leakage_concerns"] = append_once(
            frame.at[idx, "possible_leakage_concerns"],
            "archive disposition/correction metadata must remain excluded from scientific features",
        )
    frame.to_csv(path, index=False)
    return frame


def patch_feature_artifacts() -> tuple[pd.DataFrame, str, str]:
    parquet = DATA / "phase2_feature_table.parquet"
    table = pd.read_parquet(parquet)
    mutable = {"normalized_label", "target_eligible", "label_source", "label_conflict_evidence"}
    scientific = [c for c in table.columns if c not in mutable]
    before = pd.util.hash_pandas_object(table[scientific], index=True).values.tobytes()
    before_hash = hashlib.sha256(before).hexdigest()
    table.loc[table["epic_id"].eq(EB_CORRECTION), "normalized_label"] = "false_positive_eb_or_variable"
    table.loc[table["epic_id"].eq(EB_CORRECTION), "label_source"] = CORRECTION_SOURCE
    table.loc[table["epic_id"].eq(EB_CORRECTION), "label_conflict_evidence"] = "candidate_like | published_eclipsing_binary"
    table.loc[table["epic_id"].isin(LOSS_INELIGIBLE), "target_eligible"] = False
    table.loc[table["epic_id"].eq(EB_CORRECTION), "target_eligible"] = True
    table.loc[table["epic_id"].eq(PENDING), "label_source"] = CORRECTION_SOURCE
    table.loc[table["epic_id"].eq(ARCHIVE_FP_CORRECTION), "normalized_label"] = "false_positive_eb_or_variable"
    table.loc[table["epic_id"].eq(ARCHIVE_FP_CORRECTION), "target_eligible"] = True
    table.loc[table["epic_id"].eq(ARCHIVE_FP_CORRECTION), "label_source"] = CORRECTION_SOURCE
    table.loc[table["epic_id"].eq(ARCHIVE_FP_CORRECTION), "label_conflict_evidence"] = "historical_candidate_like | archive_false_positive (adjudicated)"
    table.to_parquet(parquet, index=False)
    after = pd.util.hash_pandas_object(table[scientific], index=True).values.tobytes()
    after_hash = hashlib.sha256(after).hexdigest()
    if before_hash != after_hash:
        raise AssertionError("Scientific/model feature values changed during label correction")

    preview_path = DATA / "phase2_feature_table_preview.csv"
    preview = pd.read_csv(preview_path, dtype=str, keep_default_na=False)
    preview.loc[preview["epic_id"].eq(EB_CORRECTION), "normalized_label"] = "false_positive_eb_or_variable"
    preview.loc[preview["epic_id"].eq(EB_CORRECTION), "label_source"] = CORRECTION_SOURCE
    preview.loc[preview["epic_id"].eq(EB_CORRECTION), "label_conflict_evidence"] = "candidate_like | published_eclipsing_binary"
    preview.loc[preview["epic_id"].isin(LOSS_INELIGIBLE), "target_eligible"] = "False"
    preview.loc[preview["epic_id"].eq(EB_CORRECTION), "target_eligible"] = "True"
    preview.loc[preview["epic_id"].eq(PENDING), "label_source"] = CORRECTION_SOURCE
    preview.loc[preview["epic_id"].eq(ARCHIVE_FP_CORRECTION), "normalized_label"] = "false_positive_eb_or_variable"
    preview.loc[preview["epic_id"].eq(ARCHIVE_FP_CORRECTION), "target_eligible"] = "True"
    preview.loc[preview["epic_id"].eq(ARCHIVE_FP_CORRECTION), "label_source"] = CORRECTION_SOURCE
    preview.loc[preview["epic_id"].eq(ARCHIVE_FP_CORRECTION), "label_conflict_evidence"] = "historical_candidate_like | archive_false_positive (adjudicated)"
    preview.to_csv(preview_path, index=False)
    return table, before_hash, after_hash


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in table.groupby("normalized_label", sort=True):
        has_diag = group[["primary_depth", "primary_depth_snr", "validation_period_days"]].notna().any(axis=1)
        has_embedding = group["cnn_embedding_000"].notna()
        eligible = group["target_eligible"].astype(bool)
        rows.append({
            "corrected_class": label, "retained_rows": len(group),
            "physical_loss_eligible": int(eligible.sum()),
            "nominal_diagnostics": int(has_diag.sum()),
            "eligible_with_nominal_diagnostics": int((eligible & has_diag).sum()),
            "cnn_embeddings": int(has_embedding.sum()),
            "eligible_with_cnn_embeddings": int((eligible & has_embedding).sum()),
        })
    return pd.DataFrame(rows)


def write_summary(inventory: pd.DataFrame, table: pd.DataFrame) -> None:
    path = DOCS / "phase2_label_inventory_summary.txt"
    counts = inventory["current_final_label"].value_counts().sort_index()
    eligible = as_bool(inventory["physical_loss_eligible"])
    hard = table["normalized_label"].isin(["false_positive_eb_or_variable", "reject_as_noise_or_artifact"])
    hard &= table["target_eligible"].astype(bool) & table["cnn_probability"].ge(0.5)
    lines = [
        "Phase 2 label inventory summary (corrected 2026-08-05)", "======================================================", "",
        f"unique_labelled_epics={len(inventory)}",
        f"physical_loss_eligible={int(eligible.sum())}",
        "confirmed_gold_positives=0", "silver_candidate_positives=4", "bronze_excluded=2",
        "external_confirmation_pending=1", "physical_loss_eligible_candidate_positive=4",
        f"physical_loss_eligible_eb_variable={int((eligible & inventory['current_final_label'].eq('false_positive_eb_or_variable')).sum())}",
        f"physical_loss_eligible_noise_artifact={int((eligible & inventory['current_final_label'].eq('reject_as_noise_or_artifact')).sum())}",
        f"uncertain_total={int(inventory['current_final_label'].eq('uncertain_hold').sum())}",
        f"high_cnn_hard_negatives={int(hard.sum())}",
        "explicit_high_value_hard_negative_corrections=2", "",
        "Normalized class counts (current corrected label; historical originals retained in CSV)",
        *(f"{label}={count}" for label, count in counts.items()), "",
        "Coverage by corrected feature-table class",
    ]
    for row in coverage(table).to_dict("records"):
        lines.append(
            f"{row['corrected_class']}: rows={row['retained_rows']}, eligible={row['physical_loss_eligible']}, "
            f"diagnostics={row['nominal_diagnostics']}, eligible_diagnostics={row['eligible_with_nominal_diagnostics']}, "
            f"embeddings={row['cnn_embeddings']}, eligible_embeddings={row['eligible_with_cnn_embeddings']}"
        )
    lines += ["", "Notes", "- Historical label/review evidence remains in all_label_evidence and the append-only correction log.",
              "- Uncertain rows and excluded/pending candidate tiers do not enter the hard physical loss.",
              "- high_cnn_hard_negatives means eligible corrected EB/variable or noise/artifact rows with accepted frozen CNN probability >= 0.5."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tier_audit(tiers: pd.DataFrame) -> None:
    silver_ids = ", ".join(f"`{x}`" for x in SILVER)
    bronze_ids = ", ".join(f"`{x}`" for x in sorted(BRONZE))
    text = f"""# Phase 2A Positive Tier Audit - Corrected

## Corrected outcome

| Evidence state | Count | Hard physical loss | Meaning |
| --- | ---: | --- | --- |
| `positive_gold` | 0 | none | no authoritative confirmed EPIC mapping has yet been reproduced |
| `positive_silver` | 4 | eligible as provisional candidate positives | candidate evidence, not confirmed planets |
| `positive_bronze` | 2 | excluded | lineage/sensitivity only |
| `external_confirmation_pending` | 1 | quarantined | exact EPIC mapping and confirmed disposition reference required |
| corrected EB/variable negatives | 2 | eligible as negatives | published EB and archive false-positive hard negatives |

The four Silver candidate positives are {silver_ids}. Their evidence distinctions remain separate: direct manual candidate-like review and recovered-known-unconfirmed-candidate provenance are not collapsed into confirmation.

The Bronze rows {bronze_ids} retain their normalized labels and diagnostics but remain excluded from the initial hard physical-class loss.

## Corrections and precedence

`EPIC_211915147` is now `false_positive_eb_or_variable`, with external disposition `published_eclipsing_binary`, training role `negative_eb_variable`, and fallback/untrusted event-spacing period provenance. The 2026-05-25 manual `candidate_like` judgment remains historical evidence. Later external physical-class evidence supersedes that judgment; the manual and CNN morphology support make the row a high-value hard negative.

`EPIC_211889692` remains normalized `candidate_like` for repository lineage but is now `external_confirmation_pending`, `unverified_mapping`, loss-ineligible, and quarantined. The recovered-known-confirmed-planet assertion remains historical evidence. Gold may be restored only through a new append-only correction after exact authoritative EPIC mapping and a confirmed disposition reference are ingested.

`EPIC_212024647` is now `false_positive_eb_or_variable`, with `archive_false_positive` disposition and `negative_archive_false_positive` training role. The authoritative `k2pandc` object `EPIC 212024647.01` (Yu et al. 2018; 3.696871 days; campaigns 5, 16, 18) supersedes the internal candidate target. The manual promotion and Bronze tier remain append-only historical evidence.

## Leakage and non-actions

Evidence tier, external disposition/status, training role, correction basis, manual labels/reasons, GateVetter outputs, final recommendations, catalogue-derived target fields, and target-derived verdicts are metadata only and are excluded from the scientific model feature matrix. CNN probabilities, embeddings, and numerical light-curve diagnostics were not changed.

No model was trained, no CNN was loaded or modified, no split was frozen, and no candidate-search batch was run.
"""
    (DOCS / "PHASE2_POSITIVE_TIER_AUDIT.md").write_text(text, encoding="utf-8")


def snapshot_rows() -> list[tuple[str, str, str]]:
    inventory = pd.read_csv(DOCS / "phase2_catalogue_source_inventory.csv", dtype=str, keep_default_na=False)
    rows = []
    for _, row in inventory.iterrows():
        local = row["exact_local_path_if_present"]
        exists = bool(local) and (ROOT / local).exists()
        status = row["local_status"]
        readiness = "ready for offline ingestion" if exists and status.startswith("present_") and "legacy" not in status and "stripped" not in status and "unversioned" not in status else "not ready for authoritative offline ingestion"
        if not exists:
            readiness = "requires download permission"
        rows.append((row["catalogue_name"], f"{status}; {'exists' if exists else 'absent'}", readiness))
    return rows


def write_correction_audit(inventory: pd.DataFrame, table: pd.DataFrame, feature_hash: str) -> None:
    cov = coverage(table)
    class_counts = inventory["current_final_label"].value_counts().to_dict()
    eligible = as_bool(inventory["physical_loss_eligible"])
    hard = table["normalized_label"].isin(["false_positive_eb_or_variable", "reject_as_noise_or_artifact"])
    hard &= table["target_eligible"].astype(bool) & table["cnn_probability"].ge(0.5)
    snapshot_table = "\n".join(f"| {name} | {status} | {ready} |" for name, status, ready in snapshot_rows())
    coverage_table = "\n".join(
        f"| `{r.corrected_class}` | {r.retained_rows} | {r.physical_loss_eligible} | {r.nominal_diagnostics} | {r.eligible_with_nominal_diagnostics} | {r.cnn_embeddings} | {r.eligible_with_cnn_embeddings} |"
        for r in cov.itertuples()
    )
    text = f"""# Phase 2A Label Correction Audit

## Outcome

- Confirmed Gold positives: **0**.
- Silver candidate positives: **4**.
- Bronze excluded: **2**.
- External-confirmation-pending: **1**.
- Physical-loss-eligible candidate positives: **4**.
- Physical-loss-eligible EB/variable rows: **{int((eligible & inventory['current_final_label'].eq('false_positive_eb_or_variable')).sum())}**.
- Physical-loss-eligible noise/artifact rows: **{int((eligible & inventory['current_final_label'].eq('reject_as_noise_or_artifact')).sum())}**.
- Uncertain rows: **{class_counts.get('uncertain_hold', 0)} total; 0 loss-eligible**.
- High-CNN hard negatives: **{int(hard.sum())}** eligible corrected negative rows with accepted frozen CNN probability >= 0.5.
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
{coverage_table}

The retained feature table has 268 rows. The three loss-ineligible candidate rows remain present for lineage. Scientific feature hash before and after the metadata-only patch: `{feature_hash}` (identical).

## Leakage audit and non-actions

The scientific matrix excludes positive evidence tier, external disposition/status, training role, correction basis, manual labels/reasons, GateVetter predictions/reasons/actions, final recommendations, catalogue-derived normalized target fields, and rule verdicts derived from target class. Only label/control metadata changed. CNN probabilities, all 128 embedding dimensions, numerical diagnostics, and missingness indicators are value-identical.

No model was trained; the CNN was not loaded, modified, or retrained; no split was frozen; no candidate search was run; and no catalogue data were downloaded. The already accepted local `k2pandc` ingestion was used only as correction evidence.

## Accepted archive evidence

The versioned local NASA Exoplanet Archive `k2pandc` snapshot and its accepted 4,064-row object-level ingestion were used offline. `EPIC 212024647.01` is retained as `FALSE POSITIVE`, reference `Yu et al. 2018`, period 3.696871 days, campaigns 5, 16, and 18. No additional catalogue was downloaded or ingested in this task.
"""
    (DOCS / "PHASE2_LABEL_CORRECTION_AUDIT.md").write_text(text, encoding="utf-8")


def patch_expansion_doc() -> None:
    path = DOCS / "PHASE2_EXPANSION_COVERAGE_ESTIMATE.md"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "| Aligned nominal diagnostics for the accepted scientific baseline | 201 | 9 positives plus 192 safe negatives |":
            "| Aligned nominal diagnostics retained in the feature artifact | 201 | 8 candidate-labelled rows plus 193 corrected negative rows; 197 are currently loss-eligible |",
        "- Gold positives: **1**.": "- Confirmed Gold positives: **0**.",
        "- Silver positives: **5**.": "- Silver candidate positives: **4** (unconfirmed candidate evidence).",
        "- Bronze positives: **3**.": "- Bronze excluded rows: **3**.",
        "- Physical-loss-eligible current positives: **6**.": "- External-confirmation-pending rows: **1**; physical-loss-eligible candidate positives: **4**.",
        "- Current positives with trusted periods: **5**.": "- Physical-loss-eligible candidate positives with trusted periods: **1**.",
        "- Current positives with fallback periods: **4**.": "- Physical-loss-eligible candidate positives with fallback periods: **3**; corrected EB/variable hard negative with fallback period: **1**.",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            raise AssertionError(f"Expected expansion-audit text not found: {old}")
    old_freeze = "The accepted 268-row CNN freeze supersedes the older `has_current_cnn_score` flags"
    new_freeze = "The accepted 268-row CNN freeze is retained unchanged; four candidate-labelled rows are now loss-ineligible. It supersedes the older `has_current_cnn_score` flags"
    if old_freeze in text:
        text = text.replace(old_freeze, new_freeze)
    elif new_freeze not in text:
        raise AssertionError("Expected accepted-freeze statement not found")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the append-only Phase 2A label corrections without training.")
    parser.add_argument("--audit-only", action="store_true", help="Required safety switch; no training or network access is implemented.")
    args = parser.parse_args()
    if not args.audit_only:
        raise ValueError("Pass --audit-only; this utility only applies label/provenance corrections")
    ensure_correction_log()
    ensure_baseline_supersession()
    tiers = patch_tiers()
    patch_verification(tiers)
    inventory = patch_inventory(tiers)
    table, before_hash, after_hash = patch_feature_artifacts()
    if before_hash != after_hash:
        raise AssertionError("Model feature hash changed")
    write_summary(inventory, table)
    write_tier_audit(tiers)
    write_correction_audit(inventory, table, before_hash)
    print(f"phase2_label_corrections_applied=true feature_hash={before_hash} rows={len(table)}")


if __name__ == "__main__":
    main()
