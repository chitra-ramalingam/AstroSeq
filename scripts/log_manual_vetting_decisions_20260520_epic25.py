from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog"
PREV_RUN = "20260520_epic20"
RUN_ID = "20260520_epic25"
OUT_DIR = CATALOG_DIR / "manual_review_updates" / RUN_ID

SOURCE_CATALOG = (
    CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"master_vetted_catalog_manual_review_update_{PREV_RUN}.csv"
)
PREVIOUS_LEDGER = (
    CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"manual_vetting_decisions_cumulative_{PREV_RUN}.csv"
)
SOURCE_RECONCILED_LEDGER = (
    CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"final_candidate_master_ledger_reconciled_manual_review_update_{PREV_RUN}.csv"
)
SOURCE_FINAL_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"

UPDATED_CATALOG = OUT_DIR / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
UPDATED_CONFLICTS = OUT_DIR / f"master_vetted_catalog_conflicts_manual_review_update_{RUN_ID}.csv"
MANUAL_LEDGER_INCREMENT = OUT_DIR / f"manual_vetting_decisions_increment_{RUN_ID}.csv"
MANUAL_LEDGER_CUMULATIVE = OUT_DIR / f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
UPDATED_RECONCILED_LEDGER = OUT_DIR / f"final_candidate_master_ledger_reconciled_manual_review_update_{RUN_ID}.csv"
LEDGER_AUDIT = OUT_DIR / f"final_candidate_master_ledger_manual_review_audit_{RUN_ID}.csv"
SUMMARY_TXT = OUT_DIR / f"manual_vetting_update_summary_{RUN_ID}.txt"
MANIFEST_JSON = OUT_DIR / f"manual_vetting_update_manifest_{RUN_ID}.json"
SOURCE_BACKUP = OUT_DIR / f"source_catalog_backup_before_manual_review_update_{RUN_ID}.csv"
SOURCE_RECONCILED_LEDGER_BACKUP = OUT_DIR / f"source_reconciled_ledger_backup_before_manual_review_update_{RUN_ID}.csv"
SOURCE_FINAL_LEDGER_BACKUP = OUT_DIR / f"source_final_candidate_master_ledger_backup_unmodified_{RUN_ID}.csv"

REVIEW_DATE = date(2026, 5, 20).isoformat()
REVIEWER = "user_provided_manual_review"
LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)

DECISIONS = [
    {
        "epic_id": "EPIC_211580521",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "OOT variability is acceptable at about 0.17x transit depth, but the rest of the evidence is weak or "
            "unfavorable. Primary SNR is low at about 4.74, event support is limited, and the period is very short "
            "at P=0.545d. Odd/even consistency is poor, with ratio about 0.611 and explicit delta about 0.464. "
            "Alias risk is high with strong half-period support, and the period source is event-spacing fallback "
            "with many candidate periods. Reject as noise/artifact rather than continue holding despite "
            "morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and does not materially conflict with manual rejection.",
    },
    {
        "epic_id": "EPIC_211581700",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Strong real eclipsing signal with excellent odd/even consistency and many repeated events, but the "
            "secondary eclipse is highly significant and deeper than expected for a planet-like occultation. "
            "Secondary SNR is about 19.83 and secondary-to-primary depth ratio is about 3.87, making this much more "
            "consistent with an eclipsing binary or variable binary system than a planet candidate. Reject from "
            "planet candidate pipeline despite strong morphology and high CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN correctly identified strong eclipse-like morphology, but morphology score is not a planet classifier."
        ),
    },
    {
        "epic_id": "EPIC_211586814",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Signal has decent primary SNR, but diagnostics are EB-like rather than planet-like. Odd/even depth "
            "consistency is poor, with ratio about 0.538 and explicit delta about 0.600. The period is very short "
            "at P=0.504d with a relatively long duration of about 3.68 hours, and there is strong half-period alias "
            "support. Stage F already held it as ambiguous due to odd/even risk. Reject from candidate pipeline as "
            "likely binary/alias-contaminated despite morphology-positive CNN score and no detected secondary eclipse."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is moderately high but not extreme; manual rejection is driven by odd/even mismatch, "
            "short-period geometry, and alias concerns."
        ),
    },
    {
        "epic_id": "EPIC_211592766",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_stage_f_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Transit-like signal has acceptable primary SNR around 8.09, good odd/even consistency with ratio about "
            "0.933 and explicit delta about 0.072, no significant secondary eclipse, and low alias risk. However "
            "the transit duration is long at about 7.36 hours for P=2.09d, OOT variability is non-trivial at about "
            "0.27x transit depth, CNN morphology score is only moderate, and the period source is event-spacing "
            "fallback with many candidate periods. Keep as uncertain hold pending Stage F period validation and "
            "improved plot/detrending review rather than promote."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and consistent with a cautious hold rather than promotion.",
    },
    {
        "epic_id": "EPIC_211619120",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium",
        "manual_reason": (
            "Some weak signal-like features are visible, but they are not clear or convincing enough for promotion "
            "or hold. Primary SNR is modest around 5.90, CNN morphology score is only moderate, and OOT variability "
            "is very high at about 2.03x the transit depth, suggesting the apparent dip may be dominated by stellar "
            "variability or structured noise. Odd/even consistency is acceptable but not strongly clean, with ratio "
            "about 0.835 and explicit delta about 0.174. Period source is event-spacing fallback with many candidate "
            "periods, so reject as noise/artifact rather than continue holding."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and does not materially conflict with manual rejection.",
    },
]

PROTECTED_COLUMNS = [
    "cnn_model_path",
    "cnn_score",
    "cnn_score_name",
    "cnn_role",
    "morphology_positive",
    "cnn_policy_version",
]


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def label_family(value: str) -> str:
    raw = value.lower()
    if not raw:
        return "unknown"
    if any(token in raw for token in ("reject", "binary", "noise", "artifact", "false_positive", "do_not_promote")):
        return "reject"
    if any(token in raw for token in ("uncertain", "hold", "needs_review", "review_only", "followup")):
        return "hold"
    if any(token in raw for token in ("candidate_like", "planet_like", "promote", "top_followup_candidate")):
        return "candidate"
    return "unknown"


def is_manual_reject(value: str) -> bool:
    return label_family(value) == "reject"


def append_unique(existing: str, value: str) -> str:
    parts = [item.strip() for item in clean(existing).split(";") if item.strip()]
    if value and value not in parts:
        parts.append(value)
    return "; ".join(parts)


def append_source_file(existing: str) -> str:
    return append_unique(existing, LEDGER_REL)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "manual_label",
        "manual_reason",
        "manual_reviewer",
        "manual_review_date",
        "manual_next_action",
        "manual_confidence",
        "manual_vetted",
        "decision_authority",
        "master_label",
        "master_reason",
        "master_next_action",
        "has_conflict",
        "conflict_reason",
        "cnn_manual_conflict",
        "cnn_manual_conflict_reason",
        "review_level",
        "autovet_only",
        "source_files",
        "last_updated",
    ]:
        if column not in df.columns:
            df[column] = ""
    return df


def build_increment_ledger() -> pd.DataFrame:
    ledger = pd.DataFrame(DECISIONS)
    ledger = ledger.rename(columns={"cnn_manual_conflict_reason": "conflict_reason"})
    ledger["reviewed_at"] = REVIEW_DATE
    ledger["reviewer"] = REVIEWER
    ledger["decision_authority"] = "manual_review"
    ledger["manual_vetted"] = "true"
    return ledger


def conflict_reasons(row: pd.Series) -> list[str]:
    manual_label = clean(row.get("manual_label"))
    checks = [
        ("autovet_label", "manual_label disagrees with autovet_label"),
        ("stage_f_label", "manual_label disagrees with stage_f_label"),
        ("stage_g_label", "manual_label disagrees with stage_g_label"),
        ("final_candidate_status", "manual_label disagrees with candidate_ledger_status"),
    ]
    return [reason for column, reason in checks if manual_label and clean(row.get(column)) and manual_label != clean(row.get(column))]


def conflict_severity(reason: str, manual_label: str, other_label: str) -> str:
    if reason == "manual_label disagrees with autovet_label":
        return "soft_conflict"
    if reason.startswith("cnn_"):
        return "expected_cnn_false_positive" if is_manual_reject(manual_label) else "soft_conflict"
    manual_family = label_family(manual_label)
    other_family = label_family(other_label)
    if manual_family != "unknown" and manual_family == other_family:
        return "vocabulary_mismatch"
    return "hard_conflict"


def write_ledgers(increment: pd.DataFrame) -> pd.DataFrame:
    increment.to_csv(MANUAL_LEDGER_INCREMENT, index=False)
    previous = pd.read_csv(PREVIOUS_LEDGER, dtype=str).fillna("")
    cumulative = pd.concat([previous, increment], ignore_index=True)
    cumulative = cumulative.drop_duplicates(subset=["epic_id"], keep="last")
    cumulative.to_csv(MANUAL_LEDGER_CUMULATIVE, index=False)
    return cumulative


def apply_decisions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_columns(df.copy())
    increment = build_increment_ledger()
    missing = sorted(set(increment["epic_id"]) - set(out["epic_id"]))
    if missing:
        raise RuntimeError(f"Missing EPICs in source catalog: {missing}")

    for decision in DECISIONS:
        idx = out.index[out["epic_id"].eq(decision["epic_id"])]
        if len(idx) != 1:
            raise RuntimeError(f"Expected one row for {decision['epic_id']}, found {len(idx)}")
        i = idx[0]
        out.loc[i, "manual_label"] = decision["manual_label"]
        out.loc[i, "manual_reason"] = decision["manual_reason"]
        out.loc[i, "manual_next_action"] = decision["manual_next_action"]
        out.loc[i, "manual_confidence"] = decision["manual_confidence"]
        out.loc[i, "manual_reviewer"] = REVIEWER
        out.loc[i, "manual_review_date"] = REVIEW_DATE
        out.loc[i, "manual_vetted"] = "true"
        out.loc[i, "decision_authority"] = "manual_review"
        out.loc[i, "master_label"] = decision["manual_label"]
        out.loc[i, "master_reason"] = decision["manual_reason"]
        out.loc[i, "master_next_action"] = decision["manual_next_action"]
        out.loc[i, "review_level"] = "manually_reviewed"
        out.loc[i, "autovet_only"] = "false"
        out.loc[i, "cnn_manual_conflict"] = decision["cnn_manual_conflict"]
        out.loc[i, "cnn_manual_conflict_reason"] = decision["cnn_manual_conflict_reason"]
        out.loc[i, "source_files"] = append_source_file(out.loc[i, "source_files"])
        out.loc[i, "last_updated"] = REVIEW_DATE
        reasons = conflict_reasons(out.loc[i])
        if decision["cnn_manual_conflict"] != "none":
            reasons.append(decision["cnn_manual_conflict_reason"])
        out.loc[i, "has_conflict"] = str(bool(reasons)).lower()
        out.loc[i, "conflict_reason"] = "; ".join(reasons)
    return out, increment


def build_conflict_table(df: pd.DataFrame) -> pd.DataFrame:
    comparison_columns = {
        "manual_label disagrees with autovet_label": "autovet_label",
        "manual_label disagrees with stage_f_label": "stage_f_label",
        "manual_label disagrees with stage_g_label": "stage_g_label",
        "manual_label disagrees with candidate_ledger_status": "final_candidate_status",
    }
    decisions = {item["epic_id"]: item for item in DECISIONS}
    rows: list[dict[str, str]] = []
    for _, row in df[df["epic_id"].isin(decisions)].iterrows():
        epic_id = clean(row.get("epic_id"))
        decision = decisions[epic_id]
        for reason in [item for item in clean(row.get("conflict_reason")).split("; ") if item]:
            other_col = comparison_columns.get(reason, "")
            conflict_type = reason if other_col else decision["cnn_manual_conflict"]
            rows.append(
                {
                    "epic_id": epic_id,
                    "conflict_type": conflict_type,
                    "manual_label": clean(row.get("manual_label")),
                    "autovet_label": clean(row.get("autovet_label")),
                    "stage_f_label": clean(row.get("stage_f_label")),
                    "stage_g_label": clean(row.get("stage_g_label")),
                    "candidate_ledger_status": clean(row.get("final_candidate_status")),
                    "cnn_score": clean(row.get("cnn_score")),
                    "cnn_manual_conflict": decision["cnn_manual_conflict"],
                    "conflict_severity": conflict_severity(conflict_type, clean(row.get("manual_label")), clean(row.get(other_col))),
                    "conflict_reason": reason,
                    "source_files": clean(row.get("source_files")),
                }
            )
    return pd.DataFrame(rows)


def assert_protected_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    left = before[["epic_id", *PROTECTED_COLUMNS]].sort_values("epic_id").reset_index(drop=True)
    right = after[["epic_id", *PROTECTED_COLUMNS]].sort_values("epic_id").reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("Protected CNN/frozen-policy columns changed")


def update_reconciled_ledger_copy(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = source.copy()
    for column in [
        "manual_label",
        "manual_next_action",
        "manual_reason",
        "manual_confidence",
        "manual_vetted",
        "decision_authority",
        "reviewed_at",
        "reviewer",
        "cnn_manual_conflict",
        "conflict_reason",
        "source_files",
        "source_labels",
        "source_actions",
    ]:
        if column not in ledger.columns:
            ledger[column] = ""

    audits: list[dict[str, str]] = []
    for decision in DECISIONS:
        epic_id = decision["epic_id"]
        idx = ledger.index[ledger["epic_id"].eq(epic_id)]
        if len(idx):
            i = idx[0]
            before_label = clean(ledger.loc[i, "reconciled_label"])
            before_status = clean(ledger.loc[i, "reconciled_status"])
            change_type = "update_existing_reconciled_ledger_row"
        else:
            i = len(ledger)
            ledger.loc[i, "epic_id"] = epic_id
            before_label = ""
            before_status = ""
            change_type = "append_manual_decision_row_missing_from_reconciled_ledger"

        ledger.loc[i, "reconciled_label"] = decision["manual_label"]
        ledger.loc[i, "reconciled_status"] = decision["manual_next_action"]
        ledger.loc[i, "rejection_flag"] = str(is_manual_reject(decision["manual_label"]))
        ledger.loc[i, "uncertainty_flag"] = str(label_family(decision["manual_label"]) == "hold")
        ledger.loc[i, "positive_retained_flag"] = "False"
        ledger.loc[i, "strongest_positive_flag"] = "False"
        ledger.loc[i, "latest_known_stage"] = "Manual review"
        ledger.loc[i, "manual_decision_present"] = "True"
        ledger.loc[i, "manual_label"] = decision["manual_label"]
        ledger.loc[i, "manual_next_action"] = decision["manual_next_action"]
        ledger.loc[i, "manual_reason"] = decision["manual_reason"]
        ledger.loc[i, "manual_confidence"] = decision["manual_confidence"]
        ledger.loc[i, "manual_vetted"] = "true"
        ledger.loc[i, "decision_authority"] = "manual_review"
        ledger.loc[i, "reviewed_at"] = REVIEW_DATE
        ledger.loc[i, "reviewer"] = REVIEWER
        ledger.loc[i, "cnn_manual_conflict"] = decision["cnn_manual_conflict"]
        ledger.loc[i, "conflict_reason"] = decision["cnn_manual_conflict_reason"]
        ledger.loc[i, "source_files"] = append_source_file(clean(ledger.loc[i, "source_files"]))
        ledger.loc[i, "source_labels"] = append_unique(clean(ledger.loc[i, "source_labels"]), decision["manual_label"])
        ledger.loc[i, "source_actions"] = append_unique(clean(ledger.loc[i, "source_actions"]), decision["manual_next_action"])
        ledger.loc[i, "reconciliation_reason"] = (
            "manual_review overrides existing reconciled ledger state; CNN morphology score is not promotion authority"
        )
        audits.append(
            {
                "epic_id": epic_id,
                "row_change_type": change_type,
                "previous_reconciled_label": before_label,
                "previous_reconciled_status": before_status,
                "manual_label": decision["manual_label"],
                "manual_next_action": decision["manual_next_action"],
                "manual_confidence": decision["manual_confidence"],
                "manual_vetted": "true",
                "decision_authority": "manual_review",
                "cnn_manual_conflict": decision["cnn_manual_conflict"],
                "conflict_reason": decision["cnn_manual_conflict_reason"],
                "reviewed_at": REVIEW_DATE,
            }
        )
    return ledger, pd.DataFrame(audits)


def write_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
    increment: pd.DataFrame,
    cumulative: pd.DataFrame,
    conflicts: pd.DataFrame,
    ledger_audit: pd.DataFrame,
) -> None:
    label_counts = increment["manual_label"].value_counts().to_dict()
    action_counts = increment["manual_next_action"].value_counts().to_dict()
    cnn_counts = increment["cnn_manual_conflict"].value_counts().to_dict()
    lines = [
        "Manual vetting append/update-copy summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"review_date={REVIEW_DATE}",
        f"source_catalog={SOURCE_CATALOG.relative_to(ROOT).as_posix()}",
        f"source_backup={SOURCE_BACKUP.relative_to(ROOT).as_posix()}",
        f"source_reconciled_ledger_backup={SOURCE_RECONCILED_LEDGER_BACKUP.relative_to(ROOT).as_posix()}",
        f"source_final_candidate_master_ledger_backup={SOURCE_FINAL_LEDGER_BACKUP.relative_to(ROOT).as_posix()}",
        f"increment_ledger={MANUAL_LEDGER_INCREMENT.relative_to(ROOT).as_posix()}",
        f"cumulative_ledger={MANUAL_LEDGER_CUMULATIVE.relative_to(ROOT).as_posix()}",
        f"updated_catalog={UPDATED_CATALOG.relative_to(ROOT).as_posix()}",
        f"updated_conflicts={UPDATED_CONFLICTS.relative_to(ROOT).as_posix()}",
        f"updated_reconciled_ledger_copy={UPDATED_RECONCILED_LEDGER.relative_to(ROOT).as_posix()}",
        f"ledger_audit={LEDGER_AUDIT.relative_to(ROOT).as_posix()}",
        "",
        "Safety confirmations",
        "- Did not retrain.",
        "- Did not edit CNN scores.",
        "- Did not change the frozen CNN policy.",
        "- Did not overwrite original master_vetted_catalog.csv.",
        "- Did not overwrite final_candidate_master_ledger.csv.",
        "- Wrote append/update-copy artifacts and timestamped source backups.",
        "- CNN score remains transit_morphology_score only.",
        "- CNN role remains morphology_scorer_only.",
        "- CNN morphology-positive is not promotion authority.",
        "- Manual review overrides auto_hold_needs_review for master label/action.",
        "",
        f"new_decision_count={len(increment)}",
        f"cumulative_decision_count={len(cumulative)}",
        f"conflict_rows_for_new_epics={len(conflicts)}",
        f"ledger_audit_rows={len(ledger_audit)}",
        "",
        "Batch label counts",
    ]
    lines.extend(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    lines.extend(["", "Batch action counts"])
    lines.extend(f"- {action}: {count}" for action, count in sorted(action_counts.items()))
    lines.extend(["", "CNN/manual conflicts"])
    lines.extend(f"- {kind}: {count}" for kind, count in sorted(cnn_counts.items()))
    lines.extend(
        [
            "",
            "Batch notes",
            "- EPIC_211581700 is a calibration case: strong real eclipsing signal, but binary_system due to highly significant secondary eclipse.",
            "- EPIC_211592766 is the only survivor from this batch, kept as uncertain_hold pending Stage F validation.",
            "",
            "Updated EPICs",
        ]
    )
    for epic in [item["epic_id"] for item in DECISIONS]:
        before_row = before.loc[before["epic_id"].eq(epic)].iloc[0]
        after_row = after.loc[after["epic_id"].eq(epic)].iloc[0]
        lines.append(
            "- "
            + epic
            + f": {clean(before_row.get('master_label'))} -> {clean(after_row.get('master_label'))}; "
            + f"authority={clean(after_row.get('decision_authority'))}; "
            + f"cnn_manual_conflict={clean(after_row.get('cnn_manual_conflict'))}; "
            + f"conflict={clean(after_row.get('has_conflict'))} {clean(after_row.get('conflict_reason'))}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [SOURCE_CATALOG, PREVIOUS_LEDGER, SOURCE_RECONCILED_LEDGER, SOURCE_FINAL_LEDGER]:
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(SOURCE_CATALOG, dtype=str).fillna("")
    shutil.copy2(SOURCE_CATALOG, SOURCE_BACKUP)
    shutil.copy2(SOURCE_RECONCILED_LEDGER, SOURCE_RECONCILED_LEDGER_BACKUP)
    shutil.copy2(SOURCE_FINAL_LEDGER, SOURCE_FINAL_LEDGER_BACKUP)

    after, increment = apply_decisions(before)
    assert_protected_unchanged(before, after)
    cumulative = write_ledgers(increment)
    conflicts = build_conflict_table(after)

    source_reconciled = pd.read_csv(SOURCE_RECONCILED_LEDGER, dtype=str).fillna("")
    updated_reconciled, ledger_audit = update_reconciled_ledger_copy(source_reconciled)

    after.to_csv(UPDATED_CATALOG, index=False)
    conflicts.to_csv(UPDATED_CONFLICTS, index=False)
    updated_reconciled.to_csv(UPDATED_RECONCILED_LEDGER, index=False)
    ledger_audit.to_csv(LEDGER_AUDIT, index=False)
    write_summary(before, after, increment, cumulative, conflicts, ledger_audit)
    manifest = {
        "run_id": RUN_ID,
        "review_date": REVIEW_DATE,
        "new_updated_epics": [item["epic_id"] for item in DECISIONS],
        "source_catalog": str(SOURCE_CATALOG),
        "source_backup": str(SOURCE_BACKUP),
        "source_reconciled_ledger": str(SOURCE_RECONCILED_LEDGER),
        "source_reconciled_ledger_backup": str(SOURCE_RECONCILED_LEDGER_BACKUP),
        "source_final_candidate_master_ledger": str(SOURCE_FINAL_LEDGER),
        "source_final_candidate_master_ledger_backup": str(SOURCE_FINAL_LEDGER_BACKUP),
        "increment_ledger": str(MANUAL_LEDGER_INCREMENT),
        "cumulative_ledger": str(MANUAL_LEDGER_CUMULATIVE),
        "updated_catalog": str(UPDATED_CATALOG),
        "updated_conflicts": str(UPDATED_CONFLICTS),
        "updated_reconciled_ledger_copy": str(UPDATED_RECONCILED_LEDGER),
        "ledger_audit": str(LEDGER_AUDIT),
        "summary": str(SUMMARY_TXT),
        "did_not_retrain": True,
        "cnn_scores_unchanged": True,
        "frozen_cnn_policy_unchanged": True,
        "original_master_catalog_overwritten": False,
        "final_candidate_master_ledger_overwritten": False,
        "cnn_score_role_note": "transit_morphology_score only; morphology_scorer_only; not promotion authority",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
