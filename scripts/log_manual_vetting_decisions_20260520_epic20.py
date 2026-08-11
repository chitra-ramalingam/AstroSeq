from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog"
SOURCE_CATALOG = (
    CATALOG_DIR
    / "manual_review_updates"
    / "20260520_epic16"
    / "master_vetted_catalog_manual_review_update_20260520_epic16.csv"
)
PREVIOUS_LEDGER = (
    CATALOG_DIR
    / "manual_review_updates"
    / "20260520_epic16"
    / "manual_vetting_decisions_cumulative_20260520_epic16.csv"
)
SOURCE_RECONCILED_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger_reconciled.csv"
SOURCE_FINAL_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"

STAMP = "20260520"
RUN_ID = f"{STAMP}_epic20"
OUT_DIR = CATALOG_DIR / "manual_review_updates" / RUN_ID
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
        "epic_id": "EPIC_211542908",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Strong odd/even depth mismatch with ratio about 0.391 and explicit delta about 1.27, making the signal "
            "EB-like or unreliable rather than planet-like. The signal is also very shallow, primary SNR is only "
            "modest around 6.38, alias risk is moderate, and the period source is event-spacing fallback with many "
            "candidate periods. Reject from candidate pipeline despite morphology-positive CNN score and no detected "
            "secondary eclipse."
        ),
        "cnn_manual_conflict": "cnn_high_manual_reject",
        "cnn_manual_conflict_reason": (
            "CNN score is high at about 0.934, but manual/diagnostic review rejects due to strong odd/even mismatch, "
            "shallow signal, alias risk, and weak fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211549072",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual plot review does not show a convincing transit-like signal despite very high CNN morphology score. "
            "The numerical diagnostics are also unfavorable: shallow primary depth, modest primary SNR around 6.92, "
            "very long duration around 7.85 hours for P=1.20d, strong odd/even mismatch with ratio about 0.472 and "
            "explicit delta about 0.675, and half-period alias support. Stage F held the object as ambiguous, but "
            "manual review overrides the CNN morphology-positive score because the dips are not visually convincing "
            "and the signal is not planet-like."
        ),
        "cnn_manual_conflict": "cnn_high_manual_reject",
        "cnn_manual_conflict_reason": (
            "Very high CNN score at about 0.991, but manual review finds no convincing dips and diagnostics are "
            "unfavorable."
        ),
    },
    {
        "epic_id": "EPIC_211569404",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual review does not find a convincing planet-like signal. The primary depth is shallow, primary SNR "
            "is only modest around 8.34, and the period is very short at P=0.600d with a relatively long duration "
            "around 3.19 hours. Odd/even consistency is poor, with ratio about 0.700 and explicit delta about 0.353, "
            "and there is half-period alias support despite low formal alias risk. Stage F kept this only as a "
            "secondary-queue hold, so reject as noise/artifact rather than continue holding."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is only moderate and does not materially conflict with manual rejection."
        ),
    },
    {
        "epic_id": "EPIC_211573482",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Signal is present, but diagnostics are unfavorable for a planet-like candidate. Odd/even depth consistency "
            "is poor, with ratio about 0.610 and explicit delta about 0.589, suggesting possible EB behavior. OOT "
            "variability is very high at about 1.16x the transit depth, meaning the apparent events may be embedded "
            "in stellar variability or structured noise. Period source is event-spacing fallback with many candidate "
            "periods, and CNN score is only morphology-positive, not promotion authority. Reject from candidate "
            "pipeline as likely binary/variable contamination rather than continue holding."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is moderate-high but not extreme; manual rejection is driven by EB/variability diagnostics."
        ),
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
    if any(token in raw for token in ("uncertain", "hold", "needs_review", "review_only", "followup", "manual_unreviewed")):
        return "hold"
    if any(token in raw for token in ("candidate_like", "planet_like", "promote", "top_followup_candidate")):
        return "candidate"
    return "unknown"


def is_manual_reject(value: str) -> bool:
    return label_family(value) == "reject"


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


def append_source_file(existing: str) -> str:
    parts = [item.strip() for item in clean(existing).split(";") if item.strip()]
    if LEDGER_REL not in parts:
        parts.append(LEDGER_REL)
    return "; ".join(parts)


def append_unique(existing: str, value: str) -> str:
    parts = [item.strip() for item in clean(existing).split(";") if item.strip()]
    if value and value not in parts:
        parts.append(value)
    return "; ".join(parts)


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


def write_ledgers(increment: pd.DataFrame) -> pd.DataFrame:
    increment.to_csv(MANUAL_LEDGER_INCREMENT, index=False)
    if PREVIOUS_LEDGER.exists():
        previous = pd.read_csv(PREVIOUS_LEDGER, dtype=str).fillna("")
        cumulative = pd.concat([previous, increment], ignore_index=True)
    else:
        cumulative = increment.copy()
    cumulative = cumulative.drop_duplicates(subset=["epic_id"], keep="last")
    cumulative.to_csv(MANUAL_LEDGER_CUMULATIVE, index=False)
    return cumulative


def decision_by_epic() -> dict[str, dict[str, str]]:
    return {item["epic_id"]: item for item in DECISIONS}


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
    rows: list[dict[str, str]] = []
    decisions = decision_by_epic()
    updated = df[df["epic_id"].isin(decisions)]
    for _, row in updated.iterrows():
        epic_id = clean(row.get("epic_id"))
        decision = decisions[epic_id]
        reasons = [item for item in clean(row.get("conflict_reason")).split("; ") if item]
        for reason in reasons:
            if reason in comparison_columns:
                other_col = comparison_columns[reason]
                conflict_type = reason
                other_value = clean(row.get(other_col))
            else:
                other_col = ""
                conflict_type = decision["cnn_manual_conflict"]
                other_value = ""
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
                    "conflict_severity": conflict_severity(conflict_type, clean(row.get("manual_label")), other_value),
                    "conflict_reason": reason,
                    "source_files": clean(row.get("source_files")),
                }
            )
    return pd.DataFrame(rows)


def assert_protected_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    before_part = before[["epic_id", *PROTECTED_COLUMNS]].sort_values("epic_id").reset_index(drop=True)
    after_part = after[["epic_id", *PROTECTED_COLUMNS]].sort_values("epic_id").reset_index(drop=True)
    if not before_part.equals(after_part):
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
            row_change_type = "update_existing_reconciled_ledger_row"
        else:
            i = len(ledger)
            before_label = ""
            before_status = ""
            row_change_type = "append_manual_decision_row_missing_from_reconciled_ledger"
            ledger.loc[i, "epic_id"] = epic_id

        ledger.loc[i, "reconciled_label"] = decision["manual_label"]
        ledger.loc[i, "reconciled_status"] = decision["manual_next_action"]
        ledger.loc[i, "rejection_flag"] = str(is_manual_reject(decision["manual_label"]))
        ledger.loc[i, "uncertainty_flag"] = "False"
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
                "row_change_type": row_change_type,
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
    updated_epics = [item["epic_id"] for item in DECISIONS]
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
    lines.extend(["", "Updated EPICs"])
    for epic in updated_epics:
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
