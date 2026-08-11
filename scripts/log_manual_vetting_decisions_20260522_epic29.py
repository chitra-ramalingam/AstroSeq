from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog"
PREV_RUN = "20260520_epic25"
RUN_ID = "20260522_epic29"
OUT_DIR = CATALOG_DIR / "manual_review_updates" / RUN_ID

SOURCE_CATALOG = CATALOG_DIR / "manual_review_updates" / PREV_RUN / f"master_vetted_catalog_manual_review_update_{PREV_RUN}.csv"
PREVIOUS_LEDGER = CATALOG_DIR / "manual_review_updates" / PREV_RUN / f"manual_vetting_decisions_cumulative_{PREV_RUN}.csv"
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

REVIEW_DATE = date(2026, 5, 22).isoformat()
REVIEWER = "user_provided_manual_review"
LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)

DECISIONS = [
    {
        "epic_id": "EPIC_211619120",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_oot_detrending_and_stage_f_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Some signal-like features are visible, but they are not clear enough for candidate promotion. Primary "
            "SNR is modest around 5.90, odd/even consistency is acceptable but not strongly clean with ratio about "
            "0.835 and explicit delta about 0.174, and there is no detected secondary eclipse. However OOT "
            "variability is very high at about 2.03x the transit depth, suggesting the apparent dip may be embedded "
            "in stellar variability or structured noise. Period source is event-spacing fallback with many candidate "
            "periods. Keep as uncertain hold pending OOT detrending review and Stage F period validation rather than "
            "reject immediately."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and consistent with cautious hold rather than promotion.",
    },
    {
        "epic_id": "EPIC_211683904",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Manual review does not show a convincing transit-like dip. The signal is very shallow with modest primary "
            "SNR around 5.09, and the apparent duration is long at about 4.90 hours for P=1.28d. Odd/even consistency "
            "is only mediocre, with ratio about 0.812 and explicit delta about 0.231. Alias risk is moderate with "
            "strong half-period support, and the period source is event-spacing fallback with many candidate periods. "
            "Reject as noise/artifact despite morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and does not materially conflict with manual rejection.",
    },
    {
        "epic_id": "EPIC_211702989",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual review favors noise/artifact rather than a planet-like candidate. The signal is very shallow "
            "despite modest/high primary SNR, with no detected secondary eclipse, but odd/even consistency is poor: "
            "odd/even ratio about 0.400 and explicit delta about 0.833. Alias risk is moderate with stronger support "
            "at the double-period alias, and the period source is event-spacing fallback with many candidate periods. "
            "Reject from candidate pipeline despite high CNN morphology score because the signal is not visually or "
            "diagnostically planet-like."
        ),
        "cnn_manual_conflict": "cnn_high_manual_reject",
        "cnn_manual_conflict_reason": (
            "CNN score is high at about 0.945, but manual/diagnostic review rejects due to shallow noisy signal, poor "
            "odd/even consistency, moderate alias risk, and fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211730838",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline_but_keep_as_real_eclipsing_signal",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Strong coherent eclipsing signal with excellent odd/even consistency, ratio about 0.996 and explicit "
            "delta about 0.004, many repeated events, saved validation period support, low alias risk, and no "
            "significant secondary eclipse. However the primary depth is extremely large at about 0.099, implying "
            "radius ratio about 0.315, which is too deep for a typical planet-like candidate and is more consistent "
            "with an eclipsing binary, stellar companion, or blend. OOT variability is also non-trivial at about "
            "0.27x transit depth. Reject from the planet candidate pipeline, but keep as a real eclipsing-signal / "
            "binary-system case rather than noise."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN correctly identified strong eclipse-like morphology, but morphology score is not a planet classifier."
        ),
    },
    {
        "epic_id": "EPIC_211745624",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_stage_f_period_validation_and_oot_detrending",
        "manual_confidence": "medium",
        "manual_reason": (
            "Transit-like signal has excellent odd/even consistency with ratio about 0.992 and explicit delta about "
            "0.0075, no significant secondary eclipse, plausible depth around 0.0043, reasonable duration around "
            "1.96 hours, and many event-family members. However primary SNR is weak around 3.49, OOT variability is "
            "non-trivial at about 0.49x transit depth, and the period source is event-spacing fallback with many "
            "candidate periods. Keep as uncertain hold pending Stage F period validation and OOT/detrending review "
            "rather than promote directly."
        ),
        "cnn_manual_conflict": "cnn_high_manual_hold",
        "cnn_manual_conflict_reason": (
            "CNN score is high at about 0.960, but manual review keeps it on hold due to weak SNR, OOT variability, "
            "and fallback period support."
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


def source_reconciled_labels(source_reconciled: pd.DataFrame) -> dict[str, str]:
    if "epic_id" not in source_reconciled.columns or "reconciled_label" not in source_reconciled.columns:
        return {}
    return {
        clean(row["epic_id"]): clean(row["reconciled_label"])
        for _, row in source_reconciled.iterrows()
        if clean(row.get("epic_id")) and clean(row.get("reconciled_label"))
    }


def conflict_reasons(row: pd.Series, prior_ledger_label: str = "") -> list[str]:
    manual_label = clean(row.get("manual_label"))
    checks = [
        ("autovet_label", "manual_label disagrees with autovet_label"),
        ("stage_f_label", "manual_label disagrees with stage_f_label"),
        ("stage_g_label", "manual_label disagrees with stage_g_label"),
        ("final_candidate_status", "manual_label disagrees with candidate_ledger_status"),
    ]
    reasons = [reason for column, reason in checks if manual_label and clean(row.get(column)) and manual_label != clean(row.get(column))]
    if prior_ledger_label and manual_label and manual_label != prior_ledger_label:
        reasons.append("manual_label updates existing_candidate_ledger_state")
    return reasons


def conflict_severity(reason: str, manual_label: str, other_label: str) -> str:
    if reason in {"manual_label disagrees with autovet_label", "manual_label updates existing_candidate_ledger_state"}:
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


def apply_decisions(df: pd.DataFrame, prior_labels: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        prior_label = prior_labels.get(decision["epic_id"], "")
        reasons = conflict_reasons(out.loc[i], prior_label)
        if decision["cnn_manual_conflict"] != "none":
            reasons.append(decision["cnn_manual_conflict_reason"])
        out.loc[i, "has_conflict"] = str(bool(reasons)).lower()
        out.loc[i, "conflict_reason"] = "; ".join(reasons)
    return out, increment


def build_conflict_table(df: pd.DataFrame, prior_labels: dict[str, str]) -> pd.DataFrame:
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
            if reason == "manual_label updates existing_candidate_ledger_state":
                conflict_type = reason
                other_value = prior_labels.get(epic_id, "")
            elif other_col:
                conflict_type = reason
                other_value = clean(row.get(other_col))
            else:
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
                    "existing_reconciled_label": prior_labels.get(epic_id, ""),
                    "cnn_score": clean(row.get("cnn_score")),
                    "cnn_manual_conflict": decision["cnn_manual_conflict"],
                    "conflict_severity": conflict_severity(conflict_type, clean(row.get("manual_label")), other_value),
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
        f"new_decision_rows={len(increment)}",
        f"cumulative_unique_decision_count={len(cumulative)}",
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
    source_reconciled = pd.read_csv(SOURCE_RECONCILED_LEDGER, dtype=str).fillna("")
    prior_labels = source_reconciled_labels(source_reconciled)

    shutil.copy2(SOURCE_CATALOG, SOURCE_BACKUP)
    shutil.copy2(SOURCE_RECONCILED_LEDGER, SOURCE_RECONCILED_LEDGER_BACKUP)
    shutil.copy2(SOURCE_FINAL_LEDGER, SOURCE_FINAL_LEDGER_BACKUP)

    after, increment = apply_decisions(before, prior_labels)
    assert_protected_unchanged(before, after)
    cumulative = write_ledgers(increment)
    conflicts = build_conflict_table(after, prior_labels)
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
