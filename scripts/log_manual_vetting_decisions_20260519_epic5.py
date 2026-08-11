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
    / "cnn_backfill"
    / "master_vetted_catalog_cnn_backfilled_review_level_refined.csv"
)
STAMP = "20260519"
RUN_ID = f"{STAMP}_epic5"
OUT_DIR = CATALOG_DIR / "manual_review_updates" / RUN_ID
UPDATED_CATALOG = OUT_DIR / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
UPDATED_CONFLICTS = OUT_DIR / f"master_vetted_catalog_conflicts_manual_review_update_{RUN_ID}.csv"
MANUAL_LEDGER = OUT_DIR / f"manual_vetting_decisions_{RUN_ID}.csv"
SUMMARY_TXT = OUT_DIR / f"manual_vetting_update_summary_{RUN_ID}.txt"
MANIFEST_JSON = OUT_DIR / f"manual_vetting_update_manifest_{RUN_ID}.json"
SOURCE_BACKUP = OUT_DIR / f"source_catalog_backup_before_manual_review_update_{RUN_ID}.csv"

REVIEW_DATE = date(2026, 5, 19).isoformat()
REVIEWER = "user_provided_manual_review"
LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_{RUN_ID}.csv"
)

DECISIONS = [
    {
        "epic_id": "EPIC_200008831",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Strong odd/even depth mismatch with ratio 0.350; high half-period alias risk; "
            "Stage F likely EB; noisy/unstable morphology; not planet-like."
        ),
    },
    {
        "epic_id": "EPIC_200008924",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck",
        "manual_confidence": "medium",
        "manual_reason": (
            "Weak primary SNR and Stage F rejected due to insignificant cached folded transit, "
            "but odd/even is not severe and no secondary eclipse is detected; hold for recheck "
            "rather than hard reject."
        ),
    },
    {
        "epic_id": "EPIC_211356395",
        "manual_label": "uncertain_hold",
        "manual_next_action": "promote_to_stage_f_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Visible shallow folded dip with good odd/even consistency and no secondary eclipse; "
            "OOT variability acceptable at about 0.30 of depth; primary SNR is modest and period "
            "support is fallback/ambiguous, so hold for Stage F validation rather than direct promotion."
        ),
    },
    {
        "epic_id": "EPIC_211357782",
        "manual_label": "candidate_like",
        "manual_next_action": "promote_to_stage_f_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Improved plots show a visible repeated dip; odd/even acceptable; no secondary eclipse; "
            "physically plausible depth; modest SNR and fallback-only period support require Stage F validation."
        ),
    },
    {
        "epic_id": "EPIC_211371463",
        "manual_label": "uncertain_hold",
        "manual_next_action": "run_stage_f_validation_with_oot_detrending_check",
        "manual_confidence": "medium",
        "manual_reason": (
            "Strong dip morphology and excellent odd/even consistency, but OOT variability is about "
            "1.9x the transit depth, duration is long for P=2.84d, and period source is fallback-only; "
            "hold pending better OOT/detrending validation."
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
    if value is None:
        return ""
    if pd.isna(value):
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
    if any(token in raw for token in ("candidate_like", "planet_like", "promote", "top_followup_candidate", "supported_by_calibrated_layer")):
        return "candidate"
    return "unknown"


def labels_disagree(manual_label: str, other_label: str) -> bool:
    if not manual_label or not other_label:
        return False
    return manual_label != other_label


def conflict_reasons(row: pd.Series) -> list[str]:
    manual_label = clean(row.get("manual_label"))
    checks = [
        ("autovet_label", "manual_label disagrees with autovet_label"),
        ("stage_f_label", "manual_label disagrees with stage_f_label"),
        ("stage_g_label", "manual_label disagrees with stage_g_label"),
        ("final_candidate_status", "manual_label disagrees with candidate_ledger_status"),
    ]
    reasons: list[str] = []
    for column, reason in checks:
        other = clean(row.get(column))
        if labels_disagree(manual_label, other):
            reasons.append(reason)
    return reasons


def conflict_severity(reason: str, manual_label: str, other_label: str) -> str:
    if reason == "manual_label disagrees with autovet_label":
        return "soft_conflict"
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


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
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
        "review_level",
        "autovet_only",
        "source_files",
        "last_updated",
    ]
    for column in needed:
        if column not in df.columns:
            df[column] = ""
    return df


def apply_decisions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_columns(df.copy())
    decision_df = pd.DataFrame(DECISIONS)
    decision_df["reviewed_at"] = REVIEW_DATE
    decision_df["reviewer"] = REVIEWER
    decision_df["decision_authority"] = "manual_review"
    decision_df["manual_vetted"] = "true"

    missing = sorted(set(decision_df["epic_id"]) - set(out["epic_id"]))
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
        out.loc[i, "source_files"] = append_source_file(out.loc[i, "source_files"])
        out.loc[i, "last_updated"] = REVIEW_DATE
        reasons = conflict_reasons(out.loc[i])
        out.loc[i, "has_conflict"] = str(bool(reasons)).lower()
        out.loc[i, "conflict_reason"] = "; ".join(reasons)
    return out, decision_df


def build_conflict_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    updated = df[df["epic_id"].isin([item["epic_id"] for item in DECISIONS])]
    comparison_columns = {
        "manual_label disagrees with autovet_label": "autovet_label",
        "manual_label disagrees with stage_f_label": "stage_f_label",
        "manual_label disagrees with stage_g_label": "stage_g_label",
        "manual_label disagrees with candidate_ledger_status": "final_candidate_status",
    }
    for _, row in updated.iterrows():
        reasons = [item for item in clean(row.get("conflict_reason")).split("; ") if item]
        for reason in reasons:
            comparison_column = comparison_columns[reason]
            rows.append(
                {
                    "epic_id": clean(row.get("epic_id")),
                    "conflict_type": reason,
                    "manual_label": clean(row.get("manual_label")),
                    "autovet_label": clean(row.get("autovet_label")),
                    "stage_f_label": clean(row.get("stage_f_label")),
                    "stage_g_label": clean(row.get("stage_g_label")),
                    "candidate_ledger_status": clean(row.get("final_candidate_status")),
                    "cnn_score": clean(row.get("cnn_score")),
                    "conflict_severity": conflict_severity(
                        reason,
                        clean(row.get("manual_label")),
                        clean(row.get(comparison_column)),
                    ),
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


def write_summary(before: pd.DataFrame, after: pd.DataFrame, ledger: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    updated_epics = [item["epic_id"] for item in DECISIONS]
    lines = [
        "Manual vetting update summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"review_date={REVIEW_DATE}",
        f"source_catalog={SOURCE_CATALOG.relative_to(ROOT).as_posix()}",
        f"source_backup={SOURCE_BACKUP.relative_to(ROOT).as_posix()}",
        f"manual_ledger={MANUAL_LEDGER.relative_to(ROOT).as_posix()}",
        f"updated_catalog={UPDATED_CATALOG.relative_to(ROOT).as_posix()}",
        f"updated_conflicts={UPDATED_CONFLICTS.relative_to(ROOT).as_posix()}",
        "",
        "Safety confirmations",
        "- Did not retrain.",
        "- Did not edit CNN scores.",
        "- Did not change the frozen CNN policy.",
        "- Did not overwrite original master_vetted_catalog.csv.",
        "- Did not overwrite final_candidate_master_ledger.csv.",
        "- Wrote a dated update-copy artifact and a source backup.",
        "",
        f"updated_epic_count={len(updated_epics)}",
        f"conflict_rows_for_updated_epics={len(conflicts)}",
        "",
        "Updated EPICs",
    ]
    for epic in updated_epics:
        before_row = before.loc[before["epic_id"].eq(epic)].iloc[0]
        after_row = after.loc[after["epic_id"].eq(epic)].iloc[0]
        lines.append(
            "- "
            + epic
            + f": {clean(before_row.get('master_label'))} -> {clean(after_row.get('master_label'))}; "
            + f"authority={clean(after_row.get('decision_authority'))}; "
            + f"conflict={clean(after_row.get('has_conflict'))} {clean(after_row.get('conflict_reason'))}"
        )
    lines.extend(["", "Manual ledger labels"])
    for _, row in ledger.iterrows():
        lines.append(f"- {row['epic_id']}: {row['manual_label']} ({row['manual_confidence']})")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_CATALOG.exists():
        raise FileNotFoundError(SOURCE_CATALOG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(SOURCE_CATALOG, dtype=str).fillna("")
    shutil.copy2(SOURCE_CATALOG, SOURCE_BACKUP)
    after, ledger = apply_decisions(before)
    assert_protected_unchanged(before, after)
    conflicts = build_conflict_table(after)

    ledger.to_csv(MANUAL_LEDGER, index=False)
    after.to_csv(UPDATED_CATALOG, index=False)
    conflicts.to_csv(UPDATED_CONFLICTS, index=False)
    write_summary(before, after, ledger, conflicts)
    manifest = {
        "run_id": RUN_ID,
        "review_date": REVIEW_DATE,
        "updated_epics": [item["epic_id"] for item in DECISIONS],
        "source_catalog": str(SOURCE_CATALOG),
        "source_backup": str(SOURCE_BACKUP),
        "manual_ledger": str(MANUAL_LEDGER),
        "updated_catalog": str(UPDATED_CATALOG),
        "updated_conflicts": str(UPDATED_CONFLICTS),
        "summary": str(SUMMARY_TXT),
        "did_not_retrain": True,
        "cnn_scores_unchanged": True,
        "frozen_cnn_policy_unchanged": True,
        "original_master_catalog_overwritten": False,
        "final_candidate_master_ledger_overwritten": False,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
