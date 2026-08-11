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
    / "20260519_epic5"
    / "master_vetted_catalog_manual_review_update_20260519_epic5.csv"
)
PREVIOUS_LEDGER = (
    CATALOG_DIR
    / "manual_review_updates"
    / "20260519_epic5"
    / "manual_vetting_decisions_20260519_epic5.csv"
)
STAMP = "20260519"
RUN_ID = f"{STAMP}_epic11"
OUT_DIR = CATALOG_DIR / "manual_review_updates" / RUN_ID
UPDATED_CATALOG = OUT_DIR / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
UPDATED_CONFLICTS = OUT_DIR / f"master_vetted_catalog_conflicts_manual_review_update_{RUN_ID}.csv"
MANUAL_LEDGER_INCREMENT = OUT_DIR / f"manual_vetting_decisions_increment_{RUN_ID}.csv"
MANUAL_LEDGER_CUMULATIVE = OUT_DIR / f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
SUMMARY_TXT = OUT_DIR / f"manual_vetting_update_summary_{RUN_ID}.txt"
MANIFEST_JSON = OUT_DIR / f"manual_vetting_update_manifest_{RUN_ID}.json"
SOURCE_BACKUP = OUT_DIR / f"source_catalog_backup_before_manual_review_update_{RUN_ID}.csv"

REVIEW_DATE = date(2026, 5, 19).isoformat()
REVIEWER = "user_provided_manual_review"
LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)

DECISIONS = [
    {
        "epic_id": "EPIC_211409713",
        "manual_label": "uncertain_hold",
        "manual_next_action": "promote_to_stage_f_period_validation",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Strong and clean folded transit morphology with high primary SNR, plausible depth, short duration, "
            "low OOT variability relative to depth, and no significant secondary eclipse. However odd/even is "
            "unavailable and alias risk is moderate, with stronger support at the double-period alias, so keep "
            "as uncertain hold pending Stage F period/odd-even validation rather than direct candidate promotion."
        ),
    },
    {
        "epic_id": "EPIC_211418290",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_detrending_and_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Transit-like signal is present with strong primary SNR and acceptable odd/even consistency, with low "
            "alias risk and no detected secondary eclipse. However the OOT variability is high at about 0.76x "
            "transit depth, transit duration is very long at about 12.0 hours for P=4.95d, CNN morphology score "
            "is only moderate, and the period source is event-spacing fallback with many candidate periods. Keep "
            "as uncertain hold pending improved detrending and period validation rather than promote."
        ),
    },
    {
        "epic_id": "EPIC_211424769",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Strong odd/even depth mismatch with ratio about 0.343 and explicit delta about 0.82, making the "
            "signal EB-like rather than planet-like. Primary SNR is only modest, CNN morphology score is moderate, "
            "and the period source is event-spacing fallback with many candidate periods, so reject from candidate "
            "pipeline despite low OOT variability and no meaningful positive secondary eclipse."
        ),
    },
    {
        "epic_id": "EPIC_211429653",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Stage F already rejects the signal because the primary transit depth is not significant in the cached "
            "folded light curve. Primary SNR is weak at about 3.25, the event count is low, and the period is very "
            "short at P=0.507d with a long apparent duration of about 5.88 hours. Odd/even is also severely "
            "inconsistent with ratio about 0.247 and explicit delta about 1.47, so the morphology is not reliable "
            "for planet promotion despite CNN morphology-positive scoring."
        ),
    },
    {
        "epic_id": "EPIC_211432321",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Weak/shallow signal with modest primary SNR around 7.52, very short period at P=0.776d, and "
            "implausibly long transit duration around 6.37 hours. Period support is unreliable from event-spacing "
            "fallback with many candidate periods, and alias risk is moderate with stronger support near the "
            "double-period alias. Odd/even is not catastrophically mismatched but still imperfect, so reject as "
            "noise/artifact rather than promote."
        ),
    },
    {
        "epic_id": "EPIC_211482556",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_stage_f_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Shallow but plausible transit-like signal with good odd/even consistency, ratio about 0.978, no "
            "detected secondary eclipse, low alias risk, and moderate primary SNR around 9.31. However the folded "
            "dip appears somewhat wobbly/noisy, OOT variability is non-trivial at about 0.27x transit depth, "
            "transit duration is fairly long at about 7.36 hours, CNN morphology score is only moderate, and the "
            "period source is event-spacing fallback with many candidate periods. Keep as uncertain hold pending "
            "Stage F period validation and cleaner detrended plot review rather than promote directly."
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
        out.loc[i, "source_files"] = append_source_file(out.loc[i, "source_files"])
        out.loc[i, "last_updated"] = REVIEW_DATE
        reasons = conflict_reasons(out.loc[i])
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
    updated = df[df["epic_id"].isin([item["epic_id"] for item in DECISIONS])]
    for _, row in updated.iterrows():
        reasons = [item for item in clean(row.get("conflict_reason")).split("; ") if item]
        for reason in reasons:
            other_col = comparison_columns[reason]
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
                    "conflict_severity": conflict_severity(reason, clean(row.get("manual_label")), clean(row.get(other_col))),
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


def write_summary(before: pd.DataFrame, after: pd.DataFrame, increment: pd.DataFrame, cumulative: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    updated_epics = [item["epic_id"] for item in DECISIONS]
    lines = [
        "Manual vetting append/update-copy summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"review_date={REVIEW_DATE}",
        f"source_catalog={SOURCE_CATALOG.relative_to(ROOT).as_posix()}",
        f"source_backup={SOURCE_BACKUP.relative_to(ROOT).as_posix()}",
        f"increment_ledger={MANUAL_LEDGER_INCREMENT.relative_to(ROOT).as_posix()}",
        f"cumulative_ledger={MANUAL_LEDGER_CUMULATIVE.relative_to(ROOT).as_posix()}",
        f"updated_catalog={UPDATED_CATALOG.relative_to(ROOT).as_posix()}",
        f"updated_conflicts={UPDATED_CONFLICTS.relative_to(ROOT).as_posix()}",
        "",
        "Safety confirmations",
        "- Did not retrain.",
        "- Did not edit CNN scores.",
        "- Did not change the frozen CNN policy.",
        "- Did not overwrite original master_vetted_catalog.csv.",
        "- Did not overwrite final_candidate_master_ledger.csv.",
        "- Wrote append/update-copy artifacts and a source backup.",
        "",
        f"new_decision_count={len(increment)}",
        f"cumulative_decision_count={len(cumulative)}",
        f"conflict_rows_for_new_epics={len(conflicts)}",
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
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_CATALOG.exists():
        raise FileNotFoundError(SOURCE_CATALOG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(SOURCE_CATALOG, dtype=str).fillna("")
    shutil.copy2(SOURCE_CATALOG, SOURCE_BACKUP)
    after, increment = apply_decisions(before)
    assert_protected_unchanged(before, after)
    cumulative = write_ledgers(increment)
    conflicts = build_conflict_table(after)

    after.to_csv(UPDATED_CATALOG, index=False)
    conflicts.to_csv(UPDATED_CONFLICTS, index=False)
    write_summary(before, after, increment, cumulative, conflicts)
    manifest = {
        "run_id": RUN_ID,
        "review_date": REVIEW_DATE,
        "new_updated_epics": [item["epic_id"] for item in DECISIONS],
        "source_catalog": str(SOURCE_CATALOG),
        "source_backup": str(SOURCE_BACKUP),
        "increment_ledger": str(MANUAL_LEDGER_INCREMENT),
        "cumulative_ledger": str(MANUAL_LEDGER_CUMULATIVE),
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
