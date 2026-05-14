from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_OUT = ROOT / "k2_stage_f_v3_manual_review_outcomes.csv"
MANUAL_REVIEWED_OUT = ROOT / "k2_stage_f_v3_manual_reviewed.csv"
SUMMARY_OUT = ROOT / "k2_stage_f_v3_manual_update_summary.json"
NEXT_NEEDS = ROOT / "next_needs_stage_f_validation.csv"

DECISIONS = [
    {
        "epic_id": "EPIC_211746706",
        "manual_stage_f_label": "reject_as_noise_or_artifact",
        "manual_science_binary": "not_science_like",
        "training_label_v3": "noise_or_artifact",
        "manual_reason": "odd/even depth mismatch; high OOT variability; noisy folded morphology; alias ambiguity",
    },
    {
        "epic_id": "EPIC_211598703",
        "manual_stage_f_label": "reject_as_noise_or_artifact",
        "manual_science_binary": "not_science_like",
        "training_label_v3": "noise_or_artifact",
        "manual_reason": "catastrophic odd/even mismatch; weak primary SNR; suspicious short duration; noisy folded morphology; alias ambiguity",
    },
    {
        "epic_id": "EPIC_211537297",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "acceptable odd/even; no secondary; low OOT variability; modest depth/SNR; non-unique alias support",
    },
    {
        "epic_id": "EPIC_211404127",
        "manual_stage_f_label": "promote_to_deeper_eval",
        "manual_science_binary": "science_like",
        "training_label_v3": "candidate_like",
        "manual_reason": "strong deep primary; high SNR; acceptable odd/even; no phase-0.5 secondary; strong event support; caveat high OOT and P/2 alias support",
    },
    {
        "epic_id": "EPIC_211759361",
        "manual_stage_f_label": "promote_to_deeper_eval",
        "manual_science_binary": "science_like",
        "training_label_v3": "candidate_like",
        "manual_reason": "good odd/even agreement, no phase-0.5 secondary, visible coherent primary, adequate event support; caveat moderate SNR, high OOT variability, and 2P/3 alias support",
    },
    {
        "epic_id": "EPIC_211423229",
        "manual_stage_f_label": "reject_as_noise_or_artifact",
        "manual_science_binary": "not_science_like",
        "training_label_v3": "noise_or_artifact",
        "manual_reason": "poor odd/even agreement; weak tiny primary; irregular non-transit morphology; weak/non-dominant alias support; low event-family support",
    },
    {
        "epic_id": "EPIC_211562803",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "shallow/visually weak primary; only moderate SNR; borderline odd/even agreement; non-dominant alias support; messy low-amplitude folded morphology",
    },
    {
        "epic_id": "EPIC_211631955",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "unclear primary transit; weak depth/SNR; poor odd/even agreement; high OOT variability; non-dominant alias support; noisy folded morphology",
    },
    {
        "epic_id": "EPIC_211791780",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "unclear primary transit; shallow depth/moderate SNR; high OOT variability; borderline odd/even agreement; P/P2 alias ambiguity; noisy folded morphology",
    },
    {
        "epic_id": "EPIC_211821200",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "not visually convincing; borderline odd/even agreement; moderate OOT variability; non-dominant alias support; folded morphology not clean enough for promotion",
    },
    {
        "epic_id": "EPIC_211306307",
        "manual_stage_f_label": "reject_as_noise_or_artifact",
        "manual_science_binary": "not_science_like",
        "training_label_v3": "noise_or_artifact",
        "manual_reason": "severe odd/even depth mismatch; weak/moderate primary SNR; inconsistent parity morphology; not convincing as a clean planet-like signal",
        "reviewed_at": "2026-05-05",
    },
    {
        "epic_id": "EPIC_211409604",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "some candidate-like signal but uncertain; borderline/poor odd-even agreement; high OOT variability; non-dominant alias support; not confident enough to promote",
        "reviewed_at": "2026-05-05",
    },
    {
        "epic_id": "EPIC_211770415",
        "manual_stage_f_label": "hold_deeper_eval",
        "manual_science_binary": "unresolved",
        "training_label_v3": "uncertain_hold",
        "manual_reason": "not visually clear; two-spike/weak-depth morphology; shallow primary; possible secondary-like structure by eye; non-dominant alias support; not confident enough to promote",
        "reviewed_at": "2026-05-05",
    },
]

COLUMNS = [
    "epic_id",
    "manual_stage_f_label",
    "manual_science_binary",
    "training_label_v3",
    "manual_reason",
    "source_batch",
    "source_review_csv",
    "best_period_days",
    "stage_f_label",
    "stage_f_reason",
    "reviewer",
    "reviewed_at",
    "phase_0_folded_path",
    "phase_05_secondary_check_path",
    "alias_period_comparison_path",
    "odd_even_zoom_path",
    "validation_summary_json_path",
]

REVIEW_COLUMNS = [
    "manual_review_label",
    "manual_review_status",
    "visual_notes",
    "reviewer",
    "reviewed_at",
]

DEFAULT_REVIEWER = "OC"
DEFAULT_REVIEWED_AT = "2026-05-04"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_source_row(epic_id: str) -> tuple[pd.Series | None, Path | None]:
    for path in sorted(ROOT.glob("k2_stage_e_v3_recovery_batch*_visual_review_sheet.csv")):
        df = read_csv(path)
        if len(df) == 0 or "epic_id" not in df.columns:
            continue
        hit = df.loc[df["epic_id"].astype(str).eq(epic_id)]
        if len(hit) > 0:
            return hit.iloc[0], path
    return None, None


def rel(path_value: Any) -> str:
    text = "" if pd.isna(path_value) else str(path_value)
    root = str(ROOT)
    return text.replace(root + "\\", "").replace(root + "/", "")


def upsert_manual_outcomes() -> dict[str, Any]:
    manual = read_csv(MANUAL_OUT)
    if len(manual) == 0:
        manual = pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in manual.columns:
            manual[col] = ""
    before_epics = set(manual["epic_id"].astype(str)) if len(manual) else set()
    added: list[str] = []
    updated: list[str] = []
    rows = manual.to_dict("records")
    by_epic = {str(row.get("epic_id", "")): i for i, row in enumerate(rows)}

    for decision in DECISIONS:
        epic_id = decision["epic_id"]
        source, source_path = find_source_row(epic_id)
        row = {col: "" for col in COLUMNS}
        if source is not None:
            for col in COLUMNS:
                if col in source.index:
                    row[col] = source.get(col, "")
            for path_col in [
                "phase_0_folded_path",
                "phase_05_secondary_check_path",
                "alias_period_comparison_path",
                "odd_even_zoom_path",
                "validation_summary_json_path",
            ]:
                row[path_col] = rel(row.get(path_col, ""))
            row["source_review_csv"] = source_path.name if source_path else ""
            if source_path:
                row["source_batch"] = source_path.stem.replace("k2_stage_e_v3_recovery_", "").replace("_visual_review_sheet", "")
        row.update(decision)
        row["reviewer"] = decision.get("reviewer", DEFAULT_REVIEWER)
        row["reviewed_at"] = decision.get("reviewed_at", DEFAULT_REVIEWED_AT)
        if epic_id in by_epic:
            rows[by_epic[epic_id]].update(row)
            updated.append(epic_id)
        else:
            rows.append(row)
            by_epic[epic_id] = len(rows) - 1
            added.append(epic_id)

    out = pd.DataFrame(rows)
    out = out[COLUMNS]
    out = out.sort_values(["source_batch", "epic_id"], na_position="last").reset_index(drop=True)
    out.to_csv(MANUAL_OUT, index=False)
    return {
        "manual_rows_before": len(before_epics),
        "manual_rows_after": len(out),
        "rows_added": added,
        "rows_updated": updated,
    }


def update_visual_review_sheets() -> list[str]:
    touched: list[str] = []
    decisions = {d["epic_id"]: d for d in DECISIONS}
    for path in sorted(ROOT.glob("k2_stage_e_v3_recovery_batch*_visual_review_sheet.csv")):
        df = read_csv(path)
        if len(df) == 0 or "epic_id" not in df.columns:
            continue
        for col in REVIEW_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object").fillna("")
        changed = False
        for idx, row in df.iterrows():
            epic_id = str(row.get("epic_id", ""))
            if epic_id not in decisions:
                continue
            decision = decisions[epic_id]
            df.at[idx, "manual_review_label"] = decision["manual_stage_f_label"]
            df.at[idx, "manual_review_status"] = decision["manual_science_binary"]
            df.at[idx, "visual_notes"] = decision["manual_reason"]
            df.at[idx, "reviewer"] = decision.get("reviewer", DEFAULT_REVIEWER)
            df.at[idx, "reviewed_at"] = decision.get("reviewed_at", DEFAULT_REVIEWED_AT)
            changed = True
        if changed:
            df.to_csv(path, index=False)
            touched.append(path.name)
    return touched


def write_reviewed_file() -> int:
    manual = read_csv(MANUAL_OUT)
    frames: list[pd.DataFrame] = []
    for path in sorted(ROOT.glob("k2_stage_e_v3_recovery_batch*_visual_review_sheet.csv")):
        df = read_csv(path)
        if len(df) == 0 or "epic_id" not in df.columns:
            continue
        df["source_review_csv"] = path.name
        frames.append(df)
    if not frames or len(manual) == 0:
        pd.DataFrame().to_csv(MANUAL_REVIEWED_OUT, index=False)
        return 0
    reviewed = pd.concat(frames, ignore_index=True)
    reviewed = reviewed.loc[reviewed["epic_id"].astype(str).isin(set(manual["epic_id"].astype(str)))].copy()
    reviewed = reviewed.merge(
        manual[["epic_id", "manual_stage_f_label", "manual_science_binary", "manual_reason"]],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    reviewed.to_csv(MANUAL_REVIEWED_OUT, index=False)
    return len(reviewed)


def main() -> None:
    queue_before = read_csv(NEXT_NEEDS)
    queue_epics_before = set(queue_before["epic_id"].astype(str)) if len(queue_before) and "epic_id" in queue_before.columns else set()
    summary = upsert_manual_outcomes()
    touched = update_visual_review_sheets()
    reviewed_rows = write_reviewed_file()
    decision_epics = {d["epic_id"] for d in DECISIONS}
    removed_candidates = sorted(decision_epics & queue_epics_before)
    manual = read_csv(MANUAL_OUT)
    label_counts = manual["manual_stage_f_label"].value_counts().to_dict() if len(manual) else {}
    science_counts = manual["manual_science_binary"].value_counts().to_dict() if len(manual) else {}
    summary.update(
        {
            "review_sheets_touched": touched,
            "reviewed_rows": reviewed_rows,
            "epics_present_in_validation_queue_before_rebuild": removed_candidates,
            "manual_stage_f_label_counts": label_counts,
            "manual_science_binary_counts": science_counts,
        }
    )
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
