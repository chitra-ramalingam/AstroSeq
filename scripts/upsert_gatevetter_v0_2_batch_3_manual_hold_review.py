from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DATE = "2026-06-22"
REVIEW_BATCH = "gatevetter_v0_2_batch_3_top_hold_review"
REVIEW_SOURCE_QUEUE = "top_20_holds"
REVIEWER = "user_provided_manual_review"

HOLD_QUEUE = ROOT / "gatevetter_v0_2_batch_3_top_20_holds.csv"
STAGE_G_QUEUE = ROOT / "gatevetter_v0_2_batch_3_stage_g_queue.csv"
VISUAL_MANIFEST = ROOT / "gatevetter_v0_2_batch_3_visual_manifest.csv"
DECISIONS_CSV = ROOT / "gatevetter_v0_2_batch_3_manual_hold_review_decisions.csv"
SUMMARY_TXT = ROOT / "gatevetter_v0_2_batch_3_manual_hold_review_summary.txt"
CNN_ERROR_LEDGER = ROOT / "cnn_manual_review_error_ledger.csv"
FINAL_CANDIDATE_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"


DECISIONS = [
    ("EPIC_211912465", "uncertain_hold_period_ambiguous"),
    ("EPIC_211384981", "reject_as_noise_or_artifact"),
    ("EPIC_211490678", "low_priority_negative"),
    ("EPIC_211504156", "variable_or_possible_eb"),
    ("EPIC_211613728", "reject_as_noise_or_artifact"),
    ("EPIC_211705654", "reject_as_noise_or_artifact"),
    ("EPIC_211322563", "reject_as_noise_or_artifact"),
    ("EPIC_211914980", "low_priority_negative"),
    ("EPIC_212029715", "low_priority_negative"),
]


DECISION_COLUMNS = [
    "review_batch",
    "review_source_queue",
    "reviewed_at",
    "reviewer",
    "epic_id",
    "hold_rank",
    "manual_label",
    "manual_label_family",
    "training_eligibility",
    "manual_reason",
    "reason_status",
    "manual_vetted",
    "gatevetter_v0_2_prediction",
    "gatevetter_v0_2_reason",
    "gatevetter_score",
    "candidate_survivor_score",
    "cnn_score",
    "primary_depth",
    "primary_depth_snr",
    "odd_even_depth_ratio",
    "secondary_depth_snr",
    "secondary_to_primary_depth_ratio",
    "oot_to_depth",
    "candidate_period_count",
    "event_family_count",
    "alias_risk",
    "fallback_period_flag",
    "duration_fraction_of_period",
    "best_period_days",
    "transit_duration_hours",
    "period_source",
    "period_ambiguity_flag",
    "period_comparison_status",
    "trusted_period_validation",
    "metric_trust_level",
    "stage_g_action",
    "primary_gate",
    "hard_gate_fired",
    "hard_gates_fired",
    "hold_gates_fired",
    "risk_gates_fired",
    "penalties_or_missing_evidence",
    "rule_trace",
    "score_components",
    "phase_0_folded_path",
    "validation_summary_json_path",
    "notes",
]


CNN_ERROR_COLUMNS = [
    "review_batch",
    "reviewed_at",
    "epic_id",
    "manual_label",
    "manual_label_family",
    "training_eligibility",
    "cnn_score",
    "cnn_interpretation",
    "manual_vs_cnn_status",
    "error_family",
    "candidate_survivor_score",
    "gatevetter_score",
    "primary_depth_snr",
    "odd_even_depth_ratio",
    "oot_to_depth",
    "candidate_period_count",
    "alias_risk",
    "trusted_period_validation",
    "manual_reason",
    "reason_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def manual_family(label: str) -> str:
    return "hold" if label.startswith("uncertain_hold") else "negative"


def manual_reason(label: str) -> str:
    return f"User-supplied batch-3 top-hold manual review label: {label}."


def science_binary(row: dict[str, str]) -> str:
    label = row["manual_label"]
    if row["manual_label_family"] == "hold":
        return "hold_exclude_for_now"
    if label == "variable_or_possible_eb":
        return "false_positive_eb_or_variable"
    if label == "low_priority_negative":
        return "not_candidate_like_low_priority"
    return "not_candidate_like_noise_or_artifact"


def final_recommendation(row: dict[str, str]) -> str:
    label = row["manual_label"]
    if row["manual_label_family"] == "hold":
        return "keep_hold_for_later_analysis_exclude_training_for_now"
    if label == "low_priority_negative":
        return "deprioritize_as_not_candidate_like"
    if label == "variable_or_possible_eb":
        return "exclude_from_planet_candidate_queue_as_eb_or_variable"
    return "exclude_from_novel_candidate_queue"


def manifest_paths() -> dict[str, dict[str, str]]:
    if not VISUAL_MANIFEST.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(VISUAL_MANIFEST):
        if clean(row.get("packet")) != "top_20_holds":
            continue
        out[clean(row.get("epic_id"))] = {
            "phase_0_folded_path": clean(row.get("plot_folded")),
            "validation_summary_json_path": clean(row.get("validation_summary_json")),
        }
    return out


def decision_rows(
    hold_by_epic: dict[str, dict[str, str]],
    manifest_by_epic: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for epic_id, label in DECISIONS:
        hold = hold_by_epic.get(epic_id, {})
        paths = manifest_by_epic.get(epic_id, {})
        family = manual_family(label)
        row = {
            "review_batch": REVIEW_BATCH,
            "review_source_queue": REVIEW_SOURCE_QUEUE,
            "reviewed_at": REVIEW_DATE,
            "reviewer": REVIEWER,
            "epic_id": epic_id,
            "hold_rank": clean(hold.get("hold_rank")),
            "manual_label": label,
            "manual_label_family": family,
            "training_eligibility": "exclude_for_now",
            "manual_reason": manual_reason(label),
            "reason_status": "complete_user_supplied",
            "manual_vetted": "true",
            "phase_0_folded_path": paths.get("phase_0_folded_path", ""),
            "validation_summary_json_path": paths.get("validation_summary_json_path", ""),
            "notes": "Manual batch-3 hold decision is ledger-only for now; no CNN retraining and no GateVetter v0.2 threshold update.",
        }
        for column in DECISION_COLUMNS:
            if column not in row:
                row[column] = clean(hold.get(column))
        rows.append(row)
    return rows


def merge_by_epic(path: Path, new_rows: list[dict[str, str]], columns: list[str]) -> None:
    existing = read_csv(path) if path.exists() else []
    keyed: dict[tuple[str, str], dict[str, str]] = {}
    for row in existing:
        keyed[(clean(row.get("review_batch")), clean(row.get("epic_id")))] = row
    for row in new_rows:
        keyed[(clean(row.get("review_batch")), clean(row.get("epic_id")))] = row
    write_csv(path, list(keyed.values()), columns)


def cnn_interpretation(score: str) -> str:
    try:
        value = float(score)
    except ValueError:
        return "cnn_score_missing"
    if value >= 0.9:
        return "very_high_morphology_score"
    if value >= 0.65:
        return "moderate_high_morphology_score"
    return "low_to_moderate_morphology_score"


def cnn_error_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        family = row["manual_label_family"]
        interpretation = cnn_interpretation(row["cnn_score"])
        if family == "negative" and interpretation in {
            "very_high_morphology_score",
            "moderate_high_morphology_score",
        }:
            status = "manual_negative_vs_cnn_positive_morphology"
            error_family = "cnn_morphology_false_positive_or_overcall"
        elif family == "negative":
            status = "manual_negative_with_nonhigh_cnn_score"
            error_family = "not_primary_cnn_error"
        else:
            status = "manual_hold_excluded_from_training_for_now"
            error_family = "not_error_hold_case"
        out.append(
            {
                "review_batch": REVIEW_BATCH,
                "reviewed_at": REVIEW_DATE,
                "epic_id": row["epic_id"],
                "manual_label": row["manual_label"],
                "manual_label_family": family,
                "training_eligibility": row["training_eligibility"],
                "cnn_score": row["cnn_score"],
                "cnn_interpretation": interpretation,
                "manual_vs_cnn_status": status,
                "error_family": error_family,
                "candidate_survivor_score": row["candidate_survivor_score"],
                "gatevetter_score": row["gatevetter_score"],
                "primary_depth_snr": row["primary_depth_snr"],
                "odd_even_depth_ratio": row["odd_even_depth_ratio"],
                "oot_to_depth": row["oot_to_depth"],
                "candidate_period_count": row["candidate_period_count"],
                "alias_risk": row["alias_risk"],
                "trusted_period_validation": row["trusted_period_validation"],
                "manual_reason": row["manual_reason"],
                "reason_status": row["reason_status"],
            }
        )
    return out


def final_ledger_row(row: dict[str, str]) -> dict[str, str]:
    recommendation = final_recommendation(row)
    return {
        "epic_id": row["epic_id"],
        "source_batch": REVIEW_BATCH,
        "best_period_days": row["best_period_days"],
        "stage_g_final_recommendation": "no_gatevetter_stage_g_action",
        "final_recommendation": recommendation,
        "stage_f_label": "",
        "stage_h_label": "",
        "visual_label": f"manual_top_hold_review_{row['manual_label']}",
        "final_candidate_status": row["manual_label"],
        "review_bin": f"manual_{REVIEW_SOURCE_QUEUE}_{row['manual_label_family']}",
        "status_reason": row["manual_reason"],
        "visual_notes": row["manual_reason"],
        "reviewer": REVIEWER,
        "reviewed_at": REVIEW_DATE,
        "recommended_next_action": recommendation,
        "stage_h_notes": "",
        "phase_0_folded_path": row["phase_0_folded_path"],
        "validation_summary_json_path": row["validation_summary_json_path"],
        "stage_f_closed_45_approved_ledger_status": "",
        "stage_f_closed_45_ledger_priority": "",
        "stage_f_closed_45_evidence_summary": "",
        "stage_f_closed_45_caveats": "",
        "stage_f_closed_45_ledger_change_type": "",
        "stage_g_v2_support_tier": "manual_batch_3_top_hold_review_not_stage_g",
        "stage_g_v2_calibrated_score": row["gatevetter_score"],
        "stage_g_v2_annotation_only": "TRUE",
        "stage_h_status": "",
        "stage_h_training_label_v3": row["training_eligibility"],
        "stage_h_science_binary_v3": science_binary(row),
        "stage_h_ledger_status": row["manual_label"],
        "stage_h_ledger_priority": row["manual_label_family"],
        "stage_h_reason": row["manual_reason"],
        "stage_h_reviewed_at": REVIEW_DATE,
        "stage_h_reviewer": REVIEWER,
        "stage_h_previous_final_candidate_status": "",
        "stage_h_previous_final_recommendation": "",
        "stage_h_previous_training_label_v3": "",
        "stage_h_previous_science_binary_v3": "",
    }


def upsert_final_ledger_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    if not FINAL_CANDIDATE_LEDGER.exists():
        return 0, 0
    with FINAL_CANDIDATE_LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        existing = list(reader)

    new_by_epic = {row["epic_id"]: final_ledger_row(row) for row in rows}
    updated = 0
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in existing:
        epic_id = clean(row.get("epic_id"))
        source_batch = clean(row.get("source_batch"))
        if epic_id in new_by_epic and source_batch == REVIEW_BATCH:
            out.append(new_by_epic[epic_id])
            seen.add(epic_id)
            updated += 1
        else:
            out.append(row)

    missing = [new_by_epic[epic_id] for epic_id in new_by_epic if epic_id not in seen]
    out.extend(missing)
    with FINAL_CANDIDATE_LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out)
    return len(missing), updated


def write_summary(rows: list[dict[str, str]], final_rows_appended: int, final_rows_refreshed: int) -> None:
    label_counts = Counter(row["manual_label"] for row in rows)
    family_counts = Counter(row["manual_label_family"] for row in rows)
    high_cnn_negative = [
        row["epic_id"]
        for row in rows
        if row["manual_label_family"] == "negative"
        and cnn_interpretation(row["cnn_score"])
        in {"very_high_morphology_score", "moderate_high_morphology_score"}
    ]
    lines = [
        "GateVetter v0.2 batch-3 manual top-hold review summary",
        f"review_batch: {REVIEW_BATCH}",
        f"reviewed_at: {REVIEW_DATE}",
        f"reviewer: {REVIEWER}",
        "",
        f"Stage G queue data rows: {count_data_rows(STAGE_G_QUEUE)}",
        f"Manual decisions supplied and logged: {len(rows)}",
        "Input note: 9 reviewed top-hold objects are now ledgered.",
        "Action: no CNN retraining performed; GateVetter v0.2 thresholds unchanged.",
        "",
        f"manual_label counts: {dict(label_counts)}",
        f"manual_label_family counts: {dict(family_counts)}",
        f"high/moderate-high CNN morphology negatives logged: {high_cnn_negative}",
        "",
        "Files updated/created:",
        f"- {DECISIONS_CSV.relative_to(ROOT).as_posix()}",
        f"- {SUMMARY_TXT.relative_to(ROOT).as_posix()}",
        f"- {CNN_ERROR_LEDGER.relative_to(ROOT).as_posix()}",
        (
            f"- {FINAL_CANDIDATE_LEDGER.relative_to(ROOT).as_posix()} "
            f"appended rows: {final_rows_appended}; refreshed review-batch rows: {final_rows_refreshed}"
        ),
        "",
        "Reviewed hold ranks:",
    ]
    for row in rows:
        lines.append(
            f"- {row['epic_id']}: hold_rank={row['hold_rank']}; "
            f"manual_label={row['manual_label']}; reason_status={row['reason_status']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hold_by_epic = {row["epic_id"]: row for row in read_csv(HOLD_QUEUE)}
    missing = [epic_id for epic_id, _ in DECISIONS if epic_id not in hold_by_epic]
    if missing:
        raise RuntimeError(f"Manual decision EPICs not found in batch-3 holds: {missing}")

    rows = decision_rows(hold_by_epic, manifest_paths())
    merge_by_epic(DECISIONS_CSV, rows, DECISION_COLUMNS)
    merge_by_epic(CNN_ERROR_LEDGER, cnn_error_rows(rows), CNN_ERROR_COLUMNS)
    final_rows_appended, final_rows_refreshed = upsert_final_ledger_rows(rows)
    write_summary(rows, final_rows_appended, final_rows_refreshed)


if __name__ == "__main__":
    main()
