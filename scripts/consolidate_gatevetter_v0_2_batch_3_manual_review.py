from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DATE = "2026-06-22"
REVIEW_BATCH = "gatevetter_v0_2_batch_3_manual_review"
REVIEWER = "user_provided_manual_review"

TOP_HOLDS = ROOT / "gatevetter_v0_2_batch_3_top_20_holds.csv"
STAGE_G_QUEUE = ROOT / "gatevetter_v0_2_batch_3_stage_g_queue.csv"
REJECT_SANITY = ROOT / "gatevetter_v0_2_batch_3_reject_sanity_sample.csv"
VISUAL_MANIFEST = ROOT / "gatevetter_v0_2_batch_3_visual_manifest.csv"
DECISIONS_CSV = ROOT / "gatevetter_v0_2_batch_3_manual_review_decisions.csv"
SUMMARY_TXT = ROOT / "gatevetter_v0_2_batch_3_manual_review_summary.txt"
CNN_ERROR_LEDGER = ROOT / "cnn_manual_review_error_ledger.csv"
FINAL_CANDIDATE_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"


MANUAL_LABELS = [
    ("EPIC_211912465", "uncertain_hold_period_ambiguous"),
    ("EPIC_211384981", "reject_as_noise_or_artifact"),
    ("EPIC_211490678", "low_priority_negative"),
    ("EPIC_211504156", "variable_or_possible_eb"),
    ("EPIC_211613728", "reject_as_noise_or_artifact"),
    ("EPIC_211705654", "reject_as_noise_or_artifact"),
    ("EPIC_211322563", "reject_as_noise_or_artifact"),
    ("EPIC_211914980", "low_priority_negative"),
    ("EPIC_212029715", "low_priority_negative"),
    ("EPIC_211959779", "false_positive_eb_or_variable"),
    ("EPIC_212011476", "false_positive_eb_or_variable"),
]


DECISION_COLUMNS = [
    "review_batch",
    "reviewed_at",
    "reviewer",
    "epic_id",
    "source_packet",
    "source_rank",
    "hold_rank",
    "manual_packet_rank",
    "packet_group",
    "manual_label",
    "manual_label_family",
    "training_eligibility",
    "training_use_note",
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


def training_eligibility(label: str) -> str:
    if label.startswith("uncertain_hold"):
        return "exclude_for_now"
    return "possible_future_training_example"


def training_use_note(label: str) -> str:
    if label.startswith("uncertain_hold"):
        return "Do not use hold for training yet."
    return "Clear negative or EB/variable false positive; possible future training example."


def science_binary(label: str) -> str:
    if label.startswith("uncertain_hold"):
        return "hold_exclude_for_now"
    if label in {"variable_or_possible_eb", "false_positive_eb_or_variable"}:
        return "false_positive_eb_or_variable"
    if label == "low_priority_negative":
        return "not_candidate_like_low_priority"
    return "not_candidate_like_noise_or_artifact"


def final_recommendation(label: str) -> str:
    if label.startswith("uncertain_hold"):
        return "keep_hold_for_later_analysis_exclude_training_for_now"
    if label == "low_priority_negative":
        return "deprioritize_as_not_candidate_like"
    if label in {"variable_or_possible_eb", "false_positive_eb_or_variable"}:
        return "exclude_from_planet_candidate_queue_as_eb_or_variable"
    return "exclude_from_novel_candidate_queue"


def manifest_paths() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(VISUAL_MANIFEST):
        epic_id = clean(row.get("epic_id"))
        out[epic_id] = {
            "phase_0_folded_path": clean(row.get("plot_folded")),
            "validation_summary_json_path": clean(row.get("validation_summary_json")),
        }
    return out


def source_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv(TOP_HOLDS):
        epic_id = clean(row.get("epic_id"))
        row["source_packet"] = "top_20_holds"
        row["source_rank"] = clean(row.get("hold_rank"))
        rows[epic_id] = row
    for row in read_csv(REJECT_SANITY):
        epic_id = clean(row.get("epic_id"))
        row["source_packet"] = "reject_sanity_high_cnn_high_snr"
        row["source_rank"] = clean(row.get("manual_packet_rank"))
        rows[epic_id] = row
    return rows


def decision_rows() -> list[dict[str, str]]:
    by_epic = source_rows()
    paths_by_epic = manifest_paths()
    rows: list[dict[str, str]] = []
    for epic_id, label in MANUAL_LABELS:
        if epic_id not in by_epic:
            raise RuntimeError(f"Manual decision EPIC not found in batch-3 source packets: {epic_id}")
        source = by_epic[epic_id]
        paths = paths_by_epic.get(epic_id, {})
        reason = (
            f"Consolidated GateVetter v0.2 Batch 3 manual review label: {label}."
        )
        row = {
            "review_batch": REVIEW_BATCH,
            "reviewed_at": REVIEW_DATE,
            "reviewer": REVIEWER,
            "epic_id": epic_id,
            "source_packet": clean(source.get("source_packet")),
            "source_rank": clean(source.get("source_rank")),
            "hold_rank": clean(source.get("hold_rank")),
            "manual_packet_rank": clean(source.get("manual_packet_rank")),
            "packet_group": clean(source.get("packet_group")),
            "manual_label": label,
            "manual_label_family": manual_family(label),
            "training_eligibility": training_eligibility(label),
            "training_use_note": training_use_note(label),
            "manual_reason": reason,
            "reason_status": "complete_user_supplied",
            "manual_vetted": "true",
            "phase_0_folded_path": paths.get("phase_0_folded_path", ""),
            "validation_summary_json_path": paths.get("validation_summary_json_path", ""),
            "notes": "Consolidated Batch 3 manual review; no GateVetter v0.2 threshold changes, no CNN retraining, CNN remains morphology_score only.",
        }
        for column in DECISION_COLUMNS:
            if column in row:
                continue
            if column == "gatevetter_v0_2_prediction":
                row[column] = clean(
                    source.get("gatevetter_v0_2_prediction")
                    or source.get("gatevetter_prediction")
                )
            elif column == "gatevetter_v0_2_reason":
                row[column] = clean(
                    source.get("gatevetter_v0_2_reason")
                    or source.get("gatevetter_v0_reason")
                )
            else:
                row[column] = clean(source.get(column))
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
        if family == "hold":
            status = "manual_hold_excluded_from_training_for_now"
            error_family = "not_error_hold_case"
        elif interpretation in {"very_high_morphology_score", "moderate_high_morphology_score"}:
            status = "manual_negative_vs_cnn_positive_morphology"
            error_family = "cnn_morphology_false_positive_or_overcall"
        else:
            status = "manual_negative_with_nonhigh_cnn_score"
            error_family = "not_primary_cnn_error"
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
    label = row["manual_label"]
    recommendation = final_recommendation(label)
    return {
        "epic_id": row["epic_id"],
        "source_batch": REVIEW_BATCH,
        "best_period_days": row["best_period_days"],
        "stage_g_final_recommendation": "no_gatevetter_stage_g_action",
        "final_recommendation": recommendation,
        "stage_f_label": "",
        "stage_h_label": "",
        "visual_label": f"manual_batch_3_review_{label}",
        "final_candidate_status": label,
        "review_bin": f"manual_batch_3_{row['manual_label_family']}",
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
        "stage_g_v2_support_tier": "manual_batch_3_review_not_stage_g",
        "stage_g_v2_calibrated_score": row["gatevetter_score"],
        "stage_g_v2_annotation_only": "TRUE",
        "stage_h_status": "",
        "stage_h_training_label_v3": row["training_eligibility"],
        "stage_h_science_binary_v3": science_binary(label),
        "stage_h_ledger_status": label,
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
    stage_g_rows = count_data_rows(STAGE_G_QUEUE)
    lines = [
        "GateVetter v0.2 Batch 3 consolidated manual review",
        f"review_batch: {REVIEW_BATCH}",
        f"reviewed_at: {REVIEW_DATE}",
        f"reviewer: {REVIEWER}",
        "",
        "Batch 3 review status:",
        f"- Stage G queue rows: {stage_g_rows}",
        "- Stage G queue was empty.",
        "- 9/9 top holds reviewed.",
        "- Reject sanity reviewed; no missed candidate risk found.",
        "- Rest of reject sanity/rejects accepted as correct rejects.",
        "- GateVetter v0.2 unchanged.",
        "- CNN not retrained.",
        "- CNN remains morphology_score only.",
        "",
        "Manual label counts:",
        f"- candidate_like: {label_counts.get('candidate_like', 0)}",
        f"- uncertain_hold_positive: {label_counts.get('uncertain_hold_positive', 0)}",
        f"- uncertain_hold_period_ambiguous: {label_counts.get('uncertain_hold_period_ambiguous', 0)}",
        (
            "- variable_or_possible_eb / false_positive_eb_or_variable: "
            f"{label_counts.get('variable_or_possible_eb', 0) + label_counts.get('false_positive_eb_or_variable', 0)}"
        ),
        f"- low_priority_negative: {label_counts.get('low_priority_negative', 0)}",
        f"- reject_as_noise_or_artifact: {label_counts.get('reject_as_noise_or_artifact', 0)}",
        "",
        f"manual_label counts raw: {dict(label_counts)}",
        f"manual_label_family counts: {dict(family_counts)}",
        "",
        "Training notes:",
        "- Do not use holds for training yet.",
        "- Clear negatives and EB/variable false positives marked possible_future_training_example.",
        "- CNN remains morphology_score only.",
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
        "Explicit manual labels:",
    ]
    for row in rows:
        lines.append(
            f"- {row['epic_id']}: {row['manual_label']} "
            f"({row['source_packet']}, rank={row['source_rank']})"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = decision_rows()
    write_csv(DECISIONS_CSV, rows, DECISION_COLUMNS)
    merge_by_epic(CNN_ERROR_LEDGER, cnn_error_rows(rows), CNN_ERROR_COLUMNS)
    final_rows_appended, final_rows_refreshed = upsert_final_ledger_rows(rows)
    write_summary(rows, final_rows_appended, final_rows_refreshed)


if __name__ == "__main__":
    main()
