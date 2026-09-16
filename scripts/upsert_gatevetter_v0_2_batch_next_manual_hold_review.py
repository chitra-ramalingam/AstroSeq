from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DATE = "2026-06-16"
REVIEW_BATCH = "gatevetter_v0_2_batch_next_top_hold_review"
REVIEW_SOURCE_QUEUE = "top_20_holds"
REVIEWER = "user_provided_manual_review"

HOLD_QUEUE = ROOT / "gatevetter_v0_2_batch_next_hold_queue.csv"
STAGE_G_QUEUE = ROOT / "gatevetter_v0_2_batch_next_stage_g_queue.csv"
DECISIONS_CSV = ROOT / "gatevetter_v0_2_batch_next_manual_hold_review_decisions.csv"
SUMMARY_TXT = ROOT / "gatevetter_v0_2_batch_next_manual_hold_review_summary.txt"
CNN_ERROR_LEDGER = ROOT / "cnn_manual_review_error_ledger.csv"
FINAL_CANDIDATE_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"


DECISIONS = [
    {
        "epic_id": "EPIC_211432946",
        "manual_label": "low_priority_negative",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period and high CNN morphology score with low alias risk and no clear secondary, but "
            "primary SNR is modest, odd/even ratio is borderline-poor at 0.727, OOT variability is high relative to "
            "depth at 0.783, and candidate_period_count is very large. Not candidate_like."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211504682",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period and low alias risk, but the signal is extremely weak with primary_depth_snr=1.12 "
            "and primary_depth only 5.56e-05. OOT variability is high relative to depth at 0.918 and "
            "candidate_period_count is very large. Odd/even is acceptable but cannot rescue a near-non-detection."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211518347",
        "manual_label": "low_priority_negative",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period, low alias risk, good odd/even ratio, and high CNN morphology score, but the "
            "transit evidence is weak. Primary_depth_snr is only 2.47, event support is low at 5, and OOT variability "
            "is high relative to depth at 0.904. Secondary metric is not a clean EB detection, but the phase/secondary "
            "region is not reassuring. Not candidate_like."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211592536",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Period is ambiguous and untrusted, duration fraction is too large at 0.201, primary_depth_snr is only "
            "0.56, odd/even and OOT diagnostics are unavailable due to period ambiguity, candidate_period_count is "
            "high at 970, and no convincing transit-like signal is visible."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211707161",
        "manual_label": "uncertain_hold_positive",
        "manual_label_family": "hold",
        "training_eligibility": "exclude_for_now",
        "manual_reason": (
            "Promising hold. Trusted refreshed period with low alias risk, good odd/even ratio, very low OOT/d"
        ),
        "reason_status": "incomplete_user_supplied_truncated",
    },
    {
        "epic_id": "EPIC_211781289",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period and acceptable odd/even/secondary metrics, but the primary signal is extremely "
            "weak with primary_depth_snr=1.15. OOT variability exceeds the transit depth at oot_to_depth=1.13, event "
            "support is low, and CNN morphology score is not enough to rescue a near-non-detection."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211792360",
        "manual_label": "low_priority_negative",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period with excellent odd/even agreement, no measured secondary, low alias risk, and "
            "good event-family support. However, primary SNR is only modest, duration is long-ish, "
            "candidate_period_count is very high, and OOT variability exceeds the transit depth at oot_to_depth=1.077. "
            "Not candidate_like."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211892055",
        "manual_label": "low_priority_negative",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period with good odd/even agreement and no measured secondary, but the primary signal "
            "is extremely weak with primary_depth_snr=1.06. OOT variability is high relative to transit depth at "
            "oot_to_depth=0.920, alias risk is moderate, event support is limited, and candidate_period_count remains "
            "high. Not candidate_like."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211894518",
        "manual_label": "low_priority_negative",
        "manual_label_family": "negative",
        "training_eligibility": "possible_negative",
        "manual_reason": (
            "Trusted refreshed period with acceptable odd/even ratio and moderate/high CNN morphology score, but the "
            "signal is weak with primary_depth_snr=1.86. OOT variability exceeds 50% of transit depth, duration "
            "fraction is high at 0.15 for a short 1.29-day period, and event support is limited. Not candidate_like."
        ),
        "reason_status": "complete_user_supplied",
    },
    {
        "epic_id": "EPIC_211939094",
        "manual_label": "uncertain_hold_period_ambiguous",
        "manual_label_family": "hold",
        "training_eligibility": "exclude_for_now",
        "manual_reason": (
            "Clear dip-like morphology and high CNN score, with moderate primary SNR and reasonable duration fraction, "
            "but period validation is untrusted/ambiguous. Odd/even and OOT diagnostics are unavailable, alias risk is "
            "period_ambiguous, and candidate_period_count is high. Do not promote to candidate_like until period is "
            "confirmed and odd/even/OOT diagnostics can be trusted."
        ),
        "reason_status": "complete_user_supplied",
    },
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


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def find_hold_rows() -> dict[str, dict[str, str]]:
    rows = read_csv(HOLD_QUEUE)
    return {row["epic_id"]: row for row in rows}


def decision_rows(hold_by_epic: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for decision in DECISIONS:
        hold = hold_by_epic.get(decision["epic_id"], {})
        row = {
            "review_batch": REVIEW_BATCH,
            "review_source_queue": REVIEW_SOURCE_QUEUE,
            "reviewed_at": REVIEW_DATE,
            "reviewer": REVIEWER,
            "epic_id": decision["epic_id"],
            "hold_rank": clean(hold.get("hold_rank")),
            "manual_label": decision["manual_label"],
            "manual_label_family": decision["manual_label_family"],
            "training_eligibility": decision["training_eligibility"],
            "manual_reason": decision["manual_reason"],
            "reason_status": decision["reason_status"],
            "manual_vetted": "true",
            "notes": "Manual decisions are for ledgering, later analysis, and future training data; no threshold update.",
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
        elif family == "hold":
            status = "manual_hold_excluded_from_training_for_now"
            error_family = "not_error_hold_case"
        else:
            status = "manual_review_logged"
            error_family = "not_classified"

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


def final_recommendation_for(row: dict[str, str]) -> str:
    if row["manual_label_family"] == "hold":
        return "keep_hold_for_later_analysis_exclude_training_for_now"
    if row["manual_label"] == "low_priority_negative":
        return "deprioritize_as_not_candidate_like"
    return "exclude_from_novel_candidate_queue"


def master_status_for(row: dict[str, str]) -> str:
    return row["manual_label"]


def final_ledger_row(row: dict[str, str]) -> dict[str, str]:
    status = master_status_for(row)
    recommendation = final_recommendation_for(row)
    reason = row["manual_reason"]
    if row["reason_status"] != "complete_user_supplied":
        reason = f"{reason} [reason_status={row['reason_status']}]"
    return {
        "epic_id": row["epic_id"],
        "source_batch": REVIEW_BATCH,
        "best_period_days": row["best_period_days"],
        "stage_g_final_recommendation": "no_gatevetter_stage_g_action",
        "final_recommendation": recommendation,
        "stage_f_label": "",
        "stage_h_label": "",
        "visual_label": f"manual_top_hold_review_{row['manual_label']}",
        "final_candidate_status": status,
        "review_bin": f"manual_{REVIEW_SOURCE_QUEUE}_{row['manual_label_family']}",
        "status_reason": reason,
        "visual_notes": reason,
        "reviewer": REVIEWER,
        "reviewed_at": REVIEW_DATE,
        "recommended_next_action": recommendation,
        "stage_h_notes": "",
        "phase_0_folded_path": "",
        "validation_summary_json_path": "",
        "stage_f_closed_45_approved_ledger_status": "",
        "stage_f_closed_45_ledger_priority": "",
        "stage_f_closed_45_evidence_summary": "",
        "stage_f_closed_45_caveats": "",
        "stage_f_closed_45_ledger_change_type": "",
        "stage_g_v2_support_tier": "manual_top_hold_review_not_stage_g",
        "stage_g_v2_calibrated_score": row["gatevetter_score"],
        "stage_g_v2_annotation_only": "TRUE",
        "stage_h_status": "",
        "stage_h_training_label_v3": row["training_eligibility"],
        "stage_h_science_binary_v3": "not_candidate_like"
        if row["manual_label_family"] == "negative"
        else "hold_exclude_for_now",
        "stage_h_ledger_status": status,
        "stage_h_ledger_priority": row["manual_label_family"],
        "stage_h_reason": reason,
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
    training_counts = Counter(row["training_eligibility"] for row in rows)
    incomplete = [row["epic_id"] for row in rows if row["reason_status"] != "complete_user_supplied"]
    high_cnn_negative = [
        row["epic_id"]
        for row in rows
        if row["manual_label_family"] == "negative"
        and cnn_interpretation(row["cnn_score"])
        in {"very_high_morphology_score", "moderate_high_morphology_score"}
    ]
    lines = [
        "GateVetter v0.2 batch-next manual top-hold review summary",
        f"review_batch: {REVIEW_BATCH}",
        f"reviewed_at: {REVIEW_DATE}",
        f"reviewer: {REVIEWER}",
        "",
        f"Stage G queue data rows: {count_data_rows(STAGE_G_QUEUE)}",
        f"Manual decisions supplied and logged: {len(rows)}",
        "Input note: 10 reviewed top-hold objects are now ledgered.",
        "Action: no CNN retraining performed; GateVetter v0.2 thresholds unchanged.",
        "",
        f"manual_label counts: {dict(label_counts)}",
        f"manual_label_family counts: {dict(family_counts)}",
        f"training_eligibility counts: {dict(training_counts)}",
        f"high/moderate-high CNN morphology negatives logged: {high_cnn_negative}",
        f"incomplete/truncated manual reasons: {incomplete}",
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
            f"- {row['epic_id']}: hold_rank={row['hold_rank']}; manual_label={row['manual_label']}; "
            f"training_eligibility={row['training_eligibility']}; reason_status={row['reason_status']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hold_by_epic = find_hold_rows()
    missing = [decision["epic_id"] for decision in DECISIONS if decision["epic_id"] not in hold_by_epic]
    if missing:
        raise RuntimeError(f"Manual decision EPICs not found in hold queue: {missing}")

    rows = decision_rows(hold_by_epic)
    merge_by_epic(DECISIONS_CSV, rows, DECISION_COLUMNS)
    merge_by_epic(CNN_ERROR_LEDGER, cnn_error_rows(rows), CNN_ERROR_COLUMNS)
    final_rows_appended, final_rows_refreshed = upsert_final_ledger_rows(rows)
    write_summary(rows, final_rows_appended, final_rows_refreshed)


if __name__ == "__main__":
    main()
