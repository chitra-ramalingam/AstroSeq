from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gatevetter_v0_2_batch_n import update_rolling_summary


BATCH_PREFIX = "gatevetter_v0_2_batch_4"
REVIEW_BATCH = f"{BATCH_PREFIX}_manual_review"
REVIEW_DATE = date.today().isoformat()
REVIEWER = "codex_visual_review"

STAGE_G_QUEUE = ROOT / f"{BATCH_PREFIX}_stage_g_queue.csv"
TOP_HOLDS = ROOT / f"{BATCH_PREFIX}_top_20_holds.csv"
REJECT_SANITY = ROOT / f"{BATCH_PREFIX}_reject_sanity_sample.csv"
VISUAL_MANIFEST = ROOT / f"{BATCH_PREFIX}_visual_manifest.csv"
DECISIONS_CSV = ROOT / f"{BATCH_PREFIX}_manual_review_decisions.csv"
SUMMARY_TXT = ROOT / f"{BATCH_PREFIX}_manual_review_summary.txt"
CNN_ERROR_LEDGER = ROOT / "cnn_manual_review_error_ledger.csv"
FINAL_CANDIDATE_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"


DECISIONS = [
    ("EPIC_211624954", "uncertain_hold_positive", "Plausible transit-like folded dip with trusted refreshed period, but conservative hold remains appropriate because evidence is not Stage-G clean."),
    ("EPIC_211524158", "variable_or_possible_eb", "Strong raw variability, OOT variability hold, and secondary/phase structure are more consistent with variable or EB contamination than a clean planet candidate."),
    ("EPIC_211687388", "uncertain_hold_positive", "Sharp transit-like event with good CNN support, but odd/even hold and remaining variability keep it as a positive hold rather than candidate-like promotion."),
    ("EPIC_212017960", "low_priority_negative", "Low primary SNR with odd/even and OOT hold gates; visual evidence is not candidate-like enough to keep as a positive hold."),
    ("EPIC_211958260", "variable_or_possible_eb", "High OOT-to-depth with missing key diagnostics and strong structured variability; treat as variable or possible EB."),
    ("EPIC_211480475", "false_positive_eb_or_variable", "Large coherent stellar/EB-like modulation and deep phase structure; no missed planet-like reject."),
    ("EPIC_211992776", "variable_or_possible_eb", "Weak shallow folded feature on structured variability with EB/variable gate context; no candidate-like recovery."),
    ("EPIC_211719484", "false_positive_eb_or_variable", "Period-ambiguous EB-like morphology with secondary/shape concerns; reject remains valid."),
    ("EPIC_211944122", "reject_as_noise_or_artifact", "No coherent transit-like signal in folded, zoom, or event-stack views; reject remains valid."),
    ("EPIC_211705299", "reject_as_noise_or_artifact", "No reliable transit-like family; period ambiguity and noisy folded structure support reject."),
    ("EPIC_211355462", "reject_as_noise_or_artifact", "Weak/ambiguous dip with period ambiguity and poor event support; reject remains valid."),
    ("EPIC_211963291", "reject_as_noise_or_artifact", "Period-ambiguous noisy structure without a convincing repeated transit family."),
    ("EPIC_211797554", "reject_as_noise_or_artifact", "No convincing transit-like signal; noisy/ambiguous folded and event-stack views."),
    ("EPIC_211330591", "reject_as_noise_or_artifact", "Noisy period-ambiguous folded structure; no missed candidate-like evidence."),
    ("EPIC_211803640", "reject_as_noise_or_artifact", "Noisy period-ambiguous folded structure and weak event support; reject remains valid."),
]


DECISION_COLUMNS = [
    "review_batch",
    "reviewed_at",
    "reviewer",
    "epic_id",
    "source_packet",
    "source_rank",
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
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(value: object) -> str:
    return "" if value is None else str(value)


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def manual_family(label: str) -> str:
    if label == "candidate_like":
        return "candidate_like"
    if label.startswith("uncertain_hold"):
        return "hold"
    return "negative"


def training_eligibility(label: str) -> str:
    if label == "candidate_like":
        return "candidate_like_manual_positive_candidate"
    if label.startswith("uncertain_hold"):
        return "exclude_for_now"
    return "possible_future_training_example"


def training_use_note(label: str) -> str:
    if label == "candidate_like":
        return "Candidate-like manual positive; do not use for retraining during GateVetter v0.2 freeze."
    if label.startswith("uncertain_hold"):
        return "Do not use hold for training yet."
    return "Clear negative or EB/variable false positive; possible future training example."


def science_binary(label: str) -> str:
    if label == "candidate_like":
        return "science_like"
    if label.startswith("uncertain_hold"):
        return "hold_exclude_for_now"
    if label in {"variable_or_possible_eb", "false_positive_eb_or_variable"}:
        return "false_positive_eb_or_variable"
    if label == "low_priority_negative":
        return "not_candidate_like_low_priority"
    return "not_candidate_like_noise_or_artifact"


def final_recommendation(label: str) -> str:
    if label == "candidate_like":
        return "promote_to_candidate_followup_queue_manual_visual_positive"
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
        out[clean(row.get("epic_id"))] = {
            "source_packet": clean(row.get("packet")),
            "source_rank": clean(row.get("rank")),
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
    source_by_epic = source_rows()
    paths_by_epic = manifest_paths()
    rows: list[dict[str, str]] = []
    for epic_id, label, reason in DECISIONS:
        if epic_id not in source_by_epic:
            raise RuntimeError(f"Manual decision EPIC not found in batch packets: {epic_id}")
        source = source_by_epic[epic_id]
        paths = paths_by_epic.get(epic_id, {})
        row = {
            "review_batch": REVIEW_BATCH,
            "reviewed_at": REVIEW_DATE,
            "reviewer": REVIEWER,
            "epic_id": epic_id,
            "source_packet": paths.get("source_packet") or clean(source.get("source_packet")),
            "source_rank": paths.get("source_rank") or clean(source.get("source_rank")),
            "manual_label": label,
            "manual_label_family": manual_family(label),
            "training_eligibility": training_eligibility(label),
            "training_use_note": training_use_note(label),
            "manual_reason": reason,
            "reason_status": "complete_codex_visual_review",
            "manual_vetted": "true",
            "phase_0_folded_path": paths.get("phase_0_folded_path", ""),
            "validation_summary_json_path": paths.get("validation_summary_json_path", ""),
            "notes": "GateVetter v0.2 unchanged; CNN not retrained; visual review logged after batch prediction.",
        }
        for column in DECISION_COLUMNS:
            if column in row:
                continue
            if column == "gatevetter_v0_2_prediction":
                row[column] = clean(source.get("gatevetter_v0_2_prediction") or source.get("gatevetter_prediction"))
            elif column == "gatevetter_v0_2_reason":
                row[column] = clean(source.get("gatevetter_v0_2_reason") or source.get("gatevetter_v0_reason"))
            else:
                row[column] = clean(source.get(column))
        rows.append(row)
    return rows


def merge_by_batch_and_epic(path: Path, new_rows: list[dict[str, str]], columns: list[str]) -> None:
    existing = read_csv(path)
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
        if family == "candidate_like":
            status = "manual_candidate_like_positive"
            error_family = "not_error_manual_positive"
        elif family == "hold":
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
        "visual_label": f"{REVIEW_BATCH}_{label}",
        "final_candidate_status": label,
        "review_bin": f"{REVIEW_BATCH}_{row['manual_label_family']}",
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
        "stage_g_v2_support_tier": f"{REVIEW_BATCH}_not_stage_g",
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
    existing = read_csv(FINAL_CANDIDATE_LEDGER)
    if not existing:
        return 0, 0
    with FINAL_CANDIDATE_LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
    new_by_epic = {row["epic_id"]: final_ledger_row(row) for row in rows}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    updated = 0
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
    write_csv(FINAL_CANDIDATE_LEDGER, out, columns)
    return len(missing), updated


def write_summary(rows: list[dict[str, str]], appended: int, updated: int) -> None:
    label_counts = Counter(row["manual_label"] for row in rows)
    family_counts = Counter(row["manual_label_family"] for row in rows)
    stage_g_count = count_data_rows(STAGE_G_QUEUE)
    top_hold_count = count_data_rows(TOP_HOLDS)
    reject_sanity_count = count_data_rows(REJECT_SANITY)
    reviewed_top_count = sum(1 for row in rows if row["source_packet"] == "top_20_holds")
    reviewed_reject_count = sum(1 for row in rows if row["source_packet"] == "reject_sanity_high_cnn_high_snr")
    reject_candidate_like_count = sum(
        1
        for row in rows
        if row["source_packet"] == "reject_sanity_high_cnn_high_snr" and row["manual_label"] == "candidate_like"
    )
    stage_g_line = "- Stage G queue was empty." if stage_g_count == 0 else "- Stage G queue was reviewed first."
    reject_candidate_line = (
        "- No missed candidate-like positives found in reject sanity sample."
        if reject_candidate_like_count == 0
        else f"- Candidate-like positives found in reject sanity sample: {reject_candidate_like_count}."
    )
    lines = [
        f"GateVetter v0.2 {BATCH_PREFIX} manual review",
        f"review_batch: {REVIEW_BATCH}",
        f"reviewed_at: {REVIEW_DATE}",
        f"reviewer: {REVIEWER}",
        "",
        "Workflow:",
        f"- Stage G queue rows: {stage_g_count}",
        stage_g_line,
        f"- {reviewed_top_count}/{top_hold_count} top holds reviewed.",
        f"- {reviewed_reject_count}/{reject_sanity_count} reject sanity rows reviewed.",
        reject_candidate_line,
        "- GateVetter v0.2 unchanged.",
        "- CNN not retrained.",
        "- Manual labels were not used during prediction.",
        "",
        f"manual_label counts: {dict(label_counts)}",
        f"manual_label_family counts: {dict(family_counts)}",
        "",
        "Files updated/created:",
        f"- {DECISIONS_CSV.relative_to(ROOT).as_posix()}",
        f"- {SUMMARY_TXT.relative_to(ROOT).as_posix()}",
        f"- {CNN_ERROR_LEDGER.relative_to(ROOT).as_posix()}",
        f"- {FINAL_CANDIDATE_LEDGER.relative_to(ROOT).as_posix()} appended rows: {appended}; refreshed review-batch rows: {updated}",
        "- gatevetter_v0_2_rolling_batch_summary.csv",
        "",
        "Manual labels:",
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
    merge_by_batch_and_epic(CNN_ERROR_LEDGER, cnn_error_rows(rows), CNN_ERROR_COLUMNS)
    appended, updated = upsert_final_ledger_rows(rows)
    write_summary(rows, appended, updated)
    update_rolling_summary()


if __name__ == "__main__":
    main()
