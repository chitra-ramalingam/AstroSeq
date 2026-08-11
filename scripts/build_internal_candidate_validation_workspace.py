from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "plots" / "k2_batch"
RECONCILED_LEDGER = BATCH_ROOT / "final_candidate_master_ledger_reconciled.csv"
QUEUE_OUT = BATCH_ROOT / "internal_candidate_validation_queue.csv"
PACKET_ROOT = BATCH_ROOT / "internal_candidate_review_packets"
SUMMARY_OUT = BATCH_ROOT / "internal_candidate_validation_summary.txt"

BENCHMARK_LABELS = {
    "recovered_known_confirmed_planet",
    "recovered_known_unconfirmed_candidate",
}
EXCLUDED_LABELS = {
    "reject_as_noise_or_artifact",
    "low_priority_negative",
    "binary_system",
    "variable_or_artifact",
    "uncertain_hold_variability",
}
PRIORITY_EPICS = {
    "EPIC_212024647",
    "EPIC_211682657",
    "EPIC_211396167",
}
PRIMARY_ASSET_NAMES = {
    "phase_0_folded.png": ("phase_0_folded.png",),
    "phase_05_secondary_check.png": ("phase_05_secondary_check.png",),
    "odd_even_check.png": ("odd_even_check.png", "odd_even_zoom.png"),
    "oot_variability_check.png": ("oot_variability_check.png",),
}
EVIDENCE_DIR_PRIORITY = (
    "stage_i_autovet_v1_hold_batch1/period_support_repair_validation",
    "stage_i_autovet_v1_hold_batch1",
    "stage_i_autovet_v1_hold_batch2",
    "stage_h_candidate_followup",
    "stage_f_next10_batch2",
    "stage_f_followup",
    "stage_i_discovery_pilot_batch1",
    "stage_f_next10",
    "stage_f_hybrid_v2",
    "stage_h_candidate_evidence_packets",
    "citizen_science_release_v1/candidates",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def normalized_label_bucket(row: dict[str, str]) -> str:
    label = row["reconciled_label"]
    if label in BENCHMARK_LABELS:
        return "confirmed_or_benchmark"
    if truthy(row.get("strongest_positive_flag")):
        return "strong_positive"
    if label == "candidate_like_hold":
        return "candidate_like_hold"
    if truthy(row.get("positive_retained_flag")):
        return "positive_retained"
    return ""


def include_in_queue(row: dict[str, str]) -> bool:
    label = row["reconciled_label"]
    if label in EXCLUDED_LABELS:
        return False
    return bool(normalized_label_bucket(row))


def sort_evidence_dirs(dirs: set[Path]) -> list[Path]:
    def score(path: Path) -> tuple[int, str]:
        rel_path = rel(path)
        for idx, token in enumerate(EVIDENCE_DIR_PRIORITY):
            if token in rel_path:
                return idx, rel_path
        return len(EVIDENCE_DIR_PRIORITY), rel_path

    return sorted(dirs, key=score)


def build_evidence_dir_index(epic_ids: set[str]) -> dict[str, list[Path]]:
    indexed: dict[str, set[Path]] = {epic_id: set() for epic_id in epic_ids}
    for path in BATCH_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if PACKET_ROOT in path.parents:
            continue
        parent_text = str(path.parent)
        for epic_id in epic_ids:
            if epic_id in parent_text:
                indexed[epic_id].add(path.parent)
    return {epic_id: sort_evidence_dirs(dirs) for epic_id, dirs in indexed.items()}


def pick_existing_asset(evidence_dirs: list[Path], candidate_names: tuple[str, ...]) -> Path | None:
    for directory in evidence_dirs:
        for name in candidate_names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def pick_metrics_source(evidence_dirs: list[Path]) -> Path | None:
    for directory in evidence_dirs:
        validation = directory / "validation_summary.json"
        if validation.exists():
            return validation
        metrics = directory / "metrics.json"
        if metrics.exists():
            return metrics
    return None


def load_metrics(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "validation" in data and isinstance(data["validation"], dict):
        return data["validation"]
    return data


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def review_disposition(row: dict[str, str]) -> str:
    epic_id = row["epic_id"]
    if epic_id == "EPIC_212024647":
        return "provisional_final_internal_candidate"
    if epic_id == "EPIC_211682657":
        return "candidate_with_secondary_alias_caveat"
    if epic_id == "EPIC_211396167":
        return "candidate_like_hold_pending_alias_duration_review"
    if row["reconciled_label"] in BENCHMARK_LABELS:
        return "benchmark_reference"
    if truthy(row.get("strongest_positive_flag")):
        return "strong_positive_internal_validation"
    return "positive_retained_internal_validation"


def packet_summary(
    row: dict[str, str],
    metrics: dict[str, Any],
    copied: dict[str, str],
    missing: list[str],
) -> str:
    lines = [
        f"# {row['epic_id']} Internal Candidate Summary",
        "",
        "## Internal Status",
        f"- validation_bucket: `{row['validation_bucket']}`",
        f"- reconciled_label: `{row['reconciled_label']}`",
        f"- reconciled_status: `{row['reconciled_status']}`",
        f"- recommended_internal_disposition: `{row['recommended_internal_disposition']}`",
        f"- latest_known_stage: `{row['latest_known_stage']}`",
        "",
        "## Transit Metrics",
        f"- best_period_days: `{fmt(metrics.get('best_period_days'))}`",
        f"- primary_depth_snr: `{fmt(metrics.get('primary_depth_snr'))}`",
        f"- transit_duration_hours: `{fmt(metrics.get('transit_duration_hours'))}`",
        f"- odd_even_depth_ratio: `{fmt(metrics.get('odd_even_depth_ratio'))}`",
        f"- secondary_to_primary_depth_ratio: `{fmt(metrics.get('secondary_to_primary_depth_ratio'))}`",
        f"- oot_variability_to_depth: `{fmt(metrics.get('oot_variability_to_depth'))}`",
        f"- alias_risk: `{fmt(metrics.get('alias_risk'))}`",
        "",
        "## Internal Review Note",
    ]

    if row["epic_id"] == "EPIC_212024647":
        lines.extend(
            [
                "Best current live promotion case from the reconciled hold-batch set. Odd/even agreement is good, OOT variability is modest, and the repaired-period evidence is strongest of the three priority rows. Keep one explicit phase-0.5 visual sign-off in the internal review because the rendered secondary panel shows an offset dip that is not captured by the scalar secondary metric.",
            ]
        )
    elif row["epic_id"] == "EPIC_211682657":
        lines.extend(
            [
                "Retain as a caveated candidate only. The secondary-like signal is materially large relative to the primary and the alias picture remains nontrivial, so this should not receive final internal candidate status before secondary/alias review.",
            ]
        )
    elif row["epic_id"] == "EPIC_211396167":
        lines.extend(
            [
                "Retain as a candidate-like hold. The transit is coherent with excellent odd/even and very low OOT variability, but the duration is long enough to require alias/duration sanity review before any promotion.",
            ]
        )
    elif row["validation_bucket"] == "confirmed_or_benchmark":
        lines.extend(
            [
                "Reference case retained for internal benchmarking. Keep it in validation as a known recovery/control rather than a novel promotion.",
            ]
        )
    elif row["validation_bucket"] == "strong_positive":
        lines.extend(
            [
                "Strong retained positive for internal validation. Keep manual caveats attached until final reviewer sign-off.",
            ]
        )
    else:
        lines.extend(
            [
                "Older manually retained positive kept in the internal validation set. It remains useful evidence, but current reconciliation does not make it a new final promotion by itself.",
            ]
        )

    lines.extend(
        [
            "",
            "## Copied Packet Artifacts",
        ]
    )
    if copied:
        lines.extend(f"- {name}: `{path}`" for name, path in copied.items())
    else:
        lines.append("- none")

    lines.extend(["", "## Missing Artifacts"])
    if missing:
        lines.extend(f"- {name}" for name in missing)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Provenance",
            f"- source_files: `{row['source_files']}`",
            f"- source_labels: `{row['source_labels']}`",
            f"- source_actions: `{row['source_actions']}`",
            "",
            "## Guardrail",
            "Internal validation only. This packet does not modify training labels, model files, or public-release artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def build_queue(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    queue: list[dict[str, str]] = []
    for row in rows:
        if not include_in_queue(row):
            continue
        queue_row = dict(row)
        queue_row["validation_bucket"] = normalized_label_bucket(row)
        queue_row["confirmed_or_benchmark_flag"] = str(row["reconciled_label"] in BENCHMARK_LABELS)
        queue_row["candidate_like_hold_flag"] = str(row["reconciled_label"] == "candidate_like_hold")
        queue_row["priority_epic_flag"] = str(row["epic_id"] in PRIORITY_EPICS)
        queue_row["recommended_internal_disposition"] = review_disposition(queue_row)
        queue.append(queue_row)

    bucket_order = {
        "confirmed_or_benchmark": 0,
        "strong_positive": 1,
        "candidate_like_hold": 2,
        "positive_retained": 3,
    }
    return sorted(
        queue,
        key=lambda row: (
            bucket_order[row["validation_bucket"]],
            0 if row["epic_id"] in PRIORITY_EPICS else 1,
            row["epic_id"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_packets(queue_rows: list[dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    PACKET_ROOT.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    missing_by_epic: dict[str, list[str]] = {}
    evidence_dir_index = build_evidence_dir_index({row["epic_id"] for row in queue_rows})

    for row in queue_rows:
        epic_id = row["epic_id"]
        packet_dir = PACKET_ROOT / epic_id
        packet_dir.mkdir(parents=True, exist_ok=True)
        evidence_dirs = evidence_dir_index[epic_id]

        copied: dict[str, str] = {}
        missing: list[str] = []
        for dest_name, candidate_names in PRIMARY_ASSET_NAMES.items():
            source = pick_existing_asset(evidence_dirs, candidate_names)
            if source is None:
                missing.append(dest_name)
                continue
            dest = packet_dir / dest_name
            shutil.copy2(source, dest)
            copied[dest_name] = rel(dest)

        metrics_source = pick_metrics_source(evidence_dirs)
        metrics = load_metrics(metrics_source)
        if metrics:
            metrics_dest = packet_dir / "metrics.json"
            with metrics_dest.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)
                handle.write("\n")
            copied["metrics.json"] = rel(metrics_dest)
        else:
            missing.append("metrics.json")

        summary_path = packet_dir / "internal_candidate_summary.md"
        summary_path.write_text(packet_summary(row, metrics, copied, missing), encoding="utf-8")
        copied["internal_candidate_summary.md"] = rel(summary_path)

        created.append(epic_id)
        missing_by_epic[epic_id] = missing

    return created, missing_by_epic


def build_summary(
    all_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    packet_epics: list[str],
    missing_by_epic: dict[str, list[str]],
) -> str:
    labels = Counter(row["reconciled_label"] for row in all_rows)
    rejected_count = sum(truthy(row["rejection_flag"]) for row in all_rows)
    uncertain_count = sum(truthy(row["uncertainty_flag"]) for row in all_rows)
    confirmed_count = sum(row["reconciled_label"] in BENCHMARK_LABELS for row in all_rows)
    strong_count = sum(truthy(row["strongest_positive_flag"]) for row in all_rows)
    positive_retained_count = sum(truthy(row["positive_retained_flag"]) for row in all_rows)
    candidate_like_hold_count = labels["candidate_like_hold"]
    warning_rows = [
        f"- {epic_id}: missing {', '.join(items)}"
        for epic_id, items in sorted(missing_by_epic.items())
        if items
    ]

    lines = [
        "Internal candidate validation summary",
        "",
        f"confirmed / benchmark count = {confirmed_count}",
        f"strong_positive count = {strong_count}",
        f"positive_retained count = {positive_retained_count}",
        f"candidate_like_hold count = {candidate_like_hold_count}",
        f"rejected count = {rejected_count}",
        f"uncertain count = {uncertain_count}",
        f"number of review packets created = {len(packet_epics)}",
        "",
        "EPICs selected for internal validation:",
    ]
    lines.extend(f"- {row['epic_id']} ({row['validation_bucket']})" for row in queue_rows)
    lines.extend(["", "Missing plot warnings:"])
    lines.extend(warning_rows or ["- none"])
    lines.extend(
        [
            "",
            "Internal deep-review readout:",
            "- provisional final internal candidate: EPIC_212024647",
            "- stays caveated: EPIC_211682657 (secondary/alias review), EPIC_211396167 (alias/duration review)",
            "- manual noise rejects from hold_batch2: EPIC_212037403, EPIC_212018340, EPIC_211950703",
            "",
            "Batch3 gates to tighten before another pass:",
            "- require explicit period-support repair before any candidate promotion from a needs-period-search hold",
            "- require secondary visual review when any phase-0.5 dip is present even if the scalar secondary metric is zero",
            "- hold or reject rows with large secondary-to-primary depth ratios or visible secondary structure",
            "- force alias/duration sanity review for long-duration signals before promotion",
            "- keep high OOT-variability rows out of promotion even when odd/even looks acceptable",
            "",
            "Safety confirmations:",
            "- final_candidate_master_ledger.csv was not overwritten.",
            "- training_labels_v3.csv was not modified.",
            "- No .keras model file was modified.",
            "- No retraining was run.",
            "- No citizen-science release artifacts were created by this workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = read_csv(RECONCILED_LEDGER)
    queue_rows = build_queue(rows)
    write_csv(QUEUE_OUT, queue_rows)
    packet_epics, missing_by_epic = build_packets(queue_rows)
    SUMMARY_OUT.write_text(
        build_summary(rows, queue_rows, packet_epics, missing_by_epic),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
