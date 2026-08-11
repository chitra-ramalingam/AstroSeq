from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog"
FROZEN_CNN_MODEL_PATH = "models/k2_nocrop_flux_seed46_split303.best.keras"
FROZEN_CNN_POLICY_VERSION = "frozen_batch3_transit_morphology_policy_2026-05-15"


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    kind: str
    priority: int
    label: str


SOURCES = [
    # Frozen CNN / morphology evidence.
    SourceSpec(ROOT / "k2_hybrid_candidate_score_v2.csv", "cnn", 10, "existing_frozen_flux_scores"),
    SourceSpec(ROOT / "freezes" / "stage_f_closed_45_existing_keras_scores.csv", "cnn", 20, "stage_f_closed_45_existing_keras_scores"),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch3" / "frozen_cnn_batch3_inference.csv",
        "cnn",
        30,
        "frozen_cnn_batch3_inference",
    ),
    # AutoVet.
    SourceSpec(ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv", "autovet", 10, "stage_i_autovet_v1_global"),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1" / "autovet_candidate_validation_ledger.csv",
        "autovet",
        20,
        "candidate_batch1",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "autovet_hold_validation_ledger.csv",
        "autovet",
        20,
        "hold_batch1",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch2" / "stage_i_hold_batch2_input.csv",
        "autovet",
        15,
        "hold_batch2",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch3" / "stage_i_hold_batch3_input.csv",
        "autovet",
        15,
        "hold_batch3",
    ),
    # Diagnostics.
    SourceSpec(ROOT / "freezes" / "k2_hybrid_candidate_score_v3_stage_f_closed_45.csv", "diagnostic", 30, "stage_f_closed_45_diagnostics"),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "post_repair_validation_ledger.csv",
        "diagnostic",
        40,
        "post_repair_validation_ledger",
    ),
    SourceSpec(ROOT / "k2_stage_f_next10_manual_reviewed.csv", "diagnostic", 50, "stage_f_next10_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_next10_batch2_manual_reviewed.csv", "diagnostic", 50, "stage_f_next10_batch2_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_v3_manual_reviewed.csv", "diagnostic", 50, "stage_f_v3_manual_reviewed"),
    # Manual vetting.
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "stage_i_post_repair_manual_decision_log.csv",
        "manual",
        40,
        "stage_i_post_repair_manual_decision_log",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "manual_post_repair_review_decisions.csv",
        "manual",
        50,
        "manual_post_repair_review_decisions",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch2" / "manual_stage_i_hold_batch2_review_decisions.csv",
        "manual",
        50,
        "manual_stage_i_hold_batch2_review_decisions",
    ),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch3" / "manual_stage_i_hold_batch3_review_decisions.csv",
        "manual",
        50,
        "manual_stage_i_hold_batch3_review_decisions",
    ),
    SourceSpec(ROOT / "k2_stage_f_v3_manual_review_outcomes.csv", "manual", 50, "stage_f_v3_manual_review_outcomes"),
    SourceSpec(ROOT / "k2_stage_f_v3_manual_reviewed.csv", "manual", 45, "stage_f_v3_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_next10_manual_reviewed.csv", "manual", 45, "stage_f_next10_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_next10_batch2_manual_reviewed.csv", "manual", 45, "stage_f_next10_batch2_manual_reviewed"),
    SourceSpec(ROOT / "manual_reject_archive.csv", "manual", 45, "manual_reject_archive"),
    SourceSpec(ROOT / "rejected_manual.csv", "manual", 45, "rejected_manual"),
    # Stage F / G / ledger.
    SourceSpec(ROOT / "k2_stage_f_v3_manual_review_outcomes.csv", "stage_f", 60, "stage_f_v3_manual_review_outcomes"),
    SourceSpec(ROOT / "k2_stage_f_v3_manual_reviewed.csv", "stage_f", 55, "stage_f_v3_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_next10_manual_reviewed.csv", "stage_f", 55, "stage_f_next10_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_next10_batch2_manual_reviewed.csv", "stage_f", 55, "stage_f_next10_batch2_manual_reviewed"),
    SourceSpec(ROOT / "k2_stage_f_followup_validation.csv", "stage_f", 40, "stage_f_followup_validation"),
    SourceSpec(ROOT / "k2_stage_f_hybrid_v2_validation.csv", "stage_f", 40, "stage_f_hybrid_v2_validation"),
    SourceSpec(ROOT / "k2_stage_f_next10_validation.csv", "stage_f", 40, "stage_f_next10_validation"),
    SourceSpec(ROOT / "k2_stage_f_next10_batch2_validation.csv", "stage_f", 40, "stage_f_next10_batch2_validation"),
    SourceSpec(ROOT / "k2_stage_f_v3_recovery_batch1_validation.csv", "stage_f", 40, "stage_f_v3_recovery_batch1_validation"),
    SourceSpec(ROOT / "k2_stage_f_v3_recovery_batch2_validation.csv", "stage_f", 40, "stage_f_v3_recovery_batch2_validation"),
    SourceSpec(ROOT / "k2_stage_f_v3_recovery_batch3_validation.csv", "stage_f", 40, "stage_f_v3_recovery_batch3_validation"),
    SourceSpec(ROOT / "k2_stage_f_v3_recovery_batch4_validation.csv", "stage_f", 40, "stage_f_v3_recovery_batch4_validation"),
    SourceSpec(ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv", "ledger", 70, "final_candidate_master_ledger"),
    SourceSpec(ROOT / "freezes" / "stage_g_calibrated_ranking_layer.csv", "stage_g", 45, "stage_g_calibrated_ranking_layer"),
    SourceSpec(ROOT / "freezes" / "stage_g_candidate_ledger_review.csv", "stage_g", 60, "stage_g_candidate_ledger_review"),
    SourceSpec(ROOT / "freezes" / "stage_g_v2_support_tier_review.csv", "stage_g", 50, "stage_g_v2_support_tier_review"),
    SourceSpec(ROOT / "k2_stage_g_candidate_dossiers.csv", "stage_g", 40, "stage_g_candidate_dossiers"),
    SourceSpec(
        ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "stage_g_post_repair_validation_ranking.csv",
        "stage_g",
        55,
        "stage_g_post_repair_validation_ranking",
    ),
    # Training labels.
    SourceSpec(ROOT / "training_labels_v3.csv", "training", 70, "training_labels_v3"),
    SourceSpec(ROOT / "freezes" / "stage_h_candidate_followup_decisions.csv", "training", 80, "stage_h_candidate_followup_decisions"),
]


OUTPUT_FIELDS = [
    "epic_id",
    "cnn_model_path",
    "cnn_score",
    "cnn_score_name",
    "cnn_role",
    "morphology_positive",
    "cnn_policy_version",
    "autovet_batch",
    "autovet_stage",
    "autovet_label",
    "autovet_reason",
    "autovet_score",
    "autovet_period_days",
    "autovet_threshold",
    "autovet_recommended_action",
    "period_gate",
    "odd_even_gate",
    "secondary_eclipse_gate",
    "oot_variability_gate",
    "depth_consistency_gate",
    "event_cluster_gate",
    "alias_gate",
    "thruster_cadence_gate",
    "diagnostic_gate_summary",
    "manual_label",
    "manual_reason",
    "manual_reviewer",
    "manual_review_date",
    "manual_next_action",
    "manual_confidence",
    "stage_f_label",
    "stage_f_reason",
    "stage_g_label",
    "stage_g_reason",
    "promotion_status",
    "final_candidate_status",
    "training_label",
    "training_label_source",
    "used_for_training",
    "eligible_for_future_vetter_training",
    "review_level",
    "manual_vetted",
    "diagnostic_vetted",
    "autovet_only",
    "master_label",
    "master_reason",
    "master_next_action",
    "decision_authority",
    "has_conflict",
    "conflict_reason",
    "source_files",
    "last_updated",
]


METRIC_COLUMNS = {
    "period_support_count",
    "n_periods_validated",
    "best_period_days",
    "primary_period",
    "odd_even_depth_ratio",
    "odd_even_depth_delta_explicit",
    "secondary_to_primary_depth_ratio",
    "secondary_depth_snr",
    "oot_variability_to_depth",
    "folded_depth_consistency",
    "event_family_count",
    "n_events",
    "alias_risk",
    "alias_best_support_ratio",
    "spike_fraction_2cadence",
    "stage_d_spike_clean",
}


def relpath(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_float(value: Any) -> float | None:
    raw = text(value)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def as_bool(value: Any) -> bool | None:
    raw = text(value).lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    return None


def as_date(value: Any) -> date | None:
    raw = text(value)
    if not raw:
        return None
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            continue
    return None


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def first(row: dict[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = text(row.get(column))
        if value:
            return value
    return ""


def normalize_epic(value: Any) -> str:
    raw = text(value).replace(" ", "_")
    if raw.upper().startswith("EPIC_"):
        return raw.upper()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"EPIC_{digits}" if digits else ""


@dataclass
class Choice:
    value: str = ""
    priority: int = -1
    source: str = ""
    observed_at: date | None = None

    def update(self, value: str, priority: int, source: str, observed_at: date | None = None) -> None:
        value = text(value)
        if not value:
            return
        if priority > self.priority:
            self.value = value
            self.priority = priority
            self.source = source
            self.observed_at = observed_at
            return
        if priority == self.priority and observed_at and (self.observed_at is None or observed_at >= self.observed_at):
            self.value = value
            self.source = source
            self.observed_at = observed_at


@dataclass
class Entry:
    epic_id: str
    source_files: set[str] = field(default_factory=set)
    last_updated_values: list[date] = field(default_factory=list)
    metrics: dict[str, Choice] = field(default_factory=lambda: defaultdict(Choice))
    manual_label: Choice = field(default_factory=Choice)
    manual_reason: Choice = field(default_factory=Choice)
    manual_reviewer: Choice = field(default_factory=Choice)
    manual_review_date: Choice = field(default_factory=Choice)
    manual_next_action: Choice = field(default_factory=Choice)
    manual_confidence: Choice = field(default_factory=Choice)
    stage_f_label: Choice = field(default_factory=Choice)
    stage_f_reason: Choice = field(default_factory=Choice)
    stage_g_label: Choice = field(default_factory=Choice)
    stage_g_reason: Choice = field(default_factory=Choice)
    final_candidate_status: Choice = field(default_factory=Choice)
    training_label: Choice = field(default_factory=Choice)
    training_label_source: Choice = field(default_factory=Choice)
    autovet_batch: Choice = field(default_factory=Choice)
    autovet_stage: Choice = field(default_factory=Choice)
    autovet_label: Choice = field(default_factory=Choice)
    autovet_reason: Choice = field(default_factory=Choice)
    autovet_score: Choice = field(default_factory=Choice)
    autovet_period_days: Choice = field(default_factory=Choice)
    autovet_threshold: Choice = field(default_factory=Choice)
    autovet_recommended_action: Choice = field(default_factory=Choice)
    cnn_model_path: Choice = field(default_factory=Choice)
    cnn_score: Choice = field(default_factory=Choice)
    manual_labels_seen: set[str] = field(default_factory=set)

    def touch(self, source_file: str, reviewed_at: date | None = None) -> None:
        self.source_files.add(source_file)
        if reviewed_at:
            self.last_updated_values.append(reviewed_at)


def read_rows(spec: SourceSpec) -> Iterable[dict[str, str]]:
    if not spec.path.exists():
        return []
    with spec.path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def absorb_metrics(entry: Entry, row: dict[str, str], priority: int, source: str) -> None:
    for column in METRIC_COLUMNS:
        value = first(row, [column])
        if value:
            entry.metrics[column].update(value, priority, source)


def absorb_cnn(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    score = first(row, ["model_score", "flux_p_science_like"])
    if not score:
        return
    entry.cnn_model_path.update(first(row, ["model_path"]) or FROZEN_CNN_MODEL_PATH, spec.priority, spec.label)
    entry.cnn_score.update(score, spec.priority, spec.label)
    absorb_metrics(entry, row, spec.priority, spec.label)


def absorb_autovet(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    label = first(row, ["autovet_label", "current_label"])
    if not label:
        return
    stage = "stage_i_internal_autovet"
    entry.autovet_batch.update(spec.label, spec.priority, spec.label)
    entry.autovet_stage.update(stage, spec.priority, spec.label)
    entry.autovet_label.update(label, spec.priority, spec.label)
    entry.autovet_reason.update(
        first(row, ["reason", "explanation_short", "reason_for_selection", "known_caveat"]),
        spec.priority,
        spec.label,
    )
    entry.autovet_score.update(first(row, ["autovet_rank_score", "review_priority_score"]), spec.priority, spec.label)
    entry.autovet_period_days.update(first(row, ["best_period_days", "primary_period"]), spec.priority, spec.label)
    entry.autovet_recommended_action.update(
        first(row, ["recommended_next_action", "proposed_autovet_action"]),
        spec.priority,
        spec.label,
    )
    absorb_metrics(entry, row, spec.priority, spec.label)


def absorb_manual(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    reviewed_at = as_date(first(row, ["reviewed_at"]))
    manual_label = first(
        row,
        [
            "manual_label",
            "manual_stage_i_post_repair_label",
            "manual_stage_f_label",
            "manual_review_label",
            "manual_review_status",
            "ledger_label",
        ],
    )
    if not manual_label:
        return
    entry.manual_labels_seen.add(manual_label)
    entry.manual_label.update(manual_label, spec.priority, spec.label, reviewed_at)
    entry.manual_reason.update(first(row, ["manual_reason", "reason", "visual_notes", "audit_note"]), spec.priority, spec.label, reviewed_at)
    entry.manual_reviewer.update(first(row, ["reviewer"]), spec.priority, spec.label, reviewed_at)
    entry.manual_review_date.update(first(row, ["reviewed_at"]), spec.priority, spec.label, reviewed_at)
    entry.manual_next_action.update(first(row, ["next_action", "recommended_next_action", "manual_action"]), spec.priority, spec.label, reviewed_at)
    entry.manual_confidence.update(first(row, ["manual_confidence", "confidence"]), spec.priority, spec.label, reviewed_at)
    absorb_metrics(entry, row, spec.priority, spec.label)
    entry.touch(relpath(spec.path), reviewed_at)


def absorb_stage_f(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    reviewed_at = as_date(first(row, ["reviewed_at"]))
    entry.stage_f_label.update(first(row, ["stage_f_label", "post_repair_stage_f_label"]), spec.priority, spec.label, reviewed_at)
    entry.stage_f_reason.update(first(row, ["stage_f_reason", "post_repair_reason"]), spec.priority, spec.label, reviewed_at)
    absorb_metrics(entry, row, spec.priority, spec.label)


def absorb_stage_g(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    entry.stage_g_label.update(
        first(
            row,
            [
                "stage_g_ledger_review_action",
                "stage_g_v2_action",
                "stage_g_ranking_recommendation",
                "stage_g_validation_status",
                "stage_g_review_tier",
                "stage_g_support_tier",
                "stage_g_final_recommendation",
                "final_recommendation",
            ],
        ),
        spec.priority,
        spec.label,
    )
    entry.stage_g_reason.update(first(row, ["stage_g_review_reason", "reason", "score_note"]), spec.priority, spec.label)
    absorb_metrics(entry, row, spec.priority, spec.label)


def absorb_ledger(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    reviewed_at = as_date(first(row, ["reviewed_at", "stage_h_reviewed_at"]))
    entry.final_candidate_status.update(first(row, ["final_candidate_status", "stage_h_status"]), spec.priority, spec.label, reviewed_at)
    entry.stage_f_label.update(first(row, ["stage_f_label"]), spec.priority, spec.label, reviewed_at)
    entry.stage_g_label.update(first(row, ["stage_g_final_recommendation"]), spec.priority, spec.label, reviewed_at)
    entry.stage_g_reason.update(first(row, ["status_reason", "stage_h_notes"]), spec.priority, spec.label, reviewed_at)
    absorb_metrics(entry, row, spec.priority, spec.label)


def absorb_training(entry: Entry, row: dict[str, str], spec: SourceSpec) -> None:
    reviewed_at = as_date(first(row, ["stage_h_reviewed_at"]))
    training_label = first(row, ["training_label_v3"])
    if training_label:
        entry.training_label.update(training_label, spec.priority, spec.label, reviewed_at)
        entry.training_label_source.update(first(row, ["training_label_rule"]) or spec.label, spec.priority, spec.label, reviewed_at)
    entry.final_candidate_status.update(first(row, ["final_candidate_status", "stage_h_status"]), spec.priority, spec.label, reviewed_at)


def ingest() -> tuple[dict[str, Entry], list[str]]:
    entries: dict[str, Entry] = {}
    included_sources: list[str] = []
    for spec in SOURCES:
        rows = read_rows(spec)
        if not rows:
            continue
        source_file = relpath(spec.path)
        used = False
        for row in rows:
            epic_id = normalize_epic(first(row, ["epic_id"]))
            if not epic_id:
                continue
            entry = entries.setdefault(epic_id, Entry(epic_id=epic_id))
            reviewed_at = as_date(first(row, ["reviewed_at", "stage_h_reviewed_at"]))
            entry.touch(source_file, reviewed_at)
            if spec.kind == "cnn":
                absorb_cnn(entry, row, spec)
            elif spec.kind == "autovet":
                absorb_autovet(entry, row, spec)
            elif spec.kind == "diagnostic":
                absorb_metrics(entry, row, spec.priority, spec.label)
            elif spec.kind == "manual":
                absorb_manual(entry, row, spec)
            elif spec.kind == "stage_f":
                absorb_stage_f(entry, row, spec)
            elif spec.kind == "stage_g":
                absorb_stage_g(entry, row, spec)
            elif spec.kind == "ledger":
                absorb_ledger(entry, row, spec)
            elif spec.kind == "training":
                absorb_training(entry, row, spec)
            used = True
        if used:
            included_sources.append(source_file)
    return entries, included_sources


def metric_float(entry: Entry, name: str) -> float | None:
    return as_float(entry.metrics[name].value)


def metric_bool(entry: Entry, name: str) -> bool | None:
    return as_bool(entry.metrics[name].value)


def gate_period(entry: Entry) -> str:
    support = metric_float(entry, "period_support_count")
    validated = metric_float(entry, "n_periods_validated")
    period = metric_float(entry, "best_period_days") or metric_float(entry, "primary_period")
    if validated == 0:
        return "fail"
    if (validated is not None and validated >= 1) or (support is not None and support >= 3) or period is not None:
        return "pass"
    return "unknown"


def gate_odd_even(entry: Entry) -> str:
    ratio = metric_float(entry, "odd_even_depth_ratio")
    delta = metric_float(entry, "odd_even_depth_delta_explicit")
    if ratio is None and delta is None:
        return "unknown"
    if (ratio is not None and ratio < 0.7) or (delta is not None and delta > 0.5):
        return "fail"
    if (ratio is None or ratio >= 0.8) and (delta is None or delta <= 0.25):
        return "pass"
    return "review"


def gate_secondary(entry: Entry) -> str:
    ratio = metric_float(entry, "secondary_to_primary_depth_ratio")
    snr = metric_float(entry, "secondary_depth_snr")
    if ratio is None and snr is None:
        return "unknown"
    if (ratio is not None and ratio >= 0.3) or (snr is not None and snr >= 5):
        return "fail"
    if (ratio is None or ratio <= 0.1) and (snr is None or snr <= 3):
        return "pass"
    return "review"


def gate_oot(entry: Entry) -> str:
    value = metric_float(entry, "oot_variability_to_depth")
    if value is None:
        return "unknown"
    if value > 1:
        return "fail"
    if value <= 0.5:
        return "pass"
    return "review"


def gate_depth_consistency(entry: Entry) -> str:
    value = metric_float(entry, "folded_depth_consistency")
    if value is None:
        return "unknown"
    if value < 0.6:
        return "fail"
    if value >= 0.8:
        return "pass"
    return "review"


def gate_event_cluster(entry: Entry) -> str:
    families = metric_float(entry, "event_family_count")
    n_events = metric_float(entry, "n_events")
    if families is None and n_events is None:
        return "unknown"
    if (families is not None and families < 2) or (n_events is not None and n_events < 3):
        return "fail"
    if (families is None or families >= 3) and (n_events is None or n_events >= 3):
        return "pass"
    return "review"


def gate_alias(entry: Entry) -> str:
    risk = entry.metrics["alias_risk"].value.lower()
    ratio = metric_float(entry, "alias_best_support_ratio")
    if not risk and ratio is None:
        return "unknown"
    if risk == "high" or (ratio is not None and ratio >= 1):
        return "fail"
    if risk == "low" or (ratio is not None and ratio <= 0.5):
        return "pass"
    return "review"


def gate_thruster(entry: Entry) -> str:
    spike_fraction = metric_float(entry, "spike_fraction_2cadence")
    spike_clean = metric_bool(entry, "stage_d_spike_clean")
    if spike_fraction is None and spike_clean is None:
        return "unknown"
    if spike_clean is False or (spike_fraction is not None and spike_fraction > 0.35):
        return "fail"
    if spike_clean is True or (spike_fraction is not None and spike_fraction <= 0.25):
        return "pass"
    return "review"


def gate_summary(gates: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in gates.items())


def is_manual_reject(label: str) -> bool:
    value = label.lower()
    return any(token in value for token in ("reject", "binary", "noise", "artifact", "false_positive"))


def promotion_status(entry: Entry, master_label: str) -> str:
    status = entry.final_candidate_status.value.lower()
    label = master_label.lower()
    combined = f"{status} {label}"
    if "promote" in combined or "candidate_like" in combined or "planet_like" in combined or "recovered_known" in combined:
        if any(token in combined for token in ("reject", "hold", "uncertain")):
            return "mixed_or_pending"
        return "promoted_or_candidate"
    if "reject" in combined or "binary" in combined or "noise" in combined or "artifact" in combined:
        return "rejected"
    if "hold" in combined or "uncertain" in combined or "followup" in combined:
        return "hold_or_followup"
    return "unresolved"


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


def families_disagree(left: str, right: str) -> bool:
    left_family = label_family(left)
    right_family = label_family(right)
    return left_family != "unknown" and right_family != "unknown" and left_family != right_family


def conflict_reasons(entry: Entry, morphology_positive: bool) -> list[str]:
    reasons: list[str] = []
    manual = entry.manual_label.value
    autovet = entry.autovet_label.value
    stage_f = entry.stage_f_label.value
    stage_g = entry.stage_g_label.value
    if len(entry.manual_labels_seen) > 1:
        reasons.append("multiple incompatible manual labels")
    if manual and autovet and manual != autovet:
        reasons.append("manual_label disagrees with autovet_label")
    if manual and stage_f and manual != stage_f:
        reasons.append("manual_label disagrees with stage_f_label")
    if manual and stage_g and manual != stage_g:
        reasons.append("manual_label disagrees with stage_g_label")
    if autovet == "candidate_like" and manual and is_manual_reject(manual):
        reasons.append("AutoVet candidate_like is manually rejected")
    if morphology_positive and manual and is_manual_reject(manual):
        reasons.append("CNN morphology-positive target is manually rejected")
    return reasons


def diagnostic_features_present(gates: dict[str, str]) -> bool:
    return any(value != "unknown" for value in gates.values())


def synthetic_only(entry: Entry) -> bool:
    source_text = " ".join(sorted(entry.source_files)).lower()
    return "synthetic" in source_text


def decision(entry: Entry, morphology_positive: bool) -> tuple[str, str, str, str]:
    if entry.manual_label.value:
        return (
            entry.manual_label.value,
            entry.manual_reason.value,
            entry.manual_next_action.value,
            "manual_review",
        )
    if entry.stage_g_label.value:
        return (
            entry.stage_g_label.value,
            entry.stage_g_reason.value,
            entry.stage_g_label.value,
            "stage_g",
        )
    if entry.stage_f_label.value:
        return (
            entry.stage_f_label.value,
            entry.stage_f_reason.value,
            "",
            "stage_f",
        )
    if entry.autovet_label.value:
        return (
            entry.autovet_label.value,
            entry.autovet_reason.value,
            entry.autovet_recommended_action.value,
            "internal_autovet",
        )
    if entry.cnn_score.value:
        return (
            "morphology_positive_only" if morphology_positive else "morphology_negative_only",
            "Frozen CNN morphology score only; no vetting authority.",
            "",
            "frozen_cnn_morphology_score",
        )
    return ("unresolved", "", "", "none")


def review_level(entry: Entry) -> str:
    if entry.manual_label.value:
        return "manually_reviewed"
    if entry.stage_g_label.value:
        return "stage_g_reviewed"
    if entry.stage_f_label.value:
        return "stage_f_reviewed"
    if entry.autovet_label.value:
        return "auto_only"
    if entry.cnn_score.value:
        return "cnn_scored_only"
    return "unresolved"


def classify_conflict_severity(reason: str, row: dict[str, str], all_reasons: list[str]) -> str:
    if reason == "CNN morphology-positive target is manually rejected":
        other_reasons = [item for item in all_reasons if item != reason]
        return "expected_cnn_false_positive" if not other_reasons else "soft_conflict"
    if reason == "multiple incompatible manual labels":
        return "hard_conflict"
    if reason == "manual_label disagrees with autovet_label":
        return "soft_conflict"
    if reason in {"manual_label disagrees with stage_f_label", "manual_label disagrees with stage_g_label"}:
        manual_family = label_family(row["manual_label"])
        comparison = row["stage_f_label"] if "stage_f" in reason else row["stage_g_label"]
        comparison_family = label_family(comparison)
        if manual_family == comparison_family:
            return "vocabulary_mismatch"
        return "hard_conflict"
    return "soft_conflict"


def row_for(entry: Entry) -> tuple[dict[str, str], list[str]]:
    cnn_score = as_float(entry.cnn_score.value)
    morphology_positive = cnn_score is not None and cnn_score > 0.5
    gates = {
        "period_gate": gate_period(entry),
        "odd_even_gate": gate_odd_even(entry),
        "secondary_eclipse_gate": gate_secondary(entry),
        "oot_variability_gate": gate_oot(entry),
        "depth_consistency_gate": gate_depth_consistency(entry),
        "event_cluster_gate": gate_event_cluster(entry),
        "alias_gate": gate_alias(entry),
        "thruster_cadence_gate": gate_thruster(entry),
    }
    conflicts = conflict_reasons(entry, morphology_positive)
    disagreement_conflicts = [reason for reason in conflicts if reason != "CNN morphology-positive target is manually rejected"]
    master_label, master_reason, master_next_action, authority = decision(entry, morphology_positive)
    training_label = entry.training_label.value
    used_for_training = bool(training_label)
    unresolved_conflict = "multiple incompatible manual labels" in conflicts
    eligible = (
        bool(entry.manual_label.value)
        and diagnostic_features_present(gates)
        and not entry.manual_label.value.startswith("uncertain_hold")
        and not synthetic_only(entry)
        and not unresolved_conflict
    )
    row = {
        "epic_id": entry.epic_id,
        "cnn_model_path": entry.cnn_model_path.value,
        "cnn_score": fmt_float(cnn_score),
        "cnn_score_name": "transit_morphology_score" if cnn_score is not None else "",
        "cnn_role": "morphology_scorer_only" if cnn_score is not None else "",
        "morphology_positive": str(morphology_positive).lower() if cnn_score is not None else "",
        "cnn_policy_version": FROZEN_CNN_POLICY_VERSION if cnn_score is not None else "",
        "autovet_batch": entry.autovet_batch.value,
        "autovet_stage": entry.autovet_stage.value,
        "autovet_label": entry.autovet_label.value,
        "autovet_reason": entry.autovet_reason.value,
        "autovet_score": entry.autovet_score.value,
        "autovet_period_days": entry.autovet_period_days.value,
        "autovet_threshold": entry.autovet_threshold.value,
        "autovet_recommended_action": entry.autovet_recommended_action.value,
        **gates,
        "diagnostic_gate_summary": gate_summary(gates),
        "manual_label": entry.manual_label.value,
        "manual_reason": entry.manual_reason.value,
        "manual_reviewer": entry.manual_reviewer.value,
        "manual_review_date": entry.manual_review_date.value,
        "manual_next_action": entry.manual_next_action.value,
        "manual_confidence": entry.manual_confidence.value,
        "stage_f_label": entry.stage_f_label.value,
        "stage_f_reason": entry.stage_f_reason.value,
        "stage_g_label": entry.stage_g_label.value,
        "stage_g_reason": entry.stage_g_reason.value,
        "promotion_status": promotion_status(entry, master_label),
        "final_candidate_status": entry.final_candidate_status.value,
        "training_label": training_label,
        "training_label_source": entry.training_label_source.value,
        "used_for_training": str(used_for_training).lower(),
        "eligible_for_future_vetter_training": str(eligible).lower(),
        "review_level": review_level(entry),
        "manual_vetted": str(bool(entry.manual_label.value)).lower(),
        "diagnostic_vetted": str(bool(entry.stage_f_label.value or entry.stage_g_label.value)).lower(),
        "autovet_only": str(
            authority == "internal_autovet"
            and not entry.manual_label.value
            and not entry.stage_f_label.value
            and not entry.stage_g_label.value
        ).lower(),
        "master_label": master_label,
        "master_reason": master_reason,
        "master_next_action": master_next_action,
        "decision_authority": authority,
        "has_conflict": str(bool(disagreement_conflicts)).lower(),
        "conflict_reason": "; ".join(disagreement_conflicts),
        "source_files": "; ".join(sorted(entry.source_files)),
        "last_updated": max(entry.last_updated_values).isoformat() if entry.last_updated_values else "",
    }
    return row, conflicts


def write_catalog(rows: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "master_vetted_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_conflicts(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "epic_id",
        "conflict_type",
        "manual_label",
        "autovet_label",
        "stage_f_label",
        "stage_g_label",
        "cnn_score",
        "conflict_severity",
        "conflict_reason",
        "source_files",
    ]
    output_rows = []
    for row in rows:
        reasons = [reason for reason in row["conflict_reason"].split("; ") if reason]
        if row["morphology_positive"] == "true" and is_manual_reject(row["manual_label"]):
            reasons.append("CNN morphology-positive target is manually rejected")
        if not reasons:
            continue
        for reason in reasons:
            output_rows.append(
                {
                    "epic_id": row["epic_id"],
                    "conflict_type": reason,
                    "manual_label": row["manual_label"],
                    "autovet_label": row["autovet_label"],
                    "stage_f_label": row["stage_f_label"],
                    "stage_g_label": row["stage_g_label"],
                    "cnn_score": row["cnn_score"],
                    "conflict_severity": classify_conflict_severity(reason, row, reasons),
                    "conflict_reason": reason,
                    "source_files": row["source_files"],
                }
            )
    with (OUTPUT_DIR / "master_vetted_catalog_conflicts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_missing_fields(rows: list[dict[str, str]]) -> None:
    fields_to_check = [
        "cnn_score",
        "autovet_label",
        "manual_label",
        "stage_f_label",
        "stage_g_label",
        "training_label",
        "master_label",
    ]
    output_rows = []
    for row in rows:
        missing = [field for field in fields_to_check if not row[field]]
        if not missing:
            continue
        output_rows.append(
            {
                "epic_id": row["epic_id"],
                "missing_fields": "; ".join(missing),
                "decision_authority": row["decision_authority"],
                "source_files": row["source_files"],
            }
        )
    with (OUTPUT_DIR / "master_vetted_catalog_missing_fields.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epic_id", "missing_fields", "decision_authority", "source_files"])
        writer.writeheader()
        writer.writerows(output_rows)


def write_label_counts(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["master_label"] for row in rows)
    with (OUTPUT_DIR / "master_vetted_catalog_label_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["master_label", "epic_count"])
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([label, count])


def write_review_level_counts(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["review_level"] for row in rows)
    with (OUTPUT_DIR / "master_vetted_catalog_review_level_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_level", "epic_count"])
        for level in ["auto_only", "cnn_scored_only", "stage_f_reviewed", "stage_g_reviewed", "manually_reviewed", "unresolved"]:
            writer.writerow([level, counts.get(level, 0)])


def write_training_ready(rows: list[dict[str, str]]) -> None:
    with (OUTPUT_DIR / "master_vetted_catalog_training_ready.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["eligible_for_future_vetter_training"] == "true")


def is_candidate_subset_row(row: dict[str, str]) -> bool:
    if row["promotion_status"] == "promoted_or_candidate":
        return True
    if row["master_label"] in {
        "candidate_like",
        "candidate_like_hold",
        "strong_candidate_like",
        "promote_to_deeper_eval",
        "promote_primary_candidate",
        "promote_candidate_alias_check",
        "top_followup_candidate",
        "followup_candidate_with_variability_caution",
    }:
        return True
    return False


def write_candidate_subset(rows: list[dict[str, str]]) -> None:
    with (OUTPUT_DIR / "master_vetted_catalog_candidate_subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if is_candidate_subset_row(row))


def write_summary(rows: list[dict[str, str]], included_sources: list[str]) -> None:
    authority_counts = Counter(row["decision_authority"] for row in rows)
    label_counts = Counter(row["master_label"] for row in rows)
    vetted_count = sum(row["decision_authority"] in {"manual_review", "stage_g", "stage_f"} for row in rows)
    manual_count = sum(bool(row["manual_label"]) for row in rows)
    cnn_count = sum(bool(row["cnn_score"]) for row in rows)
    autovet_count = sum(bool(row["autovet_label"]) for row in rows)
    conflict_count = sum(row["has_conflict"] == "true" for row in rows)
    future_training_count = sum(row["eligible_for_future_vetter_training"] == "true" for row in rows)
    review_level_counts = Counter(row["review_level"] for row in rows)
    lines = [
        "Master vetted catalog summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"unique_epics={len(rows)}",
        f"actually_vetted_epics={vetted_count}",
        f"manual_vetted_epics={manual_count}",
        f"cnn_scored_epics={cnn_count}",
        f"autovet_labeled_epics={autovet_count}",
        f"conflicted_epics={conflict_count}",
        f"eligible_for_future_vetter_training={future_training_count}",
        "",
        "decision_authority_counts",
    ]
    lines.extend(f"{key}={value}" for key, value in sorted(authority_counts.items()))
    lines.extend(["", "master_label_counts"])
    lines.extend(f"{key}={value}" for key, value in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])))
    lines.extend(["", "review_level_counts"])
    lines.extend(f"{key}={value}" for key, value in sorted(review_level_counts.items()))
    lines.extend(["", "included_sources"])
    lines.extend(included_sources)
    (OUTPUT_DIR / "master_vetted_catalog_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(included_sources: list[str]) -> None:
    lines = [
        "Master vetted catalog",
        "",
        "Purpose",
        "One row per EPIC across the curated K2 review corpus, showing what the frozen CNN said, what AutoVet said, what manual review said, and which authority wins.",
        "",
        "Decision authority order",
        "1. Manual vet result",
        "2. Stage G result",
        "3. Stage F result",
        "4. Internal AutoVet diagnostics",
        "5. Frozen CNN morphology score only",
        "",
        "Master decision rules",
        "- Manual review is the highest authority.",
        "- The frozen CNN is preserved as `transit_morphology_score` only.",
        "- `cnn_score > 0.5` means morphology-positive only; it never assigns candidate or planet status by itself.",
        "- Conflicts are not silently resolved. Manual review still wins the master label, and every configured disagreement is emitted to `master_vetted_catalog_conflicts.csv`.",
        "- `eligible_for_future_vetter_training` is true only for manual labels with diagnostic evidence, no unresolved conflict, no synthetic-only source, and no plain `uncertain_hold` label.",
        "",
        "Outputs",
        "- master_vetted_catalog.csv",
        "- master_vetted_catalog_summary.txt",
        "- master_vetted_catalog_conflicts.csv",
        "- master_vetted_catalog_missing_fields.csv",
        "- master_vetted_catalog_label_counts.csv",
        "- master_vetted_catalog_review_level_counts.csv",
        "- master_vetted_catalog_training_ready.csv",
        "- master_vetted_catalog_candidate_subset.csv",
        "- master_vetted_catalog_readme.txt",
        "",
        "Conflict rules",
        "- manual_label disagrees with autovet_label",
        "- manual_label disagrees with stage_f_label",
        "- manual_label disagrees with stage_g_label",
        "- CNN morphology-positive target is manually rejected",
        "- AutoVet candidate_like is manually rejected",
        "- same EPIC has multiple incompatible manual labels",
        "",
        "Review levels",
        "- auto_only",
        "- cnn_scored_only",
        "- stage_f_reviewed",
        "- stage_g_reviewed",
        "- manually_reviewed",
        "- unresolved",
        "",
        "Derived booleans",
        "- manual_vetted: true only when a manual label exists",
        "- diagnostic_vetted: true when a Stage F or Stage G label exists",
        "- autovet_only: true only when decision_authority is internal_autovet and no manual / Stage F / Stage G label exists",
        "",
        "Source scope",
        "The catalog uses curated K2 review sources only, including frozen CNN score tables, Stage I AutoVet outputs, manual Stage I/Stage F review artifacts, Stage G review artifacts, the final ledger, and training label tables.",
        "",
        "Included sources",
        *included_sources,
    ]
    (OUTPUT_DIR / "master_vetted_catalog_readme.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    entries, included_sources = ingest()
    rows = [row_for(entries[epic_id])[0] for epic_id in sorted(entries)]
    write_catalog(rows)
    write_conflicts(rows)
    write_missing_fields(rows)
    write_label_counts(rows)
    write_review_level_counts(rows)
    write_training_ready(rows)
    write_candidate_subset(rows)
    write_summary(rows, included_sources)
    write_readme(included_sources)


if __name__ == "__main__":
    main()
