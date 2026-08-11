from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260601_epic65"
BATCH_ID = "manual_vetting_batch_64"
BATCH_SIZE = 64

UPDATE_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "manual_review_updates" / RUN_ID
MANUAL_LEDGER = UPDATE_DIR / f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
UPDATED_MASTER = UPDATE_DIR / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
UPDATED_RECONCILED = UPDATE_DIR / f"final_candidate_master_ledger_reconciled_manual_review_update_{RUN_ID}.csv"
QUEUE_64 = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "cnn_backfill" / "manual_vetting_priority_queue_next64.csv"

FINAL_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"
CANONICAL_MANUAL_LEDGER = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "manual_vetting_decisions_ledger.csv"
CANONICAL_RECONCILED_LEDGER = (
    ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "final_candidate_master_ledger_reconciled_current.csv"
)

FREEZE_DIR = ROOT / "freezes" / BATCH_ID

OUT_TRAINING_VIEW = ROOT / "manual_64_training_view.csv"
OUT_SUMMARY = ROOT / "manual_vetting_64_summary.txt"
OUT_STAGE_G_QUEUE = ROOT / "manual_vetting_64_stage_g_deeper_eval_queue.csv"
OUT_DISAGREE_CSV = ROOT / "manual_vetting_64_model_disagreement_report.csv"
OUT_DISAGREE_TXT = ROOT / "manual_vetting_64_model_disagreement_report.txt"
OUT_AUTOVET_V2 = ROOT / "autovet_v2_manual_64_scores.csv"
OUT_AUTOVET_V2_EVAL = ROOT / "autovet_v2_manual_64_evaluation.txt"


METRIC_COLUMNS = [
    "queue_rank",
    "cnn_score",
    "morphology_positive",
    "flux_p_science_like",
    "flux_p_top3_mean",
    "flux_p_top10_mean",
    "flux_num_segments",
    "autovet_rank_score",
    "review_priority_score",
    "triage_score_global",
    "triage_step_score",
    "triage_whiteness_score",
    "best_shape_score",
    "best_depth_snr",
    "primary_depth_snr",
    "n_events",
    "n_periods_proposed",
    "n_periods_validated",
    "best_period_days",
    "period_support_count",
    "event_family_count",
    "primary_depth",
    "radius_ratio_sqrt_depth",
    "transit_duration_hours",
    "odd_even_depth_ratio",
    "odd_even_depth_delta_explicit",
    "secondary_to_primary_depth_ratio",
    "secondary_depth_snr",
    "oot_variability_to_depth",
    "alias_best_support_ratio",
    "spike_fraction_2cadence",
    "pilot_noise_like_flag",
    "obvious_low_signal_flag",
    "funnel_bucket",
    "prefilter_rank",
    "prefilter_rank_score",
    "recommended_prefilter_action",
    "prefilter_reason",
    "key_metrics_used",
    "triggered_rules",
]


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clip01(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def label_family(label: str) -> str:
    raw = label.lower()
    if raw in {"candidate_like", "planet_like"}:
        return "positive"
    if raw in {"uncertain_hold", "uncertain_hold_positive"}:
        return "hold"
    return "negative"


def training_class(label: str) -> tuple[str, str, bool]:
    raw = label.lower()
    if raw == "candidate_like":
        return "candidate_like_positive", "science_like", True
    if raw == "planet_like":
        return "planet_like_positive", "science_like", True
    if raw in {"binary_system", "variable_or_possible_eb"}:
        return "false_positive_eb_or_variable", "not_science_like", True
    if raw == "reject_as_noise_or_artifact":
        return "negative_reject_as_noise_or_artifact", "not_science_like", True
    if raw == "noise_or_artifact":
        return "negative_noise_or_artifact", "not_science_like", True
    if raw == "low_priority_negative":
        return "negative_low_priority", "not_science_like", True
    if raw == "uncertain_hold_positive":
        return "excluded_uncertain_hold_positive", "unresolved", False
    if raw == "uncertain_hold":
        return "excluded_uncertain_hold", "unresolved", False
    return f"negative_{raw or 'unknown'}", "not_science_like", True


def stage_g_action(label: str) -> tuple[bool, str]:
    if label.lower() in {"candidate_like", "planet_like"}:
        return True, "promote_to_stage_g_deeper_evaluation"
    return False, "do_not_promote_to_stage_g"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [MANUAL_LEDGER, UPDATED_MASTER, UPDATED_RECONCILED, QUEUE_64, FINAL_LEDGER]:
        if not path.exists():
            raise FileNotFoundError(path)

    manual = pd.read_csv(MANUAL_LEDGER, dtype=str).fillna("")
    queue = pd.read_csv(QUEUE_64, dtype=str).fillna("")
    master = pd.read_csv(UPDATED_MASTER, dtype=str).fillna("")
    reconciled = pd.read_csv(UPDATED_RECONCILED, dtype=str).fillna("")

    if manual["epic_id"].nunique() != BATCH_SIZE or len(manual) != BATCH_SIZE:
        raise RuntimeError(f"Expected {BATCH_SIZE} unique manual decisions, got rows={len(manual)} unique={manual['epic_id'].nunique()}")
    if set(manual["epic_id"]) != set(queue["epic_id"]):
        missing_queue = sorted(set(manual["epic_id"]) - set(queue["epic_id"]))
        missing_manual = sorted(set(queue["epic_id"]) - set(manual["epic_id"]))
        raise RuntimeError(f"Manual ledger and queue mismatch: missing_queue={missing_queue}; missing_manual={missing_manual}")
    if set(manual["epic_id"]) - set(master["epic_id"]):
        raise RuntimeError("Updated master catalog does not contain every manual-vetting EPIC")

    return manual, queue, master, reconciled


def build_training_view(manual: pd.DataFrame, queue: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in METRIC_COLUMNS if col in queue.columns]
    merged = manual.merge(queue[["epic_id", *metric_cols]], on="epic_id", how="left", validate="one_to_one")
    master_cols = [
        col
        for col in [
            "epic_id",
            "autovet_label",
            "autovet_reason",
            "period_gate",
            "odd_even_gate",
            "secondary_eclipse_gate",
            "oot_variability_gate",
            "depth_consistency_gate",
            "event_cluster_gate",
            "alias_gate",
            "thruster_cadence_gate",
            "diagnostic_gate_summary",
            "stage_f_label",
            "stage_f_reason",
            "stage_g_label",
            "stage_g_reason",
            "source_files",
        ]
        if col in master.columns
    ]
    merged = merged.merge(master[master_cols], on="epic_id", how="left", validate="one_to_one", suffixes=("", "_master"))

    classes = merged["manual_label"].map(training_class)
    merged["training_class"] = [item[0] for item in classes]
    merged["science_binary"] = [item[1] for item in classes]
    merged["training_use"] = [bool_text(item[2]) for item in classes]
    merged["training_exclusion_reason"] = np.where(
        merged["training_use"].eq("false"),
        "excluded_uncertain_manual_hold",
        "",
    )
    actions = merged["manual_label"].map(stage_g_action)
    merged["stage_g_deeper_eval"] = [bool_text(item[0]) for item in actions]
    merged["stage_g_deeper_eval_action"] = [item[1] for item in actions]
    merged["manual_label_family"] = merged["manual_label"].map(label_family)
    merged["freeze_batch_id"] = BATCH_ID

    front = [
        "freeze_batch_id",
        "epic_id",
        "manual_label",
        "manual_label_family",
        "training_class",
        "science_binary",
        "training_use",
        "training_exclusion_reason",
        "stage_g_deeper_eval",
        "stage_g_deeper_eval_action",
        "manual_next_action",
        "manual_confidence",
        "manual_reason",
        "reviewed_at",
        "reviewer",
        "cnn_manual_conflict",
        "conflict_reason",
    ]
    rest = [col for col in merged.columns if col not in front]
    return merged[front + rest].sort_values("queue_rank", key=lambda s: numeric(s).fillna(999999)).reset_index(drop=True)


def final_ledger_rows(training_view: pd.DataFrame, existing_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in training_view.iterrows():
        stage_g, action = stage_g_action(clean(row["manual_label"]))
        training_label = clean(row["training_class"]) if clean(row["training_use"]) == "true" else clean(row["manual_label"])
        science = clean(row["science_binary"])
        ledger_row = {col: "" for col in existing_columns}
        ledger_row.update(
            {
                "epic_id": clean(row["epic_id"]),
                "source_batch": BATCH_ID,
                "best_period_days": clean(row.get("best_period_days")),
                "stage_g_final_recommendation": action if stage_g else "",
                "final_recommendation": action,
                "stage_f_label": clean(row.get("stage_f_label")),
                "stage_h_label": clean(row["manual_label"]),
                "visual_label": clean(row["manual_label"]),
                "final_candidate_status": clean(row["manual_label"]),
                "review_bin": BATCH_ID,
                "status_reason": clean(row["manual_reason"]),
                "visual_notes": clean(row["manual_reason"]),
                "reviewer": clean(row["reviewer"]),
                "reviewed_at": clean(row["reviewed_at"]),
                "recommended_next_action": clean(row["manual_next_action"]),
                "stage_h_notes": clean(row["manual_reason"]),
                "stage_h_status": clean(row["manual_label"]),
                "stage_h_training_label_v3": training_label,
                "stage_h_science_binary_v3": science,
                "stage_h_ledger_status": action,
                "stage_h_ledger_priority": "stage_g_deeper_eval" if stage_g else clean(row["training_class"]),
                "stage_h_reason": clean(row["manual_reason"]),
                "stage_h_reviewed_at": clean(row["reviewed_at"]),
                "stage_h_reviewer": clean(row["reviewer"]),
            }
        )
        rows.append(ledger_row)
    return pd.DataFrame(rows, columns=existing_columns)


def update_final_candidate_ledger(training_view: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(FINAL_LEDGER, dtype=str).fillna("")
    backup = FREEZE_DIR / "source_final_candidate_master_ledger_before_manual_vetting_batch_64.csv"
    if not backup.exists():
        shutil.copy2(FINAL_LEDGER, backup)

    new_rows = final_ledger_rows(training_view, list(existing.columns))
    keep = existing[~existing["epic_id"].isin(set(training_view["epic_id"]))].copy()
    updated = pd.concat([keep, new_rows], ignore_index=True)
    updated.to_csv(FINAL_LEDGER, index=False)
    updated.to_csv(FREEZE_DIR / "final_candidate_master_ledger_after_manual_vetting_batch_64.csv", index=False)
    return updated


def metric_score(row: pd.Series) -> tuple[float, str, str]:
    cnn = clip01(row.get("cnn_score"))
    period = clip01(float(row.get("period_support_count", 0) or 0) / 3.0) if finite(row.get("period_support_count")) else 0.0
    events = clip01(float(row.get("event_family_count", 0) or 0) / 12.0) if finite(row.get("event_family_count")) else 0.0
    shape = clip01((float(row.get("best_shape_score")) - 0.60) / 0.25) if finite(row.get("best_shape_score")) else 0.0
    snr_raw = row.get("primary_depth_snr") if finite(row.get("primary_depth_snr")) else row.get("best_depth_snr")
    snr = clip01(np.log1p(float(snr_raw)) / np.log1p(30.0)) if finite(snr_raw) else 0.0

    odd_ratio = row.get("odd_even_depth_ratio")
    odd_delta = row.get("odd_even_depth_delta_explicit")
    odd_clean = 0.5
    if finite(odd_delta):
        odd_clean = clip01(1.0 - float(odd_delta) / 0.35)
    elif finite(odd_ratio):
        odd_clean = clip01(1.0 - abs(float(odd_ratio) - 1.0) / 0.45)

    secondary = row.get("secondary_to_primary_depth_ratio")
    secondary_snr = row.get("secondary_depth_snr")
    secondary_clean = 1.0
    if finite(secondary):
        secondary_clean = min(secondary_clean, clip01(1.0 - float(secondary) / 0.35))
    if finite(secondary_snr):
        secondary_clean = min(secondary_clean, clip01(1.0 - float(secondary_snr) / 7.0))

    oot = row.get("oot_variability_to_depth")
    oot_clean = clip01(1.0 - float(oot) / 0.75) if finite(oot) else 0.5
    alias = row.get("alias_best_support_ratio")
    alias_clean = clip01(1.0 - float(alias) / 0.75) if finite(alias) else 0.5

    depth = row.get("primary_depth")
    radius = row.get("radius_ratio_sqrt_depth")
    duration = row.get("transit_duration_hours")
    duration_ok = 1.0 if not finite(duration) or 0.5 <= float(duration) <= 10.0 else 0.0

    base = (
        0.28 * cnn
        + 0.14 * period
        + 0.10 * events
        + 0.10 * shape
        + 0.10 * snr
        + 0.10 * odd_clean
        + 0.07 * secondary_clean
        + 0.06 * oot_clean
        + 0.03 * alias_clean
        + 0.02 * duration_ok
    )

    flags: list[str] = []
    if finite(depth) and float(depth) >= 0.02:
        flags.append("deep_primary")
    if finite(radius) and float(radius) >= 0.15:
        flags.append("large_radius_ratio")
    if finite(secondary) and float(secondary) >= 0.35:
        flags.append("secondary_like")
    if finite(odd_delta) and float(odd_delta) >= 0.30:
        flags.append("odd_even_mismatch")
    if finite(oot) and float(oot) >= 0.75:
        flags.append("high_oot_variability")
    if finite(alias) and float(alias) >= 0.75:
        flags.append("strong_alias_support")
    validation_text = " ".join(
        clean(row.get(column)).lower()
        for column in [
            "autovet_reason",
            "stage_f_reason",
            "triggered_rules",
            "diagnostic_gate_summary",
            "prefilter_reason",
            "recommended_prefilter_action",
        ]
    )

    if "event_spacing_fallback" in validation_text or "no_saved_period_support" in validation_text:
        flags.append("fallback_period_context")
    if "stage f likely eb" in validation_text or "likely eb" in validation_text:
        flags.append("validation_eb_context")
    if "stage f reject" in validation_text or "artifact/reject" in validation_text:
        flags.append("validation_reject_context")
    if "odd_even_ratio=0.350" in validation_text or "odd_even_delta=0.962" in validation_text:
        flags.append("validation_odd_even_failure")
    if "alias_risk=high" in validation_text:
        flags.append("validation_high_alias")
    if finite(duration) and float(duration) > 10.0:
        flags.append("long_duration")

    penalty = 0.0
    if "deep_primary" in flags or "large_radius_ratio" in flags:
        penalty += 0.10
    if "secondary_like" in flags or "odd_even_mismatch" in flags:
        penalty += 0.12
    if "high_oot_variability" in flags:
        penalty += 0.08
    if "strong_alias_support" in flags:
        penalty += 0.08
    if "fallback_period_context" in flags:
        penalty += 0.06
    if "long_duration" in flags:
        penalty += 0.04
    if "validation_eb_context" in flags or "validation_odd_even_failure" in flags:
        penalty += 0.16
    if "validation_reject_context" in flags:
        penalty += 0.18
    if "validation_high_alias" in flags:
        penalty += 0.06

    score = float(np.clip(base - penalty, 0.0, 1.0))
    eb_like = (
        any(
            flag in flags
            for flag in ["deep_primary", "large_radius_ratio", "secondary_like", "odd_even_mismatch", "validation_eb_context", "validation_odd_even_failure"]
        )
        and (
            "high_oot_variability" in flags
            or "strong_alias_support" in flags
            or "fallback_period_context" in flags
            or "validation_high_alias" in flags
            or "validation_eb_context" in flags
        )
    )
    no_period_but_strong_signal = period < 0.34 and cnn >= 0.65 and events >= 0.75 and shape >= 0.60 and snr >= 0.45
    weak_support = score < 0.42 or (period < 0.34 and not no_period_but_strong_signal) or events < 0.25

    if eb_like:
        label = "autovet_v2_false_positive_eb_or_variable"
    elif "validation_reject_context" in flags:
        label = "autovet_v2_reject_noise_or_artifact"
    elif weak_support:
        label = "autovet_v2_reject_noise_or_artifact"
    elif score >= 0.72 and not flags:
        label = "autovet_v2_candidate_priority"
    elif score >= 0.58 or no_period_but_strong_signal:
        label = "autovet_v2_review_signal_with_caveats"
    else:
        label = "autovet_v2_hold_for_review"

    components = (
        f"cnn={cnn:.3f}; period={period:.3f}; events={events:.3f}; shape={shape:.3f}; "
        f"snr={snr:.3f}; odd_even_clean={odd_clean:.3f}; secondary_clean={secondary_clean:.3f}; "
        f"oot_clean={oot_clean:.3f}; alias_clean={alias_clean:.3f}; duration_ok={duration_ok:.3f}; penalty={penalty:.3f}"
    )
    return score, label, "; ".join(flags) if flags else "no_major_penalty_flags", components


def build_autovet_v2(training_view: pd.DataFrame) -> pd.DataFrame:
    out = training_view.copy()
    scored = [metric_score(row) for _, row in out.iterrows()]
    out["autovet_v2_score"] = [round(item[0], 6) for item in scored]
    out["autovet_v2_label"] = [item[1] for item in scored]
    out["autovet_v2_flags"] = [item[2] for item in scored]
    out["autovet_v2_components"] = [item[3] for item in scored]
    out["autovet_v2_positive_call"] = out["autovet_v2_label"].isin(
        {"autovet_v2_candidate_priority"}
    ).map(bool_text)
    out["autovet_v2_review_call"] = out["autovet_v2_label"].isin(
        {"autovet_v2_candidate_priority", "autovet_v2_review_signal_with_caveats", "autovet_v2_hold_for_review"}
    ).map(bool_text)
    out["manual_positive_for_eval"] = out["manual_label"].isin({"candidate_like", "planet_like"}).map(bool_text)
    out["manual_hold_for_eval"] = out["manual_label"].isin({"uncertain_hold", "uncertain_hold_positive"}).map(bool_text)
    return out


def build_disagreement_report(training_view: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    reject_labels = {
        "binary_system",
        "variable_or_possible_eb",
        "noise_or_artifact",
        "reject_as_noise_or_artifact",
        "low_priority_negative",
    }
    hold_or_candidate = {"candidate_like", "planet_like", "uncertain_hold", "uncertain_hold_positive"}
    for _, row in training_view.iterrows():
        label = clean(row["manual_label"])
        reason = clean(row["manual_reason"]).lower()
        cnn = float(row["cnn_score"]) if finite(row.get("cnn_score")) else np.nan
        snr = row.get("primary_depth_snr") if finite(row.get("primary_depth_snr")) else row.get("best_depth_snr")
        snr_value = float(snr) if finite(snr) else np.nan
        odd_ratio = float(row["odd_even_depth_ratio"]) if finite(row.get("odd_even_depth_ratio")) else np.nan
        odd_delta = float(row["odd_even_depth_delta_explicit"]) if finite(row.get("odd_even_depth_delta_explicit")) else np.nan
        ratio_match = re.search(r"odd/even[^.;]*(?:ratio|agreement)[^0-9]*(0\.\d+|1\.\d+)", reason)
        delta_match = re.search(r"(?:explicit delta|delta)[^0-9]*(0\.\d+)", reason)
        if not finite(odd_ratio) and ratio_match:
            odd_ratio = float(ratio_match.group(1))
        if not finite(odd_delta) and delta_match:
            odd_delta = float(delta_match.group(1))
        reason_good_odd_even = any(
            phrase in reason
            for phrase in [
                "odd/even agreement is excellent",
                "odd/even excellent",
                "odd/even is good",
                "odd/even good",
                "odd/even fairly good",
                "odd/even is acceptable",
                "odd/even acceptable",
                "odd/even consistency is excellent",
                "odd/even consistency is acceptable",
            ]
        )
        good_odd_even = (finite(odd_ratio) and 0.85 <= odd_ratio <= 1.15) or (finite(odd_delta) and odd_delta <= 0.15)
        categories: list[str] = []
        if label in reject_labels and finite(cnn) and cnn >= 0.85:
            categories.append("high_cnn_score_manual_reject")
        if label in hold_or_candidate and finite(cnn) and cnn <= 0.70:
            categories.append("low_cnn_score_manual_candidate_or_hold")
        if label in reject_labels and finite(snr_value) and snr_value >= 50:
            categories.append("high_snr_manual_reject")
        if label in reject_labels and (good_odd_even or reason_good_odd_even):
            categories.append("good_odd_even_manual_reject")
        if label in reject_labels and (
            "event_spacing_fallback" in reason
            or "fallback period" in reason
            or "candidate period count" in reason
            or "period source = event_spacing_fallback" in reason
        ):
            categories.append("fallback_period_dominated_reject")
        if label in {"binary_system", "variable_or_possible_eb"}:
            categories.append("eb_variable_false_positive")

        for category in categories:
            rows.append(
                {
                    "category": category,
                    "epic_id": clean(row["epic_id"]),
                    "manual_label": label,
                    "training_class": clean(row["training_class"]),
                    "cnn_score": clean(row.get("cnn_score")),
                    "primary_depth_snr": clean(snr_value) if finite(snr_value) else "",
                    "odd_even_depth_ratio": clean(row.get("odd_even_depth_ratio")),
                    "odd_even_depth_delta_explicit": clean(row.get("odd_even_depth_delta_explicit")),
                    "oot_variability_to_depth": clean(row.get("oot_variability_to_depth")),
                    "alias_best_support_ratio": clean(row.get("alias_best_support_ratio")),
                    "best_period_days": clean(row.get("best_period_days")),
                    "manual_reason": clean(row["manual_reason"]),
                }
            )
    return pd.DataFrame(rows)


def update_ledgers(manual: pd.DataFrame, reconciled: pd.DataFrame, training_view: pd.DataFrame) -> pd.DataFrame:
    canonical = manual.copy()
    canonical["freeze_batch_id"] = BATCH_ID
    canonical["training_class"] = training_view.set_index("epic_id").loc[canonical["epic_id"], "training_class"].to_numpy()
    canonical["training_use"] = training_view.set_index("epic_id").loc[canonical["epic_id"], "training_use"].to_numpy()
    canonical["stage_g_deeper_eval"] = training_view.set_index("epic_id").loc[canonical["epic_id"], "stage_g_deeper_eval"].to_numpy()
    canonical.to_csv(CANONICAL_MANUAL_LEDGER, index=False)
    reconciled.to_csv(CANONICAL_RECONCILED_LEDGER, index=False)
    canonical.to_csv(FREEZE_DIR / "manual_vetting_decisions_ledger_manual_vetting_batch_64.csv", index=False)
    reconciled.to_csv(FREEZE_DIR / "final_candidate_master_ledger_reconciled_manual_vetting_batch_64.csv", index=False)
    return canonical


def write_summary(
    training_view: pd.DataFrame,
    updated_final: pd.DataFrame,
    autovet_v2: pd.DataFrame,
    disagreements: pd.DataFrame,
    canonical_manual: pd.DataFrame,
) -> None:
    label_counts = Counter(training_view["manual_label"])
    training_counts = Counter(training_view["training_class"])
    stage_g_count = int(training_view["stage_g_deeper_eval"].eq("true").sum())
    training_use_count = int(training_view["training_use"].eq("true").sum())
    hold_excluded = int(training_view["training_use"].eq("false").sum())
    v2_counts = Counter(autovet_v2["autovet_v2_label"])

    eval_rows = autovet_v2[autovet_v2["manual_hold_for_eval"].eq("false")].copy()
    pred_pos = eval_rows["autovet_v2_positive_call"].eq("true")
    actual_pos = eval_rows["manual_positive_for_eval"].eq("true")
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    review_truth = autovet_v2["manual_label"].isin({"candidate_like", "planet_like", "uncertain_hold", "uncertain_hold_positive"})
    review_call = autovet_v2["autovet_v2_review_call"].eq("true")
    review_capture = int((review_truth & review_call).sum())
    review_total = int(review_truth.sum())
    review_capture_rate = review_capture / review_total if review_total else 0.0

    lines = [
        "Manual vetting batch 64 summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"freeze_batch_id={BATCH_ID}",
        f"source_manual_ledger={MANUAL_LEDGER.relative_to(ROOT).as_posix()}",
        f"source_queue_64={QUEUE_64.relative_to(ROOT).as_posix()}",
        f"manual_epics={len(training_view)}",
        f"manual_unique_epics={training_view['epic_id'].nunique()}",
        f"canonical_final_candidate_master_ledger_rows={len(updated_final)}",
        f"canonical_manual_decisions_ledger_rows={len(canonical_manual)}",
        "",
        "Safety",
        "- Did not retrain the CNN.",
        "- AutoVet v2 is deterministic and uses frozen CNN score plus validation metrics only.",
        "- uncertain_hold and uncertain_hold_positive are excluded from training_use.",
        "- Only candidate_like / planet_like rows are flagged for Stage G deeper evaluation.",
        "",
        "Exact manual_label counts across all 64",
    ]
    lines.extend(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    lines.extend(["", "Training class counts"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(training_counts.items()))
    lines.extend(
        [
            "",
            f"training_use_true={training_use_count}",
            f"training_use_false_uncertain_holds={hold_excluded}",
            f"stage_g_deeper_eval_true={stage_g_count}",
            "",
            "AutoVet v2 label counts",
        ]
    )
    lines.extend(f"- {label}: {count}" for label, count in sorted(v2_counts.items()))
    lines.extend(
        [
            "",
            "AutoVet v2 binary evaluation, holds excluded",
            f"- eval_rows={len(eval_rows)}",
            f"- true_positive={tp}",
            f"- false_positive={fp}",
            f"- false_negative={fn}",
            f"- true_negative={tn}",
            f"- precision={precision:.3f}",
            f"- recall={recall:.3f}",
            f"- candidate_or_hold_review_capture={review_capture}/{review_total} ({review_capture_rate:.3f})",
            "",
            "Disagreement category counts",
        ]
    )
    if len(disagreements):
        for label, count in sorted(Counter(disagreements["category"]).items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none: 0")
    lines.extend(
        [
            "",
            "Outputs",
            f"- {OUT_TRAINING_VIEW.relative_to(ROOT).as_posix()}",
            f"- {OUT_STAGE_G_QUEUE.relative_to(ROOT).as_posix()}",
            f"- {OUT_AUTOVET_V2.relative_to(ROOT).as_posix()}",
            f"- {OUT_AUTOVET_V2_EVAL.relative_to(ROOT).as_posix()}",
            f"- {OUT_DISAGREE_CSV.relative_to(ROOT).as_posix()}",
            f"- {OUT_DISAGREE_TXT.relative_to(ROOT).as_posix()}",
            f"- {FINAL_LEDGER.relative_to(ROOT).as_posix()}",
            f"- {CANONICAL_MANUAL_LEDGER.relative_to(ROOT).as_posix()}",
            f"- {FREEZE_DIR.relative_to(ROOT).as_posix()}",
        ]
    )
    text = "\n".join(lines) + "\n"
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    (FREEZE_DIR / "manual_vetting_64_summary.txt").write_text(text, encoding="utf-8")


def write_disagreement_text(disagreements: pd.DataFrame) -> None:
    lines = [
        "Manual 64 model disagreement report",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if disagreements.empty:
        lines.append("No disagreement rows matched configured categories.")
    else:
        for category, group in disagreements.groupby("category", sort=True):
            lines.append(category)
            lines.append(f"count={len(group)}")
            for _, row in group.sort_values("epic_id").iterrows():
                lines.append(
                    "- "
                    + clean(row["epic_id"])
                    + f": label={clean(row['manual_label'])}; cnn={clean(row['cnn_score'])}; "
                    + f"snr={clean(row['primary_depth_snr'])}; odd_even={clean(row['odd_even_depth_ratio'])}"
                )
            lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    OUT_DISAGREE_TXT.write_text(text, encoding="utf-8")
    (FREEZE_DIR / "manual_vetting_64_model_disagreement_report.txt").write_text(text, encoding="utf-8")


def write_autovet_eval(autovet_v2: pd.DataFrame) -> None:
    eval_rows = autovet_v2[autovet_v2["manual_hold_for_eval"].eq("false")].copy()
    pred_pos = eval_rows["autovet_v2_positive_call"].eq("true")
    actual_pos = eval_rows["manual_positive_for_eval"].eq("true")
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    review_truth = autovet_v2["manual_label"].isin({"candidate_like", "planet_like", "uncertain_hold", "uncertain_hold_positive"})
    review_call = autovet_v2["autovet_v2_review_call"].eq("true")
    review_capture = int((review_truth & review_call).sum())
    review_total = int(review_truth.sum())
    review_capture_rate = review_capture / review_total if review_total else 0.0
    lines = [
        "AutoVet v2 manual-64 deterministic evaluation",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "scope=manual_vetting_batch_64",
        "training=not_run",
        "cnn_role=frozen transit_morphology_score only",
        "",
        "Confusion matrix, holds excluded",
        f"true_positive={tp}",
        f"false_positive={fp}",
        f"false_negative={fn}",
        f"true_negative={tn}",
        f"precision={precision:.3f}",
        f"recall={recall:.3f}",
        f"candidate_or_hold_review_capture={review_capture}/{review_total}",
        f"candidate_or_hold_review_capture_rate={review_capture_rate:.3f}",
        "",
        "Label counts",
    ]
    lines.extend(f"- {label}: {count}" for label, count in sorted(Counter(autovet_v2["autovet_v2_label"]).items()))
    lines.extend(
        [
            "",
            "Positive prediction labels",
            "- autovet_v2_candidate_priority",
            "",
            "Review, not promotion, labels",
            "- autovet_v2_review_signal_with_caveats",
            "- autovet_v2_hold_for_review",
        ]
    )
    text = "\n".join(lines) + "\n"
    OUT_AUTOVET_V2_EVAL.write_text(text, encoding="utf-8")
    (FREEZE_DIR / "autovet_v2_manual_64_evaluation.txt").write_text(text, encoding="utf-8")


def copy_freeze_artifacts(training_view: pd.DataFrame, manual: pd.DataFrame, master: pd.DataFrame) -> None:
    manual.to_csv(FREEZE_DIR / "manual_vetting_batch_64_decisions.csv", index=False)
    training_view.to_csv(FREEZE_DIR / "manual_64_training_view.csv", index=False)
    master[master["epic_id"].isin(set(manual["epic_id"]))].to_csv(
        FREEZE_DIR / "manual_vetting_batch_64_master_catalog_view.csv",
        index=False,
    )
    shutil.copy2(MANUAL_LEDGER, FREEZE_DIR / MANUAL_LEDGER.name)
    shutil.copy2(QUEUE_64, FREEZE_DIR / QUEUE_64.name)


def write_manifest(files: list[Path]) -> None:
    manifest = {
        "freeze_batch_id": BATCH_ID,
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manual_epic_count": BATCH_SIZE,
        "did_not_retrain_cnn": True,
        "canonical_updates": {
            "final_candidate_master_ledger": str(FINAL_LEDGER),
            "manual_decisions_ledger": str(CANONICAL_MANUAL_LEDGER),
            "reconciled_current_ledger": str(CANONICAL_RECONCILED_LEDGER),
        },
        "outputs": [str(path) for path in files],
    }
    (FREEZE_DIR / "manual_vetting_batch_64_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    manual, queue, master, reconciled = read_inputs()
    training_view = build_training_view(manual, queue, master)
    training_view.to_csv(OUT_TRAINING_VIEW, index=False)

    stage_g_queue = training_view[training_view["stage_g_deeper_eval"].eq("true")].copy()
    stage_g_queue.to_csv(OUT_STAGE_G_QUEUE, index=False)

    updated_final = update_final_candidate_ledger(training_view)
    canonical_manual = update_ledgers(manual, reconciled, training_view)

    autovet_v2 = build_autovet_v2(training_view)
    autovet_v2.to_csv(OUT_AUTOVET_V2, index=False)
    write_autovet_eval(autovet_v2)

    disagreements = build_disagreement_report(training_view)
    disagreements.to_csv(OUT_DISAGREE_CSV, index=False)
    write_disagreement_text(disagreements)

    write_summary(training_view, updated_final, autovet_v2, disagreements, canonical_manual)
    copy_freeze_artifacts(training_view, manual, master)
    for source, target_name in [
        (OUT_TRAINING_VIEW, "manual_64_training_view.csv"),
        (OUT_STAGE_G_QUEUE, "manual_vetting_64_stage_g_deeper_eval_queue.csv"),
        (OUT_DISAGREE_CSV, "manual_vetting_64_model_disagreement_report.csv"),
        (OUT_AUTOVET_V2, "autovet_v2_manual_64_scores.csv"),
    ]:
        shutil.copy2(source, FREEZE_DIR / target_name)

    outputs = [
        OUT_TRAINING_VIEW,
        OUT_SUMMARY,
        OUT_STAGE_G_QUEUE,
        OUT_DISAGREE_CSV,
        OUT_DISAGREE_TXT,
        OUT_AUTOVET_V2,
        OUT_AUTOVET_V2_EVAL,
        FINAL_LEDGER,
        CANONICAL_MANUAL_LEDGER,
        CANONICAL_RECONCILED_LEDGER,
        FREEZE_DIR / "manual_vetting_batch_64_manifest.json",
    ]
    write_manifest(outputs)
    print(
        json.dumps(
            {
                "freeze_batch_id": BATCH_ID,
                "manual_epics": len(training_view),
                "label_counts": dict(sorted(Counter(training_view["manual_label"]).items())),
                "training_use_true": int(training_view["training_use"].eq("true").sum()),
                "stage_g_deeper_eval_true": int(training_view["stage_g_deeper_eval"].eq("true").sum()),
                "autovet_v2_counts": dict(sorted(Counter(autovet_v2["autovet_v2_label"]).items())),
                "disagreement_rows": int(len(disagreements)),
                "did_not_retrain_cnn": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
