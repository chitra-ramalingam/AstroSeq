from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"
REVIEW_SHEET = BATCH_DIR / "review_sheet.csv"
OUT_LEDGER = BATCH_DIR / "autovet_hold_validation_ledger.csv"
OUT_SUMMARY = BATCH_DIR / "autovet_hold_validation_summary.txt"

REQUIRED_ARTIFACTS = [
    "validation_summary.json",
    "events.csv",
    "period_candidates.csv",
    "phase_0_folded.png",
    "phase_05_secondary_check.png",
    "odd_even_zoom.png",
    "alias_period_comparison.png",
]

OUTPUT_COLUMNS = [
    "epic_id",
    "autovet_label",
    "stage_f_validation_label",
    "recommended_ledger_label",
    "recommended_training_label",
    "training_safe",
    "promotion_safe",
    "rejection_safe",
    "human_review_required",
    "label_update_allowed",
    "primary_period",
    "primary_snr",
    "primary_depth",
    "num_events",
    "odd_even_ratio",
    "secondary_to_primary_ratio",
    "alias_risk_flag",
    "eb_risk_flag",
    "artifact_risk_flag",
    "recommended_next_action",
    "reason",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fmt(value: Any, digits: int = 4) -> str:
    value = as_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_completeness(epic_dir: Path) -> tuple[bool, list[str]]:
    missing = []
    for name in REQUIRED_ARTIFACTS:
        path = epic_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(name)
    return len(missing) == 0, missing


def value_from(row: pd.Series, validation: dict[str, Any], key: str) -> Any:
    if key in validation and validation[key] is not None:
        return validation[key]
    return row.get(key)


def classify_hold(row: pd.Series) -> dict[str, Any]:
    epic_id = str(row["epic_id"])
    epic_dir = BATCH_DIR / epic_id
    complete, missing = artifact_completeness(epic_dir)
    summary = read_json(epic_dir / "validation_summary.json")
    validation = summary.get("validation", {})

    autovet_label = str(row.get("autovet_label", ""))
    stage_f_label = str(value_from(row, validation, "stage_f_label") or "")
    stage_f_reason = str(value_from(row, validation, "stage_f_reason") or "")
    primary_period = as_float(value_from(row, validation, "best_period_days"))
    primary_snr = as_float(value_from(row, validation, "primary_depth_snr"))
    primary_depth = as_float(value_from(row, validation, "primary_depth"))
    num_events = int(as_float(row.get("period_support_count"))) if np.isfinite(as_float(row.get("period_support_count"))) else 0
    if num_events <= 0 and np.isfinite(as_float(summary.get("family_event_count"))):
        num_events = int(as_float(summary.get("family_event_count")))
    odd_even_ratio = as_float(value_from(row, validation, "odd_even_depth_ratio"))
    odd_even_delta = as_float(value_from(row, validation, "odd_even_depth_delta_explicit"))
    secondary_ratio = as_float(value_from(row, validation, "secondary_to_primary_depth_ratio"))
    secondary_snr = as_float(value_from(row, validation, "secondary_depth_snr"))
    alias_ratio = as_float(value_from(row, validation, "alias_best_support_ratio"))
    alias_risk = str(value_from(row, validation, "alias_risk") or "").lower()
    original_validated = as_float(row.get("n_periods_validated"))
    original_reason = str(row.get("triggered_rules", "")) + " " + str(row.get("prefilter_reason", ""))

    missing_or_weak_period = (
        (np.isfinite(original_validated) and original_validated < 1)
        or "no_reliable_period_support" in original_reason
        or "no_saved_period_support" in original_reason
    )
    weak_primary = (np.isfinite(primary_snr) and primary_snr < 5.0) or "not significant" in stage_f_reason.lower()
    alias_risk_flag = alias_risk in {"moderate", "high"} or (np.isfinite(alias_ratio) and alias_ratio >= 0.85)
    severe_odd_even = (np.isfinite(odd_even_ratio) and odd_even_ratio < 0.55) or (np.isfinite(odd_even_delta) and odd_even_delta >= 0.70)
    strong_secondary = (np.isfinite(secondary_ratio) and secondary_ratio >= 0.25) or (np.isfinite(secondary_snr) and secondary_snr >= 7.0)
    deep_signal = np.isfinite(primary_depth) and primary_depth > 0.02
    eb_risk_flag = bool(severe_odd_even or strong_secondary or (deep_signal and alias_risk_flag))
    artifact_risk_flag = bool(stage_f_label == "stage_f_reject" and (weak_primary or num_events < 3))
    evidence_incomplete = (not complete) or not validation or not np.isfinite(primary_period)

    recommended_ledger_label = "needs_period_review"
    recommended_training_label = "do_not_train_yet"
    training_safe = False
    promotion_safe = False
    rejection_safe = False
    recommended_next_action = "manual_hold_review_before_label_update"
    reasons: list[str] = []

    if evidence_incomplete:
        recommended_next_action = "regenerate_validation_artifacts"
        reasons.append("validation evidence incomplete: " + ", ".join(missing or ["missing required metrics"]))
    elif missing_or_weak_period:
        recommended_ledger_label = "needs_period_review"
        recommended_training_label = "do_not_train_yet"
        recommended_next_action = "run_period_search_before_label_update"
        reasons.append("original AutoVet evidence had missing/weak saved period support")
    elif stage_f_label == "stage_f_hold":
        recommended_ledger_label = "uncertain_hold"
        recommended_training_label = "uncertain_hold"
        recommended_next_action = "manual_hold_review_before_label_update"
        reasons.append("Stage F hold remains uncertain; do not promote from hold audit")
    elif stage_f_label == "stage_f_reject":
        recommended_ledger_label = "reject_or_low_priority_negative"
        recommended_training_label = "noise_or_artifact" if artifact_risk_flag and alias_risk_flag and weak_primary else "low_priority_negative"
        rejection_safe = bool(artifact_risk_flag)
        training_safe = bool(rejection_safe)
        recommended_next_action = "add_to_reject_audit_queue_before_training"
        reasons.append("Stage F reject in hold audit; route to reject audit before any label update")
    else:
        recommended_next_action = "manual_hold_review_before_label_update"
        reasons.append(f"Stage F label {stage_f_label or 'missing'} is not promotion-safe in hold audit")

    if eb_risk_flag:
        reasons.append(
            f"EB risk features: odd_even_ratio={fmt(odd_even_ratio)}, "
            f"odd_even_delta={fmt(odd_even_delta)}, secondary_ratio={fmt(secondary_ratio)}, "
            f"alias_risk={alias_risk or 'NA'}"
        )
    if artifact_risk_flag:
        reasons.append(f"artifact/reject features: primary_snr={fmt(primary_snr)}, num_events={num_events}")

    return {
        "epic_id": epic_id,
        "autovet_label": autovet_label,
        "stage_f_validation_label": stage_f_label,
        "recommended_ledger_label": recommended_ledger_label,
        "recommended_training_label": recommended_training_label,
        "training_safe": bool(training_safe),
        "promotion_safe": bool(promotion_safe),
        "rejection_safe": bool(rejection_safe),
        "human_review_required": True,
        "label_update_allowed": False,
        "primary_period": primary_period,
        "primary_snr": primary_snr,
        "primary_depth": primary_depth,
        "num_events": int(num_events),
        "odd_even_ratio": odd_even_ratio,
        "secondary_to_primary_ratio": secondary_ratio,
        "alias_risk_flag": bool(alias_risk_flag),
        "eb_risk_flag": bool(eb_risk_flag),
        "artifact_risk_flag": bool(artifact_risk_flag),
        "recommended_next_action": recommended_next_action,
        "reason": "; ".join(reasons),
        "artifact_complete": bool(complete),
        "missing_artifacts": "; ".join(missing),
        "alias_best_support_ratio": alias_ratio,
        "alias_risk": alias_risk,
        "n_periods_validated_original": original_validated,
        "validation_summary_json": str(epic_dir / "validation_summary.json"),
        "events_csv": str(epic_dir / "events.csv"),
        "period_candidates_csv": str(epic_dir / "period_candidates.csv"),
    }


def write_summary(ledger: pd.DataFrame) -> None:
    stage_counts = ledger["stage_f_validation_label"].value_counts()
    period_search_epics = ledger.loc[
        ledger["recommended_next_action"].eq("run_period_search_before_label_update"), "epic_id"
    ].astype(str).tolist()
    reject_epics = ledger.loc[
        ledger["recommended_next_action"].eq("add_to_reject_audit_queue_before_training"), "epic_id"
    ].astype(str).tolist()
    lines = [
        "Stage I AutoVet v1 hold validation summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input: {rel(REVIEW_SHEET)}",
        f"Output: {rel(OUT_LEDGER)}",
        "",
        f"Total hold rows processed: {len(ledger)}",
        f"stage_f_hold count: {int(stage_counts.get('stage_f_hold', 0))}",
        f"stage_f_reject count: {int(stage_counts.get('stage_f_reject', 0))}",
        f"promotion_safe count: {int(ledger['promotion_safe'].map(as_bool).sum())}",
        f"rejection_safe count: {int(ledger['rejection_safe'].map(as_bool).sum())}",
        f"training_safe count: {int(ledger['training_safe'].map(as_bool).sum())}",
        f"Rows requiring human review: {int(ledger['human_review_required'].map(as_bool).sum())}",
        f"Automatic label updates allowed: {int(ledger['label_update_allowed'].map(as_bool).sum())}",
        f"EPICs needing period search: {', '.join(period_search_epics) if period_search_epics else 'none'}",
        f"EPICs suitable for reject audit: {', '.join(reject_epics) if reject_epics else 'none'}",
        "",
        "No training labels, final candidate ledger rows, or model artifacts were modified.",
        "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not REVIEW_SHEET.exists():
        raise FileNotFoundError(REVIEW_SHEET)
    review = pd.read_csv(REVIEW_SHEET)
    review["epic_id"] = review["epic_id"].astype(str)
    ledger = pd.DataFrame([classify_hold(row) for _, row in review.iterrows()])
    ledger = ledger[OUTPUT_COLUMNS + [c for c in ledger.columns if c not in OUTPUT_COLUMNS]]
    ledger.to_csv(OUT_LEDGER, index=False)
    write_summary(ledger)

    print(f"Hold validation ledger: {rel(OUT_LEDGER)}")
    print(f"Hold validation summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(ledger[OUTPUT_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
