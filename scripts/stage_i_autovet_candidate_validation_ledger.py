from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1"
REVIEW_SHEET = BATCH_DIR / "review_sheet.csv"
OUT_LEDGER = BATCH_DIR / "autovet_candidate_validation_ledger.csv"
OUT_SUMMARY = BATCH_DIR / "autovet_candidate_validation_summary.txt"

REQUIRED_ARTIFACTS = [
    "validation_summary.json",
    "events.csv",
    "period_candidates.csv",
    "phase_0_folded.png",
    "phase_05_secondary_check.png",
    "odd_even_zoom.png",
    "alias_period_comparison.png",
]

LEDGER_COLUMNS = [
    "epic_id",
    "autovet_label",
    "stage_f_validation_label",
    "recommended_ledger_label",
    "recommended_training_label",
    "training_safe",
    "primary_period",
    "primary_depth",
    "odd_even_ratio",
    "secondary_depth",
    "secondary_to_primary_ratio",
    "num_events",
    "event_depth_cv",
    "alias_risk_flag",
    "eb_risk_flag",
    "artifact_risk_flag",
    "promotion_safe",
    "rejection_safe",
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


def json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def circular_phase_distance(phase: pd.Series, center: float) -> pd.Series:
    return ((phase - float(center) + 0.5) % 1.0) - 0.5


def event_depth_cv(events: pd.DataFrame, period: float, center: float) -> tuple[int, float]:
    if events.empty or not np.isfinite(period) or period <= 0 or not np.isfinite(center):
        return 0, float("nan")
    if "t_mid" not in events.columns or "depth" not in events.columns:
        return 0, float("nan")

    work = events.copy()
    work["t_mid"] = pd.to_numeric(work["t_mid"], errors="coerce")
    work["depth"] = pd.to_numeric(work["depth"], errors="coerce")
    work = work.loc[work["t_mid"].notna() & work["depth"].notna()].copy()
    if work.empty:
        return 0, float("nan")

    phase = (work["t_mid"] % float(period)) / float(period)
    family = work.loc[circular_phase_distance(phase, center).abs() <= 0.03].copy()
    if family.empty:
        family = work

    depths = family["depth"].abs().to_numpy(dtype=float)
    depths = depths[np.isfinite(depths)]
    if len(depths) == 0:
        return int(len(family)), float("nan")
    mean = float(np.nanmean(depths))
    cv = float(np.nanstd(depths, ddof=0) / mean) if mean > 0 else float("nan")
    return int(len(family)), cv


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


def validation_center(summary: dict[str, Any]) -> float:
    metrics = summary.get("stage_d_summary", {}).get("stage_r_and_stage_d_metrics", {})
    return as_float(metrics.get("cluster_center_phase"))


def assess_row(row: pd.Series) -> dict[str, Any]:
    epic_id = str(row["epic_id"])
    epic_dir = BATCH_DIR / epic_id
    complete, missing = artifact_completeness(epic_dir)
    summary_path = epic_dir / "validation_summary.json"
    summary = json_load(summary_path)
    validation = summary.get("validation", {})
    events = read_csv(epic_dir / "events.csv")
    period_candidates = read_csv(epic_dir / "period_candidates.csv")

    autovet_label = str(row.get("autovet_label", ""))
    stage_f_label = str(value_from(row, validation, "stage_f_label") or "")
    primary_period = as_float(value_from(row, validation, "best_period_days"))
    primary_depth = as_float(value_from(row, validation, "primary_depth"))
    primary_snr = as_float(value_from(row, validation, "primary_depth_snr"))
    odd_even_ratio = as_float(value_from(row, validation, "odd_even_depth_ratio"))
    odd_even_delta = as_float(value_from(row, validation, "odd_even_depth_delta_explicit"))
    secondary_depth = as_float(value_from(row, validation, "secondary_depth_phase_05"))
    secondary_ratio = as_float(value_from(row, validation, "secondary_to_primary_depth_ratio"))
    alias_ratio = as_float(value_from(row, validation, "alias_best_support_ratio"))
    alias_risk = str(value_from(row, validation, "alias_risk") or "").lower()
    stage_f_reason = str(value_from(row, validation, "stage_f_reason") or "")
    center = validation_center(summary)
    num_events, depth_cv = event_depth_cv(events, primary_period, center)
    if num_events == 0:
        num_events = int(as_float(summary.get("family_event_count", row.get("period_support_count", np.nan)))) if np.isfinite(as_float(summary.get("family_event_count", row.get("period_support_count", np.nan)))) else 0

    alias_risk_flag = alias_risk in {"moderate", "high"} or (np.isfinite(alias_ratio) and alias_ratio >= 1.0)
    severe_odd_even = (np.isfinite(odd_even_ratio) and odd_even_ratio < 0.55) or (np.isfinite(odd_even_delta) and odd_even_delta >= 0.75)
    deep_or_large_radius = np.isfinite(primary_depth) and primary_depth > 0.02
    weak_primary = (np.isfinite(primary_snr) and primary_snr < 5.0) or "not significant" in stage_f_reason.lower()
    strong_secondary = np.isfinite(secondary_ratio) and secondary_ratio >= 0.25

    evidence_incomplete = (not complete) or not validation or not np.isfinite(primary_period) or not np.isfinite(primary_depth)
    eb_evidence_strong = stage_f_label == "stage_f_likely_eb" and (deep_or_large_radius or severe_odd_even or strong_secondary or alias_risk == "high")
    artifact_evidence = stage_f_label == "stage_f_reject" and (weak_primary or num_events < 3 or not np.isfinite(primary_snr))

    promotion_safe = False
    rejection_safe = False
    eb_risk_flag = False
    artifact_risk_flag = False
    recommended_ledger_label = "manual_review_required"
    recommended_training_label = "do_not_train_yet"
    training_safe = False
    next_action = "manual_review_before_any_label_or_ledger_change"
    reason_parts: list[str] = []

    if evidence_incomplete:
        next_action = "regenerate_validation_artifacts"
        reason_parts.append("validation evidence incomplete: " + ", ".join(missing or ["missing required metrics"]))
    elif autovet_label == "auto_candidate_with_caveat":
        reason_parts.append("auto_candidate_with_caveat is never directly promotable by policy")

    if not evidence_incomplete and stage_f_label == "stage_f_likely_eb":
        eb_risk_flag = True
        recommended_ledger_label = "likely_eb_or_binary_system"
        recommended_training_label = "binary_system" if eb_evidence_strong else "do_not_train_yet"
        training_safe = bool(eb_evidence_strong)
        next_action = "review_secondary_odd_even_and_eb_features_before_label_update"
        reason_parts.append(
            "Stage F likely EB; "
            f"odd_even_ratio={fmt(odd_even_ratio)}, odd_even_delta={fmt(odd_even_delta)}, "
            f"alias_risk={alias_risk or 'NA'}, primary_depth={fmt(primary_depth)}"
        )
    elif not evidence_incomplete and stage_f_label == "stage_f_reject":
        artifact_risk_flag = True
        rejection_safe = bool(artifact_evidence)
        recommended_ledger_label = "reject_or_low_priority_negative"
        recommended_training_label = "low_priority_negative"
        training_safe = bool(rejection_safe)
        next_action = "add_to_reject_audit_queue_before_training"
        reason_parts.append(
            "Stage F reject; "
            f"primary_snr={fmt(primary_snr)}, num_events={num_events}, "
            f"stage_f_reason={stage_f_reason}"
        )
    elif not evidence_incomplete and stage_f_label:
        next_action = "manual_review_before_any_label_or_ledger_change"
        reason_parts.append(f"Stage F label {stage_f_label} is not promotion-safe in candidate-with-caveat batch")

    if not reason_parts:
        reason_parts.append("insufficient validated evidence for promotion or training")

    return {
        "epic_id": epic_id,
        "autovet_label": autovet_label,
        "stage_f_validation_label": stage_f_label,
        "recommended_ledger_label": recommended_ledger_label,
        "recommended_training_label": recommended_training_label,
        "training_safe": bool(training_safe),
        "primary_period": primary_period,
        "primary_depth": primary_depth,
        "odd_even_ratio": odd_even_ratio,
        "secondary_depth": secondary_depth,
        "secondary_to_primary_ratio": secondary_ratio,
        "num_events": int(num_events),
        "event_depth_cv": depth_cv,
        "alias_risk_flag": bool(alias_risk_flag),
        "eb_risk_flag": bool(eb_risk_flag),
        "artifact_risk_flag": bool(artifact_risk_flag),
        "promotion_safe": bool(promotion_safe),
        "rejection_safe": bool(rejection_safe),
        "recommended_next_action": next_action,
        "reason": "; ".join(reason_parts),
        "validation_summary_json": str(summary_path),
        "events_csv": str(epic_dir / "events.csv"),
        "period_candidates_csv": str(epic_dir / "period_candidates.csv"),
        "period_candidate_rows": int(len(period_candidates)),
        "artifact_complete": bool(complete),
        "missing_artifacts": "; ".join(missing),
    }


def fmt(value: Any, digits: int = 4) -> str:
    value = as_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def write_summary(ledger: pd.DataFrame) -> None:
    n_rows = int(len(ledger))
    n_promoted = int(ledger["promotion_safe"].map(as_bool).sum())
    n_likely_eb = int(ledger["eb_risk_flag"].map(as_bool).sum())
    n_rejected = int(ledger["recommended_ledger_label"].eq("reject_or_low_priority_negative").sum())
    n_training_safe = int(ledger["training_safe"].map(as_bool).sum())
    human_review = ledger.loc[
        ledger["recommended_next_action"].astype(str).str.contains("review|audit|regenerate", case=False, regex=True),
        "epic_id",
    ].astype(str).tolist()

    lines = [
        "Stage I AutoVet v1 candidate validation summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Candidate-with-caveat rows reviewed: {n_rows}",
        f"Promoted: {n_promoted}",
        f"Likely EB / binary-system review: {n_likely_eb}",
        f"Rejected / low-priority negative review: {n_rejected}",
        f"Training-safe recommendations: {n_training_safe}",
        f"EPICs requiring human review: {', '.join(human_review) if human_review else 'none'}",
        "",
        "Per-EPIC recommendations",
    ]
    for _, row in ledger.iterrows():
        lines.append(
            "- "
            f"{row['epic_id']}: stage_f={row['stage_f_validation_label']}; "
            f"ledger={row['recommended_ledger_label']}; "
            f"training={row['recommended_training_label']}; "
            f"promotion_safe={row['promotion_safe']}; "
            f"next={row['recommended_next_action']}"
        )
    lines.extend(
        [
            "",
            "No training labels, final candidate ledger rows, or model artifacts were modified.",
            "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not REVIEW_SHEET.exists():
        raise FileNotFoundError(REVIEW_SHEET)

    review = pd.read_csv(REVIEW_SHEET)
    review["epic_id"] = review["epic_id"].astype(str)
    rows = [assess_row(row) for _, row in review.iterrows()]
    ledger = pd.DataFrame(rows)
    ledger = ledger[LEDGER_COLUMNS + [c for c in ledger.columns if c not in LEDGER_COLUMNS]]
    ledger.to_csv(OUT_LEDGER, index=False)
    write_summary(ledger)

    print(f"Candidate validation ledger: {rel(OUT_LEDGER)}")
    print(f"Candidate validation summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
