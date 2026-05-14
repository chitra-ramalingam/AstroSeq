from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_stage_i_autovet_validation_artifacts import choose_period_candidates
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"
REPAIR_QUEUE = BATCH_DIR / "period_support_repair_queue.csv"
MANUAL_PRIORITY_OVERLAY = BATCH_DIR / "manual_repair_priority_overlay.csv"
OUT_DIR = BATCH_DIR / "period_support_repair_validation"
OUT_LEDGER = BATCH_DIR / "post_repair_validation_ledger.csv"
OUT_SUMMARY = BATCH_DIR / "post_repair_validation_summary.txt"

PRIORITY_ORDER = {"high": 0, "medium_caution": 1, "normal": 2}


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


def json_safe(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def apply_manual_priority_overlay(queue: pd.DataFrame) -> pd.DataFrame:
    queue = queue.copy()
    queue["_queue_order"] = np.arange(len(queue))
    if not MANUAL_PRIORITY_OVERLAY.exists():
        queue["manual_priority"] = "normal"
        queue["manual_note"] = "manual priority overlay not present"
        queue["_priority_rank"] = PRIORITY_ORDER["normal"]
        return queue.sort_values(["_priority_rank", "_queue_order"]).drop(
            columns=["_priority_rank", "_queue_order"]
        )

    overlay = pd.read_csv(MANUAL_PRIORITY_OVERLAY)
    required = {
        "epic_id",
        "manual_priority",
        "manual_note",
        "human_review_required",
        "label_update_allowed",
    }
    missing = required.difference(overlay.columns)
    if missing:
        raise ValueError(f"Manual priority overlay missing columns: {sorted(missing)}")

    overlay = overlay[
        [
            "epic_id",
            "manual_priority",
            "manual_note",
            "human_review_required",
            "label_update_allowed",
        ]
    ].copy()
    overlay["epic_id"] = overlay["epic_id"].astype(str)
    overlay["_priority_overlay_order"] = np.arange(len(overlay))
    if overlay["epic_id"].duplicated().any():
        dupes = overlay.loc[overlay["epic_id"].duplicated(), "epic_id"].tolist()
        raise ValueError(f"Manual priority overlay has duplicate EPICs: {dupes}")
    if not overlay["human_review_required"].map(as_bool).all():
        raise ValueError("Manual priority overlay must require human review for every row")
    if overlay["label_update_allowed"].map(as_bool).any():
        raise ValueError("Manual priority overlay must not allow label updates")

    overlay = overlay[
        ["epic_id", "manual_priority", "manual_note", "_priority_overlay_order"]
    ]
    merged = queue.merge(
        overlay,
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    merged["manual_priority"] = merged["manual_priority"].fillna("normal")
    merged["manual_note"] = merged["manual_note"].fillna("not manually prioritized in this pass")
    unknown_priorities = sorted(set(merged["manual_priority"]).difference(PRIORITY_ORDER))
    if unknown_priorities:
        raise ValueError(f"Unknown manual priorities: {unknown_priorities}")

    merged["_priority_overlay_order"] = merged["_priority_overlay_order"].fillna(len(overlay))
    merged["_priority_rank"] = merged["manual_priority"].map(PRIORITY_ORDER)
    return merged.sort_values(
        ["_priority_rank", "_priority_overlay_order", "_queue_order"]
    ).drop(columns=["_priority_rank", "_priority_overlay_order", "_queue_order"])


def source_events_path(epic_id: str) -> Path:
    return BATCH_DIR / str(epic_id) / "events.csv"


def prepare_validation_row(row: pd.Series, order: int) -> pd.Series:
    epic_id = str(row["epic_id"])
    epic_dir = OUT_DIR / epic_id
    epic_dir.mkdir(parents=True, exist_ok=True)

    src_events = source_events_path(epic_id)
    if not src_events.exists():
        raise FileNotFoundError(f"Missing source events for {epic_id}: {src_events}")
    events_path = epic_dir / "events.csv"
    shutil.copy2(src_events, events_path)

    events = pd.read_csv(events_path)
    period_input = row.copy()
    period_input["best_period_days"] = np.nan
    candidates = choose_period_candidates(events, period_input, max_rows=40)
    if len(candidates) == 0:
        raise RuntimeError(f"No repair period candidates found for {epic_id}")
    period_candidates_path = epic_dir / "period_candidates.csv"
    candidates.to_csv(period_candidates_path, index=False)
    chosen = candidates.iloc[0]
    period = as_float(chosen["period_days"])
    support = int(as_float(chosen["support_count"])) if np.isfinite(as_float(chosen["support_count"])) else 0
    center = as_float(chosen["cluster_center_phase"])

    summary_path = epic_dir / "stage_i_period_support_repair_input_summary.json"
    metrics = {str(k): json_safe(v) for k, v in row.to_dict().items()}
    metrics.update(
        {
            "best_period_days": period,
            "period_support_count": support,
            "cluster_center_phase": center,
            "events_csv": str(events_path),
            "period_candidates_csv": str(period_candidates_path),
        }
    )
    summary = {
        "epic_id": epic_id,
        "query": epic_id.replace("_", " "),
        "stage_i_period_support_repair_context": {
            "batch_order": int(order),
            "source_repair_queue": rel(REPAIR_QUEUE),
            "repair_reason": json_safe(row.get("repair_reason")),
            "human_review_required": True,
            "label_update_allowed": False,
            "promotion_safe": False,
        },
        "stage_r_and_stage_d_metrics": metrics,
        "artifacts": {
            "events_csv": str(events_path),
            "period_candidates_csv": str(period_candidates_path),
            "light_curve_cache_path": "",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    prepared = row.copy()
    prepared["batch_order"] = int(order)
    prepared["batch_name"] = "period_support_repair_validation"
    prepared["best_period_days"] = period
    prepared["period_support_count"] = support
    prepared["visual_label"] = "stage_i_period_support_repair_needs_manual_review"
    prepared["summary_json_path"] = str(summary_path)
    prepared["events_csv"] = str(events_path)
    prepared["period_candidates_csv"] = str(period_candidates_path)
    prepared["period_search_source"] = str(chosen.get("period_source", "event_pair_spacing_search"))
    return prepared


def route_post_repair(row: pd.Series, preserve_source_labels: bool = False) -> dict[str, Any]:
    label = str(row.get("post_repair_stage_f_label", ""))
    original_label = str(row.get("stage_f_validation_label", ""))
    stage_reason = str(row.get("post_repair_stage_f_reason", ""))
    primary_snr = as_float(row.get("post_repair_primary_depth_snr"))
    alias_risk = str(row.get("post_repair_alias_risk", "")).lower()
    odd_even_ratio = as_float(row.get("post_repair_odd_even_depth_ratio"))
    secondary_ratio = as_float(row.get("post_repair_secondary_to_primary_depth_ratio"))

    eb_risk = bool(
        label == "stage_f_likely_eb"
        or alias_risk in {"moderate", "high"}
        or (np.isfinite(odd_even_ratio) and odd_even_ratio < 0.55)
        or (np.isfinite(secondary_ratio) and secondary_ratio >= 0.25)
    )
    artifact_risk = bool(label == "stage_f_reject" and (primary_snr < 5 or "not significant" in stage_reason.lower()))
    if preserve_source_labels:
        return {
            "recommended_ledger_label": row.get("recommended_ledger_label", "needs_period_review"),
            "recommended_training_label": row.get("recommended_training_label", "do_not_train_yet"),
            "training_safe": False,
            "promotion_safe": False,
            "rejection_safe": False,
            "human_review_required": True,
            "label_update_allowed": False,
            "recommended_next_action": "manual_review_repaired_period_support_before_label_update",
            "post_repair_eb_risk_flag": bool(eb_risk),
            "post_repair_artifact_risk_flag": bool(artifact_risk),
            "post_repair_reason": (
                f"pre_repair_stage_f={original_label}; post_repair_stage_f={label}; "
                f"post_repair_reason={stage_reason}; manual priority overlay run preserved source "
                "recommended labels and requires human review before any label update"
            ),
        }

    promotion_safe = False
    label_update_allowed = False
    human_review_required = True
    if label == "stage_f_reject":
        recommended_ledger_label = "reject_or_low_priority_negative"
        recommended_training_label = "low_priority_negative" if primary_snr < 5 else "do_not_train_yet"
        rejection_safe = bool(primary_snr < 5 or "not significant" in stage_reason.lower())
        training_safe = bool(rejection_safe)
        next_action = "add_to_reject_audit_queue_before_training" if rejection_safe else "manual_review_before_label_update"
    elif label == "stage_f_hold":
        recommended_ledger_label = "uncertain_hold"
        recommended_training_label = "do_not_train_yet"
        rejection_safe = False
        training_safe = False
        next_action = "manual_review_repaired_period_support_before_label_update"
    elif label == "stage_f_likely_eb":
        recommended_ledger_label = "likely_eb_or_binary_system"
        recommended_training_label = "binary_system" if alias_risk == "high" or (np.isfinite(odd_even_ratio) and odd_even_ratio < 0.55) else "do_not_train_yet"
        rejection_safe = False
        training_safe = recommended_training_label == "binary_system"
        next_action = "review_secondary_odd_even_and_eb_features_before_label_update"
    else:
        recommended_ledger_label = "needs_manual_review"
        recommended_training_label = "do_not_train_yet"
        rejection_safe = False
        training_safe = False
        next_action = "manual_review_before_label_update"

    return {
        "recommended_ledger_label": recommended_ledger_label,
        "recommended_training_label": recommended_training_label,
        "training_safe": bool(training_safe),
        "promotion_safe": bool(promotion_safe),
        "rejection_safe": bool(rejection_safe),
        "human_review_required": bool(human_review_required),
        "label_update_allowed": bool(label_update_allowed),
        "recommended_next_action": next_action,
        "post_repair_eb_risk_flag": bool(eb_risk),
        "post_repair_artifact_risk_flag": bool(artifact_risk),
        "post_repair_reason": (
            f"pre_repair_stage_f={original_label}; post_repair_stage_f={label}; "
            f"post_repair_reason={stage_reason}; human review required before any label update"
        ),
    }


def main() -> None:
    if not REPAIR_QUEUE.exists():
        raise FileNotFoundError(REPAIR_QUEUE)
    queue = pd.read_csv(REPAIR_QUEUE)
    queue["epic_id"] = queue["epic_id"].astype(str)
    preserve_source_labels = MANUAL_PRIORITY_OVERLAY.exists()
    queue = apply_manual_priority_overlay(queue)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validator = K2StageFFollowupValidation()
    prepared_rows = []
    validation_rows = []
    for order, (_, row) in enumerate(queue.iterrows(), start=1):
        prepared = prepare_validation_row(row, order)
        prepared_rows.append(prepared)
        existing_summary = OUT_DIR / str(prepared["epic_id"]) / "validation_summary.json"
        if existing_summary.exists():
            payload = json.loads(existing_summary.read_text(encoding="utf-8"))
            validation_rows.append(payload.get("validation", {}))
        else:
            validation_rows.append(validator._validate_one(row=prepared, out_dir=OUT_DIR))

    prepared_df = pd.DataFrame(prepared_rows)
    validation_df = pd.DataFrame(validation_rows).rename(
        columns={
            "stage_f_label": "post_repair_stage_f_label",
            "stage_f_reason": "post_repair_stage_f_reason",
            "primary_depth": "post_repair_primary_depth",
            "primary_depth_snr": "post_repair_primary_depth_snr",
            "odd_even_depth_ratio": "post_repair_odd_even_depth_ratio",
            "odd_even_depth_delta_explicit": "post_repair_odd_even_depth_delta_explicit",
            "secondary_to_primary_depth_ratio": "post_repair_secondary_to_primary_depth_ratio",
            "secondary_depth_snr": "post_repair_secondary_depth_snr",
            "alias_best_support_ratio": "post_repair_alias_best_support_ratio",
            "alias_risk": "post_repair_alias_risk",
            "phase_0_folded_path": "post_repair_phase_0_folded_path",
            "phase_05_secondary_check_path": "post_repair_phase_05_secondary_check_path",
            "alias_period_comparison_path": "post_repair_alias_period_comparison_path",
            "odd_even_zoom_path": "post_repair_odd_even_zoom_path",
            "validation_summary_json_path": "post_repair_validation_summary_json_path",
        }
    )
    merged = prepared_df.merge(validation_df, on=["epic_id", "best_period_days"], how="left", validate="one_to_one")
    routes = pd.DataFrame(
        [route_post_repair(row, preserve_source_labels=preserve_source_labels) for _, row in merged.iterrows()]
    )
    merged = merged.drop(columns=[c for c in routes.columns if c in merged.columns], errors="ignore")
    ledger = pd.concat([merged.reset_index(drop=True), routes], axis=1)

    ordered_cols = [
        "epic_id",
        "stage_f_validation_label",
        "post_repair_stage_f_label",
        "recommended_ledger_label",
        "recommended_training_label",
        "training_safe",
        "promotion_safe",
        "rejection_safe",
        "human_review_required",
        "label_update_allowed",
        "best_period_days",
        "period_support_count",
        "period_search_source",
        "post_repair_primary_depth",
        "post_repair_primary_depth_snr",
        "post_repair_odd_even_depth_ratio",
        "post_repair_secondary_to_primary_depth_ratio",
        "post_repair_alias_best_support_ratio",
        "post_repair_alias_risk",
        "post_repair_eb_risk_flag",
        "post_repair_artifact_risk_flag",
        "recommended_next_action",
        "post_repair_reason",
        "events_csv",
        "period_candidates_csv",
        "post_repair_phase_0_folded_path",
        "post_repair_phase_05_secondary_check_path",
        "post_repair_odd_even_zoom_path",
        "post_repair_alias_period_comparison_path",
        "post_repair_validation_summary_json_path",
    ]
    ledger = ledger[[c for c in ordered_cols if c in ledger.columns] + [c for c in ledger.columns if c not in ordered_cols]]
    ledger.to_csv(OUT_LEDGER, index=False)

    lines = [
        "Stage I period-support post-repair validation summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input repair queue: {rel(REPAIR_QUEUE)}",
        f"Manual priority overlay: {rel(MANUAL_PRIORITY_OVERLAY) if MANUAL_PRIORITY_OVERLAY.exists() else 'not present'}",
        f"Output artifact directory: {rel(OUT_DIR)}",
        f"Output ledger: {rel(OUT_LEDGER)}",
        "",
        f"Rows processed: {len(ledger)}",
        "Post-repair Stage F labels:",
    ]
    for label, count in ledger["post_repair_stage_f_label"].value_counts(dropna=False).items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(
        [
            f"promotion_safe count: {int(ledger['promotion_safe'].map(as_bool).sum())}",
            f"rejection_safe count: {int(ledger['rejection_safe'].map(as_bool).sum())}",
            f"training_safe count: {int(ledger['training_safe'].map(as_bool).sum())}",
            f"human_review_required count: {int(ledger['human_review_required'].map(as_bool).sum())}",
            f"automatic label updates allowed count: {int(ledger['label_update_allowed'].map(as_bool).sum())}",
            "",
            "No training labels, final candidate ledger rows, or model artifacts were modified.",
            "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
            "Manual priority overlay was used for processing order only; it did not permit label updates.",
            "Source recommended_ledger_label and recommended_training_label values were preserved.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Post-repair validation ledger: {rel(OUT_LEDGER)}")
    print(f"Post-repair validation summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(ledger[ordered_cols[:23]].to_string(index=False))


if __name__ == "__main__":
    main()
