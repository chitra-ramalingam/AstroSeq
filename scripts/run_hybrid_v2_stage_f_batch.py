from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


ROOT = Path(".")
HYBRID_V2 = ROOT / "k2_hybrid_candidate_score_v2.csv"
INPUT_CSV = ROOT / "k2_stage_f_hybrid_v2_batch_input.csv"
OUTPUT_CSV = ROOT / "k2_stage_f_hybrid_v2_validation.csv"
OUT_DIR = ROOT / "plots" / "k2_batch" / "stage_f_hybrid_v2"
SUMMARY_DIR = ROOT / "plots" / "k2_batch" / "stage_f_hybrid_v2_input_summaries"

PRIMARY_EPICS = [
    "EPIC_211732801",
    "EPIC_211425098",
    "EPIC_211530033",
    "EPIC_211396246",
    "EPIC_211833851",
]

SECONDARY_EPICS = [
    "EPIC_211972767",
    "EPIC_211910082",
    "EPIC_211569204",
    "EPIC_211947686",
]


def _json_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _epic_digits(epic_id: str) -> str:
    return "".join(ch for ch in str(epic_id) if ch.isdigit())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _cluster_center(events_csv: Path, period: float) -> tuple[int, float]:
    events = _read_csv(events_csv)
    if len(events) == 0:
        return 0, 0.0
    try:
        filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
            events_df=filtered,
            period=float(period),
            tol_phase=0.03,
        )
        if not math.isfinite(float(center)):
            center = 0.0
        return int(support), float(center)
    except Exception:
        return 0, 0.0


def _summary_for_row(row: pd.Series) -> Path:
    epic_id = str(row["epic_id"])
    period = float(row["best_period_days"])
    events_csv = Path(str(row.get("events_csv", "")))
    if not events_csv.exists():
        events_csv = ROOT / "plots" / "k2_batch" / "epics" / f"EPIC_{_epic_digits(epic_id)}" / "events.csv"
    support, center = _cluster_center(events_csv, period)

    summary_path = SUMMARY_DIR / epic_id / f"{epic_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {k: _json_scalar(row.get(k)) for k in row.index}
    metrics["cluster_center_phase"] = _json_scalar(row.get("cluster_center_phase", center)) or center
    metrics["period_support_count"] = int(_json_scalar(row.get("period_support_count")) or support)
    metrics["events_csv"] = str(events_csv)

    ranked_keys = [
        "epic_id",
        "best_period_days",
        "period_support_count",
        "event_family_count",
        "folded_depth_consistency",
        "duration_consistency",
        "odd_even_depth_delta",
        "coverage_rate",
        "hit_rate_snr",
        "hit_rate_shape",
        "soft_hit_rate",
        "stage_d_label",
        "stage_d_reason",
    ]
    summary = {
        "epic_id": epic_id,
        "query": f"EPIC {_epic_digits(epic_id)}",
        "source_ranked_row": {k: _json_scalar(row.get(k)) for k in ranked_keys if k in row.index},
        "stage_r_and_stage_d_metrics": metrics,
        "artifacts": {
            "events_csv": str(events_csv),
            "light_curve_cache_path": "",
        },
        "hybrid_v2_context": {
            "candidate_score": _json_scalar(row.get("candidate_score")),
            "flux_p_science_like": _json_scalar(row.get("flux_p_science_like")),
            "stage_d_quality_score": _json_scalar(row.get("stage_d_quality_score")),
            "stage_f_quality_score_before_validation": _json_scalar(row.get("stage_f_quality_score")),
            "false_positive_penalty": _json_scalar(row.get("false_positive_penalty")),
            "needs_stage_f_validation": _json_scalar(row.get("needs_stage_f_validation")),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_input() -> pd.DataFrame:
    hybrid = pd.read_csv(HYBRID_V2)
    wanted = PRIMARY_EPICS + SECONDARY_EPICS
    priority = {epic: "primary" for epic in PRIMARY_EPICS}
    priority.update({epic: "secondary_optional" for epic in SECONDARY_EPICS})
    order = {epic: i + 1 for i, epic in enumerate(wanted)}

    selected = hybrid.loc[hybrid["epic_id"].astype(str).isin(wanted)].copy()
    missing = [epic for epic in wanted if epic not in set(selected["epic_id"].astype(str))]
    if missing:
        raise ValueError(f"Missing requested EPICs from {HYBRID_V2}: {missing}")

    selected["batch_priority"] = selected["epic_id"].map(priority)
    selected["batch_order"] = selected["epic_id"].map(order)
    selected = selected.sort_values("batch_order").reset_index(drop=True)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in selected.iterrows():
        summary_path = _summary_for_row(row)
        rows.append(
            {
                "batch_order": int(row["batch_order"]),
                "batch_priority": row["batch_priority"],
                "epic_id": row["epic_id"],
                "best_period_days": row["best_period_days"],
                "period_support_count": row["period_support_count"],
                "visual_label": "visual_needs_manual_review",
                "summary_json_path": str(summary_path),
                "hybrid_rank": row.get("hybrid_rank"),
                "candidate_score": row.get("candidate_score"),
                "flux_p_science_like": row.get("flux_p_science_like"),
                "stage_d_quality_score": row.get("stage_d_quality_score"),
                "false_positive_penalty": row.get("false_positive_penalty"),
                "events_csv": row.get("events_csv"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(INPUT_CSV, index=False)
    return out


def main() -> None:
    batch_input = build_input()
    result = K2StageFFollowupValidation().run(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        out_dir=OUT_DIR,
    )

    validation = pd.read_csv(OUTPUT_CSV)
    enriched = batch_input.merge(validation, on=["epic_id", "best_period_days"], how="left", validate="one_to_one")
    ordered_cols = [
        "batch_order",
        "batch_priority",
        "epic_id",
        "best_period_days",
        "period_support_count",
        "hybrid_rank",
        "candidate_score",
        "flux_p_science_like",
        "stage_d_quality_score",
        "false_positive_penalty",
        "visual_label",
        "stage_f_label",
        "stage_f_reason",
    ]
    remaining = [c for c in enriched.columns if c not in ordered_cols]
    enriched[[c for c in ordered_cols if c in enriched.columns] + remaining].to_csv(OUTPUT_CSV, index=False)

    print(f"input_csv: {INPUT_CSV}")
    print(f"output_csv: {OUTPUT_CSV}")
    print(f"out_dir: {OUT_DIR}")
    print(f"rows_input: {result['rows_input']}")
    print(f"rows_output: {result['rows_output']}")
    print(f"label_counts: {result['label_counts']}")
    print(enriched[ordered_cols].to_string(index=False))


if __name__ == "__main__":
    main()
