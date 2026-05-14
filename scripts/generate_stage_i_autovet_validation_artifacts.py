from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


REVIEW_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
CANDIDATE_OUT = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1"
HOLD_OUT = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"

CANDIDATE_TARGETS = ["EPIC_200008831", "EPIC_200008924"]
HOLD_BATCH_SIZE = 20


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


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


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def epic_digits(epic_id: str) -> str:
    return "".join(ch for ch in str(epic_id) if ch.isdigit())


def source_events_path(epic_id: str) -> Path:
    digits = epic_digits(epic_id)
    return ROOT / "plots" / "k2_batch" / "epics" / f"EPIC_{digits}" / "events.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def choose_period_candidates(events: pd.DataFrame, row: pd.Series, max_rows: int = 25) -> pd.DataFrame:
    saved_period = as_float(row.get("best_period_days"))
    rows: list[dict[str, Any]] = []
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)

    def add_candidate(period: float, source: str) -> None:
        if not np.isfinite(period) or period <= 0:
            return
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
            events_df=filtered,
            period=float(period),
            tol_phase=0.03,
        )
        rows.append(
            {
                "period_days": float(period),
                "period_source": source,
                "selection_priority": 0 if source == "saved_autovet_best_period" else 1,
                "support_count": int(support),
                "cluster_center_phase": float(center),
            }
        )

    if np.isfinite(saved_period) and saved_period > 0:
        add_candidate(saved_period, "saved_autovet_best_period")

    t = pd.to_numeric(filtered.get("t_mid", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    t = np.sort(t[np.isfinite(t)])
    min_period = 0.5
    max_period = 40.0
    generated: list[float] = []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            delta = float(t[j] - t[i])
            if delta < min_period:
                continue
            max_harmonic = int(math.floor(delta / min_period))
            for harmonic in range(1, max_harmonic + 1):
                period = delta / float(harmonic)
                if min_period <= period <= max_period:
                    generated.append(period)

    if generated:
        # Coarse de-duplication keeps this deterministic while avoiding thousands of near-identical periods.
        rounded = sorted(set(round(float(p), 5) for p in generated if np.isfinite(p)))
        scored = []
        for period in rounded:
            support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
                events_df=filtered,
                period=float(period),
                tol_phase=0.03,
            )
            if support >= 2:
                scored.append((int(support), float(period), float(center)))
        scored = sorted(scored, key=lambda x: (-x[0], x[1]))[: max_rows * 2]
        for support, period, center in scored:
            rows.append(
                {
                    "period_days": float(period),
                    "period_source": "event_pair_spacing_search",
                    "selection_priority": 1,
                    "support_count": int(support),
                    "cluster_center_phase": float(center),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["period_days", "period_source", "support_count", "cluster_center_phase"])

    out = pd.DataFrame(rows)
    out["period_days_round"] = out["period_days"].round(5)
    out = out.sort_values(["selection_priority", "support_count", "period_days"], ascending=[True, False, True])
    out = out.drop_duplicates("period_days_round", keep="first").drop(columns=["period_days_round"])
    return out.head(max_rows).reset_index(drop=True)


def row_for_validation(row: pd.Series, batch_dir: Path, batch_name: str, batch_order: int) -> tuple[pd.Series, dict[str, Any]]:
    epic_id = str(row["epic_id"])
    epic_dir = batch_dir / epic_id
    epic_dir.mkdir(parents=True, exist_ok=True)

    events_src = source_events_path(epic_id)
    if not events_src.exists():
        raise FileNotFoundError(f"Missing events.csv for {epic_id}: {events_src}")
    events_dst = epic_dir / "events.csv"
    shutil.copy2(events_src, events_dst)
    events = read_csv(events_dst)
    candidates = choose_period_candidates(events, row)
    period_candidates_path = epic_dir / "period_candidates.csv"
    candidates.to_csv(period_candidates_path, index=False)
    if len(candidates) == 0:
        raise RuntimeError(f"No period candidates available for {epic_id}")

    chosen = candidates.iloc[0]
    period = as_float(chosen["period_days"])
    center = as_float(chosen["cluster_center_phase"])
    support = int(as_float(chosen["support_count"])) if np.isfinite(as_float(chosen["support_count"])) else 0

    summary_path = epic_dir / "stage_i_autovet_validation_input_summary.json"
    metrics = {str(k): json_safe(v) for k, v in row.to_dict().items()}
    metrics.update(
        {
            "best_period_days": period,
            "period_support_count": support,
            "cluster_center_phase": center,
            "events_csv": str(events_dst),
            "period_candidates_csv": str(period_candidates_path),
        }
    )
    summary = {
        "epic_id": epic_id,
        "query": f"EPIC {epic_digits(epic_id)}",
        "stage_i_autovet_context": {
            "batch_name": batch_name,
            "batch_order": int(batch_order),
            "source_review_queue": rel(REVIEW_QUEUE),
            "source_results": rel(RESULTS),
            "autovet_label": json_safe(row.get("autovet_label")),
            "autovet_confidence": json_safe(row.get("autovet_confidence")),
            "review_priority_score": json_safe(row.get("review_priority_score")),
            "triggered_rules": json_safe(row.get("triggered_rules")),
            "recommended_next_action": json_safe(row.get("recommended_next_action")),
        },
        "stage_r_and_stage_d_metrics": metrics,
        "artifacts": {
            "events_csv": str(events_dst),
            "period_candidates_csv": str(period_candidates_path),
            "light_curve_cache_path": "",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validation = row.copy()
    validation["batch_order"] = int(batch_order)
    validation["batch_name"] = batch_name
    validation["best_period_days"] = period
    validation["period_support_count"] = support
    validation["visual_label"] = "stage_i_autovet_needs_manual_review"
    validation["summary_json_path"] = str(summary_path)
    validation["events_csv"] = str(events_dst)
    validation["period_candidates_csv"] = str(period_candidates_path)
    return validation, {
        "chosen_period_days": period,
        "chosen_period_source": str(chosen["period_source"]),
        "chosen_period_support_count": support,
        "chosen_cluster_center_phase": center,
        "events_csv": str(events_dst),
        "period_candidates_csv": str(period_candidates_path),
        "summary_json_path": str(summary_path),
    }


def build_batch(rows: pd.DataFrame, batch_dir: Path, batch_name: str) -> pd.DataFrame:
    batch_dir.mkdir(parents=True, exist_ok=True)
    validator = K2StageFFollowupValidation()
    validation_inputs: list[pd.Series] = []
    prep_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for order, (_, row) in enumerate(rows.iterrows(), start=1):
        prepared, prep = row_for_validation(row, batch_dir, batch_name, order)
        validation_inputs.append(prepared)
        prep_rows.append({"epic_id": prepared["epic_id"], "batch_order": order, **prep})
        validation_rows.append(validator._validate_one(row=prepared, out_dir=batch_dir))

    input_df = pd.DataFrame(validation_inputs)
    input_df.to_csv(batch_dir / "validation_input.csv", index=False)
    pd.DataFrame(prep_rows).to_csv(batch_dir / "period_selection_audit.csv", index=False)
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(batch_dir / "validation_metrics.csv", index=False)

    review = input_df.merge(validation_df, on=["epic_id", "best_period_days"], how="left", validate="one_to_one")
    for col in [
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "radius_ratio_sqrt_depth",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
    ]:
        y_col = f"{col}_y"
        x_col = f"{col}_x"
        if y_col in review.columns:
            review[col] = review[y_col]
        elif x_col in review.columns:
            review[col] = review[x_col]
    review = review.drop(columns=[c for c in review.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    ordered = [
        "batch_order",
        "epic_id",
        "autovet_label",
        "autovet_confidence",
        "review_priority_score",
        "best_period_days",
        "period_support_count",
        "flux_p_science_like",
        "best_shape_score",
        "best_depth_snr",
        "n_events",
        "stage_f_label",
        "stage_f_reason",
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "radius_ratio_sqrt_depth",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "alias_risk",
        "events_csv",
        "period_candidates_csv",
        "phase_0_folded_path",
        "phase_05_secondary_check_path",
        "odd_even_zoom_path",
        "alias_period_comparison_path",
        "validation_summary_json_path",
        "triggered_rules",
        "recommended_next_action",
    ]
    remaining = [c for c in review.columns if c not in ordered]
    review = review[[c for c in ordered if c in review.columns] + remaining]
    review.to_csv(batch_dir / "review_sheet.csv", index=False)
    return review


def main() -> None:
    queue = pd.read_csv(REVIEW_QUEUE)
    results = pd.read_csv(RESULTS)
    queue["epic_id"] = queue["epic_id"].astype(str)
    results["epic_id"] = results["epic_id"].astype(str)

    candidate_rows = queue.loc[queue["epic_id"].isin(CANDIDATE_TARGETS)].copy()
    order = {epic: i for i, epic in enumerate(CANDIDATE_TARGETS)}
    candidate_rows["target_order"] = candidate_rows["epic_id"].map(order)
    candidate_rows = candidate_rows.sort_values("target_order").drop(columns=["target_order"])
    missing = [epic for epic in CANDIDATE_TARGETS if epic not in set(candidate_rows["epic_id"])]
    if missing:
        raise ValueError(f"Missing candidate targets from review queue: {missing}")

    hold_rows = (
        queue.loc[queue["autovet_label"].eq("auto_hold_needs_review")]
        .copy()
        .sort_values("review_priority_score", ascending=False)
        .head(HOLD_BATCH_SIZE)
    )

    candidate_review = build_batch(candidate_rows, CANDIDATE_OUT, "candidate_batch1")
    hold_review = build_batch(hold_rows, HOLD_OUT, "hold_batch1")

    print("Candidate validation summary metrics")
    summary_cols = [
        "epic_id",
        "autovet_label",
        "review_priority_score",
        "best_period_days",
        "period_support_count",
        "stage_f_label",
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "radius_ratio_sqrt_depth",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "alias_risk",
    ]
    print(candidate_review[[c for c in summary_cols if c in candidate_review.columns]].to_string(index=False))
    print()
    print("Hold validation summary metrics")
    print(hold_review[[c for c in summary_cols if c in hold_review.columns]].to_string(index=False))
    print()
    print(f"Candidate review sheet: {rel(CANDIDATE_OUT / 'review_sheet.csv')}")
    print(f"Hold review sheet: {rel(HOLD_OUT / 'review_sheet.csv')}")
    print("Funnel: 12,140 unresolved -> AutoVet review queue 252 -> immediate candidate batch 2 -> hold audit batch 20")


if __name__ == "__main__":
    main()
