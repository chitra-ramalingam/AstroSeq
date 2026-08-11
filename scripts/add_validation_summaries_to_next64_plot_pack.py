from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_manual_vetting_next64_plot_pack import (  # noqa: E402
    EPICS_DIR,
    KNOWN_VALIDATION_DIRS,
    OUT_DIR,
    QUEUE_CSV,
    candidate_periods_from_events,
    choose_period,
    duration_days,
    family_events,
    load_light_curve,
    load_events,
    phase_centered,
)
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner  # noqa: E402
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation  # noqa: E402

INDEX_CSV = OUT_DIR / "manual_vetting_next64_plot_index.csv"
SUMMARY_TXT = OUT_DIR / "manual_vetting_next64_validation_summary_manifest.txt"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def as_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def row_json(row: pd.Series) -> dict[str, Any]:
    return {str(k): as_json(v) for k, v in row.to_dict().items()}


def existing_validation_summary_path(epic_id: str) -> str:
    for base in KNOWN_VALIDATION_DIRS:
        candidate = base / epic_id / "validation_summary.json"
        if candidate.exists():
            return str(candidate)
    return ""


def existing_validation_payload(epic_id: str) -> dict[str, Any]:
    path = existing_validation_summary_path(epic_id)
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    validation = payload.get("validation", {})
    return validation if isinstance(validation, dict) else {}


def csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)
    return [{str(k): as_json(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def epoch_depth_stats(family: pd.DataFrame) -> dict[str, Any]:
    if len(family) == 0 or "depth" not in family.columns or "event_number" not in family.columns:
        return {
            "odd_depth_median": None,
            "even_depth_median": None,
            "odd_even_depth_ratio": None,
            "odd_even_depth_delta_explicit": None,
        }
    work = family.copy()
    work["depth"] = pd.to_numeric(work["depth"], errors="coerce")
    work = work.loc[work["depth"].notna()].copy()
    if len(work) == 0:
        return {
            "odd_depth_median": None,
            "even_depth_median": None,
            "odd_even_depth_ratio": None,
            "odd_even_depth_delta_explicit": None,
        }
    epoch_depths = work.groupby("event_number", dropna=True)["depth"].median().reset_index()
    even = epoch_depths.loc[(epoch_depths["event_number"].astype(int) % 2) == 0, "depth"].to_numpy(dtype=float)
    odd = epoch_depths.loc[(epoch_depths["event_number"].astype(int) % 2) == 1, "depth"].to_numpy(dtype=float)
    if len(even) == 0 or len(odd) == 0:
        return {
            "odd_depth_median": None,
            "even_depth_median": None,
            "odd_even_depth_ratio": None,
            "odd_even_depth_delta_explicit": None,
        }
    odd_med = float(np.nanmedian(odd))
    even_med = float(np.nanmedian(even))
    hi = max(abs(odd_med), abs(even_med))
    lo = min(abs(odd_med), abs(even_med))
    ref = float(np.nanmedian(np.abs(epoch_depths["depth"].to_numpy(dtype=float))))
    return {
        "odd_depth_median": odd_med,
        "even_depth_median": even_med,
        "odd_even_depth_ratio": float(lo / hi) if hi > 0 else None,
        "odd_even_depth_delta_explicit": float(abs(odd_med - even_med) / ref) if np.isfinite(ref) and ref > 0 else None,
    }


def alias_stats(events: pd.DataFrame, period: float, primary_support_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not np.isfinite(period) or period <= 0 or len(events) == 0:
        return [], {
            "alias_best_period_days": None,
            "alias_best_support_count": None,
            "alias_best_support_ratio": None,
            "half_period_support_count": None,
            "double_period_support_count": None,
            "alias_risk": None,
        }
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    candidates = [
        ("P/2", float(period) / 2.0),
        ("2P/3", float(period) * 2.0 / 3.0),
        ("P", float(period)),
        ("3P/2", float(period) * 1.5),
        ("P*2", float(period) * 2.0),
    ]
    rows: list[dict[str, Any]] = []
    for name, candidate_period in candidates:
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
            events_df=filtered,
            period=float(candidate_period),
            tol_phase=0.03,
        )
        rows.append(
            {
                "alias_name": name,
                "period_days": float(candidate_period),
                "support_count": int(support),
                "cluster_center_phase": as_json(center),
            }
        )
    p_support = int(max(0, int(primary_support_count)))
    if p_support <= 0:
        p_support = int(next(item["support_count"] for item in rows if item["alias_name"] == "P"))
    alternatives = [item for item in rows if item["alias_name"] != "P"]
    best_alt = sorted(alternatives, key=lambda item: (-int(item["support_count"]), float(item["period_days"])))[0]
    ratio = float(best_alt["support_count"] / p_support) if p_support > 0 else None
    alias_risk = "low"
    if ratio is not None and ratio >= 1.20:
        alias_risk = "high"
    elif ratio is not None and ratio >= 0.85:
        alias_risk = "moderate"
    return rows, {
        "alias_best_period_days": float(best_alt["period_days"]),
        "alias_best_support_count": int(best_alt["support_count"]),
        "alias_best_support_ratio": ratio,
        "half_period_support_count": int(next(item["support_count"] for item in rows if item["alias_name"] == "P/2")),
        "double_period_support_count": int(next(item["support_count"] for item in rows if item["alias_name"] == "P*2")),
        "alias_risk": alias_risk,
    }


def compute_stage_f_style_metrics(row: pd.Series, events: pd.DataFrame, family: pd.DataFrame, period: float, center_phase: float) -> dict[str, Any]:
    if not np.isfinite(period) or period <= 0:
        return {}
    lc = load_light_curve(str(row["epic_id"]))
    time = lc["time"]
    resid = lc["resid"]
    dur_days = duration_days(row, family, period)
    half_width_phase = float(np.clip(1.6 * dur_days / float(period), 0.015, 0.08)) if np.isfinite(dur_days) else 0.04
    phase0 = phase_centered(time, period, center_phase)
    folded_primary = K2StageFFollowupValidation._folded_depth(
        phase=phase0,
        resid=resid,
        half_width_phase=half_width_phase,
    )
    secondary_phase = phase_centered(time, period, (center_phase + 0.5) % 1.0)
    secondary = K2StageFFollowupValidation._folded_depth(
        phase=secondary_phase,
        resid=resid,
        half_width_phase=half_width_phase,
    )

    family_depth = pd.to_numeric(family.get("depth", pd.Series(dtype=float)), errors="coerce")
    family_snr = pd.to_numeric(family.get("depth_snr", pd.Series(dtype=float)), errors="coerce")
    primary_depth = as_json(family_depth.median()) if family_depth.notna().any() else None
    primary_snr = as_json(family_snr.median()) if family_snr.notna().any() else None
    if primary_depth is None or float(primary_depth) <= 0:
        primary_depth = as_json(folded_primary.get("depth"))
    if primary_snr is None or float(primary_snr) <= 0:
        primary_snr = as_json(folded_primary.get("snr"))
    secondary_depth = as_json(secondary.get("depth"))
    secondary_snr = as_json(secondary.get("snr"))
    secondary_ratio = (
        float(secondary_depth) / float(primary_depth)
        if primary_depth is not None and float(primary_depth) > 0 and secondary_depth is not None
        else None
    )
    radius_ratio = float(np.sqrt(float(primary_depth))) if primary_depth is not None and float(primary_depth) > 0 else None

    odd_even = epoch_depth_stats(family)
    oot = K2StageFFollowupValidation._oot_variability(phase0=phase0, resid=resid, primary_half_width=half_width_phase)
    oot_amp = as_json(oot.get("oot_variability_amp"))
    oot_to_depth = (
        float(oot_amp) / float(primary_depth)
        if oot_amp is not None and primary_depth is not None and float(primary_depth) > 0
        else None
    )
    primary_support_count = int(len(family))
    alias_rows, alias = alias_stats(events, period, primary_support_count=primary_support_count)

    return {
        "primary_depth": primary_depth,
        "primary_depth_snr": primary_snr,
        "transit_duration_days": as_json(dur_days),
        "transit_duration_hours": as_json(dur_days * 24.0 if np.isfinite(dur_days) else None),
        "radius_ratio_sqrt_depth": as_json(radius_ratio),
        "secondary_depth_phase_05": secondary_depth,
        "secondary_depth_snr": secondary_snr,
        "secondary_to_primary_depth_ratio": as_json(secondary_ratio),
        "odd_depth_median": as_json(odd_even["odd_depth_median"]),
        "even_depth_median": as_json(odd_even["even_depth_median"]),
        "odd_even_depth_ratio": as_json(odd_even["odd_even_depth_ratio"]),
        "odd_even_depth_delta_explicit": as_json(odd_even["odd_even_depth_delta_explicit"]),
        "oot_variability_amp": oot_amp,
        "oot_variability_to_depth": as_json(oot_to_depth),
        "alias_best_period_days": as_json(alias["alias_best_period_days"]),
        "alias_best_support_count": as_json(alias["alias_best_support_count"]),
        "alias_best_support_ratio": as_json(alias["alias_best_support_ratio"]),
        "half_period_support_count": as_json(alias["half_period_support_count"]),
        "double_period_support_count": as_json(alias["double_period_support_count"]),
        "alias_risk": alias["alias_risk"],
        "alias_periods": alias_rows,
    }


def build_summary(row: pd.Series, index_row: pd.Series) -> dict[str, Any]:
    epic_id = str(row["epic_id"])
    epic_dir = OUT_DIR / epic_id
    events = load_events(epic_id)
    period, center_phase, candidates, period_source = choose_period(row, events)
    family = family_events(events, period, center_phase)
    existing_summary = existing_validation_summary_path(epic_id)
    prior_validation = existing_validation_payload(epic_id)
    metrics = compute_stage_f_style_metrics(row, events, family, period, center_phase)
    events_csv = EPICS_DIR / epic_id / "events.csv"

    artifacts = {
        "summary_panel_path": str(ROOT / str(index_row["summary_panel_path"])),
        "raw_light_curve_path": str(ROOT / str(index_row["raw_light_curve_path"])),
        "detrended_light_curve_path": str(epic_dir / "detrended_light_curve.png"),
        "folded_light_curve_best_period_path": str(ROOT / str(index_row["folded_light_curve_path"])),
        "transit_window_zoom_path": str(ROOT / str(index_row["transit_zoom_path"])),
        "odd_even_transits_path": str(ROOT / str(index_row["odd_even_path"])),
        "secondary_eclipse_check_path": str(ROOT / str(index_row["secondary_check_path"])),
        "event_stack_path": str(ROOT / str(index_row["event_stack_path"])),
        "periodogram_or_period_search_path": str(epic_dir / "periodogram_or_period_search.png"),
        "oot_variability_check_path": str(epic_dir / "oot_variability_check.png"),
        "events_csv": str(events_csv) if events_csv.exists() else "",
        "previous_stage_i_validation_summary_json_path": existing_summary,
        "validation_summary_json_path": str(epic_dir / "validation_summary.json"),
    }

    validation = {
        "epic_id": epic_id,
        "queue_rank": as_json(row.get("queue_rank")),
        "best_period_days": as_json(period),
        "stage_e_visual_label": prior_validation.get("stage_e_visual_label"),
        "stage_f_label": prior_validation.get("stage_f_label"),
        "stage_f_reason": prior_validation.get("stage_f_reason"),
        "primary_depth": metrics.get("primary_depth"),
        "primary_depth_snr": metrics.get("primary_depth_snr"),
        "transit_duration_days": metrics.get("transit_duration_days"),
        "transit_duration_hours": metrics.get("transit_duration_hours"),
        "radius_ratio_sqrt_depth": metrics.get("radius_ratio_sqrt_depth"),
        "secondary_depth_phase_05": metrics.get("secondary_depth_phase_05"),
        "secondary_depth_snr": metrics.get("secondary_depth_snr"),
        "secondary_to_primary_depth_ratio": metrics.get("secondary_to_primary_depth_ratio"),
        "odd_depth_median": metrics.get("odd_depth_median"),
        "even_depth_median": metrics.get("even_depth_median"),
        "odd_even_depth_ratio": metrics.get("odd_even_depth_ratio"),
        "odd_even_depth_delta_explicit": metrics.get("odd_even_depth_delta_explicit"),
        "oot_variability_amp": metrics.get("oot_variability_amp"),
        "oot_variability_to_depth": metrics.get("oot_variability_to_depth"),
        "alias_best_period_days": metrics.get("alias_best_period_days"),
        "alias_best_support_count": metrics.get("alias_best_support_count"),
        "alias_best_support_ratio": metrics.get("alias_best_support_ratio"),
        "half_period_support_count": metrics.get("half_period_support_count"),
        "double_period_support_count": metrics.get("double_period_support_count"),
        "alias_risk": metrics.get("alias_risk"),
        "period_source": period_source,
        "cluster_center_phase": as_json(center_phase),
        "event_family_count": int(len(family)),
        "candidate_period_count": int(len(candidates)),
        "cnn_score": as_json(row.get("cnn_score")),
        "cnn_score_name": as_json(row.get("cnn_score_name")),
        "cnn_role": as_json(row.get("cnn_role")),
        "morphology_positive": as_json(row.get("morphology_positive")),
        "autovet_label": as_json(row.get("autovet_label")),
        "autovet_reason": as_json(row.get("explanation_short")),
        "master_label": as_json(row.get("master_label")),
        "master_reason": as_json(row.get("master_reason")),
        "master_next_action": as_json(row.get("master_next_action")),
        "review_level": as_json(row.get("review_level")),
        "decision_authority": as_json(row.get("decision_authority")),
        "manual_vetted": as_json(row.get("manual_vetted")),
        "manual_label": None,
        "manual_notes": None,
        "validation_summary_json_path": str(epic_dir / "validation_summary.json"),
    }

    return {
        "validation": validation,
        "queue_row": row_json(row),
        "plot_index_row": row_json(index_row),
        "artifacts": artifacts,
        "period_candidates": csv_records(epic_dir / "period_candidates.csv") if (epic_dir / "period_candidates.csv").exists() else [
            {str(k): as_json(v) for k, v in rec.items()} for rec in candidates.head(30).to_dict(orient="records")
        ],
        "alias_periods": metrics.get("alias_periods", []),
        "event_family": [{str(k): as_json(v) for k, v in rec.items()} for rec in family.to_dict(orient="records")],
        "notes": [
            "Generated for manual-vetting next64 plot pack.",
            "Plot-generation only; no labels, master catalog, or final candidate ledger were updated.",
            "CNN score is morphology_scorer_only evidence and is not decision authority.",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    if not QUEUE_CSV.exists():
        raise FileNotFoundError(QUEUE_CSV)
    if not INDEX_CSV.exists():
        raise FileNotFoundError(INDEX_CSV)

    queue = pd.read_csv(QUEUE_CSV).sort_values("queue_rank").reset_index(drop=True)
    index = pd.read_csv(INDEX_CSV)
    index_by_epic = {str(row["epic_id"]): pd.Series(row) for row in index.to_dict(orient="records")}

    written: list[Path] = []
    missing_index: list[str] = []
    for _, row in queue.iterrows():
        epic_id = str(row["epic_id"])
        if epic_id not in index_by_epic:
            missing_index.append(epic_id)
            continue
        epic_dir = OUT_DIR / epic_id
        epic_dir.mkdir(parents=True, exist_ok=True)
        summary = build_summary(row, index_by_epic[epic_id])
        out_path = epic_dir / "validation_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written.append(out_path)

    lines = [
        "Manual vetting next64 validation summary manifest",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"requested_epics={len(queue)}",
        f"validation_summary_json_written={len(written)}",
        f"missing_index_rows={len(missing_index)}",
        f"output_folder={rel(OUT_DIR)}",
        "",
        "Safety",
        "- JSON manifest generation only.",
        "- No retraining.",
        "- No label edits.",
        "- No master catalog or final candidate ledger updates.",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"validation_summary_json_written={len(written)}")
    print(f"missing_index_rows={len(missing_index)}")


if __name__ == "__main__":
    main()
