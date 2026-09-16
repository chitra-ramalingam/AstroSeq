from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_manual_vetting_next64_plot_pack as plot_pack
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


PACKETS = [
    ("stage_g_review", ROOT / "unseen_stage_g_review_manual_packet.csv"),
    ("top_holds", ROOT / "unseen_top_holds_manual_packet.csv"),
    ("reject_sanity_sample", ROOT / "unseen_reject_sanity_sample.csv"),
]
METRICS_CSV = ROOT / "unseen_full_validation_metrics.csv"
METRICS_SUMMARY = ROOT / "unseen_full_validation_metric_summary.txt"
PERIOD_REPORT = ROOT / "unseen_period_confirmation_report.csv"
PERIOD_SUMMARY = ROOT / "unseen_period_confirmation_summary.txt"
MANIFEST_CSV = ROOT / "unseen_manual_vetting_plot_manifest_refreshed.csv"
PLOT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_1_unseen_manual_review_refreshed"
EPICS_DIR = ROOT / "plots" / "k2_batch" / "epics"

REQUESTED_METRICS = [
    "primary_depth",
    "primary_depth_snr",
    "transit_duration_days",
    "transit_duration_hours",
    "duration_fraction_of_period",
    "odd_depth_median",
    "even_depth_median",
    "odd_even_depth_ratio",
    "secondary_depth_phase_05",
    "secondary_depth_snr",
    "secondary_to_primary_depth_ratio",
    "oot_to_depth",
    "alias_risk",
    "event_family_count",
    "candidate_period_count",
    "period_ambiguity_flag",
]


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def packet_rank(row: pd.Series, fallback: int) -> int:
    for col in ("manual_packet_rank", "sanity_sample_rank", "source_source_queue_rank"):
        value = as_float(row.get(col))
        if np.isfinite(value):
            return int(value)
    return fallback


def cached_fits_path(epic_id: str) -> Path:
    digits = "".join(ch for ch in epic_id if ch.isdigit())
    cache_root = Path.home() / ".lightkurve" / "cache" / "mastDownload"
    patterns = [
        f"HLSP/*{digits}*/*.fits",
        f"K2/*{digits}*/*.fits",
        f"**/*{digits}*/*.fits",
    ]
    for pattern in patterns:
        matches = sorted(cache_root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No cached FITS light curve for {epic_id}")


def load_cached_light_curve(epic_id: str) -> dict[str, Any]:
    path = cached_fits_path(epic_id)
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names)
        time = np.asarray(data["TIME"], dtype=float)
        if "FLUX" in names:
            flux = np.asarray(data["FLUX"], dtype=float)
            quality = np.asarray(data["QUALITY"], dtype=float) if "QUALITY" in names else np.zeros(len(time))
        else:
            flux_col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in names else "SAP_FLUX"
            quality_col = "SAP_QUALITY" if "SAP_QUALITY" in names else "QUALITY"
            flux = np.asarray(data[flux_col], dtype=float)
            quality = np.asarray(data[quality_col], dtype=float) if quality_col in names else np.zeros(len(time))
    ok = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
    time = time[ok]
    raw_flux = flux[ok]
    norm = K2SNR().normalize(time=time, flux=raw_flux)
    resid = np.asarray(norm["resid"], dtype=float)
    ok = np.isfinite(time) & np.isfinite(raw_flux) & np.isfinite(resid)
    return {
        "time": time[ok],
        "raw_flux": raw_flux[ok],
        "resid": resid[ok],
        "cache_path": str(path),
    }


def load_events(epic_id: str) -> pd.DataFrame:
    path = EPICS_DIR / epic_id / "events.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path)
    for col in ("t_start", "t_end", "t_mid", "duration_days", "depth", "depth_snr"):
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def event_candidates(events: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    candidates = plot_pack.candidate_periods_from_events(events)
    if len(candidates) == 0:
        return candidates
    return candidates.head(limit).assign(period_source="event_period_search")


def bls_candidates(lc: dict[str, Any], limit: int = 12) -> pd.DataFrame:
    time = lc["time"]
    resid = lc["resid"]
    baseline = float(np.nanmax(time) - np.nanmin(time))
    max_period = min(40.0, max(0.6, baseline / 2.0))
    periods = np.geomspace(0.5, max_period, 3500)
    durations = np.array([0.06, 0.09, 0.13, 0.20, 0.30])
    model = BoxLeastSquares(time, resid)
    power = model.power(periods, durations, objective="snr")
    order = np.argsort(np.nan_to_num(power.power, nan=-np.inf))[::-1]
    rows: list[dict[str, Any]] = []
    for idx in order:
        period = float(power.period[idx])
        if any(abs(period / row["period_days"] - 1.0) < 0.015 for row in rows):
            continue
        rows.append(
            {
                "period_days": period,
                "period_source": "bls_period_search",
                "bls_power": float(power.power[idx]),
                "bls_duration_days": float(power.duration[idx]),
                "bls_transit_time": float(power.transit_time[idx]),
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def support_at(events: pd.DataFrame, period: float) -> tuple[int, float]:
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
        events_df=filtered,
        period=float(period),
        tol_phase=0.03,
    )
    return int(support), float(center if np.isfinite(center) else 0.0)


def bls_at(lc: dict[str, Any], period: float) -> dict[str, float]:
    model = BoxLeastSquares(lc["time"], lc["resid"])
    durations = np.array([0.06, 0.09, 0.13, 0.20, 0.30])
    result = model.power(np.array([period]), durations, objective="snr")
    idx = int(np.nanargmax(result.power))
    return {
        "bls_power": float(result.power[idx]),
        "bls_duration_days": float(result.duration[idx]),
        "bls_transit_time": float(result.transit_time[idx]),
    }


def period_equivalent(a: float, b: float, tolerance: float = 0.025) -> bool:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return False
    ratio = a / b
    return any(abs(ratio - target) <= tolerance * target for target in (0.5, 1.0, 2.0))


def period_confirmation(
    row: pd.Series,
    events: pd.DataFrame,
    lc: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    saved = as_float(row.get("best_period_days"))
    saved_valid = bool(np.isfinite(saved) and 0.5 <= saved <= 40.0)
    ev = event_candidates(events)
    bls = bls_candidates(lc)
    fallback = as_float(ev.iloc[0]["period_days"]) if len(ev) else float("nan")

    seed_rows: list[dict[str, Any]] = []
    if saved_valid:
        seed_rows.append({"period_days": saved, "candidate_origin": "saved_best_period"})
        seed_rows.append({"period_days": saved / 2.0, "candidate_origin": "saved_half_period"})
        seed_rows.append({"period_days": saved * 2.0, "candidate_origin": "saved_double_period"})
    if np.isfinite(fallback):
        seed_rows.append({"period_days": fallback, "candidate_origin": "event_spacing_fallback"})
        seed_rows.append({"period_days": fallback / 2.0, "candidate_origin": "fallback_half_period"})
        seed_rows.append({"period_days": fallback * 2.0, "candidate_origin": "fallback_double_period"})
    for _, candidate in ev.iterrows():
        seed_rows.append({"period_days": candidate["period_days"], "candidate_origin": "top_event_period"})
    for _, candidate in bls.iterrows():
        seed_rows.append({"period_days": candidate["period_days"], "candidate_origin": "top_bls_period"})

    unique: list[dict[str, Any]] = []
    for seed in seed_rows:
        period = as_float(seed["period_days"])
        if not np.isfinite(period) or period < 0.5 or period > 40.0:
            continue
        if any(abs(period / item["period_days"] - 1.0) < 0.002 for item in unique):
            continue
        support, center = support_at(events, period)
        bls_metrics = bls_at(lc, period)
        unique.append(
            {
                **seed,
                "period_days": period,
                "event_support_count": support,
                "cluster_center_phase": center,
                **bls_metrics,
            }
        )
    candidates = pd.DataFrame(unique)
    if len(candidates) == 0:
        return {
            "saved_historical_period_days": saved,
            "event_spacing_fallback_period_days": fallback,
            "validation_period_days": float("nan"),
            "validation_period_source": "period_ambiguous",
            "period_ambiguity_flag": True,
            "period_confirmation_reason": "no_valid_period_candidates",
        }, candidates

    max_support = max(1.0, float(candidates["event_support_count"].max()))
    bls_values = pd.to_numeric(candidates["bls_power"], errors="coerce")
    finite_bls = bls_values[np.isfinite(bls_values)]
    if len(finite_bls) and float(finite_bls.max()) > float(finite_bls.min()):
        bls_score = (bls_values - float(finite_bls.min())) / (
            float(finite_bls.max()) - float(finite_bls.min())
        )
    else:
        bls_score = pd.Series(0.0, index=candidates.index)
    candidates["event_score"] = candidates["event_support_count"] / max_support
    candidates["bls_score"] = bls_score.fillna(0.0)
    candidates["combined_score"] = 0.58 * candidates["event_score"] + 0.42 * candidates["bls_score"]
    candidates["saved_period_match"] = (
        candidates["period_days"].map(lambda p: period_equivalent(float(p), saved))
        if saved_valid
        else False
    )
    if saved_valid:
        candidates.loc[
            (candidates["period_days"] - saved).abs() / saved < 0.002,
            "combined_score",
        ] += 0.18
    candidates = candidates.sort_values(
        ["combined_score", "event_support_count", "bls_power"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if saved_valid:
        saved_matches = candidates.loc[(candidates["period_days"] - saved).abs() / saved < 0.002]
        chosen = saved_matches.iloc[0] if len(saved_matches) else candidates.iloc[0]
    else:
        chosen = candidates.iloc[0]
    chosen_period = float(chosen["period_days"])

    alternatives = candidates.loc[
        (candidates["period_days"] - chosen_period).abs() / chosen_period >= 0.002
    ]
    second = alternatives.iloc[0] if len(alternatives) else None
    close_competitor = bool(
        second is not None
        and float(second["combined_score"]) >= 0.92 * float(chosen["combined_score"])
        and not period_equivalent(float(second["period_days"]), chosen_period)
    )
    harmonic_competitors = candidates.loc[
        candidates["period_days"].map(
            lambda p: period_equivalent(float(p), chosen_period)
            and abs(float(p) / chosen_period - 1.0) > 0.02
        )
    ]
    harmonic_ambiguous = bool(
        len(harmonic_competitors)
        and float(harmonic_competitors["combined_score"].max())
        >= 0.90 * float(chosen["combined_score"])
    )
    top_event_period = as_float(ev.iloc[0]["period_days"]) if len(ev) else float("nan")
    top_bls_period = as_float(bls.iloc[0]["period_days"]) if len(bls) else float("nan")
    search_disagreement = bool(
        not saved_valid
        and np.isfinite(top_event_period)
        and np.isfinite(top_bls_period)
        and not period_equivalent(top_event_period, top_bls_period)
    )
    weak_support = int(chosen["event_support_count"]) < 2
    ambiguous = close_competitor or harmonic_ambiguous or search_disagreement or weak_support

    exact_saved = saved_valid and abs(chosen_period / saved - 1.0) < 0.002
    if ambiguous:
        source = "period_ambiguous"
    elif exact_saved:
        source = "saved_best_period"
    elif len(bls):
        source = "refreshed_period_search"
    else:
        source = "event_spacing_fallback_only"
    reasons = []
    if close_competitor:
        reasons.append("near_tied_nonharmonic_candidate")
    if harmonic_ambiguous:
        reasons.append("half_or_double_period_competitor")
    if search_disagreement:
        reasons.append("bls_event_search_disagreement")
    if weak_support:
        reasons.append("event_support_below_two")
    if not reasons:
        reasons.append("best_candidate_separated")

    report = {
        "saved_historical_period_days": saved,
        "event_spacing_fallback_period_days": fallback,
        "half_period_days": chosen_period / 2.0,
        "double_period_days": chosen_period * 2.0,
        "validation_period_days": chosen_period,
        "validation_period_source": source,
        "period_ambiguity_flag": ambiguous,
        "period_confirmation_reason": "|".join(reasons),
        "validation_period_event_support": int(chosen["event_support_count"]),
        "validation_period_bls_power": float(chosen["bls_power"]),
        "validation_cluster_center_phase": float(chosen["cluster_center_phase"]),
        "validation_bls_duration_days": float(chosen["bls_duration_days"]),
        "candidate_period_count": int(len(candidates)),
        "top_event_periods": "|".join(f"{x:.8g}" for x in ev["period_days"].head(5)) if len(ev) else "",
        "top_bls_periods": "|".join(f"{x:.8g}" for x in bls["period_days"].head(5)) if len(bls) else "",
    }
    return report, candidates


def metric_missing_reasons(metrics: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for metric in REQUESTED_METRICS:
        value = metrics.get(metric)
        missing = value is None or (
            not isinstance(value, str) and not isinstance(value, bool) and not np.isfinite(as_float(value))
        )
        reasons[f"{metric}_missing_reason"] = (
            str(metrics.get("_missing_context", "recomputation_failed")) if missing else ""
        )
    return reasons


def recompute_metrics(
    epic_id: str,
    period_info: dict[str, Any],
    events: pd.DataFrame,
    lc: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    period = as_float(period_info["validation_period_days"])
    center = as_float(period_info["validation_cluster_center_phase"])
    family = plot_pack.family_events(events, period, center)
    duration = as_float(pd.to_numeric(family.get("duration_days", pd.Series(dtype=float)), errors="coerce").median())
    if not np.isfinite(duration) or duration <= 0:
        duration = as_float(period_info.get("validation_bls_duration_days"))
    if not np.isfinite(duration) or duration <= 0:
        duration = max(0.08, min(0.30, 0.03 * period))
    half_width = float(np.clip(1.6 * duration / period, 0.015, 0.08))

    phase0 = K2StageFFollowupValidation._phase_centered(lc["time"], period, center)
    primary = K2StageFFollowupValidation._folded_depth(
        phase=phase0,
        resid=lc["resid"],
        half_width_phase=half_width,
    )
    phase05 = K2StageFFollowupValidation._phase_centered(
        lc["time"], period, (center + 0.5) % 1.0
    )
    secondary = K2StageFFollowupValidation._folded_depth(
        phase=phase05,
        resid=lc["resid"],
        half_width_phase=half_width,
    )
    primary_depth = as_float(primary["depth"])
    secondary_depth = as_float(secondary["depth"])
    secondary_ratio = (
        secondary_depth / primary_depth
        if np.isfinite(primary_depth) and primary_depth > 0 and np.isfinite(secondary_depth)
        else float("nan")
    )
    odd_even_family = family.copy()
    if "event_number" in odd_even_family.columns:
        odd_even_family["family_epoch"] = odd_even_family["event_number"]
    odd_even = K2StageFFollowupValidation._epoch_depth_stats(odd_even_family)
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    alias_df, alias = K2StageFFollowupValidation._alias_stats(
        filtered,
        period=period,
        primary_support_count=int(period_info["validation_period_event_support"]),
    )
    oot = K2StageFFollowupValidation._oot_variability(
        phase0=phase0,
        resid=lc["resid"],
        primary_half_width=half_width,
    )
    oot_amp = as_float(oot["oot_variability_amp"])
    oot_ratio = (
        oot_amp / primary_depth
        if np.isfinite(oot_amp) and np.isfinite(primary_depth) and primary_depth > 0
        else float("nan")
    )
    ambiguous = bool(period_info["period_ambiguity_flag"])
    metrics = {
        "epic_id": epic_id,
        "validation_period_days": period,
        "validation_period_source": period_info["validation_period_source"],
        "primary_depth": primary_depth,
        "primary_depth_snr": as_float(primary["snr"]),
        "transit_duration_days": duration,
        "transit_duration_hours": duration * 24.0,
        "duration_fraction_of_period": duration / period,
        "odd_depth_median": as_float(odd_even["odd_depth_median"]),
        "even_depth_median": as_float(odd_even["even_depth_median"]),
        "odd_even_depth_ratio": as_float(odd_even["odd_even_depth_ratio"]),
        "secondary_depth_phase_05": secondary_depth,
        "secondary_depth_snr": as_float(secondary["snr"]),
        "secondary_to_primary_depth_ratio": secondary_ratio,
        "oot_to_depth": oot_ratio,
        "alias_risk": str(alias["alias_risk"]),
        "event_family_count": int(len(family)),
        "candidate_period_count": int(period_info["candidate_period_count"]),
        "period_ambiguity_flag": ambiguous,
        "alias_best_period_days": as_float(alias["alias_best_period_days"]),
        "alias_best_support_count": int(alias["alias_best_support_count"]),
        "alias_best_support_ratio": as_float(alias["alias_best_support_ratio"]),
        "half_period_support_count": int(alias["half_period_support_count"]),
        "double_period_support_count": int(alias["double_period_support_count"]),
        "light_curve_cache_path": lc["cache_path"],
        "events_csv": rel(EPICS_DIR / epic_id / "events.csv"),
        "_missing_context": "diagnostic_unavailable_after_recomputation",
    }
    if ambiguous:
        metrics.update(
            {
                "odd_depth_median": float("nan"),
                "even_depth_median": float("nan"),
                "odd_even_depth_ratio": float("nan"),
                "oot_to_depth": float("nan"),
                "alias_risk": "period_ambiguous",
                "_missing_context": "period_ambiguous_untrusted",
            }
        )
    metrics.update(metric_missing_reasons(metrics))
    metrics["missing_reason"] = "|".join(
        sorted(
            {
                reason
                for key, reason in metrics.items()
                if key.endswith("_missing_reason") and reason
            }
        )
    )
    return metrics, family


def plotting_row(packet_row: pd.Series, metrics: dict[str, Any], rank: int) -> pd.Series:
    return pd.Series(
        {
            "epic_id": packet_row["epic_id"],
            "queue_rank": rank,
            "cnn_score": packet_row.get("cnn_score", ""),
            "morphology_positive": "",
            "autovet_label": packet_row.get("source_autovet_label", ""),
            "explanation_short": packet_row.get("source_prefilter_reason", ""),
            "best_period_days": metrics["validation_period_days"],
            "validation_period_days": metrics["validation_period_days"],
            "validation_period_source": metrics["validation_period_source"],
            "period_ambiguity_flag": metrics["period_ambiguity_flag"],
            "primary_depth": metrics["primary_depth"],
            "primary_depth_snr": metrics["primary_depth_snr"],
            "transit_duration_hours": metrics["transit_duration_hours"],
            "odd_even_depth_ratio": metrics["odd_even_depth_ratio"],
            "secondary_depth_snr": metrics["secondary_depth_snr"],
            "secondary_to_primary_depth_ratio": metrics["secondary_to_primary_depth_ratio"],
            "oot_to_depth": metrics["oot_to_depth"],
            "alias_risk": metrics["alias_risk"],
            "event_family_count": metrics["event_family_count"],
            "candidate_period_count": metrics["candidate_period_count"],
            "master_label": packet_row.get("gatevetter_prediction", ""),
            "review_level": packet_row.get("stage_g_action", "") or "manual_packet_review",
            "decision_authority": "diagnostic_refresh_only_no_label_change",
        }
    )


def generate_plots(
    packet_source: str,
    packet_row: pd.Series,
    metrics: dict[str, Any],
    period_info: dict[str, Any],
    candidates: pd.DataFrame,
    family: pd.DataFrame,
    lc: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    epic_id = str(packet_row["epic_id"])
    epic_dir = PLOT_ROOT / packet_source / epic_id
    epic_dir.mkdir(parents=True, exist_ok=True)
    row = plotting_row(packet_row, metrics, rank)
    period = as_float(metrics["validation_period_days"])
    center = as_float(period_info["validation_cluster_center_phase"])
    paths = {
        "plot_full_lc_path": epic_dir / "raw_light_curve.png",
        "plot_detrended_lc_path": epic_dir / "detrended_light_curve.png",
        "plot_folded_path": epic_dir / "folded_light_curve_best_period.png",
        "plot_transit_zoom_path": epic_dir / "transit_window_zoom.png",
        "plot_secondary_path": epic_dir / "secondary_eclipse_check.png",
        "plot_odd_even_path": epic_dir / "odd_even_transits.png",
        "plot_event_stack_path": epic_dir / "event_stack.png",
        "plot_period_search_path": epic_dir / "periodogram_or_period_search.png",
        "plot_metadata_panel_path": epic_dir / "summary_panel.png",
    }
    plot_pack.plot_raw(paths["plot_full_lc_path"], row, lc, period)
    plot_pack.plot_detrended(paths["plot_detrended_lc_path"], row, lc, period)
    plot_pack.plot_folded(paths["plot_folded_path"], row, lc, period, center)
    plot_pack.plot_transit_zoom(paths["plot_transit_zoom_path"], row, lc, period, center, family)
    plot_pack.plot_secondary(paths["plot_secondary_path"], row, lc, period, center)
    plot_pack.plot_odd_even(paths["plot_odd_even_path"], row, lc, period, center, family)
    plot_pack.plot_event_stack(paths["plot_event_stack_path"], row, lc, period, family)
    plot_candidates = candidates.rename(columns={"event_support_count": "support_count"})
    plot_pack.plot_period_search(paths["plot_period_search_path"], row, period, plot_candidates)
    plot_pack.plot_summary(
        paths["plot_metadata_panel_path"],
        row,
        lc,
        period,
        center,
        family,
        plot_candidates,
        metrics["validation_period_source"],
    )
    candidates_path = epic_dir / "period_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    summary_path = epic_dir / "validation_summary.json"
    payload = {
        "validation": {k: json_safe(v) for k, v in metrics.items() if not k.startswith("_")},
        "period_confirmation": {k: json_safe(v) for k, v in period_info.items()},
        "packet_source": packet_source,
        "packet_row": {str(k): json_safe(v) for k, v in packet_row.to_dict().items()},
        "period_candidates_csv": rel(candidates_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "Diagnostics recomputed from cached light curve, period search, and events.csv.",
            "Packet metric columns were not used as diagnostic inputs.",
            "Manual labels and GateVetter thresholds were not changed.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "epic_id": epic_id,
        "packet_source": packet_source,
        **{key: rel(path) for key, path in paths.items()},
        "period_candidates_csv": rel(candidates_path),
        "validation_summary_json": rel(summary_path),
        "missing_plot_types": "none",
    }


def refreshed_packet(
    packet: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    packet_source: str,
) -> pd.DataFrame:
    diagnostic_columns = [
        "validation_period_days",
        "validation_period_source",
        *REQUESTED_METRICS,
        "missing_reason",
    ]
    missing_cols = [f"{name}_missing_reason" for name in REQUESTED_METRICS]
    replacement = metrics[["epic_id", *diagnostic_columns, *missing_cols]].copy()
    out = packet.drop(
        columns=[col for col in diagnostic_columns + missing_cols if col in packet.columns],
        errors="ignore",
    ).merge(replacement, on="epic_id", how="left", validate="one_to_one")
    plot_cols = [col for col in manifest.columns if col not in {"packet_source"}]
    out = out.merge(
        manifest.loc[manifest["packet_source"].eq(packet_source), plot_cols],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    return out


def write_summaries(metrics: pd.DataFrame, periods: pd.DataFrame) -> None:
    metric_lines = [
        "Unseen full validation metric refresh",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"epics={len(metrics)}",
        f"period_ambiguous={int(metrics['period_ambiguity_flag'].sum())}",
        f"rows_with_missing_reason={int(metrics['missing_reason'].fillna('').ne('').sum())}",
        "",
        "validation_period_source",
    ]
    metric_lines.extend(
        f"{key}={value}"
        for key, value in metrics["validation_period_source"].value_counts(dropna=False).items()
    )
    metric_lines.extend(["", "missing diagnostics"])
    for name in REQUESTED_METRICS:
        col = f"{name}_missing_reason"
        metric_lines.append(f"{name}={int(metrics[col].fillna('').ne('').sum())}")
    METRICS_SUMMARY.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    period_lines = [
        "Unseen period confirmation refresh",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"epics={len(periods)}",
        f"saved_best_period={int(periods['validation_period_source'].eq('saved_best_period').sum())}",
        f"refreshed_period_search={int(periods['validation_period_source'].eq('refreshed_period_search').sum())}",
        f"event_spacing_fallback_only={int(periods['validation_period_source'].eq('event_spacing_fallback_only').sum())}",
        f"period_ambiguous={int(periods['validation_period_source'].eq('period_ambiguous').sum())}",
        "",
        "Ambiguous EPICs",
    ]
    ambiguous = periods.loc[periods["period_ambiguity_flag"]]
    period_lines.extend(
        f"{row.epic_id}: P={row.validation_period_days:.8g}; {row.period_confirmation_reason}"
        for row in ambiguous.itertuples()
    )
    PERIOD_SUMMARY.write_text("\n".join(period_lines) + "\n", encoding="utf-8")


def main() -> None:
    packet_frames: dict[str, pd.DataFrame] = {}
    combined_rows = []
    for packet_source, path in PACKETS:
        packet = pd.read_csv(path)
        packet["epic_id"] = packet["epic_id"].astype(str)
        packet_frames[packet_source] = packet
        combined_rows.append(packet.assign(packet_source=packet_source))
    combined = pd.concat(combined_rows, ignore_index=True)
    if combined["epic_id"].duplicated().any():
        raise ValueError("Unseen packet EPICs must be unique across packets")

    metric_rows = []
    period_rows = []
    manifest_rows = []
    for idx, packet_row in combined.iterrows():
        epic_id = str(packet_row["epic_id"])
        packet_source = str(packet_row["packet_source"])
        print(f"[{idx + 1:02d}/{len(combined)}] {epic_id} {packet_source}", flush=True)
        lc = load_cached_light_curve(epic_id)
        events = load_events(epic_id)
        period_info, candidates = period_confirmation(packet_row, events, lc)
        if not np.isfinite(as_float(period_info["validation_period_days"])):
            raise RuntimeError(f"No validation period for {epic_id}")
        metrics, family = recompute_metrics(epic_id, period_info, events, lc)
        metric_rows.append({"packet_source": packet_source, **metrics})
        period_rows.append(
            {
                "epic_id": epic_id,
                "packet_source": packet_source,
                **period_info,
            }
        )
        manifest_rows.append(
            generate_plots(
                packet_source,
                packet_row,
                metrics,
                period_info,
                candidates,
                family,
                lc,
                packet_rank(packet_row, idx + 1),
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    periods_df = pd.DataFrame(period_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    periods_df.to_csv(PERIOD_REPORT, index=False)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    write_summaries(metrics_df, periods_df)

    output_names = {
        "stage_g_review": ROOT / "unseen_stage_g_review_manual_packet_refreshed.csv",
        "top_holds": ROOT / "unseen_top_holds_manual_packet_refreshed.csv",
        "reject_sanity_sample": ROOT / "unseen_reject_sanity_sample_refreshed.csv",
    }
    for packet_source, packet in packet_frames.items():
        refreshed_packet(packet, metrics_df, manifest_df, packet_source).to_csv(
            output_names[packet_source],
            index=False,
        )
    print(f"Wrote {len(metrics_df)} refreshed validation rows", flush=True)


if __name__ == "__main__":
    main()
