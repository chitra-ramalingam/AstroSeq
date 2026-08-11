from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2_NoiseHandler
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


QUEUE_CSV = (
    ROOT
    / "plots"
    / "k2_batch"
    / "master_vetted_catalog"
    / "cnn_backfill"
    / "manual_vetting_priority_queue_next64.csv"
)
OUT_DIR = ROOT / "plots" / "k2_batch" / "manual_vetting_next64_plot_pack"
INDEX_CSV = OUT_DIR / "manual_vetting_next64_plot_index.csv"
SUMMARY_TXT = OUT_DIR / "manual_vetting_next64_plot_pack_summary.txt"
EPICS_DIR = ROOT / "plots" / "k2_batch" / "epics"
KNOWN_VALIDATION_DIRS = [
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1",
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1",
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch2",
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch3",
]

BIN_COLOR = "#c1121f"
POINT_COLOR = "#264653"
ACCENT = "#e76f51"
EVEN = "#2a9d8f"
ODD = "#e76f51"

INDEX_COLUMNS = [
    "epic_id",
    "queue_rank",
    "cnn_score",
    "morphology_positive",
    "autovet_label",
    "autovet_reason",
    "period_days",
    "master_label",
    "review_level",
    "summary_panel_path",
    "raw_light_curve_path",
    "folded_light_curve_path",
    "transit_zoom_path",
    "odd_even_path",
    "secondary_check_path",
    "event_stack_path",
    "recommended_manual_action_blank",
    "manual_label_blank",
    "manual_notes_blank",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def fmt(value: Any, digits: int = 5) -> str:
    x = as_float(value)
    return f"{x:.{digits}g}" if np.isfinite(x) else "NA"


def save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def phase_centered(time: np.ndarray, period: float, center_phase: float) -> np.ndarray:
    phase = np.mod(np.asarray(time, dtype=float) / float(period), 1.0)
    return ((phase - float(center_phase) + 0.5) % 1.0) - 0.5


def phase_bin_median(phase: np.ndarray, flux: np.ndarray, bins: int = 120) -> pd.DataFrame:
    p = np.asarray(phase, dtype=float)
    f = np.asarray(flux, dtype=float)
    ok = np.isfinite(p) & np.isfinite(f)
    if not np.any(ok):
        return pd.DataFrame(columns=["phase", "median"])
    edges = np.linspace(-0.5, 0.5, int(bins) + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    cats = pd.cut(p[ok], bins=edges, labels=labels, include_lowest=True)
    tmp = pd.DataFrame({"bin": cats, "flux": f[ok]})
    out = tmp.groupby("bin", observed=False)["flux"].median().dropna().reset_index()
    out["phase"] = out["bin"].astype(float)
    return out[["phase", "flux"]].rename(columns={"flux": "median"})


def running_gap_threshold(time: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.asarray(time, dtype=float)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return float("inf")
    return float(max(np.nanmedian(diffs) * 8.0, 0.5))


def title_for(row: pd.Series, period: float) -> str:
    return (
        f"{row['epic_id']} | rank {int(row['queue_rank'])} | "
        f"CNN {fmt(row['cnn_score'], 4)} | {row['autovet_label']} | "
        f"P={fmt(period, 6)} d | master={row['master_label']} | authority={row['decision_authority']}"
    )


def load_light_curve(epic_id: str) -> dict[str, Any]:
    digits = epic_id.replace("EPIC_", "")
    query = f"EPIC {digits}"
    handler = K2_NoiseHandler(quality_strict=True)
    fetched = handler.fetch_best(query=query, cache_only=True)
    if str(fetched.get("status", "")).lower() != "ok":
        raise RuntimeError(str(fetched.get("status", "unavailable")))

    cleaned = handler.clean(
        fetched["lc"],
        normalize=False,
        remove_nans=True,
        quality_mask=True,
        sigma_clip=False,
        flatten=False,
    )
    time = np.asarray(cleaned["time"], dtype=float)
    raw_flux = np.asarray(cleaned["flux"], dtype=float)
    norm = K2SNR().normalize(time=time, flux=raw_flux)
    resid = np.asarray(norm["resid"], dtype=float)
    ok = np.isfinite(time) & np.isfinite(raw_flux) & np.isfinite(resid)
    return {
        "time": time[ok],
        "raw_flux": raw_flux[ok],
        "resid": resid[ok],
        "cache_path": str(fetched.get("cache_path", "")),
    }


def load_events(epic_id: str) -> pd.DataFrame:
    path = EPICS_DIR / epic_id / "events.csv"
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_csv(path)
    for col in ["t_start", "t_end", "t_mid", "duration_days", "depth", "depth_snr"]:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def candidate_periods_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if len(events) == 0 or "t_mid" not in events.columns:
        return pd.DataFrame(columns=["period_days", "support_count", "cluster_center_phase"])
    work = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    times = pd.to_numeric(work.get("t_mid", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(times) < 2:
        return pd.DataFrame(columns=["period_days", "support_count", "cluster_center_phase"])

    candidates: list[float] = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            delta = float(times[j] - times[i])
            if not np.isfinite(delta) or delta <= 0:
                continue
            for divisor in range(1, min(6, j - i + 2)):
                p = delta / float(divisor)
                if 0.2 <= p <= 40.0:
                    candidates.append(round(p, 5))
    if not candidates:
        return pd.DataFrame(columns=["period_days", "support_count", "cluster_center_phase"])

    rows: list[dict[str, Any]] = []
    for period in sorted(set(candidates)):
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(work, period=float(period), tol_phase=0.03)
        rows.append(
            {
                "period_days": float(period),
                "support_count": int(support),
                "cluster_center_phase": float(center),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["support_count", "period_days"], ascending=[False, True]).reset_index(drop=True)


def validation_period_artifacts(epic_id: str) -> tuple[float, float, pd.DataFrame]:
    for base in KNOWN_VALIDATION_DIRS:
        summary_path = base / epic_id / "validation_summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            validation = payload.get("validation", {})
            period = as_float(validation.get("best_period_days"))
            period_candidates_path = base / epic_id / "period_candidates.csv"
            candidates = pd.read_csv(period_candidates_path) if period_candidates_path.exists() else pd.DataFrame()
            center = float("nan")
            if len(candidates) > 0:
                candidates = candidates.rename(columns={"period_days": "period_days"})
                nearest = candidates.iloc[(pd.to_numeric(candidates["period_days"], errors="coerce") - period).abs().argsort()[:1]]
                if len(nearest):
                    center = as_float(nearest.iloc[0].get("cluster_center_phase"))
            if np.isfinite(period):
                return period, center, candidates
        except Exception:
            continue
    return float("nan"), float("nan"), pd.DataFrame()


def choose_period(row: pd.Series, events: pd.DataFrame) -> tuple[float, float, pd.DataFrame, str]:
    saved = as_float(row.get("best_period_days"))
    candidates = candidate_periods_from_events(events)
    if np.isfinite(saved):
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(events, period=saved, tol_phase=0.03)
        if not np.isfinite(center):
            center = 0.0
        if len(candidates) == 0:
            candidates = pd.DataFrame(
                [{"period_days": saved, "support_count": int(support), "cluster_center_phase": float(center)}]
            )
        return saved, float(center), candidates, "saved_best_period"
    validation_period, validation_center, validation_candidates = validation_period_artifacts(str(row["epic_id"]))
    if np.isfinite(validation_period):
        center = validation_center
        if not np.isfinite(center):
            support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
                events,
                period=validation_period,
                tol_phase=0.03,
            )
        if len(validation_candidates) > 0:
            candidates = validation_candidates.rename(
                columns={
                    "period_source": "period_source",
                    "support_count": "support_count",
                    "cluster_center_phase": "cluster_center_phase",
                }
            )
        return validation_period, float(center if np.isfinite(center) else 0.0), candidates, "saved_validation_period"
    if len(candidates) > 0:
        best = candidates.iloc[0]
        return float(best["period_days"]), float(best["cluster_center_phase"]), candidates, "event_spacing_fallback"
    return float("nan"), 0.0, candidates, "unavailable"


def family_events(events: pd.DataFrame, period: float, center_phase: float) -> pd.DataFrame:
    if len(events) == 0 or not np.isfinite(period):
        return pd.DataFrame()
    work = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    family = work.copy()
    family["folded_phase"] = phase_centered(family["t_mid"].to_numpy(dtype=float), period, center_phase)
    family = family.loc[family["folded_phase"].abs() <= 0.03].copy()
    if len(family) == 0:
        return family
    family = family.sort_values("t_mid").reset_index(drop=True)
    t0 = float(family["t_mid"].min())
    family["event_number"] = np.rint((family["t_mid"] - t0) / float(period)).astype(int)
    family["odd_even"] = family["event_number"].map(lambda n: "odd" if int(n) % 2 else "even")
    return family


def duration_days(row: pd.Series, family: pd.DataFrame, period: float) -> float:
    val = as_float(row.get("transit_duration_hours"))
    if np.isfinite(val) and val > 0:
        return val / 24.0
    if len(family) > 0 and "duration_days" in family.columns:
        med = as_float(pd.to_numeric(family["duration_days"], errors="coerce").median())
        if np.isfinite(med) and med > 0:
            return med
    if np.isfinite(period) and period > 0:
        return min(0.25, max(0.08, 0.03 * period))
    return float("nan")


def plot_raw(path: Path, row: pd.Series, lc: dict[str, Any], period: float) -> None:
    time = lc["time"]
    raw_flux = lc["raw_flux"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, raw_flux, lw=0.55, color=POINT_COLOR)
    threshold = running_gap_threshold(time)
    for left, right in zip(time[:-1], time[1:]):
        if right - left > threshold:
            ax.axvspan(left, right, color="#d9d9d9", alpha=0.5)
    ax.set_xlabel("Time [BKJD]")
    ax.set_ylabel("Flux")
    ax.set_title(title_for(row, period) + "\nRaw full light curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_detrended(path: Path, row: pd.Series, lc: dict[str, Any], period: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lc["time"], lc["resid"], lw=0.55, color=POINT_COLOR)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time [BKJD]")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nDetrended / normalized light curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_folded(path: Path, row: pd.Series, lc: dict[str, Any], period: float, center_phase: float) -> None:
    if not np.isfinite(period):
        save_placeholder(path, title_for(row, period), "No usable period available for folding")
        return
    phase = phase_centered(lc["time"], period, center_phase)
    binned = phase_bin_median(phase, lc["resid"], bins=140)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phase, lc["resid"], s=4, alpha=0.18, color=POINT_COLOR, linewidths=0)
    if len(binned):
        ax.plot(binned["phase"], binned["median"], color=BIN_COLOR, lw=1.5)
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Folded phase")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nFolded light curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transit_zoom(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
) -> None:
    if not np.isfinite(period):
        save_placeholder(path, title_for(row, period), "No usable period available for transit zoom")
        return
    phase = phase_centered(lc["time"], period, center_phase)
    binned = phase_bin_median(phase, lc["resid"], bins=160)
    dur_days = duration_days(row, family, period)
    half_width = float(np.clip((dur_days / period) * 1.6, 0.02, 0.12)) if np.isfinite(dur_days) else 0.08
    in_transit = np.abs(phase) <= half_width
    oot = np.abs(phase) >= min(0.45, half_width * 3.0)
    depth = float(np.nanmedian(lc["resid"][oot]) - np.nanmedian(lc["resid"][in_transit])) if np.any(in_transit) and np.any(oot) else float("nan")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phase, lc["resid"], s=5, alpha=0.18, color=POINT_COLOR, linewidths=0)
    if len(binned):
        ax.plot(binned["phase"], binned["median"], color=BIN_COLOR, lw=1.5)
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.axvspan(-half_width, half_width, color=ACCENT, alpha=0.12)
    ax.set_xlim(-max(0.18, half_width * 3.0), max(0.18, half_width * 3.0))
    ax.set_xlabel("Folded phase near transit")
    ax.set_ylabel("Detrended normalized flux")
    note = f"estimated depth={fmt(depth, 4)} | duration={fmt(dur_days * 24.0 if np.isfinite(dur_days) else np.nan, 4)} h"
    ax.set_title(title_for(row, period) + "\nTransit-window zoom | " + note)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_odd_even(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
) -> None:
    if not np.isfinite(period) or len(family) < 2 or family["odd_even"].nunique() < 2:
        save_placeholder(path, title_for(row, period), "Odd/even transit split unavailable")
        return
    phase = phase_centered(lc["time"], period, center_phase)
    fig, ax = plt.subplots(figsize=(10, 5))
    dur_days = duration_days(row, family, period)
    event_half_window = min(0.35, max(0.08, dur_days * 2.0 if np.isfinite(dur_days) else 0.12))
    for label, color in [("even", EVEN), ("odd", ODD)]:
        subset = family.loc[family["odd_even"].eq(label)]
        mask = np.zeros(len(lc["time"]), dtype=bool)
        for _, ev in subset.iterrows():
            tm = as_float(ev.get("t_mid"))
            if np.isfinite(tm):
                mask |= np.abs(lc["time"] - tm) <= event_half_window
        ax.scatter(phase[mask], lc["resid"][mask], s=8, alpha=0.32, color=color, label=label, linewidths=0)
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.set_xlim(-0.18, 0.18)
    ax.set_xlabel("Folded phase near transit")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nOdd/even transit comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_secondary(path: Path, row: pd.Series, lc: dict[str, Any], period: float, center_phase: float) -> None:
    if not np.isfinite(period):
        save_placeholder(path, title_for(row, period), "No usable period available for secondary-eclipse check")
        return
    secondary_center = (center_phase + 0.5) % 1.0
    phase = phase_centered(lc["time"], period, secondary_center)
    binned = phase_bin_median(phase, lc["resid"], bins=160)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phase, lc["resid"], s=5, alpha=0.18, color=POINT_COLOR, linewidths=0)
    if len(binned):
        ax.plot(binned["phase"], binned["median"], color=BIN_COLOR, lw=1.5)
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.set_xlim(-0.18, 0.18)
    ax.set_xlabel("Phase relative to 0.5")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nSecondary eclipse check")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_event_stack(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    family: pd.DataFrame,
) -> None:
    if not np.isfinite(period) or len(family) == 0:
        save_placeholder(path, title_for(row, period), "No repeating event family available for event stack")
        return
    dur_days = duration_days(row, family, period)
    window = min(0.5, max(0.12, dur_days * 2.5 if np.isfinite(dur_days) else 0.2))
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for idx, (_, ev) in enumerate(family.sort_values("t_mid").iterrows()):
        tm = as_float(ev.get("t_mid"))
        if not np.isfinite(tm):
            continue
        mask = np.abs(lc["time"] - tm) <= window
        if not np.any(mask):
            continue
        x = lc["time"][mask] - tm
        y = lc["resid"][mask] + idx * 0.002
        ax.plot(x, y, lw=0.7, alpha=0.75)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        save_placeholder(path, title_for(row, period), "No event windows available for stacking")
        return
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.set_xlabel("Time from event center [days]")
    ax.set_ylabel("Detrended flux + event offset")
    ax.set_title(title_for(row, period) + "\nEvent stack by event number")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_period_search(path: Path, row: pd.Series, period: float, candidates: pd.DataFrame) -> None:
    if len(candidates) == 0:
        save_placeholder(path, title_for(row, period), "No period-search candidates available")
        return
    top = candidates.head(30).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(top["period_days"], top["support_count"], color=POINT_COLOR, s=28)
    if np.isfinite(period):
        ax.axvline(period, color=BIN_COLOR, lw=1.5, label=f"chosen P={period:.6f} d")
        ax.legend(loc="best")
    ax.set_xlabel("Candidate period [days]")
    ax.set_ylabel("Event-family support count")
    ax.set_title(title_for(row, period) + "\nEvent-spacing period search")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_oot_variability(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
) -> None:
    if not np.isfinite(period):
        save_placeholder(path, title_for(row, period), "No usable period available for OOT variability check")
        return
    phase = phase_centered(lc["time"], period, center_phase)
    dur_days = duration_days(row, family, period)
    half_width = float(np.clip((dur_days / period) * 1.6, 0.02, 0.12)) if np.isfinite(dur_days) else 0.08
    secondary_phase = ((phase - 0.5 + 0.5) % 1.0) - 0.5
    oot = (np.abs(phase) > max(0.08, 2.0 * half_width)) & (np.abs(secondary_phase) > max(0.08, 2.0 * half_width))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lc["time"][oot], lc["resid"][oot], lw=0.55, color=POINT_COLOR)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time [BKJD]")
    ax.set_ylabel("Out-of-transit detrended flux")
    ax.set_title(title_for(row, period) + "\nOut-of-transit variability check")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
    candidates: pd.DataFrame,
    period_source: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.ravel()
    title = title_for(row, period)
    fig.suptitle(title, fontsize=14)

    axes[0].plot(lc["time"], lc["raw_flux"], lw=0.45, color=POINT_COLOR)
    axes[0].set_title("Raw light curve")
    axes[0].set_xlabel("BKJD")

    axes[1].plot(lc["time"], lc["resid"], lw=0.45, color=POINT_COLOR)
    axes[1].axhline(0.0, color="#111111", lw=0.7)
    axes[1].set_title("Detrended flux")
    axes[1].set_xlabel("BKJD")

    if np.isfinite(period):
        phase = phase_centered(lc["time"], period, center_phase)
        binned = phase_bin_median(phase, lc["resid"], bins=120)
        axes[2].scatter(phase, lc["resid"], s=2, alpha=0.14, color=POINT_COLOR)
        if len(binned):
            axes[2].plot(binned["phase"], binned["median"], color=BIN_COLOR, lw=1.2)
        axes[2].axvline(0.0, color="#111111", lw=0.8)
        axes[2].set_xlim(-0.5, 0.5)
        axes[2].set_title("Folded")

        dur_days = duration_days(row, family, period)
        half_width = float(np.clip((dur_days / period) * 1.6, 0.02, 0.12)) if np.isfinite(dur_days) else 0.08
        axes[3].scatter(phase, lc["resid"], s=2, alpha=0.14, color=POINT_COLOR)
        if len(binned):
            axes[3].plot(binned["phase"], binned["median"], color=BIN_COLOR, lw=1.2)
        axes[3].axvspan(-half_width, half_width, color=ACCENT, alpha=0.12)
        axes[3].axvline(0.0, color="#111111", lw=0.8)
        axes[3].set_xlim(-max(0.18, half_width * 3.0), max(0.18, half_width * 3.0))
        axes[3].set_title("Transit zoom")

        sec = phase_centered(lc["time"], period, (center_phase + 0.5) % 1.0)
        sec_bin = phase_bin_median(sec, lc["resid"], bins=120)
        axes[4].scatter(sec, lc["resid"], s=2, alpha=0.14, color=POINT_COLOR)
        if len(sec_bin):
            axes[4].plot(sec_bin["phase"], sec_bin["median"], color=BIN_COLOR, lw=1.2)
        axes[4].axvline(0.0, color="#111111", lw=0.8)
        axes[4].set_xlim(-0.18, 0.18)
        axes[4].set_title("Secondary check")
    else:
        for ax, name in zip(axes[2:5], ["Folded", "Transit zoom", "Secondary check"]):
            ax.text(0.5, 0.5, "No usable period", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)

    if np.isfinite(period) and len(family) >= 2 and family["odd_even"].nunique() >= 2:
        phase = phase_centered(lc["time"], period, center_phase)
        for label, color in [("even", EVEN), ("odd", ODD)]:
            subset = family.loc[family["odd_even"].eq(label)]
            mask = np.zeros(len(lc["time"]), dtype=bool)
            for _, ev in subset.iterrows():
                tm = as_float(ev.get("t_mid"))
                if np.isfinite(tm):
                    mask |= np.abs(lc["time"] - tm) <= 0.12
            axes[5].scatter(phase[mask], lc["resid"][mask], s=4, alpha=0.25, color=color, label=label)
        axes[5].axvline(0.0, color="#111111", lw=0.8)
        axes[5].set_xlim(-0.18, 0.18)
        axes[5].legend(loc="best", fontsize=8)
    else:
        axes[5].text(0.5, 0.5, "Odd/even unavailable", ha="center", va="center", transform=axes[5].transAxes)
    axes[5].set_title("Odd / even")

    if len(family) > 0:
        for idx, (_, ev) in enumerate(family.sort_values("t_mid").iterrows()):
            tm = as_float(ev.get("t_mid"))
            if not np.isfinite(tm):
                continue
            mask = np.abs(lc["time"] - tm) <= 0.18
            if np.any(mask):
                axes[6].plot(lc["time"][mask] - tm, lc["resid"][mask] + idx * 0.002, lw=0.6)
        axes[6].axvline(0.0, color="#111111", lw=0.8)
    else:
        axes[6].text(0.5, 0.5, "No event family", ha="center", va="center", transform=axes[6].transAxes)
    axes[6].set_title("Event stack")

    if len(candidates) > 0:
        top = candidates.head(20)
        axes[7].scatter(top["period_days"], top["support_count"], s=18, color=POINT_COLOR)
        if np.isfinite(period):
            axes[7].axvline(period, color=BIN_COLOR, lw=1.0)
    else:
        axes[7].text(0.5, 0.5, "No period candidates", ha="center", va="center", transform=axes[7].transAxes)
    axes[7].set_title("Period search")
    axes[7].set_xlabel("days")

    meta_lines = [
        f"queue_rank: {int(row['queue_rank'])}",
        f"cnn_score: {fmt(row['cnn_score'], 6)}",
        f"morphology_positive: {row['morphology_positive']}",
        f"autovet_label: {row['autovet_label']}",
        f"period_days: {fmt(period, 7)}",
        f"period_source: {period_source}",
        f"master_label: {row['master_label']}",
        f"review_level: {row['review_level']}",
        f"decision_authority: {row['decision_authority']}",
        f"event_family_count: {len(family)}",
    ]
    axes[8].text(0.02, 0.98, "\n".join(meta_lines), ha="left", va="top", transform=axes[8].transAxes)
    axes[8].set_axis_off()
    axes[8].set_title("Metadata")

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_one(row: pd.Series) -> dict[str, Any]:
    epic_id = str(row["epic_id"])
    epic_dir = OUT_DIR / epic_id
    epic_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw": epic_dir / "raw_light_curve.png",
        "detrended": epic_dir / "detrended_light_curve.png",
        "folded": epic_dir / "folded_light_curve_best_period.png",
        "zoom": epic_dir / "transit_window_zoom.png",
        "odd_even": epic_dir / "odd_even_transits.png",
        "secondary": epic_dir / "secondary_eclipse_check.png",
        "stack": epic_dir / "event_stack.png",
        "period": epic_dir / "periodogram_or_period_search.png",
        "oot": epic_dir / "oot_variability_check.png",
        "summary": epic_dir / "summary_panel.png",
    }

    try:
        lc = load_light_curve(epic_id)
    except Exception as exc:
        for path in paths.values():
            save_placeholder(path, title_for(row, float("nan")), f"Light curve unavailable: {exc}")
        return {
            "epic_id": epic_id,
            "queue_rank": int(row["queue_rank"]),
            "success": False,
            "missing_light_curve": True,
            "missing_saved_period": not np.isfinite(as_float(row.get("best_period_days"))),
            "period_days": float("nan"),
            "paths": paths,
        }

    events = load_events(epic_id)
    period, center_phase, candidates, period_source = choose_period(row, events)
    family = family_events(events, period, center_phase)

    plot_raw(paths["raw"], row, lc, period)
    plot_detrended(paths["detrended"], row, lc, period)
    plot_folded(paths["folded"], row, lc, period, center_phase)
    plot_transit_zoom(paths["zoom"], row, lc, period, center_phase, family)
    plot_odd_even(paths["odd_even"], row, lc, period, center_phase, family)
    plot_secondary(paths["secondary"], row, lc, period, center_phase)
    plot_event_stack(paths["stack"], row, lc, period, family)
    plot_period_search(paths["period"], row, period, candidates)
    plot_oot_variability(paths["oot"], row, lc, period, center_phase, family)
    plot_summary(paths["summary"], row, lc, period, center_phase, family, candidates, period_source)

    return {
        "epic_id": epic_id,
        "queue_rank": int(row["queue_rank"]),
        "success": True,
        "missing_light_curve": False,
        "missing_saved_period": not np.isfinite(as_float(row.get("best_period_days"))),
        "period_days": period,
        "paths": paths,
    }


def build_index(queue: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    result_by_epic = {item["epic_id"]: item for item in results}
    rows: list[dict[str, Any]] = []
    for _, row in queue.iterrows():
        result = result_by_epic[str(row["epic_id"])]
        paths = result["paths"]
        rows.append(
            {
                "epic_id": row["epic_id"],
                "queue_rank": int(row["queue_rank"]),
                "cnn_score": row["cnn_score"],
                "morphology_positive": row["morphology_positive"],
                "autovet_label": row["autovet_label"],
                "autovet_reason": row.get("explanation_short", ""),
                "period_days": result["period_days"],
                "master_label": row["master_label"],
                "review_level": row["review_level"],
                "summary_panel_path": rel(paths["summary"]),
                "raw_light_curve_path": rel(paths["raw"]),
                "folded_light_curve_path": rel(paths["folded"]),
                "transit_zoom_path": rel(paths["zoom"]),
                "odd_even_path": rel(paths["odd_even"]),
                "secondary_check_path": rel(paths["secondary"]),
                "event_stack_path": rel(paths["stack"]),
                "recommended_manual_action_blank": "",
                "manual_label_blank": "",
                "manual_notes_blank": "",
            }
        )
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def write_summary(queue: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    total = len(queue)
    successes = sum(bool(item["success"]) for item in results)
    failed = total - successes
    missing_lc = sum(bool(item["missing_light_curve"]) for item in results)
    missing_period = sum(bool(item["missing_saved_period"]) for item in results)
    lines = [
        "Manual vetting next-64 plot pack summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"total EPICs requested: {total}",
        f"plots generated successfully: {successes}",
        f"plots failed: {failed}",
        f"missing light curves: {missing_lc}",
        f"missing period information: {missing_period}",
        f"output folder: {rel(OUT_DIR)}",
        "",
        "Notes",
        "- Plot generation only; no retraining, labels, master catalog, or final ledger were updated.",
        "- Saved AutoVet/best periods are used when present.",
        "- Rows without saved period information use event-spacing period candidates for diagnostic plotting only.",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not QUEUE_CSV.exists():
        raise FileNotFoundError(QUEUE_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE_CSV).sort_values("queue_rank").reset_index(drop=True)
    results: list[dict[str, Any]] = []
    for _, row in queue.iterrows():
        results.append(build_one(row))

    index = build_index(queue, results)
    index.to_csv(INDEX_CSV, index=False)
    write_summary(queue, results)

    print(f"total_epics={len(queue)}")
    print(f"successes={sum(bool(item['success']) for item in results)}")
    print(f"failures={sum(not bool(item['success']) for item in results)}")
    print(f"missing_light_curves={sum(bool(item['missing_light_curve']) for item in results)}")
    print(f"missing_saved_periods={sum(bool(item['missing_saved_period']) for item in results)}")


if __name__ == "__main__":
    main()
