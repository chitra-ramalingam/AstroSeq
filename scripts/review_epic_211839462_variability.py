from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compare_epic_211340132_periods as period_tools
import scripts.refresh_gatevetter_unseen_full_validation as refresh


EPIC_ID = "EPIC_211839462"
SAVED_PERIOD = 5.21937
PERIODS = [
    (SAVED_PERIOD / 2.0, "half_period"),
    (SAVED_PERIOD, "saved_period"),
    (SAVED_PERIOD * 2.0, "double_period"),
]
DURATION_DAYS = 0.1736689327008207

METRICS_CSV = ROOT / f"{EPIC_ID}_period_comparison_metrics.csv"
BASELINE_CSV = ROOT / f"{EPIC_ID}_local_baseline_metrics.csv"
VARIABILITY_CSV = ROOT / f"{EPIC_ID}_variability_metrics.csv"
PANEL_PNG = ROOT / f"{EPIC_ID}_period_comparison_panel.png"
VARIABILITY_PNG = ROOT / f"{EPIC_ID}_variability_diagnostic.png"
SUMMARY_TXT = ROOT / f"{EPIC_ID}_deeper_review_summary.txt"


COLORS = {
    "raw": "#6a4c93",
    "corrected": "#1d3557",
    "removed": "#e76f51",
    "binned": "#c1121f",
}


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    median = float(np.nanmedian(values))
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def normalize_flux(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    return values / median - 1.0


def load_everest_series() -> dict[str, Any]:
    path = refresh.cached_fits_path(EPIC_ID)
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        time = np.asarray(data["TIME"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=float)
        corrected = np.asarray(data["FLUX"], dtype=float)
        raw = np.asarray(data["FRAW"], dtype=float)
    ok = (
        np.isfinite(time)
        & np.isfinite(corrected)
        & np.isfinite(raw)
        & (quality == 0)
    )
    time = time[ok]
    raw_norm = normalize_flux(raw[ok])
    corrected_norm = normalize_flux(corrected[ok])
    return {
        "time": time,
        "raw": raw_norm,
        "corrected": corrected_norm,
        "removed": raw_norm - corrected_norm,
        "cache_path": str(path),
    }


def phase_centered(time: np.ndarray, period: float, center: float) -> np.ndarray:
    return ((time / period - center + 0.5) % 1.0) - 0.5


def phase_bin(
    phase: np.ndarray,
    values: np.ndarray,
    bins: int = 120,
) -> pd.DataFrame:
    frame = pd.DataFrame({"phase": phase, "value": values}).dropna()
    frame["bin"] = pd.cut(
        frame["phase"],
        bins=np.linspace(-0.5, 0.5, bins + 1),
        include_lowest=True,
        labels=False,
    )
    grouped = (
        frame.dropna(subset=["bin"])
        .groupby("bin", observed=True)
        .agg(phase=("phase", "median"), value=("value", "median"))
        .reset_index(drop=True)
    )
    return grouped


def predicted_centers(
    time: np.ndarray,
    period: float,
    center_phase: float,
) -> np.ndarray:
    epoch_min = int(np.floor(time.min() / period - center_phase)) - 1
    epoch_max = int(np.ceil(time.max() / period - center_phase)) + 1
    centers = period * (
        np.arange(epoch_min, epoch_max + 1, dtype=float) + center_phase
    )
    margin = 3.5 * DURATION_DAYS
    return centers[
        (centers >= time.min() + margin)
        & (centers <= time.max() - margin)
    ]


def local_event_metrics(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    period_role: str,
    center_phase: float,
) -> pd.DataFrame:
    rows = []
    event_half = 0.5 * DURATION_DAYS
    side_inner = 1.25 * DURATION_DAYS
    side_outer = 3.0 * DURATION_DAYS
    for event_index, center in enumerate(
        predicted_centers(time, period, center_phase)
    ):
        delta = time - center
        event = np.abs(delta) <= event_half
        left = (delta >= -side_outer) & (delta <= -side_inner)
        right = (delta >= side_inner) & (delta <= side_outer)
        side = left | right
        if event.sum() < 3 or side.sum() < 8:
            continue
        coefficients = np.polyfit(delta[side], flux[side], deg=1)
        baseline = np.polyval(coefficients, delta)
        normalized = flux - baseline
        left_median = (
            float(np.nanmedian(normalized[left]))
            if left.sum() >= 3
            else float("nan")
        )
        right_median = (
            float(np.nanmedian(normalized[right]))
            if right.sum() >= 3
            else float("nan")
        )
        depth = -float(np.nanmedian(normalized[event]))
        scatter = robust_sigma(normalized[side])
        rows.append(
            {
                "epic_id": EPIC_ID,
                "period_role": period_role,
                "period_days": period,
                "center_phase": center_phase,
                "event_index": event_index,
                "predicted_center_bkjd": center,
                "event_points": int(event.sum()),
                "sideband_points": int(side.sum()),
                "left_sideband_points": int(left.sum()),
                "right_sideband_points": int(right.sum()),
                "both_sidebands_available": bool(
                    left.sum() >= 3 and right.sum() >= 3
                ),
                "local_depth": depth,
                "local_depth_snr": depth / scatter
                if np.isfinite(scatter) and scatter > 0
                else float("nan"),
                "sideband_sigma": scatter,
                "baseline_slope_per_day": float(coefficients[0]),
                "left_right_baseline_offset": right_median - left_median,
            }
        )
    return pd.DataFrame(rows)


def periodogram(
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, power = LombScargle(time, values).autopower(
        minimum_frequency=1.0 / 40.0,
        maximum_frequency=8.0,
        samples_per_peak=12,
    )
    return 1.0 / frequencies, power


def power_at(
    time: np.ndarray,
    values: np.ndarray,
    period: float,
) -> float:
    return float(LombScargle(time, values).power(1.0 / period))


def strongest_periods(
    periods: np.ndarray,
    power: np.ndarray,
    limit: int = 5,
) -> list[tuple[float, float]]:
    order = np.argsort(np.nan_to_num(power, nan=-np.inf))[::-1]
    peaks: list[tuple[float, float]] = []
    for index in order:
        period = float(periods[index])
        if all(abs(period / prior - 1.0) > 0.02 for prior, _ in peaks):
            peaks.append((period, float(power[index])))
        if len(peaks) >= limit:
            break
    return peaks


def event_mask(
    time: np.ndarray,
    period: float,
    center_phase: float,
) -> np.ndarray:
    phase = phase_centered(time, period, center_phase)
    return np.abs(phase) <= 0.75 * DURATION_DAYS / period


def variability_metrics(
    series: dict[str, Any],
    saved_center: float,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    time = series["time"]
    outside_events = ~event_mask(time, SAVED_PERIOD, saved_center)
    periodograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for name in ("raw", "corrected", "removed"):
        values = series[name]
        periods, power = periodogram(time[outside_events], values[outside_events])
        periodograms[name] = (periods, power)
        peaks = strongest_periods(periods, power)
        rows.append(
            {
                "series": name,
                "robust_sigma": robust_sigma(values[outside_events]),
                "p95_minus_p05": float(
                    np.nanpercentile(values[outside_events], 95)
                    - np.nanpercentile(values[outside_events], 5)
                ),
                "power_half_period": power_at(
                    time[outside_events],
                    values[outside_events],
                    SAVED_PERIOD / 2.0,
                ),
                "power_saved_period": power_at(
                    time[outside_events],
                    values[outside_events],
                    SAVED_PERIOD,
                ),
                "power_double_period": power_at(
                    time[outside_events],
                    values[outside_events],
                    SAVED_PERIOD * 2.0,
                ),
                "power_k2_thruster_0p245d": power_at(
                    time[outside_events],
                    values[outside_events],
                    0.245,
                ),
                "strongest_period_days": peaks[0][0],
                "strongest_power": peaks[0][1],
                "top_periods": "|".join(
                    f"{period:.7g}:{peak_power:.6g}"
                    for period, peak_power in peaks
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    raw = series["raw"][outside_events]
    corrected = series["corrected"][outside_events]
    removed = series["removed"][outside_events]
    metrics["raw_corrected_correlation"] = float(
        np.corrcoef(raw, corrected)[0, 1]
    )
    metrics["corrected_removed_correlation"] = float(
        np.corrcoef(corrected, removed)[0, 1]
    )
    return metrics, periodograms


def neighboring_trough_metrics(
    series: dict[str, Any],
    center_phase: float,
) -> dict[str, float]:
    phase = phase_centered(
        series["time"],
        SAVED_PERIOD,
        center_phase,
    )
    binned = phase_bin(phase, series["corrected"], bins=240)
    selected: list[tuple[float, float]] = []
    for row in binned.sort_values("value").itertuples():
        candidate_phase = float(row.phase)
        if all(
            abs(((candidate_phase - prior_phase + 0.5) % 1.0) - 0.5)
            > 0.08
            for prior_phase, _ in selected
        ):
            selected.append((candidate_phase, float(row.value)))
        if len(selected) >= 2:
            break
    if len(selected) < 2:
        return {
            "neighbor_trough_1_phase": float("nan"),
            "neighbor_trough_1_depth": float("nan"),
            "neighbor_trough_2_phase": float("nan"),
            "neighbor_trough_2_depth": float("nan"),
            "neighbor_trough_phase_separation": float("nan"),
        }
    first, second = selected
    separation = abs(((second[0] - first[0] + 0.5) % 1.0) - 0.5)
    return {
        "neighbor_trough_1_phase": first[0],
        "neighbor_trough_1_depth": -first[1],
        "neighbor_trough_2_phase": second[0],
        "neighbor_trough_2_depth": -second[1],
        "neighbor_trough_phase_separation": separation,
    }


def summarize_baselines(baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_role, group in baselines.groupby("period_role", sort=False):
        positive = group["local_depth"] > 0
        rows.append(
            {
                "period_role": period_role,
                "predicted_event_count": len(group),
                "positive_depth_count": int(positive.sum()),
                "positive_depth_fraction": float(positive.mean()),
                "median_local_depth": float(group["local_depth"].median()),
                "median_local_depth_snr": float(
                    group["local_depth_snr"].median()
                ),
                "depth_robust_sigma": robust_sigma(
                    group["local_depth"].to_numpy()
                ),
                "median_sideband_sigma": float(
                    group["sideband_sigma"].median()
                ),
                "median_abs_baseline_slope_per_day": float(
                    group["baseline_slope_per_day"].abs().median()
                ),
                "median_abs_left_right_offset": float(
                    group["left_right_baseline_offset"].abs().median()
                ),
                "both_sidebands_fraction": float(
                    group["both_sidebands_available"].mean()
                ),
                "left_sideband_available_fraction": float(
                    (group["left_sideband_points"] >= 3).mean()
                ),
                "right_sideband_available_fraction": float(
                    (group["right_sideband_points"] >= 3).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_event_stack(
    ax: plt.Axes,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    center_phase: float,
) -> None:
    event_half = 0.5 * DURATION_DAYS
    side_inner = 1.25 * DURATION_DAYS
    side_outer = 3.0 * DURATION_DAYS
    plotted = 0
    for index, center in enumerate(
        predicted_centers(time, period, center_phase)
    ):
        delta = time - center
        window = np.abs(delta) <= side_outer
        side = (
            ((delta >= -side_outer) & (delta <= -side_inner))
            | ((delta >= side_inner) & (delta <= side_outer))
        )
        if window.sum() < 8 or side.sum() < 6:
            continue
        coefficients = np.polyfit(delta[side], flux[side], deg=1)
        local = flux[window] - np.polyval(coefficients, delta[window])
        ax.plot(
            delta[window],
            local + index * 0.003,
            lw=0.75,
            alpha=0.82,
        )
        plotted += 1
    ax.axvline(0.0, color="#111111", lw=0.8)
    ax.axvspan(-event_half, event_half, color="#f4a261", alpha=0.14)
    ax.set_xlim(-side_outer, side_outer)
    ax.set_title(f"Predicted-epoch stack ({plotted} windows)")
    ax.set_xlabel("Days from predicted center")
    ax.set_ylabel("Local residual + offset")


def create_period_panel(
    period_metrics: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    centers: dict[str, float],
    series: dict[str, Any],
) -> None:
    time = series["time"]
    corrected = series["corrected"]
    fig, axes = plt.subplots(3, 3, figsize=(17, 16))
    for row_index, (period, role) in enumerate(PERIODS):
        center = centers[role]
        phase = phase_centered(time, period, center)
        binned = phase_bin(phase, corrected)
        axes[row_index, 0].scatter(
            phase,
            corrected,
            s=3,
            alpha=0.13,
            color=COLORS["corrected"],
            linewidths=0,
        )
        axes[row_index, 0].plot(
            binned["phase"],
            binned["value"],
            color=COLORS["binned"],
            lw=1.5,
        )
        axes[row_index, 0].axvline(0.0, color="#111111", lw=0.8)
        axes[row_index, 0].set_xlim(-0.5, 0.5)
        axes[row_index, 0].set_title(f"{role}: P={period:.6f} d")
        axes[row_index, 0].set_xlabel("Phase")
        axes[row_index, 0].set_ylabel("EVEREST corrected flux")

        plot_event_stack(
            axes[row_index, 1],
            time,
            corrected,
            period,
            center,
        )

        period_row = period_metrics.loc[
            period_metrics["period_role"].eq(role)
        ].iloc[0]
        baseline_row = baseline_summary.loc[
            baseline_summary["period_role"].eq(role)
        ].iloc[0]
        lines = [
            f"support family: {int(period_row['event_family_count'])}",
            f"primary depth: {period_row['primary_depth']:.6g}",
            f"primary SNR: {period_row['primary_depth_snr']:.4g}",
            f"odd/even ratio: {period_row['odd_even_depth_ratio']:.4g}",
            f"OOT/depth: {period_row['oot_to_depth']:.4g}",
            f"stack coherence: {period_row['event_stack_coherence']}",
            f"coherence score: {period_row['event_stack_coherence_score']:.4g}",
            "",
            f"predicted epochs: {int(baseline_row['predicted_event_count'])}",
            f"positive depth fraction: {baseline_row['positive_depth_fraction']:.3f}",
            f"median local depth: {baseline_row['median_local_depth']:.6g}",
            f"median local SNR: {baseline_row['median_local_depth_snr']:.3f}",
            f"median sideband sigma: {baseline_row['median_sideband_sigma']:.6g}",
            f"median |slope|/day: {baseline_row['median_abs_baseline_slope_per_day']:.6g}",
            f"median L/R offset: {baseline_row['median_abs_left_right_offset']:.6g}",
            f"both sidebands: {baseline_row['both_sidebands_fraction']:.3f}",
            f"left/right available: "
            f"{baseline_row['left_sideband_available_fraction']:.3f}/"
            f"{baseline_row['right_sideband_available_fraction']:.3f}",
        ]
        axes[row_index, 2].text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            transform=axes[row_index, 2].transAxes,
        )
        axes[row_index, 2].set_axis_off()
    fig.suptitle(
        f"{EPIC_ID}: saved, half, and double period comparison",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PANEL_PNG, dpi=170)
    plt.close(fig)


def create_variability_plot(
    series: dict[str, Any],
    variability: pd.DataFrame,
    periodograms: dict[str, tuple[np.ndarray, np.ndarray]],
    saved_center: float,
) -> None:
    time = series["time"]
    fig, axes = plt.subplots(3, 2, figsize=(17, 13))
    for name, offset in (("raw", 0.016), ("corrected", 0.0), ("removed", -0.012)):
        axes[0, 0].plot(
            time,
            series[name] + offset,
            lw=0.55,
            color=COLORS[name],
            label=name,
        )
    axes[0, 0].set_title("Raw, EVEREST-corrected, and removed component")
    axes[0, 0].set_xlabel("Time [BKJD]")
    axes[0, 0].set_ylabel("Normalized flux + offset")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(
        time,
        series["raw"] - series["corrected"],
        lw=0.65,
        color=COLORS["removed"],
    )
    axes[0, 1].axhline(0.0, color="#111111", lw=0.7)
    axes[0, 1].set_title("Component removed by EVEREST correction")
    axes[0, 1].set_xlabel("Time [BKJD]")
    axes[0, 1].set_ylabel("Raw - corrected")

    for name in ("raw", "corrected", "removed"):
        periods, power = periodograms[name]
        axes[1, 0].plot(
            periods,
            power,
            lw=1.0,
            color=COLORS[name],
            label=name,
        )
    for period, role in PERIODS:
        axes[1, 0].axvline(period, lw=0.8, ls="--", label=role)
    axes[1, 0].axvline(0.245, color="#777777", lw=0.8, ls=":", label="K2 0.245 d")
    axes[1, 0].set_xlim(0.2, 12.0)
    axes[1, 0].set_title("Lomb-Scargle periodogram, saved events masked")
    axes[1, 0].set_xlabel("Period [days]")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].legend(loc="upper right", ncol=2, fontsize=8)

    corrected_row = variability.loc[variability["series"].eq("corrected")].iloc[0]
    removed_row = variability.loc[variability["series"].eq("removed")].iloc[0]
    lines = [
        f"raw/corrected correlation: {corrected_row['raw_corrected_correlation']:.6f}",
        f"corrected/removed correlation: {corrected_row['corrected_removed_correlation']:.6f}",
        f"corrected p95-p05: {corrected_row['p95_minus_p05']:.6g}",
        f"removed p95-p05: {removed_row['p95_minus_p05']:.6g}",
        f"corrected strongest P: {corrected_row['strongest_period_days']:.6f} d",
        f"corrected power P/2: {corrected_row['power_half_period']:.5f}",
        f"corrected power P: {corrected_row['power_saved_period']:.5f}",
        f"corrected power 2P: {corrected_row['power_double_period']:.5f}",
        f"corrected power 0.245 d: {corrected_row['power_k2_thruster_0p245d']:.5f}",
        f"removed power P/2: {removed_row['power_half_period']:.5f}",
        f"removed power P: {removed_row['power_saved_period']:.5f}",
        f"neighbor trough 1: phase={corrected_row['neighbor_trough_1_phase']:.4f}, "
        f"depth={corrected_row['neighbor_trough_1_depth']:.5f}",
        f"neighbor trough 2: phase={corrected_row['neighbor_trough_2_phase']:.4f}, "
        f"depth={corrected_row['neighbor_trough_2_depth']:.5f}",
        f"trough separation: {corrected_row['neighbor_trough_phase_separation']:.4f}",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_axis_off()

    for column, period in enumerate((SAVED_PERIOD / 2.0, SAVED_PERIOD)):
        phase = phase_centered(time, period, saved_center)
        binned = phase_bin(phase, series["corrected"], bins=100)
        axes[2, column].scatter(
            phase,
            series["corrected"],
            s=3,
            alpha=0.12,
            color=COLORS["corrected"],
            linewidths=0,
        )
        axes[2, column].plot(
            binned["phase"],
            binned["value"],
            lw=1.6,
            color=COLORS["binned"],
        )
        axes[2, column].set_xlim(-0.5, 0.5)
        axes[2, column].set_title(f"Corrected variability folded at P={period:.6f} d")
        axes[2, column].set_xlabel("Phase")
        axes[2, column].set_ylabel("Normalized flux")
    fig.suptitle(
        f"{EPIC_ID}: astrophysical variability versus K2/systematics diagnostic",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(VARIABILITY_PNG, dpi=170)
    plt.close(fig)


def write_summary(
    period_metrics: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    variability: pd.DataFrame,
    series: dict[str, Any],
) -> None:
    corrected = variability.loc[variability["series"].eq("corrected")].iloc[0]
    raw = variability.loc[variability["series"].eq("raw")].iloc[0]
    removed = variability.loc[variability["series"].eq("removed")].iloc[0]
    saved = period_metrics.loc[
        period_metrics["period_role"].eq("saved_period")
    ].iloc[0]
    half = period_metrics.loc[
        period_metrics["period_role"].eq("half_period")
    ].iloc[0]
    double = period_metrics.loc[
        period_metrics["period_role"].eq("double_period")
    ].iloc[0]
    saved_baseline = baseline_summary.loc[
        baseline_summary["period_role"].eq("saved_period")
    ].iloc[0]
    half_baseline = baseline_summary.loc[
        baseline_summary["period_role"].eq("half_period")
    ].iloc[0]
    double_baseline = baseline_summary.loc[
        baseline_summary["period_role"].eq("double_period")
    ].iloc[0]

    correction_amp_ratio = (
        removed["p95_minus_p05"] / corrected["p95_minus_p05"]
        if corrected["p95_minus_p05"] > 0
        else float("nan")
    )
    target_power_ratio = (
        removed["power_half_period"] / corrected["power_half_period"]
        if corrected["power_half_period"] > 0
        else float("nan")
    )
    raw_corrected_persistence = (
        corrected["raw_corrected_correlation"] > 0.95
        and correction_amp_ratio < 0.35
        and target_power_ratio < 0.35
        and corrected["power_half_period"]
        > 5.0 * corrected["power_k2_thruster_0p245d"]
    )
    eclipse_like_neighbors = (
        corrected["neighbor_trough_1_depth"] > 0.01
        and corrected["neighbor_trough_2_depth"] > 0.01
        and abs(
            corrected["neighbor_trough_phase_separation"] - 0.5
        )
        < 0.08
    )
    quality_gap_edge_artifact = (
        saved_baseline["both_sidebands_fraction"] < 0.25
        and saved_baseline["left_sideband_available_fraction"] < 0.25
        and saved_baseline["right_sideband_available_fraction"] > 0.75
    )
    variability_class = (
        "k2_or_everest_quality_gap_edge_artifact_favored"
        if quality_gap_edge_artifact
        else "astrophysical_eclipsing_or_strong_periodic_variable_favored"
        if raw_corrected_persistence and eclipse_like_neighbors
        else "astrophysical_periodic_variability_favored"
        if raw_corrected_persistence
        else "mixed_or_systematic_origin_not_excluded"
    )

    lines = [
        f"{EPIC_ID} deeper period and variability review",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"light_curve={series['cache_path']}",
        "",
        "Period comparison",
        (
            f"- Half P={SAVED_PERIOD / 2.0:.6f} d: family={int(half['event_family_count'])}, "
            f"depth={half['primary_depth']:.7g}, SNR={half['primary_depth_snr']:.4g}, "
            f"odd/even={half['odd_even_depth_ratio']:.4g}, "
            f"coherence={half['event_stack_coherence']} "
            f"({half['event_stack_coherence_score']:.4f})."
        ),
        (
            f"- Saved P={SAVED_PERIOD:.6f} d: family={int(saved['event_family_count'])}, "
            f"depth={saved['primary_depth']:.7g}, SNR={saved['primary_depth_snr']:.4g}, "
            f"odd/even={saved['odd_even_depth_ratio']:.4g}, "
            f"coherence={saved['event_stack_coherence']} "
            f"({saved['event_stack_coherence_score']:.4f})."
        ),
        (
            f"- Double P={SAVED_PERIOD * 2.0:.6f} d: family={int(double['event_family_count'])}, "
            f"depth={double['primary_depth']:.7g}, SNR={double['primary_depth_snr']:.4g}, "
            f"odd/even={double['odd_even_depth_ratio']:.4g}, "
            f"coherence={double['event_stack_coherence']} "
            f"({double['event_stack_coherence_score']:.4f})."
        ),
        "",
        "Local baseline stability",
        (
            f"- Half period: positive-depth epochs={half_baseline['positive_depth_fraction']:.3f}, "
            f"median depth={half_baseline['median_local_depth']:.7g}, "
            f"median SNR={half_baseline['median_local_depth_snr']:.3f}, "
            f"median sideband sigma={half_baseline['median_sideband_sigma']:.7g}, "
            f"median |slope|/day={half_baseline['median_abs_baseline_slope_per_day']:.7g}, "
            f"both-sideband fraction={half_baseline['both_sidebands_fraction']:.3f}, "
            f"left/right available={half_baseline['left_sideband_available_fraction']:.3f}/"
            f"{half_baseline['right_sideband_available_fraction']:.3f}."
        ),
        (
            f"- Saved period: positive-depth epochs={saved_baseline['positive_depth_fraction']:.3f}, "
            f"median depth={saved_baseline['median_local_depth']:.7g}, "
            f"median SNR={saved_baseline['median_local_depth_snr']:.3f}, "
            f"median sideband sigma={saved_baseline['median_sideband_sigma']:.7g}, "
            f"median |slope|/day={saved_baseline['median_abs_baseline_slope_per_day']:.7g}, "
            f"both-sideband fraction={saved_baseline['both_sidebands_fraction']:.3f}, "
            f"left/right available={saved_baseline['left_sideband_available_fraction']:.3f}/"
            f"{saved_baseline['right_sideband_available_fraction']:.3f}."
        ),
        (
            f"- Double period: positive-depth epochs={double_baseline['positive_depth_fraction']:.3f}, "
            f"median depth={double_baseline['median_local_depth']:.7g}, "
            f"median SNR={double_baseline['median_local_depth_snr']:.3f}, "
            f"median sideband sigma={double_baseline['median_sideband_sigma']:.7g}, "
            f"median |slope|/day={double_baseline['median_abs_baseline_slope_per_day']:.7g}, "
            f"both-sideband fraction={double_baseline['both_sidebands_fraction']:.3f}, "
            f"left/right available={double_baseline['left_sideband_available_fraction']:.3f}/"
            f"{double_baseline['right_sideband_available_fraction']:.3f}."
        ),
        "",
        "Variability origin",
        f"- classification={variability_class}",
        f"- raw/corrected correlation={corrected['raw_corrected_correlation']:.6f}.",
        (
            f"- Raw and corrected strongest periods are {raw['strongest_period_days']:.6f} d "
            f"and {corrected['strongest_period_days']:.6f} d."
        ),
        (
            f"- Corrected P/2 power={corrected['power_half_period']:.6f}; "
            f"saved-P power={corrected['power_saved_period']:.6f}; "
            f"K2 0.245-d power={corrected['power_k2_thruster_0p245d']:.6f}."
        ),
        (
            f"- Removed correction amplitude / corrected amplitude={correction_amp_ratio:.4f}; "
            f"removed/corrected P/2 power={target_power_ratio:.4f}."
        ),
        (
            f"- Two neighboring folded troughs occur at phases "
            f"{corrected['neighbor_trough_1_phase']:.4f} and "
            f"{corrected['neighbor_trough_2_phase']:.4f}, with depths "
            f"{corrected['neighbor_trough_1_depth']:.5f} and "
            f"{corrected['neighbor_trough_2_depth']:.5f}; phase separation="
            f"{corrected['neighbor_trough_phase_separation']:.4f}."
        ),
        (
            "- All saved-period windows are one-sided after quality filtering: the "
            "pre-event sideband is absent while the post-event sideband is retained. "
            "The deep neighboring troughs and apparent 2.6/5.2-day periodicity therefore "
            "track recurring quality-gap edges and residual ramps. Raw/corrected agreement "
            "does not establish an astrophysical origin because the boundary artifact is "
            "present before EVEREST correction."
            if quality_gap_edge_artifact
            else "- The large OOT/depth ratio is dominated by deep neighboring eclipse-like "
            "events plus coherent modulation present in both raw and corrected light "
            "curves, not by the EVEREST-removed component."
            if raw_corrected_persistence
            else "- The raw/corrected/correction comparison does not cleanly isolate the "
            "OOT structure; a mixed astrophysical and systematic origin remains possible."
        ),
        "- Neighboring event detections and sloped local baselines are quantified in the event-level CSV.",
        "- This review does not modify manual labels or trusted-period fields.",
        "",
        f"period_metrics={METRICS_CSV.name}",
        f"local_baselines={BASELINE_CSV.name}",
        f"variability_metrics={VARIABILITY_CSV.name}",
        f"period_comparison_panel={PANEL_PNG.name}",
        f"variability_diagnostic={VARIABILITY_PNG.name}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    events = refresh.load_events(EPIC_ID)
    lc = refresh.load_cached_light_curve(EPIC_ID)
    series = load_everest_series()

    rows = []
    baseline_frames = []
    centers: dict[str, float] = {}
    for period, role in PERIODS:
        row, _family = period_tools.evaluate_period(
            events,
            lc,
            period,
            role,
        )
        row["epic_id"] = EPIC_ID
        row["period_role"] = role
        rows.append(row)
        center = float(row["cluster_center_phase"])
        centers[role] = center
        baseline_frames.append(
            local_event_metrics(
                series["time"],
                series["corrected"],
                period,
                role,
                center,
            )
        )

    period_metrics = pd.DataFrame(rows)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    baseline_summary = summarize_baselines(baselines)
    period_metrics = period_metrics.merge(
        baseline_summary,
        on="period_role",
        how="left",
        validate="one_to_one",
    )
    variability, periodograms = variability_metrics(
        series,
        centers["saved_period"],
    )
    troughs = neighboring_trough_metrics(
        series,
        centers["saved_period"],
    )
    for column, value in troughs.items():
        variability[column] = value

    period_metrics.to_csv(METRICS_CSV, index=False)
    baselines.to_csv(BASELINE_CSV, index=False)
    variability.to_csv(VARIABILITY_CSV, index=False)
    create_period_panel(period_metrics, baseline_summary, centers, series)
    create_variability_plot(
        series,
        variability,
        periodograms,
        centers["saved_period"],
    )
    write_summary(
        period_metrics,
        baseline_summary,
        variability,
        series,
    )

    print(period_metrics.to_string(index=False))
    print()
    print(variability.to_string(index=False))
    print()
    print(f"Wrote {PANEL_PNG.name}")
    print(f"Wrote {VARIABILITY_PNG.name}")
    print(f"Wrote {SUMMARY_TXT.name}")


if __name__ == "__main__":
    main()
