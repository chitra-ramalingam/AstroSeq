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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_manual_vetting_next64_plot_pack as plot_pack
import scripts.refresh_gatevetter_unseen_full_validation as refresh
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


EPIC_ID = "EPIC_211340132"
PERIODS = [
    (3.02388, "visual_packet_fallback_provisional"),
    (9.153408, "historical_period_search"),
    (1.51194, "half_visual_fallback"),
    (6.04776, "double_visual_fallback"),
    (18.306816, "double_historical_period"),
]
METRICS_CSV = ROOT / "EPIC_211340132_period_comparison_metrics.csv"
PANEL_PNG = ROOT / "EPIC_211340132_period_comparison_panel.png"
SUMMARY_TXT = ROOT / "EPIC_211340132_period_debug_summary.txt"


def as_float(value: Any) -> float:
    return refresh.as_float(value)


def robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    median = float(np.nanmedian(arr))
    return float(1.4826 * np.nanmedian(np.abs(arr - median)))


def family_for_period(events: pd.DataFrame, period: float, center: float) -> pd.DataFrame:
    family = plot_pack.family_events(events, period, center)
    if len(family):
        family = family.copy()
        family["family_epoch"] = family["event_number"]
    return family


def event_stack_coherence(family: pd.DataFrame) -> dict[str, Any]:
    if len(family) < 2:
        return {
            "event_stack_coherence_score": 0.0,
            "event_stack_coherence": "insufficient_events",
            "event_timing_rms_phase": float("nan"),
            "event_epoch_coverage": float("nan"),
            "event_depth_mad_fraction": float("nan"),
        }

    phases = pd.to_numeric(family["folded_phase"], errors="coerce").to_numpy(dtype=float)
    epochs = pd.to_numeric(family["event_number"], errors="coerce").dropna().astype(int)
    depths = pd.to_numeric(family.get("depth", pd.Series(dtype=float)), errors="coerce")
    timing_rms = float(np.sqrt(np.nanmean(phases**2)))
    predicted = int(epochs.max() - epochs.min() + 1) if len(epochs) else 0
    coverage = float(epochs.nunique() / predicted) if predicted > 0 else float("nan")
    depth_median = float(np.nanmedian(np.abs(depths))) if depths.notna().any() else float("nan")
    depth_scatter = robust_sigma(depths.to_numpy(dtype=float))
    depth_mad_fraction = (
        float(depth_scatter / depth_median)
        if np.isfinite(depth_scatter) and np.isfinite(depth_median) and depth_median > 0
        else float("nan")
    )

    phase_score = float(np.clip(1.0 - timing_rms / 0.03, 0.0, 1.0))
    coverage_score = float(np.clip(coverage, 0.0, 1.0)) if np.isfinite(coverage) else 0.0
    depth_score = (
        float(1.0 / (1.0 + depth_mad_fraction))
        if np.isfinite(depth_mad_fraction)
        else 0.0
    )
    score = 0.45 * phase_score + 0.30 * coverage_score + 0.25 * depth_score
    if score >= 0.75:
        label = "strong"
    elif score >= 0.55:
        label = "moderate"
    else:
        label = "weak"
    return {
        "event_stack_coherence_score": score,
        "event_stack_coherence": label,
        "event_timing_rms_phase": timing_rms,
        "event_epoch_coverage": coverage,
        "event_depth_mad_fraction": depth_mad_fraction,
    }


def evaluate_period(
    events: pd.DataFrame,
    lc: dict[str, Any],
    period: float,
    period_source: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    support, center = refresh.support_at(events, period)
    family = family_for_period(events, period, center)
    duration = as_float(
        pd.to_numeric(
            family.get("duration_days", pd.Series(dtype=float)), errors="coerce"
        ).median()
    )
    if not np.isfinite(duration) or duration <= 0:
        duration = max(0.08, min(0.30, 0.03 * period))
    half_width = float(np.clip(1.6 * duration / period, 0.015, 0.08))

    phase0 = K2StageFFollowupValidation._phase_centered(
        lc["time"], period, center
    )
    folded_primary = K2StageFFollowupValidation._folded_depth(
        phase=phase0,
        resid=lc["resid"],
        half_width_phase=half_width,
    )
    family_depth = pd.to_numeric(
        family.get("depth", pd.Series(dtype=float)), errors="coerce"
    )
    family_snr = pd.to_numeric(
        family.get("depth_snr", pd.Series(dtype=float)), errors="coerce"
    )
    primary_depth = as_float(family_depth.median())
    primary_snr = as_float(family_snr.median())
    if not np.isfinite(primary_depth) or primary_depth <= 0:
        primary_depth = as_float(folded_primary["depth"])
    if not np.isfinite(primary_snr) or primary_snr <= 0:
        primary_snr = as_float(folded_primary["snr"])

    phase05 = K2StageFFollowupValidation._phase_centered(
        lc["time"], period, (center + 0.5) % 1.0
    )
    secondary = K2StageFFollowupValidation._folded_depth(
        phase=phase05,
        resid=lc["resid"],
        half_width_phase=half_width,
    )
    secondary_depth = as_float(secondary["depth"])
    secondary_ratio = (
        float(secondary_depth / primary_depth)
        if np.isfinite(secondary_depth)
        and np.isfinite(primary_depth)
        and primary_depth > 0
        else float("nan")
    )

    odd_even = K2StageFFollowupValidation._epoch_depth_stats(family)
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    _, alias = K2StageFFollowupValidation._alias_stats(
        filtered,
        period=period,
        primary_support_count=max(support, len(family)),
    )
    oot = K2StageFFollowupValidation._oot_variability(
        phase0=phase0,
        resid=lc["resid"],
        primary_half_width=half_width,
    )
    oot_amp = as_float(oot["oot_variability_amp"])
    oot_to_depth = (
        float(oot_amp / primary_depth)
        if np.isfinite(oot_amp) and np.isfinite(primary_depth) and primary_depth > 0
        else float("nan")
    )
    coherence = event_stack_coherence(family)
    row = {
        "epic_id": EPIC_ID,
        "period_days": period,
        "period_source": period_source,
        "period_status": (
            "provisional_not_trusted"
            if period_source == "visual_packet_fallback_provisional"
            else "comparison_candidate"
        ),
        "cluster_center_phase": center,
        "event_family_count": int(len(family)),
        "primary_depth": primary_depth,
        "primary_depth_snr": primary_snr,
        "folded_primary_depth": as_float(folded_primary["depth"]),
        "folded_primary_depth_snr": as_float(folded_primary["snr"]),
        "transit_duration_days": duration,
        "odd_depth_median": as_float(odd_even["odd_depth_median"]),
        "even_depth_median": as_float(odd_even["even_depth_median"]),
        "odd_even_depth_ratio": as_float(odd_even["odd_even_depth_ratio"]),
        "oot_to_depth": oot_to_depth,
        "secondary_depth_phase_05": secondary_depth,
        "secondary_depth_snr": as_float(secondary["snr"]),
        "secondary_to_primary_depth_ratio": secondary_ratio,
        "alias_risk": str(alias["alias_risk"]),
        "alias_best_period_days": as_float(alias["alias_best_period_days"]),
        "alias_best_support_count": int(alias["alias_best_support_count"]),
        "alias_best_support_ratio": as_float(alias["alias_best_support_ratio"]),
        "half_period_support_count": int(alias["half_period_support_count"]),
        "double_period_support_count": int(alias["double_period_support_count"]),
        **coherence,
    }
    return row, family


def plot_folded(
    ax: plt.Axes,
    lc: dict[str, Any],
    row: dict[str, Any],
) -> None:
    phase = plot_pack.phase_centered(
        lc["time"], row["period_days"], row["cluster_center_phase"]
    )
    binned = plot_pack.phase_bin_median(phase, lc["resid"], bins=140)
    ax.scatter(phase, lc["resid"], s=3, alpha=0.15, color="#264653", linewidths=0)
    if len(binned):
        ax.plot(binned["phase"], binned["median"], color="#c1121f", lw=1.4)
    ax.axvline(0.0, color="#111111", lw=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Residual")
    ax.set_title(
        f"P={row['period_days']:.6f} d | {row['period_source']}\n"
        f"depth={row['primary_depth']:.4g}, SNR={row['primary_depth_snr']:.3g}"
    )


def plot_stack(
    ax: plt.Axes,
    lc: dict[str, Any],
    row: dict[str, Any],
    family: pd.DataFrame,
) -> None:
    duration = as_float(row["transit_duration_days"])
    window = min(0.6, max(0.12, 2.0 * duration))
    for idx, (_, event) in enumerate(family.sort_values("t_mid").iterrows()):
        midpoint = as_float(event.get("t_mid"))
        mask = np.abs(lc["time"] - midpoint) <= window
        if not np.any(mask):
            continue
        x = lc["time"][mask] - midpoint
        local = lc["resid"][mask]
        local = local - float(np.nanmedian(local))
        ax.plot(x, local + idx * 0.0012, lw=0.7, alpha=0.8)
    ax.axvline(0.0, color="#111111", lw=0.8)
    ax.set_xlim(-window, window)
    ax.set_title(
        f"Event stack: {row['event_stack_coherence']} "
        f"({row['event_stack_coherence_score']:.3f})\n"
        f"family={row['event_family_count']}, coverage={row['event_epoch_coverage']:.3f}"
    )


def plot_metrics(ax: plt.Axes, row: dict[str, Any]) -> None:
    lines = [
        f"center phase: {row['cluster_center_phase']:.6f}",
        f"primary depth: {row['primary_depth']:.7g}",
        f"primary SNR: {row['primary_depth_snr']:.4g}",
        f"odd/even ratio: {row['odd_even_depth_ratio']:.4g}",
        f"OOT/depth: {row['oot_to_depth']:.4g}",
        f"secondary/primary: {row['secondary_to_primary_depth_ratio']:.4g}",
        f"alias risk: {row['alias_risk']}",
        f"alias best P: {row['alias_best_period_days']:.6g}",
        f"alias support ratio: {row['alias_best_support_ratio']:.4g}",
        f"timing RMS phase: {row['event_timing_rms_phase']:.4g}",
        f"depth MAD fraction: {row['event_depth_mad_fraction']:.4g}",
        f"status: {row['period_status']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", transform=ax.transAxes)
    ax.set_axis_off()


def create_panel(
    metrics: pd.DataFrame,
    families: dict[float, pd.DataFrame],
    lc: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(len(metrics), 3, figsize=(17, 23))
    for idx, row_series in metrics.iterrows():
        row = row_series.to_dict()
        plot_folded(axes[idx, 0], lc, row)
        plot_stack(axes[idx, 1], lc, row, families[float(row["period_days"])])
        plot_metrics(axes[idx, 2], row)
    axes[-1, 0].set_xlabel("Folded phase")
    axes[-1, 1].set_xlabel("Days from event midpoint")
    fig.suptitle(
        f"{EPIC_ID} five-period diagnostic comparison\n"
        "P=3.02388 d remains provisional; no automatic period selection",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(PANEL_PNG, dpi=160)
    plt.close(fig)


def write_summary(metrics: pd.DataFrame) -> None:
    ranked = metrics.sort_values(
        ["event_stack_coherence_score", "primary_depth_snr"],
        ascending=[False, False],
    )
    fallback = metrics.loc[np.isclose(metrics["period_days"], 3.02388)].iloc[0]
    historical = metrics.loc[np.isclose(metrics["period_days"], 9.153408)].iloc[0]
    double_fallback = metrics.loc[np.isclose(metrics["period_days"], 6.04776)].iloc[0]
    lines = [
        f"{EPIC_ID} period debug summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "",
        "Guardrail",
        "- P=3.02388 d is a visual-packet event-spacing fallback and remains provisional.",
        "- This comparison does not update labels, GateVetter thresholds, or validation_period_days.",
        "",
        "Historical provenance",
        "- Historical period-search output: P=9.153408014513843 d, center_phase=0.3300506872548972.",
        "- Source: plots/k2_batch/20260227_192148/period_shortlist_best.csv.",
        "",
        "Comparison",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"- P={row.period_days:.6f} d ({row.period_source}): "
            f"depth={row.primary_depth:.8g}; SNR={row.primary_depth_snr:.5g}; "
            f"odd_even={row.odd_even_depth_ratio:.5g}; OOT/depth={row.oot_to_depth:.5g}; "
            f"secondary/primary={row.secondary_to_primary_depth_ratio:.5g}; "
            f"event_family={row.event_family_count}; coherence={row.event_stack_coherence} "
            f"({row.event_stack_coherence_score:.4f}); alias_risk={row.alias_risk}."
        )
    lines.extend(
        [
            "",
            "Debug interpretation",
            f"- Highest event-stack coherence in this five-period comparison: "
            f"P={ranked.iloc[0]['period_days']:.6f} d "
            f"({ranked.iloc[0]['event_stack_coherence_score']:.4f}).",
            f"- Fallback P=3.02388 d: coherence={fallback.event_stack_coherence_score:.4f}, "
            f"alias_risk={fallback.alias_risk}, odd_even={fallback.odd_even_depth_ratio:.5g}.",
            f"- Historical P=9.153408 d: coherence={historical.event_stack_coherence_score:.4f}, "
            f"alias_risk={historical.alias_risk}, odd_even={historical.odd_even_depth_ratio:.5g}.",
            f"- P=6.04776 d has the highest coherence, but its folded-primary SNR is only "
            f"{double_fallback.folded_primary_depth_snr:.4g}; its strong event-family SNR "
            "does not independently confirm a coherent folded transit.",
            f"- P=9.153408 d has event-family SNR={historical.primary_depth_snr:.5g}, but "
            f"folded-primary depth={historical.folded_primary_depth:.5g} and "
            f"SNR={historical.folded_primary_depth_snr:.5g}.",
            "- Debug conclusion: P=3.02388 d is not trusted. The period remains ambiguous; "
            "P=6.04776 d and historical P=9.153408 d are the leading manual-review comparisons, "
            "with neither confirmed by the folded-light-curve diagnostic.",
            "- Harmonic structure remains part of the diagnosis; no trusted period is forced by this script.",
            "",
            f"metrics_csv={METRICS_CSV.name}",
            f"comparison_panel={PANEL_PNG.name}",
        ]
    )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    lc = refresh.load_cached_light_curve(EPIC_ID)
    events = refresh.load_events(EPIC_ID)
    rows = []
    families: dict[float, pd.DataFrame] = {}
    for period, source in PERIODS:
        row, family = evaluate_period(events, lc, period, source)
        rows.append(row)
        families[period] = family
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_CSV, index=False)
    create_panel(metrics, families, lc)
    write_summary(metrics)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
