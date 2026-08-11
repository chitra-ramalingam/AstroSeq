from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_manual_vetting_next64_plot_pack import (
    ACCENT,
    BIN_COLOR,
    EVEN,
    ODD,
    OUT_DIR,
    POINT_COLOR,
    QUEUE_CSV,
    ROOT,
    as_float,
    candidate_periods_from_events,
    duration_days,
    family_events,
    fmt,
    load_events,
    load_light_curve,
    phase_centered,
    title_for,
)


EPIC_ID = "EPIC_211357782"
BEST_PERIOD_DAYS = 4.20484
PERIODS = [
    ("best_period", BEST_PERIOD_DAYS),
    ("half_period", 2.10242),
    ("double_period", 8.40968),
]
ENHANCED_DIR = OUT_DIR / EPIC_ID / "enhanced_manual_vetting"


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bin_phase(phase: np.ndarray, flux: np.ndarray, bins: int = 80) -> pd.DataFrame:
    p = np.asarray(phase, dtype=float)
    f = np.asarray(flux, dtype=float)
    ok = np.isfinite(p) & np.isfinite(f)
    if not np.any(ok):
        return pd.DataFrame(columns=["phase", "median", "err", "n"])

    edges = np.linspace(-0.5, 0.5, bins + 1)
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = ok & (p >= left) & (p < right)
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        vals = f[mask]
        med = float(np.nanmedian(vals))
        if n > 1:
            mad = float(np.nanmedian(np.abs(vals - med)))
            err = 1.4826 * mad / np.sqrt(n)
            if not np.isfinite(err) or err == 0:
                err = float(np.nanstd(vals) / np.sqrt(n))
        else:
            err = float("nan")
        rows.append(
            {
                "phase": float(0.5 * (left + right)),
                "median": med,
                "err": err,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def binned_for_window(phase: np.ndarray, flux: np.ndarray, xlim: tuple[float, float], bins: int) -> pd.DataFrame:
    binned = bin_phase(phase, flux, bins=bins)
    if len(binned) == 0:
        return binned
    return binned.loc[binned["phase"].between(xlim[0], xlim[1])].reset_index(drop=True)


def robust_ylim(values: np.ndarray, pad_frac: float = 0.18) -> tuple[float, float]:
    vals = _finite(values)
    if len(vals) == 0:
        return (-0.005, 0.005)
    lo, hi = np.nanpercentile(vals, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(vals))
        return center - 0.002, center + 0.002
    pad = max((hi - lo) * pad_frac, 0.00025)
    return float(lo - pad), float(hi + pad)


def shared_phase_ylim(lc: dict[str, Any], periods: list[float], centers: dict[float, float]) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for period in periods:
        center = centers[period]
        phase = phase_centered(lc["time"], period, center)
        chunks.append(lc["resid"][np.abs(phase) <= 0.22])
        sec = phase_centered(lc["time"], period, (center + 0.5) % 1.0)
        chunks.append(lc["resid"][np.abs(sec) <= 0.22])
    if not chunks:
        return robust_ylim(lc["resid"])
    return robust_ylim(np.concatenate(chunks))


def annotate_bin_counts(ax: plt.Axes, binned: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    if len(binned) == 0:
        return
    visible = binned.loc[binned["phase"].between(xlim[0], xlim[1])]
    if len(visible) == 0:
        return
    step = max(1, int(np.ceil(len(visible) / 28)))
    y = ylim[0] + 0.035 * (ylim[1] - ylim[0])
    for _, item in visible.iloc[::step].iterrows():
        ax.text(
            float(item["phase"]),
            y,
            str(int(item["n"])),
            ha="center",
            va="bottom",
            fontsize=6,
            color="#555555",
            rotation=90,
        )
    ax.text(
        0.995,
        0.02,
        "bin n",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )


def plot_phase_panel(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
    title_suffix: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bins: int,
    mark_transit: bool = True,
) -> None:
    phase = phase_centered(lc["time"], period, center_phase)
    binned = binned_for_window(phase, lc["resid"], xlim, bins=bins)
    dur = duration_days(row, family, period)
    half_width = float(np.clip((dur / period) * 1.6, 0.02, 0.12)) if np.isfinite(dur) else 0.08

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(phase, lc["resid"], s=5, alpha=0.16, color=POINT_COLOR, linewidths=0)
    if len(binned):
        ax.errorbar(
            binned["phase"],
            binned["median"],
            yerr=binned["err"].fillna(0.0),
            fmt="o",
            ms=4.2,
            lw=0.8,
            capsize=2,
            color=BIN_COLOR,
            ecolor=BIN_COLOR,
            label="binned median +/- robust SE",
        )
    if mark_transit:
        ax.axvspan(-half_width, half_width, color=ACCENT, alpha=0.16, label="expected transit window")
        ax.axvline(-half_width, color=ACCENT, lw=0.9, ls="--")
        ax.axvline(half_width, color=ACCENT, lw=0.9, ls="--")
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    annotate_bin_counts(ax, binned, xlim, ylim)
    ax.set_xlabel("Folded phase")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + f"\n{title_suffix}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_secondary_panel(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    ylim: tuple[float, float],
) -> None:
    secondary_center = (center_phase + 0.5) % 1.0
    phase = phase_centered(lc["time"], period, secondary_center)
    xlim = (-0.18, 0.18)
    binned = binned_for_window(phase, lc["resid"], xlim, bins=90)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(phase, lc["resid"], s=5, alpha=0.16, color=POINT_COLOR, linewidths=0)
    if len(binned):
        ax.errorbar(
            binned["phase"],
            binned["median"],
            yerr=binned["err"].fillna(0.0),
            fmt="o",
            ms=4.2,
            lw=0.8,
            capsize=2,
            color=BIN_COLOR,
            ecolor=BIN_COLOR,
        )
    ax.axvspan(-0.025, 0.025, color=ACCENT, alpha=0.13, label="phase-0.5 check window")
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    annotate_bin_counts(ax, binned, xlim, ylim)
    ax.set_xlabel("Phase relative to expected secondary")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nSecondary eclipse check with binned error bars")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_odd_even_panel(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    center_phase: float,
    family: pd.DataFrame,
    ylim: tuple[float, float],
) -> None:
    phase = phase_centered(lc["time"], period, center_phase)
    xlim = (-0.18, 0.18)
    dur = duration_days(row, family, period)
    event_half_window = min(0.35, max(0.08, dur * 2.0 if np.isfinite(dur) else 0.12))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, color in [("even", EVEN), ("odd", ODD)]:
        subset = family.loc[family["odd_even"].eq(label)]
        mask = np.zeros(len(lc["time"]), dtype=bool)
        for _, ev in subset.iterrows():
            tm = as_float(ev.get("t_mid"))
            if np.isfinite(tm):
                mask |= np.abs(lc["time"] - tm) <= event_half_window
        ax.scatter(phase[mask], lc["resid"][mask], s=9, alpha=0.27, color=color, label=f"{label} events", linewidths=0)
        binned = binned_for_window(phase[mask], lc["resid"][mask], xlim, bins=45)
        if len(binned):
            ax.errorbar(
                binned["phase"],
                binned["median"],
                yerr=binned["err"].fillna(0.0),
                fmt="o",
                ms=4.4,
                lw=0.8,
                capsize=2,
                color=color,
                ecolor=color,
            )
            annotate_bin_counts(ax, binned, xlim, ylim)
    ax.axvspan(-0.025, 0.025, color=ACCENT, alpha=0.12, label="expected transit core")
    ax.axvline(0.0, color="#111111", lw=1.0)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Folded phase near transit")
    ax.set_ylabel("Detrended normalized flux")
    ax.set_title(title_for(row, period) + "\nOdd/even comparison with binned error bars")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_marked_light_curve(
    path: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    family: pd.DataFrame,
    flux_key: str,
    ylabel: str,
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(lc["time"], lc[flux_key], lw=0.55, color=POINT_COLOR)
    for idx, (_, ev) in enumerate(family.sort_values("t_mid").iterrows(), start=1):
        tm = as_float(ev.get("t_mid"))
        if not np.isfinite(tm):
            continue
        ax.axvline(tm, color=ACCENT, lw=1.0, alpha=0.9)
        ax.text(tm, 0.97, f"E{idx}", rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform(), fontsize=8)
    if flux_key == "resid":
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.45)
    ax.set_xlabel("Time [BKJD]")
    ax.set_ylabel(ylabel)
    ax.set_title(title_for(row, period) + f"\n{title_suffix}: individual event centers marked")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_event_metric(path: Path, row: pd.Series, family: pd.DataFrame, period: float, metric: str, ylabel: str) -> None:
    ordered = family.sort_values("t_mid").reset_index(drop=True).copy()
    x = np.arange(1, len(ordered) + 1)
    vals = pd.to_numeric(ordered[metric], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x, vals, color=POINT_COLOR, alpha=0.82)
    ax.plot(x, vals, color=BIN_COLOR, lw=1.2, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}\n{tm:.2f}" for i, tm in zip(x, ordered["t_mid"].to_numpy(dtype=float))])
    ax.set_xlabel("Event number and center [BKJD]")
    ax.set_ylabel(ylabel)
    ax.set_title(title_for(row, period) + f"\nEvent-by-event {ylabel}")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_event_cutouts(path: Path, row: pd.Series, lc: dict[str, Any], period: float, family: pd.DataFrame) -> None:
    ordered = family.sort_values("t_mid").reset_index(drop=True)
    n = len(ordered)
    cols = 1
    fig, axes = plt.subplots(n, cols, figsize=(10, max(2.0 * n, 6.0)), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    dur = duration_days(row, family, period)
    window = max(0.18, min(0.7, (dur * 4.0) if np.isfinite(dur) else 0.35))
    ylim = robust_ylim(lc["resid"])
    for idx, (ax, (_, ev)) in enumerate(zip(axes_arr, ordered.iterrows()), start=1):
        tm = as_float(ev.get("t_mid"))
        mask = np.abs(lc["time"] - tm) <= window
        ax.scatter(lc["time"][mask] - tm, lc["resid"][mask], s=12, alpha=0.65, color=POINT_COLOR, linewidths=0)
        ax.plot(lc["time"][mask] - tm, lc["resid"][mask], lw=0.75, color=POINT_COLOR, alpha=0.55)
        ax.axvspan(-duration_days(row, family, period) / 2.0, duration_days(row, family, period) / 2.0, color=ACCENT, alpha=0.13)
        ax.axvline(0.0, color=ACCENT, lw=1.0)
        ax.axhline(0.0, color="#111111", lw=0.7, alpha=0.45)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"E{idx}")
        ax.text(
            0.99,
            0.92,
            f"t={tm:.5f} | depth={fmt(ev.get('depth'), 4)} | SNR={fmt(ev.get('depth_snr'), 4)}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
        )
    axes_arr[-1].set_xlabel("Time from event center [days]")
    fig.suptitle(title_for(row, period) + "\nLocal detrended cutouts for the 5-event family", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_individual_event_cutouts(
    out_dir: Path,
    row: pd.Series,
    lc: dict[str, Any],
    period: float,
    family: pd.DataFrame,
) -> dict[str, str]:
    ordered = family.sort_values("t_mid").reset_index(drop=True)
    dur = duration_days(row, family, period)
    window = max(0.18, min(0.7, (dur * 4.0) if np.isfinite(dur) else 0.35))
    ylim = robust_ylim(lc["resid"])
    paths: dict[str, str] = {}
    for idx, (_, ev) in enumerate(ordered.iterrows(), start=1):
        tm = as_float(ev.get("t_mid"))
        mask = np.abs(lc["time"] - tm) <= window
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.scatter(lc["time"][mask] - tm, lc["resid"][mask], s=15, alpha=0.7, color=POINT_COLOR, linewidths=0)
        ax.plot(lc["time"][mask] - tm, lc["resid"][mask], lw=0.85, color=POINT_COLOR, alpha=0.58)
        if np.isfinite(dur):
            ax.axvspan(-dur / 2.0, dur / 2.0, color=ACCENT, alpha=0.15, label="event duration window")
            ax.axvline(-dur / 2.0, color=ACCENT, lw=0.8, ls="--")
            ax.axvline(dur / 2.0, color=ACCENT, lw=0.8, ls="--")
        ax.axvline(0.0, color=ACCENT, lw=1.1, label="event center")
        ax.axhline(0.0, color="#111111", lw=0.7, alpha=0.45)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Time from event center [days]")
        ax.set_ylabel("Detrended normalized flux")
        ax.set_title(
            title_for(row, period)
            + f"\nLocal detrended cutout E{idx}: t={tm:.5f}, depth={fmt(ev.get('depth'), 4)}, SNR={fmt(ev.get('depth_snr'), 4)}"
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"local_detrended_cutout_event_{idx}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"local_detrended_cutout_event_{idx}"] = str(path)
    return paths


def diagnostic_text() -> str:
    return "\n".join(
        [
            "Enhanced manual-vetting diagnostic block",
            f"epic_id = {EPIC_ID}",
            "period_source = event_spacing_fallback",
            "event_family_count = 5",
            "candidate_period_count = 619",
            "best_period_days = 4.20484",
            "half_period_days = 2.10242",
            "double_period_days = 8.40968",
            "primary_depth_snr = 4.739",
            "cnn_score = 0.6889",
            "manual_review_status = uncertain_hold",
            "",
            "Why this remains uncertain_hold:",
            "- The period is not a saved validated period; it comes from event-spacing fallback.",
            "- The 5-event family is suggestive but sparse, and 619 candidate periods indicate alias ambiguity.",
            "- The primary folded depth SNR is only moderate at 4.739, below a clean-promotion comfort zone.",
            "- The CNN morphology score is supportive but not decisive; it is a morphology scorer only.",
            "- Half/double-period folds and odd/even behavior need manual comparison before promotion.",
            "- The correct action is to hold as uncertain until a stronger period solution or follow-up vetting resolves the alias risk.",
        ]
    )


def plot_diagnostic_text(path: Path, row: pd.Series, period: float) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.text(0.03, 0.97, diagnostic_text(), ha="left", va="top", transform=ax.transAxes, fontsize=11, linespacing=1.35)
    ax.set_axis_off()
    ax.set_title(title_for(row, period) + "\nDiagnostic decision note", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_row() -> pd.Series:
    queue = pd.read_csv(QUEUE_CSV)
    match = queue.loc[queue["epic_id"].eq(EPIC_ID)]
    if len(match) == 0:
        raise RuntimeError(f"{EPIC_ID} not found in {QUEUE_CSV}")
    return match.iloc[0]


def center_for_period(candidates: pd.DataFrame, events: pd.DataFrame, period: float) -> float:
    if len(candidates) and "period_days" in candidates.columns:
        periods = pd.to_numeric(candidates["period_days"], errors="coerce")
        idx = (periods - period).abs().idxmin()
        if np.isfinite(periods.loc[idx]) and abs(float(periods.loc[idx]) - period) < 1e-3:
            center = as_float(candidates.loc[idx].get("cluster_center_phase"))
            if np.isfinite(center):
                return center
    family_base = family_events(events, BEST_PERIOD_DAYS, 0.3320136155078343)
    if len(family_base):
        phases = np.mod(family_base["t_mid"].to_numpy(dtype=float) / period, 1.0)
        return float(np.nanmedian(phases))
    return 0.0


def main() -> None:
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    row = load_row()
    lc = load_light_curve(EPIC_ID)
    events = load_events(EPIC_ID)
    candidates = candidate_periods_from_events(events)
    best_center = center_for_period(candidates, events, BEST_PERIOD_DAYS)
    family = family_events(events, BEST_PERIOD_DAYS, best_center)
    centers = {period: center_for_period(candidates, events, period) for _, period in PERIODS}
    centers[BEST_PERIOD_DAYS] = best_center
    ylim = shared_phase_ylim(lc, [period for _, period in PERIODS], centers)

    paths: dict[str, str] = {}
    plot_marked_light_curve(
        ENHANCED_DIR / "raw_light_curve_event_centers.png",
        row,
        lc,
        BEST_PERIOD_DAYS,
        family,
        "raw_flux",
        "Flux",
        "Raw full light curve",
    )
    paths["raw_light_curve_event_centers"] = str(ENHANCED_DIR / "raw_light_curve_event_centers.png")
    plot_marked_light_curve(
        ENHANCED_DIR / "detrended_light_curve_event_centers.png",
        row,
        lc,
        BEST_PERIOD_DAYS,
        family,
        "resid",
        "Detrended normalized flux",
        "Detrended full light curve",
    )
    paths["detrended_light_curve_event_centers"] = str(ENHANCED_DIR / "detrended_light_curve_event_centers.png")

    for label, period in PERIODS:
        center = centers[period]
        this_family = family_events(events, period, center)
        plot_phase_panel(
            ENHANCED_DIR / f"phase_folded_{label}.png",
            row,
            lc,
            period,
            center,
            this_family,
            f"Phase-folded diagnostic at {label.replace('_', ' ')}",
            (-0.5, 0.5),
            ylim,
            100,
        )
        paths[f"phase_folded_{label}"] = str(ENHANCED_DIR / f"phase_folded_{label}.png")
        plot_phase_panel(
            ENHANCED_DIR / f"transit_zoom_{label}.png",
            row,
            lc,
            period,
            center,
            this_family,
            f"Transit zoom at {label.replace('_', ' ')}",
            (-0.18, 0.18),
            ylim,
            80,
        )
        paths[f"transit_zoom_{label}"] = str(ENHANCED_DIR / f"transit_zoom_{label}.png")

    plot_secondary_panel(
        ENHANCED_DIR / "secondary_eclipse_check_best_period.png",
        row,
        lc,
        BEST_PERIOD_DAYS,
        best_center,
        ylim,
    )
    paths["secondary_eclipse_check_best_period"] = str(ENHANCED_DIR / "secondary_eclipse_check_best_period.png")
    for label, period in [("best_period", BEST_PERIOD_DAYS), ("half_period", 2.10242)]:
        center = centers[period]
        this_family = family_events(events, period, center)
        plot_odd_even_panel(
            ENHANCED_DIR / f"odd_even_{label}.png",
            row,
            lc,
            period,
            center,
            this_family,
            ylim,
        )
        paths[f"odd_even_{label}"] = str(ENHANCED_DIR / f"odd_even_{label}.png")

    plot_event_metric(ENHANCED_DIR / "event_by_event_depth.png", row, family, BEST_PERIOD_DAYS, "depth", "depth")
    paths["event_by_event_depth"] = str(ENHANCED_DIR / "event_by_event_depth.png")
    plot_event_metric(ENHANCED_DIR / "event_by_event_snr.png", row, family, BEST_PERIOD_DAYS, "depth_snr", "depth SNR")
    paths["event_by_event_snr"] = str(ENHANCED_DIR / "event_by_event_snr.png")
    plot_event_cutouts(ENHANCED_DIR / "local_detrended_cutouts_5_events.png", row, lc, BEST_PERIOD_DAYS, family)
    paths["local_detrended_cutouts_5_events"] = str(ENHANCED_DIR / "local_detrended_cutouts_5_events.png")
    paths.update(plot_individual_event_cutouts(ENHANCED_DIR, row, lc, BEST_PERIOD_DAYS, family))
    plot_diagnostic_text(ENHANCED_DIR / "diagnostic_text_block.png", row, BEST_PERIOD_DAYS)
    paths["diagnostic_text_block"] = str(ENHANCED_DIR / "diagnostic_text_block.png")
    (ENHANCED_DIR / "diagnostic_text_block.txt").write_text(diagnostic_text() + "\n", encoding="utf-8")
    paths["diagnostic_text_block_txt"] = str(ENHANCED_DIR / "diagnostic_text_block.txt")

    manifest = {
        "epic_id": EPIC_ID,
        "best_period_days": BEST_PERIOD_DAYS,
        "half_period_days": 2.10242,
        "double_period_days": 8.40968,
        "period_source": "event_spacing_fallback",
        "event_family_count": int(len(family)),
        "candidate_period_count": int(len(candidates)),
        "primary_depth_snr": 4.739,
        "cnn_score": 0.6889,
        "manual_review_status": "uncertain_hold",
        "shared_phase_ylim": list(ylim),
        "centers": {label: centers[period] for label, period in PERIODS},
        "paths": paths,
    }
    (ENHANCED_DIR / "enhanced_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
