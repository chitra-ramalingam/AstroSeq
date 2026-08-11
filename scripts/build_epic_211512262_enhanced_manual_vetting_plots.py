from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_epic_211357782_enhanced_manual_vetting_plots import (
    OUT_DIR,
    candidate_periods_from_events,
    family_events,
    load_events,
    load_light_curve,
    phase_centered,
    plot_event_cutouts,
    plot_event_metric,
    plot_individual_event_cutouts,
    plot_marked_light_curve,
    plot_odd_even_panel,
    plot_phase_panel,
    plot_secondary_panel,
    robust_ylim,
    shared_phase_ylim,
)
from build_manual_vetting_next64_plot_pack import QUEUE_CSV, as_float


EPIC_ID = "EPIC_211512262"
BEST_PERIOD_DAYS = 9.11255
HALF_PERIOD_DAYS = 4.556275
DOUBLE_PERIOD_DAYS = 18.22510
PERIODS = [
    ("best_period", BEST_PERIOD_DAYS),
    ("half_period", HALF_PERIOD_DAYS),
    ("double_period", DOUBLE_PERIOD_DAYS),
]

EPIC_DIR = OUT_DIR / EPIC_ID
ENHANCED_DIR = EPIC_DIR / "enhanced_manual_vetting"
VALIDATION_JSON = EPIC_DIR / "validation_summary.json"


def load_row() -> pd.Series:
    queue = pd.read_csv(QUEUE_CSV)
    match = queue.loc[queue["epic_id"].eq(EPIC_ID)]
    if len(match) == 0:
        raise RuntimeError(f"{EPIC_ID} not found in {QUEUE_CSV}")
    return match.iloc[0]


def load_validation() -> dict[str, Any]:
    if not VALIDATION_JSON.exists():
        return {}
    payload = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
    return payload.get("validation", {})


def center_for_period(candidates: pd.DataFrame, events: pd.DataFrame, period: float, best_center: float) -> float:
    if len(candidates) and "period_days" in candidates.columns:
        periods = pd.to_numeric(candidates["period_days"], errors="coerce")
        idx = (periods - period).abs().idxmin()
        if np.isfinite(periods.loc[idx]) and abs(float(periods.loc[idx]) - period) < 1e-3:
            center = as_float(candidates.loc[idx].get("cluster_center_phase"))
            if np.isfinite(center):
                return center

    best_family = family_events(events, BEST_PERIOD_DAYS, best_center)
    if len(best_family):
        phases = np.mod(best_family["t_mid"].to_numpy(dtype=float) / period, 1.0)
        return float(np.nanmedian(phases))
    return 0.0


def diagnostic_text(validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Enhanced plot-generation notes",
            f"epic_id = {EPIC_ID}",
            "manual_vetted = false",
            "manual_label = not set",
            "decision_authority = unchanged",
            "",
            f"best_period_days = {BEST_PERIOD_DAYS:.6f}",
            f"half_period_days = {HALF_PERIOD_DAYS:.6f}",
            f"double_period_days = {DOUBLE_PERIOD_DAYS:.6f}",
            f"period_source = {validation.get('period_source', 'NA')}",
            f"event_family_count = {validation.get('event_family_count', 'NA')}",
            f"candidate_period_count = {validation.get('candidate_period_count', 'NA')}",
            f"primary_depth_snr = {validation.get('primary_depth_snr', 'NA')}",
            f"odd_even_depth_ratio = {validation.get('odd_even_depth_ratio', 'NA')}",
            f"alias_best_period_days = {validation.get('alias_best_period_days', 'NA')}",
            f"alias_best_support_ratio = {validation.get('alias_best_support_ratio', 'NA')}",
            f"cnn_score = {validation.get('cnn_score', 'NA')}",
            "",
            "Why these plots were regenerated:",
            "- The original plots are review support only and the target has not been manually vetted.",
            "- The cached validation has low folded primary SNR despite a high CNN morphology score.",
            "- The period is event-spacing fallback only, so best/half/double folds are shown side by side.",
            "- Binned points include error bars and bin counts to make the average folded morphology easier to judge.",
        ]
    )


def write_diagnostic_plot(path: Path, text: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.text(0.03, 0.97, text, ha="left", va="top", transform=ax.transAxes, fontsize=11, linespacing=1.35)
    ax.set_axis_off()
    ax.set_title(f"{EPIC_ID} enhanced review-support notes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    row = load_row()
    validation = load_validation()
    lc = load_light_curve(EPIC_ID)
    events = load_events(EPIC_ID)
    candidates = candidate_periods_from_events(events)

    best_center = as_float(validation.get("cluster_center_phase"))
    if not np.isfinite(best_center):
        best_center = center_for_period(candidates, events, BEST_PERIOD_DAYS, 0.0)
    centers = {period: center_for_period(candidates, events, period, best_center) for _, period in PERIODS}
    centers[BEST_PERIOD_DAYS] = float(best_center)
    family = family_events(events, BEST_PERIOD_DAYS, float(best_center))
    ylim = shared_phase_ylim(lc, [period for _, period in PERIODS], centers)
    raw_ylim = robust_ylim(lc["resid"])

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
        this_family = family_events(events, period, centers[period])
        plot_phase_panel(
            ENHANCED_DIR / f"phase_folded_{label}.png",
            row,
            lc,
            period,
            centers[period],
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
            centers[period],
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
        float(best_center),
        ylim,
    )
    paths["secondary_eclipse_check_best_period"] = str(ENHANCED_DIR / "secondary_eclipse_check_best_period.png")

    for label, period in [("best_period", BEST_PERIOD_DAYS), ("half_period", HALF_PERIOD_DAYS)]:
        this_family = family_events(events, period, centers[period])
        plot_odd_even_panel(
            ENHANCED_DIR / f"odd_even_{label}.png",
            row,
            lc,
            period,
            centers[period],
            this_family,
            ylim,
        )
        paths[f"odd_even_{label}"] = str(ENHANCED_DIR / f"odd_even_{label}.png")

    plot_event_metric(ENHANCED_DIR / "event_by_event_depth.png", row, family, BEST_PERIOD_DAYS, "depth", "depth")
    paths["event_by_event_depth"] = str(ENHANCED_DIR / "event_by_event_depth.png")
    plot_event_metric(ENHANCED_DIR / "event_by_event_snr.png", row, family, BEST_PERIOD_DAYS, "depth_snr", "depth SNR")
    paths["event_by_event_snr"] = str(ENHANCED_DIR / "event_by_event_snr.png")
    plot_event_cutouts(ENHANCED_DIR / "local_detrended_cutouts_events.png", row, lc, BEST_PERIOD_DAYS, family)
    paths["local_detrended_cutouts_events"] = str(ENHANCED_DIR / "local_detrended_cutouts_events.png")
    paths.update(plot_individual_event_cutouts(ENHANCED_DIR, row, lc, BEST_PERIOD_DAYS, family))

    notes = diagnostic_text(validation)
    (ENHANCED_DIR / "review_support_notes.txt").write_text(notes + "\n", encoding="utf-8")
    write_diagnostic_plot(ENHANCED_DIR / "review_support_notes.png", notes)
    paths["review_support_notes_txt"] = str(ENHANCED_DIR / "review_support_notes.txt")
    paths["review_support_notes"] = str(ENHANCED_DIR / "review_support_notes.png")

    manifest = {
        "epic_id": EPIC_ID,
        "manual_vetted": False,
        "manual_label": None,
        "decision_authority": "unchanged",
        "best_period_days": BEST_PERIOD_DAYS,
        "half_period_days": HALF_PERIOD_DAYS,
        "double_period_days": DOUBLE_PERIOD_DAYS,
        "period_source": validation.get("period_source"),
        "event_family_count": int(len(family)),
        "candidate_period_count": int(len(candidates)),
        "primary_depth_snr": validation.get("primary_depth_snr"),
        "cnn_score": validation.get("cnn_score"),
        "shared_phase_ylim": list(ylim),
        "raw_resid_ylim": list(raw_ylim),
        "centers": {label: centers[period] for label, period in PERIODS},
        "paths": paths,
    }
    (ENHANCED_DIR / "enhanced_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
