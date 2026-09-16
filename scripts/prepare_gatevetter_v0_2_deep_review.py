from __future__ import annotations

import json
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
import scripts.compare_epic_211340132_periods as period_debug
import scripts.refresh_gatevetter_unseen_all as refresh_all
import scripts.refresh_gatevetter_unseen_full_validation as refresh
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


TARGETS = [
    (1, "uncertain_hold_positive", "EPIC_211624954"),
    (1, "uncertain_hold_positive", "EPIC_211687388"),
    (1, "uncertain_hold_positive", "EPIC_211768304"),
    (1, "uncertain_hold_positive", "EPIC_211996306"),
    (1, "uncertain_hold_positive", "EPIC_211959909"),
    (2, "uncertain_hold_period_ambiguous", "EPIC_211912465"),
    (2, "uncertain_hold_period_ambiguous", "EPIC_211485867"),
    (2, "uncertain_hold_period_ambiguous", "EPIC_211387236"),
    (2, "uncertain_hold_period_ambiguous", "EPIC_211889082"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_211816343"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_212029934"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_211351798"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_211845034"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_211431812"),
    (3, "highest_scoring_unreviewed_hold", "EPIC_211696209"),
]

OUT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_deep_review"
SHORTLIST_CSV = ROOT / "gatevetter_v0_2_deep_review_shortlist.csv"
PERIOD_CSV = ROOT / "gatevetter_v0_2_deep_review_period_comparison.csv"
MANIFEST_CSV = ROOT / "gatevetter_v0_2_deep_review_manifest.csv"
SUMMARY_TXT = ROOT / "gatevetter_v0_2_deep_review_summary.txt"


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe(value: Any) -> Any:
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
    return value


def source_rows() -> pd.DataFrame:
    frames = []
    for batch in range(3, 11):
        path = ROOT / f"gatevetter_v0_2_batch_{batch}_predictions.csv"
        frame = pd.read_csv(path)
        frame["source_batch"] = batch
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["epic_id"] = combined["epic_id"].astype(str)
    requested = pd.DataFrame(TARGETS, columns=["priority", "review_category", "epic_id"])
    out = requested.merge(combined, on="epic_id", how="left", validate="one_to_one")
    missing = out.loc[out["source_batch"].isna(), "epic_id"].tolist()
    if missing:
        raise RuntimeError(f"Targets missing from batch 3-10 predictions: {missing}")
    return out


def robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = float(np.nanmedian(arr))
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def local_baseline_metrics(
    epic_id: str,
    lc: dict[str, Any],
    period: float,
    center: float,
    duration: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    time = np.asarray(lc["time"], dtype=float)
    flux = np.asarray(lc["resid"], dtype=float)
    epoch_min = int(np.floor(time.min() / period - center)) - 1
    epoch_max = int(np.ceil(time.max() / period - center)) + 1
    centers = period * (np.arange(epoch_min, epoch_max + 1, dtype=float) + center)
    margin = 3.5 * duration
    centers = centers[(centers >= time.min() + margin) & (centers <= time.max() - margin)]
    rows = []
    for idx, event_center in enumerate(centers):
        delta = time - event_center
        event = np.abs(delta) <= 0.5 * duration
        left = (delta >= -3.0 * duration) & (delta <= -1.25 * duration)
        right = (delta >= 1.25 * duration) & (delta <= 3.0 * duration)
        side = left | right
        if event.sum() < 3 or side.sum() < 8:
            continue
        coef = np.polyfit(delta[side], flux[side], 1)
        normalized = flux - np.polyval(coef, delta)
        scatter = robust_sigma(normalized[side])
        left_median = float(np.nanmedian(normalized[left])) if left.sum() >= 3 else np.nan
        right_median = float(np.nanmedian(normalized[right])) if right.sum() >= 3 else np.nan
        depth = -float(np.nanmedian(normalized[event]))
        rows.append({
            "epic_id": epic_id,
            "event_index": idx,
            "predicted_center_bkjd": event_center,
            "local_depth": depth,
            "local_depth_snr": depth / scatter if np.isfinite(scatter) and scatter > 0 else np.nan,
            "sideband_sigma": scatter,
            "baseline_slope_per_day": float(coef[0]),
            "left_right_baseline_offset": right_median - left_median,
            "both_sidebands_available": bool(left.sum() >= 3 and right.sum() >= 3),
        })
    events = pd.DataFrame(rows)
    if len(events) == 0:
        return {
            "local_baseline_stability": "insufficient_events",
            "local_baseline_event_count": 0,
            "median_abs_baseline_slope_per_day": np.nan,
            "median_abs_left_right_offset": np.nan,
            "both_sidebands_fraction": 0.0,
        }, events
    depth_scale = float(np.nanmedian(np.abs(events["local_depth"])))
    slope = float(events["baseline_slope_per_day"].abs().median())
    offset = float(events["left_right_baseline_offset"].abs().median())
    both = float(events["both_sidebands_available"].mean())
    slope_fraction = slope * duration / depth_scale if depth_scale > 0 else np.inf
    offset_fraction = offset / depth_scale if depth_scale > 0 else np.inf
    if both >= 0.8 and slope_fraction <= 0.25 and offset_fraction <= 0.5:
        label = "stable"
    elif both >= 0.6 and slope_fraction <= 0.6 and offset_fraction <= 1.0:
        label = "mixed"
    else:
        label = "unstable"
    return {
        "local_baseline_stability": label,
        "local_baseline_event_count": int(len(events)),
        "median_abs_baseline_slope_per_day": slope,
        "median_abs_left_right_offset": offset,
        "both_sidebands_fraction": both,
        "baseline_slope_depth_fraction": slope_fraction,
        "baseline_offset_depth_fraction": offset_fraction,
    }, events


def evaluate_period(
    epic_id: str,
    role: str,
    period: float,
    events: pd.DataFrame,
    lc: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    support, center = refresh.support_at(events, period)
    family = plot_pack.family_events(events, period, center).copy()
    if len(family) and "event_number" in family:
        family["family_epoch"] = family["event_number"]
    duration = refresh.as_float(pd.to_numeric(family.get("duration_days", pd.Series(dtype=float)), errors="coerce").median())
    if not np.isfinite(duration) or duration <= 0:
        duration = max(0.08, min(0.30, 0.03 * period))
    half_width = float(np.clip(1.6 * duration / period, 0.015, 0.08))
    phase0 = K2StageFFollowupValidation._phase_centered(lc["time"], period, center)
    primary = K2StageFFollowupValidation._folded_depth(
        phase=phase0, resid=lc["resid"], half_width_phase=half_width
    )
    phase05 = K2StageFFollowupValidation._phase_centered(lc["time"], period, (center + 0.5) % 1.0)
    secondary = K2StageFFollowupValidation._folded_depth(
        phase=phase05, resid=lc["resid"], half_width_phase=half_width
    )
    primary_depth = refresh.as_float(primary["depth"])
    secondary_depth = refresh.as_float(secondary["depth"])
    odd_even = K2StageFFollowupValidation._epoch_depth_stats(family)
    oot = K2StageFFollowupValidation._oot_variability(
        phase0=phase0, resid=lc["resid"], primary_half_width=half_width
    )
    oot_amp = refresh.as_float(oot["oot_variability_amp"])
    filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
    _, alias = K2StageFFollowupValidation._alias_stats(
        filtered,
        period=period,
        primary_support_count=max(support, len(family)),
    )
    coherence = period_debug.event_stack_coherence(family)
    baseline, baseline_events = local_baseline_metrics(epic_id, lc, period, center, duration)
    odd_ratio = refresh.as_float(odd_even["odd_even_depth_ratio"])
    secondary_ratio = secondary_depth / primary_depth if primary_depth > 0 else np.nan
    row = {
        "epic_id": epic_id,
        "period_role": role,
        "period_days": period,
        "cluster_center_phase": center,
        "event_support_count": int(support),
        "event_family_count": int(len(family)),
        "primary_depth": primary_depth,
        "primary_depth_snr": refresh.as_float(primary["snr"]),
        "duration_days": duration,
        "odd_depth_median": refresh.as_float(odd_even["odd_depth_median"]),
        "even_depth_median": refresh.as_float(odd_even["even_depth_median"]),
        "odd_even_depth_ratio": odd_ratio,
        "odd_even_assessment": "consistent" if np.isfinite(odd_ratio) and 0.8 <= odd_ratio <= 1.25 else "concerning_or_incomplete",
        "oot_to_depth": oot_amp / primary_depth if primary_depth > 0 else np.nan,
        "secondary_depth": secondary_depth,
        "secondary_depth_snr": refresh.as_float(secondary["snr"]),
        "secondary_to_primary_depth_ratio": secondary_ratio,
        "secondary_assessment": "concerning" if (np.isfinite(secondary_ratio) and secondary_ratio >= 0.5) or abs(refresh.as_float(secondary["snr"])) >= 5 else "no_strong_secondary",
        "alias_risk": clean(alias["alias_risk"]),
        **coherence,
        **baseline,
    }
    return row, family, baseline_events


def plot_folded(ax: plt.Axes, lc: dict[str, Any], row: dict[str, Any]) -> None:
    phase = plot_pack.phase_centered(lc["time"], row["period_days"], row["cluster_center_phase"])
    binned = plot_pack.phase_bin_median(phase, lc["resid"], bins=140)
    ax.scatter(phase, lc["resid"], s=2, alpha=0.12, color="#315b6d", linewidths=0)
    if len(binned):
        ax.plot(binned["phase"], binned["median"], color="#b42318", lw=1.3)
    ax.axvline(0, color="#111111", lw=0.7)
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(f"{row['period_role']}: {row['period_days']:.7g} d | SNR {row['primary_depth_snr']:.3g}")


def plot_stack(ax: plt.Axes, lc: dict[str, Any], row: dict[str, Any], family: pd.DataFrame) -> None:
    window = min(0.6, max(0.12, 2.0 * row["duration_days"]))
    for idx, (_, event) in enumerate(family.sort_values("t_mid").iterrows()):
        midpoint = refresh.as_float(event.get("t_mid"))
        mask = np.abs(lc["time"] - midpoint) <= window
        if np.any(mask):
            local = lc["resid"][mask] - np.nanmedian(lc["resid"][mask])
            ax.plot(lc["time"][mask] - midpoint, local + idx * 0.0012, lw=0.6)
    ax.axvline(0, color="#111111", lw=0.7)
    ax.set_xlim(-window, window)
    ax.set_title(f"Stack {row['event_stack_coherence']} ({row['event_stack_coherence_score']:.2f}), n={row['event_family_count']}")


def one_page_panel(
    path: Path,
    source: pd.Series,
    period_info: dict[str, Any],
    comparisons: list[dict[str, Any]],
    families: list[pd.DataFrame],
    lc: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(16, 11), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.05, 1.05, 0.82],
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.925,
        hspace=0.27,
        wspace=0.20,
    )
    for idx, row in enumerate(comparisons):
        plot_folded(fig.add_subplot(grid[0, idx]), lc, row)
        plot_stack(fig.add_subplot(grid[1, idx]), lc, row, families[idx])
    p = comparisons[1]
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    trusted = not bool(period_info["period_ambiguity_flag"]) and clean(period_info["period_comparison_status"]).startswith("trusted")
    lines = [
        f"Priority {int(source['priority'])} · {source['review_category']} · source batch {int(source['source_batch'])}",
        f"Trusted period decision: {'TRUSTED' if trusted else 'NOT TRUSTED'} · {period_info['period_comparison_status']} · {period_info['period_confirmation_reason']}",
        f"P={p['period_days']:.9g} d · odd/even={p['odd_even_depth_ratio']:.4g} ({p['odd_even_assessment']}) · OOT/depth={p['oot_to_depth']:.4g}",
        f"Secondary SNR={p['secondary_depth_snr']:.4g} · secondary/primary={p['secondary_to_primary_depth_ratio']:.4g} ({p['secondary_assessment']})",
        f"Event stack={p['event_stack_coherence']} ({p['event_stack_coherence_score']:.3f}) · local baseline={p['local_baseline_stability']} · median |slope|/day={p['median_abs_baseline_slope_per_day']:.4g} · median L/R offset={p['median_abs_left_right_offset']:.4g}",
        "Diagnostic refresh only: GateVetter v0.2 unchanged; CNN unchanged; no automatic promotion.",
    ]
    ax.text(0.01, 0.94, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle(
        f"{source['epic_id']} — GateVetter v0.2 deep review",
        fontsize=16,
        x=0.5,
        y=0.975,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    shortlist = source_rows()
    history = refresh_all.historical_period_index(set(shortlist["epic_id"]))
    missing_history = sorted(set(shortlist["epic_id"]) - set(history))
    if missing_history:
        raise RuntimeError(f"Missing historical period-search output: {missing_history}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    manifest_rows = []
    shortlist_rows = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    for idx, source in shortlist.iterrows():
        epic_id = str(source["epic_id"])
        print(f"[{idx + 1:02d}/{len(shortlist)}] {epic_id}", flush=True)
        epic_dir = OUT_ROOT / epic_id
        epic_dir.mkdir(parents=True, exist_ok=True)
        lc = refresh.load_cached_light_curve(epic_id)
        events = refresh.load_events(epic_id)
        period_info, candidates = refresh_all.choose_all_period(source, events, lc, history[epic_id])
        period = float(period_info["validation_period_days"])
        comparisons = []
        families = []
        baseline_frames = []
        for role, candidate_period in (("P/2", period / 2), ("P", period), ("2P", period * 2)):
            row, family, baselines = evaluate_period(epic_id, role, candidate_period, events, lc)
            comparisons.append(row)
            families.append(family)
            if len(baselines):
                baselines["period_role"] = role
                baseline_frames.append(baselines)
            comparison_rows.append(row)
        panel_path = epic_dir / "deep_review_panel.png"
        json_path = epic_dir / "validation_summary.json"
        period_path = epic_dir / "period_comparison.csv"
        baseline_path = epic_dir / "local_baseline_events.csv"
        pd.DataFrame(comparisons).to_csv(period_path, index=False)
        pd.concat(baseline_frames, ignore_index=True).to_csv(baseline_path, index=False) if baseline_frames else pd.DataFrame().to_csv(baseline_path, index=False)
        one_page_panel(panel_path, source, period_info, comparisons, families, lc)
        trusted = not bool(period_info["period_ambiguity_flag"]) and clean(period_info["period_comparison_status"]).startswith("trusted")
        payload = {
            "epic_id": epic_id,
            "priority": int(source["priority"]),
            "review_category": source["review_category"],
            "source_batch": int(source["source_batch"]),
            "source_gatevetter": {k: safe(v) for k, v in source.to_dict().items()},
            "trusted_period_decision": {
                "trusted": trusted,
                **{k: safe(v) for k, v in period_info.items()},
            },
            "period_comparison": [{k: safe(v) for k, v in row.items()} for row in comparisons],
            "artifacts": {
                "one_page_visual_panel": refresh.rel(panel_path),
                "period_comparison_csv": refresh.rel(period_path),
                "local_baseline_events_csv": refresh.rel(baseline_path),
            },
            "generated_at": generated_at,
            "safeguards": [
                "GateVetter v0.2 rules were not changed.",
                "CNN was not retrained or changed.",
                "This packet does not promote or relabel the target.",
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        p_row = comparisons[1]
        shortlist_rows.append({
            "priority": int(source["priority"]),
            "review_category": source["review_category"],
            "epic_id": epic_id,
            "source_batch": int(source["source_batch"]),
            "candidate_survivor_score": source.get("candidate_survivor_score"),
            "trusted_period": trusted,
            "period_decision": period_info["period_comparison_status"],
            "validation_period_days": period,
            "odd_even_depth_ratio": p_row["odd_even_depth_ratio"],
            "oot_to_depth": p_row["oot_to_depth"],
            "secondary_depth_snr": p_row["secondary_depth_snr"],
            "secondary_to_primary_depth_ratio": p_row["secondary_to_primary_depth_ratio"],
            "event_stack_coherence": p_row["event_stack_coherence"],
            "event_stack_coherence_score": p_row["event_stack_coherence_score"],
            "local_baseline_stability": p_row["local_baseline_stability"],
            "validation_summary_json": refresh.rel(json_path),
            "one_page_visual_panel": refresh.rel(panel_path),
        })
        manifest_rows.append({
            "priority": int(source["priority"]),
            "epic_id": epic_id,
            "validation_summary_json": refresh.rel(json_path),
            "one_page_visual_panel": refresh.rel(panel_path),
            "period_comparison_csv": refresh.rel(period_path),
            "local_baseline_events_csv": refresh.rel(baseline_path),
        })
    shortlist_out = pd.DataFrame(shortlist_rows)
    shortlist_out.to_csv(SHORTLIST_CSV, index=False)
    pd.DataFrame(comparison_rows).to_csv(PERIOD_CSV, index=False)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)
    lines = [
        "GateVetter v0.2 deep-review shortlist",
        f"generated_at={generated_at}",
        f"targets={len(shortlist_out)}",
        f"trusted_periods={int(shortlist_out['trusted_period'].sum())}",
        f"period_ambiguous={int((~shortlist_out['trusted_period']).sum())}",
        "new_batches=stopped",
        "gatevetter_v0_2=unchanged",
        "cnn=unchanged_not_retrained",
        f"shortlist={SHORTLIST_CSV.name}",
        f"period_comparison={PERIOD_CSV.name}",
        f"manifest={MANIFEST_CSV.name}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {SHORTLIST_CSV.name} ({len(shortlist_out)} rows)")
    print(f"Wrote {MANIFEST_CSV.name} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
