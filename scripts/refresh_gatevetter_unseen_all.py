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
import scripts.compare_epic_211340132_periods as period_debug
import scripts.refresh_gatevetter_unseen_full_validation as refresh


PACKETS = [
    ("stage_g_review", ROOT / "unseen_stage_g_review_manual_packet.csv"),
    ("top_holds", ROOT / "unseen_top_holds_manual_packet.csv"),
    ("reject_sanity_sample", ROOT / "unseen_reject_sanity_sample.csv"),
]
PREDICTIONS = ROOT / "gatevetter_v0_1_unseen_predictions.csv"
METRICS_CSV = ROOT / "unseen_full_validation_metrics_refreshed_all.csv"
COMPARISON_CSV = ROOT / "unseen_period_comparison_summary_all.csv"
AMBIGUOUS_CSV = ROOT / "unseen_period_ambiguous_queue.csv"
MANIFEST_CSV = ROOT / "unseen_manual_vetting_plot_manifest_refreshed_all.csv"
SUMMARY_TXT = ROOT / "unseen_refreshed_all_summary.txt"
PLOT_ROOT = (
    ROOT
    / "plots"
    / "k2_batch"
    / "gatevetter_v0_1_unseen_manual_review_refreshed_all"
)
HIGH_CANDIDATE_PERIOD_COUNT = 20

OUTPUT_PACKETS = {
    "stage_g_review": ROOT / "unseen_stage_g_review_manual_packet_refreshed_all.csv",
    "top_holds": ROOT / "unseen_top_holds_manual_packet_refreshed_all.csv",
    "reject_sanity_sample": ROOT / "unseen_reject_sanity_sample_refreshed_all.csv",
}


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean(value).lower() in {"true", "1", "yes", "y"}


def epic_id_from_row(row: pd.Series) -> str:
    for column in ("epic_id", "epic", "query"):
        raw = clean(row.get(column))
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return f"EPIC_{digits}"
    return ""


def historical_period_index(epics: set[str]) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in (ROOT / "plots" / "k2_batch").rglob("period_shortlist_best.csv"):
        try:
            data = pd.read_csv(path)
        except Exception:
            continue
        period_col = "P" if "P" in data.columns else (
            "best_period_days" if "best_period_days" in data.columns else ""
        )
        if not period_col:
            continue
        for _, row in data.iterrows():
            epic_id = epic_id_from_row(row)
            period = refresh.as_float(row.get(period_col))
            if epic_id not in epics or not np.isfinite(period):
                continue
            reason = clean(row.get("reason"))
            rows.append(
                {
                    "epic_id": epic_id,
                    "period_days": period,
                    "historical_reason": reason,
                    "historical_validated": reason == "validated",
                    "historical_center_phase": refresh.as_float(
                        row.get("cluster_center_phase")
                    ),
                    "historical_source_file": refresh.rel(path),
                }
            )
    history = pd.DataFrame(rows)
    if len(history) == 0:
        return {}
    history["period_round"] = history["period_days"].round(5)
    history = history.sort_values(
        ["historical_validated", "historical_source_file"],
        ascending=[False, False],
    ).drop_duplicates(["epic_id", "period_round"], keep="first")
    return {
        epic_id: group.drop(columns=["period_round"]).reset_index(drop=True)
        for epic_id, group in history.groupby("epic_id")
    }


def add_historical_candidates(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    events: pd.DataFrame,
    lc: dict[str, Any],
) -> pd.DataFrame:
    rows = candidates.to_dict(orient="records")
    for historical in history.to_dict(orient="records"):
        period = float(historical["period_days"])
        existing = next(
            (
                row
                for row in rows
                if abs(float(row["period_days"]) / period - 1.0) < 0.002
            ),
            None,
        )
        if existing is not None:
            existing["historical_match"] = True
            existing["historical_validated"] = bool(
                historical["historical_validated"]
            )
            existing["historical_reason"] = historical["historical_reason"]
            existing["historical_source_file"] = historical[
                "historical_source_file"
            ]
            continue
        support, center = refresh.support_at(events, period)
        rows.append(
            {
                "period_days": period,
                "candidate_origin": "historical_period_search",
                "event_support_count": support,
                "cluster_center_phase": center,
                **refresh.bls_at(lc, period),
                "historical_match": True,
                "historical_validated": bool(
                    historical["historical_validated"]
                ),
                "historical_reason": historical["historical_reason"],
                "historical_source_file": historical[
                    "historical_source_file"
                ],
            }
        )
    out = pd.DataFrame(rows)
    for column, default in [
        ("historical_match", False),
        ("historical_validated", False),
        ("historical_reason", ""),
        ("historical_source_file", ""),
    ]:
        if column not in out:
            out[column] = default
        out[column] = out[column].fillna(default)
    max_support = max(1.0, float(out["event_support_count"].max()))
    bls = pd.to_numeric(out["bls_power"], errors="coerce")
    finite = bls[np.isfinite(bls)]
    if len(finite) and float(finite.max()) > float(finite.min()):
        out["bls_score"] = (
            (bls - float(finite.min())) / (float(finite.max()) - float(finite.min()))
        ).fillna(0.0)
    else:
        out["bls_score"] = 0.0
    out["event_score"] = out["event_support_count"] / max_support
    out["combined_score"] = 0.58 * out["event_score"] + 0.42 * out["bls_score"]
    out.loc[out["historical_validated"].map(as_bool), "combined_score"] += 0.12
    return out.sort_values(
        ["combined_score", "event_support_count", "bls_power"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def exact_match(period: float, candidate: float) -> bool:
    return abs(period / candidate - 1.0) < 0.002


def choose_all_period(
    packet_row: pd.Series,
    events: pd.DataFrame,
    lc: dict[str, Any],
    history: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_info, base_candidates = refresh.period_confirmation(packet_row, events, lc)
    candidates = add_historical_candidates(
        base_candidates, history, events, lc
    )
    saved = refresh.as_float(packet_row.get("best_period_days"))
    saved_valid = bool(np.isfinite(saved) and 0.5 <= saved <= 40.0)

    validated_history = history.loc[
        history["historical_validated"].map(as_bool)
    ].copy()
    if saved_valid:
        saved_rows = candidates.loc[
            candidates["period_days"].map(lambda p: exact_match(float(p), saved))
        ]
        chosen = saved_rows.iloc[0] if len(saved_rows) else candidates.iloc[0]
    elif len(validated_history):
        historical_periods = validated_history["period_days"].to_numpy(dtype=float)
        historical_rows = candidates.loc[
            candidates["period_days"].map(
                lambda p: any(exact_match(float(p), h) for h in historical_periods)
            )
        ]
        chosen = historical_rows.iloc[0] if len(historical_rows) else candidates.iloc[0]
    else:
        chosen = candidates.iloc[0]
    chosen_period = float(chosen["period_days"])

    top_event = clean(base_info.get("top_event_periods")).split("|")[0]
    top_bls = clean(base_info.get("top_bls_periods")).split("|")[0]
    event_period = refresh.as_float(top_event)
    bls_period = refresh.as_float(top_bls)
    event_bls_disagree = bool(
        np.isfinite(event_period)
        and np.isfinite(bls_period)
        and not refresh.period_equivalent(event_period, bls_period)
    )
    history_conflicts = validated_history.loc[
        ~validated_history["period_days"].map(
            lambda p: refresh.period_equivalent(float(p), chosen_period)
        )
    ]
    historical_disagreement = len(history_conflicts) > 0
    alternatives = candidates.loc[
        ~candidates["period_days"].map(
            lambda p: exact_match(float(p), chosen_period)
        )
    ]
    second = alternatives.iloc[0] if len(alternatives) else None
    near_tied_nonharmonic = bool(
        second is not None
        and float(second["combined_score"])
        >= 0.92 * float(chosen["combined_score"])
        and not refresh.period_equivalent(
            float(second["period_days"]), chosen_period
        )
    )
    harmonic = candidates.loc[
        candidates["period_days"].map(
            lambda p: refresh.period_equivalent(float(p), chosen_period)
            and not exact_match(float(p), chosen_period)
        )
    ]
    harmonic_competitor = bool(
        len(harmonic)
        and float(harmonic["combined_score"].max())
        >= 0.90 * float(chosen["combined_score"])
    )
    weak_support = int(chosen["event_support_count"]) < 2
    ambiguous = bool(
        event_bls_disagree
        or historical_disagreement
        or near_tied_nonharmonic
        or harmonic_competitor
        or weak_support
    )

    chosen_is_saved = saved_valid and exact_match(chosen_period, saved)
    chosen_is_history = bool(
        len(history)
        and history["period_days"].map(
            lambda p: exact_match(float(p), chosen_period)
        ).any()
    )
    if ambiguous:
        source = "period_ambiguous"
        status = "no_automatic_period_selection"
    elif chosen_is_saved:
        source = "saved_best_period"
        status = "trusted_saved_period"
    elif chosen_is_history:
        source = "historical_period_search"
        status = "trusted_historical_period"
    elif clean(chosen.get("candidate_origin")) == "event_spacing_fallback":
        source = "event_spacing_fallback_only"
        status = "provisional_not_trusted"
    else:
        source = "refreshed_period_search"
        status = "trusted_refreshed_period"

    reasons = []
    if event_bls_disagree:
        reasons.append("bls_event_search_disagreement")
    if historical_disagreement:
        reasons.append("validated_historical_period_disagreement")
    if near_tied_nonharmonic:
        reasons.append("near_tied_nonharmonic_candidate")
    if harmonic_competitor:
        reasons.append("half_or_double_period_competitor")
    if weak_support:
        reasons.append("event_support_below_two")
    if not reasons:
        reasons.append("period_evidence_consistent")

    historical_periods = "|".join(
        f"{period:.9g}" for period in history["period_days"]
    )
    historical_files = "|".join(
        sorted(set(history["historical_source_file"].astype(str)))
    )
    info = {
        "saved_historical_period_days": saved,
        "historical_period_search_periods": historical_periods,
        "historical_period_search_files": historical_files,
        "event_spacing_fallback_period_days": refresh.as_float(
            base_info.get("event_spacing_fallback_period_days")
        ),
        "half_period_days": chosen_period / 2.0,
        "double_period_days": chosen_period * 2.0,
        "validation_period_days": chosen_period,
        "validation_period_source": source,
        "period_ambiguity_flag": ambiguous or status == "provisional_not_trusted",
        "period_comparison_status": status,
        "period_confirmation_reason": "|".join(reasons),
        "validation_period_event_support": int(
            chosen["event_support_count"]
        ),
        "validation_period_bls_power": float(chosen["bls_power"]),
        "validation_cluster_center_phase": float(
            chosen["cluster_center_phase"]
        ),
        "validation_bls_duration_days": float(
            chosen["bls_duration_days"]
        ),
        "candidate_period_count": int(len(candidates)),
        "top_event_periods": base_info.get("top_event_periods", ""),
        "top_bls_periods": base_info.get("top_bls_periods", ""),
        "top_comparison_periods": "|".join(
            f"{period:.9g}"
            for period in [
                chosen_period,
                *candidates.loc[
                    ~candidates["period_days"].map(
                        lambda p: exact_match(float(p), chosen_period)
                    ),
                    "period_days",
                ].head(7),
            ]
        ),
        "selected_candidate_origin": clean(chosen.get("candidate_origin")),
        "selected_combined_score": float(chosen["combined_score"]),
        "second_candidate_period_days": (
            float(second["period_days"]) if second is not None else np.nan
        ),
        "second_candidate_combined_score": (
            float(second["combined_score"]) if second is not None else np.nan
        ),
    }
    return info, candidates


def comparison_trigger_reasons(
    packet_row: pd.Series,
    period_info: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    reasons = []
    if as_bool(packet_row.get("fallback_period_flag")):
        reasons.append("input_period_source=event_spacing_fallback")
    if clean(period_info.get("validation_period_source")) == "period_ambiguous":
        reasons.append("validation_period_source=period_ambiguous")
    if not np.isfinite(refresh.as_float(metrics.get("odd_even_depth_ratio"))):
        reasons.append("odd_even_depth_ratio_null")
    if not np.isfinite(refresh.as_float(metrics.get("oot_to_depth"))):
        reasons.append("oot_to_depth_null")
    if int(period_info["candidate_period_count"]) >= HIGH_CANDIDATE_PERIOD_COUNT:
        reasons.append(
            f"candidate_period_count>={HIGH_CANDIDATE_PERIOD_COUNT}"
        )
    prediction = clean(packet_row.get("gatevetter_prediction"))
    action = clean(packet_row.get("stage_g_action"))
    if prediction == "candidate_like_positive":
        reasons.append("gatevetter_candidate_like_positive")
    if "caveated_candidate_stage_g_review" in action:
        reasons.append("gatevetter_caveated_candidate_stage_g_review")
    return reasons


def plot_comparison_panel(
    path: Path,
    epic_id: str,
    candidates: pd.DataFrame,
    events: pd.DataFrame,
    lc: dict[str, Any],
    period_info: dict[str, Any],
) -> None:
    selected = float(period_info["validation_period_days"])
    selected_rows = candidates.loc[
        candidates["period_days"].map(
            lambda p: exact_match(float(p), selected)
        )
    ]
    other_rows = candidates.loc[
        ~candidates["period_days"].map(
            lambda p: exact_match(float(p), selected)
        )
    ].head(4)
    top = pd.concat([selected_rows.head(1), other_rows], ignore_index=True)
    fig, axes = plt.subplots(len(top), 2, figsize=(14, 3.6 * len(top)))
    if len(top) == 1:
        axes = np.asarray([axes])
    for idx, (_, candidate) in enumerate(top.iterrows()):
        period = float(candidate["period_days"])
        center = float(candidate["cluster_center_phase"])
        phase = plot_pack.phase_centered(lc["time"], period, center)
        binned = plot_pack.phase_bin_median(phase, lc["resid"], bins=140)
        ax = axes[idx, 0]
        ax.scatter(
            phase,
            lc["resid"],
            s=3,
            alpha=0.14,
            color="#264653",
            linewidths=0,
        )
        if len(binned):
            ax.plot(
                binned["phase"],
                binned["median"],
                color="#c1121f",
                lw=1.3,
            )
        ax.axvline(0.0, color="#111111", lw=0.8)
        ax.set_xlim(-0.5, 0.5)
        marker = "SELECTED" if exact_match(period, selected) else "candidate"
        ax.set_title(
            f"{marker}: P={period:.7g} d | "
            f"{clean(candidate.get('candidate_origin'))}\n"
            f"support={int(candidate['event_support_count'])}; "
            f"BLS={float(candidate['bls_power']):.4g}; "
            f"score={float(candidate['combined_score']):.3f}"
        )
        family = plot_pack.family_events(events, period, center)
        coherence = period_debug.event_stack_coherence(family)
        stack = axes[idx, 1]
        duration = refresh.as_float(
            pd.to_numeric(
                family.get("duration_days", pd.Series(dtype=float)),
                errors="coerce",
            ).median()
        )
        window = min(
            0.6,
            max(0.12, 2.0 * duration if np.isfinite(duration) else 0.2),
        )
        for event_idx, (_, event) in enumerate(
            family.sort_values("t_mid").iterrows()
        ):
            midpoint = refresh.as_float(event.get("t_mid"))
            mask = np.abs(lc["time"] - midpoint) <= window
            if not np.any(mask):
                continue
            local = lc["resid"][mask]
            local = local - float(np.nanmedian(local))
            stack.plot(
                lc["time"][mask] - midpoint,
                local + event_idx * 0.0012,
                lw=0.65,
            )
        stack.axvline(0.0, color="#111111", lw=0.8)
        stack.set_xlim(-window, window)
        stack.set_title(
            f"event stack: {coherence['event_stack_coherence']} "
            f"({coherence['event_stack_coherence_score']:.3f}); "
            f"family={len(family)}"
        )
    axes[-1, 0].set_xlabel("Folded phase")
    axes[-1, 1].set_xlabel("Days from event midpoint")
    fig.suptitle(
        f"{epic_id} period comparison | "
        f"{period_info['period_comparison_status']}\n"
        f"{period_info['period_confirmation_reason']}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def triage_fields(
    packet_row: pd.Series,
    period_info: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    status = clean(period_info.get("period_comparison_status"))
    missing = clean(metrics.get("missing_reason"))
    if status in {
        "provisional_not_trusted",
        "no_automatic_period_selection",
    }:
        route = "period_ambiguous_hold"
        classification = "period_ambiguous"
        reason = (
            f"period_comparison_status={status}; "
            f"{period_info['period_confirmation_reason']}"
        )
    elif missing:
        route = "diagnostics_incomplete_hold"
        classification = "diagnostics_incomplete"
        reason = f"missing diagnostics are not passed evidence: {missing}"
    else:
        route = "manual_review_with_refreshed_metrics"
        classification = "manual_review_with_refreshed_metrics"
        reason = "trusted period selected; retain packet for manual review"
    return {
        "pre_refresh_stage_g_action": packet_row.get("stage_g_action"),
        "pre_refresh_gatevetter_prediction": packet_row.get(
            "gatevetter_prediction"
        ),
        "refreshed_triage_classification": classification,
        "refreshed_triage_route": route,
        "refreshed_triage_reason": reason,
        "refreshed_candidate_like_allowed": False,
        "pending_manual_final_label": True,
        "refreshed_triage_authority": (
            "refreshed_validation_packet_triage_not_final_manual_label"
        ),
    }


def refreshed_packet(
    packet: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    manifest: pd.DataFrame,
    packet_source: str,
) -> pd.DataFrame:
    diagnostic_columns = [
        "validation_period_days",
        "validation_period_source",
        "period_ambiguity_flag",
        "period_comparison_status",
        "primary_depth",
        "primary_depth_snr",
        "odd_depth_median",
        "even_depth_median",
        "odd_even_depth_ratio",
        "secondary_depth_phase_05",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "oot_to_depth",
        "alias_risk",
        "candidate_period_count",
        "event_family_count",
        "duration_fraction_of_period",
        "missing_reason",
        "refreshed_triage_classification",
        "refreshed_triage_route",
        "refreshed_triage_reason",
        "refreshed_candidate_like_allowed",
        "pending_manual_final_label",
        "refreshed_triage_authority",
        "pre_refresh_stage_g_action",
        "pre_refresh_gatevetter_prediction",
    ]
    replacement = metrics[["epic_id", *diagnostic_columns]]
    comparison_columns = [
        "epic_id",
        "period_comparison_triggered",
        "period_comparison_trigger_reasons",
        "period_confirmation_reason",
        "historical_period_search_periods",
        "top_event_periods",
        "top_bls_periods",
        "top_comparison_periods",
    ]
    plot_columns = [
        col for col in manifest.columns if col not in {"packet_source"}
    ]
    out = packet.drop(
        columns=[
            col
            for col in diagnostic_columns
            + comparison_columns[1:]
            + plot_columns[1:]
            if col in packet.columns
        ],
        errors="ignore",
    )
    out = out.merge(replacement, on="epic_id", how="left", validate="one_to_one")
    out = out.merge(
        comparisons[comparison_columns],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        manifest.loc[
            manifest["packet_source"].eq(packet_source), plot_columns
        ],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    out["stage_g_action"] = out["refreshed_triage_route"]
    return out


def main() -> None:
    packet_frames: dict[str, pd.DataFrame] = {}
    combined_rows = []
    for packet_source, path in PACKETS:
        packet = pd.read_csv(path)
        packet["epic_id"] = packet["epic_id"].astype(str)
        packet_frames[packet_source] = packet
        combined_rows.append(packet.assign(packet_source=packet_source))
    combined = pd.concat(combined_rows, ignore_index=True)
    if len(combined) != 59 or combined["epic_id"].nunique() != 59:
        raise RuntimeError("Expected 59 unique unseen packet EPICs")

    predictions = pd.read_csv(PREDICTIONS)
    predictions["epic_id"] = predictions["epic_id"].astype(str)
    prediction_columns = [
        col
        for col in predictions.columns
        if col not in combined.columns or col == "epic_id"
    ]
    combined = combined.merge(
        predictions[prediction_columns],
        on="epic_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    history = historical_period_index(set(combined["epic_id"]))
    missing_history = sorted(set(combined["epic_id"]).difference(history))
    if missing_history:
        raise RuntimeError(
            f"Missing historical period-search output: {missing_history}"
        )

    refresh.PLOT_ROOT = PLOT_ROOT
    metric_rows = []
    comparison_rows = []
    manifest_rows = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    for idx, packet_row in combined.iterrows():
        epic_id = str(packet_row["epic_id"])
        packet_source = str(packet_row["packet_source"])
        print(
            f"[{idx + 1:02d}/{len(combined)}] {epic_id} {packet_source}",
            flush=True,
        )
        lc = refresh.load_cached_light_curve(epic_id)
        events = refresh.load_events(epic_id)
        period_info, candidates = choose_all_period(
            packet_row, events, lc, history[epic_id]
        )
        metrics, family = refresh.recompute_metrics(
            epic_id, period_info, events, lc
        )
        metrics["period_comparison_status"] = period_info[
            "period_comparison_status"
        ]
        triage = triage_fields(packet_row, period_info, metrics)
        metrics.update(triage)
        triggers = comparison_trigger_reasons(
            packet_row, period_info, metrics
        )

        manifest = refresh.generate_plots(
            packet_source,
            packet_row,
            metrics,
            period_info,
            candidates,
            family,
            lc,
            refresh.packet_rank(packet_row, idx + 1),
        )
        comparison_path = (
            PLOT_ROOT
            / packet_source
            / epic_id
            / "period_comparison_panel.png"
        )
        if triggers:
            plot_comparison_panel(
                comparison_path,
                epic_id,
                candidates,
                events,
                lc,
                period_info,
            )
            manifest["plot_period_comparison_path"] = refresh.rel(
                comparison_path
            )
        else:
            manifest["plot_period_comparison_path"] = ""
        manifest_rows.append(manifest)
        metric_rows.append(
            {"packet_source": packet_source, **metrics}
        )
        comparison_rows.append(
            {
                "epic_id": epic_id,
                "packet_source": packet_source,
                **period_info,
                "input_period_source": (
                    "event_spacing_fallback"
                    if as_bool(packet_row.get("fallback_period_flag"))
                    else "saved_best_period"
                ),
                "gatevetter_prediction": packet_row.get(
                    "gatevetter_prediction"
                ),
                "period_comparison_triggered": bool(triggers),
                "period_comparison_trigger_reasons": "|".join(triggers),
                "plot_period_comparison_path": manifest[
                    "plot_period_comparison_path"
                ],
                "refreshed_triage_route": triage[
                    "refreshed_triage_route"
                ],
                "generated_at": generated_at,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    comparisons_df = pd.DataFrame(comparison_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    comparisons_df.to_csv(COMPARISON_CSV, index=False)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    ambiguous = metrics_df.loc[
        metrics_df["period_ambiguity_flag"].map(as_bool)
        | metrics_df["period_comparison_status"].isin(
            ["provisional_not_trusted", "no_automatic_period_selection"]
        )
    ].copy()
    ambiguous.to_csv(AMBIGUOUS_CSV, index=False)

    for packet_source, packet in packet_frames.items():
        output = refreshed_packet(
            packet,
            metrics_df,
            comparisons_df,
            manifest_df,
            packet_source,
        )
        output.to_csv(OUTPUT_PACKETS[packet_source], index=False)

    lines = [
        "GateVetter v0.1 unseen all-packet refreshed validation summary",
        f"generated_at={generated_at}",
        "scope=59 unique EPICs across the three unseen manual packets",
        f"prediction_rows_available={len(predictions)}",
        f"packet_epics_refreshed={len(metrics_df)}",
        f"historical_period_search_coverage={len(history)}/59",
        f"period_comparison_triggered={int(comparisons_df['period_comparison_triggered'].sum())}",
        f"period_ambiguous_queue={len(ambiguous)}",
        f"high_candidate_period_count_threshold={HIGH_CANDIDATE_PERIOD_COUNT}",
        "",
        "Period comparison status",
    ]
    lines.extend(
        f"{key}={value}"
        for key, value in metrics_df[
            "period_comparison_status"
        ].value_counts().items()
    )
    lines.extend(["", "Validation period source"])
    lines.extend(
        f"{key}={value}"
        for key, value in metrics_df[
            "validation_period_source"
        ].value_counts().items()
    )
    lines.extend(["", "Packet triage routes"])
    lines.extend(
        f"{key}={value}"
        for key, value in metrics_df[
            "refreshed_triage_route"
        ].value_counts().items()
    )
    lines.extend(
        [
            "",
            "Safeguards",
            "- GateVetter rules and thresholds were not changed.",
            "- Manual labels were not updated.",
            "- Missing diagnostics were not treated as passed evidence.",
            "- no_automatic_period_selection and provisional_not_trusted route to period_ambiguous_hold.",
            "- Odd/even and OOT metrics are withheld when the period is ambiguous.",
            "",
            "Outputs",
            f"- {METRICS_CSV.name}",
            f"- {COMPARISON_CSV.name}",
            f"- {AMBIGUOUS_CSV.name}",
            f"- {MANIFEST_CSV.name}",
        ]
    )
    lines.extend(f"- {path.name}" for path in OUTPUT_PACKETS.values())
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"metrics_rows={len(metrics_df)}")
    print(f"ambiguous_rows={len(ambiguous)}")


if __name__ == "__main__":
    main()
