from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compare_epic_211340132_periods as period_debug
import scripts.refresh_gatevetter_unseen_full_validation as refresh


COMPARISON_CSV = ROOT / "unseen_period_comparison_summary_all.csv"
PLOT_ROOT = (
    ROOT
    / "plots"
    / "k2_batch"
    / "gatevetter_v0_1_unseen_manual_review_refreshed_all"
)
OUTPUT_CSV = ROOT / "unseen_period_comparison_metrics_all_untrusted.csv"
SUMMARY_TXT = ROOT / "unseen_period_comparison_metrics_all_untrusted_summary.txt"

OUTPUT_COLUMNS = [
    "epic_id",
    "packet_source",
    "tested_period_days",
    "period_role",
    "is_selected_period",
    "is_trusted_period",
    "period_ambiguity_flag",
    "metric_trust_level",
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
    "event_stack_score",
    "event_family_count",
    "coverage",
    "missing_reason",
]

ORIGIN_ROLES = {
    "event_spacing_fallback": "visual_packet_fallback_period",
    "saved_best_period": "saved_best_period",
    "saved_half_period": "half_saved_best_period",
    "saved_double_period": "double_saved_best_period",
    "fallback_half_period": "half_visual_fallback",
    "fallback_double_period": "double_visual_fallback",
    "historical_period_search": "historical_period_search",
    "top_event_period": "top_event_period_search_candidate",
    "top_bls_period": "top_bls_period_search_candidate",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def as_bool(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def split_periods(value: Any) -> list[float]:
    periods = []
    for part in clean(value).split("|"):
        period = refresh.as_float(part)
        if np.isfinite(period) and period > 0:
            periods.append(period)
    return periods


def same_period(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-9, atol=1e-10))


def add_period(
    periods: list[dict[str, Any]],
    period: Any,
    role: str,
    candidate: dict[str, Any] | None = None,
) -> None:
    value = refresh.as_float(period)
    if not np.isfinite(value) or value <= 0:
        return
    existing = next(
        (item for item in periods if same_period(item["period_days"], value)),
        None,
    )
    if existing is None:
        existing = {
            "period_days": value,
            "roles": [],
            "candidate": {},
        }
        periods.append(existing)
    if role and role not in existing["roles"]:
        existing["roles"].append(role)
    if candidate:
        for key, item in candidate.items():
            if clean(item) and not clean(existing["candidate"].get(key)):
                existing["candidate"][key] = item


def candidate_table_path(row: pd.Series) -> Path:
    return (
        PLOT_ROOT
        / clean(row["packet_source"])
        / clean(row["epic_id"])
        / "period_candidates.csv"
    )


def tested_periods(row: pd.Series) -> list[dict[str, Any]]:
    path = candidate_table_path(row)
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = pd.read_csv(path)
    periods: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        origin = clean(candidate.get("candidate_origin"))
        add_period(
            periods,
            candidate.get("period_days"),
            ORIGIN_ROLES.get(origin, origin or "period_search_candidate"),
            candidate,
        )

    saved = refresh.as_float(row.get("saved_historical_period_days"))
    if np.isfinite(saved) and saved > 0:
        add_period(periods, saved, "saved_best_period")
        add_period(periods, saved / 2.0, "half_saved_best_period")
        add_period(periods, saved * 2.0, "double_saved_best_period")

    fallback = refresh.as_float(row.get("event_spacing_fallback_period_days"))
    if np.isfinite(fallback) and fallback > 0:
        add_period(periods, fallback, "visual_packet_fallback_period")
        add_period(periods, fallback / 2.0, "half_visual_fallback")
        add_period(periods, fallback * 2.0, "double_visual_fallback")

    for historical in split_periods(row.get("historical_period_search_periods")):
        add_period(periods, historical, "historical_period_search")
        add_period(periods, historical / 2.0, "half_historical_period")
        add_period(periods, historical * 2.0, "double_historical_period")

    return sorted(periods, key=lambda item: item["period_days"])


def failure_reasons(metrics: dict[str, Any], coverage: float) -> list[str]:
    reasons = []
    checks = [
        ("primary_depth", "primary_depth_unavailable"),
        ("primary_depth_snr", "primary_depth_snr_unavailable"),
        ("odd_depth_median", "insufficient_odd_even_event_depths"),
        ("even_depth_median", "insufficient_odd_even_event_depths"),
        ("odd_even_depth_ratio", "insufficient_odd_even_event_depths"),
        ("secondary_depth_phase_05", "secondary_depth_unavailable"),
        ("secondary_depth_snr", "secondary_depth_snr_unavailable"),
        (
            "secondary_to_primary_depth_ratio",
            "primary_or_secondary_depth_unavailable",
        ),
        ("oot_to_depth", "oot_variability_or_primary_depth_unavailable"),
    ]
    for column, reason in checks:
        if not np.isfinite(refresh.as_float(metrics.get(column))):
            reasons.append(reason)
    if not np.isfinite(coverage):
        reasons.append("insufficient_event_epoch_span_for_coverage")
    return list(dict.fromkeys(reasons))


def evaluate_period(
    epic_id: str,
    packet_source: str,
    period_item: dict[str, Any],
    comparison: pd.Series,
    events: pd.DataFrame,
    lc: dict[str, Any],
    candidate_count: int,
) -> dict[str, Any]:
    period = float(period_item["period_days"])
    candidate = period_item["candidate"]
    support = refresh.as_float(candidate.get("event_support_count"))
    center = refresh.as_float(candidate.get("cluster_center_phase"))
    if not np.isfinite(support) or not np.isfinite(center):
        support, center = refresh.support_at(events, period)
    duration = refresh.as_float(candidate.get("bls_duration_days"))
    calculation_info = {
        "validation_period_days": period,
        "validation_period_source": "manual_review_period_diagnostic",
        "validation_cluster_center_phase": center,
        "validation_bls_duration_days": duration,
        "validation_period_event_support": int(max(0, support)),
        "candidate_period_count": candidate_count,
        # Deliberately false only while calculating so period ambiguity cannot
        # mask otherwise measurable period-dependent diagnostics.
        "period_ambiguity_flag": False,
    }
    metrics, family = refresh.recompute_metrics(
        epic_id,
        calculation_info,
        events,
        lc,
    )
    coherence = period_debug.event_stack_coherence(family)
    coverage = refresh.as_float(coherence["event_epoch_coverage"])
    reasons = failure_reasons(metrics, coverage)

    selected_period = refresh.as_float(comparison.get("validation_period_days"))
    is_selected = bool(
        np.isfinite(selected_period)
        and np.isclose(period, selected_period, rtol=2e-6, atol=1e-9)
    )
    ambiguous = as_bool(comparison.get("period_ambiguity_flag"))
    trusted_status = clean(comparison.get("period_comparison_status")).startswith(
        "trusted_"
    )
    is_trusted = bool(is_selected and trusted_status and not ambiguous)
    return {
        "epic_id": epic_id,
        "packet_source": packet_source,
        "tested_period_days": period,
        "period_role": "|".join(period_item["roles"]),
        "is_selected_period": is_selected,
        "is_trusted_period": is_trusted,
        "period_ambiguity_flag": ambiguous,
        "metric_trust_level": (
            "trusted_period_dependent"
            if is_trusted
            else "untrusted_period_dependent"
        ),
        "primary_depth": metrics["primary_depth"],
        "primary_depth_snr": metrics["primary_depth_snr"],
        "odd_depth_median": metrics["odd_depth_median"],
        "even_depth_median": metrics["even_depth_median"],
        "odd_even_depth_ratio": metrics["odd_even_depth_ratio"],
        "secondary_depth_phase_05": metrics["secondary_depth_phase_05"],
        "secondary_depth_snr": metrics["secondary_depth_snr"],
        "secondary_to_primary_depth_ratio": metrics[
            "secondary_to_primary_depth_ratio"
        ],
        "oot_to_depth": metrics["oot_to_depth"],
        "alias_risk": metrics["alias_risk"],
        "event_stack_score": refresh.as_float(
            coherence["event_stack_coherence_score"]
        ),
        "event_family_count": metrics["event_family_count"],
        "coverage": coverage,
        "missing_reason": "|".join(reasons),
    }


def failed_row(
    epic_id: str,
    packet_source: str,
    period_item: dict[str, Any],
    comparison: pd.Series,
    error: Exception,
) -> dict[str, Any]:
    period = float(period_item["period_days"])
    selected = refresh.as_float(comparison.get("validation_period_days"))
    ambiguous = as_bool(comparison.get("period_ambiguity_flag"))
    row = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update(
        {
            "epic_id": epic_id,
            "packet_source": packet_source,
            "tested_period_days": period,
            "period_role": "|".join(period_item["roles"]),
            "is_selected_period": bool(
                np.isfinite(selected)
                and np.isclose(period, selected, rtol=2e-6, atol=1e-9)
            ),
            "is_trusted_period": False,
            "period_ambiguity_flag": ambiguous,
            "metric_trust_level": "untrusted_period_dependent",
            "event_family_count": 0,
            "missing_reason": f"period_evaluation_exception:{type(error).__name__}",
        }
    )
    return row


def normalize_selected_periods(
    table: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    out = table.copy()
    out["is_selected_period"] = False
    out["is_trusted_period"] = False
    out["metric_trust_level"] = "untrusted_period_dependent"
    comparison_by_epic = comparisons.set_index("epic_id", drop=False)
    for epic_id, group in out.groupby("epic_id"):
        comparison = comparison_by_epic.loc[epic_id]
        selected = refresh.as_float(comparison.get("validation_period_days"))
        if not np.isfinite(selected):
            continue
        relative_distance = (
            pd.to_numeric(group["tested_period_days"], errors="coerce") - selected
        ).abs() / selected
        selected_index = relative_distance.idxmin()
        out.loc[selected_index, "is_selected_period"] = True
        ambiguous = as_bool(comparison.get("period_ambiguity_flag"))
        trusted_status = clean(
            comparison.get("period_comparison_status")
        ).startswith("trusted_")
        if trusted_status and not ambiguous:
            out.loc[selected_index, "is_trusted_period"] = True
            out.loc[
                selected_index,
                "metric_trust_level",
            ] = "trusted_period_dependent"
    return out


def write_summary(table: pd.DataFrame, generated_at: str) -> None:
    failed = table["missing_reason"].fillna("").ne("")
    reason_counts = Counter()
    for value in table.loc[failed, "missing_reason"].astype(str):
        reason_counts.update(part for part in value.split("|") if part)
    lines = [
        "Untrusted per-period manual-review diagnostic summary",
        f"generated_at={generated_at}",
        "scope=manual-review visibility only; no validation, prediction, label, or queue outputs modified",
        f"number_of_epics_tested={table['epic_id'].nunique()}",
        f"number_of_tested_periods={len(table)}",
        "number_of_rows_with_odd_even_depth_ratio_computed="
        f"{int(pd.to_numeric(table['odd_even_depth_ratio'], errors='coerce').notna().sum())}",
        "number_of_rows_with_oot_to_depth_computed="
        f"{int(pd.to_numeric(table['oot_to_depth'], errors='coerce').notna().sum())}",
        f"number_of_rows_where_calculation_failed={int(failed.sum())}",
        "",
        "top_reasons_for_calculation_failure",
    ]
    if reason_counts:
        lines.extend(
            f"{reason}={count}" for reason, count in reason_counts.most_common(10)
        )
    else:
        lines.append("none=0")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    comparisons = pd.read_csv(COMPARISON_CSV)
    if comparisons["epic_id"].duplicated().any():
        raise ValueError("Expected one comparison row per EPIC")

    rows = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    for index, comparison in comparisons.iterrows():
        epic_id = clean(comparison["epic_id"])
        packet_source = clean(comparison["packet_source"])
        periods = tested_periods(comparison)
        print(
            f"[{index + 1:02d}/{len(comparisons)}] "
            f"{epic_id}: {len(periods)} tested periods",
            flush=True,
        )
        events = refresh.load_events(epic_id)
        lc = refresh.load_cached_light_curve(epic_id)
        for period_item in periods:
            try:
                rows.append(
                    evaluate_period(
                        epic_id,
                        packet_source,
                        period_item,
                        comparison,
                        events,
                        lc,
                        len(periods),
                    )
                )
            except Exception as error:
                rows.append(
                    failed_row(
                        epic_id,
                        packet_source,
                        period_item,
                        comparison,
                        error,
                    )
                )

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    table = normalize_selected_periods(table, comparisons)
    table.to_csv(OUTPUT_CSV, index=False)
    write_summary(table, generated_at)
    print(f"Wrote {len(table)} per-period diagnostic rows", flush=True)


if __name__ == "__main__":
    main()
