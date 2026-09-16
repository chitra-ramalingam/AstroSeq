from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PACKET = ROOT / "unseen_stage_g_review_manual_packet_refreshed.csv"
OUTPUT_PACKET = ROOT / "unseen_stage_g_review_manual_packet_refreshed_reclassified.csv"
PERIOD_NEEDED = ROOT / "unseen_period_confirmation_needed.csv"
SUMMARY = ROOT / "unseen_refreshed_metric_reclassification_summary.txt"
SPECIAL_EPIC = "EPIC_211340132"


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def period_ambiguous(row: pd.Series) -> bool:
    return (
        clean(row.get("validation_period_source")) == "period_ambiguous"
        or as_bool(row.get("period_ambiguity_flag"))
        or clean(row.get("odd_even_depth_ratio_missing_reason"))
        == "period_ambiguous_untrusted"
        or clean(row.get("oot_to_depth_missing_reason"))
        == "period_ambiguous_untrusted"
    )


def classify(row: pd.Series) -> dict[str, Any]:
    epic_id = clean(row.get("epic_id"))
    if epic_id == SPECIAL_EPIC:
        return {
            "refreshed_triage_classification": "variable_or_artifact",
            "refreshed_triage_route": "uncertain_hold_variable",
            "refreshed_triage_reason": (
                "refreshed OOT variability exceeds transit depth: "
                f"oot_to_depth={row.get('oot_to_depth')}; "
                f"primary_depth_snr={row.get('primary_depth_snr')}; "
                f"odd_even_depth_ratio={row.get('odd_even_depth_ratio')}; "
                f"candidate_period_count={row.get('candidate_period_count')}; "
                "pending manual final label"
            ),
            "refreshed_candidate_like_allowed": False,
            "pending_manual_final_label": True,
        }
    if period_ambiguous(row):
        reasons = []
        if clean(row.get("validation_period_source")) == "period_ambiguous":
            reasons.append("validation_period_source=period_ambiguous")
        if as_bool(row.get("period_ambiguity_flag")):
            reasons.append("period_ambiguity_flag=true")
        if clean(row.get("odd_even_depth_ratio_missing_reason")) == "period_ambiguous_untrusted":
            reasons.append("odd_even_depth_ratio_untrusted")
        if clean(row.get("oot_to_depth_missing_reason")) == "period_ambiguous_untrusted":
            reasons.append("oot_to_depth_untrusted")
        return {
            "refreshed_triage_classification": "period_confirmation_needed",
            "refreshed_triage_route": "period_confirmation_needed",
            "refreshed_triage_reason": "|".join(reasons),
            "refreshed_candidate_like_allowed": False,
            "pending_manual_final_label": True,
        }
    return {
        "refreshed_triage_classification": "manual_review",
        "refreshed_triage_route": "manual_review",
        "refreshed_triage_reason": "no_reclassification_rule_fired",
        "refreshed_candidate_like_allowed": False,
        "pending_manual_final_label": True,
    }


def main() -> None:
    packet = pd.read_csv(INPUT_PACKET)
    packet["epic_id"] = packet["epic_id"].astype(str)
    manual_columns = [
        col
        for col in packet.columns
        if "manual" in col
        or col
        in {
            "posthoc_reviewed_at",
            "posthoc_reviewer",
            "posthoc_decision_authority",
        }
    ]
    manual_before = packet[["epic_id", *manual_columns]].copy()

    packet["pre_refresh_stage_g_action"] = packet["stage_g_action"]
    packet["pre_refresh_gatevetter_prediction"] = packet["gatevetter_prediction"]
    classifications = packet.apply(classify, axis=1, result_type="expand")
    for column in classifications.columns:
        packet[column] = classifications[column]
    packet["stage_g_action"] = packet["refreshed_triage_route"]
    packet["refreshed_triage_authority"] = (
        "refreshed_validation_packet_triage_not_final_manual_label"
    )
    packet["refreshed_triage_at"] = datetime.now().isoformat(timespec="seconds")

    period_needed = packet.loc[
        packet["refreshed_triage_route"].eq("period_confirmation_needed")
    ].copy()

    packet.to_csv(OUTPUT_PACKET, index=False)
    period_needed.to_csv(PERIOD_NEEDED, index=False)

    manual_after = packet[["epic_id", *manual_columns]]
    if not manual_before.equals(manual_after):
        raise RuntimeError("Manual label columns changed during packet triage")
    if len(packet) != 11:
        raise RuntimeError(f"Expected 11 Stage G rows, found {len(packet)}")
    if len(period_needed) != 10:
        raise RuntimeError(
            f"Expected 10 period-confirmation rows, found {len(period_needed)}"
        )
    special = packet.loc[packet["epic_id"].eq(SPECIAL_EPIC)]
    if len(special) != 1:
        raise RuntimeError(f"Missing or duplicate {SPECIAL_EPIC}")
    if special.iloc[0]["refreshed_triage_route"] != "uncertain_hold_variable":
        raise RuntimeError(f"{SPECIAL_EPIC} did not receive required variable hold route")
    if packet["refreshed_candidate_like_allowed"].map(as_bool).any():
        raise RuntimeError("A refreshed Stage G row remains candidate-like allowed")

    lines = [
        "Unseen refreshed-metric packet reclassification summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"input_packet={INPUT_PACKET.name}",
        f"output_packet={OUTPUT_PACKET.name}",
        f"rows_input={len(packet)}",
        f"rows_output={len(packet)}",
        f"period_confirmation_needed={len(period_needed)}",
        "uncertain_hold_variable=1",
        "candidate_like_allowed_after_refresh=0",
        "",
        "Routing counts",
    ]
    lines.extend(
        f"{route}={count}"
        for route, count in packet["refreshed_triage_route"].value_counts().items()
    )
    lines.extend(
        [
            "",
            "Policy applied",
            "- Period-ambiguous rows were removed from Stage G candidate-like routing.",
            "- Rows with period_ambiguous_untrusted odd/even or OOT diagnostics were routed to period_confirmation_needed.",
            "- Original GateVetter predictions were retained for provenance; refreshed_triage_route controls this packet.",
            "- GateVetter thresholds and final manual labels were not changed.",
            "",
            f"{SPECIAL_EPIC}",
            "- refreshed_triage_classification=variable_or_artifact",
            "- refreshed_triage_route=uncertain_hold_variable",
            "- basis: odd_even_depth_ratio=0.959229; oot_to_depth=1.6082; primary_depth_snr=5.328567; candidate_period_count=14",
            "- pending manual final label",
            "",
            "Period confirmation EPICs",
        ]
    )
    lines.extend(f"- {epic}" for epic in period_needed["epic_id"])
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"reclassified_rows={len(packet)}")
    print(f"period_confirmation_needed={len(period_needed)}")
    print(packet[["epic_id", "refreshed_triage_classification", "refreshed_triage_route"]].to_string(index=False))


if __name__ == "__main__":
    main()
