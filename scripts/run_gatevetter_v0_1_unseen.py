from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gatevetter_v0_1_rules as frozen

SOURCE_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
MANUAL_64_DECISIONS = ROOT / "freezes" / "manual_vetting_batch_64" / "manual_vetting_batch_64_decisions.csv"
MANUAL_REVIEW_LEDGER = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "manual_vetting_decisions_ledger.csv"

VALIDATION_LEDGERS = [
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1" / "autovet_candidate_validation_ledger.csv",
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "autovet_hold_validation_ledger.csv",
    ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1" / "post_repair_validation_ledger.csv",
]

OUT_PREDICTIONS = ROOT / "gatevetter_v0_1_unseen_predictions.csv"
OUT_STAGE_G_QUEUE = ROOT / "gatevetter_v0_1_unseen_stage_g_review_queue.csv"
OUT_RULE_TRACE = ROOT / "gatevetter_v0_1_unseen_rule_trace.csv"

BATCH_SIZE = 64


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def first_nonempty(row: dict[str, Any], columns: list[str]) -> str:
    for column in columns:
        value = clean(row.get(column))
        if value:
            return value
    return ""


def truthy_text(row: dict[str, Any], columns: list[str], tokens: list[str]) -> bool:
    text = " ".join(clean(row.get(column)).lower() for column in columns)
    return any(token in text for token in tokens)


def alias_risk(row: dict[str, Any]) -> str:
    explicit = first_nonempty(row, ["alias_risk", "post_repair_alias_risk"])
    if explicit:
        return explicit
    gate = clean(row.get("alias_gate")).lower()
    if gate in {"pass", "fail", "review"}:
        return gate
    support = first_nonempty(row, ["post_repair_alias_best_support_ratio", "alias_best_support_ratio"])
    if not finite(support):
        return ""
    ratio = float(support)
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "moderate"
    return "low"


def fallback_period_flag(row: dict[str, Any]) -> str:
    if truthy_text(
        row,
        [
            "prefilter_reason",
            "recommended_prefilter_action",
            "triggered_rules",
            "autovet_reason",
            "reason",
            "repair_reason",
            "funnel_bucket",
        ],
        [
            "event_spacing_fallback",
            "fallback period",
            "fallback-only",
            "fallback/ambiguous",
            "no_saved_period_support",
            "needs_period_search",
            "run_period_search",
        ],
    ):
        return "true"
    if clean(row.get("period_gate")).lower() == "fail":
        return "true"
    return "false"


def duration_fraction(row: dict[str, Any]) -> str:
    existing = first_nonempty(row, ["duration_fraction_of_period"])
    if finite(existing):
        return f"{float(existing):.6g}"
    duration_hours = first_nonempty(row, ["transit_duration_hours"])
    period_days = first_nonempty(row, ["best_period_days", "primary_period", "autovet_period_days"])
    if finite(duration_hours) and finite(period_days) and float(period_days) > 0:
        return f"{float(duration_hours) / (float(period_days) * 24.0):.6g}"
    return ""


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def validation_rows_by_epic() -> dict[str, dict[str, Any]]:
    by_epic: dict[str, dict[str, Any]] = {}
    for path in VALIDATION_LEDGERS:
        if not path.exists():
            continue
        frame = load_csv(path)
        for _, row in frame.iterrows():
            epic_id = clean(row.get("epic_id"))
            if not epic_id:
                continue
            by_epic.setdefault(epic_id, {}).update(row.to_dict())
    return by_epic


def build_unseen_source_batch() -> pd.DataFrame:
    queue = load_csv(SOURCE_QUEUE)
    manual_64 = load_csv(MANUAL_64_DECISIONS)
    manual_64_ids = set(manual_64["epic_id"].map(clean))

    queue = queue.copy()
    queue.insert(0, "source_queue_rank", range(1, len(queue) + 1))
    unseen = queue[~queue["epic_id"].map(clean).isin(manual_64_ids)].head(BATCH_SIZE).copy()

    if len(unseen) != BATCH_SIZE:
        raise RuntimeError(f"Expected {BATCH_SIZE} unseen rows, found {len(unseen)}")
    if unseen["epic_id"].map(clean).isin(manual_64_ids).any():
        raise RuntimeError("Manual-64 EPIC leaked into unseen batch")
    return unseen


def build_features(source: pd.DataFrame) -> pd.DataFrame:
    diagnostics = validation_rows_by_epic()
    rows: list[dict[str, Any]] = []
    for _, source_row in source.iterrows():
        epic_id = clean(source_row.get("epic_id"))
        row = source_row.to_dict()
        row.update(diagnostics.get(epic_id, {}))
        rows.append(
            {
                "epic_id": epic_id,
                "cnn_score": first_nonempty(row, ["cnn_score", "flux_p_science_like"]),
                "primary_depth": first_nonempty(row, ["post_repair_primary_depth", "primary_depth"]),
                "primary_depth_snr": first_nonempty(
                    row,
                    ["post_repair_primary_depth_snr", "primary_snr", "primary_depth_snr", "best_depth_snr"],
                ),
                "odd_even_depth_ratio": first_nonempty(row, ["post_repair_odd_even_depth_ratio", "odd_even_ratio", "odd_even_depth_ratio"]),
                "secondary_depth_snr": first_nonempty(row, ["post_repair_secondary_depth_snr", "secondary_depth_snr"]),
                "secondary_to_primary_depth_ratio": first_nonempty(
                    row,
                    ["post_repair_secondary_to_primary_depth_ratio", "secondary_to_primary_ratio", "secondary_to_primary_depth_ratio"],
                ),
                "oot_to_depth": first_nonempty(row, ["oot_variability_to_depth", "oot_to_depth"]),
                "candidate_period_count": first_nonempty(row, ["candidate_period_count", "n_periods_proposed"]),
                "event_family_count": first_nonempty(row, ["event_family_count", "num_events", "n_events"]),
                "alias_risk": alias_risk(row),
                "fallback_period_flag": fallback_period_flag(row),
                "duration_fraction_of_period": duration_fraction(row),
                "best_period_days": first_nonempty(row, ["best_period_days", "primary_period", "autovet_period_days"]),
                "transit_duration_hours": first_nonempty(row, ["transit_duration_hours"]),
                "stage_f_label": first_nonempty(row, ["stage_f_label", "post_repair_stage_f_label", "stage_f_validation_label"]),
                "stage_f_reason": first_nonempty(row, ["stage_f_reason", "post_repair_stage_f_reason", "reason", "post_repair_reason"]),
            }
        )

    features = pd.DataFrame(rows, columns=["epic_id", *frozen.ALLOWED_FEATURES])
    frozen.assert_no_forbidden_prediction_columns(features)
    return features


def add_source_context(predictions: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    source_context = source[
        [
            "epic_id",
            "source_queue_rank",
            "autovet_label",
            "recommended_next_action",
            "review_priority_score",
            "autovet_rank_score",
            "funnel_bucket",
            "prefilter_rank",
            "prefilter_rank_score",
            "prefilter_reason",
        ]
    ].copy()
    source_context = source_context.rename(columns={column: f"source_{column}" for column in source_context.columns if column != "epic_id"})
    return predictions.merge(source_context, on="epic_id", how="left", validate="one_to_one")


def attach_manual_review_after_blind_queue(frame: pd.DataFrame) -> pd.DataFrame:
    manual_columns = [
        "epic_id",
        "manual_label",
        "manual_next_action",
        "manual_confidence",
        "manual_reason",
        "manual_vetted",
        "reviewed_at",
        "reviewer",
        "decision_authority",
    ]
    if MANUAL_REVIEW_LEDGER.exists():
        manual = load_csv(MANUAL_REVIEW_LEDGER)
        available = [column for column in manual_columns if column in manual.columns]
        manual = manual[available].copy()
        rename = {column: f"posthoc_{column}" for column in available if column != "epic_id"}
        manual = manual.rename(columns=rename)
        out = frame.merge(manual, on="epic_id", how="left", validate="one_to_one")
    else:
        out = frame.copy()

    for column in manual_columns:
        if column == "epic_id":
            continue
        posthoc = f"posthoc_{column}"
        if posthoc not in out.columns:
            out[posthoc] = ""
    return out.fillna("")


def build_stage_g_queue(scored: pd.DataFrame) -> pd.DataFrame:
    queue = scored[
        scored["gatevetter_prediction"].isin({"candidate_like_positive", "caveated_candidate_stage_g_review"})
    ].copy()
    if queue.empty:
        columns = [
            "stage_g_rank",
            "epic_id",
            "stage_g_action",
            "gatevetter_prediction",
            "gatevetter_score",
            "gatevetter_v0_reason",
            "rule_trace",
        ]
        return pd.DataFrame(columns=columns)
    queue = queue.sort_values("gatevetter_score", ascending=False).reset_index(drop=True)
    queue.insert(0, "stage_g_rank", range(1, len(queue) + 1))
    front = [
        "stage_g_rank",
        "epic_id",
        "stage_g_action",
        "gatevetter_prediction",
        "gatevetter_score",
        "gatevetter_v0_reason",
        "cnn_score",
        "primary_depth_snr",
        "odd_even_depth_ratio",
        "oot_to_depth",
        "candidate_period_count",
        "fallback_period_flag",
        "duration_fraction_of_period",
        "primary_gate",
        "hard_gate_fired",
        "hard_gates_fired",
        "hold_gates_fired",
        "penalties_or_missing_evidence",
        "rule_trace",
    ]
    trailing = [column for column in queue.columns if column not in front]
    return queue[front + trailing]


def write_rule_trace(scored: pd.DataFrame) -> None:
    columns = [
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_score",
        "primary_gate",
        "hard_gate_fired",
        "gatevetter_v0_reason",
        "rule_trace",
        "hard_gates_fired",
        "hold_gates_fired",
        "risk_gates_fired",
        "penalties_or_missing_evidence",
        "score_components",
        "source_source_queue_rank",
        "source_autovet_label",
        "posthoc_manual_label",
        "posthoc_manual_next_action",
        "posthoc_manual_vetted",
    ]
    scored[[column for column in columns if column in scored.columns]].to_csv(OUT_RULE_TRACE, index=False)


def main() -> None:
    source = build_unseen_source_batch()
    features = build_features(source)
    blind_predictions = pd.DataFrame([frozen.gatevet(row) for _, row in features.iterrows()])
    blind_scored = add_source_context(blind_predictions, source)
    blind_queue = build_stage_g_queue(blind_scored)

    scored = attach_manual_review_after_blind_queue(blind_scored)
    stage_g_queue = attach_manual_review_after_blind_queue(blind_queue)

    scored.to_csv(OUT_PREDICTIONS, index=False)
    stage_g_queue.to_csv(OUT_STAGE_G_QUEUE, index=False)
    write_rule_trace(scored)

    print(f"Wrote {OUT_PREDICTIONS.name} ({len(scored)} rows)")
    print(f"Wrote {OUT_STAGE_G_QUEUE.name} ({len(stage_g_queue)} rows)")
    print(f"Wrote {OUT_RULE_TRACE.name}")
    print()
    print("Prediction counts:")
    print(scored["gatevetter_prediction"].value_counts().to_string())
    print()
    print("Stage G queue:")
    if len(stage_g_queue):
        print(stage_g_queue[["stage_g_rank", "epic_id", "gatevetter_prediction", "gatevetter_score", "gatevetter_v0_reason"]].to_string(index=False))
    else:
        print("empty")


if __name__ == "__main__":
    main()
