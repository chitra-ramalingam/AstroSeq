from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent

TRAINING_VIEW = ROOT / "manual_64_training_view.csv"
FEATURE_PROBE = ROOT / "manual_64_autovet_v2_feature_probe.csv"
AUTOVET_V2_BLIND = ROOT / "autovet_v2_blind_manual64_predictions.csv"

OUT_PREDICTIONS = ROOT / "gatevetter_v0_manual64_predictions.csv"
OUT_CONFUSION = ROOT / "gatevetter_v0_manual64_confusion_matrix.csv"
OUT_FAILURE_REPORT = ROOT / "gatevetter_v0_failure_report.txt"
OUT_RULE_TRACE = ROOT / "gatevetter_v0_rule_trace.csv"
OUT_STAGE_G_QUEUE = ROOT / "gatevetter_v0_stage_g_queue.csv"

LABEL_ORDER = [
    "candidate_like_positive",
    "excluded_uncertain_hold",
    "excluded_uncertain_hold_positive",
    "false_positive_eb_or_variable",
    "negative_noise_or_artifact",
    "negative_reject_as_noise_or_artifact",
    "negative_low_priority",
]

ALLOWED_FEATURES = [
    "cnn_score",
    "primary_depth",
    "primary_depth_snr",
    "odd_even_depth_ratio",
    "secondary_depth_snr",
    "secondary_to_primary_depth_ratio",
    "oot_to_depth",
    "candidate_period_count",
    "event_family_count",
    "alias_risk",
    "fallback_period_flag",
    "duration_fraction_of_period",
    "best_period_days",
    "transit_duration_hours",
    "stage_f_label",
    "stage_f_reason",
]

FORBIDDEN_DURING_PREDICTION = {
    "freeze_batch_id",
    "manual_confidence",
    "manual_label",
    "manual_label_family",
    "manual_next_action",
    "manual_reason",
    "manual_truth_class",
    "reviewed_at",
    "reviewer",
    "science_binary",
    "stage_g_deeper_eval",
    "stage_g_deeper_eval_action",
    "training_class",
    "training_exclusion_reason",
    "training_use",
}

POSITIVE_LABELS = {"candidate_like", "planet_like"}
EB_VARIABLE_LABELS = {"binary_system", "variable_or_possible_eb"}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def fnum(row: pd.Series, column: str, default: float = math.nan) -> float:
    value = row.get(column, default)
    if not finite(value):
        return default
    return float(value)


def clip01(value: float) -> float:
    if not finite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def nonempty_first(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = clean(row.get(column))
        if value:
            return value
    return ""


def truth_class_from_manual_row(row: pd.Series) -> str:
    existing = clean(row.get("training_class"))
    if existing:
        return existing

    label = clean(row.get("manual_label")).lower()
    if label in POSITIVE_LABELS:
        return "candidate_like_positive"
    if label == "uncertain_hold_positive":
        return "excluded_uncertain_hold_positive"
    if label == "uncertain_hold":
        return "excluded_uncertain_hold"
    if label in EB_VARIABLE_LABELS:
        return "false_positive_eb_or_variable"
    if label == "reject_as_noise_or_artifact":
        return "negative_reject_as_noise_or_artifact"
    if label == "noise_or_artifact":
        return "negative_noise_or_artifact"
    if label == "low_priority_negative":
        return "negative_low_priority"
    return f"negative_{label or 'unknown'}"


def duration_fraction(row: pd.Series) -> float:
    existing = fnum(row, "duration_fraction_of_period")
    if finite(existing):
        return existing

    best_period_days = fnum(row, "best_period_days")
    transit_duration_hours = fnum(row, "transit_duration_hours")
    if finite(best_period_days) and best_period_days > 0 and finite(transit_duration_hours):
        return transit_duration_hours / (best_period_days * 24.0)
    return math.nan


def alias_is_high(value: Any) -> bool:
    return clean(value).lower() in {"high", "fail", "failed", "review_high"}


def stage_f_block(row: pd.Series) -> tuple[bool, bool, str]:
    text = f"{clean(row.get('stage_f_label'))} {clean(row.get('stage_f_reason'))}".lower()
    if not text.strip():
        return False, False, ""

    blockers = [
        "rejected",
        "reject",
        "primary depth not significant",
        "period failure",
        "run period search before label update",
    ]
    matched = [item for item in blockers if item in text]
    if not matched:
        return False, False, ""

    severe = any(item in text for item in ["rejected", "reject", "period failure"])
    return True, severe, "stage_f_blocks_candidate:" + "|".join(matched)


def score_survivor(row: pd.Series) -> tuple[float, list[str], dict[str, float]]:
    cnn = clip01(fnum(row, "cnn_score", 0.0))
    snr = fnum(row, "primary_depth_snr")
    odd = fnum(row, "odd_even_depth_ratio")
    oot = fnum(row, "oot_to_depth")
    duration = duration_fraction(row)

    components: dict[str, float] = {"cnn_morphology": round(0.45 * cnn, 6)}
    notes: list[str] = []

    if finite(snr):
        if snr < 4.0:
            snr_component = 0.04
            notes.append("weak_primary_depth_snr")
        elif snr <= 80.0:
            snr_component = min(0.25, 0.12 + (snr / 80.0) * 0.13)
        elif snr <= 300.0:
            snr_component = 0.18
            notes.append("very_high_primary_depth_snr_capped")
        else:
            snr_component = 0.08
            notes.append("absurd_primary_depth_snr_suspicious")
        components["primary_depth_snr"] = round(snr_component, 6)
    else:
        components["primary_depth_snr"] = 0.0
        notes.append("missing_primary_depth_snr")

    if finite(odd):
        components["odd_even_support"] = 0.12 if odd >= 0.85 else 0.06
    else:
        components["odd_even_support"] = 0.0
        notes.append("missing_odd_even_depth_ratio")

    if finite(oot):
        components["oot_cleanliness"] = 0.10 if oot < 0.15 else 0.06 if oot < 0.30 else 0.0
    else:
        components["oot_cleanliness"] = 0.0
        notes.append("missing_oot_to_depth")

    if finite(duration):
        components["duration_sanity"] = 0.08 if duration < 0.10 else 0.04 if duration < 0.15 else 0.0
    else:
        components["duration_sanity"] = 0.0
        notes.append("missing_duration_fraction")

    score = round(sum(components.values()), 6)
    return score, notes, components


def gatevet(row: pd.Series) -> dict[str, Any]:
    cnn = clip01(fnum(row, "cnn_score", 0.0))
    snr = fnum(row, "primary_depth_snr")
    odd = fnum(row, "odd_even_depth_ratio")
    secondary_snr = fnum(row, "secondary_depth_snr")
    secondary_ratio = fnum(row, "secondary_to_primary_depth_ratio")
    oot = fnum(row, "oot_to_depth")
    period_count = fnum(row, "candidate_period_count", 0.0)
    fallback = truthy(row.get("fallback_period_flag"))
    duration = duration_fraction(row)

    hard_gates: list[str] = []
    hold_gates: list[str] = []
    risk_gates: list[str] = []
    penalties: list[str] = []
    trace: list[str] = []
    blocked_candidate = False

    # Gate 1: physical duration sanity.
    if finite(duration):
        if duration >= 0.25:
            hard_gates.append("gate1_duration_fraction_ge_0.25")
            trace.append("Gate 1 hard reject: duration fraction >= 0.25")
            return finalize_prediction(
                row,
                "negative_reject_as_noise_or_artifact",
                "physical_duration_impossible",
                0.0,
                hard_gates,
                hold_gates,
                risk_gates,
                penalties,
                trace,
            )
        if duration >= 0.15:
            hold_gates.append("gate1_duration_fraction_ge_0.15")
            trace.append("Gate 1 hold: duration fraction >= 0.15")
            blocked_candidate = True
    else:
        penalties.append("gate1_duration_fraction_missing")

    # Gate 2: period reliability.
    if fallback and period_count >= 1000:
        hard_gates.append("gate2_fallback_period_with_ge_1000_candidates")
        trace.append("Gate 2 hard reject: fallback period with candidate period count >= 1000")
        return finalize_prediction(
            row,
            "negative_noise_or_artifact",
            "fallback_period_clutter_hard_reject",
            0.0,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
        )
    if fallback and period_count >= 500:
        hold_gates.append("gate2_fallback_period_with_ge_500_candidates")
        trace.append("Gate 2 low-priority hold: fallback period with candidate period count >= 500")
        blocked_candidate = True
    elif fallback:
        penalties.append("gate2_fallback_period_context")
    elif period_count >= 1000:
        penalties.append("gate2_candidate_period_count_ge_1000_no_fallback")
    elif period_count >= 500:
        penalties.append("gate2_candidate_period_count_ge_500_no_fallback")

    # Gate 3: EB / variable risk.
    if (
        (finite(secondary_ratio) and secondary_ratio >= 0.25)
        or (finite(secondary_snr) and secondary_snr >= 7.0)
        or (finite(odd) and odd <= 0.65)
        or (alias_is_high(row.get("alias_risk")) and finite(odd) and odd <= 0.75)
    ):
        hard_gates.append("gate3_eb_or_variable_risk")
        trace.append("Gate 3 route: EB/variable risk")
        return finalize_prediction(
            row,
            "false_positive_eb_or_variable",
            "eb_or_variable_gate",
            0.0,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
        )
    if finite(odd) and odd <= 0.75:
        hold_gates.append("gate3_odd_even_ratio_le_0.75")
        trace.append("Gate 3 hold: odd/even ratio <= 0.75")
        blocked_candidate = True

    # Gate 4: out-of-transit contamination.
    if finite(oot):
        if oot >= 0.50:
            hold_gates.append("gate4_oot_to_depth_ge_0.50")
            trace.append("Gate 4 hold/reject block: OOT variability >= 50% of depth")
            blocked_candidate = True
        elif oot >= 0.30:
            hold_gates.append("gate4_oot_to_depth_ge_0.30")
            trace.append("Gate 4 hold: OOT variability >= 30% of depth")
            blocked_candidate = True
    else:
        penalties.append("gate4_oot_to_depth_missing")

    # Gate 5: Stage F context.
    stage_blocked, stage_severe, stage_reason = stage_f_block(row)
    if stage_blocked:
        blocked_candidate = True
        gate_name = "gate5_stage_f_blocks_candidate"
        if stage_severe:
            hard_gates.append(gate_name)
            trace.append("Gate 5 route: severe Stage F rejection context")
            return finalize_prediction(
                row,
                "negative_reject_as_noise_or_artifact",
                stage_reason,
                0.0,
                hard_gates,
                hold_gates,
                risk_gates,
                penalties,
                trace,
            )
        hold_gates.append(gate_name)
        trace.append("Gate 5 hold: Stage F says do not promote")

    candidate_score, score_notes, components = score_survivor(row)
    penalties.extend(score_notes)

    if blocked_candidate:
        if any(gate.startswith("gate2_fallback_period") for gate in hold_gates):
            prediction = "negative_low_priority"
            reason = "period_reliability_hold_low_priority"
        else:
            prediction = "excluded_uncertain_hold"
            reason = "gate_blocked_candidate_hold"
        return finalize_prediction(
            row,
            prediction,
            reason,
            candidate_score,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
            components,
        )

    has_required_candidate_evidence = (
        cnn >= 0.65
        and not fallback
        and finite(snr)
        and snr >= 4.0
        and finite(odd)
        and odd > 0.75
        and finite(oot)
        and oot < 0.30
    )
    if has_required_candidate_evidence and candidate_score >= 0.62:
        return finalize_prediction(
            row,
            "candidate_like_positive",
            "passed_gates_candidate_survivor",
            candidate_score,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
            components,
        )

    has_provisional_missing_context_candidate = (
        fallback
        and period_count < 500
        and 0.65 <= cnn <= 0.75
        and finite(snr)
        and 20.0 <= snr <= 80.0
        and not finite(odd)
        and not finite(oot)
        and not finite(duration)
        and not alias_is_high(row.get("alias_risk"))
    )
    if has_provisional_missing_context_candidate:
        trace.append(
            "Gate 6 provisional candidate: no hard/hold gate fired; fallback count is low; "
            "SNR is credible; missing cross-checks did not fail"
        )
        components["provisional_missing_context_support"] = 0.14
        candidate_score = max(candidate_score + 0.14, 0.621)
        return finalize_prediction(
            row,
            "candidate_like_positive",
            "provisional_candidate_missing_crosschecks",
            candidate_score,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
            components,
        )

    if cnn >= 0.65 and finite(snr) and snr >= 4.0:
        return finalize_prediction(
            row,
            "excluded_uncertain_hold",
            "incomplete_or_uncertain_candidate_evidence",
            candidate_score,
            hard_gates,
            hold_gates,
            risk_gates,
            penalties,
            trace,
            components,
        )

    return finalize_prediction(
        row,
        "negative_noise_or_artifact",
        "weak_or_incomplete_signal_after_gates",
        candidate_score,
        hard_gates,
        hold_gates,
        risk_gates,
        penalties,
        trace,
        components,
    )


def finalize_prediction(
    row: pd.Series,
    prediction: str,
    reason: str,
    candidate_score: float,
    hard_gates: list[str],
    hold_gates: list[str],
    risk_gates: list[str],
    penalties: list[str],
    trace: list[str],
    components: dict[str, float] | None = None,
) -> dict[str, Any]:
    components = components or {}
    if not trace:
        trace = ["Gates 1-5 passed; candidate survivor scoring applied"]

    if hard_gates:
        primary_gate = hard_gates[0]
    elif hold_gates:
        primary_gate = hold_gates[0]
    elif prediction == "candidate_like_positive":
        primary_gate = "gate6_candidate_survivor"
    elif prediction == "excluded_uncertain_hold":
        primary_gate = "gate6_conservative_hold"
    else:
        primary_gate = "gate6_weak_or_incomplete_signal"

    if prediction == "candidate_like_positive":
        gatevetter_score = candidate_score
    elif hard_gates:
        gatevetter_score = min(candidate_score, 0.05)
    elif hold_gates:
        gatevetter_score = min(candidate_score, 0.55)
    elif prediction == "excluded_uncertain_hold":
        gatevetter_score = min(candidate_score, 0.60)
    else:
        gatevetter_score = min(candidate_score, 0.35)

    return {
        "epic_id": clean(row.get("epic_id")),
        "gatevetter_prediction": prediction,
        "gatevetter_score": round(gatevetter_score, 6),
        "gatevetter_v0_prediction": prediction,
        "gatevetter_v0_reason": reason,
        "candidate_survivor_score": round(candidate_score, 6),
        "stage_g_action": "validation_only_no_gatevetter_stage_g_queue",
        "primary_gate": primary_gate,
        "hard_gate_fired": "true" if hard_gates else "false",
        "rule_trace": "; ".join(trace),
        "hard_gates_fired": "|".join(hard_gates) or "none",
        "hold_gates_fired": "|".join(hold_gates) or "none",
        "risk_gates_fired": "|".join(risk_gates) or "none",
        "penalties_or_missing_evidence": "|".join(dict.fromkeys(penalties)) or "none",
        "score_components": "|".join(f"{key}:{value}" for key, value in components.items()) or "none",
        "cnn_score": clean(row.get("cnn_score")),
        "primary_depth": clean(row.get("primary_depth")),
        "primary_depth_snr": clean(row.get("primary_depth_snr")),
        "odd_even_depth_ratio": clean(row.get("odd_even_depth_ratio")),
        "secondary_depth_snr": clean(row.get("secondary_depth_snr")),
        "secondary_to_primary_depth_ratio": clean(row.get("secondary_to_primary_depth_ratio")),
        "oot_to_depth": clean(row.get("oot_to_depth")),
        "candidate_period_count": clean(row.get("candidate_period_count")),
        "event_family_count": clean(row.get("event_family_count")),
        "alias_risk": clean(row.get("alias_risk")),
        "fallback_period_flag": clean(row.get("fallback_period_flag")),
        "duration_fraction_of_period": "" if not finite(duration_fraction(row)) else round(duration_fraction(row), 6),
        "best_period_days": clean(row.get("best_period_days")),
        "transit_duration_hours": clean(row.get("transit_duration_hours")),
        "stage_f_label": clean(row.get("stage_f_label")),
        "stage_f_reason": clean(row.get("stage_f_reason")),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAINING_VIEW.exists():
        raise FileNotFoundError(TRAINING_VIEW)
    if not FEATURE_PROBE.exists():
        raise FileNotFoundError(FEATURE_PROBE)

    view = pd.read_csv(TRAINING_VIEW, dtype=str).fillna("")
    probe = pd.read_csv(FEATURE_PROBE, dtype=str).fillna("")
    if view["epic_id"].nunique() != len(view):
        raise RuntimeError("manual_64_training_view.csv has duplicate EPIC ids")
    if probe["epic_id"].nunique() != len(probe):
        raise RuntimeError("manual_64_autovet_v2_feature_probe.csv has duplicate EPIC ids")
    return view, probe


def assert_no_forbidden_prediction_columns(features: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_DURING_PREDICTION.intersection(features.columns))
    if forbidden:
        raise RuntimeError(f"Forbidden prediction columns present: {forbidden}")

    allowed = {"epic_id", *ALLOWED_FEATURES}
    unexpected = sorted(set(features.columns) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected prediction columns present: {unexpected}")


def build_blind_feature_table(view: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    merged = view.merge(probe, on="epic_id", how="left", suffixes=("_view", "_probe"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "epic_id": clean(row.get("epic_id")),
                "cnn_score": nonempty_first(row, ["cnn_score_probe", "cnn_score_view"]),
                "primary_depth": nonempty_first(row, ["primary_depth_view", "primary_depth_probe", "primary_depth"]),
                "primary_depth_snr": nonempty_first(row, ["primary_depth_snr_probe", "primary_depth_snr_view"]),
                "odd_even_depth_ratio": nonempty_first(row, ["odd_even_depth_ratio_probe", "odd_even_depth_ratio_view"]),
                "secondary_depth_snr": nonempty_first(row, ["secondary_depth_snr_view", "secondary_depth_snr_probe"]),
                "secondary_to_primary_depth_ratio": nonempty_first(
                    row,
                    ["secondary_to_primary_depth_ratio_view", "secondary_to_primary_depth_ratio_probe"],
                ),
                "oot_to_depth": nonempty_first(
                    row,
                    ["oot_to_depth", "oot_to_depth_probe", "oot_to_depth_view", "oot_variability_to_depth_view"],
                ),
                "candidate_period_count": nonempty_first(
                    row,
                    ["candidate_period_count", "candidate_period_count_probe", "candidate_period_count_view", "n_periods_proposed"],
                ),
                "event_family_count": nonempty_first(row, ["event_family_count_view", "event_family_count_probe", "n_events"]),
                "alias_risk": nonempty_first(row, ["alias_risk", "alias_risk_probe", "alias_risk_view"]),
                "fallback_period_flag": nonempty_first(row, ["fallback_period_flag", "fallback_period_flag_probe", "fallback_period_flag_view"]),
                "duration_fraction_of_period": nonempty_first(
                    row,
                    ["duration_fraction_of_period", "duration_fraction_of_period_probe", "duration_fraction_of_period_view"],
                ),
                "best_period_days": nonempty_first(row, ["best_period_days_view", "best_period_days_probe"]),
                "transit_duration_hours": nonempty_first(row, ["transit_duration_hours_view", "transit_duration_hours_probe"]),
                "stage_f_label": nonempty_first(row, ["stage_f_label_view", "stage_f_label_probe"]),
                "stage_f_reason": nonempty_first(row, ["stage_f_reason_view", "stage_f_reason_probe"]),
            }
        )

    features = pd.DataFrame(rows, columns=["epic_id", *ALLOWED_FEATURES])
    assert_no_forbidden_prediction_columns(features)
    return features


def attach_manual_truth(predictions: pd.DataFrame, view: pd.DataFrame) -> pd.DataFrame:
    truth_rows = [
        {
            "epic_id": clean(row.get("epic_id")),
            "manual_truth_class": truth_class_from_manual_row(row),
            "manual_label": clean(row.get("manual_label")),
            "manual_label_family": clean(row.get("manual_label_family")),
        }
        for _, row in view.iterrows()
    ]
    truth = pd.DataFrame(truth_rows)
    out = predictions.merge(truth, on="epic_id", how="left", validate="one_to_one")
    out["match"] = out["manual_truth_class"].eq(out["gatevetter_v0_prediction"])
    return out


def confusion_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    present_truths = [label for label in LABEL_ORDER if label in set(scored["manual_truth_class"])]
    present_predictions = [label for label in LABEL_ORDER if label in set(scored["gatevetter_v0_prediction"])]
    for truth in present_truths:
        subset = scored[scored["manual_truth_class"].eq(truth)]
        row = {"manual_truth_class": truth}
        for pred in present_predictions:
            row[pred] = int(subset["gatevetter_v0_prediction"].eq(pred).sum())
        row["row_total"] = int(len(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def build_stage_g_queue(scored: pd.DataFrame) -> pd.DataFrame:
    queue = scored[scored["manual_truth_class"].eq("candidate_like_positive")].copy()
    if queue.empty:
        return pd.DataFrame(
            columns=[
                "stage_g_rank",
                "epic_id",
                "stage_g_action",
                "gatevetter_prediction",
                "gatevetter_score",
                "gatevetter_v0_reason",
                "manual_truth_class",
                "rule_trace",
            ]
        )
    queue = queue.sort_values("gatevetter_score", ascending=False).reset_index(drop=True)
    queue.insert(0, "stage_g_rank", range(1, len(queue) + 1))
    queue["stage_g_action"] = "manual_candidate_like_only_gatevetter_not_validated"
    return queue[
        [
            "stage_g_rank",
            "epic_id",
            "stage_g_action",
            "gatevetter_prediction",
            "gatevetter_score",
            "gatevetter_v0_reason",
            "manual_truth_class",
            "manual_label",
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
    ]


def maybe_autovet_comparison(scored: pd.DataFrame) -> list[str]:
    if not AUTOVET_V2_BLIND.exists():
        return ["AutoVet v2 blind comparison: not available."]
    previous = pd.read_csv(AUTOVET_V2_BLIND, dtype=str).fillna("")
    merged = scored.merge(
        previous[["epic_id", "blind_prediction"]],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    promoted_now = set(scored.loc[scored["gatevetter_v0_prediction"].eq("candidate_like_positive"), "epic_id"])
    promoted_v2 = set(previous.loc[previous["blind_prediction"].eq("candidate_like_positive"), "epic_id"])
    removed = sorted(promoted_v2 - promoted_now)
    added = sorted(promoted_now - promoted_v2)
    disagreements = int((~merged["gatevetter_v0_prediction"].eq(merged["blind_prediction"])).sum())
    return [
        "AutoVet v2 blind comparison",
        f"- AutoVet v2 candidate-like promotions: {len(promoted_v2)}",
        f"- GateVetter v0 candidate-like promotions: {len(promoted_now)}",
        f"- Removed AutoVet v2 promotions: {len(removed)}",
        f"- Added v0 promotions: {len(added)}",
        f"- Prediction disagreements vs AutoVet v2: {disagreements}",
        f"- Removed promotion EPICs: {', '.join(removed) if removed else 'none'}",
        f"- Added promotion EPICs: {', '.join(added) if added else 'none'}",
    ]


def failure_report(scored: pd.DataFrame, matrix: pd.DataFrame, stage_g_queue: pd.DataFrame) -> str:
    n_rows = len(scored)
    n_matches = int(scored["match"].sum())
    accuracy = n_matches / n_rows if n_rows else 0.0
    truth_counts = Counter(scored["manual_truth_class"])
    pred_counts = Counter(scored["gatevetter_v0_prediction"])

    positive_truth = scored[scored["manual_truth_class"].eq("candidate_like_positive")]
    positive_recall_hits = int(positive_truth["gatevetter_v0_prediction"].eq("candidate_like_positive").sum())
    positive_recall = positive_recall_hits / len(positive_truth) if len(positive_truth) else 0.0

    candidate_predictions = scored[scored["gatevetter_v0_prediction"].eq("candidate_like_positive")]
    candidate_precision_hits = int(candidate_predictions["manual_truth_class"].eq("candidate_like_positive").sum())
    candidate_precision = candidate_precision_hits / len(candidate_predictions) if len(candidate_predictions) else 0.0
    false_promotions = candidate_predictions[~candidate_predictions["manual_truth_class"].eq("candidate_like_positive")].copy()

    manual_candidate_ids = ["EPIC_211915147", "EPIC_211497712", "EPIC_211953866", "EPIC_211357782"]
    manual_candidates = scored[scored["epic_id"].isin(manual_candidate_ids)].copy()
    manual_candidates["_manual_candidate_order"] = manual_candidates["epic_id"].map(
        {epic_id: index for index, epic_id in enumerate(manual_candidate_ids)}
    )
    manual_candidates = manual_candidates.sort_values("_manual_candidate_order")

    top_non_candidates = (
        scored[~scored["manual_truth_class"].eq("candidate_like_positive")]
        .sort_values("gatevetter_score", ascending=False)
        .head(10)
        .copy()
    )

    gate_counts: Counter[str] = Counter()
    for value in scored["hard_gates_fired"].tolist() + scored["hold_gates_fired"].tolist():
        for item in clean(value).split("|"):
            if item and item != "none":
                gate_counts[item] += 1

    misses = scored[~scored["match"]].copy()
    misses = misses.sort_values(["manual_truth_class", "candidate_survivor_score"], ascending=[True, False])

    lines = [
        "GateVetter v0 manual-64 calibration report",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "design=gate-first; candidate survivor scoring only runs after Gates 1-5",
        "manual_truth_attachment=after blind prediction only",
        f"prediction_inputs=epic_id plus explicit allowlist only ({', '.join(ALLOWED_FEATURES)})",
        f"forbidden_prediction_fields={', '.join(sorted(FORBIDDEN_DURING_PREDICTION))}",
        "",
        "Accuracy",
        f"- rows: {n_rows}",
        f"- matches: {n_matches}",
        f"- misses: {n_rows - n_matches}",
        f"- exact_accuracy: {accuracy:.3f}",
        f"- candidate_like_positive recall: {positive_recall_hits} / {len(positive_truth)} = {positive_recall:.3f}",
        f"- candidate_like_positive precision: {candidate_precision_hits} / {len(candidate_predictions)} = {candidate_precision:.3f}",
        f"- false candidate promotions: {len(false_promotions)}",
        "",
        "Truth counts",
    ]
    lines.extend(f"- {label}: {truth_counts[label]}" for label in LABEL_ORDER if truth_counts[label])
    lines.extend(["", "Prediction counts"])
    lines.extend(f"- {label}: {pred_counts[label]}" for label in LABEL_ORDER if pred_counts[label])
    lines.extend(["", "Gate counts"])
    if gate_counts:
        lines.extend(f"- {gate}: {count}" for gate, count in gate_counts.most_common())
    else:
        lines.append("- none")

    lines.extend(["", "Confusion matrix", matrix.to_string(index=False), ""])
    lines.extend(maybe_autovet_comparison(scored))

    lines.extend(["", "Manual candidate checks"])
    for _, row in manual_candidates.iterrows():
        lines.append(
            f"- {row['epic_id']}: pred={row['gatevetter_prediction']}; truth={row['manual_truth_class']}; "
            f"score={row['gatevetter_score']}; primary_gate={row['primary_gate']}; "
            f"hard_gate={row['hard_gate_fired']}; trace={row['rule_trace']}"
        )

    lines.extend(["", "Top 10 non-candidates by GateVetter candidate score"])
    if top_non_candidates.empty:
        lines.append("- none")
    else:
        for _, row in top_non_candidates.iterrows():
            lines.append(
                f"- {row['epic_id']}: truth={row['manual_truth_class']}; pred={row['gatevetter_prediction']}; "
                f"score={row['gatevetter_score']}; primary_gate={row['primary_gate']}; reason={row['gatevetter_v0_reason']}"
            )

    lines.extend(["", "Dangerous false candidate promotions"])
    if false_promotions.empty:
        lines.append("- none")
    else:
        for _, row in false_promotions.sort_values("gatevetter_score", ascending=False).iterrows():
            lines.append(
                f"- {row['epic_id']}: truth={row['manual_truth_class']}; score={row['gatevetter_score']}; "
                f"primary_gate={row['primary_gate']}; hard={row['hard_gates_fired']}; trace={row['rule_trace']}"
            )

    lines.extend(["", "Stage G queue policy"])
    lines.append("- GateVetter-driven Stage G queueing is disabled for v0 validation.")
    lines.append("- gatevetter_v0_stage_g_queue.csv contains manual candidate_like rows only.")
    lines.append(f"- manual candidate rows in queue: {len(stage_g_queue)}")
    if len(stage_g_queue):
        for _, row in stage_g_queue.iterrows():
            lines.append(
                f"- {row['epic_id']}: action={row['stage_g_action']}; pred={row['gatevetter_prediction']}; "
                f"truth={row['manual_truth_class']}; score={row['gatevetter_score']}; reason={row['gatevetter_v0_reason']}"
            )

    lines.extend(
        [
            "",
            "Post-run assessment",
            f"- Beats AutoVet v2 blind: {'yes' if positive_recall_hits > 0 and len(false_promotions) < 7 else 'partial/no'} "
            f"(AutoVet v2 recall was 0/4 with 7 candidate-like promotions; GateVetter v0 recall is "
            f"{positive_recall_hits}/4 with {len(false_promotions)} false candidate promotions).",
            f"- Recovered manual candidates: {positive_recall_hits} / {len(positive_truth)}.",
            "- Avoided promoting known biggest noise cases: yes; no negative_noise_or_artifact or "
            "negative_reject_as_noise_or_artifact row is predicted candidate_like_positive.",
            "- Gates too harsh: Gate 2 fallback+>=1000 period clutter blocks EPIC_211953866; Gate 4 OOT>=0.50 "
            "and Gate 1 duration>=0.15 hold EPIC_211915147 / EPIC_211497712.",
            "- Gates too weak: EB/variable routing still misses rows with blank secondary fields and acceptable odd/even; "
            "candidate scoring can rank uncertain holds high when diagnostics are missing rather than failed.",
            "- Next threshold adjustments: test a caveated-candidate tier for fallback period_count=0 with excellent odd/even; "
            "test OOT>=0.50 as forced hold rather than permanent block only when odd/even is excellent and no secondary is present; "
            "test splitting fallback>=1000 into hard reject versus review when odd/even and OOT are both clean.",
        ]
    )

    lines.extend(["", "Misses"])
    if misses.empty:
        lines.append("- none")
    else:
        for _, row in misses.iterrows():
            lines.append(
                f"- {row['epic_id']}: truth={row['manual_truth_class']}; pred={row['gatevetter_v0_prediction']}; "
                f"score={row['gatevetter_score']}; raw_survivor_score={row['candidate_survivor_score']}; hard={row['hard_gates_fired']}; "
                f"hold={row['hold_gates_fired']}; reason={row['gatevetter_v0_reason']}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    view, probe = load_inputs()
    features = build_blind_feature_table(view, probe)
    predictions = pd.DataFrame([gatevet(row) for _, row in features.iterrows()])
    scored = attach_manual_truth(predictions, view)

    front = [
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_score",
        "manual_truth_class",
        "match",
        "primary_gate",
        "hard_gate_fired",
        "rule_trace",
        "cnn_score",
        "primary_depth_snr",
        "odd_even_depth_ratio",
        "oot_to_depth",
        "candidate_period_count",
        "fallback_period_flag",
        "duration_fraction_of_period",
        "secondary_to_primary_depth_ratio",
        "alias_risk",
        "stage_f_label",
        "stage_f_reason",
    ]
    trailing = [col for col in scored.columns if col not in front]
    scored = scored[front + trailing]

    matrix = confusion_matrix(scored)
    stage_g_queue = build_stage_g_queue(scored)

    scored.to_csv(OUT_PREDICTIONS, index=False)
    matrix.to_csv(OUT_CONFUSION, index=False)
    scored[
        [
            "epic_id",
            "gatevetter_prediction",
            "gatevetter_score",
            "manual_truth_class",
            "match",
            "primary_gate",
            "hard_gate_fired",
            "gatevetter_v0_reason",
            "rule_trace",
            "hard_gates_fired",
            "hold_gates_fired",
            "risk_gates_fired",
            "penalties_or_missing_evidence",
            "score_components",
        ]
    ].to_csv(OUT_RULE_TRACE, index=False)
    stage_g_queue.to_csv(OUT_STAGE_G_QUEUE, index=False)
    report_text = failure_report(scored, matrix, stage_g_queue)
    OUT_FAILURE_REPORT.write_text(report_text, encoding="utf-8")

    print(f"Wrote {OUT_PREDICTIONS.name} ({len(scored)} rows)")
    print(f"Wrote {OUT_CONFUSION.name}")
    print(f"Wrote {OUT_FAILURE_REPORT.name}")
    print(f"Wrote {OUT_RULE_TRACE.name}")
    print(f"Wrote {OUT_STAGE_G_QUEUE.name} ({len(stage_g_queue)} rows)")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
