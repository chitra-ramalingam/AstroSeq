from __future__ import annotations

import math
from typing import Any

import pandas as pd

import gatevetter_v0_1_rules as v0_1


CORE_DIAGNOSTICS = [
    "primary_depth",
    "primary_depth_snr",
    "odd_even_depth_ratio",
    "secondary_depth_snr",
    "secondary_to_primary_depth_ratio",
    "oot_to_depth",
    "duration_fraction_of_period",
]

STAGE_G_PREDICTIONS = {
    "candidate_like_positive",
    "caveated_candidate_stage_g_review",
}

ALLOWED_FEATURES = [
    *v0_1.ALLOWED_FEATURES,
    "period_source",
    "period_ambiguity_flag",
    "validation_period_source",
    "period_comparison_status",
    "trusted_period_validation",
    "metric_trust_level",
    "odd_even_depth_ratio_missing_reason",
]

FORBIDDEN_DURING_PREDICTION = v0_1.FORBIDDEN_DURING_PREDICTION


def clean(value: Any) -> str:
    return v0_1.clean(value)


def finite(value: Any) -> bool:
    return v0_1.finite(value)


def fnum(row: pd.Series, column: str, default: float = math.nan) -> float:
    return v0_1.fnum(row, column, default)


def truthy(value: Any) -> bool:
    return v0_1.truthy(value)


def odd_even_justified_unavailable(row: pd.Series) -> bool:
    return (
        trusted_period(row)
        and not finite(row.get("odd_even_depth_ratio"))
        and bool(clean(row.get("odd_even_depth_ratio_missing_reason")))
    )


def missing_core_diagnostics(row: pd.Series) -> list[str]:
    missing = []
    for column in CORE_DIAGNOSTICS:
        if finite(row.get(column)):
            continue
        if column == "odd_even_depth_ratio" and odd_even_justified_unavailable(row):
            continue
        missing.append(column)
    return missing


def stage_g_prerequisite_failures(row: pd.Series) -> list[str]:
    failures = [f"missing_{column}" for column in missing_core_diagnostics(row)]
    primary_depth = fnum(row, "primary_depth")
    primary_snr = fnum(row, "primary_depth_snr")
    alias = clean(row.get("alias_risk")).lower()
    if finite(primary_depth) and primary_depth <= 0:
        failures.append("primary_depth_not_positive")
    if finite(primary_snr) and primary_snr < 4.0:
        failures.append("primary_depth_snr_not_credible")
    if not alias:
        failures.append("alias_risk_missing")
    elif alias == "period_ambiguous":
        failures.append("alias_risk_period_ambiguous")
    return failures


def trusted_period(row: pd.Series) -> bool:
    if truthy(row.get("period_ambiguity_flag")):
        return False
    if not truthy(row.get("trusted_period_validation")):
        return False
    status = clean(row.get("period_comparison_status")).lower()
    source = clean(row.get("validation_period_source")).lower()
    return status.startswith("trusted_") and source not in {
        "",
        "period_ambiguous",
        "event_spacing_fallback_only",
    }


def period_source_is_fallback_only(row: pd.Series) -> bool:
    period_source = clean(row.get("period_source")).lower()
    validation_source = clean(row.get("validation_period_source")).lower()
    return period_source in {
        "event_spacing_fallback",
        "event_spacing_fallback_only",
    } or validation_source == "event_spacing_fallback_only"


def period_is_ambiguous(row: pd.Series) -> bool:
    return (
        truthy(row.get("period_ambiguity_flag"))
        or clean(row.get("validation_period_source")).lower()
        == "period_ambiguous"
        or clean(row.get("alias_risk")).lower() == "period_ambiguous"
    )


def finalize_policy_block(
    row: pd.Series,
    *,
    prediction: str,
    reason: str,
    gate: str,
    trace: str,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    score, score_notes, components = v0_1.score_survivor(row)
    penalties = list(score_notes)
    for item in missing or []:
        if item.startswith(("missing_", "primary_", "alias_")):
            penalties.append(item)
        else:
            penalties.append(f"missing_{item}")
    result = v0_1.finalize_prediction(
        row,
        prediction,
        reason,
        score,
        [gate],
        [],
        [],
        penalties,
        [trace],
        components,
    )
    return version_result(result, row)


def version_result(
    result: dict[str, Any],
    row: pd.Series | None = None,
) -> dict[str, Any]:
    out = dict(result)
    prediction = clean(out.get("gatevetter_prediction"))
    reason = clean(out.get("gatevetter_v0_reason"))
    out["gatevetter_v0_2_prediction"] = prediction
    out["gatevetter_v0_2_reason"] = reason
    if row is not None:
        for column in [
            "period_source",
            "period_ambiguity_flag",
            "validation_period_source",
            "period_comparison_status",
            "trusted_period_validation",
            "metric_trust_level",
            "odd_even_depth_ratio_missing_reason",
        ]:
            out[column] = clean(row.get(column))
    return out


def annotate_nonpromotion_result(
    result: dict[str, Any],
    row: pd.Series,
    missing: list[str],
) -> dict[str, Any]:
    out = version_result(result, row)
    annotations = []
    if period_is_ambiguous(row):
        annotations.append(
            "Gate v0.2 Stage G block: period or alias diagnostics are "
            "period-ambiguous"
        )
    if not trusted_period(row):
        annotations.append(
            "Gate v0.2 Stage G block: trusted period validation unavailable"
        )
    if missing:
        annotations.append(
            "Gate v0.2 Stage G block: missing diagnostics are not passes "
            f"({','.join(missing)})"
        )
    if period_source_is_fallback_only(row):
        annotations.append(
            "Gate v0.2 Stage G block: period is event-spacing fallback-only"
        )
    if annotations:
        prior = clean(out.get("rule_trace"))
        out["rule_trace"] = "; ".join([prior, *annotations] if prior else annotations)
    return out


def gatevet(row: pd.Series) -> dict[str, Any]:
    duration = v0_1.duration_fraction(row)
    missing = missing_core_diagnostics(row)
    period_count = fnum(row, "candidate_period_count", 0.0)

    if finite(duration) and duration <= 0.0:
        return finalize_policy_block(
            row,
            prediction="negative_reject_as_noise_or_artifact",
            reason="invalid_nonpositive_duration_fraction",
            gate="gate_v0_2_duration_fraction_nonpositive",
            trace="Gate v0.2 hard reject: duration fraction is nonpositive",
        )

    if finite(duration) and duration >= 0.35:
        return finalize_policy_block(
            row,
            prediction="negative_reject_as_noise_or_artifact",
            reason="hard_duration_fraction_ge_0_35",
            gate="gate_v0_2_duration_fraction_ge_0_35",
            trace=(
                "Gate v0.2 hard reject: duration fraction is invalid or "
                "greater than or equal to 0.35"
            ),
        )

    if finite(duration) and duration >= 0.20:
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="duration_fraction_ge_0_20_hold",
            gate="gate_v0_2_duration_fraction_ge_0_20",
            trace=(
                "Gate v0.2 hold: duration fraction is greater than or equal "
                "to 0.20 and below 0.35"
            ),
        )

    # v0.1 used 0.15/0.25 duration thresholds. v0.2 owns duration policy, so
    # values below the new 0.20 hold threshold must not trigger the old gates.
    v0_1_row = row.copy()
    if finite(duration) and duration < 0.20:
        v0_1_row["duration_fraction_of_period"] = min(duration, 0.149999)
    result = v0_1.gatevet(v0_1_row)
    prediction = clean(result.get("gatevetter_prediction"))

    if (
        period_source_is_fallback_only(row)
        and period_count >= 500
        and missing
        and prediction
        not in {
            "false_positive_eb_or_variable",
            "negative_noise_or_artifact",
            "negative_reject_as_noise_or_artifact",
        }
    ):
        return finalize_policy_block(
            row,
            prediction="negative_low_priority",
            reason="fallback_period_clutter_missing_core_diagnostics",
            gate="gate_v0_2_fallback_ge_500_missing_core",
            trace=(
                "Gate v0.2 promotion block: event-spacing fallback with "
                "candidate period count >= 500 and missing core diagnostics"
            ),
            missing=missing,
        )

    if prediction not in STAGE_G_PREDICTIONS:
        return annotate_nonpromotion_result(result, row, missing)

    if period_is_ambiguous(row):
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="period_ambiguity_blocks_stage_g_promotion",
            gate="gate_v0_2_period_ambiguity",
            trace=(
                "Gate v0.2 promotion block: period or alias diagnostics are "
                "period-ambiguous"
            ),
            missing=missing,
        )

    if period_source_is_fallback_only(row):
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="fallback_only_period_blocks_stage_g_promotion",
            gate="gate_v0_2_fallback_only_period",
            trace=(
                "Gate v0.2 promotion block: period is event-spacing "
                "fallback-only"
            ),
            missing=missing,
        )

    if not trusted_period(row):
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="trusted_period_validation_required",
            gate="gate_v0_2_untrusted_period",
            trace=(
                "Gate v0.2 promotion block: trusted period validation is "
                "required before candidate promotion"
            ),
            missing=missing,
        )

    prerequisite_failures = stage_g_prerequisite_failures(row)
    if prerequisite_failures:
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="missing_core_diagnostics_block_promotion",
            gate="gate_v0_2_missing_core_diagnostics",
            trace=(
                "Gate v0.2 promotion block: missing diagnostics are not "
                "counted as passed diagnostics"
            ),
            missing=prerequisite_failures,
        )

    if clean(result.get("gatevetter_v0_reason")) == (
        "provisional_candidate_missing_crosschecks"
    ):
        return finalize_policy_block(
            row,
            prediction="excluded_uncertain_hold",
            reason="provisional_missing_crosschecks_disabled",
            gate="gate_v0_2_provisional_route_disabled",
            trace=(
                "Gate v0.2 promotion block: provisional missing-crosscheck "
                "route is disabled"
            ),
            missing=missing_core_diagnostics(row),
        )
    return version_result(result, row)


def assert_no_forbidden_prediction_columns(features: pd.DataFrame) -> None:
    forbidden = sorted(
        FORBIDDEN_DURING_PREDICTION.intersection(features.columns)
    )
    if forbidden:
        raise RuntimeError(
            f"Forbidden prediction columns present: {forbidden}"
        )
    unexpected = set(features.columns).difference({"epic_id", *ALLOWED_FEATURES})
    if unexpected:
        raise AssertionError(
            f"Unexpected GateVetter v0.2 prediction columns: {sorted(unexpected)}"
        )
