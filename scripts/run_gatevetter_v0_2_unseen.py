from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gatevetter_v0_2_rules as rules
import scripts.run_gatevetter_v0_1_unseen as v0_1_runner


REFRESHED_METRICS = ROOT / "unseen_full_validation_metrics_refreshed_all.csv"
OUT_PREDICTIONS = ROOT / "gatevetter_v0_2_unseen_predictions.csv"
OUT_STAGE_G_QUEUE = ROOT / "gatevetter_v0_2_unseen_stage_g_queue.csv"
OUT_HOLD_QUEUE = ROOT / "gatevetter_v0_2_unseen_hold_queue.csv"
OUT_REJECT_SUMMARY = ROOT / "gatevetter_v0_2_unseen_reject_summary.csv"
OUT_COMPARISON = ROOT / "gatevetter_v0_2_failure_mode_comparison.txt"
OUT_RULE_TRACE = ROOT / "gatevetter_v0_2_unseen_rule_trace.csv"
OUT_SUMMARY = ROOT / "gatevetter_v0_2_unseen_summary.txt"
V0_1_PREDICTIONS = ROOT / "gatevetter_v0_1_unseen_predictions.csv"

MANUALLY_REVIEWED_V0_1_STAGE_G = {
    "EPIC_211340132",
    "EPIC_211348972",
    "EPIC_211376143",
    "EPIC_211523843",
    "EPIC_211719075",
    "EPIC_211749084",
    "EPIC_211813179",
    "EPIC_211957368",
    "EPIC_211977585",
    "EPIC_212019207",
    "EPIC_212022078",
}


def clean(value: Any) -> str:
    return v0_1_runner.clean(value)


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def refreshed_diagnostics() -> pd.DataFrame:
    frame = v0_1_runner.load_csv(REFRESHED_METRICS)
    columns = [
        "epic_id",
        "validation_period_days",
        "validation_period_source",
        "odd_even_depth_ratio_missing_reason",
        "period_ambiguity_flag",
        "period_comparison_status",
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "duration_fraction_of_period",
        "odd_even_depth_ratio",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "oot_to_depth",
        "alias_risk",
        "event_family_count",
        "candidate_period_count",
        "missing_reason",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].copy()


def trusted_period_validation(row: pd.Series) -> str:
    status = clean(row.get("period_comparison_status")).lower()
    source = clean(row.get("validation_period_source")).lower()
    ambiguous = truthy(row.get("period_ambiguity_flag"))
    trusted = (
        not ambiguous
        and status.startswith("trusted_")
        and source not in {"", "period_ambiguous", "event_spacing_fallback_only"}
    )
    return "true" if trusted else "false"


def build_features(source: pd.DataFrame) -> pd.DataFrame:
    base = v0_1_runner.build_features(source).set_index("epic_id", drop=False)
    refreshed = refreshed_diagnostics().set_index("epic_id", drop=False)
    overlay_columns = [
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "duration_fraction_of_period",
        "odd_even_depth_ratio",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "oot_to_depth",
        "alias_risk",
        "event_family_count",
        "candidate_period_count",
    ]
    rows: list[dict[str, Any]] = []
    for epic_id, base_row in base.iterrows():
        row = base_row.to_dict()
        if epic_id in refreshed.index:
            diagnostic = refreshed.loc[epic_id]
            for column in overlay_columns:
                row[column] = clean(diagnostic.get(column))
            row["best_period_days"] = clean(
                diagnostic.get("validation_period_days")
            )
            row["period_ambiguity_flag"] = clean(
                diagnostic.get("period_ambiguity_flag")
            )
            validation_source = clean(
                diagnostic.get("validation_period_source")
            )
            row["validation_period_source"] = clean(
                diagnostic.get("validation_period_source")
            )
            row["period_source"] = (
                "event_spacing_fallback"
                if validation_source == "event_spacing_fallback_only"
                else validation_source
            )
            row["period_comparison_status"] = clean(
                diagnostic.get("period_comparison_status")
            )
            row["odd_even_depth_ratio_missing_reason"] = clean(
                diagnostic.get("odd_even_depth_ratio_missing_reason")
            )
            row["metric_trust_level"] = (
                "untrusted_period_dependent"
                if truthy(diagnostic.get("period_ambiguity_flag"))
                else "trusted_period_dependent"
            )
            row["trusted_period_validation"] = trusted_period_validation(
                diagnostic
            )
        else:
            row.update(
                {
                    "period_ambiguity_flag": "",
                    "period_source": (
                        "event_spacing_fallback"
                        if truthy(row.get("fallback_period_flag"))
                        else ""
                    ),
                    "validation_period_source": "",
                    "period_comparison_status": "",
                    "trusted_period_validation": "false",
                    "metric_trust_level": "period_validation_unavailable",
                    "odd_even_depth_ratio_missing_reason": "",
                }
            )
        rows.append(row)

    features = pd.DataFrame(rows, columns=["epic_id", *rules.ALLOWED_FEATURES])
    rules.assert_no_forbidden_prediction_columns(features)
    return features


def build_stage_g_queue(scored: pd.DataFrame) -> pd.DataFrame:
    queue = scored[
        scored["gatevetter_prediction"].isin(
            {"candidate_like_positive", "caveated_candidate_stage_g_review"}
        )
    ].copy()
    if queue.empty:
        return pd.DataFrame(
            columns=[
                "stage_g_rank",
                "epic_id",
                "stage_g_action",
                "gatevetter_prediction",
                "gatevetter_score",
                "gatevetter_v0_2_reason",
                "rule_trace",
            ]
        )
    queue = queue.sort_values("gatevetter_score", ascending=False).reset_index(
        drop=True
    )
    queue.insert(0, "stage_g_rank", range(1, len(queue) + 1))
    front = [
        "stage_g_rank",
        "epic_id",
        "stage_g_action",
        "gatevetter_prediction",
        "gatevetter_score",
        "gatevetter_v0_2_reason",
        "period_ambiguity_flag",
        "trusted_period_validation",
        "period_comparison_status",
        "validation_period_source",
        "primary_gate",
        "hard_gate_fired",
        "hard_gates_fired",
        "penalties_or_missing_evidence",
        "rule_trace",
    ]
    trailing = [column for column in queue.columns if column not in front]
    return queue[front + trailing]


def build_hold_queue(scored: pd.DataFrame) -> pd.DataFrame:
    holds = scored[
        scored["gatevetter_prediction"].isin(
            {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
        )
    ].copy()
    holds = holds.sort_values(
        ["gatevetter_score", "epic_id"], ascending=[False, True]
    ).reset_index(drop=True)
    holds.insert(0, "hold_rank", range(1, len(holds) + 1))
    front = [
        "hold_rank",
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_score",
        "gatevetter_v0_2_reason",
        "primary_gate",
        "period_ambiguity_flag",
        "trusted_period_validation",
        "validation_period_source",
        "penalties_or_missing_evidence",
        "rule_trace",
    ]
    return holds[front + [c for c in holds.columns if c not in front]]


def build_reject_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rejects = scored[
        ~scored["gatevetter_prediction"].isin(
            {
                "candidate_like_positive",
                "caveated_candidate_stage_g_review",
                "excluded_uncertain_hold",
                "excluded_uncertain_hold_positive",
            }
        )
    ].copy()
    if rejects.empty:
        return pd.DataFrame(
            columns=[
                "gatevetter_prediction",
                "gatevetter_v0_2_reason",
                "reject_count",
                "epic_ids",
            ]
        )
    summary = (
        rejects.groupby(
            ["gatevetter_prediction", "gatevetter_v0_2_reason"],
            dropna=False,
        )["epic_id"]
        .agg(
            reject_count="size",
            epic_ids=lambda values: "|".join(sorted(values)),
        )
        .reset_index()
        .sort_values(
            ["reject_count", "gatevetter_prediction", "gatevetter_v0_2_reason"],
            ascending=[False, True, True],
        )
    )
    return summary


def apply_manual_queue_disposition(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "posthoc_manual_label" not in frame.columns:
        return frame
    manual_label = frame["posthoc_manual_label"].map(clean)
    manual_reject = manual_label.map(
        lambda value: any(
            token in value.lower()
            for token in (
                "reject",
                "binary",
                "variable_or_possible_eb",
                "noise_or_artifact",
                "false_positive",
            )
        )
    )
    out = frame.loc[~manual_reject].copy()
    if "hold_rank" in out.columns:
        out = out.reset_index(drop=True)
        out["hold_rank"] = range(1, len(out) + 1)
    if "stage_g_rank" in out.columns:
        out = out.reset_index(drop=True)
        out["stage_g_rank"] = range(1, len(out) + 1)
    return out


def write_rule_trace(scored: pd.DataFrame) -> None:
    columns = [
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_score",
        "gatevetter_v0_2_reason",
        "primary_gate",
        "hard_gate_fired",
        "hard_gates_fired",
        "hold_gates_fired",
        "penalties_or_missing_evidence",
        "period_ambiguity_flag",
        "trusted_period_validation",
        "period_comparison_status",
        "validation_period_source",
        "metric_trust_level",
        "rule_trace",
        "score_components",
        "source_source_queue_rank",
        "posthoc_manual_label",
        "posthoc_manual_next_action",
        "posthoc_manual_vetted",
    ]
    scored[
        [column for column in columns if column in scored.columns]
    ].to_csv(OUT_RULE_TRACE, index=False)


def write_summary(scored: pd.DataFrame, queue: pd.DataFrame) -> None:
    reviewed = scored[
        scored["epic_id"].isin(MANUALLY_REVIEWED_V0_1_STAGE_G)
    ].copy()
    reviewed_promoted = reviewed[
        reviewed["gatevetter_prediction"].isin(
            {"candidate_like_positive", "caveated_candidate_stage_g_review"}
        )
    ]
    period_ambiguity_blocks = sum(
        rules.period_is_ambiguous(row) for _, row in scored.iterrows()
    )
    untrusted_period_blocks = sum(
        not rules.trusted_period(row) for _, row in scored.iterrows()
    )
    missing_core_blocks = sum(
        bool(rules.missing_core_diagnostics(row))
        for _, row in scored.iterrows()
    )
    lines = [
        "GateVetter v0.2 unseen rerun summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "source_batch=same 64-object unseen packet used by GateVetter v0.1",
        f"prediction_rows={len(scored)}",
        f"stage_g_queue_rows={len(queue)}",
        "manual_v0_1_stage_g_review_rows=11",
        f"manual_v0_1_stage_g_rows_repromoted={len(reviewed_promoted)}",
        f"period_ambiguity_stage_g_blocks={period_ambiguity_blocks}",
        f"untrusted_period_stage_g_blocks={untrusted_period_blocks}",
        f"missing_core_stage_g_blocks={missing_core_blocks}",
        f"fallback_ge_500_missing_core_blocks={int(scored['gatevetter_v0_2_reason'].eq('fallback_period_clutter_missing_core_diagnostics').sum())}",
        f"hard_duration_rejects={int(scored['gatevetter_v0_2_reason'].eq('hard_duration_fraction_ge_0_35').sum())}",
        "",
        "Prediction counts",
    ]
    lines.extend(
        f"{label}={count}"
        for label, count in scored["gatevetter_prediction"].value_counts().items()
    )
    lines.extend(["", "Previously reviewed v0.1 Stage G objects"])
    for _, row in reviewed.sort_values("epic_id").iterrows():
        lines.append(
            f"{row['epic_id']}={row['gatevetter_prediction']}|"
            f"{row['gatevetter_v0_2_reason']}"
        )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(
    scored: pd.DataFrame,
    stage_g_queue: pd.DataFrame,
    hold_queue: pd.DataFrame,
) -> None:
    v0_1 = v0_1_runner.load_csv(V0_1_PREDICTIONS)
    old_columns = [
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_v0_reason",
        "rule_trace",
    ]
    old = v0_1[old_columns].rename(
        columns={
            "gatevetter_prediction": "v0_1_prediction",
            "gatevetter_v0_reason": "v0_1_reason",
            "rule_trace": "v0_1_rule_trace",
        }
    )
    new_columns = [
        "epic_id",
        "gatevetter_prediction",
        "gatevetter_v0_2_reason",
        "rule_trace",
    ]
    new = scored[new_columns].rename(
        columns={
            "gatevetter_prediction": "v0_2_prediction",
            "gatevetter_v0_2_reason": "v0_2_reason",
            "rule_trace": "v0_2_rule_trace",
        }
    )
    comparison = old.merge(new, on="epic_id", validate="one_to_one")
    changed = comparison[
        comparison["v0_1_prediction"].ne(comparison["v0_2_prediction"])
        | comparison["v0_1_reason"].ne(comparison["v0_2_reason"])
    ].copy()
    old_stage_g = v0_1[
        v0_1["gatevetter_prediction"].isin(rules.STAGE_G_PREDICTIONS)
    ]
    known = scored[
        scored["epic_id"].isin(MANUALLY_REVIEWED_V0_1_STAGE_G)
    ]
    known_blocked = ~known["gatevetter_prediction"].isin(
        rules.STAGE_G_PREDICTIONS
    )

    lines = [
        "GateVetter v0.1 vs v0.2 unseen failure-mode comparison",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"v0_1_stage_g_rows={len(old_stage_g)}",
        f"v0_2_stage_g_rows={len(stage_g_queue)}",
        f"v0_1_stage_g_rows_removed={len(old_stage_g) - len(stage_g_queue)}",
        f"known_11_rejected_v0_1_stage_g_rows_blocked={bool(known_blocked.all())}",
        f"known_11_blocked_count={int(known_blocked.sum())}/11",
        f"changed_epic_count={len(changed)}",
        "",
        "Top 20 v0.2 holds",
    ]
    if hold_queue.empty:
        lines.append("none")
    else:
        for _, row in hold_queue.head(20).iterrows():
            lines.append(
                f"{int(row['hold_rank'])}. {row['epic_id']} | "
                f"score={row['gatevetter_score']} | "
                f"reason={row['gatevetter_v0_2_reason']}"
            )
    lines.extend(["", "Rule trace for every changed EPIC"])
    if changed.empty:
        lines.append("none")
    else:
        for _, row in changed.sort_values("epic_id").iterrows():
            lines.extend(
                [
                    f"{row['epic_id']}",
                    f"  v0.1={row['v0_1_prediction']}|{row['v0_1_reason']}",
                    f"  v0.1_rule_trace={row['v0_1_rule_trace']}",
                    f"  v0.2={row['v0_2_prediction']}|{row['v0_2_reason']}",
                    f"  v0.2_rule_trace={row['v0_2_rule_trace']}",
                ]
            )
    OUT_COMPARISON.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    scored: pd.DataFrame,
    stage_g_queue: pd.DataFrame,
) -> None:
    if len(scored) != v0_1_runner.BATCH_SIZE:
        raise RuntimeError(
            f"Expected {v0_1_runner.BATCH_SIZE} predictions, found {len(scored)}"
        )
    if scored["epic_id"].duplicated().any():
        raise RuntimeError("Duplicate EPIC ids in v0.2 predictions")
    known = scored[
        scored["epic_id"].isin(MANUALLY_REVIEWED_V0_1_STAGE_G)
    ]
    if len(known) != len(MANUALLY_REVIEWED_V0_1_STAGE_G):
        raise RuntimeError("Not all 11 known Stage G failures are in the rerun")
    if known["gatevetter_prediction"].isin(rules.STAGE_G_PREDICTIONS).any():
        raise RuntimeError("A known rejected v0.1 Stage G row was re-promoted")
    for _, row in stage_g_queue.iterrows():
        failures = rules.stage_g_prerequisite_failures(row)
        if not rules.trusted_period(row):
            failures.append("trusted_period_validation_required")
        if rules.period_is_ambiguous(row):
            failures.append("period_ambiguous")
        if rules.period_source_is_fallback_only(row):
            failures.append("fallback_only_period")
        if failures:
            raise RuntimeError(
                f"Invalid Stage G promotion for {row['epic_id']}: {failures}"
            )


def main() -> None:
    source = v0_1_runner.build_unseen_source_batch()
    features = build_features(source)
    blind_predictions = pd.DataFrame(
        [rules.gatevet(row) for _, row in features.iterrows()]
    )
    blind_scored = v0_1_runner.add_source_context(blind_predictions, source)
    blind_queue = build_stage_g_queue(blind_scored)
    blind_holds = build_hold_queue(blind_scored)
    reject_summary = build_reject_summary(blind_scored)

    scored = v0_1_runner.attach_manual_review_after_blind_queue(blind_scored)
    stage_g_queue = v0_1_runner.attach_manual_review_after_blind_queue(
        blind_queue
    )
    hold_queue = v0_1_runner.attach_manual_review_after_blind_queue(
        blind_holds
    )
    stage_g_queue = apply_manual_queue_disposition(stage_g_queue)
    hold_queue = apply_manual_queue_disposition(hold_queue)
    validate_outputs(scored, stage_g_queue)
    scored.to_csv(OUT_PREDICTIONS, index=False)
    stage_g_queue.to_csv(OUT_STAGE_G_QUEUE, index=False)
    hold_queue.to_csv(OUT_HOLD_QUEUE, index=False)
    reject_summary.to_csv(OUT_REJECT_SUMMARY, index=False)
    write_rule_trace(scored)
    write_summary(scored, stage_g_queue)
    write_comparison(scored, stage_g_queue, hold_queue)

    print(f"Wrote {OUT_PREDICTIONS.name} ({len(scored)} rows)")
    print(f"Wrote {OUT_STAGE_G_QUEUE.name} ({len(stage_g_queue)} rows)")
    print(f"Wrote {OUT_HOLD_QUEUE.name} ({len(hold_queue)} rows)")
    print(f"Wrote {OUT_REJECT_SUMMARY.name} ({len(reject_summary)} rows)")
    print(f"Wrote {OUT_COMPARISON.name}")
    print(f"Wrote {OUT_RULE_TRACE.name}")
    print(f"Wrote {OUT_SUMMARY.name}")
    print()
    print(scored["gatevetter_prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()
