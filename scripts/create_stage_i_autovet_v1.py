from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "plots" / "k2_batch" / "stage_i_prefilter_funnel.csv"
MODEL_PATH = ROOT / "models" / "k2_nocrop_flux_seed46_split303.best.keras"
INFER_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"
INFER_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"
LABELS_CSV = ROOT / "training_labels_v3.csv"
OUT_RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
OUT_POLICY = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_policy_note.txt"
OUT_REVIEW = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"

ACTIVE_MODEL = "models/k2_nocrop_flux_seed46_split303.best.keras"
REVIEW_HOLD_LIMIT = 250
LABEL_ORDER = [
    "auto_reject_likely_eb_or_artifact",
    "auto_reject_noise_or_artifact",
    "auto_hold_needs_review",
    "auto_candidate_with_caveat",
    "auto_high_priority_candidate",
]

OPTIONAL_METRICS = [
    "primary_depth",
    "radius_ratio_sqrt_depth",
    "primary_depth_snr",
    "transit_duration_hours",
    "odd_even_depth_ratio",
    "odd_even_depth_delta_explicit",
    "secondary_to_primary_depth_ratio",
    "secondary_depth_snr",
    "oot_variability_to_depth",
    "alias_best_support_ratio",
    "spike_fraction_2cadence",
    "depth_ratio",
    "period_support_count",
    "event_family_count",
    "best_period_days",
]

OUTPUT_COLUMNS = [
    "epic_id",
    "autovet_label",
    "autovet_confidence",
    "recommended_next_action",
    "explanation_short",
    "triggered_rules",
    "key_metrics_used",
    "autovet_rank_score",
    "review_priority_score",
    "flux_p_science_like",
    "flux_p_top3_mean",
    "flux_p_top10_mean",
    "flux_num_segments",
    "triage_score_global",
    "triage_step_score",
    "triage_whiteness_score",
    "best_shape_score",
    "best_depth_snr",
    "n_events",
    "n_periods_proposed",
    "n_periods_validated",
    "best_period_days",
    "period_support_count",
    "event_family_count",
    "primary_depth",
    "radius_ratio_sqrt_depth",
    "primary_depth_snr",
    "transit_duration_hours",
    "odd_even_depth_ratio",
    "odd_even_depth_delta_explicit",
    "secondary_to_primary_depth_ratio",
    "secondary_depth_snr",
    "oot_variability_to_depth",
    "alias_best_support_ratio",
    "spike_fraction_2cadence",
    "pilot_noise_like_flag",
    "obvious_low_signal_flag",
    "funnel_bucket",
    "prefilter_rank",
    "prefilter_rank_score",
    "recommended_prefilter_action",
    "prefilter_reason",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def as_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def as_bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    raw = df[col]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def clip01(value: float) -> float:
    if not finite(value):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def load_flux_scores(epic_ids: set[str]) -> pd.DataFrame:
    meta = pd.read_parquet(INFER_META, columns=["star_id", "start", "end", "seg_mid_time"]).reset_index(drop=True)
    meta["star_id"] = meta["star_id"].astype(str)
    mask = meta["star_id"].isin(epic_ids).to_numpy()
    idx = np.flatnonzero(mask)
    matched = set(meta.loc[mask, "star_id"])
    missing = sorted(epic_ids - matched)
    if missing:
        raise ValueError(f"Missing {len(missing)} Stage I EPICs in {rel(INFER_META)}; first={missing[:10]}")

    x_mem = np.load(INFER_X, mmap_mode="r")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    expected = tuple(model.input_shape[1:])
    flux_shape = (int(x_mem.shape[1]), 1)
    if expected != flux_shape:
        raise ValueError(f"Model input {model.input_shape} does not match flux-only tensor shape {flux_shape}")

    preds = np.empty(len(idx), dtype="float32")
    batch_size = 2048
    for start in range(0, len(idx), batch_size):
        stop = min(start + batch_size, len(idx))
        x = np.asarray(x_mem[idx[start:stop], :, :1], dtype=np.float32)
        preds[start:stop] = np.asarray(model.predict(x, batch_size=256, verbose=0)).reshape(-1).astype("float32")

    seg = meta.loc[mask].copy().reset_index(drop=True)
    seg["segment_model_score"] = preds

    rows: list[dict[str, Any]] = []
    for epic_id, grp in seg.groupby("star_id", sort=True):
        probs = grp["segment_model_score"].to_numpy(float)
        order = np.argsort(-probs)
        best = grp.iloc[int(order[0])]
        rows.append(
            {
                "epic_id": epic_id,
                "flux_p_science_like": float(np.max(probs)),
                "flux_p_top3_mean": float(np.mean(np.sort(probs)[-min(3, len(probs)) :])),
                "flux_p_top10_mean": float(np.mean(np.sort(probs)[-min(10, len(probs)) :])),
                "flux_num_segments": int(len(grp)),
                "flux_best_segment_start": int(best["start"]),
                "flux_best_segment_end": int(best["end"]),
                "flux_best_segment_mid_time": float(best["seg_mid_time"]),
            }
        )
    return pd.DataFrame(rows)


def add_optional_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "best_period" in out.columns and "best_period_days" not in out.columns:
        out["best_period_days"] = out["best_period"]
    for col in OPTIONAL_METRICS:
        if col not in out.columns:
            out[col] = np.nan
    out["period_support_count"] = as_num(out, "period_support_count").combine_first(as_num(out, "n_periods_validated"))
    out["event_family_count"] = as_num(out, "event_family_count").combine_first(as_num(out, "n_events"))
    out["primary_depth_snr"] = as_num(out, "primary_depth_snr").combine_first(as_num(out, "best_depth_snr"))
    return out


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = add_optional_metric_columns(df)
    for col in [
        "flux_p_science_like",
        "flux_p_top3_mean",
        "flux_p_top10_mean",
        "triage_score_global",
        "triage_step_score",
        "triage_whiteness_score",
        "best_shape_score",
        "best_depth_snr",
        "n_events",
        "n_periods_proposed",
        "n_periods_validated",
        "best_period_days",
        "period_support_count",
        "event_family_count",
        "primary_depth",
        "radius_ratio_sqrt_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "spike_fraction_2cadence",
        "prefilter_rank_score",
    ]:
        out[col] = as_num(out, col)

    out["pilot_noise_like_flag"] = as_bool(out, "pilot_noise_like_flag")
    out["obvious_low_signal_flag"] = as_bool(out, "obvious_low_signal_flag")
    out["triage_usable_bool"] = as_bool(out, "triage_usable")

    decisions = [decide(row) for _, row in out.iterrows()]
    decision_df = pd.DataFrame(decisions, index=out.index)
    out = pd.concat([out, decision_df], axis=1)

    out["autovet_rank_score"] = (
        0.35 * out["flux_p_science_like"].fillna(0.0)
        + 0.20 * np.clip(out["period_support_count"].fillna(0.0) / 3.0, 0.0, 1.0)
        + 0.15 * np.clip(out["event_family_count"].fillna(0.0) / 12.0, 0.0, 1.0)
        + 0.15 * np.clip((out["best_shape_score"].fillna(0.0) - 0.58) / 0.25, 0.0, 1.0)
        + 0.10 * np.clip(np.log1p(out["best_depth_snr"].fillna(0.0)) / np.log1p(30.0), 0.0, 1.0)
        + 0.05 * np.clip(out["prefilter_rank_score"].fillna(0.0) / 0.75, 0.0, 1.0)
    )
    penalty = (
        0.10 * (out["obvious_low_signal_flag"].astype(float))
        + 0.15 * (out["pilot_noise_like_flag"].astype(float))
        + 0.10 * np.clip((out["triage_step_score"].fillna(0.0) - 0.20) / 0.60, 0.0, 1.0)
    )
    out["review_priority_score"] = np.clip(out["autovet_rank_score"] - penalty, 0.0, 1.0)
    return out


def decide(row: pd.Series) -> dict[str, Any]:
    flux = row["flux_p_science_like"]
    shape = row["best_shape_score"]
    depth_snr = row["best_depth_snr"]
    primary_depth = row["primary_depth"]
    radius_ratio = row["radius_ratio_sqrt_depth"]
    period = row["best_period_days"]
    period_count = row["period_support_count"]
    proposed = row["n_periods_proposed"]
    validated = row["n_periods_validated"]
    events = row["event_family_count"]
    step = row["triage_step_score"]
    white = row["triage_whiteness_score"]
    odd_delta = row["odd_even_depth_delta_explicit"]
    odd_ratio = row["odd_even_depth_ratio"]
    secondary_ratio = row["secondary_to_primary_depth_ratio"]
    secondary_snr = row["secondary_depth_snr"]
    oot = row["oot_variability_to_depth"]
    alias = row["alias_best_support_ratio"]
    spike = row["spike_fraction_2cadence"]
    pilot_noise = bool(row["pilot_noise_like_flag"])
    low_signal = bool(row["obvious_low_signal_flag"])
    triage_usable = bool(row["triage_usable_bool"])

    no_period = not finite(period_count) or period_count < 1 or not finite(validated) or validated < 1
    weak_period = not no_period and period_count < 2
    strong_period = finite(period_count) and period_count >= 2 and finite(validated) and validated >= 2
    event_ok = finite(events) and events >= 3
    event_strong = finite(events) and events >= 8
    shape_ok = finite(shape) and shape >= 0.66
    shape_strong = finite(shape) and shape >= 0.75
    snr_ok = finite(depth_snr) and depth_snr >= 7
    snr_moderate = finite(depth_snr) and 5 <= depth_snr < 12
    extreme_snr = finite(depth_snr) and depth_snr >= 75
    poor_shape = not finite(shape) or shape < 0.64
    high_step = finite(step) and step >= 0.35
    very_high_step = finite(step) and step >= 0.60
    strong_flux = finite(flux) and flux >= 0.75
    signal_flux = finite(flux) and flux >= 0.60
    weak_flux = not finite(flux) or flux < 0.45
    short_period = finite(period) and period < 5.0

    rules: list[str] = []
    caveats: list[str] = []

    if not triage_usable:
        rules.append("triage_unusable_or_insufficient_data")
        return build_decision(
            "auto_reject_noise_or_artifact",
            0.78,
            "deprioritize_until_detector_features_are_recovered",
            "Detector triage is unusable, so AutoVet cannot find reliable signal evidence.",
            rules,
            row,
        )

    if finite(primary_depth) and primary_depth > 0.02:
        rules.append("primary_depth_gt_2pct")
        if short_period:
            rules.append("deep_short_period_eb_risk")
    if finite(radius_ratio) and radius_ratio > 0.15:
        rules.append("radius_ratio_sqrt_depth_gt_0p15")
    if finite(odd_delta) and odd_delta >= 0.50:
        rules.append("severe_odd_even_depth_mismatch")
    if finite(odd_ratio) and (odd_ratio <= 0.50 or odd_ratio >= 1.80):
        rules.append("severe_odd_even_depth_ratio")
    if finite(secondary_ratio) and secondary_ratio >= 0.50:
        rules.append("strong_secondary_to_primary_depth_ratio")
    if finite(secondary_snr) and secondary_snr >= 7.0:
        rules.append("strong_secondary_depth_snr")
    if finite(spike) and spike >= 0.35:
        rules.append("spike_or_outlier_dominated")
    if pilot_noise and extreme_snr:
        rules.append("stage_i_pilot_spike_like_extreme_snr_pattern")
    if extreme_snr and poor_shape:
        rules.append("extreme_snr_with_poor_shape_support")

    eb_rules = [
        "primary_depth_gt_2pct",
        "deep_short_period_eb_risk",
        "radius_ratio_sqrt_depth_gt_0p15",
        "severe_odd_even_depth_mismatch",
        "severe_odd_even_depth_ratio",
        "strong_secondary_to_primary_depth_ratio",
        "strong_secondary_depth_snr",
        "spike_or_outlier_dominated",
        "stage_i_pilot_spike_like_extreme_snr_pattern",
        "extreme_snr_with_poor_shape_support",
    ]
    eb_hits = [r for r in rules if r in eb_rules]
    if eb_hits and (len(eb_hits) >= 2 or any(r in eb_hits for r in ["deep_short_period_eb_risk", "strong_secondary_depth_snr", "spike_or_outlier_dominated"])):
        return build_decision(
            "auto_reject_likely_eb_or_artifact",
            min(0.95, 0.72 + 0.05 * len(eb_hits)),
            "deprioritize_as_likely_eb_or_artifact_review_only_if_external_evidence_exists",
            "EB/artifact indicators dominate the candidate evidence.",
            rules,
            row,
        )

    noise_rules: list[str] = []
    if no_period:
        noise_rules.append("no_reliable_period_support")
    if low_signal:
        noise_rules.append("obvious_low_signal_flag")
    if weak_flux:
        noise_rules.append("weak_frozen_flux_model_support")
    if finite(depth_snr) and depth_snr < 5:
        noise_rules.append("weak_primary_depth_snr")
    if poor_shape:
        noise_rules.append("poor_shape_or_hit_support")
    if very_high_step:
        noise_rules.append("strong_step_systematic_indicator")
    if pilot_noise:
        noise_rules.append("stage_i_pilot_noise_like_pattern")

    if (
        pilot_noise
        or (no_period and low_signal)
        or (no_period and weak_flux and (poor_shape or not event_ok))
        or (no_period and len(noise_rules) >= 4)
        or (very_high_step and weak_flux and poor_shape)
    ):
        rules.extend(noise_rules)
        return build_decision(
            "auto_reject_noise_or_artifact",
            min(0.94, 0.68 + 0.04 * len(set(noise_rules))),
            "deprioritize_as_noise_or_artifact_keep_out_of_candidate_review_queue",
            "Noise/artifact evidence is stronger than the unresolved signal evidence.",
            rules,
            row,
        )

    if no_period:
        rules.append("no_reliable_period_support")
        if signal_flux or snr_ok or shape_ok or event_ok:
            rules.append("possible_signal_requires_period_search")
        return build_decision(
            "auto_hold_needs_review",
            0.58 if signal_flux else 0.52,
            "run_period_search_or_stage_i_plot_generation_before_candidate_promotion",
            "Some signal evidence exists, but there is no reliable saved period support.",
            rules,
            row,
        )

    if finite(oot) and oot >= 1.25:
        caveats.append("high_oot_variability_to_depth")
    if finite(alias) and alias >= 0.65:
        caveats.append("alias_ambiguity")
    if weak_period:
        caveats.append("limited_period_support")
    if snr_moderate:
        caveats.append("moderate_primary_snr")
    if high_step:
        caveats.append("step_systematic_caution")
    if finite(secondary_ratio) and secondary_ratio >= 0.25:
        caveats.append("borderline_secondary_depth_ratio")
    if finite(odd_delta) and odd_delta >= 0.25:
        caveats.append("borderline_odd_even_depth_delta")

    clean_secondary = (not finite(secondary_ratio) or secondary_ratio < 0.20) and (not finite(secondary_snr) or secondary_snr < 5.0)
    clean_odd_even = (not finite(odd_delta) or odd_delta < 0.20) and (not finite(odd_ratio) or 0.70 < odd_ratio < 1.45)
    low_oot = not finite(oot) or oot < 0.75
    low_alias = not finite(alias) or alias < 0.50
    plausible_depth = not finite(primary_depth) or primary_depth <= 0.02
    plausible_duration = not finite(row["transit_duration_hours"]) or 0.5 <= row["transit_duration_hours"] <= 15.0
    not_spiky = not finite(spike) or spike < 0.20

    if (
        strong_flux
        and strong_period
        and event_strong
        and shape_strong
        and snr_ok
        and depth_snr < 75
        and clean_secondary
        and clean_odd_even
        and low_oot
        and low_alias
        and plausible_depth
        and plausible_duration
        and not_spiky
        and not caveats
    ):
        rules.extend(["strong_flux_model_support", "strong_period_event_support", "clean_secondary_odd_even_oot_alias_metrics"])
        return build_decision(
            "auto_high_priority_candidate",
            0.88,
            "generate_or_review_stage_i_evidence_packet_then_manual_candidate_followup",
            "Strong frozen-model, period/event, and cleanliness metrics support high-priority candidate review.",
            rules,
            row,
        )

    if signal_flux and event_ok and shape_ok and snr_ok and clean_secondary and plausible_depth and plausible_duration and not_spiky:
        rules.extend(["flux_model_supports_signal", "adequate_period_event_support", "no_fatal_secondary_or_depth_flags"])
        rules.extend(caveats)
        confidence = 0.70 if caveats else 0.78
        return build_decision(
            "auto_candidate_with_caveat",
            confidence,
            "review_candidate_with_caveats_before_any_ledger_or_label_change",
            "Candidate evidence is present, but one or more caveats still need manual review.",
            rules,
            row,
        )

    rules.extend(caveats or ["possible_signal_without_clean_promotion_evidence"])
    if not event_ok:
        rules.append("weak_event_family_support")
    if not shape_ok:
        rules.append("borderline_shape_support")
    if not snr_ok:
        rules.append("weak_or_moderate_primary_snr")
    return build_decision(
        "auto_hold_needs_review",
        0.60 if signal_flux else 0.55,
        "hold_for_manual_review_or_period_validation_before_promotion",
        "The row has possible signal evidence, but not enough clean support for candidate promotion.",
        rules,
        row,
    )


def build_decision(
    label: str,
    confidence: float,
    next_action: str,
    explanation: str,
    rules: list[str],
    row: pd.Series,
) -> dict[str, Any]:
    seen: list[str] = []
    for rule in rules:
        if rule and rule not in seen:
            seen.append(rule)
    return {
        "autovet_label": label,
        "autovet_confidence": round(float(np.clip(confidence, 0.0, 0.99)), 3),
        "recommended_next_action": next_action,
        "explanation_short": explanation,
        "triggered_rules": "; ".join(seen),
        "key_metrics_used": metric_summary(row),
    }


def metric_summary(row: pd.Series) -> str:
    fields = [
        ("flux_p_science_like", row.get("flux_p_science_like")),
        ("triage_score_global", row.get("triage_score_global")),
        ("triage_step_score", row.get("triage_step_score")),
        ("triage_whiteness_score", row.get("triage_whiteness_score")),
        ("best_shape_score", row.get("best_shape_score")),
        ("best_depth_snr", row.get("best_depth_snr")),
        ("n_events", row.get("n_events")),
        ("n_periods_validated", row.get("n_periods_validated")),
        ("best_period_days", row.get("best_period_days")),
        ("primary_depth", row.get("primary_depth")),
        ("radius_ratio_sqrt_depth", row.get("radius_ratio_sqrt_depth")),
        ("secondary_to_primary_depth_ratio", row.get("secondary_to_primary_depth_ratio")),
        ("odd_even_depth_delta_explicit", row.get("odd_even_depth_delta_explicit")),
        ("oot_variability_to_depth", row.get("oot_variability_to_depth")),
        ("alias_best_support_ratio", row.get("alias_best_support_ratio")),
        ("spike_fraction_2cadence", row.get("spike_fraction_2cadence")),
    ]
    return "; ".join(f"{name}={fmt(value)}" for name, value in fields)


def build_review_queue(scored: pd.DataFrame) -> pd.DataFrame:
    include_labels = {"auto_high_priority_candidate", "auto_candidate_with_caveat"}
    direct = scored[scored["autovet_label"].isin(include_labels)].copy()
    holds = scored[scored["autovet_label"].eq("auto_hold_needs_review")].copy()
    holds = holds.sort_values(["review_priority_score", "autovet_rank_score"], ascending=False).head(REVIEW_HOLD_LIMIT)
    review = pd.concat([direct, holds], ignore_index=True)
    order = pd.Categorical(
        review["autovet_label"],
        categories=["auto_high_priority_candidate", "auto_candidate_with_caveat", "auto_hold_needs_review"],
        ordered=True,
    )
    review = review.assign(_label_order=order)
    review = review.sort_values(["_label_order", "review_priority_score", "autovet_rank_score"], ascending=[True, False, False])
    return review.drop(columns=["_label_order"])


def label_audit_summary() -> list[str]:
    lines: list[str] = []
    if not LABELS_CSV.exists():
        return ["Manual calibration labels unavailable: training_labels_v3.csv not found."]
    labels = pd.read_csv(LABELS_CSV)
    lines.append(f"training_labels_v3.csv rows={len(labels)}")
    if "training_label_v3" in labels.columns:
        lines.append("Manual label buckets: " + ", ".join(f"{k}={v}" for k, v in labels["training_label_v3"].value_counts().sort_index().items()))
    if "source_batch" in labels.columns:
        stage_i = labels[labels["source_batch"].astype(str).str.contains("stage_i", case=False, na=False)]
        stage_f = labels[labels["source_batch"].astype(str).str.contains("stage_f|batch", case=False, regex=True, na=False)]
        lines.append(f"Stage I pilot/calibration examples in labels: {len(stage_i)}")
        lines.append(f"Stage F/manual-review-like examples in labels: {len(stage_f)}")
        if len(stage_i):
            status_counts = stage_i.get("final_candidate_status", pd.Series(dtype=str)).value_counts()
            lines.append("Stage I pilot outcomes: " + ", ".join(f"{k}={v}" for k, v in status_counts.items()))
    for path in [
        ROOT / "freezes" / "stage_h_candidate_followup_decisions.csv",
        ROOT / "freezes" / "final_candidate_master_ledger_stage_h_candidate_followup.csv",
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                lines.append(f"{rel(path)} audit rows={len(df)}")
            except Exception as exc:
                lines.append(f"{rel(path)} audit read failed: {exc}")
    return lines


def write_policy_note(scored: pd.DataFrame, review: pd.DataFrame) -> None:
    counts = scored["autovet_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    raw_cols = set(pd.read_csv(INPUT_CSV, nrows=0).columns)
    direct_optional = [col for col in OPTIONAL_METRICS if col in raw_cols]
    missing_optional = [
        col
        for col in OPTIONAL_METRICS
        if col not in raw_cols and col not in {"primary_depth_snr", "period_support_count", "event_family_count", "best_period_days"}
    ]
    auto_deprioritized = int(scored["autovet_label"].isin(["auto_reject_likely_eb_or_artifact", "auto_reject_noise_or_artifact"]).sum())
    lines = [
        "Stage I AutoVet v1 policy note",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Scope",
        f"- Input: {rel(INPUT_CSV)}",
        f"- Active frozen model: {ACTIVE_MODEL}",
        "- The Keras model is loaded for inference only; this script does not train, fine-tune, relabel, or update any ledger.",
        "- Output labels are rule-based queue/vetting decisions, not manual labels.",
        "",
        "Outputs",
        f"- Results: {rel(OUT_RESULTS)} rows={len(scored)}",
        f"- Review queue: {rel(OUT_REVIEW)} rows={len(review)}",
        f"- Auto-deprioritized rows: {auto_deprioritized}",
        "",
        "Count by autovet_label",
    ]
    lines.extend(f"- {label}: {count}" for label, count in counts.items())
    lines.extend(
        [
            "",
            "Inputs used",
            "- Fresh flux_p_science_like, flux_p_top3_mean, and flux_p_top10_mean are computed from the frozen split303 flux model over splits/infer_c5.",
            "- Stage I funnel diagnostics used directly: triage_score_global, triage_step_score, triage_whiteness_score, best_shape_score, best_depth_snr, n_events, n_periods_proposed, n_periods_validated, best_period, pilot_noise_like_flag, obvious_low_signal_flag.",
            "- Optional Stage F-style metrics are consumed when present in the input table; missing optional metrics are left as NA and cannot trigger their corresponding rules.",
            f"- Optional Stage F-style metrics present directly in the input: {', '.join(direct_optional) if direct_optional else 'none'}",
            "- Fallback aliases used for requested fields: best_period_days=best_period, period_support_count=n_periods_validated, event_family_count=n_events, primary_depth_snr=best_depth_snr.",
            f"- Optional Stage F-style metrics absent/all-NA in this run: {', '.join(missing_optional) if missing_optional else 'none'}",
            "",
            "Rule order",
            "- First, triage-unusable rows are deprioritized as noise/artifact until detector features can be recovered.",
            "- EB/artifact rules fire on deep/short-period, large radius-ratio, severe odd/even, strong-secondary, spike/outlier, or extreme-SNR-with-poor-shape evidence.",
            "- Noise/artifact rules fire on no period support plus weak flux/shape/SNR/event evidence, Stage I pilot noise-like patterns, or strong systematic indicators.",
            "- Holds preserve possible signals that need period search, alias checks, OOT checks, or manual review before promotion.",
            "- Candidate-with-caveat requires frozen-model support, adequate period/event support, acceptable odd/even/secondary/depth checks, and no fatal caveat.",
            "- High-priority candidates require strong frozen-model support, strong period/event support, clean odd/even/secondary/alias/OOT checks, plausible depth/duration, and no spike dominance.",
            "",
            "Review queue policy",
            "- Include all auto_high_priority_candidate and auto_candidate_with_caveat rows.",
            f"- Include only the top {REVIEW_HOLD_LIMIT} auto_hold_needs_review rows by review_priority_score.",
            "- Exclude auto_reject_likely_eb_or_artifact and auto_reject_noise_or_artifact rows.",
            "",
            "Manual calibration/audit examples referenced, not modified",
        ]
    )
    lines.extend(f"- {line}" for line in label_audit_summary())
    OUT_POLICY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(scored: pd.DataFrame, review: pd.DataFrame) -> None:
    print("Count by autovet_label")
    print(scored["autovet_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())
    print()
    print("Top 50 review queue")
    show_cols = [
        "epic_id",
        "autovet_label",
        "autovet_confidence",
        "review_priority_score",
        "flux_p_science_like",
        "n_periods_validated",
        "best_period_days",
        "best_shape_score",
        "best_depth_snr",
        "n_events",
        "recommended_next_action",
    ]
    print(review[show_cols].head(50).to_string(index=False))
    print()
    auto_deprioritized = int(scored["autovet_label"].isin(["auto_reject_likely_eb_or_artifact", "auto_reject_noise_or_artifact"]).sum())
    print(f"Rows auto-deprioritized: {auto_deprioritized}")
    print()
    print("Examples from each label bucket")
    example_cols = [
        "epic_id",
        "autovet_label",
        "autovet_confidence",
        "flux_p_science_like",
        "n_periods_validated",
        "best_shape_score",
        "best_depth_snr",
        "triggered_rules",
    ]
    examples = scored.sort_values(["autovet_label", "review_priority_score"], ascending=[True, False]).groupby("autovet_label").head(5)
    print(examples[example_cols].to_string(index=False))


def main() -> None:
    for path in [INPUT_CSV, MODEL_PATH, INFER_X, INFER_META]:
        if not path.exists():
            raise FileNotFoundError(path)

    base = pd.read_csv(INPUT_CSV)
    base["epic_id"] = base["epic_id"].astype(str)
    flux = load_flux_scores(set(base["epic_id"]))
    merged = base.merge(flux, on="epic_id", how="left", validate="one_to_one")
    if merged["flux_p_science_like"].isna().any():
        missing = merged.loc[merged["flux_p_science_like"].isna(), "epic_id"].head(10).tolist()
        raise ValueError(f"Missing frozen flux scores after merge; first={missing}")

    scored = score_rows(merged)
    for col in OUTPUT_COLUMNS:
        if col not in scored.columns:
            scored[col] = np.nan
    scored = scored.sort_values(["autovet_label", "review_priority_score", "autovet_rank_score"], ascending=[True, False, False])
    scored[OUTPUT_COLUMNS].to_csv(OUT_RESULTS, index=False)

    review = build_review_queue(scored)
    review[OUTPUT_COLUMNS].to_csv(OUT_REVIEW, index=False)
    write_policy_note(scored, review)
    print_report(scored, review)


if __name__ == "__main__":
    main()
