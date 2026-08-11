from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "manual_64_training_view.csv"
BATCH_ID = "manual_vetting_batch_64"

OUT_SUMMARY = ROOT / "manual_vetting_64_summary.txt"
OUT_COUNTS = ROOT / "manual_64_label_counts.csv"
OUT_DISAGREE = ROOT / "manual_64_model_disagreement_report.txt"
OUT_STAGE_G = ROOT / "stage_g_deeper_eval_queue_manual_64.csv"
OUT_AUTOVET = ROOT / "autovet_v2_manual_64_scores.csv"
OUT_AUTOVET_EVAL = ROOT / "autovet_v2_manual_64_evaluation.txt"
FREEZE_DIR = ROOT / "freezes" / BATCH_ID

EXPECTED_ROW_COUNT = 64
EXPECTED_FAMILY_COUNTS = {"negative": 44, "hold": 16, "positive": 4}
EXPECTED_TRAINING_USE = {"true": 48, "false": 16}
EXPECTED_STAGE_G_EPICS = {
    "EPIC_211915147",
    "EPIC_211497712",
    "EPIC_211953866",
    "EPIC_211357782",
}

UNCERTAIN_LABELS = {"uncertain_hold", "uncertain_hold_positive"}
POSITIVE_LABELS = {"candidate_like", "planet_like"}
NEGATIVE_LABELS = {
    "binary_system",
    "variable_or_possible_eb",
    "reject_as_noise_or_artifact",
    "noise_or_artifact",
    "low_priority_negative",
}
EB_VARIABLE_LABELS = {"binary_system", "variable_or_possible_eb"}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def fnum(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column)
    if not finite(value):
        return default
    return float(value)


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def assert_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise AssertionError(f"{name}: observed={observed!r}, expected={expected!r}")


def load_view() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df = pd.read_csv(INPUT, dtype=str).fillna("")
    assert_equal("row_count", len(df), EXPECTED_ROW_COUNT)
    assert_equal("unique_epics", df["epic_id"].nunique(), EXPECTED_ROW_COUNT)
    assert set(df["freeze_batch_id"]) == {BATCH_ID}, sorted(set(df["freeze_batch_id"]))

    family_counts = df["manual_label_family"].value_counts().to_dict()
    assert_equal("label_family_counts", family_counts, EXPECTED_FAMILY_COUNTS)

    training_counts = df["training_use"].str.lower().value_counts().to_dict()
    assert_equal("training_use_counts", training_counts, EXPECTED_TRAINING_USE)

    stage_g_epics = set(df.loc[df["stage_g_deeper_eval"].str.lower().eq("true"), "epic_id"])
    assert_equal("stage_g_epics", stage_g_epics, EXPECTED_STAGE_G_EPICS)

    bad_holds = df[df["manual_label"].isin(UNCERTAIN_LABELS) & df["training_use"].str.lower().ne("false")]
    if len(bad_holds):
        raise AssertionError(f"uncertain holds unexpectedly included in training: {bad_holds['epic_id'].tolist()}")

    bad_nonholds = df[~df["manual_label"].isin(UNCERTAIN_LABELS) & df["training_use"].str.lower().ne("true")]
    if len(bad_nonholds):
        raise AssertionError(f"non-hold rows unexpectedly excluded from training: {bad_nonholds['epic_id'].tolist()}")

    bad_eb = df[df["manual_label"].isin(EB_VARIABLE_LABELS) & df["training_class"].ne("false_positive_eb_or_variable")]
    if len(bad_eb):
        raise AssertionError(f"EB/variable labels not mapped to false-positive negative class: {bad_eb['epic_id'].tolist()}")

    expected_negative_classes = {
        "reject_as_noise_or_artifact": "negative_reject_as_noise_or_artifact",
        "noise_or_artifact": "negative_noise_or_artifact",
        "low_priority_negative": "negative_low_priority",
    }
    for manual_label, training_class in expected_negative_classes.items():
        bad = df[df["manual_label"].eq(manual_label) & df["training_class"].ne(training_class)]
        if len(bad):
            raise AssertionError(f"{manual_label} not kept separate as {training_class}: {bad['epic_id'].tolist()}")

    return df


def autovet_v2_score(row: pd.Series) -> tuple[float, str, str, str]:
    cnn = clip01(fnum(row, "cnn_score"))
    shape = clip01((fnum(row, "best_shape_score") - 0.60) / 0.25)
    snr_source = row.get("primary_depth_snr") if finite(row.get("primary_depth_snr")) else row.get("best_depth_snr")
    snr = clip01(np.log1p(fnum(pd.Series({"x": snr_source}), "x")) / np.log1p(30.0))
    events = clip01(fnum(row, "event_family_count") / 12.0)
    period = clip01(fnum(row, "period_support_count") / 3.0)

    odd_ratio = row.get("odd_even_depth_ratio")
    odd_delta = row.get("odd_even_depth_delta_explicit")
    odd_clean = 0.5
    if finite(odd_delta):
        odd_clean = clip01(1.0 - fnum(row, "odd_even_depth_delta_explicit") / 0.35)
    elif finite(odd_ratio):
        odd_clean = clip01(1.0 - abs(fnum(row, "odd_even_depth_ratio") - 1.0) / 0.45)

    secondary_clean = 1.0
    if finite(row.get("secondary_to_primary_depth_ratio")):
        secondary_clean = min(secondary_clean, clip01(1.0 - fnum(row, "secondary_to_primary_depth_ratio") / 0.35))
    if finite(row.get("secondary_depth_snr")):
        secondary_clean = min(secondary_clean, clip01(1.0 - fnum(row, "secondary_depth_snr") / 7.0))

    oot_clean = clip01(1.0 - fnum(row, "oot_variability_to_depth") / 0.75) if finite(row.get("oot_variability_to_depth")) else 0.5
    alias_clean = clip01(1.0 - fnum(row, "alias_best_support_ratio") / 0.75) if finite(row.get("alias_best_support_ratio")) else 0.5
    duration_ok = 1.0
    if finite(row.get("transit_duration_hours")):
        duration = fnum(row, "transit_duration_hours")
        duration_ok = 1.0 if 0.5 <= duration <= 10.0 else 0.0

    score = (
        0.28 * cnn
        + 0.14 * period
        + 0.10 * events
        + 0.10 * shape
        + 0.10 * snr
        + 0.10 * odd_clean
        + 0.07 * secondary_clean
        + 0.06 * oot_clean
        + 0.03 * alias_clean
        + 0.02 * duration_ok
    )

    context = " ".join(
        clean(row.get(column)).lower()
        for column in [
            "manual_reason",
            "autovet_reason",
            "stage_f_reason",
            "triggered_rules",
            "diagnostic_gate_summary",
            "prefilter_reason",
            "recommended_prefilter_action",
        ]
    )
    flags: list[str] = []
    if finite(row.get("primary_depth")) and fnum(row, "primary_depth") >= 0.02:
        flags.append("deep_primary")
    if finite(row.get("radius_ratio_sqrt_depth")) and fnum(row, "radius_ratio_sqrt_depth") >= 0.15:
        flags.append("large_radius_ratio")
    if "event_spacing_fallback" in context or "no_saved_period_support" in context:
        flags.append("fallback_period_context")
    if "likely eb" in context or "binary" in context:
        flags.append("eb_or_binary_context")
    if "stage f reject" in context or "artifact/reject" in context:
        flags.append("validation_reject_context")
    if finite(row.get("alias_best_support_ratio")) and fnum(row, "alias_best_support_ratio") >= 0.75:
        flags.append("strong_alias_support")
    if finite(row.get("oot_variability_to_depth")) and fnum(row, "oot_variability_to_depth") >= 0.75:
        flags.append("high_oot_variability")

    penalty = 0.0
    if "deep_primary" in flags or "large_radius_ratio" in flags:
        penalty += 0.10
    if "fallback_period_context" in flags:
        penalty += 0.06
    if "eb_or_binary_context" in flags:
        penalty += 0.14
    if "validation_reject_context" in flags:
        penalty += 0.18
    if "strong_alias_support" in flags:
        penalty += 0.08
    if "high_oot_variability" in flags:
        penalty += 0.08

    score = clip01(score - penalty)
    no_period_but_strong_signal = period < 0.34 and cnn >= 0.65 and events >= 0.75 and shape >= 0.60 and snr >= 0.45

    if "eb_or_binary_context" in flags and score < 0.62:
        label = "autovet_v2_false_positive_eb_or_variable"
    elif "validation_reject_context" in flags or score < 0.42 or (period < 0.34 and not no_period_but_strong_signal):
        label = "autovet_v2_reject_noise_or_artifact"
    elif score >= 0.72 and not flags:
        label = "autovet_v2_candidate_priority"
    elif score >= 0.58 or no_period_but_strong_signal:
        label = "autovet_v2_review_signal_with_caveats"
    else:
        label = "autovet_v2_hold_for_review"

    components = (
        f"cnn={cnn:.3f}; period={period:.3f}; events={events:.3f}; shape={shape:.3f}; "
        f"snr={snr:.3f}; odd_even_clean={odd_clean:.3f}; secondary_clean={secondary_clean:.3f}; "
        f"oot_clean={oot_clean:.3f}; alias_clean={alias_clean:.3f}; duration_ok={duration_ok:.3f}; penalty={penalty:.3f}"
    )
    return round(score, 6), label, "; ".join(flags) if flags else "no_major_penalty_flags", components


def build_autovet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scored = [autovet_v2_score(row) for _, row in out.iterrows()]
    out["autovet_v2_score"] = [item[0] for item in scored]
    out["autovet_v2_label"] = [item[1] for item in scored]
    out["autovet_v2_flags"] = [item[2] for item in scored]
    out["autovet_v2_components"] = [item[3] for item in scored]
    out["autovet_v2_positive_call"] = out["autovet_v2_label"].eq("autovet_v2_candidate_priority").map(lambda v: str(v).lower())
    out["autovet_v2_review_call"] = out["autovet_v2_label"].isin(
        {"autovet_v2_candidate_priority", "autovet_v2_review_signal_with_caveats", "autovet_v2_hold_for_review"}
    ).map(lambda v: str(v).lower())
    out["manual_positive_for_eval"] = out["manual_label"].isin(POSITIVE_LABELS).map(lambda v: str(v).lower())
    out["manual_hold_for_eval"] = out["manual_label"].isin(UNCERTAIN_LABELS).map(lambda v: str(v).lower())
    return out


def write_counts(df: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for count_type, column in [
        ("manual_label_family", "manual_label_family"),
        ("manual_label", "manual_label"),
        ("training_use", "training_use"),
        ("training_class", "training_class"),
    ]:
        for label, count in sorted(Counter(df[column].str.lower() if column == "training_use" else df[column]).items()):
            rows.append({"count_type": count_type, "label": label, "count": count})
    pd.DataFrame(rows).to_csv(OUT_COUNTS, index=False)
    pd.DataFrame(rows).to_csv(FREEZE_DIR / OUT_COUNTS.name, index=False)


def parsed_good_odd_even(row: pd.Series) -> bool:
    reason = clean(row.get("manual_reason")).lower()
    ratio = fnum(row, "odd_even_depth_ratio", np.nan)
    delta = fnum(row, "odd_even_depth_delta_explicit", np.nan)
    ratio_match = re.search(r"odd/even[^.;]*(?:ratio|agreement)[^0-9]*(0\.\d+|1\.\d+)", reason)
    delta_match = re.search(r"(?:explicit delta|delta)[^0-9]*(0\.\d+)", reason)
    if not finite(ratio) and ratio_match:
        ratio = float(ratio_match.group(1))
    if not finite(delta) and delta_match:
        delta = float(delta_match.group(1))
    phrase_good = any(
        phrase in reason
        for phrase in [
            "odd/even agreement is excellent",
            "odd/even excellent",
            "odd/even is good",
            "odd/even good",
            "odd/even fairly good",
            "odd/even is acceptable",
            "odd/even acceptable",
            "odd/even consistency is excellent",
            "odd/even consistency is acceptable",
        ]
    )
    return phrase_good or (finite(ratio) and 0.85 <= ratio <= 1.15) or (finite(delta) and delta <= 0.15)


def disagreement_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        label = clean(row["manual_label"])
        reason = clean(row["manual_reason"]).lower()
        cnn = fnum(row, "cnn_score", np.nan)
        snr = fnum(row, "primary_depth_snr", np.nan) if finite(row.get("primary_depth_snr")) else fnum(row, "best_depth_snr", np.nan)
        categories: list[str] = []
        if label in NEGATIVE_LABELS and finite(cnn) and cnn >= 0.85:
            categories.append("high_cnn_score_manual_reject")
        if label in POSITIVE_LABELS | UNCERTAIN_LABELS and finite(cnn) and cnn <= 0.70:
            categories.append("low_cnn_score_manual_candidate_or_hold")
        if label in NEGATIVE_LABELS and finite(snr) and snr >= 50:
            categories.append("high_snr_manual_reject")
        if label in NEGATIVE_LABELS and parsed_good_odd_even(row):
            categories.append("good_odd_even_manual_reject")
        if label in NEGATIVE_LABELS and (
            "event_spacing_fallback" in reason or "fallback period" in reason or "candidate period count" in reason
        ):
            categories.append("fallback_period_dominated_reject")
        if label in EB_VARIABLE_LABELS:
            categories.append("eb_variable_false_positive")
        for category in categories:
            rows.append(
                {
                    "category": category,
                    "epic_id": clean(row["epic_id"]),
                    "manual_label": label,
                    "training_class": clean(row["training_class"]),
                    "cnn_score": clean(row.get("cnn_score")),
                    "primary_depth_snr": "" if not finite(snr) else f"{snr:.12g}",
                    "manual_reason": clean(row["manual_reason"]),
                }
            )
    return rows


def write_disagreement(df: pd.DataFrame) -> Counter:
    rows = disagreement_rows(df)
    counts = Counter(row["category"] for row in rows)
    lines = [
        "Manual 64 model disagreement report",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"source={INPUT.relative_to(ROOT).as_posix()}",
        "",
    ]
    for category in [
        "high_cnn_score_manual_reject",
        "low_cnn_score_manual_candidate_or_hold",
        "high_snr_manual_reject",
        "good_odd_even_manual_reject",
        "fallback_period_dominated_reject",
        "eb_variable_false_positive",
    ]:
        group = [row for row in rows if row["category"] == category]
        lines.append(category)
        lines.append(f"count={len(group)}")
        for row in sorted(group, key=lambda item: item["epic_id"]):
            lines.append(
                f"- {row['epic_id']}: label={row['manual_label']}; cnn={row['cnn_score']}; snr={row['primary_depth_snr']}"
            )
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    OUT_DISAGREE.write_text(text, encoding="utf-8")
    (FREEZE_DIR / OUT_DISAGREE.name).write_text(text, encoding="utf-8")
    return counts


def write_stage_g(df: pd.DataFrame) -> None:
    queue = df[df["stage_g_deeper_eval"].str.lower().eq("true")].copy()
    assert_equal("stage_g_queue_epics", set(queue["epic_id"]), EXPECTED_STAGE_G_EPICS)
    queue.to_csv(OUT_STAGE_G, index=False)
    queue.to_csv(FREEZE_DIR / OUT_STAGE_G.name, index=False)


def write_autovet_eval(scored: pd.DataFrame) -> Counter:
    scored.to_csv(OUT_AUTOVET, index=False)
    scored.to_csv(FREEZE_DIR / OUT_AUTOVET.name, index=False)

    eval_rows = scored[scored["manual_hold_for_eval"].eq("false")]
    pred_pos = eval_rows["autovet_v2_positive_call"].eq("true")
    actual_pos = eval_rows["manual_positive_for_eval"].eq("true")
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    review_truth = scored["manual_label"].isin(POSITIVE_LABELS | UNCERTAIN_LABELS)
    review_call = scored["autovet_v2_review_call"].eq("true")
    review_capture = int((review_truth & review_call).sum())
    review_total = int(review_truth.sum())
    review_capture_rate = review_capture / review_total if review_total else 0.0
    counts = Counter(scored["autovet_v2_label"])

    lines = [
        "AutoVet v2 manual-64 deterministic evaluation",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"source={INPUT.relative_to(ROOT).as_posix()}",
        "training=not_run",
        "cnn_role=frozen transit_morphology_score only",
        "",
        "Confusion matrix, holds excluded",
        f"true_positive={tp}",
        f"false_positive={fp}",
        f"false_negative={fn}",
        f"true_negative={tn}",
        f"precision={precision:.3f}",
        f"recall={recall:.3f}",
        f"candidate_or_hold_review_capture={review_capture}/{review_total}",
        f"candidate_or_hold_review_capture_rate={review_capture_rate:.3f}",
        "",
        "Label counts",
    ]
    lines.extend(f"- {label}: {count}" for label, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "Policy",
            "- No CNN retraining was run.",
            "- `autovet_v2_candidate_priority` is the only automatic positive prediction label.",
            "- `autovet_v2_review_signal_with_caveats` is review routing, not promotion authority.",
        ]
    )
    text = "\n".join(lines) + "\n"
    OUT_AUTOVET_EVAL.write_text(text, encoding="utf-8")
    (FREEZE_DIR / OUT_AUTOVET_EVAL.name).write_text(text, encoding="utf-8")
    return counts


def write_summary(df: pd.DataFrame, autovet_counts: Counter, disagreement_counts: Counter) -> None:
    family_counts = Counter(df["manual_label_family"])
    label_counts = Counter(df["manual_label"])
    training_counts = Counter(df["training_use"].str.lower())
    training_class_counts = Counter(df["training_class"])
    lines = [
        "Manual vetting batch 64 summary",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"freeze_batch_id={BATCH_ID}",
        f"source={INPUT.relative_to(ROOT).as_posix()}",
        f"row_count={len(df)}",
        f"unique_epics={df['epic_id'].nunique()}",
        "",
        "Verified label-family counts",
    ]
    lines.extend(f"- {label}: {family_counts[label]}" for label in ["negative", "hold", "positive"])
    lines.extend(["", "Manual label counts"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    lines.extend(["", "Training use counts"])
    lines.extend(f"- {label}: {training_counts[label]}" for label in ["true", "false"])
    lines.extend(["", "Training class counts"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(training_class_counts.items()))
    lines.extend(
        [
            "",
            "Stage G deeper evaluation EPICs",
            *[f"- {epic}" for epic in sorted(EXPECTED_STAGE_G_EPICS)],
            "",
            "Policy confirmations",
            "- uncertain_hold and uncertain_hold_positive are excluded from training.",
            "- binary_system and variable_or_possible_eb remain false_positive_eb_or_variable negatives.",
            "- reject_as_noise_or_artifact, noise_or_artifact, and low_priority_negative remain separate negative classes.",
            "- CNN was not retrained.",
            "- AutoVet v2 is a deterministic tabular/rules layer over frozen CNN score plus validation metrics.",
            "",
            "AutoVet v2 label counts",
        ]
    )
    lines.extend(f"- {label}: {count}" for label, count in sorted(autovet_counts.items()))
    lines.extend(["", "Model disagreement counts"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(disagreement_counts.items()))
    lines.extend(
        [
            "",
            "Outputs",
            f"- {OUT_SUMMARY.relative_to(ROOT).as_posix()}",
            f"- {OUT_COUNTS.relative_to(ROOT).as_posix()}",
            f"- {OUT_DISAGREE.relative_to(ROOT).as_posix()}",
            f"- {OUT_STAGE_G.relative_to(ROOT).as_posix()}",
            f"- {OUT_AUTOVET.relative_to(ROOT).as_posix()}",
            f"- {OUT_AUTOVET_EVAL.relative_to(ROOT).as_posix()}",
        ]
    )
    text = "\n".join(lines) + "\n"
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    (FREEZE_DIR / OUT_SUMMARY.name).write_text(text, encoding="utf-8")


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_view()
    write_counts(df)
    write_stage_g(df)
    disagreement_counts = write_disagreement(df)
    scored = build_autovet(df)
    autovet_counts = write_autovet_eval(scored)
    write_summary(df, autovet_counts, disagreement_counts)
    print(
        {
            "row_count": len(df),
            "label_family_counts": dict(Counter(df["manual_label_family"])),
            "training_use_counts": dict(Counter(df["training_use"].str.lower())),
            "stage_g_epics": sorted(EXPECTED_STAGE_G_EPICS),
            "autovet_v2_counts": dict(sorted(autovet_counts.items())),
            "did_not_retrain_cnn": True,
        }
    )


if __name__ == "__main__":
    main()
