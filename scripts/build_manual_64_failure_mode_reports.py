from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_VIEW = ROOT / "manual_64_training_view.csv"
AUTOVET_V2 = ROOT / "autovet_v2_manual_64_scores.csv"

OUT_TXT = ROOT / "manual_64_model_disagreement_report.txt"
OUT_FAILURE = ROOT / "manual_64_failure_modes.csv"
OUT_PROBE = ROOT / "manual_64_autovet_v2_feature_probe.csv"
FREEZE_DIR = ROOT / "freezes" / "manual_vetting_batch_64"

TABLE_COLUMNS = [
    "epic_id",
    "manual_label",
    "cnn_score",
    "primary_depth_snr",
    "odd_even_depth_ratio",
    "oot_to_depth",
    "candidate_period_count",
    "alias_risk",
    "fallback_period_flag",
    "duration_fraction_of_period",
    "failure_mode",
    "autovet_v2_penalty",
]

NEGATIVE_LABELS = {
    "binary_system",
    "variable_or_possible_eb",
    "reject_as_noise_or_artifact",
    "noise_or_artifact",
    "low_priority_negative",
}
HOLD_LABELS = {"uncertain_hold", "uncertain_hold_positive"}
POSITIVE_LABELS = {"candidate_like", "planet_like"}
EB_VARIABLE_LABELS = {"binary_system", "variable_or_possible_eb"}
NOISE_LABELS = {"reject_as_noise_or_artifact", "noise_or_artifact", "low_priority_negative"}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def as_float(value: Any) -> float | None:
    if not finite(value):
        return None
    return float(value)


def fmt(value: Any) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.6g}"


def find_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).rstrip(".")
            if raw:
                return float(raw)
    return None


def odd_even_ratio(row: pd.Series) -> float | None:
    value = as_float(row.get("odd_even_depth_ratio"))
    if value is not None:
        return value
    text = clean(row.get("manual_reason"))
    return find_float(
        [
            r"odd/even[^.;]{0,140}ratio[^0-9]*(0\.\d+|1\.\d+)",
        ],
        text,
    )


def oot_to_depth(row: pd.Series) -> float | None:
    value = as_float(row.get("oot_variability_to_depth"))
    if value is not None:
        return value
    text = clean(row.get("manual_reason"))
    return find_float(
        [
            r"oot_variability_to_depth\s*=\s*([0-9.]+)",
            r"oot/depth\s*=\s*([0-9.]+)",
            r"oot variability[^.;]*?([0-9.]+)\s*x",
            r"oot variability[^.;]*?([0-9.]+)x",
            r"oot behaviour[^.;]*?([0-9.]+)\s*x",
        ],
        text,
    )


def candidate_period_count(row: pd.Series) -> float | None:
    text = clean(row.get("manual_reason"))
    value = find_float(
        [
            r"candidate[_ ]period[_ ]count\s*(?:is|=)\s*([0-9.]+)",
            r"candidate periods?[^0-9]*([0-9.]+)",
        ],
        text,
    )
    if value is not None:
        return value
    return as_float(row.get("n_periods_proposed"))


def alias_risk(row: pd.Series) -> str:
    text = clean(row.get("manual_reason")).lower()
    match = re.search(r"alias risk\s*(?:=|is)?\s*(low|moderate|high)", text)
    if match:
        return match.group(1)
    if "low alias risk" in text:
        return "low"
    if "moderate alias" in text:
        return "moderate"
    if "high alias" in text:
        return "high"
    gate = clean(row.get("alias_gate")).lower()
    if gate in {"pass", "fail", "review"}:
        return gate
    support = as_float(row.get("alias_best_support_ratio"))
    if support is None:
        return ""
    if support >= 0.8:
        return "high"
    if support >= 0.5:
        return "moderate"
    return "low"


def fallback_period_flag(row: pd.Series) -> bool:
    text = " ".join(
        clean(row.get(column)).lower()
        for column in ["manual_reason", "prefilter_reason", "recommended_prefilter_action", "triggered_rules", "autovet_reason"]
    )
    return any(
        token in text
        for token in [
            "event_spacing_fallback",
            "fallback period",
            "fallback-only",
            "fallback/ambiguous",
            "no_saved_period_support",
            "needs_period_search",
            "run_period_search",
        ]
    )


def duration_fraction_of_period(row: pd.Series) -> float | None:
    text = clean(row.get("manual_reason"))
    duration = as_float(row.get("transit_duration_hours"))
    period = as_float(row.get("best_period_days"))
    parsed_duration = find_float(
        [
            r"duration\s*(?:=|is|around|about)?\s*([0-9.]+)\s*h",
            r"duration[^.;]*?([0-9.]+)\s*hours",
        ],
        text,
    )
    parsed_period = find_float(
        [
            r"P\s*=\s*([0-9.]+)\s*d",
            r"P\s*=\s*([0-9.]+)d",
            r"period:\s*([0-9.]+)\s*d",
            r"period is very short at P=([0-9.]+)d",
        ],
        text,
    )
    if parsed_duration is not None:
        duration = parsed_duration
    if parsed_period is not None:
        period = parsed_period
    if duration is None or period is None or period <= 0:
        return None
    return duration / (period * 24.0)


def autovet_penalty(row: pd.Series) -> float | None:
    components = clean(row.get("autovet_v2_components"))
    match = re.search(r"penalty=([0-9.]+)", components)
    if match:
        return float(match.group(1))
    return None


def has_good_odd_even(row: pd.Series) -> bool:
    ratio = odd_even_ratio(row)
    text = clean(row.get("manual_reason")).lower()
    return (
        (ratio is not None and 0.85 <= ratio <= 1.15)
        or "odd/even agreement is excellent" in text
        or "odd/even excellent" in text
        or "odd/even is good" in text
        or "odd/even good" in text
        or "odd/even fairly good" in text
        or "odd/even is acceptable" in text
        or "odd/even acceptable" in text
        or "odd/even consistency is excellent" in text
        or "odd/even consistency is acceptable" in text
    )


def noise_signature(row: pd.Series) -> bool:
    text = clean(row.get("manual_reason")).lower()
    label = clean(row.get("manual_label"))
    return label in NOISE_LABELS or any(
        token in text
        for token in [
            "noise",
            "artifact",
            "not significant",
            "weak primary",
            "shallow",
            "oot variability is high",
            "period solution not trustworthy",
            "not visually convincing",
        ]
    )


def useful_rejection_feature(row: pd.Series) -> bool:
    if clean(row.get("manual_label")) not in NEGATIVE_LABELS:
        return False
    return (
        fallback_period_flag(row)
        or clean(row.get("manual_label")) in EB_VARIABLE_LABELS
        or (oot_to_depth(row) is not None and oot_to_depth(row) >= 0.35)
        or (duration_fraction_of_period(row) is not None and duration_fraction_of_period(row) >= 0.12)
        or alias_risk(row) in {"moderate", "high", "fail", "review"}
        or not has_good_odd_even(row)
    )


def feature_row(row: pd.Series) -> dict[str, Any]:
    snr = as_float(row.get("primary_depth_snr")) or as_float(row.get("best_depth_snr"))
    return {
        "epic_id": clean(row.get("epic_id")),
        "manual_label": clean(row.get("manual_label")),
        "manual_label_family": clean(row.get("manual_label_family")),
        "training_class": clean(row.get("training_class")),
        "training_use": clean(row.get("training_use")).lower(),
        "cnn_score": fmt(row.get("cnn_score")),
        "primary_depth_snr": fmt(snr),
        "odd_even_depth_ratio": fmt(odd_even_ratio(row)),
        "oot_to_depth": fmt(oot_to_depth(row)),
        "candidate_period_count": fmt(candidate_period_count(row)),
        "alias_risk": alias_risk(row),
        "fallback_period_flag": str(fallback_period_flag(row)).lower(),
        "duration_fraction_of_period": fmt(duration_fraction_of_period(row)),
        "autovet_v2_score": fmt(row.get("autovet_v2_score")),
        "autovet_v2_label": clean(row.get("autovet_v2_label")),
        "autovet_v2_flags": clean(row.get("autovet_v2_flags")),
        "autovet_v2_penalty": fmt(autovet_penalty(row)),
        "manual_reason": clean(row.get("manual_reason")),
    }


def section_memberships(row: pd.Series) -> list[tuple[str, str]]:
    label = clean(row.get("manual_label"))
    cnn = as_float(row.get("cnn_score")) or 0.0
    snr = as_float(row.get("primary_depth_snr")) or as_float(row.get("best_depth_snr")) or 0.0
    memberships: list[tuple[str, str]] = []
    if label in NEGATIVE_LABELS and cnn >= 0.85:
        memberships.append(("A. CNN high-confidence false positives", "cnn_high_confidence_false_positive"))
    if label in HOLD_LABELS and cnn >= 0.85:
        memberships.append(("B. CNN high-confidence holds", "cnn_high_confidence_hold"))
    if label in POSITIVE_LABELS and cnn < 0.75:
        memberships.append(("C. Low/moderate CNN positives", "low_moderate_cnn_positive"))
    if label in NEGATIVE_LABELS and snr >= 50:
        memberships.append(("D. High-SNR false positives", "high_snr_false_positive"))
    if label in NEGATIVE_LABELS and has_good_odd_even(row):
        memberships.append(("E. Good odd/even false positives", "good_odd_even_false_positive"))
    if label in EB_VARIABLE_LABELS:
        memberships.append(("F. EB/variable false positives", "eb_variable_false_positive"))
    if noise_signature(row):
        memberships.append(("G. Noise/artifact signatures", "noise_artifact_signature"))
    if useful_rejection_feature(row):
        memberships.append(("H. Most useful rejection features", "useful_rejection_feature"))
    return memberships


def load() -> pd.DataFrame:
    if not MANUAL_VIEW.exists():
        raise FileNotFoundError(MANUAL_VIEW)
    manual = pd.read_csv(MANUAL_VIEW, dtype=str).fillna("")
    if len(manual) != 64:
        raise AssertionError(f"manual_64_training_view.csv row count is {len(manual)}, expected 64")
    if AUTOVET_V2.exists():
        auto = pd.read_csv(AUTOVET_V2, dtype=str).fillna("")
        cols = ["epic_id", "autovet_v2_score", "autovet_v2_label", "autovet_v2_flags", "autovet_v2_components"]
        auto = auto[[col for col in cols if col in auto.columns]].drop_duplicates("epic_id", keep="last")
        manual = manual.drop(columns=[col for col in auto.columns if col != "epic_id" and col in manual.columns], errors="ignore")
        manual = manual.merge(auto, on="epic_id", how="left", validate="one_to_one")
    return manual


def table_line(row: dict[str, Any]) -> str:
    return " | ".join(clean(row.get(column)) for column in TABLE_COLUMNS)


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    probe = pd.DataFrame([feature_row(row) for _, row in df.iterrows()])
    probe.to_csv(OUT_PROBE, index=False)
    probe.to_csv(FREEZE_DIR / OUT_PROBE.name, index=False)

    failure_rows: list[dict[str, Any]] = []
    section_rows: dict[str, list[dict[str, Any]]] = {}
    probe_by_epic = {row["epic_id"]: row for row in probe.to_dict("records")}
    for _, row in df.iterrows():
        epic_id = clean(row.get("epic_id"))
        for section, failure_mode in section_memberships(row):
            out = dict(probe_by_epic[epic_id])
            out["section"] = section
            out["failure_mode"] = failure_mode
            failure_rows.append(out)
            section_rows.setdefault(section, []).append(out)

    failure = pd.DataFrame(failure_rows)
    ordered_cols = ["section", *TABLE_COLUMNS, "autovet_v2_score", "autovet_v2_label", "autovet_v2_flags", "manual_reason"]
    failure = failure[[col for col in ordered_cols if col in failure.columns]].sort_values(["section", "epic_id"])
    failure.to_csv(OUT_FAILURE, index=False)
    failure.to_csv(FREEZE_DIR / OUT_FAILURE.name, index=False)

    lines = [
        "Manual 64 model disagreement / failure-mode report",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "source=manual_64_training_view.csv",
        "",
        "columns=" + ", ".join(TABLE_COLUMNS),
    ]
    for section in [
        "A. CNN high-confidence false positives",
        "B. CNN high-confidence holds",
        "C. Low/moderate CNN positives",
        "D. High-SNR false positives",
        "E. Good odd/even false positives",
        "F. EB/variable false positives",
        "G. Noise/artifact signatures",
        "H. Most useful rejection features",
    ]:
        rows = sorted(section_rows.get(section, []), key=lambda item: item["epic_id"])
        lines.extend(["", section, f"count={len(rows)}", table_line({column: column for column in TABLE_COLUMNS})])
        lines.extend(table_line(row) for row in rows)

    counts = Counter(row["section"] for row in failure_rows)
    lines.extend(["", "section_counts"])
    lines.extend(f"{section}={counts.get(section, 0)}" for section in [
        "A. CNN high-confidence false positives",
        "B. CNN high-confidence holds",
        "C. Low/moderate CNN positives",
        "D. High-SNR false positives",
        "E. Good odd/even false positives",
        "F. EB/variable false positives",
        "G. Noise/artifact signatures",
        "H. Most useful rejection features",
    ])
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (FREEZE_DIR / OUT_TXT.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        {
            "probe_rows": len(probe),
            "failure_mode_rows": len(failure),
            "section_counts": dict(counts),
            "outputs": [OUT_TXT.name, OUT_FAILURE.name, OUT_PROBE.name],
        }
    )


if __name__ == "__main__":
    main()
