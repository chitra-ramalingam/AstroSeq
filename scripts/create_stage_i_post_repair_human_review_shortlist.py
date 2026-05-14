from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"
INPUT_LEDGER = BATCH_DIR / "post_repair_validation_ledger.csv"
OUT_SHORTLIST = BATCH_DIR / "post_repair_human_review_shortlist.csv"
OUT_SUMMARY = BATCH_DIR / "post_repair_human_review_shortlist_summary.txt"

OUTPUT_COLUMNS = [
    "epic_id",
    "review_tier",
    "manual_priority",
    "post_repair_stage_f_label",
    "best_period_days",
    "period_support_count",
    "post_repair_primary_depth_snr",
    "post_repair_primary_depth",
    "post_repair_odd_even_depth_ratio",
    "post_repair_secondary_to_primary_depth_ratio",
    "post_repair_alias_risk",
    "post_repair_eb_risk_flag",
    "post_repair_artifact_risk_flag",
    "oot_variability_to_depth",
    "promotion_safe",
    "training_safe",
    "rejection_safe",
    "human_review_required",
    "label_update_allowed",
    "recommended_next_action",
    "review_reason",
    "post_repair_phase_0_folded_path",
    "post_repair_phase_05_secondary_check_path",
    "post_repair_odd_even_zoom_path",
    "post_repair_alias_period_comparison_path",
]

TIER_ORDER = {
    "tier_1_promising_hold": 0,
    "tier_2_caution_hold": 1,
    "tier_3_metric_interesting": 2,
}

MANUAL_PRIORITY_ORDER = {"high": 0, "medium_caution": 1, "normal": 2}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def clean_hold(row: pd.Series) -> bool:
    return (
        str(row.get("post_repair_stage_f_label", "")) == "stage_f_hold"
        and str(row.get("post_repair_alias_risk", "")).lower() == "low"
        and not as_bool(row.get("post_repair_eb_risk_flag"))
        and not as_bool(row.get("post_repair_artifact_risk_flag"))
    )


def strong_metric(row: pd.Series) -> bool:
    return (
        as_float(row.get("period_support_count")) >= 10
        and as_float(row.get("post_repair_primary_depth_snr")) >= 5
    )


def caution_hold(row: pd.Series) -> bool:
    return (
        str(row.get("post_repair_stage_f_label", "")) != "stage_f_reject"
        and (
            str(row.get("post_repair_alias_risk", "")).lower() in {"moderate", "high"}
            or as_bool(row.get("post_repair_eb_risk_flag"))
            or as_float(row.get("post_repair_secondary_to_primary_depth_ratio")) >= 0.25
            or as_float(row.get("post_repair_odd_even_depth_ratio")) < 0.55
        )
    )


def select_row(row: pd.Series) -> bool:
    manual_priority = str(row.get("manual_priority", "normal"))
    return (
        manual_priority in {"high", "medium_caution"}
        or clean_hold(row)
        or strong_metric(row)
    )


def tier_for(row: pd.Series) -> str:
    manual_priority = str(row.get("manual_priority", "normal"))
    if manual_priority == "medium_caution" or caution_hold(row):
        return "tier_2_caution_hold"
    if manual_priority == "high" or (clean_hold(row) and strong_metric(row)):
        return "tier_1_promising_hold"
    return "tier_3_metric_interesting"


def review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    manual_priority = str(row.get("manual_priority", "normal"))
    if manual_priority in {"high", "medium_caution"}:
        reasons.append(f"manual_priority={manual_priority}")
    if clean_hold(row):
        reasons.append("clean post-repair hold with low alias risk and no EB/artifact flags")
    if strong_metric(row):
        reasons.append(
            "period_support_count>=10 and post_repair_primary_depth_snr>=5"
        )
    if caution_hold(row):
        cautions: list[str] = []
        alias_risk = str(row.get("post_repair_alias_risk", "")).lower()
        if alias_risk in {"moderate", "high"}:
            cautions.append(f"alias_risk={alias_risk}")
        if as_bool(row.get("post_repair_eb_risk_flag")):
            cautions.append("post_repair_eb_risk_flag=True")
        secondary_ratio = as_float(row.get("post_repair_secondary_to_primary_depth_ratio"))
        if secondary_ratio >= 0.25:
            cautions.append(f"secondary_ratio={secondary_ratio:.4f}")
        odd_even_ratio = as_float(row.get("post_repair_odd_even_depth_ratio"))
        if odd_even_ratio < 0.55:
            cautions.append(f"odd_even_ratio={odd_even_ratio:.4f}")
        if cautions:
            reasons.append("caution: " + ", ".join(cautions))
    oot_to_depth = as_float(row.get("oot_variability_to_depth"))
    if oot_to_depth >= 5:
        reasons.append(f"OOT/depth caution={oot_to_depth:.3f}")
    if not reasons:
        reasons.append("metric-interesting row selected by shortlist rules")
    reasons.append("shortlist only; human review required before any label update")
    return "; ".join(reasons)


def main() -> None:
    if not INPUT_LEDGER.exists():
        raise FileNotFoundError(f"Missing post-repair validation ledger: {INPUT_LEDGER}")

    ledger = pd.read_csv(INPUT_LEDGER)
    for col in OUTPUT_COLUMNS:
        if col not in ledger.columns and col != "review_tier" and col != "review_reason":
            ledger[col] = pd.NA

    shortlist = ledger.loc[ledger.apply(select_row, axis=1)].copy()
    shortlist["review_tier"] = shortlist.apply(tier_for, axis=1)
    shortlist["review_reason"] = shortlist.apply(review_reason, axis=1)

    if not shortlist["human_review_required"].map(as_bool).all():
        raise ValueError("Shortlist contains rows without human_review_required=True")
    if shortlist["label_update_allowed"].map(as_bool).any():
        raise ValueError("Shortlist contains rows with label_update_allowed=True")

    shortlist["_manual_priority_order"] = shortlist["manual_priority"].map(
        MANUAL_PRIORITY_ORDER
    ).fillna(MANUAL_PRIORITY_ORDER["normal"])
    shortlist["_tier_order"] = shortlist["review_tier"].map(TIER_ORDER)
    shortlist["_oot_variability_to_depth"] = shortlist["oot_variability_to_depth"].map(
        as_float
    )
    shortlist["_period_support_count"] = shortlist["period_support_count"].map(as_float)
    shortlist["_post_repair_primary_depth_snr"] = shortlist[
        "post_repair_primary_depth_snr"
    ].map(as_float)
    shortlist = shortlist.sort_values(
        [
            "_manual_priority_order",
            "_tier_order",
            "_oot_variability_to_depth",
            "_period_support_count",
            "_post_repair_primary_depth_snr",
            "epic_id",
        ],
        ascending=[True, True, True, False, False, True],
    )
    shortlist = shortlist[OUTPUT_COLUMNS]
    shortlist.to_csv(OUT_SHORTLIST, index=False)

    tier_1 = shortlist.loc[
        shortlist["review_tier"].eq("tier_1_promising_hold"), "epic_id"
    ].tolist()
    tier_2 = shortlist.loc[
        shortlist["review_tier"].eq("tier_2_caution_hold"), "epic_id"
    ].tolist()
    tier_3 = shortlist.loc[
        shortlist["review_tier"].eq("tier_3_metric_interesting"), "epic_id"
    ].tolist()

    lines = [
        "Stage I post-repair human review shortlist summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input post-repair ledger: {rel(INPUT_LEDGER)}",
        f"Output shortlist: {rel(OUT_SHORTLIST)}",
        "",
        f"Total post-repair rows: {len(ledger)}",
        f"Shortlist rows: {len(shortlist)}",
        f"tier_1 count: {len(tier_1)}",
        f"tier_2 count: {len(tier_2)}",
        f"tier_3 count: {len(tier_3)}",
        f"tier_1 EPICs: {', '.join(tier_1) if tier_1 else 'none'}",
        f"tier_2 EPICs: {', '.join(tier_2) if tier_2 else 'none'}",
        f"tier_3 EPICs: {', '.join(tier_3) if tier_3 else 'none'}",
        "",
        "This is a human review shortlist only.",
        "No rows were auto-promoted, auto-rejected, or auto-trained.",
        "Safety flags were carried through from the post-repair ledger and not changed by this shortlist.",
        "No labels, final candidate ledgers, models, model artifacts, or training files were modified.",
        "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Post-repair human review shortlist: {rel(OUT_SHORTLIST)}")
    print(f"Post-repair human review shortlist summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(shortlist.to_string(index=False))


if __name__ == "__main__":
    main()
