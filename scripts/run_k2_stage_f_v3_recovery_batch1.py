from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


NEEDS_STAGE_F_CSV = ROOT / "next_needs_stage_f_validation.csv"
SCORES_V3_CSV = ROOT / "k2_hybrid_candidate_score_v3.csv"

BATCH_NAME = "batch1"
BATCH_INPUT_OUT = ROOT / f"k2_stage_f_v3_recovery_{BATCH_NAME}_input.csv"
VALIDATION_OUT = ROOT / f"k2_stage_f_v3_recovery_{BATCH_NAME}_validation.csv"
VISUAL_REVIEW_OUT = ROOT / f"k2_stage_e_v3_recovery_{BATCH_NAME}_visual_review_sheet.csv"
PLOTS_OUT_DIR = ROOT / "plots" / "k2_batch" / f"stage_f_v3_recovery_{BATCH_NAME}"
SUMMARY_OUT_DIR = ROOT / "plots" / "k2_batch" / f"stage_f_v3_recovery_{BATCH_NAME}_input_summaries"


def configure_batch(batch_name: str) -> None:
    global BATCH_NAME, BATCH_INPUT_OUT, VALIDATION_OUT, VISUAL_REVIEW_OUT, PLOTS_OUT_DIR, SUMMARY_OUT_DIR
    BATCH_NAME = str(batch_name).strip()
    if not BATCH_NAME:
        raise ValueError("batch name must not be blank")
    BATCH_INPUT_OUT = ROOT / f"k2_stage_f_v3_recovery_{BATCH_NAME}_input.csv"
    VALIDATION_OUT = ROOT / f"k2_stage_f_v3_recovery_{BATCH_NAME}_validation.csv"
    VISUAL_REVIEW_OUT = ROOT / f"k2_stage_e_v3_recovery_{BATCH_NAME}_visual_review_sheet.csv"
    PLOTS_OUT_DIR = ROOT / "plots" / "k2_batch" / f"stage_f_v3_recovery_{BATCH_NAME}"
    SUMMARY_OUT_DIR = ROOT / "plots" / "k2_batch" / f"stage_f_v3_recovery_{BATCH_NAME}_input_summaries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage F V3 recovery batch from next_needs_stage_f_validation.csv.")
    parser.add_argument("--batch-name", default="batch1", help="Output batch suffix, e.g. batch2.")
    parser.add_argument("--limit", type=int, default=5, help="Number of unresolved rows to select.")
    return parser.parse_args()


def json_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def epic_digits(epic_id: str) -> str:
    return "".join(ch for ch in str(epic_id) if ch.isdigit())


def select_batch(limit: int = 5) -> pd.DataFrame:
    needs = pd.read_csv(NEEDS_STAGE_F_CSV)
    scores = pd.read_csv(SCORES_V3_CSV)
    needs = needs.loc[
        needs["is_unresolved_v3"].astype(str).eq("True")
        & needs["needs_stage_f_validation"].astype(str).eq("True")
    ].copy()
    needs["hybrid_v3_strict_score"] = pd.to_numeric(needs["hybrid_v3_strict_score"], errors="coerce")
    selected = needs.sort_values("hybrid_v3_strict_score", ascending=False).head(int(limit)).copy()
    selected = selected.merge(
        scores.drop(columns=[c for c in selected.columns if c in scores.columns and c != "epic_id"]),
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    if len(selected) != int(limit):
        raise RuntimeError(f"Expected {int(limit)} unresolved Stage F recovery rows; got {len(selected)}")
    return selected.reset_index(drop=True)


def write_stage_f_input(selected: pd.DataFrame) -> pd.DataFrame:
    SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for batch_idx, (_, row) in enumerate(selected.iterrows(), start=1):
        epic_id = str(row["epic_id"]).strip()
        summary_path = SUMMARY_OUT_DIR / epic_id / f"{epic_id}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {k: json_scalar(row.get(k)) for k in row.index}
        events_csv = str(row.get("events_csv", "")).strip()
        if not events_csv:
            events_csv = str(ROOT / "plots" / "k2_batch" / "epics" / epic_id / "events.csv")
        summary = {
            "epic_id": epic_id,
            "query": f"EPIC {epic_digits(epic_id)}",
            "stage_r_and_stage_d_metrics": metrics,
            "artifacts": {
                "events_csv": events_csv,
                "light_curve_cache_path": "",
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(
            {
                "batch_order": batch_idx,
                "epic_id": epic_id,
                "best_period_days": row["best_period_days"],
                "period_support_count": row.get("period_support_count", ""),
                "visual_label": "visual_planet_like_candidate",
                "summary_json_path": str(summary_path.relative_to(ROOT)),
                "hybrid_v3_rank": row.get("hybrid_v3_rank", ""),
                "strict_rank": row.get("strict_rank", ""),
                "hybrid_v3_strict_score": row.get("hybrid_v3_strict_score", ""),
                "v3_base_score": row.get("v3_base_score", ""),
                "hybrid_v3_strict_fp_penalty": row.get("hybrid_v3_strict_fp_penalty", ""),
                "flux_p_science_like": row.get("flux_p_science_like", ""),
                "stage_d_quality_score": row.get("stage_d_quality_score", ""),
                "stage_d_label": row.get("stage_d_label", ""),
                "stage_d_reason": row.get("stage_d_reason", ""),
                "events_csv": events_csv,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(BATCH_INPUT_OUT, index=False)
    return out


def run_stage_f() -> dict[str, Any]:
    return K2StageFFollowupValidation().run(
        input_csv=BATCH_INPUT_OUT,
        output_csv=VALIDATION_OUT,
        out_dir=PLOTS_OUT_DIR,
    )


def write_visual_review_sheet(batch_input: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(VALIDATION_OUT)
    review = validation.merge(
        batch_input[
            [
                "batch_order",
                "epic_id",
                "hybrid_v3_rank",
                "strict_rank",
                "period_support_count",
                "hybrid_v3_strict_score",
                "v3_base_score",
                "hybrid_v3_strict_fp_penalty",
                "flux_p_science_like",
                "stage_d_quality_score",
                "stage_d_label",
                "stage_d_reason",
                "events_csv",
            ]
        ],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    review["manual_review_label"] = ""
    review["manual_review_status"] = ""
    review["visual_notes"] = ""
    review["reviewer"] = ""
    review["reviewed_at"] = ""
    columns = [
        "batch_order",
        "epic_id",
        "hybrid_v3_rank",
        "strict_rank",
        "hybrid_v3_strict_score",
        "v3_base_score",
        "hybrid_v3_strict_fp_penalty",
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_d_label",
        "best_period_days",
        "period_support_count",
        "stage_e_visual_label",
        "stage_f_label",
        "stage_f_reason",
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "radius_ratio_sqrt_depth",
        "secondary_depth_phase_05",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_period_days",
        "alias_best_support_count",
        "alias_best_support_ratio",
        "alias_risk",
        "phase_0_folded_path",
        "phase_05_secondary_check_path",
        "alias_period_comparison_path",
        "odd_even_zoom_path",
        "validation_summary_json_path",
        "events_csv",
        "stage_d_reason",
        "manual_review_label",
        "manual_review_status",
        "visual_notes",
        "reviewer",
        "reviewed_at",
    ]
    review = review[[c for c in columns if c in review.columns]]
    review.to_csv(VISUAL_REVIEW_OUT, index=False)
    return review


def main() -> None:
    args = parse_args()
    configure_batch(args.batch_name)
    selected = select_batch(args.limit)
    batch_input = write_stage_f_input(selected)
    result = run_stage_f()
    review = write_visual_review_sheet(batch_input)
    print(f"selected_epics={','.join(batch_input['epic_id'].astype(str))}")
    print(f"wrote {BATCH_INPUT_OUT.relative_to(ROOT)} rows={len(batch_input)}")
    print(f"wrote {VALIDATION_OUT.relative_to(ROOT)} rows={result.get('rows_output')}")
    print(f"wrote {VISUAL_REVIEW_OUT.relative_to(ROOT)} rows={len(review)}")
    print(f"plots_dir={PLOTS_OUT_DIR.relative_to(ROOT)}")
    print(f"stage_f_label_counts={pd.read_csv(VALIDATION_OUT)['stage_f_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
