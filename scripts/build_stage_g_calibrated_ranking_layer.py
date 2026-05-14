from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCORE_CSV = ROOT / "freezes" / "k2_hybrid_candidate_score_v3_stage_f_closed_45.csv"
FROZEN_LABELS_CSV = ROOT / "freezes" / "training_labels_v3_stage_f_closed_45.csv"
MASTER_LEDGER_CSV = ROOT / "k2_master_candidate_ledger.csv"
STAGE_G_DOSSIERS_CSV = ROOT / "k2_stage_g_candidate_dossiers.csv"
STAGE_F_QUEUE_CSV = ROOT / "next_needs_stage_f_validation.csv"

OUT_RANKING = ROOT / "freezes" / "stage_g_calibrated_ranking_layer.csv"
OUT_SWEEP = ROOT / "freezes" / "stage_g_calibrated_ranking_layer_sweep.csv"
OUT_RULE = ROOT / "freezes" / "stage_g_calibrated_ranking_layer_rule.json"
OUT_LEDGER_REVIEW = ROOT / "freezes" / "stage_g_candidate_ledger_review.csv"
OUT_SUMMARY = ROOT / "freezes" / "stage_g_candidate_ledger_review_summary.txt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def boolish(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def metric_row(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    actual = actual.astype(bool)
    pred = pred.astype(bool)
    tp = int((actual & pred).sum())
    fp = int((~actual & pred).sum())
    tn = int((~actual & ~pred).sum())
    fn = int((actual & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def topk_rows(scored: pd.DataFrame, k: int) -> dict[str, Any]:
    binary = scored[scored["binary_eval_included"]].sort_values("stage_g_calibrated_score", ascending=False)
    top = binary.head(k)
    tp = int(top["target_science_like"].sum())
    return {
        f"top_{k}_tp": tp,
        f"top_{k}_fp": int(len(top) - tp),
        f"top_{k}_precision": tp / len(top) if len(top) else 0.0,
    }


def load_base() -> pd.DataFrame:
    if not FROZEN_SCORE_CSV.exists():
        raise FileNotFoundError(f"Missing frozen score table: {FROZEN_SCORE_CSV}")
    if not FROZEN_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing frozen labels: {FROZEN_LABELS_CSV}")

    df = pd.read_csv(FROZEN_SCORE_CSV)
    df["epic_id"] = df["epic_id"].astype(str)
    labels = pd.read_csv(FROZEN_LABELS_CSV)[["epic_id", "training_label_v3", "science_binary_v3"]].copy()
    labels["epic_id"] = labels["epic_id"].astype(str)

    df = df.drop(columns=[c for c in ["training_label_v3", "science_binary_v3"] if c in df.columns])
    df = df.merge(labels, on="epic_id", how="left", validate="many_to_one")
    df["science_binary_v3"] = df["science_binary_v3"].fillna("unresolved")
    df["target_science_like"] = df["science_binary_v3"].eq("science_like")
    df["binary_eval_included"] = df["science_binary_v3"].isin(["science_like", "not_science_like"])
    return df


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stage_g_flux_component"] = clip01(num(df, "flux_p_science_like") / 0.80)
    df["stage_g_flux_gate_component"] = clip01((num(df, "flux_p_science_like") - 0.60) / 0.20)
    df["stage_g_stage_d_component"] = clip01(num(df, "stage_d_quality_score"))
    df["stage_g_stage_f_component"] = clip01(num(df, "stage_f_quality_score"))
    df["stage_g_event_support_component"] = pd.DataFrame(
        {
            "period_support": clip01(num(df, "period_support_count") / 20.0),
            "event_family": clip01(num(df, "event_family_count") / 6.0),
            "long_events": clip01(num(df, "n_events_long_good") / 12.0),
            "good_events": clip01(num(df, "n_events_ge_10_cadences") / 20.0),
            "coverage": clip01(num(df, "coverage_rate")),
            "soft_hit_rate": clip01(num(df, "soft_hit_rate")),
        }
    ).mean(axis=1)
    df["stage_g_shape_consistency_component"] = pd.DataFrame(
        {
            "folded_depth": clip01(num(df, "folded_depth_consistency")),
            "duration": clip01(num(df, "duration_consistency")),
            "max_shape": clip01(num(df, "max_shape_score") / 0.88),
        }
    ).mean(axis=1)

    df["stage_g_odd_even_penalty_component"] = np.maximum(
        clip01(num(df, "stage_d_odd_even_penalty")),
        clip01(num(df, "stage_f_odd_even_penalty")),
    )
    df["stage_g_secondary_penalty_component"] = clip01(num(df, "stage_f_secondary_penalty"))
    df["stage_g_oot_penalty_component"] = clip01(num(df, "stage_f_oot_penalty"))
    df["stage_g_alias_penalty_component"] = clip01(num(df, "stage_f_alias_penalty"))
    df["stage_g_spike_penalty_component"] = clip01(num(df, "spike_fraction_2cadence") / 0.35)
    return df


BASE_WEIGHT_SETS = [
    {"flux": 0.20, "flux_gate": 0.10, "stage_d": 0.25, "stage_f": 0.25, "event": 0.10, "shape": 0.10},
    {"flux": 0.15, "flux_gate": 0.10, "stage_d": 0.30, "stage_f": 0.25, "event": 0.10, "shape": 0.10},
    {"flux": 0.15, "flux_gate": 0.10, "stage_d": 0.25, "stage_f": 0.30, "event": 0.10, "shape": 0.10},
    {"flux": 0.25, "flux_gate": 0.10, "stage_d": 0.20, "stage_f": 0.25, "event": 0.10, "shape": 0.10},
    {"flux": 0.20, "flux_gate": 0.05, "stage_d": 0.30, "stage_f": 0.25, "event": 0.10, "shape": 0.10},
]

PENALTY_WEIGHT_SETS = [
    {"odd_even": 0.28, "secondary": 0.22, "oot": 0.22, "alias": 0.16, "spike": 0.08},
    {"odd_even": 0.35, "secondary": 0.20, "oot": 0.20, "alias": 0.15, "spike": 0.10},
    {"odd_even": 0.25, "secondary": 0.30, "oot": 0.20, "alias": 0.15, "spike": 0.10},
    {"odd_even": 0.25, "secondary": 0.20, "oot": 0.30, "alias": 0.15, "spike": 0.10},
    {"odd_even": 0.25, "secondary": 0.20, "oot": 0.20, "alias": 0.25, "spike": 0.10},
]


def score_with_rule(df: pd.DataFrame, base: dict[str, float], penalty: dict[str, float], penalty_scale: float) -> pd.Series:
    base_score = (
        base["flux"] * df["stage_g_flux_component"]
        + base["flux_gate"] * df["stage_g_flux_gate_component"]
        + base["stage_d"] * df["stage_g_stage_d_component"]
        + base["stage_f"] * df["stage_g_stage_f_component"]
        + base["event"] * df["stage_g_event_support_component"]
        + base["shape"] * df["stage_g_shape_consistency_component"]
    )
    penalty_score = (
        penalty["odd_even"] * df["stage_g_odd_even_penalty_component"]
        + penalty["secondary"] * df["stage_g_secondary_penalty_component"]
        + penalty["oot"] * df["stage_g_oot_penalty_component"]
        + penalty["alias"] * df["stage_g_alias_penalty_component"]
        + penalty["spike"] * df["stage_g_spike_penalty_component"]
    )
    return pd.Series(clip01(base_score - penalty_scale * penalty_score), index=df.index)


def sweep_rules(df: pd.DataFrame) -> pd.DataFrame:
    binary = df[df["binary_eval_included"]].copy()
    rows: list[dict[str, Any]] = []
    for base_idx, penalty_idx, penalty_scale, threshold in product(
        range(len(BASE_WEIGHT_SETS)),
        range(len(PENALTY_WEIGHT_SETS)),
        [0.55, 0.70, 0.85, 1.00, 1.15],
        np.round(np.arange(0.35, 0.751, 0.025), 3),
    ):
        scores = score_with_rule(df, BASE_WEIGHT_SETS[base_idx], PENALTY_WEIGHT_SETS[penalty_idx], penalty_scale)
        pred = (scores.loc[binary.index] >= threshold) & (num(binary, "flux_p_science_like") >= 0.60)
        metrics = metric_row(binary["target_science_like"], pred)
        scored_tmp = binary.assign(stage_g_calibrated_score=scores.loc[binary.index].to_numpy())
        row = {
            "base_weight_set": base_idx,
            "penalty_weight_set": penalty_idx,
            "penalty_scale": penalty_scale,
            "threshold": threshold,
            "predicted_positives": int(pred.sum()),
            **metrics,
            **topk_rows(scored_tmp, 5),
            **topk_rows(scored_tmp, 10),
        }
        row["meets_precision_target"] = row["precision"] >= 0.50
        row["meets_recall_floor"] = row["recall"] >= 0.25
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep["selection_bucket"] = np.select(
        [
            sweep["meets_precision_target"] & sweep["meets_recall_floor"],
            sweep["meets_precision_target"],
        ],
        [0, 1],
        default=2,
    )
    return sweep.sort_values(
        [
            "selection_bucket",
            "precision",
            "recall",
            "f1",
            "top_5_precision",
            "fp",
            "threshold",
        ],
        ascending=[True, False, False, False, False, True, False],
    ).reset_index(drop=True)


def classify_rows(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    flux = num(df, "flux_p_science_like")
    hard_penalty = (
        (df["stage_g_odd_even_penalty_component"] >= 0.90)
        | (df["stage_g_secondary_penalty_component"] >= 0.90)
        | (df["stage_g_oot_penalty_component"] >= 0.90)
        | (df["stage_g_alias_penalty_component"] >= 0.90)
    )
    df["stage_g_flux_send_to_vetting"] = flux >= 0.60
    df["stage_g_interesting_flux"] = flux >= 0.70
    df["stage_g_calibrated_pass"] = (df["stage_g_calibrated_score"] >= threshold) & df["stage_g_flux_send_to_vetting"]
    df["stage_g_review_tier"] = np.select(
        [
            df["stage_g_calibrated_pass"] & ~hard_penalty,
            df["stage_g_calibrated_pass"] & hard_penalty,
            df["stage_g_flux_send_to_vetting"],
            df["stage_g_interesting_flux"],
        ],
        [
            "A_calibrated_followup_review",
            "B_followup_with_red_flag_review",
            "C_flux_triage_deeper_vetting",
            "C_flux_triage_deeper_vetting",
        ],
        default="D_hold_or_reject_review",
    )
    df["stage_g_review_reason"] = [
        "; ".join(
            part
            for part in [
                f"flux={float(f):.3f}",
                f"score={float(s):.3f}",
                "flux>=0.60" if f >= 0.60 else "flux<0.60",
                "hard_fp_penalty" if hp else "",
            ]
            if part
        )
        for f, s, hp in zip(flux, df["stage_g_calibrated_score"], hard_penalty)
    ]
    return df


def build_ledger_review(scored: pd.DataFrame) -> pd.DataFrame:
    review = scored.copy()
    if MASTER_LEDGER_CSV.exists():
        ledger = pd.read_csv(MASTER_LEDGER_CSV)
        ledger["epic_id"] = ledger["epic_id"].astype(str)
        ledger_cols = [
            "epic_id",
            "source_batch",
            "final_candidate_status",
            "review_bin",
            "visual_label",
            "visual_notes",
        ]
        review = review.merge(
            ledger[[c for c in ledger_cols if c in ledger.columns]],
            on="epic_id",
            how="left",
            suffixes=("", "_ledger"),
        )
    if STAGE_G_DOSSIERS_CSV.exists():
        dossiers = pd.read_csv(STAGE_G_DOSSIERS_CSV)
        dossiers["epic_id"] = dossiers["epic_id"].astype(str)
        review = review.merge(
            dossiers[["epic_id", "final_candidate_rank", "final_recommendation", "candidate_summary_md_path"]],
            on="epic_id",
            how="left",
        )
    status = review.get("final_candidate_status", pd.Series("", index=review.index)).fillna("").astype(str)
    binary = review["science_binary_v3"].fillna("unresolved").astype(str)
    calibrated_pass = review["stage_g_calibrated_pass"].astype(bool)
    flux_vetting = review["stage_g_flux_send_to_vetting"].astype(bool)
    negative_status = status.str.contains("reject|deprioritize|binary|secondary_hold|low_priority", case=False, regex=True)
    promoted_status = status.str.contains("promote|recovered_known", case=False, regex=True)
    review["stage_g_ledger_review_action"] = np.select(
        [
            calibrated_pass & binary.eq("not_science_like"),
            flux_vetting & negative_status,
            calibrated_pass & binary.eq("science_like"),
            flux_vetting & binary.eq("science_like") & ~calibrated_pass,
            calibrated_pass & binary.eq("unresolved"),
            flux_vetting & binary.eq("unresolved"),
            promoted_status,
        ],
        [
            "calibration_false_positive_do_not_promote",
            "closed_negative_do_not_reopen_without_new_evidence",
            "candidate_supported_by_calibrated_layer",
            "known_positive_below_pass_threshold_recheck_features",
            "unresolved_high_rank_needs_stage_g_review",
            "unresolved_flux_triage_when_queue_reopens",
            "existing_promoted_or_recovered_candidate",
        ],
        default="hold_or_reject_review",
    )
    keep = [
        "stage_g_rank",
        "epic_id",
        "stage_g_calibrated_score",
        "stage_g_review_tier",
        "stage_g_ledger_review_action",
        "stage_g_review_reason",
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "stage_g_odd_even_penalty_component",
        "stage_g_secondary_penalty_component",
        "stage_g_oot_penalty_component",
        "stage_g_alias_penalty_component",
        "training_label_v3",
        "science_binary_v3",
        "source_batch",
        "final_candidate_status",
        "review_bin",
        "visual_label",
        "final_candidate_rank",
        "final_recommendation",
        "candidate_summary_md_path",
        "stage_d_reason",
        "stage_r_reason",
        "events_csv",
    ]
    return review[[c for c in keep if c in review.columns]]


def main() -> None:
    base = add_components(load_base())
    sweep = sweep_rules(base)
    best = sweep.iloc[0].to_dict()
    base_weights = BASE_WEIGHT_SETS[int(best["base_weight_set"])]
    penalty_weights = PENALTY_WEIGHT_SETS[int(best["penalty_weight_set"])]
    threshold = float(best["threshold"])
    penalty_scale = float(best["penalty_scale"])

    scored = base.copy()
    scored["stage_g_calibrated_score"] = score_with_rule(scored, base_weights, penalty_weights, penalty_scale)
    scored = classify_rows(scored, threshold)
    scored = scored.sort_values("stage_g_calibrated_score", ascending=False).reset_index(drop=True)
    scored.insert(0, "stage_g_rank", np.arange(1, len(scored) + 1))

    ledger_review = build_ledger_review(scored)
    stage_f_queue_count = len(pd.read_csv(STAGE_F_QUEUE_CSV)) if STAGE_F_QUEUE_CSV.exists() else None
    binary = scored[scored["binary_eval_included"]].copy()
    metrics = metric_row(binary["target_science_like"], binary["stage_g_calibrated_pass"])
    top5 = topk_rows(scored, 5)
    top10 = topk_rows(scored, 10)

    rule = {
        "model_policy": "Keras flux model is frozen; this is a tabular Stage G ranking layer only.",
        "active_flux_model": "models/k2_nocrop_flux_seed46_split303.best.keras",
        "input_score": "flux_p_science_like",
        "flux_send_to_vetting_threshold": 0.60,
        "base_weights": base_weights,
        "penalty_weights": penalty_weights,
        "penalty_scale": penalty_scale,
        "calibrated_pass_threshold": threshold,
        "calibration_rows": {
            "total_scored_rows": int(len(scored)),
            "binary_eval_rows": int(len(binary)),
            "positives": int(binary["target_science_like"].sum()),
            "negatives": int((~binary["target_science_like"]).sum()),
            "excluded_unresolved": int((~scored["binary_eval_included"]).sum()),
        },
        "binary_metrics_at_pass_threshold": metrics,
        "top_k_precision": {**top5, **top10},
        "stage_f_queue_rows": stage_f_queue_count,
    }

    scored.to_csv(OUT_RANKING, index=False)
    sweep.to_csv(OUT_SWEEP, index=False)
    OUT_RULE.write_text(json.dumps(rule, indent=2), encoding="utf-8")
    ledger_review.to_csv(OUT_LEDGER_REVIEW, index=False)

    tier_counts = ledger_review["stage_g_review_tier"].value_counts().to_dict()
    action_counts = ledger_review["stage_g_ledger_review_action"].value_counts().to_dict()
    lines = [
        "Stage G Candidate Ledger Review and Calibrated Ranking Layer",
        "=" * 62,
        "",
        "Keras model status: frozen; no Keras training or fine-tuning performed.",
        f"Active flux model: {rule['active_flux_model']}",
        f"Frozen labels: {rel(FROZEN_LABELS_CSV)}",
        f"Stage F queue rows: {stage_f_queue_count}",
        "",
        "Calibrated ranking layer:",
        f"- flux send-to-vetting threshold: {rule['flux_send_to_vetting_threshold']:.2f}",
        f"- calibrated pass threshold: {threshold:.3f}",
        f"- penalty scale: {penalty_scale:.2f}",
        f"- binary metrics: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']} "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} F1={metrics['f1']:.3f}",
        f"- top-5 precision: {top5['top_5_precision']:.3f}",
        f"- top-10 precision: {top10['top_10_precision']:.3f}",
        "",
        "Stage G review tiers:",
        *[f"- {tier}: {count}" for tier, count in tier_counts.items()],
        "",
        "Stage G ledger-review actions:",
        *[f"- {action}: {count}" for action, count in action_counts.items()],
        "",
        "Top Stage G review rows:",
        ledger_review[
            [
                "stage_g_rank",
                "epic_id",
                "stage_g_calibrated_score",
                "stage_g_review_tier",
                "stage_g_ledger_review_action",
                "flux_p_science_like",
                "training_label_v3",
                "science_binary_v3",
                "final_candidate_status",
            ]
        ]
        .head(15)
        .to_string(index=False),
        "",
        f"Wrote: {rel(OUT_RANKING)}",
        f"Wrote: {rel(OUT_SWEEP)}",
        f"Wrote: {rel(OUT_RULE)}",
        f"Wrote: {rel(OUT_LEDGER_REVIEW)}",
        f"Wrote: {rel(OUT_SUMMARY)}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
