from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HYBRID_V2_CSV = ROOT / "k2_hybrid_candidate_score_v2.csv"
LABELS_V3_CSV = ROOT / "training_labels_v3.csv"
STAGE_F_HYBRID_V2_CSV = ROOT / "k2_stage_f_hybrid_v2_validation.csv"

OUT_ALL = ROOT / "k2_hybrid_candidate_score_v3.csv"
OUT_TOP_UNRESOLVED = ROOT / "k2_hybrid_top50_unresolved_v3.csv"
OUT_RULE_SEARCH = ROOT / "k2_hybrid_score_v3_rule_search.csv"
OUT_BEST_RULE = ROOT / "k2_hybrid_score_v3_best_rule.csv"
OUT_LABEL_EVAL = ROOT / "k2_hybrid_score_v3_label_eval.csv"
OUT_TOP_FALSE_POSITIVES = ROOT / "k2_hybrid_score_v3_top_false_positives.csv"
OUT_MISSED_POSITIVES = ROOT / "k2_hybrid_score_v3_missed_positives.csv"

SCIENCE_LABELS = {"planet_like", "candidate_like"}


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def as_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def penalty_up(df: pd.DataFrame, col: str, bad_at: float) -> pd.Series:
    return pd.Series(clip01(as_num(df, col) / bad_at), index=df.index)


def stage_f_quality(raw: pd.DataFrame) -> pd.Series:
    label_score = raw.get("stage_f_label", pd.Series("", index=raw.index)).map(
        {
            "stage_f_planet_like": 1.0,
            "stage_f_hold": 0.55,
            "stage_f_likely_eb": 0.05,
            "stage_f_reject": 0.10,
        }
    ).fillna(0.5)
    depth_snr_score = clip01(np.log1p(as_num(raw, "primary_depth_snr")) / np.log1p(20.0))
    secondary_clean = 1.0 - clip01(as_num(raw, "secondary_to_primary_depth_ratio") / 0.50)
    odd_even_clean = 1.0 - clip01(as_num(raw, "odd_even_depth_delta_explicit") / 0.50)
    oot_clean = 1.0 - clip01(as_num(raw, "oot_variability_to_depth") / 2.00)
    alias_clean = 1.0 - clip01((as_num(raw, "alias_best_support_ratio") - 0.35) / 0.50)
    return pd.Series(
        0.25 * depth_snr_score
        + 0.25 * secondary_clean
        + 0.20 * odd_even_clean
        + 0.15 * oot_clean
        + 0.10 * alias_clean
        + 0.05 * label_score,
        index=raw.index,
    )


def load_stage_f_overrides() -> pd.DataFrame:
    raw = pd.read_csv(STAGE_F_HYBRID_V2_CSV)
    raw = raw.copy()
    raw["stage_f_quality_score_override"] = stage_f_quality(raw)
    raw["stage_f_odd_even_penalty_override"] = penalty_up(raw, "odd_even_depth_delta_explicit", 0.50)
    raw["stage_f_secondary_penalty_override"] = np.maximum(
        penalty_up(raw, "secondary_to_primary_depth_ratio", 0.50),
        penalty_up(raw, "secondary_depth_snr", 7.0),
    )
    raw["stage_f_oot_penalty_override"] = penalty_up(raw, "oot_variability_to_depth", 2.00)
    raw["stage_f_alias_penalty_override"] = clip01((as_num(raw, "alias_best_support_ratio") - 0.35) / 0.50)
    keep = [
        "epic_id",
        "stage_f_label",
        "primary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "stage_f_quality_score_override",
        "stage_f_odd_even_penalty_override",
        "stage_f_secondary_penalty_override",
        "stage_f_oot_penalty_override",
        "stage_f_alias_penalty_override",
    ]
    return raw[[c for c in keep if c in raw.columns]]


def load_base() -> pd.DataFrame:
    df = pd.read_csv(HYBRID_V2_CSV)
    labels = pd.read_csv(LABELS_V3_CSV)[["epic_id", "training_label_v3", "science_binary_v3"]]
    overrides = load_stage_f_overrides()

    stale_label_cols = [
        "training_label_v2",
        "training_label_rule",
        "final_candidate_status",
        "review_bin",
        "is_labeled_in_ledger",
        "is_unresolved",
    ]
    df = df.drop(columns=[c for c in stale_label_cols if c in df.columns])
    df = df.merge(labels, on="epic_id", how="left", validate="one_to_one")
    df = df.merge(overrides, on="epic_id", how="left", suffixes=("", "_stage_f_hybrid_v2"))

    override_map = {
        "stage_f_quality_score": "stage_f_quality_score_override",
        "stage_f_odd_even_penalty": "stage_f_odd_even_penalty_override",
        "stage_f_secondary_penalty": "stage_f_secondary_penalty_override",
        "stage_f_oot_penalty": "stage_f_oot_penalty_override",
        "stage_f_alias_penalty": "stage_f_alias_penalty_override",
    }
    for target, source in override_map.items():
        if target not in df.columns:
            df[target] = np.nan
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[source] = pd.to_numeric(df[source], errors="coerce")
        df[target] = df[source].combine_first(df[target])

    for col in [
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
        "stage_d_odd_even_penalty",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["stage_f_quality_score"] = df["stage_f_quality_score"].replace(0.0, np.nan).fillna(0.25)
    df["science_binary_v3"] = df["science_binary_v3"].fillna("unresolved")
    df["is_labeled_v3"] = df["training_label_v3"].notna()
    df["is_unresolved_v3"] = ~df["is_labeled_v3"]
    df["is_tuning_row_v3"] = df["is_labeled_v3"] & df["training_label_v3"].ne("uncertain_hold")
    df["actual_science_like_v3"] = df["training_label_v3"].isin(SCIENCE_LABELS)
    df["v3_base_score"] = clip01(
        0.35 * df["flux_p_science_like"]
        + 0.25 * df["stage_d_quality_score"]
        + 0.25 * df["stage_f_quality_score"]
    )
    return df


def score_with_rule(
    df: pd.DataFrame,
    odd_even_weight: float,
    oot_weight: float,
    secondary_weight: float,
    alias_weight: float,
) -> pd.Series:
    odd_even_penalty = np.maximum(df["stage_d_odd_even_penalty"], df["stage_f_odd_even_penalty"])
    fp_penalty = clip01(
        odd_even_weight * odd_even_penalty
        + oot_weight * df["stage_f_oot_penalty"]
        + secondary_weight * df["stage_f_secondary_penalty"]
        + alias_weight * df["stage_f_alias_penalty"]
    )
    return pd.Series(clip01(df["v3_base_score"] - fp_penalty), index=df.index)


def metrics_for(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
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


def search_rules(df: pd.DataFrame) -> pd.DataFrame:
    tuning = df[df["is_tuning_row_v3"]].copy()
    actual = tuning["actual_science_like_v3"]
    rows: list[dict[str, Any]] = []
    for odd_even_weight, oot_weight, secondary_weight, alias_weight, min_candidate_score in product(
        [0.15, 0.25, 0.35, 0.45, 0.55, 0.70],
        [0.10, 0.20, 0.30, 0.40, 0.55],
        [0.10, 0.20, 0.35, 0.50, 0.70],
        [0.05, 0.10, 0.20, 0.35],
        np.round(np.arange(0.30, 0.701, 0.025), 3),
    ):
        score = score_with_rule(df, odd_even_weight, oot_weight, secondary_weight, alias_weight).loc[tuning.index]
        metrics = metrics_for(actual, score >= min_candidate_score)
        rows.append(
            {
                "odd_even_weight": odd_even_weight,
                "oot_weight": oot_weight,
                "secondary_weight": secondary_weight,
                "alias_weight": alias_weight,
                "min_candidate_score": min_candidate_score,
                "meets_precision_target": metrics["precision"] >= 0.50,
                "meets_recall_target": metrics["recall"] >= 0.75,
                **metrics,
            }
        )
    search = pd.DataFrame(rows)
    search["target_rank_bucket"] = np.select(
        [
            search["meets_precision_target"] & search["meets_recall_target"],
            search["meets_precision_target"],
        ],
        [0, 1],
        default=2,
    )
    search = search.sort_values(
        [
            "target_rank_bucket",
            "recall",
            "precision",
            "f1",
            "fp",
            "min_candidate_score",
        ],
        ascending=[True, False, False, False, True, False],
    ).reset_index(drop=True)
    search.insert(0, "rule_rank", np.arange(1, len(search) + 1))
    return search


def apply_best_rule(df: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["hybrid_score_v3"] = score_with_rule(
        df,
        float(best["odd_even_weight"]),
        float(best["oot_weight"]),
        float(best["secondary_weight"]),
        float(best["alias_weight"]),
    )
    odd_even_penalty = np.maximum(df["stage_d_odd_even_penalty"], df["stage_f_odd_even_penalty"])
    df["hybrid_v3_false_positive_penalty"] = clip01(
        float(best["odd_even_weight"]) * odd_even_penalty
        + float(best["oot_weight"]) * df["stage_f_oot_penalty"]
        + float(best["secondary_weight"]) * df["stage_f_secondary_penalty"]
        + float(best["alias_weight"]) * df["stage_f_alias_penalty"]
    )
    df["hybrid_v3_candidate_pass"] = df["hybrid_score_v3"] >= float(best["min_candidate_score"])
    df = df.sort_values("hybrid_score_v3", ascending=False).reset_index(drop=True)
    df.insert(0, "hybrid_v3_rank", np.arange(1, len(df) + 1))
    return df


def write_eval_outputs(scored: pd.DataFrame, best: pd.Series) -> None:
    tuning = scored[scored["is_tuning_row_v3"]].copy()
    metrics = metrics_for(tuning["actual_science_like_v3"], tuning["hybrid_v3_candidate_pass"])
    eval_row = {
        "model": "hybrid_score_v3",
        "decision_rule": "hybrid_score_v3 >= min_candidate_score",
        "excluded_from_metric_optimization": "uncertain_hold",
        **best[
            [
                "odd_even_weight",
                "oot_weight",
                "secondary_weight",
                "alias_weight",
                "min_candidate_score",
            ]
        ].to_dict(),
        **metrics,
    }
    pd.DataFrame([eval_row]).to_csv(OUT_LABEL_EVAL, index=False)

    fp = tuning[(~tuning["actual_science_like_v3"]) & tuning["hybrid_v3_candidate_pass"]].copy()
    fp = fp.sort_values("hybrid_score_v3", ascending=False)
    missed = tuning[tuning["actual_science_like_v3"] & ~tuning["hybrid_v3_candidate_pass"]].copy()
    missed = missed.sort_values("hybrid_score_v3", ascending=False)
    keep = [
        "hybrid_v3_rank",
        "epic_id",
        "training_label_v3",
        "hybrid_score_v3",
        "v3_base_score",
        "hybrid_v3_false_positive_penalty",
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
    ]
    fp[[c for c in keep if c in fp.columns]].to_csv(OUT_TOP_FALSE_POSITIVES, index=False)
    missed[[c for c in keep if c in missed.columns]].to_csv(OUT_MISSED_POSITIVES, index=False)


def main() -> None:
    df = load_base()
    search = search_rules(df)
    best = search.iloc[0]
    scored = apply_best_rule(df, best)

    search.to_csv(OUT_RULE_SEARCH, index=False)
    pd.DataFrame([best]).to_csv(OUT_BEST_RULE, index=False)
    scored.to_csv(OUT_ALL, index=False)
    top_unresolved = scored[scored["is_unresolved_v3"]].head(50).copy()
    top_unresolved.insert(0, "unresolved_hybrid_v3_rank", np.arange(1, len(top_unresolved) + 1))
    top_unresolved.to_csv(OUT_TOP_UNRESOLVED, index=False)
    write_eval_outputs(scored, best)

    print(f"wrote {OUT_ALL.relative_to(ROOT)} rows={len(scored)}")
    print(f"wrote {OUT_RULE_SEARCH.relative_to(ROOT)} rows={len(search)}")
    print(f"wrote {OUT_BEST_RULE.relative_to(ROOT)}")
    print(f"wrote {OUT_TOP_UNRESOLVED.relative_to(ROOT)} rows={len(top_unresolved)}")
    print(f"wrote {OUT_LABEL_EVAL.relative_to(ROOT)}")
    print("\nbest rule")
    print(pd.DataFrame([best]).to_string(index=False))
    print("\ntop unresolved")
    cols = [
        "unresolved_hybrid_v3_rank",
        "hybrid_v3_rank",
        "epic_id",
        "hybrid_score_v3",
        "hybrid_v3_candidate_pass",
        "v3_base_score",
        "hybrid_v3_false_positive_penalty",
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "needs_stage_f_validation",
    ]
    print(top_unresolved[[c for c in cols if c in top_unresolved.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
