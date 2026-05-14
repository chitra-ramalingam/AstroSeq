from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_k2_hybrid_candidate_score_v1 import (
    LABELS_CSV,
    OUT_ALL as OUT_V1_ALL,
    ROOT,
    STAGE_D_CSV,
    clip01,
    load_flux_scores,
    load_stage_f,
    manual_penalty,
    penalty_up,
    score_down,
    score_up,
)


OUT_ALL = ROOT / "k2_hybrid_candidate_score_v2.csv"
OUT_TOP50_UNRESOLVED = ROOT / "k2_hybrid_top50_unresolved_v2.csv"
OUT_NEEDS_STAGE_F = ROOT / "k2_hybrid_top50_needs_stage_f.csv"
OUT_STAGE_F_TOP10 = ROOT / "k2_hybrid_stage_f_validation_top10.csv"


def main() -> None:
    stage_d = pd.read_csv(STAGE_D_CSV)
    labels = pd.read_csv(LABELS_CSV)
    flux = load_flux_scores(set(stage_d["epic_id"].astype(str)))
    stage_f = load_stage_f()

    df = stage_d.merge(flux, on="epic_id", how="left", validate="one_to_one")
    df = df.merge(stage_f, on="epic_id", how="left")
    df = df.merge(
        labels[
            [
                "epic_id",
                "training_label_v2",
                "final_candidate_status",
                "review_bin",
                "training_label_rule",
            ]
        ],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )

    df["stage_f_available"] = df["stage_f_available"].fillna(False).astype(bool)
    df["stage_f_quality_score_raw"] = df["stage_f_quality_score"]
    df["stage_f_quality_score"] = df["stage_f_quality_score"].fillna(0.25)
    for col in [
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
    ]:
        df[col] = df[col].fillna(0.0)

    stage_d_positive_terms = pd.DataFrame(
        {
            "n_events_long_good": score_up(df["n_events_long_good"], 15.0),
            "n_events_ge_10_cadences": score_up(df["n_events_ge_10_cadences"], 20.0),
            "max_shape_score": score_up(df["max_shape_score"], 0.88),
            "period_support_count": score_up(df["period_support_count"], 20.0),
            "event_family_count": score_up(df["event_family_count"], 8.0),
            "folded_depth_consistency": score_up(df["folded_depth_consistency"], 1.0),
            "duration_consistency": score_up(df["duration_consistency"], 1.0),
            "coverage_rate": score_up(df["coverage_rate"], 1.0),
            "hit_rate_snr": score_up(df["hit_rate_snr"], 1.0),
            "hit_rate_shape": score_up(df["hit_rate_shape"], 1.0),
            "soft_hit_rate": score_up(df["soft_hit_rate"], 1.0),
        }
    )
    df["stage_d_positive_score"] = stage_d_positive_terms.mean(axis=1)
    df["stage_d_spike_clean"] = score_down(df["spike_fraction_2cadence"], 0.35)
    df["stage_d_odd_even_clean"] = score_down(df["odd_even_depth_delta"], 0.50)
    df["stage_d_quality_score"] = clip01(
        0.82 * df["stage_d_positive_score"]
        + 0.09 * df["stage_d_spike_clean"]
        + 0.09 * df["stage_d_odd_even_clean"]
        - np.where(df["stage_d_label"].eq("hold_deeper_eval"), 0.12, 0.0)
    )

    df["manual_label_penalty"] = [
        manual_penalty(label, status, review_bin)
        for label, status, review_bin in zip(
            df["training_label_v2"],
            df["final_candidate_status"],
            df["review_bin"],
        )
    ]
    df["stage_d_odd_even_penalty"] = penalty_up(df["odd_even_depth_delta"], 0.50)
    df["false_positive_penalty"] = clip01(
        0.22 * np.maximum(df["stage_d_odd_even_penalty"], df["stage_f_odd_even_penalty"])
        + 0.20 * df["stage_f_secondary_penalty"]
        + 0.18 * df["stage_f_oot_penalty"]
        + 0.15 * df["stage_f_alias_penalty"]
        + 0.25 * df["manual_label_penalty"]
    )
    df["flux_recall_filter_pass"] = df["flux_p_science_like"] >= 0.5
    df["candidate_score"] = clip01(
        0.35 * df["flux_p_science_like"]
        + 0.25 * df["stage_d_quality_score"]
        + 0.25 * df["stage_f_quality_score"]
        - 0.15 * df["false_positive_penalty"]
    )
    df["is_labeled_in_ledger"] = df["training_label_v2"].notna()
    df["is_unresolved"] = ~df["is_labeled_in_ledger"]
    df["needs_stage_f_validation"] = (
        df["is_unresolved"] & df["flux_recall_filter_pass"] & ~df["stage_f_available"]
    )
    df["hybrid_v2_policy_note"] = np.where(
        df["needs_stage_f_validation"],
        "high_flux_or_stage_d_but_missing_stage_f; validate_before_discovery_ranking",
        "",
    )

    df = df.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    df.insert(0, "hybrid_rank", np.arange(1, len(df) + 1))

    preferred_cols = [
        "hybrid_rank",
        "epic_id",
        "candidate_score",
        "needs_stage_f_validation",
        "hybrid_v2_policy_note",
        "flux_p_science_like",
        "flux_recall_filter_pass",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "stage_f_available",
        "false_positive_penalty",
        "manual_label_penalty",
        "is_unresolved",
        "training_label_v2",
        "final_candidate_status",
        "review_bin",
        "stage_d_label",
        "stage_f_conservative_label",
        "best_period_days",
        "period_support_count",
        "event_family_count",
        "folded_depth_consistency",
        "duration_consistency",
        "odd_even_depth_delta",
        "primary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
        "flux_num_segments",
        "flux_p_top3_mean",
        "flux_p_top10_mean",
        "flux_best_segment_start",
        "flux_best_segment_end",
        "flux_best_segment_mid_time",
        "events_csv",
        "stage_f_source_csv",
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    cols = [c for c in preferred_cols if c in df.columns] + remaining
    df[cols].to_csv(OUT_ALL, index=False)

    top_unresolved = df[df["is_unresolved"] & df["flux_recall_filter_pass"]].head(50).copy()
    top_unresolved.insert(0, "unresolved_hybrid_rank", np.arange(1, len(top_unresolved) + 1))
    top_unresolved[[c for c in ["unresolved_hybrid_rank"] + preferred_cols if c in top_unresolved.columns] + remaining].to_csv(
        OUT_TOP50_UNRESOLVED,
        index=False,
    )

    needs_stage_f = df[df["needs_stage_f_validation"]].head(50).copy()
    needs_stage_f.insert(0, "needs_stage_f_rank", np.arange(1, len(needs_stage_f) + 1))
    needs_stage_f[[c for c in ["needs_stage_f_rank"] + preferred_cols if c in needs_stage_f.columns] + remaining].to_csv(
        OUT_NEEDS_STAGE_F,
        index=False,
    )

    top10 = needs_stage_f.head(10).copy()
    top10.to_csv(OUT_STAGE_F_TOP10, index=False)

    print(f"v1 reference: {OUT_V1_ALL}")
    print(f"wrote {OUT_ALL} rows={len(df)}")
    print(f"wrote {OUT_TOP50_UNRESOLVED} rows={len(top_unresolved)}")
    print(f"wrote {OUT_NEEDS_STAGE_F} rows={len(needs_stage_f)}")
    print(f"wrote {OUT_STAGE_F_TOP10} rows={len(top10)}")
    print(
        top10[
            [
                "needs_stage_f_rank",
                "hybrid_rank",
                "epic_id",
                "candidate_score",
                "flux_p_science_like",
                "stage_d_quality_score",
                "stage_f_quality_score",
                "false_positive_penalty",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
