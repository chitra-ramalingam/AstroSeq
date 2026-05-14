from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = ROOT / "k2_hybrid_candidate_score_v3.csv"

STRICT_PROMOTED_OUT = ROOT / "strict_promoted_candidates.csv"
STRICT_NOT_PROMOTED_OUT = ROOT / "strict_rejected_or_not_promoted.csv"
MISSED_POSITIVE_AUDIT_OUT = ROOT / "missed_positive_audit.csv"
NEXT_NEEDS_STAGE_F_OUT = ROOT / "next_needs_stage_f_validation.csv"

ODD_EVEN_WEIGHT = 0.15
OOT_WEIGHT = 0.10
SECONDARY_WEIGHT = 0.10
ALIAS_WEIGHT = 0.05
MIN_CANDIDATE_SCORE = 0.575

SCIENCE_LABELS = {"planet_like", "candidate_like"}
EXPLICIT_NON_PROMOTION_LABELS = {
    "uncertain_hold",
    "low_priority_negative",
    "noise_or_artifact",
    "binary_system",
    "variable_or_artifact",
}
MANUAL_HOLD_EPICS = {
    "EPIC_211936906",
    "EPIC_211538087",
    "EPIC_211818652",
    "EPIC_211552564",
}


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def as_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).astype(float)


def apply_frozen_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    odd_even_penalty = np.maximum(as_num(df, "stage_d_odd_even_penalty"), as_num(df, "stage_f_odd_even_penalty"))
    df["hybrid_v3_strict_fp_penalty"] = clip01(
        ODD_EVEN_WEIGHT * odd_even_penalty
        + OOT_WEIGHT * as_num(df, "stage_f_oot_penalty")
        + SECONDARY_WEIGHT * as_num(df, "stage_f_secondary_penalty")
        + ALIAS_WEIGHT * as_num(df, "stage_f_alias_penalty")
    )
    df["hybrid_v3_strict_score"] = clip01(as_num(df, "v3_base_score") - df["hybrid_v3_strict_fp_penalty"])
    df["strict_score_pass"] = df["hybrid_v3_strict_score"] >= MIN_CANDIDATE_SCORE
    df["is_explicit_non_promotion_label"] = df["training_label_v3"].isin(EXPLICIT_NON_PROMOTION_LABELS)
    df["is_manual_hold_epic"] = df["epic_id"].isin(MANUAL_HOLD_EPICS)
    df["is_science_label_v3"] = df["training_label_v3"].isin(SCIENCE_LABELS)
    df["strict_promoted"] = df["strict_score_pass"] & ~df["is_explicit_non_promotion_label"] & ~df["is_manual_hold_epic"]
    df["strict_policy_reason"] = np.select(
        [
            df["strict_promoted"] & df["is_science_label_v3"],
            df["strict_promoted"] & df["is_unresolved_v3"].astype(str).eq("True"),
            df["strict_score_pass"] & df["training_label_v3"].eq("uncertain_hold"),
            df["is_manual_hold_epic"],
            df["is_science_label_v3"] & ~df["strict_score_pass"],
            df["is_unresolved_v3"].astype(str).eq("True") & df["needs_stage_f_validation"].astype(str).eq("True"),
            df["training_label_v3"].eq("uncertain_hold"),
            df["is_explicit_non_promotion_label"],
        ],
        [
            "strict_score_pass_known_science_like",
            "strict_score_pass_unresolved_candidate",
            "strict_score_pass_but_existing_uncertain_hold",
            "manual_hold_from_stage_f_v3_recovery_batch1",
            "known_science_like_missed_by_strict_rule_keep_for_audit",
            "unresolved_needs_stage_f_validation_not_strict_promoted",
            "existing_uncertain_hold_not_strict_promoted",
            "explicit_negative_or_artifact_not_promoted",
        ],
        default="not_strict_promoted",
    )
    df["strict_route"] = np.select(
        [
            df["strict_promoted"],
            df["is_manual_hold_epic"],
            df["training_label_v3"].eq("uncertain_hold"),
            df["is_science_label_v3"] & ~df["strict_score_pass"],
            df["is_unresolved_v3"].astype(str).eq("True") & df["needs_stage_f_validation"].astype(str).eq("True"),
        ],
        [
            "strict_promoted",
            "hold_queue",
            "hold_queue",
            "hold_queue",
            "needs_stage_f_validation",
        ],
        default="not_promoted_reject_or_archive",
    )
    return df.sort_values("hybrid_v3_strict_score", ascending=False).reset_index(drop=True)


def output_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "hybrid_v3_rank",
        "epic_id",
        "strict_route",
        "strict_policy_reason",
        "strict_promoted",
        "strict_score_pass",
        "hybrid_v3_strict_score",
        "hybrid_v3_strict_fp_penalty",
        "v3_base_score",
        "hybrid_score_v3",
        "training_label_v3",
        "science_binary_v3",
        "final_candidate_status",
        "is_unresolved_v3",
        "needs_stage_f_validation",
        "flux_p_science_like",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "stage_f_odd_even_penalty",
        "stage_f_oot_penalty",
        "stage_f_secondary_penalty",
        "stage_f_alias_penalty",
        "best_period_days",
        "phase_0_folded_path",
        "validation_summary_json_path",
    ]
    return [c for c in cols if c in df.columns]


def main() -> None:
    df = pd.read_csv(SCORES_CSV)
    scored = apply_frozen_rule(df)
    scored.insert(0, "strict_rank", np.arange(1, len(scored) + 1))

    cols = output_columns(scored)
    promoted = scored[scored["strict_promoted"]].copy()
    not_promoted = scored[~scored["strict_promoted"]].copy()
    missed = scored[scored["is_science_label_v3"] & ~scored["strict_promoted"]].copy()
    next_stage_f = scored[scored["strict_route"].eq("needs_stage_f_validation")].copy()

    promoted[["strict_rank"] + cols].to_csv(STRICT_PROMOTED_OUT, index=False)
    not_promoted[["strict_rank"] + cols].to_csv(STRICT_NOT_PROMOTED_OUT, index=False)
    missed[["strict_rank"] + cols].to_csv(MISSED_POSITIVE_AUDIT_OUT, index=False)
    next_stage_f[["strict_rank"] + cols].to_csv(NEXT_NEEDS_STAGE_F_OUT, index=False)

    print(f"strict rule: odd_even={ODD_EVEN_WEIGHT} oot={OOT_WEIGHT} secondary={SECONDARY_WEIGHT} alias={ALIAS_WEIGHT} min_score={MIN_CANDIDATE_SCORE}")
    print(f"wrote {STRICT_PROMOTED_OUT.relative_to(ROOT)} rows={len(promoted)}")
    print(f"wrote {STRICT_NOT_PROMOTED_OUT.relative_to(ROOT)} rows={len(not_promoted)}")
    print(f"wrote {MISSED_POSITIVE_AUDIT_OUT.relative_to(ROOT)} rows={len(missed)}")
    print(f"wrote {NEXT_NEEDS_STAGE_F_OUT.relative_to(ROOT)} rows={len(next_stage_f)}")
    print("\nroute counts")
    print(scored["strict_route"].value_counts().to_string())
    print("\nstrict promoted")
    print(promoted[["strict_rank", "epic_id", "hybrid_v3_strict_score", "training_label_v3", "strict_route"]].to_string(index=False))
    print("\nnext needs stage f")
    print(next_stage_f[["strict_rank", "epic_id", "hybrid_v3_strict_score", "v3_base_score", "hybrid_v3_strict_fp_penalty"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
