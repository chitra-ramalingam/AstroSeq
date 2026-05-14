from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER_CSV = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"
FROZEN_LEDGER_CSV = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger_stage_f_closed.csv"
MANUAL_STAGE_F_V3_CSV = ROOT / "k2_stage_f_v3_manual_review_outcomes.csv"
LABELS_OUT = ROOT / "training_labels_v3.csv"
HYBRID_V2_CSV = ROOT / "k2_hybrid_candidate_score_v2.csv"
STAGE_D_RANKED_CSV = ROOT / "k2_stage_d_tier_a_results.csv"

LABEL_COUNTS_OUT = ROOT / "training_labels_v3_label_counts.csv"
CONFUSION_OUT = ROOT / "training_labels_v3_eval_confusion_matrix.csv"
PR_OUT = ROOT / "training_labels_v3_eval_precision_recall.csv"
TOP_FP_OUT = ROOT / "training_labels_v3_eval_top_false_positives.csv"
TOP_MISSED_OUT = ROOT / "training_labels_v3_eval_top_missed_positives.csv"
SCORED_OUT = ROOT / "training_labels_v3_eval_scored_rows.csv"


STATUS_TO_LABEL = {
    "promote_primary_candidate": "planet_like",
    "promote_candidate_alias_check": "candidate_like",
    "recovered_known_confirmed_planet": "planet_like",
    "recovered_known_unconfirmed_candidate": "candidate_like",
    "secondary_hold": "uncertain_hold",
    "hold_low_priority_candidate": "uncertain_hold",
    "deprioritize_after_manual_visual_review": "low_priority_negative",
    "reject_manual_false_positive": "noise_or_artifact",
    "reject_as_planet_binary_system": "binary_system",
    "reject_likely_eclipsing_binary": "binary_system",
    "reject_likely_variable_or_binary": "variable_or_artifact",
}

SCIENCE_LABELS = {"planet_like", "candidate_like"}
MANUAL_STAGE_F_TO_LABEL = {
    "promote_to_deeper_eval": "candidate_like",
    "hold_deeper_eval": "uncertain_hold",
    "reject_as_noise_or_artifact": "noise_or_artifact",
}


def science_class(label: str) -> str:
    return "science_like" if label in SCIENCE_LABELS else "not_science_like"


def confusion_counts(actual: pd.Series, predicted: pd.Series) -> dict[str, int]:
    actual_sci = actual.eq("science_like")
    pred_sci = predicted.eq("science_like")
    return {
        "tp": int((actual_sci & pred_sci).sum()),
        "fp": int((~actual_sci & pred_sci).sum()),
        "tn": int((~actual_sci & ~pred_sci).sum()),
        "fn": int((actual_sci & ~pred_sci).sum()),
    }


def precision_recall(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def build_labels() -> pd.DataFrame:
    ledger = pd.read_csv(LEDGER_CSV)
    ledger["training_label_v3"] = ledger["final_candidate_status"].map(STATUS_TO_LABEL)
    if ledger["training_label_v3"].isna().any():
        bad = ledger.loc[
            ledger["training_label_v3"].isna(),
            ["epic_id", "final_candidate_status", "review_bin"],
        ]
        raise ValueError(f"Unmapped v3 statuses:\n{bad.to_string(index=False)}")
    ledger["science_binary_v3"] = ledger["training_label_v3"].map(science_class)
    ledger["training_label_rule"] = "final_candidate_status:" + ledger["final_candidate_status"].astype(str)

    keep = [
        "epic_id",
        "training_label_v3",
        "science_binary_v3",
        "training_label_rule",
        "final_candidate_status",
        "review_bin",
        "source_batch",
        "best_period_days",
        "stage_f_label",
        "stage_h_label",
        "visual_label",
        "reviewer",
        "reviewed_at",
    ]
    out = ledger[[c for c in keep if c in ledger.columns]].copy()
    if MANUAL_STAGE_F_V3_CSV.exists():
        manual = pd.read_csv(MANUAL_STAGE_F_V3_CSV)
        if len(manual) > 0:
            manual = manual.copy()
            manual["training_label_v3"] = manual["manual_stage_f_label"].map(MANUAL_STAGE_F_TO_LABEL)
            if manual["training_label_v3"].isna().any():
                bad = manual.loc[
                    manual["training_label_v3"].isna(),
                    ["epic_id", "manual_stage_f_label", "manual_reason"],
                ]
                raise ValueError(f"Unmapped manual Stage F V3 labels:\n{bad.to_string(index=False)}")
            manual["science_binary_v3"] = manual["manual_science_binary"]
            manual["training_label_rule"] = "manual_stage_f_label:" + manual["manual_stage_f_label"].astype(str)
            manual_out = pd.DataFrame(
                {
                    "epic_id": manual["epic_id"].astype(str),
                    "training_label_v3": manual["training_label_v3"],
                    "science_binary_v3": manual["science_binary_v3"],
                    "training_label_rule": manual["training_label_rule"],
                    "final_candidate_status": manual["manual_stage_f_label"],
                    "review_bin": "manual_stage_f_v3_review",
                    "source_batch": manual.get("source_batch", "stage_f_v3_manual_review"),
                    "best_period_days": manual.get("best_period_days", ""),
                    "stage_f_label": manual.get("stage_f_label", ""),
                    "stage_h_label": "",
                    "visual_label": manual.get("manual_stage_f_label", ""),
                    "reviewer": manual.get("reviewer", ""),
                    "reviewed_at": manual.get("reviewed_at", ""),
                }
            )
            out = out.loc[~out["epic_id"].astype(str).isin(set(manual_out["epic_id"].astype(str)))].copy()
            out = pd.concat([out, manual_out], ignore_index=True)
    out.to_csv(LABELS_OUT, index=False)
    (
        out["training_label_v3"]
        .value_counts()
        .rename_axis("training_label_v3")
        .reset_index(name="count")
        .to_csv(LABEL_COUNTS_OUT, index=False)
    )
    return out


def load_scored(labels: pd.DataFrame) -> pd.DataFrame:
    hybrid = pd.read_csv(HYBRID_V2_CSV)
    stage_d = pd.read_csv(STAGE_D_RANKED_CSV).reset_index(drop=True)
    stage_d["stage_d_rank"] = np.arange(1, len(stage_d) + 1)

    df = labels.merge(
        hybrid[["epic_id", "flux_p_science_like", "candidate_score", "hybrid_rank"]],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        stage_d[["epic_id", "stage_d_rank"]],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )

    if df[["flux_p_science_like", "candidate_score", "stage_d_rank"]].isna().any().any():
        missing = df.loc[
            df[["flux_p_science_like", "candidate_score", "stage_d_rank"]].isna().any(axis=1),
            ["epic_id", "flux_p_science_like", "candidate_score", "stage_d_rank"],
        ]
        raise ValueError(f"Missing evaluation scores:\n{missing.to_string(index=False)}")

    n_science = int(df["science_binary_v3"].eq("science_like").sum())
    df["stage_d_rank_score"] = 1.0 - (df["stage_d_rank"].astype(float) - 1.0) / max(len(stage_d) - 1, 1)
    df["old_flux_pred_binary"] = np.where(df["flux_p_science_like"] >= 0.5, "science_like", "not_science_like")
    df["hybrid_v2_pred_binary"] = np.where(df["candidate_score"] >= 0.5, "science_like", "not_science_like")
    top_stage_d_epics = set(df.nsmallest(n_science, "stage_d_rank")["epic_id"])
    df["stage_d_rank_pred_binary"] = np.where(
        df["epic_id"].isin(top_stage_d_epics),
        "science_like",
        "not_science_like",
    )
    return df


def model_specs() -> list[dict[str, str]]:
    return [
        {
            "model": "old_flux_model",
            "score_col": "flux_p_science_like",
            "pred_col": "old_flux_pred_binary",
            "rank_col": "flux_p_science_like",
            "sort_ascending": "False",
            "decision_rule": "score >= 0.5",
        },
        {
            "model": "hybrid_score_v2",
            "score_col": "candidate_score",
            "pred_col": "hybrid_v2_pred_binary",
            "rank_col": "candidate_score",
            "sort_ascending": "False",
            "decision_rule": "score >= 0.5",
        },
        {
            "model": "stage_d_ranking",
            "score_col": "stage_d_rank_score",
            "pred_col": "stage_d_rank_pred_binary",
            "rank_col": "stage_d_rank",
            "sort_ascending": "True",
            "decision_rule": "top N ranked rows, where N=true science_like count",
        },
    ]


def evaluate(df: pd.DataFrame) -> None:
    confusion_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    false_positive_rows: list[pd.DataFrame] = []
    missed_rows: list[pd.DataFrame] = []

    for spec in model_specs():
        model = spec["model"]
        pred_col = spec["pred_col"]
        rank_col = spec["rank_col"]
        ascending = spec["sort_ascending"] == "True"
        counts = confusion_counts(df["science_binary_v3"], df[pred_col])
        pr = precision_recall(counts)
        pr_rows.append({"model": model, "decision_rule": spec["decision_rule"], **pr})
        for actual in ["science_like", "not_science_like"]:
            for predicted in ["science_like", "not_science_like"]:
                confusion_rows.append(
                    {
                        "model": model,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(((df["science_binary_v3"] == actual) & (df[pred_col] == predicted)).sum()),
                    }
                )

        fp = df[(df["science_binary_v3"] == "not_science_like") & (df[pred_col] == "science_like")].copy()
        fp = fp.sort_values(rank_col, ascending=ascending).head(10)
        fp.insert(0, "model", model)
        false_positive_rows.append(fp)

        missed = df[(df["science_binary_v3"] == "science_like") & (df[pred_col] == "not_science_like")].copy()
        missed = missed.sort_values(rank_col, ascending=not ascending).head(10)
        missed.insert(0, "model", model)
        missed_rows.append(missed)

    pd.DataFrame(confusion_rows).to_csv(CONFUSION_OUT, index=False)
    pd.DataFrame(pr_rows).to_csv(PR_OUT, index=False)
    common_cols = [
        "model",
        "epic_id",
        "training_label_v3",
        "science_binary_v3",
        "final_candidate_status",
        "flux_p_science_like",
        "candidate_score",
        "stage_d_rank",
        "stage_d_rank_score",
    ]
    pd.concat(false_positive_rows, ignore_index=True)[common_cols].to_csv(TOP_FP_OUT, index=False)
    pd.concat(missed_rows, ignore_index=True)[common_cols].to_csv(TOP_MISSED_OUT, index=False)
    df.to_csv(SCORED_OUT, index=False)


def main() -> None:
    if not FROZEN_LEDGER_CSV.exists():
        raise FileNotFoundError(f"Expected frozen ledger at {FROZEN_LEDGER_CSV}")
    labels = build_labels()
    scored = load_scored(labels)
    evaluate(scored)

    print(f"wrote {LABELS_OUT.relative_to(ROOT)} rows={len(labels)}")
    print(f"wrote {LABEL_COUNTS_OUT.relative_to(ROOT)}")
    print(f"wrote {CONFUSION_OUT.relative_to(ROOT)}")
    print(f"wrote {PR_OUT.relative_to(ROOT)}")
    print(f"wrote {TOP_FP_OUT.relative_to(ROOT)}")
    print(f"wrote {TOP_MISSED_OUT.relative_to(ROOT)}")
    print(f"wrote {SCORED_OUT.relative_to(ROOT)}")
    print("\nlabel counts")
    print(labels["training_label_v3"].value_counts().to_string())
    print("\nprecision/recall")
    print(pd.read_csv(PR_OUT).to_string(index=False))


if __name__ == "__main__":
    main()
