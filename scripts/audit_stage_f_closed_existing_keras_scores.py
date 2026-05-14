from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "k2_nocrop_flux_seed46_split303.best.keras"
LABELS_CSV = ROOT / "freezes" / "training_labels_v3_stage_f_closed_45.csv"
INFER_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"
INFER_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"
HYBRID_V2_CSV = ROOT / "k2_hybrid_candidate_score_v2.csv"

OUT_SCORES = ROOT / "freezes" / "stage_f_closed_45_existing_keras_scores.csv"
OUT_SWEEP = ROOT / "freezes" / "stage_f_closed_45_existing_keras_threshold_sweep.csv"
OUT_AUDIT = ROOT / "freezes" / "stage_f_closed_45_existing_keras_precision_audit.txt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def is_positive(series: pd.Series) -> pd.Series:
    return series.astype(str).eq("science_like")


def is_binary(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["science_like", "not_science_like"])


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    actual = y_true.astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(float(threshold), 2),
        "predicted_positives": int(pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def topk_precision(binary_ranked: pd.DataFrame, k: int) -> dict[str, Any]:
    top = binary_ranked.head(k)
    tp = int(top["target_science_like"].sum())
    return {
        "top_k": int(k),
        "rows_available": int(len(top)),
        "true_positives": tp,
        "false_positives": int(len(top) - tp),
        "precision": tp / len(top) if len(top) else 0.0,
        "recall": tp / int(binary_ranked["target_science_like"].sum()) if len(binary_ranked) else 0.0,
    }


def format_table(df: pd.DataFrame, cols: list[str], float_cols: list[str] | None = None) -> str:
    out = df[cols].copy()
    for col in float_cols or []:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
    return out.to_string(index=False)


def choose_operating_points(sweep: pd.DataFrame) -> dict[str, dict[str, Any]]:
    nonempty = sweep[sweep["predicted_positives"] > 0].copy()
    high_precision = nonempty.sort_values(
        ["precision", "true_positives", "recall", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    best_f1 = sweep.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    shortlist_pool = sweep[sweep["predicted_positives"].between(3, 10)].copy()
    if shortlist_pool.empty:
        shortlist_pool = nonempty
    clean_shortlist = shortlist_pool.sort_values(
        ["precision", "true_positives", "f1", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    return {
        "best_high_precision_threshold": high_precision.to_dict(),
        "best_balanced_f1_threshold": best_f1.to_dict(),
        "cleanest_candidate_shortlist_threshold": clean_shortlist.to_dict(),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels: {LABELS_CSV}")
    if not INFER_X.exists() or not INFER_META.exists():
        raise FileNotFoundError(f"Missing inference tensors: {INFER_X}, {INFER_META}")

    labels = pd.read_csv(LABELS_CSV)
    labels["epic_id"] = labels["epic_id"].astype(str)
    wanted = set(labels["epic_id"])

    meta = pd.read_parquet(INFER_META).reset_index(drop=True)
    meta["star_id"] = meta["star_id"].astype(str)
    mask = meta["star_id"].isin(wanted).to_numpy()
    idx = np.flatnonzero(mask)
    matched = set(meta.loc[mask, "star_id"])
    missing = sorted(wanted - matched)
    if missing:
        raise ValueError(f"Frozen-label EPICs missing from {rel(INFER_META)}: {missing}")

    x_mem = np.load(INFER_X, mmap_mode="r")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    expected = tuple(model.input_shape[1:])
    flux_shape = (int(x_mem.shape[1]), 1)
    if expected != flux_shape:
        raise ValueError(f"Model input {model.input_shape} does not match flux-only tensor shape {flux_shape}")

    x = np.asarray(x_mem[idx, :, :1], dtype=np.float32)
    preds = np.asarray(model.predict(x, batch_size=256, verbose=0)).reshape(-1).astype(float)

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
                "model_score": float(np.max(probs)),
                "model_score_top3_mean": float(np.mean(np.sort(probs)[-min(3, len(probs)) :])),
                "model_score_top10_mean": float(np.mean(np.sort(probs)[-min(10, len(probs)) :])),
                "model_score_mean": float(np.mean(probs)),
                "model_score_median": float(np.median(probs)),
                "flux_num_segments": int(len(grp)),
                "best_segment_start": int(best["start"]),
                "best_segment_end": int(best["end"]),
                "best_segment_mid_time": float(best["seg_mid_time"]),
                "best_segment_raw_score": float(best["segment_model_score"]),
            }
        )

    scores = labels.merge(pd.DataFrame(rows), on="epic_id", how="left", validate="one_to_one")
    scores["target_science_like"] = is_positive(scores["science_binary_v3"]).astype(int)
    scores["binary_eval_included"] = is_binary(scores["science_binary_v3"])
    scores["rank_by_model_score"] = scores["model_score"].rank(method="first", ascending=False).astype(int)

    if HYBRID_V2_CSV.exists():
        hybrid = pd.read_csv(HYBRID_V2_CSV)
        hybrid["epic_id"] = hybrid["epic_id"].astype(str)
        compare_cols = [
            "epic_id",
            "flux_p_science_like",
            "flux_p_top3_mean",
            "flux_p_top10_mean",
            "flux_num_segments",
            "flux_best_segment_start",
            "flux_best_segment_end",
            "flux_best_segment_mid_time",
        ]
        scores = scores.merge(
            hybrid[[c for c in compare_cols if c in hybrid.columns]],
            on="epic_id",
            how="left",
            validate="one_to_one",
        )
        scores["score_minus_existing_flux_p_science_like"] = (
            pd.to_numeric(scores["model_score"], errors="coerce")
            - pd.to_numeric(scores["flux_p_science_like"], errors="coerce")
        )
        scores["abs_score_delta_vs_existing_flux"] = scores["score_minus_existing_flux_p_science_like"].abs()
    else:
        scores["score_minus_existing_flux_p_science_like"] = np.nan
        scores["abs_score_delta_vs_existing_flux"] = np.nan

    binary = scores[scores["binary_eval_included"]].copy()
    binary = binary.sort_values("model_score", ascending=False).reset_index(drop=True)
    y_true = binary["target_science_like"].to_numpy(int)
    y_score = binary["model_score"].to_numpy(float)
    thresholds = np.round(np.arange(0.10, 0.951, 0.05), 2)
    sweep = pd.DataFrame([metrics_at_threshold(y_true, y_score, t) for t in thresholds])
    op = choose_operating_points(sweep)
    clean_threshold = float(op["cleanest_candidate_shortlist_threshold"]["threshold"])
    f1_threshold = float(op["best_balanced_f1_threshold"]["threshold"])

    scores["pred_at_clean_shortlist_threshold"] = scores["model_score"] >= clean_threshold
    scores["pred_at_best_f1_threshold"] = scores["model_score"] >= f1_threshold
    scores["correct_at_clean_shortlist_threshold"] = np.where(
        scores["binary_eval_included"],
        scores["pred_at_clean_shortlist_threshold"].astype(int).eq(scores["target_science_like"]).map(
            {True: "correct", False: "incorrect"}
        ),
        "excluded",
    )
    scores["correct_at_best_f1_threshold"] = np.where(
        scores["binary_eval_included"],
        scores["pred_at_best_f1_threshold"].astype(int).eq(scores["target_science_like"]).map(
            {True: "correct", False: "incorrect"}
        ),
        "excluded",
    )

    reason_cols = [
        "training_label_rule",
        "final_candidate_status",
        "review_bin",
        "source_batch",
        "stage_f_label",
        "stage_h_label",
        "visual_label",
    ]
    scores["notes_reason"] = scores[[c for c in reason_cols if c in scores.columns]].apply(
        lambda row: "; ".join(f"{k}={v}" for k, v in row.items() if pd.notna(v) and str(v).strip()),
        axis=1,
    )
    scores = scores.sort_values("model_score", ascending=False).reset_index(drop=True)
    scores["rank_by_model_score"] = np.arange(1, len(scores) + 1)

    sweep.to_csv(OUT_SWEEP, index=False)
    scores.to_csv(OUT_SCORES, index=False)

    topk = pd.DataFrame([topk_precision(binary, k) for k in [5, 10, 15, 20]])
    existing_delta = pd.to_numeric(scores["abs_score_delta_vs_existing_flux"], errors="coerce")
    max_delta = float(existing_delta.max()) if existing_delta.notna().any() else float("nan")
    mean_delta = float(existing_delta.mean()) if existing_delta.notna().any() else float("nan")
    close_match = bool(existing_delta.notna().all() and max_delta <= 1e-6)

    label_counts = scores["science_binary_v3"].value_counts(dropna=False).to_dict()
    ranked_cols = [
        "rank_by_model_score",
        "epic_id",
        "model_score",
        "training_label_v3",
        "science_binary_v3",
        "correct_at_clean_shortlist_threshold",
        "notes_reason",
    ]

    lines: list[str] = []
    lines.append("Stage F closed-45 existing Keras score audit")
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"Model exists: {MODEL_PATH.exists()} ({rel(MODEL_PATH)})")
    lines.append(f"Model input shape: {model.input_shape}")
    lines.append(f"Source tensor: {rel(INFER_X)} shape={tuple(x_mem.shape)}")
    lines.append("Input used for scoring: X_infer[matched_indices, :, :1] flux-only")
    lines.append(f"Frozen labels: {rel(LABELS_CSV)} rows={len(scores)}")
    lines.append(f"Label counts: {label_counts}")
    lines.append(
        "Binary rows used in threshold metrics: "
        f"{len(binary)} (positives={int(binary['target_science_like'].sum())}, "
        f"negatives={int((1 - binary['target_science_like']).sum())}, "
        f"excluded={int((~scores['binary_eval_included']).sum())})"
    )
    lines.append("")
    lines.append("Threshold sweep")
    lines.append("-" * 16)
    lines.append(
        format_table(
            sweep,
            [
                "threshold",
                "predicted_positives",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
            ],
            ["threshold", "precision", "recall", "f1"],
        )
    )
    lines.append("")
    lines.append("Top-k precision")
    lines.append("-" * 15)
    lines.append(
        format_table(
            topk,
            ["top_k", "rows_available", "true_positives", "false_positives", "precision", "recall"],
            ["precision", "recall"],
        )
    )
    lines.append("")
    lines.append("Recommended operating points")
    lines.append("-" * 28)
    for name, row in op.items():
        lines.append(
            f"{name}: threshold={float(row['threshold']):.2f}, "
            f"pred_pos={int(row['predicted_positives'])}, tp={int(row['true_positives'])}, "
            f"fp={int(row['false_positives'])}, fn={int(row['false_negatives'])}, "
            f"precision={float(row['precision']):.3f}, recall={float(row['recall']):.3f}, "
            f"f1={float(row['f1']):.3f}"
        )
    lines.append("")
    lines.append("Fresh score vs existing k2_hybrid_candidate_score_v2.csv flux_p_science_like")
    lines.append("-" * 73)
    lines.append(f"Matched closely: {close_match}")
    lines.append(f"Max absolute delta: {max_delta:.12g}")
    lines.append(f"Mean absolute delta: {mean_delta:.12g}")
    if close_match:
        lines.append("Conclusion: freshly computed scores match the existing flux_p_science_like values closely.")
    else:
        lines.append(
            "Conclusion: scores do not match exactly; a preprocessing/input mismatch would be likely if deltas are large. "
            "Inspect channel selection, tensor version, and segment aggregation before interpreting thresholds."
        )
    lines.append("")
    lines.append("Ranked EPIC list by model score")
    lines.append("-" * 32)
    lines.append(format_table(scores, ranked_cols, ["model_score"]))
    lines.append("")
    lines.append(f"Wrote: {rel(OUT_SCORES)}")
    lines.append(f"Wrote: {rel(OUT_SWEEP)}")
    lines.append(f"Wrote: {rel(OUT_AUDIT)}")

    audit_text = "\n".join(lines) + "\n"
    OUT_AUDIT.write_text(audit_text, encoding="utf-8")
    print(audit_text)


if __name__ == "__main__":
    main()
