from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILY = "k2_nocrop_flux_seed46_split303"
MODEL_PATH = ROOT / "models" / f"{MODEL_FAMILY}.best.keras"
LABELS_CSV = ROOT / "training_labels_v2.csv"
INFER_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"
INFER_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"

EVAL_CSV = ROOT / f"{MODEL_FAMILY}_baseline_eval.csv"
METRICS_JSON = ROOT / f"{MODEL_FAMILY}_baseline_metrics.json"
CONFUSION_CSV = ROOT / f"{MODEL_FAMILY}_baseline_confusion_matrix.csv"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def paths_for_patterns(patterns: list[str], roots: list[Path]) -> list[str]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(p for p in root.glob(pattern) if p.is_file())
    return [str(p.resolve()) for p in sorted(set(found))]


def discover_artifacts() -> dict[str, list[str]]:
    roots = [ROOT, ROOT / "models", ROOT / "splits", ROOT / "splits" / "seed303"]
    family_patterns = [
        f"*{MODEL_FAMILY}*",
        "*split303*",
        "*seed303*",
    ]
    return {
        "keras_files": paths_for_patterns([f"*{MODEL_FAMILY}*.keras"], [ROOT / "models", ROOT]),
        "history_files": paths_for_patterns(
            [f"*history*{p}*" for p in family_patterns] + [f"{p}*history*" for p in family_patterns],
            roots,
        ),
        "metrics_files": paths_for_patterns(
            [f"*metric*{p}*" for p in family_patterns] + [f"{p}*metric*" for p in family_patterns],
            roots,
        ),
        "config_files": paths_for_patterns(
            [f"*config*{p}*" for p in family_patterns]
            + [f"{p}*config*" for p in family_patterns]
            + [f"*{MODEL_FAMILY}*.json", f"*{MODEL_FAMILY}*.yaml", f"*{MODEL_FAMILY}*.yml"],
            roots,
        ),
        "scaler_normalizer_tensor_metadata_files": paths_for_patterns(
            [
                "*scaler*303*",
                "*normalizer*303*",
                "*shape*303*",
                "X_*.npy",
                "meta_*.parquet",
            ],
            [ROOT / "splits" / "seed303"],
        ),
    }


def binary_target(label: str) -> float | None:
    if label in {"planet_like", "candidate_like"}:
        return 1.0
    if label in {"noise_or_artifact", "binary_system"}:
        return 0.0
    return None


def safe_metric(fn, y_true: np.ndarray, y_score_or_pred: np.ndarray) -> float | None:
    try:
        return float(fn(y_true, y_score_or_pred))
    except ValueError:
        return None


def main() -> None:
    artifacts = discover_artifacts()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels: {LABELS_CSV}")
    if not INFER_X.exists() or not INFER_META.exists():
        raise FileNotFoundError(f"Missing inference tensors: {INFER_X}, {INFER_META}")

    labels = pd.read_csv(LABELS_CSV)
    infer_meta = pd.read_parquet(INFER_META).reset_index(drop=True)
    wanted = set(labels["epic_id"].astype(str))
    matched_mask = infer_meta["star_id"].astype(str).isin(wanted).to_numpy()
    matched_idx = np.flatnonzero(matched_mask)
    matched_epics = sorted(set(infer_meta.loc[matched_mask, "star_id"].astype(str)))
    missing_epics = sorted(wanted - set(matched_epics))
    if missing_epics:
        raise ValueError(f"Missing labeled EPICs from {INFER_META}: {missing_epics}")

    x_mem = np.load(INFER_X, mmap_mode="r")
    model = tf.keras.models.load_model(MODEL_PATH)
    if tuple(model.input_shape[1:]) != (int(x_mem.shape[1]), 1):
        raise ValueError(f"Model input {model.input_shape} does not match X flux-only shape {(x_mem.shape[1], 1)}")

    x_eval = np.asarray(x_mem[matched_idx, :, :1], dtype=np.float32)
    p = model.predict(x_eval, batch_size=256, verbose=0).reshape(-1).astype(float)

    seg = infer_meta.loc[matched_mask].copy().reset_index(drop=True)
    seg["segment_probability"] = p
    rows: list[dict[str, Any]] = []
    for epic_id, grp in seg.groupby("star_id", sort=True):
        order = np.argsort(-grp["segment_probability"].to_numpy())
        best = grp.iloc[int(order[0])]
        probs = grp["segment_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "epic_id": epic_id,
                "num_segments": int(len(grp)),
                "p_max": float(np.max(probs)),
                "p_top3_mean": float(np.mean(np.sort(probs)[-min(3, len(probs)) :])),
                "p_top10_mean": float(np.mean(np.sort(probs)[-min(10, len(probs)) :])),
                "p_mean": float(np.mean(probs)),
                "p_median": float(np.median(probs)),
                "best_segment_start": int(best["start"]),
                "best_segment_end": int(best["end"]),
                "best_segment_mid_time": float(best["seg_mid_time"]),
            }
        )

    eval_df = pd.DataFrame(rows)
    eval_df = labels.merge(eval_df, on="epic_id", how="left", validate="one_to_one")
    eval_df["binary_true"] = eval_df["training_label_v2"].map(binary_target)
    eval_df["binary_eval_aligned"] = eval_df["binary_true"].notna()
    eval_df["binary_pred_at_0p5"] = np.where(eval_df["p_max"] >= 0.5, 1.0, 0.0)
    eval_df["binary_true_label"] = eval_df["binary_true"].map({1.0: "science_like", 0.0: "not_science_like"})
    eval_df["binary_pred_label_at_0p5"] = eval_df["binary_pred_at_0p5"].map(
        {1.0: "science_like", 0.0: "not_science_like"}
    )
    eval_df = eval_df.sort_values("p_max", ascending=False)
    eval_df.insert(0, "baseline_rank", np.arange(1, len(eval_df) + 1))
    eval_df.to_csv(EVAL_CSV, index=False)

    aligned = eval_df[eval_df["binary_eval_aligned"]].copy()
    y_true = aligned["binary_true"].to_numpy(dtype=float)
    y_pred = aligned["binary_pred_at_0p5"].to_numpy(dtype=float)
    y_score = aligned["p_max"].to_numpy(dtype=float)

    cm_labels = ["not_science_like", "science_like"]
    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0])
    cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in cm_labels], columns=[f"pred_{x}" for x in cm_labels])
    cm_df.to_csv(CONFUSION_CSV)

    metrics: dict[str, Any] = {
        "model_family": MODEL_FAMILY,
        "loaded_model_path": str(MODEL_PATH.resolve()),
        "labels_csv": str(LABELS_CSV.resolve()),
        "inference_x": str(INFER_X.resolve()),
        "inference_meta": str(INFER_META.resolve()),
        "outputs": {
            "eval_csv": str(EVAL_CSV.resolve()),
            "metrics_json": str(METRICS_JSON.resolve()),
            "confusion_matrix_csv": str(CONFUSION_CSV.resolve()),
        },
        "artifact_scan": artifacts,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "source_tensor_shape": list(x_mem.shape),
        "tensor_input_used": "X_infer[:, :, :1] flux-only, matching K2TransitTrainerV2.load_split",
        "n_labeled_epics": int(len(labels)),
        "n_matched_labeled_epics": int(eval_df["p_max"].notna().sum()),
        "missing_labeled_epics": missing_epics,
        "label_counts": labels["training_label_v2"].value_counts().to_dict(),
        "binary_alignment": {
            "positive_labels": ["planet_like", "candidate_like"],
            "negative_labels": ["noise_or_artifact", "binary_system"],
            "excluded_labels": ["uncertain_hold"],
            "threshold": 0.5,
            "n_aligned": int(len(aligned)),
            "n_excluded": int((~eval_df["binary_eval_aligned"]).sum()),
        },
        "binary_metrics_on_aligned_labels": {
            "accuracy_at_0p5": safe_metric(accuracy_score, y_true, y_pred),
            "balanced_accuracy_at_0p5": safe_metric(balanced_accuracy_score, y_true, y_pred),
            "roc_auc_p_max": safe_metric(roc_auc_score, y_true, y_score),
            "average_precision_p_max": safe_metric(average_precision_score, y_true, y_score),
            "confusion_matrix_labels": cm_labels,
            "confusion_matrix": cm.tolist(),
        },
        "top_labeled_by_p_max": eval_df[
            ["baseline_rank", "epic_id", "training_label_v2", "p_max", "p_top3_mean", "num_segments"]
        ].head(10).to_dict(orient="records"),
        "notes": [
            "No retraining was performed.",
            "The model is binary sigmoid; multiclass training_label_v2 values only partially align, so confusion is binary and excludes uncertain_hold.",
            "EPIC-level score uses max segment probability, matching the existing best-window-per-star inference ranking style.",
        ],
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
