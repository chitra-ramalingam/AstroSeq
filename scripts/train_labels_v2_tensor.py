from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LEDGER_CSV = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"
FEATURES_CSV = ROOT / "k2_stage_d_tier_a_results.csv"
STAGE_D_RANKED_CSV = ROOT / "k2_stage_d_pass_ranked.csv"

LABELS_OUT = ROOT / "training_labels_v2.csv"
METRICS_OUT = ROOT / "training_labels_v2_metrics.json"
CONFUSION_OUT = ROOT / "training_labels_v2_confusion_matrix.csv"
PROBS_OUT = ROOT / "training_labels_v2_probability_scores.csv"
TOP50_OUT = ROOT / "training_labels_v2_top50_unresolved_epic_predictions.csv"
STAGE_D_COMPARE_OUT = ROOT / "training_labels_v2_stage_d_comparison.csv"
MODEL_OUT = ROOT / "models" / "training_labels_v2_tensor.keras"


FEATURE_COLUMNS = [
    "n_events_long_good",
    "n_events_ge_10_cadences",
    "max_shape_score",
    "spike_fraction_2cadence",
    "depth_ratio",
    "n_events_raw",
    "n_events_after_filters",
    "best_period_days",
    "period_support_count",
    "event_family_count",
    "folded_depth_consistency",
    "duration_consistency",
    "odd_even_depth_delta",
    "cluster_center_phase",
    "n_predicted",
    "n_covered",
    "coverage_rate",
    "hit_rate_snr",
    "hit_rate_shape",
    "soft_hit_rate",
    "cache_hits",
    "cache_misses",
    "downloads_done",
    "validations_run",
]


def map_label(row: pd.Series) -> tuple[str, str]:
    status = str(row.get("final_candidate_status", "")).strip()
    review_bin = str(row.get("review_bin", "")).strip()

    if status == "recovered_known_confirmed_planet":
        return "planet_like", "final_candidate_status:recovered_known_confirmed_planet"
    if status == "recovered_known_unconfirmed_candidate":
        return "candidate_like", "final_candidate_status:recovered_known_unconfirmed_candidate"
    if status == "promote_primary_candidate":
        return "planet_like", "final_candidate_status:promote_primary_candidate"
    if status == "promote_candidate_alias_check":
        return "candidate_like", "final_candidate_status:promote_candidate_alias_check"
    if status.startswith("hold_"):
        return "uncertain_hold", "final_candidate_status:hold_*"
    if status == "reject_as_planet_binary_system":
        return "binary_system", "final_candidate_status:reject_as_planet_binary_system"
    if status.startswith("manual_false_positive_"):
        return "noise_or_artifact", "final_candidate_status:manual_false_positive_*"
    if status.startswith("reject_low_confidence_"):
        return "noise_or_artifact", "final_candidate_status:reject_low_confidence_*"

    # The local ledger stores several requested wildcard concepts in review_bin
    # or with a slightly different status spelling. Keep these explicit.
    if review_bin.startswith("manual_false_positive") or status == "reject_manual_false_positive":
        return "noise_or_artifact", "review_bin/manual_status:manual_false_positive_*"
    if "hold" in status or "hold" in review_bin:
        return "uncertain_hold", "status_or_review_bin:hold"
    if status.startswith("deprioritize_") or review_bin.startswith("deprioritized_"):
        return "uncertain_hold", "status_or_review_bin:deprioritized_hold"

    return "unmapped", "unmapped"


def build_labels() -> pd.DataFrame:
    ledger = pd.read_csv(LEDGER_CSV)
    mapped = ledger.apply(map_label, axis=1, result_type="expand")
    ledger["training_label_v2"] = mapped[0]
    ledger["training_label_rule"] = mapped[1]
    if (ledger["training_label_v2"] == "unmapped").any():
        bad = ledger.loc[
            ledger["training_label_v2"] == "unmapped",
            ["epic_id", "final_candidate_status", "review_bin"],
        ]
        raise ValueError(f"Unmapped ledger labels:\n{bad.to_string(index=False)}")

    keep = [
        "epic_id",
        "training_label_v2",
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
    out.to_csv(LABELS_OUT, index=False)
    return out


def deterministic_split(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(46)
    val_idx: list[int] = []
    for _, grp in labels.groupby("training_label_v2", sort=True):
        indices = grp.index.to_numpy()
        if len(indices) >= 2:
            val_idx.append(int(rng.choice(indices, size=1, replace=False)[0]))

    val = np.array(sorted(set(val_idx)), dtype=int)
    train = np.array([i for i in labels.index.to_numpy() if i not in set(val)], dtype=int)
    return train, val


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(46)
    inputs = tf.keras.Input(shape=(n_features,), name="stage_d_features")
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def metrics_for(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict[str, Any]:
    y_pred = probs.argmax(axis=1)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else None,
    }
    for i, label in enumerate(labels):
        mask = y_true == i
        out[f"support_{label}"] = int(mask.sum())
        out[f"mean_p_true_for_{label}"] = float(probs[mask, i].mean()) if mask.any() else None
    return out


def main() -> None:
    labels = build_labels()
    features = pd.read_csv(FEATURES_CSV)
    stage_d_ranked = pd.read_csv(STAGE_D_RANKED_CSV)
    stage_d_ranked = stage_d_ranked.reset_index(drop=True)
    stage_d_ranked["stage_d_rank"] = np.arange(1, len(stage_d_ranked) + 1)

    train_df = features.merge(
        labels[["epic_id", "training_label_v2"]],
        on="epic_id",
        how="inner",
        validate="one_to_one",
    )
    if len(train_df) != len(labels):
        missing = sorted(set(labels["epic_id"]) - set(train_df["epic_id"]))
        raise ValueError(f"Missing feature rows for labeled EPICs: {missing}")

    feature_matrix = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    labeled_feature_matrix = train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_labeled = scaler.fit_transform(imputer.fit_transform(labeled_feature_matrix)).astype("float32")
    x_all = scaler.transform(imputer.transform(feature_matrix)).astype("float32")

    encoder = LabelEncoder()
    y = encoder.fit_transform(train_df["training_label_v2"])
    class_names = encoder.classes_.tolist()

    train_idx, val_idx = deterministic_split(train_df)
    model = build_model(n_features=x_labeled.shape[1], n_classes=len(class_names))

    counts = np.bincount(y[train_idx], minlength=len(class_names)).astype(float)
    class_weight = {
        i: float(len(train_idx) / (len(class_names) * max(counts[i], 1.0)))
        for i in range(len(class_names))
    }

    history = model.fit(
        x_labeled[train_idx],
        y[train_idx],
        validation_data=(x_labeled[val_idx], y[val_idx]),
        epochs=250,
        batch_size=8,
        verbose=0,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True,
            )
        ],
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)

    p_train = model.predict(x_labeled[train_idx], verbose=0)
    p_val = model.predict(x_labeled[val_idx], verbose=0)
    p_labeled = model.predict(x_labeled, verbose=0)
    p_all = model.predict(x_all, verbose=0)

    train_df = train_df.copy()
    train_df["split_v2"] = "train"
    train_df.loc[train_df.index[val_idx], "split_v2"] = "validation"

    cm = confusion_matrix(y[val_idx], p_val.argmax(axis=1), labels=np.arange(len(class_names)))
    cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(CONFUSION_OUT)

    all_probs = features[["epic_id", "promote_tier", "stage_d_label"]].copy()
    for i, c in enumerate(class_names):
        all_probs[f"p_{c}"] = p_all[:, i]
    all_probs["p_science_like"] = all_probs.get("p_planet_like", 0.0) + all_probs.get("p_candidate_like", 0.0)
    all_probs["predicted_label_v2"] = encoder.inverse_transform(p_all.argmax(axis=1))
    all_probs["max_probability"] = p_all.max(axis=1)

    labeled_meta = labels[["epic_id", "training_label_v2", "final_candidate_status", "review_bin"]]
    all_probs = all_probs.merge(labeled_meta, on="epic_id", how="left")
    all_probs = all_probs.merge(stage_d_ranked[["epic_id", "stage_d_rank"]], on="epic_id", how="left")
    all_probs = all_probs.sort_values(["p_science_like", "max_probability"], ascending=False)
    all_probs.insert(0, "model_rank_all", np.arange(1, len(all_probs) + 1))
    all_probs.to_csv(PROBS_OUT, index=False)

    unresolved = all_probs[all_probs["training_label_v2"].isna()].copy()
    top50 = unresolved.head(50).copy()
    top50.insert(0, "unresolved_model_rank", np.arange(1, len(top50) + 1))
    top50.to_csv(TOP50_OUT, index=False)

    compare = all_probs[all_probs["stage_d_rank"].notna()].copy()
    compare["stage_d_rank"] = compare["stage_d_rank"].astype(int)
    compare["rank_delta_model_minus_stage_d"] = compare["model_rank_all"] - compare["stage_d_rank"]
    compare = compare.sort_values("stage_d_rank")
    compare.to_csv(STAGE_D_COMPARE_OUT, index=False)

    train_probs_in_original_order = p_labeled
    train_metrics = metrics_for(y[train_idx], p_train, class_names)
    val_metrics = metrics_for(y[val_idx], p_val, class_names)
    labeled_metrics = metrics_for(y, train_probs_in_original_order, class_names)

    metrics = {
        "input_ledger_csv": str(LEDGER_CSV.relative_to(ROOT)),
        "feature_csv": str(FEATURES_CSV.relative_to(ROOT)),
        "model_csv_outputs": {
            "labels": str(LABELS_OUT.relative_to(ROOT)),
            "metrics": str(METRICS_OUT.relative_to(ROOT)),
            "confusion_matrix": str(CONFUSION_OUT.relative_to(ROOT)),
            "probability_scores": str(PROBS_OUT.relative_to(ROOT)),
            "top50_unresolved": str(TOP50_OUT.relative_to(ROOT)),
            "stage_d_comparison": str(STAGE_D_COMPARE_OUT.relative_to(ROOT)),
            "model": str(MODEL_OUT.relative_to(ROOT)),
        },
        "features": FEATURE_COLUMNS,
        "classes": class_names,
        "label_counts": labels["training_label_v2"].value_counts().to_dict(),
        "train_epics": train_df.loc[train_idx, "epic_id"].tolist(),
        "validation_epics": train_df.loc[val_idx, "epic_id"].tolist(),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "labeled_fit_metrics": labeled_metrics,
        "epochs_run": int(len(history.history["loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "n_unresolved_scored": int(len(unresolved)),
        "n_stage_d_ranked_compared": int(len(compare)),
        "note": (
            "Validation is a deterministic tiny holdout: one example from each class with at least two labels. "
            "binary_system has only one label and is train-only."
        ),
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
