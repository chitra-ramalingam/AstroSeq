from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/k2_nocrop_flux_seed46_split303.best.keras"
DEFAULT_LABELS = ROOT / "docs/phase2/phase2_label_inventory.csv"
DEFAULT_X = ROOT / "splits/infer_c5/X_infer.npy"
DEFAULT_META = ROOT / "splits/infer_c5/meta_infer.parquet"
DEFAULT_NPZ = ROOT / "data/phase2/phase2_cnn_embeddings.npz"
DEFAULT_SCORES = ROOT / "data/phase2/phase2_cnn_scores.csv"
EMBEDDING_LAYER = "global_average_pooling1d_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_epics(labels: pd.DataFrame) -> list[str]:
    safe = labels["safe_for_supervised_training"].astype(str).str.lower().eq("true")
    accessible = labels["has_accessible_light_curve_tensor"].astype(str).str.lower().eq("true")
    return sorted(labels.loc[safe & accessible, "epic_id"].astype(str).unique())


def aggregate_star_outputs(
    meta: pd.DataFrame,
    probabilities: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select the max-probability segment and its aligned embedding per EPIC."""
    if len(meta) != len(probabilities) or len(meta) != len(embeddings):
        raise ValueError("meta, probabilities, and embeddings must have equal row counts")
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    work = meta.reset_index(drop=True).copy()
    work["_p"] = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    for epic_id, group in work.groupby("star_id", sort=True):
        positions = group.index.to_numpy(dtype=int)
        local = int(np.nanargmax(group["_p"].to_numpy(dtype=float)))
        pos = int(positions[local])
        row = group.loc[pos]
        rows.append({
            "epic_id": str(epic_id),
            "cnn_probability": float(row["_p"]),
            "cnn_embedding_layer": EMBEDDING_LAYER,
            "cnn_embedding_dim": int(embeddings.shape[1]),
            "cnn_embedding_aggregation": "max_probability_segment",
            "cnn_segment_count": int(len(group)),
            "cnn_best_segment_start": int(row["start"]),
            "cnn_best_segment_end": int(row["end"]),
            "cnn_best_segment_mid_time": float(row["seg_mid_time"]),
        })
        vectors.append(np.asarray(embeddings[pos], dtype=np.float32))
    return pd.DataFrame(rows), np.vstack(vectors).astype(np.float32)


def export_embeddings(
    *, labels_path: Path, model_path: Path, x_path: Path, meta_path: Path,
    output_npz: Path, output_scores: Path, batch_size: int = 512,
) -> tuple[pd.DataFrame, np.ndarray]:
    import tensorflow as tf

    labels = pd.read_csv(labels_path)
    targets = eligible_epics(labels)
    meta_all = pd.read_parquet(meta_path, columns=["star_id", "start", "end", "seg_mid_time"])
    meta_all["star_id"] = meta_all["star_id"].astype(str)
    mask = meta_all["star_id"].isin(targets).to_numpy()
    matched = set(meta_all.loc[mask, "star_id"])
    missing = sorted(set(targets) - matched)
    if missing:
        raise RuntimeError(f"Eligible EPICs missing inference tensors: {missing}")
    indices = np.flatnonzero(mask)
    meta = meta_all.loc[mask].reset_index(drop=True)
    x_mem = np.load(x_path, mmap_mode="r")

    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (512, 1):
        raise ValueError(f"Unexpected frozen model input: {model.input_shape}")
    model.trainable = False
    encoder = tf.keras.Model(model.input, model.get_layer(EMBEDDING_LAYER).output)
    encoder.trainable = False
    if int(encoder.output_shape[-1]) != 128:
        raise ValueError(f"Unexpected embedding dimension: {encoder.output_shape}")

    probs = np.empty(len(indices), dtype=np.float32)
    embeds = np.empty((len(indices), 128), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        x = np.asarray(x_mem[indices[start:stop], :, :1], dtype=np.float32)
        # Explicit inference mode: dropout is disabled and no weights can update.
        probs[start:stop] = np.asarray(model(x, training=False)).reshape(-1)
        embeds[start:stop] = np.asarray(encoder(x, training=False), dtype=np.float32)

    scores, star_embeddings = aggregate_star_outputs(meta, probs, embeds)
    scores["cnn_model_path"] = model_path.relative_to(ROOT).as_posix()
    scores["cnn_model_sha256"] = sha256(model_path)
    scores["cnn_inference_training"] = False
    scores["cnn_weights_modified"] = False
    class_map = labels.set_index("epic_id")["current_final_label"].astype(str)
    scores["normalized_label"] = scores["epic_id"].map(class_map)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        epic_id=scores["epic_id"].to_numpy(dtype=str),
        cnn_probability=scores["cnn_probability"].to_numpy(np.float32),
        embedding=star_embeddings,
        embedding_layer=np.array(EMBEDDING_LAYER),
        model_sha256=np.array(sha256(model_path)),
    )
    scores.to_csv(output_scores, index=False)
    return scores, star_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen Phase 2 CNN probabilities and 128-D embeddings.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--x", type=Path, default=DEFAULT_X)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores, embeddings = export_embeddings(
        labels_path=args.labels, model_path=args.model, x_path=args.x, meta_path=args.meta,
        output_npz=args.output_npz, output_scores=args.output_scores, batch_size=args.batch_size,
    )
    print(f"exported_epics={len(scores)} embedding_shape={embeddings.shape}")
    print(scores.groupby("normalized_label").size().to_string())


if __name__ == "__main__":
    main()
