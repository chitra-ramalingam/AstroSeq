from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from src.Classifiers.Builders.BuilderHelper import BuilderHelper


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 10
    lr: float = 3e-4
    seed: int = 42


class K2TransitTrainerV2:
    def __init__(self, cfg: Optional[TrainConfig] = None, verbose: bool = True) -> None:
        self.cfg = cfg or TrainConfig()
        self.verbose = bool(verbose)
        self.helper = BuilderHelper()
        tf.random.set_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def declareHigherDimModel(self, w: int, channels: int = 1) -> tf.keras.Model:
       return self.helper.declareHigherDimModel(w, channels)

    def load_split(self, X_path: str | Path, meta_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        X = np.load(Path(X_path), mmap_mode="r")  # (N,w,c)
        df = pd.read_parquet(Path(meta_path))
        y = df["label"].astype(np.float32).to_numpy()

        if self.verbose:
            print(f"Loaded X={X.shape}  y_pos={int(y.sum())}/{len(y)} from {X_path}")
        return X, y

    def make_tfdata(self, X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            ds = ds.shuffle(min(len(y), 20000), seed=self.cfg.seed, reshuffle_each_iteration=True)
        ds = ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def compute_class_weight(self, y: np.ndarray) -> dict:
        y = y.astype(np.float32)
        pos = float(np.sum(y == 1.0))
        neg = float(np.sum(y == 0.0))
        if pos <= 0:
            return {0: 1.0, 1: 1.0}
        return {0: 1.0, 1: (neg / pos)}

    def train(
        self,
        X_train_path: str | Path, meta_train_path: str | Path,
        X_val_path: str | Path,   meta_val_path: str | Path,
        X_test_path: str | Path,  meta_test_path: str | Path,
        out_model_path: str | Path,
    ) -> tf.keras.Model:
        Xtr, ytr = self.load_split(X_train_path, meta_train_path)
        Xva, yva = self.load_split(X_val_path, meta_val_path)
        Xte, yte = self.load_split(X_test_path, meta_test_path)

        w = int(Xtr.shape[1])
        c = int(Xtr.shape[2])

        batch = min(256, len(Xtr)) if len(Xtr) else 1

        model = self.declareHigherDimModel(w=w, channels=c)

        ds_tr = self.make_tfdata(Xtr, ytr, training=True)
        ds_va = self.make_tfdata(Xva, yva, training=False)
        ds_te = self.make_tfdata(Xte, yte, training=False)

        class_weight = self.compute_class_weight(ytr)
        if self.verbose:
            print("class_weight:", class_weight)
            print("model input:", model.input_shape)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=2, factor=0.5),
        ]

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=self.cfg.epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        if len(Xte) == 0:
            print("[test] Skipping evaluation: empty test set")
        else:
            metrics = model.evaluate(ds_te, verbose=1)
            if self.verbose:
                print("Test:", dict(zip(model.metrics_names, metrics)))

        out_model_path = Path(out_model_path)
        out_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_model_path)
        if self.verbose:
            print("Saved:", out_model_path)

        return model
