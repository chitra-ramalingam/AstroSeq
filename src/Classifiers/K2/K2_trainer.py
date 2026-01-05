from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math
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
       return self.helper.declareLayerNormalizedModel(w = w, 
                                                channels = channels, 
                                                lr = self.cfg.lr, 
                                                dropout=.2, 
                                                label_smoothing=0.0)

    def load_split(self, X_path: str | Path, meta_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        X = np.load(Path(X_path), mmap_mode="r")  # (N,w,c)
        #X = X[:, :, :1]
        #X = self.min_center_crop(X, crop=256)
        df = pd.read_parquet(Path(meta_path))
        y = df["label"].astype(np.float32).to_numpy()
        
        if self.verbose:
            print(f"Loaded X={X.shape}  y_pos={int(y.sum())}/{len(y)} from {X_path}")
        return X, y

    def min_center_crop(self, X: np.ndarray, crop: int = 256, flux_ch: int = 0) -> np.ndarray:
        N, W, C = X.shape
        half = crop // 2

        flux = X[:, :, flux_ch]
        imin = np.argmin(flux, axis=1)
        starts = np.clip(imin - half, 0, W - crop)

        out = np.empty((N, crop, C), dtype=X.dtype)
        for i in range(N):
            s = int(starts[i])
            out[i] = X[i, s:s + crop, :]
        return out


    def make_tfdata_balanced(self, X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        if not training:
            return ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)

        # Split streams
        ds_pos = ds.filter(lambda x, yy: yy > 0.5).repeat()
        ds_neg = ds.filter(lambda x, yy: yy <= 0.5).repeat()

        # Mix 50/50 (you can change these weights later)
        ds_bal = tf.data.Dataset.sample_from_datasets(
            [ds_pos, ds_neg],
            weights=[0.5, 0.5],
            seed=self.cfg.seed,
            stop_on_empty_dataset=False,
        )

        # Optional shuffle AFTER mixing (keeps batches varied)
        ds_bal = ds_bal.shuffle(20000, seed=self.cfg.seed, reshuffle_each_iteration=True)

        return ds_bal.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)


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
        print("In K2_training: val_pos_frac:", yva.mean())

        w = int(Xtr.shape[1])
        c = int(Xtr.shape[2])

        batch = min(256, len(Xtr)) if len(Xtr) else 1

        model = self.declareHigherDimModel(w=w, channels=c)

        ds_tr = self.make_tfdata(Xtr, ytr, training=True)
        ds_va = self.make_tfdata(Xva, yva, training=False)
        ds_te = self.make_tfdata(Xte, yte, training=False)

        xb, yb = next(iter(ds_tr))
        print("train batch pos:", float(tf.reduce_mean(yb)))
        print("TRAIN y shape:", ytr.shape, "dtype:", ytr.dtype,
            "min/max:", ytr.min(), ytr.max())

        print("VAL   y shape:", yva.shape, "dtype:", yva.dtype,
            "min/max:", yva.min(), yva.max())

        class_weight = self.compute_class_weight(ytr)
        if self.verbose:
            print("class_weight:", class_weight)
            print("model input:", model.input_shape)

        xb, yb = next(iter(ds_tr))
        print("ds_tr batch shapes:", xb.shape, yb.shape)

        xb2, yb2 = next(iter(ds_va))
        print("ds_va batch shapes:", xb2.shape, yb2.shape)

        
        val_batch = next(iter(ds_va))
        print("val batch len:", len(val_batch))
        if len(val_batch) == 3:
            X, y, sw = val_batch
            print("sw min/max:", sw.min(), sw.max())

        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", patience=1, factor=0.5),
            ]
        steps_per_epoch = math.ceil(len(ytr) / self.cfg.batch_size)

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=None,
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

        p = model.predict(ds_va, verbose=0).ravel()
        print("pred min/max:", p.min(), p.max())
        print("pred pcts:", np.percentile(p, [0.1, 1, 50, 99, 99.9]))

        return model
    
    def make_tfdata(self, X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        # Ensure dtypes
        X = X.astype("float32", copy=False)
        y = y.astype("float32", copy=False)

        # CRITICAL: force y to (N, 1) to match model output (batch, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"Expected y shape (N,1) but got {y.shape}")

        ds = tf.data.Dataset.from_tensor_slices((X, y))

        if training:
            buf = min(len(y), 20000)
            ds = ds.shuffle(buf, seed=self.cfg.seed, reshuffle_each_iteration=True)

        ds = ds.batch(self.cfg.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

