from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import precision_recall_curve

from src.Classifiers.Builders.BuilderHelper import BuilderHelper


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 10
    lr: float = 3e-4
    seed: int = 42

    # Suggested: make cropping a first-class toggle
    crop_len: Optional[int] = None   # set to 128 for tighter focus, or None to disable


class K2TransitTrainerV2:
    def __init__(self, cfg: Optional[TrainConfig] = None, verbose: bool = True) -> None:
        self.cfg = cfg or TrainConfig()
        self.verbose = bool(verbose)
        self.helper = BuilderHelper()
        tf.random.set_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def declareHigherDimModel(self, w: int, channels: int = 1) -> tf.keras.Model:
        return self.helper.declareLayerNormalizedModel(
            w=w,
            channels=channels,
            lr=self.cfg.lr,
            dropout=0.2,
            label_smoothing=0.0
        )

    def load_split(self, X_path: str | Path, meta_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        X = np.load(Path(X_path), mmap_mode="r")  # (N,w,c)
        X = X[:, :, :1]  # flux only

        df = pd.read_parquet(Path(meta_path))
        y = df["label"].astype(np.float32).to_numpy()  # keep 1D here; we’ll shape later in tf.data

        # Suggested: optional center crop around minimum flux
        if self.cfg.crop_len is not None:
            if "center_idx" in df.columns:
                centers = df["center_idx"].to_numpy()
                X = self.center_crop_at_index(X, centers, crop=int(self.cfg.crop_len))
            else:
                X = self.min_center_crop(X, crop=int(self.cfg.crop_len))  # fallback

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
        """
        Balanced 50/50 sampling for training.
        Critical fixes:
          - cast x/y to float32
          - ensure y is (batch, 1) via expand_dims
        """
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        # Always enforce dtypes and label rank
        ds = ds.map(
            lambda x, yy: (tf.cast(x, tf.float32), tf.expand_dims(tf.cast(yy, tf.float32), axis=-1)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if not training:
            return ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)

        # Split streams (note yy is now shape (1,), so compare yy[0])
        ds_pos = ds.filter(lambda x, yy: yy[0] > 0.5).repeat()
        ds_neg = ds.filter(lambda x, yy: yy[0] <= 0.5).repeat()

        ds_bal = tf.data.Dataset.sample_from_datasets(
            [ds_pos, ds_neg],
            weights=[0.5, 0.5],
            seed=self.cfg.seed,
            stop_on_empty_dataset=False,
        )

        ds_bal = ds_bal.shuffle(20000, seed=self.cfg.seed, reshuffle_each_iteration=True)
        return ds_bal.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    def make_tfdata(self, X: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
        # Ensure dtypes
        X = X.astype("float32", copy=False)
        y = y.astype("float32", copy=False)

        # Force y to (N, 1) to match model output (batch, 1)
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
        print("In K2_training: val_pos_frac:", float(np.mean(yva)))

        w = int(Xtr.shape[1])
        c = int(Xtr.shape[2])
        # Diagnostics: check how often the min-flux index lands near the center
        imin = np.argmin(Xva[:, :, 0], axis=1)
        center = Xva.shape[1] // 2
        win = 16  # +/- 16 samples around center

        def frac_center(mask: np.ndarray) -> float:
            m = imin[mask]
            return float(np.mean((m >= center - win) & (m <= center + win))) if len(m) else float("nan")

        yva_1d = yva.reshape(-1)
        print("imin median:", int(np.median(imin)))
        print("centered pos:", frac_center(yva_1d == 1.0))
        print("centered neg:", frac_center(yva_1d == 0.0))

        model = self.declareHigherDimModel(w=w, channels=c)

        ds_tr = self.make_tfdata_balanced(Xtr, ytr, training=True)
        ds_va = self.make_tfdata(Xva, yva, training=False)
        ds_te = self.make_tfdata(Xte, yte, training=False)

        xb, yb = next(iter(ds_tr))
        print("train batch pos:", float(tf.reduce_mean(yb)))
        print("ds_tr batch shapes:", xb.shape, yb.shape)

        xb2, yb2 = next(iter(ds_va))
        print("ds_va batch shapes:", xb2.shape, yb2.shape)

        out_model_path = Path(out_model_path)
        out_model_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(out_model_path.with_suffix(".best.keras"))


        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_pr_auc", mode="max",
                save_best_only=True, save_weights_only=False, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", patience=3, factor=0.5, min_lr=1e-5, verbose=1
            ),
        ]

        # With repeat()+balanced sampling, you must define steps_per_epoch explicitly.
        steps_per_epoch = max(1, math.ceil(len(ytr) / self.cfg.batch_size))

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=None,  # already balancing via sampling
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate test set
        if len(Xte) == 0:
            print("[test] Skipping evaluation: empty test set")
        else:
            metrics = model.evaluate(ds_te, verbose=0, return_dict=True)
            print("Test (restored best):", metrics)

            if Path(ckpt_path).exists():
                best_model = tf.keras.models.load_model(ckpt_path)
                best_metrics = best_model.evaluate(ds_te, verbose=0, return_dict=True)
                print("Test (checkpoint best):", best_metrics)


        # Save model
        out_model_path = Path(out_model_path)
        out_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_model_path)
        if self.verbose:
            print("Saved:", out_model_path)

        # Validation predictions + diagnostics
        p = model.predict(ds_va, verbose=0).ravel().astype(np.float32)
        self.print_metrics(yva=yva, p=p)
        print("pred min/max:", float(p.min()), float(p.max()))
        print("pred pcts:", np.percentile(p, [0.1, 1, 50, 99, 99.9]))

        val_pos = float(np.mean(yva_1d))
        print("val_pos_frac:", val_pos, "baseline_pr_auc~", val_pos)

        if np.any(yva_1d == 1):
            print("pred mean pos:", float(p[yva_1d == 1].mean()))
            print("pred median pos:", float(np.median(p[yva_1d == 1])))
        else:
            print("pred mean pos:", None)
            print("pred median pos:", None)

        if np.any(yva_1d == 0):
            print("pred mean neg:", float(p[yva_1d == 0].mean()))
            print("pred median neg:", float(np.median(p[yva_1d == 0])))
        else:
            print("pred mean neg:", None)
            print("pred median neg:", None)

        # Top-K precision
        k = 50
        top = np.argsort(-p)[:k]
        print("top50 precision:", float(yva_1d[top].mean()), "expected baseline:", float(yva_1d.mean()))

        def topk_prec(score: np.ndarray, kk: int = 50) -> float:
            topk = np.argsort(-score)[:kk]
            return float(yva_1d[topk].mean())

        print("top50 prec using p   :", topk_prec(p, 50))
        print("top50 prec using 1-p :", topk_prec(1.0 - p, 50))
        print("baseline:", float(yva_1d.mean()))

        # PR-AUC checks (direct + inverted)
        def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = tf.keras.metrics.AUC(curve="PR")
            m.update_state(y_true, y_pred)
            return float(m.result().numpy())

        y_float = yva_1d.astype(np.float32)
        print("PR(y, p)      =", pr_auc(y_float, p))
        print("PR(y, 1-p)    =", pr_auc(y_float, 1.0 - p))
        print("PR(1-y, p)    =", pr_auc(1.0 - y_float, p))
        print("PR(1-y, 1-p)  =", pr_auc(1.0 - y_float, 1.0 - p))

        return model

    def print_metrics(self, yva, p):

        y = yva.reshape(-1).astype(np.int32)
        p = p.reshape(-1).astype(np.float32)

        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        best = f1.argmax()
        best_thr = thr[max(best-1, 0)]  # thr has len-1 vs prec/rec

        print("best F1:", float(f1[best]))
        print("best threshold:", float(best_thr))
        print("precision@best:", float(prec[best]), "recall@best:", float(rec[best]))

    def _eval_pr_auc(self, model: tf.keras.Model, ds: tf.data.Dataset) -> float:
        m = tf.keras.metrics.AUC(curve="PR")
        for xb, yb in ds:
            p = model(xb, training=False)
            m.update_state(yb, p)
        return float(m.result().numpy())


    def _small_loss_epoch(
        self,
        model: tf.keras.Model,
        opt: tf.keras.optimizers.Optimizer,
        ds_tr: tf.data.Dataset,
        steps_per_epoch: int,
        keep_frac: float = 0.8,
        batch_size: int = 256,
    ) -> None:
        # collect ONE epoch worth of batches into memory
        Xs, Ys = [], []
        it = iter(ds_tr)
        for _ in range(steps_per_epoch):
            xb, yb = next(it)
            Xs.append(xb.numpy())
            Ys.append(yb.numpy())
        X = np.concatenate(Xs, axis=0).astype("float32", copy=False)
        Y = np.concatenate(Ys, axis=0).astype("float32", copy=False)

        # forward pass to score per-sample losses
        P = model.predict(X, batch_size=batch_size, verbose=0)
        loss_vec = tf.keras.losses.binary_crossentropy(Y, P).numpy().reshape(-1)
          
        y_flat = Y.reshape(-1)

        pos_idx = np.where(y_flat > 0.5)[0]
        neg_idx = np.where(y_flat <= 0.5)[0]

        # keep ALL positives
        keep_pos = pos_idx

        # keep a fraction of negatives (small-loss negatives for noise filtering)
        kneg = max(1, int(len(neg_idx) * keep_frac))
        neg_losses = loss_vec[neg_idx]
        if steps_per_epoch <= 1:
            keep_neg = neg_idx[np.argpartition(neg_losses, kneg - 1)[:kneg]]
        else:
            keep_neg = neg_idx[np.argpartition(neg_losses, -kneg)[-kneg:]]

        idx = np.concatenate([keep_pos, keep_neg])

        # train on filtered subset (one pass)
        ds_f = (
            tf.data.Dataset.from_tensor_slices((X[idx], Y[idx]))
            .shuffle(min(20000, len(idx)))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        bce = tf.keras.losses.BinaryCrossentropy()

        for xb, yb in ds_f:
            with tf.GradientTape() as tape:
                p = model(xb, training=True)
                loss = bce(yb, p)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    def center_crop_at_index(self, X: np.ndarray, centers: np.ndarray, crop: int) -> np.ndarray:
        N, W, C = X.shape
        half = crop // 2
        centers = centers.astype(np.int64)
        starts = np.clip(centers - half, 0, W - crop)

        out = np.empty((N, crop, C), dtype=X.dtype)
        for i in range(N):
            s = int(starts[i])
            out[i] = X[i, s:s + crop, :]
        return out
