import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd


class K2PrintModelOutputs:
    def __init__(self):
        pass


    def make_tfdata(self,X, y, batch_size=256):
        X = X.astype("float32", copy=False)
        y = y.astype("float32", copy=False)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def pr_auc(self,y_true, y_pred):
        m = tf.keras.metrics.AUC(curve="PR")
        m.update_state(y_true, y_pred)
        return float(m.result().numpy())

    def topk_precision(self,y_true, y_pred, k=50):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        top = np.argsort(-y_pred)[:k]
        return float(y_true[top].mean())
    
    def loadall(self, dataDir:str):
        #dataDir = ""  # change if needed
        Xva = np.load(f"{dataDir}/X_val.npy", mmap_mode="r")
        
        dfva = pd.read_parquet(f"{dataDir}/meta_val.parquet")
        yva = dfva["label"].astype(np.float32).to_numpy()
        Xte = np.load(f"{dataDir}/X_test.npy", mmap_mode="r")
        dftr = pd.read_parquet(f"{dataDir}/meta_train.parquet")

        dfte = pd.read_parquet(f"{dataDir}/meta_test.parquet")
        yte = dfte["label"].astype(np.float32).to_numpy()

        print("VAL label_star counts:\n", dfva["label_star"].value_counts(dropna=False).head(10))
        print("TEST label_star counts:\n", dfte["label_star"].value_counts(dropna=False).head(10))
        print("TRAIN label_star counts:\n", dftr["label_star"].value_counts(dropna=False).head(10))
        print("Xva:", Xva.shape, "yva:", yva.shape, "yva sum:", float(yva.sum()), "min/max:", float(yva.min()), float(yva.max()))
        print("Xte:", Xte.shape, "yte:", yte.shape, "yte sum:", float(yte.sum()), "min/max:", float(yte.min()), float(yte.max()))
        print("unique yva (first 10):", np.unique(yva)[:10])
        print(dfva.columns)
        print(dfva["label"].value_counts(dropna=False).head(10))

        models_dir = Path("models")
        paths = sorted(models_dir.glob("*.keras"))

        ds_va = self.make_tfdata(Xva[:, :, :1], yva)  # flux-only—match your training
        ds_te = self.make_tfdata(Xte[:, :, :1], yte)

        print("MODEL | val_pr | val_top50 | test_pr")
        for pth in paths:
            try:
                m = tf.keras.models.load_model(pth)
                p_val = m.predict(ds_va, verbose=0).ravel().astype(np.float32)
                p_tes = m.predict(ds_te, verbose=0).ravel().astype(np.float32)

                val_pr = self.pr_auc(yva.astype(np.float32), p_val)
                val_top50 = self.topk_precision(yva.astype(np.float32), p_val, k=50)
                test_pr = self.pr_auc(yte.astype(np.float32), p_tes)

                print(f"{pth.name} | {val_pr:.4f} | {val_top50:.3f} | {test_pr:.4f}")
            except Exception as e:
                print(f"{pth.name} | FAILED: {e}")

