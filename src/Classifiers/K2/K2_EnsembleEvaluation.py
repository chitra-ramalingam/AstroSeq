from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig



@dataclass(frozen=True)
class EnsembleResult:
    p_ens: np.ndarray              # (N,)
    p_models: np.ndarray           # (M, N)
    meta: pd.DataFrame


class K2EnsembleEvaluator:
    def __init__(self, trainer, model_paths: List[str | Path]) -> None:
        self.trainer = trainer
        self.model_paths = [Path(p) for p in model_paths]
        for p in self.model_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing model: {p}")

        # Load once—don’t reload per split
        self.models = [tf.keras.models.load_model(p) for p in self.model_paths]

    @staticmethod
    def _pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = tf.keras.metrics.AUC(curve="PR")
        m.update_state(y_true.astype(np.float32), y_pred.astype(np.float32))
        return float(m.result().numpy())

    @staticmethod
    def _topk_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int = 50) -> float:
        top = np.argsort(-y_pred)[:k]
        return float(y_true[top].mean())

    def predict_ensemble(
        self,
        X_path: str | Path,
        meta_path: str | Path,
        batch_size: Optional[int] = None,
    ) -> EnsembleResult:
        # Load split exactly like you do elsewhere
        X, y = self.trainer.load_split(X_path, meta_path)
        if batch_size is not None:
            self.trainer.cfg.batch_size = int(batch_size)

        ds = self.trainer.make_tfdata(X, y, training=False)

        preds = []
        for m in self.models:
            p = m.predict(ds, verbose=0).ravel().astype(np.float32)
            preds.append(p)

        P = np.stack(preds, axis=0)          # (M, N)
        p_ens = P.mean(axis=0)               # (N,)

        meta = pd.read_parquet(Path(meta_path)).copy()
        meta["p_ens"] = p_ens

        # Optional—store individual model preds too
        for i in range(P.shape[0]):
            meta[f"p_m{i}"] = P[i]

        return EnsembleResult(p_ens=p_ens, p_models=P, meta=meta)

    def evaluate(
        self,
        X_path: str | Path,
        meta_path: str | Path,
        k: int = 50,
    ) -> Dict[str, Any]:
        X, y = self.trainer.load_split(X_path, meta_path)
        ds = self.trainer.make_tfdata(X, y, training=False)

        preds = [m.predict(ds, verbose=0).ravel().astype(np.float32) for m in self.models]
        P = np.stack(preds, axis=0)
        p_ens = P.mean(axis=0)

        y1 = y.reshape(-1).astype(np.float32)
        out = {
            "n": int(len(y1)),
            "pos": int(y1.sum()),
            "pos_frac": float(y1.mean()) if len(y1) else float("nan"),
            "pr_auc_ens": self._pr_auc(y1, p_ens),
            "topk_prec_ens": self._topk_precision(y1, p_ens, k=k),
        }

        # Per-model metrics too—handy to see who’s carrying
        for i in range(P.shape[0]):
            out[f"pr_auc_m{i}"] = self._pr_auc(y1, P[i])
            out[f"topk_prec_m{i}"] = self._topk_precision(y1, P[i], k=k)

        return out
