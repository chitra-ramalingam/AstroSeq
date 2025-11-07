# isolation_forest_scorer.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestScorer:
    """
    Unsupervised anomaly scorer for light-curve segments using IsolationForest.
    Works on z-scored segments (e.g., from segment_with_idx), with optional PCA.

    Conventions:
      - We define 'score' = -iforest.score_samples(Zs); higher => more anomalous.
      - Thresholding uses a quantile (e.g., 0.99) by default.

    Typical use:
      scorer = IsolationForestScorer(window=200)
      segs, spans = segment_with_idx(flux)        # your helper
      out = scorer.fit_score_segments(segs, spans)
      # out["scores"], out["flags"], out["spans"]
    """
    window: int = 200
    n_components: int = 20          # PCA components; set 0 or None to disable PCA
    contamination: float = 0.01     # expected anomaly fraction (used only for model's internal threshold)
    n_estimators: int = 300
    random_state: int = 0
    n_jobs: int = -1
    threshold_quantile: float = 0.99  # quantile over scores for flagging anomalies

    # Fitted transformers/model (populated after fit)
    _pca: Optional[PCA] = None
    _scaler: Optional[StandardScaler] = None
    _iforest: Optional[IsolationForest] = None

    # ---------- core utilities ----------
    def _maybe_pca(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Apply (and optionally fit) PCA, then standardize."""
        Z = X
        if self.n_components and self.n_components > 0:
            ncomp = min(self.n_components, Z.shape[1])
            if fit or self._pca is None:
                self._pca = PCA(n_components=ncomp, random_state=self.random_state)
                Z = self._pca.fit_transform(Z)
            else:
                Z = self._pca.transform(Z)
        # Standardize after PCA (or raw)
        if fit or self._scaler is None:
            self._scaler = StandardScaler().fit(Z)
            Zs = self._scaler.transform(Z)
        else:
            Zs = self._scaler.transform(Z)
        return Zs

    def _ensure_fitted(self):
        if self._iforest is None:
            raise RuntimeError("IsolationForestScorer is not fitted. Call fit(...) first or use fit_score_segments(...).")

    # ---------- public API ----------
    def fit(self, X_segments: np.ndarray) -> "IsolationForestScorer":
        """
        Fit PCA+scaler+IsolationForest on the provided segment matrix.
        X_segments: shape (n_segments, window)
        """
        if X_segments.size == 0:
            raise ValueError("Empty X_segments passed to fit().")
        Zs = self._maybe_pca(X_segments, fit=True)
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples="auto",
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        ).fit(Zs)
        return self

    def score(self, X_segments: np.ndarray) -> np.ndarray:
        """
        Score segments using the fitted model.
        Returns 'scores' where higher => more anomalous.
        """
        self._ensure_fitted()
        Zs = self._maybe_pca(X_segments, fit=False)
        # sklearn's score_samples: higher ~ more normal; we invert
        return -self._iforest.score_samples(Zs)

    def predict_flags(self, scores: np.ndarray, quantile: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Convert continuous scores to flags via a quantile threshold.
        Returns (flags, threshold_value). flags is boolean, True => anomaly.
        """
        q = self.threshold_quantile if quantile is None else quantile
        thr = np.quantile(scores, q) if scores.size > 0 else np.inf
        flags = scores >= thr
        return flags.astype(bool), thr

    def fit_score_segments(self, X_segments: np.ndarray, spans: Optional[np.ndarray] = None,
                           quantile: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        One-shot: fit on X_segments, then score and flag them.
        Returns dict with 'scores', 'flags', 'threshold', 'spans'.
        """
        self.fit(X_segments)
        scores = self.score(X_segments)
        flags, thr = self.predict_flags(scores, quantile=quantile)
        return {
            "scores": scores,
            "flags": flags,
            "threshold": np.array([thr]),
            "spans": spans if spans is not None else None,
            "segs": X_segments
        }

    def score_from_flux(self, flux: np.ndarray, segment_with_idx_fn, stride: Optional[int] = None,
                        quantile: Optional[float] = None, fit_on_self: bool = True) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: do segmentation on a single light curve, then fit+score (or just score if already fitted).
        - segment_with_idx_fn: your function (flux, w, stride) -> (segments, spans)
        - fit_on_self: if True, fit on this star's segments; if False, assume model already fitted globally.
        """
        segs, spans = segment_with_idx_fn(flux, w=self.window, stride=stride or self.window // 4)
        if segs.size == 0:
            return {"scores": np.empty((0,), float), "flags": np.empty((0,), bool), "threshold": np.array([np.nan]), "spans": spans, "segs": segs   }
        if fit_on_self or self._iforest is None:
            return self.fit_score_segments(segs, spans=spans, quantile=quantile)
        else:
            scores = self.score(segs)
            flags, thr = self.predict_flags(scores, quantile=quantile)
            return {"scores": scores, "flags": flags, "threshold": np.array([thr]), "spans": spans, "segs": segs}
