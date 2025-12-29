from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class SegmentFilterConfig:
    """
    Optional filtering to drop "broken" windows:
      - too many zeros (usually padding or NaNs turned into zeros)
      - too little energy (flat / dead segments)
    """
    max_zero_frac: float = 0.50
    min_mean_abs: float = 0.05
    verbose: bool = True


class K2ScoreLoader:
    """
    K2ScoreLoader

    Your parquet files are metadata-only:
      ['star_id', 'mission', 'start', 'end', 'label', 'seg_mid_time', ...]
    They do NOT contain the 1024 flux window vectors.

    This class reconstructs X by downloading/stitching each EPIC lightcurve,
    applies length-preserving preprocessing (so start/end indices stay valid),
    slices windows, builds correct channel count for the model, and scores.

    Typical usage:
        loader = K2ScoreLoader(window_len=1024, quality_bitmask="none")
        df_scored = loader.score_meta_parquet("k2_segments_test.parquet", "k2_window1024_base.keras")
    """

    def __init__(
        self,
        window_len: int = 1024,
        stride: int = 256,
        quality_bitmask: str = "none",  # "none" keeps cadence indexing stable
        provenance_priority: Tuple[str, ...] = ("K2", "EVEREST", "K2SFF"),
        banned_provenance: Tuple[str, ...] = ("K2SC", "K2VARCAT"),
        cache_flux: bool = True,
        verbose: bool = True,
    ) -> None:
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.quality_bitmask = str(quality_bitmask)
        self.provenance_priority = tuple(provenance_priority)
        self.banned_provenance = tuple(banned_provenance)
        self.cache_flux = bool(cache_flux)
        self.verbose = bool(verbose)

        self._model: Optional[tf.keras.Model] = None
        self._model_path: Optional[str] = None
        self._flux_cache: Dict[str, np.ndarray] = {}

    # -------------------------
    # Lightcurve fetching
    # -------------------------
    def _fetch_k2_flux(self, star_id: str) -> np.ndarray:
        """
        Download and stitch K2 LC, return flux as float32.
        Keeps NaNs and keeps cadence count stable by using quality_bitmask="none".
        Skips K2SC because older LK versions cannot read it.
        """
        star_id = str(star_id)
        if self.cache_flux and star_id in self._flux_cache:
            return self._flux_cache[star_id]

        import lightkurve as lk  # lazy import

        query = star_id.replace("EPIC_", "EPIC ")
        sr = lk.search_lightcurve(query, mission="K2")
        if len(sr) == 0:
            raise RuntimeError(f"No K2 lightcurve found for {star_id} (query='{query}')")

        tbl = sr.table
        prov = np.asarray(tbl["provenance_name"]).astype(str) if "provenance_name" in tbl.colnames else None

        banned_upper = {b.upper() for b in self.banned_provenance}

        # Build download order: priority first, then remaining (excluding banned)
        idxs: List[int] = []
        if prov is not None:
            prov_u = np.char.upper(prov)
            for p in self.provenance_priority:
                idxs.extend([i for i in range(len(sr)) if prov_u[i] == p.upper()])
            idxs.extend([i for i in range(len(sr)) if prov_u[i] not in banned_upper and i not in idxs])
        else:
            idxs = list(range(len(sr)))

        last_err: Optional[Exception] = None

        for i in idxs:
            if prov is not None and str(prov[i]).upper() in banned_upper:
                continue
            try:
                lc = sr[i].download(quality_bitmask=self.quality_bitmask)
                if lc is None:
                    continue
                try:
                    lc = lc.stitch()
                except Exception:
                    pass

                flux = np.asarray(lc.flux.value, dtype=np.float32)  # keep NaNs, do NOT filter
                if flux.size == 0:
                    continue

                if self.cache_flux:
                    self._flux_cache[star_id] = flux
                return flux

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to download a supported LC for {star_id}. Last error: {last_err}")

    # -------------------------
    # Preprocessing (MUST preserve length)
    # -------------------------
    def preprocess_flux(self, flux: np.ndarray) -> np.ndarray:
        """
        Length-preserving preprocessing so (start,end) indices stay valid.

        Replace this with your training preprocessing if needed,
        but DO NOT drop/trim elements or indices will shift.

        Current behavior:
          - nanmedian center
          - nanstd scale
          - set non-finite to 0
        """
        flux = np.asarray(flux, dtype=np.float32)
        if flux.size == 0:
            return flux

        med = np.nanmedian(flux)
        std = np.nanstd(flux) + 1e-8
        x = (flux - med) / std
        x[~np.isfinite(x)] = 0.0
        return x.astype(np.float32)

    # -------------------------
    # Window slicing
    # -------------------------
    def _slice_segment(self, series: np.ndarray, start: int, end: int, on_bad_index: str) -> Optional[np.ndarray]:
        """
        Returns segment of length window_len or None (if skip).
        on_bad_index: "raise" | "pad" | "skip"
        """
        n = int(series.shape[0])
        start = int(start)
        end = int(end)

        if (end - start) != self.window_len:
            if on_bad_index == "raise":
                raise ValueError(f"end-start != window_len: start={start}, end={end}, L={self.window_len}")
            if on_bad_index == "skip":
                return None

        if start < 0 or end <= 0:
            if on_bad_index == "raise":
                raise ValueError(f"Invalid start/end: start={start}, end={end}")
            if on_bad_index == "skip":
                return None

        if end > n:
            if on_bad_index == "raise":
                raise ValueError(f"end={end} > len(series)={n}. Indices mismatch or preprocessing mismatch.")
            if on_bad_index == "skip":
                return None

            # pad
            seg = np.zeros(self.window_len, dtype=np.float32)
            cut = series[start:min(end, n)]
            seg[: cut.shape[0]] = cut
            return seg

        seg = series[start:end].astype(np.float32, copy=False)
        if seg.shape[0] == self.window_len:
            return seg

        if on_bad_index == "raise":
            raise ValueError(f"Segment length {seg.shape[0]} != {self.window_len}")
        if on_bad_index == "skip":
            return None

        seg2 = np.zeros(self.window_len, dtype=np.float32)
        seg2[: min(self.window_len, seg.shape[0])] = seg[: min(self.window_len, seg.shape[0])]
        return seg2

    def build_X_from_meta(
        self,
        df_meta: pd.DataFrame,
        on_bad_index: str = "raise",
        filter_cfg: Optional[SegmentFilterConfig] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Rebuild X from metadata rows: (star_id, start, end)

        Returns:
          df_used: rows that became windows (same order as X)
          X: (N, window_len) float32
        """
        required = {"star_id", "start", "end"}
        missing = required - set(df_meta.columns)
        if missing:
            raise ValueError(f"df_meta missing columns: {sorted(missing)}")

        df = df_meta.copy()
        df["star_id"] = df["star_id"].astype(str)

        X_list: List[np.ndarray] = []
        keep_idx: List[int] = []

        # download once per star
        for star_id, g in df.groupby("star_id", sort=False):
            raw_flux = self._fetch_k2_flux(star_id)
            series = self.preprocess_flux(raw_flux)

            for idx, row in g.iterrows():
                seg = self._slice_segment(series, row["start"], row["end"], on_bad_index=on_bad_index)
                if seg is None:
                    continue
                X_list.append(seg)
                keep_idx.append(idx)

        df_used = df.loc[keep_idx].reset_index(drop=True)
        X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, self.window_len), dtype=np.float32)

        if filter_cfg is not None and X.shape[0] > 0:
            df_used, X = self._filter_segments(df_used, X, filter_cfg)

        return df_used, X

    def _filter_segments(
        self,
        df_used: pd.DataFrame,
        X: np.ndarray,
        cfg: SegmentFilterConfig,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Drop windows with too many zeros or too low mean abs.
        """
        zero_frac = (X == 0).mean(axis=1)
        mean_abs = np.mean(np.abs(X), axis=1)

        keep = (zero_frac <= cfg.max_zero_frac) & (mean_abs >= cfg.min_mean_abs)

        if cfg.verbose:
            print(f"Filtered segments: {int(keep.sum())} / {len(keep)}")
            if len(keep) > 0:
                print("X %zeros:", float((X == 0).mean()))
                print("X mean abs:", float(np.mean(np.abs(X))))
                print("X std:", float(np.std(X)))
                print("zero_per_row min/mean/max:", float(zero_frac.min()), float(zero_frac.mean()), float(zero_frac.max()))
                print("energy min/mean/max:", float(mean_abs.min()), float(mean_abs.mean()), float(mean_abs.max()))

        df_f = df_used.loc[keep].reset_index(drop=True)
        X_f = X[keep].astype(np.float32, copy=False)
        return df_f, X_f

    # -------------------------
    # Model scoring
    # -------------------------
    def _load_model(self, keras_model_path: str | Path) -> tf.keras.Model:
        keras_model_path = str(keras_model_path)
        if self._model is None or self._model_path != keras_model_path:
            self._model = tf.keras.models.load_model(keras_model_path, compile=False)
            self._model_path = keras_model_path
        return self._model

    def _ensure_model_channels(self, X: np.ndarray, model: tf.keras.Model) -> np.ndarray:
        """
        Adapts X to the model's expected channels.

        Model input shape often is (None, 1024, 2).

        Supported:
          - X (N, L) -> becomes (N, L, 1) and then adapted if model expects 2
          - X (N, L, 1) or (N, L, 2) allowed

        If model expects 2 and X has 1:
          channel0 = X
          channel1 = diff(X)  (first difference, padded)
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 2:
            X = X[..., None]  # (N, L, 1)

        if X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D. Got shape={X.shape}")

        in_shape = model.input_shape  # e.g. (None, 1024, 2)
        exp_channels = in_shape[-1] if isinstance(in_shape, tuple) else None
        if exp_channels is None:
            return X

        if X.shape[-1] == exp_channels:
            return X

        if exp_channels == 2 and X.shape[-1] == 1:
            x1 = X[..., 0]
            dx = np.diff(x1, axis=1, prepend=x1[:, :1])
            X2 = np.stack([x1, dx], axis=-1).astype(np.float32)
            return X2

        if exp_channels == 1 and X.shape[-1] == 2:
            return X[..., :1]

        raise ValueError(f"Channel mismatch: model expects {exp_channels}, got {X.shape[-1]} (X shape={X.shape})")

    def score_X(
        self,
        X: np.ndarray,
        keras_model_path: str | Path,
        batch_size: int = 256,
        verbose: int = 1,
    ) -> np.ndarray:
        model = self._load_model(keras_model_path)
        X_in = self._ensure_model_channels(X, model)

        if self.verbose:
            print("MODEL INPUT SHAPE:", model.input_shape)
            print("X going into predict:", X_in.shape, X_in.dtype, "min/max:",
                  float(np.min(X_in)), float(np.max(X_in)))

        preds = model.predict(X_in, batch_size=int(batch_size), verbose=verbose)
        preds = np.asarray(preds).reshape(-1)
        return preds

    # -------------------------
    # Public API
    # -------------------------
    def score_meta_parquet(
        self,
        meta_parquet_path: str | Path,
        keras_model_path: str | Path,
        batch_size: int = 256,
        on_bad_index: str = "raise",
        filter_cfg: Optional[SegmentFilterConfig] = None,
        split_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        meta parquet -> rebuild X -> score -> return df_scored segments with 'score'

        Expects meta parquet to include:
          star_id, start, end
        Optional:
          label, seg_mid_time, mission...
        """
        df_meta = pd.read_parquet(Path(meta_parquet_path)).copy()

        if "seg_mid_time" not in df_meta.columns:
            df_meta["seg_mid_time"] = np.nan

        df_used, X = self.build_X_from_meta(df_meta, on_bad_index=on_bad_index, filter_cfg=filter_cfg)

        if X.shape[0] == 0:
            raise RuntimeError("No windows reconstructed. Check (start,end) and preprocessing alignment.")

        scores = self.score_X(X, keras_model_path, batch_size=batch_size, verbose=1)
        df_used["score"] = scores

        if split_name is not None:
            df_used["split"] = split_name

        if self.verbose:
            print("preds min/max/mean/std:",
                  float(np.min(scores)), float(np.max(scores)),
                  float(np.mean(scores)), float(np.std(scores)))

        return df_used
