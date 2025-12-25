from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import lightkurve as lk
from lightkurve.utils import LightkurveError

import numpy as np
import pandas as pd
import tensorflow as tf
import lightkurve as lk
import numpy as np


class K2ScoreLoader:
    """
    K2ScoreLoader

    Responsibility: produce segment-level scores by building/loading X.

    Supports two ways to get X:
      A) Feature parquet that already contains the 1024-window vectors (recommended)
      B) Metadata-only parquet (star_id, start, end, seg_mid_time, ...) where X is rebuilt
         by downloading the K2 lightcurve and slicing by (start,end).

    Returns scored DataFrames with a 'score' column.
    """

    def __init__(
        self,
        window_len: int = 1024,
        quality_bitmask: str = "none",
        cache_lightcurves: bool = True,
    ) -> None:
        self.window_len = int(window_len)
        self.quality_bitmask = str(quality_bitmask)
        self.cache_lightcurves = bool(cache_lightcurves)

        self._model: Optional[tf.keras.Model] = None
        self._model_path: Optional[str] = None

        self._lc_cache: Dict[str, np.ndarray] = {}

    # -----------------------------
    # Model scoring
    # -----------------------------
    def _load_model(self, keras_model_path: str | Path) -> tf.keras.Model:
        keras_model_path = str(keras_model_path)
        if self._model is None or self._model_path != keras_model_path:
            self._model = tf.keras.models.load_model(keras_model_path, compile=False)
            self._model_path = keras_model_path
            print("MODEL INPUT SHAPE:", self._model.input_shape)

        return self._model


    def _ensure_model_channels(self, X: np.ndarray, model: tf.keras.Model) -> np.ndarray:
        """
        Force X to match model.input_shape channels.

        Common case:
          model expects (None, 1024, 2) but X is (N, 1024, 1)
        We build a 2nd channel.

        channel_strategy:
          - "duplicate": ch2 = ch1
          - "derivative": ch2 = first difference (edge emphasis)
          - "zeros": ch2 = 0
        """
        # Normalize to 3D
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[..., None]  # (N, L, 1)

        expected = model.input_shape
        if isinstance(expected, list):
            raise ValueError(f"Multi-input model not supported by this loader. input_shape={expected}")

        expected_c = expected[-1]
        if expected_c is None:
            return X  # can't infer

        got_c = X.shape[-1]
        if got_c == expected_c:
            return X

        if expected_c == 2 and got_c == 1:
            strat = getattr(self, "channel_strategy", "duplicate")

            if strat == "zeros":
                ch2 = np.zeros_like(X)
            elif strat == "derivative":
                ch2 = np.diff(X, axis=1, prepend=X[:, :1, :])
            else:
                # "duplicate" default
                ch2 = X

            return np.concatenate([X, ch2], axis=-1)  # (N, L, 2)

        if expected_c == 1 and got_c == 2:
            return X[..., :1]

        raise ValueError(f"Cannot adapt channels: model expects {expected_c} but got {got_c}")

    def score_X(
        self,
        X: np.ndarray,
        keras_model_path: str | Path,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Score X using keras model, automatically matching the model's expected channels.
        """
        model = self._load_model(keras_model_path)

        X = self._ensure_model_channels(X, model)
        print("X going into predict:", X.shape, X.dtype,
            "min/max:", float(X.min()), float(X.max()))

        preds = model.predict(X, batch_size=int(batch_size), verbose=1)
        preds = preds.reshape(-1)
        print("preds min/max/mean/std:",
            float(preds.min()), float(preds.max()),
            float(preds.mean()), float(preds.std()))
        return np.asarray(preds).reshape(-1)


    # -----------------------------
    # Feature parquet helpers (contains vectors)
    # -----------------------------
    def debug_parquet_schema(self, path: str | Path, n: int = 2) -> None:
        """
        Prints columns + any vector-like columns (window arrays).
        Run this on segments_all_W1024_S256_k2*.parquet to discover the feature column name.
        """
        df = pd.read_parquet(Path(path))
        print("Columns:", list(df.columns))

        vector_like = []
        for c in df.columns:
            v = df[c].iloc[0]
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.ndim == 1 and arr.size >= 50:
                    vector_like.append((c, int(arr.size), str(arr.dtype)))
        print("Vector-like columns:", vector_like)
        print(df.head(n))

    def _auto_find_feature_col(self, df: pd.DataFrame, min_len: int = 50) -> str:
        for c in df.columns:
            v = df[c].iloc[0]
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.ndim == 1 and arr.size >= min_len:
                    return c
        raise ValueError("No vector feature column found in this parquet (no window arrays).")

    def load_feature_parquet(
        self,
        path: str | Path,
        feature_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads a parquet that CONTAINS the window vectors.
        Returns: (df_meta_without_vector_col, X)
        """
        df = pd.read_parquet(Path(path)).copy()

        if "star_id" not in df.columns:
            raise ValueError("Expected 'star_id' in the feature parquet.")
        df["star_id"] = df["star_id"].astype(str)

        if "seg_mid_time" not in df.columns:
            df["seg_mid_time"] = np.nan

        if feature_col is None:
            feature_col = self._auto_find_feature_col(df)

        if feature_col not in df.columns:
            raise ValueError(f"feature_col '{feature_col}' not found in columns.")

        X = np.stack(df[feature_col].apply(lambda v: np.asarray(v, dtype=np.float32)).to_list()).astype(np.float32)
        df_meta = df.drop(columns=[feature_col]).copy()
        return df_meta, X

    def score_feature_parquet(
        self,
        feature_parquet_path: str | Path,
        keras_model_path: str | Path,
        feature_col: Optional[str] = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Feature parquet -> scores -> returns df_scored (segment-level)
        """
        df_meta, X = self.load_feature_parquet(feature_parquet_path, feature_col=feature_col)
        df_meta["score"] = self.score_X(X, keras_model_path, batch_size=batch_size)
        return df_meta

    def _fetch_k2_time_flux(self, star_id: str):
        """
        Download a K2 lightcurve and return (flux, time), skipping unsupported products like K2SC.
        """
        
        star_id = str(star_id)

        # lazy init caches
        if not hasattr(self, "_lc_flux_cache"):
            self._lc_flux_cache = {}
        if not hasattr(self, "_lc_time_cache"):
            self._lc_time_cache = {}

        if self.cache_lightcurves and star_id in self._lc_flux_cache:
            return self._lc_flux_cache[star_id], self._lc_time_cache[star_id]

        query = star_id.replace("EPIC_", "EPIC ").strip()
        sr = lk.search_lightcurve(query, mission="K2")
        if len(sr) == 0:
            raise RuntimeError(f"No K2 lightcurve found for {star_id} (query={query})")

        # Filter out provenances that Lightkurve can't read in your environment
        banned = {"K2SC"}  # this is your crash
        # you can add more if needed:
        # banned |= {"K2VARCAT"}

        tbl = sr.table
        prov = None
        if "provenance_name" in tbl.colnames:
            prov = np.asarray(tbl["provenance_name"]).astype(str)
            keep = np.array([p.upper() not in banned for p in prov], dtype=bool)
            sr = sr[keep]
            prov = prov[keep]

        if len(sr) == 0:
            raise RuntimeError(f"Only unsupported K2 products found for {star_id}. Banned={sorted(banned)}")

        # Prefer certain products first (adjust if you know what you trained on)
        priority = ["K2", "EVEREST", "K2SFF"]

        # Build download order
        idxs = []
        if prov is not None:
            for p in priority:
                idxs.extend([i for i in range(len(sr)) if prov[i].upper() == p])
            idxs.extend([i for i in range(len(sr)) if i not in idxs])  # the rest
        else:
            idxs = list(range(len(sr)))

        last_err = None
        for i in idxs:
            try:
                lc = sr[i].download(quality_bitmask=self.quality_bitmask)
                if lc is None:
                    continue

                # If it's a collection-like object, stitch; otherwise just use it
                try:
                    lc = lc.stitch()
                except Exception:
                    pass

                flux = np.asarray(lc.flux.value, dtype=np.float32)  # keep NaNs
                time = np.asarray(lc.time.value, dtype=np.float64)

                if flux.size == 0 or time.size == 0:
                    continue

                if self.cache_lightcurves:
                    self._lc_flux_cache[star_id] = flux
                    self._lc_time_cache[star_id] = time

                return flux, time

            except LightkurveError as e:
                # Unsupported product, corrupted FITS, etc.
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to download a usable LC for {star_id}. Last error: {last_err}")

    def _fetch_k2_flux(self, star_id: str) -> np.ndarray:
        """
        Backwards-compatible wrapper: existing code calls _fetch_k2_flux().
        """
        flux, _t = self._fetch_k2_time_flux(star_id)
        return flux

    def _looks_like_time_bounds(self, df_meta: pd.DataFrame) -> bool:
        """
        If seg_mid_time is mostly inside [start,end], then start/end are TIME bounds.
        """
        if "seg_mid_time" not in df_meta.columns:
            return False

        s = pd.to_numeric(df_meta["start"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(df_meta["end"], errors="coerce").to_numpy(dtype=float)
        m = pd.to_numeric(df_meta["seg_mid_time"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(s) & np.isfinite(e) & np.isfinite(m)
        if ok.sum() < 50:
            return False

        pct_inside = float(np.mean((m[ok] >= s[ok]) & (m[ok] <= e[ok])))
        return pct_inside > 0.70

    def _robust_norm_fill(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = mad * 1.4826 + 1e-6
        x = (x - med) / scale
        x[~np.isfinite(x)] = 0.0
        return x

    def _slice_segment_time(self, t: np.ndarray, flux: np.ndarray, start_t: float, end_t: float) -> np.ndarray:
        """
        Slice using time bounds, then resample to window_len and robust-normalize.
        """
        t = np.asarray(t, dtype=float)
        flux = np.asarray(flux, dtype=np.float32)

        mask = np.isfinite(t) & np.isfinite(flux) & (t >= float(start_t)) & (t <= float(end_t))
        seg_t = t[mask]
        seg_y = flux[mask]

        if seg_y.size == 0:
            return np.zeros(self.window_len, dtype=np.float32)

        if seg_y.size == 1:
            x = np.full(self.window_len, float(seg_y[0]), dtype=np.float32)
            return self._robust_norm_fill(x)

        u_old = (seg_t - seg_t.min()) / (seg_t.max() - seg_t.min() + 1e-12)
        u_new = np.linspace(0.0, 1.0, self.window_len, dtype=np.float64)

        x = np.interp(u_new, u_old, seg_y).astype(np.float32)
        return self._robust_norm_fill(x)

    def preprocess_flux(self, flux: np.ndarray) -> np.ndarray:
        """
        Preserve array length to keep (start,end) indexing valid.
        Normalize using nan-safe stats, then fill NaNs with 0.
        """
        flux = np.asarray(flux, dtype=np.float32)

        if flux.size == 0:
            return flux

        med = np.nanmedian(flux)
        std = np.nanstd(flux) + 1e-8
        x = (flux - med) / std

        # keep length the same: replace NaNs/Infs after normalization
        x[~np.isfinite(x)] = 0.0
        return x.astype(np.float32)

    def build_X_from_meta(
        self,
        df_meta: pd.DataFrame,
        on_bad_index: str = "pad",   # "pad" | "skip" | "raise"
        slice_mode: str = "auto",    # "auto" | "index" | "time"
        min_points_time: int = 16,   # minimum points required for a time-slice to be considered valid
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Reconstruct X for each row using (star_id, start, end).

        Robust behavior:
          - decides per-star whether start/end are INDEX or TIME
          - if TIME, tries common time zero-point offsets (raw, BJD->BKJD, BJD->BTJD)
          - if TIME slicing yields empty segments, falls back to INDEX for that star
        """
        required = {"star_id", "start", "end"}
        missing = required - set(df_meta.columns)
        if missing:
            raise ValueError(f"df_meta missing columns: {sorted(missing)}")

        if slice_mode not in {"auto", "index", "time"}:
            raise ValueError("slice_mode must be one of: 'auto', 'index', 'time'")

        df = df_meta.copy()
        df["star_id"] = df["star_id"].astype(str)

        # helper: normalize a window without changing length
        def _norm_window(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            med = np.nanmedian(x)
            std = np.nanstd(x) + 1e-8
            y = (x - med) / std
            y[~np.isfinite(y)] = 0.0
            return y

        # helper: slice by time and resample to fixed window_len
        def _slice_time_resample(t: np.ndarray, f: np.ndarray, st: float, en: float) -> Optional[np.ndarray]:
            m = np.isfinite(t) & np.isfinite(f) & (t >= st) & (t <= en)
            tt = t[m]
            ff = f[m]
            if ff.size < int(min_points_time):
                return None

            # resample to self.window_len using normalized time parameter
            if ff.size == 1:
                w = np.full(self.window_len, float(ff[0]), dtype=np.float32)
                return _norm_window(w)

            u_old = (tt - tt.min()) / (tt.max() - tt.min() + 1e-12)
            u_new = np.linspace(0.0, 1.0, self.window_len, dtype=np.float64)
            w = np.interp(u_new, u_old, ff).astype(np.float32)
            return _norm_window(w)

        X_list: List[np.ndarray] = []
        keep_rows: List[int] = []

        for star_id, g in df.groupby("star_id", sort=False):
            flux_raw, time_raw = self._fetch_k2_time_flux(star_id)

            # numeric versions of start/end
            s_arr = pd.to_numeric(g["start"], errors="coerce").to_numpy(dtype=float)
            e_arr = pd.to_numeric(g["end"], errors="coerce").to_numpy(dtype=float)

            # ---- decide mode for this star
            tmin = float(np.nanmin(time_raw))
            tmax = float(np.nanmax(time_raw))
            n = int(len(flux_raw))

            # index plausibility
            s_intlike = np.nanmean(np.isclose(s_arr, np.round(s_arr), atol=1e-6))
            e_intlike = np.nanmean(np.isclose(e_arr, np.round(e_arr), atol=1e-6))
            idx_ok = np.isfinite(s_arr) & np.isfinite(e_arr) & (s_arr >= 0) & (e_arr > s_arr) & (e_arr <= n)
            idx_ok_frac = float(np.mean(idx_ok)) if idx_ok.size else 0.0

            # time plausibility (try a few offsets)
            # 0: assume already in lc.time units (often BKJD)
            # -2454833: if start/end stored as BJD, convert to BKJD
            # -2457000: if start/end stored as BJD, convert to BTJD (more TESS-like, but cheap to test)
            offsets = [0.0, -2454833.0, -2457000.0]
            best_time = (0.0, 0.0)  # (time_ok_frac, offset)

            for off in offsets:
                st = s_arr + off
                en = e_arr + off
                time_ok = np.isfinite(st) & np.isfinite(en) & (en > st) & (st <= tmax) & (en >= tmin)
                frac = float(np.mean(time_ok)) if time_ok.size else 0.0
                if frac > best_time[0]:
                    best_time = (frac, off)

            time_ok_frac, best_off = best_time

            # choose
            if slice_mode == "index":
                use_time = False
            elif slice_mode == "time":
                use_time = True
            else:
                # AUTO: prefer index if it looks “cadence-like”
                # (your seg_mid_time containment check fooled us because seg_mid_time can also be an index)
                use_time = (time_ok_frac > 0.80) and (idx_ok_frac < 0.50)

            # prepare index-mode flux (normalized, length preserved)
            flux_idx = self.preprocess_flux(flux_raw)

            # ---- build X rows
            empty_time_slices = 0
            total_rows = 0

            for idx, row in g.iterrows():
                total_rows += 1
                s = row["start"]
                e = row["end"]

                if use_time:
                    st = float(s) + float(best_off)
                    en = float(e) + float(best_off)
                    w = _slice_time_resample(time_raw, flux_raw, st, en)

                    if w is None:
                        empty_time_slices += 1
                        if on_bad_index == "raise":
                            raise ValueError(f"{star_id}: empty time slice (start={s}, end={e}, off={best_off})")
                        if on_bad_index == "skip":
                            continue
                        w = np.zeros(self.window_len, dtype=np.float32)
                    X_list.append(w)
                    keep_rows.append(idx)
                    continue

                # index mode
                si = int(float(s))
                ei = int(float(e))

                if si < 0 or ei <= 0 or ei <= si:
                    if on_bad_index == "raise":
                        raise ValueError(f"{star_id}: invalid start/end (start={si}, end={ei})")
                    if on_bad_index == "skip":
                        continue
                    seg = np.zeros(self.window_len, dtype=np.float32)
                else:
                    if si >= n:
                        if on_bad_index == "raise":
                            raise ValueError(f"{star_id}: start={si} >= len(flux)={n}")
                        if on_bad_index == "skip":
                            continue
                        seg = np.zeros(self.window_len, dtype=np.float32)
                    else:
                        seg = self._slice_segment(flux_idx, si, ei, self.window_len)

                X_list.append(seg)
                keep_rows.append(idx)

            # If we chose time but almost everything was empty, auto-fallback to index for THIS star
            if slice_mode == "auto" and use_time and total_rows > 0:
                empty_frac = empty_time_slices / total_rows
                if empty_frac > 0.70:
                    # rebuild those rows in index mode instead (fast, same star already fetched)
                    # remove the rows we just appended for this star
                    rollback = sum(1 for _ in keep_rows[-total_rows:])
                    del X_list[-rollback:]
                    del keep_rows[-rollback:]

                    for idx, row in g.iterrows():
                        s = int(float(row["start"]))
                        e = int(float(row["end"]))
                        if s < 0 or e <= 0 or e <= s:
                            if on_bad_index == "raise":
                                raise ValueError(f"{star_id}: invalid start/end (start={s}, end={e})")
                            if on_bad_index == "skip":
                                continue
                            seg = np.zeros(self.window_len, dtype=np.float32)
                        else:
                            if s >= n:
                                if on_bad_index == "raise":
                                    raise ValueError(f"{star_id}: start={s} >= len(flux)={n}")
                                if on_bad_index == "skip":
                                    continue
                                seg = np.zeros(self.window_len, dtype=np.float32)
                            else:
                                seg = self._slice_segment(flux_idx, s, e, self.window_len)

                        X_list.append(seg)
                        keep_rows.append(idx)

        df_used = df.loc[keep_rows].reset_index(drop=True)
        X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, self.window_len), dtype=np.float32)

        print("X shape:", X.shape)
        print("X mean abs:", float(np.mean(np.abs(X))))
        print("X %zeros:", float(np.mean(X == 0.0)))
        print("X std:", float(np.std(X)))

        return df_used, X


    def score_meta_parquet(
        self,
        meta_parquet_path: str | Path,
        keras_model_path: str | Path,
        batch_size: int = 256,
        on_bad_index: str = "pad",
    ) -> pd.DataFrame:
        """
        Meta parquet (no vectors) -> rebuild X -> score -> returns df_scored (segment-level)
        """
        df_meta = pd.read_parquet(Path(meta_parquet_path)).copy()
        if "seg_mid_time" not in df_meta.columns:
            df_meta["seg_mid_time"] = np.nan

        df_used, X = self.build_X_from_meta(df_meta, on_bad_index=on_bad_index)
        if len(df_used) == 0:
            raise RuntimeError("No segments could be reconstructed. Likely index mismatch or missing lightcurves.")
        # --- filter out junk windows (too many zeros)
        
        
        zero_per_row = (X == 0.0).mean(axis=1)
        keep = zero_per_row <= 0.50   # keep windows with <=50% zeros

        df_used = df_used.loc[keep].reset_index(drop=True)
        X = X[keep]


        print("Filtered segments:", int(keep.sum()), "/", int(len(keep)))
        print("Filtered X %zeros:", float((X == 0.0).mean()))
        # ---------
        self.quick_sanity_X(X, name="X")

        df_used["score"] = self.score_X(X, keras_model_path, batch_size=batch_size)
        return df_used

    def quick_sanity_X(self, X, name="X"):
        X = np.asarray(X)
        print(f"{name} shape:", X.shape)
        print(f"{name} dtype:", X.dtype)
        print(f"{name} %zeros:", float(np.mean(X == 0)))
        print(f"{name} mean abs:", float(np.mean(np.abs(X))))
        print(f"{name} std:", float(np.std(X)))
        if X.ndim >= 2:
            energy = np.mean(np.abs(X.reshape(X.shape[0], -1)), axis=1)
            print(f"{name} per-row energy min/mean/max:",
                float(np.min(energy)), float(np.mean(energy)), float(np.max(energy)))
        zero_per_row = (X == 0).mean(axis=1)
        print("zero_per_row min/mean/max:",
            float(zero_per_row.min()), float(zero_per_row.mean()), float(zero_per_row.max()))

        print("rows with >50% zeros:", int((zero_per_row > 0.5).sum()), "/", len(zero_per_row))

        
    def _slice_segment(
        self,
        flux: np.ndarray,
        start: int,
        end: int,
        window_len: int,
        ) -> np.ndarray:
        """
        Slice a fixed-length window from a 1D flux array using start/end.

        Handles common conventions:
          - end exclusive:  end - start == window_len
          - end inclusive:  end - start == window_len - 1

        If start/end don't match window_len exactly, falls back to:
          - slice starting at `start` for `window_len` points

        Always returns shape (window_len,) padded with zeros if needed.
        """
        flux = np.asarray(flux, dtype=np.float32)
        n = int(flux.shape[0])
        w = int(window_len)

        s = int(start)
        e = int(end)

        # Clamp start/end to sensible bounds
        s = max(0, min(s, n))          # allow s==n (empty slice -> padded)
        e = max(0, min(e, n))          # if end is exclusive, e can be n

        # Decide inclusive vs exclusive if it matches perfectly
        d = e - s
        if d == w:
            # end exclusive
            seg = flux[s:e]
        elif d == w - 1:
            # end inclusive (make e inclusive by +1)
            e_incl = min(e + 1, n)
            seg = flux[s:e_incl]
        else:
            # Fallback: ignore end and take a window_len slice from start
            seg = flux[s : min(s + w, n)]

        seg = np.asarray(seg, dtype=np.float32)

        # Force fixed length via pad/truncate
        if seg.size == w:
            return seg

        out = np.zeros(w, dtype=np.float32)
        out[: min(w, seg.size)] = seg[: min(w, seg.size)]
        return out
    
    def quick_score_diagnostics(self, df_scored: pd.DataFrame) -> None:
        """
        df_scored must contain: label (0/1), score
        Prints separation stats + AUCs (if sklearn available).
        """
        if "label" not in df_scored.columns or "score" not in df_scored.columns:
            raise ValueError("Need columns: label, score")

        y = df_scored["label"].astype(int).to_numpy()
        s = df_scored["score"].astype(float).to_numpy()

        pos = s[y == 1]
        neg = s[y == 0]

        print("counts pos/neg:", len(pos), len(neg))
        print("score mean pos/neg:", float(pos.mean()) if len(pos) else None, float(neg.mean()) if len(neg) else None)
        print("score p95 pos/neg:", float(np.percentile(pos, 95)) if len(pos) else None, float(np.percentile(neg, 95)) if len(neg) else None)
        print("score min/max:", float(np.min(s)), float(np.max(s)))

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            print("ROC AUC:", float(roc_auc_score(y, s)))
            print("PR  AUC:", float(average_precision_score(y, s)))
            print("baseline PR (pos frac):", float(y.mean()))
        except Exception as e:
            print("sklearn not available for AUCs:", e)


