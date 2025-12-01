import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
import os

class CommonHelper:
    def __init__(selfa):
        pass

    def row_to_target_and_mission(self, row):
        # Kepler (KIC)
        for k in ["kepid", "KIC", "kic"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"KIC {n}", "Kepler"
        # K2 (EPIC)
        for k in ["epic", "EPIC"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"EPIC {n}", "K2"
        # TESS (TIC)
        for k in ["tid", "TIC ID", "tic_id", "tic"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"TIC {n}", "TESS"
        # TOI / names
        for k in ["toi", "TOI"]:
            if k in row and pd.notna(row[k]):
                return f"TOI {str(row[k]).strip()}", "TESS"
        if "hostname" in row and pd.notna(row["hostname"]):
            return str(row["hostname"]).strip(), None
        return None, None
    
    def _intish(self, x):
        try:
            return int(float(x))
        except Exception:
            return None
        
    def fetch_flux_row(self, row, use_all=True, max_files=4, any_author=True):
        """Return (time, flux, mission) on success; else (None, None, None)."""
        try:
            target, pref = self.row_to_target_and_mission(row)
            if not target:
                return None, None, None, None
            print("Fetching from lightcurve:", target)
            missions_try = [pref] if pref else []
            for m in ["TESS", "Kepler", "K2"]:
                if m and m not in missions_try:
                    missions_try.append(m)

            for m in missions_try:
                sr = lk.search_lightcurve(target, mission=m, author=None if any_author else self.author)
                if len(sr) == 0:
                    continue
                if use_all:
                    lcc = sr[:max_files].download_all()
                    lc = lcc.stitch().remove_nans().normalize()
                else:
                    lc = sr[0].download().remove_nans().normalize()
                try:
                    lc_flat = lc.flatten(window_length=201, polyorder=2)
                except Exception:
                    lc_flat = lc.copy()

                time = lc.time.value.astype(np.float64)
                fl_raw = lc.flux.value.astype(np.float32)
                flux_flat = lc_flat.flux.value.astype(np.float32)
                flux = np.stack([fl_raw, flux_flat], axis=-1)  # (T, 2)
                return time, flux, m, target

            return None, None, None, None
        except Exception:
            return None, None, None, None

    def segment_with_idx(self, flux, w=None, stride=None):
            if flux.ndim == 1: flux = flux[:, None]
            T, C = flux.shape
            if stride is None: stride = max(1, w // 4)
            if w <= 0 or T < w:
                return np.empty((0, w, C), np.float32), np.empty((0, 2), int)

            segs, spans = [], []
            for i in range(0, T - w + 1, stride):
                seg = flux[i:i+w].astype(np.float32)          # (w, C)
                seg = seg - np.median(seg, axis=0, keepdims=True)  # center only; keep amplitude
                segs.append(seg)
                spans.append((i, i+w))
            return np.asarray(segs, np.float32), np.asarray(spans, int)
    
    def cache_path_for_target(self, cache_dir, target):
        # e.g. "TIC 123456" -> "TIC_123456.npz"
        safe_target = str(target).replace(" ", "_").replace("/", "_")
        return os.path.join(cache_dir, f"{safe_target}.npz")

    def normalize_flux(self, flux):
        """Median/MAD normalization, same as training."""
        if flux.ndim == 1:
            flux = flux[:, None]  # (T, 1)
        med = np.median(flux, axis=0, keepdims=True)
        mad = np.median(np.abs(flux - med), axis=0, keepdims=True) + 1e-6
        flux_norm = (flux - med) / mad
        return flux_norm
    

    def fetch_with_cache(self, row):
        target, _ = self.row_to_target_and_mission(row)
        if not target:
            return None, None, None, None
        cache_path = self.cache_path_for_target(cache_dir="lc_cache", target= target)

        # 1) Try to load from cache
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)

            if "empty" in data and bool(data["empty"]):
                print("Cached EMPTY star, skipping:", target)
                return None, None, None, target

            time   = data["time"]
            flux   = data["flux"]
            mission = str(data["mission"])
            target  = str(data["target"])
            return time, flux, mission, target
        
    def fetch_with_targetId_FromCache(self, target):
        if not target:
            return None, None, None, None
        cache_path = self.cache_path_for_target(cache_dir="lc_cache", target= target)

        # 1) Try to load from cache
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)

            if "empty" in data and bool(data["empty"]):
                print("Cached EMPTY star, skipping:", target)
                return None, None, None, target

            time   = data["time"]
            flux   = data["flux"]
            mission = str(data["mission"])
            target  = str(data["target"])
            return time, flux, mission, target