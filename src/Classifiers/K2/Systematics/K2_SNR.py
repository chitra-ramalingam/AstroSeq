from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class K2SNR:
    """
    Canonical K2 dip-SNR utility.

    Canonical definitions:
      flux_rel = flux / median(flux) - 1
      baseline = rolling median(flux_rel)
      resid = flux_rel - baseline
      local_sigma = 1.4826 * rolling_MAD(resid), clamped by sigma_floor
      dip_depth = -min(resid over dip segment), positive for dips
      dip_snr = dip_depth / local_sigma, positive for dips
    """

    def __init__(
        self,
        window_days: float = 1.0,
        sigma_floor_quantile: Optional[float] = 0.05,
        sigma_floor_abs: float = 1e-5,
        min_points: int = 20,
    ) -> None:
        if (not np.isfinite(window_days)) or float(window_days) <= 0:
            raise ValueError("window_days must be finite and > 0")
        if int(min_points) < 3:
            raise ValueError("min_points must be >= 3")
        if (not np.isfinite(sigma_floor_abs)) or float(sigma_floor_abs) <= 0:
            raise ValueError("sigma_floor_abs must be finite and > 0")
        if sigma_floor_quantile is not None:
            q = float(sigma_floor_quantile)
            if (not np.isfinite(q)) or q < 0.0 or q > 1.0:
                raise ValueError("sigma_floor_quantile must be in [0, 1] or None")

        self.window_days = float(window_days)
        self.sigma_floor_quantile = None if sigma_floor_quantile is None else float(sigma_floor_quantile)
        self.sigma_floor_abs = float(sigma_floor_abs)
        self.min_points = int(min_points)

    @staticmethod
    def _rolling_median(values: np.ndarray, window_len: int) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        n = len(x)
        if n == 0:
            return np.asarray([], dtype=float)

        out = np.full(n, np.nan, dtype=float)
        half = int(max(1, window_len // 2))
        global_med = float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0

        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            seg = x[a:b]
            seg = seg[np.isfinite(seg)]
            out[i] = float(np.nanmedian(seg)) if len(seg) > 0 else global_med
        return out

    @staticmethod
    def _rolling_mad(values: np.ndarray, window_len: int) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        n = len(x)
        if n == 0:
            return np.asarray([], dtype=float)

        out = np.full(n, np.nan, dtype=float)
        half = int(max(1, window_len // 2))

        if np.any(np.isfinite(x)):
            gmed = float(np.nanmedian(x))
            gmad = float(np.nanmedian(np.abs(x - gmed)))
        else:
            gmad = 0.0

        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            seg = x[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) == 0:
                out[i] = gmad
                continue
            med = float(np.nanmedian(seg))
            out[i] = float(np.nanmedian(np.abs(seg - med)))

        return out

    def _window_len_from_days(self, time: np.ndarray) -> int:
        t = np.asarray(time, dtype=float)
        t = t[np.isfinite(t)]
        if len(t) < 3:
            w = int(max(5, self.min_points))
            return w if w % 2 == 1 else w + 1

        dt = np.diff(np.sort(t))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            raw = int(self.min_points)
        else:
            raw = int(np.round(self.window_days / float(np.nanmedian(dt))))

        w = int(max(5, self.min_points, raw))
        if w % 2 == 0:
            w += 1
        return w

    def _compute_sigma_floor(self, local_sigma_raw: np.ndarray) -> float:
        s = np.asarray(local_sigma_raw, dtype=float)
        pos = s[np.isfinite(s) & (s > 0)]

        floor = float(self.sigma_floor_abs)
        if self.sigma_floor_quantile is not None and len(pos) > 0:
            q_floor = float(np.nanquantile(pos, self.sigma_floor_quantile))
            if np.isfinite(q_floor) and q_floor > 0:
                floor = max(floor, q_floor)

        if (not np.isfinite(floor)) or floor <= 0:
            floor = float(self.sigma_floor_abs)
        return float(floor)

    @staticmethod
    def _coerce_float(value: Any, default: float = float("nan")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if np.isfinite(out) else float(default)

    def _sanitize_dip_snr(self, dip_snr: float, debug_row: Dict[str, Any]) -> float:
        snr = float(dip_snr) if np.isfinite(dip_snr) else 0.0
        if snr < 0:
            snr = 0.0
        if snr > 1e4:
            payload = dict(debug_row)
            payload["dip_snr_raw"] = float(snr)
            print(f"[K2SNR] dip_snr clamp (>1e4): {payload}")
            snr = 1e4
        if not np.isfinite(snr):
            snr = 0.0
        return float(snr)

    def normalize(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, Any]:
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        if t.shape != f.shape:
            raise ValueError("time and flux must have the same shape")

        n = len(t)
        if n == 0:
            return {
                "flux_rel": np.asarray([], dtype=float),
                "baseline": np.asarray([], dtype=float),
                "resid": np.asarray([], dtype=float),
                "local_sigma": np.asarray([], dtype=float),
                "sigma_floor": float(self.sigma_floor_abs),
            }

        flux_rel = np.full(n, np.nan, dtype=float)
        baseline = np.full(n, np.nan, dtype=float)
        resid = np.full(n, np.nan, dtype=float)

        finite = np.isfinite(t) & np.isfinite(f)
        if not np.any(finite):
            sigma_floor = float(self.sigma_floor_abs)
            local_sigma = np.full(n, sigma_floor, dtype=float)
            return {
                "flux_rel": flux_rel,
                "baseline": baseline,
                "resid": resid,
                "local_sigma": local_sigma,
                "sigma_floor": sigma_floor,
            }

        t_valid = t[finite]
        f_valid = f[finite]
        med_flux = float(np.nanmedian(f_valid))
        if (not np.isfinite(med_flux)) or abs(med_flux) < 1e-12:
            med_flux = 1.0

        flux_rel_valid = (f_valid / med_flux) - 1.0
        window_len = self._window_len_from_days(t_valid)
        baseline_valid = self._rolling_median(flux_rel_valid, window_len)
        resid_valid = flux_rel_valid - baseline_valid

        mad_valid = self._rolling_mad(resid_valid, window_len)
        local_sigma_raw = 1.4826 * mad_valid
        sigma_floor = self._compute_sigma_floor(local_sigma_raw)
        local_sigma_valid = np.where(
            np.isfinite(local_sigma_raw) & (local_sigma_raw > 0),
            local_sigma_raw,
            sigma_floor,
        )
        local_sigma_valid = np.maximum(local_sigma_valid, sigma_floor)
        local_sigma_valid = np.where(np.isfinite(local_sigma_valid), local_sigma_valid, sigma_floor)

        flux_rel[finite] = flux_rel_valid
        baseline[finite] = baseline_valid
        resid[finite] = resid_valid

        local_sigma = np.full(n, sigma_floor, dtype=float)
        local_sigma[finite] = local_sigma_valid
        local_sigma = np.where(np.isfinite(local_sigma) & (local_sigma >= sigma_floor), local_sigma, sigma_floor)

        return {
            "flux_rel": flux_rel.astype(float),
            "baseline": baseline.astype(float),
            "resid": resid.astype(float),
            "local_sigma": local_sigma.astype(float),
            "sigma_floor": float(sigma_floor),
        }

    def depth_snr_for_segment(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        i0: int,
        i1: int,
    ) -> Dict[str, float]:
        norm = self.normalize(time=time, flux=flux)
        resid = np.asarray(norm["resid"], dtype=float)
        sigma = np.asarray(norm["local_sigma"], dtype=float)
        sigma_floor = float(norm["sigma_floor"])

        n = len(resid)
        a = int(max(0, min(int(i0), n)))
        b = int(max(0, min(int(i1), n)))
        if b <= a:
            return {
                "dip_depth": 0.0,
                "local_sigma_med": float(sigma_floor),
                "dip_snr": 0.0,
            }

        seg_resid = resid[a:b]
        seg_sigma = sigma[a:b]
        seg_resid = seg_resid[np.isfinite(seg_resid)]
        seg_sigma = seg_sigma[np.isfinite(seg_sigma) & (seg_sigma > 0)]

        if len(seg_resid) == 0:
            dip_depth = 0.0
        else:
            min_resid = float(np.nanmin(seg_resid))
            dip_depth = float(max(0.0, -min_resid))

        if len(seg_sigma) == 0:
            local_sigma_med = float(sigma_floor)
        else:
            local_sigma_med = float(np.nanmedian(seg_sigma))
            if (not np.isfinite(local_sigma_med)) or local_sigma_med <= 0:
                local_sigma_med = float(sigma_floor)
            local_sigma_med = float(max(local_sigma_med, sigma_floor))

        dip_snr_raw = float(dip_depth / local_sigma_med) if local_sigma_med > 0 else float("inf")
        dip_snr = self._sanitize_dip_snr(
            dip_snr_raw,
            {
                "method": "depth_snr_for_segment",
                "i0": int(a),
                "i1": int(b),
                "dip_depth": float(dip_depth),
                "local_sigma_med": float(local_sigma_med),
            },
        )

        return {
            "dip_depth": float(dip_depth),
            "local_sigma_med": float(local_sigma_med),
            "dip_snr": float(dip_snr),
        }

    @staticmethod
    def _duration_below_threshold(resid: np.ndarray, threshold: np.ndarray, idx_min: int) -> int:
        x = np.asarray(resid, dtype=float)
        thr = np.asarray(threshold, dtype=float)
        n = len(x)
        if n == 0 or len(thr) != n or idx_min < 0 or idx_min >= n:
            return 0
        if (not np.isfinite(x[idx_min])) or (not np.isfinite(thr[idx_min])) or (x[idx_min] > thr[idx_min]):
            return 0

        run = 1
        i = idx_min - 1
        while i >= 0 and np.isfinite(x[i]) and np.isfinite(thr[i]) and (x[i] <= thr[i]):
            run += 1
            i -= 1

        i = idx_min + 1
        while i < n and np.isfinite(x[i]) and np.isfinite(thr[i]) and (x[i] <= thr[i]):
            run += 1
            i += 1
        return int(run)

    def soft_dip_snr_at_time(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        t_center: float,
        tol_days: float,
    ) -> Dict[str, Any]:
        if (not np.isfinite(tol_days)) or float(tol_days) <= 0:
            raise ValueError("tol_days must be finite and > 0")

        t = np.asarray(time, dtype=float)
        if len(t) == 0:
            return {
                "min_resid": float("nan"),
                "dip_depth": 0.0,
                "local_sigma_med": float(self.sigma_floor_abs),
                "dip_snr": 0.0,
                "duration_below_threshold": 0,
                "n_points": 0,
            }

        norm = self.normalize(time=t, flux=np.asarray(flux, dtype=float))
        resid = np.asarray(norm["resid"], dtype=float)
        sigma = np.asarray(norm["local_sigma"], dtype=float)
        sigma_floor = float(norm["sigma_floor"])

        in_win = np.isfinite(t) & (np.abs(t - float(t_center)) <= float(tol_days))
        n_points = int(np.sum(in_win))
        if n_points == 0:
            return {
                "min_resid": float("nan"),
                "dip_depth": 0.0,
                "local_sigma_med": float(sigma_floor),
                "dip_snr": 0.0,
                "duration_below_threshold": 0,
                "n_points": 0,
            }

        resid_win = resid[in_win]
        sigma_win = sigma[in_win]
        valid = np.isfinite(resid_win)
        if not np.any(valid):
            return {
                "min_resid": float("nan"),
                "dip_depth": 0.0,
                "local_sigma_med": float(sigma_floor),
                "dip_snr": 0.0,
                "duration_below_threshold": 0,
                "n_points": int(n_points),
            }

        valid_idx = np.where(valid)[0]
        idx_local = int(valid_idx[int(np.argmin(resid_win[valid]))])
        min_resid = float(resid_win[idx_local])
        dip_depth = float(max(0.0, -min_resid))

        pos_sigma = sigma_win[np.isfinite(sigma_win) & (sigma_win > 0)]
        if len(pos_sigma) == 0:
            local_sigma_med = float(sigma_floor)
        else:
            local_sigma_med = float(np.nanmedian(pos_sigma))
            if (not np.isfinite(local_sigma_med)) or local_sigma_med <= 0:
                local_sigma_med = float(sigma_floor)
            local_sigma_med = float(max(local_sigma_med, sigma_floor))

        dip_snr_raw = float(dip_depth / local_sigma_med) if local_sigma_med > 0 else float("inf")
        dip_snr = self._sanitize_dip_snr(
            dip_snr_raw,
            {
                "method": "soft_dip_snr_at_time",
                "t_center": float(t_center),
                "tol_days": float(tol_days),
                "dip_depth": float(dip_depth),
                "local_sigma_med": float(local_sigma_med),
                "n_points": int(n_points),
            },
        )

        threshold = -np.where(np.isfinite(sigma_win) & (sigma_win > 0), sigma_win, local_sigma_med)
        duration = self._duration_below_threshold(resid=resid_win, threshold=threshold, idx_min=idx_local)

        return {
            "min_resid": float(min_resid),
            "dip_depth": float(dip_depth),
            "local_sigma_med": float(local_sigma_med),
            "dip_snr": float(dip_snr),
            "duration_below_threshold": int(duration),
            "n_points": int(n_points),
        }

    @staticmethod
    def _event_bounds_from_row(row: pd.Series, time: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        # Preferred explicit index pairs.
        for left_key, right_key in (("start_idx", "end_idx"), ("i0", "i1"), ("start", "end")):
            if left_key in row and right_key in row:
                i0 = pd.to_numeric(pd.Series([row[left_key]]), errors="coerce").iloc[0]
                i1 = pd.to_numeric(pd.Series([row[right_key]]), errors="coerce").iloc[0]
                if np.isfinite(i0) and np.isfinite(i1):
                    return int(i0), int(i1)

        # Fallback to time bounds if indices are absent.
        if ("t_start" in row) and ("t_end" in row) and len(time) > 0:
            t0 = pd.to_numeric(pd.Series([row["t_start"]]), errors="coerce").iloc[0]
            t1 = pd.to_numeric(pd.Series([row["t_end"]]), errors="coerce").iloc[0]
            if np.isfinite(t0) and np.isfinite(t1):
                arr_t = np.asarray(time, dtype=float)
                i0 = int(np.argmin(np.abs(arr_t - float(t0))))
                i1 = int(np.argmin(np.abs(arr_t - float(t1)))) + 1
                if i1 <= i0:
                    i1 = i0 + 1
                return i0, i1

        return None, None

    def audit_events(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(events_df, pd.DataFrame):
            raise ValueError("events_df must be a pandas DataFrame")

        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        if t.shape != f.shape:
            raise ValueError("time and flux must have the same shape")

        rows = []
        for idx, row in events_df.iterrows():
            i0, i1 = self._event_bounds_from_row(row, t)
            if i0 is None or i1 is None:
                depth_stats = {
                    "dip_depth": float("nan"),
                    "local_sigma_med": float("nan"),
                    "dip_snr": float("nan"),
                }
            else:
                depth_stats = self.depth_snr_for_segment(time=t, flux=f, i0=int(i0), i1=int(i1))

            old_depth_snr = self._coerce_float(row.get("depth_snr", float("nan")))
            new_dip_snr = self._coerce_float(depth_stats.get("dip_snr", float("nan")))

            rows.append(
                {
                    "event_index": idx,
                    "i0": int(i0) if i0 is not None else np.nan,
                    "i1": int(i1) if i1 is not None else np.nan,
                    "old_depth_snr": float(old_depth_snr),
                    "new_dip_snr": float(new_dip_snr),
                    "delta_snr": float(new_dip_snr - old_depth_snr)
                    if np.isfinite(new_dip_snr) and np.isfinite(old_depth_snr)
                    else float("nan"),
                    "dip_depth": self._coerce_float(depth_stats.get("dip_depth", float("nan"))),
                    "local_sigma_med": self._coerce_float(depth_stats.get("local_sigma_med", float("nan"))),
                }
            )

        audit_df = pd.DataFrame(rows)
        out_path = Path("snr_audit.csv")
        audit_df.to_csv(out_path, index=False)
        return audit_df
