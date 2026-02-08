from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class K2PeriodValidator:
    """
    Detector-consistent period validator.

    This validator scores predicted transit windows by reusing the same
    preprocess + dip segmentation + shape scoring path as the detector.
    """

    def __init__(
        self,
        detector: Any,
        tol_days: float = 0.12,
        inner_tol_days: Optional[float] = None,
        outer_window_days: float = 2.0,
        min_duration_cadences: int = 3,
        shape_threshold: float = 0.6,
        snr_threshold: float = 4.0,
    ) -> None:
        if detector is None:
            raise ValueError("detector must be provided")
        inner_tol = float(tol_days) if inner_tol_days is None else float(inner_tol_days)
        if not np.isfinite(inner_tol) or inner_tol <= 0:
            raise ValueError("inner_tol_days/tol_days must be finite and > 0")
        if not np.isfinite(outer_window_days) or outer_window_days <= 0:
            raise ValueError("outer_window_days must be finite and > 0")
        if float(outer_window_days) < float(inner_tol):
            raise ValueError("outer_window_days must be >= inner_tol_days")
        if int(min_duration_cadences) < 1:
            raise ValueError("min_duration_cadences must be >= 1")

        self.detector = detector
        # Inner evaluation window around each predicted transit.
        self.tol_days = float(inner_tol)
        self.inner_tol_days = float(inner_tol)
        # Outer window used for local normalization/sigma context.
        self.outer_window_days = float(outer_window_days)
        self.min_duration_cadences = int(min_duration_cadences)
        self.shape_threshold = float(shape_threshold)
        self.snr_threshold = float(snr_threshold)
        self.detect_sigma_threshold = self._infer_detect_sigma(detector)

        self._preprocess_fn, self._rank_fn = self._resolve_detector_functions(detector)

    @staticmethod
    def _row_category(row: pd.Series) -> str:
        if not bool(row.get("covered", False)):
            return "uncovered"
        if not bool(row.get("has_candidate", False)):
            return "no_candidate"
        if bool(row.get("hit_both", False)):
            return "hit"
        return "miss"

    @staticmethod
    def _resolve_detector_functions(detector: Any) -> Tuple[Any, Any]:
        preprocess_fn = None
        rank_fn = None

        if hasattr(detector, "preprocessor") and hasattr(detector, "ranker"):
            preprocess_fn = getattr(detector.preprocessor, "preprocess", None)
            rank_fn = getattr(detector.ranker, "rank_windows", None)
        elif hasattr(detector, "preprocess") and hasattr(detector, "rank_windows"):
            preprocess_fn = getattr(detector, "preprocess", None)
            rank_fn = getattr(detector, "rank_windows", None)
        elif isinstance(detector, dict):
            preprocess_fn = detector.get("preprocess")
            rank_fn = detector.get("rank_windows")

        if not callable(preprocess_fn) or not callable(rank_fn):
            raise ValueError(
                "detector must expose preprocess + rank_windows, "
                "or provide .preprocessor/.ranker with those methods"
            )
        return preprocess_fn, rank_fn

    @staticmethod
    def _candidate_value(candidate: Any, key: str, default: float = float("nan")) -> float:
        if isinstance(candidate, dict):
            value = candidate.get(key, default)
        else:
            value = getattr(candidate, key, default)
        try:
            out = float(value)
        except Exception:
            out = float("nan")
        return out if np.isfinite(out) else float("nan")

    @staticmethod
    def _candidate_int(candidate: Any, key: str, default: int = 0) -> int:
        if isinstance(candidate, dict):
            value = candidate.get(key, default)
        else:
            value = getattr(candidate, key, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _safe_nanmean(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float("nan")
        return float(np.nanmean(arr))

    @staticmethod
    def _sigma_floor(local_sigma: np.ndarray) -> float:
        s = np.asarray(local_sigma, dtype=float)
        pos = s[np.isfinite(s) & (s > 0)]
        if len(pos) == 0:
            return 1e-5
        floor = float(np.nanpercentile(pos, 5))
        if not np.isfinite(floor) or floor <= 0:
            floor = 1e-5
        return floor

    @staticmethod
    def _infer_detect_sigma(detector: Any) -> float:
        detect_sigma = float("nan")
        try:
            if hasattr(detector, "ranker"):
                cfg = getattr(detector.ranker, "config", None)
                if cfg is not None and hasattr(cfg, "detect_sigma"):
                    detect_sigma = float(getattr(cfg, "detect_sigma"))
            elif hasattr(detector, "config") and hasattr(detector.config, "detect_sigma"):
                detect_sigma = float(getattr(detector.config, "detect_sigma"))
            elif isinstance(detector, dict) and "detect_sigma" in detector:
                detect_sigma = float(detector["detect_sigma"])
        except Exception:
            detect_sigma = float("nan")

        if not np.isfinite(detect_sigma) or detect_sigma <= 0:
            detect_sigma = 2.5
        return detect_sigma

    @staticmethod
    def _duration_below_threshold_around_min(
        resid: np.ndarray,
        sigma: np.ndarray,
        k_sigma: float,
    ) -> int:
        x = np.asarray(resid, dtype=float)
        s = np.asarray(sigma, dtype=float)
        if len(x) == 0 or len(s) == 0 or len(x) != len(s):
            return 0
        if not np.any(np.isfinite(x)):
            return 0

        idx = int(np.nanargmin(x))
        if idx < 0 or idx >= len(x):
            return 0

        thr = -float(k_sigma) * s
        if not np.isfinite(thr[idx]) or not np.isfinite(x[idx]) or not (x[idx] <= thr[idx]):
            return 0

        length = 1
        i = idx - 1
        while i >= 0:
            if not np.isfinite(x[i]) or not np.isfinite(thr[i]) or not (x[i] <= thr[i]):
                break
            length += 1
            i -= 1

        i = idx + 1
        while i < len(x):
            if not np.isfinite(x[i]) or not np.isfinite(thr[i]) or not (x[i] <= thr[i]):
                break
            length += 1
            i += 1
        return int(length)

    def _compute_soft_stats(
        self,
        resid_inner: np.ndarray,
        sigma_inner: np.ndarray,
        n_points_in_window: int,
    ) -> Dict[str, Any]:
        x = np.asarray(resid_inner, dtype=float)
        s = np.asarray(sigma_inner, dtype=float)
        npts = int(n_points_in_window)
        if len(x) == 0 or len(s) == 0 or len(x) != len(s):
            return {
                "min_resid_inner": float("nan"),
                "dip_snr_at_min": float("nan"),
                "duration_below_threshold": 0,
                "n_points_in_window": npts,
            }

        sigma_floor = self._sigma_floor(s)
        s = np.where(np.isfinite(s) & (s > 0), s, sigma_floor)
        s = np.maximum(s, sigma_floor)

        if not np.any(np.isfinite(x)):
            return {
                "min_resid_inner": float("nan"),
                "dip_snr_at_min": float("nan"),
                "duration_below_threshold": 0,
                "n_points_in_window": npts,
            }

        idx = int(np.nanargmin(x))
        min_resid = float(x[idx]) if np.isfinite(x[idx]) else float("nan")
        sigma_at_min = float(s[idx]) if np.isfinite(s[idx]) else sigma_floor
        if not np.isfinite(sigma_at_min) or sigma_at_min <= 0:
            sigma_at_min = sigma_floor
        dip_snr = float((-min_resid) / sigma_at_min) if np.isfinite(min_resid) else float("nan")
        if not np.isfinite(dip_snr):
            dip_snr = float("nan")

        duration = self._duration_below_threshold_around_min(
            resid=x,
            sigma=s,
            k_sigma=self.detect_sigma_threshold,
        )
        return {
            "min_resid_inner": float(min_resid),
            "dip_snr_at_min": float(dip_snr),
            "duration_below_threshold": int(duration),
            "n_points_in_window": npts,
        }

    @staticmethod
    def _duration_distribution(series: pd.Series) -> Dict[int, int]:
        if len(series) == 0:
            return {}
        vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        counts = vals.value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.to_dict().items()}

    @staticmethod
    def _fallback_outer_normalize(
        time_outer: np.ndarray,
        flux_outer: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Fallback normalization for very small outer windows.
        """
        t = np.asarray(time_outer, dtype=float)
        f = np.asarray(flux_outer, dtype=float)
        if len(t) == 0:
            return {
                "time": np.asarray([], dtype=float),
                "flux": np.asarray([], dtype=float),
                "local_sigma": np.asarray([], dtype=float),
            }

        med = float(np.nanmedian(f))
        if (not np.isfinite(med)) or abs(med) < 1e-12:
            med = 1.0
        rel = (f / med) - 1.0
        rel_med = float(np.nanmedian(rel))
        resid = rel - rel_med
        mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))))
        sigma = float(1.4826 * mad)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(resid))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1e-5
        sigma_vec = np.full_like(resid, sigma, dtype=float)
        return {
            "time": t.astype(float),
            "flux": resid.astype(float),
            "local_sigma": sigma_vec.astype(float),
        }

    def _run_ranker(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        sigma_local: np.ndarray,
    ) -> List[Any]:
        try:
            out = self._rank_fn(
                query="PERIOD_VALIDATOR",
                author="WINDOW",
                time=time,
                flux=flux,
                sigma_local=sigma_local,
            )
        except TypeError:
            try:
                out = self._rank_fn(time=time, flux=flux, sigma_local=sigma_local)
            except TypeError:
                out = self._rank_fn(time, flux, sigma_local)
        return list(out) if out is not None else []

    def validate(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        P: float,
        t0: Optional[float] = None,
        quality_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validate a proposed period by scoring detector-consistent per-window events.
        """
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)

        if t.shape != f.shape:
            raise ValueError("time and flux must have the same shape")
        if not np.isfinite(P) or P <= 0:
            raise ValueError("P must be finite and > 0")

        keep = np.isfinite(t) & np.isfinite(f)
        if quality_mask is not None:
            q = np.asarray(quality_mask)
            if q.shape != t.shape:
                raise ValueError("quality_mask must have the same shape as time/flux")
            keep &= q.astype(bool)

        t = t[keep]
        f = f[keep]

        if len(t) == 0:
            empty_hits = pd.DataFrame(
                columns=[
                    "tk",
                    "best_shape_score",
                    "best_depth_snr",
                    "best_duration",
                    "best_tmid_offset",
                    "min_resid_inner",
                    "dip_snr_at_min",
                    "duration_below_threshold",
                    "n_points_in_window",
                    "has_candidate",
                ]
            )
            empty_miss = pd.DataFrame(
                columns=[
                    "tk",
                    "best_shape_score",
                    "best_depth_snr",
                    "best_duration",
                    "best_tmid_offset",
                    "min_resid_inner",
                    "dip_snr_at_min",
                    "duration_below_threshold",
                    "n_points_in_window",
                    "has_candidate",
                    "hit_shape",
                    "hit_snr",
                ]
            )
            empty_uncovered = pd.DataFrame(columns=["tk"])
            return {
                "P": float(P),
                "t0": float("nan"),
                "n_predicted": 0,
                "n_covered": 0,
                "coverage_rate": float("nan"),
                "hit_rate_shape": float("nan"),
                "hit_rate_snr": float("nan"),
                "mean_best_shape": float("nan"),
                "mean_best_snr": float("nan"),
                "n_windows_with_no_candidates": 0,
                "frac_no_cand_dip_snr_gt3": float("nan"),
                "duration_below_threshold_dist_no_cand": {},
                "hits_df": empty_hits,
                "misses_df": empty_miss,
                "uncovered_df": empty_uncovered,
                "scores_df": pd.DataFrame(),
            }

        order = np.argsort(t)
        t = t[order]
        f = f[order]

        t_min = float(np.min(t))
        t_max = float(np.max(t))
        base_t0 = float(t_min) if t0 is None else float(t0)

        k_start = int(np.ceil((t_min - base_t0) / float(P)))
        k_end = int(np.floor((t_max - base_t0) / float(P)))

        if k_end < k_start:
            empty_hits = pd.DataFrame(
                columns=[
                    "tk",
                    "best_shape_score",
                    "best_depth_snr",
                    "best_duration",
                    "best_tmid_offset",
                    "min_resid_inner",
                    "dip_snr_at_min",
                    "duration_below_threshold",
                    "n_points_in_window",
                    "has_candidate",
                ]
            )
            empty_miss = pd.DataFrame(
                columns=[
                    "tk",
                    "best_shape_score",
                    "best_depth_snr",
                    "best_duration",
                    "best_tmid_offset",
                    "min_resid_inner",
                    "dip_snr_at_min",
                    "duration_below_threshold",
                    "n_points_in_window",
                    "has_candidate",
                    "hit_shape",
                    "hit_snr",
                ]
            )
            empty_uncovered = pd.DataFrame(columns=["tk"])
            return {
                "P": float(P),
                "t0": float(base_t0),
                "n_predicted": 0,
                "n_covered": 0,
                "coverage_rate": float("nan"),
                "hit_rate_shape": float("nan"),
                "hit_rate_snr": float("nan"),
                "mean_best_shape": float("nan"),
                "mean_best_snr": float("nan"),
                "n_windows_with_no_candidates": 0,
                "frac_no_cand_dip_snr_gt3": float("nan"),
                "duration_below_threshold_dist_no_cand": {},
                "hits_df": empty_hits,
                "misses_df": empty_miss,
                "uncovered_df": empty_uncovered,
                "scores_df": pd.DataFrame(),
            }

        tk_vals = base_t0 + np.arange(k_start, k_end + 1, dtype=float) * float(P)
        rows: List[Dict[str, Any]] = []
        n_windows_with_no_candidates = 0

        for tk in tk_vals:
            in_inner = np.abs(t - float(tk)) <= self.tol_days
            n_points_in_window = int(np.sum(in_inner))
            if not np.any(in_inner):
                rows.append(
                    {
                        "tk": float(tk),
                        "covered": False,
                        "best_shape_score": 0.0,
                        "best_depth_snr": 0.0,
                        "best_duration": float("nan"),
                        "best_tmid_offset": float("nan"),
                        "min_resid_inner": float("nan"),
                        "dip_snr_at_min": float("nan"),
                        "duration_below_threshold": 0,
                        "n_points_in_window": 0,
                        "has_candidate": False,
                        "hit_shape": False,
                        "hit_snr": False,
                        "hit_both": False,
                    }
                )
                continue

            in_outer = np.abs(t - float(tk)) <= self.outer_window_days
            if not np.any(in_outer):
                rows.append(
                    {
                        "tk": float(tk),
                        "covered": False,
                        "best_shape_score": 0.0,
                        "best_depth_snr": 0.0,
                        "best_duration": float("nan"),
                        "best_tmid_offset": float("nan"),
                        "min_resid_inner": float("nan"),
                        "dip_snr_at_min": float("nan"),
                        "duration_below_threshold": 0,
                        "n_points_in_window": n_points_in_window,
                        "has_candidate": False,
                        "hit_shape": False,
                        "hit_snr": False,
                        "hit_both": False,
                    }
                )
                continue

            t_outer = t[in_outer]
            f_outer = f[in_outer]
            min_outer_points = max(int(self.min_duration_cadences) * 2 + 1, 7)
            if len(t_outer) < min_outer_points:
                pre = self._fallback_outer_normalize(t_outer, f_outer)
            else:
                pre = self._preprocess_fn(t_outer, f_outer)

            t_proc_all = np.asarray(pre.get("time", []), dtype=float)
            f_proc_all = np.asarray(pre.get("flux", []), dtype=float)
            s_proc_all = np.asarray(pre.get("local_sigma", np.full_like(f_proc_all, np.nan)), dtype=float)
            if s_proc_all.shape != f_proc_all.shape:
                s_proc_all = np.full_like(f_proc_all, np.nan)

            if len(t_proc_all) == 0 or len(f_proc_all) == 0:
                pre = self._fallback_outer_normalize(t_outer, f_outer)
                t_proc_all = np.asarray(pre.get("time", []), dtype=float)
                f_proc_all = np.asarray(pre.get("flux", []), dtype=float)
                s_proc_all = np.asarray(pre.get("local_sigma", np.full_like(f_proc_all, np.nan)), dtype=float)
                if s_proc_all.shape != f_proc_all.shape:
                    s_proc_all = np.full_like(f_proc_all, np.nan)

            in_inner_proc = np.abs(t_proc_all - float(tk)) <= self.tol_days
            t_proc = t_proc_all[in_inner_proc]
            f_proc = f_proc_all[in_inner_proc]
            s_proc = s_proc_all[in_inner_proc]

            soft = self._compute_soft_stats(
                resid_inner=f_proc,
                sigma_inner=s_proc,
                n_points_in_window=n_points_in_window,
            )

            if len(t_proc) == 0 or len(f_proc) == 0:
                rows.append(
                    {
                        "tk": float(tk),
                        "covered": True,
                        "best_shape_score": 0.0,
                        "best_depth_snr": 0.0,
                        "best_duration": float("nan"),
                        "best_tmid_offset": float("nan"),
                        "min_resid_inner": float(soft["min_resid_inner"]),
                        "dip_snr_at_min": float(soft["dip_snr_at_min"]),
                        "duration_below_threshold": int(soft["duration_below_threshold"]),
                        "n_points_in_window": int(soft["n_points_in_window"]),
                        "has_candidate": False,
                        "hit_shape": False,
                        "hit_snr": False,
                        "hit_both": False,
                    }
                )
                n_windows_with_no_candidates += 1
                continue

            sigma_floor = self._sigma_floor(s_proc_all)
            s_proc = np.where(np.isfinite(s_proc) & (s_proc > 0), s_proc, sigma_floor)
            s_proc = np.maximum(s_proc, sigma_floor)

            candidates = self._run_ranker(t_proc, f_proc, s_proc)
            valid_candidates: List[Any] = []
            for cand in candidates:
                dur = self._candidate_int(cand, "duration_cadences", 0)
                if dur >= int(self.min_duration_cadences):
                    valid_candidates.append(cand)

            if len(valid_candidates) == 0:
                rows.append(
                    {
                        "tk": float(tk),
                        "covered": True,
                        "best_shape_score": 0.0,
                        "best_depth_snr": 0.0,
                        "best_duration": float("nan"),
                        "best_tmid_offset": float("nan"),
                        "min_resid_inner": float(soft["min_resid_inner"]),
                        "dip_snr_at_min": float(soft["dip_snr_at_min"]),
                        "duration_below_threshold": int(soft["duration_below_threshold"]),
                        "n_points_in_window": int(soft["n_points_in_window"]),
                        "has_candidate": False,
                        "hit_shape": False,
                        "hit_snr": False,
                        "hit_both": False,
                    }
                )
                n_windows_with_no_candidates += 1
                continue

            def _rank_key(cand: Any) -> Tuple[float, float]:
                shape = self._candidate_value(cand, "shape_score", float("nan"))
                snr = self._candidate_value(cand, "depth_snr", float("nan"))
                k_shape = shape if np.isfinite(shape) else -np.inf
                k_snr = snr if np.isfinite(snr) else -np.inf
                return (k_shape, k_snr)

            best = max(valid_candidates, key=_rank_key)
            best_shape = self._candidate_value(best, "shape_score", float("nan"))
            best_snr = self._candidate_value(best, "depth_snr", float("nan"))
            if not np.isfinite(best_shape):
                best_shape = 0.0
            if not np.isfinite(best_snr):
                best_snr = 0.0
            best_duration = self._candidate_int(best, "duration_cadences", 0)
            best_tmid = self._candidate_value(best, "t_mid", float("nan"))
            best_tmid_offset = float(best_tmid - float(tk)) if np.isfinite(best_tmid) else float("nan")

            hit_shape = bool(np.isfinite(best_shape) and (best_shape > self.shape_threshold))
            hit_snr = bool(np.isfinite(best_snr) and (best_snr > self.snr_threshold))

            rows.append(
                {
                    "tk": float(tk),
                    "covered": True,
                    "best_shape_score": float(best_shape),
                    "best_depth_snr": float(best_snr),
                    "best_duration": int(best_duration),
                    "best_tmid_offset": float(best_tmid_offset),
                    "min_resid_inner": float(soft["min_resid_inner"]),
                    "dip_snr_at_min": float(soft["dip_snr_at_min"]),
                    "duration_below_threshold": int(soft["duration_below_threshold"]),
                    "n_points_in_window": int(soft["n_points_in_window"]),
                    "has_candidate": True,
                    "hit_shape": hit_shape,
                    "hit_snr": hit_snr,
                    "hit_both": bool(hit_shape and hit_snr),
                }
            )

        scores_df = pd.DataFrame(rows)
        n_predicted = int(len(scores_df))
        covered_df = scores_df[scores_df["covered"]].copy()
        n_covered = int(len(covered_df))
        coverage_rate = float(n_covered / n_predicted) if n_predicted > 0 else float("nan")

        hit_rate_shape = float(covered_df["hit_shape"].mean()) if n_covered > 0 else float("nan")
        hit_rate_snr = float(covered_df["hit_snr"].mean()) if n_covered > 0 else float("nan")
        mean_best_shape = self._safe_nanmean(covered_df["best_shape_score"].to_numpy(dtype=float)) if n_covered > 0 else float("nan")
        mean_best_snr = self._safe_nanmean(covered_df["best_depth_snr"].to_numpy(dtype=float)) if n_covered > 0 else float("nan")

        no_cand_df = covered_df[~covered_df["has_candidate"]].copy() if len(covered_df) > 0 else covered_df
        if len(no_cand_df) > 0:
            frac_no_cand_dip_snr_gt3 = float((pd.to_numeric(no_cand_df["dip_snr_at_min"], errors="coerce") > 3.0).mean())
            duration_below_threshold_dist_no_cand = self._duration_distribution(no_cand_df["duration_below_threshold"])
        else:
            frac_no_cand_dip_snr_gt3 = float("nan")
            duration_below_threshold_dist_no_cand = {}

        base_cols = [
            "tk",
            "best_shape_score",
            "best_depth_snr",
            "best_duration",
            "best_tmid_offset",
            "min_resid_inner",
            "dip_snr_at_min",
            "duration_below_threshold",
            "n_points_in_window",
            "has_candidate",
        ]
        hits_df = covered_df[covered_df["hit_both"]][base_cols].reset_index(drop=True)
        misses_df = covered_df[~covered_df["hit_both"]][base_cols + ["hit_shape", "hit_snr"]].reset_index(drop=True)
        uncovered_df = scores_df[~scores_df["covered"]][["tk"]].reset_index(drop=True)

        return {
            "P": float(P),
            "t0": float(base_t0),
            "n_predicted": n_predicted,
            "n_covered": n_covered,
            "coverage_rate": coverage_rate,
            "hit_rate_shape": hit_rate_shape,
            "hit_rate_snr": hit_rate_snr,
            "mean_best_shape": mean_best_shape,
            "mean_best_snr": mean_best_snr,
            "n_windows_with_no_candidates": int(n_windows_with_no_candidates),
            "frac_no_cand_dip_snr_gt3": float(frac_no_cand_dip_snr_gt3),
            "duration_below_threshold_dist_no_cand": duration_below_threshold_dist_no_cand,
            "hits_df": hits_df,
            "misses_df": misses_df,
            "uncovered_df": uncovered_df,
            "scores_df": scores_df,
        }

    def sweep_periods(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        periods: Iterable[float],
        t0: Optional[float] = None,
        quality_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Evaluate multiple periods and return one summary row per period.
        """
        rows: List[Dict[str, Any]] = []
        for p in periods:
            result = self.validate(
                time=time,
                flux=flux,
                P=float(p),
                t0=t0,
                quality_mask=quality_mask,
            )
            rows.append(
                {
                    "P": float(result["P"]),
                    "t0": float(result["t0"]),
                    "n_predicted": int(result["n_predicted"]),
                    "n_covered": int(result["n_covered"]),
                    "coverage_rate": float(result["coverage_rate"]),
                    "hit_rate_shape": float(result["hit_rate_shape"]),
                    "hit_rate_snr": float(result["hit_rate_snr"]),
                    "mean_best_shape": float(result["mean_best_shape"]),
                    "mean_best_snr": float(result["mean_best_snr"]),
                    "n_windows_with_no_candidates": int(result["n_windows_with_no_candidates"]),
                    "frac_no_cand_dip_snr_gt3": float(result["frac_no_cand_dip_snr_gt3"]),
                }
            )

        out = pd.DataFrame(rows)
        if len(out) == 0:
            return out
        return out.sort_values(
            ["hit_rate_shape", "hit_rate_snr", "coverage_rate", "mean_best_shape", "mean_best_snr"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

    def plot_validation_hitmap(
        self,
        validation_result: Dict[str, Any],
        outpath: Union[str, Path],
    ) -> None:
        """
        Save hit/miss/no-candidate/uncovered map vs predicted transit times.
        """
        import matplotlib.pyplot as plt

        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        scores = validation_result.get("scores_df", pd.DataFrame())
        fig, ax = plt.subplots(figsize=(11, 3.2))

        if isinstance(scores, pd.DataFrame) and len(scores) > 0:
            work = scores.copy()
            work["category"] = work.apply(self._row_category, axis=1)
            y_map = {"uncovered": 0, "no_candidate": 1, "miss": 2, "hit": 3}
            color_map = {
                "uncovered": "gray",
                "no_candidate": "tab:orange",
                "miss": "tab:red",
                "hit": "tab:green",
            }
            for cat in ["uncovered", "no_candidate", "miss", "hit"]:
                sub = work[work["category"] == cat]
                if len(sub) == 0:
                    continue
                ax.scatter(
                    sub["tk"].to_numpy(dtype=float),
                    np.full(len(sub), y_map[cat], dtype=float),
                    s=24,
                    c=color_map[cat],
                    label=cat,
                    alpha=0.85,
                )

            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["uncovered", "no_candidate", "miss", "hit"])
            p = validation_result.get("P", float("nan"))
            t0 = validation_result.get("t0", float("nan"))
            ax.set_title(f"Validation Hitmap P={float(p):.6f}, t0={float(t0):.6f}")
            ax.grid(axis="x", alpha=0.2)
            ax.legend(loc="upper right", frameon=False, ncol=4)
        else:
            ax.text(0.5, 0.5, "No validation rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])

        ax.set_xlabel("Predicted transit time tk")
        ax.set_ylabel("Outcome")
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)

    def plot_scores_vs_phase(
        self,
        validation_result: Dict[str, Any],
        outpath: Union[str, Path],
        score_col: str = "best_shape_score",
    ) -> None:
        """
        Save score-vs-phase scatter for validated windows.
        """
        import matplotlib.pyplot as plt

        if score_col not in {"best_shape_score", "best_depth_snr"}:
            raise ValueError("score_col must be 'best_shape_score' or 'best_depth_snr'")

        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        scores = validation_result.get("scores_df", pd.DataFrame())
        p = float(validation_result.get("P", float("nan")))
        t0 = float(validation_result.get("t0", float("nan")))

        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        if isinstance(scores, pd.DataFrame) and len(scores) > 0 and np.isfinite(p) and p > 0 and np.isfinite(t0):
            work = scores.copy()
            work["category"] = work.apply(self._row_category, axis=1)
            work["phase"] = np.mod((pd.to_numeric(work["tk"], errors="coerce") - t0), p) / p
            work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
            color_map = {
                "uncovered": "gray",
                "no_candidate": "tab:orange",
                "miss": "tab:red",
                "hit": "tab:green",
            }
            for cat in ["uncovered", "no_candidate", "miss", "hit"]:
                sub = work[work["category"] == cat]
                if len(sub) == 0:
                    continue
                ax.scatter(
                    sub["phase"].to_numpy(dtype=float),
                    sub[score_col].to_numpy(dtype=float),
                    s=22,
                    c=color_map[cat],
                    label=cat,
                    alpha=0.85,
                )
            ax.set_xlim(0.0, 1.0)
            ax.grid(alpha=0.2)
            ax.legend(loc="best", frameon=False, ncol=2)
            ax.set_title(f"{score_col} vs phase, P={p:.6f}")
        else:
            ax.text(0.5, 0.5, "No phase-score data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Phase = ((tk - t0) % P) / P")
        ax.set_ylabel(score_col)
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
