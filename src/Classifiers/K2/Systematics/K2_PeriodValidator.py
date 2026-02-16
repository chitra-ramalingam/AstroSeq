from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


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
        snr_window_days, snr_min_points = self._infer_snr_settings(detector)
        self.snr = K2SNR(
            window_days=snr_window_days,
            min_points=snr_min_points,
        )

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
    def _infer_snr_settings(detector: Any) -> Tuple[float, int]:
        window_days = 1.0
        min_points = 20
        try:
            pre_cfg = None
            if hasattr(detector, "preprocessor") and hasattr(detector.preprocessor, "config"):
                pre_cfg = detector.preprocessor.config
            elif hasattr(detector, "config"):
                pre_cfg = detector.config
            elif isinstance(detector, dict):
                pre_cfg = detector.get("preprocess_config", None)

            if pre_cfg is not None:
                if hasattr(pre_cfg, "local_window_days"):
                    v = float(getattr(pre_cfg, "local_window_days"))
                    if np.isfinite(v) and v > 0:
                        window_days = float(v)
                if hasattr(pre_cfg, "local_min_window_cadences"):
                    w = int(getattr(pre_cfg, "local_min_window_cadences"))
                    if w >= 3:
                        min_points = int(w)
        except Exception:
            pass
        return float(window_days), int(min_points)

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

    def _candidate_segment_snr(self, time: np.ndarray, resid: np.ndarray, candidate: Any) -> float:
        i0 = self._candidate_int(candidate, "start_idx", default=-1)
        i1 = self._candidate_int(candidate, "end_idx", default=-1)
        if i0 < 0 or i1 <= i0:
            return 0.0
        stats = self.snr.depth_snr_for_segment(
            time=np.asarray(time, dtype=float),
            flux=np.asarray(resid, dtype=float) + 1.0,
            i0=int(i0),
            i1=int(i1),
        )
        dip_snr = float(stats.get("dip_snr", 0.0))
        return dip_snr if np.isfinite(dip_snr) else 0.0

    def _compute_soft_stats(
        self,
        time_inner: np.ndarray,
        flux_inner: np.ndarray,
        t_center: float,
        n_points_in_window: int,
    ) -> Dict[str, Any]:
        t = np.asarray(time_inner, dtype=float)
        x = np.asarray(flux_inner, dtype=float)
        npts = int(n_points_in_window)
        if len(t) == 0 or len(x) == 0 or len(t) != len(x):
            return {
                "min_resid_inner": float("nan"),
                "dip_snr_at_min": float("nan"),
                "best_t_min_time": float("nan"),
                "duration_below_threshold": 0,
                "n_points_in_window": npts,
            }

        flux_like = x + 1.0
        soft = self.snr.soft_dip_snr_at_time(
            time=t,
            flux=flux_like,
            t_center=float(t_center),
            tol_days=self.tol_days,
        )
        norm = self.snr.normalize(time=t, flux=flux_like)
        resid = np.asarray(norm["resid"], dtype=float)
        in_inner = np.isfinite(t) & np.isfinite(resid) & (np.abs(t - float(t_center)) <= self.tol_days)
        if not np.any(in_inner):
            return {
                "min_resid_inner": float("nan"),
                "dip_snr_at_min": float(soft.get("dip_snr", float("nan"))),
                "best_t_min_time": float("nan"),
                "duration_below_threshold": int(soft.get("duration_below_threshold", 0)),
                "n_points_in_window": npts,
            }

        local_idx = np.where(in_inner)[0]
        idx_min = int(local_idx[int(np.argmin(resid[in_inner]))])
        best_t_min_time = float(t[idx_min]) if np.isfinite(t[idx_min]) else float("nan")
        return {
            "min_resid_inner": float(soft.get("min_resid", float("nan"))),
            "dip_snr_at_min": float(soft.get("dip_snr", float("nan"))),
            "best_t_min_time": float(best_t_min_time),
            "duration_below_threshold": int(soft.get("duration_below_threshold", 0)),
            "n_points_in_window": npts,
        }

    @staticmethod
    def _duration_distribution(series: pd.Series) -> Dict[int, int]:
        if len(series) == 0:
            return {}
        vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        counts = vals.value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.to_dict().items()}

    def _fallback_outer_normalize(
        self,
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

        norm = self.snr.normalize(time=t, flux=f)
        return {
            "time": t.astype(float),
            "flux": np.asarray(norm["resid"], dtype=float),
            "local_sigma": np.asarray(norm["local_sigma"], dtype=float),
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
                    "best_t_min_time",
                    "best_time",
                    "phase_best",
                    "phase_offset",
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
                    "best_t_min_time",
                    "best_time",
                    "phase_best",
                    "phase_offset",
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
                    "best_t_min_time",
                    "best_time",
                    "phase_best",
                    "phase_offset",
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
                    "best_t_min_time",
                    "best_time",
                    "phase_best",
                    "phase_offset",
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
                        "best_t_min_time": float("nan"),
                        "best_time": float("nan"),
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
                        "best_t_min_time": float("nan"),
                        "best_time": float("nan"),
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
                time_inner=t_proc,
                flux_inner=f_proc,
                t_center=float(tk),
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
                        "best_t_min_time": float(soft["best_t_min_time"]),
                        "best_time": float("nan"),
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

            sigma_floor = (
                float(np.nanmedian(s_proc_all[np.isfinite(s_proc_all) & (s_proc_all > 0)]))
                if np.any(np.isfinite(s_proc_all) & (s_proc_all > 0))
                else float(self.snr.sigma_floor_abs)
            )
            if (not np.isfinite(sigma_floor)) or sigma_floor <= 0:
                sigma_floor = float(self.snr.sigma_floor_abs)
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
                        "best_t_min_time": float(soft["best_t_min_time"]),
                        "best_time": float("nan"),
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
                snr = self._candidate_segment_snr(time=t_proc, resid=f_proc, candidate=cand)
                k_shape = shape if np.isfinite(shape) else -np.inf
                k_snr = snr if np.isfinite(snr) else -np.inf
                return (k_shape, k_snr)

            best = max(valid_candidates, key=_rank_key)
            best_shape = self._candidate_value(best, "shape_score", float("nan"))
            best_snr = self._candidate_segment_snr(time=t_proc, resid=f_proc, candidate=best)
            if not np.isfinite(best_shape):
                best_shape = 0.0
            if not np.isfinite(best_snr):
                best_snr = 0.0
            best_duration = self._candidate_int(best, "duration_cadences", 0)
            best_tmid = self._candidate_value(best, "t_mid", float("nan"))
            best_t_min_time = float(soft["best_t_min_time"])
            best_time = float(best_tmid) if np.isfinite(best_tmid) else float(best_t_min_time)
            if not np.isfinite(best_time):
                best_time = float("nan")
            best_tmid_offset = float(best_time - float(tk)) if np.isfinite(best_time) else float("nan")

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
                    "best_t_min_time": float(best_t_min_time),
                    "best_time": float(best_time),
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
        if len(scores_df) > 0:
            scores_df["tk"] = pd.to_numeric(scores_df["tk"], errors="coerce")
            scores_df["best_time"] = pd.to_numeric(scores_df.get("best_time", np.nan), errors="coerce")
            scores_df["phase_best"] = np.mod(scores_df["best_time"] - float(base_t0), float(P)) / float(P)
            phase_offset_raw = (scores_df["best_time"] - scores_df["tk"]) / float(P)
            scores_df["phase_offset"] = np.mod(phase_offset_raw + 0.5, 1.0) - 0.5
        else:
            scores_df["phase_best"] = np.nan
            scores_df["phase_offset"] = np.nan

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
            "best_t_min_time",
            "best_time",
            "phase_best",
            "phase_offset",
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

        fig, (ax_best, ax_offset) = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
        if isinstance(scores, pd.DataFrame) and len(scores) > 0 and np.isfinite(p) and p > 0 and np.isfinite(t0):
            work = scores.copy()
            work["category"] = work.apply(self._row_category, axis=1)
            work["tk"] = pd.to_numeric(work["tk"], errors="coerce")
            work["best_time"] = pd.to_numeric(work.get("best_time", np.nan), errors="coerce")
            work["phase_best"] = np.mod(work["best_time"] - t0, p) / p
            phase_offset_raw = (work["best_time"] - work["tk"]) / p
            work["phase_offset"] = np.mod(phase_offset_raw + 0.5, 1.0) - 0.5
            work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
            color_map = {
                "uncovered": "gray",
                "no_candidate": "tab:orange",
                "miss": "tab:red",
                "hit": "tab:green",
            }
            has_points = False
            for cat in ["uncovered", "no_candidate", "miss", "hit"]:
                sub = work[
                    (work["category"] == cat)
                    & np.isfinite(work["phase_best"])
                    & np.isfinite(work[score_col])
                ]
                if len(sub) == 0:
                    continue
                has_points = True
                ax_best.scatter(
                    sub["phase_best"].to_numpy(dtype=float),
                    sub[score_col].to_numpy(dtype=float),
                    s=22,
                    c=color_map[cat],
                    label=cat,
                    alpha=0.85,
                )
            for cat in ["uncovered", "no_candidate", "miss", "hit"]:
                sub = work[
                    (work["category"] == cat)
                    & np.isfinite(work["phase_offset"])
                    & np.isfinite(work[score_col])
                ]
                if len(sub) == 0:
                    continue
                has_points = True
                ax_offset.scatter(
                    sub["phase_offset"].to_numpy(dtype=float),
                    sub[score_col].to_numpy(dtype=float),
                    s=22,
                    c=color_map[cat],
                    label=cat,
                    alpha=0.85,
                )

            if has_points:
                ax_best.set_xlim(0.0, 1.0)
                ax_offset.set_xlim(-0.5, 0.5)
                ax_best.grid(alpha=0.2)
                ax_offset.grid(alpha=0.2)
                ax_best.legend(loc="best", frameon=False, ncol=2)
                ax_best.set_title(f"{score_col} vs phase_best, P={p:.6f}")
                ax_offset.set_title(f"{score_col} vs phase_offset, P={p:.6f}")
            else:
                ax_best.text(0.5, 0.5, "No phase-score data", ha="center", va="center", transform=ax_best.transAxes)
                ax_offset.text(0.5, 0.5, "No phase-score data", ha="center", va="center", transform=ax_offset.transAxes)
        else:
            ax_best.text(0.5, 0.5, "No phase-score data", ha="center", va="center", transform=ax_best.transAxes)
            ax_offset.text(0.5, 0.5, "No phase-score data", ha="center", va="center", transform=ax_offset.transAxes)

        ax_best.set_xlabel("phase_best = ((best_time - t0) % P) / P")
        ax_offset.set_xlabel("phase_offset = wrap((best_time - tk) / P)")
        ax_best.set_ylabel(score_col)
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)

    @staticmethod
    def _read_plot_csv(path: Union[str, Path]) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _split_plot_frames(
        hits_df: pd.DataFrame,
        misses_df: pd.DataFrame,
        uncovered_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        hits = hits_df.copy()
        misses = misses_df.copy()
        uncovered = uncovered_df.copy()

        if "has_candidate" in misses.columns:
            raw_has_candidate = misses["has_candidate"]
            parsed_numeric = pd.to_numeric(raw_has_candidate, errors="coerce")
            if parsed_numeric.notna().any():
                has_candidate = parsed_numeric.fillna(0).astype(int).astype(bool)
            else:
                as_text = raw_has_candidate.astype(str).str.strip().str.lower()
                has_candidate = as_text.isin({"true", "t", "yes", "y", "1"})
        else:
            has_candidate = pd.Series(True, index=misses.index)

        miss_only = misses[has_candidate].copy()
        no_candidate = misses[~has_candidate].copy()

        return {
            "hit": hits,
            "miss": miss_only,
            "no_candidate": no_candidate,
            "uncovered": uncovered,
        }

    @staticmethod
    def _assert_hit_frame_matches_source(
        hits_df: pd.DataFrame,
        frames: Dict[str, pd.DataFrame],
        context: str,
    ) -> None:
        hit_frame = frames.get("hit", pd.DataFrame())
        if len(hit_frame) != len(hits_df):
            raise AssertionError(
                f"[{context}] hit dataframe mismatch: source_rows={len(hits_df)} split_rows={len(hit_frame)}"
            )

    def plot_hitmap(
        self,
        hits_csv: Union[str, Path],
        misses_csv: Union[str, Path],
        uncovered_csv: Union[str, Path],
        outpath: Union[str, Path],
        P: Optional[float] = None,
    ) -> Path:
        """
        Save hit/no-candidate/miss/uncovered map vs predicted transit time tk from CSVs.
        """
        import matplotlib.pyplot as plt

        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        hits_df = self._read_plot_csv(hits_csv)
        misses_df = self._read_plot_csv(misses_csv)
        uncovered_df = self._read_plot_csv(uncovered_csv)
        frames = self._split_plot_frames(hits_df, misses_df, uncovered_df)
        self._assert_hit_frame_matches_source(hits_df=hits_df, frames=frames, context="plot_hitmap")

        y_map = {"uncovered": 0, "no_candidate": 1, "miss": 2, "hit": 3}
        color_map = {
            "uncovered": "gray",
            "no_candidate": "tab:orange",
            "miss": "tab:red",
            "hit": "tab:green",
        }

        fig, ax = plt.subplots(figsize=(11.0, 3.4))
        has_points = False
        plotted_hit_rows = 0
        hit_tk = pd.to_numeric(hits_df.get("tk", np.nan), errors="coerce")
        expected_hit_rows = int(np.isfinite(hit_tk).sum()) if len(hits_df) > 0 else 0

        for cat in ["uncovered", "no_candidate", "miss", "hit"]:
            sub = frames[cat].copy()
            if len(sub) == 0:
                continue
            sub["tk"] = pd.to_numeric(sub.get("tk", np.nan), errors="coerce")
            sub = sub[np.isfinite(sub["tk"])].copy()
            if len(sub) == 0:
                continue

            y = np.full(len(sub), y_map[cat], dtype=float)
            if cat == "no_candidate":
                dip_snr = pd.to_numeric(sub.get("dip_snr_at_min", 0.0), errors="coerce").fillna(0.0)
                dip_snr = dip_snr.clip(lower=0.0, upper=10.0)
                sizes = 20.0 + 12.0 * dip_snr.to_numpy(dtype=float)

                duration = pd.to_numeric(sub.get("duration_below_threshold", 0.0), errors="coerce").fillna(0.0)
                duration = duration.clip(lower=0.0, upper=10.0)
                linewidths = 0.2 + 0.2 * duration.to_numpy(dtype=float)

                ax.scatter(
                    sub["tk"].to_numpy(dtype=float),
                    y,
                    s=sizes,
                    c=color_map[cat],
                    edgecolors="black",
                    linewidths=linewidths,
                    label=cat,
                    alpha=0.85,
                )
            else:
                ax.scatter(
                    sub["tk"].to_numpy(dtype=float),
                    y,
                    s=28,
                    c=color_map[cat],
                    label=cat,
                    alpha=0.85,
                )
            if cat == "hit":
                plotted_hit_rows += int(len(sub))
            has_points = True

        if expected_hit_rows > 0:
            if plotted_hit_rows <= 0:
                raise AssertionError(
                    f"[plot_hitmap] hit rows exist in CSV ({expected_hit_rows}) but none were plotted"
                )
            if plotted_hit_rows != expected_hit_rows:
                raise AssertionError(
                    f"[plot_hitmap] hit plot count mismatch: expected={expected_hit_rows} plotted={plotted_hit_rows}"
                )

        if has_points:
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["uncovered", "no_candidate", "miss", "hit"])
            ax.grid(axis="x", alpha=0.2)
            ax.legend(loc="upper right", frameon=False, ncol=4)
            if P is not None and np.isfinite(float(P)):
                ax.set_title(f"Validation Hitmap P={float(P):.6f}")
            else:
                ax.set_title("Validation Hitmap")
        else:
            ax.text(0.5, 0.5, "No hitmap data", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])

        ax.set_xlabel("Predicted transit time tk")
        ax.set_ylabel("Outcome")
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        return out_file

    def plot_phase_offset(
        self,
        hits_csv: Union[str, Path],
        misses_csv: Union[str, Path],
        uncovered_csv: Union[str, Path],
        outpath: Union[str, Path],
        P: Optional[float] = None,
        score_col: str = "best_shape_score",
    ) -> Path:
        """
        Save score-vs-phase_offset scatter from CSVs, using x=wrap((best_time - tk)/P).
        """
        import matplotlib.pyplot as plt

        if score_col not in {"best_shape_score", "best_depth_snr"}:
            raise ValueError("score_col must be 'best_shape_score' or 'best_depth_snr'")

        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        hits_df = self._read_plot_csv(hits_csv)
        misses_df = self._read_plot_csv(misses_csv)
        uncovered_df = self._read_plot_csv(uncovered_csv)
        frames = self._split_plot_frames(hits_df, misses_df, uncovered_df)
        self._assert_hit_frame_matches_source(hits_df=hits_df, frames=frames, context="plot_phase_offset")

        color_map = {
            "uncovered": "gray",
            "no_candidate": "tab:orange",
            "miss": "tab:red",
            "hit": "tab:green",
        }

        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        has_points = False
        plotted_hit_rows = 0
        p_value = float(P) if P is not None and np.isfinite(float(P)) and float(P) > 0 else float("nan")
        expected_hit_rows = 0
        if len(frames["hit"]) > 0:
            hit_sub = frames["hit"].copy()
            phase_offset = pd.to_numeric(hit_sub.get("phase_offset", np.nan), errors="coerce")
            if np.isfinite(p_value):
                best_time = pd.to_numeric(hit_sub.get("best_time", np.nan), errors="coerce")
                tk = pd.to_numeric(hit_sub.get("tk", np.nan), errors="coerce")
                phase_offset_raw = (best_time - tk) / p_value
                phase_offset_fallback = np.mod(phase_offset_raw + 0.5, 1.0) - 0.5
                phase_offset = phase_offset.where(np.isfinite(phase_offset), phase_offset_fallback)
            score_vals = pd.to_numeric(hit_sub.get(score_col, np.nan), errors="coerce")
            expected_hit_rows = int(np.sum(np.isfinite(phase_offset) & np.isfinite(score_vals)))

        for cat in ["uncovered", "no_candidate", "miss", "hit"]:
            sub = frames[cat].copy()
            if len(sub) == 0:
                continue

            phase_offset = pd.to_numeric(sub.get("phase_offset", np.nan), errors="coerce")
            if np.isfinite(p_value):
                best_time = pd.to_numeric(sub.get("best_time", np.nan), errors="coerce")
                tk = pd.to_numeric(sub.get("tk", np.nan), errors="coerce")
                phase_offset_raw = (best_time - tk) / p_value
                phase_offset_fallback = np.mod(phase_offset_raw + 0.5, 1.0) - 0.5
                missing_phase = ~np.isfinite(phase_offset)
                phase_offset = phase_offset.where(~missing_phase, phase_offset_fallback)
            sub["phase_offset"] = phase_offset

            sub[score_col] = pd.to_numeric(sub.get(score_col, np.nan), errors="coerce")
            sub = sub[np.isfinite(sub["phase_offset"]) & np.isfinite(sub[score_col])].copy()
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["phase_offset"].to_numpy(dtype=float),
                sub[score_col].to_numpy(dtype=float),
                s=28,
                c=color_map[cat],
                label=cat,
                alpha=0.85,
            )
            if cat == "hit":
                plotted_hit_rows += int(len(sub))
            has_points = True

        if len(frames["hit"]) > 0:
            if expected_hit_rows <= 0:
                raise AssertionError(
                    "[plot_phase_offset] hit rows exist in CSV but none have plottable phase/score values"
                )
            if plotted_hit_rows <= 0:
                raise AssertionError(
                    f"[plot_phase_offset] hit rows exist in CSV ({len(frames['hit'])}) but none were plotted"
                )
            if plotted_hit_rows != expected_hit_rows:
                raise AssertionError(
                    f"[plot_phase_offset] hit plot count mismatch: expected={expected_hit_rows} plotted={plotted_hit_rows}"
                )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        if has_points:
            ax.set_xlim(-0.5, 0.5)
            ax.grid(alpha=0.2)
            ax.legend(loc="best", frameon=False, ncol=2)
            if np.isfinite(p_value):
                ax.set_title(f"{score_col} vs phase_offset, P={p_value:.6f}")
            else:
                ax.set_title(f"{score_col} vs phase_offset")
        else:
            ax.text(0.5, 0.5, "No phase-offset data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("phase_offset = wrap((best_time - tk) / P)")
        ax.set_ylabel(score_col)
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        return out_file
