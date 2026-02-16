from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.Classifiers.K2.Systematics.K2NoiseLoader import (
    K2NoiseConfig,
    K2NoiseLoader,
    K2NoiseLoaderConfig,
)
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


@dataclass
class K2TimeDomainPreprocessConfig:
    """
    Time-domain preprocessing knobs:
    1) local robust normalization
    2) strict thruster-like step masking
    3) asymmetric outlier handling
    """

    local_window_days: float = 1.0
    local_min_window_cadences: int = 31
    thruster_step_sigma: float = 8.0
    thruster_expand_cadences: int = 1
    positive_outlier_sigma: float = 4.0
    positive_clip_sigma: float = 3.0
    negative_outlier_sigma: float = 5.0
    min_negative_run_keep: int = 2


@dataclass
class K2TimeDomainRankConfig:
    """
    Time-domain ranking knobs:
    4) minimum dip duration + ingress/egress coherence
    5) transit-shape score (depth/symmetry/curvature/continuity)
    """

    detect_sigma: float = 2.5
    min_dip_cadences: int = 2
    max_dip_cadences: int = 120
    rank_window_cadences: int = 256
    depth_weight: float = 0.40
    symmetry_weight: float = 0.20
    curvature_weight: float = 0.20
    continuity_weight: float = 0.15
    duration_weight: float = 0.05
    depth_snr_scale: float = 3.0


@dataclass
class K2TransitCandidate:
    query: str
    author: str
    start_idx: int
    end_idx: int
    min_idx: int
    window_start: int
    window_end: int
    t_start: float
    t_end: float
    t_mid: float
    duration_cadences: int
    duration_days: float
    depth: float
    depth_snr: float
    symmetry: float
    curvature: float
    continuity: float
    ingress_egress_ok: bool
    shape_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class K2TimeDomainPreprocessor:
    def __init__(
        self,
        config: Optional[K2TimeDomainPreprocessConfig] = None,
        snr: Optional[K2SNR] = None,
    ) -> None:
        self.config = config if config is not None else K2TimeDomainPreprocessConfig()
        self.snr = snr if snr is not None else K2SNR(
            window_days=float(self.config.local_window_days),
            min_points=int(self.config.local_min_window_cadences),
        )

    def preprocess(
        self,
        time: np.ndarray,
        flux: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)

        if len(t) != len(f):
            raise ValueError("time and flux must have the same length")
        if len(t) == 0:
            return {
                "time": np.asarray([], dtype=float),
                "flux": np.asarray([], dtype=float),
                "local_sigma": np.asarray([], dtype=float),
                "local_baseline": np.asarray([], dtype=float),
                "thruster_mask": np.asarray([], dtype=bool),
            }

        finite = np.isfinite(t) & np.isfinite(f)
        t = t[finite]
        f = f[finite]
        if len(t) == 0:
            return {
                "time": np.asarray([], dtype=float),
                "flux": np.asarray([], dtype=float),
                "local_sigma": np.asarray([], dtype=float),
                "local_baseline": np.asarray([], dtype=float),
                "thruster_mask": np.asarray([], dtype=bool),
            }

        # Apply canonical SNR normalization before artifact handling.
        norm = self.snr.normalize(t, f)
        rel = np.asarray(norm["flux_rel"], dtype=float)

        # Remove step-like cadence artifacts (thruster-style).
        keep = self._thruster_mask(rel)
        t = t[keep]
        f = f[keep]

        # Recompute canonical residual/sigma after masking for cleaner thresholds.
        norm2 = self.snr.normalize(t, f)
        resid = np.asarray(norm2["resid"], dtype=float)
        baseline2 = np.asarray(norm2["baseline"], dtype=float)
        sigma_local = np.asarray(norm2["local_sigma"], dtype=float)
        f_clean = self._asymmetric_outlier_handle(resid, sigma_local)

        return {
            "time": t,
            "flux": f_clean,
            "local_sigma": sigma_local,
            "local_baseline": baseline2,
            "thruster_mask": keep,
        }

    def _thruster_mask(self, rel_flux: np.ndarray) -> np.ndarray:
        cfg = self.config
        x = np.asarray(rel_flux, dtype=float)
        n = len(x)
        if n == 0:
            return np.asarray([], dtype=bool)

        dx = np.diff(x, prepend=x[:1])
        med = float(np.nanmedian(dx))
        mad = float(np.nanmedian(np.abs(dx - med)) + 1e-12)
        sig = 1.4826 * mad
        if not np.isfinite(sig) or sig <= 0:
            return np.ones(n, dtype=bool)

        step_idx = np.where(np.abs(dx - med) > (cfg.thruster_step_sigma * sig))[0]
        keep = np.ones(n, dtype=bool)
        pad = int(max(0, cfg.thruster_expand_cadences))
        for idx in step_idx:
            a = max(0, int(idx) - pad)
            b = min(n, int(idx) + pad + 1)
            keep[a:b] = False

        keep &= np.isfinite(x)
        return keep

    def _asymmetric_outlier_handle(self, rel_flux: np.ndarray, sigma_local: np.ndarray) -> np.ndarray:
        cfg = self.config
        x = np.asarray(rel_flux, dtype=float).copy()
        s = np.asarray(sigma_local, dtype=float).copy()
        if len(x) == 0:
            return x

        fallback = float(np.nanmedian(s[np.isfinite(s) & (s > 0)])) if np.any(np.isfinite(s) & (s > 0)) else 1.0
        s = np.where(np.isfinite(s) & (s > 0), s, fallback)

        # Positive spikes are clipped aggressively.
        pos_limit = cfg.positive_outlier_sigma * s
        pos_clip = cfg.positive_clip_sigma * s
        pos_mask = x > pos_limit
        x[pos_mask] = np.minimum(x[pos_mask], pos_clip[pos_mask])

        # Negative outliers are preserved unless they are isolated 1-cadence glitches.
        neg_mask = x < (-cfg.negative_outlier_sigma * s)
        for a, b in _contiguous_true_runs(neg_mask):
            if (b - a) < int(max(1, cfg.min_negative_run_keep)):
                x[a:b] = 0.0

        return x



class K2TimeDomainTransitRanker:
    def __init__(
        self,
        config: Optional[K2TimeDomainRankConfig] = None,
        snr: Optional[K2SNR] = None,
    ) -> None:
        self.config = config if config is not None else K2TimeDomainRankConfig()
        self.snr = snr if snr is not None else K2SNR()

    def rank_windows(
        self,
        query: str,
        author: str,
        time: np.ndarray,
        flux: np.ndarray,
        sigma_local: np.ndarray,
    ) -> List[K2TransitCandidate]:
        t = np.asarray(time, dtype=float)
        x = np.asarray(flux, dtype=float)
        s = np.asarray(sigma_local, dtype=float)
        cfg = self.config

        if len(t) == 0:
            return []
        if len(t) != len(x) or len(x) != len(s):
            raise ValueError("time/flux/sigma_local must have same length")

        fallback_sigma = float(np.nanmedian(s[np.isfinite(s) & (s > 0)])) if np.any(np.isfinite(s) & (s > 0)) else 1.0
        s = np.where(np.isfinite(s) & (s > 0), s, fallback_sigma)

        dip_mask = x <= (-cfg.detect_sigma * s)
        candidates: List[K2TransitCandidate] = []
        n = len(x)

        for a, b in _contiguous_true_runs(dip_mask):
            run_len = int(b - a)
            if run_len < int(cfg.min_dip_cadences):
                continue
            if run_len > int(cfg.max_dip_cadences):
                continue

            ingress_egress_ok = self._ingress_egress_coherent(x, a, b)
            if not ingress_egress_ok:
                continue

            seg = x[a:b]
            if len(seg) == 0 or not np.any(np.isfinite(seg)):
                continue
            min_local = int(np.nanargmin(seg))
            min_idx = int(a + min_local)

            # Canonical depth/SNR calculation from shared K2SNR utility.
            depth_stats = self.snr.depth_snr_for_segment(
                time=t,
                flux=x + 1.0,
                i0=a,
                i1=b,
            )
            depth = float(depth_stats["dip_depth"])
            depth_snr = float(depth_stats["dip_snr"])
            seg_sigma = float(depth_stats["local_sigma_med"])

            symmetry = self._symmetry_score(min_idx=min_idx, start=a, end=b)
            curvature = self._curvature_score(x=x, min_idx=min_idx, sigma=seg_sigma)
            continuity = self._continuity_score(seg)
            duration_score = min(1.0, run_len / max(int(cfg.min_dip_cadences), 1))
            depth_term = depth_snr / (depth_snr + float(cfg.depth_snr_scale))

            shape_score = (
                cfg.depth_weight * depth_term
                + cfg.symmetry_weight * symmetry
                + cfg.curvature_weight * curvature
                + cfg.continuity_weight * continuity
                + cfg.duration_weight * duration_score
            )
            shape_score = float(np.clip(shape_score, 0.0, 1.0))

            ws, we = self._window_bounds(center=min_idx, n=n, width=int(cfg.rank_window_cadences))
            t0 = float(t[a])
            t1 = float(t[b - 1])
            duration_days = float(t1 - t0) if b > a else 0.0
            tmid = float(t[min_idx])

            candidates.append(
                K2TransitCandidate(
                    query=str(query),
                    author=str(author),
                    start_idx=int(a),
                    end_idx=int(b),
                    min_idx=int(min_idx),
                    window_start=int(ws),
                    window_end=int(we),
                    t_start=t0,
                    t_end=t1,
                    t_mid=tmid,
                    duration_cadences=run_len,
                    duration_days=duration_days,
                    depth=depth,
                    depth_snr=depth_snr,
                    symmetry=float(symmetry),
                    curvature=float(curvature),
                    continuity=float(continuity),
                    ingress_egress_ok=bool(ingress_egress_ok),
                    shape_score=shape_score,
                )
            )

        candidates.sort(key=lambda c: c.shape_score, reverse=True)
        return candidates

    @staticmethod
    def _ingress_egress_coherent(x: np.ndarray, start: int, end: int) -> bool:
        n = len(x)
        if start >= end:
            return False
        pre = x[max(0, start - 2):start]
        post = x[end:min(n, end + 2)]
        left_edge = x[start:min(end, start + 2)]
        right_edge = x[max(start, end - 2):end]

        pre_med = float(np.nanmedian(pre)) if len(pre) > 0 else 0.0
        post_med = float(np.nanmedian(post)) if len(post) > 0 else 0.0
        left_med = float(np.nanmedian(left_edge))
        right_med = float(np.nanmedian(right_edge))

        return (left_med < pre_med) and (right_med < post_med)

    @staticmethod
    def _symmetry_score(min_idx: int, start: int, end: int) -> float:
        left = int(max(1, min_idx - start + 1))
        right = int(max(1, end - min_idx))
        return float(np.clip(1.0 - abs(left - right) / max(left + right, 1), 0.0, 1.0))

    @staticmethod
    def _curvature_score(x: np.ndarray, min_idx: int, sigma: float) -> float:
        n = len(x)
        a = max(0, min_idx - 1)
        b = min(n, min_idx + 2)
        bottom = x[a:b]
        spread = float(np.nanstd(bottom)) if len(bottom) > 0 else 0.0
        return float(1.0 / (1.0 + (spread / (sigma + 1e-12))))

    @staticmethod
    def _continuity_score(seg: np.ndarray) -> float:
        y = np.asarray(seg, dtype=float)
        if len(y) < 3:
            return 1.0
        d = np.diff(y)
        s = np.sign(d)
        flips = np.sum(s[1:] != s[:-1])
        score = 1.0 - float(flips) / max(len(s) - 1, 1)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _window_bounds(center: int, n: int, width: int) -> Tuple[int, int]:
        w = int(max(3, width))
        half = w // 2
        s = max(0, int(center) - half)
        e = min(n, s + w)
        if (e - s) < w:
            s = max(0, e - w)
        return int(s), int(e)


class K2TimeDomainTransitPipeline:
    """
    Time-domain K2 candidate finder with strict quality handling and morphology ranking.

    This class intentionally composes existing K2 noise tooling:
    - K2NoiseLoader for orchestration config and summary compatibility
    - K2_NoiseHandler (via loader.handler) for strict quality masking and K2 product selection
    """

    def __init__(
        self,
        loader: Optional[K2NoiseLoader] = None,
        loader_config: Optional[K2NoiseLoaderConfig] = None,
        noise_config: Optional[K2NoiseConfig] = None,
        preprocess_config: Optional[K2TimeDomainPreprocessConfig] = None,
        rank_config: Optional[K2TimeDomainRankConfig] = None,
        handler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        pre_cfg = preprocess_config if preprocess_config is not None else K2TimeDomainPreprocessConfig()
        shared_snr = K2SNR(
            window_days=float(pre_cfg.local_window_days),
            min_points=int(pre_cfg.local_min_window_cadences),
        )

        if loader is not None:
            self.loader = loader
        else:
            kwargs = dict(handler_kwargs or {})
            kwargs.setdefault("quality_strict", True)
            self.loader = K2NoiseLoader(
                loader_config=loader_config if loader_config is not None else K2NoiseLoaderConfig(per_segment=True),
                noise_config=noise_config if noise_config is not None else K2NoiseConfig(mode="strict"),
                handler_kwargs=kwargs,
            )

        self.snr = shared_snr
        self.preprocessor = K2TimeDomainPreprocessor(config=pre_cfg, snr=shared_snr)
        self.ranker = K2TimeDomainTransitRanker(config=rank_config, snr=shared_snr)

    def run_one(
        self,
        query: str,
        limit: Optional[int] = None,
        exptime: Optional[Union[str, float]] = None,
        cache_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        q = str(query).strip()
        if q == "":
            raise ValueError("query must be non-empty")

        # Reuse existing loader summary for compatibility with existing reporting CSVs.
        noise_row = self.loader.run_one(
            query=q,
            limit=limit,
            exptime=exptime,
            flatten=False,
            per_segment=True,
            cache_only=cache_only,
        )
        if noise_row.get("status") != "ok":
            out = dict(noise_row)
            out["n_candidates"] = 0
            out["best_shape_score"] = np.nan
            out["shape_rank_method"] = "time_domain"
            return {"summary": out, "candidates": []}

        fetched = self.loader.handler.fetch_best(
            query=q,
            limit=limit or self.loader.loader_config.limit,
            exptime=exptime,
            cache_only=bool(self.loader.loader_config.cache_only if cache_only is None else cache_only),
        )
        if str(fetched.get("status", "ok")).lower() != "ok":
            out = dict(noise_row)
            out["status"] = str(fetched.get("status", "error"))
            out["n_candidates"] = 0
            out["best_shape_score"] = np.nan
            out["shape_rank_method"] = "time_domain"
            return {"summary": out, "candidates": []}

        # Strict quality filtering is forced here, and we disable symmetric sigma clipping.
        cleaned = self.loader.handler.clean(
            fetched["lc"],
            normalize=False,
            remove_nans=True,
            quality_mask=True,
            sigma_clip=False,
            flatten=False,
        )

        pre = self.preprocessor.preprocess(cleaned["time"], cleaned["flux"])
        t = pre["time"]
        x = pre["flux"]
        s = pre["local_sigma"]

        candidates = self.ranker.rank_windows(
            query=q,
            author=str(fetched.get("author", "")),
            time=t,
            flux=x,
            sigma_local=s,
        )

        cand_rows = [c.to_dict() for c in candidates]
        best = float(candidates[0].shape_score) if len(candidates) > 0 else float("nan")
        mean_score = float(np.nanmean([c.shape_score for c in candidates])) if len(candidates) > 0 else float("nan")

        summary = dict(noise_row)
        summary.update(
            {
                "author": str(fetched.get("author", summary.get("author", ""))),
                "n_points_after_preprocess": int(len(t)),
                "n_candidates": int(len(candidates)),
                "best_shape_score": best,
                "mean_shape_score": mean_score,
                "shape_rank_method": "time_domain",
            }
        )

        return {"summary": summary, "candidates": cand_rows}

    def run_queries(
        self,
        queries: Iterable[str],
        limit: Optional[int] = None,
        exptime: Optional[Union[str, float]] = None,
        cache_only: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        summary_rows: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []

        for q in queries:
            result = self.run_one(query=str(q), limit=limit, exptime=exptime, cache_only=cache_only)
            summary_rows.append(result["summary"])
            candidate_rows.extend(result["candidates"])

        return {
            "summary": pd.DataFrame(summary_rows),
            "candidates": pd.DataFrame(candidate_rows),
        }


def infer_periods_from_events(
    events_df: pd.DataFrame,
    max_period: float = 40.0,
    min_hits: int = 3,
    tol_frac: float = 0.01,
) -> Tuple[List[Tuple[float, int, List[Any]]], pd.DataFrame]:
    """
    Infer candidate periods from detected event mid-times.

    Args:
        events_df:
            DataFrame that must include a `t_mid` column.
            Input index values are preserved as event identifiers.
        max_period:
            Maximum candidate period to consider from pairwise differences.
        min_hits:
            Minimum number of supporting events required for ranked output.
        tol_frac:
            Relative clustering tolerance. A period `p` is assigned to a cluster
            center `c` if `abs(p-c) <= tol_frac * c`.

    Returns:
        ranked_periods:
            List of tuples:
            (period, count_hits, supporting_event_indices)
            sorted by count_hits desc, pair_count desc, period asc.
        hist_df:
            Plot-ready cluster summary DataFrame with columns:
            period, period_low, period_high, pair_count, count_hits,
            supporting_event_indices, passes_min_hits.
    """
    if "t_mid" not in events_df.columns:
        raise ValueError("events_df must contain a 't_mid' column")
    if max_period <= 0:
        raise ValueError("max_period must be > 0")
    if min_hits < 1:
        raise ValueError("min_hits must be >= 1")
    if tol_frac <= 0:
        raise ValueError("tol_frac must be > 0")

    times = pd.to_numeric(events_df["t_mid"], errors="coerce")
    ok = np.isfinite(times.to_numpy(dtype=float))
    if not np.any(ok):
        empty = pd.DataFrame(
            columns=[
                "period",
                "period_low",
                "period_high",
                "pair_count",
                "count_hits",
                "supporting_event_indices",
                "passes_min_hits",
            ]
        )
        return [], empty

    work = events_df.loc[ok].copy()
    work = work.assign(t_mid=pd.to_numeric(work["t_mid"], errors="coerce"))
    work = work.sort_values("t_mid").copy()
    t = work["t_mid"].to_numpy(dtype=float)
    event_ids = list(work.index)

    period_samples: List[float] = []
    sample_pairs: List[Tuple[Any, Any]] = []
    n = len(t)
    for i in range(n - 1):
        ti = float(t[i])
        for j in range(i + 1, n):
            dt = float(t[j] - ti)
            if dt <= 0.0 or dt > float(max_period):
                continue
            period_samples.append(dt)
            sample_pairs.append((event_ids[i], event_ids[j]))

    if len(period_samples) == 0:
        empty = pd.DataFrame(
            columns=[
                "period",
                "period_low",
                "period_high",
                "pair_count",
                "count_hits",
                "supporting_event_indices",
                "passes_min_hits",
            ]
        )
        return [], empty

    samples_sorted = sorted(zip(period_samples, sample_pairs), key=lambda x: x[0])
    clusters: List[Dict[str, Any]] = []

    for period, pair in samples_sorted:
        best_idx = -1
        best_gap = float("inf")

        for idx, c in enumerate(clusters):
            center = float(c["center"])
            tol = float(tol_frac) * max(center, 1e-12)
            gap = abs(float(period) - center)
            if gap <= tol and gap < best_gap:
                best_idx = idx
                best_gap = gap

        if best_idx < 0:
            clusters.append(
                {
                    "center": float(period),
                    "periods": [float(period)],
                    "pairs": [pair],
                    "support": {pair[0], pair[1]},
                }
            )
            continue

        c = clusters[best_idx]
        c["periods"].append(float(period))
        c["pairs"].append(pair)
        c["support"].update(pair)
        c["center"] = float(np.median(np.asarray(c["periods"], dtype=float)))

    hist_rows: List[Dict[str, Any]] = []
    ranked_rows: List[Tuple[float, int, List[Any], int]] = []

    for c in clusters:
        periods = np.asarray(c["periods"], dtype=float)
        support_indices = sorted(c["support"])
        pair_count = int(len(c["pairs"]))
        count_hits = int(len(support_indices))
        period_center = float(np.median(periods))
        low = float(np.min(periods))
        high = float(np.max(periods))

        hist_rows.append(
            {
                "period": period_center,
                "period_low": low,
                "period_high": high,
                "pair_count": pair_count,
                "count_hits": count_hits,
                "supporting_event_indices": support_indices,
                "passes_min_hits": bool(count_hits >= int(min_hits)),
            }
        )

        if count_hits >= int(min_hits):
            ranked_rows.append((period_center, count_hits, support_indices, pair_count))

    ranked_rows.sort(key=lambda x: (-x[1], -x[3], x[0]))
    ranked_periods = [(p, hits, idxs) for (p, hits, idxs, _pair_count) in ranked_rows]

    hist_df = pd.DataFrame(hist_rows).sort_values("period").reset_index(drop=True)
    return ranked_periods, hist_df


def phase_cluster_score(
    events_df: pd.DataFrame,
    P: float,
    tol_phase: float = 0.03,
) -> Tuple[int, float, List[Any]]:
    """
    Score phase concentration for a proposed period.

    Args:
        events_df:
            DataFrame that must include `t_mid`.
            Input index values are preserved as event identifiers.
        P:
            Proposed period (same units as t_mid).
        tol_phase:
            Cluster width on [0, 1) phase circle.

    Returns:
        (cluster_count, cluster_center_phase, in_cluster_indices)
    """
    if "t_mid" not in events_df.columns:
        raise ValueError("events_df must contain a 't_mid' column")
    if not np.isfinite(P) or P <= 0:
        raise ValueError("P must be a finite positive number")
    if not np.isfinite(tol_phase) or tol_phase <= 0 or tol_phase > 1:
        raise ValueError("tol_phase must be in (0, 1]")

    t = pd.to_numeric(events_df["t_mid"], errors="coerce")
    ok = np.isfinite(t.to_numpy(dtype=float))
    if not np.any(ok):
        print(f"[phase_cluster_score] P={P:.8f} no valid t_mid values")
        return 0, float("nan"), []

    work = events_df.loc[ok].copy()
    work = work.assign(t_mid=pd.to_numeric(work["t_mid"], errors="coerce"))
    work = work.assign(phase=(np.mod(work["t_mid"].to_numpy(dtype=float), float(P)) / float(P)))
    work = work.sort_values("phase")

    phases = work["phase"].to_numpy(dtype=float)
    ids = list(work.index)
    n = len(phases)

    phase_print = work[["t_mid", "phase"]].copy()
    phase_print.insert(0, "event_index", phase_print.index)
    print(f"\n[phase_cluster_score] P={P:.8f} tol_phase={tol_phase:.4f}")
    print("[phase_cluster_score] phases sorted:")
    print(phase_print.to_string(index=False))

    # Circular sweep: duplicate phases shifted by +1.
    phase2 = np.concatenate([phases, phases + 1.0])
    id2 = ids + ids

    best_count = 0
    best_start = 0
    best_end = -1

    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < i + n and (phase2[j + 1] - phase2[i]) <= float(tol_phase) + 1e-12:
            j += 1

        count = int(j - i + 1)
        if count > best_count:
            best_count = count
            best_start = i
            best_end = j

    if best_count <= 0:
        return 0, float("nan"), []

    cluster_phases = np.asarray(phase2[best_start:best_end + 1], dtype=float)
    cluster_ids = id2[best_start:best_end + 1]
    # Unique while preserving order.
    seen = set()
    in_cluster_indices: List[Any] = []
    for idx in cluster_ids:
        if idx in seen:
            continue
        seen.add(idx)
        in_cluster_indices.append(idx)

    # Circular mean in [0,1).
    theta = 2.0 * np.pi * np.mod(cluster_phases, 1.0)
    mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    cluster_center_phase = float(mean_angle / (2.0 * np.pi))

    return int(best_count), cluster_center_phase, in_cluster_indices


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    n = len(m)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < n and m[j]:
            j += 1
        runs.append((int(i), int(j)))
        i = j
    return runs
