from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import lightkurve as lk
except Exception as e:
    lk = None  # allows import in environments without lightkurve


@dataclass
class K2NoiseMetrics:
    n_points: int
    baseline_days: float
    duty_cycle: float
    mad: float
    robust_sigma: float
    outlier_rate_6sigma: float
    step_score: float
    whiteness_score: float  # interpreted by K2NoiseConfig.whiteness_score_definition
    outlier_rate_global: float = np.nan
    notes: str = ""


@dataclass
class K2NoiseConfig:
    mode: str = "strict"
    min_points: int = 800
    min_baseline_days: float = 10.0
    min_robust_sigma: float = 0.0
    max_outlier_rate_6sigma: Optional[float] = None
    catastrophic_outlier_rate_6sigma: float = 0.10
    max_step_score: float = 1.5
    max_whiteness_score: Optional[float] = None
    whiteness_score_definition: str = "pvalue"  # "pvalue" or "statistic"
    whiteness_alpha: Optional[float] = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower().strip()
        if self.mode not in {"strict", "discovery"}:
            raise ValueError(f"Unsupported K2NoiseConfig mode: {self.mode!r}. Expected 'strict' or 'discovery'.")
        self.whiteness_score_definition = str(self.whiteness_score_definition).lower().strip()
        if self.whiteness_score_definition not in {"pvalue", "statistic"}:
            raise ValueError(
                f"Unsupported whiteness_score_definition: {self.whiteness_score_definition!r}. "
                "Expected 'pvalue' or 'statistic'."
            )
        if self.whiteness_alpha is not None:
            self.whiteness_alpha = float(self.whiteness_alpha)
            if (not np.isfinite(self.whiteness_alpha)) or (self.whiteness_alpha <= 0.0) or (self.whiteness_alpha >= 1.0):
                raise ValueError("whiteness_alpha must be a finite value in (0, 1).")
        self.apply_mode_preset(overwrite=False)

    def apply_mode_preset(self, overwrite: bool = True) -> None:
        if self.mode not in {"strict", "discovery"}:
            raise ValueError(f"Unsupported K2NoiseConfig mode: {self.mode!r}. Expected 'strict' or 'discovery'.")
        if self.mode == "discovery":
            preset_outlier = 0.08
            preset_whiteness_stat = 0.8
            preset_whiteness_alpha = 0.05
        else:
            preset_outlier = 0.02
            preset_whiteness_stat = 0.6
            preset_whiteness_alpha = 0.01

        if overwrite or (self.max_outlier_rate_6sigma is None):
            self.max_outlier_rate_6sigma = preset_outlier
        if self.whiteness_score_definition == "pvalue":
            if overwrite or (self.whiteness_alpha is None):
                self.whiteness_alpha = preset_whiteness_alpha
        else:
            if overwrite or (self.max_whiteness_score is None):
                self.max_whiteness_score = preset_whiteness_stat


class K2PipelineStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        exc: Exception,
        author_selected: str = "",
        campaign_selected: str = "",
    ) -> None:
        self.stage = str(stage)
        self.error_type = type(exc).__name__
        self.error_msg = str(exc)[:200]
        self.author_selected = str(author_selected)
        self.campaign_selected = str(campaign_selected)
        super().__init__(f"{self.stage}:{self.error_type}:{self.error_msg}")
        self.__cause__ = exc


class K2_NoiseHandler:
    """
    K2_NoiseHandler — a "clean room" for K2 lightcurves.

    Goals:
    - pick best available provenance (EVEREST → K2SFF → K2)
    - hard-mask poisoned cadences (quality != 0)
    - normalize robustly
    - optionally flatten/detrend without murdering transits
    - compute noise metrics for gating and ranking

    It’s intentionally conservative — better to discard cursed stars than
    to feed AstroSeq garbage and wonder why everything looks like dips.
    """

    AUTHOR_PRIORITY = ("EVEREST", "K2SFF", "K2")

    def __init__(
        self,
        author_priority: Tuple[str, ...] = AUTHOR_PRIORITY,
        quality_strict: bool = True,
        sigma_clip: float = 6.0,
        max_gap_days: float = 0.5,
        flatten_window_days: float = 1.0,
        min_points: Optional[int] = None,
        min_baseline_days: Optional[float] = None,
        mode: Optional[str] = None,
        verbose: bool = False,
        noise_config: Optional[K2NoiseConfig] = None,
        min_robust_sigma: Optional[float] = None,
        max_outlier_rate_6sigma: Optional[float] = None,
        catastrophic_outlier_rate_6sigma: Optional[float] = None,
        max_step_score: Optional[float] = None,
        max_whiteness_score: Optional[float] = None,
        whiteness_score_definition: Optional[str] = None,
        whiteness_alpha: Optional[float] = None,
    ):
        self.author_priority = author_priority
        self.quality_strict = quality_strict
        self.sigma_clip = float(sigma_clip)
        self.max_gap_days = float(max_gap_days)
        self.flatten_window_days = float(flatten_window_days)

        cfg = replace(noise_config) if noise_config is not None else K2NoiseConfig()
        if mode is not None:
            cfg.mode = str(mode).lower().strip()
        if whiteness_score_definition is not None:
            cfg.whiteness_score_definition = str(whiteness_score_definition).lower().strip()
        cfg.whiteness_score_definition = str(cfg.whiteness_score_definition).lower().strip()
        if cfg.whiteness_score_definition not in {"pvalue", "statistic"}:
            raise ValueError(
                f"Unsupported whiteness_score_definition: {cfg.whiteness_score_definition!r}. "
                "Expected 'pvalue' or 'statistic'."
            )
        if min_points is not None:
            cfg.min_points = int(min_points)
        if min_baseline_days is not None:
            cfg.min_baseline_days = float(min_baseline_days)
        if min_robust_sigma is not None:
            cfg.min_robust_sigma = float(min_robust_sigma)
        if max_outlier_rate_6sigma is not None:
            cfg.max_outlier_rate_6sigma = float(max_outlier_rate_6sigma)
        if catastrophic_outlier_rate_6sigma is not None:
            cfg.catastrophic_outlier_rate_6sigma = float(catastrophic_outlier_rate_6sigma)
        if max_step_score is not None:
            cfg.max_step_score = float(max_step_score)
        if max_whiteness_score is not None:
            cfg.max_whiteness_score = float(max_whiteness_score)
        if whiteness_alpha is not None:
            cfg.whiteness_alpha = float(whiteness_alpha)
        if cfg.whiteness_alpha is not None:
            if (not np.isfinite(cfg.whiteness_alpha)) or (cfg.whiteness_alpha <= 0.0) or (cfg.whiteness_alpha >= 1.0):
                raise ValueError("whiteness_alpha must be a finite value in (0, 1).")
        cfg.apply_mode_preset(overwrite=False)
        self.noise_config = cfg

        # Backwards-compatible mirrors for existing code that reads these attrs.
        self.min_points = int(self.noise_config.min_points)
        self.min_baseline_days = float(self.noise_config.min_baseline_days)
        self.verbose = bool(verbose)

        if lk is None:
            raise ImportError("lightkurve is required to use K2_NoiseHandler")

    # ----------------------------
    # Public API
    # ----------------------------

    def fetch_best(
        self,
        query: str,
        limit: int = 50,
        exptime: Optional[Union[str, float]] = None,  # "long" / "short" or seconds
        download_dir: Optional[str] = None,
        cache_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Download the best K2 lightcurve for a target.

        Returns:
            {
                "status": "ok" | "cache_miss",
                "lc": LightCurve,
                "author": str,
                "search_result": dict (serializable metadata about the search),
                "cache_path": Optional[str],
                "cache_source": str,
            }
        """
        author = ""
        campaign = ""
        direct_local = self._find_direct_local_product(query=query, download_dir=download_dir, exptime=exptime)
        if direct_local is not None:
            path = str(direct_local["cache_path"])
            print(
                f"[select local] EPIC {direct_local['epic']} "
                f"author={direct_local['author']} mission=K2 campaign={direct_local['campaign']}"
            )
            print(f"[lc path] EPIC {direct_local['epic']} -> {path}")
            print(f"[cache source] source={self._cache_source_from_path(path)}")
            try:
                lc = lk.read(direct_local["cache_path"])
            except Exception as e:
                raise K2PipelineStageError(
                    stage="load",
                    exc=e,
                    author_selected=str(direct_local["author"]),
                    campaign_selected=str(direct_local["campaign"]),
                )
            return {
                "status": "ok",
                "lc": lc,
                "author": str(direct_local["author"]),
                "author_selected": str(direct_local["author"]),
                "campaign_selected": str(direct_local["campaign"]),
                "search_result": dict(direct_local["search_result"]),
                "cache_path": path,
                "cache_source": self._cache_source_from_path(path),
            }
        if cache_only:
            epic = self._infer_epic_id(query=query, selected_row=None)
            print(f"[lc path] EPIC {epic} -> None")
            print(f"[lc path] EPIC {epic} -> no direct local cached file; skipping search because cache_only=True")
            print("[cache source] source=local_direct_cache_miss")
            return {
                "status": "cache_miss",
                "lc": None,
                "author": "",
                "author_selected": "",
                "campaign_selected": "",
                "search_result": {
                    "query": str(query),
                    "selected_index": -1,
                    "selection_reason": "direct_local_cache_miss_no_search",
                    "n_results": 0,
                    "author_priority": [str(a) for a in self.author_priority],
                    "results": [],
                },
                "cache_path": None,
                "cache_source": "local_direct_cache_miss",
            }
        try:
            sr = lk.search_lightcurve(
                query,
                mission="K2",
                author=list(self.author_priority),
                limit=limit,
                exptime=exptime,
            )
        except Exception as e:
            raise K2PipelineStageError(stage="search", exc=e)

        if len(sr) == 0:
            raise K2PipelineStageError(stage="search", exc=ValueError(f"No K2 lightcurve found for query={query!r}"))

        picked = self.choose_best(sr)
        best = picked["product"]
        epic = self._infer_epic_id(query=query, selected_row=best)
        author = self._author_as_str(picked.get("author", ""))
        mission = self._extract_search_field(best, "mission", default="")
        campaign = str(self._extract_search_field(best, "campaign", default=""))
        print(f"[select] EPIC {epic} author={author} mission={mission} campaign={campaign}")

        search_meta = self._serialize_search_result(sr, picked["index"], picked["reason"], query=query)

        if cache_only:
            cached_path = self._find_cached_product_path(best, download_dir=download_dir)
            if cached_path is None:
                print(f"[lc path] EPIC {epic} -> None")
                print(f"[lc path] EPIC {epic} -> no local cached file for selected product; download_dir={download_dir}")
                print(f"[cache source] source=custom_or_unknown")
                return {
                    "status": "cache_miss",
                    "lc": None,
                    "author": author,
                    "author_selected": author,
                    "campaign_selected": campaign,
                    "search_result": search_meta,
                    "cache_path": None,
                    "cache_source": "custom_or_unknown",
                }
            try:
                lc = lk.read(cached_path)
            except Exception as e:
                raise K2PipelineStageError(
                    stage="load",
                    exc=e,
                    author_selected=author,
                    campaign_selected=campaign,
                )
            path = str(cached_path)
            print(f"[lc path] EPIC {epic} -> {path}")
            print(f"[cache source] source={self._cache_source_from_path(path)}")
            return {
                "status": "ok",
                "lc": lc,
                "author": author,
                "author_selected": author,
                "campaign_selected": campaign,
                "search_result": search_meta,
                "cache_path": path,
                "cache_source": self._cache_source_from_path(path),
            }

        try:
            if download_dir is None:
                lc = best.download()
            else:
                lc = best.download(download_dir=download_dir)
        except Exception as e:
            raise K2PipelineStageError(
                stage="download",
                exc=e,
                author_selected=author,
                campaign_selected=campaign,
            )

        has_path_attr = any(hasattr(lc, attr) for attr in ("path", "local_path"))
        path = self._extract_download_path(lc)
        print(f"[lc path] EPIC {epic} -> {path}")
        if not has_path_attr:
            print(f"[lc path] EPIC {epic} -> no path attribute; download_dir={download_dir}")
        elif path is None:
            print(f"[lc path] EPIC {epic} -> path attribute exists but is empty; download_dir={download_dir}")
        print(f"[cache source] source={self._cache_source_from_path(path)}")

        if lc is None:
            raise K2PipelineStageError(
                stage="download",
                exc=ValueError(f"Download returned no light curve for query={query!r}"),
                author_selected=author,
                campaign_selected=campaign,
            )
        return {
            "status": "ok",
            "lc": lc,
            "author": author,
            "author_selected": author,
            "campaign_selected": campaign,
            "search_result": search_meta,
            "cache_path": path,
            "cache_source": self._cache_source_from_path(path),
        }

    def clean(
        self,
        lc,
        normalize: bool = True,
        remove_nans: bool = True,
        quality_mask: bool = True,
        sigma_clip: bool = True,
        flatten: bool = False,
        flatten_mask_transits: Optional[np.ndarray] = None,
    ):
        """
        Clean a LightCurve into time/flux arrays that are stable for triage.

        Args:
            flatten — if True, flatten long-term trends (transit-preserving-ish if masked)
            flatten_mask_transits — boolean mask over points to exclude from flattening

        Returns:
            dict with keys:
              time, flux, flux_raw, mask_good, lc_clean (LightCurve), notes
        """
        notes: List[str] = []

        lc2 = lc

        if remove_nans:
            lc2 = lc2.remove_nans()

        # Quality mask — the “no haunted cadences” rule
        if quality_mask and hasattr(lc2, "quality") and lc2.quality is not None:
            q = np.asarray(lc2.quality)
            if self.quality_strict:
                good = q == 0
            else:
                # lenient option — keep small flags sometimes, but still ditch big mess
                good = q < 2**10
            lc2 = lc2[good]
            notes.append(f"quality_masked={np.sum(~good)}")

        t, f = self._to_arrays(lc2)

        if len(t) == 0:
            raise ValueError("Lightcurve became empty after masking/NaN removal")

        f_raw = f.copy()

        # Robust normalize — median to 0-centered relative flux
        if normalize:
            med = np.nanmedian(f)
            if not np.isfinite(med) or med == 0:
                notes.append("normalize_skipped_bad_median")
            else:
                f = (f / med) - 1.0

        # Sigma clip in residual space — robustly remove spikes
        mask_good = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask_good], f[mask_good]

        if sigma_clip and len(f) > 10:
            med = np.nanmedian(f)
            mad = np.nanmedian(np.abs(f - med)) + 1e-12
            rsig = 1.4826 * mad
            keep = np.abs(f - med) <= self.sigma_clip * rsig
            notes.append(f"sigma_clipped={np.sum(~keep)}")
            t, f = t[keep], f[keep]
            mask_good = mask_good.copy()
            # mask_good refers to pre-clip indices — we’ll return final arrays anyway

        # Optional flatten — beware: flattening can distort transits if not masked
        if flatten:
            try:
                lc_tmp = self._from_arrays(lc2, t, f)
                if flatten_mask_transits is not None:
                    # flatten expects mask where True means "in-transit" to be excluded
                    m = np.asarray(flatten_mask_transits).astype(bool)
                    if len(m) == len(lc_tmp.flux):
                        lc_tmp = lc_tmp.flatten(
                            window_length=self._window_length_from_days(lc_tmp, self.flatten_window_days),
                            mask=m,
                        )
                    else:
                        notes.append("flatten_mask_ignored_length_mismatch")
                        lc_tmp = lc_tmp.flatten(
                            window_length=self._window_length_from_days(lc_tmp, self.flatten_window_days)
                        )
                else:
                    lc_tmp = lc_tmp.flatten(
                        window_length=self._window_length_from_days(lc_tmp, self.flatten_window_days)
                    )
                t, f = self._to_arrays(lc_tmp)
                notes.append("flattened=True")
                lc2 = lc_tmp
            except Exception as e:
                notes.append(f"flatten_failed={type(e).__name__}")

        return {
            "time": t,
            "flux": f,
            "flux_raw": f_raw,
            "mask_good": mask_good,
            "lc_clean": self._from_arrays(lc2, t, f),
            "notes": ";".join(notes),
        }

    def flatten_transit_safe(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        window_days: float,
        sigma: float = 4.0,
        iters: int = 2,
        expand_cadences: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten while preserving narrow negative events (transit-like dips).

        Workflow:
        1) Initial flatten
        2) Flag dips where residual < -sigma * robust_sigma
        3) Expand each dip mask by +/- expand_cadences cadences
        4) Re-flatten with mask (iterative)

        Returns:
            (flattened_flux, final_mask)
            flattened_flux is aligned to input order; invalid points are NaN.
            final_mask is True for cadences masked as candidate dips.
        """
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        if len(t) != len(f):
            raise ValueError("time and flux must have the same length")

        n_total = len(t)
        if n_total == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=bool)

        finite = np.isfinite(t) & np.isfinite(f)
        flat_out = np.full(n_total, np.nan, dtype=float)
        mask_out = np.zeros(n_total, dtype=bool)
        if not np.any(finite):
            return flat_out, mask_out

        idx = np.arange(n_total, dtype=int)[finite]
        t_work = t[finite]
        f_work = f[finite]

        order = np.argsort(t_work, kind="mergesort")
        idx = idx[order]
        t_work = t_work[order]
        f_work = f_work[order]

        if len(f_work) < 5:
            flat_out[idx] = f_work
            return flat_out, mask_out

        def _safe_window_length(window_len: int, n_points: int) -> int:
            max_odd = n_points if (n_points % 2 == 1) else (n_points - 1)
            wl = int(max(5, min(window_len, max_odd)))
            if wl % 2 == 0:
                wl -= 1
            return max(5, wl)

        def _flatten_once(t_arr: np.ndarray, f_arr: np.ndarray, mask_arr: Optional[np.ndarray]) -> np.ndarray:
            lc_tmp = lk.LightCurve(time=t_arr, flux=f_arr)
            wl = _safe_window_length(self._window_length_from_days(lc_tmp, float(window_days)), len(f_arr))

            try:
                if mask_arr is None:
                    lc_flat = lc_tmp.flatten(window_length=wl)
                else:
                    lc_flat = lc_tmp.flatten(window_length=wl, mask=mask_arr.astype(bool))
                f_flat = lc_flat.flux.value if hasattr(lc_flat.flux, "value") else np.asarray(lc_flat.flux)
                return np.asarray(f_flat, dtype=float)
            except Exception:
                # Simple fallback that removes low-frequency trend if flatten fails.
                try:
                    from scipy.signal import savgol_filter

                    trend = savgol_filter(f_arr, window_length=wl, polyorder=2, mode="interp")
                    return np.asarray(f_arr - trend + np.nanmedian(trend), dtype=float)
                except Exception:
                    return np.asarray(f_arr, dtype=float)

        iters = int(max(1, iters))
        expand = int(max(0, expand_cadences))
        mask_work = np.zeros(len(f_work), dtype=bool)
        flat_work = _flatten_once(t_work, f_work, None)

        kernel = np.ones(2 * expand + 1, dtype=int)
        for _ in range(iters):
            resid = flat_work - float(np.nanmedian(flat_work))
            mad = float(np.nanmedian(np.abs(resid)) + 1e-12)
            rsig = float(1.4826 * mad)
            dip_raw = resid < (-float(sigma) * rsig)
            if expand > 0 and np.any(dip_raw):
                dip = np.convolve(dip_raw.astype(int), kernel, mode="same") > 0
            else:
                dip = dip_raw
            mask_work = mask_work | dip
            flat_work = _flatten_once(t_work, f_work, mask_work)

        flat_out[idx] = flat_work
        mask_out[idx] = mask_work
        return flat_out, mask_out

    def segment_by_gaps(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        max_gap_days: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split arrays into contiguous segments whenever time gaps exceed max_gap_days.

        Returns:
            [(t_seg, f_seg, idx_seg), ...]
            idx_seg indexes into the original input arrays.
        """
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)

        if len(t) != len(f):
            raise ValueError("time and flux must have the same length")
        if len(t) == 0:
            return []

        idx = np.arange(len(t), dtype=int)
        good = np.isfinite(t) & np.isfinite(f)
        t, f, idx = t[good], f[good], idx[good]
        if len(t) == 0:
            return []

        order = np.argsort(t, kind="mergesort")
        t, f, idx = t[order], f[order], idx[order]

        dt = np.diff(t)
        split_points = np.where(dt > float(max_gap_days))[0] + 1
        bounds = np.concatenate(([0], split_points, [len(t)]))

        segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            if b > a:
                segments.append((t[a:b], f[a:b], idx[a:b]))
        return segments

    def metrics(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        notes: str = "",
        per_segment: bool = False,
    ) -> Union[K2NoiseMetrics, Dict[str, Union[K2NoiseMetrics, List[K2NoiseMetrics]]]]:
        """
        Compute robust noise metrics for gating/ranking.

        If per_segment=True, returns:
            {"global": K2NoiseMetrics, "segments": [K2NoiseMetrics, ...]}
        """
        segments = self.segment_by_gaps(time, flux, self.max_gap_days)

        if len(segments) == 0:
            t_global = np.asarray([], dtype=float)
            f_global = np.asarray([], dtype=float)
        else:
            t_global = np.concatenate([s[0] for s in segments])
            f_global = np.concatenate([s[1] for s in segments])

        m_global = self._metrics_single(t_global, f_global, notes=notes)
        if not per_segment:
            return m_global

        m_segments = [self._metrics_single(ts, fs, notes=notes) for ts, fs, _ in segments]
        return {"global": m_global, "segments": m_segments}

    def _metrics_single(self, time: np.ndarray, flux: np.ndarray, notes: str = "") -> K2NoiseMetrics:
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)

        n = int(len(f))
        if n < 5:
            return K2NoiseMetrics(
                n_points=n,
                baseline_days=0.0,
                duty_cycle=0.0,
                mad=np.nan,
                robust_sigma=np.nan,
                outlier_rate_6sigma=np.nan,
                step_score=np.nan,
                whiteness_score=np.nan,
                outlier_rate_global=np.nan,
                notes=notes + ";too_few_points",
            )

        baseline = float(np.nanmax(t) - np.nanmin(t))

        # duty cycle relative to gaps
        gaps = np.diff(np.sort(t))
        if len(gaps) > 0:
            duty = float(np.mean(gaps <= self.max_gap_days))
        else:
            duty = 1.0

        med = float(np.nanmedian(f))
        mad = float(np.nanmedian(np.abs(f - med)) + 1e-12)
        rsig = float(1.4826 * mad)
        resid = f - med
        outlier_rate_global = float(np.mean(np.abs(resid) > 6.0 * rsig))
        outlier_rate_local = self._local_outlier_rate(
            time=t,
            residual=resid,
            k_sigma=6.0,
            window_days=1.0,
            fallback_sigma=rsig,
        )

        # step_score: median absolute first-difference normalized by sigma
        df = np.diff(f)
        step = float(np.nanmedian(np.abs(df)) / (rsig + 1e-12))

        # whiteness_score:
        # - statistic mode: |rho_1| where rho_1 is lag-1 autocorrelation (lower is whiter)
        # - pvalue mode: p = 2*(1-Phi(|rho_1|*sqrt(n-1))) = erfc(|rho_1|*sqrt(n-1)/sqrt(2))
        #   This is a two-sided normal-approximation p-value for H0: rho_1 = 0.
        #   Higher p indicates residuals are more consistent with whiteness at lag 1.
        fr = f - med
        if np.all(~np.isfinite(fr)) or np.nanstd(fr) == 0:
            w = np.nan
        else:
            fr0 = fr[:-1]
            fr1 = fr[1:]
            fr0 = fr0 - np.nanmean(fr0)
            fr1 = fr1 - np.nanmean(fr1)
            denom = (np.nanstd(fr0) * np.nanstd(fr1)) + 1e-12
            rho = float(np.nanmean(fr0 * fr1) / denom)
            if self.noise_config.whiteness_score_definition == "pvalue":
                z = abs(rho) * np.sqrt(max(float(n - 1), 1.0))
                w = float(math.erfc(float(z) / np.sqrt(2.0)))
            else:
                w = float(np.abs(rho))
        print(
            f"[whiteness calc] func=_metrics_single test={self.whiteness_definition()} "
            f"value_type={self.noise_config.whiteness_score_definition} lags=[1] value={w}"
        )

        return K2NoiseMetrics(
            n_points=n,
            baseline_days=baseline,
            duty_cycle=duty,
            mad=mad,
            robust_sigma=rsig,
            outlier_rate_6sigma=outlier_rate_local,
            step_score=step,
            whiteness_score=w,
            outlier_rate_global=outlier_rate_global,
            notes=notes,
        )

    def score(self, m: K2NoiseMetrics) -> float:
        """
        Noise quality score where higher is better.

        Score is the minimum normalized margin to configured thresholds.
        This makes `score > 0` equivalent to passing all configured gates.
        """
        cfg = self.noise_config

        required_vals = [m.robust_sigma, m.baseline_days, m.outlier_rate_6sigma, m.step_score]
        if not all(np.isfinite(v) for v in required_vals):
            return -np.inf

        margins = [
            (float(m.n_points) - float(cfg.min_points)) / max(float(cfg.min_points), 1.0),
            (float(m.baseline_days) - float(cfg.min_baseline_days)) / max(float(cfg.min_baseline_days), 1e-12),
            (float(m.robust_sigma) - float(cfg.min_robust_sigma)) / max(abs(float(cfg.min_robust_sigma)), 1.0),
            (float(cfg.max_outlier_rate_6sigma) - float(m.outlier_rate_6sigma))
            / max(float(cfg.max_outlier_rate_6sigma), 1e-12),
            (float(cfg.max_step_score) - float(m.step_score)) / max(float(cfg.max_step_score), 1e-12),
        ]

        if np.isfinite(m.whiteness_score):
            wt = float(self.whiteness_threshold())
            if cfg.whiteness_score_definition == "pvalue":
                margins.append((float(m.whiteness_score) - wt) / max(wt, 1e-12))
            else:
                margins.append((wt - float(m.whiteness_score)) / max(wt, 1e-12))

        score = float(np.nanmin(np.asarray(margins, dtype=float)))
        return score if np.isfinite(score) else -np.inf

    def explain(self, m: K2NoiseMetrics) -> Dict[str, Any]:
        """
        Explain threshold failures for a metric object using active config.
        """
        cfg = self.noise_config
        thresholds: Dict[str, Any] = {
            "mode": str(cfg.mode),
            "min_points": float(cfg.min_points),
            "min_baseline_days": float(cfg.min_baseline_days),
            "min_robust_sigma": float(cfg.min_robust_sigma),
            "max_outlier_rate_6sigma": float(cfg.max_outlier_rate_6sigma),
            "catastrophic_outlier_rate_6sigma": float(cfg.catastrophic_outlier_rate_6sigma),
            "max_step_score": float(cfg.max_step_score),
            "max_whiteness_score": float(cfg.max_whiteness_score) if cfg.max_whiteness_score is not None else np.nan,
            "whiteness_score_definition": str(cfg.whiteness_score_definition),
            "whiteness_alpha": float(cfg.whiteness_alpha) if cfg.whiteness_alpha is not None else np.nan,
            "whiteness_threshold": float(self.whiteness_threshold()),
        }
        values: Dict[str, float] = {
            "n_points": float(m.n_points),
            "baseline_days": float(m.baseline_days),
            "robust_sigma": float(m.robust_sigma),
            "outlier_rate_6sigma": float(m.outlier_rate_6sigma),
            "outlier_rate_global": float(m.outlier_rate_global),
            "step_score": float(m.step_score),
            "whiteness_score": float(m.whiteness_score),
        }

        fail_reasons: List[str] = []
        if m.n_points < cfg.min_points:
            fail_reasons.append(f"n_points<{cfg.min_points}")
        if m.baseline_days < cfg.min_baseline_days:
            fail_reasons.append(f"baseline_days<{cfg.min_baseline_days}")
        if (not np.isfinite(m.robust_sigma)) or (m.robust_sigma < cfg.min_robust_sigma):
            fail_reasons.append(f"robust_sigma<{cfg.min_robust_sigma}")
        if (not np.isfinite(m.outlier_rate_6sigma)) or (m.outlier_rate_6sigma > cfg.max_outlier_rate_6sigma):
            fail_reasons.append(f"outlier_rate_6sigma>{cfg.max_outlier_rate_6sigma}")
        if (not np.isfinite(m.step_score)) or (m.step_score > cfg.max_step_score):
            fail_reasons.append(f"step_score>{cfg.max_step_score}")
        if np.isfinite(m.whiteness_score):
            wt = float(self.whiteness_threshold())
            if cfg.whiteness_score_definition == "pvalue":
                if m.whiteness_score < wt:
                    fail_reasons.append(f"whiteness_pvalue={m.whiteness_score:.6g}<{wt}")
            elif m.whiteness_score > wt:
                fail_reasons.append(f"whiteness_score={m.whiteness_score:.6g}>{wt}")

        return {"fail_reasons": fail_reasons, "thresholds": thresholds, "values": values}

    def whiteness_definition(self) -> str:
        if self.noise_config.whiteness_score_definition == "pvalue":
            return "lag1_autocorr_pvalue_normal_approx"
        return "lag1_abs_autocorr_statistic"

    def whiteness_threshold(self) -> float:
        if self.noise_config.whiteness_score_definition == "pvalue":
            if self.noise_config.whiteness_alpha is None:
                return 0.01
            return float(self.noise_config.whiteness_alpha)
        if self.noise_config.max_whiteness_score is None:
            return float("inf")
        return float(self.noise_config.max_whiteness_score)

    def score_segments(self, metrics: List[K2NoiseMetrics], policy: str) -> float:
        """
        Aggregate per-segment scores using policy: best | median | worst.
        """
        if len(metrics) == 0:
            return -np.inf

        seg_scores = np.asarray([self.score(m) for m in metrics], dtype=float)
        p = str(policy).lower().strip()
        if p == "best":
            return float(np.nanmax(seg_scores))
        if p == "median":
            return float(np.nanmedian(seg_scores))
        if p == "worst":
            return float(np.nanmin(seg_scores))
        raise ValueError(f"Unknown segment score policy: {policy!r}. Expected one of ('best', 'median', 'worst').")

    def is_usable(self, m: K2NoiseMetrics) -> bool:
        """
        Hard gate derived from score: usable iff score > 0.
        """
        return self.score(m) > 0.0

    def batch_report(
        self,
        queries: Iterable[str],
        limit: int = 50,
        exptime: Optional[Union[str, float]] = None,
        flatten: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run fetch → clean → metrics → usable for a list of targets.
        Returns a list of dict rows suitable for CSV writing.
        """
        rows: List[Dict[str, Any]] = []

        for q in queries:
            try:
                fetched = self.fetch_best(q, limit=limit, exptime=exptime)
                lc = fetched["lc"]
                author = fetched["author"]
                cleaned = self.clean(lc, flatten=flatten)
                m = self.metrics(cleaned["time"], cleaned["flux"], notes=cleaned["notes"])
                rows.append(
                    {
                        "query": q,
                        "author": author,
                        **m.__dict__,
                        "usable": self.is_usable(m),
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "query": q,
                        "author": "",
                        "n_points": 0,
                        "baseline_days": 0.0,
                        "duty_cycle": 0.0,
                        "mad": np.nan,
                        "robust_sigma": np.nan,
                        "outlier_rate_6sigma": np.nan,
                        "outlier_rate_global": np.nan,
                        "step_score": np.nan,
                        "whiteness_score": np.nan,
                        "notes": f"failed={type(e).__name__}",
                        "usable": False,
                    }
                )

        return rows

    # ----------------------------
    # Internals
    # ----------------------------

    def choose_best(self, sr) -> Dict[str, Any]:
        """
        Pick the best product from a Lightkurve SearchResult and log why it won.
        """
        if len(sr) == 0:
            raise ValueError("choose_best() received an empty SearchResult")

        for preferred_author in self.author_priority:
            for i in range(len(sr)):
                row = sr[i]
                author = self._author_as_str(self._extract_search_field(row, "author", default=""))
                if author == str(preferred_author):
                    reason = (
                        f"matched author priority '{preferred_author}' at index {i}; "
                        "kept first match by search ordering"
                    )
                    self._log_choice(reason)
                    return {"index": i, "product": row, "author": author, "reason": reason}

        fallback = sr[0]
        fallback_author = self._author_as_str(self._extract_search_field(fallback, "author", default=""))
        reason = "no preferred author found in SearchResult; fell back to index 0"
        self._log_choice(reason)
        return {"index": 0, "product": fallback, "author": fallback_author, "reason": reason}

    def _log_choice(self, message: str) -> None:
        if self.verbose:
            print(f"[K2_NoiseHandler.choose_best] {message}")

    def _extract_search_field(self, row: Any, key: str, default: Any = None) -> Any:
        try:
            if hasattr(row, "colnames") and key in row.colnames:
                return self._to_serializable(row[key])
            if hasattr(row, "keys") and key in row.keys():
                return self._to_serializable(row[key])
            if hasattr(row, key):
                return self._to_serializable(getattr(row, key))
        except Exception:
            pass
        return default

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [self._to_serializable(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if hasattr(value, "value"):
            try:
                return self._to_serializable(value.value)
            except Exception:
                pass

        return str(value)

    def _author_as_str(self, value: Any) -> str:
        v = self._to_serializable(value)
        if isinstance(v, list):
            if len(v) == 0:
                return ""
            return str(v[0])
        return str(v)

    def _extract_download_path(self, downloaded_obj: Any) -> Optional[str]:
        if downloaded_obj is None:
            return None
        for attr in ("path", "local_path"):
            try:
                p = getattr(downloaded_obj, attr, None)
            except Exception:
                p = None
            if p is not None and str(p).strip() != "":
                return str(p)
        return None

    def _cache_source_from_path(self, path: Optional[str]) -> str:
        if path is None:
            return "custom_or_unknown"
        p = str(path).lower().replace("/", "\\")
        if "\\mastdownload\\" in p:
            return "astroquery_mastDownload"
        if "\\.lightkurve\\cache\\" in p:
            return "lightkurve_cache"
        return "custom_or_unknown"

    def _infer_epic_id(self, query: str, selected_row: Any) -> str:
        candidates = [
            query,
            self._extract_search_field(selected_row, "target_name", default=""),
            self._extract_search_field(selected_row, "obs_id", default=""),
        ]
        for c in candidates:
            txt = str(self._to_serializable(c))
            m = re.search(r"(\d{6,})", txt)
            if m is not None:
                return str(m.group(1))
        return str(query)

    def _cache_roots(self, download_dir: Optional[str]) -> List[Path]:
        roots: List[Path] = []
        if download_dir is not None and str(download_dir).strip() != "":
            roots.append(Path(str(download_dir)).expanduser())
        env_root = os.environ.get("LIGHTKURVE_CACHE_DIR", "")
        if env_root.strip() != "":
            roots.append(Path(env_root).expanduser())
        roots.append(Path.home() / ".lightkurve" / "cache")
        roots.append(Path.home() / ".lightkurve")

        out: List[Path] = []
        seen = set()
        for r in roots:
            key = str(r).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _find_cached_product_path(self, selected_row: Any, download_dir: Optional[str]) -> Optional[Path]:
        product_filename = str(self._extract_search_field(selected_row, "productFilename", default="")).strip()
        obs_id = str(self._extract_search_field(selected_row, "obs_id", default="")).strip()
        obs_collection = str(self._extract_search_field(selected_row, "obs_collection", default="")).strip()
        mission = str(self._extract_search_field(selected_row, "mission", default="")).strip()
        mission_fallback = mission.split(" ")[0] if mission else "K2"
        collection = obs_collection if obs_collection else mission_fallback

        if product_filename == "":
            return None

        roots = self._cache_roots(download_dir=download_dir)
        candidates: List[Path] = []
        for root in roots:
            candidates.append(root / product_filename)
            if obs_id != "":
                candidates.append(root / collection / obs_id / product_filename)
                candidates.append(root / "mastDownload" / collection / obs_id / product_filename)
            if root.name.lower() == "mastdownload" and obs_id != "":
                candidates.append(root / collection / obs_id / product_filename)

        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return c
            except Exception:
                pass

        # Narrow glob fallback for common mast cache layouts.
        for root in roots:
            try:
                patterns = [
                    f"mastDownload/*/*/{product_filename}",
                    f"*/*/{product_filename}",
                    product_filename,
                ]
                for pat in patterns:
                    for match in root.glob(pat):
                        if match.exists() and match.is_file():
                            return match
            except Exception:
                continue
        return None

    def _find_direct_local_product(
        self,
        query: str,
        download_dir: Optional[str],
        exptime: Optional[Union[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        epic = self._infer_epic_id(query=query, selected_row=None).strip()
        if epic == "":
            return None

        roots = self._cache_roots(download_dir=download_dir)
        candidates = self._direct_local_candidate_paths(epic=epic, download_dir=download_dir, exptime=exptime)
        if len(candidates) == 0:
            return None

        ranked = sorted(candidates, key=lambda path: self._direct_local_candidate_sort_key(path, roots))
        best_path = ranked[0]
        author = self._direct_local_author(best_path)
        campaign = self._direct_local_campaign(best_path)
        return {
            "epic": epic,
            "author": author,
            "campaign": campaign,
            "cache_path": best_path,
            "search_result": {
                "query": str(query),
                "selected_index": -1,
                "selection_reason": "direct_local_cache_lookup",
                "n_results": int(len(ranked)),
                "author_priority": [str(a) for a in self.author_priority],
                "results": [
                    {
                        "author": self._direct_local_author(path),
                        "campaign": self._direct_local_campaign(path),
                        "path": str(path),
                    }
                    for path in ranked[:20]
                ],
            },
        }

    def _direct_local_candidate_paths(
        self,
        epic: str,
        download_dir: Optional[str],
        exptime: Optional[Union[str, float]] = None,
    ) -> List[Path]:
        del exptime
        roots = self._cache_roots(download_dir=download_dir)
        patterns = [
            f"mastDownload/HLSP/*{epic}*/*.fits*",
            f"mastDownload/*/*{epic}*/*.fits*",
            f"*{epic}*/*.fits*",
            f"**/*{epic}*.fits*",
        ]
        matches: List[Path] = []
        seen = set()
        for root in roots:
            for pat in patterns:
                try:
                    for match in root.glob(pat):
                        if (not match.exists()) or (not match.is_file()):
                            continue
                        text = str(match).lower()
                        if ("k2" not in text) and ("ktwo" not in text):
                            continue
                        key = str(match).lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        matches.append(match)
                except Exception:
                    continue
        return matches

    def _direct_local_candidate_sort_key(self, path: Path, roots: List[Path]) -> Tuple[int, int, int, str]:
        author = self._direct_local_author(path)
        try:
            author_rank = list(self.author_priority).index(author)
        except ValueError:
            author_rank = len(self.author_priority)
        campaign_rank = self._direct_local_campaign_sort_key(path)
        root_rank = self._direct_local_root_rank(path=path, roots=roots)
        return (author_rank, root_rank, campaign_rank, str(path).lower())

    @staticmethod
    def _direct_local_root_rank(path: Path, roots: List[Path]) -> int:
        path_text = str(path).lower()
        for idx, root in enumerate(roots):
            root_text = str(root).lower()
            if path_text.startswith(root_text):
                return int(idx)
        return int(len(roots) + 1)

    @staticmethod
    def _direct_local_campaign(path: Path) -> str:
        text = str(path).lower()
        m = re.search(r"-c(\d{2,3})(?:[^0-9]|$)", text)
        if m is not None:
            return f"c{m.group(1)}"
        m = re.search(r"(?:^|[^a-z0-9])c(\d{2,3})(?:[^0-9]|$)", text)
        if m is not None:
            return f"c{m.group(1)}"
        return ""

    def _direct_local_campaign_sort_key(self, path: Path) -> int:
        campaign = self._direct_local_campaign(path)
        m = re.search(r"c(\d{2,3})", campaign.lower())
        if m is None:
            return 10_000
        try:
            return int(m.group(1))
        except Exception:
            return 10_000

    def _direct_local_author(self, path: Path) -> str:
        text = str(path).upper()
        for author in self.author_priority:
            if str(author).upper() in text:
                return str(author)
        return "K2"

    def _serialize_search_result(
        self,
        sr: Any,
        selected_index: int,
        selection_reason: str,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        keys = [
            "author",
            "target_name",
            "mission",
            "exptime",
            "distance",
            "year",
            "campaign",
            "sequence_number",
            "obs_id",
            "proposal_id",
            "productFilename",
        ]
        results: List[Dict[str, Any]] = []
        for i in range(len(sr)):
            row = sr[i]
            item: Dict[str, Any] = {"index": i}
            for key in keys:
                item[key] = self._extract_search_field(row, key, default=None)
            item["author"] = self._author_as_str(item.get("author", ""))
            results.append(item)

        selected = results[selected_index] if 0 <= selected_index < len(results) else {}
        return {
            "query": query,
            "n_results": int(len(sr)),
            "author_priority": [str(a) for a in self.author_priority],
            "selected_index": int(selected_index),
            "selection_reason": selection_reason,
            "selected": selected,
            "results": results,
        }

    def _local_outlier_rate(
        self,
        time: np.ndarray,
        residual: np.ndarray,
        k_sigma: float = 6.0,
        window_days: float = 1.0,
        fallback_sigma: float = 1.0,
    ) -> float:
        t = np.asarray(time, dtype=float)
        r = np.asarray(residual, dtype=float)
        if len(t) == 0:
            return np.nan

        n = len(r)
        window_len = self._window_len_from_time_days(t, float(window_days), n)
        local_med, local_sig = self._rolling_robust_stats(r, window_len)

        local_sig = np.where(np.isfinite(local_sig) & (local_sig > 0), local_sig, max(float(fallback_sigma), 1e-12))
        local_med = np.where(np.isfinite(local_med), local_med, 0.0)

        flags = np.abs(r - local_med) > (float(k_sigma) * local_sig)
        valid = np.isfinite(r) & np.isfinite(local_med) & np.isfinite(local_sig)
        if not np.any(valid):
            return np.nan
        return float(np.mean(flags[valid]))

    def _window_len_from_time_days(self, time: np.ndarray, window_days: float, n_points: int) -> int:
        t = np.asarray(time, dtype=float)
        if len(t) < 3:
            return int(max(5, min(11, n_points if n_points % 2 == 1 else max(1, n_points - 1))))

        dt = np.diff(np.sort(t))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            raw = 101
        else:
            med_dt = float(np.nanmedian(dt))
            raw = int(np.round(float(window_days) / med_dt))

        w = max(11, raw)
        max_odd = n_points if n_points % 2 == 1 else (n_points - 1)
        w = min(w, max(max_odd, 5))
        if w % 2 == 0:
            w -= 1
        return int(max(w, 5))

    def _rolling_robust_stats(self, values: np.ndarray, window_len: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(values, dtype=float)
        n = len(x)
        med = np.zeros(n, dtype=float)
        sig = np.zeros(n, dtype=float)
        half = int(window_len // 2)

        global_med = float(np.nanmedian(x)) if n > 0 else 0.0
        global_mad = float(np.nanmedian(np.abs(x - global_med)) + 1e-12) if n > 0 else 1e-12
        global_sig = float(1.4826 * global_mad)

        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            seg = x[a:b]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 5:
                med[i] = global_med
                sig[i] = max(global_sig, 1e-12)
                continue
            m = float(np.nanmedian(seg))
            mad = float(np.nanmedian(np.abs(seg - m)) + 1e-12)
            med[i] = m
            sig[i] = max(1.4826 * mad, 1e-12)

        return med, sig

    def _to_arrays(self, lc) -> Tuple[np.ndarray, np.ndarray]:
        t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
        f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
        return np.asarray(t, dtype=float), np.asarray(f, dtype=float)

    def _from_arrays(self, lc_template, t: np.ndarray, f: np.ndarray):
        # Keep Lightkurve object type when possible — helps downstream tooling
        try:
            return lk.LightCurve(time=t, flux=f)
        except Exception:
            # fallback — return original object sliced if possible
            return lc_template

    def _window_length_from_days(self, lc, days: float) -> int:
        # Convert a days window into an odd integer number of cadences
        t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
        dt = np.nanmedian(np.diff(np.sort(t)))
        if not np.isfinite(dt) or dt <= 0:
            return 401  # safe-ish default
        n = int(np.round(days / dt))
        n = max(31, n)
        if n % 2 == 0:
            n += 1
        return n
