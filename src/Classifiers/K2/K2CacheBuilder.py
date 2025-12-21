# k2_cached_scorer.py

from __future__ import annotations
import os
import glob
from astropy.io.fits.verify import VerifyError
import re
import os
import time
import math
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import lightkurve as lk
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.Classifiers.CommonHelper import CommonHelper


log = logging.getLogger(__name__)
@dataclass
class ScoreRow:
    epic_id: str
    mission: str
    n_points: int
    n_segments: int
    best_seg_score: float
    mean_seg_score: float
    median_seg_score: float
    std_seg_score: float
    status: str
    error: str = ""
    best_seg_idx: int = -1
    best_start: int = -1
    best_end: int = -1
    best_t0: float = float("nan")
    best_t1: float = float("nan")
    p95_seg_score: float = float("nan")


class K2CachedScorer:
    """
    Cache-first K2 inference runner for a 1D CNN model trained on 2-channel inputs:
      channel0 = lc.flux after remove_nans().normalize()
      channel1 = flattened flux using lc.flatten(window_length=201, polyorder=2)
    Then optionally apply Median/MAD normalization per channel (same as your normalize_flux).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        cache_dir: Union[str, Path] = "k2_cache",
        author: Optional[str] = None,
        any_author: bool = True,
        use_all: bool = False,
        max_files: int = 12,
        flatten_window_length: int = 201,
        flatten_polyorder: int = 2,
        seg_len: int = 1024,
        stride: Optional[int] = None,
        use_mad_norm: bool = True,
        clip_after_mad: Optional[float] = None,
        verbose: bool = True,
    ):
        self.model_path = str(model_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.commonHelper = CommonHelper()
        self.author = author
        self.any_author = any_author
        self.use_all = use_all
        self.max_files = int(max_files)

        self.flatten_window_length = int(flatten_window_length)
        self.flatten_polyorder = int(flatten_polyorder)

        self.seg_len = int(seg_len)
        self.stride = int(stride) if stride is not None else int(seg_len)
        self.use_mad_norm = bool(use_mad_norm)
        self.clip_after_mad = float(clip_after_mad) if clip_after_mad is not None else None

        self.verbose = False
        self.model = tf.keras.models.load_model(self.model_path)

        print(f"[K2CachedScorer] Loaded model: {self.model_path}")
        print(f"[K2CachedScorer] Cache dir: {self.cache_dir.resolve()}")
        print(f"[K2CachedScorer] seg_len={self.seg_len}, stride={self.stride}, use_all={self.use_all}")

    # ---------- Public API ----------

    def download_targets(
        self,
        epic_ids: Iterable[Union[int, str]],
        workers: int = 6,
        sleep_s: float = 0.0,
    ) -> pd.DataFrame:
        """
        Downloads light curves into download_dir cache. This is I/O-bound; use a few threads.
        Returns a dataframe with per-target download status.
        """
        epic_ids = [str(e).strip() for e in epic_ids]
        rows = []

        def _dl(epic: str) -> Dict[str, Any]:
            try:
                t0 = time.time()
                sr = self._search(epic)
                if len(sr) == 0:
                    return {"epic_id": epic, "status": "no_search_results", "files": 0, "seconds": time.time() - t0}

                if self.use_all:
                    _ = sr[: self.max_files].download_all(download_dir=str(self.cache_dir))
                else:
                    _ = sr[0].download(download_dir=str(self.cache_dir))

                if sleep_s > 0:
                    time.sleep(sleep_s)

                return {"epic_id": epic, "status": "ok", "files": int(min(len(sr), self.max_files if self.use_all else 1)),
                        "seconds": time.time() - t0}
            except Exception as e:
                return {"epic_id": epic, "status": "error", "files": 0, "seconds": 0.0, "error": repr(e)}

        if self.verbose:
            print(f"[download_targets] Downloading {len(epic_ids)} targets with {workers} workers...")

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = {ex.submit(_dl, epic): epic for epic in epic_ids}
            for fut in as_completed(futs):
                rows.append(fut.result())

        df = pd.DataFrame(rows).sort_values(["status", "epic_id"])
        return df

    def score_targets(
        self,
        epic_ids: Iterable[Union[int, str]],
        out_csv: Union[str, Path] = "k2_inference_scores.csv",
        save_every: int = 50,
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Scores targets (using cached downloads) and saves incremental CSV.
        If resume=True and out_csv exists, it will skip EPIC IDs already scored.
        """
        epic_ids = [str(e).strip() for e in epic_ids]
        out_csv = Path(out_csv)

        done = set()
        if resume and out_csv.exists():
            try:
                prev = pd.read_csv(out_csv)
                if "epic_id" in prev.columns:
                    done = set(prev["epic_id"].astype(str).tolist())
                    if self.verbose:
                        print(f"[score_targets] Resuming: found {len(done)} already in {out_csv}")
            except Exception:
                pass

        rows: List[Dict[str, Any]] = []
        scored_count = 0
        total = len(epic_ids)

        for i, epic in enumerate(epic_ids, start=1):
            if epic in done:
                continue

            r = self.score_one(epic)
            rows.append(r.__dict__)
            scored_count += 1

            if self.verbose and (i % 10 == 0 or i == total):
               log.info("[score_targets] %s/%s processed. Newly scored this run: %s", i, total, scored_count)


            if scored_count > 0 and (scored_count % int(save_every) == 0):
                self._append_rows(out_csv, rows)
                rows = []

        if rows:
            self._append_rows(out_csv, rows)

        df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
        return df

    def score_one(self, epic_id: Union[int, str]) -> ScoreRow:
        """
        Score a single EPIC target and record which segment was best (for plotting/vetting packs).
        """
        epic = str(epic_id).strip()
        try:
            time_arr, flux = self._get_time_and_flux(epic)
            if time_arr is None or flux is None or flux.shape[0] < self.seg_len:
                return ScoreRow(
                    epic_id=epic,
                    mission="K2",
                    n_points=0 if flux is None else int(flux.shape[0]),
                    n_segments=0,
                    best_seg_score=float("nan"),
                    mean_seg_score=float("nan"),
                    median_seg_score=float("nan"),
                    std_seg_score=float("nan"),
                    p95_seg_score=float("nan"),
                    best_seg_idx=-1,
                    best_start=-1,
                    best_end=-1,
                    best_t0=float("nan"),
                    best_t1=float("nan"),
                    status="too_short_or_missing",
                    error="",
                )

            X = self._segment(flux, seg_len=self.seg_len, stride=self.stride)  # (N, L, 2)
            if X.shape[0] == 0:
                return ScoreRow(
                    epic_id=epic,
                    mission="K2",
                    n_points=int(flux.shape[0]),
                    n_segments=0,
                    best_seg_score=float("nan"),
                    mean_seg_score=float("nan"),
                    median_seg_score=float("nan"),
                    std_seg_score=float("nan"),
                    p95_seg_score=float("nan"),
                    best_seg_idx=-1,
                    best_start=-1,
                    best_end=-1,
                    best_t0=float("nan"),
                    best_t1=float("nan"),
                    status="no_segments",
                    error="",
                )

            seg_scores = self._predict_segments(X)
            seg_scores = np.asarray(seg_scores).reshape(-1)

            best_idx = int(np.argmax(seg_scores))
            best_start = int(best_idx * self.stride)
            best_end = int(best_start + self.seg_len)

            # clamp end to available samples
            n = int(len(time_arr))
            if n > 0:
                s = min(max(best_start, 0), n - 1)
                e = min(max(best_end - 1, 0), n - 1)
                best_t0 = float(time_arr[s])
                best_t1 = float(time_arr[e])
            else:
                best_t0 = float("nan")
                best_t1 = float("nan")

            return ScoreRow(
                epic_id=epic,
                mission="K2",
                n_points=int(flux.shape[0]),
                n_segments=int(len(seg_scores)),
                best_seg_score=float(np.max(seg_scores)),
                mean_seg_score=float(np.mean(seg_scores)),
                median_seg_score=float(np.median(seg_scores)),
                std_seg_score=float(np.std(seg_scores)),
                p95_seg_score=float(np.quantile(seg_scores, 0.95)),
                best_seg_idx=best_idx,
                best_start=best_start,
                best_end=best_end,
                best_t0=best_t0,
                best_t1=best_t1,
                status="ok",
                error="",
            )

        except Exception as e:
            return ScoreRow(
                epic_id=epic,
                mission="K2",
                n_points=0,
                n_segments=0,
                best_seg_score=float("nan"),
                mean_seg_score=float("nan"),
                median_seg_score=float("nan"),
                std_seg_score=float("nan"),
                p95_seg_score=float("nan"),
                best_seg_idx=-1,
                best_start=-1,
                best_end=-1,
                best_t0=float("nan"),
                best_t1=float("nan"),
                status="error",
                error=repr(e),
            )

    # ---------- Internals ----------

    def _search(self, epic: str):
        target = f"{epic}"
        return lk.search_lightcurve(target, mission="K2", author=None if self.any_author else self.author)

    def _get_time_and_flux(self, epic: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        glob it
        """
        epic_num = self._clean_epic(epic)
        sr = self._search(epic_num)
        if len(sr) == 0:
            return None, None

        if self.use_all:
            # (optional: add retry logic for download_all too, but start simple)
            lcc = sr[: self.max_files].download_all(download_dir=str(self.cache_dir))
            lc = lcc.stitch().remove_nans().normalize()
        else:
            lc = self._download_lc_with_retry(sr, epic_num)
        try:
            lc_flat = lc.flatten(window_length=self.flatten_window_length, polyorder=self.flatten_polyorder)
        except Exception:
            lc_flat = lc.copy()

        time_arr = np.asarray(lc.time.value, dtype=np.float64)
        fl_raw = np.asarray(lc.flux.value, dtype=np.float32)
        fl_flat = np.asarray(lc_flat.flux.value, dtype=np.float32)

        # align lengths just in case (defensive)
        n = min(len(time_arr), len(fl_raw), len(fl_flat))
        time_arr = time_arr[:n]
        fl_raw = fl_raw[:n]
        fl_flat = fl_flat[:n]

        flux = np.stack([fl_raw, fl_flat], axis=-1)  # (T, 2)

        # Optional: your median/MAD normalization on both channels
        if self.use_mad_norm:
            flux = self.commonHelper.normalize_flux(flux)
            if self.clip_after_mad is not None:
                flux = np.clip(flux, -self.clip_after_mad, self.clip_after_mad)

        return time_arr, flux


    @staticmethod
    def _segment(flux: np.ndarray, seg_len: int, stride: int) -> np.ndarray:
        """
        Segment (T, C) -> (N, seg_len, C), dropping incomplete last segment.
        """
        T, C = flux.shape
        if T < seg_len:
            return np.zeros((0, seg_len, C), dtype=np.float32)

        starts = list(range(0, T - seg_len + 1, stride))
        X = np.empty((len(starts), seg_len, C), dtype=np.float32)
        for i, s in enumerate(starts):
            X[i] = flux[s : s + seg_len].astype(np.float32)
        return X

    def _predict_segments(self, X: np.ndarray) -> np.ndarray:
        """
        Handles model outputs of shape (N, 1) or (N,) etc.
        """
        y = self.model.predict(X, verbose=0)
        y = np.asarray(y).reshape(-1)
        return y

    @staticmethod
    def _append_rows(out_csv: Path, rows: List[Dict[str, Any]]) -> None:
        df_new = pd.DataFrame(rows)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(out_csv, index=False)


    def _purge_cached_epic_files(self, epic_num: str):
        # Lightkurve cache filenames vary, so we glob EPIC id.
        patterns = [
            str(self.cache_dir / f"*{epic_num}*.fits"),
            str(self.cache_dir / f"*{epic_num}*.fits.gz"),
        ]
        for p in patterns:
            for fp in glob.glob(p):
                try:
                    os.remove(fp)
                except OSError:
                    pass

    def _download_lc_with_retry(self, sr, epic_num: str):
        # 1st attempt
        try:
            return sr[0].download(download_dir=str(self.cache_dir)).remove_nans().normalize()
        except Exception:
            # purge and retry once
            self._purge_cached_epic_files(epic_num)
            return sr[0].download(download_dir=str(self.cache_dir)).remove_nans().normalize()


    def _clean_epic(self, epic: str) -> str:
        """
        Normalize an EPIC identifier to digits only.
        Examples:
        'EPIC 247418783' -> '247418783'
        '247418783'      -> '247418783'
        'EPIC247418783'  -> '247418783'
        """
        s = str(epic).strip().upper()
        s = s.replace("EPIC", "").strip()
        s = re.sub(r"[^0-9]", "", s)
        return s
