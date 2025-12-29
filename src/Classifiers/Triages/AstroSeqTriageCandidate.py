from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class AstroSeqCandidateTriage:
    """
    AstroSeqCandidateTriage

    Input: segment-level dataframe with at least:
      - star_id
      - score
    Recommended:
      - seg_mid_time (for periodic/secondary flags)
      - mission, campaign/sector/quarter, split (passthrough metadata)
    """

    def __init__(
        self,
        score_threshold: float = 0.80,
        top_n: int = 10,
        min_period_days: float = 0.3,
        max_period_days: float = 50.0,
        epoch_tolerance: float = 0.35,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.top_n = int(top_n)
        self.min_period_days = float(min_period_days)
        self.max_period_days = float(max_period_days)
        self.epoch_tolerance = float(epoch_tolerance)

    # -----------------------------
    # Save
    # -----------------------------
    def write_triage(self, df: pd.DataFrame, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() == ".csv":
            df.to_csv(out_path, index=False)
        elif out_path.suffix.lower() == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            raise ValueError("Unsupported output type. Use .csv or .parquet")

    # -----------------------------
    # Threshold helper
    # -----------------------------
    def set_threshold_from_scores(self, scores: np.ndarray, percentile: float = 99.0) -> float:
        """
        Sets score_threshold based on score distribution.
        Useful when model outputs are compressed (e.g. 0.24 to 0.49).
        """
        scores = np.asarray(scores, dtype=float)
        thr = float(np.percentile(scores, percentile))
        self.score_threshold = thr
        return thr

    # -----------------------------
    # Period estimation
    # -----------------------------
    def estimate_period_from_times(self, times: np.ndarray) -> float:
        times = np.asarray(times, dtype=float)
        times = times[np.isfinite(times)]
        if times.size < 3:
            return np.nan

        times = np.sort(times)

        diffs_list: List[np.ndarray] = []
        for i in range(len(times)):
            d = times[i + 1 :] - times[i]
            if d.size:
                diffs_list.append(d)

        if not diffs_list:
            return np.nan

        diffs = np.concatenate(diffs_list)
        diffs = diffs[(diffs >= self.min_period_days) & (diffs <= self.max_period_days)]
        if diffs.size < 5:
            return np.nan

        bins = np.linspace(self.min_period_days, self.max_period_days, 201)
        hist, edges = np.histogram(diffs, bins=bins)
        if int(hist.max()) < 3:
            return np.nan

        best_idx = int(hist.argmax())
        return float((edges[best_idx] + edges[best_idx + 1]) / 2.0)

    def assign_epochs(self, times: np.ndarray, period: float) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if not np.isfinite(period) or period <= 0:
            return np.full(len(times), -1, dtype=int)
        t0 = float(np.nanmin(times))
        return np.rint((times - t0) / period).astype(int)

    # -----------------------------
    # Flags
    # -----------------------------
    def compute_flags(self, times: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
        times = np.asarray(times, dtype=float)
        scores = np.asarray(scores, dtype=float)

        m = np.isfinite(scores)
        if times.shape[0] == scores.shape[0]:
            m = m & np.isfinite(times)

        times = times[m] if times.shape[0] == scores.shape[0] else np.full(np.sum(m), np.nan)
        scores = scores[m]

        above = scores >= self.score_threshold
        singleton = int(np.sum(above) == 1)

        # If no times, periodic flags are not meaningful
        if not np.isfinite(times).any():
            return {
                "flag_singleton_spike": singleton,
                "cand_period_days": np.nan,
                "flag_periodic": 0,
                "odd_even_score_gap": np.nan,
                "flag_secondary_proxy": 0,
            }

        high_times = times[above]
        high_scores = scores[above]

        period = self.estimate_period_from_times(high_times) if high_times.size >= 3 else np.nan

        periodic = 0
        odd_even_proxy = np.nan
        secondary_proxy = 0

        if np.isfinite(period):
            epochs = self.assign_epochs(high_times, period)

            t0 = float(np.min(high_times))
            frac = np.abs(((high_times - t0) / period) - np.rint((high_times - t0) / period))
            periodic = int(np.nanmedian(frac) <= self.epoch_tolerance)

            if np.unique(epochs).size >= 4:
                odd = high_scores[epochs % 2 == 1]
                even = high_scores[epochs % 2 == 0]
                if odd.size >= 2 and even.size >= 2:
                    odd_even_proxy = float(np.abs(np.nanmean(odd) - np.nanmean(even)))

            # Secondary proxy: support near phase ~0.5
            phases = (((times - t0) / period) % 1.0)
            dist_half = np.minimum(np.abs(phases - 0.5), 1.0 - np.abs(phases - 0.5))
            dist_zero = np.minimum(np.abs(phases - 0.0), 1.0 - np.abs(phases - 0.0))
            near_half = dist_half <= 0.08
            near_zero = dist_zero <= 0.08

            secondary_proxy = int(
                (scores[near_half] >= self.score_threshold * 0.85).sum() >= 2
                and near_zero.sum() >= 2
            )

        return {
            "flag_singleton_spike": singleton,
            "cand_period_days": float(period) if np.isfinite(period) else np.nan,
            "flag_periodic": int(periodic),
            "odd_even_score_gap": float(odd_even_proxy) if np.isfinite(odd_even_proxy) else np.nan,
            "flag_secondary_proxy": int(secondary_proxy),
        }

    # -----------------------------
    # Star-level triage table
    # -----------------------------
    def build_triage_table(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        df_scored must include:
          - star_id
          - score
        Recommended:
          - seg_mid_time
        """
        if "star_id" not in df_scored.columns or "score" not in df_scored.columns:
            raise ValueError("build_triage_table expects columns: star_id, score")

        df = df_scored.copy()
        df["star_id"] = df["star_id"].astype(str)

        if "seg_mid_time" not in df.columns:
            df["seg_mid_time"] = np.nan

        # deterministic top-k
        df = df.sort_values(["star_id", "score"], ascending=[True, False])

        passthrough_cols = [c for c in ["mission", "campaign", "sector", "quarter", "split"] if c in df.columns]

        rows: List[Dict[str, object]] = []

        for star_id, g in df.groupby("star_id", sort=False):
            scores = g["score"].to_numpy(dtype=float)
            times = g["seg_mid_time"].to_numpy(dtype=float)

            top_n = min(self.top_n, len(scores))
            top_scores = scores[:top_n]
            above = scores >= self.score_threshold

            starwise_score = float(np.nanmax(scores)) if len(scores) else np.nan
            top_mean = float(np.nanmean(top_scores)) if top_scores.size else np.nan
            top_std = float(np.nanstd(top_scores)) if top_scores.size else np.nan
            consistency = float(top_mean - top_std) if np.isfinite(top_mean) and np.isfinite(top_std) else np.nan

            out: Dict[str, object] = {
                "star_id": star_id,
                "starwise_score": starwise_score,
                "n_segments": int(len(scores)),
                "n_segments_above_thr": int(np.nansum(above)),
                "best_segment_score": float(top_scores[0]) if top_scores.size else np.nan,
                "topN_mean": top_mean,
                "topN_std": top_std,
                "consistency_score": consistency,
                "top_scores": ",".join([f"{s:.4f}" for s in top_scores[:5]]),
                "top_times": ",".join([f"{t:.6f}" for t in times[:5]]) if np.isfinite(times[: min(5, len(times))]).any() else "",
            }

            for c in passthrough_cols:
                mode = g[c].mode()
                out[c] = mode.iloc[0] if not mode.empty else g[c].iloc[0]

            out.update(self.compute_flags(times, scores))
            rows.append(out)

        triage = pd.DataFrame(rows)

        # Suggested ranking
        sort_cols = ["starwise_score", "consistency_score", "flag_periodic", "n_segments_above_thr"]
        sort_cols = [c for c in sort_cols if c in triage.columns]
        triage = triage.sort_values(sort_cols, ascending=False).reset_index(drop=True)

        return triage
