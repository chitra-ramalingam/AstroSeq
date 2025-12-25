from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
class AstroSeqCandidateTriage:
    """
    AstroSeqCandidateTriage

    Responsibility: triage and aggregation only.
    Input must already contain a 'score' column.

    This is your original triage class, with the X-building/scoring moved out
    into K2ScoreLoader. Based on your shared implementation. :contentReference[oaicite:0]{index=0}
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
    # Split assignment from your existing meta parquets
    # -----------------------------
    def assign_split_from_meta_parquets(
        self,
        df: pd.DataFrame,
        train_meta_path: str | Path,
        val_meta_path: str | Path,
        test_meta_path: str | Path,
        key_cols: Tuple[str, str, str] = ("star_id", "start", "end"),
    ) -> pd.DataFrame:
        """
        Adds df['split'] by matching keys against:
          - k2_segments_train.parquet
          - k2_segments_val.parquet
          - k2_segments_test.parquet
        """
        df = df.copy()

        for c in key_cols:
            if c not in df.columns:
                raise ValueError(f"Key column '{c}' not found in feature dataframe.")

        def _load_keys(p: str | Path, split_name: str) -> pd.DataFrame:
            m = pd.read_parquet(Path(p))
            for c in key_cols:
                if c not in m.columns:
                    raise ValueError(f"Key column '{c}' not found in meta parquet: {p}")
            m = m[list(key_cols)].copy()
            m["star_id"] = m["star_id"].astype(str)
            m["split"] = split_name
            return m.drop_duplicates()

        keys = pd.concat(
            [
                _load_keys(train_meta_path, "train"),
                _load_keys(val_meta_path, "val"),
                _load_keys(test_meta_path, "test"),
            ],
            ignore_index=True,
        )

        df["star_id"] = df["star_id"].astype(str)
        df = df.merge(keys, on=list(key_cols), how="left")
        return df

    # -----------------------------
    # Period + epoch utilities
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
    # Cheap triage flags
    # -----------------------------
    def compute_flags(self, times: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
        times = np.asarray(times, dtype=float)
        scores = np.asarray(scores, dtype=float)

        m = np.isfinite(times) & np.isfinite(scores)
        times = times[m]
        scores = scores[m]

        above = scores >= self.score_threshold
        singleton = int(above.sum() == 1)

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
    # Star-level triage
    # -----------------------------
    def build_triage_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["star_id"] = df["star_id"].astype(str)

        if "score" not in df.columns:
            raise ValueError("build_triage_table expects a 'score' column.")

        if "seg_mid_time" not in df.columns:
            df["seg_mid_time"] = np.nan

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
                "top_times": (
                    ",".join([f"{t:.6f}" for t in times[:5]])
                    if np.isfinite(times[: min(5, len(times))]).any()
                    else ""
                ),
            }

            for c in passthrough_cols:
                mode = g[c].mode()
                out[c] = mode.iloc[0] if not mode.empty else g[c].iloc[0]

            out.update(self.compute_flags(times, scores))
            rows.append(out)

        triage = pd.DataFrame(rows)

        sort_cols = ["starwise_score", "consistency_score", "flag_periodic", "n_segments_above_thr"]
        sort_cols = [c for c in sort_cols if c in triage.columns]
        return triage.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # -----------------------------
    # One-shot pipeline (meta parquet -> scored -> triage)
    # -----------------------------
    def meta_parquet_to_triage(
        self,
        score_loader: K2ScoreLoader,
        meta_parquet_path: str | Path,
        keras_model_path: str | Path,
        split_name: str,
        out_triage_path: str | Path | None = None,
        batch_size: int = 256,
        on_bad_index: str = "pad",
    ) -> pd.DataFrame:
        df_scored = score_loader.score_meta_parquet(
            meta_parquet_path=meta_parquet_path,
            keras_model_path=keras_model_path,
            batch_size=batch_size,
            on_bad_index=on_bad_index,
        )
        df_scored["split"] = split_name
        triage = self.build_triage_table(df_scored)
        if out_triage_path is not None:
            self.write_triage(triage, out_triage_path)
        return triage

    # -----------------------------
    # One-shot pipeline (feature parquet -> scored -> split -> triage)
    # -----------------------------
    def feature_parquet_to_triage(
        self,
        score_loader: K2ScoreLoader,
        feature_parquet_path: str | Path,
        keras_model_path: str | Path,
        train_meta_path: str | Path,
        val_meta_path: str | Path,
        test_meta_path: str | Path,
        out_triage_path: str | Path,
        feature_col: Optional[str] = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        df_scored = score_loader.score_feature_parquet(
            feature_parquet_path=feature_parquet_path,
            keras_model_path=keras_model_path,
            feature_col=feature_col,
            batch_size=batch_size,
        )

        df_scored = self.assign_split_from_meta_parquets(
            df_scored,
            train_meta_path=train_meta_path,
            val_meta_path=val_meta_path,
            test_meta_path=test_meta_path,
            key_cols=("star_id", "start", "end"),
        )

        df_scored = df_scored[df_scored["split"].notna()].copy()
        triage = self.build_triage_table(df_scored)
        self.write_triage(triage, out_triage_path)
        return triage

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

