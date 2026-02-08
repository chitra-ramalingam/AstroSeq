from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class K2Validation_Prediction:
    def validate_period_by_prediction(
        self,
        time: np.ndarray,
        resid: np.ndarray,
        local_sigma: Optional[np.ndarray],
        events_df: pd.DataFrame,
        P: float,
        t0: Optional[float] = None,
        tol_days: float = 0.08,
        sigma_floor: Optional[float] = None,
        snr_threshold: float = 3.0,
        max_rows: int = 6,
        do_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a proposed period by predicting transit times and checking local dip SNR.

        Returns:
            Dict with coverage-aware metrics and rows=[(tk, dip_snr, min_resid), ...].
        """
        t = np.asarray(time, dtype=float)
        r = np.asarray(resid, dtype=float)
        if local_sigma is None:
            s = np.full_like(r, np.nan, dtype=float)
        else:
            s = np.asarray(local_sigma, dtype=float)
        if t.shape != r.shape:
            raise ValueError("time and resid must have the same shape")
        if s.shape != r.shape:
            raise ValueError("local_sigma must be None or have the same shape as time/resid")
        if not np.isfinite(P) or P <= 0:
            raise ValueError("P must be finite and > 0")
        if not np.isfinite(tol_days) or tol_days <= 0:
            raise ValueError("tol_days must be finite and > 0")
        if "t_mid" not in events_df.columns:
            raise ValueError("events_df must contain 't_mid'")

        finite = np.isfinite(t) & np.isfinite(r)
        t = t[finite]
        r = r[finite]
        s = s[finite]
        if len(t) == 0:
            return {
                "hit_rate": float("nan"),
                "hit_rate_3": float("nan"),
                "hit_rate_4": float("nan"),
                "hit_rate_overall": float("nan"),
                "mean_hit_snr": float("nan"),
                "n_predicted": 0,
                "n_covered": 0,
                "coverage_rate": float("nan"),
                "sigma_floor": float("nan"),
                "t0": float("nan"),
                "rows": [],
                "covered_mask": [],
                "worst_misses": [],
                "best_hits": [],
                "uncovered_windows": [],
            }

        if sigma_floor is None:
            pos_sigma = s[np.isfinite(s) & (s > 0)]
            sigma_floor_value = float(np.nanpercentile(pos_sigma, 5)) if len(pos_sigma) > 0 else 1e-5
        else:
            sigma_floor_value = float(sigma_floor)
        if (not np.isfinite(sigma_floor_value)) or (sigma_floor_value <= 0):
            sigma_floor_value = 1e-5

        t_min = float(np.min(t))
        t_max = float(np.max(t))
        t_mid_all = pd.to_numeric(events_df["t_mid"], errors="coerce")

        if t0 is None:
            cluster_idx = events_df.attrs.get("in_cluster_indices", None)
            if cluster_idx is None and "in_cluster" in events_df.columns:
                cluster_idx = events_df.index[events_df["in_cluster"].astype(bool)].tolist()
            if cluster_idx is None:
                cluster_idx = list(events_df.index)

            cluster_df = events_df.loc[events_df.index.intersection(cluster_idx)].copy()
            if cluster_df.empty:
                cluster_df = events_df.copy()

            if "shape_score" in cluster_df.columns and cluster_df["shape_score"].notna().any():
                best_idx = pd.to_numeric(cluster_df["shape_score"], errors="coerce").idxmax()
                t_mid_ref = float(pd.to_numeric(cluster_df.loc[best_idx, "t_mid"], errors="coerce"))
            else:
                t_mid_ref = float(np.nanmedian(pd.to_numeric(cluster_df["t_mid"], errors="coerce")))

            if not np.isfinite(t_mid_ref):
                t_mid_ref = float(np.nanmedian(t_mid_all.to_numpy(dtype=float)))
            if not np.isfinite(t_mid_ref):
                t_mid_ref = float(t_min)

            t0 = float(t_mid_ref - np.round((t_mid_ref - t_min) / float(P)) * float(P))

        k_start = int(np.ceil((t_min - float(t0)) / float(P)))
        k_end = int(np.floor((t_max - float(t0)) / float(P)))
        if k_end < k_start:
            return {
                "hit_rate": float("nan"),
                "hit_rate_3": float("nan"),
                "hit_rate_4": float("nan"),
                "hit_rate_overall": float("nan"),
                "mean_hit_snr": float("nan"),
                "n_predicted": 0,
                "n_covered": 0,
                "coverage_rate": float("nan"),
                "sigma_floor": float(sigma_floor_value),
                "t0": float(t0),
                "rows": [],
                "covered_mask": [],
                "worst_misses": [],
                "best_hits": [],
                "uncovered_windows": [],
            }

        k_vals = np.arange(k_start, k_end + 1, dtype=int)
        tk_vals = float(t0) + (k_vals.astype(float) * float(P))

        rows: List[Tuple[float, float, float]] = []
        covered_mask: List[bool] = []
        for tk in tk_vals:
            near = np.abs(t - float(tk)) <= float(tol_days)
            if not np.any(near):
                rows.append((float(tk), float("nan"), float("nan")))
                covered_mask.append(False)
                continue

            seg = r[near]
            seg_sigma = s[near]
            seg_sigma_pos = seg_sigma[np.isfinite(seg_sigma) & (seg_sigma > 0)]
            if len(seg_sigma_pos) > 0:
                sigma = float(np.nanmedian(seg_sigma_pos))
            else:
                seg_center = float(np.nanmedian(seg))
                sigma = float(1.4826 * np.nanmedian(np.abs(seg - seg_center)))
            if not np.isfinite(sigma):
                sigma = sigma_floor_value
            sigma = float(max(sigma, sigma_floor_value))

            min_resid = float(np.nanmin(seg))
            dip_snr = float((-min_resid) / sigma)
            if not np.isfinite(dip_snr):
                dip_snr = float("nan")
            rows.append((float(tk), dip_snr, min_resid))
            covered_mask.append(True)

        out_df = pd.DataFrame(rows, columns=["tk", "dip_snr", "min_resid"])
        out_df["covered"] = np.asarray(covered_mask, dtype=bool)
        out_df["hit3"] = out_df["dip_snr"] > 3.0
        out_df["hit4"] = out_df["dip_snr"] > 4.0
        out_df["hit_thr"] = out_df["dip_snr"] > float(snr_threshold)

        n_predicted = int(len(out_df))
        n_covered = int(out_df["covered"].sum())
        coverage_rate = float(n_covered / n_predicted) if n_predicted > 0 else float("nan")

        covered_df = out_df[out_df["covered"]].copy()
        hit_rate = float(covered_df["hit_thr"].mean()) if len(covered_df) > 0 else float("nan")
        hit_rate3 = float(covered_df["hit3"].mean()) if len(covered_df) > 0 else float("nan")
        hit_rate4 = float(covered_df["hit4"].mean()) if len(covered_df) > 0 else float("nan")
        hit_rate_overall = float(out_df["hit_thr"].mean()) if len(out_df) > 0 else float("nan")

        hit_snrs = covered_df.loc[covered_df["hit_thr"] & np.isfinite(covered_df["dip_snr"]), "dip_snr"].to_numpy(dtype=float)
        mean_hit_snr = float(np.mean(hit_snrs)) if len(hit_snrs) > 0 else float("nan")

        best_hits_df = covered_df[covered_df["hit_thr"]].sort_values("dip_snr", ascending=False).head(int(max_rows))
        worst_misses = (
            covered_df[~covered_df["hit_thr"]]
            .assign(_sort_snr=lambda x: x["dip_snr"].fillna(-np.inf))
            .sort_values("_sort_snr", ascending=True)
            .drop(columns="_sort_snr")
            .head(int(max_rows))
        )
        uncovered_df = out_df[~out_df["covered"]].copy()

        print(
            f"[validate_period_by_prediction] P={P:.8f} t0={float(t0):.6f} "
            f"n_predicted={n_predicted} n_covered={n_covered} coverage_rate={coverage_rate:.3f} "
            f"sigma_floor={sigma_floor_value:.6g} "
            f"hit_rate@>3={hit_rate3:.3f} hit_rate@>4={hit_rate4:.3f} "
            f"hit_rate@>{snr_threshold:.1f}={hit_rate:.3f} hit_rate_overall={hit_rate_overall:.3f} "
            f"mean_hit_snr={mean_hit_snr:.3f}"
        )
        if len(worst_misses) > 0:
            print("[validate_period_by_prediction] worst misses:")
            print(worst_misses[["tk", "dip_snr", "min_resid"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("[validate_period_by_prediction] worst misses: none")
        if len(best_hits_df) > 0:
            print("[validate_period_by_prediction] best hits:")
            print(best_hits_df[["tk", "dip_snr", "min_resid"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("[validate_period_by_prediction] best hits: none")
        if len(uncovered_df) > 0:
            print("[validate_period_by_prediction] uncovered windows:")
            print(uncovered_df[["tk"]].head(int(max_rows)).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("[validate_period_by_prediction] uncovered windows: none")

        if do_plot and len(out_df) > 0:
            try:
                import matplotlib.pyplot as plt

                hit_times = out_df.loc[out_df["covered"] & out_df["hit_thr"], "tk"].to_numpy(dtype=float)
                miss_times = out_df.loc[out_df["covered"] & (~out_df["hit_thr"]), "tk"].to_numpy(dtype=float)
                uncovered_times = out_df.loc[~out_df["covered"], "tk"].to_numpy(dtype=float)

                fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                axes[0].plot(t, r, ".", ms=1.5, alpha=0.30, color="black", label="resid")
                if len(hit_times) > 0:
                    axes[0].vlines(hit_times, ymin=float(np.nanmin(r)), ymax=float(np.nanmax(r)), color="tab:green", alpha=0.18, lw=0.8)
                if len(miss_times) > 0:
                    axes[0].vlines(miss_times, ymin=float(np.nanmin(r)), ymax=float(np.nanmax(r)), color="tab:red", alpha=0.18, lw=0.8)
                if len(uncovered_times) > 0:
                    axes[0].vlines(uncovered_times, ymin=float(np.nanmin(r)), ymax=float(np.nanmax(r)), color="gray", alpha=0.18, lw=0.8)
                axes[0].set_ylabel("resid")
                axes[0].set_title(f"Predicted transits for P={P:.8f} d, t0={float(t0):.6f}")

                axes[1].plot(out_df["tk"], out_df["dip_snr"], "o-", ms=3, lw=1, color="tab:blue")
                axes[1].axhline(3.0, color="tab:orange", ls="--", lw=1)
                axes[1].axhline(4.0, color="tab:green", ls="--", lw=1)
                axes[1].set_ylabel("dip_snr")
                axes[1].set_xlabel("time")
                axes[1].set_ylim(bottom=0.0)
                fig.tight_layout()
                plt.close(fig)
            except Exception as exc:
                print(f"[validate_period_by_prediction] plot skipped: {exc}")

        worst_miss_rows = [
            (float(row.tk), float(row.dip_snr), float(row.min_resid))
            for row in worst_misses.itertuples(index=False)
        ]
        best_hit_rows = [
            (float(row.tk), float(row.dip_snr), float(row.min_resid))
            for row in best_hits_df.itertuples(index=False)
        ]
        uncovered_windows = [float(x) for x in uncovered_df["tk"].to_numpy(dtype=float)]

        return {
            "hit_rate": hit_rate,
            "hit_rate_3": hit_rate3,
            "hit_rate_4": hit_rate4,
            "hit_rate_overall": hit_rate_overall,
            "mean_hit_snr": mean_hit_snr,
            "n_predicted": n_predicted,
            "n_covered": n_covered,
            "coverage_rate": coverage_rate,
            "sigma_floor": float(sigma_floor_value),
            "t0": float(t0),
            "rows": rows,
            "covered_mask": covered_mask,
            "worst_misses": worst_miss_rows,
            "best_hits": best_hit_rows,
            "uncovered_windows": uncovered_windows,
        }
