from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union
from src.Classifiers.K2.K2CacheBuilder import K2CachedScorer

log = logging.getLogger(__name__)
matplotlib.use("Agg")

class K2Loader:
    def __init__(self):
        pass

    @staticmethod
    def _restore_stdio():
        """
        Fix broken stdout/stderr if some other code redirected+closed them.
        Call this BEFORE anything prints.
        """
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    @staticmethod
    def load_epic_ids_from_csv(
        csv_path: Union[str, Path],
        epic_col: str = "star_id",
        limit: Optional[int] = None,
    ) -> List[str]:
        df = pd.read_csv(csv_path)

        if epic_col not in df.columns:
            raise ValueError(f"Column '{epic_col}' not found. Columns: {list(df.columns)}")

        # Clean EPIC ids: allow "EPIC 123", "123", "123.0"
        ids = df[epic_col].astype(str).str.upper()
        ids = ids.str.replace(r"\.0$", "", regex=True)
        ids = ids.str.replace("EPIC", "", regex=False)
        ids = ids.str.replace(r"[^0-9]", "", regex=True).str.strip()
        ids = ids[ids != ""].tolist()

        # de-dupe while preserving order
        seen = set()
        out = []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)

        if limit is not None:
            out = out[: int(limit)]
        return out

    def callK2_LoadData(self):
        # 1) restore stdout/stderr FIRST (before model init / prints)
        self._restore_stdio()

        # 2) configure logging ONCE (do this in main.py ideally)
        # Keeping here as a safe fallback:
        logging.basicConfig(level=logging.INFO)

        scorer = K2CachedScorer(
            model_path="k2_window1024_base.keras",
            cache_dir="k2_cache",
            use_all=False,
            any_author=True,
            seg_len=1024,
            stride=256,
            use_mad_norm=True,
            clip_after_mad=10.0,
            verbose=False,   # IMPORTANT: stop internal print spam
        )

        epic_csv = "data/K2_epic_ids.csv"
        epic_ids = self.load_epic_ids_from_csv(epic_csv, epic_col="star_id", limit=1000)

        # Download phase
        #dl_df = scorer.download_targets(epic_ids, workers=3, sleep_s=0.0)
        #log.info("Download status counts:\n%s", dl_df["status"].value_counts())
        #dl_df.to_csv("k2_download_status.csv", index=False)

        # Score phase
        scores = scorer.score_targets(
            epic_ids,
            out_csv="k2_inference_scores.csv",
            save_every=50,
            resume=True,
        )

        # No print (stdout might still be fragile somewhere). Log instead:
        top = scores.sort_values("best_seg_score", ascending=False).head(25)
        log.info("Top 25 candidates:\n%s", top.to_string(index=False))
        print (scores["status"].value_counts())
        print (scores["n_segments"].describe())
        print( scores.sort_values("best_seg_score", ascending=False).head(50)[["epic_id","n_segments","best_seg_score","median_seg_score","std_seg_score"]] )
        self.printscore(scores)

    def printscore(self, scores):

        df = pd.read_csv("k2_inference_scores.csv")

        ok = df[df["status"].astype(str).str.lower() == "ok"].copy()

        # composite score for stability
        ok["priority"] = ok["best_seg_score"] - 0.5 * ok["std_seg_score"]

        normal = ok[ok["n_segments"] <= 20].sort_values(["priority", "best_seg_score"], ascending=False)
        longs  = ok[ok["n_segments"] > 20].sort_values("best_seg_score", ascending=False)

        normal.head(50).to_csv("k2_top50_normal.csv", index=False)
        longs.head(50).to_csv("k2_top50_long.csv", index=False)

        print("Normal top 10:\n", normal.head(10)[["epic_id","n_segments","best_seg_score","median_seg_score","std_seg_score","priority"]])
        print("\nLong top 10:\n", longs.head(10)[["epic_id","n_segments","best_seg_score","median_seg_score","std_seg_score","priority"]])

    def score_runner(self):
        scorer = K2CachedScorer(
                    model_path="k2_window1024_base.keras",
                    cache_dir="k2_cache",
                    use_all=False,
                    any_author=True,
                    seg_len=1024,
                    stride=256,
                    use_mad_norm=True,
                    clip_after_mad=10.0,
                    verbose=False,   # IMPORTANT: stop internal print spam
                )
        normal = pd.read_csv("k2_top50_normal.csv")["epic_id"].astype(str).tolist()
        longs  = pd.read_csv("k2_top50_long.csv")["epic_id"].astype(str).tolist()

        epic_ids = list(dict.fromkeys(normal + longs))  # de-dupe preserve order

        scores = scorer.score_targets(
        epic_ids,
        out_csv="k2_top_vetting_scores.csv",
        save_every=5,
        resume=False
        )
        print(scores[["epic_id","best_seg_score","best_seg_idx","best_t0","best_t1","status"]].head(10))
        self.make_vetting_packs(
            scorer,
            scores_csv="k2_top_vetting_scores.csv",
            out_dir="vetting_k2",
            neighbor_residuals_for_all=False
        )

    def make_vetting_packs(
        self,
        scorer,
        scores_csv="k2_top_vetting_scores.csv",
        out_dir="vetting_k2",
        neighbor_residuals_for_all=False,   # set True if you want ±1 for everyone
        long_segments_threshold=20,         # treat as "long" if n_segments > this
         ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- helper: clip residual for visualization only (robust) ---
        def _viz_clip(x, k=6.0):
            x = np.asarray(x, dtype=np.float64)
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-6
            lo = med - k * mad
            hi = med + k * mad
            return np.clip(x, lo, hi)

        df = pd.read_csv(scores_csv)
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

        for _, r in df.iterrows():
            epic = str(r["epic_id"])
            best_start = int(r["best_start"])
            best_end = int(r["best_end"])
            best_idx = int(r.get("best_seg_idx", -1))
            n_segments = int(r.get("n_segments", 0))

            time_arr, flux = scorer._get_time_and_flux(epic)
            if time_arr is None or flux is None:
                continue

            raw = flux[:, 0]
            flat = flux[:, 1]
            resid = raw - flat

            star_dir = out_dir / f"EPIC_{epic}"
            star_dir.mkdir(parents=True, exist_ok=True)

            # ---------- Full LC ----------
            plt.figure()
            plt.plot(time_arr, raw, linewidth=0.6, label="raw_norm")
            plt.plot(time_arr, flat, linewidth=0.6, label="flat")
            if 0 <= best_start < len(time_arr):
                plt.axvline(time_arr[best_start], linewidth=1)
            if 0 <= best_end - 1 < len(time_arr):
                plt.axvline(time_arr[min(best_end - 1, len(time_arr) - 1)], linewidth=1)
            plt.title(f"EPIC {epic} full LC")
            plt.xlabel("Time")
            plt.ylabel("Flux (processed)")
            plt.legend(loc="best", fontsize=8, frameon=False)
            plt.tight_layout()
            plt.savefig(star_dir / "full.png", dpi=160)
            plt.close()

            # ---------- Best segment ----------
            s = max(best_start, 0)
            e = min(best_end, len(time_arr))
            if e - s >= 10:
                plt.figure()
                plt.plot(time_arr[s:e], raw[s:e], linewidth=0.8, label="raw_norm")
                plt.plot(time_arr[s:e], flat[s:e], linewidth=0.8, label="flat")
                plt.legend(loc="best", fontsize=8, frameon=False)
                plt.title(f"EPIC {epic} best segment [{s}:{e}]")
                plt.xlabel("Time")
                plt.ylabel("Flux (processed)")
                plt.tight_layout()
                plt.savefig(star_dir / "best_segment.png", dpi=180)
                plt.close()

                # ---------- Best residual (clipped for visibility) ----------
                rr = _viz_clip(resid[s:e], k=6.0)
                plt.figure()
                plt.plot(time_arr[s:e], rr, linewidth=0.8, label="raw - flat (clipped)")
                plt.axhline(0.0, linewidth=0.8)
                plt.legend(loc="best", fontsize=8, frameon=False)
                plt.title(f"EPIC {epic} residual best segment [{s}:{e}]")
                plt.xlabel("Time")
                plt.ylabel("Residual")
                plt.tight_layout()
                plt.savefig(star_dir / "best_residual.png", dpi=180)
                plt.close()

            # ---------- Neighbor residuals (best_idx ± 1) ----------
            do_neighbors = neighbor_residuals_for_all or (n_segments > long_segments_threshold)
            if do_neighbors and best_idx >= 0:
                for delta in (-1, 1):
                    nb_idx = best_idx + delta
                    if nb_idx < 0:
                        continue

                    nb_start = nb_idx * scorer.stride
                    nb_end = nb_start + scorer.seg_len
                    ns = max(nb_start, 0)
                    ne = min(nb_end, len(time_arr))
                    if ne - ns < 10:
                        continue

                    rrn = _viz_clip(resid[ns:ne], k=6.0)
                    plt.figure()
                    plt.plot(time_arr[ns:ne], rrn, linewidth=0.8, label="raw - flat (clipped)")
                    plt.axhline(0.0, linewidth=0.8)
                    plt.legend(loc="best", fontsize=8, frameon=False)
                    plt.title(f"EPIC {epic} residual seg {nb_idx} [{ns}:{ne}]")
                    plt.xlabel("Time")
                    plt.ylabel("Residual")
                    plt.tight_layout()

                    fname = "neighbor_minus1_residual.png" if delta == -1 else "neighbor_plus1_residual.png"
                    plt.savefig(star_dir / fname, dpi=180)
                    plt.close()

        print(f"Done. Wrote vetting packs to: {out_dir}")
