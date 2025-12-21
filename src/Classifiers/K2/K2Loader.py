# src/Classifiers/K2/K2Loader.py

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from src.Classifiers.K2.K2CacheBuilder import K2CachedScorer


log = logging.getLogger(__name__)


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
