import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.Classifiers.LargeWindow.SegmentDataset import SegmentDataset
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.CommonHelper import CommonHelper
from src.Classifiers.LargeWindow.HardNegativeMiner import HardNegativeMiner


class ModelSplitter:
    def __init__(self, window=1024, stride=256):
        self.window = window
        self.stride = stride
        self.commonHelper = CommonHelper()
        self.helper = BuilderHelper()
        self.ephBuilder = EphermisBuilder(window=self.window, stride=self.stride)

    def split_model(
        self,
        segments_path: str,
        mission: str = "tess",
        catalog_path: str | None = None,
        out_prefix: str | None = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state_test: int = 42,
        random_state_val: int = 43,
        run_diagnostic: bool = True,
    ):
        """
        Star-level split (prevents leakage): train/val/test are split by star_id and stratified
        by whether a star has ANY positive segment.
        """

        mission = mission.strip().lower()
        segments = pd.read_parquet(segments_path).copy()

        if "mission" not in segments.columns:
            raise KeyError("segments parquet missing 'mission' column")
        if "label" not in segments.columns:
            raise KeyError("segments parquet missing 'label' column. Did you label it?")

        segments["mission"] = segments["mission"].astype(str).str.lower()

        seg_m = segments[segments["mission"] == mission].copy()
        if len(seg_m) == 0:
            raise ValueError(f"No segments found for mission='{mission}' in {segments_path}")

        # Star-level stratification target: does this star have any positive segment?
        star_table = (
            seg_m.groupby("star_id")["label"]
            .max()
            .reset_index()
            .rename(columns={"label": "has_pos"})
        )

        # Split stars into train+val vs test
        trainval_ids, test_ids = train_test_split(
            star_table["star_id"],
            test_size=test_size,
            random_state=random_state_test,
            stratify=star_table["has_pos"],
        )

        # Split train+val into train vs val
        trainval_table = star_table.set_index("star_id").loc[trainval_ids]
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=val_size,
            random_state=random_state_val,
            stratify=trainval_table["has_pos"],
        )

        train_df = seg_m[seg_m["star_id"].isin(train_ids)].reset_index(drop=True)
        val_df   = seg_m[seg_m["star_id"].isin(val_ids)].reset_index(drop=True)
        test_df  = seg_m[seg_m["star_id"].isin(test_ids)].reset_index(drop=True)

        print("Mission:", mission)
        print("Stars:  train =", len(train_ids), " val =", len(val_ids), " test =", len(test_ids))
        print("Segs:   train =", len(train_df), " val =", len(val_df), " test =", len(test_df))

        if out_prefix is None:
            out_prefix = f"{mission}_segments"

        train_df.to_parquet(f"{out_prefix}_train.parquet", index=False)
        val_df.to_parquet(f"{out_prefix}_val.parquet", index=False)
        test_df.to_parquet(f"{out_prefix}_test.parquet", index=False)

        # Optional diagnostic (highly recommended when changing labeling logic)
        if run_diagnostic:
            if catalog_path is None:
                print("Skipping diagnostic: provide catalog_path to enable OR-mask check.")
            else:
                self.print_label_stats_with_catalog(
                    train_df=train_df,
                    catalog_path=catalog_path,
                    mission=mission,
                    n_rows=5000,
                )

        return train_df, val_df, test_df

    def print_label_stats_with_catalog(
        self,
        train_df: pd.DataFrame,
        catalog_path: str,
        mission: str,
        n_rows: int = 5000
    ):
        """
        Diagnostic: recompute OR-mask across *all planets* for stars and compare
        in_transit_count distributions between label=0 and label=1.
        """
        mission = mission.strip().lower()

        if len(train_df) == 0:
            print("print_label_stats_with_catalog: train_df empty")
            return

        sample = train_df.sample(n=min(n_rows, len(train_df)), random_state=0)

        catalog = pd.read_csv(catalog_path, low_memory=False)
        eph = self.ephBuilder._prepare_ephemeris(catalog)
        eph["mission"] = eph["mission"].astype(str).str.lower()

        counts = []

        for star_id, g in sample.groupby("star_id"):
            try:
                time, flux, _, _ = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except Exception:
                continue

            if time is None or len(time) == 0:
                continue

            planets = eph[(eph["star_id"] == star_id) & (eph["mission"] == mission)]
            if len(planets) == 0:
                continue

            masks = []
            for _, p in planets.iterrows():
                masks.append(
                    self.ephBuilder._in_transit_mask(
                        time=np.asarray(time),
                        period=float(p["period_days"]),
                        t0=float(p["t0_mission"]),
                        duration_days=float(p["duration_days"]),
                    )
                )

            if not masks:
                continue

            mask_any = np.logical_or.reduce(masks)

            for _, row in g.iterrows():
                start, end = int(row["start"]), int(row["end"])
                if start < 0 or end > len(time):
                    continue
                itc = int(mask_any[start:end].sum())
                counts.append((int(row["label"]), itc))

        if not counts:
            print("No diagnostic counts computed (missing cache or ephemerides).")
            return

        counts = pd.DataFrame(counts, columns=["label", "in_transit_count"])
        print(counts.groupby("label")["in_transit_count"].describe())

    def generateSegments(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame,
            batch_size: int = 64,
            seed: int = 42,
            mission: str = "tess",
            hard_neg_path: str | None = None,
            balance: bool = True,
            neg_pos_ratio: int = 3,
            hard_neg_frac: float = 0.0,   # <- DEFAULT OFF (important!)
        ):
        mission = mission.strip().lower()

        val_gen  = SegmentDataset(val_df,  batch_size=batch_size, shuffle=False, window=self.window)
        test_gen = SegmentDataset(test_df, batch_size=batch_size, shuffle=False, window=self.window)

        if not balance:
            train_use = train_df.reset_index(drop=True)
            print("Unbalanced train pos frac:", float(train_use["label"].mean()))
            print("steps/epoch:", int(np.ceil(len(train_use) / batch_size)))
            train_gen = SegmentDataset(train_use, batch_size=batch_size, shuffle=True, window=self.window)
            return train_gen, val_gen, test_gen

        pos = train_df[train_df["label"] == 1].copy()
        neg = train_df[train_df["label"] == 0].copy()
        if len(pos) == 0:
            raise RuntimeError("No positive segments in train_df!")

        if neg_pos_ratio <= 0:
            neg_pos_ratio = 1

        n_pos = len(pos)
        n_neg_total = min(len(neg), neg_pos_ratio * n_pos)

        pos_use = pos.sample(n=n_pos, random_state=seed, replace=False).reset_index(drop=True)

        hard_use = pd.DataFrame()
        n_hard = 0

        # --- SAFE guard: only use hard negs if file exists AND hard_neg_frac > 0 ---
        if hard_neg_path and hard_neg_frac > 0.0:
            import os
            if os.path.exists(hard_neg_path):
                hard = pd.read_parquet(hard_neg_path).copy()

                if "mission" in hard.columns:
                    hard["mission"] = hard["mission"].astype(str).str.lower()
                    hard = hard[hard["mission"] == mission]

                if "label" in hard.columns:
                    hard = hard[hard["label"] == 0].copy()
                else:
                    hard["label"] = 0

                if "score" in hard.columns:
                    hard = hard.drop(columns=["score"])

                n_hard = min(len(hard), int(round(hard_neg_frac * n_neg_total)))
                if n_hard > 0:
                    hard_use = hard.sample(n=n_hard, random_state=seed + 1, replace=False).reset_index(drop=True)
            else:
                print(f"[generateSegments] hard_neg_path not found, ignoring: {hard_neg_path}")

        n_rand = n_neg_total - n_hard
        rand_use = neg.sample(n=n_rand, random_state=seed + 2, replace=False).reset_index(drop=True)

        train_use = (
            pd.concat([pos_use, hard_use, rand_use], ignore_index=True)
            .sample(frac=1.0, random_state=seed + 3)
            .reset_index(drop=True)
        )

        print("Train label counts:\n", train_use["label"].value_counts())
        print("Train pos frac:", float(train_use["label"].mean()))
        print("Neg:Pos:", f"{n_neg_total}:{n_pos}", "| hard_used:", n_hard, "| hard_neg_frac:", hard_neg_frac)
        print("steps/epoch:", int(np.ceil(len(train_use) / batch_size)))

        train_gen = SegmentDataset(train_use, batch_size=batch_size, shuffle=True, window=self.window)
        return train_gen, val_gen, test_gen

   
    def mine_hard_negatives(
            self,
            train_df: pd.DataFrame,
            model_path: str = "tess_window1024.keras",
            mission: str = "tess",
            max_neg_pool: int = 300_000,
            topk: int = 80_000,
            max_neg_per_star: int | None = 200,
            output_path: str = "tess_hard_neg_top80k.parquet"):
        miner = HardNegativeMiner(window=self.window, batch_size=64)
        hard = miner.mine(
            model_path=model_path,
            train_df=train_df,
            mission=mission,
            max_neg_pool=max_neg_pool,
            topk=topk,
            max_neg_per_star=max_neg_per_star,
            output_path=output_path,
        )
        return hard

