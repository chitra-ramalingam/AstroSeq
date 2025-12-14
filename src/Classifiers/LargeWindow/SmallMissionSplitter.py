import pandas as pd
import numpy as np
from src.Classifiers.LargeWindow.SegmentDataset import SegmentDataset
from sklearn.model_selection import train_test_split

from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.CommonHelper import CommonHelper


class SmallDataSetMissionSplitter:
    """
    For small missions like K2 where stratified splitting often fails or yields 0 positive stars in val/test.

    - Star-level split (no leakage).
    - Tries multiple random seeds until train/val/test each have >=1 positive star (when feasible).
    - If infeasible (e.g. only 1-2 positive stars total), does best-effort split.
    """

    def __init__(self, window=1024, stride=256):
        self.window = window
        self.stride = stride
        self.commonHelper = CommonHelper()
        self.ephBuilder = EphermisBuilder(window=self.window, stride=self.stride)

    def split_model(
        self,
        segments_path: str,
        mission: str = "k2",
        catalog_path: str | None = None,
        out_prefix: str | None = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 123,
        max_tries: int = 500,
        run_diagnostic: bool = True,
    ):
        mission = (mission or "").strip().lower()
        segments = pd.read_parquet(segments_path).copy()

        if "mission" not in segments.columns:
            raise KeyError("segments parquet missing 'mission' column")
        if "label" not in segments.columns:
            raise KeyError("segments parquet missing 'label' column. Did you label it?")

        segments["mission"] = segments["mission"].astype(str).str.strip().str.lower()
        segments["label"] = segments["label"].fillna(0).astype(int)

        seg_m = segments[segments["mission"] == mission].copy()
        if seg_m.empty:
            raise ValueError(f"No segments found for mission='{mission}' in {segments_path}")

        # Star-level table and has_pos for each star
        star_table = (
            seg_m.groupby("star_id", as_index=False)["label"]
            .max()
            .rename(columns={"label": "has_pos"})
        )
        star_table["has_pos"] = star_table["has_pos"].astype(int)

        stars = star_table["star_id"].to_numpy()
        y = star_table["has_pos"].to_numpy()

        n_stars = len(stars)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())

        if n_stars < 5:
            raise ValueError(f"Too few stars ({n_stars}) to split safely.")

        # map for quick counts
        has_pos_map = dict(zip(stars, y))

        def count_pos(star_ids):
            return int(sum(has_pos_map.get(s, 0) for s in star_ids))

        # When positive stars are tiny, strict stratification can break.
        # We'll *attempt* stratification only if it won't error.
        can_stratify = (n_pos >= 2) and (n_neg >= 2)

        def attempt(seed: int):
            strat1 = y if can_stratify and len(np.unique(y)) > 1 else None

            # split test
            trainval, test = train_test_split(
                stars,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=strat1,
            )

            # split val from trainval
            y_trainval = np.array([has_pos_map[s] for s in trainval], dtype=int)
            strat2 = y_trainval if can_stratify and len(np.unique(y_trainval)) > 1 else None

            train, val = train_test_split(
                trainval,
                test_size=val_size,
                random_state=seed + 999,
                shuffle=True,
                stratify=strat2,
            )

            # If feasible, require >=1 positive star in each split
            if n_pos >= 3:
                if count_pos(train) == 0 or count_pos(val) == 0 or count_pos(test) == 0:
                    return None

            return train, val, test

        split = None
        for i in range(max_tries):
            split = attempt(random_state + i)
            if split is not None:
                break

        # If we couldn't satisfy constraints, fall back to a best-effort split
        if split is None:
            split = attempt(random_state)
            if split is None:
                # Last fallback: no stratification at all
                trainval, test = train_test_split(
                    stars,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=True,
                    stratify=None,
                )
                train, val = train_test_split(
                    trainval,
                    test_size=val_size,
                    random_state=random_state + 999,
                    shuffle=True,
                    stratify=None,
                )
                split = (train, val, test)

        train_ids, val_ids, test_ids = split

        train_df = seg_m[seg_m["star_id"].isin(train_ids)].reset_index(drop=True)
        val_df   = seg_m[seg_m["star_id"].isin(val_ids)].reset_index(drop=True)
        test_df  = seg_m[seg_m["star_id"].isin(test_ids)].reset_index(drop=True)

        if out_prefix is None:
            out_prefix = f"{mission}_segments_small"

        train_df.to_parquet(f"{out_prefix}_train.parquet", index=False)
        val_df.to_parquet(f"{out_prefix}_val.parquet", index=False)
        test_df.to_parquet(f"{out_prefix}_test.parquet", index=False)

        print("Mission:", mission)
        print("Stars:  train =", len(train_ids), " val =", len(val_ids), " test =", len(test_ids))
        print("PosStars:", "train =", count_pos(train_ids), " val =", count_pos(val_ids), " test =", count_pos(test_ids))
        print("Segs:   train =", len(train_df), " val =", len(val_df), " test =", len(test_df))
        print("PosSegs:", "train =", int((train_df["label"] == 1).sum()),
              " val =", int((val_df["label"] == 1).sum()),
              " test =", int((test_df["label"] == 1).sum()))
        print("PosFrac:", "train =", float(train_df["label"].mean() if len(train_df) else 0.0),
              " val =", float(val_df["label"].mean() if len(val_df) else 0.0),
              " test =", float(test_df["label"].mean() if len(test_df) else 0.0))

        if run_diagnostic:
            if catalog_path is None:
                print("Skipping diagnostic: provide catalog_path to enable OR-mask check.")
            else:
                self.print_label_stats_with_catalog(
                    train_df=train_df,
                    catalog_path=catalog_path,
                    mission=mission,
                    n_rows=min(5000, len(train_df)),
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
        Same diagnostic as your big splitter, but K2-safe:
        converts t0 to the cache time base before computing masks.
        """
        mission = (mission or "").strip().lower()

        if train_df.empty:
            print("print_label_stats_with_catalog: train_df empty")
            return

        sample = train_df.sample(n=min(n_rows, len(train_df)), random_state=0)

        catalog = pd.read_csv(catalog_path, low_memory=False)
        eph = self.ephBuilder._prepare_ephemeris(catalog).copy()
        eph["mission"] = eph["mission"].astype(str).str.strip().str.lower()

        counts = []

        for star_id, g in sample.groupby("star_id"):
            try:
                time, flux, _, _ = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except Exception:
                continue

            if time is None or len(time) == 0:
                continue

            time = np.asarray(time)
            time_base = self.ephBuilder._guess_time_base(time)

            planets = eph[(eph["star_id"] == star_id) & (eph["mission"] == mission)]
            if len(planets) == 0:
                continue

            masks = []
            for _, p in planets.iterrows():
                period = float(p["period_days"])
                t0_mission = float(p["t0_mission"])
                dur = float(p["duration_days"])

                t0 = self.ephBuilder._convert_t0_to_timebase(
                    mission=mission,
                    t0_mission=t0_mission,
                    time_base=time_base
                )

                masks.append(
                    self.ephBuilder._in_transit_mask(
                        time=time,
                        period=period,
                        t0=t0,
                        duration_days=dur,
                    )
                )

            if not masks:
                continue

            mask_any = np.logical_or.reduce(masks)

            for _, row in g.iterrows():
                start, end = int(row["start"]), int(row["end"])
                if start < 0 or end > len(time) or end <= start:
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
        hard_neg_frac: float = 0.0,
    ):
        mission = mission.strip().lower()

        # Ensure label is numeric
        for df in (train_df, val_df, test_df):
            df["label"] = df["label"].fillna(0).astype(int)

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

        # Optional hard negatives
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


