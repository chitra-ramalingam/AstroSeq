import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.Classifiers.LargeWindow.SegmentDataset import SegmentDataset
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.CommonHelper import CommonHelper


class ModelSplitter:
    def __init__(self, window=1024, stride=256):
        self.window = window
        self.stride = stride
        self.commonHelper = CommonHelper()
        self.helper = BuilderHelper()
        self.ephBuilder = EphermisBuilder(window=self.window, stride=self.stride)

    def split_model(
            self,
            segments_path: str = "segments_all_W1024_S256_tess_labeled.parquet",
            catalog_path: str | None = None,
        ):
        segments = pd.read_parquet(segments_path)
        tess_segments = segments[segments["mission"].str.lower() == "tess"].copy()

        if "label" not in tess_segments.columns:
            raise KeyError("Column 'label' not found. Did you label the segments parquet?")

        star_table = (
            tess_segments.groupby("star_id")["label"].max().reset_index().rename(columns={"label": "has_pos"})
        )

        trainval_ids, test_ids = train_test_split(
            star_table["star_id"],
            test_size=0.15,
            random_state=42,
            stratify=star_table["has_pos"],
        )

        trainval_table = star_table.set_index("star_id").loc[trainval_ids]
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=0.15,
            random_state=43,
            stratify=trainval_table["has_pos"],
        )

        train_df = tess_segments[tess_segments["star_id"].isin(train_ids)].reset_index(drop=True)
        val_df   = tess_segments[tess_segments["star_id"].isin(val_ids)].reset_index(drop=True)
        test_df  = tess_segments[tess_segments["star_id"].isin(test_ids)].reset_index(drop=True)

        print("Stars:  train =", len(train_ids), " val =", len(val_ids), " test =", len(test_ids))
        print("Segs:   train =", len(train_df), " val =", len(val_df), " test =", len(test_df))

        train_df.to_parquet("tess_segments_train.parquet", index=False)
        val_df.to_parquet("tess_segments_val.parquet", index=False)
        test_df.to_parquet("tess_segments_test.parquet", index=False)

        if catalog_path is not None:
            self.print_label_stats_with_catalog(train_df, catalog_path=catalog_path)
        else:
            print("Skipping print_label_stats: provide catalog_path to enable diagnostic OR-mask check.")

        return train_df, val_df, test_df

    def print_label_stats(self, train_df: pd.DataFrame, n_rows: int = 5000):
        """
        Diagnostic: compute in_transit_count inside each window using OR-mask across ALL planets.
        This must match the multi-planet labeling logic, otherwise you'll see identical distributions.
        """
        if len(train_df) == 0:
            print("print_label_stats: train_df empty")
            return

        sample = train_df.sample(n=min(n_rows, len(train_df)), random_state=0)

        # Build ephemerides for all stars in this sample from your catalog-derived logic.
        # NOTE: This requires you have access to the same catalog in EphermisBuilder labeling step.
        # If EphermisBuilder already wrote a labeled parquet WITHOUT ephemeris columns,
        # we recompute eph from the same catalog used earlier.
        #
        # Easiest: store catalog_path on the class or pass it in.
        #
        # For now, we assume EphermisBuilder has an ephemeris table cached/available
        # by reloading it from the catalog you used for labeling.
        #
        # If you want this diagnostic to work without catalog, keep ephemeris columns in labeled parquet.
        raise NotImplementedError(
            "print_label_stats needs catalog ephemerides to compute OR-mask. "
            "Use print_label_stats_with_catalog(train_df, catalog_path) instead."
        )

    def print_label_stats_with_catalog(self, train_df: pd.DataFrame, catalog_path: str, n_rows: int = 5000):
        """
        Same diagnostic, but you provide the catalog path so we can rebuild ephemerides and OR them.
        """
        if len(train_df) == 0:
            print("print_label_stats_with_catalog: train_df empty")
            return

        sample = train_df.sample(n=min(n_rows, len(train_df)), random_state=0)

        catalog = pd.read_csv(catalog_path)
        eph = self.ephBuilder._prepare_ephemeris(catalog)
        eph["mission"] = eph["mission"].str.lower()

        counts = []

        for star_id, g in sample.groupby("star_id"):
            try:
                time, flux, _, _ = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except Exception:
                continue

            if time is None or len(time) == 0:
                continue

            planets = eph[(eph["star_id"] == star_id) & (eph["mission"] == "tess")]
            if len(planets) == 0:
                # no ephemeris rows found for this star
                continue

            masks = []
            for _, p in planets.iterrows():
                masks.append(
                    self.ephBuilder._in_transit_mask(
                        time=time,
                        period=float(p["period_days"]),
                        t0=float(p["t0_mission"]),
                        duration_days=float(p["duration_days"]),
                    )
                )

            if len(masks) == 0:
                continue

            mask_any = np.logical_or.reduce(masks)

            for _, row in g.iterrows():
                start, end = int(row["start"]), int(row["end"])
                if start < 0 or end > len(time):
                    continue
                itc = int(mask_any[start:end].sum())
                counts.append((int(row["label"]), itc))

        if len(counts) == 0:
            print("No diagnostic counts computed (missing cache or ephemerides).")
            return

        counts = pd.DataFrame(counts, columns=["label", "in_transit_count"])
        print(counts.groupby("label")["in_transit_count"].describe())

    def generateSegments(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        BATCH_SIZE = 64

        pos_train = train_df[train_df["label"] == 1]
        neg_train = train_df[train_df["label"] == 0]

        print("Raw train pos:", len(pos_train), "neg:", len(neg_train))

        if len(pos_train) == 0:
            raise RuntimeError("No positive segments in train_df!")

        # 50/50 balancing (strongest signal for learning)
        neg_sampled = neg_train.sample(n=min(len(neg_train), len(pos_train)), random_state=42)

        balanced_train_df = (
            pd.concat([pos_train, neg_sampled])
            .sample(frac=1.0, random_state=43)
            .reset_index(drop=True)
        )

        print("Balanced counts:\n", balanced_train_df["label"].value_counts())
        print("Balanced pos frac:", balanced_train_df["label"].mean())

        print("train segments:", len(balanced_train_df))

        MAX_TRAIN_SEGS = 50_000
        if len(balanced_train_df) > MAX_TRAIN_SEGS:
            balanced_train_df = (
                balanced_train_df
                .sample(n=MAX_TRAIN_SEGS, random_state=123)
                .reset_index(drop=True)
            )

        print("Final train segments:", len(balanced_train_df))
        print("Balanced train label counts:\n", balanced_train_df["label"].value_counts())
        print("Balanced train pos fraction:", balanced_train_df["label"].mean())

        train_gen = SegmentDataset(
            balanced_train_df,
            batch_size=BATCH_SIZE,
            shuffle=True,
            window=self.window,
        )

        print("steps per epoch:", len(train_gen))

        val_gen = SegmentDataset(val_df, batch_size=BATCH_SIZE, shuffle=False, window=self.window)
        test_gen = SegmentDataset(test_df, batch_size=BATCH_SIZE, shuffle=False, window=self.window)

        return train_gen, val_gen, test_gen
