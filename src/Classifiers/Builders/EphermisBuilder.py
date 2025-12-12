import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, models
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from src.Classifiers.isolation_forest import IsolationForestScorer
from src.Classifiers.CnnNNet import CnnNNet
from src.Classifiers.CommonHelper import CommonHelper
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
)

class EphermisBuilder:

    def __init__(self, window: int = 1024, stride: int = 256):
        self.window = window
        self.stride = stride
        self.helper = BuilderHelper()
        self.commonHelper = CommonHelper()

    def _prepare_ephemeris(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Returns one row per *planet* (not one row per star).
        That way, multi-planet hosts keep all ephemerides.
        """
        df = self.helper.add_star_id(catalog)
        df = df.dropna(subset=["star_id", "mission"])

        # normalize mission casing early
        df["mission"] = df["mission"].str.lower()

        def get_period(row):
            m = row["mission"]
            if m == "tess" and not pd.isna(row.get("pl_orbper")):
                return float(row["pl_orbper"])
            if m == "kepler" and not pd.isna(row.get("koi_period")):
                return float(row["koi_period"])
            if m == "k2" and not pd.isna(row.get("koi_period")):
                return float(row["koi_period"])
            return np.nan

        def get_t0_mission(row):
            m = row["mission"]
            # TESS: pl_tranmid is BJD_TDB; lightkurve uses BTJD = BJD - 2457000
            if m == "tess" and not pd.isna(row.get("pl_tranmid")):
                return float(row["pl_tranmid"]) - 2457000.0
            # Kepler/K2: koi_time0bk is BKJD (already aligned with LK Kepler/K2 time)
            if m == "kepler" and not pd.isna(row.get("koi_time0bk")):
                return float(row["koi_time0bk"])
            if m == "k2" and not pd.isna(row.get("koi_time0bk")):
                return float(row["koi_time0bk"])
            return np.nan

        def get_duration_days(row):
            m = row["mission"]
            # TESS: pl_trandurh in hours
            if m == "tess" and not pd.isna(row.get("pl_trandurh")):
                return float(row["pl_trandurh"]) / 24.0
            # Kepler/K2: koi_duration in hours
            if m in ("kepler", "k2") and not pd.isna(row.get("koi_duration")):
                return float(row["koi_duration"]) / 24.0
            return np.nan

        df["period_days"] = df.apply(get_period, axis=1)
        df["t0_mission"] = df.apply(get_t0_mission, axis=1)
        df["duration_days"] = df.apply(get_duration_days, axis=1)

        eph = df[["star_id", "mission", "period_days", "t0_mission", "duration_days"]].copy()
        eph = eph.dropna(subset=["period_days", "t0_mission", "duration_days"])

        # sanity filters: remove nonsense ephemerides
        eph = eph[(eph["period_days"] > 0) & (eph["duration_days"] > 0)]
        # remove absurd duty cycles (transit can't take 20%+ of orbit)
        eph = eph[(eph["duration_days"] / eph["period_days"]) < 0.2]

        # IMPORTANT: do NOT drop_duplicates on (star_id, mission) anymore.
        # We only drop exact duplicate planet rows (same ephemeris).
        eph = eph.drop_duplicates(subset=["star_id", "mission", "period_days", "t0_mission", "duration_days"])

        return eph

    def _in_transit_mask(self, time: np.ndarray, period: float, t0: float, duration_days: float) -> np.ndarray:
        """
        time and t0 must be in same time system (BTJD for TESS; BKJD for Kepler/K2).
        """
        if np.isnan(period) or np.isnan(t0) or np.isnan(duration_days):
            return np.zeros_like(time, dtype=bool)
        if period <= 0 or duration_days <= 0:
            return np.zeros_like(time, dtype=bool)
        # safety: reject absurd duty cycle
        if (duration_days / period) >= 0.2:
            return np.zeros_like(time, dtype=bool)

        phase_day = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        return np.abs(phase_day) < (duration_days / 2.0)

    def label_segments_from_catalog(
            self,
            segments_path: str,
            catalog_path: str,
            output_path: str | None = None,
        ) -> pd.DataFrame:
        segments_df = pd.read_parquet(segments_path).copy()
        catalog = pd.read_csv(catalog_path, low_memory=False)

        # normalize
        segments_df["mission"] = segments_df["mission"].str.lower()

        eph = self._prepare_ephemeris(catalog).copy()
        eph["mission"] = eph["mission"].str.lower()

        MIN_IN_TRANSIT_POINTS = int(0.10 * self.window)  # 10% of 1024 = 102

        labeled_rows = []

        # Loop per star+mission, compute mask_any once, then label all its segments
        for (star_id, mission), segs in segments_df.groupby(["star_id", "mission"], sort=False):
            # get all ephemerides (planets) for this star
            planets = eph[(eph["star_id"] == star_id) & (eph["mission"] == mission)]
            if len(planets) == 0:
                # no ephemeris -> all 0
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            # load cached lightcurve for this star
            try:
                time, flux, m_cache, target = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except Exception:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            if time is None or len(time) == 0:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            # OR mask across all planets
            masks = []
            for _, p in planets.drop_duplicates(subset=["period_days","t0_mission","duration_days"]).iterrows():
                period = float(p["period_days"])
                t0 = float(p["t0_mission"])
                dur = float(p["duration_days"])
                mask_p = self._in_transit_mask(time, period, t0, dur)
                if mask_p is not None and mask_p.shape[0] == len(time):
                    masks.append(mask_p)

            if len(masks) == 0:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            mask_any = np.logical_or.reduce(masks)

            # Label each segment by counting in-transit points
            for _, r in segs.iterrows():
                start = int(r["start"])
                end = int(r["end"])

                if start < 0 or end > len(time) or end <= start:
                    label = 0
                else:
                    itc = int(mask_any[start:end].sum())
                    label = 1 if itc >= MIN_IN_TRANSIT_POINTS else 0

                labeled_rows.append((star_id, mission, start, end, label))

        labels_df = pd.DataFrame(labeled_rows, columns=["star_id","mission","start","end","label"])

        # Merge by keys (guaranteed correct alignment)
        out_df = segments_df.merge(labels_df, on=["star_id","mission","start","end"], how="left")
        out_df["label"] = out_df["label"].fillna(0).astype(int)

        if output_path is None:
            output_path = segments_path.replace(".parquet", "_labeled.parquet")

        out_df.to_parquet(output_path, index=False)
        return out_df
