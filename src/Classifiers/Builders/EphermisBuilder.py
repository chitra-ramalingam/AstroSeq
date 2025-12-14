import pandas as pd
import numpy as np
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from src.Classifiers.CommonHelper import CommonHelper


class EphermisBuilder:
    KEPLER_BJD0 = 2454833.0  # BKJD = BJD - 2454833.0
    TESS_BJD0   = 2457000.0  # BTJD = BJD - 2457000.0

    def __init__(self, window: int = 1024, stride: int = 256):
        self.window = window
        self.stride = stride
        self.helper = BuilderHelper()
        self.commonHelper = CommonHelper()

    # ---------- small normalizers ----------

    def _normalize_star_id(self, s) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.nan
        s = str(s).strip()
        # Normalize "EPIC 123" / "EPIC-123" -> "EPIC_123" etc.
        s = pd.Series([s]).str.replace(r"^(KIC|KOI|EPIC|TIC|TOI)[\s\-]+", r"\1_", regex=True).iloc[0]
        s = pd.Series([s]).str.replace(r"__+", "_", regex=True).iloc[0]
        return s

    def _mission_from_star_id(self, star_id: str) -> str:
        if not isinstance(star_id, str) or not star_id:
            return np.nan
        u = star_id.upper()
        if u.startswith("EPIC_") or u.startswith("EPIC"):
            return "k2"
        if u.startswith("KIC_") or u.startswith("KOI_") or u.startswith("KOI"):
            return "kepler"
        if u.startswith("TIC_") or u.startswith("TOI_") or u.startswith("TIC") or u.startswith("TOI"):
            return "tess"
        return np.nan

    # ---------- ephemeris prep ----------

    def _prepare_ephemeris(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Returns one row per *planet* (not one row per star).
        Keeps multi-planet hosts.
        """
        df = self.helper.add_star_id(catalog)
        df = df.dropna(subset=["star_id"]).copy()

        # Normalize IDs + mission early
        df["star_id"] = df["star_id"].map(self._normalize_star_id)

        if "mission" not in df.columns:
            df["mission"] = np.nan
        df["mission"] = df["mission"].astype(str).str.strip().str.lower()
        df.loc[df["mission"].isin(["nan", "none", ""]), "mission"] = np.nan

        # If mission missing or nonsense, infer from star_id prefix
        df.loc[df["mission"].isna(), "mission"] = df.loc[df["mission"].isna(), "star_id"].map(self._mission_from_star_id)

        # If mission contradicts star_id prefix (your EPIC-as-TESS bug), fix it
        inferred = df["star_id"].map(self._mission_from_star_id)
        bad = inferred.notna() & df["mission"].notna() & (df["mission"] != inferred)
        df.loc[bad, "mission"] = inferred[bad]

        df = df.dropna(subset=["mission"])

        def get_period(row):
            m = row["mission"]

            # Prefer unified NASA columns when present
            if not pd.isna(row.get("pl_orbper")):
                return float(row["pl_orbper"])

            # Legacy Kepler / KOI catalog columns
            if m in ("kepler", "k2") and not pd.isna(row.get("koi_period")):
                return float(row["koi_period"])

            return np.nan

        def get_t0_mission(row):
            m = row["mission"]

            # TESS: pl_tranmid is BJD -> store as BTJD
            if m == "tess" and not pd.isna(row.get("pl_tranmid")):
                return float(row["pl_tranmid"]) - self.TESS_BJD0

            # Kepler/K2: if koi_time0bk exists, it's already BKJD
            if m in ("kepler", "k2") and not pd.isna(row.get("koi_time0bk")):
                return float(row["koi_time0bk"])

            # K2 from Exoplanet Archive (pscomppars join) often gives pl_tranmid (BJD)
            # Store as BKJD so it matches LK Kepler/K2 reduced time
            if m == "k2" and not pd.isna(row.get("pl_tranmid")):
                return float(row["pl_tranmid"]) - self.KEPLER_BJD0

            # Kepler confirmed planets may also have pl_tranmid
            if m == "kepler" and not pd.isna(row.get("pl_tranmid")):
                return float(row["pl_tranmid"]) - self.KEPLER_BJD0

            return np.nan

        def get_duration_days(row):
            m = row["mission"]

            # TESS unified column: pl_trandurh hours
            if m == "tess" and not pd.isna(row.get("pl_trandurh")):
                return float(row["pl_trandurh"]) / 24.0

            # Exoplanet Archive ps/pscomppars uses pl_trandur in hours (no trailing h)
            if m == "tess" and not pd.isna(row.get("pl_trandur")):
                return float(row["pl_trandur"]) / 24.0

            # Kepler/K2 KOI duration in hours
            if m in ("kepler", "k2") and not pd.isna(row.get("koi_duration")):
                return float(row["koi_duration"]) / 24.0

            # K2 (and sometimes Kepler) confirmed planets: pl_trandur(h) hours
            if m in ("kepler", "k2") and not pd.isna(row.get("pl_trandurh")):
                return float(row["pl_trandurh"]) / 24.0
            if m in ("kepler", "k2") and not pd.isna(row.get("pl_trandur")):
                return float(row["pl_trandur"]) / 24.0

            return np.nan

        df["period_days"] = df.apply(get_period, axis=1)
        df["t0_mission"] = df.apply(get_t0_mission, axis=1)
        df["duration_days"] = df.apply(get_duration_days, axis=1)

        eph = df[["star_id", "mission", "period_days", "t0_mission", "duration_days"]].copy()
        eph = eph.dropna(subset=["period_days", "t0_mission", "duration_days"])

        # sanity filters
        eph = eph[(eph["period_days"] > 0) & (eph["duration_days"] > 0)]
        eph = eph[(eph["duration_days"] / eph["period_days"]) < 0.2]

        eph = eph.drop_duplicates(subset=["star_id", "mission", "period_days", "t0_mission", "duration_days"])
        return eph

    # ---------- transit mask + labeling ----------

    def _in_transit_mask(self, time: np.ndarray, period: float, t0: float, duration_days: float) -> np.ndarray:
        if np.isnan(period) or np.isnan(t0) or np.isnan(duration_days):
            return np.zeros_like(time, dtype=bool)
        if period <= 0 or duration_days <= 0:
            return np.zeros_like(time, dtype=bool)
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

        # Normalize star_id + mission in segments (this fixes EPIC_* incorrectly tagged as TESS)
        segments_df["star_id"] = segments_df["star_id"].map(self._normalize_star_id)
        if "mission" not in segments_df.columns:
            segments_df["mission"] = np.nan
        segments_df["mission"] = segments_df["mission"].astype(str).str.strip().str.lower()
        segments_df.loc[segments_df["mission"].isin(["nan", "none", ""]), "mission"] = np.nan

        inferred_seg = segments_df["star_id"].map(self._mission_from_star_id)
        # Fill missing mission, and also override contradictory mission
        segments_df.loc[segments_df["mission"].isna(), "mission"] = inferred_seg[segments_df["mission"].isna()]
        bad = inferred_seg.notna() & segments_df["mission"].notna() & (segments_df["mission"] != inferred_seg)
        segments_df.loc[bad, "mission"] = inferred_seg[bad]

        eph = self._prepare_ephemeris(catalog).copy()
        eph["star_id"] = eph["star_id"].map(self._normalize_star_id)
        eph["mission"] = eph["mission"].astype(str).str.strip().str.lower()

        MIN_IN_TRANSIT_POINTS = int(0.10 * self.window)

        labeled_rows: list[tuple[str, str, int, int, int]] = []

        for (star_id, mission), segs in segments_df.groupby(["star_id", "mission"], sort=False):
            planets = eph[(eph["star_id"] == star_id) & (eph["mission"] == mission)]
            if len(planets) == 0:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            try:
                time, flux, _m_cache, _target = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except Exception:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            if time is None or len(time) == 0:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            time = np.asarray(time)
            time_base = self._guess_time_base(time)

            masks = []
            for _, p in planets.drop_duplicates(subset=["period_days", "t0_mission", "duration_days"]).iterrows():
                period = float(p["period_days"])
                t0_mission = float(p["t0_mission"])
                dur = float(p["duration_days"])

                t0 = self._convert_t0_to_timebase(mission=mission, t0_mission=t0_mission, time_base=time_base)
                mask_p = self._in_transit_mask(time, period, t0, dur)
                if mask_p is not None and mask_p.shape[0] == len(time):
                    masks.append(mask_p)

            if len(masks) == 0:
                for _, r in segs.iterrows():
                    labeled_rows.append((star_id, mission, int(r["start"]), int(r["end"]), 0))
                continue

            mask_any = np.logical_or.reduce(masks)

            for _, r in segs.iterrows():
                start = int(r["start"])
                end = int(r["end"])
                if start < 0 or end > len(time) or end <= start:
                    label = 0
                else:
                    itc = int(mask_any[start:end].sum())
                    label = 1 if itc >= MIN_IN_TRANSIT_POINTS else 0

                labeled_rows.append((star_id, mission, start, end, label))

        labels_df = pd.DataFrame(labeled_rows, columns=["star_id", "mission", "start", "end", "label"])
        out_df = segments_df.merge(labels_df, on=["star_id", "mission", "start", "end"], how="left")
        out_df["label"] = out_df["label"].fillna(0).astype(int)

        if output_path is None:
            output_path = segments_path.replace(".parquet", "_labeled.parquet")

        out_df.to_parquet(output_path, index=False)
        return out_df

    def _guess_time_base(self, time: np.ndarray) -> str:
        tmed = float(np.nanmedian(time))
        if tmed > 2_000_000:
            return "BJD"
        if 10_000 < tmed < 200_000:
            return "BJD_OFFSET_UNKNOWN"
        return "REDUCED"

    def _convert_t0_to_timebase(self, mission: str, t0_mission: float, time_base: str) -> float:
        mission = (mission or "").strip().lower()
        if np.isnan(t0_mission):
            return np.nan

        if time_base == "REDUCED":
            # We store: TESS=BTJD, Kepler/K2=BKJD
            return float(t0_mission)

        if time_base == "BJD":
            # Need full BJD
            if mission == "tess":
                return float(t0_mission) + self.TESS_BJD0
            if mission in ("kepler", "k2"):
                return float(t0_mission) + self.KEPLER_BJD0

        return float(t0_mission)
