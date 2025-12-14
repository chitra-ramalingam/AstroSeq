import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, models
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from src.Classifiers.isolation_forest import IsolationForestScorer
from src.Classifiers.CnnNNet import CnnNNet
from src.Classifiers.CommonHelper import CommonHelper
from src.Classifiers.Loaders.k2Loader import K2Loader
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
)


class MissionSegmentBuilder:
    def __init__(
        self,
        window: int = 1024,
        mission: str | None = "tess",   # use lowercase internally
        author: str = "SPOC",
        stride: int = 256,
        output_path: str | None = None,
    ):
        self.window = window
        self.mission = mission.lower() if mission is not None else None
        self.stride = stride
        self.author = author
        self.helper = BuilderHelper()
        self.commonHelper = CommonHelper()
        self.k2CsvFilePath = "K2_ephemerides.csv"
        self.output_path = (
            output_path
            if output_path is not None
            else f"segments_all_W{self.window}_S{self.stride}"
            + (f"_{self.mission}" if self.mission is not None else "")
            + ".parquet"
        )

    def read_from_file(self, main_csvfile_path: str) -> pd.DataFrame:
        """Read main catalog, add star_id + mission, dedupe, shuffle, then build segments."""
        
        catalog = pd.read_csv(main_csvfile_path)
        if self.mission == "k2":
            loader = K2Loader()
            catalog = loader.loadfile(self.k2CsvFilePath)

        df = self.helper.add_star_id(catalog)
        df = df.dropna(subset=["star_id", "mission"])
        print(df.columns.tolist())
        print(df.head())

        if self.mission is not None:
            df = df[df["mission"].str.lower() == self.mission]

        df = df.drop_duplicates(subset=["mission", "star_id"])
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        segments_df = self._load_segments_for_mission(df)
        return segments_df

    def _load_segments_for_mission(self, df: pd.DataFrame) -> pd.DataFrame:
        segment_rows: list[dict] = []

        # mission truth source = catalog (df), not cache
        star_to_mission = (
            df.dropna(subset=["star_id", "mission"])
            .assign(mission=lambda x: x["mission"].astype(str).str.strip().str.lower())
            .drop_duplicates(subset=["star_id"])
            .set_index("star_id")["mission"]
            .to_dict()
        )

        for star_id in df["star_id"].unique():
            try:
                time, flux, _cache_mission, target = self.commonHelper.fetch_with_targetId_FromCache(star_id)
            except FileNotFoundError:
                continue
            except Exception:
                continue

            if flux is None or time is None:
                continue

            n_points = len(time)
            segments = self._generate_segments(n_points)

            mission = star_to_mission.get(star_id, self.mission)  # fallback to builder mission
            if mission is None:
                continue

            for start, end in segments:
                segment_rows.append(
                    {
                        "star_id": star_id,
                        "mission": str(mission).lower(),
                        "start": start,
                        "end": end,
                    }
                )

        segments_df = pd.DataFrame(segment_rows)
        segments_df.to_parquet(self.output_path, index=False)
        return segments_df

    def _generate_segments(self, n_points: int) -> list[tuple[int, int]]:
        """Generate (start, end) index pairs for this star."""
        segments: list[tuple[int, int]] = []
        if n_points < self.window:
            return segments

        for start in range(0, n_points - self.window + 1, self.stride):
            end = start + self.window
            segments.append((start, end))
        return segments

    