import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from src.Classifiers.CommonHelper import CommonHelper


class SegmentDataset(Sequence):
    def __init__(
        self,
        df,
        batch_size=64,
        shuffle=True,
        window=1024,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window = window
        self.commonHelper = CommonHelper()
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_rows = self.df.iloc[batch_idx]

        X_list = []
        y_list = []
        flux_cache = {}

        for _, row in batch_rows.iterrows():
            star_id = row["star_id"]
            start = int(row["start"])
            end = int(row["end"])
            label = float(row["label"])

            if star_id not in flux_cache:
                time, flux, mission, target = self.commonHelper.fetch_with_targetId_FromCache(star_id)
                flux_cache[star_id] = flux  # (T, 2)

            flux = flux_cache[star_id]

            if end > flux.shape[0]:
                continue

            seg_flux = flux[start:end, :]  # (window, 2)

            # --- CRITICAL CLEANUP ---
            # Replace NaN/Inf with 0 and optionally clip extremes
            seg_flux = np.nan_to_num(seg_flux, nan=0.0, posinf=0.0, neginf=0.0)
            seg_flux = np.clip(seg_flux, -10.0, 10.0)

            if seg_flux.shape[0] != self.window:
                continue

            X_list.append(seg_flux)
            y_list.append(label)

        X = np.stack(X_list, axis=0).astype("float32")
        y = np.array(y_list, dtype="float32")

        # Extra safety assertions (optional but useful while debugging)
        if not np.isfinite(X).all():
            raise ValueError("Found non-finite values in X batch")
        if not np.isfinite(y).all():
            raise ValueError("Found non-finite values in y batch")

        return X, y
