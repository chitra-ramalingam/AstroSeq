import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

class BuilderHelper:
    def __init__(self):
        pass
    
    def add_star_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        def infer_mission(row):
            # Order matters: check which ID columns are actually populated
            if not pd.isna(row.get("kepid")) or not pd.isna(row.get("koi_name")) or not pd.isna(row.get("koi")):
                return "kepler"
            if not pd.isna(row.get("epic")):
                return "k2"
            if not pd.isna(row.get("tid")) or not pd.isna(row.get("toi")):
                return "tess"
            return None

        df["mission"] = df.apply(infer_mission, axis=1)

        def canon_row(row):
            m = row["mission"]
            if m == "kepler":
                if not pd.isna(row.get("kepid")):
                    return f"KIC_{int(row['kepid'])}"
                if not pd.isna(row.get("koi_name")):
                    return row["koi_name"]
                if not pd.isna(row.get("koi")):
                    return f"KOI_{int(row['koi'])}"
            if m == "k2":
                if not pd.isna(row.get("epic")):
                    return f"EPIC_{int(row['epic'])}"
            if m == "tess":
                if not pd.isna(row.get("tid")):
                    return f"TIC_{int(row['tid'])}"
                if not pd.isna(row.get("toi")):
                    return f"TOI_{int(row['toi'])}"
            # fallback
            if not pd.isna(row.get("hostname")):
                return row["hostname"]
            return None

        df["star_id"] = df.apply(canon_row, axis=1)
        return df
    
    def declareHigherDimModel(self,w, channels=1):

        inputs = layers.Input(shape=(w, channels))

        x = layers.Conv1D(64, 11, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 7, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 5, padding='same', activation='relu', dilation_rate=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 3, padding='same', activation='relu', dilation_rate=4)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                ],
            )
        return model


    def declareModel(self,window, channels=1):
            w = window
            inputs = layers.Input(shape=(w, channels))
            x = layers.Conv1D(32, 11, padding='same', activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 5, padding='same', activation='relu', dilation_rate=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 3, padding='same', activation='relu', dilation_rate=4)(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(3e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                ],
            )
            return model
