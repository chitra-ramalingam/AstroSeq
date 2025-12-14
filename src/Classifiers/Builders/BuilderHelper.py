import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

class BuilderHelper:
    def __init__(self):
        pass

    def add_star_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- 1) Prefer existing mission if present ---
        if "mission" in df.columns:
            df["mission"] = df["mission"].astype(str).str.strip().str.lower()
            df.loc[df["mission"].isin(["nan", "none", ""]), "mission"] = np.nan
        else:
            df["mission"] = np.nan

        # --- 2) Prefer existing star_id if present (normalize it) ---
        if "star_id" in df.columns:
            df["star_id"] = df["star_id"].astype(str).str.strip()
            df.loc[df["star_id"].isin(["nan", "None", ""]), "star_id"] = np.nan

            # normalize common formats: "EPIC 123" -> "EPIC_123", "TIC 123" -> "TIC_123", etc.
            df["star_id"] = (
                df["star_id"]
                .str.replace(r"^(KIC|KOI|EPIC|TIC|TOI)[\s\-]+", r"\1_", regex=True)
                .str.replace(r"__+", "_", regex=True)
            )
        else:
            df["star_id"] = np.nan

        # --- 3) If mission missing, infer it from available IDs OR star_id prefix ---
        def infer_mission(row):
            m = row.get("mission")
            if isinstance(m, str) and m.strip():
                return m.strip().lower()

            sid = row.get("star_id")
            if isinstance(sid, str):
                s = sid.upper()
                if s.startswith("KIC_") or s.startswith("KOI_") or s.startswith("KOI"):
                    return "kepler"
                if s.startswith("EPIC_") or s.startswith("EPIC"):
                    return "k2"
                if s.startswith("TIC_") or s.startswith("TOI_") or s.startswith("TIC") or s.startswith("TOI"):
                    return "tess"

            # fallback: infer from numeric ID columns (your original logic)
            if not pd.isna(row.get("kepid")) or not pd.isna(row.get("koi_name")) or not pd.isna(row.get("koi")):
                return "kepler"
            if not pd.isna(row.get("epic")):
                return "k2"
            if not pd.isna(row.get("tid")) or not pd.isna(row.get("toi")):
                return "tess"
            return np.nan

        df["mission"] = df.apply(infer_mission, axis=1)

        # --- 4) If star_id missing, build it using mission + known columns ---
        def canon_row(row):
            # if already present, keep it
            sid = row.get("star_id")
            if isinstance(sid, str) and sid.strip() and sid.lower() not in ("nan", "none"):
                return sid.strip()

            m = row.get("mission")

            if m == "kepler":
                if not pd.isna(row.get("kepid")):
                    return f"KIC_{int(row['kepid'])}"
                if not pd.isna(row.get("koi_name")):
                    return str(row["koi_name"]).strip()
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

            # last fallback
            if not pd.isna(row.get("hostname")):
                return str(row["hostname"]).strip()

            return np.nan

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
