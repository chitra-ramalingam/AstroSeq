import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from src.Classifiers.Builders.MissionSegmentBuilder import MissionSegmentBuilder
from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.LargeWindow.ModelSplitter import ModelSplitter
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from sklearn.metrics import roc_auc_score, average_precision_score


class LargeWindowCnnModel:
    def __init__(self):
        self.window_size = 1024
        self.helper = BuilderHelper()
        self.stride = 256
        self.catalog_path = "CombinedExoplanetData.csv"

    def build_model(self, mission: str = "tess", do_hard_neg: bool = False):
        mission = mission.strip().lower()

        seg_path = f"segments_all_W1024_S256_{mission}.parquet"
        labeled_path = f"segments_all_W1024_S256_{mission}_labeled.parquet"

        # models
        base_model_path = f"{mission}_window1024_base.keras"     # baseline output (kepler will create this)
        hard_model_path = f"{mission}_window1024_hard.keras"     # optional (only if do_hard_neg=True)

        # hard-neg artifacts (optional)
        hard_neg_path = f"{mission}_hard_neg_top80k.parquet"

        # 1) Build segments + label only if missing
        if not os.path.exists(seg_path):
            missionBuilder = MissionSegmentBuilder(window=self.window_size, mission=mission)
            missionBuilder.read_from_file(self.catalog_path)

        if not os.path.exists(labeled_path):
            eBuilder = EphermisBuilder(window=self.window_size, stride=self.stride)
            labeled_df = eBuilder.label_segments_from_catalog(
                segments_path=seg_path,
                catalog_path=self.catalog_path,
                output_path=labeled_path,
            )
            print("New label counts:\n", labeled_df["label"].value_counts())

        # 2) Split (star-level, no leakage)
        modelSplit = ModelSplitter(self.window_size)
        train_df, val_df, test_df = modelSplit.split_model(
            segments_path=labeled_path,
            mission=mission,
            catalog_path=self.catalog_path,
        )

        for df in (train_df, val_df, test_df):
            df["label"] = df["label"].fillna(0).astype(int)

        print("Train pos frac:", float(train_df["label"].mean()))
        print("Val positive fraction:", float(val_df["label"].mean()))
        print("Test positive fraction:", float(test_df["label"].mean()))

        # 3) BASELINE TRAIN (THIS is what you want for Kepler)
        print(f"\n=== BASELINE TRAIN ({mission}) ===")
        train_gen, val_gen, test_gen = modelSplit.generateSegments(
            train_df, val_df, test_df,
            batch_size=64,
            balance=True,
            neg_pos_ratio=3,        # start with 1:3
            hard_neg_path=None,     # <- OFF (important)
            hard_neg_frac=0.0,      # <- OFF
            mission=mission,
        )

        # Train from scratch for Kepler baseline
        self.fit_and_evaluate(
            train_gen, val_gen, test_gen,
            init_model_path=None,
            out_model_path=base_model_path
        )

        # 4) OPTIONAL: Hard-neg step (leave OFF for Kepler until baseline is decent)
        if not do_hard_neg:
            print(f"[Info] do_hard_neg=False -> skipping hard negative mining for {mission}")
            return

        print(f"\n=== HARD NEGATIVE MINING ({mission}) ===")
        modelSplit.mine_hard_negatives(
            train_df=train_df,
            model_path=base_model_path,
            mission=mission,
            max_neg_pool=300_000,
            topk=80_000,
            max_neg_per_star=200,
            output_path=hard_neg_path,
        )

        print(f"\n=== HARD-NEG FINE-TUNE ({mission}) ===")
        train_gen_hn, val_gen, test_gen = modelSplit.generateSegments(
            train_df, val_df, test_df,
            batch_size=64,
            balance=True,
            neg_pos_ratio=5,        # gentler
            hard_neg_path=hard_neg_path,
            hard_neg_frac=0.10,     # tiny spice, not the meal
            mission=mission,
        )

        self.fit_and_evaluate(
            train_gen_hn, val_gen, test_gen,
            init_model_path=base_model_path,
            out_model_path=hard_model_path
        )


    def fit_and_evaluate(self, train_gen, val_gen, test_gen,
                         init_model_path:str,
                          out_model_path: str):
        if init_model_path:
            model = tf.keras.models.load_model(init_model_path)
            # recompile with a smaller LR for fine-tuning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                        tf.keras.metrics.AUC(curve="PR",  name="pr_auc")],
            )
        else:
            model = self.helper.declareHigherDimModel(self.window_size, channels=2)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=out_model_path,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        )

        callbacks = [
            checkpoint,
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=3,
                min_delta=1e-3,
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=callbacks,
            verbose=1,
        )

        best_model = tf.keras.models.load_model(out_model_path)

        test_metrics = best_model.evaluate(test_gen, verbose=0)
        print("Test metrics:", dict(zip(best_model.metrics_names, test_metrics)))

        y_true, y_pred = [], []
        for Xb, yb in test_gen:
            preds = np.asarray(best_model.predict_on_batch(Xb)).ravel()
            y_true.extend(yb.tolist())
            y_pred.extend(preds.tolist())

        print("Segment ROC AUC:", roc_auc_score(y_true, y_pred))
        print("Segment PR AUC:", average_precision_score(y_true, y_pred))

        return history

    def checkSavedModel(self, savedModelPath: str, test_gen):
        m = tf.keras.models.load_model(savedModelPath)
        print("Saved Loaded:", m.name)
        print("Saved Keras file Input shape:", m.input_shape)
        print("Output shape:", m.output_shape)
        print("Num params:", m.count_params())

        y_true, y_pred = [], []
        for Xb, yb in test_gen:
            preds = np.asarray(m.predict_on_batch(Xb)).ravel()
            y_true.extend(yb.tolist())
            y_pred.extend(preds.tolist())

        print("Saved ROC:", roc_auc_score(y_true, y_pred))
        print("Saved PR :", average_precision_score(y_true, y_pred))
