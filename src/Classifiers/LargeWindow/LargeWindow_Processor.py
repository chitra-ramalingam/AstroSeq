import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src.Classifiers.Builders.MissionSegmentBuilder import MissionSegmentBuilder
from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.LargeWindow.ModelSplitter import ModelSplitter
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from sklearn.metrics import roc_auc_score, average_precision_score
from src.Classifiers.LargeWindow.SmallMissionSplitter import SmallDataSetMissionSplitter


class LargeWindowCnnModel:
    def __init__(self):
        self.window_size = 1024
        self.helper = BuilderHelper()
        self.stride = 256
        self.catalog_path = "CombinedExoplanetData.csv"
        self.k2CsvFilePath = "K2_ephemerides.csv"

    def build_model(
        self,
        mission: str = "tess",
        neg_pos_ratio: int = 3,
        do_hard_neg: bool = False,
    ):
        mission = mission.strip().lower()

        seg_path = f"segments_all_W1024_S256_{mission}.parquet"
        labeled_path = f"segments_all_W1024_S256_{mission}_labeled.parquet"

        base_model_path = f"{mission}_window1024_base.keras"
        hard_model_path = f"{mission}_window1024_hard.keras"

        hard_neg_path = f"{mission}_hard_neg_top80k.parquet"

        catpath = self.k2CsvFilePath if mission == "k2" else self.catalog_path

        # ---- K2 small-dataset defaults (only if user didn't override) ----
        if mission == "k2":
            # K2 usually benefits from a gentler neg:pos because star count is tiny
            neg_pos_ratio = min(neg_pos_ratio, 2)

        # 1) Build segments + label only if missing
        if not os.path.exists(seg_path):
            missionBuilder = MissionSegmentBuilder(window=self.window_size, mission=mission)
            missionBuilder.read_from_file(catpath)

        if not os.path.exists(labeled_path):
            eBuilder = EphermisBuilder(window=self.window_size, stride=self.stride)
            labeled_df = eBuilder.label_segments_from_catalog(
                segments_path=seg_path,
                catalog_path=catpath,
                output_path=labeled_path,
            )
            print("New label counts:\n", labeled_df["label"].value_counts())

        # 2) Split (star-level, no leakage)
        if mission == "k2":
            splitter = SmallDataSetMissionSplitter(self.window_size)
        else:
            splitter = ModelSplitter(self.window_size)

        train_df, val_df, test_df = splitter.split_model(
            segments_path=labeled_path,
            mission=mission,
            catalog_path=catpath,
        )

        for df in (train_df, val_df, test_df):
            df["label"] = df["label"].fillna(0).astype(int)

        print("Train pos frac:", float(train_df["label"].mean()))
        print("Val positive fraction:", float(val_df["label"].mean()))
        print("Test positive fraction:", float(test_df["label"].mean()))

        # 3) BASELINE TRAIN
        print(f"\n=== BASELINE TRAIN ({mission}) ===")

        # K2 tweak: smaller batch = more gradient steps, less “all-one-star” batches
        batch_size = 32 if mission == "k2" else 64

        train_gen, val_gen, test_gen = splitter.generateSegments(
            train_df, val_df, test_df,
            batch_size=batch_size,
            balance=True,
            neg_pos_ratio=neg_pos_ratio,
            hard_neg_path=None,
            hard_neg_frac=0.0,
            mission=mission,
        )

        # For K2, PR-AUC on val can be wildly unstable when val has few positive stars.
        # ROC-AUC is usually more stable for early stopping on tiny datasets.
        if mission == "k2":
            monitor = "val_roc_auc"
            mode = "max"
            patience = 6
            epochs = 25
            lr = 3e-4
        else:
            monitor = "val_pr_auc"
            mode = "max"
            patience = 3
            epochs = 15
            lr = 3e-4

        self.fit_and_evaluate(
            train_gen, val_gen, test_gen,
            init_model_path=None,
            out_model_path=base_model_path,
            monitor=monitor,
            mode=mode,
            patience=patience,
            epochs=epochs,
            lr=lr,
        )

        # 4) OPTIONAL: Hard-neg step
        if not do_hard_neg:
            print(f"[Info] do_hard_neg=False -> skipping hard negative mining for {mission}")
            return

        print(f"\n=== HARD NEGATIVE MINING ({mission}) ===")
        splitter.mine_hard_negatives(
            train_df=train_df,
            model_path=base_model_path,
            mission=mission,
            max_neg_pool=300_000,
            topk=80_000,
            max_neg_per_star=200,
            output_path=hard_neg_path,
        )

        print(f"\n=== HARD-NEG FINE-TUNE ({mission}) ===")
        train_gen_hn, val_gen, test_gen = splitter.generateSegments(
            train_df, val_df, test_df,
            batch_size=batch_size,
            balance=True,
            neg_pos_ratio=5,
            hard_neg_path=hard_neg_path,
            hard_neg_frac=0.10,
            mission=mission,
        )

        # Fine-tune with smaller LR
        self.fit_and_evaluate(
            train_gen_hn, val_gen, test_gen,
            init_model_path=base_model_path,
            out_model_path=hard_model_path,
            monitor=monitor,
            mode=mode,
            patience=patience,
            epochs=epochs,
            lr=1e-4,
        )
        print("TEST label counts:\n", test_df["label"].value_counts(dropna=False))
        print("TEST pos frac:", float(test_df["label"].mean()))


    def fit_and_evaluate(
        self,
        train_gen, val_gen, test_gen,
        init_model_path: str | None,
        out_model_path: str,
        monitor: str = "val_pr_auc",
        mode: str = "max",
        patience: int = 3,
        epochs: int = 15,
        lr: float = 3e-4,
    ):
        if init_model_path:
            model = tf.keras.models.load_model(init_model_path)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                ],
            )
        else:
            model = self.helper.declareHigherDimModel(self.window_size, channels=2)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                ],
            )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=out_model_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
        )

        callbacks = [
            checkpoint,
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                factor=0.5,
                patience=max(1, patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                min_delta=1e-3,
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
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

        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(float)

        # If test set has only one class, roc_auc_score will crash—guard it.
        if len(np.unique(y_true)) > 1:
            print("Segment ROC AUC:", roc_auc_score(y_true, y_pred))
        else:
            print("Segment ROC AUC: undefined (only one class in test labels)")

        print("Segment PR AUC:", average_precision_score(y_true, y_pred))
        y_true = np.asarray(y_true).astype(int)
        print("TEST (generator) label counts:", np.bincount(y_true, minlength=2))
        print("TEST baseline PR (pos frac):", y_true.mean())
        return history
