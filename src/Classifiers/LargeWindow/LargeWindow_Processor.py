import numpy as np
import pandas as pd
import tensorflow as tf
from src.Classifiers.Builders.MissionSegmentBuilder import MissionSegmentBuilder
from src.Classifiers.Builders.EphermisBuilder import EphermisBuilder
from src.Classifiers.LargeWindow.ModelSplitter import ModelSplitter
from src.Classifiers.Builders.BuilderHelper import BuilderHelper
from sklearn.metrics import roc_auc_score, average_precision_score
from src.Classifiers.CommonHelper import CommonHelper
import matplotlib.pyplot as plt

class LargeWindowCnnModel:
    def __init__(self):
        self.window_size = 1024
        self.helper = BuilderHelper()
        self.stride = 256
        self.catalog_path = "CombinedExoplanetData.csv"

    def build_model(self):
        missionBuilder = MissionSegmentBuilder(window=1024, mission="tess")
        eBuilder = EphermisBuilder(window=1024, stride=256)
        segments_df = missionBuilder.read_from_file(self.catalog_path)
        # this writes something like "segments_all_W1024_S256_tess.parquet"

        # 2) label them using the same catalog
        labeled_df = eBuilder.label_segments_from_catalog(
            segments_path="segments_all_W1024_S256_tess.parquet",
            catalog_path=self.catalog_path,
            output_path="segments_all_W1024_S256_tess_labeled.parquet",
        )

        print(labeled_df.columns)
        print(labeled_df.head(5))
        print("New label counts:\n", labeled_df["label"].value_counts())


        modelSplit = ModelSplitter(self.window_size)
        train_df, val_df, test_df = modelSplit.split_model(
            segments_path="segments_all_W1024_S256_tess_labeled.parquet",
            catalog_path=self.catalog_path,
        )   
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            print(name, "unique labels:", sorted(df["label"].dropna().unique()))
      
        for df in (train_df, val_df, test_df):
            df["label"] = df["label"].fillna(0).astype(int)

        train_gen, val_gen, test_gen = modelSplit.generateSegments(train_df, val_df, test_df)   
        Xb, yb = next(iter(train_gen))
        print("Xb shape:", Xb.shape)              # expect (B, 1024, 2)
        print("yb shape:", yb.shape)              # expect (B,)
        print("yb unique:", np.unique(yb))
        print("batch pos fraction:", yb.mean())
        print("any NaN in X?", np.isnan(Xb).any())

        print("Train pos frac:", train_df["label"].mean())
        val_pos_frac = val_df["label"].mean()
        print("Val positive fraction:", val_pos_frac)
        print("Test positive fraction:", test_df["label"].mean())

        # model = self.helper.declareHigherDimModel(self.window_size ,channels=2)
        # preds = model.predict(Xb, verbose=0).ravel()

        # print("yb[:20]:", yb[:20])
        # print("preds[:20]:", np.round(preds[:20], 4))
        # print("preds min/max/mean:", preds.min(), preds.max(), preds.mean())
        # # 1) Fresh model
       
        
        # 3) Check predictions again
        #preds_toy = model_toy.predict(Xb, verbose=0).ravel()

        #print("yb:", yb[:20])
        #print("preds_toy:", np.round(preds_toy[:20], 3))
        #print("preds_toy min/max/mean:", preds_toy.min(), preds_toy.max(), preds_toy.mean())

        #print("Toy ROC AUC:", roc_auc_score(yb, preds_toy))
        #print("Toy PR AUC:", average_precision_score(yb, preds_toy))
        #self.plot_toy(Xb, yb)
        self.fit_and_evaluate(train_gen, val_gen, test_gen)

    def fit_and_evaluate(self, train_gen, val_gen, test_gen):
        model = self.helper.declareHigherDimModel(self.window_size ,channels=2)
        class_weight = {0: 1.0, 1: 1.0}  # if 50/50 sampling

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="tess_window1024.keras",
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
                min_delta=1e-3,       # require +0.001 improvement
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=callbacks,
            class_weight=class_weight
        )
        test_metrics = model.evaluate(test_gen)
        print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))

        y_true = []
        y_pred = []
        star_ids = []

        for X_batch, y_batch in test_gen:
            preds = model.predict(X_batch, verbose=0).ravel()
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())

        # segment-level ROC AUC

        print("Segment ROC AUC:", roc_auc_score(y_true, y_pred))
        print("Segment PR AUC:", average_precision_score(y_true, y_pred))

        #model.save(".keras")



    def check_val(self,val_df, model):
        # 1) grab one star from val that *has* positives
        val_with_pos = val_df.groupby("star_id")["label"].max()
        star_id = val_with_pos[val_with_pos > 0].index[0]

        star_segments = val_df[val_df["star_id"] == star_id].sort_values("start")

        ch = CommonHelper()

        time, flux, mission, target = ch.fetch_with_targetId_FromCache(star_id)

        # 2) build all segment arrays for that star
        import numpy as np

        X_list = []
        for _, row in star_segments.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            X_list.append(flux[start:end, :])  # (1024, 2)

        X_star = np.stack(X_list, axis=0)
        y_star = star_segments["label"].values

        # 3) predict
        preds_star = model.predict(X_star, verbose=0).ravel()

        print("star:", star_id)
        print("labels:", y_star[:30])
        print("preds:",  np.round(preds_star[:30], 3))
        print("preds min/max/mean:", preds_star.min(), preds_star.max(), preds_star.mean())

    def plot_toy(self, Xb, yb):

        idx_pos = np.where(yb == 1)[0][0]
        seg_pos = Xb[idx_pos]        # (1024, 2)
        flat = seg_pos[:, 1]         # flattened channel

        plt.plot(flat)
        plt.gca().invert_yaxis()
        plt.title("Example positive segment (flattened flux)")
        plt.show()
