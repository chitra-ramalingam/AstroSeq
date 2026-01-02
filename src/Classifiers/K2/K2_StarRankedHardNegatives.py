import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

class K2_StarRankedHardNegatives:
    """
    Identify hard negative segments based on star-ranked model predictions.
    """

    def __init__(self) -> None:
        pass

    def mine_hard_negatives(self) -> None:
        """
        Identify hard negative segments from training data based on star-ranked scores.
        Saves the mined hard negatives to disk.
        """
        # Load v2 model (the one that still "reacts" strongly)
        model = tf.keras.models.load_model("k2_window1024_centralized_v2.keras")

        meta_train = pd.read_parquet("k2_dataset_centered_v2/meta_train.parquet").reset_index(drop=True)
        X_train = np.load("k2_dataset_centered_v2/X_train.npy")

        p_train = model.predict(X_train, batch_size=256, verbose=1).reshape(-1)
        meta_train["p"] = p_train

        # Star score in TRAIN (top-10 mean)
        star_df = meta_train.groupby("star_id").agg(
            y_star=("label_star", "max"),
            score_top10=("p", lambda s: float(np.mean(np.sort(s)[-10:]))),
            nseg=("p","size")
        ).reset_index()

        # Pick the worst offenders: negative stars ranked highest
        N_STARS = 500
        bad_stars = (star_df[star_df["y_star"] == 0]
                    .sort_values("score_top10", ascending=False)
                    .head(N_STARS)["star_id"].tolist())

        # From those stars, take highest-scoring negative segments
        N_SEGS = 20000  # tune 5k–50k
        hn = (meta_train[(meta_train["star_id"].isin(bad_stars)) & (meta_train["label"] == 0)]
            .sort_values("p", ascending=False)
            .head(N_SEGS))

        hn_idx = hn.index.to_numpy()
        X_hn = X_train[hn_idx]
        y_hn = np.zeros(len(hn_idx), dtype=np.int32)

        np.save("k2_dataset_centered_v2/X_hardneg_star.npy", X_hn)
        hn.drop(columns=["p"]).to_parquet("k2_dataset_centered_v2/meta_hardneg_star.parquet", index=False)

        print("Saved star-hardneg bank:", X_hn.shape, "from", len(bad_stars), "stars")
        print("Top bad stars (train):")
        print(star_df[star_df["star_id"].isin(bad_stars)].head(10))


        #self.create_model()
        #self.displayModel()

    def displayModel(self):
        
        # ---- paths ----
        DATA_DIR = "k2_dataset_centered_v2"
        MODEL_PATH = "k2_window1024_v4_starHN_W3.keras"

        # ---- load test set ----
        meta_test = pd.read_parquet(f"{DATA_DIR}/meta_test.parquet").reset_index(drop=True)
        X_test = np.load(f"{DATA_DIR}/X_test.npy")
        y_seg = meta_test["label"].to_numpy().astype(int)

        # ---- load model v4 + predict ----
        model = tf.keras.models.load_model(MODEL_PATH)
        p = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
        meta_test["p"] = p

        # ---- 1) segment metrics ----
        print("Segment PR-AUC:", average_precision_score(y_seg, p))
        print("Segment ROC-AUC:", roc_auc_score(y_seg, p))

        # ---- 2) star aggregation ----
        star_y = meta_test.groupby("star_id")["label_star"].max().astype(int)
        star_max = meta_test.groupby("star_id")["p"].max()
        star_top10 = meta_test.groupby("star_id")["p"].apply(lambda s: float(np.mean(np.sort(s)[-10:])))

        star_df = pd.DataFrame({
            "star_id": star_y.index,
            "y_star": star_y.values,
            "score_max": star_max.values,
            "score_top10mean": star_top10.values,
        })

        print("Star PR-AUC (max):", average_precision_score(star_df["y_star"], star_df["score_max"]))
        print("Star PR-AUC (top10mean):", average_precision_score(star_df["y_star"], star_df["score_top10mean"]))

        # ---- 3) precision@K ----
        def precision_at_k(df, score_col, k):
            return df.nlargest(k, score_col)["y_star"].mean()

        base = float(star_df["y_star"].mean())
        for k in [50, 100, 200, 500, 1000]:
            pk = float(precision_at_k(star_df, "score_top10mean", k))
            print(k, "precision", round(pk, 3), "lift", round(pk / base, 2))

        # ---- 4) top stars table ----
        print(star_df.sort_values("score_top10mean", ascending=False).head(20))

        # ---- save outputs (optional but recommended) ----
        star_df.to_csv("k2_v4_star_scores.csv", index=False)
        meta_test.to_parquet("k2_v4_meta_test_with_preds.parquet", index=False)
        print("Saved: k2_v4_star_scores.csv, k2_v4_meta_test_with_preds.parquet")


    def create_model(self):

        X_train = np.load("k2_dataset_centered_v2/X_train.npy")
        y_train = pd.read_parquet("k2_dataset_centered_v2/meta_train.parquet")["label"].to_numpy().astype(np.int32)

        X_val = np.load("k2_dataset_centered_v2/X_val.npy")
        y_val = pd.read_parquet("k2_dataset_centered_v2/meta_val.parquet")["label"].to_numpy().astype(np.int32)

        X_hn = np.load("k2_dataset_centered_v2/X_hardneg_star.npy")
        y_hn = np.zeros(len(X_hn), dtype=np.int32)

        X_aug = np.concatenate([X_train, X_hn], axis=0)
        y_aug = np.concatenate([y_train, y_hn], axis=0)

        w = np.ones(len(X_aug), dtype=np.float32)
        w[len(X_train):] = 3.0   # start gentle

        model = tf.keras.models.load_model("k2_dataset_centered_v2.keras", compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ],
        )

        model.fit(
            X_aug, y_aug,
            sample_weight=w,
            batch_size=256,
            epochs=3,
            validation_data=(X_val, y_val),
            verbose=1
        )
        model.save("k2_window1024_v4_starHN_W3.keras")
        print("Saved k2_window1024_v4_starHN_W3.keras")
