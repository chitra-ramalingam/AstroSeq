from xml.parsers.expat import model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from src.Classifiers.K2.K2_StarRankedHardNegatives import K2_StarRankedHardNegatives


class K2_PrintSets:
    def __init__(self):
        pass

    def print_meta_test(self):
        meta_test = pd.read_parquet("k2_dataset_v2/meta_test.parquet")
        print("meta_test columns:", meta_test.columns.tolist())

        # Find the label column (common names)
        for c in ["y", "label", "is_transit", "target", "y_seg"]:
            if c in meta_test.columns:
                ycol = c
                break
        else:
            raise ValueError("Can't find label column. Show meta_test.columns and I’ll point it out.")

        y = meta_test[ycol].to_numpy().astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_frac = n_pos / (n_pos + n_neg)
        print("TEST label counts [neg, pos]:", [n_neg, n_pos])
        print("Baseline PR (pos fraction):", pos_frac)
        print("Val counts", meta_test["provenance"].value_counts().head())

    def print_preds(self):
        
        meta_test = pd.read_parquet("k2_dataset_v2/meta_test.parquet").reset_index(drop=True)
        X_test = np.load("k2_dataset_v2/X_test.npy")

        model = tf.keras.models.load_model("k2_window1024_v2.keras")
        p = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)

        meta_test["p"] = p

        # Star scores
        star_max = meta_test.groupby("star_id")["p"].max()
        star_topk = meta_test.groupby("star_id")["p"].apply(lambda s: float(np.mean(np.sort(s)[-10:])))

        # Star labels
        star_y = meta_test.groupby("star_id")["label_star"].max().astype(int)

        star_df = pd.DataFrame({
            "y_star": star_y,
            "score_max": star_max,
            "score_top10mean": star_topk,
        }).reset_index()

        star_df.to_csv("k2_test_star_scores.csv", index=False)
        print(star_df.sort_values("score_top10mean", ascending=False).head(20))

        print("Star PR-AUC (max):", average_precision_score(star_df["y_star"], star_df["score_max"]))
        print("Star PR-AUC (top10mean):", average_precision_score(star_df["y_star"], star_df["score_top10mean"]))
        
        X_test = np.load("k2_dataset_v2/X_test.npy")
        mtrain = pd.read_parquet("k2_dataset_v2/meta_train.parquet")
        mval   = pd.read_parquet("k2_dataset_v2/meta_val.parquet")
        mtest  = pd.read_parquet("k2_dataset_v2/meta_test.parquet")

        s_train = set(mtrain["star_id"])
        s_val   = set(mval["star_id"])
        s_test  = set(mtest["star_id"])

        print("train∩val:", len(s_train & s_val))
        print("train∩test:", len(s_train & s_test))
        print("val∩test:", len(s_val & s_test))


        star_hn = K2_StarRankedHardNegatives()
        star_hn.displayModel()
        #self.precision(star_df)
        #self.precision_after_hardnegs()
        #self.hard_negatives(model)


    def hard_negatives(self, model):
        meta_train = pd.read_parquet("k2_dataset_v2/meta_train.parquet").reset_index(drop=True)
        X_train = np.load("k2_dataset_v2/X_train.npy")

        p_train = model.predict(X_train, batch_size=256, verbose=1).reshape(-1)
        meta_train["p"] = p_train
        thr = 0.65
        hn = meta_train[(meta_train["label"] == 0) & (meta_train["p"] > thr)] \
        .sort_values("p", ascending=False)

        print("Hard negatives:", len(hn))
        hn.head(20)[["star_id","p","start","end","seg_mid_time"]]
        N = 50000  # start with 50k, tune later
        hn_idx = hn.index.values[:N]

        X_hn = X_train[hn_idx]
        y_hn = np.zeros(len(hn_idx), dtype=np.int32)
        meta_hn = meta_train.loc[hn_idx].copy()

        np.save("k2_dataset_v2/X_hardneg.npy", X_hn)
        np.save("k2_dataset_v2/y_hardneg.npy", y_hn)
        meta_hn.to_parquet("k2_dataset_v2/meta_hardneg.parquet", index=False)

        print("Saved hard negative bank:", X_hn.shape)
        star_df = pd.read_csv("k2_test_star_scores.csv")  # from your earlier step
        fp = star_df[(star_df.y_star==0) & (star_df.score_top10mean > thr)] \
                .sort_values("score_top10mean", ascending=False)
        print("FP:", fp.head(30))
        self.train_hardneg(model)

    def train_hardneg(self, model):
        model = tf.keras.models.load_model("k2_window1024_v2.keras", compile=False)
        # model.summary()
        # print("------------- Summary Total params:", model.count_params())

        # # 3) Weights: count + shapes (this is the useful part)
        # weights = model.get_weights()
        # print("-----------Num weight arrays:", len(weights))
        # for i, w in enumerate(weights[:30]):  # show first 30 only
        #     print(i, "shape:", w.shape, "dtype:", w.dtype, "min/max:", float(np.min(w)), float(np.max(w)))

        # # 4) Even nicer: includes layer names for trainable vars
        # for v in model.trainable_variables[:30]:
        #     print(v.name, v.shape, v.dtype)
        # # Load base train
        meta_train = pd.read_parquet("k2_dataset_v2/meta_train.parquet").reset_index(drop=True)
        X_train = np.load("k2_dataset_v2/X_train.npy")
        y_train = meta_train["label"].to_numpy().astype(np.int32)

        # Load hard negatives (y=0)
        X_hn = np.load("k2_dataset_v2/X_hardneg.npy")
        y_hn = np.zeros(len(X_hn), dtype=np.int32)

        # Combine
        X_aug = np.concatenate([X_train, X_hn], axis=0)
        y_aug = np.concatenate([y_train, y_hn], axis=0)

        # Sample weights: make hard negatives "louder"
        w = np.ones(len(X_aug), dtype=np.float32)
        w[len(X_train):] = 8.0   # try 5.0, 8.0, 10.0


        # Fine-tune gently
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),  # assumes sigmoid output
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
            validation_data=(np.load("k2_dataset_v2/X_val.npy"), pd.read_parquet("k2_dataset_v2/meta_val.parquet")["label"].to_numpy().astype(np.int32)),
            verbose=1
        )

        model.save("k2_window1024_v3_hardnegW8.keras")
        print("Saved k2_window1024_v3_hardnegW8.keras")
        print("Output shape:", model.output_shape)
        print("Last layer:", model.layers[-1].name, type(model.layers[-1]))


    def precision_at_k(self,df, score_col, k):
        return df.nlargest(k, score_col)["y_star"].mean()

    def lift_at_k(self, df, score_col, k):
        return self.precision_at_k(df, score_col, k) / df["y_star"].mean()

    def precision(self, star_df):
        for k in [50, 100, 200, 500, 1000]:
            p = self.precision_at_k(star_df, "score_top10mean", k)
            lift = self.lift_at_k(star_df, "score_top10mean", k)
            print(k, "precision", round(p,3), "lift", round(lift,2))

    def precision_after_hardnegs(self):
        
        # Load test
        meta_test = pd.read_parquet("k2_dataset_v2/meta_test.parquet").reset_index(drop=True)
        X_test = np.load("k2_dataset_v2/X_test.npy")
        y_seg = meta_test["label"].to_numpy().astype(int)

        # Load model v3
        model = tf.keras.models.load_model("k2_window1024_v3_hardnegW8.keras")
        p = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
        meta_test["p"] = p

        # Segment metrics (sanity)
        print("Segment PR-AUC:", average_precision_score(y_seg, p))
        print("Segment ROC-AUC:", roc_auc_score(y_seg, p))

        # Star aggregation
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

        # Precision@K (your real KPI)
        def precision_at_k(df, score_col, k):
            return df.nlargest(k, score_col)["y_star"].mean()

        base = star_df["y_star"].mean()
        for k in [50, 100, 200, 500, 1000]:
            pk = precision_at_k(star_df, "score_top10mean", k)
            print(k, "precision", round(pk,3), "lift", round(pk/base,2))

        print(star_df.sort_values("score_top10mean", ascending=False).head(20))

        star_df.to_csv("k2_v3_star_scores.csv", index=False)
        meta_test.to_parquet("k2_dataset_v2/k2_v3_meta_test_with_preds.parquet", index=False)
        # top false-positive stars in test (or do this on train/val stars later)
        fp_stars = star_df[(star_df.y_star==0)].nlargest(200, "score_top10mean")["star_id"].tolist()

        # Look at which segments inside them are scoring high
        fp_segs = meta_test[meta_test["star_id"].isin(fp_stars)].sort_values("p", ascending=False).head(5000)
        print("segs scoring", fp_segs[["star_id","label","p","start","end","seg_mid_time"]].head(20))

        # ----------------- FInanly arrived ----------------- #
        # mined negatives you fed it were either:
        # not representative of the false positives that matter for star ranking, or
        # too few / too heavily weighted, causing calibration shrink without learning discriminative features.