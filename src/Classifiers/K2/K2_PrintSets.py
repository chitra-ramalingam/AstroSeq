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
        meta_test = pd.read_parquet("k2_dataset_centered_v2/meta_test.parquet")
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
        
        meta_test = pd.read_parquet("k2_dataset_centered_v2/meta_test.parquet").reset_index(drop=True)
        X_test = np.load("k2_dataset_centered_v2/X_test.npy")

        model = tf.keras.models.load_model("k2_window1024_centralized_v2.keras")
        p = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
        y_seg = meta_test["label"].to_numpy().astype(int)
        print("Segment baseline PR (pos frac):", y_seg.mean())
        print("Segment PR-AUC:", average_precision_score(y_seg, p))
        print("Segment ROC-AUC:", roc_auc_score(y_seg, p))
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
        
        X_test = np.load("k2_dataset_centered_v2/X_test.npy")
        mtrain = pd.read_parquet("k2_dataset_centered_v2/meta_train.parquet")
        mval   = pd.read_parquet("k2_dataset_centered_v2/meta_val.parquet")
        mtest  = pd.read_parquet("k2_dataset_centered_v2/meta_test.parquet")

        s_train = set(mtrain["star_id"])
        s_val   = set(mval["star_id"])
        s_test  = set(mtest["star_id"])

        print("train∩val:", len(s_train & s_val))
        print("train∩test:", len(s_train & s_test))
        print("val∩test:", len(s_val & s_test))

        g = meta_test.groupby("star_id")["p"]

        star_max = g.max()
        star_top10mean = g.apply(lambda s: float(np.mean(np.sort(s)[-10:])))
        star_median_top10 = g.apply(lambda s: float(np.median(np.sort(s)[-10:])))
        star_consistency = g.apply(lambda s: float(np.mean(s > 0.6)))  # tune 0.6/0.65 later
        star_gap = star_max - star_top10mean

        star_y = meta_test.groupby("star_id")["label_star"].max().astype(int)

        star_df = pd.DataFrame({
            "y_star": star_y,
            "score_max": star_max,
            "score_top10mean": star_top10mean,
            "score_median_top10": star_median_top10,
            "consistency_gt_0p6": star_consistency,
            "gap": star_gap,
        }).reset_index()
        star_df["score_fused"] = (
                star_df["score_median_top10"]
                + 0.10 * star_df["consistency_gt_0p6"]
                - 0.20 * star_df["gap"]
            )
        print("Star PR-AUC (fused):", average_precision_score(star_df["y_star"], star_df["score_fused"]))



        star_hn = K2_StarRankedHardNegatives()
        star_hn.displayModel()
        self.precision(star_df)
        DO_MINE_HARDNEGS = True  
        if DO_MINE_HARDNEGS:
            self.hard_negatives(model)
        else:
            print("Skipping hard negative mining (using existing bank).")
        self.precision_after_hardnegs()

        meta_val = pd.read_parquet("k2_dataset_centered_v2/meta_val.parquet").reset_index(drop=True)
        y_val = meta_val["label"].to_numpy().astype(int)
        print("VAL label counts [neg,pos]:", [(y_val==0).sum(), (y_val==1).sum()])
        print("VAL baseline PR (pos frac):", y_val.mean())



    def hard_negatives(self, model):
        meta_train = pd.read_parquet("k2_dataset_centered_v2/meta_train.parquet").reset_index(drop=True)
        X_train = np.load("k2_dataset_centered_v2/X_train.npy")

        p_train = model.predict(X_train, batch_size=256, verbose=0).reshape(-1)
        meta_train["p"] = p_train
        meta_train = meta_train.copy()
        meta_train["p"] = p_train

        BANK_SIZE = 50000
        PER_STAR_CAP = 30

        neg0 = meta_train[(meta_train["label"]==0) & (meta_train["label_star"]==0)]
        neg1 = meta_train[(meta_train["label"]==0) & (meta_train["label_star"]==1)]

        hn = (neg0.sort_values("p", ascending=False)
            .groupby("star_id", group_keys=False)
            .head(PER_STAR_CAP)
            .head(BANK_SIZE))
        
        hn = hn.sort_values("p", ascending=False)


        print("HN bank:", len(hn),
            "unique stars:", hn["star_id"].nunique(),
            "p range:", float(hn["p"].min()), "to", float(hn["p"].max()),
            "top star cap:", int(hn["star_id"].value_counts().iloc[0]))

        print("Train neg candidates (y_star=0):", len(neg0), "max p:", float(neg0["p"].max()))
        print("Train neg candidates (y_star=1):", len(neg1), "max p:", float(neg1["p"].max()))

        #hn = meta_train[(meta_train["label"]==0) & (meta_train["label_star"]==0) & (meta_train["p"]>=thr)]
        print("HN mined:", len(hn), "unique stars:", hn["star_id"].nunique())
        print("HN top p:", float(hn["p"].max()) if len(hn) else None)

        print("\nHN head:")
        print(hn.sort_values("p", ascending=False).head(10)[["star_id","p","start","end","seg_mid_time"]])

        print("Hard negatives:", len(hn))
        print(hn.head(20)[["star_id","p","start","end","seg_mid_time"]])
        print("HN from y_star=1 stars:", int((hn["label_star"] == 1).sum()))
        print("HN from y_star=0 stars:", int((hn["label_star"] == 0).sum()))

        BANK_N = 50000   # bank size saved to disk
        hn_idx = hn.index.values[:BANK_N]

        X_hn_bank = X_train[hn_idx]
        y_hn_bank = np.zeros(len(hn_idx), dtype=np.int32)
        meta_hn_bank = meta_train.loc[hn_idx].copy()

        TAKE = 10000     # or 20000 if stable
        X_hn = X_hn_bank[:TAKE]
        y_hn = y_hn_bank[:TAKE]
        meta_hn = meta_hn_bank.iloc[:TAKE].copy()


        np.save("k2_dataset_centered_v2/X_hardneg.npy", X_hn)
        np.save("k2_dataset_centered_v2/y_hardneg.npy", y_hn)
        meta_hn.to_parquet("k2_dataset_centered_v2/meta_hardneg.parquet", index=False)

        np.save("k2_dataset_centered_v2/X_hardneg_bank.npy", X_hn_bank)   # 50k
        meta_hn_bank.to_parquet("k2_dataset_centered_v2/meta_hardneg_bank.parquet", index=False)

        np.save("k2_dataset_centered_v2/X_hardneg_take.npy", X_hn)        # 10k
        meta_hn.to_parquet("k2_dataset_centered_v2/meta_hardneg_take.parquet", index=False)

        print("Saved hardneg BANK:", X_hn_bank.shape, "TAKE:", X_hn.shape)

        star_df = pd.read_csv("k2_test_star_scores.csv")  # from your earlier step
       
        print("p quantiles:", hn["p"].quantile([0.5,0.9,0.99]).to_dict())
        self.train_hardneg(model,X_train, X_hn_bank, X_hn, meta_hn)

    def train_hardneg(self, model, X_train, X_hn_bank, X_hn, meta_hn):
        model = tf.keras.models.load_model("k2_window1024_centralized_v2.keras", compile=False)
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
        meta_train = pd.read_parquet("k2_dataset_centered_v2/meta_train.parquet").reset_index(drop=True)
        X_train = np.load("k2_dataset_centered_v2/X_train.npy", mmap_mode="r")
        y_train = meta_train["label"].to_numpy().astype(np.int32)

        # Load hard negatives (y=0)
        X_hn = np.load("k2_dataset_centered_v2/X_hardneg_take.npy")
        y_hn = np.zeros(len(X_hn), dtype=np.int32)

        # Combine
        X_aug = np.concatenate([X_train, X_hn], axis=0)
        y_aug = np.concatenate([y_train, y_hn], axis=0)

        # Sample weights: make hard negatives "louder"
        w = np.ones(len(X_aug), dtype=np.float32)
        w[len(X_train):] = 2.0   # try 5.0, 8.0, 10.0


        # Fine-tune gently
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),  # assumes sigmoid output
            metrics=[
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                    tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                ],

        )
        print("Model:", model)
        print("Bank size:", X_hn_bank.shape, "Take:", X_hn.shape)
        print("Take p range:", float(meta_hn["p"].min()), "to", float(meta_hn["p"].max()))
        print("Weights: base=1.0 hardneg=", float(w[len(X_train)]))

        model.fit(
            X_aug, y_aug,
            sample_weight=w,
            batch_size=256,
            epochs=2,
            validation_data=(np.load("k2_dataset_centered_v2/X_val.npy"), pd.read_parquet("k2_dataset_centered_v2/meta_val.parquet")["label"].to_numpy().astype(np.int32)),
            verbose=1
        )

        model.save("k2_window1024_v3_hardnegW2.keras")
        self.print_eval_report("k2_window1024_v3_hardnegW2.keras")

        print("Saved k2_window1024_v3_hardnegW2.keras")
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
        meta_test = pd.read_parquet("k2_dataset_centered_v2/meta_test.parquet").reset_index(drop=True)
        X_test = np.load("k2_dataset_centered_v2/X_test.npy")
        y_seg = meta_test["label"].to_numpy().astype(int)

        # Load model v3
        model = tf.keras.models.load_model("k2_window1024_v3_hardnegW2.keras")
        p = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
        meta_test["p"] = p
        y_seg = meta_test["label"].to_numpy().astype(int)
        base_pr = y_seg.mean()
        print("Segment  baseline PR (pos frac):", base_pr)
        print("Segment  PR-AUC:", average_precision_score(y_seg, p))
        print("Segment  ROC-AUC:", roc_auc_score(y_seg, p))
     
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
        meta_test.to_parquet("k2_dataset_centered_v2/k2_v3_meta_test_with_preds.parquet", index=False)
        # top false-positive stars in test (or do this on train/val stars later)
        fp_stars = star_df[(star_df.y_star==0)].nlargest(200, "score_top10mean")["star_id"].tolist()

        # Look at which segments inside them are scoring high
        fp_segs = meta_test[meta_test["star_id"].isin(fp_stars)].sort_values("p", ascending=False).head(5000)
        print("segs scoring", fp_segs[["star_id","label","p","start","end","seg_mid_time"]].head(20))
        # ----------------- FInanly arrived ----------------- #
        # mined negatives you fed it were either:
        # not representative of the false positives that matter for star ranking, or
        # too few / too heavily weighted, causing calibration shrink without learning discriminative features.

    def print_eval_report(self, model_path: str):

        # Load splits
        meta_val  = pd.read_parquet("k2_dataset_centered_v2/meta_val.parquet").reset_index(drop=True)
        X_val     = np.load("k2_dataset_centered_v2/X_val.npy")

        meta_test = pd.read_parquet("k2_dataset_centered_v2/meta_test.parquet").reset_index(drop=True)
        X_test    = np.load("k2_dataset_centered_v2/X_test.npy")

        model = tf.keras.models.load_model(model_path)

        def eval_split(name, X, meta):
            p = model.predict(X, batch_size=256, verbose=0).ravel()
            y = meta["label"].to_numpy().astype(int)
            print(f"\n[{name}] baseline PR:", float(y.mean()))
            print(f"[{name}] PR-AUC:", float(average_precision_score(y, p)))
            print(f"[{name}] ROC-AUC:", float(roc_auc_score(y, p)))
            print(f"[{name}] p range:", float(p.min()), "to", float(p.max()), "mean:", float(p.mean()))
            return p

        print("\nEvaluating:", model_path)
        p_val  = eval_split("VAL",  X_val,  meta_val)
        p_test = eval_split("TEST", X_test, meta_test)

        # Star-level aggregation on TEST
        mt = meta_test.copy()
        mt["p"] = p_test

        g = mt.groupby("star_id")["p"]
        star_df = (pd.DataFrame({
            "star_id": g.size().index,
            "y_star": mt.groupby("star_id")["label_star"].max().astype(int).values,
            "score_max": g.max().values,
            "score_top10mean": g.apply(lambda s: float(np.mean(np.sort(s)[-10:]))).values,
            "score_median_top10": g.apply(lambda s: float(np.median(np.sort(s)[-10:]))).values,
        }))

        print("\nStar PR-AUC (max):", float(average_precision_score(star_df["y_star"], star_df["score_max"])))
        print("Star PR-AUC (top10mean):", float(average_precision_score(star_df["y_star"], star_df["score_top10mean"])))
        print("Star PR-AUC (median_top10):", float(average_precision_score(star_df["y_star"], star_df["score_median_top10"])))

        base = float(star_df["y_star"].mean())
        for k in [50, 100, 200, 500, 1000]:
            pk = float(star_df.nlargest(k, "score_top10mean")["y_star"].mean())
            print(k, "precision", round(pk, 3), "lift", round(pk / base, 2))

        print("\nTop 20 stars:")
        print(star_df.sort_values("score_top10mean", ascending=False).head(20))

        star_df.to_csv("k2_v3_star_scores.csv", index=False)
        mt.to_parquet("k2_dataset_centered_v2/k2_v3_meta_test_with_preds.parquet", index=False)
