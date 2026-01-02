from xml.parsers.expat import model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from src.Classifiers.K2.K2_StarRankedHardNegatives import K2_StarRankedHardNegatives
import os
import matplotlib.pyplot as plt


class K2_PrintAnalysis:
    def __init__(self):
        pass


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


    def save_galleries(
        self,
        model_path: str,
        data_dir: str = "k2_dataset_centered_v2",
        out_dir: str = "galleries",
        split: str = "test",          # "test" or "val"
        n: int = 25,                  # images per gallery
        batch_size: int = 256,
        ylim_pct: tuple = (1, 99),    # robust limits so spikes don't flatten everything
        ):

        os.makedirs(out_dir, exist_ok=True)

        split = split.lower()
        if split not in {"test", "val"}:
            raise ValueError("split must be 'test' or 'val'")

        X_path = os.path.join(data_dir, f"X_{split}.npy")
        meta_path = os.path.join(data_dir, f"meta_{split}.parquet")

        meta = pd.read_parquet(meta_path).reset_index(drop=True)
        X = np.load(X_path)  # (N, 1024, 2)

        # sanity check
        if X.ndim != 3 or X.shape[2] != 2:
            raise ValueError(f"Expected X shape (N,1024,2). Got {X.shape}")

        model = tf.keras.models.load_model(model_path)
        p = model.predict(X, batch_size=batch_size, verbose=0).ravel()

        m = meta.copy()
        m["p"] = p

        def plot_grid(df, title, fname):
            k = min(n, len(df))
            if k == 0:
                print(f"[save_galleries] No samples for: {title}")
                return

            ncols = 5
            nrows = (k + ncols - 1) // ncols

            plt.figure(figsize=(ncols * 4, nrows * 3))
            plt.suptitle(title)

            for i, row in enumerate(df.head(k).itertuples(index=True), start=1):
                idx = int(row.Index)

                y0 = X[idx, :, 0]
                y1 = X[idx, :, 1]

                ax = plt.subplot(nrows, ncols, i)

                # plot BOTH channels
                ax.plot(y0, linewidth=0.8)
                ax.plot(y1, linewidth=0.8)

                # robust y-limits to reveal dips (avoid spike-dominated scaling)
                allv = np.concatenate([y0, y1])
                lo, hi = np.percentile(allv, [ylim_pct[0], ylim_pct[1]])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    ax.set_ylim(lo, hi)

                ax.set_title(
                    f"{row.star_id}\nlabel={int(row.label)} y_star={int(row.label_star)} p={row.p:.3f}",
                    fontsize=9
                )
                ax.set_xticks([])
                ax.set_yticks([])

            out_path = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()
            print("Saved:", out_path)

        # 1) Hard negatives
        hardneg = (m[(m["label"] == 0) & (m["label_star"] == 0)]
                .sort_values("p", ascending=False))
        plot_grid(
            hardneg,
            f"{split.upper()} Hard negatives: label=0,y_star=0 sorted by p (top {n})",
            f"{split}_hardneg_gallery.png",
        )

        # 2) True positives
        truepos = (m[m["label"] == 1]
                .sort_values("p", ascending=False))
        plot_grid(
            truepos,
            f"{split.upper()} True positives: label=1 sorted by p (top {n})",
            f"{split}_truepos_gallery.png",
        )

        # 3) False negatives
        falseneg = (m[m["label"] == 1]
                    .sort_values("p", ascending=True))
        plot_grid(
            falseneg,
            f"{split.upper()} False negatives: label=1 sorted by p ASC (top {n})",
            f"{split}_falseneg_gallery.png",
        )

        scored_path = os.path.join(out_dir, f"{split}_scored_meta.parquet")
        m.to_parquet(scored_path, index=False)
        print("Saved:", scored_path)
