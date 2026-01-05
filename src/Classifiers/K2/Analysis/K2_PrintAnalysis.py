from xml.parsers.expat import model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from src.Classifiers.K2.K2_StarRankedHardNegatives import K2_StarRankedHardNegatives
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        self.star_veto_metrics(mt=mt)

    def star_veto_metrics(self, mt):
        g = mt.groupby("star_id")["p"]

        # windows with high p are what matters for triage
        top10 = g.apply(lambda s: np.sort(s.to_numpy())[-10:] if len(s) >= 10 else np.sort(s.to_numpy()))
        top10mean = top10.apply(lambda a: float(np.mean(a)))

        # spike proxy: fraction of points that are extreme in channel-1 for top windows
        # (fast, crude, useful)
        return top10mean


    def save_galleries(
        self,
        model_path: str,
        data_dir: str = "k2_dataset_centered_v2",
        out_dir: str = "galleries",
        split: str = "test",          # "test" or "val"
        n: int = 25,                  # images per gallery
        batch_size: int = 256,
        ylim_pct: tuple = (1, 99),    # robust limits
        mode: str = "both",           # "both" | "ch0" | "ch1"   (3 modes)
        bin: int = 1,                 # downsample for plotting (8 is nice)
        center_frac: float | None = None,  # e.g. 0.35 to zoom into center
    ):
            os.makedirs(out_dir, exist_ok=True)

            split = split.lower()
            if split not in {"test", "val"}:
                raise ValueError("split must be 'test' or 'val'")

            mode = mode.lower()
            if mode not in {"both", "ch0", "ch1"}:
                raise ValueError("mode must be one of: 'both', 'ch0', 'ch1'")

            if bin < 1:
                raise ValueError("bin must be >= 1")

            if center_frac is not None:
                if not (0 < center_frac <= 1.0):
                    raise ValueError("center_frac must be in (0, 1] or None")

            X_path = os.path.join(data_dir, f"X_{split}.npy")
            meta_path = os.path.join(data_dir, f"meta_{split}.parquet")

            meta = pd.read_parquet(meta_path).reset_index(drop=True)
            X = np.load(X_path)  # (N, 1024, 2)

            if X.ndim != 3 or X.shape[2] != 2:
                raise ValueError(f"Expected X shape (N,1024,2). Got {X.shape}")

            model = tf.keras.models.load_model(model_path)
            p = model.predict(X, batch_size=batch_size, verbose=0).ravel()

            m = meta.copy()
            m["p"] = p

            def _prep(y: np.ndarray) -> np.ndarray:
                y = np.asarray(y, dtype=np.float32)

                # optional center crop (useful because transits are centered in your dataset)
                if center_frac is not None and center_frac < 1.0:
                    L = len(y)
                    w = max(2, int(round(L * center_frac)))
                    s = (L - w) // 2
                    y = y[s:s+w]

                # optional binning for readability
                if bin > 1:
                    L = len(y) - (len(y) % bin)
                    if L <= 0:
                        return y
                    y = y[:L].reshape(-1, bin).mean(axis=1)

                return y

            def plot_grid(df: pd.DataFrame, title: str, fname: str):
                k = min(n, len(df))
                if k == 0:
                    print(f"[save_galleries] No samples for: {title}")
                    return

                ncols = 5
                nrows = (k + ncols - 1) // ncols

                plt.figure(figsize=(ncols * 4, nrows * 3))
                plt.suptitle(f"{title} | mode={mode} | bin={bin}", fontsize=14)

                # IMPORTANT: iterrows gives (idx, Series) where idx is the DataFrame index
                # Since you did meta.reset_index(drop=True), idx is exactly the row into X.
                for i, (idx, row) in enumerate(df.head(k).iterrows(), start=1):
                    idx = int(idx)

                    y0 = _prep(X[idx, :, 0])
                    y1 = _prep(X[idx, :, 1])

                    ax = plt.subplot(nrows, ncols, i)

                    if mode == "both":
                        ax.plot(y0, linewidth=0.8)
                        ax.plot(y1, linewidth=0.8)
                        allv = np.concatenate([y0, y1])
                    elif mode == "ch0":
                        ax.plot(y0, linewidth=0.8)
                        allv = y0
                    else:  # "ch1"
                        ax.plot(y1, linewidth=0.8)
                        allv = y1

                    lo, hi = np.percentile(allv, [ylim_pct[0], ylim_pct[1]])
                    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                        ax.set_ylim(lo, hi)

                    y_star = int(row["label_star"]) if "label_star" in row else -1
                    ax.set_title(
                        f"{row['star_id']}\nlabel={int(row['label'])} y_star={y_star} p={row['p']:.3f}",
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
            hardneg = (m[(m["label"] == 0) & (m.get("label_star", 0) == 0)]
                    .sort_values("p", ascending=False))
            plot_grid(
                hardneg,
                f"{split.upper()} Hard negatives: label=0,y_star=0 sorted by p (top {n})",
                f"{split}_hardneg_{mode}.png",
            )

            # 2) True positives
            truepos = (m[m["label"] == 1]
                    .sort_values("p", ascending=False))
            plot_grid(
                truepos,
                f"{split.upper()} True positives: label=1 sorted by p (top {n})",
                f"{split}_truepos_{mode}.png",
            )

            # 3) False negatives
            falseneg = (m[m["label"] == 1]
                        .sort_values("p", ascending=True))
            plot_grid(
                falseneg,
                f"{split.upper()} False negatives: label=1 sorted by p ASC (top {n})",
                f"{split}_falseneg_{mode}.png",
            )

            scored_path = os.path.join(out_dir, f"{split}_scored_meta.parquet")
            m.to_parquet(scored_path, index=False)
            print("Saved:", scored_path)


   
    def plot_star_top_windows(self, model_path, data_dir, star_id, split="test", topk=12, batch_size=256, out_png=None):
        X = np.load(f"{data_dir}/X_{split}.npy")
        meta = pd.read_parquet(f"{data_dir}/meta_{split}.parquet").reset_index(drop=True)
        model = tf.keras.models.load_model(model_path)

        p = model.predict(X, batch_size=batch_size, verbose=0).ravel()
        mt = meta.copy()
        mt["p"] = p

        rows = mt[mt["star_id"] == star_id].copy()
        if len(rows) == 0:
            print("No rows for", star_id)
            return

        rows = rows.sort_values("p", ascending=False).head(topk)
        idxs = rows.index.to_list()

        ncols = 4
        nrows = (len(idxs) + ncols - 1) // ncols
        plt.figure(figsize=(ncols * 4.5, nrows * 4.2))
        plt.suptitle(f"{star_id} top-{topk} windows by p ({split})")

        for i, idx in enumerate(idxs, start=1):
            y0 = X[idx, :, 0]
            y1 = X[idx, :, 1]

            # robust scaling per window
            allv = np.concatenate([y0, y1])
            lo, hi = np.percentile(allv, [1, 99])

            ax = plt.subplot(nrows, ncols, i)
            ax.plot(np.clip(y0, lo, hi), linewidth=0.8)
            ax.plot(np.clip(y1, lo, hi), linewidth=0.8)
            ax.set_title(f"p={rows.loc[idx,'p']:.3f} label={int(rows.loc[idx,'label'])}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        if out_png:
            plt.savefig(out_png, dpi=170)
            print("Saved:", out_png)
            plt.close()
        else:
            plt.show()

    def furtherPrints(self, X_test_npy:str, meta_tst_pqtfile):
        X = np.load(X_test_npy, mmap_mode="r")  # (N,1024,2)
        for ch in range(X.shape[2]):
            v = X[:2000, :, ch].ravel()
            print(
                "ch", ch,
                "mean", float(v.mean()),
                "std", float(v.std()),
                "near_0_or_1_frac", float(np.mean(np.isclose(v,0,atol=1e-3) | np.isclose(v,1,atol=1e-3)))
            )

        meta = pd.read_parquet(meta_tst_pqtfile).reset_index(drop=True)
        X = np.load(X_test_npy, mmap_mode="r")

        ch = 0  # try 0 then 1
        q05 = np.percentile(X[:,:,ch], 5, axis=1)     # “how low does it go”
        mn  = np.min(X[:,:,ch], axis=1)

        df = meta.copy()
        df["q05"] = q05
        df["mn"]  = mn

        print(df.groupby("label")[["q05","mn"]].median())
        print(df.groupby("label")[["q05","mn"]].mean())

        meta = pd.read_parquet(meta_tst_pqtfile).reset_index(drop=True)
        X = np.load(X_test_npy, mmap_mode="r")

        ch = 0
        q05 = np.percentile(X[:,:,ch], 5, axis=1)

        pos_idx = np.where(meta["label"].to_numpy() == 1)[0]
        pos_sorted = pos_idx[np.argsort(q05[pos_idx])]  # smallest q05 = deepest

        top = pos_sorted[:12]
        plt.figure(figsize=(14,8))
        for i, idx in enumerate(top, 1):
            y = np.array(X[idx, :, ch], dtype=float)

            # simple binning to reveal shallow dips
            b = 8
            yb = y[:len(y)//b*b].reshape(-1, b).mean(axis=1)

            ax = plt.subplot(3,4,i)
            ax.plot(yb, linewidth=1.0)
            lo, hi = np.percentile(yb, [5,95])
            ax.set_ylim(lo, hi)
            ax.set_title(f"{meta.loc[idx,'star_id']}  p?  label=1", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        plt.show()
