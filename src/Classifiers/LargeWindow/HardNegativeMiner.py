import numpy as np
import pandas as pd
import tensorflow as tf

from src.Classifiers.LargeWindow.SegmentDataset import SegmentDataset


class HardNegativeMiner:
    """
    Mines "hard negatives" from TRAIN segments only:
    - take label==0 segments
    - score them with a trained model
    - keep top-K highest predicted probabilities

    This is mission-aware (filters on mission).
    """

    def __init__(self, window: int = 1024, batch_size: int = 64, stride: int = 256):
        self.window = window
        self.batch_size = batch_size
        self.stride = stride

    def mine(
                self,
                model_path: str,
                train_df: pd.DataFrame,
                mission: str = "tess",
                max_neg_pool: int = 300_000,
                topk: int = 80_000,
                max_neg_per_star: int | None = 200,
                seed: int = 0,
                output_path: str | None = None,
            ) -> pd.DataFrame:
        """
        Returns a dataframe of hard negatives (label=0) with an extra 'score' column.
        If output_path is provided, writes parquet.
        """

        mission = mission.strip().lower()

        df = train_df.copy()
        df["mission"] = df["mission"].astype(str).str.lower()

        # 1) Negative pool from TRAIN only
        neg_pool = df[(df["mission"] == mission) & (df["label"] == 0)].copy()
        if len(neg_pool) == 0:
            raise RuntimeError(f"No negatives found for mission='{mission}' in train_df")

        # Optional: cap negatives per star for diversity (NO groupby.apply warning)
        if max_neg_per_star is not None and max_neg_per_star > 0:
            rng = np.random.RandomState(seed)
            parts = []
            for _, g in neg_pool.groupby("star_id", sort=False):
                k = min(len(g), max_neg_per_star)
                # different seed per star but still reproducible overall
                s = int(rng.randint(0, 2**31 - 1))
                parts.append(g.sample(n=k, random_state=s, replace=False))
            neg_pool = pd.concat(parts, ignore_index=True)

        # Optional: cap total pool size (keeps runtime reasonable)
        if max_neg_pool is not None and len(neg_pool) > max_neg_pool:
            neg_pool = neg_pool.sample(n=max_neg_pool, random_state=seed).reset_index(drop=True)

        print(f"[HardNegativeMiner] Neg pool size: {len(neg_pool)} (mission={mission})")

        # 2) Load trained model
        model = tf.keras.models.load_model(model_path)

        # 3) Score in batches
        neg_gen = SegmentDataset(
            neg_pool,
            batch_size=self.batch_size,
            shuffle=False,
            window=self.window,
        )

        scores = []
        for Xb, _ in neg_gen:
            preds = np.asarray(model.predict_on_batch(Xb)).ravel()
            scores.append(preds)

        scores = np.concatenate(scores, axis=0)
        if len(scores) != len(neg_pool):
            raise RuntimeError(
                f"Score length mismatch: got {len(scores)} scores for {len(neg_pool)} rows."
            )

        neg_pool = neg_pool.reset_index(drop=True)
        neg_pool["score"] = scores

        # 4) Take top-K hardest
        topk = min(topk, len(neg_pool))
        hard = neg_pool.nlargest(topk, "score").reset_index(drop=True)

        print(f"[HardNegativeMiner] Hard negatives selected: {len(hard)} (topk={topk})")
        print("[HardNegativeMiner] score stats:", hard["score"].describe().to_dict())

        if output_path:

            hard.to_parquet(output_path, index=False)
            print(f"[HardNegativeMiner] Saved: {output_path}")

        return hard

