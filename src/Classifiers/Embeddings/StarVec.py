import os
import numpy as np
import pandas as pd
from src.Classifiers.CommonHelper import CommonHelper
import tensorflow as tf
from tensorflow.keras import layers, models


class StarVec:
    
    def __init__(self, window, model_path="best_ref.keras"):
        
        self.window = window
        self.commonHelper = CommonHelper()
        # load trained model
        self.trained_model = tf.keras.models.load_model(model_path)

        # build encoder that outputs the penultimate layer (Dropout output)
        # model.layers[-1] = Dense(1, sigmoid)
        # model.layers[-2] = Dropout(0.3) -> this is our embedding layer (64-dim)
        embedding_output = self.trained_model.layers[-2].output
        self.encoder = tf.keras.Model(
            inputs=self.trained_model.input,
            outputs=embedding_output
        )
        pass

    def featurize_star(self,csv_path , 
                       output_path="star_embeddings.npz", 
                       batch_size=128, use_topk=5):
        df = pd.read_csv(csv_path)
        subset_cols = [c for c in ["tid","hostname","toi","kepid","kepler_name",
                                   "koi_name","koi","epic"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)

        df = df.sample(frac=1.0, random_state=42)  # shuffle

        star_to_embs = {}
        star_to_scores = {}
        for index, row in df.iterrows():
            time, flux, mission, target = self._fetch_with_cache(row)
           
            if flux is None or time is None:
                continue

            flux = np.asarray(flux, dtype=np.float32)
            if flux.ndim == 1:
                flux = flux[:, None]
            try:
                if flux.size == 0 or flux.shape[0] < self.window:
                    continue
            except Exception:
                continue
            flux_norm = self.commonHelper.normalize_flux(flux)
            segs, spans = self.commonHelper.segment_with_idx(flux= flux_norm, w=self.window)
            seg_embs = self.encoder.predict(segs, batch_size=batch_size, verbose=0)
            seg_scores = self.trained_model.predict(segs, batch_size=batch_size, verbose=0)
            seg_scores = seg_scores.reshape(-1)  # (num_segs, 1) -> (num_segs,)

            # accumulate
            if target not in star_to_embs:
                star_to_embs[target] = []
                star_to_scores[target] = []

            star_to_embs[target].append(seg_embs)    # list of arrays
            star_to_scores[target].append(seg_scores)


        star_ids = []
        star_vecs = []
        bad = []  # (target, reason)

        for target, emb_list in star_to_embs.items():
            # concat all segments of the star
            H = np.vstack(emb_list)  # (num_total_segs, emb_dim)
            S = np.concatenate(star_to_scores[target])  # (num_total_segs,)

            if use_topk is not None and use_topk > 0:
                k = min(use_topk, H.shape[0])
                idx = np.argsort(S)[-k:]
                H_used = H[idx]
            else:
                H_used = H

            # --- NEW: drop any segments with NaN/Inf ---
            seg_ok = np.isfinite(H_used).all(axis=1)
            H_used = H_used[seg_ok]
            if H_used.shape[0] == 0:
                bad.append((target, "all segments non-finite"))
                continue

            # --- NEW: mean pool in float64 to avoid overflow ---
            z_star = H_used.astype(np.float64).mean(axis=0)

            # --- NEW: validate star embedding ---
            if not np.isfinite(z_star).all():
                bad.append((target, "star embedding non-finite after mean"))
                continue

            # --- NEW (recommended): L2-normalize the star embedding ---
            n = np.linalg.norm(z_star)
            if not np.isfinite(n) or n == 0:
                bad.append((target, "bad norm"))
                continue
            z_star = (z_star / (n + 1e-12)).astype(np.float32)

            star_ids.append(target)
            star_vecs.append(z_star)

        # ---- AFTER the loop ----
        star_ids = np.array(star_ids)
        star_vecs = np.vstack(star_vecs).astype(np.float32)

        np.savez_compressed(output_path, star_ids=star_ids, star_vecs=star_vecs)

        if bad:
            with open(output_path.replace(".npz", "_bad_embeddings.txt"), "w", encoding="utf-8") as f:
                for sid, reason in bad:
                    f.write(f"{sid}\t{reason}\n")

        print(f"Saved {len(star_ids)} star embeddings to {output_path}")
        print(f"Dropped {len(bad)} stars (see *_bad_embeddings.txt)")
        return star_ids, star_vecs

    

    def _fetch_with_cache(self, row):
        target, _ = self.commonHelper.row_to_target_and_mission(row)
        if not target:
            return None, None, None, None
        cache_path = self.commonHelper.cache_path_for_target(cache_dir="lc_cache", target= target)

        # 1) Try to load from cache
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)

            if "empty" in data and bool(data["empty"]):
                print("Cached EMPTY star, skipping:", target)
                return None, None, None, target

            time   = data["time"]
            flux   = data["flux"]
            mission = str(data["mission"])
            target  = str(data["target"])
            return time, flux, mission, target
        
    

        

        

