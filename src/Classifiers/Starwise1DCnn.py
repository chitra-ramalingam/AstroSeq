import numpy as np
import pandas as pd
from src.Classifiers.CommonHelper import CommonHelper
from src.Classifiers.CnnNNet import CnnNNet

class Starwise1DCnn(CnnNNet):
    def __init__(self, window=201, stride=None, top_k=5):
        super().__init__(window=window, channels=1)
        self.commonHelper = CommonHelper()
        self.window = window
        self.net = CnnNNet(window=window, channels=1)
        self.stride = stride or window // 4
        self.top_k = top_k

    # --- core utils ---

    def _normalize_flux(self, flux):
        """Median/MAD normalization, same as training."""
        if flux.ndim == 1:
            flux = flux[:, None]  # (T, 1)
        med = np.median(flux, axis=0, keepdims=True)
        mad = np.median(np.abs(flux - med), axis=0, keepdims=True) + 1e-6
        flux_norm = (flux - med) / mad
        return flux_norm

    def _segment_flux(self, flux):
        segs, spans = self.commonHelper.segment_with_idx(flux, w=self.window, stride=self.stride)
        # segs: (num_segments, window), or (num_segments, window, C)
        return segs, spans

    def _aggregate_segments(self, p_seg):
        """Star-level score from segment probabilities."""
        if len(p_seg) == 0:
            return np.nan

        # Example: top-k mean (or you can use np.max(p_seg))
        k = min(self.top_k, len(p_seg))
        top_idx = np.argpartition(-p_seg, k - 1)[:k]
        return float(p_seg[top_idx].mean())

    # --- main entry for new stars ---

    def predict_stars_from_csv(self, csv_path,
                               use_all=True, max_files=4, any_author=True,
                               return_segments=False):
        
        df = pd.read_csv(csv_path)

        subset_cols = [c for c in ["tid","hostname","toi","kepid","kepler_name",
                                   "koi_name","koi","epic"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)

        df = df.sample(frac=1.0, random_state=42)  # shuffle

        star_rows = []
        segments_info = {}  # optional detailed outputs

        for _, row in df.iterrows():
            time, flux, mission, target = self.commonHelper.fetch_flux_row(
                row,
                use_all=use_all,
                max_files=max_files,
                any_author=any_author
            )

            if flux is None or flux is None:
                continue

            # Normalize like training
            flux = np.asarray(flux, dtype=np.float32)
            if flux.ndim == 1:
                flux = flux[:, None]

            try:
                if flux.size == 0 or flux.shape[0] < self.window:
                    continue
            except Exception:
                continue

            flux_norm = self._normalize_flux(flux)

            # Segment
            segs, spans = self._segment_flux(flux_norm)
            if segs.shape[0] == 0:
                star_score = np.nan
                p_seg = np.array([], dtype=np.float32)
            else:
                # Make sure shape matches model: (N, window, channels)
                if segs.ndim == 2:
                    segs_in = segs[..., None]  # (N, window, 1)
                else:
                    segs_in = segs

                # Predict segment probabilities with  trained CNN
                p_seg = self.model.predict(segs_in, verbose=0).ravel().astype(np.float32)

                # Aggregate to star-level probability
                star_score = self._aggregate_segments(p_seg)

            # Build a star row
            star_rows.append({
                "target_id": target,
                "mission": mission,
                "num_segments": len(p_seg),
                "star_score": star_score,
            })

            if return_segments:
                segments_info[target] = {
                    "time": time,
                    "spans": spans,
                    "segment_scores": p_seg,
                }

        df_stars = pd.DataFrame(star_rows)
        if return_segments:
            return df_stars, segments_info
        return df_stars
