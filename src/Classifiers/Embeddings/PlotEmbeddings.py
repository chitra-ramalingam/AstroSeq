import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.Classifiers.CommonHelper import CommonHelper

class PlotEmbeddings:
    def __init__(self, window=200, trained_model_path="best_ref.keras"):
        self.window = window
        self.commonHelper = CommonHelper()
        self.model  = tf.keras.models.load_model("best_ref.keras")
        pass

    def inspect_star(self,target_id, window, topk_segments=20):
        # 1) load from cache
        row_dummy = {"target": target_id}   
        time, flux, mission, target = self.commonHelper.fetch_with_targetId_FromCache(target_id)
        if flux is None or time is None:
            print("No cache to inspect", target_id)
            return

        flux = np.asarray(flux, dtype=np.float32)
        if flux.ndim == 1:
            flux = flux[:, None]

        if flux.shape[0] < window:
            print("Too short:", target_id)
            return

        flux_norm = self.commonHelper.normalize_flux(flux)

        # 2) segment
        segs, spans = self.commonHelper.segment_with_idx(flux_norm, self.window)  # spans: list of (start,end) indices
        if segs is None or len(segs) == 0:
            print("No segments for", target_id)
            return

        segs = np.asarray(segs, dtype=np.float32)
        if segs.ndim == 2:
            segs = segs[..., None]  # (n_seg, window, 1)

        # 3) segment scores
        seg_scores = self.model.predict(segs, batch_size=256, verbose=0).reshape(-1)

        # pick top-k segments by score
        k = min(topk_segments, len(seg_scores))
        idx_top = np.argsort(seg_scores)[-k:]

        # 4) plot: full light curve + highlight top segments
        t = np.asarray(time)
        f = flux_norm.squeeze()

        plt.figure(figsize=(10, 5))
        plt.plot(t, f, ".", markersize=1, alpha=0.5, label="flux")

        for i in idx_top:
            start, end = spans[i]
            plt.axvspan(t[start], t[end-1], color="red", alpha=0.2)

        plt.title(f"{target_id} (top {k} segments highlighted)")
        plt.xlabel("time")
        plt.ylabel("normalized flux")
        plt.tight_layout()
        plt.show()

        # Optional: plot the top segments themselves stacked
        plt.figure(figsize=(6, 6))
        for j, i in enumerate(idx_top):
            plt.plot(segs[i,:,0] + j*5)  # offset each for clarity
        plt.title(f"{target_id} top {k} segments (offset)")
        plt.xlabel("time index within window")
        plt.tight_layout()
        plt.show()
