import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.Classifiers.Embeddings.RegressorNet import RegressorNet
from src.Classifiers.Embeddings.PlotEmbeddings import PlotEmbeddings
class EmbeddingPresentation:
    def __init__ (self, 
                  npz_path: str = "star_embeddings_1dcnn.npz",
                  star_scores_csv : str = "starwise_score_1dcnn.csv"): 
        self.star_scores_csv = star_scores_csv
        self.npz_path = npz_path
        pass

    def load_embeddings_and_check(self):
        data = np.load(self.npz_path, allow_pickle=True)
        print(data.files)  # sanity check keys

        
        star_ids  = data["star_ids"]   # (N,)
        star_vecs = data["star_vecs"]  # (N, emb_dim)

        print("num stars:", star_ids.shape[0])
        print("embedding dim:", star_vecs.shape[1])
        print("dtype:", star_vecs.dtype)


        n_nan = np.isnan(star_vecs).sum()
        n_inf = np.isinf(star_vecs).sum()
        print("nan count:", n_nan, "inf count:", n_inf)

        abs_max = np.nanmax(np.abs(star_vecs))
        print("max |value| in embeddings:", abs_max)

        # 3) Find which stars are broken (rows with any non-finite)
        bad_rows = ~np.isfinite(star_vecs).all(axis=1)
        print("bad rows:", bad_rows.sum(), "/", star_vecs.shape[0])

        for i in np.where(bad_rows)[0][:10]:
            print("bad star:", star_ids[i], "first vals:", star_vecs[i][:6])


        vec64 = star_vecs.astype(np.float64)
        norms = np.linalg.norm(vec64, axis=1)

        finite_norms = norms[np.isfinite(norms)]
        print("norm mean/std/min/max (finite only):",
          finite_norms.mean(), finite_norms.std(),
             finite_norms.min(), finite_norms.max())
        
        return star_vecs
     
    def plot_embeddings(self, star_vecs):
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(star_vecs)  # (N,2)
        plt.figure(figsize=(6, 6))
        plt.scatter(Z2[:,0], Z2[:,1], s=4, alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Star embeddings (PCA)")
        plt.tight_layout()
        plt.show()
    
    def runRegressorNet(self):
        reg_net = RegressorNet()
        smooth_scores, star_ids, star_vecs = reg_net.runModel(
            embedding_path=self.npz_path,
            star_scores_csv= self.star_scores_csv
        )
        reg_net.calculate_cosinedistance(smooth_scores, star_ids, star_vecs)
        wierd_stars = reg_net.find_outliers(smooth_scores, star_ids, star_vecs)
        # plotEmbeddings = PlotEmbeddings()
        # for index, row in wierd_stars.iterrows():
        #     target_id = row["target"]
        #     plotEmbeddings.inspect_star(
        #         target_id=target_id,
        #         window=200,
        #         topk_segments=20
        #     )
        