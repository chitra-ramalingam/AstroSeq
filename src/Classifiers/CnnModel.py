from src.Classifiers.Embeddings.EmbeddingPresentation import EmbeddingPresentation
from src.Classifiers.SimpleLightCurve import SimpleLightCurve
from src.Classifiers.Astro1DCNN import Astro1DCNN
from src.Classifiers.CNNPlots import CNNPlots
from src.Classifiers.Starwise1DCnn import Starwise1DCnn
from src.Classifiers.isolation_forest import IsolationForestScorer
from src.Classifiers.CnnNNet import CnnNNet
from src.Classifiers.CommonHelper import CommonHelper
from src.Classifiers.Embeddings.StarVec import StarVec
import tensorflow as tf
import numpy as np
class CnnModel:
    def __init__(self):
        self.starwise1DCsv = "starwise_score_1dcnn.csv"
        self.starEmbeddingsNpz = "star_embeddings_1dcnn.npz"
        pass

    def runSampleModels(self):
        slC = SimpleLightCurve()
        flux =slC.normalize()
        segments = slC.segmentLightCurve(flux)
        model = slC.declareModel(window=200)
        trainRespones = slC.trainModel(model, segments, labels=[1]*len(segments))  # Dummy labels for illustration
        slC.evaluateModel(model, trainRespones[1], trainRespones[2])  # Dummy labels for illustration
        slC.plotAll(trainRespones[0])
        print("Welcome to Exo-Planets")

    def runStarbased1DCNN(self):
       
        model = Starwise1DCnn( window=200, stride=50, top_k=5)
        model.model = tf.keras.models.load_model("best_ref.keras")

        df_stars, seg_info = model.predict_stars_from_csv("CombinedExoplanetData.csv", return_segments=True)
        if 'star_score' in df_stars.columns:
           df_stars = df_stars.sort_values('star_score', ascending=False).reset_index(drop=True)
        print(df_stars.head(100))
        out_path = self.starwise1DCsv
        df_stars.to_csv(out_path, index=False)
        print(f"Saved df_stars to {out_path}")

    def runStarVecEmbeddings(self):
        starvec = StarVec(window=200, model_path="best_ref.keras")
        starvec.featurize_star(
            csv_path="CombinedExoplanetData.csv",
            output_path=self.starEmbeddingsNpz,
            batch_size=128,
            use_topk=5
        )

    def runTestOnStarVecEmbeddings(self):
        embed_presenter = EmbeddingPresentation(npz_path=self.starEmbeddingsNpz,
                                                star_scores_csv=self.starwise1DCsv)
        star_vecs = embed_presenter.load_embeddings_and_check()
        embed_presenter.plot_embeddings(star_vecs=star_vecs)
        embed_presenter.runRegressorNet()


    def runAstro1DCNN(self):
        epoch = 30
        astro_cnn = Astro1DCNN(window=200, mission="TESS", author="SPOC")
        X, y, groups, sample_w,cache = astro_cnn.build_from_csv(
                        "CombinedExoplanetData.csv",
                        min_pos_groups=12,     # stars with at least one positive segment
                        min_neg_groups=12,     # stars with no positive segments (or no ephemeris)
                        any_author=True,       # accept SPOC/QLP/etc.
                        use_all=True,          # stitch multiple files (e.g., sectors)
                        max_files=4,
                        stride=50,             # overlap windows to catch dips
                        per_star_cap=40        # cap segments per star to avoid dominance
                    )
        model = astro_cnn.declareModel(channels=X.shape[2])   # <-- not 1 anymore

        hist, X_test, y_test,X_val,y_val = astro_cnn.trainModel(model,
                                                                 X=X,
                                                                 y=y,
                                                                 groups=groups,
                                                                 sample_w=sample_w,
                                                                epochs=epoch, batch_size=128)

        eval_res = astro_cnn.evaluateModel(model,X_val, y_val, X_test, y_test, mode="balanced_accuracy")
        astro_cnn.set_threshold(eval_res["threshold"])

        df_pred = astro_cnn.predict_from_cache(model, cache, agg="topk_mean", k=3)

        plots = CNNPlots()

        # after training: doesnt work on threshold, uses default 0.5
        plots.plot_history(hist)

        # curves
        plots.plot_roc_pr(model, X_val,  y_val,  set_name="VAL")
        plots.plot_roc_pr(model, X_test, y_test, set_name="TEST")

        # robust ROC + reports
        plots.safe_plot_roc(model, X_val,  y_val,  set_name="VAL")
        plots.safe_plot_roc(model, X_test, y_test, set_name="TEST")

        # pick a good threshold on VAL, then evaluate on TEST
        thr = plots.evaluate_with_threshold(model, X_val, y_val, X_test, y_test, mode="balanced_accuracy")
