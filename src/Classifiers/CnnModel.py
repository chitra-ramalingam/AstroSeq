from src.Classifiers.SimpleLightCurve import SimpleLightCurve
from src.Classifiers.Astro1DCNN import Astro1DCNN
from src.Classifiers.CNNPlots import CNNPlots
import numpy as np
class CnnModel:
    def __init__(self):
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

    def runAstro1DCNN(self):
        epoch = 30
        astro_cnn = Astro1DCNN(window=200, mission="TESS", author="SPOC")
        X, y, groups, sample_w = astro_cnn.build_from_csv(
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

        astro_cnn.evaluateModel(model,X_val, y_val, X_test, y_test, mode="balanced_accuracy")


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
