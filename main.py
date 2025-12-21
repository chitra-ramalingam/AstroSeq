import re
import shutil
from pathlib import Path

import pandas as pd

CSV_PATH = "k2_inference_scores.csv"
CACHE_DIR = Path("k2_cache")

import logging
logging.basicConfig(level=logging.INFO)
from src.Classifiers.K2.K2Loader import K2Loader
from src.Classifiers.CnnModel import CnnModel
from src.Classifiers.LargeWindow.LargeWindow_Processor import LargeWindowCnnModel
def main():
            #     cnnModel = CnnModel()
            #     #this creates the .keras model file
            #     #cnnModel.runAstro1DCNN()
            #     # this one caches the star segments in lccache and then does star-based prediction
            #     # the star scores are saved to starwise_score_1dcnn.csv
            #     # the better the scrore implies the higher the chance of an exoplanet transit
                

            #     # cnnModel.runStarbased1DCNN()
            #     # this one creates star embeddings and saves to star_embeddings_1dcnn.npz
            #    # cnnModel.runStarVecEmbeddings()
            #    #-------- purely for running tests on the saved embeddings and star scores
            #     #cnnModel.runTestOnStarVecEmbeddings()
            #     #-------- Binary classifier on top of star embeddings
            #     cnnModel.runBinaryEmbeddingsClassifier()
        #largeWindowMain()




       k2Processors()

def k2Processors():
    # df = pd.read_csv("k2_inference_scores.csv")
    # err = df[df["status"].astype(str).str.lower() == "error"]

    # # show top error types (first 120 chars of message)
    # print(err["error"].astype(str).str[:120].value_counts().head(20))
    # print("Errors:", len(err), "out of", len(df))
    # from pathlib import Path
    # cache = Path("k2_cache")
    # bad = 0
    # for fp in cache.rglob("*.fits*"):
    #     try:
    #         if fp.is_file() and fp.stat().st_size == 65536:
    #             fp.unlink()
    #             bad += 1
    #     except Exception:
    #         pass

    # print("Deleted truncated 64KB files:", bad)
    # from astropy.config import paths
    # print("Astropy cache:", paths.get_cache_dir())
    loader = K2Loader()
    #loader.callK2_LoadData()    
    loader.score_runner()   

def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)




if __name__ == "__main__":
    main()