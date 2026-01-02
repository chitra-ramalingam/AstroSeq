import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from src.Classifiers.K2.K2_PrintSets import K2_PrintSets

from src.Classifiers.Triages.AstroSeqTriageCandidate import AstroSeqCandidateTriage
from src.Classifiers.Triages.K2_Score_loader import K2ScoreLoader, SegmentFilterConfig
import pandas as pd
from src.Classifiers.K2.K2_Dataset_builder import K2SegmentDatasetBuilder, InjectionConfig, PreprocessConfig
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig
from src.Classifiers.K2.K2CampaignSource import K2CampaignEpicSource
from src.Classifiers.K2.Analysis.K2_PrintAnalysis import K2_PrintAnalysis
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
        
        #triageCandidates()
        #printValues()
        K2_Analysis()


def K2_ModelCreationAndTraining_Printing():
    triageCandidates()
    printValues()
     
def k2Processors():
        #############Not used after lots issues needs abandoning
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
    ##################### This didnt really work, need abandon###########
    loader = K2Loader()
    #loader.callK2_LoadData()    
    loader.score_runner()   

def largeWindowMain():
    largeWindowModel = LargeWindowCnnModel()
    #largeWindowModel.build_model(mission="tess",neg_pos_ratio= 3, do_hard_neg=True)
   # largeWindowModel.build_model(mission="kepler",neg_pos_ratio= 7 , do_hard_neg=False)
    largeWindowModel.build_model(mission="k2", neg_pos_ratio=2,do_hard_neg=False)


def triageCandidates():
    # epics = [
    # "EPIC_206317286",
    # "EPIC_206024342",
    # "EPIC_211822797",
    # # ...
    # ]

    src = K2CampaignEpicSource(campaign=5)
    epics = src.fetch_epic_ids(prefix=True)   # ["EPIC_211822797", ...]
    print("N EPICs:", len(epics))
    print(epics[:20])
    dataDir ="k2_dataset_centered_v2"

    builder = K2SegmentDatasetBuilder(
        out_dir=dataDir,
        window_len=1024,
        stride=256,
        preprocess_cfg=PreprocessConfig(use_flatten=True),
        inject_cfg=InjectionConfig(enabled=True, positive_star_fraction=0.2),
    )

    
    train_ids, val_ids, test_ids = builder.split_epics_min(epics)

    # if Path(f"{dataDir}/X_train.npy").exists():
    #     print("Dataset already exists, skipping build.")
    # else:
    builder.build_split(train_ids, "train")
    builder.build_split(val_ids, "val")
    builder.build_split(test_ids, "test")

    #---------
    for split in ["train", "val", "test"]:
        m = pd.read_parquet(f"{dataDir}/meta_{split}.parquet")
        n_pos = int(m["label"].sum())
        n_tot = len(m)
        print(split, "pos", n_pos, "tot", n_tot, "pos_frac", n_pos/n_tot)

    trainer = K2TransitTrainerV2(TrainConfig(epochs=10, batch_size=256, lr=3e-4))

    trainer.train(
        f"{dataDir}/X_train.npy", f"{dataDir}/meta_train.parquet",
        f"{dataDir}/X_val.npy",   f"{dataDir}/meta_val.parquet",
        f"{dataDir}/X_test.npy",  f"{dataDir}/meta_test.parquet",
        out_model_path="k2_window1024_centralized_v2.keras",
    )

def printValues():
    printK2 = K2_PrintSets()
    printK2.print_meta_test()
    printK2.print_preds()
    #printK2.print_eval_report("k2_window1024_v3_hardnegW2.keras")

def K2_Analysis() :
     k2Analy = K2_PrintAnalysis()
     model_path = "k2_w1024_c05_center015_cov070_base.keras"
     k2Analy.print_eval_report(model_path)
     k2Analy.save_galleries(model_path=model_path, split="test", n=25)


if __name__ == "__main__":
    main()