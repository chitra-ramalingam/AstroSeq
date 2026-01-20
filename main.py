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
from src.Classifiers.K2.K2_PrintModelOutputs import K2PrintModelOutputs
from src.Classifiers.K2.K2_SplitSeedSweep import ValSplitSweep
from src.Classifiers.K2.K2_SplitSeedDataFactory import K2SplitSeedDatasetFactory
from src.Classifiers.K2.K2_EnsembleEvaluation import K2EnsembleEvaluator
from src.Classifiers.K2.K2_InferenceBuilder import K2_inferenceBuilder

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
        #K2_Analysis()
        K2_ModelCreationAndTraining_Printing()


def K2_ModelCreationAndTraining_Printing():
    #triageCandidates()
    #printdata = K2PrintModelOutputs()
    #printdata.loadall("k2_dataset_centered_v4")
    #k2_model_splitting_eval()
    #K2_trainsplits()
    #K2_printEnsembleEvals()
    k2_RunFinalAll()
    
def k2_model_splitting_eval():
    src = K2CampaignEpicSource(campaign=5)
    fetched = src.fetch_epic_ids(prefix=True) 
    epics = sorted(fetched)[:3000]
    src.save_epics_list(epics=epics, out_path="splits/epics_used.txt")

    Path("splits/epics_used.txt").write_text("\n".join(epics), encoding="utf-8")
    factory = K2SplitSeedDatasetFactory(
        base_out_root="splits",
        window_len=512,
        stride=256,
        preprocess_cfg=PreprocessConfig(),
        inject_cfg=InjectionConfig(enabled=True, rng_seed=42),
        verbose=True,
        injection_seed_offset=False,
        )

    paths_by_seed = factory.build_many(epics, split_seeds=[101, 202, 303])
    print(paths_by_seed[101].X_val, paths_by_seed[101].meta_val)
    trainer = K2TransitTrainerV2(TrainConfig(seed=46), verbose=True)
    sweep = ValSplitSweep(
        trainer=trainer,
        model_path="models/k2_nocrop_flux_seed46.best.keras",
        split_root="splits",
        split_seeds=[101, 202, 303],
    )

    sweep.run(out_csv="split_seed_sweep/seed46_eval_only.csv")

def K2_trainsplits():
    trainer = K2TransitTrainerV2(TrainConfig(seed=46), verbose=True)
    sweep = ValSplitSweep(
        trainer=trainer,
        model_path="models/k2_nocrop_flux_seed46.best.keras",
        split_root="splits",
        split_seeds=[101, 202, 303],
    )
    
    for s in [101, 202, 303]:
      sweep.train_on_split(s)

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
    dataDir ="k2_dataset_centered_v4"

    builder = K2SegmentDatasetBuilder(
        out_dir=dataDir,
        window_len=512,
        stride=128,
        preprocess_cfg=PreprocessConfig(use_flatten=True),
        inject_cfg=InjectionConfig(enabled=True,
                                    positive_star_fraction=0.2),
    )
    builder.max_products_per_star = 1
    #builder.download_dir = None  # or point to your existing Lightkurve cache root
    builder.remote_timeout = 120

    epics_small = epics[:3000]  # or 1000 if you feel brave

    train_ids, val_ids, test_ids = builder.split_epics_min(epics_small)

    # if Path(f"{dataDir}/X_train.npy").exists():
    #     print("Dataset already exists, skipping build.")
    # else:
    builder.build_split(train_ids, "train")
    builder.build_split(val_ids, "val")
    builder.build_split(test_ids, "test")
    for split in ["train", "val", "test"]:
        m = pd.read_parquet(f"{dataDir}/meta_{split}.parquet")
        n_pos = int(m["label"].sum())
        n_tot = len(m)
        print(split, "pos", n_pos, "tot", n_tot, "pos_frac", n_pos/n_tot)

    #---------
    for split in ["train", "val", "test"]:
        m = pd.read_parquet(f"{dataDir}/meta_{split}.parquet")
        n_pos = int(m["label"].sum())
        n_tot = len(m)
        print(split, "pos", n_pos, "tot", n_tot, "pos_frac", n_pos/n_tot)

    for seed in [42, 43, 44, 45, 46]:
        trainer = K2TransitTrainerV2(
            TrainConfig(
                epochs=50,          # 50 is enough with EarlyStopping
                batch_size=256,
                lr=1e-4,
                seed=seed,
                crop_len=None,      # keep the “no-crop” baseline
            )
        )

        trainer.train(
            f"{dataDir}/X_train.npy", f"{dataDir}/meta_train.parquet",
            f"{dataDir}/X_val.npy",   f"{dataDir}/meta_val.parquet",
            f"{dataDir}/X_test.npy",  f"{dataDir}/meta_test.parquet",
            out_model_path=f"models/k2_nocrop_flux_seed{seed}.keras",
        )

def K2_printEnsembleEvals():
    models = [
        "models/k2_nocrop_flux_seed46_split101.best.keras",
        "models/k2_nocrop_flux_seed46_split202.best.keras",
        "models/k2_nocrop_flux_seed46_split303.best.keras",
    ]

    trainer = K2TransitTrainerV2(TrainConfig(seed=46), verbose=False)

    ens = K2EnsembleEvaluator(trainer=trainer, model_paths=models)
    print("Ensemble Eval..")
    metrics = ens.evaluate(
        X_path="splits/seed303/X_val.npy",
        meta_path="splits/seed303/meta_val.parquet",
        k=50,
    )

    print(metrics)
    res = ens.predict_ensemble(
        X_path="splits/seed303/X_val.npy",
        meta_path="splits/seed303/meta_val.parquet",
    )

    df = res.meta

    # Keep best window per star
    best_per_star = (
        df.sort_values("p_ens", ascending=False)
        .groupby("star_id", as_index=False)
        .head(1)
        .sort_values("p_ens", ascending=False)
    )

    best_per_star.head(50).to_csv("candidates_top50.csv", index=False)
    print("Wrote candidates_top50.csv")
    print(best_per_star[["star_id","p_ens","start","end","seg_mid_time"]].head(20))
    cols = ["star_id","p_ens","label_star","label","start","end","seg_mid_time"]
    print(best_per_star[cols].head(50))

    print("Top50 star-level precision (label_star):",
        float(best_per_star.head(50)["label_star"].mean()))

    print("Top50 window-level precision (label):",
        float(best_per_star.head(50)["label"].mean()))
    
def k2_RunFinalAll():
    inf  = K2_inferenceBuilder()
    inf.runEvals()


def printValues():
    printK2 = K2_PrintSets()
    printK2.print_meta_test()
    printK2.print_preds()
    #printK2.print_eval_report("k2_window1024_v3_hardnegW2.keras")

def K2_Analysis() :
    MODEL = "k2_w1024_c05_center015_cov070_base.keras"
    DATA = "k2_dataset_centered_v2"

    k2Analy = K2_PrintAnalysis()
    model_path = MODEL
    #k2Analy.print_eval_report(model_path)
    #k2Analy.save_galleries(MODEL, split="test", mode="stacked", downsample=4, n=25)
    #k2Analy.save_galleries(MODEL, split="test", mode="single", channel=0, downsample=4, n=25)
    #k2Analy.save_galleries(MODEL, split="test", mode="single", channel=1, downsample=4, n=25)
    # k2Analy.save_galleries(
    # model_path=model_path,
    # split="test",
    # plot_mode="stacked",   # best for 2-channel debugging
    # n=25,
    # )
   
  
    #k2Analy.plot_star_top_windows(MODEL, DATA, "EPIC_212160557", out_png="galleries/badboy_EPIC_212160557.png")
    #k2Analy.plot_star_top_windows(MODEL, DATA, "EPIC_228682495", out_png="galleries/hero_EPIC_228682495.png")
    k2Analy.furtherPrints("k2_dataset_centered_v2/X_test.npy", "k2_dataset_centered_v2/meta_test.parquet")
    k2Analy.save_galleries(
    model_path=model_path,
    data_dir=DATA,
    out_dir="galleries",
    split="val",
    mode="ch1",
    bin=8,
    center_frac=0.35,   # try this; it makes “centered transit” actually visible
    )
    k2Analy.save_galleries(
        model_path=model_path,
        out_dir="galaries",
        split="test",
        mode="ch1",
        bin=8,
        center_frac=0.35
    )




if __name__ == "__main__":
    main()