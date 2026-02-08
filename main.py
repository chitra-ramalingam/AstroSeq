import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from src.Classifiers.K2.K2_PrintSets import K2_PrintSets

from src.Classifiers.Triages.AstroSeqTriageCandidate import AstroSeqCandidateTriage
from src.Classifiers.Triages.K2_Score_loader import K2ScoreLoader, SegmentFilterConfig
from src.Classifiers.K2.K2_Dataset_builder import K2SegmentDatasetBuilder, InjectionConfig, PreprocessConfig
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig
from src.Classifiers.K2.K2CampaignSource import K2CampaignEpicSource
from src.Classifiers.K2.Analysis.K2_PrintAnalysis import K2_PrintAnalysis
from src.Classifiers.K2.K2_PrintModelOutputs import K2PrintModelOutputs
from src.Classifiers.K2.K2_SplitSeedSweep import ValSplitSweep
from src.Classifiers.K2.K2_SplitSeedDataFactory import K2SplitSeedDatasetFactory
from src.Classifiers.K2.K2_EnsembleEvaluation import K2EnsembleEvaluator
from src.Classifiers.K2.K2_InferenceBuilder import K2_inferenceBuilder
from src.Classifiers.K2.K2PlotStars import K2PlotStars
from src.Classifiers.K2.Systematics.K2NoiseLoader import K2NoiseLoader, K2NoiseLoaderConfig, K2NoiseConfig
from src.Classifiers.K2.Systematics.K2_PeriodValidator import K2PeriodValidator
from src.Classifiers.K2.Systematics.K2Validation_Prediction import K2Validation_Prediction

CSV_PATH = "k2_inference_scores.csv"
CACHE_DIR = Path("k2_cache")

import logging
logging.basicConfig(level=logging.INFO)
from src.Classifiers.K2.K2Loader import K2Loader
from src.Classifiers.CnnModel import CnnModel
from src.Classifiers.LargeWindow.LargeWindow_Processor import LargeWindowCnnModel


def validate_period_by_prediction(
    time: np.ndarray,
    resid: np.ndarray,
    local_sigma: Optional[np.ndarray],
    events_df: pd.DataFrame,
    P: float,
    t0: Optional[float] = None,
    tol_days: float = 0.08,
    sigma_floor: Optional[float] = None,
    snr_threshold: float = 3.0,
    max_rows: int = 6,
    do_plot: bool = True,
) -> Dict[str, Any]:
    validator = K2Validation_Prediction()
    return validator.validate_period_by_prediction(
        time=time,
        resid=resid,
        local_sigma=local_sigma,
        events_df=events_df,
        P=P,
        t0=t0,
        tol_days=tol_days,
        sigma_floor=sigma_floor,
        snr_threshold=snr_threshold,
        max_rows=max_rows,
        do_plot=do_plot,
    )


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
        #K2_ModelCreationAndTraining_Printing()
        #k2_noise_loader_runner()
        #k2_plot_star_from_candidates("211797674")

        k2DomainTransit()

def k2DomainTransit():
    from src.Classifiers.K2.K2_TimeDomainTransitPipeline import (
        K2TimeDomainTransitPipeline,
        phase_cluster_score,
    )

    pipe = K2TimeDomainTransitPipeline()
    result = pipe.run_one("EPIC 211797674")
    summary = result["summary"]
    candidates = result["candidates"]  # list of ranked time-domain candidates
    print(pd.DataFrame([summary]).to_string(index=False))
    if candidates:
        cand_df = pd.DataFrame(candidates)
        cols = [
            "query", "author", "shape_score", "depth_snr", "duration_cadences",
            "symmetry", "curvature", "continuity", "t_start", "t_end", "t_mid",
            "start_idx", "end_idx", "window_start", "window_end",
        ]
        show_cols = [c for c in cols if c in cand_df.columns]
        print(cand_df.reindex(columns=show_cols).head(10).to_string(index=False))

        fetched = pipe.loader.handler.fetch_best(
            query="EPIC 211797674",
            limit=pipe.loader.loader_config.limit,
            exptime=None,
        )
        cleaned = pipe.loader.handler.clean(
            fetched["lc"],
            normalize=False,
            remove_nans=True,
            quality_mask=True,
            sigma_clip=False,
            flatten=False,
        )
        time = np.asarray(cleaned["time"], dtype=float)
        flux = np.asarray(cleaned["flux"], dtype=float)
        validator = K2PeriodValidator(
            detector=pipe,
            tol_days=0.12,
            min_duration_cadences=3,
            shape_threshold=0.6,
            snr_threshold=4.0,
        )

        period_grid = [1.2668, 1.254132, 1.279468]
        phase_rows: List[Dict[str, Any]] = []
        print("\nPhase-cluster scan:")
        for p in period_grid:
            cluster_count, cluster_center_phase, in_cluster_indices = phase_cluster_score(
                cand_df,
                P=p,
                tol_phase=0.03,
            )
            print(
                f"P={p:.8f} cluster_count={cluster_count} "
                f"cluster_center_phase={cluster_center_phase:.6f} "
                f"in_cluster_indices={in_cluster_indices}"
            )
            phase_rows.append(
                {
                    "P": float(p),
                    "cluster_count": int(cluster_count),
                    "cluster_center_phase": float(cluster_center_phase),
                    "in_cluster_indices": list(in_cluster_indices),
                }
            )

        phase_df = pd.DataFrame(phase_rows).sort_values(["cluster_count", "P"], ascending=[False, True]).reset_index(drop=True)
        top_k = min(3, len(phase_df))
        top_phase = phase_df.head(top_k).copy()
        print("\nTop periods by phase clustering:")
        print(top_phase[["P", "cluster_count", "cluster_center_phase"]].to_string(index=False))

        out_dir = Path("plots/period_validation")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        print("\nDetector-consistent period validation:")
        for row in top_phase.itertuples(index=False):
            p = float(row.P)
            in_cluster_indices = list(row.in_cluster_indices)

            t0_guess: Optional[float] = None
            if len(in_cluster_indices) > 0 and "shape_score" in cand_df.columns and "t_mid" in cand_df.columns:
                cluster_df = cand_df.loc[cand_df.index.intersection(in_cluster_indices)].copy()
                if not cluster_df.empty and cluster_df["shape_score"].notna().any():
                    best_idx = pd.to_numeric(cluster_df["shape_score"], errors="coerce").idxmax()
                    t_mid_ref = float(pd.to_numeric(cluster_df.loc[best_idx, "t_mid"], errors="coerce"))
                    t_min = float(np.nanmin(time)) if np.any(np.isfinite(time)) else float("nan")
                    if np.isfinite(t_mid_ref) and np.isfinite(t_min):
                        t0_guess = float(t_mid_ref - np.round((t_mid_ref - t_min) / p) * p)

            val = validator.validate(
                time=time,
                flux=flux,
                P=p,
                t0=t0_guess,
                quality_mask=None,
            )
            p_tag = f"{p:.6f}".replace(".", "p")
            hits_csv = out_dir / f"period_{p_tag}_hits.csv"
            misses_csv = out_dir / f"period_{p_tag}_misses.csv"
            uncovered_csv = out_dir / f"period_{p_tag}_uncovered.csv"
            hitmap_png = out_dir / f"period_{p_tag}_hitmap.png"
            phase_png = out_dir / f"period_{p_tag}_phase.png"
            val["hits_df"].to_csv(hits_csv, index=False)
            val["misses_df"].to_csv(misses_csv, index=False)
            val["uncovered_df"].to_csv(uncovered_csv, index=False)
            validator.plot_validation_hitmap(val, hitmap_png)
            validator.plot_scores_vs_phase(val, phase_png, score_col="best_shape_score")

            print(
                f"P={p:.8f} n_pred={val['n_predicted']} n_cov={val['n_covered']} "
                f"cov={val['coverage_rate']:.3f} hit_shape={val['hit_rate_shape']:.3f} "
                f"hit_snr={val['hit_rate_snr']:.3f} mean_shape={val['mean_best_shape']:.3f} "
                f"mean_snr={val['mean_best_snr']:.3f} no_cand={val['n_windows_with_no_candidates']} "
                f"frac_no_cand_dip_gt3={val['frac_no_cand_dip_snr_gt3']:.3f} "
                f"dur_dist_no_cand={val['duration_below_threshold_dist_no_cand']} "
                f"csvs=[{hits_csv.name}, {misses_csv.name}, {uncovered_csv.name}] "
                f"plots=[{hitmap_png.name}, {phase_png.name}]"
            )

            summary_rows.append(
                {
                    "P": float(val["P"]),
                    "t0": float(val["t0"]),
                    "n_predicted": int(val["n_predicted"]),
                    "n_covered": int(val["n_covered"]),
                    "coverage_rate": float(val["coverage_rate"]),
                    "hit_rate_shape": float(val["hit_rate_shape"]),
                    "hit_rate_snr": float(val["hit_rate_snr"]),
                    "mean_best_shape": float(val["mean_best_shape"]),
                    "mean_best_snr": float(val["mean_best_snr"]),
                    "n_windows_with_no_candidates": int(val["n_windows_with_no_candidates"]),
                    "frac_no_cand_dip_snr_gt3": float(val["frac_no_cand_dip_snr_gt3"]),
                }
            )

        if len(summary_rows) > 0:
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ["hit_rate_shape", "hit_rate_snr", "coverage_rate", "mean_best_shape", "mean_best_snr"],
                ascending=[False, False, False, False, False],
            )
            print("\nValidation summary:")
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    else:
        print("No time-domain candidates found.")


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


def k2_noise_loader_runner():
    cfg = K2NoiseLoaderConfig(flatten=False, per_segment=True)
    noise_cfg = K2NoiseConfig(mode="discovery")  # or preset="discovery"
    loader = K2NoiseLoader(loader_config=cfg, noise_config=noise_cfg)
    rows = loader.run_queries(
        queries=["EPIC 211797674"],
        limit=10,
        flatten=False,
        per_segment=True,
    )
    df = loader.to_dataframe(rows)
    cols = [
        "query","status","author","usable",
        "score","score_global","score_best_seg","score_median_seg","score_worst_seg",
        "n_points","baseline_days","why_not_usable","notes"
        ]
    print(df.reindex(columns=[c for c in cols if c in df.columns]).to_string(index=False))
    print(df[["query", "status", "author", "usable", "score", "n_points", "baseline_days"]].head())
    #k2_noise_loader_runner()

def k2_noise_compact_report(query: str = "EPIC 211797674") -> None:
    loader = K2NoiseLoader()
    row = loader.run_one(query=query, limit=10, flatten=False, per_segment=True)

    compact = {
        "query": row.get("query", ""),
        "author": row.get("author", ""),
        "score_global": row.get("score_global", float("nan")),
        "score_best_seg": row.get("score_best_seg", float("nan")),
        "score_median_seg": row.get("score_median_seg", float("nan")),
        "usable": row.get("usable", False),
        "why_not_usable": row.get("why_not_usable", ""),
        "robust_sigma": row.get("robust_sigma", float("nan")),
        "outlier_rate_6sigma": row.get("outlier_rate_6sigma", float("nan")),
        "outlier_rate_global": row.get("outlier_rate_global", float("nan")),
        "step_score": row.get("step_score", float("nan")),
        "whiteness_score": row.get("whiteness_score", float("nan")),
    }
    print(pd.DataFrame([compact]).to_string(index=False))

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
    # 1) Build infer dataset + run ensemble —writes:
    #    - candidates_c5_top3_windows_per_star.csv
    #    - candidates_c5_best_per_star.csv (optional)
    inf = K2_inferenceBuilder()
    inf.runEvals(limit=10_000, out_dir_name="infer_c5_v2_10k")

    plotter = K2PlotStars()

    # 2) Build stable star ranking from top-3 windows per star
    star_rank = plotter.create_star_ranking_from_top3(
        top3_csv="candidates_c5_top3_windows_per_star.csv",
        out_csv="candidates_c5_star_ranking.csv",
    )

    # 3) Vet only the top-N ranked stars (fast) using cached flux
    plotter.vet_top_candidates(
         cache_root="splits/infer_c5_v2_10k/_cache/infer",  # swap to infer_c5_v2 for full
         in_star_csv="candidates_c5_star_ranking.csv",
         out_star_csv="candidates_c5_star_ranking_vetted.csv",
         top_n=2000,
         deep_cut=-7.0,   # more useful than -8 for planet-like filtering
     )

    print("Done — wrote:")
    print("  candidates_c5_top3_windows_per_star.csv")
    print("  candidates_c5_star_ranking.csv")
    print("  candidates_c5_star_ranking_vetted.csv")
    plotter = K2PlotStars()

    cache_root = "splits/infer_c5_v2_10k/_cache/infer"

    for sid in ["EPIC_211327055", "EPIC_211378898", "EPIC_211320303"]:
        plotter.inspect_cached_star(sid, cache_root)




def k2_plot_star_from_candidates(star_id: Union[str, int] = "211797674") -> None:
    # try:
    #     plotter.run()
    
    # except FileNotFoundError as exc:
    #     logging.warning("Unable to rank stars: %s", exc)
    #     return
    #plotter.plot_star_windows(star_id)
    #plotter.plot_star_flux_from_cache("211797674")
    plotter = K2PlotStars(source_csv="candidates_c5_top3_windows_per_star.csv")

    cache_root = "splits/infer_c5_v2_10k/_cache/infer"
    vetted = pd.read_csv("candidates_c5_star_ranking_vetted.csv")

    top = vetted.head(50)["star_id"].tolist()
    for sid in top:
        plotter.plot_star_flux_from_cache(
            sid,
            cache_root=cache_root,
            save_dir="plots/vetted_top50",
            zoom_pad=256,
        )




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
