import pandas as pd
from pathlib import Path
from src.Classifiers.K2.K2CampaignSource import K2CampaignEpicSource
from src.Classifiers.K2.K2_EnsembleEvaluation import K2EnsembleEvaluator
from src.Classifiers.K2.K2_Dataset_builder import K2SegmentDatasetBuilder, InjectionConfig, PreprocessConfig
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig


class K2_inferenceBuilder:
    def __init__(self):
        pass

    def _build_c5_infer_dataset(self, limit: int | None = None):
        src = K2CampaignEpicSource(campaign=5)
        epics = src.fetch_epic_ids(prefix=True)   # keep as-is (same population behavior)
        if limit is not None:
            epics = epics[:limit]

        out_dir = Path("splits/infer_c5")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "epics_used.txt").write_text("\n".join(epics) + "\n", encoding="utf-8")
        print("epics:", len(epics), "saved to", out_dir / "epics_used.txt")

        builder = K2SegmentDatasetBuilder(
            out_dir=out_dir,
            window_len=512,
            stride=256,
            preprocess_cfg=PreprocessConfig(),
            inject_cfg=InjectionConfig(enabled=False, rng_seed=42),  # <- OFF
            verbose=True,
        )

        X_path, meta_path = builder.build_split(epics, split_name="infer")
        print("Wrote:", X_path, meta_path)
        return X_path, meta_path

    # stage it first (recommended), then go full
    # X_path, meta_path = build_c5_infer_dataset(limit=5000)
    # X_path, meta_path = build_c5_infer_dataset(limit=10000)

    def runEvals(self):
        X_path, meta_path = self._build_c5_infer_dataset(limit=None)  # full run (~25k)

        models = [
            "models/k2_nocrop_flux_seed46_split101.best.keras",
            "models/k2_nocrop_flux_seed46_split202.best.keras",
            "models/k2_nocrop_flux_seed46_split303.best.keras",
        ]

        trainer = K2TransitTrainerV2(TrainConfig(seed=46), verbose=False)
        ens = K2EnsembleEvaluator(trainer=trainer, model_paths=models)

        res = ens.predict_ensemble(X_path=X_path, meta_path=meta_path)
        df = res.meta  # contains p_ens and meta columns
        print("rows:", len(df), "unique stars:", df["star_id"].nunique())
        # Best window per star
        best_per_star = (
            df.sort_values("p_ens", ascending=False)
            .groupby("star_id", as_index=False)
            .head(1)
            .sort_values("p_ens", ascending=False)
        )

        best_per_star.insert(0, "rank", range(1, len(best_per_star) + 1))
        best_per_star[["rank","star_id","p_ens","start","end","seg_mid_time","provenance"]].to_csv(
            "candidates_c5_best_per_star.csv", index=False
        )

        # Top 3 windows per star (useful for inspection)
        top3_per_star = (
            df.sort_values("p_ens", ascending=False)
            .groupby("star_id", as_index=False)
            .head(3)
        )

        top3_per_star.to_csv("candidates_c5_top3_windows_per_star.csv", index=False)

        print(best_per_star.head(20)[["rank","star_id","p_ens","start","end","seg_mid_time"]])
