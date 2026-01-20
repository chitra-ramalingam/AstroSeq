from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List
import csv
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig
@dataclass(frozen=True)
class SplitPaths:
    X_val: Path
    meta_val: Path


class ValSplitSweep:
    """
    Evaluate ONE frozen model across multiple validation splits.
    """

    def __init__(
        self,
        trainer: "K2TransitTrainerV2",
        model_path: str | Path,
        split_root: str | Path,
        split_seeds: Iterable[int],
    ) -> None:
        self.trainer = trainer
        self.model_path = Path(model_path)
        self.split_root = Path(split_root)
        self.split_seeds = list(split_seeds)

    def _paths_for_seed(self, split_seed: int) -> SplitPaths:
        d = self.split_root / f"seed{split_seed}"
        return SplitPaths(
            X_val=d / "X_val.npy",
            meta_val=d / "meta_val.parquet",
        )

    def run(self, out_csv: str | Path) -> List[Dict[str, Any]]:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split_seed", "val_pr_auc", "val_top50", "n", "pos", "pos_frac"],
            )
            writer.writeheader()

            for s in self.split_seeds:
                paths = self._paths_for_seed(s)
                if not paths.X_val.exists() or not paths.meta_val.exists():
                    raise FileNotFoundError(f"Missing split files for seed {s}: {paths}")

                metrics = self.trainer.evaluate_pretrained(
                    model_path=self.model_path,
                    X_path=paths.X_val,
                    meta_path=paths.meta_val,
                    split_name="val",
                )

                row = {
                    "split_seed": s,
                    "val_pr_auc": metrics["val_pr_auc"],
                    "val_top50": metrics["val_top50"],
                    "n": metrics["n"],
                    "pos": metrics["pos"],
                    "pos_frac": metrics["pos_frac"],
                }
                writer.writerow(row)
                rows.append(row)

        return rows
    
    def train_on_split(self, split_seed: int):
        base = Path(f"splits/seed{split_seed}")
        tnr = K2TransitTrainerV2(TrainConfig(seed=46), verbose=True)

        out = Path("models") / f"k2_nocrop_flux_seed46_split{split_seed}.keras"

        tnr.train(
            X_train_path=base/"X_train.npy", meta_train_path=base/"meta_train.parquet",
            X_val_path=base/"X_val.npy",     meta_val_path=base/"meta_val.parquet",
            X_test_path=base/"X_test.npy",   meta_test_path=base/"meta_test.parquet",
            out_model_path=out,
        )

