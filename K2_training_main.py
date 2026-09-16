from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.Classifiers.K2.K2CampaignSource import K2CampaignEpicSource
from src.Classifiers.K2.K2_Dataset_builder import InjectionConfig, PreprocessConfig
from src.Classifiers.K2.K2_SplitSeedDataFactory import BuiltSplitPaths, K2SplitSeedDatasetFactory
from src.Classifiers.K2.K2_trainer import K2TransitTrainerV2, TrainConfig


DEFAULT_SPLIT_SEED = 303
DEFAULT_TRAIN_SEED = 46
DEFAULT_LIMIT = 3000
DEFAULT_SPLIT_ROOT = Path("splits")
DEFAULT_MODEL_PATH = Path("models/k2_nocrop_flux_seed46_split303.keras")


def split_paths(split_root: Path, split_seed: int) -> BuiltSplitPaths:
    base = split_root / f"seed{split_seed}"
    return BuiltSplitPaths(
        X_train=base / "X_train.npy",
        meta_train=base / "meta_train.parquet",
        X_val=base / "X_val.npy",
        meta_val=base / "meta_val.parquet",
        X_test=base / "X_test.npy",
        meta_test=base / "meta_test.parquet",
    )


def all_paths_exist(paths: BuiltSplitPaths) -> bool:
    return all(Path(p).exists() for p in _iter_split_paths(paths))


def _iter_split_paths(paths: BuiltSplitPaths) -> Iterable[Path]:
    yield paths.X_train
    yield paths.meta_train
    yield paths.X_val
    yield paths.meta_val
    yield paths.X_test
    yield paths.meta_test


def build_seed303_split_if_needed(
    *,
    split_root: Path,
    split_seed: int,
    limit: int,
    rebuild: bool,
    verbose: bool,
) -> BuiltSplitPaths:
    paths = split_paths(split_root=split_root, split_seed=split_seed)
    if all_paths_exist(paths) and not rebuild:
        print(f"Using existing split tensors in {split_root / f'seed{split_seed}'}")
        return paths

    src = K2CampaignEpicSource(campaign=5)
    fetched = src.fetch_epic_ids(prefix=True)

    # This preserves the original main.py behavior used for the split-seed run.
    epics = sorted(fetched)[: int(limit)]
    src.save_epics_list(epics=epics, out_path=split_root / "epics_used.txt")

    factory = K2SplitSeedDatasetFactory(
        base_out_root=split_root,
        window_len=512,
        stride=256,
        preprocess_cfg=PreprocessConfig(),
        inject_cfg=InjectionConfig(enabled=True, rng_seed=42),
        verbose=verbose,
        injection_seed_offset=False,
    )
    return factory.build_for_seed(epics, split_seed=split_seed)


def train_k2_nocrop_flux_seed46_split303(
    *,
    paths: BuiltSplitPaths,
    model_path: Path,
    train_seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool,
) -> Path:
    trainer = K2TransitTrainerV2(
        TrainConfig(
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            seed=int(train_seed),
            crop_len=None,
        ),
        verbose=verbose,
    )

    trainer.train(
        X_train_path=paths.X_train,
        meta_train_path=paths.meta_train,
        X_val_path=paths.X_val,
        meta_val_path=paths.meta_val,
        X_test_path=paths.X_test,
        meta_test_path=paths.meta_test,
        out_model_path=model_path,
    )
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the K2 no-crop flux-only seed46 split303 training run. "
            "The best checkpoint is written beside the model as .best.keras."
        )
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Rebuild splits/seed303 instead of reusing existing tensors.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce script-level verbosity. Keras training still prints progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    print("K2 no-crop flux training")
    print(f"split_seed={args.split_seed} train_seed={args.train_seed}")
    print(f"model_path={args.model_path}")
    print(f"best_checkpoint={args.model_path.with_suffix('.best.keras')}")

    paths = build_seed303_split_if_needed(
        split_root=args.split_root,
        split_seed=args.split_seed,
        limit=args.limit,
        rebuild=args.rebuild_split,
        verbose=verbose,
    )

    train_k2_nocrop_flux_seed46_split303(
        paths=paths,
        model_path=args.model_path,
        train_seed=args.train_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
