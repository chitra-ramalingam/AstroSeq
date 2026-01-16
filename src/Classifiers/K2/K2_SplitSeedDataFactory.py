from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np

# Import your builder + configs
from src.Classifiers.K2.K2_Dataset_builder import (
     K2SegmentDatasetBuilder, PreprocessConfig, InjectionConfig
 )

@dataclass(frozen=True)
class SplitSeedSpec:
    split_seed: int
    out_dir: Path

@dataclass(frozen=True)
class BuiltSplitPaths:
    X_train: Path
    meta_train: Path
    X_val: Path
    meta_val: Path
    X_test: Path
    meta_test: Path

class K2SplitSeedDatasetFactory:
    """
    Builds train/val/test datasets for multiple split seeds.

    Key rules:
    - Each split_seed gets its own out_dir (prevents cache collisions).
    - EPIC IDs are shuffled deterministically using split_seed.
    - Injection RNG can be tied to split_seed for reproducible but distinct injections.
    """

    def __init__(
        self,
        base_out_root: str | Path,
        window_len: int = 512,
        stride: int = 256,
        quality_bitmask: str = "none",
        provenance_priority: Tuple[str, ...] = ("K2", "EVEREST", "K2SFF"),
        banned_provenance: Tuple[str, ...] = ("K2SC", "K2VARCAT"),
        preprocess_cfg=None,
        inject_cfg=None,
        verbose: bool = True,
        injection_seed_offset: bool = True,
    ) -> None:
        self.base_out_root = Path(base_out_root)
        self.base_out_root.mkdir(parents=True, exist_ok=True)

        self.window_len = int(window_len)
        self.stride = int(stride)
        self.quality_bitmask = str(quality_bitmask)
        self.provenance_priority = tuple(provenance_priority)
        self.banned_provenance = tuple(banned_provenance)
        self.preprocess_cfg = preprocess_cfg
        self.inject_cfg = inject_cfg
        self.verbose = bool(verbose)

        # If True—each dataset seed uses (base_injection_seed + split_seed)
        # so your injections/pos-star choices differ per split_seed deterministically.
        self.injection_seed_offset = bool(injection_seed_offset)

    def build_for_seed(self, epic_ids: List[str], split_seed: int) -> BuiltSplitPaths:
        epic_ids = list(epic_ids)

        # 1) deterministically shuffle ids for this split_seed
        rng = np.random.default_rng(int(split_seed))
        ids_shuf = list(rng.permutation(epic_ids))

        # 2) compute train/val/test lists
        # Uses your existing splitting logic, but now on shuffled IDs
        builder = self._make_builder(split_seed=split_seed)
        train_ids, val_ids, test_ids = builder.split_epics_min(ids_shuf)

        if self.verbose:
            print(f"[seed{split_seed}] epics: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

        # 3) build splits to disk
        Xtr, mtr = builder.build_split(train_ids, "train")
        Xva, mva = builder.build_split(val_ids, "val")
        Xte, mte = builder.build_split(test_ids, "test")

        return BuiltSplitPaths(
            X_train=Path(Xtr), meta_train=Path(mtr),
            X_val=Path(Xva), meta_val=Path(mva),
            X_test=Path(Xte), meta_test=Path(mte),
        )

    def build_many(self, epic_ids: List[str], split_seeds: Iterable[int]) -> Dict[int, BuiltSplitPaths]:
        out: Dict[int, BuiltSplitPaths] = {}
        for s in split_seeds:
            out[int(s)] = self.build_for_seed(epic_ids, int(s))
        return out

    def _make_builder(self, split_seed: int):
        # Create an isolated output folder per split_seed
        out_dir = self.base_out_root / f"seed{int(split_seed)}"

        # Clone injection config and optionally offset RNG seed per split_seed
        inj_cfg = self.inject_cfg
        if inj_cfg is not None and getattr(inj_cfg, "enabled", False) and self.injection_seed_offset:
            # shallow copy via dataclass replace-style
            from dataclasses import replace
            base = int(getattr(inj_cfg, "rng_seed", 42))
            inj_cfg = replace(inj_cfg, rng_seed=base + int(split_seed))

        return K2SegmentDatasetBuilder(
            out_dir=out_dir,
            window_len=self.window_len,
            stride=self.stride,
            quality_bitmask=self.quality_bitmask,
            provenance_priority=self.provenance_priority,
            banned_provenance=self.banned_provenance,
            preprocess_cfg=self.preprocess_cfg,
            inject_cfg=inj_cfg,
            verbose=self.verbose,
        )
