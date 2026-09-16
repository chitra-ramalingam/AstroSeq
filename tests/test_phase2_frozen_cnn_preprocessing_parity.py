from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from src.Classifiers.K2.K2_FrozenCnnPreprocessing import FrozenK2CNNTensorBuilder


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/phase2/phase2_feature_generation_manifest.parquet"
TRUSTED_X = ROOT / "splits/infer_c5/X_infer.npy"
TRUSTED_META = ROOT / "splits/infer_c5/meta_infer.parquet"
TRUSTED_HASHES = {
    "EPIC_200008725": "b3880cffb6adf61c076396e70aa51a934220654f26995196a6a0ed3975bcdd15",
    "EPIC_200008785": "bdba1b513c2cc21f628a778e16a55f416d7cc758f9013d765806ec9d1915115e",
    "EPIC_200008831": "735364c01a2338bb5281aab6b5e348e3658a3804844d98cb413b625cae06af15",
}


def numeric_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def fits_time_flux(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names or [])
        flux_column = next((name for name in ("FLUX", "PDCSAP_FLUX", "SAP_FLUX") if name in names), None)
        if flux_column is None:
            raise ValueError(f"no supported flux column in {path}")
        return (
            np.asarray(data["TIME"], dtype=np.float64).reshape(-1),
            np.asarray(data[flux_column], dtype=np.float32).reshape(-1),
        )


class FrozenCNNPreprocessingUnitTests(unittest.TestCase):
    def test_time_cleaning_drops_only_invalid_time(self) -> None:
        time = np.array([1.0, np.nan, 2.0, 3.0])
        flux = np.array([10.0, 20.0, np.nan, 40.0], dtype=np.float32)
        cleaned_time, cleaned_flux = FrozenK2CNNTensorBuilder.clean_time_cadences(time, flux)
        np.testing.assert_array_equal(cleaned_time, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(cleaned_flux, [10.0, np.nan, 40.0])

    def test_historical_small_scale_is_normalized_not_zeroed(self) -> None:
        builder = FrozenK2CNNTensorBuilder()
        phase = np.linspace(0.0, 8.0 * np.pi, 1024, dtype=np.float32)
        relative_flux = 1.0 + 1e-3 * np.sin(phase)
        standardized = builder.standardize_flux(relative_flux)
        self.assertGreater(float(np.std(standardized)), 0.5)
        self.assertFalse(np.all(standardized == 0.0))
        self.assertLessEqual(float(np.max(np.abs(standardized))), 10.0)

    def test_segmentation_discards_tail_without_padding(self) -> None:
        builder = FrozenK2CNNTensorBuilder()
        time = np.arange(1025, dtype=np.float64)
        flux = np.arange(1025, dtype=np.float32)
        tensor, starts, ends, mids = builder.segment(time, flux)
        self.assertEqual(tensor.shape, (3, 512, 2))
        np.testing.assert_array_equal(starts, [0, 256, 512])
        np.testing.assert_array_equal(ends, [512, 768, 1024])
        np.testing.assert_array_equal(tensor[:, 0, 1], 0.0)
        self.assertEqual(float(tensor[-1, -1, 0]), 1023.0)
        self.assertNotIn(1024.0, tensor[:, :, 0])
        np.testing.assert_allclose(mids, [255.5, 511.5, 767.5])


@unittest.skipUnless(
    MANIFEST.exists() and TRUSTED_X.exists() and TRUSTED_META.exists(),
    "trusted Phase 2 reuse artifacts are not present",
)
class FrozenCNNTrustedTensorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FrozenK2CNNTensorBuilder()
        cls.manifest = pd.read_parquet(MANIFEST)
        cls.manifest["epic_id"] = cls.manifest["epic_id"].astype(str)
        cls.meta = pd.read_parquet(TRUSTED_META)
        cls.meta["star_id"] = cls.meta["star_id"].astype(str)
        cls.tensor = np.load(TRUSTED_X, mmap_mode="r")

    def test_phase2_builder_exactly_reproduces_trusted_reused_tensors(self) -> None:
        for epic, expected_hash in TRUSTED_HASHES.items():
            with self.subTest(epic=epic):
                row = self.manifest.loc[self.manifest["epic_id"].eq(epic)].iloc[0]
                self.assertEqual(str(row["tensor_action"]), "reuse_existing")
                paths = [Path(value) for value in json.loads(str(row["local_light_curve_path"]))]
                if len(paths) != 1 or not paths[0].exists():
                    self.skipTest(f"trusted local FITS is unavailable for {epic}")

                time, flux = fits_time_flux(paths[0])
                actual, starts, ends, mid_times = self.builder.build_tensor(time, flux)

                indices = self.meta.index[self.meta["star_id"].eq(epic)].to_numpy(dtype=int)
                expected = np.asarray(self.tensor[indices], dtype=np.float32)
                expected_meta = self.meta.loc[indices]

                self.assertEqual(numeric_sha256(expected), expected_hash)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(starts, expected_meta["start"].to_numpy(np.int64))
                np.testing.assert_array_equal(ends, expected_meta["end"].to_numpy(np.int64))
                np.testing.assert_array_equal(mid_times, expected_meta["seg_mid_time"].to_numpy(float))


if __name__ == "__main__":
    unittest.main()
