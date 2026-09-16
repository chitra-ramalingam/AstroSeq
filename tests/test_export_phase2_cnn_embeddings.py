from __future__ import annotations

import numpy as np
import pandas as pd
import unittest
from pathlib import Path

from scripts.export_phase2_cnn_embeddings import aggregate_star_outputs, eligible_epics


class ExportEmbeddingTests(unittest.TestCase):
    def test_eligible_epics_requires_safe_and_accessible(self) -> None:
        labels = pd.DataFrame({
            "epic_id": ["EPIC_1", "EPIC_2", "EPIC_3"],
            "safe_for_supervised_training": [True, True, False],
            "has_accessible_light_curve_tensor": [True, False, True],
        })
        self.assertEqual(eligible_epics(labels), ["EPIC_1"])

    def test_aggregate_uses_probability_aligned_segment_embedding(self) -> None:
        meta = pd.DataFrame({
            "star_id": ["EPIC_1", "EPIC_1", "EPIC_2"],
            "start": [0, 256, 0], "end": [512, 768, 512], "seg_mid_time": [1.0, 2.0, 3.0],
        })
        probabilities = np.array([0.2, 0.9, 0.4], dtype=np.float32)
        embeddings = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
        scores, vectors = aggregate_star_outputs(meta, probabilities, embeddings)
        self.assertEqual(scores["epic_id"].tolist(), ["EPIC_1", "EPIC_2"])
        self.assertTrue(np.allclose(scores["cnn_probability"], [0.9, 0.4]))
        self.assertTrue(np.allclose(vectors, [[2, 0], [3, 0]]))

    def test_generated_embedding_bundle_contract_when_present(self) -> None:
        path = Path("data/phase2/phase2_cnn_embeddings.npz")
        if not path.exists():
            self.skipTest("generated audit artifact not present")
        bundle = np.load(path)
        self.assertEqual(bundle["embedding"].shape, (268, 128))
        self.assertEqual(len(set(bundle["epic_id"].astype(str))), 268)
        self.assertEqual(str(bundle["embedding_layer"]), "global_average_pooling1d_2")


if __name__ == "__main__":
    unittest.main()
