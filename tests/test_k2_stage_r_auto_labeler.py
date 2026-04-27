from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageRAutoLabeler import K2StageRAutoLabeler


class K2StageRAutoLabelerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_r_auto_labeler_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    @staticmethod
    def _event(
        epic_id,
        query: str,
        duration_cadences: float,
        shape_score: float,
        depth: float = 0.001,
        depth_snr: float = 10.0,
    ) -> dict:
        return {
            "epic_id": epic_id,
            "query": query,
            "author": "EVEREST",
            "duration_cadences": duration_cadences,
            "duration_days": duration_cadences * 0.0204,
            "depth": depth,
            "depth_snr": depth_snr,
            "symmetry": 0.8,
            "curvature": 0.7,
            "continuity": 0.7,
            "shape_score": shape_score,
        }

    def test_labels_manual_calibration_examples_and_spike_reject(self) -> None:
        rows = []
        rows.extend(
            [
                self._event("EPIC_212023491", "EPIC 212023491", 5, 0.759, 0.00117),
                self._event("EPIC_212023491", "EPIC 212023491", 7, 0.700, 0.00100),
                self._event("EPIC_212023491", "EPIC 212023491", 2, 0.710, 0.00028),
                self._event("EPIC_212023491", "EPIC 212023491", 2, 0.708, 0.00026),
                self._event("EPIC_212023491", "EPIC 212023491", 2, 0.703, 0.00023),
            ]
        )
        rows.extend(
            [
                self._event("EPIC_211633247", "EPIC 211633247", 5, 0.748, 0.00078),
                *[
                    self._event("EPIC_211633247", "EPIC 211633247", 2, 0.690, 0.00010 + i * 0.00002)
                    for i in range(36)
                ],
                self._event("EPIC_211633247", "EPIC 211633247", 6, 0.700, 0.00040),
                self._event("EPIC_211633247", "EPIC 211633247", 7, 0.705, 0.00050),
                self._event("EPIC_211633247", "EPIC 211633247", 5, 0.710, 0.00060),
                self._event("EPIC_211633247", "EPIC 211633247", 5, 0.715, 0.00070),
                self._event("EPIC_211633247", "EPIC 211633247", 4, 0.720, 0.00080),
            ]
        )
        rows.extend(
            [
                self._event("", "EPIC 211791780", 17, 0.822, 0.00078),
                self._event("", "EPIC 211791780", 18, 0.815, 0.00107),
                self._event("", "EPIC 211791780", 2, 0.690, 0.00030),
            ]
        )
        rows.extend(
            [
                self._event("EPIC_211945111", "EPIC 211945111", 20, 0.765, 0.00064),
                self._event("EPIC_211945111", "EPIC 211945111", 22, 0.753, 0.00086),
                self._event("EPIC_211945111", "EPIC 211945111", 19, 0.744, 0.00043),
            ]
        )
        rows.extend(
            [
                self._event(211836788, "EPIC 211836788", 21, 0.775, 0.00110),
                self._event(211836788, "EPIC 211836788", 24, 0.766, 0.00115),
                self._event(211836788, "EPIC 211836788", 2, 0.670, 0.00042),
                self._event(211836788, "EPIC 211836788", 2, 0.666, 0.00037),
            ]
        )
        rows.extend(
            [
                self._event("EPIC_211529255", "EPIC 211529255", 2, 0.700, 0.00020),
                self._event("EPIC_211529255", "EPIC 211529255", 2, 0.690, 0.00080),
                self._event("EPIC_211529255", "EPIC 211529255", 3, 0.680, 0.00010),
                self._event("EPIC_211529255", "EPIC 211529255", 2, 0.670, 0.00150),
            ]
        )

        labels = K2StageRAutoLabeler.label_events(pd.DataFrame(rows)).set_index("epic_id")

        self.assertEqual(labels.loc["EPIC_212023491", "stage_r_label"], "hold_for_review")
        self.assertTrue(bool(labels.loc["EPIC_212023491", "stage_r_needs_manual_review"]))
        self.assertGreater(labels.loc["EPIC_212023491", "spike_fraction_2cadence"], 0.5)
        self.assertEqual(labels.loc["EPIC_211633247", "stage_r_label"], "reject_as_noise_or_artifact")
        self.assertEqual(labels.loc["EPIC_211791780", "stage_r_label"], "promote_to_deeper_eval")
        self.assertEqual(labels.loc["EPIC_211945111", "stage_r_label"], "promote_to_deeper_eval")
        self.assertEqual(labels.loc["EPIC_211836788", "stage_r_label"], "promote_to_deeper_eval")
        self.assertEqual(labels.loc["EPIC_211529255", "stage_r_label"], "reject_as_noise_or_artifact")

        debug = json.loads(labels.loc["EPIC_211791780", "stage_r_debug_json"])
        self.assertEqual(debug["policy_version"], "stage_r_manual_calibration_v2")
        self.assertEqual(debug["counts"]["n_events_long_good"], 2)
        self.assertTrue(debug["flags"]["promote_rule_passed"])

    def test_run_cli_writes_requested_output_columns(self) -> None:
        input_csv = self.case_dir / "events.csv"
        output_csv = self.case_dir / "stage_r_labels.csv"
        pd.DataFrame(
            [
                self._event(None, "EPIC 211945111", 20, 0.765),
                self._event(None, "EPIC 211945111", 22, 0.753),
            ]
        ).to_csv(input_csv, index=False)

        out = K2StageRAutoLabeler.run_cli(["--input", str(input_csv), "--output", str(output_csv)])
        labels = pd.read_csv(output_csv)

        self.assertEqual(out["rows_input"], 2)
        self.assertEqual(out["rows_output"], 1)
        self.assertEqual(list(labels.columns), K2StageRAutoLabeler.OUTPUT_COLUMNS)
        self.assertEqual(labels.loc[0, "epic_id"], "EPIC_211945111")
        self.assertEqual(labels.loc[0, "stage_r_label"], "promote_to_deeper_eval")


if __name__ == "__main__":
    unittest.main()
