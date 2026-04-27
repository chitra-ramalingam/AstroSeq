from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageDDeeperEvalRunner import K2StageDDeeperEvalRunner


class K2StageDDeeperEvalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_d_deeper_eval_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    @staticmethod
    def _input_row(epic_id: str) -> dict:
        return {
            "epic_id": epic_id,
            "promote_tier": "Tier_A",
            "n_events_long_good": 4,
            "n_events_ge_10_cadences": 4,
            "max_shape_score": 0.9,
            "spike_fraction_2cadence": 0.0,
            "depth_ratio": 1.1,
            "stage_r_reason": "promote",
        }

    @staticmethod
    def _event_row(epic: str, t_mid: float, depth: float = 0.01, duration: int = 12) -> dict:
        return {
            "query": epic.replace("_", " "),
            "author": "TEST",
            "start_idx": 0,
            "end_idx": duration,
            "min_idx": 5,
            "window_start": 0,
            "window_end": 20,
            "t_start": t_mid - 0.1,
            "t_end": t_mid + 0.1,
            "t_mid": t_mid,
            "duration_cadences": duration,
            "duration_days": 0.25,
            "depth": depth,
            "depth_snr": 10.0,
            "symmetry": 0.9,
            "curvature": 0.8,
            "continuity": 0.8,
            "ingress_egress_ok": True,
            "shape_score": 0.85,
        }

    def test_stage_d_deeper_eval_writes_one_result_per_input_row(self) -> None:
        input_csv = self.case_dir / "k2_stage_d_input_tier_a.csv"
        output_csv = self.case_dir / "k2_stage_d_tier_a_results.csv"
        epics_dir = self.case_dir / "epics"
        epics_dir.mkdir()

        pd.DataFrame(
            [
                self._input_row("EPIC_100001"),
                self._input_row("EPIC_100002"),
            ]
        ).to_csv(input_csv, index=False)

        good_dir = epics_dir / "EPIC_100001"
        good_dir.mkdir()
        pd.DataFrame(
            [
                self._event_row("EPIC_100001", 0.0),
                self._event_row("EPIC_100001", 2.0, depth=0.011),
                self._event_row("EPIC_100001", 4.0, depth=0.0105),
                self._event_row("EPIC_100001", 6.0, depth=0.0102),
            ]
        ).to_csv(good_dir / "events.csv", index=False)

        weak_dir = epics_dir / "EPIC_100002"
        weak_dir.mkdir()
        pd.DataFrame([self._event_row("EPIC_100002", 1.0)]).to_csv(weak_dir / "events.csv", index=False)

        out = K2StageDDeeperEvalRunner().run(
            input_csv=input_csv,
            output_csv=output_csv,
            epics_dir=epics_dir,
            min_cluster_count=2,
            validation_enabled=False,
        )

        self.assertEqual(int(out["rows_input"]), 2)
        self.assertEqual(int(out["rows_output"]), 2)
        self.assertEqual(int(out["pass_count"]), 1)
        self.assertEqual(int(out["fail_count"]), 1)

        results = pd.read_csv(output_csv)
        self.assertEqual(list(results["epic_id"].astype(str)), ["EPIC_100001", "EPIC_100002"])
        self.assertIn("best_period_days", results.columns)
        self.assertIn("period_support_count", results.columns)
        self.assertIn("folded_depth_consistency", results.columns)
        self.assertIn("duration_consistency", results.columns)
        self.assertIn("odd_even_depth_delta", results.columns)
        labels = dict(zip(results["epic_id"], results["stage_d_label"]))
        self.assertEqual(labels["EPIC_100001"], "pass_deeper_eval")
        self.assertEqual(labels["EPIC_100002"], "fail_deeper_eval")


if __name__ == "__main__":
    unittest.main()
