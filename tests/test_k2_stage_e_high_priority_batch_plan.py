from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageEHighPriorityBatchPlan import K2StageEHighPriorityBatchPlan


class K2StageEHighPriorityBatchPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_e_high_priority_batch_plan_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_e_builds_sequential_batches_without_reshuffling(self) -> None:
        stage_d_csv = self.case_dir / "k2_stage_d_process_now_high_priority.csv"
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_301",
                    "query": "EPIC 301",
                    "execution_order": 1,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 9.0,
                    "n_events": 5,
                    "n_periods_proposed": 1,
                    "extra_col": "a",
                },
                {
                    "epic_id": "EPIC_302",
                    "query": "EPIC 302",
                    "execution_order": 2,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 8.0,
                    "n_events": 4,
                    "n_periods_proposed": 1,
                    "extra_col": "b",
                },
                {
                    "epic_id": "EPIC_303",
                    "query": "EPIC 303",
                    "execution_order": 3,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 7.0,
                    "n_events": 3,
                    "n_periods_proposed": 1,
                    "extra_col": "c",
                },
                {
                    "epic_id": "EPIC_304",
                    "query": "EPIC 304",
                    "execution_order": 4,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 6.0,
                    "n_events": 2,
                    "n_periods_proposed": 0,
                    "extra_col": "d",
                },
                {
                    "epic_id": "EPIC_305",
                    "query": "EPIC 305",
                    "execution_order": 5,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 5.0,
                    "n_events": 1,
                    "n_periods_proposed": 0,
                    "extra_col": "e",
                },
            ]
        ).to_csv(stage_d_csv, index=False)

        out = K2StageEHighPriorityBatchPlan().run(
            stage_d_high_priority_csv=stage_d_csv,
            out_dir=self.case_dir,
            batch_size=2,
        )

        self.assertEqual(int(out["total_rows"]), 5)
        self.assertEqual(int(out["batch_size"]), 2)
        self.assertEqual(int(out["total_batches"]), 3)
        self.assertEqual(out["first_10_epics_batch_1"], ["EPIC_301", "EPIC_302"])
        self.assertEqual(
            out["rows_per_batch_summary"],
            "high_priority_batch_001:2 | high_priority_batch_002:2 | high_priority_batch_003:1",
        )

        plan_df = pd.read_csv(out["batch_plan_csv"])
        self.assertEqual(
            list(plan_df.columns[:10]),
            [
                "epic_id",
                "query",
                "execution_order",
                "batch_id",
                "batch_position",
                "next_action",
                "priority",
                "best_depth_snr",
                "n_events",
                "n_periods_proposed",
            ],
        )
        self.assertEqual(list(plan_df["epic_id"].astype(str)), ["EPIC_301", "EPIC_302", "EPIC_303", "EPIC_304", "EPIC_305"])
        self.assertEqual(
            list(plan_df["batch_id"].astype(str)),
            [
                "high_priority_batch_001",
                "high_priority_batch_001",
                "high_priority_batch_002",
                "high_priority_batch_002",
                "high_priority_batch_003",
            ],
        )
        self.assertEqual(list(plan_df["batch_position"].astype(int)), [1, 2, 1, 2, 1])
        self.assertEqual(list(plan_df["execution_order"].astype(int)), [1, 2, 3, 4, 5])
        self.assertEqual(list(plan_df["extra_col"].astype(str)), ["a", "b", "c", "d", "e"])


if __name__ == "__main__":
    unittest.main()
