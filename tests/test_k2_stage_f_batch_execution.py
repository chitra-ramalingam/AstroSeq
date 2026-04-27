from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageFBatchExecution import K2StageFBatchExecution


class _FakeBatchRunner:
    def __init__(self, *, out_dir, input_csv, query_col, **kwargs):
        self.out_dir = Path(out_dir)
        self.input_csv = Path(input_csv)
        self.query_col = query_col

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        batch_results_csv = self.out_dir / "batch_results.csv"
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_401",
                    "query": "EPIC 401",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 3,
                    "best_shape_score": 0.8,
                    "best_depth_snr": 9.1,
                    "n_periods_proposed": 2,
                    "n_periods_validated": 1,
                    "best_period": 12.0,
                    "label": "Sparse_or_mono",
                    "label_reason": "strong shape but weak periodic hit rate",
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_401"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_401" / "events.csv"),
                },
                {
                    "epic_id": "EPIC_402",
                    "query": "EPIC 402",
                    "triage_status": "error",
                    "triage_usable": False,
                    "triage_why_not_usable": "",
                    "n_events": 0,
                    "best_shape_score": None,
                    "best_depth_snr": None,
                    "n_periods_proposed": 0,
                    "n_periods_validated": 0,
                    "best_period": None,
                    "label": "No_events",
                    "label_reason": "0 events detected",
                    "error_stage": "download",
                    "error_type": "RuntimeError",
                    "error_msg": "fetch_status=error",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_402"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_402" / "events.csv"),
                },
            ]
        ).to_csv(batch_results_csv, index=False)
        return {
            "batch_results_csv": batch_results_csv,
            "results_df": pd.read_csv(batch_results_csv),
        }


class K2StageFBatchExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_f_batch_execution_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_f_runs_one_batch_and_builds_summary(self) -> None:
        stage_e_csv = self.case_dir / "k2_stage_e_high_priority_batch_plan.csv"
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_401",
                    "query": "EPIC 401",
                    "execution_order": 1,
                    "batch_id": "high_priority_batch_001",
                    "batch_position": 1,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 10.0,
                    "n_events": 4,
                    "n_periods_proposed": 2,
                },
                {
                    "epic_id": "EPIC_402",
                    "query": "EPIC 402",
                    "execution_order": 2,
                    "batch_id": "high_priority_batch_001",
                    "batch_position": 2,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 9.0,
                    "n_events": 3,
                    "n_periods_proposed": 1,
                },
                {
                    "epic_id": "EPIC_403",
                    "query": "EPIC 403",
                    "execution_order": 3,
                    "batch_id": "high_priority_batch_002",
                    "batch_position": 1,
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 8.0,
                    "n_events": 2,
                    "n_periods_proposed": 1,
                },
            ]
        ).to_csv(stage_e_csv, index=False)

        out = K2StageFBatchExecution(runner_factory=_FakeBatchRunner).run(
            stage_e_plan_csv=stage_e_csv,
            out_dir=self.case_dir,
            batch_id="high_priority_batch_001",
        )

        self.assertEqual(int(out["rows_attempted"]), 2)
        self.assertEqual(int(out["rows_completed"]), 1)
        self.assertEqual(int(out["rows_failed"]), 1)
        self.assertEqual(int(out["rows_with_candidate_signal"]), 1)
        self.assertEqual(int(out["rows_without_candidate_signal"]), 1)
        self.assertEqual(int(out["rows_flagged_for_manual_review"]), 0)
        self.assertEqual(int(out["rows_requiring_rescue_followup"]), 1)
        self.assertEqual(out["representative_for_batch_002"], "no")

        results_df = pd.read_csv(out["results_csv"])
        self.assertEqual(list(results_df["epic_id"].astype(str)), ["EPIC_401", "EPIC_402"])
        self.assertEqual(list(results_df["execution_order"].astype(int)), [1, 2])
        self.assertEqual(list(results_df["batch_position"].astype(int)), [1, 2])
        self.assertIn("planned_best_depth_snr", results_df.columns)
        self.assertIn("best_depth_snr", results_df.columns)

        summary_df = pd.read_csv(out["summary_csv"])
        self.assertEqual(int(summary_df.loc[0, "rows_attempted"]), 2)
        self.assertEqual(str(summary_df.loc[0, "failure_modes_encountered"]), "download:RuntimeError=1")
        self.assertIn("main.py --out-dir", str(summary_df.loc[0, "command_used"]))


if __name__ == "__main__":
    unittest.main()
