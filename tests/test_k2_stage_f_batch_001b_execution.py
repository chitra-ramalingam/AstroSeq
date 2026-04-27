from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageFBatch001bExecution import K2StageFBatch001bExecution


class _FakeBatch001bRunner:
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
                    "epic_id": "EPIC_901",
                    "query": "EPIC 901",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 4,
                    "best_shape_score": 0.9,
                    "best_depth_snr": 12.0,
                    "n_periods_proposed": 1,
                    "n_periods_validated": 1,
                    "best_period": 10.0,
                    "label": "Sparse_or_mono",
                    "label_reason": "strong shape but weak periodic hit rate",
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_901"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_901" / "events.csv"),
                },
                {
                    "epic_id": "EPIC_902",
                    "query": "EPIC 902",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "whiteness_pvalue=0<0.01",
                    "n_events": 2,
                    "best_shape_score": 0.6,
                    "best_depth_snr": 8.0,
                    "n_periods_proposed": 0,
                    "n_periods_validated": 0,
                    "best_period": None,
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_902"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_902" / "events.csv"),
                },
            ]
        ).to_csv(batch_results_csv, index=False)
        return {"batch_results_csv": batch_results_csv, "results_df": pd.read_csv(batch_results_csv)}


class K2StageFBatch001bExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_f_batch_001b_execution_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_f_batch_001b_execution_builds_summary_with_comparison(self) -> None:
        input_csv = self.case_dir / "k2_stage_f_batch_001b_input.csv"
        original_results_csv = self.case_dir / "k2_stage_f_batch_001_results.csv"
        original_summary_csv = self.case_dir / "k2_stage_f_batch_001_summary.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_901",
                    "query": "EPIC 901",
                    "old_execution_order": 10,
                    "new_execution_order": 1,
                    "rerank_score": 77.0,
                    "rerank_reason": "x",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.99,
                    "keepability_risk_flag": "low",
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 100.0,
                    "n_events": 5,
                    "n_periods_proposed": 1,
                },
                {
                    "epic_id": "EPIC_902",
                    "query": "EPIC 902",
                    "old_execution_order": 11,
                    "new_execution_order": 2,
                    "rerank_score": 76.0,
                    "rerank_reason": "y",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.95,
                    "keepability_risk_flag": "risk",
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 90.0,
                    "n_events": 4,
                    "n_periods_proposed": 0,
                },
            ]
        ).to_csv(input_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_X1", "query": "EPIC X1", "label": "Noisy_trash", "label_reason": "usable=False:whiteness_pvalue=0<0.01"},
                {"epic_id": "EPIC_X2", "query": "EPIC X2", "label": "Noisy_trash", "label_reason": "usable=False:whiteness_pvalue=0<0.01"},
            ]
        ).to_csv(original_results_csv, index=False)
        pd.DataFrame([{"runtime_seconds": 10.0}]).to_csv(original_summary_csv, index=False)

        out = K2StageFBatch001bExecution(runner_factory=_FakeBatch001bRunner).run(
            input_csv=input_csv,
            out_dir=self.case_dir,
            original_results_csv=original_results_csv,
            original_summary_csv=original_summary_csv,
        )

        self.assertEqual(int(out["rows_attempted"]), 2)
        self.assertEqual(int(out["rows_completed"]), 2)
        self.assertEqual(int(out["rows_failed"]), 0)
        self.assertEqual(int(out["rows_with_candidate_signal"]), 2)
        self.assertEqual(int(out["rows_without_candidate_signal"]), 0)
        self.assertEqual(int(out["rows_flagged_for_manual_review"]), 0)
        self.assertEqual(int(out["rows_requiring_rescue_followup"]), 1)
        self.assertIn("Sparse_or_mono=1", out["final_label_counts"])
        self.assertIn("Noisy_trash=1", out["final_label_counts"])
        self.assertEqual(out["comparison"]["original_noisy_trash_count"], 2)
        self.assertEqual(out["comparison"]["current_noisy_trash_count"], 1)
        self.assertEqual(out["comparison"]["whiteness_rejection_frequency_improved"], "yes")
        self.assertEqual(out["comparison"]["enough_evidence_to_proceed"], "yes")

        summary_df = pd.read_csv(out["summary_csv"])
        self.assertEqual(int(summary_df.loc[0, "patched_batch_001b_noisy_trash_count"]), 1)
        self.assertEqual(str(summary_df.loc[0, "whiteness_rejection_frequency_improved"]), "yes")
        self.assertIn("main.py --out-dir", str(summary_df.loc[0, "command_used"]))


if __name__ == "__main__":
    unittest.main()
