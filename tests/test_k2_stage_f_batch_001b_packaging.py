from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageFBatch001bPackaging import K2StageFBatch001bPackaging


class K2StageFBatch001bPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_f_batch_001b_packaging_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_f_batch_001b_packaging_builds_input_and_summary(self) -> None:
        rerank_csv = self.case_dir / "k2_stage_e1_high_priority_rerank_preview.csv"
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_801",
                    "query": "EPIC 801",
                    "old_execution_order": 11,
                    "new_execution_order": 1,
                    "rerank_score": 75.0,
                    "rerank_reason": "r1",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.99,
                    "keepability_risk_flag": "low",
                    "next_action": "process_now",
                    "priority": "high",
                    "epic_id_norm": "801",
                    "current_status": "ready",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "routing_rule_id": "usable_with_saved_signal_support",
                    "priority_rule_id": "high_priority_process_now",
                    "period_source_reason": "x",
                    "period_terminal_reason": "other",
                    "n_events": 30,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 100.0,
                    "saved_triage_usable": True,
                    "saved_triage_whiteness_definition": "lag1",
                    "saved_triage_why_not_usable": "",
                    "saved_triage_step_score": 0.01,
                    "saved_triage_score_global": -0.5,
                },
                {
                    "epic_id": "EPIC_802",
                    "query": "EPIC 802",
                    "old_execution_order": 21,
                    "new_execution_order": 2,
                    "rerank_score": 74.0,
                    "rerank_reason": "r2",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.98,
                    "keepability_risk_flag": "low",
                    "next_action": "process_now",
                    "priority": "high",
                    "epic_id_norm": "802",
                    "current_status": "ready",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "routing_rule_id": "usable_with_saved_signal_support",
                    "priority_rule_id": "high_priority_process_now",
                    "period_source_reason": "x",
                    "period_terminal_reason": "other",
                    "n_events": 28,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 90.0,
                    "saved_triage_usable": True,
                    "saved_triage_whiteness_definition": "lag1",
                    "saved_triage_why_not_usable": "",
                    "saved_triage_step_score": 0.02,
                    "saved_triage_score_global": -0.4,
                },
            ]
        ).to_csv(rerank_csv, index=False)

        out = K2StageFBatch001bPackaging().run(rerank_preview_csv=rerank_csv, out_dir=self.case_dir)

        self.assertEqual(int(out["rows_total"]), 2)
        self.assertEqual(out["first_10_epics"], ["EPIC_801", "EPIC_802"])
        self.assertAlmostEqual(float(out["median_whiteness_proxy_value"]), 0.985)
        self.assertAlmostEqual(float(out["median_saved_triage_step_score"]), 0.015)
        self.assertAlmostEqual(float(out["median_n_events"]), 29.0)
        self.assertAlmostEqual(float(out["median_best_depth_snr"]), 95.0)

        input_df = pd.read_csv(out["input_csv"])
        self.assertEqual(list(input_df["epic_id"].astype(str)), ["EPIC_801", "EPIC_802"])
        self.assertIn("old_execution_order", input_df.columns)
        self.assertIn("new_execution_order", input_df.columns)
        self.assertIn("rerank_score", input_df.columns)
        self.assertIn("rerank_reason", input_df.columns)
        self.assertIn("whiteness_proxy_value", input_df.columns)
        self.assertIn("keepability_risk_flag", input_df.columns)

        summary_df = pd.read_csv(out["plan_summary_csv"])
        self.assertEqual(str(summary_df.loc[0, "batch_id"]), "high_priority_batch_001b")
        self.assertIn("replaces the original batch 001", str(summary_df.loc[0, "calibration_replacement_note"]))


if __name__ == "__main__":
    unittest.main()
