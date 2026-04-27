from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageE1HighPriorityRerank import K2StageE1HighPriorityRerank


class K2StageE1HighPriorityRerankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_e1_high_priority_rerank_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_e1_builds_preview_and_summary(self) -> None:
        stage_d_csv = self.case_dir / "k2_stage_d_process_now_high_priority.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_701",
                    "query": "EPIC 701",
                    "epic_id_norm": "701",
                    "n_events": 20,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 300.0,
                    "execution_order": 1,
                },
                {
                    "epic_id": "EPIC_702",
                    "query": "EPIC 702",
                    "epic_id_norm": "702",
                    "n_events": 18,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 250.0,
                    "execution_order": 2,
                },
                {
                    "epic_id": "EPIC_703",
                    "query": "EPIC 703",
                    "epic_id_norm": "703",
                    "n_events": 25,
                    "n_periods_proposed": 1,
                    "best_depth_snr": 120.0,
                    "execution_order": 3,
                },
            ]
        ).to_csv(stage_d_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_701",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.90,
                    "triage_step_score": 0.20,
                    "triage_score_global": -0.20,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
                {
                    "epic_id": "EPIC_702",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.95,
                    "triage_step_score": 0.10,
                    "triage_score_global": -0.10,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
                {
                    "epic_id": "EPIC_703",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.99,
                    "triage_step_score": 0.01,
                    "triage_score_global": 0.10,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
            ]
        ).to_csv(whiteness_csv, index=False)

        out = K2StageE1HighPriorityRerank().run(
            stage_d_high_priority_csv=stage_d_csv,
            whiteness_csv=whiteness_csv,
            out_dir=self.case_dir,
        )

        self.assertEqual(int(out["rows_total"]), 3)
        self.assertEqual(int(out["whiteness_proxy_coverage"]), 3)

        preview_df = pd.read_csv(out["preview_csv"])
        self.assertIn("old_execution_order", preview_df.columns)
        self.assertIn("new_execution_order", preview_df.columns)
        self.assertIn("rerank_score", preview_df.columns)
        self.assertIn("rerank_reason", preview_df.columns)
        self.assertIn("whiteness_proxy_available", preview_df.columns)
        self.assertIn("whiteness_proxy_value", preview_df.columns)
        self.assertIn("keepability_risk_flag", preview_df.columns)
        self.assertEqual(list(preview_df["epic_id"].astype(str)), ["EPIC_703", "EPIC_701", "EPIC_702"])
        self.assertEqual(list(preview_df["new_execution_order"].astype(int)), [1, 2, 3])
        self.assertEqual(list(preview_df["old_execution_order"].astype(int)), [3, 1, 2])

        summary_df = pd.read_csv(out["summary_csv"])
        rec = summary_df.loc[(summary_df["section"] == "recommendation") & (summary_df["metric"] == "action"), "value_text"].iloc[0]
        proxy = summary_df.loc[
            (summary_df["section"] == "metadata") & (summary_df["metric"] == "usable_whiteness_stability_proxy_exists_upstream"),
            "value_text",
        ].iloc[0]
        self.assertEqual(str(rec), "rerun a patched batch 001 before proceeding")
        self.assertEqual(str(proxy), "yes")


if __name__ == "__main__":
    unittest.main()
