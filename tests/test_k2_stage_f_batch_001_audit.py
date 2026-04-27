from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageFBatch001Audit import K2StageFBatch001Audit


class K2StageFBatch001AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_f_batch_001_audit_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_f1_builds_audit_and_summary(self) -> None:
        stage_f_results_csv = self.case_dir / "k2_stage_f_batch_001_results.csv"
        stage_e_plan_csv = self.case_dir / "k2_stage_e_high_priority_batch_plan.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_501",
                    "query": "EPIC 501",
                    "execution_order": 1,
                    "batch_position": 1,
                    "planned_best_depth_snr": 200.0,
                    "planned_n_events": 20,
                    "planned_n_periods_proposed": 0,
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
                {
                    "epic_id": "EPIC_502",
                    "query": "EPIC 502",
                    "execution_order": 2,
                    "batch_position": 2,
                    "planned_best_depth_snr": 150.0,
                    "planned_n_events": 22,
                    "planned_n_periods_proposed": 0,
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
            ]
        ).to_csv(stage_f_results_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_501",
                    "query": "EPIC 501",
                    "execution_order": 1,
                    "batch_id": "high_priority_batch_001",
                    "batch_position": 1,
                    "best_depth_snr": 200.0,
                    "n_events": 20,
                    "n_periods_proposed": 0,
                },
                {
                    "epic_id": "EPIC_601",
                    "query": "EPIC 601",
                    "execution_order": 101,
                    "batch_id": "high_priority_batch_002",
                    "batch_position": 1,
                    "best_depth_snr": 100.0,
                    "n_events": 21,
                    "n_periods_proposed": 0,
                },
            ]
        ).to_csv(stage_e_plan_csv, index=False)

        out = K2StageFBatch001Audit().run(
            stage_f_results_csv=stage_f_results_csv,
            stage_e_plan_csv=stage_e_plan_csv,
            out_dir=self.case_dir,
        )

        self.assertEqual(int(out["rows_total"]), 2)
        self.assertEqual(int(out["upstream_triage_usable_true_but_final_noisy_trash"]), 2)
        self.assertEqual(out["rejection_dominated_by_one_single_gate"], "yes")
        self.assertEqual(out["batch_002_likely_to_behave_similarly"], "yes")
        self.assertEqual(out["recommendation"], "pause and patch execution ordering for future batches")

        audit_df = pd.read_csv(out["audit_csv"])
        self.assertEqual(
            list(audit_df.columns[:12]),
            [
                "epic_id",
                "query",
                "execution_order",
                "batch_position",
                "planned_best_depth_snr",
                "planned_n_events",
                "planned_n_periods_proposed",
                "upstream_triage_status",
                "upstream_triage_usable",
                "triage_status",
                "triage_usable",
                "triage_whiteness_score",
            ],
        )
        self.assertTrue((audit_df["final_label"] == "Noisy_trash").all())

        summary_df = pd.read_csv(out["audit_summary_csv"])
        rec = summary_df.loc[summary_df["metric"].eq("recommendation"), "value_text"].iloc[0]
        dominated = summary_df.loc[
            summary_df["metric"].eq("rejection_dominated_by_one_single_gate"), "value_text"
        ].iloc[0]
        self.assertEqual(str(rec), "pause and patch execution ordering for future batches")
        self.assertEqual(str(dominated), "yes")


if __name__ == "__main__":
    unittest.main()
