from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageGHighPriorityWhitenessAudit import K2StageGHighPriorityWhitenessAudit


class K2StageGHighPriorityWhitenessAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_g_high_priority_whiteness_audit_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_g_builds_population_audit_and_summary(self) -> None:
        stage_d_csv = self.case_dir / "k2_stage_d_process_now_high_priority.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"
        rerank_preview_csv = self.case_dir / "k2_stage_e1_high_priority_rerank_preview.csv"
        batch_001_summary_csv = self.case_dir / "k2_stage_f_batch_001_summary.csv"
        batch_001b_summary_csv = self.case_dir / "k2_stage_f_batch_001b_summary.csv"

        pd.DataFrame(
            [
                {"epic_id": "EPIC_901", "query": "EPIC 901", "epic_id_norm": "901", "execution_order": 1, "best_depth_snr": 200.0, "n_events": 20, "n_periods_proposed": 0},
                {"epic_id": "EPIC_902", "query": "EPIC 902", "epic_id_norm": "902", "execution_order": 2, "best_depth_snr": 150.0, "n_events": 22, "n_periods_proposed": 0},
                {"epic_id": "EPIC_903", "query": "EPIC 903", "epic_id_norm": "903", "execution_order": 3, "best_depth_snr": 90.0, "n_events": 24, "n_periods_proposed": 0},
                {"epic_id": "EPIC_904", "query": "EPIC 904", "epic_id_norm": "904", "execution_order": 4, "best_depth_snr": 60.0, "n_events": 26, "n_periods_proposed": 0},
                {"epic_id": "EPIC_905", "query": "EPIC 905", "epic_id_norm": "905", "execution_order": 5, "best_depth_snr": 40.0, "n_events": 28, "n_periods_proposed": 1},
            ]
        ).to_csv(stage_d_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_901", "triage_status": "ok", "triage_usable": True, "triage_whiteness_pvalue": 0.99950, "triage_whiteness_definition": "lag1", "triage_why_not_usable": "", "triage_whiteness_interpretation": "white", "triage_score_global": -0.40, "triage_step_score": 0.020, "triage_whiteness_one_minus_pvalue": 0.00050},
                {"epic_id": "EPIC_902", "triage_status": "ok", "triage_usable": True, "triage_whiteness_pvalue": 0.99960, "triage_whiteness_definition": "lag1", "triage_why_not_usable": "", "triage_whiteness_interpretation": "white", "triage_score_global": -0.35, "triage_step_score": 0.018, "triage_whiteness_one_minus_pvalue": 0.00040},
                {"epic_id": "EPIC_903", "triage_status": "ok", "triage_usable": True, "triage_whiteness_pvalue": 0.99970, "triage_whiteness_definition": "lag1", "triage_why_not_usable": "", "triage_whiteness_interpretation": "white", "triage_score_global": -0.30, "triage_step_score": 0.015, "triage_whiteness_one_minus_pvalue": 0.00030},
                {"epic_id": "EPIC_904", "triage_status": "ok", "triage_usable": True, "triage_whiteness_pvalue": 0.99980, "triage_whiteness_definition": "lag1", "triage_why_not_usable": "", "triage_whiteness_interpretation": "white", "triage_score_global": -0.25, "triage_step_score": 0.012, "triage_whiteness_one_minus_pvalue": 0.00020},
                {"epic_id": "EPIC_905", "triage_status": "ok", "triage_usable": True, "triage_whiteness_pvalue": 0.99990, "triage_whiteness_definition": "lag1", "triage_why_not_usable": "", "triage_whiteness_interpretation": "white", "triage_score_global": -0.20, "triage_step_score": 0.010, "triage_whiteness_one_minus_pvalue": 0.00010},
            ]
        ).to_csv(whiteness_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id_norm": "904", "new_execution_order": 1},
                {"epic_id_norm": "905", "new_execution_order": 2},
                {"epic_id_norm": "903", "new_execution_order": 3},
                {"epic_id_norm": "902", "new_execution_order": 4},
                {"epic_id_norm": "901", "new_execution_order": 5},
            ]
        ).to_csv(rerank_preview_csv, index=False)

        pd.DataFrame(
            [
                {
                    "label_counts": "Noisy_trash=5",
                    "dominant_label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000) (5)",
                }
            ]
        ).to_csv(batch_001_summary_csv, index=False)

        pd.DataFrame(
            [
                {
                    "final_label_counts": "Noisy_trash=5",
                    "patched_batch_001b_whiteness_rejection_count": 5,
                }
            ]
        ).to_csv(batch_001b_summary_csv, index=False)

        out = K2StageGHighPriorityWhitenessAudit().run(
            stage_d_high_priority_csv=stage_d_csv,
            whiteness_csv=whiteness_csv,
            rerank_preview_csv=rerank_preview_csv,
            batch_001_summary_csv=batch_001_summary_csv,
            batch_001b_summary_csv=batch_001b_summary_csv,
            out_dir=self.case_dir,
        )

        self.assertEqual(int(out["rows_total"]), 5)
        self.assertEqual(int(out["proxy_coverage"]), 5)
        self.assertEqual(int(out["rows_with_n_periods_proposed_gt0"]), 1)
        self.assertEqual(str(out["recommendation"]), "D: revisit whiteness policy scientifically")

        audit_df = pd.read_csv(out["audit_csv"])
        self.assertIn("saved_triage_whiteness_pvalue", audit_df.columns)
        self.assertIn("saved_triage_step_score", audit_df.columns)
        self.assertIn("saved_triage_score_global", audit_df.columns)
        self.assertIn("whiteness_risk_bucket", audit_df.columns)
        self.assertIn("runtime_survivability_proxy", audit_df.columns)
        self.assertEqual(len(audit_df), 5)

        summary_df = pd.read_csv(out["summary_csv"])
        rec = summary_df.loc[
            (summary_df["section"] == "recommendation") & (summary_df["metric"] == "primary_next_lever"),
            "value_text",
        ].iloc[0]
        n_periods_gt0 = summary_df.loc[
            (summary_df["section"] == "counts") & (summary_df["metric"] == "rows_with_n_periods_proposed_gt0"),
            "count",
        ].iloc[0]
        coverage = summary_df.loc[
            (summary_df["section"] == "coverage") & (summary_df["metric"] == "saved_triage_whiteness_pvalue"),
            "count",
        ].iloc[0]

        self.assertEqual(str(rec), "D: revisit whiteness policy scientifically")
        self.assertEqual(int(n_periods_gt0), 1)
        self.assertEqual(int(coverage), 5)


if __name__ == "__main__":
    unittest.main()
