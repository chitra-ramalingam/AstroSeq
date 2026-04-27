from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageHWhitenessPolicyDiagnosis import K2StageHWhitenessPolicyDiagnosis


class K2StageHWhitenessPolicyDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_h_whiteness_policy_diagnosis_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_h_builds_diagnosis_and_recommends_redefinition(self) -> None:
        original_csv = self.case_dir / "k2_stage_f_batch_001_results.csv"
        patched_csv = self.case_dir / "k2_stage_f_batch_001b_results.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_1001",
                    "query": "EPIC 1001",
                    "execution_order": 1,
                    "batch_id": "high_priority_batch_001",
                    "planned_best_depth_snr": 100.0,
                    "planned_n_events": 20.0,
                    "planned_n_periods_proposed": 0.0,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_score_global": -1.0,
                    "triage_step_score": 0.01,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable_pipeline": "whiteness_pvalue=0<0.01",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "n_events": 20,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 100.0,
                    "epic_id_norm": "1001",
                },
                {
                    "epic_id": "EPIC_1002",
                    "query": "EPIC 1002",
                    "execution_order": 2,
                    "batch_id": "high_priority_batch_001",
                    "planned_best_depth_snr": 90.0,
                    "planned_n_events": 18.0,
                    "planned_n_periods_proposed": 0.0,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_score_global": -1.0,
                    "triage_step_score": 0.02,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable_pipeline": "whiteness_pvalue=0<0.01",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "n_events": 18,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 90.0,
                    "epic_id_norm": "1002",
                },
            ]
        ).to_csv(original_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_1001",
                    "query": "EPIC 1001",
                    "batch_id": "high_priority_batch_001b",
                    "old_execution_order": 10,
                    "new_execution_order": 1,
                    "planned_best_depth_snr": 100.0,
                    "planned_n_events": 20.0,
                    "planned_n_periods_proposed": 0.0,
                    "saved_triage_whiteness_pvalue": 0.999,
                    "saved_triage_step_score": 0.01,
                    "saved_triage_score_global": -0.66,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_usable": True,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_score_global": -1.0,
                    "triage_step_score": 0.01,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable_pipeline": "whiteness_pvalue=0<0.01",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "n_events": 20,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 100.0,
                    "epic_id_norm": "1001",
                },
                {
                    "epic_id": "EPIC_1003",
                    "query": "EPIC 1003",
                    "batch_id": "high_priority_batch_001b",
                    "old_execution_order": 11,
                    "new_execution_order": 2,
                    "planned_best_depth_snr": 80.0,
                    "planned_n_events": 17.0,
                    "planned_n_periods_proposed": 0.0,
                    "saved_triage_whiteness_pvalue": 0.998,
                    "saved_triage_step_score": 0.03,
                    "saved_triage_score_global": -0.65,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_usable": True,
                    "triage_status_pipeline": "ok",
                    "triage_usable_pipeline": False,
                    "triage_score_global": -1.0,
                    "triage_step_score": 0.03,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable_pipeline": "whiteness_pvalue=0<0.01",
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "n_events": 17,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 80.0,
                    "epic_id_norm": "1003",
                },
            ]
        ).to_csv(patched_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_1001", "triage_usable": True, "triage_whiteness_pvalue": 0.999, "triage_step_score": 0.01, "triage_score_global": -0.66, "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx"},
                {"epic_id": "EPIC_1002", "triage_usable": True, "triage_whiteness_pvalue": 0.997, "triage_step_score": 0.02, "triage_score_global": -0.64, "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx"},
                {"epic_id": "EPIC_1003", "triage_usable": True, "triage_whiteness_pvalue": 0.998, "triage_step_score": 0.03, "triage_score_global": -0.65, "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx"},
            ]
        ).to_csv(whiteness_csv, index=False)

        out = K2StageHWhitenessPolicyDiagnosis().run(
            original_results_csv=original_csv,
            patched_results_csv=patched_csv,
            whiteness_csv=whiteness_csv,
            out_dir=self.case_dir,
        )

        self.assertEqual(int(out["rows_total"]), 4)
        self.assertEqual(int(out["saved_proxy_pass_runtime_fail_count"]), 4)
        self.assertEqual(int(out["same_definition_count"]), 4)
        self.assertEqual(int(out["step_score_exact_match_count"]), 4)
        self.assertEqual(int(out["runtime_whiteness_zero_count"]), 4)
        self.assertEqual(str(out["recommendation"]), "C: redefine how whiteness is computed/interpreted")

        diagnosis_df = pd.read_csv(out["diagnosis_csv"])
        self.assertIn("saved_triage_whiteness_pvalue", diagnosis_df.columns)
        self.assertIn("runtime_triage_whiteness_score", diagnosis_df.columns)
        self.assertIn("whiteness_definition_same", diagnosis_df.columns)
        self.assertIn("proxy_non_equivalence_flag", diagnosis_df.columns)

        summary_df = pd.read_csv(out["summary_csv"])
        rec = summary_df.loc[
            (summary_df["section"] == "recommendation") & (summary_df["metric"] == "primary_scientific_action"),
            "value_text",
        ].iloc[0]
        same_def = summary_df.loc[
            (summary_df["section"] == "diagnostic") & (summary_df["metric"] == "saved_and_runtime_whiteness_definition_same"),
            "count",
        ].iloc[0]

        self.assertEqual(str(rec), "C: redefine how whiteness is computed/interpreted")
        self.assertEqual(int(same_def), 4)

    def test_stage_h_prefers_explicit_runtime_pvalue_when_present(self) -> None:
        original_csv = self.case_dir / "k2_stage_f_batch_001_results.csv"
        patched_csv = self.case_dir / "k2_stage_f_batch_001b_results.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"

        base_row = {
            "epic_id": "EPIC_2001",
            "query": "EPIC 2001",
            "planned_best_depth_snr": 50.0,
            "planned_n_events": 5.0,
            "planned_n_periods_proposed": 0.0,
            "triage_status_pipeline": "ok",
            "triage_usable_pipeline": False,
            "triage_score_global": -1.0,
            "triage_step_score": 0.01,
            "triage_whiteness_score": 0.0,
            "triage_whiteness_pvalue": 0.9,
            "triage_whiteness_log10_pvalue": -0.045757490560675115,
            "triage_whiteness_statistic_abs_rho": 0.01,
            "triage_whiteness_z": 0.45,
            "triage_whiteness_mode": "pvalue",
            "triage_whiteness_underflowed": False,
            "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
            "triage_why_not_usable_pipeline": "whiteness_pvalue=0<0.01",
            "label": "Noisy_trash",
            "label_reason": "usable=False:whiteness_pvalue=0<0.01",
            "n_events": 5,
            "n_periods_proposed": 0,
            "best_depth_snr": 50.0,
            "epic_id_norm": "2001",
        }
        pd.DataFrame(
            [
                {
                    **base_row,
                    "execution_order": 1,
                    "batch_id": "high_priority_batch_001",
                }
            ]
        ).to_csv(original_csv, index=False)
        pd.DataFrame(
            [
                {
                    **base_row,
                    "batch_id": "high_priority_batch_001b",
                    "old_execution_order": 1,
                    "new_execution_order": 1,
                    "saved_triage_whiteness_pvalue": 0.9,
                    "saved_triage_step_score": 0.01,
                    "saved_triage_score_global": 0.1,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_usable": True,
                }
            ]
        ).to_csv(patched_csv, index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_2001",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.9,
                    "triage_step_score": 0.01,
                    "triage_score_global": 0.1,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                }
            ]
        ).to_csv(whiteness_csv, index=False)

        out = K2StageHWhitenessPolicyDiagnosis().run(
            original_results_csv=original_csv,
            patched_results_csv=patched_csv,
            whiteness_csv=whiteness_csv,
            out_dir=self.case_dir,
        )

        diagnosis_df = pd.read_csv(out["diagnosis_csv"])
        self.assertAlmostEqual(float(diagnosis_df.loc[0, "runtime_triage_whiteness_pvalue"]), 0.9, places=12)
        self.assertAlmostEqual(float(diagnosis_df.loc[0, "whiteness_gap"]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
