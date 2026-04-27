from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageLPostPatchCalibrationRerun import K2StageLPostPatchCalibrationRerun


class _FakeStageLRunner:
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
                    "epic_id": "EPIC_1001",
                    "query": "EPIC 1001",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "whiteness_pvalue=0<0.01",
                    "triage_score_global": -1.0,
                    "triage_n_points": 120,
                    "triage_step_score": 0.0,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_pvalue": 0.0,
                    "triage_whiteness_log10_pvalue": -320.0,
                    "triage_whiteness_statistic_abs_rho": 0.8,
                    "triage_whiteness_z": 26.0,
                    "triage_whiteness_mode": "pvalue",
                    "triage_whiteness_underflowed": True,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "n_points_after_preprocess": 120,
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "author_selected": "SPOC",
                    "campaign_selected": "5",
                    "cache_only": False,
                    "n_events": 7,
                    "best_shape_score": 0.9,
                    "best_depth_snr": 30.0,
                    "n_periods_proposed": 0,
                    "n_periods_validated": 0,
                    "best_period": None,
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0<0.01; whiteness_pvalue<0.010 (0.000)",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_1001"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_1001" / "events.csv"),
                    "best_hits_csv": "",
                    "best_misses_csv": "",
                    "best_uncovered_csv": "",
                    "best_hitmap_png": "",
                    "best_phase_offset_png": "",
                },
                {
                    "epic_id": "EPIC_1002",
                    "query": "EPIC 1002",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "whiteness_pvalue=0.005<0.01",
                    "triage_score_global": -0.8,
                    "triage_n_points": 90,
                    "triage_step_score": 0.005,
                    "triage_whiteness_score": 0.005,
                    "triage_whiteness_pvalue": 0.005,
                    "triage_whiteness_log10_pvalue": -2.3010299956639813,
                    "triage_whiteness_statistic_abs_rho": 0.2,
                    "triage_whiteness_z": 2.8,
                    "triage_whiteness_mode": "pvalue",
                    "triage_whiteness_underflowed": False,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "n_points_after_preprocess": 90,
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "author_selected": "SPOC",
                    "campaign_selected": "5",
                    "cache_only": False,
                    "n_events": 5,
                    "best_shape_score": 0.8,
                    "best_depth_snr": 15.0,
                    "n_periods_proposed": 0,
                    "n_periods_validated": 0,
                    "best_period": None,
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness_pvalue=0.005<0.01",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_1002"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_1002" / "events.csv"),
                    "best_hits_csv": "",
                    "best_misses_csv": "",
                    "best_uncovered_csv": "",
                    "best_hitmap_png": "",
                    "best_phase_offset_png": "",
                },
                {
                    "epic_id": "EPIC_1003",
                    "query": "EPIC 1003",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "whiteness unavailable after upstream error",
                    "triage_score_global": -0.4,
                    "triage_n_points": 0,
                    "triage_step_score": float("nan"),
                    "triage_whiteness_score": float("nan"),
                    "triage_whiteness_pvalue": float("nan"),
                    "triage_whiteness_log10_pvalue": float("nan"),
                    "triage_whiteness_statistic_abs_rho": float("nan"),
                    "triage_whiteness_z": float("nan"),
                    "triage_whiteness_mode": "pvalue",
                    "triage_whiteness_underflowed": False,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "n_points_after_preprocess": 0,
                    "error_stage": "",
                    "error_type": "",
                    "error_msg": "",
                    "author_selected": "",
                    "campaign_selected": "",
                    "cache_only": False,
                    "n_events": 0,
                    "best_shape_score": float("nan"),
                    "best_depth_snr": float("nan"),
                    "n_periods_proposed": 0,
                    "n_periods_validated": 0,
                    "best_period": None,
                    "label": "Noisy_trash",
                    "label_reason": "usable=False:whiteness unavailable after upstream error",
                    "epic_dir": str(self.out_dir / "epics" / "EPIC_1003"),
                    "events_csv": str(self.out_dir / "epics" / "EPIC_1003" / "events.csv"),
                    "best_hits_csv": "",
                    "best_misses_csv": "",
                    "best_uncovered_csv": "",
                    "best_hitmap_png": "",
                    "best_phase_offset_png": "",
                },
            ]
        ).to_csv(batch_results_csv, index=False)
        return {"batch_results_csv": batch_results_csv, "results_df": pd.read_csv(batch_results_csv)}


class K2StageLPostPatchCalibrationRerunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_l_postpatch_calibration_rerun_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_l_builds_results_summary_and_audit(self) -> None:
        input_csv = self.case_dir / "k2_stage_f_batch_001b_input.csv"
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_1001",
                    "query": "EPIC 1001",
                    "old_execution_order": 10,
                    "new_execution_order": 1,
                    "rerank_score": 80.0,
                    "rerank_reason": "x",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.999,
                    "keepability_risk_flag": "low",
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 30.0,
                    "n_events": 7,
                    "n_periods_proposed": 0,
                    "saved_triage_whiteness_pvalue": 0.999,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_step_score": 0.001,
                    "saved_triage_score_global": -0.6,
                },
                {
                    "epic_id": "EPIC_1002",
                    "query": "EPIC 1002",
                    "old_execution_order": 11,
                    "new_execution_order": 2,
                    "rerank_score": 79.0,
                    "rerank_reason": "y",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.995,
                    "keepability_risk_flag": "low",
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 15.0,
                    "n_events": 5,
                    "n_periods_proposed": 0,
                    "saved_triage_whiteness_pvalue": 0.995,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_step_score": 0.005,
                    "saved_triage_score_global": -0.5,
                },
                {
                    "epic_id": "EPIC_1003",
                    "query": "EPIC 1003",
                    "old_execution_order": 12,
                    "new_execution_order": 3,
                    "rerank_score": 78.0,
                    "rerank_reason": "z",
                    "whiteness_proxy_available": True,
                    "whiteness_proxy_value": 0.99,
                    "keepability_risk_flag": "medium",
                    "next_action": "process_now",
                    "priority": "high",
                    "best_depth_snr": 10.0,
                    "n_events": 0,
                    "n_periods_proposed": 0,
                    "saved_triage_whiteness_pvalue": 0.990,
                    "saved_triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "saved_triage_step_score": 0.008,
                    "saved_triage_score_global": -0.4,
                },
            ]
        ).to_csv(input_csv, index=False)

        out = K2StageLPostPatchCalibrationRerun(runner_factory=_FakeStageLRunner).run(
            input_csv=input_csv,
            out_dir=self.case_dir,
        )

        results_df = pd.read_csv(out["results_csv"])
        self.assertIn("triage_whiteness_pvalue", results_df.columns)
        self.assertIn("triage_whiteness_log10_pvalue", results_df.columns)
        self.assertIn("triage_whiteness_underflowed", results_df.columns)

        audit_df = pd.read_csv(out["audit_csv"])
        self.assertEqual(list(audit_df.columns[:13]), K2StageLPostPatchCalibrationRerun.AUDIT_REQUIRED_COLUMNS)
        self.assertEqual(int(audit_df["runtime_whiteness_zero"].sum()), 1)
        self.assertEqual(int(audit_df["runtime_whiteness_finite_positive"].sum()), 1)
        self.assertEqual(int(audit_df["runtime_whiteness_missing"].sum()), 1)
        self.assertEqual(int(audit_df["legacy_explicit_agree_pvalue_mode"].sum()), 3)

        summary_df = pd.read_csv(out["summary_csv"])
        underflow_count = summary_df.loc[
            summary_df["metric"].eq("triage_whiteness_underflowed_true"), "count"
        ].iloc[0]
        pvalue_zero_count = summary_df.loc[
            summary_df["metric"].eq("triage_whiteness_pvalue_eq_0_0"), "count"
        ].iloc[0]
        finite_positive_count = summary_df.loc[
            summary_df["metric"].eq("triage_whiteness_pvalue_finite_positive"), "count"
        ].iloc[0]
        pvalue_nan_count = summary_df.loc[
            summary_df["metric"].eq("triage_whiteness_pvalue_nan"), "count"
        ].iloc[0]
        agree_count = summary_df.loc[
            summary_df["metric"].eq("legacy_score_explicit_pvalue_agree_in_pvalue_mode"), "count"
        ].iloc[0]
        resolved = summary_df.loc[
            summary_df["metric"].eq("patch_successfully_resolved_saved_runtime_comparability_problem"), "value_text"
        ].iloc[0]
        rejected = summary_df.loc[
            summary_df["metric"].eq("batch_still_scientifically_rejected_after_representation_fix"), "value_text"
        ].iloc[0]

        self.assertEqual(int(underflow_count), 1)
        self.assertEqual(int(pvalue_zero_count), 1)
        self.assertEqual(int(finite_positive_count), 1)
        self.assertEqual(int(pvalue_nan_count), 1)
        self.assertEqual(int(agree_count), 3)
        self.assertEqual(str(resolved), "yes")
        self.assertEqual(str(rejected), "yes")


if __name__ == "__main__":
    unittest.main()
