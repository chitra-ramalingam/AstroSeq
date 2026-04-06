from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2KnownConfirmedFalseNegativeAudit import K2KnownConfirmedFalseNegativeAudit


class K2KnownConfirmedFalseNegativeAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_known_confirmed_false_negative_audit_{uuid4().hex}"
        self.current_dir = self.case_dir / "current"
        self.baseline_dir = self.case_dir / "baseline"
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_audit_classifies_min_cluster_count_case_and_policy_flags(self) -> None:
        epic = "200008693"

        pd.DataFrame(
            [
                {
                    "query": "EPIC 200008693",
                    "epic_id": epic,
                    "triage_status": "ok",
                    "n_events": 3,
                    "best_shape_score": 0.659073,
                    "best_depth_snr": 4.110326,
                }
            ]
        ).to_csv(self.current_dir / "merged_batch_results.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": epic,
                    "query": "EPIC 200008693",
                    "reason": "P_null_or_missing",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_category": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "n_events_after_filters": 3,
                    "min_cluster_count": 3,
                    "period_cap_days": 20.0,
                    "hist_in_period_range": 3.0,
                    "hist_pass_cluster_count": 0.0,
                }
            ]
        ).to_csv(self.current_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(columns=["epic"]).to_csv(self.current_dir / "period_shortlist_best.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": epic,
                    "query": "EPIC 200008693",
                    "terminal_reason": "no_cluster_periods",
                    "stage_reached": "period_inference",
                    "selected_for_period_stage": True,
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "details_json": (
                        '{"n_periods_proposed": 0, "n_periods_validated": 0, '
                        '"period_failure_category": "candidate_filter_rejection", '
                        '"period_failure_detail": "all_candidate_periods_below_min_cluster_count", '
                        '"period_hist_in_period_range": 3.0, '
                        '"period_hist_pass_cluster_count": 0.0, '
                        '"period_min_cluster_count": 3.0, '
                        '"period_n_events_after_filters": 3.0}'
                    ),
                }
            ]
        ).to_csv(self.current_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [
                {
                    "operating_mode_requested": "scale_validation_conditional_mcc2_experiment",
                    "min_cluster_count": 3,
                    "conditional_min_cluster_count_relax_enabled": True,
                    "conditional_min_cluster_count_relax_to": 2,
                    "conditional_min_cluster_count_min_events_after_filters": 4,
                    "conditional_min_cluster_count_min_hist_in_range": 2,
                    "period_cap_days": 20.0,
                }
            ]
        ).to_csv(self.current_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(
            [
                {
                    "query": "EPIC 200008693",
                    "epic_id": epic,
                    "triage_status": "ok",
                    "n_events": 3,
                    "best_shape_score": 0.659073,
                    "best_depth_snr": 4.110326,
                }
            ]
        ).to_csv(self.baseline_dir / "merged_batch_results.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic": epic,
                    "query": "EPIC 200008693",
                    "reason": "cluster_only",
                    "P": 1.777588,
                    "cluster_count": 2,
                    "manual_review_required": False,
                    "n_events_after_filters": 3,
                }
            ]
        ).to_csv(self.baseline_dir / "Apr1_period_shortlist_best.csv", index=False)
        pd.DataFrame(columns=["epic_id"]).to_csv(self.baseline_dir / "Apr1_period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": epic,
                    "query": "EPIC 200008693",
                    "terminal_reason": "other",
                    "stage_reached": "candidate_period_generation",
                    "selected_for_period_stage": True,
                    "details_json": '{"n_periods_proposed": 0, "n_periods_validated": 0}',
                }
            ]
        ).to_csv(self.baseline_dir / "Apr1_epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [
                {
                    "operating_mode_requested": "supported_high_recall",
                    "min_cluster_count": 2,
                    "period_cap_days": 20.0,
                }
            ]
        ).to_csv(self.baseline_dir / "Apr1_period_shortlist_diagnostics.csv", index=False)

        out = K2KnownConfirmedFalseNegativeAudit().run(
            epic_ids=[epic],
            current_run_dir=self.current_dir,
            baseline_run_dir=self.baseline_dir,
            analysis_csv=self.current_dir / "audit.csv",
            report_txt=self.current_dir / "audit.txt",
        )

        self.assertTrue(Path(out["analysis_csv"]).exists())
        self.assertTrue(Path(out["report_txt"]).exists())
        row = pd.read_csv(out["analysis_csv"]).iloc[0]
        self.assertEqual(str(row["epic_id"]), "EPIC_200008693")
        self.assertEqual(str(row["current_outcome_group"]), "quarantine")
        self.assertEqual(str(row["saved_default_outcome_group"]), "best")
        self.assertEqual(str(row["primary_rejection_bucket"]), "minimum cluster count")
        self.assertTrue(bool(row["survives_under_saved_default_policy"]))
        self.assertFalse(bool(row["survives_under_conditional_mcc2_carveout"]))
        self.assertFalse(bool(row["conditional_mcc2_carveout_eligible_from_existing_diagnostics"]))
        self.assertFalse(bool(row["survives_under_larger_period_cap_from_existing_diagnostics"]))
        self.assertIn("n_events_after_filters=3 < 4", str(row["conditional_mcc2_carveout_eligibility_reason"]))
        self.assertIn("period-cap rejection", str(row["larger_period_cap_assessment"]))


if __name__ == "__main__":
    unittest.main()
