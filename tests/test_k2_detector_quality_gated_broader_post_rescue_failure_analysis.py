from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderPostRescueFailureAnalysis import (
    K2DetectorQualityGatedBroaderPostRescueFailureAnalysis,
)


class K2DetectorQualityGatedBroaderPostRescueFailureAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_detector_quality_gated_broader_post_rescue_failure_analysis_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_analysis_buckets_failures_and_recommends_histogram_handling(self) -> None:
        qw_csv = self.case_dir / "detector_quality_gated_broader_quarantined_winners.csv"
        q_csv = self.case_dir / "Apr1_period_shortlist_quarantine.csv"
        d_csv = self.case_dir / "Apr1_period_shortlist_diagnostics.csv"
        f_csv = self.case_dir / "Apr1_epic_funnel_reasons.csv"
        analysis_csv = self.case_dir / "detector_quality_gated_broader_post_rescue_failure_analysis.csv"
        rollup_csv = self.case_dir / "detector_quality_gated_broader_post_rescue_failure_rollup.csv"

        pd.DataFrame(
            [
                {"epic_id": "EPIC_101", "failure_category": "insufficient_events", "shortlist_rejection_reason": "insufficient_events", "terminal_reason": "no_cluster_periods", "quarantine_n_events_after_filters": 1},
                {"epic_id": "EPIC_102", "failure_category": "empty_histogram", "shortlist_rejection_reason": "empty_histogram", "terminal_reason": "no_cluster_periods", "quarantine_n_events_after_filters": 2},
                {"epic_id": "EPIC_103", "failure_category": "candidate_filter_rejection", "shortlist_rejection_reason": "candidate_filter_rejection", "terminal_reason": "no_cluster_periods", "quarantine_n_events_after_filters": 2, "quarantine_hist_total": 1, "quarantine_hist_in_period_range": 0, "quarantine_hist_pass_cluster_count": 1, "quarantine_hist_pass_all_filters": 0},
                {"epic_id": "EPIC_104", "failure_category": "other", "shortlist_rejection_reason": "other", "terminal_reason": "no_cluster_periods", "quarantine_n_events_after_filters": 4, "quarantine_hist_total": 5, "quarantine_hist_in_period_range": 2, "quarantine_hist_pass_cluster_count": 0, "quarantine_hist_pass_all_filters": 0},
            ]
        ).to_csv(qw_csv, index=False)
        pd.DataFrame(
            [
                {"epic_id": "101", "failure_category": "insufficient_events", "failure_detail": "insufficient_events_for_period_inference", "n_events_after_filters": 1},
                {"epic_id": "102", "failure_category": "empty_histogram", "failure_detail": "infer_periods_from_events_returned_empty_hist", "n_events_after_filters": 2},
                {"epic_id": "103", "failure_category": "candidate_filter_rejection", "failure_detail": "all_candidate_periods_outside_period_bounds", "n_events_after_filters": 2, "hist_total": 1, "hist_in_period_range": 0, "hist_pass_cluster_count": 1, "hist_pass_all_filters": 0},
                {"epic_id": "104", "failure_category": "other", "failure_detail": "ambiguous_case", "n_events_after_filters": 4, "hist_total": 5, "hist_in_period_range": 2, "hist_pass_cluster_count": 0, "hist_pass_all_filters": 0},
            ]
        ).to_csv(q_csv, index=False)
        pd.DataFrame(
            [
                {"epic_id": "101", "terminal_reason": "no_cluster_periods", "source_reason": "no_cluster_periods", "stage_reached": "period_inference"},
                {"epic_id": "102", "terminal_reason": "no_cluster_periods", "source_reason": "empty_histogram", "stage_reached": "period_inference"},
                {"epic_id": "103", "terminal_reason": "no_cluster_periods", "source_reason": "candidate_filter_rejection", "stage_reached": "period_inference", "details_json": '{"period_failure_category":"candidate_filter_rejection","period_failure_detail":"all_candidate_periods_outside_period_bounds","period_hist_total":1,"period_hist_in_period_range":0,"period_hist_pass_cluster_count":1,"period_hist_pass_all_filters":0}'},
                {"epic_id": "104", "terminal_reason": "no_cluster_periods", "source_reason": "other", "stage_reached": "period_inference"},
            ]
        ).to_csv(f_csv, index=False)
        pd.DataFrame([{"min_cluster_count": 2, "operating_mode_requested": "supported_high_recall", "n_quarantined_no_cluster_periods": 455}]).to_csv(d_csv, index=False)

        out = K2DetectorQualityGatedBroaderPostRescueFailureAnalysis().run(
            quarantined_winners_csv=qw_csv,
            quarantine_csv=q_csv,
            diagnostics_csv=d_csv,
            funnel_csv=f_csv,
            analysis_csv=analysis_csv,
            rollup_csv=rollup_csv,
            examples_per_bucket=2,
        )

        self.assertEqual(int(out["quarantined_winners_total"]), 4)
        self.assertEqual(out["recommended_next_lever"], "histogram construction / handling")

        analysis_df = pd.read_csv(analysis_csv)
        bucket_map = dict(zip(analysis_df["epic_id"], analysis_df["actionable_bucket"]))
        lever_map = dict(zip(analysis_df["epic_id"], analysis_df["suggested_lever"]))
        self.assertEqual(bucket_map["EPIC_101"], "true insufficient signal")
        self.assertEqual(bucket_map["EPIC_102"], "likely recoverable with histogram handling changes")
        self.assertEqual(bucket_map["EPIC_103"], "likely recoverable with looser cluster/period policy")
        self.assertEqual(bucket_map["EPIC_104"], "likely unrecoverable / noise")
        self.assertEqual(lever_map["EPIC_103"], "candidate filter policy")

        rollup_df = pd.read_csv(rollup_csv)
        row = rollup_df.loc[(rollup_df["section"] == "recommendation") & (rollup_df["metric"] == "recommended_next_lever")].iloc[0]
        self.assertEqual(str(row["value"]), "histogram construction / handling")

    def test_analysis_fails_clearly_when_required_columns_are_missing(self) -> None:
        qw_csv = self.case_dir / "detector_quality_gated_broader_quarantined_winners.csv"
        q_csv = self.case_dir / "Apr1_period_shortlist_quarantine.csv"
        d_csv = self.case_dir / "Apr1_period_shortlist_diagnostics.csv"
        f_csv = self.case_dir / "Apr1_epic_funnel_reasons.csv"

        pd.DataFrame([{"epic_id": "EPIC_101", "failure_category": "empty_histogram"}]).to_csv(qw_csv, index=False)
        pd.DataFrame([{"epic_id": "101", "failure_category": "empty_histogram", "failure_detail": "x", "n_events_after_filters": 2}]).to_csv(q_csv, index=False)
        pd.DataFrame([{"epic_id": "101", "terminal_reason": "no_cluster_periods", "source_reason": "x", "stage_reached": "period_inference"}]).to_csv(f_csv, index=False)
        pd.DataFrame([{"min_cluster_count": 2, "operating_mode_requested": "supported_high_recall", "n_quarantined_no_cluster_periods": 1}]).to_csv(d_csv, index=False)

        with self.assertRaisesRegex(ValueError, "quarantined_winners CSV missing required columns"):
            K2DetectorQualityGatedBroaderPostRescueFailureAnalysis().run(
                quarantined_winners_csv=qw_csv,
                quarantine_csv=q_csv,
                diagnostics_csv=d_csv,
                funnel_csv=f_csv,
                analysis_csv=self.case_dir / "analysis.csv",
                rollup_csv=self.case_dir / "rollup.csv",
            )


if __name__ == "__main__":
    unittest.main()
