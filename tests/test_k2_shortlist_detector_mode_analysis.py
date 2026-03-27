from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistDetectorModeAnalysis import K2ShortlistDetectorModeAnalysis


class TestK2ShortlistDetectorModeAnalysis(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("tmp_pycache") / f"k2_shortlist_detector_mode_analysis_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    def _write_raw_csv(self, path: Path) -> None:
        pd.DataFrame(
            [
                {
                    "query": "EPIC 100",
                    "epic_id": "100",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.7,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
                {
                    "query": "EPIC 200",
                    "epic_id": "200",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.6,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
                {
                    "query": "EPIC 300",
                    "epic_id": "300",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.5,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                },
            ]
        ).to_csv(path, index=False)

    def test_detector_mode_analysis_writes_requested_csvs(self) -> None:
        case_dir = self._make_case_dir()
        baseline_dir = case_dir / "baseline"
        mcc2_dir = case_dir / "mcc2"
        detector_dir = case_dir / "detector"
        out_dir = case_dir / "out"
        for d in [baseline_dir, mcc2_dir, detector_dir]:
            d.mkdir(parents=True, exist_ok=False)

        baseline_raw = case_dir / "baseline_whiteness.csv"
        detector_raw = case_dir / "detector_whiteness.csv"
        self._write_raw_csv(baseline_raw)
        self._write_raw_csv(detector_raw)

        pd.DataFrame(
            [
                {"epic": "100", "query": "EPIC 100", "P": 12.0, "manual_review_required": False},
            ]
        ).to_csv(baseline_dir / "period_shortlist_best.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": "200",
                    "query": "EPIC 200",
                    "failure_category": "insufficient_events",
                    "source_reason": "no_cluster_periods",
                    "failure_detail": "insufficient_events_for_period_inference",
                    "n_events_raw": 1,
                    "n_events_after_filters": 1,
                },
                {
                    "epic_id": "300",
                    "query": "EPIC 300",
                    "failure_category": "events_filtered_to_zero",
                    "source_reason": "events_filtered_to_zero",
                    "n_events_raw": 0,
                    "n_events_after_filters": 0,
                },
            ]
        ).to_csv(baseline_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "n_events_after_filters=1", "period_n_events_raw": 1, "period_n_events_after_filters": 1},
                {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "events_filtered_to_zero", "period_n_events_raw": 0, "period_n_events_after_filters": 0},
            ]
        ).to_csv(baseline_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [{"mcc_policy_mode": "precision_first_default", "raw_epic_list_csv": str(baseline_raw)}]
        ).to_csv(baseline_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(
            [
                {"epic": "100", "query": "EPIC 100", "P": 12.0, "manual_review_required": False},
                {"epic": "200", "query": "EPIC 200", "P": 6.0, "manual_review_required": True},
            ]
        ).to_csv(mcc2_dir / "period_shortlist_best.csv", index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": "300",
                    "query": "EPIC 300",
                    "failure_category": "events_filtered_to_zero",
                    "source_reason": "events_filtered_to_zero",
                    "n_events_raw": 0,
                    "n_events_after_filters": 0,
                },
            ]
        ).to_csv(mcc2_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "events_filtered_to_zero", "period_n_events_raw": 0, "period_n_events_after_filters": 0},
            ]
        ).to_csv(mcc2_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [{"mcc_policy_mode": "supported_high_recall", "raw_epic_list_csv": str(baseline_raw)}]
        ).to_csv(mcc2_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(
            [
                {"epic": "100", "query": "EPIC 100", "P": 12.0, "manual_review_required": False},
                {"epic": "200", "query": "EPIC 200", "P": 6.0, "manual_review_required": True},
                {"epic": "300", "query": "EPIC 300", "P": 18.0, "manual_review_required": True},
            ]
        ).to_csv(detector_dir / "period_shortlist_best.csv", index=False)
        pd.DataFrame(columns=["epic_id"]).to_csv(detector_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
            ]
        ).to_csv(detector_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [{"mcc_policy_mode": "supported_high_recall", "raw_epic_list_csv": str(detector_raw)}]
        ).to_csv(detector_dir / "period_shortlist_diagnostics.csv", index=False)

        out = K2ShortlistDetectorModeAnalysis().run(
            baseline_run_dir=baseline_dir,
            mcc2_run_dir=mcc2_dir,
            detector_run_dir=detector_dir,
            out_dir=out_dir,
        )

        self.assertTrue(Path(out["detector_mode_comparison_csv"]).exists())
        self.assertTrue(Path(out["rescued_by_detector_mode_csv"]).exists())
        self.assertTrue(Path(out["rescued_by_detector_mode_by_period_bin_csv"]).exists())
        self.assertEqual(int(out["detector_added_vs_mcc2"]), 1)
        self.assertEqual(int(out["zero_event_delta_vs_mcc2"]), -1)
        self.assertEqual(int(out["insufficient_support_delta_vs_mcc2"]), 0)
        self.assertEqual(int(out["manual_review_delta_vs_mcc2"]), 1)

        comparison_df = pd.read_csv(out["detector_mode_comparison_csv"])
        detector_row = comparison_df.loc[
            comparison_df["mode"].astype(str) == "detector_high_recall_experimental"
        ].iloc[0]
        self.assertEqual(int(detector_row["shortlisted_count"]), 3)
        self.assertEqual(int(detector_row["added_vs_mcc2"]), 1)
        self.assertEqual(int(detector_row["zero_event_count"]), 0)

        rescued_df = pd.read_csv(out["rescued_by_detector_mode_csv"])
        row_300 = rescued_df.loc[rescued_df["epic_id"].astype(str) == "300"].iloc[0]
        self.assertTrue(bool(row_300["rescued_by_detector_experimental"]))
        self.assertEqual(str(row_300["prior_first_failed_upstream_stage"]), "event_detection_produced_zero_events")
        self.assertEqual(str(row_300["prior_suspected_cause"]), "detector_sensitivity_or_candidate_generation")


if __name__ == "__main__":
    unittest.main()
