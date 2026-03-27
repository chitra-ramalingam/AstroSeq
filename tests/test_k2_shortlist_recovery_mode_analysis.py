from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock
import uuid

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class TestK2ShortlistRecoveryModeAnalysis(unittest.TestCase):
    def test_run_cli_applies_threshold_relaxed_operating_mode(self) -> None:
        with mock.patch.object(K2ShortlistPeriodRunner, "run", autospec=True, return_value={"ok": True}) as run_mock:
            out = K2ShortlistPeriodRunner.run_cli(
                [
                    "--operating-mode",
                    K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME,
                    "--period-stage-n",
                    "2000",
                    "--run-id",
                    "threshold_relaxed_test",
                ]
            )

        self.assertEqual(out, {"ok": True})
        runner_self = run_mock.call_args.args[0]
        self.assertEqual(str(runner_self.config.OPERATING_MODE), K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME)
        self.assertEqual(int(runner_self.config.MIN_CLUSTER_COUNT), 2)
        self.assertEqual(float(runner_self.config.CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE), 0.05)
        self.assertEqual(float(runner_self.config.CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE), 0.05)
        self.assertEqual(int(runner_self.config.PERIOD_STAGE_N), 2000)

    def test_recovery_mode_analysis_writes_requested_csvs(self) -> None:
        root = Path("tmp_pycache") / f"k2_recovery_mode_analysis_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            baseline_dir = root / "baseline"
            mcc2_dir = root / "mcc2"
            threshold_dir = root / "threshold"
            out_dir = root / "out"
            for d in [baseline_dir, mcc2_dir, threshold_dir]:
                d.mkdir(parents=True, exist_ok=False)

            pd.DataFrame(
                [
                    {"epic": "100", "P": 12.0, "manual_review_required": False},
                    {"epic": "200", "P": 16.0, "manual_review_required": False},
                ]
            ).to_csv(baseline_dir / "period_shortlist_best.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "300", "failure_category": "events_filtered_to_zero", "source_reason": "events_filtered_to_zero", "n_events_raw": 0, "n_events_after_filters": 0, "P": pd.NA},
                    {"epic_id": "400", "failure_category": "insufficient_events", "source_reason": "no_cluster_periods", "failure_detail": "insufficient_events_for_period_inference", "n_events_raw": 2, "n_events_after_filters": 1, "P": pd.NA},
                    {"epic_id": "500", "failure_category": "candidate_filter_rejection", "source_reason": "no_cluster_periods", "n_events_raw": 3, "n_events_after_filters": 3, "P": pd.NA},
                ]
            ).to_csv(baseline_dir / "period_shortlist_quarantine.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "events_filtered_to_zero"},
                    {"epic_id": "400", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "n_events_after_filters=1"},
                    {"epic_id": "500", "selected_for_period_stage": True, "terminal_reason": "no_cluster_periods", "source_reason": "no_cluster_periods"},
                ]
            ).to_csv(baseline_dir / "epic_funnel_reasons.csv", index=False)
            pd.DataFrame([{"mcc_policy_mode": "precision_first_default"}]).to_csv(baseline_dir / "period_shortlist_diagnostics.csv", index=False)

            pd.DataFrame(
                [
                    {"epic": "100", "P": 12.0, "manual_review_required": False},
                    {"epic": "200", "P": 16.0, "manual_review_required": False},
                    {"epic": "500", "P": 4.0, "manual_review_required": True},
                ]
            ).to_csv(mcc2_dir / "period_shortlist_best.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "300", "failure_category": "events_filtered_to_zero", "source_reason": "events_filtered_to_zero", "n_events_raw": 0, "n_events_after_filters": 0, "P": pd.NA},
                    {"epic_id": "400", "failure_category": "insufficient_events", "source_reason": "no_cluster_periods", "failure_detail": "insufficient_events_for_period_inference", "n_events_raw": 2, "n_events_after_filters": 1, "P": pd.NA},
                    {"epic_id": "600", "failure_category": "cluster2_guardrail_rejection", "source_reason": "cluster2_guardrail_rejection", "n_events_raw": 2, "n_events_after_filters": 2, "P": 18.0},
                ]
            ).to_csv(mcc2_dir / "period_shortlist_quarantine.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "events_filtered_to_zero"},
                    {"epic_id": "400", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "n_events_after_filters=1"},
                    {"epic_id": "500", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "600", "selected_for_period_stage": True, "terminal_reason": "validated_guardrail_reject", "source_reason": "cluster2_guardrail_rejection"},
                ]
            ).to_csv(mcc2_dir / "epic_funnel_reasons.csv", index=False)
            pd.DataFrame([{"mcc_policy_mode": "supported_high_recall"}]).to_csv(mcc2_dir / "period_shortlist_diagnostics.csv", index=False)

            pd.DataFrame(
                [
                    {"epic": "100", "P": 12.0, "manual_review_required": False},
                    {"epic": "200", "P": 16.0, "manual_review_required": False},
                    {"epic": "500", "P": 4.0, "manual_review_required": True},
                    {"epic": "600", "P": 18.0, "manual_review_required": True},
                ]
            ).to_csv(threshold_dir / "period_shortlist_best.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "300", "failure_category": "events_filtered_to_zero", "source_reason": "events_filtered_to_zero", "n_events_raw": 0, "n_events_after_filters": 0, "P": pd.NA},
                    {"epic_id": "400", "failure_category": "insufficient_events", "source_reason": "no_cluster_periods", "failure_detail": "insufficient_events_for_period_inference", "n_events_raw": 2, "n_events_after_filters": 1, "P": pd.NA},
                ]
            ).to_csv(threshold_dir / "period_shortlist_quarantine.csv", index=False)
            pd.DataFrame(
                [
                    {"epic_id": "100", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "200", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "300", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "events_filtered_to_zero"},
                    {"epic_id": "400", "selected_for_period_stage": True, "terminal_reason": "too_few_events_after_filters", "source_reason": "n_events_after_filters=1"},
                    {"epic_id": "500", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                    {"epic_id": "600", "selected_for_period_stage": True, "terminal_reason": "validated_period", "source_reason": "validated_period"},
                ]
            ).to_csv(threshold_dir / "epic_funnel_reasons.csv", index=False)
            pd.DataFrame([{"mcc_policy_mode": "supported_high_recall_threshold_relaxed"}]).to_csv(threshold_dir / "period_shortlist_diagnostics.csv", index=False)

            out = K2ShortlistRecoveryModeAnalysis().run(
                baseline_run_dir=baseline_dir,
                mcc2_run_dir=mcc2_dir,
                threshold_run_dir=threshold_dir,
                out_dir=out_dir,
            )

            self.assertTrue(Path(out["post_mcc_remaining_failures_by_reason_csv"]).exists())
            self.assertTrue(Path(out["post_mcc_remaining_failures_by_period_bin_csv"]).exists())
            self.assertTrue(Path(out["recovery_mode_comparison_csv"]).exists())
            self.assertTrue(Path(out["rescued_by_mode_csv"]).exists())
            self.assertTrue(Path(out["post_mcc_no_p_available_whiteness_diagnostics_csv"]).exists())
            self.assertTrue(Path(out["no_p_available_upstream_blocker_summary_csv"]).exists())
            self.assertTrue(Path(out["no_p_available_upstream_blocker_by_period_bin_csv"]).exists())
            self.assertTrue(Path(out["no_upstream_events_detected_diagnostics_csv"]).exists())
            self.assertTrue(Path(out["too_few_events_remaining_after_filtering_diagnostics_csv"]).exists())
            self.assertTrue(Path(out["first_failed_upstream_stage_summary_csv"]).exists())
            self.assertTrue(Path(out["first_failed_upstream_stage_by_period_bin_csv"]).exists())
            self.assertTrue(Path(out["event_detection_zero_events_diagnostics_csv"]).exists())
            self.assertTrue(Path(out["event_detection_insufficient_support_diagnostics_csv"]).exists())
            self.assertTrue(Path(out["suspected_zero_event_cause_summary_csv"]).exists())
            self.assertTrue(Path(out["suspected_zero_event_cause_by_period_bin_csv"]).exists())
            self.assertTrue(Path(out["suspected_insufficient_support_cause_summary_csv"]).exists())
            self.assertTrue(Path(out["suspected_insufficient_support_cause_by_period_bin_csv"]).exists())
            self.assertEqual(int(out["threshold_added_vs_mcc2"]), 1)
            self.assertEqual(int(out["period_bin_15_20_delta_vs_mcc2"]), 1)
            self.assertEqual(int(out["manual_review_delta_vs_mcc2"]), 1)

            reasons_df = pd.read_csv(out["post_mcc_remaining_failures_by_reason_csv"])
            reason_map = dict(zip(reasons_df["failure_reason_bucket"], reasons_df["count"]))
            self.assertEqual(int(reason_map["events_filtered_to_zero"]), 1)
            self.assertEqual(int(reason_map["insufficient_events"]), 1)
            self.assertEqual(int(reason_map["other"]), 1)

            compare_df = pd.read_csv(out["recovery_mode_comparison_csv"])
            threshold_row = compare_df.loc[compare_df["mode"] == K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME].iloc[0]
            self.assertEqual(int(threshold_row["shortlisted_count"]), 4)
            self.assertEqual(int(threshold_row["added_vs_mcc2"]), 1)
            self.assertEqual(int(threshold_row["best_count_bin_15_20"]), 2)

            rescued_df = pd.read_csv(out["rescued_by_mode_csv"])
            row_600 = rescued_df.loc[rescued_df["epic_id"].astype(str) == "600"].iloc[0]
            self.assertFalse(bool(row_600["rescued_by_mcc2"]))
            self.assertTrue(bool(row_600["rescued_by_threshold_relaxed"]))
            self.assertEqual(str(row_600["dominant_prior_failure_reason"]), "other")

            no_p_df = pd.read_csv(out["post_mcc_no_p_available_whiteness_diagnostics_csv"])
            self.assertIn("dominant_upstream_blocker", no_p_df.columns)
            self.assertIn("first_failed_upstream_stage", no_p_df.columns)
            row_300 = no_p_df.loc[no_p_df["epic_id"].astype(str) == "300"].iloc[0]
            row_400 = no_p_df.loc[no_p_df["epic_id"].astype(str) == "400"].iloc[0]
            self.assertEqual(str(row_300["dominant_upstream_blocker"]), "no_upstream_events_detected")
            self.assertEqual(str(row_400["dominant_upstream_blocker"]), "too_few_events_remaining_after_filtering")
            self.assertEqual(str(row_300["first_failed_upstream_stage"]), "event_detection_produced_zero_events")
            self.assertEqual(str(row_400["first_failed_upstream_stage"]), "event_detection_produced_insufficient_support")

            blocker_summary_df = pd.read_csv(out["no_p_available_upstream_blocker_summary_csv"])
            blocker_map = dict(zip(blocker_summary_df["dominant_upstream_blocker"], blocker_summary_df["count"]))
            self.assertEqual(int(blocker_map["no_upstream_events_detected"]), 1)
            self.assertEqual(int(blocker_map["too_few_events_remaining_after_filtering"]), 1)

            no_upstream_df = pd.read_csv(out["no_upstream_events_detected_diagnostics_csv"])
            too_few_df = pd.read_csv(out["too_few_events_remaining_after_filtering_diagnostics_csv"])
            self.assertEqual(len(no_upstream_df), 1)
            self.assertEqual(len(too_few_df), 1)
            self.assertEqual(str(no_upstream_df.iloc[0]["suspected_zero_event_cause"]), "detector_sensitivity_or_candidate_generation")
            self.assertEqual(str(too_few_df.iloc[0]["suspected_insufficient_support_cause"]), "downstream_event_retention_before_shortlist")

            stage_summary_df = pd.read_csv(out["first_failed_upstream_stage_summary_csv"])
            stage_map = dict(zip(stage_summary_df["first_failed_upstream_stage"], stage_summary_df["count"]))
            self.assertEqual(int(stage_map["event_detection_produced_zero_events"]), 1)
            self.assertEqual(int(stage_map["event_detection_produced_insufficient_support"]), 1)
            zero_cause_summary_df = pd.read_csv(out["suspected_zero_event_cause_summary_csv"])
            insufficient_cause_summary_df = pd.read_csv(out["suspected_insufficient_support_cause_summary_csv"])
            zero_cause_map = dict(zip(zero_cause_summary_df["suspected_zero_event_cause"], zero_cause_summary_df["count"]))
            insufficient_cause_map = dict(zip(insufficient_cause_summary_df["suspected_insufficient_support_cause"], insufficient_cause_summary_df["count"]))
            self.assertEqual(int(zero_cause_map["detector_sensitivity_or_candidate_generation"]), 1)
            self.assertEqual(int(insufficient_cause_map["downstream_event_retention_before_shortlist"]), 1)
        finally:
            if root.exists():
                for path in sorted(root.rglob("*"), reverse=True):
                    try:
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            path.rmdir()
                    except OSError:
                        pass
                try:
                    root.rmdir()
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
