from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageDExecutionPackaging import K2StageDExecutionPackaging


class K2StageDExecutionPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_d_execution_packaging_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_d_packages_stage_c_into_execution_files_with_stable_order(self) -> None:
        stage_c_csv = self.case_dir / "k2_stage_c_action_queue.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_201",
                    "query": "EPIC 201",
                    "current_status": "ready_for_default_pass_high_signal",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "process_now",
                    "priority": "high",
                    "reason_detail": "high-1",
                    "source_file": "x",
                    "epic_id_norm": "201",
                    "routing_rule_id": "usable_with_saved_signal_support",
                    "priority_rule_id": "high_priority_process_now",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 4,
                    "n_periods_proposed": 2,
                    "best_depth_snr": 8.0,
                },
                {
                    "epic_id": "EPIC_202",
                    "query": "EPIC 202",
                    "current_status": "ready_for_default_pass_high_signal",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "process_now",
                    "priority": "high",
                    "reason_detail": "high-2",
                    "source_file": "x",
                    "epic_id_norm": "202",
                    "routing_rule_id": "usable_with_saved_signal_support",
                    "priority_rule_id": "high_priority_process_now",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 9,
                    "n_periods_proposed": 5,
                    "best_depth_snr": None,
                },
                {
                    "epic_id": "EPIC_203",
                    "query": "EPIC 203",
                    "current_status": "ready_for_default_pass",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "process_now",
                    "priority": "medium",
                    "reason_detail": "medium",
                    "source_file": "x",
                    "epic_id_norm": "203",
                    "routing_rule_id": "usable_default_queue",
                    "priority_rule_id": "medium_priority_process_now",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 3,
                    "n_periods_proposed": 1,
                    "best_depth_snr": 6.5,
                },
                {
                    "epic_id": "EPIC_204",
                    "query": "EPIC 204",
                    "current_status": "quality_flagged_but_signal_present",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "rescue_path_candidate",
                    "priority": "high",
                    "reason_detail": "rescue",
                    "source_file": "x",
                    "epic_id_norm": "204",
                    "routing_rule_id": "quality_flag_with_saved_signal",
                    "priority_rule_id": "high_priority_rescue_queue",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "quality_flagged",
                    "triage_usable": False,
                    "triage_why_not_usable": "x",
                    "n_events": 2,
                    "n_periods_proposed": 1,
                    "best_depth_snr": 7.4,
                },
                {
                    "epic_id": "EPIC_205",
                    "query": "EPIC 205",
                    "current_status": "quality_flagged_needs_manual_review",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "needs_manual_review",
                    "priority": "medium",
                    "reason_detail": "manual",
                    "source_file": "x",
                    "epic_id_norm": "205",
                    "routing_rule_id": "quality_flag_without_saved_signal",
                    "priority_rule_id": "medium_priority_manual_review",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "quality_flagged",
                    "triage_usable": False,
                    "triage_why_not_usable": "x",
                    "n_events": None,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 2.2,
                },
                {
                    "epic_id": "EPIC_206",
                    "query": "EPIC 206",
                    "current_status": "single_event_no_period_signal",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "low_priority_or_defer",
                    "priority": "low",
                    "reason_detail": "defer-1",
                    "source_file": "x",
                    "epic_id_norm": "206",
                    "routing_rule_id": "single_event_and_no_proposed_periods",
                    "priority_rule_id": "low_priority_defer_queue",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 1,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 2.0,
                },
                {
                    "epic_id": "EPIC_207",
                    "query": "EPIC 207",
                    "current_status": "single_event_no_period_signal",
                    "stage_b_bucket": "unresolved",
                    "data_availability": "light_curve_available_in_saved_outputs",
                    "scope_status": "in_scope_unresolved",
                    "next_action": "low_priority_or_defer",
                    "priority": "low",
                    "reason_detail": "defer-2",
                    "source_file": "x",
                    "epic_id_norm": "207",
                    "routing_rule_id": "single_event_and_no_proposed_periods",
                    "priority_rule_id": "low_priority_defer_queue",
                    "derived_from_columns": "x",
                    "period_source_reason": "x",
                    "period_terminal_reason": "x",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 1,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 1.0,
                },
            ]
        ).to_csv(stage_c_csv, index=False)

        out = K2StageDExecutionPackaging().run(stage_c_csv=stage_c_csv, out_dir=self.case_dir)

        self.assertEqual(int(out["rows_total"]), 7)
        self.assertEqual(int(out["process_now_high_priority_count"]), 2)
        self.assertEqual(int(out["process_now_medium_priority_count"]), 1)
        self.assertEqual(int(out["rescue_candidates_count"]), 1)
        self.assertEqual(int(out["manual_review_count"]), 1)
        self.assertEqual(int(out["deferred_count"]), 2)
        self.assertEqual(int(out["missing_ranking_fields"]["best_depth_snr"]), 1)
        self.assertEqual(int(out["missing_ranking_fields"]["n_events"]), 1)
        self.assertEqual(int(out["missing_ranking_fields"]["rows_with_any_missing_ranking_field"]), 2)

        high_df = pd.read_csv(out["process_now_high_priority_csv"])
        self.assertEqual(list(high_df["epic_id_norm"].astype(str)), ["201", "202"])
        self.assertEqual(list(high_df["execution_order"].astype(int)), [1, 2])

        medium_df = pd.read_csv(out["process_now_medium_priority_csv"])
        self.assertEqual(list(medium_df["epic_id_norm"].astype(str)), ["203"])
        self.assertIn("execution_order", medium_df.columns)

        rescue_df = pd.read_csv(out["rescue_candidates_csv"])
        self.assertEqual(list(rescue_df["epic_id_norm"].astype(str)), ["204"])

        manual_df = pd.read_csv(out["manual_review_csv"])
        self.assertEqual(list(manual_df["epic_id_norm"].astype(str)), ["205"])

        deferred_df = pd.read_csv(out["deferred_csv"])
        self.assertEqual(list(deferred_df["epic_id_norm"].astype(str)), ["206", "207"])
        self.assertEqual(list(deferred_df["execution_order"].astype(int)), [1, 2])


if __name__ == "__main__":
    unittest.main()
