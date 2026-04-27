from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageCActionQueue import K2StageCActionQueue


class K2StageCActionQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_c_action_queue_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_c_routes_unresolved_rows_into_operational_lanes(self) -> None:
        stage_b_csv = self.case_dir / "stage_b_unresolved.csv"
        batch_csv = self.case_dir / "batch_results.csv"
        funnel_csv = self.case_dir / "epic_funnel_reasons.csv"
        action_queue_csv = self.case_dir / "k2_stage_c_action_queue.csv"
        summary_csv = self.case_dir / "k2_stage_c_action_queue_summary.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_101",
                    "epic_id_norm": "101",
                    "current_status_bucket": "unresolved and still needing triage/classification",
                    "unresolved": True,
                    "outside_current_scope": False,
                    "load_failed_or_missing_light_curve": False,
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
                {
                    "epic_id": "EPIC_102",
                    "epic_id_norm": "102",
                    "current_status_bucket": "unresolved and still needing triage/classification",
                    "unresolved": True,
                    "outside_current_scope": False,
                    "load_failed_or_missing_light_curve": False,
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
                {
                    "epic_id": "EPIC_103",
                    "epic_id_norm": "103",
                    "current_status_bucket": "unresolved and still needing triage/classification",
                    "unresolved": True,
                    "outside_current_scope": False,
                    "load_failed_or_missing_light_curve": False,
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
                {
                    "epic_id": "EPIC_104",
                    "epic_id_norm": "104",
                    "current_status_bucket": "unresolved and still needing triage/classification",
                    "unresolved": True,
                    "outside_current_scope": False,
                    "load_failed_or_missing_light_curve": False,
                    "period_source_reason": "not_in_period_stage_random_sample_n5000",
                    "period_terminal_reason": "other",
                },
            ]
        ).to_csv(stage_b_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_101",
                    "query": "EPIC 101",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 5,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 4.0,
                },
                {
                    "epic_id": "EPIC_102",
                    "query": "EPIC 102",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_why_not_usable": "",
                    "n_events": 1,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 3.0,
                },
                {
                    "epic_id": "EPIC_103",
                    "query": "EPIC 103",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "outlier_rate_global>0.1",
                    "n_events": 9,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 9.5,
                },
                {
                    "epic_id": "EPIC_104",
                    "query": "EPIC 104",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_why_not_usable": "outlier_rate_6sigma>0.02",
                    "n_events": 2,
                    "n_periods_proposed": 0,
                    "best_depth_snr": 3.2,
                },
            ]
        ).to_csv(batch_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "101", "terminal_reason": "other", "source_reason": "not_in_period_stage_random_sample_n5000"},
                {"epic_id": "102", "terminal_reason": "other", "source_reason": "not_in_period_stage_random_sample_n5000"},
                {"epic_id": "103", "terminal_reason": "other", "source_reason": "not_in_period_stage_random_sample_n5000"},
                {"epic_id": "104", "terminal_reason": "other", "source_reason": "not_in_period_stage_random_sample_n5000"},
            ]
        ).to_csv(funnel_csv, index=False)

        out = K2StageCActionQueue().run(
            stage_b_unresolved_csv=stage_b_csv,
            batch_results_csv=batch_csv,
            funnel_csv=funnel_csv,
            action_queue_csv=action_queue_csv,
            summary_csv=summary_csv,
        )

        self.assertEqual(int(out["rows_total"]), 4)
        self.assertEqual(int(out["process_now_count"]), 1)
        self.assertEqual(int(out["low_priority_or_defer_count"]), 1)
        self.assertEqual(int(out["rescue_path_candidate_count"]), 1)
        self.assertEqual(int(out["needs_manual_review_count"]), 1)
        self.assertEqual(int(out["blocked_missing_light_curve_count"]), 0)
        self.assertEqual(int(out["outside_scope_count"]), 0)

        queue_df = pd.read_csv(action_queue_csv)
        action_map = dict(zip(queue_df["epic_id_norm"].astype(str), queue_df["next_action"]))
        priority_map = dict(zip(queue_df["epic_id_norm"].astype(str), queue_df["priority"]))
        rule_map = dict(zip(queue_df["epic_id_norm"].astype(str), queue_df["routing_rule_id"]))

        self.assertEqual(action_map["101"], K2StageCActionQueue.ACTION_PROCESS_NOW)
        self.assertEqual(action_map["102"], K2StageCActionQueue.ACTION_LOW_PRIORITY_OR_DEFER)
        self.assertEqual(action_map["103"], K2StageCActionQueue.ACTION_RESCUE_PATH_CANDIDATE)
        self.assertEqual(action_map["104"], K2StageCActionQueue.ACTION_NEEDS_MANUAL_REVIEW)
        self.assertEqual(priority_map["101"], K2StageCActionQueue.PRIORITY_MEDIUM)
        self.assertEqual(priority_map["102"], K2StageCActionQueue.PRIORITY_LOW)
        self.assertEqual(priority_map["103"], K2StageCActionQueue.PRIORITY_HIGH)
        self.assertEqual(rule_map["103"], "quality_flag_with_saved_signal")

        summary_df = pd.read_csv(summary_csv)
        next_action_counts = summary_df.loc[summary_df["dimension"].eq("next_action")].set_index("value")["count"].to_dict()
        self.assertEqual(int(next_action_counts[K2StageCActionQueue.ACTION_PROCESS_NOW]), 1)
        self.assertEqual(int(next_action_counts[K2StageCActionQueue.ACTION_LOW_PRIORITY_OR_DEFER]), 1)
        self.assertEqual(int(next_action_counts[K2StageCActionQueue.ACTION_RESCUE_PATH_CANDIDATE]), 1)
        self.assertEqual(int(next_action_counts[K2StageCActionQueue.ACTION_NEEDS_MANUAL_REVIEW]), 1)


if __name__ == "__main__":
    unittest.main()
