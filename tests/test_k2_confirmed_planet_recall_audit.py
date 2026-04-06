from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2ConfirmedPlanetRecallAudit import K2ConfirmedPlanetRecallAudit


class K2ConfirmedPlanetRecallAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_confirmed_planet_recall_audit_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_audit_classifies_confirmed_epics_and_writes_outputs(self) -> None:
        reference_csv = self.case_dir / "reference.csv"
        batch_csv = self.case_dir / "batch_results.csv"
        best_csv = self.case_dir / "period_shortlist_best.csv"
        quarantine_csv = self.case_dir / "period_shortlist_quarantine.csv"
        funnel_csv = self.case_dir / "epic_funnel_reasons.csv"
        diagnostics_csv = self.case_dir / "period_shortlist_diagnostics.csv"
        audit_csv = self.case_dir / "k2_confirmed_planet_recall_audit.csv"
        rollup_csv = self.case_dir / "k2_confirmed_planet_recall_rollup.csv"
        false_negatives_csv = self.case_dir / "k2_confirmed_false_negatives.csv"

        pd.DataFrame(
            [
                {"pl_name": "K2-1 b", "hostname": "K2-1", "default_flag": 1, "disc_facility": "K2", "epic_id": "EPIC 201", "k2_name": "K2-1 b"},
                {"pl_name": "K2-2 b", "hostname": "K2-2", "default_flag": 1, "disc_facility": "K2", "epic_id": "EPIC 202", "k2_name": "K2-2 b"},
                {"pl_name": "K2-3 b", "hostname": "K2-3", "default_flag": 1, "disc_facility": "K2", "epic_id": "EPIC 203", "k2_name": "K2-3 b"},
                {"pl_name": "K2-4 b", "hostname": "K2-4", "default_flag": 1, "disc_facility": "K2", "epic_id": "EPIC 204", "k2_name": "K2-4 b"},
                {"pl_name": "K2-5 b", "hostname": "K2-5", "default_flag": 1, "disc_facility": "K2", "epic_id": "EPIC 205", "k2_name": "K2-5 b"},
                {"pl_name": "K2-X b", "hostname": "K2-X", "default_flag": 1, "disc_facility": "K2", "epic_id": "", "k2_name": ""},
            ]
        ).to_csv(reference_csv, index=False)

        pd.DataFrame(
            [
                {"query": "EPIC 201", "epic_id": "EPIC_201", "triage_status": "ok", "n_events": 4, "best_shape_score": 0.8, "best_depth_snr": 5.0},
                {"query": "EPIC 202", "epic_id": "EPIC_202", "triage_status": "ok", "n_events": 3, "best_shape_score": 0.7, "best_depth_snr": 4.0},
                {"query": "EPIC 203", "epic_id": "EPIC_203", "triage_status": "ok", "n_events": 2, "best_shape_score": 0.6, "best_depth_snr": 3.0},
                {"query": "EPIC 204", "epic_id": "EPIC_204", "triage_status": "ok", "n_events": 0, "best_shape_score": 0.5, "best_depth_snr": 2.0},
            ]
        ).to_csv(batch_csv, index=False)

        pd.DataFrame([{"epic": "201", "query": "EPIC 201", "reason": "validated", "P": 5.0, "n_events_after_filters": 4}]).to_csv(best_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "202",
                    "query": "EPIC 202",
                    "reason": "P_null_or_missing",
                    "source_reason": "no_cluster_periods",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "n_events_after_filters": 3,
                },
                {
                    "epic_id": "204",
                    "query": "EPIC 204",
                    "reason": "P_null_or_missing",
                    "source_reason": "events_filtered_to_zero",
                    "failure_category": "events_filtered_to_zero",
                    "shortlist_rejection_reason": "events_filtered_to_zero",
                    "n_events_after_filters": 0,
                },
            ]
        ).to_csv(quarantine_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "202",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": '{"query":"EPIC 202","selected_for_period_stage":true}',
                },
                {
                    "epic_id": "203",
                    "terminal_reason": "other",
                    "source_reason": "not_in_period_stage_randomN",
                    "stage_reached": "pre_period_gate",
                    "details_json": '{"query":"EPIC 203","selected_for_period_stage":false,"period_stage_mode":"randomN"}',
                },
                {
                    "epic_id": "204",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "events_filtered_to_zero",
                    "stage_reached": "period_inference",
                    "details_json": '{"query":"EPIC 204","selected_for_period_stage":true}',
                },
            ]
        ).to_csv(funnel_csv, index=False)

        pd.DataFrame([{"period_cap_days": 20.0}]).to_csv(diagnostics_csv, index=False)

        out = K2ConfirmedPlanetRecallAudit().run(
            batch_results_csv=batch_csv,
            best_csv=best_csv,
            quarantine_csv=quarantine_csv,
            funnel_csv=funnel_csv,
            diagnostics_csv=diagnostics_csv,
            reference_csv=reference_csv,
            refresh_reference=False,
            audit_csv=audit_csv,
            rollup_csv=rollup_csv,
            false_negatives_csv=false_negatives_csv,
        )

        self.assertEqual(int(out["confirmed_total"]), 5)
        self.assertEqual(int(out["confirmed_in_best"]), 1)
        self.assertEqual(int(out["confirmed_in_quarantine"]), 1)
        self.assertEqual(int(out["confirmed_detected_but_failed_downstream"]), 1)
        self.assertEqual(int(out["confirmed_no_events_after_filters"]), 1)
        self.assertEqual(int(out["confirmed_not_seen"]), 1)
        self.assertAlmostEqual(float(out["confirmed_recall_best_only"]), 1.0 / 5.0)
        self.assertAlmostEqual(float(out["confirmed_recall_best_plus_quarantine"]), 2.0 / 5.0)

        audit_df = pd.read_csv(audit_csv)
        outcome_map = dict(zip(audit_df["epic_id"], audit_df["outcome_label"]))
        self.assertEqual(outcome_map["EPIC_201"], "recovered_in_best")
        self.assertEqual(outcome_map["EPIC_202"], "recovered_in_quarantine")
        self.assertEqual(outcome_map["EPIC_203"], "detected_but_failed_downstream")
        self.assertEqual(outcome_map["EPIC_204"], "no_events_after_filters")
        self.assertEqual(outcome_map["EPIC_205"], "not_seen / not_matched")

        rollup_df = pd.read_csv(rollup_csv)
        rollup_map = dict(zip(rollup_df["metric"], rollup_df["value"]))
        self.assertEqual(int(rollup_map["reference_rows_without_epic_mapping"]), 1)
        self.assertEqual(float(rollup_map["confirmed_recall_best_only"]), 1.0 / 5.0)

        false_negatives_df = pd.read_csv(false_negatives_csv)
        self.assertEqual(len(false_negatives_df), 4)


if __name__ == "__main__":
    unittest.main()
