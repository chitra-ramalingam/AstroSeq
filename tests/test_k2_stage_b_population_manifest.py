from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2StageBPopulationManifest import K2StageBPopulationManifest


class K2StageBPopulationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_stage_b_population_manifest_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_stage_b_manifest_builds_expected_buckets_and_unresolved_subset(self) -> None:
        batch_csv = self.case_dir / "batch_results.csv"
        retriaged_csv = self.case_dir / "batch_results_retriaged.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"
        best_csv = self.case_dir / "period_shortlist_best.csv"
        quarantine_csv = self.case_dir / "period_shortlist_quarantine.csv"
        funnel_csv = self.case_dir / "epic_funnel_reasons.csv"
        reference_csv = self.case_dir / "reference.csv"
        master_csv = self.case_dir / "stage_b_master.csv"
        unresolved_csv = self.case_dir / "stage_b_unresolved.csv"
        rollup_csv = self.case_dir / "stage_b_rollup.csv"

        processed_rows = [
            {"epic_id": "EPIC_101", "query": "EPIC 101", "triage_status": "ok", "triage_usable": True},
            {"epic_id": "EPIC_102", "query": "EPIC 102", "triage_status": "ok", "triage_usable": True},
            {"epic_id": "EPIC_103", "query": "EPIC 103", "triage_status": "ok", "triage_usable": True},
            {"epic_id": "EPIC_104", "query": "EPIC 104", "triage_status": "ok", "triage_usable": True},
            {"epic_id": "EPIC_105", "query": "EPIC 105", "triage_status": "ok", "triage_usable": True},
        ]
        pd.DataFrame(processed_rows).to_csv(batch_csv, index=False)
        pd.DataFrame(processed_rows).to_csv(retriaged_csv, index=False)
        pd.DataFrame(processed_rows).to_csv(whiteness_csv, index=False)

        pd.DataFrame([{"epic": "101", "query": "EPIC 101", "reason": "validated"}]).to_csv(best_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "105",
                    "query": "EPIC 105",
                    "reason": "P_null_or_missing",
                    "source_reason": "candidate_filter_rejection",
                    "failure_category": "candidate_filter_rejection",
                    "failure_detail": "cluster_count_below_minimum",
                }
            ]
        ).to_csv(quarantine_csv, index=False)

        pd.DataFrame(
            [
                {
                    "epic_id": "101",
                    "terminal_reason": "other",
                    "source_reason": "validated_period",
                    "stage_reached": "validated_period",
                    "details_json": '{"selected_for_period_stage": true}',
                },
                {
                    "epic_id": "102",
                    "terminal_reason": "other",
                    "source_reason": "not_in_period_stage_random_sample_n5000",
                    "stage_reached": "pre_period_gate",
                    "details_json": '{"selected_for_period_stage": false}',
                },
                {
                    "epic_id": "103",
                    "terminal_reason": "no_lightcurve/load_failed",
                    "source_reason": "triage_status=error",
                    "stage_reached": "lightcurve_load",
                    "details_json": '{"selected_for_period_stage": false, "load_failed_exception_type": "FileNotFoundError"}',
                },
                {
                    "epic_id": "104",
                    "terminal_reason": "other",
                    "source_reason": "not_in_period_stage_random_sample_n5000",
                    "stage_reached": "pre_period_gate",
                    "details_json": '{"selected_for_period_stage": false}',
                },
                {
                    "epic_id": "105",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": '{"selected_for_period_stage": true}',
                },
            ]
        ).to_csv(funnel_csv, index=False)

        pd.DataFrame(
            [
                {"pl_name": "K2-102 b", "hostname": "K2-102", "k2_name": "K2-102 b", "epic_id": "EPIC 102"},
                {"pl_name": "K2-106 b", "hostname": "K2-106", "k2_name": "K2-106 b", "epic_id": "EPIC 106"},
            ]
        ).to_csv(reference_csv, index=False)

        out = K2StageBPopulationManifest().run(
            batch_results_csv=batch_csv,
            batch_results_retriaged_csv=retriaged_csv,
            batch_results_whiteness_csv=whiteness_csv,
            best_csv=best_csv,
            quarantine_csv=quarantine_csv,
            funnel_csv=funnel_csv,
            reference_csv=reference_csv,
            master_csv=master_csv,
            unresolved_csv=unresolved_csv,
            rollup_csv=rollup_csv,
        )

        self.assertEqual(int(out["total_relevant_epics"]), 6)
        self.assertEqual(int(out["resolved_already_classified"]), 2)
        self.assertEqual(int(out["known_confirmed_calibration_cases"]), 1)
        self.assertEqual(int(out["unresolved_needing_triage"]), 1)
        self.assertEqual(int(out["load_failed_missing_light_curve"]), 1)
        self.assertEqual(int(out["outside_current_scope"]), 1)

        master_df = pd.read_csv(master_csv)
        bucket_map = dict(zip(master_df["epic_id_norm"].astype(str), master_df["current_status_bucket"]))
        unresolved_map = dict(zip(master_df["epic_id_norm"].astype(str), master_df["unresolved"].astype(bool)))
        shortlist_map = dict(zip(master_df["epic_id_norm"].astype(str), master_df["already_shortlisted"].astype(bool)))
        quarantine_map = dict(zip(master_df["epic_id_norm"].astype(str), master_df["already_quarantined"].astype(bool)))
        confirmed_map = dict(zip(master_df["epic_id_norm"].astype(str), master_df["known_confirmed"].astype(bool)))

        self.assertEqual(bucket_map["101"], K2StageBPopulationManifest.BUCKET_RESOLVED)
        self.assertEqual(bucket_map["102"], K2StageBPopulationManifest.BUCKET_KNOWN_CONFIRMED)
        self.assertEqual(bucket_map["103"], K2StageBPopulationManifest.BUCKET_LOAD_FAILED)
        self.assertEqual(bucket_map["104"], K2StageBPopulationManifest.BUCKET_UNRESOLVED)
        self.assertEqual(bucket_map["105"], K2StageBPopulationManifest.BUCKET_RESOLVED)
        self.assertEqual(bucket_map["106"], K2StageBPopulationManifest.BUCKET_OUTSIDE_SCOPE)

        self.assertTrue(shortlist_map["101"])
        self.assertTrue(quarantine_map["105"])
        self.assertTrue(confirmed_map["102"])
        self.assertTrue(confirmed_map["106"])
        self.assertFalse(unresolved_map["102"])
        self.assertTrue(unresolved_map["104"])

        unresolved_df = pd.read_csv(unresolved_csv)
        self.assertEqual(list(unresolved_df["epic_id_norm"].astype(str)), ["104"])

        rollup_df = pd.read_csv(rollup_csv)
        bucket_rollup = rollup_df.loc[rollup_df["section"].eq("bucket_counts")].set_index("metric")["value"].to_dict()
        self.assertEqual(int(bucket_rollup[K2StageBPopulationManifest.BUCKET_RESOLVED]), 2)
        self.assertEqual(int(bucket_rollup[K2StageBPopulationManifest.BUCKET_KNOWN_CONFIRMED]), 1)
        self.assertEqual(int(bucket_rollup[K2StageBPopulationManifest.BUCKET_UNRESOLVED]), 1)
        self.assertEqual(int(bucket_rollup[K2StageBPopulationManifest.BUCKET_LOAD_FAILED]), 1)
        self.assertEqual(int(bucket_rollup[K2StageBPopulationManifest.BUCKET_OUTSIDE_SCOPE]), 1)


if __name__ == "__main__":
    unittest.main()
