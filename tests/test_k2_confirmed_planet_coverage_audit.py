from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2ConfirmedPlanetCoverageAudit import K2ConfirmedPlanetCoverageAudit


class K2ConfirmedPlanetCoverageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_confirmed_planet_coverage_audit_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_coverage_audit_classifies_core_buckets(self) -> None:
        reference_csv = self.case_dir / "reference.csv"
        recall_audit_csv = self.case_dir / "recall.csv"
        batch_csv = self.case_dir / "batch_results.csv"
        retriaged_csv = self.case_dir / "batch_results_retriaged.csv"
        whiteness_csv = self.case_dir / "batch_results_whiteness.csv"
        audit_csv = self.case_dir / "coverage.csv"
        rollup_csv = self.case_dir / "coverage_rollup.csv"

        pd.DataFrame(
            [
                {"pl_name": "K2-1 b", "hostname": "K2-1", "k2_name": "K2-1 b", "epic_id": "EPIC 201"},
                {"pl_name": "K2-2 b", "hostname": "K2-2", "k2_name": "K2-2 b", "epic_id": "EPIC 202"},
                {"pl_name": "K2-3 b", "hostname": "K2-3", "k2_name": "K2-3 b", "epic_id": "EPIC 203"},
                {"pl_name": "K2-4 b", "hostname": "K2-4", "k2_name": "K2-4 b", "epic_id": "EPIC 204"},
            ]
        ).to_csv(reference_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_201", "epic_id_norm": "201", "outcome_label": "recovered_in_best"},
                {"epic_id": "EPIC_202", "epic_id_norm": "202", "outcome_label": "no_events_after_filters"},
                {"epic_id": "EPIC_203", "epic_id_norm": "203", "outcome_label": "not_seen / not_matched"},
                {"epic_id": "EPIC_204", "epic_id_norm": "204", "outcome_label": "not_seen / not_matched"},
            ]
        ).to_csv(recall_audit_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_201", "query": "EPIC 201", "triage_status": "ok"},
                {"epic_id": "EPIC_202", "query": "EPIC 202", "triage_status": "error", "error_stage": "load_lc"},
            ]
        ).to_csv(batch_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_201", "query": "EPIC 201"},
                {"epic_id": "EPIC_202", "query": "EPIC 202"},
                {"epic_id": "EPIC_203", "query": "EPIC 203"},
            ]
        ).to_csv(retriaged_csv, index=False)

        pd.DataFrame(
            [
                {"epic_id": "EPIC_201", "query": "EPIC 201"},
                {"epic_id": "EPIC_202", "query": "EPIC 202"},
                {"epic_id": "EPIC_203", "query": "EPIC 203"},
            ]
        ).to_csv(whiteness_csv, index=False)

        out = K2ConfirmedPlanetCoverageAudit().run(
            reference_csv=reference_csv,
            recall_audit_csv=recall_audit_csv,
            batch_results_csv=batch_csv,
            batch_results_retriaged_csv=retriaged_csv,
            batch_results_whiteness_csv=whiteness_csv,
            audit_csv=audit_csv,
            rollup_csv=rollup_csv,
        )

        self.assertEqual(int(out["confirmed_total"]), 4)
        self.assertEqual(int(out["matched_to_processed_universe"]), 1)
        self.assertEqual(int(out["not_processed"]), 1)
        self.assertEqual(int(out["load_failed"]), 1)
        self.assertEqual(int(out["outside_scope"]), 1)
        self.assertEqual(int(out["id_mismatch"]), 0)
        self.assertIn(
            str(out["final_dominant_coverage_blocker"]),
            {
                "present in raw K2 manifest but never processed",
                "outside current pipeline universe / campaign scope",
            },
        )
        self.assertEqual(str(out["coverage_vs_science_conclusion"]), "incomplete population coverage")

        audit_df = pd.read_csv(audit_csv)
        bucket_map = dict(zip(audit_df["epic_id"], audit_df["coverage_bucket"]))
        self.assertEqual(bucket_map["EPIC_201"], "present in AstroSeq processed universe and matched")
        self.assertEqual(bucket_map["EPIC_202"], "processed but no light curve / load failed")
        self.assertEqual(bucket_map["EPIC_203"], "present in raw K2 manifest but never processed")
        self.assertEqual(bucket_map["EPIC_204"], "outside current pipeline universe / campaign scope")


if __name__ == "__main__":
    unittest.main()
