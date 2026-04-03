from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderCachedFailedDownstreamReport import (
    K2DetectorQualityGatedBroaderCachedFailedDownstreamReport,
)


class K2DetectorQualityGatedBroaderCachedFailedDownstreamReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_detector_quality_gated_broader_cached_failed_report_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def test_report_joins_normalized_epics_and_writes_requested_outputs(self) -> None:
        winners_csv = self.case_dir / "detector_quality_gated_broader_winners.csv"
        best_csv = self.case_dir / "Apr1_period_shortlist_best.csv"
        quarantine_csv = self.case_dir / "Apr1_period_shortlist_quarantine.csv"
        funnel_csv = self.case_dir / "Apr1_epic_funnel_reasons.csv"
        summary_csv = self.case_dir / "detector_quality_gated_broader_downstream_summary.csv"
        quarantined_winners_csv = self.case_dir / "detector_quality_gated_broader_quarantined_winners.csv"
        best_winners_csv = self.case_dir / "detector_quality_gated_broader_best_winners.csv"

        pd.DataFrame(
            [
                {"epic_id": "EPIC_101", "gained_extra_events": True},
                {"epic_id": "EPIC_102", "gained_extra_events": True},
                {"epic_id": "EPIC_103", "gained_extra_events": True},
            ]
        ).to_csv(winners_csv, index=False)
        pd.DataFrame(
            [
                {"epic": "101", "query": "EPIC 101", "P": 7.5, "reason": "validated", "manual_review_required": False},
                {"epic": "999", "query": "EPIC 999", "P": 3.0, "reason": "validated", "manual_review_required": False},
            ]
        ).to_csv(best_csv, index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_102",
                    "query": "EPIC 102",
                    "reason": "P_found_but_rejected",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "all_candidate_periods_below_min_cluster_count",
                },
                {
                    "epic_id": "777",
                    "query": "EPIC 777",
                    "reason": "P_null_or_missing",
                    "failure_category": "events_filtered_to_zero",
                    "shortlist_rejection_reason": "events_filtered_to_zero",
                },
            ]
        ).to_csv(quarantine_csv, index=False)
        pd.DataFrame(
            [
                {
                    "epic_id": "102",
                    "query": "EPIC 102",
                    "terminal_reason": "no_cluster_periods",
                    "shortlist_rejection_reason": "all_candidate_periods_below_min_cluster_count",
                },
                {
                    "epic_id": "103",
                    "query": "EPIC 103",
                    "terminal_reason": "other",
                    "shortlist_rejection_reason": "",
                },
            ]
        ).to_csv(funnel_csv, index=False)

        out = K2DetectorQualityGatedBroaderCachedFailedDownstreamReport().run(
            winners_csv=winners_csv,
            best_csv=best_csv,
            quarantine_csv=quarantine_csv,
            funnel_csv=funnel_csv,
            summary_csv=summary_csv,
            quarantined_winners_csv=quarantined_winners_csv,
            best_winners_csv=best_winners_csv,
        )

        self.assertEqual(int(out["winners_total"]), 3)
        self.assertEqual(int(out["winners_in_best"]), 1)
        self.assertEqual(int(out["winners_in_quarantine"]), 1)
        self.assertAlmostEqual(float(out["downstream_conversion_rate"]), 1.0 / 3.0)

        self.assertTrue(summary_csv.exists())
        self.assertTrue(quarantined_winners_csv.exists())
        self.assertTrue(best_winners_csv.exists())

        best_winners_df = pd.read_csv(best_winners_csv)
        self.assertEqual(list(best_winners_df["epic_id"]), ["EPIC_101"])
        self.assertEqual(list(best_winners_df["best_epic"]), [101])
        self.assertEqual(list(best_winners_df["epic_id_norm"]), [101])

        quarantined_df = pd.read_csv(quarantined_winners_csv)
        self.assertEqual(list(quarantined_df["epic_id"]), ["EPIC_102"])
        self.assertEqual(list(quarantined_df["epic_id_norm"]), [102])
        row_102 = quarantined_df.iloc[0]
        self.assertEqual(str(row_102["failure_category"]), "candidate_filter_rejection")
        self.assertEqual(
            str(row_102["shortlist_rejection_reason"]),
            "all_candidate_periods_below_min_cluster_count",
        )
        self.assertEqual(str(row_102["terminal_reason"]), "no_cluster_periods")

        summary_df = pd.read_csv(summary_csv)
        summary_map = dict(
            zip(
                summary_df.loc[summary_df["section"] == "summary", "metric"],
                summary_df.loc[summary_df["section"] == "summary", "value"],
            )
        )
        self.assertEqual(int(summary_map["winners_total"]), 3)
        self.assertEqual(int(summary_map["winners_in_best"]), 1)
        self.assertEqual(int(summary_map["winners_in_quarantine"]), 1)

        failure_rows = summary_df.loc[
            (summary_df["section"] == "top_failure_reasons")
            & (summary_df["reason_column"] == "terminal_reason")
        ]
        self.assertEqual(list(failure_rows["metric"]), ["no_cluster_periods"])
        self.assertEqual(int(failure_rows.iloc[0]["value"]), 1)

    def test_report_fails_clearly_when_required_columns_are_missing(self) -> None:
        winners_csv = self.case_dir / "detector_quality_gated_broader_winners.csv"
        best_csv = self.case_dir / "Apr1_period_shortlist_best.csv"
        quarantine_csv = self.case_dir / "Apr1_period_shortlist_quarantine.csv"
        funnel_csv = self.case_dir / "Apr1_epic_funnel_reasons.csv"

        pd.DataFrame([{"epic_id": "EPIC_101"}]).to_csv(winners_csv, index=False)
        pd.DataFrame([{"epic": "101"}]).to_csv(best_csv, index=False)
        pd.DataFrame([{"epic_id": "101"}]).to_csv(quarantine_csv, index=False)
        pd.DataFrame([{"epic_id": "101", "terminal_reason": "validated_period"}]).to_csv(funnel_csv, index=False)

        with self.assertRaisesRegex(ValueError, "quarantine CSV missing required columns"):
            K2DetectorQualityGatedBroaderCachedFailedDownstreamReport().run(
                winners_csv=winners_csv,
                best_csv=best_csv,
                quarantine_csv=quarantine_csv,
                funnel_csv=funnel_csv,
                summary_csv=self.case_dir / "summary.csv",
                quarantined_winners_csv=self.case_dir / "quarantined.csv",
                best_winners_csv=self.case_dir / "best.csv",
            )


if __name__ == "__main__":
    unittest.main()
