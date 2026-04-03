from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis import (
    K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis,
)


class K2DetectorQualityGatedBroaderWinnerDownstreamAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_detector_quality_gated_broader_winner_downstream_analysis_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def _write_run(
        self,
        run_dir: Path,
        *,
        best_rows: list[dict],
        quarantine_rows: list[dict],
        funnel_rows: list[dict],
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(best_rows).to_csv(run_dir / "period_shortlist_best.csv", index=False)
        pd.DataFrame(quarantine_rows).to_csv(run_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame([{"n_total_epics": 3}]).to_csv(run_dir / "period_shortlist_diagnostics.csv", index=False)
        pd.DataFrame(funnel_rows).to_csv(run_dir / "epic_funnel_reasons.csv", index=False)

    def test_analysis_classifies_broader_winners_and_writes_outputs(self) -> None:
        winners_csv = self.case_dir / "detector_quality_gated_broader_winners.csv"
        comparison_csv = self.case_dir / "detector_quality_gated_broader_comparison.csv"
        rollup_csv = self.case_dir / "detector_quality_gated_broader_rollup.csv"
        default_run_dir = self.case_dir / "default"
        quality_gated_run_dir = self.case_dir / "quality_gated"
        analysis_csv = self.case_dir / "detector_quality_gated_broader_winner_downstream_analysis.csv"
        analysis_rollup_csv = self.case_dir / "detector_quality_gated_broader_winner_downstream_rollup.csv"
        real_rescues_csv = self.case_dir / "detector_quality_gated_broader_real_rescues.csv"

        pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_101",
                    "gained_extra_events": True,
                    "improved_best_shape_score": True,
                    "default_n_events": 1,
                    "quality_gated_n_events": 3,
                    "delta_n_events": 2,
                    "default_best_shape_score": 0.50,
                    "quality_gated_best_shape_score": 0.60,
                    "delta_best_shape_score": 0.10,
                    "default_best_depth_snr": 4.0,
                    "quality_gated_best_depth_snr": 4.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id": "EPIC_102",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "default_n_events": 1,
                    "quality_gated_n_events": 2,
                    "delta_n_events": 1,
                    "default_best_shape_score": 0.52,
                    "quality_gated_best_shape_score": 0.52,
                    "delta_best_shape_score": 0.0,
                    "default_best_depth_snr": 4.2,
                    "quality_gated_best_depth_snr": 4.2,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id": "EPIC_103",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "default_n_events": 1,
                    "quality_gated_n_events": 2,
                    "delta_n_events": 1,
                    "default_best_shape_score": 0.48,
                    "quality_gated_best_shape_score": 0.48,
                    "delta_best_shape_score": 0.0,
                    "default_best_depth_snr": 3.9,
                    "quality_gated_best_depth_snr": 3.9,
                    "delta_best_depth_snr": 0.0,
                },
            ]
        ).to_csv(winners_csv, index=False)

        comparison_rows = []
        for epic_id in ["EPIC_101", "EPIC_102", "EPIC_103"]:
            comparison_rows.append(
                {
                    "epic_id": epic_id,
                    "query": epic_id.replace("_", " "),
                    "mode": "detector_default",
                    "n_events": 1,
                }
            )
            comparison_rows.append(
                {
                    "epic_id": epic_id,
                    "query": epic_id.replace("_", " "),
                    "mode": "detector_high_recall_quality_gated_experimental",
                    "n_events": 2,
                }
            )
        pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)
        pd.DataFrame(
            [
                {"metric": "count_with_extra_events_vs_default", "value": 3},
                {"metric": "recommendation", "value": "continue"},
            ]
        ).to_csv(rollup_csv, index=False)

        self._write_run(
            default_run_dir,
            best_rows=[],
            quarantine_rows=[
                {
                    "epic_id": "101",
                    "query": "EPIC 101",
                    "reason": "P_null_or_missing",
                    "source_reason": "events_filtered_to_zero",
                    "failure_category": "events_filtered_to_zero",
                    "failure_detail": "all events removed",
                    "P": "",
                },
                {
                    "epic_id": "103",
                    "query": "EPIC 103",
                    "reason": "P_null_or_missing",
                    "source_reason": "events_filtered_to_zero",
                    "failure_category": "events_filtered_to_zero",
                    "failure_detail": "all events removed",
                    "P": "",
                },
            ],
            funnel_rows=[
                {"epic_id": "101", "selected_for_period_stage": True},
                {"epic_id": "102", "selected_for_period_stage": True},
                {"epic_id": "103", "selected_for_period_stage": True},
            ],
        )
        self._write_run(
            quality_gated_run_dir,
            best_rows=[
                {
                    "epic": "101",
                    "query": "EPIC 101",
                    "reason": "validated",
                    "P": 8.5,
                    "manual_review_required": False,
                }
            ],
            quarantine_rows=[
                {
                    "epic_id": "102",
                    "query": "EPIC 102",
                    "reason": "P_found_but_rejected",
                    "source_reason": "no_cluster_periods",
                    "failure_category": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "P": 12.0,
                },
                {
                    "epic_id": "103",
                    "query": "EPIC 103",
                    "reason": "P_null_or_missing",
                    "source_reason": "events_filtered_to_zero",
                    "failure_category": "events_filtered_to_zero",
                    "failure_detail": "all events removed",
                    "P": "",
                },
            ],
            funnel_rows=[
                {"epic_id": "101", "selected_for_period_stage": True},
                {"epic_id": "102", "selected_for_period_stage": True},
                {"epic_id": "103", "selected_for_period_stage": True},
            ],
        )

        out = K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis().run(
            winners_csv=winners_csv,
            comparison_csv=comparison_csv,
            rollup_csv=rollup_csv,
            default_run_dir=default_run_dir,
            default_best_csv=None,
            default_quarantine_csv=None,
            default_diagnostics_csv=None,
            default_funnel_csv=None,
            quality_gated_run_dir=quality_gated_run_dir,
            quality_gated_best_csv=None,
            quality_gated_quarantine_csv=None,
            quality_gated_diagnostics_csv=None,
            quality_gated_funnel_csv=None,
            analysis_csv=analysis_csv,
            analysis_rollup_csv=analysis_rollup_csv,
            real_rescues_csv=real_rescues_csv,
        )

        self.assertEqual(int(out["winners_total"]), 3)
        self.assertEqual(int(out["real_rescues"]), 1)
        self.assertEqual(int(out["detector_only_gains"]), 1)
        self.assertEqual(int(out["still_blocked"]), 1)
        self.assertEqual(int(out["rescue_counts_by_period_bin"]["(5,10]"]), 1)
        self.assertEqual(
            out["top_failure_reasons"]["period clustering found no usable period candidate"],
            1,
        )

        self.assertTrue(analysis_csv.exists())
        self.assertTrue(analysis_rollup_csv.exists())
        self.assertTrue(real_rescues_csv.exists())

        analysis_df = pd.read_csv(analysis_csv)
        bucket_map = dict(zip(analysis_df["epic_id"], analysis_df["winner_bucket"]))
        self.assertEqual(bucket_map["EPIC_101"], "real_rescue")
        self.assertEqual(bucket_map["EPIC_102"], "still_blocked")
        self.assertEqual(bucket_map["EPIC_103"], "detector_only_gain")

        row_102 = analysis_df.loc[analysis_df["epic_id"] == "EPIC_102"].iloc[0]
        self.assertEqual(str(row_102["quality_gated_failure_reason_bucket"]), "cluster_related_failures")
        self.assertIn("progressed further downstream", str(row_102["non_rescue_explanation"]))
        self.assertEqual(str(row_102["default_detector_mode"]), "detector_default")
        self.assertEqual(
            str(row_102["quality_gated_detector_mode"]),
            "detector_high_recall_quality_gated_experimental",
        )

        rescues_df = pd.read_csv(real_rescues_csv)
        self.assertEqual(list(rescues_df["epic_id"]), ["EPIC_101"])

        rollup_df = pd.read_csv(analysis_rollup_csv)
        summary_map = dict(
            zip(
                rollup_df.loc[rollup_df["section"] == "summary", "metric"],
                rollup_df.loc[rollup_df["section"] == "summary", "value"],
            )
        )
        self.assertEqual(int(summary_map["winners_total"]), 3)
        self.assertEqual(int(summary_map["real_rescues"]), 1)


if __name__ == "__main__":
    unittest.main()
