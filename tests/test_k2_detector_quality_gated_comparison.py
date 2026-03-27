from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedComparison import K2DetectorQualityGatedComparison
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class TestK2DetectorQualityGatedComparison(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("tmp_pycache") / f"k2_detector_quality_gated_comparison_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    @staticmethod
    def _write_batch_results(path: Path, mode: str, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        df["detector_operating_mode"] = mode
        df.to_csv(path, index=False)

    def test_quality_gated_comparison_writes_requested_csv_and_summary(self) -> None:
        case_dir = self._make_case_dir()
        baseline_dir = case_dir / "baseline"
        experimental_dir = case_dir / "experimental"
        quality_gated_dir = case_dir / "quality_gated"
        for d in [baseline_dir, experimental_dir, quality_gated_dir]:
            d.mkdir(parents=True, exist_ok=False)

        common_rows = [
            {
                "epic_id": "EPIC_100",
                "query": "EPIC 100",
                "n_events": 10,
                "best_shape_score": 0.70,
                "best_depth_snr": 5.0,
                "triage_usable": False,
                "triage_why_not_usable": "whiteness_pvalue=0<0.01",
            },
            {
                "epic_id": "EPIC_200",
                "query": "EPIC 200",
                "n_events": 12,
                "best_shape_score": 0.60,
                "best_depth_snr": 4.0,
                "triage_usable": False,
                "triage_why_not_usable": "outlier_rate_6sigma>0.02",
            },
        ]
        baseline_rows = common_rows
        experimental_rows = [
            dict(common_rows[0], n_events=14, best_shape_score=0.68, best_depth_snr=4.8),
            dict(common_rows[1], n_events=14, best_shape_score=0.60, best_depth_snr=4.1),
        ]
        quality_gated_rows = [
            dict(common_rows[0], n_events=12, best_shape_score=0.71, best_depth_snr=5.2),
            dict(common_rows[1], n_events=12, best_shape_score=0.60, best_depth_snr=4.0),
        ]

        self._write_batch_results(
            baseline_dir / "batch_results.csv",
            str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
            baseline_rows,
        )
        self._write_batch_results(
            experimental_dir / "batch_results.csv",
            str(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE),
            experimental_rows,
        )
        self._write_batch_results(
            quality_gated_dir / "batch_results.csv",
            str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
            quality_gated_rows,
        )

        out_csv = case_dir / "detector_quality_gated_comparison.csv"
        epic_summary_csv = case_dir / "detector_quality_gated_epic_summary.csv"
        rollup_csv = case_dir / "detector_quality_gated_rollup.csv"
        out = K2DetectorQualityGatedComparison().run(
            baseline_run_dir=baseline_dir,
            experimental_run_dir=experimental_dir,
            quality_gated_run_dir=quality_gated_dir,
            out_csv=out_csv,
            epic_summary_csv=epic_summary_csv,
            rollup_csv=rollup_csv,
        )

        self.assertTrue(out_csv.exists())
        self.assertTrue(epic_summary_csv.exists())
        self.assertTrue(rollup_csv.exists())
        self.assertEqual(str(out["out_csv"]), str(out_csv))
        self.assertTrue(bool(out["keeps_some_event_count_gain"]))
        self.assertTrue(bool(out["improves_best_shape_score_any"]))
        self.assertTrue(bool(out["improves_best_depth_snr_any"]))
        self.assertTrue(bool(out["looks_better_for_scaling"]))
        self.assertEqual(int(out["qg_event_gain_epic_count_vs_default"]), 1)
        self.assertEqual(int(out["qg_shape_improved_epic_count_vs_experimental"]), 1)
        self.assertEqual(int(out["qg_depth_improved_epic_count_vs_experimental"]), 1)
        self.assertEqual(float(out["qg_event_gain_total_vs_default"]), 2.0)
        self.assertEqual(float(out["qg_event_delta_total_vs_experimental"]), -4.0)
        self.assertEqual(int(out["experimental_extra_events_vs_default_count"]), 2)
        self.assertEqual(int(out["quality_gated_extra_events_vs_default_count"]), 1)
        self.assertEqual(int(out["any_best_shape_score_improvement_vs_default_count"]), 1)
        self.assertEqual(int(out["any_best_depth_snr_improvement_vs_default_count"]), 2)
        self.assertEqual(int(out["plain_high_recall_regressed_quality_count"]), 1)
        self.assertEqual(int(out["quality_gated_avoided_plain_regression_count"]), 1)

        df = pd.read_csv(out_csv)
        self.assertEqual(
            df.columns.tolist(),
            [
                "epic_id",
                "query",
                "mode",
                "n_events",
                "best_shape_score",
                "best_depth_snr",
                "triage_usable",
                "triage_why_not_usable",
            ],
        )
        self.assertEqual(len(df), 6)
        self.assertEqual(
            df["mode"].tolist(),
            [
                str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
                str(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE),
                str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
                str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
                str(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE),
                str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
            ],
        )

        epic_df = pd.read_csv(epic_summary_csv)
        self.assertEqual(len(epic_df), 2)
        row_100 = epic_df.loc[epic_df["epic_id"].astype(str) == "EPIC_100"].iloc[0]
        self.assertTrue(bool(row_100["quality_gated_avoided_plain_regression"]))

        rollup_df = pd.read_csv(rollup_csv)
        rollup_map = dict(zip(rollup_df["metric"], rollup_df["value"]))
        self.assertEqual(int(rollup_map["experimental_extra_events_vs_default_count"]), 2)
        self.assertEqual(int(rollup_map["quality_gated_extra_events_vs_default_count"]), 1)
        self.assertEqual(int(rollup_map["plain_high_recall_regressed_quality_count"]), 1)
        self.assertEqual(int(rollup_map["quality_gated_avoided_plain_regression_count"]), 1)


if __name__ == "__main__":
    unittest.main()
