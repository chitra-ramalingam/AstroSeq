from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Pipeline.K2CampaignRunner import K2CampaignRunner
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class TestK2BatchRunnerDetectorOnlyAnalysis(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("tmp_pycache") / f"k2_batch_runner_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    @staticmethod
    def _detector_output(usable: bool = False, why_not: str = "n_points<800;whiteness_pvalue=1e-10<0.01") -> dict:
        return {
            "summary": {
                "status": "ok",
                "usable": usable,
                "why_not_usable": why_not,
                "score_global": -1.0,
                "n_points": 3200,
                "step_score": 0.2,
                "whiteness_score": 1e-10,
                "whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                "n_points_after_preprocess": 3000,
                "author_selected": "EVEREST",
                "campaign_selected": "c05",
            },
            "candidates": [
                {
                    "shape_score": 0.82,
                    "depth_snr": 5.1,
                    "t_mid": 12.5,
                }
            ],
        }

    def test_default_path_hard_fail_still_skips_period_validation(self) -> None:
        case_dir = self._make_case_dir()
        runner = K2BatchRunner(
            out_dir=case_dir,
            queries=["EPIC 211301344"],
            cache_only=True,
            skip_existing_epics=False,
        )
        with mock.patch.object(runner.detector, "run_one", return_value=self._detector_output()), \
            mock.patch.object(runner, "_fetch_clean_time_flux") as fetch_mock, \
            mock.patch.object(runner.validator, "validate") as validate_mock:
            out = runner.run()

        fetch_mock.assert_not_called()
        validate_mock.assert_not_called()
        self.assertNotIn("detector_candidate_results_csv", out)
        df = pd.read_csv(out["batch_results_csv"])
        self.assertEqual(int(df.iloc[0]["n_periods_validated"]), 0)
        self.assertEqual(str(df.iloc[0]["label"]), "Noisy_trash")
        self.assertIn("usable=False:", str(df.iloc[0]["label_reason"]))

    def test_detector_only_mode_records_candidates_without_hard_fail_block(self) -> None:
        case_dir = self._make_case_dir()
        runner = K2BatchRunner(
            out_dir=case_dir,
            queries=["EPIC 211301344"],
            detector_only_analysis=True,
            cache_only=True,
            skip_existing_epics=False,
        )
        with mock.patch.object(runner.detector, "run_one", return_value=self._detector_output()), \
            mock.patch.object(runner, "_fetch_clean_time_flux") as fetch_mock, \
            mock.patch.object(runner.validator, "validate") as validate_mock:
            out = runner.run()

        fetch_mock.assert_not_called()
        validate_mock.assert_not_called()
        detector_csv = Path(out["detector_candidate_results_csv"])
        self.assertTrue(detector_csv.exists())
        df = pd.read_csv(detector_csv)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(str(row["query"]), "EPIC 211301344")
        self.assertEqual(int(row["n_events"]), 1)
        self.assertEqual(bool(row["cache_only"]), True)
        self.assertEqual(str(row["triage_status"]), "ok")
        self.assertEqual(bool(row["triage_usable"]), False)
        self.assertIn("whiteness_pvalue", str(row["triage_why_not_usable"]))
        self.assertAlmostEqual(float(row["best_shape_score"]), 0.82, places=6)
        self.assertAlmostEqual(float(row["best_depth_snr"]), 5.1, places=6)
        self.assertEqual(int(row["n_points_after_preprocess"]), 3000)

    def test_detector_only_csv_contains_expected_columns(self) -> None:
        case_dir = self._make_case_dir()
        runner = K2BatchRunner(
            out_dir=case_dir,
            queries=["EPIC 211301344"],
            detector_only_analysis=True,
            cache_only=True,
            skip_existing_epics=False,
        )
        with mock.patch.object(runner.detector, "run_one", return_value=self._detector_output()):
            out = runner.run()

        df = pd.read_csv(out["detector_candidate_results_csv"])
        expected = [
            "query",
            "epic_id",
            "detector_operating_mode",
            "detector_detect_sigma",
            "triage_status",
            "triage_usable",
            "triage_why_not_usable",
            "n_events",
            "best_shape_score",
            "best_depth_snr",
            "n_points_after_preprocess",
            "cache_only",
        ]
        self.assertEqual(df.columns.tolist(), expected)

    def test_campaign_runner_threads_detector_only_flag(self) -> None:
        parser = K2CampaignRunner().build_parser()
        args = parser.parse_args(["--detector-only-analysis", "--query", "EPIC 211301344"])
        runner = K2CampaignRunner()._build_batch_runner(args=args, queries=["EPIC 211301344"], input_csv=None)
        self.assertTrue(runner.detector_only_analysis)

    def test_quality_gated_mode_keeps_baseline_events_and_filters_weak_extras(self) -> None:
        case_dir = self._make_case_dir()
        runner = K2BatchRunner(
            out_dir=case_dir,
            queries=["EPIC 211301344"],
            detector_only_analysis=True,
            detector_operating_mode=K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE,
            cache_only=True,
            skip_existing_epics=False,
        )
        baseline_output = {
            "summary": dict(self._detector_output()["summary"]),
            "candidates": [
                {"shape_score": 0.82, "depth_snr": 5.1, "t_mid": 12.5, "min_idx": 10, "start_idx": 9, "end_idx": 11},
            ],
        }
        experimental_output = {
            "summary": dict(self._detector_output()["summary"]),
            "candidates": [
                {"shape_score": 0.82, "depth_snr": 5.1, "t_mid": 12.5, "min_idx": 10, "start_idx": 9, "end_idx": 11},
                {"shape_score": 0.70, "depth_snr": 3.4, "t_mid": 20.0, "min_idx": 20, "start_idx": 19, "end_idx": 21},
                {"shape_score": 0.60, "depth_snr": 3.5, "t_mid": 30.0, "min_idx": 30, "start_idx": 29, "end_idx": 31},
                {"shape_score": 0.75, "depth_snr": 2.9, "t_mid": 40.0, "min_idx": 40, "start_idx": 39, "end_idx": 41},
            ],
        }
        default_detector = runner._get_default_detector()
        with mock.patch.object(runner.detector, "run_one", return_value=experimental_output), \
            mock.patch.object(default_detector, "run_one", return_value=baseline_output):
            out = runner.run()

        df = pd.read_csv(out["detector_candidate_results_csv"])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(int(row["n_events"]), 2)
        self.assertAlmostEqual(float(row["best_shape_score"]), 0.82, places=6)
        self.assertAlmostEqual(float(row["best_depth_snr"]), 5.1, places=6)

        events_df = pd.read_csv(case_dir / "epics" / "EPIC_211301344" / "events.csv")
        self.assertEqual(len(events_df), 2)
        self.assertEqual(sorted(events_df["t_mid"].tolist()), [12.5, 20.0])

    def test_quality_gated_mode_uses_relaxed_detect_sigma(self) -> None:
        runner = K2BatchRunner(
            out_dir=self._make_case_dir(),
            queries=["EPIC 211301344"],
            detector_operating_mode=K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE,
            cache_only=True,
            skip_existing_epics=False,
        )
        self.assertEqual(
            float(runner.detector_detect_sigma),
            float(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_DETECT_SIGMA),
        )

    def test_quality_gated_mode_handles_empty_candidate_frames(self) -> None:
        case_dir = self._make_case_dir()
        runner = K2BatchRunner(
            out_dir=case_dir,
            queries=["EPIC 211301344"],
            detector_only_analysis=True,
            detector_operating_mode=K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE,
            cache_only=True,
            skip_existing_epics=False,
        )
        empty_output = {
            "summary": dict(self._detector_output()["summary"]),
            "candidates": [],
        }
        default_detector = runner._get_default_detector()
        with mock.patch.object(runner.detector, "run_one", return_value=empty_output), \
            mock.patch.object(default_detector, "run_one", return_value=empty_output):
            out = runner.run()

        df = pd.read_csv(out["detector_candidate_results_csv"])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(int(row["n_events"]), 0)
        self.assertTrue(pd.isna(row["best_shape_score"]))
        self.assertTrue(pd.isna(row["best_depth_snr"]))

    def test_retriage_prefers_explicit_pvalue_field_when_present(self) -> None:
        runner = K2BatchRunner(
            out_dir=self._make_case_dir(),
            queries=["EPIC 211301344"],
            cache_only=True,
            skip_existing_epics=False,
        )
        df = pd.DataFrame(
            [
                {
                    "query": "EPIC 211301344",
                    "triage_status": "ok",
                    "triage_step_score": 0.2,
                    "triage_whiteness_score": 0.0,
                    "triage_whiteness_pvalue": 0.95,
                    "triage_whiteness_mode": "pvalue",
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                    "n_events": 1,
                    "best_shape_score": 0.82,
                    "best_depth_snr": 5.1,
                }
            ]
        )

        out = runner.retriage_results_df(df)

        self.assertTrue(bool(out.iloc[0]["triage_usable"]))
        self.assertEqual(str(out.iloc[0]["triage_why_not_usable"]), "")


if __name__ == "__main__":
    unittest.main()
