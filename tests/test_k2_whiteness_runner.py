import unittest
import uuid
from pathlib import Path
import shutil
from unittest import mock

import pandas as pd

from src.Classifiers.K2.Pipeline.K2WhitenessRunner import K2WhitenessRunner


class TestK2WhitenessRunner(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("plots") / "k2_batch" / f"test_whiteness_runner_{uuid.uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    def test_run_cli_writes_sliced_whiteness_csv_and_counts(self) -> None:
        tmp = self._make_case_dir()
        input_csv = tmp / "batch_results.csv"
        output_csv = tmp / "batch_results_whiteness.csv"
        pd.DataFrame(
            [
                {"query": "EPIC 1", "triage_status": "ok"},
                {"query": "EPIC 2", "triage_status": "ok"},
                {"query": "EPIC 3", "triage_status": "error"},
                {"query": "EPIC 4", "triage_status": "ok"},
            ]
        ).to_csv(input_csv, index=False)

        captured: dict[str, object] = {}

        class FakeBatchRunner:
            def __init__(self, out_dir, whiteness_alpha, whiteness_score_definition, **kwargs):  # noqa: ANN001
                captured["out_dir"] = Path(out_dir)
                captured["whiteness_alpha"] = float(whiteness_alpha)
                captured["whiteness_score_definition"] = str(whiteness_score_definition)

            def retriage_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
                work = df.copy()
                work["triage_whiteness_score"] = [0.005, 0.50]
                work["triage_whiteness_definition"] = ["pvalue", "pvalue"]
                work["triage_usable"] = [False, False]
                work["triage_why_not_usable"] = [
                    "whiteness_pvalue<0.010 (0.005)",
                    "triage_status=error",
                ]
                return work

        with mock.patch("src.Classifiers.K2.Pipeline.K2WhitenessRunner.K2BatchRunner", FakeBatchRunner):
            out = K2WhitenessRunner.run_cli(
                [
                    "--input",
                    str(input_csv),
                    "--out",
                    str(output_csv),
                    "--whiteness-alpha",
                    "0.02",
                    "--start-index",
                    "1",
                    "--end-index",
                    "4",
                    "--max-rows",
                    "2",
                ]
            )

        self.assertEqual(out["input_csv"], input_csv)
        self.assertEqual(out["out_csv"], output_csv)
        self.assertEqual(out["total_rows"], 2)
        self.assertEqual(out["usable_rows"], 0)
        self.assertEqual(out["whiteness_value_column"], "triage_whiteness_pvalue")
        self.assertTrue(bool(out["whiteness_is_pvalue"]))
        self.assertEqual(out["whiteness_interpretation_column"], "triage_whiteness_interpretation")
        self.assertAlmostEqual(float(out["whiteness_min"]), 0.005, places=9)
        self.assertAlmostEqual(float(out["whiteness_median"]), 0.2525, places=9)
        self.assertAlmostEqual(float(out["whiteness_max"]), 0.50, places=9)
        self.assertEqual(captured["out_dir"], output_csv.parent)
        self.assertAlmostEqual(float(captured["whiteness_alpha"]), 0.02, places=9)
        self.assertEqual(captured["whiteness_score_definition"], "pvalue")

        written = pd.read_csv(output_csv)
        self.assertEqual(written["query"].tolist(), ["EPIC 2", "EPIC 3"])
        self.assertEqual(written["epic_id"].tolist(), [2, 3])
        self.assertEqual(written["triage_whiteness_definition"].tolist(), ["pvalue", "pvalue"])
        self.assertIn("triage_whiteness_pvalue", written.columns)
        self.assertIn("triage_whiteness_interpretation", written.columns)
        self.assertNotIn("triage_whiteness_score", written.columns)

    def test_run_cli_places_output_in_run_subdir(self) -> None:
        tmp = self._make_case_dir()
        input_csv = tmp / "batch_results.csv"
        output_csv = tmp / "batch_results_whiteness.csv"
        pd.DataFrame([{"query": "EPIC 1", "triage_status": "ok"}]).to_csv(input_csv, index=False)

        class FakeBatchRunner:
            def __init__(self, **kwargs):  # noqa: ANN003
                pass

            def retriage_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
                work = df.copy()
                work["triage_whiteness_score"] = [0.50]
                work["triage_whiteness_definition"] = ["pvalue"]
                work["triage_usable"] = [True]
                work["triage_why_not_usable"] = [""]
                return work

        with mock.patch("src.Classifiers.K2.Pipeline.K2WhitenessRunner.K2BatchRunner", FakeBatchRunner):
            out = K2WhitenessRunner.run_cli(
                [
                    "--input",
                    str(input_csv),
                    "--output",
                    str(output_csv),
                    "--use-run-subdir",
                    "--run-id",
                    "slice_200",
                ]
            )

        self.assertEqual(out["out_csv"].parent.name, "slice_200")
        self.assertTrue(out["out_csv"].exists())

    def test_run_keeps_score_name_when_definitions_are_mixed(self) -> None:
        tmp = self._make_case_dir()
        input_csv = tmp / "batch_results.csv"
        output_csv = tmp / "batch_results_whiteness.csv"
        pd.DataFrame(
            [
                {"query": "EPIC 1", "triage_status": "ok"},
                {"query": "EPIC 2", "triage_status": "ok"},
            ]
        ).to_csv(input_csv, index=False)

        class FakeBatchRunner:
            def __init__(self, **kwargs):  # noqa: ANN003
                pass

            def retriage_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
                work = df.copy()
                work["triage_whiteness_score"] = [0.12, 0.33]
                work["triage_whiteness_definition"] = ["lag1_abs_autocorr_statistic", "lag1_autocorr_pvalue_normal_approx"]
                work["triage_usable"] = [True, True]
                work["triage_why_not_usable"] = ["", ""]
                return work

        with mock.patch("src.Classifiers.K2.Pipeline.K2WhitenessRunner.K2BatchRunner", FakeBatchRunner):
            out = K2WhitenessRunner.run_cli(["--input", str(input_csv), "--out", str(output_csv)])

        self.assertEqual(out["whiteness_value_column"], "triage_whiteness_score")
        self.assertFalse(bool(out["whiteness_is_pvalue"]))
        self.assertEqual(out["whiteness_interpretation_column"], "triage_whiteness_higher_is_better")
        written = pd.read_csv(output_csv)
        self.assertIn("triage_whiteness_score", written.columns)
        self.assertIn("triage_whiteness_higher_is_better", written.columns)
        self.assertNotIn("triage_whiteness_pvalue", written.columns)

    def test_run_exports_null_ok_rows_for_anomaly_audit(self) -> None:
        tmp = self._make_case_dir()
        input_csv = tmp / "batch_results.csv"
        output_csv = tmp / "batch_results_whiteness.csv"
        pd.DataFrame([{"query": "EPIC 1", "triage_status": "ok"}]).to_csv(input_csv, index=False)

        class FakeBatchRunner:
            def __init__(self, **kwargs):  # noqa: ANN003
                pass

            def retriage_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
                work = df.copy()
                work["epic_id"] = ["1"]
                work["triage_whiteness_score"] = [float("nan")]
                work["triage_whiteness_definition"] = ["lag1_autocorr_pvalue_normal_approx"]
                work["triage_usable"] = [False]
                work["triage_why_not_usable"] = ["n_points<800"]
                work["n_events"] = [float("nan")]
                return work

        with mock.patch("src.Classifiers.K2.Pipeline.K2WhitenessRunner.K2BatchRunner", FakeBatchRunner):
            out = K2WhitenessRunner.run_cli(["--input", str(input_csv), "--out", str(output_csv)])

        self.assertEqual(int(out["null_ok_rows_count"]), 1)
        self.assertTrue(Path(out["null_ok_rows_csv"]).exists())
        audit_df = pd.read_csv(out["null_ok_rows_csv"])
        self.assertEqual(audit_df.iloc[0]["shortlist_rejection_reason"], "whiteness_null_and_triage_unusable")


if __name__ == "__main__":
    unittest.main()
