import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner


class TestK2ShortlistPeriodRunner(unittest.TestCase):
    def test_validate_period_rows_quarantines_invalid_p(self) -> None:
        cfg = K2ShortlistPeriodConfig(
            MIN_PERIOD_DAYS=0.5,
            MAX_PERIOD_DAYS=20.0,
            PERIOD_HARD_MAX_DAYS=20.0,
        )
        runner = K2ShortlistPeriodRunner(config=cfg)
        df = pd.DataFrame(
            [
                {"epic": "1", "query": "EPIC 1", "reason": "validated", "P": 3.0},
                {"epic": "2", "query": "EPIC 2", "reason": "no_cluster_periods", "P": np.nan},
                {"epic": "3", "query": "EPIC 3", "reason": "validated", "P": 0.0},
                {"epic": "4", "query": "EPIC 4", "reason": "validated", "P": 25.0},
            ]
        ).reindex(columns=runner.SUMMARY_COLUMNS)

        valid_df, quarantine_df, diagnostics = runner._validate_period_rows(df)

        self.assertEqual(len(valid_df), 1)
        self.assertEqual(valid_df.iloc[0]["epic"], "1")
        self.assertEqual(len(quarantine_df), 3)
        self.assertEqual(diagnostics["rows_total"], 4)
        self.assertEqual(diagnostics["rows_null_p"], 1)
        self.assertEqual(diagnostics["rows_dropped"], 3)
        self.assertEqual(diagnostics["rows_valid"], 1)
        source_map = dict(zip(quarantine_df["epic_id"], quarantine_df["missing_upstream_source"]))
        self.assertEqual(source_map.get("2"), "infer_periods_from_events(events_df)")

    def test_null_p_rate_fail_fast_raises_with_epics(self) -> None:
        cfg = K2ShortlistPeriodConfig(NULL_P_RATE_MAX=0.001)
        runner = K2ShortlistPeriodRunner(config=cfg)
        diagnostics = {"rows_total": 1000, "rows_null_p": 2}
        quarantine_df = pd.DataFrame(
            [
                {"epic_id": "111", "reason": "P_null_or_missing", "source_reason": "missing_events_csv"},
                {"epic_id": "222", "reason": "P_null_or_missing", "source_reason": "missing_events_csv"},
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "top_20_epics=\\['111', '222'\\]"):
            runner._enforce_null_p_rate_threshold(diagnostics=diagnostics, quarantine_df=quarantine_df)

    def test_null_p_rate_exempt_source_reason_does_not_raise(self) -> None:
        cfg = K2ShortlistPeriodConfig(NULL_P_RATE_MAX=0.001, NULL_P_RATE_EXEMPT_SOURCE_REASONS=("no_cluster_periods",))
        runner = K2ShortlistPeriodRunner(config=cfg)
        diagnostics = {"rows_total": 1000, "rows_null_p": 2}
        quarantine_df = pd.DataFrame(
            [
                {"epic_id": "111", "reason": "P_null_or_missing", "source_reason": "no_cluster_periods"},
                {"epic_id": "222", "reason": "P_null_or_missing", "source_reason": "no_cluster_periods"},
            ]
        )
        runner._enforce_null_p_rate_threshold(diagnostics=diagnostics, quarantine_df=quarantine_df)

    def test_null_p_rate_exempt_inferred_from_missing_upstream_source_does_not_raise(self) -> None:
        cfg = K2ShortlistPeriodConfig(NULL_P_RATE_MAX=0.001, NULL_P_RATE_EXEMPT_SOURCE_REASONS=("no_cluster_periods",))
        runner = K2ShortlistPeriodRunner(config=cfg)
        diagnostics = {"rows_total": 1000, "rows_null_p": 2}
        quarantine_df = pd.DataFrame(
            [
                {
                    "epic_id": "111",
                    "reason": "P_null_or_missing",
                    "source_reason": "",
                    "missing_upstream_source": "infer_periods_from_events(events_df)",
                },
                {
                    "epic_id": "222",
                    "reason": "P_null_or_missing",
                    "source_reason": "",
                    "missing_upstream_source": "infer_periods_from_events(events_df)",
                },
            ]
        )
        runner._enforce_null_p_rate_threshold(diagnostics=diagnostics, quarantine_df=quarantine_df)

    def test_stratified_best_selection_returns_one_row_per_epic(self) -> None:
        runner = K2ShortlistPeriodRunner()
        work = pd.DataFrame(
            [
                {"epic": "A", "P": 4.0, "score_raw": 70.0, "_row_order": 0},
                {"epic": "A", "P": 18.0, "score_raw": 80.0, "_row_order": 1},
                {"epic": "B", "P": 3.5, "score_raw": 25.0, "_row_order": 2},
                {"epic": "C", "P": 4.5, "score_raw": 30.0, "_row_order": 3},
                {"epic": "D", "P": 17.0, "score_raw": 75.0, "_row_order": 4},
                {"epic": "E", "P": 19.0, "score_raw": 90.0, "_row_order": 5},
            ]
        )

        best_df, quotas, achieved, summary_counts = runner._select_best_rows_stratified(work)
        self.assertEqual(len(best_df), 5)
        self.assertEqual(best_df["epic"].nunique(), 5)
        self.assertTrue(len(quotas) > 0)
        self.assertTrue(len(achieved) > 0)
        self.assertTrue(len(summary_counts) > 0)

    def test_stratified_best_selection_supports_equal_mode(self) -> None:
        cfg = K2ShortlistPeriodConfig(BEST_SELECTION_BIN_MODE="equal_per_bin")
        runner = K2ShortlistPeriodRunner(config=cfg)
        work = pd.DataFrame(
            [
                {"epic": "A", "P": 4.0, "score_raw": 90.0, "_row_order": 0},
                {"epic": "B", "P": 7.0, "score_raw": 80.0, "_row_order": 1},
                {"epic": "C", "P": 12.0, "score_raw": 70.0, "_row_order": 2},
                {"epic": "D", "P": 18.0, "score_raw": 60.0, "_row_order": 3},
            ]
        )
        best_df, quotas, achieved, _ = runner._select_best_rows_stratified(work)
        self.assertEqual(len(best_df), 4)
        self.assertEqual(sum(quotas.values()), 4)
        self.assertEqual(sum(achieved.values()), 4)

    def test_dedupe_epic_period_prefers_validated_row(self) -> None:
        runner = K2ShortlistPeriodRunner()
        df = pd.DataFrame(
            [
                {"epic": "1", "query": "EPIC 1", "reason": "cluster_only", "P": 8.0, "soft_hit_rate": 0.0},
                {"epic": "1", "query": "EPIC 1", "reason": "validated", "P": 8.0, "soft_hit_rate": 0.3},
                {"epic": "1", "query": "EPIC 1", "reason": "validated", "P": 12.0, "soft_hit_rate": 0.1},
            ]
        ).reindex(columns=runner.SUMMARY_COLUMNS)

        out = runner._dedupe_epic_period_rows(df)
        self.assertEqual(len(out), 2)
        sub = out.loc[pd.to_numeric(out["P"], errors="coerce").round(6) == 8.0]
        self.assertEqual(len(sub), 1)
        self.assertEqual(str(sub.iloc[0]["reason"]), "validated")

    def test_save_period_histograms_writes_png_and_counts(self) -> None:
        runner = K2ShortlistPeriodRunner()
        out_dir = Path("tmp_pycache") / f"k2_shortlist_period_hist_test_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame({"P": [2.0, 4.0, 7.0, 12.0, 17.5]})
        best_df = pd.DataFrame({"P": [4.0, 12.0, 18.0]})
        out_png = out_dir / "hist.png"
        out_counts_csv = out_dir / "hist_counts.csv"

        meta = runner._save_period_histograms(
            summary_df=summary_df,
            best_df=best_df,
            out_png=out_png,
            out_counts_csv=out_counts_csv,
        )

        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 0)
        self.assertTrue(out_counts_csv.exists())
        counts_df = pd.read_csv(out_counts_csv)
        self.assertIn("summary_count", counts_df.columns)
        self.assertIn("best_count", counts_df.columns)
        self.assertEqual(int(counts_df["summary_count"].sum()), int(meta["summary_hist_total"]))
        self.assertEqual(int(counts_df["best_count"].sum()), int(meta["best_hist_total"]))
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
