import shutil
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodCompare import K2ShortlistPeriodCompare
from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner


class TestK2ShortlistPeriodRunner(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("tmp_pycache") / f"k2_shortlist_period_runner_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    def test_raw_epic_list_csv_path_uses_configured_whiteness_path(self) -> None:
        cfg = K2ShortlistPeriodConfig()
        self.assertEqual(
            cfg.raw_epic_list_csv_path,
            Path(r"plots\k2_batch\batch_results_whiteness.csv"),
        )

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

    def test_load_raw_epic_table_requires_whiteness_precompute_columns(self) -> None:
        case_dir = self._make_case_dir()
        raw_csv = case_dir / "batch_results_whiteness.csv"
        pd.DataFrame(
            [
                {
                    "query": "EPIC 1",
                    "epic_id": "1",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                }
            ]
        ).to_csv(raw_csv, index=False)
        runner = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(RAW_EPIC_LIST_CSV=str(raw_csv)))
        with self.assertRaisesRegex(ValueError, "must contain one whiteness value column"):
            runner._load_raw_epic_table(shortlist_df=pd.DataFrame({"query": []}))

    def test_load_raw_epic_table_missing_file_raises_with_hint(self) -> None:
        case_dir = self._make_case_dir()
        raw_csv = case_dir / "missing_whiteness.csv"
        runner = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(RAW_EPIC_LIST_CSV=str(raw_csv)))
        with self.assertRaisesRegex(FileNotFoundError, "k2_whiteness"):
            runner._load_raw_epic_table(shortlist_df=pd.DataFrame({"query": []}))

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

    def test_period_stage_selector_topk_all_random(self) -> None:
        raw = pd.DataFrame(
            [
                {"epic_id": "1", "query": "EPIC 1", "triage_status": "ok", "n_events": 5, "best_shape_score": 0.9, "best_depth_snr": 10.0},
                {"epic_id": "2", "query": "EPIC 2", "triage_status": "ok", "n_events": 4, "best_shape_score": 0.8, "best_depth_snr": 9.0},
                {"epic_id": "3", "query": "EPIC 3", "triage_status": "ok", "n_events": 3, "best_shape_score": 0.7, "best_depth_snr": 8.0},
                {"epic_id": "4", "query": "EPIC 4", "triage_status": "error", "n_events": 0, "best_shape_score": 0.6, "best_depth_snr": 7.0},
            ]
        )
        shortlist = pd.DataFrame({"query": ["EPIC 1", "EPIC 2"]})

        topk_runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(PERIOD_STAGE_SELECTION_MODE="topk", PERIOD_STAGE_K=2)
        )
        topk_selected, topk_meta = topk_runner._select_period_stage_queries(raw_epics_df=raw, shortlist_df=shortlist)
        self.assertEqual(len(topk_selected), 2)
        self.assertEqual(int(topk_meta["n_excluded_by_topk_gate"]), 1)

        all_runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(PERIOD_STAGE_SELECTION_MODE="all")
        )
        all_selected, all_meta = all_runner._select_period_stage_queries(raw_epics_df=raw, shortlist_df=shortlist)
        self.assertEqual(len(all_selected), 4)
        self.assertEqual(int(all_meta["n_excluded_by_topk_gate"]), 0)

        random_runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(
                PERIOD_STAGE_SELECTION_MODE="randomN",
                PERIOD_STAGE_N=2,
                PERIOD_STAGE_RANDOM_SEED=1,
            )
        )
        random_selected, random_meta = random_runner._select_period_stage_queries(raw_epics_df=raw, shortlist_df=shortlist)
        self.assertEqual(len(random_selected), 2)
        self.assertEqual(int(random_meta["n_excluded_by_topk_gate"]), 0)

    def test_null_whiteness_and_unusable_is_rejected_before_candidate_scoring(self) -> None:
        runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(PERIOD_STAGE_SELECTION_MODE="all")
        )
        raw = pd.DataFrame(
            [
                {
                    "epic_id": "1",
                    "query": "EPIC 1",
                    "triage_status": "ok",
                    "triage_usable": False,
                    "triage_whiteness_pvalue": np.nan,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "n_points<800",
                    "n_events": np.nan,
                    "best_shape_score": np.nan,
                    "best_depth_snr": np.nan,
                    "shortlist_rejection_reason": "whiteness_null_and_triage_unusable",
                    "shortlist_rejection_stage": "pre_candidate_scoring",
                    "whiteness_missing": True,
                    "whiteness_null_reason_category": "noncomputable_quality_metrics",
                    "rejected_before_candidate_scoring": True,
                    "rejected_after_candidate_scoring": False,
                },
                {
                    "epic_id": "2",
                    "query": "EPIC 2",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_pvalue": 0.9,
                    "triage_whiteness_definition": "lag1_autocorr_pvalue_normal_approx",
                    "triage_why_not_usable": "",
                    "n_events": 4,
                    "best_shape_score": 0.8,
                    "best_depth_snr": 9.0,
                },
            ]
        )
        selected, meta = runner._select_period_stage_queries(raw_epics_df=raw, shortlist_df=pd.DataFrame({"query": []}))
        self.assertEqual(selected, ["EPIC 2"])
        self.assertEqual(int(meta["n_excluded_by_shortlist_precheck"]), 1)

        funnel, reasons = runner._build_epic_funnel_and_reasons(
            raw_epics_df=raw,
            selected_queries=selected,
            selection_meta=meta,
            df_summary_raw=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            df_summary_valid=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            df_summary_unique=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            df_summary_validated_only=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            best_df=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            quarantine_df=pd.DataFrame(columns=runner.QUARANTINE_COLUMNS),
        )
        row = reasons.loc[reasons["epic_id"].astype(str) == "1"].iloc[0]
        self.assertEqual(str(row["query"]), "EPIC 1")
        self.assertEqual(str(row["terminal_reason"]), "shortlist_precheck_reject")
        self.assertEqual(str(row["shortlist_rejection_reason"]), "whiteness_null_and_triage_unusable")
        self.assertTrue(bool(row["rejected_before_candidate_scoring"]))
        self.assertEqual(int(funnel["n_excluded_by_shortlist_precheck"]), 1)

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

    def test_build_epic_funnel_and_reasons_assigns_terminal_reason(self) -> None:
        runner = K2ShortlistPeriodRunner()
        raw_epics_df = pd.DataFrame(
            [
                {"epic_id": "1", "query": "EPIC 1", "triage_status": "error", "triage_why_not_usable": "triage_status=error", "n_events": 0},
                {"epic_id": "2", "query": "EPIC 2", "triage_status": "ok", "triage_why_not_usable": "", "n_events": 0},
                {"epic_id": "3", "query": "EPIC 3", "triage_status": "ok", "triage_why_not_usable": "", "n_events": 5},
            ]
        )
        df_summary_raw = pd.DataFrame(
            [
                {"epic": "3", "query": "EPIC 3", "reason": "cluster_only_validation_error", "P": 7.0},
            ]
        ).reindex(columns=runner.SUMMARY_COLUMNS)
        df_summary_valid = df_summary_raw.copy()
        df_summary_unique = df_summary_raw.copy()
        df_summary_validated_only = pd.DataFrame(columns=runner.SUMMARY_COLUMNS)
        best_df = pd.DataFrame([{"epic": "3", "P": 7.0}])
        quarantine_df = pd.DataFrame(columns=runner.QUARANTINE_COLUMNS)

        funnel, reasons = runner._build_epic_funnel_and_reasons(
            raw_epics_df=raw_epics_df,
            selected_queries=["EPIC 3"],
            selection_meta={
                "period_stage_selection_mode": "topk",
                "period_stage_max_epics": 200,
                "n_enter_period_stage": 1,
                "n_excluded_by_topk_gate": 2,
                "ranking_basis": "best_shape_score desc, best_depth_snr desc",
            },
            df_summary_raw=df_summary_raw,
            df_summary_valid=df_summary_valid,
            df_summary_unique=df_summary_unique,
            df_summary_validated_only=df_summary_validated_only,
            best_df=best_df,
            quarantine_df=quarantine_df,
        )

        reason_map = dict(zip(reasons["epic_id"].astype(str), reasons["terminal_reason"].astype(str)))
        self.assertEqual(reason_map["1"], "no_lightcurve/load_failed")
        self.assertEqual(reason_map["2"], "no_events")
        self.assertEqual(reason_map["3"], "fails_validation")
        self.assertEqual(int(funnel["n_total_epics"]), 3)
        self.assertEqual(int(funnel["n_best_unique_epics"]), 1)
        self.assertEqual(int(funnel["n_enter_period_stage"]), 1)
        self.assertEqual(int(funnel["n_excluded_by_topk_gate"]), 2)

    def test_period_failure_context_surfaces_in_quarantine_and_funnel_details(self) -> None:
        runner = K2ShortlistPeriodRunner()
        quarantine_df = pd.DataFrame(
            [
                {
                    "epic_id": "42",
                    "query": "EPIC 42",
                    "reason": "P_null_or_missing",
                    "source_reason": "no_cluster_periods",
                    "failure_category": "",
                    "failure_detail": "",
                }
            ]
        )
        payload = {
            "failure_category": "candidate_filter_rejection",
            "detail": "all_candidate_periods_failed_filters",
            "n_events_raw": 17,
            "n_events_after_filters": 9,
            "params": {
                "infer_max_period_days": 20.0,
                "infer_min_hits": 1,
                "infer_tol_frac": 0.01,
                "min_cluster_count": 3,
                "period_cap_days": 20.0,
                "min_period_days": 0.5,
                "top_k_periods": 3,
            },
            "extra": {
                "hist_total": 30,
                "hist_finite_period": 29,
                "hist_in_period_range": 11,
                "hist_pass_cluster_count": 2,
                "hist_pass_all_filters": 0,
            },
        }
        q_aug = runner._augment_quarantine_with_failure_diagnostics(
            quarantine_df=quarantine_df,
            inference_failures_by_epic={"42": payload},
        )
        self.assertEqual(float(q_aug.iloc[0]["min_period_days"]), 0.5)
        self.assertEqual(float(q_aug.iloc[0]["top_k_periods"]), 3.0)
        self.assertEqual(float(q_aug.iloc[0]["hist_pass_all_filters"]), 0.0)

        raw_epics_df = pd.DataFrame(
            [
                {
                    "epic_id": "42",
                    "query": "EPIC 42",
                    "triage_status": "ok",
                    "triage_why_not_usable": "",
                    "triage_usable": True,
                    "n_events": 9,
                }
            ]
        )
        empty_summary = pd.DataFrame(columns=runner.SUMMARY_COLUMNS)
        funnel, reasons = runner._build_epic_funnel_and_reasons(
            raw_epics_df=raw_epics_df,
            selected_queries=["EPIC 42"],
            selection_meta={
                "period_stage_selection_mode": "all",
                "n_enter_period_stage": 1,
                "n_excluded_by_topk_gate": 0,
                "ranking_basis": "best_shape_score desc, best_depth_snr desc",
            },
            df_summary_raw=empty_summary,
            df_summary_valid=empty_summary,
            df_summary_unique=empty_summary,
            df_summary_validated_only=empty_summary,
            best_df=pd.DataFrame(columns=runner.SUMMARY_COLUMNS),
            quarantine_df=q_aug,
        )
        self.assertEqual(int(funnel["n_total_epics"]), 1)
        row = reasons.iloc[0]
        details = row["details_json"]
        self.assertIn('"period_failure_category": "candidate_filter_rejection"', details)
        self.assertIn('"period_hist_pass_all_filters": 0.0', details)

    def test_cluster2_validated_rows_are_flagged_for_review_and_guardrails(self) -> None:
        runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(
                CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE=0.20,
                CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE=0.10,
            )
        )
        df = pd.DataFrame(
            [
                {
                    "epic": "1",
                    "query": "EPIC 1",
                    "reason": "validated",
                    "P": 0.8,
                    "cluster_count": 2,
                    "n_events_after_filters": 2,
                    "hit_rate_shape": 0.15,
                    "soft_hit_rate": 0.05,
                },
                {
                    "epic": "2",
                    "query": "EPIC 2",
                    "reason": "validated",
                    "P": 8.0,
                    "cluster_count": 2,
                    "n_events_after_filters": 3,
                    "hit_rate_shape": 0.30,
                    "soft_hit_rate": 0.20,
                },
                {
                    "epic": "3",
                    "query": "EPIC 3",
                    "reason": "validated",
                    "P": 12.0,
                    "cluster_count": 3,
                    "n_events_after_filters": 8,
                    "hit_rate_shape": 0.01,
                    "soft_hit_rate": 0.01,
                },
            ]
        )

        annotated = runner._annotate_validated_review_flags(df)

        row1 = annotated.loc[annotated["epic"] == "1"].iloc[0]
        self.assertTrue(bool(row1["manual_review_required"]))
        self.assertIn("supported high-recall mode candidate requires review", str(row1["manual_review_reason"]))
        self.assertTrue(bool(row1["cluster2_watch_very_short_period"]))
        self.assertTrue(bool(row1["cluster2_watch_low_event_support"]))
        self.assertFalse(bool(row1["cluster2_guardrail_hit_rate_shape_pass"]))
        self.assertFalse(bool(row1["cluster2_guardrail_soft_hit_rate_pass"]))
        self.assertFalse(bool(row1["cluster2_guardrail_pass"]))
        self.assertIn("hit_rate_shape<0.200", str(row1["cluster2_guardrail_reason"]))
        self.assertIn("soft_hit_rate<0.100", str(row1["cluster2_guardrail_reason"]))

        row2 = annotated.loc[annotated["epic"] == "2"].iloc[0]
        self.assertTrue(bool(row2["manual_review_required"]))
        self.assertTrue(bool(row2["cluster2_guardrail_pass"]))

        row3 = annotated.loc[annotated["epic"] == "3"].iloc[0]
        self.assertFalse(bool(row3["manual_review_required"]))
        self.assertTrue(bool(row3["cluster2_guardrail_pass"]))

    def test_cluster2_guardrail_rejection_moves_validated_rows_to_quarantine(self) -> None:
        runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(
                MIN_CLUSTER_COUNT=2,
                CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE=0.20,
            )
        )
        df_summary_valid = pd.DataFrame(
            [
                {
                    "epic": "1",
                    "query": "EPIC 1",
                    "reason": "validated",
                    "cluster_count": 2,
                    "hit_rate_shape": 0.10,
                    "soft_hit_rate": 0.30,
                    "n_events_raw": 3,
                    "n_events_after_filters": 3,
                },
                {
                    "epic": "2",
                    "query": "EPIC 2",
                    "reason": "validated",
                    "cluster_count": 2,
                    "hit_rate_shape": 0.25,
                    "soft_hit_rate": 0.30,
                    "n_events_raw": 4,
                    "n_events_after_filters": 4,
                },
            ]
        )

        kept, quarantine_df, rejected = runner._apply_cluster2_validated_guardrails(
            df_summary_valid=df_summary_valid,
            quarantine_df=pd.DataFrame(columns=runner.QUARANTINE_COLUMNS),
        )

        self.assertEqual(rejected, 1)
        self.assertEqual(set(kept["epic"].astype(str)), {"2"})
        self.assertEqual(len(quarantine_df), 1)
        qrow = quarantine_df.iloc[0]
        self.assertEqual(str(qrow["epic_id"]), "1")
        self.assertEqual(str(qrow["reason"]), "validated_guardrail_reject")
        self.assertEqual(str(qrow["source_reason"]), "cluster2_guardrail_rejection")
        self.assertIn("hit_rate_shape<0.200", str(qrow["failure_detail"]))

    def test_mcc_policy_mode_and_note(self) -> None:
        default_runner = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(MIN_CLUSTER_COUNT=3))
        experimental_runner = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(MIN_CLUSTER_COUNT=2))

        self.assertEqual(default_runner._mcc_policy_mode(), "precision_first_default")
        self.assertIn("precision-first default", default_runner._mcc_policy_note())
        self.assertEqual(experimental_runner._mcc_policy_mode(), "supported_high_recall")
        self.assertIn("supported high-recall mode", experimental_runner._mcc_policy_note())

    def test_run_cli_accepts_period_stage_n_override(self) -> None:
        with mock.patch.object(K2ShortlistPeriodRunner, "run", autospec=True, return_value={"ok": True}) as run_mock:
            out = K2ShortlistPeriodRunner.run_cli(
                ["--min-cluster-count", "2", "--period-stage-n", "2000", "--run-id", "cli_override_test"]
            )

        self.assertEqual(out, {"ok": True})
        runner_self = run_mock.call_args.args[0]
        self.assertEqual(int(runner_self.config.MIN_CLUSTER_COUNT), 2)
        self.assertEqual(int(runner_self.config.PERIOD_STAGE_N), 2000)
        self.assertEqual(str(runner_self.config.RUN_ID), "cli_override_test")

    def test_compare_report_marks_validation_capable_runs(self) -> None:
        case_dir = self._make_case_dir()
        baseline_dir = case_dir / "baseline"
        trial_dir = case_dir / "trial"
        out_dir = case_dir / "out"
        baseline_dir.mkdir(parents=True, exist_ok=False)
        trial_dir.mkdir(parents=True, exist_ok=False)

        baseline_best = pd.DataFrame(
            [
                {
                    "epic": "100",
                    "query": "EPIC 100",
                    "reason": "validated",
                    "P": 12.0,
                    "cluster_count": 3,
                    "n_events_after_filters": 8,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.30,
                    "hit_rate_shape": 0.30,
                    "soft_hit_rate": 0.30,
                }
            ]
        )
        trial_best = pd.DataFrame(
            [
                {
                    "epic": "100",
                    "query": "EPIC 100",
                    "reason": "validated",
                    "P": 12.0,
                    "cluster_count": 3,
                    "n_events_after_filters": 8,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.30,
                    "hit_rate_shape": 0.30,
                    "soft_hit_rate": 0.30,
                },
                {
                    "epic": "200",
                    "query": "EPIC 200",
                    "reason": "validated",
                    "P": 0.8,
                    "cluster_count": 2,
                    "n_events_after_filters": 2,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.04,
                    "hit_rate_shape": 0.04,
                    "soft_hit_rate": 0.04,
                },
            ]
        )
        baseline_validated = pd.DataFrame(
            [
                {
                    "epic": "100",
                    "query": "EPIC 100",
                    "P": 12.0,
                    "reason": "validated",
                    "cluster_count": 3,
                    "n_events_after_filters": 8,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.30,
                    "hit_rate_shape": 0.30,
                    "soft_hit_rate": 0.30,
                }
            ]
        )
        trial_validated = pd.DataFrame(
            [
                {
                    "epic": "100",
                    "query": "EPIC 100",
                    "P": 12.0,
                    "reason": "validated",
                    "cluster_count": 3,
                    "n_events_after_filters": 8,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.30,
                    "hit_rate_shape": 0.30,
                    "soft_hit_rate": 0.30,
                },
                {
                    "epic": "200",
                    "query": "EPIC 200",
                    "P": 0.8,
                    "reason": "validated",
                    "cluster_count": 2,
                    "n_events_after_filters": 2,
                    "coverage_rate": 1.0,
                    "hit_rate_snr": 0.04,
                    "hit_rate_shape": 0.04,
                    "soft_hit_rate": 0.04,
                },
            ]
        )
        baseline_funnel = pd.DataFrame([{"epic_id": "200", "terminal_reason": "no_cluster_periods", "source_reason": "candidate_filter_rejection"}])
        trial_funnel = pd.DataFrame([{"epic_id": "200", "terminal_reason": "validated", "source_reason": "validated"}])
        baseline_diag = pd.DataFrame([{
            "mcc_policy_mode": "precision_first_default",
            "min_cluster_count": 3,
            "default_min_cluster_count": 3,
            "manual_review_cluster_count_eq": 2,
            "cluster2_guardrail_hit_rate_shape_min": 0.1,
            "cluster2_guardrail_soft_hit_rate_min": 0.1,
            "rows_best": 1,
            "rows_validated_only": 1,
            "n_validated_period": 1,
            "n_quarantined_no_cluster_periods": 1,
        }])
        trial_diag = pd.DataFrame([{
            "mcc_policy_mode": "supported_high_recall",
            "min_cluster_count": 2,
            "default_min_cluster_count": 3,
            "manual_review_cluster_count_eq": 2,
            "cluster2_guardrail_hit_rate_shape_min": 0.1,
            "cluster2_guardrail_soft_hit_rate_min": 0.1,
            "rows_best": 2,
            "rows_validated_only": 2,
            "n_validated_period": 2,
            "n_quarantined_no_cluster_periods": 0,
        }])

        baseline_best.to_csv(baseline_dir / "period_shortlist_best.csv", index=False)
        trial_best.to_csv(trial_dir / "period_shortlist_best.csv", index=False)
        baseline_validated.to_csv(baseline_dir / "period_shortlist_summary_validated_only.csv", index=False)
        trial_validated.to_csv(trial_dir / "period_shortlist_summary_validated_only.csv", index=False)
        baseline_funnel.to_csv(baseline_dir / "epic_funnel_reasons.csv", index=False)
        trial_funnel.to_csv(trial_dir / "epic_funnel_reasons.csv", index=False)
        baseline_diag.to_csv(baseline_dir / "period_shortlist_diagnostics.csv", index=False)
        trial_diag.to_csv(trial_dir / "period_shortlist_diagnostics.csv", index=False)
        pd.DataFrame(columns=["failure_category"]).to_csv(baseline_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(columns=["failure_category"]).to_csv(trial_dir / "period_shortlist_quarantine.csv", index=False)

        out = K2ShortlistPeriodCompare().run(
            baseline_run_dir=baseline_dir,
            trial_run_dir=trial_dir,
            out_dir=out_dir,
        )

        report_df = pd.read_csv(out["report_csv"])
        validation_row = report_df.loc[report_df["section"] == "validation_diagnosis"].iloc[0]
        self.assertEqual(str(validation_row["baseline_value"]), "validated_outputs_present")
        self.assertEqual(str(validation_row["trial_value"]), "validated_outputs_present")
        self.assertIn("Validation-capable comparison", str(validation_row["note"]))
        policy_row = report_df.loc[
            (report_df["section"] == "policy") & (report_df["metric"] == "mcc_policy_mode")
        ].iloc[0]
        self.assertEqual(str(policy_row["baseline_value"]), "precision_first_default")
        self.assertEqual(str(policy_row["trial_value"]), "supported_high_recall")
        cluster2_review_df = pd.read_csv(out["cluster2_review_csv"])
        self.assertEqual(list(cluster2_review_df["epic"].astype(str)), ["200"])
        self.assertTrue(bool(cluster2_review_df.iloc[0]["cluster2_watch_low_event_support"]))
        self.assertTrue(bool(cluster2_review_df.iloc[0]["candidate_stage_rescued"]))
        self.assertTrue(bool(cluster2_review_df.iloc[0]["validated_stage_rescued"]))


if __name__ == "__main__":
    unittest.main()
