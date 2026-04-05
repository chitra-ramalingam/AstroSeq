from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidation import K2DetectorQualityGatedScaleValidation
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis import (
    K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis import (
    K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment import (
    K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment,
)
from src.Classifiers.K2.Batch.K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation import (
    K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation,
)
from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner


class TestK2DetectorQualityGatedScaleValidation(unittest.TestCase):
    def _make_case_dir(self) -> Path:
        case_dir = Path("tmp_pycache") / f"k2_detector_quality_gated_scale_validation_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(case_dir, ignore_errors=True))
        return case_dir

    def _make_population_df(self) -> pd.DataFrame:
        specs = [
            ("A_unusable_error", 5),
            ("B_unusable_quality_gate", 4),
            ("C_usable_n_events_0", 5),
            ("D_usable_n_events_1", 4),
            ("E_usable_n_events_ge2_no_valid_period", 6),
            ("F_usable_validated_period", 3),
        ]
        rows = []
        epic = 200000000
        for label, count in specs:
            for _ in range(count):
                epic += 1
                base = {
                    "query": f"EPIC {epic}",
                    "epic_id": f"EPIC_{epic}",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "n_events": 2,
                    "n_periods_validated": 0,
                }
                if label == "A_unusable_error":
                    base["triage_status"] = "error"
                    base["triage_usable"] = False
                elif label == "B_unusable_quality_gate":
                    base["triage_status"] = "ok"
                    base["triage_usable"] = False
                elif label == "C_usable_n_events_0":
                    base["n_events"] = 0
                elif label == "D_usable_n_events_1":
                    base["n_events"] = 1
                elif label == "F_usable_validated_period":
                    base["n_periods_validated"] = 1
                rows.append(base)
        return pd.DataFrame(rows)

    def test_sample_manifest_is_nested_and_preserves_weights(self) -> None:
        runner = K2DetectorQualityGatedScaleValidation()
        population = runner._load_population_from_df_for_test(self._make_population_df())

        manifest_small = runner._build_sample_manifest_df(population=population, target_n=12, random_seed=7)
        manifest_large = runner._build_sample_manifest_df(population=population, target_n=18, random_seed=7)

        self.assertEqual(len(manifest_small), 12)
        self.assertEqual(len(manifest_large), 18)
        self.assertTrue(
            set(manifest_small["epic_id_canonical"].astype(str)).issubset(
                set(manifest_large["epic_id_canonical"].astype(str))
            )
        )
        self.assertIn("stratum_label", manifest_small.columns)
        self.assertIn("sample_weight", manifest_small.columns)
        for stratum_label, sub in manifest_small.groupby("stratum_label"):
            pop_n = int(sub["stratum_population_n"].iloc[0])
            sample_n = int(sub["stratum_sample_n"].iloc[0])
            self.assertAlmostEqual(float(sub["sample_weight"].iloc[0]), float(pop_n / sample_n))

    def test_summary_and_go_no_go_report_are_weighted_and_gate_on_winner_count(self) -> None:
        runner = K2DetectorQualityGatedScaleValidation()
        analysis_df = pd.DataFrame(
            [
                {
                    "stratum_label": "A_unusable_error",
                    "stratum_population_n": 10,
                    "sample_weight": 5.0,
                    "detector_winner": 1,
                    "improved_best_shape_score": 0,
                    "improved_best_depth_snr": 1,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 1,
                    "winner_and_qg_shortlisted": 1,
                    "winner_and_qg_quarantined": 0,
                    "winner_in_usable_strata": 0,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "",
                },
                {
                    "stratum_label": "A_unusable_error",
                    "stratum_population_n": 10,
                    "sample_weight": 5.0,
                    "detector_winner": 1,
                    "improved_best_shape_score": 1,
                    "improved_best_depth_snr": 0,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 1,
                    "winner_and_qg_shortlisted": 1,
                    "winner_and_qg_quarantined": 0,
                    "winner_in_usable_strata": 0,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "",
                },
                {
                    "stratum_label": "A_unusable_error",
                    "stratum_population_n": 10,
                    "sample_weight": 5.0,
                    "detector_winner": 0,
                    "improved_best_shape_score": 0,
                    "improved_best_depth_snr": 0,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 0,
                    "winner_and_qg_shortlisted": 0,
                    "winner_and_qg_quarantined": 0,
                    "winner_in_usable_strata": 0,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "",
                },
                {
                    "stratum_label": "C_usable_n_events_0",
                    "stratum_population_n": 12,
                    "sample_weight": 4.0,
                    "detector_winner": 1,
                    "improved_best_shape_score": 0,
                    "improved_best_depth_snr": 1,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 0,
                    "winner_and_qg_shortlisted": 0,
                    "winner_and_qg_quarantined": 1,
                    "winner_in_usable_strata": 1,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "empty_histogram",
                },
                {
                    "stratum_label": "C_usable_n_events_0",
                    "stratum_population_n": 12,
                    "sample_weight": 4.0,
                    "detector_winner": 0,
                    "improved_best_shape_score": 0,
                    "improved_best_depth_snr": 0,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 0,
                    "winner_and_qg_shortlisted": 0,
                    "winner_and_qg_quarantined": 0,
                    "winner_in_usable_strata": 0,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "",
                },
                {
                    "stratum_label": "C_usable_n_events_0",
                    "stratum_population_n": 12,
                    "sample_weight": 4.0,
                    "detector_winner": 0,
                    "improved_best_shape_score": 0,
                    "improved_best_depth_snr": 0,
                    "default_shortlisted": 0,
                    "quality_gated_shortlisted": 0,
                    "winner_and_qg_shortlisted": 0,
                    "winner_and_qg_quarantined": 0,
                    "winner_in_usable_strata": 0,
                    "default_shortlisted_but_qg_not_shortlisted": 0,
                    "quality_gated_failure_category_norm": "",
                },
            ]
        )

        summary_df = runner._build_downstream_summary_df(analysis_df=analysis_df.copy())
        go_no_go_df = runner._build_go_no_go_report_df(analysis_df=analysis_df.copy(), summary_df=summary_df)

        conversion_row = summary_df.loc[
            (summary_df["metric"].astype(str) == "downstream_conversion_rate")
            & (summary_df["stratum_label"].astype(str) == "overall")
        ].iloc[0]
        self.assertGreater(float(conversion_row["estimate"]), 0.60)
        self.assertIn("qg_quarantined_failure_share__empty_histogram", summary_df["metric"].astype(str).tolist())

        final_row = go_no_go_df.loc[go_no_go_df["metric"].astype(str) == "final_recommendation"].iloc[0]
        self.assertEqual(str(final_row["observed_value"]), "hold")

    def test_run_expands_once_when_initial_winner_count_is_too_sparse(self) -> None:
        runner = K2DetectorQualityGatedScaleValidation()
        population = runner._load_population_from_df_for_test(self._make_population_df())
        case_dir = self._make_case_dir()
        stage_calls = []

        def stage_stub(*, manifest_df, sample_manifest_csv, stage_dir, **_kwargs):
            stage_calls.append(int(len(manifest_df)))
            stage_dir = Path(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            observed_winners = 1 if len(manifest_df) == 6 else 3
            pairwise_df = manifest_df.copy()
            pairwise_df["gained_extra_events"] = False
            pairwise_df.loc[pairwise_df.index[:observed_winners], "gained_extra_events"] = True
            pairwise_df["improved_best_shape_score"] = False
            pairwise_df["improved_best_depth_snr"] = False
            pairwise_csv = stage_dir / "paired_detector_comparison.csv"
            pairwise_df.to_csv(pairwise_csv, index=False)
            detector_summary_csv = stage_dir / "paired_detector_summary.csv"
            pd.DataFrame([{"metric": "detector_winners_observed", "value": observed_winners}]).to_csv(
                detector_summary_csv, index=False
            )

            analysis_df = manifest_df.copy()
            analysis_df["detector_winner"] = False
            analysis_df.loc[analysis_df.index[:observed_winners], "detector_winner"] = True
            analysis_df["improved_best_shape_score"] = False
            analysis_df["improved_best_depth_snr"] = False
            analysis_df["default_shortlisted"] = False
            analysis_df["quality_gated_shortlisted"] = False
            analysis_df["winner_and_qg_shortlisted"] = False
            analysis_df["winner_and_qg_quarantined"] = False
            analysis_df["winner_in_usable_strata"] = False
            analysis_df["default_shortlisted_but_qg_not_shortlisted"] = False
            analysis_df["quality_gated_failure_category_norm"] = ""
            analysis_df.loc[analysis_df.index[: max(1, observed_winners - 1)], "winner_and_qg_shortlisted"] = True
            analysis_df.loc[analysis_df.index[: max(1, observed_winners - 1)], "quality_gated_shortlisted"] = True
            analysis_df.loc[analysis_df.index[: max(1, observed_winners - 1)], "winner_in_usable_strata"] = True
            if observed_winners > 1:
                analysis_df.loc[analysis_df.index[observed_winners - 1], "winner_and_qg_quarantined"] = True
                analysis_df.loc[analysis_df.index[observed_winners - 1], "quality_gated_failure_category_norm"] = "empty_histogram"
            downstream_pairwise_csv = stage_dir / "paired_downstream_analysis.csv"
            analysis_df.to_csv(downstream_pairwise_csv, index=False)

            summary_df = pd.DataFrame(
                [
                    {
                        "scope": "overall",
                        "stratum_label": "overall",
                        "metric": "downstream_conversion_rate",
                        "estimate": 0.66,
                        "ci_low": 0.52,
                        "ci_high": 0.80,
                    }
                ]
            )
            summary_csv = stage_dir / "downstream_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            go_no_go_df = pd.DataFrame(
                [
                    {
                        "stratum_label": "overall",
                        "weighting": "stratified_design_weight",
                        "metric": "final_recommendation",
                        "observed_value": "go",
                        "min_allowed": "",
                        "max_allowed": "",
                        "passed": True,
                    }
                ]
            )
            go_no_go_csv = stage_dir / "go_no_go_report.csv"
            go_no_go_txt = stage_dir / "go_no_go_report.txt"
            go_no_go_df.to_csv(go_no_go_csv, index=False)
            go_no_go_txt.write_text("recommendation: go\n", encoding="utf-8")
            return {
                "pairwise_df": pairwise_df,
                "pairwise_csv": pairwise_csv,
                "detector_summary_csv": detector_summary_csv,
                "downstream_pairwise_df": analysis_df,
                "downstream_pairwise_csv": downstream_pairwise_csv,
                "summary_df": summary_df,
                "summary_csv": summary_csv,
                "go_no_go_df": go_no_go_df,
                "go_no_go_csv": go_no_go_csv,
                "go_no_go_txt": go_no_go_txt,
            }

        runner._load_population = lambda population_batch_csv: population.copy()
        runner._run_stage = stage_stub
        out = runner.run(
            population_batch_csv=case_dir / "unused.csv",
            out_dir=case_dir / "out",
            initial_sample_n=6,
            expanded_sample_n=8,
            max_sample_n=10,
            random_seed=13,
            min_winners_for_reliable_conversion=3,
            conversion_ci_width_threshold=0.30,
        )

        self.assertEqual(stage_calls, [6, 8])
        self.assertEqual(int(out["final_sample_n"]), 8)
        self.assertEqual(int(out["observed_winners"]), 3)
        self.assertEqual(str(out["final_recommendation"]), "go")

    def test_run_detector_mode_reuses_renamed_batch_results_csv(self) -> None:
        runner = K2DetectorQualityGatedScaleValidation()
        case_dir = self._make_case_dir()
        run_dir = case_dir / "detector_default_run"
        run_dir.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(
            [
                {
                    "query": "EPIC 200000001",
                    "epic_id": "EPIC_200000001",
                    "detector_operating_mode": "detector_default",
                    "triage_status": "ok",
                    "triage_usable": True,
                    "n_events": 1,
                }
            ]
        ).to_csv(run_dir / "Apr3_batch_results_def_run.csv", index=False)

        out = runner._run_detector_mode(
            sample_manifest_csv=case_dir / "unused_manifest.csv",
            run_dir=run_dir,
            detector_operating_mode="detector_default",
            cache_only=True,
            max_workers=1,
        )
        self.assertEqual(
            Path(out["batch_results_csv"]).name,
            "Apr3_batch_results_def_run.csv",
        )

    def test_build_downstream_pairwise_df_normalizes_downstream_epic_keys(self) -> None:
        runner = K2DetectorQualityGatedScaleValidation()
        manifest_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000001",
                    "query": "EPIC 200000001",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 5,
                    "sample_weight": 1.0,
                    "is_usable_stratum": True,
                }
            ]
        )
        detector_pairwise_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000001",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                }
            ]
        )

        runner.downstream_helper._load_downstream_state = lambda **kwargs: {"label": kwargs["label"]}
        runner.downstream_helper._build_state_outcome_frame = lambda state, prefix: pd.DataFrame(
            [
                {
                    "epic_id_canonical": "200000001",
                    f"{prefix}_terminal_group": "shortlisted" if prefix == "quality_gated" else "failed_downstream",
                    f"{prefix}_downstream_outcome": "validated" if prefix == "quality_gated" else "failed_no_period",
                    f"{prefix}_downstream_outcome_score": 5 if prefix == "quality_gated" else 1,
                    f"{prefix}_failure_reason_bucket": "",
                    f"{prefix}_failure_category": "",
                    f"{prefix}_failure_detail": "",
                    f"{prefix}_reason": "",
                    f"{prefix}_source_reason": "",
                    f"{prefix}_shortlist_rejection_reason": "",
                    f"{prefix}_terminal_reason": "",
                    f"{prefix}_failure_P": pd.NA,
                    f"{prefix}_failure_period_bin": "no_P_available",
                    f"{prefix}_best_reason": "validated" if prefix == "quality_gated" else "",
                    f"{prefix}_best_P": 4.2 if prefix == "quality_gated" else pd.NA,
                    f"{prefix}_best_period_bin": "(1,5]" if prefix == "quality_gated" else "no_P_available",
                    f"{prefix}_manual_review_required": False,
                    f"{prefix}_best_query": "EPIC 200000001" if prefix == "quality_gated" else "",
                }
            ]
        )

        analysis_df = runner._build_downstream_pairwise_df(
            manifest_df=manifest_df,
            detector_pairwise_df=detector_pairwise_df,
            default_downstream_run_dir=Path("unused_default"),
            quality_gated_downstream_run_dir=Path("unused_qg"),
        )

        self.assertEqual(str(analysis_df.loc[0, "quality_gated_terminal_group"]), "shortlisted")
        self.assertTrue(bool(analysis_df.loc[0, "winner_and_qg_shortlisted"]))
        self.assertFalse(bool(analysis_df.loc[0, "winner_and_qg_no_record"]))

    def test_post_hold_failure_analysis_rolls_up_saved_stage_outputs(self) -> None:
        case_dir = self._make_case_dir()
        out_dir = case_dir / "scale_validation"
        stage_dir = out_dir / "stage_n600"
        default_downstream_dir = stage_dir / "default_downstream"
        qg_downstream_dir = stage_dir / "quality_gated_downstream"
        default_downstream_dir.mkdir(parents=True, exist_ok=False)
        qg_downstream_dir.mkdir(parents=True, exist_ok=False)

        manifest_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000001",
                    "query": "EPIC 200000001",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
                {
                    "epic_id_canonical": "EPIC_200000002",
                    "query": "EPIC 200000002",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
                {
                    "epic_id_canonical": "EPIC_200000003",
                    "query": "EPIC 200000003",
                    "stratum_label": "D_usable_n_events_1",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
            ]
        )
        manifest_df.to_csv(stage_dir / "sampled_epic_manifest.csv", index=False)

        paired_detector_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000001",
                    "query": "EPIC 200000001",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id_canonical": "EPIC_200000002",
                    "query": "EPIC 200000002",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id_canonical": "EPIC_200000003",
                    "query": "EPIC 200000003",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
            ]
        )
        paired_detector_df.to_csv(stage_dir / "paired_detector_comparison.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            default_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame([{"min_cluster_count": 3}]).to_csv(default_downstream_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            qg_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "epic_id": "200000001",
                    "query": "EPIC 200000001",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "n_events_after_filters": 4,
                    "hist_total": 3,
                    "hist_in_period_range": 3,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                },
                {
                    "epic_id": "200000002",
                    "query": "EPIC 200000002",
                    "failure_category": "empty_histogram",
                    "shortlist_rejection_reason": "empty_histogram",
                    "failure_detail": "infer_periods_from_events_returned_empty_hist",
                    "source_reason": "no_cluster_periods",
                    "n_events_after_filters": 3,
                    "hist_total": pd.NA,
                    "hist_in_period_range": pd.NA,
                    "hist_pass_cluster_count": pd.NA,
                    "hist_pass_all_filters": pd.NA,
                    "min_cluster_count": 3,
                },
                {
                    "epic_id": "200000003",
                    "query": "EPIC 200000003",
                    "failure_category": "insufficient_events",
                    "shortlist_rejection_reason": "insufficient_events",
                    "failure_detail": "insufficient_events_for_period_inference",
                    "source_reason": "no_cluster_periods",
                    "n_events_after_filters": 1,
                    "hist_total": pd.NA,
                    "hist_in_period_range": pd.NA,
                    "hist_pass_cluster_count": pd.NA,
                    "hist_pass_all_filters": pd.NA,
                    "min_cluster_count": 3,
                },
            ]
        ).to_csv(qg_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {
                    "query": "EPIC 200000001",
                    "epic_id": "200000001",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":3.0,"period_hist_in_period_range":3.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":4.0,"selected_for_period_stage":true}'
                    ),
                },
                {
                    "query": "EPIC 200000002",
                    "epic_id": "200000002",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "empty_histogram",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"empty_histogram",'
                        '"period_failure_detail":"infer_periods_from_events_returned_empty_hist",'
                        '"period_n_events_after_filters":3.0,"selected_for_period_stage":true}'
                    ),
                },
                {
                    "query": "EPIC 200000003",
                    "epic_id": "200000003",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "insufficient_events",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"insufficient_events",'
                        '"period_failure_detail":"insufficient_events_for_period_inference",'
                        '"period_n_events_after_filters":1.0,"selected_for_period_stage":true}'
                    ),
                },
            ]
        ).to_csv(qg_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame([{"min_cluster_count": 3, "operating_mode_requested": "precision_first_default"}]).to_csv(
            qg_downstream_dir / "period_shortlist_diagnostics.csv", index=False
        )

        analysis_runner = K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis()
        out = analysis_runner.run(
            manifest_csv=stage_dir / "sampled_epic_manifest.csv",
            paired_detector_csv=stage_dir / "paired_detector_comparison.csv",
            default_downstream_dir=default_downstream_dir,
            quality_gated_downstream_dir=qg_downstream_dir,
            analysis_csv=out_dir / "analysis.csv",
            rollup_csv=out_dir / "rollup.csv",
            examples_per_group=2,
        )

        self.assertEqual(int(out["quarantined_winners_total"]), 3)
        self.assertEqual(int(out["bucket_counts"]["cluster / period policy"]), 1)
        self.assertEqual(int(out["bucket_counts"]["histogram construction / handling"]), 1)
        self.assertEqual(int(out["bucket_counts"]["true insufficient signal"]), 1)
        self.assertEqual(str(out["recommended_next_lever"]), "keep quality-gated limited to cached-failed setting")
        self.assertTrue((out_dir / "analysis.csv").exists())
        self.assertTrue((out_dir / "rollup.csv").exists())

    def test_cluster_policy_analysis_identifies_conditional_mcc2_carveout(self) -> None:
        case_dir = self._make_case_dir()
        out_dir = case_dir / "scale_validation"
        stage_dir = out_dir / "stage_n600"
        default_downstream_dir = stage_dir / "default_downstream"
        qg_downstream_dir = stage_dir / "quality_gated_downstream"
        default_downstream_dir.mkdir(parents=True, exist_ok=False)
        qg_downstream_dir.mkdir(parents=True, exist_ok=False)

        manifest_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000011",
                    "query": "EPIC 200000011",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
                {
                    "epic_id_canonical": "EPIC_200000012",
                    "query": "EPIC 200000012",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
                {
                    "epic_id_canonical": "EPIC_200000013",
                    "query": "EPIC 200000013",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
                {
                    "epic_id_canonical": "EPIC_200000014",
                    "query": "EPIC 200000014",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                },
            ]
        )
        manifest_df.to_csv(stage_dir / "sampled_epic_manifest.csv", index=False)

        paired_detector_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000011",
                    "query": "EPIC 200000011",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id_canonical": "EPIC_200000012",
                    "query": "EPIC 200000012",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id_canonical": "EPIC_200000013",
                    "query": "EPIC 200000013",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
                {
                    "epic_id_canonical": "EPIC_200000014",
                    "query": "EPIC 200000014",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                },
            ]
        )
        paired_detector_df.to_csv(stage_dir / "paired_detector_comparison.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            default_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame([{"min_cluster_count": 3}]).to_csv(default_downstream_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            qg_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "epic_id": "200000011",
                    "query": "EPIC 200000011",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "n_events_after_filters": 5,
                    "hist_total": 4,
                    "hist_in_period_range": 3,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                },
                {
                    "epic_id": "200000012",
                    "query": "EPIC 200000012",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "n_events_after_filters": 4,
                    "hist_total": 1,
                    "hist_in_period_range": 1,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                },
                {
                    "epic_id": "200000013",
                    "query": "EPIC 200000013",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "n_events_after_filters": 3,
                    "hist_total": 2,
                    "hist_in_period_range": 1,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                },
                {
                    "epic_id": "200000014",
                    "query": "EPIC 200000014",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "n_events_after_filters": 2,
                    "hist_total": 1,
                    "hist_in_period_range": 1,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                },
            ]
        ).to_csv(qg_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {
                    "query": "EPIC 200000011",
                    "epic_id": "200000011",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":4.0,"period_hist_in_period_range":3.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":5.0,"n_periods_proposed":0.0,'
                        '"n_periods_validated":0.0,"selected_for_period_stage":true}'
                    ),
                },
                {
                    "query": "EPIC 200000012",
                    "epic_id": "200000012",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":1.0,"period_hist_in_period_range":1.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":4.0,"n_periods_proposed":0.0,'
                        '"n_periods_validated":0.0,"selected_for_period_stage":true}'
                    ),
                },
                {
                    "query": "EPIC 200000013",
                    "epic_id": "200000013",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":2.0,"period_hist_in_period_range":1.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":3.0,"n_periods_proposed":0.0,'
                        '"n_periods_validated":0.0,"selected_for_period_stage":true}'
                    ),
                },
                {
                    "query": "EPIC 200000014",
                    "epic_id": "200000014",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":1.0,"period_hist_in_period_range":1.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":2.0,"n_periods_proposed":0.0,'
                        '"n_periods_validated":0.0,"selected_for_period_stage":true}'
                    ),
                },
            ]
        ).to_csv(qg_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame(
            [
                {
                    "min_cluster_count": 3,
                    "operating_mode_requested": "precision_first_default",
                    "mcc_policy_mode": "precision_first_default",
                    "manual_review_cluster_count_eq": 2,
                    "cluster2_guardrail_hit_rate_shape_min": 0.1,
                    "cluster2_guardrail_soft_hit_rate_min": 0.1,
                }
            ]
        ).to_csv(qg_downstream_dir / "period_shortlist_diagnostics.csv", index=False)

        analysis_runner = K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis()
        out = analysis_runner.run(
            manifest_csv=stage_dir / "sampled_epic_manifest.csv",
            paired_detector_csv=stage_dir / "paired_detector_comparison.csv",
            default_downstream_dir=default_downstream_dir,
            quality_gated_downstream_dir=qg_downstream_dir,
            analysis_csv=out_dir / "cluster_analysis.csv",
            rollup_csv=out_dir / "cluster_rollup.csv",
            examples_per_group=2,
        )

        self.assertEqual(int(out["cluster_policy_cases_total"]), 4)
        self.assertEqual(int(out["bucket_counts"]["supported MCC=2 carve-out candidate"]), 1)
        self.assertEqual(int(out["bucket_counts"]["single-candidate near-miss"]), 1)
        self.assertEqual(int(out["bucket_counts"]["three-event borderline"]), 1)
        self.assertEqual(int(out["bucket_counts"]["two-event low-support"]), 1)
        self.assertEqual(str(out["dominant_gate"]), "minimum cluster count")
        self.assertIn("MIN_CLUSTER_COUNT=2", str(out["recommended_smallest_safe_change"]))
        self.assertTrue((out_dir / "cluster_analysis.csv").exists())
        self.assertTrue((out_dir / "cluster_rollup.csv").exists())

    def test_shortlist_policy_resolves_conditional_mcc2_only_for_supported_cases(self) -> None:
        runner = K2ShortlistPeriodRunner(
            config=K2ShortlistPeriodConfig(
                OPERATING_MODE=str(K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME),
                MIN_CLUSTER_COUNT=3,
                CONDITIONAL_MIN_CLUSTER_COUNT_RELAX_ENABLED=True,
                CONDITIONAL_MIN_CLUSTER_COUNT_RELAX_TO=2,
                CONDITIONAL_MIN_CLUSTER_COUNT_MIN_EVENTS_AFTER_FILTERS=4,
                CONDITIONAL_MIN_CLUSTER_COUNT_MIN_HIST_IN_RANGE=2,
            )
        )
        hist_df = pd.DataFrame(
            [
                {"period": 3.0, "count_hits": 2, "pair_count": 10},
                {"period": 5.0, "count_hits": 2, "pair_count": 8},
                {"period": 7.0, "count_hits": 1, "pair_count": 5},
            ]
        )
        policy_hit = runner._resolve_candidate_filter_policy(
            hist_df=hist_df,
            n_events_after_filters=4,
            min_period_days=0.5,
            max_period_days=20.0,
        )
        policy_miss = runner._resolve_candidate_filter_policy(
            hist_df=hist_df,
            n_events_after_filters=3,
            min_period_days=0.5,
            max_period_days=20.0,
        )

        self.assertTrue(bool(policy_hit["conditional_min_cluster_count_applied"]))
        self.assertEqual(int(policy_hit["effective_min_cluster_count"]), 2)
        self.assertFalse(bool(policy_miss["conditional_min_cluster_count_applied"]))
        self.assertEqual(int(policy_miss["effective_min_cluster_count"]), 3)

    def test_conditional_mcc2_experiment_reuses_stage_and_reports_paired_gain(self) -> None:
        case_dir = self._make_case_dir()
        out_dir = case_dir / "scale_validation"
        stage_dir = out_dir / "stage_n600"
        default_downstream_dir = stage_dir / "default_downstream"
        baseline_qg_downstream_dir = stage_dir / "quality_gated_downstream"
        detector_default_run_dir = out_dir / "detector_default_run"
        detector_quality_gated_run_dir = out_dir / "detector_quality_gated_run"
        default_downstream_dir.mkdir(parents=True, exist_ok=False)
        baseline_qg_downstream_dir.mkdir(parents=True, exist_ok=False)
        detector_default_run_dir.mkdir(parents=True, exist_ok=False)
        detector_quality_gated_run_dir.mkdir(parents=True, exist_ok=False)

        manifest_df = pd.DataFrame(
            [
                {
                    "epic_id_canonical": "EPIC_200000021",
                    "query": "EPIC 200000021",
                    "stratum_label": "E_usable_n_events_ge2_no_valid_period",
                    "stratum_population_n": 10,
                    "sample_weight": 2.0,
                    "is_usable_stratum": True,
                }
            ]
        )
        manifest_df.to_csv(stage_dir / "sampled_epic_manifest.csv", index=False)
        pd.DataFrame(
            [
                {
                    "query": "EPIC 200000021",
                    "epic_id_canonical": "EPIC_200000021",
                    "gained_extra_events": True,
                    "improved_best_shape_score": False,
                    "improved_best_depth_snr": False,
                    "delta_n_events": 1,
                    "delta_best_shape_score": 0.0,
                    "delta_best_depth_snr": 0.0,
                }
            ]
        ).to_csv(stage_dir / "paired_detector_comparison.csv", index=False)
        pd.DataFrame([{"query": "EPIC 200000021"}]).to_csv(detector_default_run_dir / "batch_results.csv", index=False)
        pd.DataFrame([{"query": "EPIC 200000021"}]).to_csv(detector_quality_gated_run_dir / "batch_results.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            default_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(columns=["epic_id"]).to_csv(default_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame([{"min_cluster_count": 3}]).to_csv(default_downstream_dir / "period_shortlist_diagnostics.csv", index=False)

        pd.DataFrame(columns=["epic", "query", "reason", "P", "manual_review_required"]).to_csv(
            baseline_qg_downstream_dir / "period_shortlist_best.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "epic_id": "200000021",
                    "query": "EPIC 200000021",
                    "failure_category": "candidate_filter_rejection",
                    "shortlist_rejection_reason": "candidate_filter_rejection",
                    "failure_detail": "all_candidate_periods_below_min_cluster_count",
                    "source_reason": "no_cluster_periods",
                    "shortlist_rejection_stage": "post_candidate_scoring",
                    "n_events_after_filters": 5,
                    "hist_total": 4,
                    "hist_in_period_range": 3,
                    "hist_pass_cluster_count": 0,
                    "hist_pass_all_filters": 0,
                    "min_cluster_count": 3,
                }
            ]
        ).to_csv(baseline_qg_downstream_dir / "period_shortlist_quarantine.csv", index=False)
        pd.DataFrame(
            [
                {
                    "query": "EPIC 200000021",
                    "epic_id": "200000021",
                    "terminal_reason": "no_cluster_periods",
                    "source_reason": "candidate_filter_rejection",
                    "stage_reached": "period_inference",
                    "details_json": (
                        '{"period_failure_category":"candidate_filter_rejection",'
                        '"period_failure_detail":"all_candidate_periods_below_min_cluster_count",'
                        '"period_hist_total":4.0,"period_hist_in_period_range":3.0,'
                        '"period_hist_pass_cluster_count":0.0,"period_hist_pass_all_filters":0.0,'
                        '"period_n_events_after_filters":5.0,"selected_for_period_stage":true}'
                    ),
                }
            ]
        ).to_csv(baseline_qg_downstream_dir / "epic_funnel_reasons.csv", index=False)
        pd.DataFrame([{"min_cluster_count": 3}]).to_csv(
            baseline_qg_downstream_dir / "period_shortlist_diagnostics.csv", index=False
        )

        experiment_runner = K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment()

        def run_downstream_stage_stub(**kwargs):
            stage_out_dir = Path(kwargs["stage_out_dir"])
            stage_out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "epic": "200000021",
                        "query": "EPIC 200000021",
                        "reason": "validated",
                        "P": 5.5,
                        "manual_review_required": True,
                        "cluster_count": 2,
                    }
                ]
            ).to_csv(stage_out_dir / "period_shortlist_best.csv", index=False)
            pd.DataFrame(columns=["epic_id"]).to_csv(stage_out_dir / "period_shortlist_quarantine.csv", index=False)
            pd.DataFrame(columns=["epic_id"]).to_csv(stage_out_dir / "epic_funnel_reasons.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "min_cluster_count": 3,
                        "conditional_min_cluster_count_enabled": True,
                        "conditional_min_cluster_count_relax_to": 2,
                    }
                ]
            ).to_csv(stage_out_dir / "period_shortlist_diagnostics.csv", index=False)
            return {"run_dir": str(stage_out_dir)}

        experiment_runner.scale_validation._run_downstream_stage = run_downstream_stage_stub
        out = experiment_runner.run(
            out_dir=out_dir,
            stage_dir=stage_dir,
            detector_default_run_dir=detector_default_run_dir,
            detector_quality_gated_run_dir=detector_quality_gated_run_dir,
            max_workers=1,
            cache_only=True,
        )

        self.assertEqual(int(out["baseline_metrics"]["winners_in_best"]), 0)
        self.assertEqual(int(out["experiment_metrics"]["winners_in_best"]), 1)
        self.assertEqual(int(out["paired_gain_cases"]), 1)
        self.assertEqual(int(out["paired_regression_cases"]), 0)
        self.assertEqual(int(out["harmful_regression_cases"]), 0)
        self.assertTrue((stage_dir / "conditional_mcc2_experiment_comparison.csv").exists())
        self.assertTrue((stage_dir / "conditional_mcc2_experiment_summary.csv").exists())
        self.assertTrue((stage_dir / "conditional_mcc2_experiment_decision_audit.csv").exists())
        self.assertTrue((stage_dir / "conditional_mcc2_experiment_decision_audit.txt").exists())
        self.assertTrue((stage_dir / "conditional_mcc2_experiment_next_limited_broader_validation_plan.txt").exists())
        summary_df = pd.read_csv(stage_dir / "conditional_mcc2_experiment_summary.csv")
        self.assertIn("performance_safety_outcomes", summary_df["section"].astype(str).tolist())
        self.assertIn("residual_failure_composition_outcomes", summary_df["section"].astype(str).tolist())

    def test_limited_broader_validation_prepare_only_writes_plan(self) -> None:
        case_dir = self._make_case_dir()
        shards_root = case_dir / "shards"
        out_dir = case_dir / "broader_out"
        winners_csv = case_dir / "winners.csv"
        shards_root.mkdir(parents=True, exist_ok=False)
        winners_csv.write_text("epic_id\nEPIC_200000001\n", encoding="utf-8")

        runner = K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation()
        out = runner.run(
            shards_root=shards_root,
            out_dir=out_dir,
            winners_csv=winners_csv,
            max_workers=1,
            disable_validation=False,
            prepare_only=True,
        )

        self.assertTrue(bool(out["prepare_only"]))
        self.assertTrue(Path(out["plan_txt"]).exists())
        plan_text = Path(out["plan_txt"]).read_text(encoding="utf-8")
        self.assertIn("scale_validation_conditional_mcc2_experiment", plan_text)
        self.assertIn("default_global_policy_changed: false", plan_text)
        self.assertIn("supported_experimental_policy: false", plan_text)
        self.assertIn("automatic_scale_up_scheduled: false", plan_text)


def _load_population_from_df_for_test(self: K2DetectorQualityGatedScaleValidation, df: pd.DataFrame) -> pd.DataFrame:
    tmp_dir = Path("tmp_pycache")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = tmp_dir / f"k2_detector_quality_gated_scale_validation_pop_{uuid4().hex}.csv"
    df.to_csv(tmp_csv, index=False)
    return self._load_population(tmp_csv)


K2DetectorQualityGatedScaleValidation._load_population_from_df_for_test = _load_population_from_df_for_test


if __name__ == "__main__":
    unittest.main()
