from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidation import K2DetectorQualityGatedScaleValidation
from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig


class K2DetectorQualityGatedScaleValidationConditionalMcc2Experiment:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_quality_gated_scale_validation")
    DEFAULT_STAGE_DIR_NAME = "stage_n600"
    DEFAULT_EXPERIMENT_QG_DIR_NAME = "quality_gated_downstream_conditional_mcc2_experiment"
    DEFAULT_EXPERIMENT_ANALYSIS_CSV_NAME = "conditional_mcc2_experiment_paired_downstream_analysis.csv"
    DEFAULT_COMPARISON_CSV_NAME = "conditional_mcc2_experiment_comparison.csv"
    DEFAULT_SUMMARY_CSV_NAME = "conditional_mcc2_experiment_summary.csv"
    DEFAULT_GO_NO_GO_CSV_NAME = "conditional_mcc2_experiment_go_no_go_report.csv"
    DEFAULT_GO_NO_GO_TXT_NAME = "conditional_mcc2_experiment_go_no_go_report.txt"
    DEFAULT_DECISION_AUDIT_CSV_NAME = "conditional_mcc2_experiment_decision_audit.csv"
    DEFAULT_DECISION_AUDIT_TXT_NAME = "conditional_mcc2_experiment_decision_audit.txt"
    DEFAULT_NEXT_BROADER_PLAN_TXT_NAME = "conditional_mcc2_experiment_next_limited_broader_validation_plan.txt"
    DEFAULT_NEXT_BROADER_OUT_DIR = Path(
        r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream_conditional_mcc2_experiment"
    )
    PERFORMANCE_SAFETY_CRITERIA = {
        "downstream_conversion_rate",
        "observed_winner_count",
        "winner_share_from_usable_strata",
        "max_new_failure_share",
        "quarantine_to_best_ratio",
        "default_shortlisted_to_qg_not_shortlisted_rate",
    }
    RESIDUAL_FAILURE_COMPOSITION_CRITERIA = {"known_failure_mix_share"}

    def __init__(self) -> None:
        self.scale_validation = K2DetectorQualityGatedScaleValidation()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Reuse the completed stage_n600 scale-validation sample and detector outputs, rerun only the "
                "quality-gated downstream stage under the conditional MCC=2 experimental policy, and compare "
                "the result against the current baseline. This mode is for guarded validation only and does not "
                "promote the carve-out to a supported policy."
            )
        )
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--stage-dir", type=Path, default=None)
        p.add_argument("--detector-default-run-dir", type=Path, default=None)
        p.add_argument("--detector-quality-gated-run-dir", type=Path, default=None)
        p.add_argument("--max-workers", type=int, default=K2DetectorQualityGatedScaleValidation.DEFAULT_MAX_WORKERS)
        p.add_argument("--cache-only", action="store_true", help="Use cache-only validation fetches.")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        stage_dir = Path(args.stage_dir) if args.stage_dir is not None else out_dir / cls.DEFAULT_STAGE_DIR_NAME
        return cls().run(
            out_dir=out_dir,
            stage_dir=stage_dir,
            detector_default_run_dir=Path(args.detector_default_run_dir)
            if args.detector_default_run_dir is not None
            else out_dir / "detector_default_run",
            detector_quality_gated_run_dir=Path(args.detector_quality_gated_run_dir)
            if args.detector_quality_gated_run_dir is not None
            else out_dir / "detector_quality_gated_run",
            max_workers=int(args.max_workers),
            cache_only=bool(args.cache_only),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _metric_estimate(summary_df: pd.DataFrame, metric: str) -> float:
        sub = summary_df.loc[
            summary_df["stratum_label"].astype(str).eq("overall")
            & summary_df["metric"].astype(str).eq(str(metric))
        ]
        if len(sub) == 0:
            return float("nan")
        return float(pd.to_numeric(sub["estimate"], errors="coerce").iloc[0])

    @staticmethod
    def _report_metric_value(go_no_go_df: pd.DataFrame, metric: str) -> float:
        sub = go_no_go_df.loc[go_no_go_df["metric"].astype(str).eq(str(metric))]
        if len(sub) == 0:
            return float("nan")
        return float(pd.to_numeric(sub["observed_value"], errors="coerce").iloc[0])

    @staticmethod
    def _metric_failed_checks(go_no_go_df: pd.DataFrame) -> str:
        failed = go_no_go_df.loc[
            (go_no_go_df["metric"].astype(str) != "final_recommendation")
            & (~go_no_go_df["passed"].fillna(False).astype(bool))
        ].copy()
        if len(failed) == 0:
            return ""
        return "|".join(failed["metric"].astype(str).tolist())

    @staticmethod
    def _final_recommendation(go_no_go_df: pd.DataFrame) -> str:
        sub = go_no_go_df.loc[go_no_go_df["metric"].astype(str).eq("final_recommendation")]
        if len(sub) == 0:
            return ""
        return str(sub["observed_value"].iloc[0])

    @classmethod
    def _criterion_group(cls, metric: str) -> str:
        if str(metric) in cls.PERFORMANCE_SAFETY_CRITERIA:
            return "performance/safety"
        if str(metric) in cls.RESIDUAL_FAILURE_COMPOSITION_CRITERIA:
            return "residual failure-composition"
        if str(metric) == "final_recommendation":
            return "decision"
        return "other"

    @staticmethod
    def _threshold_text(min_allowed: Any, max_allowed: Any) -> str:
        min_num = pd.to_numeric(pd.Series([min_allowed]), errors="coerce").iloc[0]
        max_num = pd.to_numeric(pd.Series([max_allowed]), errors="coerce").iloc[0]
        if pd.notna(min_num) and pd.notna(max_num):
            if min_num == float("-inf"):
                return f"<= {max_num}"
            if max_num == float("inf"):
                return f">= {min_num}"
            return f"[{min_num}, {max_num}]"
        if pd.notna(min_num):
            return f">= {min_num}"
        if pd.notna(max_num):
            return f"<= {max_num}"
        return ""

    @classmethod
    def _decision_explanation(cls, go_no_go_df: pd.DataFrame) -> Dict[str, str]:
        failed = go_no_go_df.loc[
            (go_no_go_df["metric"].astype(str) != "final_recommendation")
            & (~go_no_go_df["passed"].fillna(False).astype(bool))
        ].copy()
        if len(failed) == 0:
            return {
                "hold_type": "all_checks_passed",
                "final_recommendation_explanation": "All go/no-go criteria passed.",
            }
        failed_metrics = failed["metric"].astype(str).tolist()
        failed_groups = {cls._criterion_group(metric) for metric in failed_metrics}
        if failed_groups == {"residual failure-composition"}:
            return {
                "hold_type": "composition_rule_only_hold",
                "final_recommendation_explanation": (
                    "Final recommendation remains hold because only residual failure-composition criteria failed: "
                    f"{'|'.join(failed_metrics)}. Performance/safety criteria passed."
                ),
            }
        if "performance/safety" in failed_groups:
            return {
                "hold_type": "performance_or_safety_hold",
                "final_recommendation_explanation": (
                    "Final recommendation remains hold because one or more performance/safety criteria failed: "
                    f"{'|'.join(failed_metrics)}."
                ),
            }
        return {
            "hold_type": "mixed_or_other_hold",
            "final_recommendation_explanation": (
                "Final recommendation remains hold because the following criteria failed: "
                f"{'|'.join(failed_metrics)}."
            ),
        }

    def _build_decision_audit_df(self, go_no_go_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        criteria = go_no_go_df.loc[go_no_go_df["metric"].astype(str) != "final_recommendation"].copy()
        for _, row in criteria.iterrows():
            metric = str(row["metric"])
            rows.append(
                {
                    "section": "criterion",
                    "criterion_group": self._criterion_group(metric),
                    "criterion_name": metric,
                    "observed_value": row["observed_value"],
                    "threshold": self._threshold_text(row["min_allowed"], row["max_allowed"]),
                    "min_allowed": row["min_allowed"],
                    "max_allowed": row["max_allowed"],
                    "passed": bool(row["passed"]),
                    "explanation": "",
                }
            )
        explanation = self._decision_explanation(go_no_go_df)
        final_row = go_no_go_df.loc[go_no_go_df["metric"].astype(str) == "final_recommendation"].iloc[0]
        rows.append(
            {
                "section": "decision",
                "criterion_group": "decision",
                "criterion_name": "final_recommendation",
                "observed_value": final_row["observed_value"],
                "threshold": "all criteria must pass",
                "min_allowed": "",
                "max_allowed": "",
                "passed": bool(final_row["passed"]),
                "explanation": explanation["final_recommendation_explanation"],
            }
        )
        rows.append(
            {
                "section": "decision",
                "criterion_group": "decision",
                "criterion_name": "hold_type",
                "observed_value": explanation["hold_type"],
                "threshold": "",
                "min_allowed": "",
                "max_allowed": "",
                "passed": bool(final_row["passed"]),
                "explanation": explanation["final_recommendation_explanation"],
            }
        )
        return pd.DataFrame(rows)

    def _write_decision_audit_report(self, audit_df: pd.DataFrame, csv_path: Path, txt_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(csv_path, index=False)
        lines: List[str] = []
        decision_rows = audit_df.loc[audit_df["section"].astype(str).eq("decision")]
        if len(decision_rows) > 0:
            final = decision_rows.loc[decision_rows["criterion_name"].astype(str).eq("final_recommendation")]
            if len(final) > 0:
                lines.append(f"final_recommendation: {final.iloc[0]['observed_value']}")
            hold_type = decision_rows.loc[decision_rows["criterion_name"].astype(str).eq("hold_type")]
            if len(hold_type) > 0:
                lines.append(f"hold_type: {hold_type.iloc[0]['observed_value']}")
                lines.append(f"explanation: {hold_type.iloc[0]['explanation']}")
        lines.append("criteria:")
        criteria = audit_df.loc[audit_df["section"].astype(str).eq("criterion")].copy()
        for _, row in criteria.iterrows():
            lines.append(
                f"- [{row['criterion_group']}] {row['criterion_name']}: "
                f"observed={row['observed_value']} threshold={row['threshold']} passed={row['passed']}"
            )
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_next_limited_broader_validation_plan(self, txt_path: Path) -> str:
        run_out_dir = str(self.DEFAULT_NEXT_BROADER_OUT_DIR)
        runner_cmd = (
            ".\\.venv\\Scripts\\python.exe main.py k2_cached_failed_broader_downstream "
            f"--operating-mode {K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME} "
            f"--out-dir \"{run_out_dir}\""
        )
        report_cmd = (
            ".\\.venv\\Scripts\\python.exe main.py k2_detector_quality_gated_broader_cached_failed_downstream_report "
            f"--out-dir \"{run_out_dir}\" "
            f"--best-csv \"{run_out_dir}\\period_shortlist_best.csv\" "
            f"--quarantine-csv \"{run_out_dir}\\period_shortlist_quarantine.csv\" "
            f"--funnel-csv \"{run_out_dir}\\epic_funnel_reasons.csv\""
        )
        analysis_cmd = (
            ".\\.venv\\Scripts\\python.exe main.py k2_detector_quality_gated_broader_post_rescue_failure_analysis "
            f"--out-dir \"{run_out_dir}\" "
            f"--quarantined-winners-csv \"{run_out_dir}\\detector_quality_gated_broader_quarantined_winners.csv\""
        )
        lines = [
            "next_step: manual follow-up only if explicitly requested",
            f"operating_mode: {K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME}",
            "policy_status: experimental_flag_only",
            "default_global_policy_changed: false",
            "supported_experimental_policy: false",
            "automatic_scale_up_scheduled: false",
            "commands:",
            f"1. {runner_cmd}",
            f"2. {report_cmd}",
            f"3. {analysis_cmd}",
        ]
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return runner_cmd

    @staticmethod
    def _failure_mix_df(analysis_df: pd.DataFrame, label: str) -> pd.DataFrame:
        winners = analysis_df.loc[analysis_df["winner_and_qg_quarantined"].fillna(False).astype(bool)].copy()
        if len(winners) == 0:
            return pd.DataFrame(
                [{"run_label": label, "failure_category": "", "count": 0, "share": 0.0, "weighted_share": 0.0}]
            )
        total = int(len(winners))
        total_weight = float(pd.to_numeric(winners["sample_weight"], errors="coerce").fillna(0.0).sum())
        rows: List[Dict[str, Any]] = []
        counts = winners["quality_gated_failure_category_norm"].fillna("").astype(str).value_counts()
        for failure_category, count in counts.items():
            sub = winners.loc[winners["quality_gated_failure_category_norm"].fillna("").astype(str).eq(str(failure_category))]
            weighted = float(pd.to_numeric(sub["sample_weight"], errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "run_label": label,
                    "failure_category": str(failure_category),
                    "count": int(count),
                    "share": float(count / max(1, total)),
                    "weighted_share": float(weighted / total_weight) if total_weight > 0 else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _comparison_outcome_label(row: pd.Series) -> str:
        baseline = str(row.get("baseline_qg_terminal_group", "") or "")
        experiment = str(row.get("experiment_qg_terminal_group", "") or "")
        if baseline == experiment:
            if experiment == "shortlisted":
                return "stayed_best"
            if experiment == "failed_downstream":
                return "stayed_quarantine"
            return "stayed_other"
        return f"{baseline or 'missing'}_to_{experiment or 'missing'}"

    def _collect_metrics(
        self,
        *,
        analysis_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        go_no_go_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "winners_total": int(analysis_df["detector_winner"].fillna(False).astype(bool).sum()),
            "winners_in_best": int(analysis_df["winner_and_qg_shortlisted"].fillna(False).astype(bool).sum()),
            "winners_in_quarantine": int(analysis_df["winner_and_qg_quarantined"].fillna(False).astype(bool).sum()),
            "downstream_conversion_rate": self._metric_estimate(summary_df, "downstream_conversion_rate"),
            "quarantine_to_best_ratio": self._metric_estimate(summary_df, "quarantine_to_best_ratio"),
            "final_recommendation": self._final_recommendation(go_no_go_df),
            "failed_checks": self._metric_failed_checks(go_no_go_df),
        }

    def _write_go_no_go_report(self, report_df: pd.DataFrame, csv_path: Path, txt_path: Path) -> None:
        report_df.to_csv(csv_path, index=False)
        final_row = report_df.loc[report_df["metric"].astype(str) == "final_recommendation"].iloc[0]
        failed = report_df.loc[(report_df["metric"].astype(str) != "final_recommendation") & (~report_df["passed"].astype(bool))]
        lines = [f"recommendation: {final_row['observed_value']}"]
        if len(failed) == 0:
            lines.append("all go/no-go checks passed")
        else:
            lines.append("failed checks:")
            for _, row in failed.iterrows():
                lines.append(f"- {row['metric']}: observed={row['observed_value']} min={row['min_allowed']} max={row['max_allowed']}")
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(
        self,
        *,
        out_dir: Path,
        stage_dir: Path,
        detector_default_run_dir: Path,
        detector_quality_gated_run_dir: Path,
        max_workers: int,
        cache_only: bool,
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir).resolve()
        stage_dir = Path(stage_dir).resolve()
        detector_default_run_dir = Path(detector_default_run_dir).resolve()
        detector_quality_gated_run_dir = Path(detector_quality_gated_run_dir).resolve()

        manifest_csv = stage_dir / "sampled_epic_manifest.csv"
        paired_detector_csv = stage_dir / "paired_detector_comparison.csv"
        default_downstream_dir = stage_dir / "default_downstream"
        baseline_qg_downstream_dir = stage_dir / "quality_gated_downstream"
        experiment_qg_downstream_dir = stage_dir / self.DEFAULT_EXPERIMENT_QG_DIR_NAME
        experiment_analysis_csv = stage_dir / self.DEFAULT_EXPERIMENT_ANALYSIS_CSV_NAME
        comparison_csv = stage_dir / self.DEFAULT_COMPARISON_CSV_NAME
        summary_csv = stage_dir / self.DEFAULT_SUMMARY_CSV_NAME
        go_no_go_csv = stage_dir / self.DEFAULT_GO_NO_GO_CSV_NAME
        go_no_go_txt = stage_dir / self.DEFAULT_GO_NO_GO_TXT_NAME
        decision_audit_csv = stage_dir / self.DEFAULT_DECISION_AUDIT_CSV_NAME
        decision_audit_txt = stage_dir / self.DEFAULT_DECISION_AUDIT_TXT_NAME
        broader_plan_txt = stage_dir / self.DEFAULT_NEXT_BROADER_PLAN_TXT_NAME

        manifest_df = self._read_required_csv(manifest_csv)
        paired_detector_df = self._read_required_csv(paired_detector_csv)

        detector_default_batch_csv = detector_default_run_dir / "batch_results.csv"
        detector_quality_gated_batch_csv = detector_quality_gated_run_dir / "batch_results.csv"
        if not detector_default_batch_csv.exists():
            raise FileNotFoundError(f"Missing detector default batch csv: {detector_default_batch_csv}")
        if not detector_quality_gated_batch_csv.exists():
            raise FileNotFoundError(f"Missing detector quality-gated batch csv: {detector_quality_gated_batch_csv}")

        reuse_existing_experiment = False
        experiment_diagnostics_csv = experiment_qg_downstream_dir / "period_shortlist_diagnostics.csv"
        if experiment_diagnostics_csv.exists():
            experiment_diagnostics = self._read_required_csv(experiment_diagnostics_csv)
            if len(experiment_diagnostics) > 0:
                reuse_existing_experiment = (
                    str(experiment_diagnostics.iloc[0].get("operating_mode_requested", "") or "").strip()
                    == str(K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME)
                )
        if not reuse_existing_experiment:
            self.scale_validation._run_downstream_stage(
                detector_run_dir=detector_quality_gated_run_dir,
                detector_batch_csv=detector_quality_gated_batch_csv,
                stage_out_dir=experiment_qg_downstream_dir,
                operating_mode=str(K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME),
                disable_validation=False,
                cache_only=bool(cache_only),
                max_workers=int(max_workers),
            )

        baseline_analysis = self.scale_validation._build_downstream_pairwise_df(
            manifest_df=manifest_df,
            detector_pairwise_df=paired_detector_df,
            default_downstream_run_dir=default_downstream_dir,
            quality_gated_downstream_run_dir=baseline_qg_downstream_dir,
        )
        experiment_analysis = self.scale_validation._build_downstream_pairwise_df(
            manifest_df=manifest_df,
            detector_pairwise_df=paired_detector_df,
            default_downstream_run_dir=default_downstream_dir,
            quality_gated_downstream_run_dir=experiment_qg_downstream_dir,
        )
        experiment_analysis.to_csv(experiment_analysis_csv, index=False)

        baseline_summary = self.scale_validation._build_downstream_summary_df(analysis_df=baseline_analysis.copy())
        baseline_go_no_go = self.scale_validation._build_go_no_go_report_df(
            analysis_df=baseline_analysis.copy(),
            summary_df=baseline_summary.copy(),
        )
        experiment_summary = self.scale_validation._build_downstream_summary_df(analysis_df=experiment_analysis.copy())
        experiment_go_no_go = self.scale_validation._build_go_no_go_report_df(
            analysis_df=experiment_analysis.copy(),
            summary_df=experiment_summary.copy(),
        )
        self._write_go_no_go_report(experiment_go_no_go, go_no_go_csv, go_no_go_txt)
        decision_audit_df = self._build_decision_audit_df(experiment_go_no_go)
        self._write_decision_audit_report(decision_audit_df, decision_audit_csv, decision_audit_txt)
        broader_validation_command = self._write_next_limited_broader_validation_plan(broader_plan_txt)

        baseline_metrics = self._collect_metrics(
            analysis_df=baseline_analysis,
            summary_df=baseline_summary,
            go_no_go_df=baseline_go_no_go,
        )
        experiment_metrics = self._collect_metrics(
            analysis_df=experiment_analysis,
            summary_df=experiment_summary,
            go_no_go_df=experiment_go_no_go,
        )

        baseline_mix = self._failure_mix_df(baseline_analysis, "baseline")
        experiment_mix = self._failure_mix_df(experiment_analysis, "experiment")
        failure_mix_comparison = baseline_mix.merge(
            experiment_mix,
            how="outer",
            on="failure_category",
            suffixes=("_baseline", "_experiment"),
        ).fillna({"count_baseline": 0, "count_experiment": 0, "share_baseline": 0.0, "share_experiment": 0.0, "weighted_share_baseline": 0.0, "weighted_share_experiment": 0.0})
        failure_mix_comparison["delta_count"] = (
            pd.to_numeric(failure_mix_comparison["count_experiment"], errors="coerce").fillna(0).astype(int)
            - pd.to_numeric(failure_mix_comparison["count_baseline"], errors="coerce").fillna(0).astype(int)
        )
        failure_mix_comparison["delta_share"] = (
            pd.to_numeric(failure_mix_comparison["share_experiment"], errors="coerce").fillna(0.0)
            - pd.to_numeric(failure_mix_comparison["share_baseline"], errors="coerce").fillna(0.0)
        )

        comparison = baseline_analysis[[
            "epic_id_canonical",
            "query",
            "detector_winner",
            "sample_weight",
            "quality_gated_terminal_group",
            "quality_gated_downstream_outcome",
            "quality_gated_downstream_outcome_score",
            "quality_gated_failure_category_norm",
            "quality_gated_failure_detail",
            "winner_and_qg_shortlisted",
            "winner_and_qg_quarantined",
            "downstream_improved_vs_default",
            "downstream_regressed_vs_default",
        ]].rename(
            columns={
                "quality_gated_terminal_group": "baseline_qg_terminal_group",
                "quality_gated_downstream_outcome": "baseline_qg_downstream_outcome",
                "quality_gated_downstream_outcome_score": "baseline_qg_downstream_outcome_score",
                "quality_gated_failure_category_norm": "baseline_qg_failure_category_norm",
                "quality_gated_failure_detail": "baseline_qg_failure_detail",
                "winner_and_qg_shortlisted": "baseline_winner_and_qg_shortlisted",
                "winner_and_qg_quarantined": "baseline_winner_and_qg_quarantined",
                "downstream_improved_vs_default": "baseline_downstream_improved_vs_default",
                "downstream_regressed_vs_default": "baseline_downstream_regressed_vs_default",
            }
        ).merge(
            experiment_analysis[[
                "epic_id_canonical",
                "quality_gated_terminal_group",
                "quality_gated_downstream_outcome",
                "quality_gated_downstream_outcome_score",
                "quality_gated_failure_category_norm",
                "quality_gated_failure_detail",
                "winner_and_qg_shortlisted",
                "winner_and_qg_quarantined",
                "downstream_improved_vs_default",
                "downstream_regressed_vs_default",
            ]].rename(
                columns={
                    "quality_gated_terminal_group": "experiment_qg_terminal_group",
                    "quality_gated_downstream_outcome": "experiment_qg_downstream_outcome",
                    "quality_gated_downstream_outcome_score": "experiment_qg_downstream_outcome_score",
                    "quality_gated_failure_category_norm": "experiment_qg_failure_category_norm",
                    "quality_gated_failure_detail": "experiment_qg_failure_detail",
                    "winner_and_qg_shortlisted": "experiment_winner_and_qg_shortlisted",
                    "winner_and_qg_quarantined": "experiment_winner_and_qg_quarantined",
                    "downstream_improved_vs_default": "experiment_downstream_improved_vs_default",
                    "downstream_regressed_vs_default": "experiment_downstream_regressed_vs_default",
                }
            ),
            how="left",
            on="epic_id_canonical",
        )
        comparison["paired_experiment_vs_baseline_score_delta"] = (
            pd.to_numeric(comparison["experiment_qg_downstream_outcome_score"], errors="coerce").fillna(0.0)
            - pd.to_numeric(comparison["baseline_qg_downstream_outcome_score"], errors="coerce").fillna(0.0)
        )
        comparison["paired_regression_vs_baseline"] = comparison["paired_experiment_vs_baseline_score_delta"] < 0.0
        comparison["paired_gain_vs_baseline"] = comparison["paired_experiment_vs_baseline_score_delta"] > 0.0
        comparison["paired_movement_label"] = comparison.apply(self._comparison_outcome_label, axis=1)
        comparison.to_csv(comparison_csv, index=False)
        winner_comparison = comparison.loc[comparison["detector_winner"].fillna(False).astype(bool)].copy()
        paired_gain_cases = int(winner_comparison["paired_gain_vs_baseline"].fillna(False).astype(bool).sum())
        paired_regression_cases = int(winner_comparison["paired_regression_vs_baseline"].fillna(False).astype(bool).sum())
        harmful_regression_cases = int(
            (
                winner_comparison["baseline_winner_and_qg_shortlisted"].fillna(False).astype(bool)
                & (~winner_comparison["experiment_winner_and_qg_shortlisted"].fillna(False).astype(bool))
            ).sum()
        )

        summary_rows: List[Dict[str, Any]] = []
        for metric_name in [
            "winners_total",
            "winners_in_best",
            "winners_in_quarantine",
            "downstream_conversion_rate",
            "quarantine_to_best_ratio",
        ]:
            base_value = baseline_metrics[metric_name]
            exp_value = experiment_metrics[metric_name]
            delta = float(exp_value - base_value) if isinstance(base_value, (int, float)) and isinstance(exp_value, (int, float)) else ""
            summary_rows.append(
                {
                    "section": "performance_safety_outcomes",
                    "metric": metric_name,
                    "baseline_value": base_value,
                    "experiment_value": exp_value,
                    "delta_value": delta,
                    "notes": "",
                }
            )

        summary_rows.extend(
            [
                {
                    "section": "decision_recommendation",
                    "metric": "baseline_final_recommendation",
                    "baseline_value": baseline_metrics["final_recommendation"],
                    "experiment_value": experiment_metrics["final_recommendation"],
                    "delta_value": "",
                    "notes": f"baseline_failed_checks={baseline_metrics['failed_checks']}",
                },
                {
                    "section": "decision_recommendation",
                    "metric": "experiment_final_recommendation",
                    "baseline_value": baseline_metrics["final_recommendation"],
                    "experiment_value": experiment_metrics["final_recommendation"],
                    "delta_value": "",
                    "notes": f"experiment_failed_checks={experiment_metrics['failed_checks']}",
                },
                {
                    "section": "performance_safety_outcomes",
                    "metric": "paired_gain_vs_baseline_cases",
                    "baseline_value": 0,
                    "experiment_value": paired_gain_cases,
                    "delta_value": paired_gain_cases,
                    "notes": "",
                },
                {
                    "section": "performance_safety_outcomes",
                    "metric": "paired_regression_vs_baseline_cases",
                    "baseline_value": 0,
                    "experiment_value": paired_regression_cases,
                    "delta_value": paired_regression_cases,
                    "notes": "Counts detector winners with lower experimental downstream outcome score than baseline.",
                },
                {
                    "section": "performance_safety_outcomes",
                    "metric": "harmful_best_to_not_best_regression_cases",
                    "baseline_value": 0,
                    "experiment_value": harmful_regression_cases,
                    "delta_value": harmful_regression_cases,
                    "notes": "",
                },
            ]
        )

        for metric_name in ["known_failure_mix_share", "max_new_failure_share"]:
            summary_rows.append(
                {
                    "section": "residual_failure_composition_outcomes",
                    "metric": metric_name,
                    "baseline_value": self._report_metric_value(baseline_go_no_go, metric_name),
                    "experiment_value": self._report_metric_value(experiment_go_no_go, metric_name),
                    "delta_value": self._report_metric_value(experiment_go_no_go, metric_name)
                    - self._report_metric_value(baseline_go_no_go, metric_name),
                    "notes": "",
                }
            )

        for _, row in failure_mix_comparison.sort_values("failure_category", kind="mergesort").iterrows():
            summary_rows.append(
                {
                    "section": "residual_failure_composition_outcomes",
                    "metric": str(row["failure_category"]),
                    "baseline_value": int(pd.to_numeric(pd.Series([row["count_baseline"]]), errors="coerce").fillna(0).iloc[0]),
                    "experiment_value": int(pd.to_numeric(pd.Series([row["count_experiment"]]), errors="coerce").fillna(0).iloc[0]),
                    "delta_value": int(pd.to_numeric(pd.Series([row["delta_count"]]), errors="coerce").fillna(0).iloc[0]),
                    "notes": (
                        f"baseline_share={float(row['share_baseline']):.6f}; "
                        f"experiment_share={float(row['share_experiment']):.6f}; "
                        f"delta_share={float(row['delta_share']):.6f}"
                    ),
                }
            )
        decision_rows = decision_audit_df.loc[decision_audit_df["section"].astype(str).eq("decision")].copy()
        for _, row in decision_rows.iterrows():
            summary_rows.append(
                {
                    "section": "decision_recommendation",
                    "metric": str(row["criterion_name"]),
                    "baseline_value": "",
                    "experiment_value": row["observed_value"],
                    "delta_value": "",
                    "notes": row["explanation"],
                }
            )
        summary_rows.append(
            {
                "section": "next_limited_broader_validation",
                "metric": "recommended_command",
                "baseline_value": "",
                "experiment_value": broader_validation_command,
                "delta_value": "",
                "notes": (
                    "Experimental conditional MCC=2 carve-out only; default global policy remains unchanged; "
                    "no further scale-up is scheduled by default."
                ),
            }
        )
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

        movement_counts = comparison.loc[
            comparison["detector_winner"].fillna(False).astype(bool), "paired_movement_label"
        ].value_counts()
        return {
            "experiment_analysis_csv": experiment_analysis_csv,
            "comparison_csv": comparison_csv,
            "summary_csv": summary_csv,
            "go_no_go_csv": go_no_go_csv,
            "go_no_go_txt": go_no_go_txt,
            "decision_audit_csv": decision_audit_csv,
            "decision_audit_txt": decision_audit_txt,
            "broader_validation_plan_txt": broader_plan_txt,
            "baseline_metrics": baseline_metrics,
            "experiment_metrics": experiment_metrics,
            "paired_regression_cases": paired_regression_cases,
            "paired_gain_cases": paired_gain_cases,
            "harmful_regression_cases": harmful_regression_cases,
            "decision_audit_rows": decision_audit_df.to_dict(orient="records"),
            "movement_counts": {str(k): int(v) for k, v in movement_counts.items()},
            "failure_mix_comparison": failure_mix_comparison.to_dict(orient="records"),
        }
