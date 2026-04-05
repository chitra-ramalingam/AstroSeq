from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidation import (
    K2DetectorQualityGatedScaleValidation,
)


class K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_quality_gated_scale_validation")
    DEFAULT_STAGE_DIR_NAME = "stage_n600"
    DEFAULT_ANALYSIS_CSV_NAME = "stage_n600_post_hold_quarantined_winners_analysis.csv"
    DEFAULT_ROLLUP_CSV_NAME = "stage_n600_post_hold_quarantined_winners_rollup.csv"

    BUCKET_TRUE_INSUFFICIENT_SIGNAL = "true insufficient signal"
    BUCKET_HISTOGRAM = "histogram construction / handling"
    BUCKET_CLUSTER_POLICY = "cluster / period policy"
    BUCKET_CANDIDATE_FILTER = "candidate filter policy"
    BUCKET_SOMETHING_ELSE = "something else"
    LEVER_LIMIT_TO_CACHED_FAILED = "keep quality-gated limited to cached-failed setting"

    def __init__(self) -> None:
        self.scale_validation = K2DetectorQualityGatedScaleValidation()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Analyze the stage_n600 scale-validation hold result using only existing CSV outputs "
                "and summarize downstream bottlenecks among quarantined detector winners."
            )
        )
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--stage-dir", type=Path, default=None)
        p.add_argument("--manifest-csv", type=Path, default=None)
        p.add_argument("--paired-detector-csv", type=Path, default=None)
        p.add_argument("--default-downstream-dir", type=Path, default=None)
        p.add_argument("--quality-gated-downstream-dir", type=Path, default=None)
        p.add_argument("--analysis-csv", type=Path, default=None)
        p.add_argument("--rollup-csv", type=Path, default=None)
        p.add_argument("--examples-per-group", type=int, default=3)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        stage_dir = Path(args.stage_dir) if args.stage_dir is not None else out_dir / cls.DEFAULT_STAGE_DIR_NAME
        return cls().run(
            manifest_csv=Path(args.manifest_csv) if args.manifest_csv is not None else stage_dir / "sampled_epic_manifest.csv",
            paired_detector_csv=Path(args.paired_detector_csv)
            if args.paired_detector_csv is not None
            else stage_dir / "paired_detector_comparison.csv",
            default_downstream_dir=Path(args.default_downstream_dir)
            if args.default_downstream_dir is not None
            else stage_dir / "default_downstream",
            quality_gated_downstream_dir=Path(args.quality_gated_downstream_dir)
            if args.quality_gated_downstream_dir is not None
            else stage_dir / "quality_gated_downstream",
            analysis_csv=Path(args.analysis_csv)
            if args.analysis_csv is not None
            else out_dir / cls.DEFAULT_ANALYSIS_CSV_NAME,
            rollup_csv=Path(args.rollup_csv)
            if args.rollup_csv is not None
            else out_dir / cls.DEFAULT_ROLLUP_CSV_NAME,
            examples_per_group=int(args.examples_per_group),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _first_nonempty_text(*values: Any) -> str:
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text != "" and text.lower() != "nan":
                return text
        return ""

    @staticmethod
    def _first_numeric(*values: Any) -> float:
        for value in values:
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(num):
                return float(num)
        return float("nan")

    @staticmethod
    def _normalize_key_series(values: pd.Series) -> pd.Series:
        return values.map(K2DetectorQualityGatedScaleValidation._canonical_epic)

    @staticmethod
    def _example_epics(df: pd.DataFrame, limit: int) -> str:
        if len(df) == 0:
            return ""
        work = df.copy()
        for col in ["sample_weight", "delta_n_events", "delta_best_shape_score", "delta_best_depth_snr"]:
            work[col] = pd.to_numeric(work.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        work = work.sort_values(
            by=["sample_weight", "delta_n_events", "delta_best_shape_score", "delta_best_depth_snr", "epic_id_canonical"],
            ascending=[False, False, False, False, True],
        )
        return "|".join(work["epic_id_canonical"].astype(str).head(max(1, int(limit))).tolist())

    def _classify_bottleneck(self, row: pd.Series) -> Dict[str, str]:
        failure_category = str(row.get("failure_category", "") or "").strip().lower()
        shortlist_reason = str(row.get("shortlist_rejection_reason", "") or "").strip().lower()
        terminal_reason = str(row.get("terminal_reason", "") or "").strip().lower()
        failure_detail = str(row.get("failure_detail", "") or "").strip().lower()
        source_reason = str(row.get("source_reason", "") or "").strip().lower()
        stage_reached = str(row.get("stage_reached", "") or "").strip().lower()

        n_after = self._first_numeric(row.get("n_events_after_filters", pd.NA))
        hist_total = self._first_numeric(row.get("hist_total", pd.NA))
        hist_in_range = self._first_numeric(row.get("hist_in_period_range", pd.NA))
        hist_pass_cluster = self._first_numeric(row.get("hist_pass_cluster_count", pd.NA))
        hist_pass_all = self._first_numeric(row.get("hist_pass_all_filters", pd.NA))

        if (
            failure_category == "insufficient_events"
            or shortlist_reason == "insufficient_events"
            or "insufficient_events" in failure_detail
            or "insufficient_events" in source_reason
            or (pd.notna(n_after) and n_after <= 1.0)
        ):
            return {
                "bottleneck_bucket": self.BUCKET_TRUE_INSUFFICIENT_SIGNAL,
                "bucket_rationale": "Too few events survived filtering for robust downstream period inference.",
                "is_fixable_policy_bottleneck": "False",
            }
        if (
            failure_category == "empty_histogram"
            or shortlist_reason == "empty_histogram"
            or "empty_hist" in failure_detail
            or "returned_empty_hist" in failure_detail
            or (pd.notna(n_after) and n_after >= 2.0 and (pd.isna(hist_total) or hist_total <= 0.0))
        ):
            return {
                "bottleneck_bucket": self.BUCKET_HISTOGRAM,
                "bucket_rationale": "Two or more events survived, but histogram construction produced no usable candidate rows.",
                "is_fixable_policy_bottleneck": "True",
            }
        if (
            "below_min_cluster_count" in failure_detail
            or "outside_period_bounds" in failure_detail
            or terminal_reason == "no_cluster_periods"
            or source_reason == "no_cluster_periods"
            or (pd.notna(hist_total) and hist_total > 0.0 and pd.notna(hist_pass_cluster) and hist_pass_cluster <= 0.0)
            or (pd.notna(hist_total) and hist_total > 0.0 and pd.notna(hist_in_range) and hist_in_range <= 0.0)
        ):
            return {
                "bottleneck_bucket": self.BUCKET_CLUSTER_POLICY,
                "bucket_rationale": (
                    "Histogram rows existed, but cluster-count or period-range policy prevented a downstream period "
                    f"candidate from surviving at stage_reached={stage_reached or 'unknown'}."
                ),
                "is_fixable_policy_bottleneck": "True",
            }
        if (
            failure_category == "candidate_filter_rejection"
            or shortlist_reason == "candidate_filter_rejection"
            or (pd.notna(hist_pass_cluster) and hist_pass_cluster > 0.0 and pd.notna(hist_pass_all) and hist_pass_all <= 0.0)
            or "candidate_filter" in failure_detail
            or "candidate_filter" in source_reason
        ):
            return {
                "bottleneck_bucket": self.BUCKET_CANDIDATE_FILTER,
                "bucket_rationale": "A period candidate reached downstream filtering but was rejected by candidate policy.",
                "is_fixable_policy_bottleneck": "True",
            }
        return {
            "bottleneck_bucket": self.BUCKET_SOMETHING_ELSE,
            "bucket_rationale": "Available saved diagnostics do not isolate a single dominant downstream policy bottleneck.",
            "is_fixable_policy_bottleneck": "False",
        }

    def _recommended_next_lever(self, analysis: pd.DataFrame) -> Dict[str, str]:
        bucket_counts = analysis["bottleneck_bucket"].value_counts()
        total = max(1, int(len(analysis)))
        dominant_bucket = str(bucket_counts.index[0]) if len(bucket_counts) > 0 else self.BUCKET_SOMETHING_ELSE
        dominant_count = int(bucket_counts.iloc[0]) if len(bucket_counts) > 0 else 0
        dominant_share = float(dominant_count / total)

        if dominant_bucket in {
            self.BUCKET_CLUSTER_POLICY,
            self.BUCKET_HISTOGRAM,
            self.BUCKET_CANDIDATE_FILTER,
        } and dominant_share >= 0.50:
            return {
                "recommended_next_lever": dominant_bucket,
                "recommendation_rationale": (
                    f"{dominant_bucket} accounts for {dominant_count}/{total} quarantined winners "
                    f"({dominant_share:.1%}), which is concentrated enough to investigate before expanding sample size."
                ),
            }
        return {
            "recommended_next_lever": self.LEVER_LIMIT_TO_CACHED_FAILED,
            "recommendation_rationale": (
                f"No single fixable downstream bottleneck dominates the {total} quarantined winners "
                f"(largest bucket={dominant_bucket} at {dominant_count}/{total}, {dominant_share:.1%})."
            ),
        }

    def build_analysis_df(
        self,
        *,
        manifest_csv: Path,
        paired_detector_csv: Path,
        default_downstream_dir: Path,
        quality_gated_downstream_dir: Path,
    ) -> pd.DataFrame:
        manifest_csv = Path(manifest_csv).resolve()
        paired_detector_csv = Path(paired_detector_csv).resolve()
        default_downstream_dir = Path(default_downstream_dir).resolve()
        quality_gated_downstream_dir = Path(quality_gated_downstream_dir).resolve()

        manifest_df = self._read_required_csv(manifest_csv)
        paired_detector_df = self._read_required_csv(paired_detector_csv)
        downstream_pairwise_df = self.scale_validation._build_downstream_pairwise_df(
            manifest_df=manifest_df,
            detector_pairwise_df=paired_detector_df,
            default_downstream_run_dir=default_downstream_dir,
            quality_gated_downstream_run_dir=quality_gated_downstream_dir,
        )

        winners = downstream_pairwise_df.loc[
            downstream_pairwise_df["detector_winner"].fillna(False).astype(bool)
            & downstream_pairwise_df["quality_gated_failed_downstream"].fillna(False).astype(bool)
        ].copy()

        quarantine_csv = quality_gated_downstream_dir / "period_shortlist_quarantine.csv"
        funnel_csv = quality_gated_downstream_dir / "epic_funnel_reasons.csv"
        diagnostics_csv = quality_gated_downstream_dir / "period_shortlist_diagnostics.csv"

        quarantine = self._read_required_csv(quarantine_csv).copy()
        if len(quarantine) > 0:
            quarantine["epic_id_canonical"] = self._normalize_key_series(quarantine.get("epic_id", pd.Series(dtype=str)))
            quarantine = quarantine.loc[quarantine["epic_id_canonical"].astype(str) != ""].drop_duplicates(
                subset=["epic_id_canonical"], keep="first"
            )
            quarantine = quarantine.rename(columns={c: f"quarantine_{c}" for c in quarantine.columns if c != "epic_id_canonical"})

        funnel = self.scale_validation.downstream_helper.helper._expand_funnel_details(self._read_required_csv(funnel_csv)).copy()
        if len(funnel) > 0:
            funnel["epic_id_canonical"] = self._normalize_key_series(funnel.get("epic_id", pd.Series(dtype=str)))
            funnel = funnel.loc[funnel["epic_id_canonical"].astype(str) != ""].drop_duplicates(
                subset=["epic_id_canonical"], keep="first"
            )
            funnel = funnel.rename(columns={c: f"funnel_{c}" for c in funnel.columns if c != "epic_id_canonical"})

        diagnostics = self._read_required_csv(diagnostics_csv).copy()
        diagnostics_row = diagnostics.iloc[0].to_dict() if len(diagnostics) > 0 else {}

        analysis = winners.copy()
        if len(quarantine) > 0:
            analysis = analysis.merge(quarantine, how="left", on="epic_id_canonical")
        if len(funnel) > 0:
            analysis = analysis.merge(funnel, how="left", on="epic_id_canonical")

        analysis["failure_category"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_failure_category", ""),
                row.get("quarantine_failure_category", ""),
                row.get("funnel_period_failure_category", ""),
                row.get("quality_gated_failure_category_norm", ""),
            ),
            axis=1,
        )
        analysis["shortlist_rejection_reason"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_shortlist_rejection_reason", ""),
                row.get("quarantine_shortlist_rejection_reason", ""),
                row.get("funnel_shortlist_rejection_reason", ""),
            ),
            axis=1,
        )
        analysis["terminal_reason"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_terminal_reason", ""),
                row.get("funnel_terminal_reason", ""),
            ),
            axis=1,
        )
        analysis["failure_detail"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_failure_detail", ""),
                row.get("quarantine_failure_detail", ""),
                row.get("funnel_period_failure_detail", ""),
            ),
            axis=1,
        )
        analysis["source_reason"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_source_reason", ""),
                row.get("quarantine_source_reason", ""),
                row.get("funnel_source_reason", ""),
            ),
            axis=1,
        )
        analysis["stage_reached"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("funnel_stage_reached", ""),
            ),
            axis=1,
        )
        analysis["n_events_after_filters"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_n_events_after_filters", pd.NA),
                row.get("funnel_period_n_events_after_filters", pd.NA),
                row.get("funnel_n_events", pd.NA),
            ),
            axis=1,
        )
        analysis["hist_total"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_hist_total", pd.NA),
                row.get("funnel_period_hist_total", pd.NA),
            ),
            axis=1,
        )
        analysis["hist_in_period_range"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_hist_in_period_range", pd.NA),
                row.get("funnel_period_hist_in_period_range", pd.NA),
            ),
            axis=1,
        )
        analysis["hist_pass_cluster_count"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_hist_pass_cluster_count", pd.NA),
                row.get("funnel_period_hist_pass_cluster_count", pd.NA),
            ),
            axis=1,
        )
        analysis["hist_pass_all_filters"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_hist_pass_all_filters", pd.NA),
                row.get("funnel_period_hist_pass_all_filters", pd.NA),
            ),
            axis=1,
        )
        analysis["min_cluster_count"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("quarantine_min_cluster_count", pd.NA),
                row.get("funnel_period_min_cluster_count", pd.NA),
                diagnostics_row.get("min_cluster_count", pd.NA),
            ),
            axis=1,
        )

        bottleneck_meta = analysis.apply(self._classify_bottleneck, axis=1, result_type="expand")
        analysis["bottleneck_bucket"] = bottleneck_meta["bottleneck_bucket"]
        analysis["bucket_rationale"] = bottleneck_meta["bucket_rationale"]
        analysis["is_fixable_policy_bottleneck"] = bottleneck_meta["is_fixable_policy_bottleneck"]
        analysis = analysis.sort_values(
            by=["bottleneck_bucket", "failure_category", "shortlist_rejection_reason", "epic_id_canonical"],
            kind="mergesort",
        ).reset_index(drop=True)
        return analysis

    def run(
        self,
        *,
        manifest_csv: Path,
        paired_detector_csv: Path,
        default_downstream_dir: Path,
        quality_gated_downstream_dir: Path,
        analysis_csv: Path,
        rollup_csv: Path,
        examples_per_group: int = 3,
    ) -> Dict[str, Any]:
        analysis_csv = Path(analysis_csv).resolve()
        rollup_csv = Path(rollup_csv).resolve()
        analysis = self.build_analysis_df(
            manifest_csv=manifest_csv,
            paired_detector_csv=paired_detector_csv,
            default_downstream_dir=default_downstream_dir,
            quality_gated_downstream_dir=quality_gated_downstream_dir,
        )

        recommendation = self._recommended_next_lever(analysis)
        total = max(1, int(len(analysis)))

        rollup_rows: List[Dict[str, Any]] = [
            {
                "section": "summary",
                "metric": "quarantined_winners_total",
                "count": int(len(analysis)),
                "share": 1.0 if len(analysis) > 0 else 0.0,
                "example_epics": "",
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "recommended_next_lever",
                "count": "",
                "share": "",
                "example_epics": "",
                "notes": recommendation["recommended_next_lever"],
            },
            {
                "section": "summary",
                "metric": "recommendation_rationale",
                "count": "",
                "share": "",
                "example_epics": "",
                "notes": recommendation["recommendation_rationale"],
            },
        ]

        for section_name, column in [
            ("bottleneck_bucket_counts", "bottleneck_bucket"),
            ("failure_category_counts", "failure_category"),
            ("shortlist_rejection_reason_counts", "shortlist_rejection_reason"),
            ("terminal_reason_counts", "terminal_reason"),
        ]:
            counts = analysis[column].fillna("").astype(str).value_counts()
            for metric, count in counts.items():
                subset = analysis.loc[analysis[column].fillna("").astype(str) == str(metric)]
                rollup_rows.append(
                    {
                        "section": section_name,
                        "metric": str(metric),
                        "count": int(count),
                        "share": float(count / total),
                        "example_epics": self._example_epics(subset, examples_per_group),
                        "notes": subset["bucket_rationale"].iloc[0] if (column == "bottleneck_bucket" and len(subset) > 0) else "",
                    }
                )

        analysis_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(analysis_csv, index=False)
        pd.DataFrame(rollup_rows).to_csv(rollup_csv, index=False)

        bucket_counts = {
            bucket: int(analysis["bottleneck_bucket"].eq(bucket).sum())
            for bucket in [
                self.BUCKET_TRUE_INSUFFICIENT_SIGNAL,
                self.BUCKET_HISTOGRAM,
                self.BUCKET_CLUSTER_POLICY,
                self.BUCKET_CANDIDATE_FILTER,
                self.BUCKET_SOMETHING_ELSE,
            ]
        }
        return {
            "analysis_csv": analysis_csv,
            "rollup_csv": rollup_csv,
            "quarantined_winners_total": int(len(analysis)),
            "bucket_counts": bucket_counts,
            "recommended_next_lever": recommendation["recommended_next_lever"],
            "recommendation_rationale": recommendation["recommendation_rationale"],
        }
