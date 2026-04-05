from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis import (
    K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis,
)


class K2DetectorQualityGatedScaleValidationClusterPolicyAnalysis:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_quality_gated_scale_validation")
    DEFAULT_STAGE_DIR_NAME = "stage_n600"
    DEFAULT_ANALYSIS_CSV_NAME = "stage_n600_cluster_policy_quarantined_winners_analysis.csv"
    DEFAULT_ROLLUP_CSV_NAME = "stage_n600_cluster_policy_quarantined_winners_rollup.csv"

    DOMINANT_GATE_MIN_CLUSTER_COUNT = "minimum cluster count"
    DOMINANT_GATE_CLUSTER_SUPPORT = "cluster support requirements"
    DOMINANT_GATE_PERIOD_BOUNDS = "period bounds / inference range"
    DOMINANT_GATE_VALIDATION = "period validation threshold"
    DOMINANT_GATE_SHORTLIST_POLICY = "shortlist candidate filter policy"
    DOMINANT_GATE_OTHER = "another clearly supported gate"

    BUCKET_SUPPORTED_MCC2 = "supported MCC=2 carve-out candidate"
    BUCKET_SINGLE_CANDIDATE = "single-candidate near-miss"
    BUCKET_THREE_EVENT = "three-event borderline"
    BUCKET_TWO_EVENT = "two-event low-support"

    SAFE_CHANGE_CONDITIONAL_MCC2 = (
        "test MIN_CLUSTER_COUNT=2 only when n_events_after_filters>=4 and hist_in_period_range>=2, "
        "while keeping existing cluster_count==2 manual-review and guardrail thresholds unchanged"
    )

    def __init__(self) -> None:
        self.post_hold = K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Analyze only the cluster / period policy quarantined winners from the stage_n600 "
                "scale-validation hold result using saved CSV outputs."
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
        return K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis._read_required_csv(Path(path))

    @staticmethod
    def _example_epics(df: pd.DataFrame, limit: int) -> str:
        return K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis._example_epics(df, limit)

    @staticmethod
    def _first_nonempty_text(*values: Any) -> str:
        return K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis._first_nonempty_text(*values)

    @staticmethod
    def _first_numeric(*values: Any) -> float:
        return K2DetectorQualityGatedScaleValidationPostHoldFailureAnalysis._first_numeric(*values)

    def _classify_dominant_gate(self, row: pd.Series) -> Dict[str, str]:
        failure_detail = str(row.get("failure_detail", "") or "").strip().lower()
        shortlist_reason = str(row.get("shortlist_rejection_reason", "") or "").strip().lower()
        terminal_reason = str(row.get("terminal_reason", "") or "").strip().lower()
        n_periods_proposed = self._first_numeric(row.get("n_periods_proposed", pd.NA), row.get("funnel_n_periods_proposed", pd.NA))
        n_periods_validated = self._first_numeric(row.get("n_periods_validated", pd.NA), row.get("funnel_n_periods_validated", pd.NA))
        hist_in_range = self._first_numeric(row.get("hist_in_period_range", pd.NA))
        hist_pass_cluster = self._first_numeric(row.get("hist_pass_cluster_count", pd.NA))

        if (
            "below_min_cluster_count" in failure_detail
            and terminal_reason == "no_cluster_periods"
            and (pd.isna(n_periods_proposed) or n_periods_proposed <= 0.0)
            and (pd.isna(n_periods_validated) or n_periods_validated <= 0.0)
        ):
            return {
                "dominant_gate": self.DOMINANT_GATE_MIN_CLUSTER_COUNT,
                "gate_rationale": (
                    "Candidate periods existed in the period-inference histogram, but none reached "
                    f"MIN_CLUSTER_COUNT={int(self._first_numeric(row.get('min_cluster_count', 3.0), 3.0))}."
                ),
            }
        if (
            terminal_reason == "no_cluster_periods"
            and pd.notna(hist_in_range)
            and hist_in_range > 0.0
            and pd.notna(hist_pass_cluster)
            and hist_pass_cluster <= 0.0
        ):
            return {
                "dominant_gate": self.DOMINANT_GATE_CLUSTER_SUPPORT,
                "gate_rationale": "In-range candidate periods existed, but cluster support stayed below the active threshold.",
            }
        if terminal_reason == "no_cluster_periods" and pd.notna(hist_in_range) and hist_in_range <= 0.0:
            return {
                "dominant_gate": self.DOMINANT_GATE_PERIOD_BOUNDS,
                "gate_rationale": "Histogram rows existed but none survived the active period-range bounds.",
            }
        if pd.notna(n_periods_proposed) and n_periods_proposed > 0.0 and (pd.isna(n_periods_validated) or n_periods_validated <= 0.0):
            return {
                "dominant_gate": self.DOMINANT_GATE_VALIDATION,
                "gate_rationale": "Candidate periods were proposed but all failed later validation thresholds.",
            }
        if shortlist_reason == "candidate_filter_rejection":
            return {
                "dominant_gate": self.DOMINANT_GATE_SHORTLIST_POLICY,
                "gate_rationale": "The saved shortlist rejection reason points to a downstream candidate policy filter.",
            }
        return {
            "dominant_gate": self.DOMINANT_GATE_OTHER,
            "gate_rationale": "Saved fields do not isolate a narrower downstream policy gate with confidence.",
        }

    def _classify_recoverability_bucket(self, row: pd.Series) -> Dict[str, str]:
        n_events_after = self._first_numeric(row.get("n_events_after_filters", pd.NA))
        hist_in_range = self._first_numeric(row.get("hist_in_period_range", pd.NA))

        if pd.notna(n_events_after) and n_events_after >= 4.0 and pd.notna(hist_in_range) and hist_in_range >= 2.0:
            return {
                "recoverability_bucket": self.BUCKET_SUPPORTED_MCC2,
                "recoverability_rationale": (
                    "At least four filtered events and at least two in-range histogram candidates survived before "
                    "the MCC=3 gate, so a narrow MCC=2 carve-out is the most plausible low-risk recovery path."
                ),
                "recommended_for_smallest_safe_change": "True",
            }
        if pd.notna(n_events_after) and n_events_after >= 4.0:
            return {
                "recoverability_bucket": self.BUCKET_SINGLE_CANDIDATE,
                "recoverability_rationale": (
                    "Filtered-event support is reasonable, but only one in-range candidate period was saved, so "
                    "lowering MCC alone is less reliable."
                ),
                "recommended_for_smallest_safe_change": "False",
            }
        if pd.notna(n_events_after) and n_events_after == 3.0:
            note = "no in-range candidate period remained" if pd.notna(hist_in_range) and hist_in_range <= 0.0 else "support stays borderline even under MCC=2"
            return {
                "recoverability_bucket": self.BUCKET_THREE_EVENT,
                "recoverability_rationale": (
                    "Exactly three filtered events remained; this is a near-miss on support, but "
                    f"{note}."
                ),
                "recommended_for_smallest_safe_change": "False",
            }
        return {
            "recoverability_bucket": self.BUCKET_TWO_EVENT,
            "recoverability_rationale": (
                "Only two filtered events remained, so any MCC=2 rescue would still be low-support and "
                "more inflation-prone."
            ),
            "recommended_for_smallest_safe_change": "False",
        }

    def _recommended_smallest_safe_change(self, analysis: pd.DataFrame) -> Dict[str, str]:
        if len(analysis) == 0:
            return {
                "dominant_gate": self.DOMINANT_GATE_OTHER,
                "recommended_smallest_safe_change": "no cluster-policy cases found",
                "recommendation_rationale": "No cluster / period policy cases were present in the saved analysis set.",
            }

        gate_counts = analysis["dominant_gate"].fillna("").astype(str).value_counts()
        dominant_gate = str(gate_counts.index[0]) if len(gate_counts) > 0 else self.DOMINANT_GATE_OTHER
        dominant_gate_count = int(gate_counts.iloc[0]) if len(gate_counts) > 0 else 0
        total = int(len(analysis))

        safe_bucket_count = int(analysis["recoverability_bucket"].eq(self.BUCKET_SUPPORTED_MCC2).sum())
        safe_bucket_share = float(safe_bucket_count / max(1, total))
        if dominant_gate == self.DOMINANT_GATE_MIN_CLUSTER_COUNT and safe_bucket_count > 0:
            return {
                "dominant_gate": dominant_gate,
                "recommended_smallest_safe_change": self.SAFE_CHANGE_CONDITIONAL_MCC2,
                "recommendation_rationale": (
                    f"{dominant_gate} is the blocker in {dominant_gate_count}/{total} cluster-policy cases, and "
                    f"{safe_bucket_count}/{total} ({safe_bucket_share:.1%}) already have >=4 filtered events and "
                    ">=2 in-range histogram candidates. That makes a conditional MCC=2 carve-out the narrowest "
                    "next policy test that is still materially sized."
                ),
            }
        return {
            "dominant_gate": dominant_gate,
            "recommended_smallest_safe_change": "no single narrow policy change is well supported by the saved fields",
            "recommendation_rationale": (
                f"The saved fields do not show a large enough concentration behind one safely narrow change "
                f"(dominant gate={dominant_gate}, {dominant_gate_count}/{total})."
            ),
        }

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
        quality_gated_downstream_dir = Path(quality_gated_downstream_dir).resolve()

        full_analysis = self.post_hold.build_analysis_df(
            manifest_csv=manifest_csv,
            paired_detector_csv=paired_detector_csv,
            default_downstream_dir=default_downstream_dir,
            quality_gated_downstream_dir=quality_gated_downstream_dir,
        )
        analysis = full_analysis.loc[
            full_analysis["bottleneck_bucket"].fillna("").astype(str) == self.post_hold.BUCKET_CLUSTER_POLICY
        ].copy()

        diagnostics = self._read_required_csv(quality_gated_downstream_dir / "period_shortlist_diagnostics.csv")
        diagnostics_row = diagnostics.iloc[0].to_dict() if len(diagnostics) > 0 else {}
        for key in [
            "operating_mode_requested",
            "mcc_policy_mode",
            "mcc_policy_note",
            "manual_review_cluster_count_eq",
            "cluster2_guardrail_hit_rate_shape_min",
            "cluster2_guardrail_soft_hit_rate_min",
        ]:
            analysis[f"diagnostics_{key}"] = diagnostics_row.get(key, "")

        analysis["n_periods_proposed"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("funnel_n_periods_proposed", pd.NA),
                row.get("quality_gated_n_periods_proposed", pd.NA),
            ),
            axis=1,
        )
        analysis["n_periods_validated"] = analysis.apply(
            lambda row: self._first_numeric(
                row.get("funnel_n_periods_validated", pd.NA),
                row.get("quality_gated_n_periods_validated", pd.NA),
            ),
            axis=1,
        )
        analysis["candidate_periods_generated"] = pd.to_numeric(analysis.get("hist_total"), errors="coerce")
        analysis["candidate_periods_in_range"] = pd.to_numeric(analysis.get("hist_in_period_range"), errors="coerce")
        analysis["candidate_periods_passing_min_cluster_count"] = pd.to_numeric(
            analysis.get("hist_pass_cluster_count"), errors="coerce"
        )
        analysis["candidate_periods_passing_all_filters"] = pd.to_numeric(
            analysis.get("hist_pass_all_filters"), errors="coerce"
        )
        analysis["shortlist_rejection_stage"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quarantine_shortlist_rejection_stage", ""),
                row.get("funnel_shortlist_rejection_stage", ""),
            ),
            axis=1,
        )

        gate_meta = analysis.apply(self._classify_dominant_gate, axis=1, result_type="expand")
        analysis["dominant_gate"] = gate_meta["dominant_gate"]
        analysis["gate_rationale"] = gate_meta["gate_rationale"]

        recoverability_meta = analysis.apply(self._classify_recoverability_bucket, axis=1, result_type="expand")
        analysis["recoverability_bucket"] = recoverability_meta["recoverability_bucket"]
        analysis["recoverability_rationale"] = recoverability_meta["recoverability_rationale"]
        analysis["recommended_for_smallest_safe_change"] = recoverability_meta["recommended_for_smallest_safe_change"]

        recommendation = self._recommended_smallest_safe_change(analysis)
        analysis["recommended_smallest_safe_change"] = recommendation["recommended_smallest_safe_change"]

        analysis = analysis.sort_values(
            by=["recoverability_bucket", "n_events_after_filters", "candidate_periods_in_range", "epic_id_canonical"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        total = max(1, int(len(analysis)))
        rollup_rows: List[Dict[str, Any]] = [
            {
                "section": "summary",
                "metric": "cluster_policy_cases_total",
                "count": int(len(analysis)),
                "share": 1.0 if len(analysis) > 0 else 0.0,
                "example_epics": "",
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "dominant_gate",
                "count": "",
                "share": "",
                "example_epics": "",
                "notes": recommendation["dominant_gate"],
            },
            {
                "section": "summary",
                "metric": "recommended_smallest_safe_change",
                "count": "",
                "share": "",
                "example_epics": "",
                "notes": recommendation["recommended_smallest_safe_change"],
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
            ("recoverability_bucket_counts", "recoverability_bucket"),
            ("dominant_gate_counts", "dominant_gate"),
            ("failure_detail_counts", "failure_detail"),
            ("shortlist_rejection_stage_counts", "shortlist_rejection_stage"),
            ("shortlist_rejection_reason_counts", "shortlist_rejection_reason"),
            ("terminal_reason_counts", "terminal_reason"),
            ("n_events_after_filters_counts", "n_events_after_filters"),
            ("candidate_periods_generated_counts", "candidate_periods_generated"),
            ("candidate_periods_in_range_counts", "candidate_periods_in_range"),
            ("candidate_periods_passing_min_cluster_count_counts", "candidate_periods_passing_min_cluster_count"),
            ("n_periods_proposed_counts", "n_periods_proposed"),
            ("n_periods_validated_counts", "n_periods_validated"),
        ]:
            counts = analysis[column].fillna("").astype(str).value_counts()
            for metric, count in counts.items():
                subset = analysis.loc[analysis[column].fillna("").astype(str) == str(metric)]
                notes = ""
                if column == "recoverability_bucket" and len(subset) > 0:
                    notes = str(subset["recoverability_rationale"].iloc[0])
                elif column == "dominant_gate" and len(subset) > 0:
                    notes = str(subset["gate_rationale"].iloc[0])
                rollup_rows.append(
                    {
                        "section": section_name,
                        "metric": str(metric),
                        "count": int(count),
                        "share": float(count / total),
                        "example_epics": self._example_epics(subset, examples_per_group),
                        "notes": notes,
                    }
                )

        analysis_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(analysis_csv, index=False)
        pd.DataFrame(rollup_rows).to_csv(rollup_csv, index=False)

        bucket_counts = {
            bucket: int(analysis["recoverability_bucket"].eq(bucket).sum())
            for bucket in [
                self.BUCKET_SUPPORTED_MCC2,
                self.BUCKET_SINGLE_CANDIDATE,
                self.BUCKET_THREE_EVENT,
                self.BUCKET_TWO_EVENT,
            ]
        }
        return {
            "analysis_csv": analysis_csv,
            "rollup_csv": rollup_csv,
            "cluster_policy_cases_total": int(len(analysis)),
            "bucket_counts": bucket_counts,
            "dominant_gate": recommendation["dominant_gate"],
            "recommended_smallest_safe_change": recommendation["recommended_smallest_safe_change"],
            "recommendation_rationale": recommendation["recommendation_rationale"],
        }
