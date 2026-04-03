from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2DetectorQualityGatedBroaderPostRescueFailureAnalysis:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream")
    DEFAULT_QUARANTINED_WINNERS_CSV = DEFAULT_OUT_DIR / "detector_quality_gated_broader_quarantined_winners.csv"
    DEFAULT_QUARANTINE_CSV = DEFAULT_OUT_DIR / "Apr1_period_shortlist_quarantine.csv"
    DEFAULT_DIAGNOSTICS_CSV = DEFAULT_OUT_DIR / "Apr1_period_shortlist_diagnostics.csv"
    DEFAULT_FUNNEL_CSV = DEFAULT_OUT_DIR / "Apr1_epic_funnel_reasons.csv"
    DEFAULT_ANALYSIS_CSV_NAME = "detector_quality_gated_broader_post_rescue_failure_analysis.csv"
    DEFAULT_ROLLUP_CSV_NAME = "detector_quality_gated_broader_post_rescue_failure_rollup.csv"

    BUCKET_TRUE_INSUFFICIENT_SIGNAL = "true insufficient signal"
    BUCKET_RECOVERABLE_CLUSTER_PERIOD = "likely recoverable with looser cluster/period policy"
    BUCKET_RECOVERABLE_HISTOGRAM = "likely recoverable with histogram handling changes"
    BUCKET_UNRECOVERABLE_NOISE = "likely unrecoverable / noise"

    LEVER_CLUSTER_POLICY = "cluster policy"
    LEVER_HISTOGRAM = "histogram construction / handling"
    LEVER_CANDIDATE_FILTER = "candidate filter policy"
    LEVER_SOMETHING_ELSE = "something else"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Analyze post-rescue downstream failures using only existing broader cached-failed CSV outputs."
        )
        p.add_argument("--quarantined-winners-csv", type=Path, default=cls.DEFAULT_QUARANTINED_WINNERS_CSV)
        p.add_argument("--quarantine-csv", type=Path, default=cls.DEFAULT_QUARANTINE_CSV)
        p.add_argument("--diagnostics-csv", type=Path, default=cls.DEFAULT_DIAGNOSTICS_CSV)
        p.add_argument("--funnel-csv", type=Path, default=cls.DEFAULT_FUNNEL_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--analysis-csv", type=Path, default=None)
        p.add_argument("--rollup-csv", type=Path, default=None)
        p.add_argument("--examples-per-bucket", type=int, default=3)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            quarantined_winners_csv=Path(args.quarantined_winners_csv),
            quarantine_csv=Path(args.quarantine_csv),
            diagnostics_csv=Path(args.diagnostics_csv),
            funnel_csv=Path(args.funnel_csv),
            analysis_csv=Path(args.analysis_csv) if args.analysis_csv is not None else out_dir / cls.DEFAULT_ANALYSIS_CSV_NAME,
            rollup_csv=Path(args.rollup_csv) if args.rollup_csv is not None else out_dir / cls.DEFAULT_ROLLUP_CSV_NAME,
            examples_per_bucket=int(args.examples_per_bucket),
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
    def _require_columns(df: pd.DataFrame, *, label: str, required_columns: Sequence[str]) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"{label} CSV missing required columns: {', '.join(missing)}")

    def _prepare_table(self, df: pd.DataFrame, *, epic_col: str, label: str, required_columns: Sequence[str]) -> pd.DataFrame:
        self._require_columns(df, label=label, required_columns=required_columns)
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self.helper._canonical_epic)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out

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
    def _diagnostic_value(df: pd.DataFrame, column: str) -> str:
        if len(df) == 0 or column not in df.columns:
            return ""
        value = df.iloc[0][column]
        return "" if pd.isna(value) else str(value)

    @staticmethod
    def _example_epics(df: pd.DataFrame, limit: int) -> str:
        if len(df) == 0:
            return ""
        work = df.copy()
        for col in ["delta_n_events", "delta_best_shape_score", "delta_best_depth_snr"]:
            work[col] = pd.to_numeric(work.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        work = work.sort_values(
            by=["delta_n_events", "delta_best_shape_score", "delta_best_depth_snr", "epic_id"],
            ascending=[False, False, False, True],
        )
        return "|".join(work["epic_id"].astype(str).head(max(1, int(limit))).tolist())

    def _choose_bucket(self, row: pd.Series) -> Dict[str, str]:
        failure = str(row.get("failure_category", "") or "").strip().lower()
        shortlist = str(row.get("shortlist_rejection_reason", "") or "").strip().lower()
        detail = str(row.get("failure_detail", "") or "").strip().lower()
        source = str(row.get("source_reason", "") or "").strip().lower()
        n_after = self._first_numeric(row.get("n_events_after_filters", pd.NA))
        hist_total = self._first_numeric(row.get("hist_total", pd.NA))
        hist_in_range = self._first_numeric(row.get("hist_in_period_range", pd.NA))
        hist_pass_cluster = self._first_numeric(row.get("hist_pass_cluster_count", pd.NA))

        if failure == "insufficient_events" or shortlist == "insufficient_events" or "insufficient_events" in detail or "insufficient_events" in source or (pd.notna(n_after) and n_after <= 1.0):
            return {
                "actionable_bucket": self.BUCKET_TRUE_INSUFFICIENT_SIGNAL,
                "suggested_lever": self.LEVER_SOMETHING_ELSE,
                "bucket_rationale": "Only one event remained after filtering, so period inference lacked enough support.",
            }
        if failure == "empty_histogram" or shortlist == "empty_histogram" or "returned_empty_hist" in detail or "empty_histogram" in detail or (pd.isna(hist_total) and pd.notna(n_after) and n_after >= 2.0):
            return {
                "actionable_bucket": self.BUCKET_RECOVERABLE_HISTOGRAM,
                "suggested_lever": self.LEVER_HISTOGRAM,
                "bucket_rationale": "Two or more events survived, but period inference produced no usable histogram rows.",
            }
        if failure == "candidate_filter_rejection" or shortlist == "candidate_filter_rejection" or "outside_period_bounds" in detail or (pd.notna(hist_pass_cluster) and hist_pass_cluster > 0.0 and pd.notna(hist_in_range) and hist_in_range <= 0.0):
            return {
                "actionable_bucket": self.BUCKET_RECOVERABLE_CLUSTER_PERIOD,
                "suggested_lever": self.LEVER_CANDIDATE_FILTER,
                "bucket_rationale": "A histogram candidate existed, but policy filters removed it before shortlist-best output.",
            }
        return {
            "actionable_bucket": self.BUCKET_UNRECOVERABLE_NOISE,
            "suggested_lever": self.LEVER_SOMETHING_ELSE,
            "bucket_rationale": "Available diagnostics do not show a clear histogram or cluster-policy recovery path.",
        }

    def run(
        self,
        *,
        quarantined_winners_csv: Path,
        quarantine_csv: Path,
        diagnostics_csv: Path,
        funnel_csv: Path,
        analysis_csv: Path,
        rollup_csv: Path,
        examples_per_bucket: int = 3,
    ) -> Dict[str, Any]:
        quarantined_winners_csv = Path(quarantined_winners_csv).resolve()
        quarantine_csv = Path(quarantine_csv).resolve()
        diagnostics_csv = Path(diagnostics_csv).resolve()
        funnel_csv = Path(funnel_csv).resolve()
        analysis_csv = Path(analysis_csv).resolve()
        rollup_csv = Path(rollup_csv).resolve()

        winners = self._prepare_table(
            self._read_required_csv(quarantined_winners_csv),
            epic_col="epic_id",
            label="quarantined_winners",
            required_columns=["epic_id", "failure_category", "shortlist_rejection_reason", "terminal_reason"],
        )
        quarantine = self._prepare_table(
            self._read_required_csv(quarantine_csv),
            epic_col="epic_id",
            label="quarantine",
            required_columns=["epic_id", "failure_category", "failure_detail", "n_events_after_filters"],
        )
        funnel = self.helper._expand_funnel_details(self._read_required_csv(funnel_csv))
        funnel = self._prepare_table(
            funnel,
            epic_col="epic_id",
            label="funnel",
            required_columns=["epic_id", "terminal_reason", "source_reason", "stage_reached"],
        )
        diagnostics = self._read_required_csv(diagnostics_csv)
        self._require_columns(
            diagnostics,
            label="diagnostics",
            required_columns=["min_cluster_count", "operating_mode_requested", "n_quarantined_no_cluster_periods"],
        )

        quarantine = quarantine.rename(columns={c: f"raw_quarantine_{c}" for c in quarantine.columns if c != "epic_id_norm"})
        funnel = funnel.rename(columns={c: f"raw_funnel_{c}" for c in funnel.columns if c != "epic_id_norm"})
        analysis = winners.merge(quarantine, on="epic_id_norm", how="left").merge(funnel, on="epic_id_norm", how="left").copy()

        text_specs = {
            "failure_category": ["failure_category", "quarantine_failure_category", "raw_quarantine_failure_category", "funnel_period_failure_category", "raw_funnel_period_failure_category"],
            "shortlist_rejection_reason": ["shortlist_rejection_reason", "quarantine_shortlist_rejection_reason", "raw_quarantine_shortlist_rejection_reason", "funnel_shortlist_rejection_reason", "raw_funnel_shortlist_rejection_reason"],
            "terminal_reason": ["terminal_reason", "funnel_terminal_reason", "raw_funnel_terminal_reason"],
            "failure_detail": ["quarantine_failure_detail", "raw_quarantine_failure_detail", "funnel_period_failure_detail", "raw_funnel_period_failure_detail"],
            "source_reason": ["quarantine_source_reason", "raw_quarantine_source_reason", "funnel_source_reason", "raw_funnel_source_reason"],
            "stage_reached": ["funnel_stage_reached", "raw_funnel_stage_reached"],
        }
        for out_col, cols in text_specs.items():
            analysis[out_col] = analysis.apply(lambda row, names=cols: self._first_nonempty_text(*[row.get(name, "") for name in names]), axis=1)

        numeric_specs = {
            "n_events_raw": ["quarantine_n_events_raw", "raw_quarantine_n_events_raw", "funnel_period_n_events_raw", "raw_funnel_period_n_events_raw"],
            "n_events_after_filters": ["quarantine_n_events_after_filters", "raw_quarantine_n_events_after_filters", "funnel_period_n_events_after_filters", "raw_funnel_period_n_events_after_filters", "funnel_n_events", "raw_funnel_n_events"],
            "hist_total": ["quarantine_hist_total", "raw_quarantine_hist_total", "funnel_period_hist_total", "raw_funnel_period_hist_total"],
            "hist_finite_period": ["quarantine_hist_finite_period", "raw_quarantine_hist_finite_period", "funnel_period_hist_finite_period", "raw_funnel_period_hist_finite_period"],
            "hist_in_period_range": ["quarantine_hist_in_period_range", "raw_quarantine_hist_in_period_range", "funnel_period_hist_in_period_range", "raw_funnel_period_hist_in_period_range"],
            "hist_pass_cluster_count": ["quarantine_hist_pass_cluster_count", "raw_quarantine_hist_pass_cluster_count", "funnel_period_hist_pass_cluster_count", "raw_funnel_period_hist_pass_cluster_count"],
            "hist_pass_all_filters": ["quarantine_hist_pass_all_filters", "raw_quarantine_hist_pass_all_filters", "funnel_period_hist_pass_all_filters", "raw_funnel_period_hist_pass_all_filters"],
            "n_periods_proposed": ["funnel_n_periods_proposed", "raw_funnel_n_periods_proposed"],
            "n_periods_validated": ["funnel_n_periods_validated", "raw_funnel_n_periods_validated"],
            "min_cluster_count": ["quarantine_min_cluster_count", "raw_quarantine_min_cluster_count", "funnel_period_min_cluster_count", "raw_funnel_period_min_cluster_count"],
            "infer_max_period_days": ["quarantine_infer_max_period_days", "raw_quarantine_infer_max_period_days", "funnel_period_infer_max_period_days", "raw_funnel_period_infer_max_period_days"],
            "infer_min_hits": ["quarantine_infer_min_hits", "raw_quarantine_infer_min_hits", "funnel_period_infer_min_hits", "raw_funnel_period_infer_min_hits"],
            "infer_tol_frac": ["quarantine_infer_tol_frac", "raw_quarantine_infer_tol_frac", "funnel_period_infer_tol_frac", "raw_funnel_period_infer_tol_frac"],
            "min_period_days": ["quarantine_min_period_days", "raw_quarantine_min_period_days", "funnel_period_min_period_days", "raw_funnel_period_min_period_days"],
            "period_cap_days": ["quarantine_period_cap_days", "raw_quarantine_period_cap_days", "funnel_period_cap_days", "raw_funnel_period_cap_days"],
            "top_k_periods": ["quarantine_top_k_periods", "raw_quarantine_top_k_periods", "funnel_period_top_k_periods", "raw_funnel_period_top_k_periods"],
        }
        for out_col, cols in numeric_specs.items():
            analysis[out_col] = analysis.apply(lambda row, names=cols: self._first_numeric(*[row.get(name, pd.NA) for name in names]), axis=1)

        analysis["has_histogram_rows"] = analysis["hist_total"].fillna(0).gt(0)
        analysis["has_cluster_support_in_histogram"] = analysis["hist_pass_cluster_count"].fillna(0).gt(0)
        analysis["has_period_in_range"] = analysis["hist_in_period_range"].fillna(0).gt(0)
        analysis["has_period_after_all_filters"] = analysis["hist_pass_all_filters"].fillna(0).gt(0)

        bucket_meta = analysis.apply(self._choose_bucket, axis=1, result_type="expand")
        analysis["actionable_bucket"] = bucket_meta["actionable_bucket"]
        analysis["suggested_lever"] = bucket_meta["suggested_lever"]
        analysis["bucket_rationale"] = bucket_meta["bucket_rationale"]
        analysis = analysis.sort_values(by=["actionable_bucket", "failure_category", "epic_id"]).reset_index(drop=True)

        min_cluster_count = self._diagnostic_value(diagnostics, "min_cluster_count")
        mode = self._diagnostic_value(diagnostics, "operating_mode_requested")
        no_cluster_total = self._diagnostic_value(diagnostics, "n_quarantined_no_cluster_periods")

        lever_counts = analysis["suggested_lever"].value_counts()
        histogram_count = int(lever_counts.get(self.LEVER_HISTOGRAM, 0))
        candidate_filter_count = int(lever_counts.get(self.LEVER_CANDIDATE_FILTER, 0))
        cluster_policy_count = int(lever_counts.get(self.LEVER_CLUSTER_POLICY, 0))
        recommended_next_lever = self.LEVER_HISTOGRAM
        recommendation_rationale = (
            f"{histogram_count} quarantined winners have 2+ surviving events but empty histogram output, while diagnostics show min_cluster_count={min_cluster_count or 'unknown'}."
        )
        if histogram_count <= 0:
            recommended_next_lever = self.LEVER_CANDIDATE_FILTER if candidate_filter_count >= cluster_policy_count else self.LEVER_CLUSTER_POLICY
            recommendation_rationale = "Histogram-related failures are not the dominant recoverable bucket."

        bucket_order = [
            self.BUCKET_TRUE_INSUFFICIENT_SIGNAL,
            self.BUCKET_RECOVERABLE_CLUSTER_PERIOD,
            self.BUCKET_RECOVERABLE_HISTOGRAM,
            self.BUCKET_UNRECOVERABLE_NOISE,
        ]
        rollup_rows: List[Dict[str, Any]] = [
            {"section": "summary", "metric": "quarantined_winners_total", "value": int(len(analysis)), "example_epics": "", "notes": ""},
            {"section": "summary", "metric": "operating_mode_requested", "value": mode, "example_epics": "", "notes": ""},
            {"section": "summary", "metric": "min_cluster_count", "value": min_cluster_count, "example_epics": "", "notes": ""},
            {"section": "summary", "metric": "n_quarantined_no_cluster_periods_global", "value": no_cluster_total, "example_epics": "", "notes": ""},
        ]
        for val in sorted(x for x in analysis["n_events_after_filters"].dropna().unique().tolist()):
            subset = analysis.loc[analysis["n_events_after_filters"].eq(val)]
            rollup_rows.append({"section": "event_count_profile", "metric": f"n_events_after_filters_eq_{int(val)}", "value": int(len(subset)), "example_epics": self._example_epics(subset, examples_per_bucket), "notes": ""})
        for bucket in bucket_order:
            subset = analysis.loc[analysis["actionable_bucket"] == bucket]
            rollup_rows.append({"section": "bucket_counts", "metric": bucket, "value": int(len(subset)), "example_epics": self._example_epics(subset, examples_per_bucket), "notes": subset["bucket_rationale"].iloc[0] if len(subset) > 0 else ""})
        for lever in [self.LEVER_HISTOGRAM, self.LEVER_CANDIDATE_FILTER, self.LEVER_CLUSTER_POLICY, self.LEVER_SOMETHING_ELSE]:
            subset = analysis.loc[analysis["suggested_lever"] == lever]
            rollup_rows.append({"section": "lever_counts", "metric": lever, "value": int(len(subset)), "example_epics": self._example_epics(subset, examples_per_bucket), "notes": ""})
        for metric, count in analysis["failure_category"].fillna("").astype(str).value_counts().items():
            subset = analysis.loc[analysis["failure_category"].fillna("").astype(str) == str(metric)]
            rollup_rows.append({"section": "failure_category_counts", "metric": str(metric), "value": int(count), "example_epics": self._example_epics(subset, examples_per_bucket), "notes": ""})
        rollup_rows.append({"section": "recommendation", "metric": "recommended_next_lever", "value": recommended_next_lever, "example_epics": self._example_epics(analysis.loc[analysis["suggested_lever"] == recommended_next_lever], examples_per_bucket), "notes": recommendation_rationale})

        analysis_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(analysis_csv, index=False)
        pd.DataFrame(rollup_rows).to_csv(rollup_csv, index=False)

        return {
            "analysis_csv": analysis_csv,
            "rollup_csv": rollup_csv,
            "quarantined_winners_total": int(len(analysis)),
            "bucket_counts": {bucket: int(analysis["actionable_bucket"].eq(bucket).sum()) for bucket in bucket_order},
            "recommended_next_lever": recommended_next_lever,
            "recommendation_rationale": recommendation_rationale,
        }
