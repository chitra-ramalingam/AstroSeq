from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig


class K2ShortlistRecoveryModeAnalysis:
    DEFAULT_BASELINE_RUN_DIR = Path(r"plots\k2_batch\compare_mcc3_2000_valcap")
    DEFAULT_MCC2_RUN_DIR = Path(r"plots\k2_batch\compare_mcc2_2000_valcap")
    DEFAULT_THRESHOLD_RUN_DIR = Path(r"plots\k2_batch\compare_mcc2_threshold_relaxed_2000")
    DEFAULT_FAIL_REASON_CSV = "post_mcc_remaining_failures_by_reason.csv"
    DEFAULT_FAIL_BIN_CSV = "post_mcc_remaining_failures_by_period_bin.csv"
    DEFAULT_MODE_COMPARISON_CSV = "recovery_mode_comparison.csv"
    DEFAULT_RESCUED_BY_MODE_CSV = "rescued_by_mode.csv"
    DEFAULT_NO_P_DIAGNOSTICS_CSV = "post_mcc_no_p_available_whiteness_diagnostics.csv"
    DEFAULT_NO_P_BLOCKER_SUMMARY_CSV = "no_p_available_upstream_blocker_summary.csv"
    DEFAULT_NO_P_BLOCKER_BY_BIN_CSV = "no_p_available_upstream_blocker_by_period_bin.csv"
    DEFAULT_NO_UPSTREAM_EVENTS_DIAGNOSTICS_CSV = "no_upstream_events_detected_diagnostics.csv"
    DEFAULT_TOO_FEW_EVENTS_DIAGNOSTICS_CSV = "too_few_events_remaining_after_filtering_diagnostics.csv"
    DEFAULT_FIRST_FAILED_STAGE_SUMMARY_CSV = "first_failed_upstream_stage_summary.csv"
    DEFAULT_FIRST_FAILED_STAGE_BY_BIN_CSV = "first_failed_upstream_stage_by_period_bin.csv"
    DEFAULT_EVENT_DETECTION_ZERO_EVENTS_DIAGNOSTICS_CSV = "event_detection_zero_events_diagnostics.csv"
    DEFAULT_EVENT_DETECTION_INSUFFICIENT_SUPPORT_DIAGNOSTICS_CSV = "event_detection_insufficient_support_diagnostics.csv"
    DEFAULT_ZERO_EVENT_CAUSE_SUMMARY_CSV = "suspected_zero_event_cause_summary.csv"
    DEFAULT_ZERO_EVENT_CAUSE_BY_BIN_CSV = "suspected_zero_event_cause_by_period_bin.csv"
    DEFAULT_INSUFFICIENT_SUPPORT_CAUSE_SUMMARY_CSV = "suspected_insufficient_support_cause_summary.csv"
    DEFAULT_INSUFFICIENT_SUPPORT_CAUSE_BY_BIN_CSV = "suspected_insufficient_support_cause_by_period_bin.csv"

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Compare K2 shortlist recovery modes and summarize remaining failure buckets.")
        p.add_argument("--baseline-run-dir", type=Path, default=cls.DEFAULT_BASELINE_RUN_DIR, help=f"Baseline precision-first run directory. Default: {cls.DEFAULT_BASELINE_RUN_DIR}")
        p.add_argument("--mcc2-run-dir", type=Path, default=cls.DEFAULT_MCC2_RUN_DIR, help=f"Supported high-recall run directory. Default: {cls.DEFAULT_MCC2_RUN_DIR}")
        p.add_argument("--threshold-run-dir", type=Path, default=cls.DEFAULT_THRESHOLD_RUN_DIR, help=f"Threshold-relaxed MCC=2 run directory. Default: {cls.DEFAULT_THRESHOLD_RUN_DIR}")
        p.add_argument("--out-dir", type=Path, default=None, help="Output directory for analysis artifacts. Default: threshold run directory.")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.threshold_run_dir)
        return cls().run(
            baseline_run_dir=Path(args.baseline_run_dir),
            mcc2_run_dir=Path(args.mcc2_run_dir),
            threshold_run_dir=Path(args.threshold_run_dir),
            out_dir=out_dir,
        )

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _canonical_epic(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return ""
        m = re.search(r"(\d+)", text)
        return m.group(1) if m is not None else text

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _coalesce_series(df: pd.DataFrame, primary: str, secondary: str, default: Any = pd.NA) -> pd.Series:
        if primary in df.columns:
            out = df[primary].copy()
        elif secondary in df.columns:
            out = df[secondary].copy()
        else:
            return pd.Series([default] * len(df), index=df.index)
        if (primary in df.columns) and (secondary in df.columns):
            out = out.where(out.notna(), df[secondary])
        return out

    @staticmethod
    def _period_bin_for_series(p: pd.Series) -> pd.Series:
        p_num = pd.to_numeric(p, errors="coerce")
        bins = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]
        labels = ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]
        out = pd.cut(p_num, bins=bins, labels=labels, include_lowest=True, right=True)
        out = out.astype("object")
        out = out.where(out.notna(), "no_P_available")
        return out.astype(str)

    @staticmethod
    def _expand_funnel_details(funnel: pd.DataFrame) -> pd.DataFrame:
        if len(funnel) == 0 or "details_json" not in funnel.columns:
            return funnel.copy()
        details_records: List[Dict[str, Any]] = []
        for raw in funnel["details_json"].tolist():
            payload: Dict[str, Any] = {}
            text = str(raw).strip()
            if text != "" and text.lower() != "nan":
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}
            details_records.append(payload)
        details_df = pd.DataFrame(details_records, index=funnel.index)
        details_df = details_df.rename(columns={c: c for c in details_df.columns if c not in funnel.columns})
        out = funnel.copy()
        for col in details_df.columns:
            if col not in out.columns:
                out[col] = details_df[col]
            else:
                out[col] = out[col].where(out[col].notna(), details_df[col])
        return out

    def _load_raw_triage_table(self, raw_path: Optional[Path] = None) -> tuple[pd.DataFrame, str]:
        raw_path = Path(raw_path) if raw_path is not None else Path(K2ShortlistPeriodConfig.RAW_EPIC_LIST_CSV)
        raw = self._read_csv(raw_path)
        if len(raw) == 0:
            return pd.DataFrame(columns=["epic_id"]), "triage_whiteness_pvalue"
        if "epic_id" in raw.columns:
            epic_series = raw["epic_id"].map(self._canonical_epic)
        else:
            query_series = raw.get("query", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str)
            epic_series = query_series.str.extract(r"(\d+)")[0].fillna("").astype(str)
        whiteness_col = "triage_whiteness_pvalue" if "triage_whiteness_pvalue" in raw.columns else "triage_whiteness_score"
        out = pd.DataFrame(
            {
                "epic_id": epic_series,
                "query_raw": raw.get("query", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "triage_status_raw": raw.get("triage_status", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "triage_usable_raw": raw.get("triage_usable", pd.Series([False] * len(raw), index=raw.index)),
                whiteness_col: raw.get(whiteness_col, pd.Series([pd.NA] * len(raw), index=raw.index)),
                "triage_whiteness_definition_raw": raw.get("triage_whiteness_definition", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "triage_why_not_usable_raw": raw.get("triage_why_not_usable", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "error_stage_raw": raw.get("error_stage", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "error_type_raw": raw.get("error_type", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "error_msg_raw": raw.get("error_msg", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
                "campaign_selected_raw": raw.get("campaign_selected", pd.Series([""] * len(raw), index=raw.index)).fillna("").astype(str),
            }
        )
        out = out.loc[out["epic_id"] != ""].drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)
        return out, whiteness_col

    @staticmethod
    def _is_whiteness_related_text(text: str) -> bool:
        value = str(text).strip().lower()
        if value == "":
            return False
        tokens = ["whiteness", "autocorr", "pvalue", "alpha", "not_white"]
        return any(tok in value for tok in tokens)

    @staticmethod
    def _is_other_quality_text(text: str) -> bool:
        value = str(text).strip().lower()
        if value == "":
            return False
        tokens = [
            "n_points<",
            "baseline_days<",
            "robust_sigma<",
            "outlier_rate",
            "all_flux_nan",
            "insufficient",
            "quality",
            "bad_quality",
        ]
        return any(tok in value for tok in tokens)

    def _dominant_upstream_blocker(self, row: Dict[str, Any]) -> str:
        raw_count = pd.to_numeric(pd.Series([row.get("raw_event_count_before_filters", pd.NA)]), errors="coerce").iloc[0]
        usable_count = pd.to_numeric(pd.Series([row.get("usable_event_count_after_filters", pd.NA)]), errors="coerce").iloc[0]
        reason_text = " | ".join(
            [
                str(row.get("shortlist_rejection_reason", "") or ""),
                str(row.get("source_reason", "") or ""),
                str(row.get("failure_detail", "") or ""),
                str(row.get("triage_why_not_usable", "") or ""),
                str(row.get("triage_whiteness_definition", "") or ""),
            ]
        ).strip().lower()

        if pd.notna(raw_count) and float(raw_count) <= 0:
            return "no_upstream_events_detected"
        if self._is_whiteness_related_text(reason_text):
            return "whiteness_related_filtering"
        if pd.notna(usable_count) and float(usable_count) <= 0 and self._is_other_quality_text(reason_text):
            return "other_quality_filtering"
        if pd.notna(usable_count) and float(usable_count) <= 1:
            return "too_few_events_remaining_after_filtering"
        if self._is_other_quality_text(reason_text):
            return "other_quality_filtering"
        return "other_known_upstream_gating"

    def _first_failed_upstream_stage(self, row: Dict[str, Any]) -> str:
        triage_status = str(row.get("triage_status", "") or "").strip().lower()
        raw_count = pd.to_numeric(pd.Series([row.get("raw_event_count_before_filters", pd.NA)]), errors="coerce").iloc[0]
        usable_count = pd.to_numeric(pd.Series([row.get("usable_event_count_after_filters", pd.NA)]), errors="coerce").iloc[0]
        blocker = str(row.get("dominant_upstream_blocker", "") or "").strip()

        if triage_status == "error":
            return "lightcurve_load_or_triage"
        if blocker == "no_upstream_events_detected":
            return "event_detection_produced_zero_events"
        if pd.notna(usable_count) and float(usable_count) < 2.0:
            return "event_detection_produced_insufficient_support"
        if blocker == "whiteness_related_filtering":
            return "whiteness_related_filtering"
        if blocker == "other_quality_filtering":
            return "other_quality_filtering"
        if pd.notna(raw_count) and pd.notna(usable_count) and float(usable_count) < float(raw_count):
            return "post_detection_event_filtering"
        return "other_known_upstream_gating"

    def _suspected_zero_event_cause(self, row: Dict[str, Any]) -> str:
        triage_status = str(row.get("triage_status", "") or "").strip().lower()
        why = str(row.get("triage_why_not_usable", "") or "").strip().lower()
        raw_count = pd.to_numeric(
            pd.Series([row.get("raw_detector_output_count", row.get("raw_detector_output_count_before_downstream_filtering", pd.NA))]),
            errors="coerce",
        ).iloc[0]

        if triage_status == "error":
            return "lightcurve_load_or_triage_error"
        if "all_flux_nan" in why:
            return "all_flux_nan_or_empty_after_cleaning"
        if ("n_points<" in why) or ("baseline_days<" in why) or ("insufficient" in why):
            return "insufficient_baseline_or_points"
        if ("outlier_rate" in why) or ("robust_sigma" in why) or ("quality" in why) or ("noisy" in why):
            return "preprocessing_or_quality_handling"
        if self._is_whiteness_related_text(why):
            return "whiteness_related_quality_gate"
        if pd.notna(raw_count) and float(raw_count) == 0.0:
            return "detector_sensitivity_or_candidate_generation"
        return "instrumentation_missing_zero_event_cause"

    def _suspected_insufficient_support_cause(self, row: Dict[str, Any]) -> str:
        why = str(row.get("triage_why_not_usable", "") or "").strip().lower()
        raw_count = pd.to_numeric(pd.Series([row.get("raw_detected_event_count", pd.NA)]), errors="coerce").iloc[0]
        usable_count = pd.to_numeric(pd.Series([row.get("usable_event_count", pd.NA)]), errors="coerce").iloc[0]
        removed = pd.to_numeric(pd.Series([row.get("events_removed_by_filtering", pd.NA)]), errors="coerce").iloc[0]

        if pd.notna(raw_count) and float(raw_count) <= 1.0:
            return "detector_sensitivity_or_candidate_generation"
        if pd.notna(removed) and float(removed) > 0.0:
            if ("outlier_rate" in why) or ("robust_sigma" in why) or ("quality" in why) or ("noisy" in why):
                return "preprocessing_or_quality_handling"
            if self._is_whiteness_related_text(why):
                return "whiteness_related_quality_gate"
            return "downstream_event_retention_before_shortlist"
        if pd.notna(raw_count) and pd.notna(usable_count) and float(usable_count) < 2.0:
            return "detector_sensitivity_or_candidate_generation"
        return "instrumentation_missing_insufficient_support_cause"

    @staticmethod
    def _failure_bucket(row: Dict[str, Any]) -> str:
        failure = str(row.get("failure_category", "") or "").strip().lower()
        reason = str(row.get("reason", "") or "").strip().lower()
        source = str(row.get("source_reason", "") or "").strip().lower()
        shortlist_reason = str(row.get("shortlist_rejection_reason", "") or "").strip().lower()
        terminal = str(row.get("terminal_reason", "") or "").strip().lower()
        detail = str(row.get("failure_detail", "") or "").strip().lower()

        if (
            failure == "events_filtered_to_zero"
            or source == "events_filtered_to_zero"
            or shortlist_reason == "events_filtered_to_zero"
            or (terminal == "too_few_events_after_filters" and source == "events_filtered_to_zero")
            or reason == "events_filtered_to_zero"
        ):
            return "events_filtered_to_zero"

        if (
            failure == "insufficient_events"
            or shortlist_reason == "insufficient_events"
            or "insufficient_events" in source
            or "insufficient_events" in detail
            or "n_events_after_filters=1" in source
        ):
            return "insufficient_events"

        if (
            terminal == "shortlist_precheck_reject"
            or source == "whiteness_null_and_triage_unusable"
            or shortlist_reason == "whiteness_null_and_triage_unusable"
            or terminal == "all_flux_nan/insufficient_points"
        ):
            return "triage_unusable_or_quality_failures"

        if (
            failure in {"candidate_filter_rejection", "empty_histogram", "below_min_cluster_count"}
            or source in {"no_cluster_periods", "below_min_cluster_count", "cluster_only_no_valid_period"}
            or terminal == "no_cluster_periods"
            or "min_cluster_count" in detail
        ):
            return "cluster_related_failures"

        return "other"

    def _load_run_state(self, run_dir: Path) -> Dict[str, Any]:
        best = self._read_csv(run_dir / "period_shortlist_best.csv").copy()
        quarantine = self._read_csv(run_dir / "period_shortlist_quarantine.csv").copy()
        funnel = self._expand_funnel_details(self._read_csv(run_dir / "epic_funnel_reasons.csv").copy())
        diagnostics = self._read_csv(run_dir / "period_shortlist_diagnostics.csv").copy()

        if "epic" in best.columns:
            best["epic"] = best["epic"].map(self._canonical_epic)
            best = best.loc[best["epic"] != ""].drop_duplicates(subset=["epic"], keep="first").reset_index(drop=True)
        else:
            best["epic"] = pd.Series(dtype=str)

        if "epic_id" in quarantine.columns:
            quarantine["epic_id"] = quarantine["epic_id"].map(self._canonical_epic)
            quarantine = quarantine.loc[quarantine["epic_id"] != ""].drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)
        else:
            quarantine["epic_id"] = pd.Series(dtype=str)

        if "epic_id" in funnel.columns:
            funnel["epic_id"] = funnel["epic_id"].map(self._canonical_epic)
        else:
            funnel["epic_id"] = pd.Series(dtype=str)

        selected_epics: Set[str] = set()
        if "selected_for_period_stage" in funnel.columns:
            selected_mask = funnel["selected_for_period_stage"].fillna(False).astype(bool)
            selected_epics = {str(x) for x in funnel.loc[selected_mask, "epic_id"].tolist() if str(x) != ""}
        if len(selected_epics) == 0:
            selected_epics = set(best["epic"].tolist()).union(set(quarantine["epic_id"].tolist()))

        best_epics = set(best["epic"].tolist())
        failed_epics = selected_epics.difference(best_epics)

        quarantine_fail = quarantine.loc[quarantine["epic_id"].isin(failed_epics)].copy()
        quarantine_fail["failure_reason_bucket"] = quarantine_fail.apply(lambda row: self._failure_bucket(row.to_dict()), axis=1)
        if "P" not in quarantine_fail.columns:
            quarantine_fail["P"] = pd.NA

        missing_fail_epics = failed_epics.difference(set(quarantine_fail["epic_id"].tolist()))
        if len(missing_fail_epics) > 0:
            funnel_fail = funnel.loc[funnel["epic_id"].isin(missing_fail_epics)].copy()
            if len(funnel_fail) > 0:
                funnel_fail["failure_reason_bucket"] = funnel_fail.apply(lambda row: self._failure_bucket(row.to_dict()), axis=1)
                if "P" not in funnel_fail.columns:
                    funnel_fail["P"] = pd.NA
                funnel_fail = funnel_fail.rename(columns={"epic_id": "epic_id"})
                quarantine_fail = pd.concat(
                    [quarantine_fail, funnel_fail.reindex(columns=list(quarantine_fail.columns.union(funnel_fail.columns)))],
                    ignore_index=True,
                )
                quarantine_fail = quarantine_fail.drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)

        if len(quarantine_fail) == 0:
            quarantine_fail = pd.DataFrame(columns=["epic_id", "failure_reason_bucket", "P"])

        quarantine_fail["period_bin"] = self._period_bin_for_series(quarantine_fail.get("P", pd.Series(dtype=float)))

        return {
            "run_dir": run_dir,
            "best": best,
            "quarantine": quarantine,
            "funnel": funnel,
            "diagnostics": diagnostics,
            "selected_epics": selected_epics,
            "best_epics": best_epics,
            "failed_rows": quarantine_fail,
        }

    @staticmethod
    def _period_bin_count_map(best_df: pd.DataFrame) -> Dict[str, int]:
        if len(best_df) == 0:
            return {label: 0 for label in ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]}
        bins = K2ShortlistRecoveryModeAnalysis._period_bin_for_series(best_df.get("P", pd.Series(dtype=float)))
        counts = bins.value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]}

    @staticmethod
    def _manual_review_count(best_df: pd.DataFrame) -> int:
        if len(best_df) == 0 or "manual_review_required" not in best_df.columns:
            return 0
        mask = best_df["manual_review_required"].fillna(False).astype(bool)
        return int(best_df.loc[mask, "epic"].nunique())

    def run(
        self,
        baseline_run_dir: Path,
        mcc2_run_dir: Path,
        threshold_run_dir: Path,
        out_dir: Path,
    ) -> Dict[str, Any]:
        baseline = self._load_run_state(baseline_run_dir)
        mcc2 = self._load_run_state(mcc2_run_dir)
        threshold = self._load_run_state(threshold_run_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        bucket_order = [
            "events_filtered_to_zero",
            "insufficient_events",
            "cluster_related_failures",
            "triage_unusable_or_quality_failures",
            "other",
        ]

        post_mcc_failed = mcc2["failed_rows"].copy()
        reason_rows = []
        for bucket in bucket_order:
            count = int((post_mcc_failed.get("failure_reason_bucket", pd.Series(dtype=str)) == bucket).sum())
            reason_rows.append({"failure_reason_bucket": bucket, "count": count})
        fail_reason_df = pd.DataFrame(reason_rows)

        fail_bin_rows: List[Dict[str, Any]] = []
        for period_bin in ["no_P_available", "(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]:
            sub = post_mcc_failed.loc[post_mcc_failed.get("period_bin", pd.Series(dtype=str)).astype(str) == period_bin]
            for bucket in bucket_order:
                count = int((sub.get("failure_reason_bucket", pd.Series(dtype=str)) == bucket).sum())
                fail_bin_rows.append(
                    {
                        "period_bin": period_bin,
                        "failure_reason_bucket": bucket,
                        "count": count,
                    }
                )
        fail_bin_df = pd.DataFrame(fail_bin_rows)

        raw_triage_df, whiteness_col = self._load_raw_triage_table()
        no_p_diag = post_mcc_failed.loc[post_mcc_failed.get("period_bin", pd.Series(dtype=str)).astype(str) == "no_P_available"].copy()
        if len(no_p_diag) == 0:
            no_p_diag = pd.DataFrame(columns=["epic_id"])
        mcc2_funnel = mcc2["funnel"].copy()
        if "epic_id" in mcc2_funnel.columns:
            mcc2_funnel["epic_id"] = mcc2_funnel["epic_id"].map(self._canonical_epic)
        funnel_cols = [
            "epic_id",
            "query",
            "source_reason",
            "shortlist_rejection_reason",
            "failure_detail",
            "period_n_events_raw",
            "period_n_events_after_filters",
        ]
        funnel_sub = mcc2_funnel.reindex(columns=[c for c in funnel_cols if c in mcc2_funnel.columns]).drop_duplicates(subset=["epic_id"], keep="first")
        no_p_diag = no_p_diag.merge(funnel_sub, how="left", on="epic_id", suffixes=("", "_funnel"))
        no_p_diag = no_p_diag.merge(raw_triage_df, how="left", on="epic_id")
        no_p_diag["query"] = self._coalesce_series(no_p_diag, "query", "query_raw", default="")
        no_p_diag["raw_event_count_before_filters"] = pd.to_numeric(
            self._coalesce_series(no_p_diag, "period_n_events_raw", "n_events_raw"), errors="coerce"
        )
        no_p_diag["usable_event_count_after_filters"] = pd.to_numeric(
            self._coalesce_series(no_p_diag, "period_n_events_after_filters", "n_events_after_filters"), errors="coerce"
        )
        no_p_diag["events_filtered_to_zero"] = no_p_diag.get("failure_reason_bucket", pd.Series(dtype=str)).astype(str).eq("events_filtered_to_zero")
        no_p_diag["insufficient_events"] = no_p_diag.get("failure_reason_bucket", pd.Series(dtype=str)).astype(str).eq("insufficient_events")
        triage_usable_series = self._coalesce_series(no_p_diag, "triage_usable_raw", "triage_usable", default=False)
        no_p_diag["triage_usable"] = triage_usable_series.map(self._to_bool)
        no_p_diag["triage_status"] = self._coalesce_series(
            no_p_diag, "triage_status_raw", "triage_status", default=""
        ).fillna("").astype(str)
        no_p_diag["triage_whiteness_definition"] = self._coalesce_series(
            no_p_diag, "triage_whiteness_definition_raw", "triage_whiteness_definition", default=""
        ).fillna("").astype(str)
        no_p_diag["triage_why_not_usable"] = self._coalesce_series(
            no_p_diag, "triage_why_not_usable_raw", "triage_why_not_usable", default=""
        ).fillna("").astype(str)
        no_p_diag["error_stage"] = self._coalesce_series(
            no_p_diag, "error_stage_raw", "error_stage", default=""
        ).fillna("").astype(str)
        no_p_diag["error_type"] = self._coalesce_series(
            no_p_diag, "error_type_raw", "error_type", default=""
        ).fillna("").astype(str)
        no_p_diag["error_msg"] = self._coalesce_series(
            no_p_diag, "error_msg_raw", "error_msg", default=""
        ).fillna("").astype(str)
        no_p_diag["campaign_selected"] = self._coalesce_series(
            no_p_diag, "campaign_selected_raw", "campaign_selected", default=""
        ).fillna("").astype(str)
        if whiteness_col not in no_p_diag.columns:
            no_p_diag[whiteness_col] = pd.NA
        no_p_diag["dominant_upstream_blocker"] = no_p_diag.apply(
            lambda row: self._dominant_upstream_blocker(row.to_dict()),
            axis=1,
        )
        no_p_diag["light_curve_loaded_successfully"] = no_p_diag["triage_status"].str.strip().str.lower().eq("ok")
        no_p_diag["upstream_event_detection_ran_successfully"] = pd.to_numeric(
            no_p_diag["raw_event_count_before_filters"], errors="coerce"
        ).notna()
        no_p_diag["raw_detector_output_count_before_downstream_filtering"] = no_p_diag["raw_event_count_before_filters"]
        no_p_diag["event_detection_ran_successfully"] = no_p_diag["upstream_event_detection_ran_successfully"]
        no_p_diag["raw_detector_output_count"] = no_p_diag["raw_detector_output_count_before_downstream_filtering"]
        no_p_diag["light_curve_quality_flags"] = no_p_diag["triage_why_not_usable"].fillna("").astype(str)
        no_p_diag["light_curve_availability_flags"] = no_p_diag["triage_status"].fillna("").astype(str)
        no_p_diag["detector_mode_identifier"] = pd.NA
        no_p_diag["detector_config_identifier"] = pd.NA
        no_p_diag["events_removed_by_filtering"] = (
            pd.to_numeric(no_p_diag["raw_event_count_before_filters"], errors="coerce").fillna(0.0)
            - pd.to_numeric(no_p_diag["usable_event_count_after_filters"], errors="coerce").fillna(0.0)
        )
        no_p_diag["first_failed_upstream_stage"] = no_p_diag.apply(
            lambda row: self._first_failed_upstream_stage(row.to_dict()),
            axis=1,
        )
        no_p_diag_output = no_p_diag.reindex(
            columns=[
                "epic_id",
                "query",
                "period_bin",
                "failure_reason_bucket",
                "raw_event_count_before_filters",
                "usable_event_count_after_filters",
                "events_filtered_to_zero",
                "insufficient_events",
                "triage_status",
                "triage_usable",
                whiteness_col,
                "triage_whiteness_definition",
                "triage_why_not_usable",
                "dominant_upstream_blocker",
                "light_curve_loaded_successfully",
                "upstream_event_detection_ran_successfully",
                "raw_detector_output_count_before_downstream_filtering",
                "events_removed_by_filtering",
                "error_stage",
                "error_type",
                "error_msg",
                "campaign_selected",
                "first_failed_upstream_stage",
            ]
        ).copy()
        blocker_summary_df = (
            no_p_diag.groupby("dominant_upstream_blocker", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "dominant_upstream_blocker"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        blocker_by_bin_df = (
            no_p_diag.groupby(["period_bin", "dominant_upstream_blocker"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["period_bin", "count", "dominant_upstream_blocker"], ascending=[True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        no_upstream_events_df = no_p_diag.loc[
            no_p_diag["dominant_upstream_blocker"].eq("no_upstream_events_detected")
        ].copy()
        too_few_events_df = no_p_diag.loc[
            no_p_diag["dominant_upstream_blocker"].eq("too_few_events_remaining_after_filtering")
        ].copy()
        for df in [no_upstream_events_df, too_few_events_df]:
            df["event_support_below_viability_after_detection"] = (
                pd.to_numeric(df["usable_event_count_after_filters"], errors="coerce").fillna(0.0) < 2.0
            )
            df["filtering_reduced_event_count"] = (
                pd.to_numeric(df["events_removed_by_filtering"], errors="coerce").fillna(0.0) > 0.0
            )
        no_upstream_events_df["suspected_zero_event_cause"] = no_upstream_events_df.apply(
            lambda row: self._suspected_zero_event_cause(row.to_dict()),
            axis=1,
        )
        no_upstream_events_df["event_detection_ran_successfully"] = no_upstream_events_df["upstream_event_detection_ran_successfully"]
        no_upstream_events_df["raw_detector_output_count"] = no_upstream_events_df["raw_detector_output_count_before_downstream_filtering"]
        no_upstream_events_df["detector_mode_identifier"] = pd.NA
        no_upstream_events_df["detector_config_identifier"] = pd.NA
        too_few_events_df["raw_detected_event_count"] = too_few_events_df["raw_event_count_before_filters"]
        too_few_events_df["usable_event_count"] = too_few_events_df["usable_event_count_after_filters"]
        too_few_events_df["suspected_insufficient_support_cause"] = too_few_events_df.apply(
            lambda row: self._suspected_insufficient_support_cause(row.to_dict()),
            axis=1,
        )
        zero_event_stage_df = no_p_diag.loc[
            no_p_diag["first_failed_upstream_stage"].eq("event_detection_produced_zero_events")
        ].copy()
        zero_event_stage_df["suspected_zero_event_cause"] = zero_event_stage_df.apply(
            lambda row: self._suspected_zero_event_cause(row.to_dict()),
            axis=1,
        )
        zero_event_stage_df["event_detection_ran_successfully"] = zero_event_stage_df["upstream_event_detection_ran_successfully"]
        zero_event_stage_df["raw_detector_output_count"] = zero_event_stage_df["raw_detector_output_count_before_downstream_filtering"]
        zero_event_stage_df["detector_mode_identifier"] = pd.NA
        zero_event_stage_df["detector_config_identifier"] = pd.NA
        zero_event_stage_df = zero_event_stage_df.reindex(
            columns=[
                "epic_id",
                "query",
                "period_bin",
                "light_curve_loaded_successfully",
                "light_curve_quality_flags",
                "light_curve_availability_flags",
                "event_detection_ran_successfully",
                "detector_mode_identifier",
                "detector_config_identifier",
                "raw_detector_output_count",
                "error_stage",
                "error_type",
                "error_msg",
                "campaign_selected",
                "triage_status",
                "triage_usable",
                whiteness_col,
                "triage_whiteness_definition",
                "triage_why_not_usable",
                "first_failed_upstream_stage",
                "suspected_zero_event_cause",
            ]
        ).copy()
        insufficient_support_stage_df = no_p_diag.loc[
            no_p_diag["first_failed_upstream_stage"].eq("event_detection_produced_insufficient_support")
        ].copy()
        insufficient_support_stage_df["event_support_below_viability_after_detection"] = (
            pd.to_numeric(insufficient_support_stage_df["usable_event_count_after_filters"], errors="coerce").fillna(0.0) < 2.0
        )
        insufficient_support_stage_df["filtering_reduced_event_count"] = (
            pd.to_numeric(insufficient_support_stage_df["events_removed_by_filtering"], errors="coerce").fillna(0.0) > 0.0
        )
        insufficient_support_stage_df["raw_detected_event_count"] = insufficient_support_stage_df["raw_event_count_before_filters"]
        insufficient_support_stage_df["usable_event_count"] = insufficient_support_stage_df["usable_event_count_after_filters"]
        insufficient_support_stage_df["suspected_insufficient_support_cause"] = insufficient_support_stage_df.apply(
            lambda row: self._suspected_insufficient_support_cause(row.to_dict()),
            axis=1,
        )
        insufficient_support_stage_df = insufficient_support_stage_df.reindex(
            columns=[
                "epic_id",
                "query",
                "period_bin",
                "raw_detected_event_count",
                "usable_event_count",
                "raw_event_count_before_filters",
                "usable_event_count_after_filters",
                "raw_detector_output_count_before_downstream_filtering",
                "events_removed_by_filtering",
                "filtering_reduced_event_count",
                "event_support_below_viability_after_detection",
                "triage_status",
                "error_stage",
                "error_type",
                "error_msg",
                "campaign_selected",
                "triage_usable",
                whiteness_col,
                "triage_whiteness_definition",
                "triage_why_not_usable",
                "first_failed_upstream_stage",
                "suspected_insufficient_support_cause",
            ]
        ).copy()
        no_upstream_events_df = zero_event_stage_df.copy()
        too_few_events_df = insufficient_support_stage_df.copy()
        zero_event_cause_summary_df = (
            zero_event_stage_df.groupby("suspected_zero_event_cause", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "suspected_zero_event_cause"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        zero_event_cause_by_bin_df = (
            zero_event_stage_df.groupby(["period_bin", "suspected_zero_event_cause"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["period_bin", "count", "suspected_zero_event_cause"], ascending=[True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        insufficient_support_cause_summary_df = (
            insufficient_support_stage_df.groupby("suspected_insufficient_support_cause", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "suspected_insufficient_support_cause"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        insufficient_support_cause_by_bin_df = (
            insufficient_support_stage_df.groupby(["period_bin", "suspected_insufficient_support_cause"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["period_bin", "count", "suspected_insufficient_support_cause"], ascending=[True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        stage_rollup_base = pd.concat(
            [zero_event_stage_df.copy(), insufficient_support_stage_df.copy()],
            ignore_index=True,
            sort=False,
        )
        first_failed_stage_summary_df = (
            stage_rollup_base.groupby("first_failed_upstream_stage", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "first_failed_upstream_stage"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        first_failed_stage_by_bin_df = (
            stage_rollup_base.groupby(["period_bin", "first_failed_upstream_stage"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["period_bin", "count", "first_failed_upstream_stage"], ascending=[True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )

        modes = [
            (str(K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME), baseline),
            (str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME), mcc2),
            (str(K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME), threshold),
        ]
        baseline_epics = baseline["best_epics"]
        mcc2_epics = mcc2["best_epics"]
        comparison_rows: List[Dict[str, Any]] = []
        for mode_name, state in modes:
            best_epics = state["best_epics"]
            period_counts = self._period_bin_count_map(state["best"])
            comparison_rows.append(
                {
                    "mode": mode_name,
                    "shortlisted_count": int(len(best_epics)),
                    "quarantined_count": int(len(state["failed_rows"])),
                    "added_vs_baseline": int(len(best_epics.difference(baseline_epics))),
                    "added_vs_mcc2": int(len(best_epics.difference(mcc2_epics))),
                    "manual_review_count": int(self._manual_review_count(state["best"])),
                    "best_count_bin_0_1": int(period_counts["(0,1]"]),
                    "best_count_bin_1_5": int(period_counts["(1,5]"]),
                    "best_count_bin_5_10": int(period_counts["(5,10]"]),
                    "best_count_bin_10_15": int(period_counts["(10,15]"]),
                    "best_count_bin_15_20": int(period_counts["(15,20]"]),
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)

        baseline_failure_map = dict(zip(baseline["failed_rows"].get("epic_id", pd.Series(dtype=str)), baseline["failed_rows"].get("failure_reason_bucket", pd.Series(dtype=str))))
        mcc2_failure_map = dict(zip(mcc2["failed_rows"].get("epic_id", pd.Series(dtype=str)), mcc2["failed_rows"].get("failure_reason_bucket", pd.Series(dtype=str))))

        baseline_period_map = dict(zip(baseline["best"].get("epic", pd.Series(dtype=str)), self._period_bin_for_series(baseline["best"].get("P", pd.Series(dtype=float)))))
        mcc2_period_map = dict(zip(mcc2["best"].get("epic", pd.Series(dtype=str)), self._period_bin_for_series(mcc2["best"].get("P", pd.Series(dtype=float)))))
        threshold_period_map = dict(zip(threshold["best"].get("epic", pd.Series(dtype=str)), self._period_bin_for_series(threshold["best"].get("P", pd.Series(dtype=float)))))

        rescued_union = sorted(
            baseline_epics.union(mcc2_epics).union(threshold["best_epics"]),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        rescued_rows: List[Dict[str, Any]] = []
        for epic in rescued_union:
            in_baseline = epic in baseline_epics
            in_mcc2 = epic in mcc2_epics
            in_threshold = epic in threshold["best_epics"]
            prior_reason = ""
            if in_threshold and (not in_mcc2):
                prior_reason = str(mcc2_failure_map.get(epic, ""))
            elif in_mcc2 and (not in_baseline):
                prior_reason = str(baseline_failure_map.get(epic, ""))

            period_bin = ""
            if in_threshold:
                period_bin = str(threshold_period_map.get(epic, ""))
            elif in_mcc2:
                period_bin = str(mcc2_period_map.get(epic, ""))
            elif in_baseline:
                period_bin = str(baseline_period_map.get(epic, ""))

            rescued_rows.append(
                {
                    "epic_id": epic,
                    "rescued_by_baseline": bool(in_baseline),
                    "rescued_by_mcc2": bool(in_mcc2),
                    "rescued_by_threshold_relaxed": bool(in_threshold),
                    "dominant_prior_failure_reason": prior_reason,
                    "period_bin": period_bin,
                }
            )
        rescued_df = pd.DataFrame(rescued_rows)

        fail_reason_csv = out_dir / self.DEFAULT_FAIL_REASON_CSV
        fail_bin_csv = out_dir / self.DEFAULT_FAIL_BIN_CSV
        comparison_csv = out_dir / self.DEFAULT_MODE_COMPARISON_CSV
        rescued_csv = out_dir / self.DEFAULT_RESCUED_BY_MODE_CSV
        no_p_diag_csv = out_dir / self.DEFAULT_NO_P_DIAGNOSTICS_CSV
        no_p_blocker_summary_csv = out_dir / self.DEFAULT_NO_P_BLOCKER_SUMMARY_CSV
        no_p_blocker_by_bin_csv = out_dir / self.DEFAULT_NO_P_BLOCKER_BY_BIN_CSV
        no_upstream_events_csv = out_dir / self.DEFAULT_NO_UPSTREAM_EVENTS_DIAGNOSTICS_CSV
        too_few_events_csv = out_dir / self.DEFAULT_TOO_FEW_EVENTS_DIAGNOSTICS_CSV
        first_failed_stage_summary_csv = out_dir / self.DEFAULT_FIRST_FAILED_STAGE_SUMMARY_CSV
        first_failed_stage_by_bin_csv = out_dir / self.DEFAULT_FIRST_FAILED_STAGE_BY_BIN_CSV
        event_detection_zero_events_csv = out_dir / self.DEFAULT_EVENT_DETECTION_ZERO_EVENTS_DIAGNOSTICS_CSV
        event_detection_insufficient_support_csv = out_dir / self.DEFAULT_EVENT_DETECTION_INSUFFICIENT_SUPPORT_DIAGNOSTICS_CSV
        zero_event_cause_summary_csv = out_dir / self.DEFAULT_ZERO_EVENT_CAUSE_SUMMARY_CSV
        zero_event_cause_by_bin_csv = out_dir / self.DEFAULT_ZERO_EVENT_CAUSE_BY_BIN_CSV
        insufficient_support_cause_summary_csv = out_dir / self.DEFAULT_INSUFFICIENT_SUPPORT_CAUSE_SUMMARY_CSV
        insufficient_support_cause_by_bin_csv = out_dir / self.DEFAULT_INSUFFICIENT_SUPPORT_CAUSE_BY_BIN_CSV
        fail_reason_df.to_csv(fail_reason_csv, index=False)
        fail_bin_df.to_csv(fail_bin_csv, index=False)
        comparison_df.to_csv(comparison_csv, index=False)
        rescued_df.to_csv(rescued_csv, index=False)
        no_p_diag_output.to_csv(no_p_diag_csv, index=False)
        blocker_summary_df.to_csv(no_p_blocker_summary_csv, index=False)
        blocker_by_bin_df.to_csv(no_p_blocker_by_bin_csv, index=False)
        no_upstream_events_df.to_csv(no_upstream_events_csv, index=False)
        too_few_events_df.to_csv(too_few_events_csv, index=False)
        first_failed_stage_summary_df.to_csv(first_failed_stage_summary_csv, index=False)
        first_failed_stage_by_bin_df.to_csv(first_failed_stage_by_bin_csv, index=False)
        zero_event_stage_df.to_csv(event_detection_zero_events_csv, index=False)
        insufficient_support_stage_df.to_csv(event_detection_insufficient_support_csv, index=False)
        zero_event_cause_summary_df.to_csv(zero_event_cause_summary_csv, index=False)
        zero_event_cause_by_bin_df.to_csv(zero_event_cause_by_bin_csv, index=False)
        insufficient_support_cause_summary_df.to_csv(insufficient_support_cause_summary_csv, index=False)
        insufficient_support_cause_by_bin_df.to_csv(insufficient_support_cause_by_bin_csv, index=False)

        top_reasons = fail_reason_df.sort_values(["count", "failure_reason_bucket"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        threshold_added = int(len(threshold["best_epics"].difference(mcc2_epics)))
        threshold_bin_15_20 = int(comparison_df.loc[comparison_df["mode"] == str(K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME), "best_count_bin_15_20"].iloc[0]) if len(comparison_df) > 0 else 0
        mcc2_bin_15_20 = int(comparison_df.loc[comparison_df["mode"] == str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME), "best_count_bin_15_20"].iloc[0]) if len(comparison_df) > 0 else 0
        threshold_manual = int(comparison_df.loc[comparison_df["mode"] == str(K2ShortlistPeriodConfig.THRESHOLD_RELAXED_MODE_NAME), "manual_review_count"].iloc[0]) if len(comparison_df) > 0 else 0
        mcc2_manual = int(comparison_df.loc[comparison_df["mode"] == str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME), "manual_review_count"].iloc[0]) if len(comparison_df) > 0 else 0

        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {fail_reason_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {fail_bin_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {comparison_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {rescued_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {no_p_diag_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {no_p_blocker_summary_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {no_p_blocker_by_bin_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {no_upstream_events_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {too_few_events_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {first_failed_stage_summary_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {first_failed_stage_by_bin_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {event_detection_zero_events_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {event_detection_insufficient_support_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {zero_event_cause_summary_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {zero_event_cause_by_bin_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {insufficient_support_cause_summary_csv}")
        print(f"[K2ShortlistRecoveryModeAnalysis] wrote {insufficient_support_cause_by_bin_csv}")

        return {
            "post_mcc_remaining_failures_by_reason_csv": fail_reason_csv,
            "post_mcc_remaining_failures_by_period_bin_csv": fail_bin_csv,
            "recovery_mode_comparison_csv": comparison_csv,
            "rescued_by_mode_csv": rescued_csv,
            "post_mcc_no_p_available_whiteness_diagnostics_csv": no_p_diag_csv,
            "no_p_available_upstream_blocker_summary_csv": no_p_blocker_summary_csv,
            "no_p_available_upstream_blocker_by_period_bin_csv": no_p_blocker_by_bin_csv,
            "no_upstream_events_detected_diagnostics_csv": no_upstream_events_csv,
            "too_few_events_remaining_after_filtering_diagnostics_csv": too_few_events_csv,
            "first_failed_upstream_stage_summary_csv": first_failed_stage_summary_csv,
            "first_failed_upstream_stage_by_period_bin_csv": first_failed_stage_by_bin_csv,
            "event_detection_zero_events_diagnostics_csv": event_detection_zero_events_csv,
            "event_detection_insufficient_support_diagnostics_csv": event_detection_insufficient_support_csv,
            "suspected_zero_event_cause_summary_csv": zero_event_cause_summary_csv,
            "suspected_zero_event_cause_by_period_bin_csv": zero_event_cause_by_bin_csv,
            "suspected_insufficient_support_cause_summary_csv": insufficient_support_cause_summary_csv,
            "suspected_insufficient_support_cause_by_period_bin_csv": insufficient_support_cause_by_bin_csv,
            "remaining_top_failure_reasons": top_reasons.head(5).to_dict(orient="records"),
            "threshold_added_vs_mcc2": threshold_added,
            "period_bin_15_20_delta_vs_mcc2": int(threshold_bin_15_20 - mcc2_bin_15_20),
            "manual_review_delta_vs_mcc2": int(threshold_manual - mcc2_manual),
            "no_p_whiteness_value_column": whiteness_col,
            "no_p_whiteness_related_blocker_count": int(
                blocker_summary_df.loc[
                    blocker_summary_df["dominant_upstream_blocker"] == "whiteness_related_filtering",
                    "count",
                ].sum()
            ),
            "events_filtered_to_zero_whiteness_related_count": int(
                len(
                    no_p_diag_output.loc[
                        no_p_diag_output["events_filtered_to_zero"].fillna(False).astype(bool)
                        & no_p_diag_output["dominant_upstream_blocker"].eq("whiteness_related_filtering")
                    ]
                )
            ),
            "dominant_no_p_upstream_blocker": str(
                blocker_summary_df.iloc[0]["dominant_upstream_blocker"] if len(blocker_summary_df) > 0 else ""
            ),
            "dominant_first_failed_upstream_stage": str(
                first_failed_stage_summary_df.iloc[0]["first_failed_upstream_stage"] if len(first_failed_stage_summary_df) > 0 else ""
            ),
            "top_suspected_zero_event_cause": str(
                zero_event_cause_summary_df.iloc[0]["suspected_zero_event_cause"] if len(zero_event_cause_summary_df) > 0 else ""
            ),
            "top_suspected_insufficient_support_cause": str(
                insufficient_support_cause_summary_df.iloc[0]["suspected_insufficient_support_cause"]
                if len(insufficient_support_cause_summary_df) > 0
                else ""
            ),
            "no_upstream_events_detected_count": int(len(no_upstream_events_df)),
            "too_few_events_remaining_after_filtering_count": int(len(too_few_events_df)),
        }
