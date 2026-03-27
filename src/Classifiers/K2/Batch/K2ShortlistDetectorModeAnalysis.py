from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2ShortlistDetectorModeAnalysis:
    DEFAULT_BASELINE_RUN_DIR = Path(r"plots\k2_batch\compare_mcc3_2000_valcap")
    DEFAULT_MCC2_RUN_DIR = Path(r"plots\k2_batch\compare_mcc2_2000_valcap")
    DEFAULT_DETECTOR_RUN_DIR = Path(r"plots\k2_batch\detector_high_recall_experimental_2000_valcap")
    DEFAULT_COMPARISON_CSV = "detector_mode_comparison.csv"
    DEFAULT_RESCUED_CSV = "rescued_by_detector_mode.csv"
    DEFAULT_RESCUED_BY_BIN_CSV = "rescued_by_detector_mode_by_period_bin.csv"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Compare baseline, MCC=2, and detector-side experimental shortlist runs.")
        p.add_argument("--baseline-run-dir", type=Path, default=cls.DEFAULT_BASELINE_RUN_DIR)
        p.add_argument("--mcc2-run-dir", type=Path, default=cls.DEFAULT_MCC2_RUN_DIR)
        p.add_argument("--detector-run-dir", type=Path, default=cls.DEFAULT_DETECTOR_RUN_DIR)
        p.add_argument("--out-dir", type=Path, default=None, help="Output directory for detector comparison artifacts. Default: detector run directory.")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.detector_run_dir)
        return cls().run(
            baseline_run_dir=Path(args.baseline_run_dir),
            mcc2_run_dir=Path(args.mcc2_run_dir),
            detector_run_dir=Path(args.detector_run_dir),
            out_dir=out_dir,
        )

    def _raw_csv_path_for_state(self, state: Dict[str, Any]) -> Path:
        diagnostics = state.get("diagnostics", pd.DataFrame()).copy()
        if len(diagnostics) > 0 and "raw_epic_list_csv" in diagnostics.columns:
            value = str(diagnostics.iloc[0].get("raw_epic_list_csv", "") or "").strip()
            if value != "":
                return Path(value)
        return Path(K2ShortlistPeriodConfig.RAW_EPIC_LIST_CSV)

    def _build_no_p_diag(self, state: Dict[str, Any]) -> pd.DataFrame:
        raw_triage_df, whiteness_col = self.helper._load_raw_triage_table(raw_path=self._raw_csv_path_for_state(state))
        failed_rows = state.get("failed_rows", pd.DataFrame()).copy()
        no_p_diag = failed_rows.loc[
            failed_rows.get("period_bin", pd.Series(dtype=str)).astype(str).eq("no_P_available")
        ].copy()
        if len(no_p_diag) == 0:
            return pd.DataFrame(
                columns=[
                    "epic_id",
                    "query",
                    "period_bin",
                    "failure_reason_bucket",
                    "raw_event_count_before_filters",
                    "usable_event_count_after_filters",
                    "first_failed_upstream_stage",
                    "suspected_zero_event_cause",
                    "suspected_insufficient_support_cause",
                    whiteness_col,
                ]
            )

        funnel = state.get("funnel", pd.DataFrame()).copy()
        if "epic_id" in funnel.columns:
            funnel["epic_id"] = funnel["epic_id"].map(self.helper._canonical_epic)
        funnel_cols = [
            "epic_id",
            "query",
            "source_reason",
            "shortlist_rejection_reason",
            "failure_detail",
            "period_n_events_raw",
            "period_n_events_after_filters",
        ]
        funnel_sub = funnel.reindex(columns=[c for c in funnel_cols if c in funnel.columns]).drop_duplicates(
            subset=["epic_id"], keep="first"
        )
        no_p_diag = no_p_diag.merge(funnel_sub, how="left", on="epic_id", suffixes=("", "_funnel"))
        no_p_diag = no_p_diag.merge(raw_triage_df, how="left", on="epic_id")
        no_p_diag["query"] = self.helper._coalesce_series(no_p_diag, "query", "query_raw", default="")
        no_p_diag["raw_event_count_before_filters"] = pd.to_numeric(
            self.helper._coalesce_series(no_p_diag, "period_n_events_raw", "n_events_raw"),
            errors="coerce",
        )
        no_p_diag["usable_event_count_after_filters"] = pd.to_numeric(
            self.helper._coalesce_series(no_p_diag, "period_n_events_after_filters", "n_events_after_filters"),
            errors="coerce",
        )
        triage_usable_series = self.helper._coalesce_series(no_p_diag, "triage_usable_raw", "triage_usable", default=False)
        no_p_diag["triage_usable"] = triage_usable_series.map(self.helper._to_bool)
        no_p_diag["triage_status"] = self.helper._coalesce_series(
            no_p_diag, "triage_status_raw", "triage_status", default=""
        ).fillna("").astype(str)
        no_p_diag["triage_whiteness_definition"] = self.helper._coalesce_series(
            no_p_diag, "triage_whiteness_definition_raw", "triage_whiteness_definition", default=""
        ).fillna("").astype(str)
        no_p_diag["triage_why_not_usable"] = self.helper._coalesce_series(
            no_p_diag, "triage_why_not_usable_raw", "triage_why_not_usable", default=""
        ).fillna("").astype(str)
        no_p_diag["error_stage"] = self.helper._coalesce_series(no_p_diag, "error_stage_raw", "error_stage", default="").fillna("").astype(str)
        no_p_diag["error_type"] = self.helper._coalesce_series(no_p_diag, "error_type_raw", "error_type", default="").fillna("").astype(str)
        no_p_diag["error_msg"] = self.helper._coalesce_series(no_p_diag, "error_msg_raw", "error_msg", default="").fillna("").astype(str)
        if whiteness_col not in no_p_diag.columns:
            no_p_diag[whiteness_col] = pd.NA
        no_p_diag["dominant_upstream_blocker"] = no_p_diag.apply(
            lambda row: self.helper._dominant_upstream_blocker(row.to_dict()),
            axis=1,
        )
        no_p_diag["events_removed_by_filtering"] = (
            pd.to_numeric(no_p_diag["raw_event_count_before_filters"], errors="coerce").fillna(0.0)
            - pd.to_numeric(no_p_diag["usable_event_count_after_filters"], errors="coerce").fillna(0.0)
        )
        no_p_diag["first_failed_upstream_stage"] = no_p_diag.apply(
            lambda row: self.helper._first_failed_upstream_stage(row.to_dict()),
            axis=1,
        )
        no_p_diag["raw_detector_output_count"] = no_p_diag["raw_event_count_before_filters"]
        no_p_diag["raw_detected_event_count"] = no_p_diag["raw_event_count_before_filters"]
        no_p_diag["usable_event_count"] = no_p_diag["usable_event_count_after_filters"]
        no_p_diag["suspected_zero_event_cause"] = no_p_diag.apply(
            lambda row: self.helper._suspected_zero_event_cause(row.to_dict()),
            axis=1,
        )
        no_p_diag["suspected_insufficient_support_cause"] = no_p_diag.apply(
            lambda row: self.helper._suspected_insufficient_support_cause(row.to_dict()),
            axis=1,
        )
        return no_p_diag.reindex(
            columns=[
                "epic_id",
                "query",
                "period_bin",
                "failure_reason_bucket",
                "raw_event_count_before_filters",
                "usable_event_count_after_filters",
                "first_failed_upstream_stage",
                "suspected_zero_event_cause",
                "suspected_insufficient_support_cause",
                whiteness_col,
            ]
        ).copy()

    @staticmethod
    def _best_query_map(best_df: pd.DataFrame) -> Dict[str, str]:
        if len(best_df) == 0:
            return {}
        query_col = best_df.get("query", pd.Series([""] * len(best_df), index=best_df.index)).fillna("").astype(str)
        return dict(zip(best_df.get("epic", pd.Series(dtype=str)).astype(str), query_col))

    def run(
        self,
        baseline_run_dir: Path,
        mcc2_run_dir: Path,
        detector_run_dir: Path,
        out_dir: Path,
    ) -> Dict[str, Any]:
        baseline = self.helper._load_run_state(baseline_run_dir)
        mcc2 = self.helper._load_run_state(mcc2_run_dir)
        detector = self.helper._load_run_state(detector_run_dir)
        baseline_no_p = self._build_no_p_diag(baseline)
        mcc2_no_p = self._build_no_p_diag(mcc2)
        detector_no_p = self._build_no_p_diag(detector)

        out_dir.mkdir(parents=True, exist_ok=True)

        modes = [
            (str(K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME), baseline, baseline_no_p),
            (str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME), mcc2, mcc2_no_p),
            (str(K2ShortlistPeriodConfig.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE_NAME), detector, detector_no_p),
        ]
        baseline_epics = baseline["best_epics"]
        mcc2_epics = mcc2["best_epics"]

        comparison_rows: List[Dict[str, Any]] = []
        for mode_name, state, no_p_diag in modes:
            best_epics = state["best_epics"]
            comparison_rows.append(
                {
                    "mode": mode_name,
                    "shortlisted_count": int(len(best_epics)),
                    "quarantined_count": int(len(state["failed_rows"])),
                    "no_p_available_count": int(len(no_p_diag)),
                    "zero_event_count": int(
                        no_p_diag["first_failed_upstream_stage"].astype(str).eq("event_detection_produced_zero_events").sum()
                    ),
                    "insufficient_support_count": int(
                        no_p_diag["first_failed_upstream_stage"].astype(str).eq("event_detection_produced_insufficient_support").sum()
                    ),
                    "added_vs_baseline": int(len(best_epics.difference(baseline_epics))),
                    "added_vs_mcc2": int(len(best_epics.difference(mcc2_epics))),
                    "manual_review_count": int(self.helper._manual_review_count(state["best"])),
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)

        baseline_period_map = dict(
            zip(
                baseline["best"].get("epic", pd.Series(dtype=str)),
                self.helper._period_bin_for_series(baseline["best"].get("P", pd.Series(dtype=float))),
            )
        )
        mcc2_period_map = dict(
            zip(
                mcc2["best"].get("epic", pd.Series(dtype=str)),
                self.helper._period_bin_for_series(mcc2["best"].get("P", pd.Series(dtype=float))),
            )
        )
        detector_period_map = dict(
            zip(
                detector["best"].get("epic", pd.Series(dtype=str)),
                self.helper._period_bin_for_series(detector["best"].get("P", pd.Series(dtype=float))),
            )
        )
        baseline_query_map = self._best_query_map(baseline["best"])
        mcc2_query_map = self._best_query_map(mcc2["best"])
        detector_query_map = self._best_query_map(detector["best"])

        baseline_stage_map = dict(zip(baseline_no_p["epic_id"].astype(str), baseline_no_p["first_failed_upstream_stage"].astype(str)))
        mcc2_stage_map = dict(zip(mcc2_no_p["epic_id"].astype(str), mcc2_no_p["first_failed_upstream_stage"].astype(str)))
        baseline_cause_map = {
            str(epic): (
                str(cause_zero)
                if str(stage) == "event_detection_produced_zero_events"
                else str(cause_insufficient)
            )
            for epic, stage, cause_zero, cause_insufficient in zip(
                baseline_no_p["epic_id"].astype(str),
                baseline_no_p["first_failed_upstream_stage"].astype(str),
                baseline_no_p["suspected_zero_event_cause"].astype(str),
                baseline_no_p["suspected_insufficient_support_cause"].astype(str),
            )
        }
        mcc2_cause_map = {
            str(epic): (
                str(cause_zero)
                if str(stage) == "event_detection_produced_zero_events"
                else str(cause_insufficient)
            )
            for epic, stage, cause_zero, cause_insufficient in zip(
                mcc2_no_p["epic_id"].astype(str),
                mcc2_no_p["first_failed_upstream_stage"].astype(str),
                mcc2_no_p["suspected_zero_event_cause"].astype(str),
                mcc2_no_p["suspected_insufficient_support_cause"].astype(str),
            )
        }

        rescued_union = sorted(
            baseline_epics.union(mcc2_epics).union(detector["best_epics"]),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        rescued_rows: List[Dict[str, Any]] = []
        for epic in rescued_union:
            in_baseline = epic in baseline_epics
            in_mcc2 = epic in mcc2_epics
            in_detector = epic in detector["best_epics"]
            if in_detector:
                period_bin = str(detector_period_map.get(epic, ""))
                query = str(detector_query_map.get(epic, ""))
            elif in_mcc2:
                period_bin = str(mcc2_period_map.get(epic, ""))
                query = str(mcc2_query_map.get(epic, ""))
            else:
                period_bin = str(baseline_period_map.get(epic, ""))
                query = str(baseline_query_map.get(epic, ""))

            prior_stage = ""
            prior_cause = ""
            if in_detector and (not in_mcc2):
                prior_stage = str(mcc2_stage_map.get(str(epic), ""))
                prior_cause = str(mcc2_cause_map.get(str(epic), ""))
            elif in_mcc2 and (not in_baseline):
                prior_stage = str(baseline_stage_map.get(str(epic), ""))
                prior_cause = str(baseline_cause_map.get(str(epic), ""))

            rescued_rows.append(
                {
                    "epic_id": str(epic),
                    "query": query,
                    "period_bin": period_bin,
                    "rescued_by_baseline": bool(in_baseline),
                    "rescued_by_mcc2": bool(in_mcc2),
                    "rescued_by_detector_experimental": bool(in_detector),
                    "prior_first_failed_upstream_stage": prior_stage,
                    "prior_suspected_cause": prior_cause,
                }
            )
        rescued_df = pd.DataFrame(rescued_rows)

        rescued_by_bin_rows = []
        first_seen_masks = [
            (
                str(K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME),
                rescued_df["rescued_by_baseline"].fillna(False).astype(bool),
            ),
            (
                str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME),
                rescued_df["rescued_by_mcc2"].fillna(False).astype(bool)
                & ~rescued_df["rescued_by_baseline"].fillna(False).astype(bool),
            ),
            (
                str(K2ShortlistPeriodConfig.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE_NAME),
                rescued_df["rescued_by_detector_experimental"].fillna(False).astype(bool)
                & ~rescued_df["rescued_by_mcc2"].fillna(False).astype(bool),
            ),
        ]
        for mode_name, mask in first_seen_masks:
            sub = rescued_df.loc[mask].copy()
            counts = sub["period_bin"].fillna("").astype(str).value_counts()
            for period_bin, count in counts.items():
                rescued_by_bin_rows.append(
                    {
                        "mode": mode_name,
                        "period_bin": str(period_bin),
                        "rescued_unique_epics": int(count),
                    }
                )
        rescued_by_bin_df = (
            pd.DataFrame(rescued_by_bin_rows)
            .sort_values(["mode", "period_bin"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )

        comparison_csv = out_dir / self.DEFAULT_COMPARISON_CSV
        rescued_csv = out_dir / self.DEFAULT_RESCUED_CSV
        rescued_by_bin_csv = out_dir / self.DEFAULT_RESCUED_BY_BIN_CSV
        comparison_df.to_csv(comparison_csv, index=False)
        rescued_df.to_csv(rescued_csv, index=False)
        rescued_by_bin_df.to_csv(rescued_by_bin_csv, index=False)

        detector_row = comparison_df.loc[
            comparison_df["mode"].astype(str).eq(str(K2ShortlistPeriodConfig.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE_NAME))
        ].iloc[0]
        mcc2_row = comparison_df.loc[
            comparison_df["mode"].astype(str).eq(str(K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME))
        ].iloc[0]
        detector_15_20 = int(
            rescued_df.loc[
                rescued_df["rescued_by_detector_experimental"].fillna(False).astype(bool),
                "period_bin",
            ]
            .fillna("")
            .astype(str)
            .eq("(15,20]")
            .sum()
        )
        mcc2_15_20 = int(
            rescued_df.loc[
                rescued_df["rescued_by_mcc2"].fillna(False).astype(bool),
                "period_bin",
            ]
            .fillna("")
            .astype(str)
            .eq("(15,20]")
            .sum()
        )

        return {
            "detector_mode_comparison_csv": comparison_csv,
            "rescued_by_detector_mode_csv": rescued_csv,
            "rescued_by_detector_mode_by_period_bin_csv": rescued_by_bin_csv,
            "detector_added_vs_mcc2": int(detector_row["added_vs_mcc2"]),
            "period_bin_15_20_delta_vs_mcc2": int(detector_15_20 - mcc2_15_20),
            "zero_event_delta_vs_mcc2": int(detector_row["zero_event_count"] - mcc2_row["zero_event_count"]),
            "insufficient_support_delta_vs_mcc2": int(detector_row["insufficient_support_count"] - mcc2_row["insufficient_support_count"]),
            "manual_review_delta_vs_mcc2": int(detector_row["manual_review_count"] - mcc2_row["manual_review_count"]),
        }
