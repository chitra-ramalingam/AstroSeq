from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2KnownConfirmedFalseNegativeAudit:
    DEFAULT_CURRENT_RUN_DIR = Path(
        r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream_conditional_mcc2_experiment"
    )
    DEFAULT_BASELINE_RUN_DIR = Path(r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream")
    DEFAULT_ANALYSIS_CSV_NAME = "known_confirmed_false_negative_audit.csv"
    DEFAULT_REPORT_TXT_NAME = "known_confirmed_false_negative_audit.txt"
    DEFAULT_EPIC_IDS = ["200008693"]

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Audit known confirmed K2 planets against saved shortlist/funnel outputs without rerunning any "
                "detector or downstream stages."
            )
        )
        p.add_argument(
            "--epic-id",
            dest="epic_ids",
            action="append",
            default=None,
            help="EPIC id to audit. Repeat for multiple EPICs. Default: EPIC 200008693",
        )
        p.add_argument("--current-run-dir", type=Path, default=cls.DEFAULT_CURRENT_RUN_DIR)
        p.add_argument("--baseline-run-dir", type=Path, default=cls.DEFAULT_BASELINE_RUN_DIR)
        p.add_argument("--analysis-csv", type=Path, default=None)
        p.add_argument("--report-txt", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        current_run_dir = Path(args.current_run_dir)
        epic_ids = list(args.epic_ids) if args.epic_ids else list(cls.DEFAULT_EPIC_IDS)
        return cls().run(
            epic_ids=epic_ids,
            current_run_dir=current_run_dir,
            baseline_run_dir=Path(args.baseline_run_dir),
            analysis_csv=Path(args.analysis_csv)
            if args.analysis_csv is not None
            else current_run_dir / cls.DEFAULT_ANALYSIS_CSV_NAME,
            report_txt=Path(args.report_txt)
            if args.report_txt is not None
            else current_run_dir / cls.DEFAULT_REPORT_TXT_NAME,
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
    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "t", "yes", "y"}

    def _prepare_table(self, df: pd.DataFrame, *, epic_col: str, label: str) -> pd.DataFrame:
        if epic_col not in df.columns:
            raise ValueError(f"{label} CSV missing required column: {epic_col}")
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self.helper._canonical_epic)
        out = out.loc[out["epic_id_norm"] != ""].reset_index(drop=True)
        return out

    def _prepare_best(self, path: Path) -> pd.DataFrame:
        best = self._prepare_table(self._read_required_csv(path), epic_col="epic", label="best")
        return best.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_quarantine(self, path: Path) -> pd.DataFrame:
        quarantine = self._prepare_table(self._read_required_csv(path), epic_col="epic_id", label="quarantine")
        return quarantine.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_funnel(self, path: Path) -> pd.DataFrame:
        funnel = self.helper._expand_funnel_details(self._read_required_csv(path))
        funnel = self._prepare_table(funnel, epic_col="epic_id", label="funnel")
        return funnel.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_merged(self, path: Path) -> pd.DataFrame:
        merged = self._read_required_csv(path)
        if "epic_id" in merged.columns:
            merged["epic_id_norm"] = merged["epic_id"].map(self.helper._canonical_epic)
        elif "query" in merged.columns:
            merged["epic_id_norm"] = merged["query"].map(self.helper._canonical_epic)
        else:
            raise ValueError(f"merged batch CSV missing epic_id/query columns: {path}")
        merged = merged.loc[merged["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return merged

    @staticmethod
    def _diagnostic_value(df: pd.DataFrame, column: str) -> Any:
        if len(df) == 0 or column not in df.columns:
            return pd.NA
        return df.iloc[0][column]

    def _load_policy_state(
        self,
        *,
        run_dir: Path,
        best_name: str,
        quarantine_name: str,
        funnel_name: str,
        diagnostics_name: str,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir).resolve()
        return {
            "run_dir": run_dir,
            "best": self._prepare_best(run_dir / best_name),
            "quarantine": self._prepare_quarantine(run_dir / quarantine_name),
            "funnel": self._prepare_funnel(run_dir / funnel_name),
            "diagnostics": self._read_required_csv(run_dir / diagnostics_name),
            "merged": self._prepare_merged(run_dir / "merged_batch_results.csv"),
        }

    @staticmethod
    def _lookup_first(df: pd.DataFrame, epic_id_norm: str) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype=object)
        sub = df.loc[df["epic_id_norm"].astype(str).eq(str(epic_id_norm))]
        if len(sub) == 0:
            return pd.Series(dtype=object)
        return sub.iloc[0]

    @staticmethod
    def _outcome_group(in_best: bool, in_quarantine: bool) -> str:
        if in_best and in_quarantine:
            return "both"
        if in_best:
            return "best"
        if in_quarantine:
            return "quarantine"
        return "neither"

    def _primary_rejection_bucket(
        self,
        *,
        terminal_reason: str,
        failure_category: str,
        failure_detail: str,
        shortlist_rejection_reason: str,
        hist_in_period_range: float,
        hist_pass_cluster_count: float,
    ) -> str:
        terminal = terminal_reason.strip().lower()
        failure = failure_category.strip().lower()
        detail = failure_detail.strip().lower()
        shortlist = shortlist_rejection_reason.strip().lower()
        if (
            "min_cluster_count" in detail
            or "below_min_cluster_count" in detail
            or (
                failure == "candidate_filter_rejection"
                and pd.notna(hist_in_period_range)
                and hist_in_period_range > 0.0
                and pd.notna(hist_pass_cluster_count)
                and hist_pass_cluster_count <= 0.0
            )
        ):
            return "minimum cluster count"
        if terminal == "period_over_cap" or "above_max_period" in detail or "period_cap" in detail:
            return "period cap"
        if failure == "candidate_filter_rejection" or shortlist == "candidate_filter_rejection":
            return "candidate filter policy"
        return "something else"

    @staticmethod
    def _format_flag(value: bool) -> str:
        return "yes" if bool(value) else "no"

    def _build_row(self, *, epic_id_norm: str, current_state: Dict[str, Any], baseline_state: Dict[str, Any]) -> Dict[str, Any]:
        current_best = self._lookup_first(current_state["best"], epic_id_norm)
        current_quarantine = self._lookup_first(current_state["quarantine"], epic_id_norm)
        current_funnel = self._lookup_first(current_state["funnel"], epic_id_norm)
        current_merged = self._lookup_first(current_state["merged"], epic_id_norm)

        baseline_best = self._lookup_first(baseline_state["best"], epic_id_norm)
        baseline_quarantine = self._lookup_first(baseline_state["quarantine"], epic_id_norm)
        baseline_funnel = self._lookup_first(baseline_state["funnel"], epic_id_norm)
        baseline_merged = self._lookup_first(baseline_state["merged"], epic_id_norm)

        detector_row = current_merged if len(current_merged) > 0 else baseline_merged

        current_in_best = len(current_best) > 0
        current_in_quarantine = len(current_quarantine) > 0
        baseline_in_best = len(baseline_best) > 0
        baseline_in_quarantine = len(baseline_quarantine) > 0

        current_query = self._first_nonempty_text(
            detector_row.get("query", ""),
            current_funnel.get("query", ""),
            current_quarantine.get("query", ""),
            current_best.get("query", ""),
            f"EPIC {epic_id_norm}",
        )

        current_n_events_after_filters = self._first_numeric(
            current_quarantine.get("n_events_after_filters", pd.NA),
            current_funnel.get("period_n_events_after_filters", pd.NA),
            current_funnel.get("n_events", pd.NA),
            current_best.get("n_events_after_filters", pd.NA),
        )
        current_hist_in_period_range = self._first_numeric(
            current_quarantine.get("hist_in_period_range", pd.NA),
            current_funnel.get("period_hist_in_period_range", pd.NA),
        )
        current_hist_pass_cluster_count = self._first_numeric(
            current_quarantine.get("hist_pass_cluster_count", pd.NA),
            current_funnel.get("period_hist_pass_cluster_count", pd.NA),
        )
        current_min_cluster_count = self._first_numeric(
            current_quarantine.get("min_cluster_count", pd.NA),
            current_funnel.get("period_min_cluster_count", pd.NA),
            self._diagnostic_value(current_state["diagnostics"], "min_cluster_count"),
        )
        current_period_cap_days = self._first_numeric(
            current_quarantine.get("period_cap_days", pd.NA),
            current_funnel.get("period_cap_days", pd.NA),
            self._diagnostic_value(current_state["diagnostics"], "period_cap_days"),
        )
        current_n_periods_proposed = self._first_numeric(current_funnel.get("n_periods_proposed", pd.NA))
        current_n_periods_validated = self._first_numeric(current_funnel.get("n_periods_validated", pd.NA))
        current_failure_category = self._first_nonempty_text(
            current_quarantine.get("failure_category", ""),
            current_funnel.get("period_failure_category", ""),
        )
        current_failure_detail = self._first_nonempty_text(
            current_quarantine.get("failure_detail", ""),
            current_funnel.get("period_failure_detail", ""),
        )
        current_terminal_reason = self._first_nonempty_text(current_funnel.get("terminal_reason", ""))
        current_shortlist_rejection_reason = self._first_nonempty_text(
            current_quarantine.get("shortlist_rejection_reason", ""),
            current_funnel.get("shortlist_rejection_reason", ""),
        )
        current_shortlist_rejection_stage = self._first_nonempty_text(
            current_quarantine.get("shortlist_rejection_stage", ""),
            current_funnel.get("shortlist_rejection_stage", ""),
        )
        current_stage_reached = self._first_nonempty_text(current_funnel.get("stage_reached", ""))
        current_selected_for_period_stage = self._to_bool(current_funnel.get("selected_for_period_stage", False))
        current_policy_mode = self._first_nonempty_text(
            self._diagnostic_value(current_state["diagnostics"], "operating_mode_requested")
        )

        baseline_policy_mode = self._first_nonempty_text(
            self._diagnostic_value(baseline_state["diagnostics"], "operating_mode_requested")
        )
        baseline_period_cap_days = self._first_numeric(
            self._diagnostic_value(baseline_state["diagnostics"], "period_cap_days")
        )
        baseline_min_cluster_count = self._first_numeric(
            self._diagnostic_value(baseline_state["diagnostics"], "min_cluster_count")
        )

        conditional_relax_enabled_raw = self._diagnostic_value(
            current_state["diagnostics"], "conditional_min_cluster_count_relax_enabled"
        )
        conditional_relax_to = self._first_numeric(
            self._diagnostic_value(current_state["diagnostics"], "conditional_min_cluster_count_relax_to")
        )
        conditional_min_events = self._first_numeric(
            self._diagnostic_value(current_state["diagnostics"], "conditional_min_cluster_count_min_events_after_filters")
        )
        conditional_min_hist = self._first_numeric(
            self._diagnostic_value(current_state["diagnostics"], "conditional_min_cluster_count_min_hist_in_range")
        )
        conditional_relax_enabled = self._to_bool(conditional_relax_enabled_raw) or (
            current_policy_mode == "scale_validation_conditional_mcc2_experiment"
            and pd.notna(conditional_relax_to)
            and conditional_relax_to <= current_min_cluster_count
        )
        conditional_eligible = (
            conditional_relax_enabled
            and pd.notna(current_n_events_after_filters)
            and pd.notna(current_hist_in_period_range)
            and current_n_events_after_filters >= conditional_min_events
            and current_hist_in_period_range >= conditional_min_hist
        )
        if not conditional_relax_enabled:
            conditional_reason = "Current saved run does not advertise a conditional MCC=2 carve-out."
        elif not pd.notna(current_n_events_after_filters):
            conditional_reason = "Current saved diagnostics do not expose n_events_after_filters for this EPIC."
        elif current_n_events_after_filters < conditional_min_events:
            conditional_reason = (
                f"Not eligible: n_events_after_filters={int(current_n_events_after_filters)} < {int(conditional_min_events)}."
            )
        elif not pd.notna(current_hist_in_period_range):
            conditional_reason = "Current saved diagnostics do not expose hist_in_period_range for this EPIC."
        elif current_hist_in_period_range < conditional_min_hist:
            conditional_reason = (
                f"Not eligible: hist_in_period_range={current_hist_in_period_range:.0f} < {int(conditional_min_hist)}."
            )
        else:
            conditional_reason = "Eligible for the conditional MCC=2 carve-out under the saved diagnostics."

        primary_rejection_bucket = self._primary_rejection_bucket(
            terminal_reason=current_terminal_reason,
            failure_category=current_failure_category,
            failure_detail=current_failure_detail,
            shortlist_rejection_reason=current_shortlist_rejection_reason,
            hist_in_period_range=current_hist_in_period_range,
            hist_pass_cluster_count=current_hist_pass_cluster_count,
        )
        if primary_rejection_bucket == "period cap":
            larger_period_cap_help = True
            larger_period_cap_reason = "Possible yes: existing diagnostics indicate a period-cap rejection."
        else:
            larger_period_cap_help = False
            larger_period_cap_reason = "No: existing diagnostics do not show a period-cap rejection for this EPIC."

        current_period_inference_status = self._first_nonempty_text(
            current_stage_reached,
            "not_reached",
        )
        if current_selected_for_period_stage and current_terminal_reason == "no_cluster_periods":
            current_period_inference_status = "selected_for_period_stage_but_no_cluster_periods"
        elif current_in_best:
            current_period_inference_status = "produced_shortlist_candidate"

        rejection_gate_summary = " | ".join(
            [
                x
                for x in [
                    f"stage={current_shortlist_rejection_stage}" if current_shortlist_rejection_stage else "",
                    f"reason={current_shortlist_rejection_reason}" if current_shortlist_rejection_reason else "",
                    f"failure_category={current_failure_category}" if current_failure_category else "",
                    f"failure_detail={current_failure_detail}" if current_failure_detail else "",
                    f"terminal_reason={current_terminal_reason}" if current_terminal_reason else "",
                    f"min_cluster_count={int(current_min_cluster_count)}" if pd.notna(current_min_cluster_count) else "",
                ]
                if x != ""
            ]
        )

        return {
            "epic_id": f"EPIC_{epic_id_norm}",
            "epic_id_norm": epic_id_norm,
            "query": current_query,
            "detector_present_in_saved_outputs": bool(len(detector_row) > 0),
            "detector_triage_status": self._first_nonempty_text(detector_row.get("triage_status", "")),
            "detector_n_events": self._first_numeric(detector_row.get("n_events", pd.NA)),
            "detector_best_shape_score": self._first_numeric(detector_row.get("best_shape_score", pd.NA)),
            "detector_best_depth_snr": self._first_numeric(detector_row.get("best_depth_snr", pd.NA)),
            "current_policy_mode_requested": current_policy_mode,
            "current_outcome_group": self._outcome_group(current_in_best, current_in_quarantine),
            "current_in_best": bool(current_in_best),
            "current_in_quarantine": bool(current_in_quarantine),
            "current_selected_for_period_stage": bool(current_selected_for_period_stage),
            "current_period_inference_status": current_period_inference_status,
            "current_stage_reached": current_stage_reached,
            "current_terminal_reason": current_terminal_reason,
            "current_shortlist_rejection_stage": current_shortlist_rejection_stage,
            "current_shortlist_rejection_reason": current_shortlist_rejection_reason,
            "current_failure_category": current_failure_category,
            "current_failure_detail": current_failure_detail,
            "current_n_events_after_filters": current_n_events_after_filters,
            "current_hist_in_period_range": current_hist_in_period_range,
            "current_hist_pass_cluster_count": current_hist_pass_cluster_count,
            "current_n_periods_proposed": current_n_periods_proposed,
            "current_n_periods_validated": current_n_periods_validated,
            "current_min_cluster_count": current_min_cluster_count,
            "current_period_cap_days": current_period_cap_days,
            "current_rejection_gate_summary": rejection_gate_summary,
            "primary_rejection_bucket": primary_rejection_bucket,
            "saved_default_policy_mode_requested": baseline_policy_mode,
            "saved_default_outcome_group": self._outcome_group(baseline_in_best, baseline_in_quarantine),
            "saved_default_in_best": bool(baseline_in_best),
            "saved_default_in_quarantine": bool(baseline_in_quarantine),
            "saved_default_best_reason": self._first_nonempty_text(baseline_best.get("reason", "")),
            "saved_default_best_P": self._first_numeric(baseline_best.get("P", pd.NA)),
            "saved_default_best_cluster_count": self._first_numeric(baseline_best.get("cluster_count", pd.NA)),
            "saved_default_best_manual_review_required": self._to_bool(baseline_best.get("manual_review_required", False)),
            "saved_default_terminal_reason": self._first_nonempty_text(baseline_funnel.get("terminal_reason", "")),
            "saved_default_failure_category": self._first_nonempty_text(
                baseline_quarantine.get("failure_category", ""),
                baseline_funnel.get("period_failure_category", ""),
            ),
            "saved_default_failure_detail": self._first_nonempty_text(
                baseline_quarantine.get("failure_detail", ""),
                baseline_funnel.get("period_failure_detail", ""),
            ),
            "saved_default_min_cluster_count": baseline_min_cluster_count,
            "saved_default_period_cap_days": baseline_period_cap_days,
            "survives_under_saved_default_policy": bool(baseline_in_best),
            "survives_under_conditional_mcc2_carveout": bool(current_in_best),
            "conditional_mcc2_carveout_eligible_from_existing_diagnostics": bool(conditional_eligible),
            "conditional_mcc2_carveout_eligibility_reason": conditional_reason,
            "survives_under_larger_period_cap_from_existing_diagnostics": bool(larger_period_cap_help),
            "larger_period_cap_assessment": larger_period_cap_reason,
        }

    def _write_report_txt(self, rows: List[Dict[str, Any]], report_txt: Path) -> None:
        lines: List[str] = []
        for row in rows:
            lines.extend(
                [
                    f"{row['epic_id']}",
                    (
                        "detector-side: "
                        f"present={self._format_flag(row['detector_present_in_saved_outputs'])} "
                        f"triage_status={row['detector_triage_status'] or 'unknown'} "
                        f"n_events={row['detector_n_events']} "
                        f"best_shape_score={row['detector_best_shape_score']} "
                        f"best_depth_snr={row['detector_best_depth_snr']}"
                    ),
                    (
                        "period-inference: "
                        f"current_mode={row['current_policy_mode_requested'] or 'unknown'} "
                        f"selected_for_period_stage={self._format_flag(row['current_selected_for_period_stage'])} "
                        f"status={row['current_period_inference_status']} "
                        f"n_events_after_filters={row['current_n_events_after_filters']} "
                        f"n_periods_proposed={row['current_n_periods_proposed']} "
                        f"n_periods_validated={row['current_n_periods_validated']}"
                    ),
                    (
                        "shortlist-outcome: "
                        f"current={row['current_outcome_group']} "
                        f"saved_default={row['saved_default_outcome_group']} "
                        f"saved_default_policy_mode={row['saved_default_policy_mode_requested'] or 'unknown'}"
                    ),
                    (
                        "exact_rejection_gates: "
                        f"{row['current_rejection_gate_summary'] or 'none recorded in current saved outputs'}"
                    ),
                    f"primary_rejection_bucket: {row['primary_rejection_bucket']}",
                    (
                        "policy_flags: "
                        f"saved_default_survive={self._format_flag(row['survives_under_saved_default_policy'])} "
                        f"conditional_mcc2_survive={self._format_flag(row['survives_under_conditional_mcc2_carveout'])} "
                        f"conditional_mcc2_eligible={self._format_flag(row['conditional_mcc2_carveout_eligible_from_existing_diagnostics'])} "
                        f"larger_period_cap_survive={self._format_flag(row['survives_under_larger_period_cap_from_existing_diagnostics'])}"
                    ),
                    f"conditional_mcc2_note: {row['conditional_mcc2_carveout_eligibility_reason']}",
                    f"larger_period_cap_note: {row['larger_period_cap_assessment']}",
                    "",
                ]
            )
        report_txt.parent.mkdir(parents=True, exist_ok=True)
        report_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def run(
        self,
        *,
        epic_ids: Sequence[str],
        current_run_dir: Path,
        baseline_run_dir: Path,
        analysis_csv: Path,
        report_txt: Path,
    ) -> Dict[str, Any]:
        current_state = self._load_policy_state(
            run_dir=current_run_dir,
            best_name="period_shortlist_best.csv",
            quarantine_name="period_shortlist_quarantine.csv",
            funnel_name="epic_funnel_reasons.csv",
            diagnostics_name="period_shortlist_diagnostics.csv",
        )
        baseline_state = self._load_policy_state(
            run_dir=baseline_run_dir,
            best_name="Apr1_period_shortlist_best.csv",
            quarantine_name="Apr1_period_shortlist_quarantine.csv",
            funnel_name="Apr1_epic_funnel_reasons.csv",
            diagnostics_name="Apr1_period_shortlist_diagnostics.csv",
        )

        rows = [
            self._build_row(
                epic_id_norm=self.helper._canonical_epic(epic_id),
                current_state=current_state,
                baseline_state=baseline_state,
            )
            for epic_id in epic_ids
        ]
        analysis_df = pd.DataFrame(rows).sort_values(by=["epic_id_norm"]).reset_index(drop=True)

        analysis_csv = Path(analysis_csv).resolve()
        report_txt = Path(report_txt).resolve()
        analysis_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(analysis_csv, index=False)
        self._write_report_txt(rows=analysis_df.to_dict(orient="records"), report_txt=report_txt)

        return {
            "analysis_csv": analysis_csv,
            "report_txt": report_txt,
            "epic_count": int(len(analysis_df)),
            "rows": analysis_df.to_dict(orient="records"),
        }
