from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2StageCActionQueue:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_B_UNRESOLVED_CSV = DEFAULT_OUT_DIR / "k2_stage_b_master_unresolved_manifest.csv"
    DEFAULT_BATCH_RESULTS_CSV = DEFAULT_OUT_DIR / "batch_results.csv"
    DEFAULT_FUNNEL_CSV = DEFAULT_OUT_DIR / "epic_funnel_reasons.csv"
    DEFAULT_ACTION_QUEUE_CSV_NAME = "k2_stage_c_action_queue.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_c_action_queue_summary.csv"

    ACTION_PROCESS_NOW = "process_now"
    ACTION_BLOCKED_MISSING_LIGHT_CURVE = "blocked_missing_light_curve"
    ACTION_OUTSIDE_SCOPE = "outside_scope"
    ACTION_NEEDS_MANUAL_REVIEW = "needs_manual_review"
    ACTION_RESCUE_PATH_CANDIDATE = "rescue_path_candidate"
    ACTION_LOW_PRIORITY_OR_DEFER = "low_priority_or_defer"

    PRIORITY_HIGH = "high"
    PRIORITY_MEDIUM = "medium"
    PRIORITY_LOW = "low"
    PRIORITY_BLOCKED = "blocked"
    PRIORITY_EXCLUDED = "excluded"

    DATA_AVAILABLE = "light_curve_available_in_saved_outputs"
    DATA_MISSING_LIGHT_CURVE = "missing_light_curve"

    SCOPE_IN_SCOPE = "in_scope_unresolved"
    SCOPE_OUTSIDE = "outside_scope"

    RESCUE_SNR_THRESHOLD = 7.0

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Build the Stage C K2 unresolved routing layer from existing Stage B and saved detector/funnel CSVs."
            )
        )
        p.add_argument("--stage-b-unresolved-csv", type=Path, default=cls.DEFAULT_STAGE_B_UNRESOLVED_CSV)
        p.add_argument("--batch-results-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_CSV)
        p.add_argument("--funnel-csv", type=Path, default=cls.DEFAULT_FUNNEL_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--action-queue-csv", type=Path, default=None)
        p.add_argument("--summary-csv", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            stage_b_unresolved_csv=Path(args.stage_b_unresolved_csv),
            batch_results_csv=Path(args.batch_results_csv),
            funnel_csv=Path(args.funnel_csv),
            action_queue_csv=Path(args.action_queue_csv)
            if args.action_queue_csv is not None
            else out_dir / cls.DEFAULT_ACTION_QUEUE_CSV_NAME,
            summary_csv=Path(args.summary_csv) if args.summary_csv is not None else out_dir / cls.DEFAULT_SUMMARY_CSV_NAME,
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
    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _to_float(value: Any) -> float:
        series = pd.to_numeric(pd.Series([value]), errors="coerce")
        num = series.iloc[0]
        if pd.isna(num):
            return float("nan")
        return float(num)

    def _canonical_epic(self, value: Any) -> str:
        return self.helper._canonical_epic(value)

    def _prepare_stage_b_unresolved(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        required = ["epic_id", "epic_id_norm", "current_status_bucket", "unresolved"]
        missing = [c for c in required if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage B unresolved CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].astype(str)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out

    def _prepare_batch(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        if "epic_id" not in df.columns and "query" not in df.columns:
            raise ValueError(f"batch results CSV missing epic_id/query columns: {path}")
        epic_col = "epic_id" if "epic_id" in df.columns else "query"
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self._canonical_epic)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out

    def _prepare_funnel(self, path: Path) -> pd.DataFrame:
        raw = self._read_required_csv(path)
        expanded = self.helper._expand_funnel_details(raw)
        if "epic_id" not in expanded.columns and "query" not in expanded.columns:
            raise ValueError(f"funnel CSV missing epic_id/query columns: {path}")
        epic_col = "epic_id" if "epic_id" in expanded.columns else "query"
        expanded["epic_id_norm"] = expanded[epic_col].map(self._canonical_epic)
        return (
            expanded.loc[expanded["epic_id_norm"] != ""]
            .drop_duplicates(subset=["epic_id_norm"], keep="first")
            .reset_index(drop=True)
        )

    @staticmethod
    def _lookup_row(df: pd.DataFrame, epic_id_norm: str) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype=object)
        sub = df.loc[df["epic_id_norm"].astype(str).eq(str(epic_id_norm))]
        if len(sub) == 0:
            return pd.Series(dtype=object)
        return sub.iloc[0]

    def _classify_row(self, *, stage_b_row: pd.Series, batch_row: pd.Series, funnel_row: pd.Series) -> Dict[str, str]:
        outside_scope = self._to_bool(stage_b_row.get("outside_current_scope", False))
        load_failed = self._to_bool(stage_b_row.get("load_failed_or_missing_light_curve", False)) or (
            self._first_nonempty_text(funnel_row.get("terminal_reason", "")).strip().lower() == "no_lightcurve/load_failed"
        )
        triage_usable = self._to_bool(batch_row.get("triage_usable", stage_b_row.get("batch_triage_usable", False)))
        triage_why_not_usable = self._first_nonempty_text(
            batch_row.get("triage_why_not_usable", ""),
            stage_b_row.get("batch_triage_why_not_usable", ""),
        )
        n_events = self._to_float(batch_row.get("n_events", pd.NA))
        n_periods_proposed = self._to_float(batch_row.get("n_periods_proposed", pd.NA))
        best_depth_snr = self._to_float(batch_row.get("best_depth_snr", pd.NA))

        if outside_scope:
            return {
                "current_status": "excluded_outside_scope",
                "data_availability": self.DATA_AVAILABLE,
                "scope_status": self.SCOPE_OUTSIDE,
                "next_action": self.ACTION_OUTSIDE_SCOPE,
                "priority": self.PRIORITY_EXCLUDED,
                "routing_rule_id": "outside_scope_from_stage_b",
                "priority_rule_id": "excluded_priority_for_scope_exclusion",
                "reason_detail": "Stage B flagged this EPIC as outside current scope.",
            }
        if load_failed:
            return {
                "current_status": "blocked_missing_light_curve",
                "data_availability": self.DATA_MISSING_LIGHT_CURVE,
                "scope_status": self.SCOPE_IN_SCOPE,
                "next_action": self.ACTION_BLOCKED_MISSING_LIGHT_CURVE,
                "priority": self.PRIORITY_BLOCKED,
                "routing_rule_id": "load_failed_from_stage_b_or_funnel",
                "priority_rule_id": "blocked_priority_for_missing_light_curve",
                "reason_detail": self._first_nonempty_text(
                    f"Saved upstream data marks light-curve access as missing/load_failed; terminal_reason={funnel_row.get('terminal_reason', '')}",
                    f"stage_b_load_failed_flag={stage_b_row.get('load_failed_or_missing_light_curve', False)}",
                ),
            }

        if (not triage_usable) and (
            (pd.notna(n_periods_proposed) and n_periods_proposed > 0.0)
            or (pd.notna(best_depth_snr) and best_depth_snr >= self.RESCUE_SNR_THRESHOLD)
        ):
            return {
                "current_status": "quality_flagged_but_signal_present",
                "data_availability": self.DATA_AVAILABLE,
                "scope_status": self.SCOPE_IN_SCOPE,
                "next_action": self.ACTION_RESCUE_PATH_CANDIDATE,
                "priority": self.PRIORITY_HIGH,
                "routing_rule_id": "quality_flag_with_saved_signal",
                "priority_rule_id": "high_priority_rescue_queue",
                "reason_detail": (
                    f"triage_usable=False due to '{triage_why_not_usable}'; "
                    f"n_periods_proposed={int(n_periods_proposed) if pd.notna(n_periods_proposed) else 'NA'}; "
                    f"best_depth_snr={best_depth_snr:.3f}."
                ),
            }

        if not triage_usable:
            return {
                "current_status": "quality_flagged_needs_manual_review",
                "data_availability": self.DATA_AVAILABLE,
                "scope_status": self.SCOPE_IN_SCOPE,
                "next_action": self.ACTION_NEEDS_MANUAL_REVIEW,
                "priority": self.PRIORITY_MEDIUM,
                "routing_rule_id": "quality_flag_without_saved_signal",
                "priority_rule_id": "medium_priority_manual_review",
                "reason_detail": (
                    f"triage_usable=False due to '{triage_why_not_usable}'; "
                    f"n_periods_proposed={int(n_periods_proposed) if pd.notna(n_periods_proposed) else 'NA'}; "
                    f"best_depth_snr={best_depth_snr:.3f}."
                ),
            }

        if (pd.notna(n_events) and n_events <= 1.0) and (pd.notna(n_periods_proposed) and n_periods_proposed <= 0.0):
            return {
                "current_status": "single_event_no_period_signal",
                "data_availability": self.DATA_AVAILABLE,
                "scope_status": self.SCOPE_IN_SCOPE,
                "next_action": self.ACTION_LOW_PRIORITY_OR_DEFER,
                "priority": self.PRIORITY_LOW,
                "routing_rule_id": "single_event_and_no_proposed_periods",
                "priority_rule_id": "low_priority_defer_queue",
                "reason_detail": (
                    f"triage_usable=True but saved detector outputs show n_events={int(n_events)} and "
                    f"n_periods_proposed={int(n_periods_proposed)}."
                ),
            }

        if (pd.notna(n_periods_proposed) and n_periods_proposed > 0.0) or (
            pd.notna(best_depth_snr) and best_depth_snr >= self.RESCUE_SNR_THRESHOLD
        ):
            return {
                "current_status": "ready_for_default_pass_high_signal",
                "data_availability": self.DATA_AVAILABLE,
                "scope_status": self.SCOPE_IN_SCOPE,
                "next_action": self.ACTION_PROCESS_NOW,
                "priority": self.PRIORITY_HIGH,
                "routing_rule_id": "usable_with_saved_signal_support",
                "priority_rule_id": "high_priority_process_now",
                "reason_detail": (
                    f"triage_usable=True; n_events={int(n_events) if pd.notna(n_events) else 'NA'}; "
                    f"n_periods_proposed={int(n_periods_proposed) if pd.notna(n_periods_proposed) else 'NA'}; "
                    f"best_depth_snr={best_depth_snr:.3f}."
                ),
            }

        return {
            "current_status": "ready_for_default_pass",
            "data_availability": self.DATA_AVAILABLE,
            "scope_status": self.SCOPE_IN_SCOPE,
            "next_action": self.ACTION_PROCESS_NOW,
            "priority": self.PRIORITY_MEDIUM,
            "routing_rule_id": "usable_default_queue",
            "priority_rule_id": "medium_priority_process_now",
            "reason_detail": (
                f"triage_usable=True; n_events={int(n_events) if pd.notna(n_events) else 'NA'}; "
                f"n_periods_proposed={int(n_periods_proposed) if pd.notna(n_periods_proposed) else 'NA'}; "
                "no saved quality block or explicit exclusion."
            ),
        }

    def run(
        self,
        *,
        stage_b_unresolved_csv: Path,
        batch_results_csv: Path,
        funnel_csv: Path,
        action_queue_csv: Path,
        summary_csv: Path,
    ) -> Dict[str, Any]:
        stage_b = self._prepare_stage_b_unresolved(Path(stage_b_unresolved_csv))
        batch = self._prepare_batch(Path(batch_results_csv))
        funnel = self._prepare_funnel(Path(funnel_csv))

        rows: List[Dict[str, Any]] = []
        for _, stage_b_row in stage_b.iterrows():
            epic_id_norm = str(stage_b_row["epic_id_norm"])
            batch_row = self._lookup_row(batch, epic_id_norm)
            funnel_row = self._lookup_row(funnel, epic_id_norm)
            route = self._classify_row(stage_b_row=stage_b_row, batch_row=batch_row, funnel_row=funnel_row)
            query = self._first_nonempty_text(
                batch_row.get("query", ""),
                f"EPIC {epic_id_norm}",
            )
            rows.append(
                {
                    "epic_id": stage_b_row.get("epic_id", f"EPIC_{epic_id_norm}"),
                    "query": query,
                    "current_status": route["current_status"],
                    "stage_b_bucket": stage_b_row.get("current_status_bucket", ""),
                    "data_availability": route["data_availability"],
                    "scope_status": route["scope_status"],
                    "next_action": route["next_action"],
                    "priority": route["priority"],
                    "reason_detail": route["reason_detail"],
                    "source_file": "|".join(
                        [
                            str(Path(stage_b_unresolved_csv)),
                            str(Path(batch_results_csv)),
                            str(Path(funnel_csv)),
                        ]
                    ),
                    "epic_id_norm": epic_id_norm,
                    "routing_rule_id": route["routing_rule_id"],
                    "priority_rule_id": route["priority_rule_id"],
                    "derived_from_columns": (
                        "current_status_bucket|outside_current_scope|load_failed_or_missing_light_curve|"
                        "triage_usable|triage_why_not_usable|n_events|n_periods_proposed|best_depth_snr|"
                        "period_source_reason|period_terminal_reason"
                    ),
                    "period_source_reason": self._first_nonempty_text(stage_b_row.get("period_source_reason", "")),
                    "period_terminal_reason": self._first_nonempty_text(stage_b_row.get("period_terminal_reason", "")),
                    "triage_status": self._first_nonempty_text(
                        batch_row.get("triage_status", ""), stage_b_row.get("batch_triage_status", "")
                    ),
                    "triage_usable": self._to_bool(
                        batch_row.get("triage_usable", stage_b_row.get("batch_triage_usable", False))
                    ),
                    "triage_why_not_usable": self._first_nonempty_text(
                        batch_row.get("triage_why_not_usable", ""),
                        stage_b_row.get("batch_triage_why_not_usable", ""),
                    ),
                    "n_events": self._to_float(batch_row.get("n_events", pd.NA)),
                    "n_periods_proposed": self._to_float(batch_row.get("n_periods_proposed", pd.NA)),
                    "best_depth_snr": self._to_float(batch_row.get("best_depth_snr", pd.NA)),
                }
            )

        queue_df = pd.DataFrame(rows).sort_values(
            by=["priority", "next_action", "n_periods_proposed", "best_depth_snr", "n_events", "epic_id_norm"],
            ascending=[True, True, False, False, False, True],
        ).reset_index(drop=True)

        summary_rows: List[Dict[str, Any]] = []
        for dimension in ["stage_b_bucket", "next_action", "priority", "data_availability", "scope_status"]:
            counts = queue_df[dimension].fillna("").astype(str).value_counts().sort_index()
            for value, count in counts.items():
                summary_rows.append(
                    {
                        "section": "counts",
                        "dimension": dimension,
                        "value": value,
                        "count": int(count),
                    }
                )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "outside_scope_from_stage_b",
                "count": 0,
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "load_failed_from_stage_b_or_funnel",
                "count": 0,
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "quality_flag_with_saved_signal",
                "count": int(queue_df["routing_rule_id"].eq("quality_flag_with_saved_signal").sum()),
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "quality_flag_without_saved_signal",
                "count": int(queue_df["routing_rule_id"].eq("quality_flag_without_saved_signal").sum()),
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "single_event_and_no_proposed_periods",
                "count": int(queue_df["routing_rule_id"].eq("single_event_and_no_proposed_periods").sum()),
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "usable_with_saved_signal_support",
                "count": int(queue_df["routing_rule_id"].eq("usable_with_saved_signal_support").sum()),
            }
        )
        summary_rows.append(
            {
                "section": "derivation",
                "dimension": "routing_rule",
                "value": "usable_default_queue",
                "count": int(queue_df["routing_rule_id"].eq("usable_default_queue").sum()),
            }
        )
        summary_df = pd.DataFrame(summary_rows)

        action_queue_csv = Path(action_queue_csv)
        summary_csv = Path(summary_csv)
        action_queue_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        queue_df.to_csv(action_queue_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

        return {
            "action_queue_csv": str(action_queue_csv),
            "summary_csv": str(summary_csv),
            "rows_total": int(len(queue_df)),
            "process_now_count": int(queue_df["next_action"].eq(self.ACTION_PROCESS_NOW).sum()),
            "blocked_missing_light_curve_count": int(
                queue_df["next_action"].eq(self.ACTION_BLOCKED_MISSING_LIGHT_CURVE).sum()
            ),
            "outside_scope_count": int(queue_df["next_action"].eq(self.ACTION_OUTSIDE_SCOPE).sum()),
            "needs_manual_review_count": int(queue_df["next_action"].eq(self.ACTION_NEEDS_MANUAL_REVIEW).sum()),
            "rescue_path_candidate_count": int(
                queue_df["next_action"].eq(self.ACTION_RESCUE_PATH_CANDIDATE).sum()
            ),
            "low_priority_or_defer_count": int(queue_df["next_action"].eq(self.ACTION_LOW_PRIORITY_OR_DEFER).sum()),
        }
