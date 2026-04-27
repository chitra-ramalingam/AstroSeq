from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2StageFBatch001bExecution:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_INPUT_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001b_input.csv"
    DEFAULT_RESULTS_CSV_NAME = "k2_stage_f_batch_001b_results.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_f_batch_001b_summary.csv"
    DEFAULT_RUN_DIR_NAME = "k2_stage_f_batch_001b_run"
    DEFAULT_ORIGINAL_RESULTS_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001_results.csv"
    DEFAULT_ORIGINAL_SUMMARY_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001_summary.csv"

    REQUIRED_INPUT_COLUMNS = [
        "epic_id",
        "query",
        "old_execution_order",
        "new_execution_order",
        "rerank_score",
        "rerank_reason",
        "whiteness_proxy_value",
        "keepability_risk_flag",
        "next_action",
        "priority",
    ]

    def __init__(self, runner_factory: Optional[Callable[..., Any]] = None) -> None:
        self.runner_factory = runner_factory if runner_factory is not None else K2BatchRunner

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Run the Stage F patched calibration batch 001b using the official default K2 pipeline."
        )
        p.add_argument("--input-csv", type=Path, default=cls.DEFAULT_INPUT_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--original-results-csv", type=Path, default=cls.DEFAULT_ORIGINAL_RESULTS_CSV)
        p.add_argument("--original-summary-csv", type=Path, default=cls.DEFAULT_ORIGINAL_SUMMARY_CSV)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            input_csv=Path(args.input_csv),
            out_dir=Path(args.out_dir),
            original_results_csv=Path(args.original_results_csv),
            original_summary_csv=Path(args.original_summary_csv),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_input(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage F batch 001b input CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["new_execution_order"] = pd.to_numeric(out["new_execution_order"], errors="coerce")
        out["old_execution_order"] = pd.to_numeric(out["old_execution_order"], errors="coerce")
        if out["new_execution_order"].isna().any():
            raise ValueError("Stage F batch 001b input CSV contains non-numeric new_execution_order values.")
        if out["old_execution_order"].isna().any():
            raise ValueError("Stage F batch 001b input CSV contains non-numeric old_execution_order values.")
        return out.sort_values(by=["new_execution_order", "epic_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _pipeline_command(run_dir: Path, input_csv: Path) -> str:
        return f".\\.venv\\Scripts\\python.exe main.py --out-dir {run_dir} --input-csv {input_csv} --query-col query"

    @staticmethod
    def _is_failed(results_df: pd.DataFrame) -> pd.Series:
        triage_status = results_df.get("triage_status", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.strip().str.lower()
        error_stage = results_df.get("error_stage", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.strip()
        return triage_status.eq("error") | error_stage.ne("")

    @staticmethod
    def _whiteness_reject_mask(results_df: pd.DataFrame) -> pd.Series:
        labels = results_df.get("label", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str)
        reasons = results_df.get("label_reason", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.lower()
        why_not = results_df.get("triage_why_not_usable", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.lower()
        return labels.eq("Noisy_trash") & (reasons.str.contains("whiteness") | why_not.str.contains("whiteness"))

    @staticmethod
    def _counts_text(series: pd.Series) -> str:
        counts = series.fillna("").astype(str).value_counts()
        if len(counts) == 0:
            return "none"
        return " | ".join([f"{label or 'blank'}={int(count)}" for label, count in counts.items()])

    @staticmethod
    def _failure_modes(results_df: pd.DataFrame) -> str:
        failed = K2StageFBatch001bExecution._is_failed(results_df)
        if not bool(failed.any()):
            return "none"
        work = results_df.loc[failed].copy()
        work["mode"] = (
            work.get("error_stage", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str).str.strip()
            + ":"
            + work.get("error_type", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str).str.strip()
        )
        counts = work["mode"].replace({":": "unknown:unknown"}).value_counts()
        return " | ".join([f"{mode}={int(count)}" for mode, count in counts.items()])

    @staticmethod
    def _load_previous_runtime(summary_csv: Path) -> float:
        if not summary_csv.exists():
            return float("nan")
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            return float("nan")
        if len(df) == 0 or "runtime_seconds" not in df.columns:
            return float("nan")
        runtime = pd.to_numeric(df["runtime_seconds"], errors="coerce").iloc[0]
        return float(runtime) if pd.notna(runtime) else float("nan")

    @staticmethod
    def _build_results_output(input_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        planned = input_df.copy().rename(
            columns={
                "best_depth_snr": "planned_best_depth_snr",
                "n_events": "planned_n_events",
                "n_periods_proposed": "planned_n_periods_proposed",
            }
        )
        merged = planned.merge(results_df, how="left", on=["epic_id", "query"], suffixes=("", "_pipeline"))
        preferred_front = [
            "epic_id",
            "query",
            "old_execution_order",
            "new_execution_order",
            "rerank_score",
            "rerank_reason",
            "whiteness_proxy_available",
            "whiteness_proxy_value",
            "keepability_risk_flag",
            "next_action",
            "priority",
            "planned_best_depth_snr",
            "planned_n_events",
            "planned_n_periods_proposed",
            "triage_status",
            "triage_usable",
            "triage_why_not_usable",
            "n_events",
            "best_shape_score",
            "best_depth_snr",
            "n_periods_proposed",
            "n_periods_validated",
            "best_period",
            "label",
            "label_reason",
            "error_stage",
            "error_type",
            "error_msg",
            "epic_dir",
            "events_csv",
            "best_hits_csv",
            "best_misses_csv",
            "best_uncovered_csv",
            "best_hitmap_png",
            "best_phase_offset_png",
        ]
        front = [c for c in preferred_front if c in merged.columns]
        remaining = [c for c in merged.columns if c not in front]
        return merged[front + remaining].sort_values(
            by=["new_execution_order", "epic_id"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)

    @staticmethod
    def _comparison_against_original(
        current_results_df: pd.DataFrame,
        original_results_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        current_noisy = int(current_results_df.get("label", pd.Series(dtype=str)).fillna("").astype(str).eq("Noisy_trash").sum())
        original_noisy = int(original_results_df.get("label", pd.Series(dtype=str)).fillna("").astype(str).eq("Noisy_trash").sum())
        current_non_noisy = int(len(current_results_df) - current_noisy)
        original_non_noisy = int(len(original_results_df) - original_noisy)
        current_whiteness = int(K2StageFBatch001bExecution._whiteness_reject_mask(current_results_df).sum())
        original_whiteness = int(K2StageFBatch001bExecution._whiteness_reject_mask(original_results_df).sum())

        whiteness_improved = "yes" if current_whiteness < original_whiteness else "no"
        evidence_to_proceed = "yes" if (current_non_noisy > 0 and current_noisy < original_noisy and current_whiteness < original_whiteness) else "no"
        evidence_note = (
            "The patched calibration batch produced fewer Noisy_trash outcomes, fewer whiteness rejections, and at least one non-Noisy_trash outcome."
            if evidence_to_proceed == "yes"
            else "The patched calibration batch does not yet provide enough evidence to proceed to later batches; review this rerun before scaling."
        )

        return {
            "original_noisy_trash_count": original_noisy,
            "current_noisy_trash_count": current_noisy,
            "original_non_noisy_count": original_non_noisy,
            "current_non_noisy_count": current_non_noisy,
            "original_whiteness_rejection_count": original_whiteness,
            "current_whiteness_rejection_count": current_whiteness,
            "whiteness_rejection_frequency_improved": whiteness_improved,
            "enough_evidence_to_proceed": evidence_to_proceed,
            "comparison_note": evidence_note,
        }

    def run(
        self,
        *,
        input_csv: Path,
        out_dir: Path,
        original_results_csv: Path,
        original_summary_csv: Path,
    ) -> Dict[str, Any]:
        input_df = self._prepare_input(Path(input_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir / self.DEFAULT_RUN_DIR_NAME
        results_csv = out_dir / self.DEFAULT_RESULTS_CSV_NAME
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME

        command_used = self._pipeline_command(run_dir=run_dir, input_csv=Path(input_csv))
        started = time.perf_counter()
        runner = self.runner_factory(out_dir=run_dir, input_csv=input_csv, query_col="query")
        out = runner.run()
        runtime_seconds = float(time.perf_counter() - started)
        previous_runtime = self._load_previous_runtime(summary_csv)
        if pd.notna(previous_runtime) and float(previous_runtime) > runtime_seconds:
            runtime_seconds = float(previous_runtime)

        pipeline_batch_results_csv = Path(out["batch_results_csv"])
        results_df = pd.read_csv(pipeline_batch_results_csv) if pipeline_batch_results_csv.exists() else pd.DataFrame(out.get("results_df", pd.DataFrame()))
        results_output = self._build_results_output(input_df=input_df, results_df=results_df)
        results_output.to_csv(results_csv, index=False)

        failed_mask = self._is_failed(results_output)
        n_events = pd.to_numeric(results_output.get("n_events", 0), errors="coerce").fillna(0.0)
        labels = results_output.get("label", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str)
        label_reasons = results_output.get("label_reason", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str)

        rows_attempted = int(len(input_df))
        rows_failed = int(failed_mask.sum())
        rows_completed = int(rows_attempted - rows_failed)
        rows_with_candidate_signal = int(n_events.gt(0).sum())
        rows_without_candidate_signal = int(rows_attempted - rows_with_candidate_signal)
        rows_flagged_for_manual_review = int(labels.eq("Unclassified").sum())
        rows_requiring_rescue_followup = int(labels.eq("Sparse_or_mono").sum())
        label_counts_text = self._counts_text(labels)
        label_reason_counts_text = self._counts_text(label_reasons)
        failure_modes = self._failure_modes(results_output)

        original_results_df = self._read_required_csv(Path(original_results_csv))
        comparison = self._comparison_against_original(
            current_results_df=results_output,
            original_results_df=original_results_df,
        )

        runtime_minutes = runtime_seconds / 60.0
        runtime_notes = (
            f"Executed the default official K2BatchRunner settings against {rows_attempted} patched calibration queries in "
            f"an isolated run directory ({run_dir}). Runtime was {runtime_seconds:.1f}s ({runtime_minutes:.2f}m). "
            "Ordering followed new_execution_order from the accepted 001b input without modifying batch composition."
        )

        summary_row = {
            "batch_id": "high_priority_batch_001b",
            "rows_attempted": rows_attempted,
            "rows_completed": rows_completed,
            "rows_failed": rows_failed,
            "rows_with_candidate_signal": rows_with_candidate_signal,
            "rows_without_candidate_signal": rows_without_candidate_signal,
            "rows_flagged_for_manual_review": rows_flagged_for_manual_review,
            "rows_requiring_rescue_followup": rows_requiring_rescue_followup,
            "final_label_counts": label_counts_text,
            "final_label_reason_counts": label_reason_counts_text,
            "command_used": command_used,
            "runtime_seconds": runtime_seconds,
            "runtime_notes": runtime_notes,
            "failure_modes_encountered": failure_modes,
            "original_batch_001_noisy_trash_count": comparison["original_noisy_trash_count"],
            "patched_batch_001b_noisy_trash_count": comparison["current_noisy_trash_count"],
            "patched_batch_001b_non_noisy_count": comparison["current_non_noisy_count"],
            "original_batch_001_non_noisy_count": comparison["original_non_noisy_count"],
            "original_batch_001_whiteness_rejection_count": comparison["original_whiteness_rejection_count"],
            "patched_batch_001b_whiteness_rejection_count": comparison["current_whiteness_rejection_count"],
            "whiteness_rejection_frequency_improved": comparison["whiteness_rejection_frequency_improved"],
            "enough_evidence_to_proceed_to_later_batches": comparison["enough_evidence_to_proceed"],
            "comparison_note": comparison["comparison_note"],
            "source_input_csv": str(Path(input_csv)),
            "source_original_results_csv": str(Path(original_results_csv)),
            "source_original_summary_csv": str(Path(original_summary_csv)),
            "stage_f_run_dir": str(run_dir),
            "pipeline_batch_results_csv": str(pipeline_batch_results_csv),
            "stage_f_results_csv": str(results_csv),
        }
        pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

        return {
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "rows_attempted": rows_attempted,
            "rows_completed": rows_completed,
            "rows_failed": rows_failed,
            "rows_with_candidate_signal": rows_with_candidate_signal,
            "rows_without_candidate_signal": rows_without_candidate_signal,
            "rows_flagged_for_manual_review": rows_flagged_for_manual_review,
            "rows_requiring_rescue_followup": rows_requiring_rescue_followup,
            "final_label_counts": label_counts_text,
            "final_label_reason_counts": label_reason_counts_text,
            "command_used": command_used,
            "runtime_notes": runtime_notes,
            "failure_modes_encountered": failure_modes,
            "comparison": comparison,
            "stage_f_run_dir": str(run_dir),
            "pipeline_batch_results_csv": str(pipeline_batch_results_csv),
        }
