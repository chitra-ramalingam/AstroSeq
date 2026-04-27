from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2StageFBatchExecution:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_E_BATCH_PLAN_CSV = DEFAULT_OUT_DIR / "k2_stage_e_high_priority_batch_plan.csv"
    DEFAULT_BATCH_ID = "high_priority_batch_001"
    DEFAULT_RESULTS_CSV_NAME = "k2_stage_f_batch_001_results.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_f_batch_001_summary.csv"
    DEFAULT_INPUT_CSV_NAME = "k2_stage_f_batch_001_input.csv"
    DEFAULT_RUN_DIR_NAME = "k2_stage_f_batch_001_run"

    REQUIRED_PLAN_COLUMNS = [
        "epic_id",
        "query",
        "execution_order",
        "batch_id",
        "batch_position",
        "next_action",
        "priority",
        "best_depth_snr",
        "n_events",
        "n_periods_proposed",
    ]

    def __init__(
        self,
        runner_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.runner_factory = runner_factory if runner_factory is not None else K2BatchRunner

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Run Stage F for a single accepted Stage E batch using the default official K2 pipeline."
        )
        p.add_argument("--stage-e-plan-csv", type=Path, default=cls.DEFAULT_STAGE_E_BATCH_PLAN_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--batch-id", type=str, default=cls.DEFAULT_BATCH_ID)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            stage_e_plan_csv=Path(args.stage_e_plan_csv),
            out_dir=Path(args.out_dir),
            batch_id=str(args.batch_id),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_plan(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_PLAN_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage E batch plan missing required columns: {missing} ({path})")
        out = df.copy()
        out["execution_order"] = pd.to_numeric(out["execution_order"], errors="coerce")
        out["batch_position"] = pd.to_numeric(out["batch_position"], errors="coerce")
        if out["execution_order"].isna().any():
            raise ValueError("Stage E plan contains non-numeric execution_order values.")
        if out["batch_position"].isna().any():
            raise ValueError("Stage E plan contains non-numeric batch_position values.")
        return out.sort_values(by=["execution_order", "batch_position"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _select_batch(plan_df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        batch_df = plan_df.loc[plan_df["batch_id"].astype(str).eq(str(batch_id))].copy()
        if len(batch_df) == 0:
            raise ValueError(f"No Stage E rows found for batch_id={batch_id}")
        return batch_df.sort_values(by=["execution_order", "batch_position"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _pipeline_command(run_dir: Path, input_csv: Path) -> str:
        return (
            f".\\.venv\\Scripts\\python.exe main.py --out-dir {run_dir} "
            f"--input-csv {input_csv} --query-col query"
        )

    @staticmethod
    def _is_failed(results_df: pd.DataFrame) -> pd.Series:
        triage_status = results_df.get("triage_status", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.strip().str.lower()
        error_stage = results_df.get("error_stage", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str).str.strip()
        return triage_status.eq("error") | error_stage.ne("")

    @staticmethod
    def _failure_modes(results_df: pd.DataFrame) -> str:
        failed = K2StageFBatchExecution._is_failed(results_df)
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
    def _label_counts_text(results_df: pd.DataFrame) -> str:
        labels = results_df.get("label", pd.Series([""] * len(results_df), index=results_df.index)).fillna("").astype(str)
        counts = labels.value_counts()
        if len(counts) == 0:
            return "none"
        return " | ".join([f"{label or 'blank'}={int(count)}" for label, count in counts.items()])

    @staticmethod
    def _dominant_label_reason(results_df: pd.DataFrame) -> str:
        if "label_reason" not in results_df.columns:
            return "none"
        reasons = results_df["label_reason"].fillna("").astype(str)
        reasons = reasons.loc[reasons.str.strip() != ""]
        if len(reasons) == 0:
            return "none"
        counts = reasons.value_counts()
        top_reason = str(counts.index[0])
        top_count = int(counts.iloc[0])
        return f"{top_reason} ({top_count})"

    @staticmethod
    def _representative_note(
        rows_attempted: int,
        rows_failed: int,
        failure_modes: str,
        label_counts_text: str,
    ) -> Dict[str, str]:
        failed_frac = (float(rows_failed) / float(rows_attempted)) if rows_attempted > 0 else 1.0
        if rows_attempted == 0:
            return {
                "representative_for_batch_002": "no",
                "representative_note": "No rows were attempted, so this batch cannot be used to judge readiness for batch 002.",
            }
        if failed_frac <= 0.10:
            return {
                "representative_for_batch_002": "yes",
                "representative_note": (
                    f"Failure rate {rows_failed}/{rows_attempted} ({failed_frac:.1%}) is low enough to treat this pass as representative "
                    f"for continuing to batch 002, barring review of any specific failure modes ({failure_modes}). "
                    f"Observed label distribution: {label_counts_text}."
                ),
            }
        return {
            "representative_for_batch_002": "no",
            "representative_note": (
                f"Failure rate {rows_failed}/{rows_attempted} ({failed_frac:.1%}) is too high to treat this pass as representative "
                f"for batch 002 without investigating the observed failure modes ({failure_modes}). "
                f"Observed label distribution: {label_counts_text}."
            ),
        }

    @staticmethod
    def _build_results_output(batch_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        planned = batch_df.copy().rename(
            columns={
                "best_depth_snr": "planned_best_depth_snr",
                "n_events": "planned_n_events",
                "n_periods_proposed": "planned_n_periods_proposed",
            }
        )
        merged = planned.merge(
            results_df,
            how="left",
            on=["epic_id", "query"],
            suffixes=("", "_pipeline"),
        )
        preferred_front = [
            "epic_id",
            "query",
            "execution_order",
            "batch_id",
            "batch_position",
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
            by=["execution_order", "batch_position"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)

    @staticmethod
    def _load_previous_summary(summary_csv: Path) -> Dict[str, Any]:
        if not summary_csv.exists():
            return {}
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            return {}
        if len(df) == 0:
            return {}
        return df.iloc[0].to_dict()

    def run(self, *, stage_e_plan_csv: Path, out_dir: Path, batch_id: str) -> Dict[str, Any]:
        plan_df = self._prepare_plan(Path(stage_e_plan_csv))
        batch_df = self._select_batch(plan_df, batch_id=batch_id)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_csv = out_dir / self.DEFAULT_INPUT_CSV_NAME
        run_dir = out_dir / self.DEFAULT_RUN_DIR_NAME
        results_csv = out_dir / self.DEFAULT_RESULTS_CSV_NAME
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME

        batch_df.to_csv(input_csv, index=False)
        command_used = self._pipeline_command(run_dir=run_dir, input_csv=input_csv)
        previous_summary = self._load_previous_summary(summary_csv)

        started = time.perf_counter()
        runner = self.runner_factory(
            out_dir=run_dir,
            input_csv=input_csv,
            query_col="query",
        )
        out = runner.run()
        runtime_seconds = float(time.perf_counter() - started)
        prev_runtime_seconds = pd.to_numeric(pd.Series([previous_summary.get("runtime_seconds", float("nan"))]), errors="coerce").iloc[0]
        if pd.notna(prev_runtime_seconds) and float(prev_runtime_seconds) > runtime_seconds:
            runtime_seconds = float(prev_runtime_seconds)

        batch_results_csv = Path(out["batch_results_csv"])
        results_df = pd.read_csv(batch_results_csv) if batch_results_csv.exists() else pd.DataFrame(out.get("results_df", pd.DataFrame()))
        results_output = self._build_results_output(batch_df=batch_df, results_df=results_df)
        results_output.to_csv(results_csv, index=False)

        failed_mask = self._is_failed(results_output)
        n_events = pd.to_numeric(results_output.get("n_events", 0), errors="coerce").fillna(0.0)
        labels = results_output.get("label", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str)

        rows_attempted = int(len(batch_df))
        rows_failed = int(failed_mask.sum())
        rows_completed = int(rows_attempted - rows_failed)
        rows_with_candidate_signal = int(n_events.gt(0).sum())
        rows_without_candidate_signal = int(rows_attempted - rows_with_candidate_signal)
        rows_flagged_for_manual_review = int(labels.eq("Unclassified").sum())
        rows_requiring_rescue_followup = int(labels.eq("Sparse_or_mono").sum())
        failure_modes = self._failure_modes(results_output)
        label_counts_text = self._label_counts_text(results_output)
        dominant_label_reason = self._dominant_label_reason(results_output)
        representative = self._representative_note(
            rows_attempted=rows_attempted,
            rows_failed=rows_failed,
            failure_modes=failure_modes,
            label_counts_text=label_counts_text,
        )

        runtime_minutes = runtime_seconds / 60.0
        runtime_notes = (
            f"Executed the default official K2BatchRunner settings against {rows_attempted} queries in an isolated run directory "
            f"({run_dir}). Runtime was {runtime_seconds:.1f}s ({runtime_minutes:.2f}m). "
            f"Ordering followed Stage E execution_order and batch_position with no reshuffle. "
            f"Label distribution was {label_counts_text}; dominant label_reason was {dominant_label_reason}."
        )

        summary_row = {
            "batch_id": str(batch_id),
            "rows_attempted": rows_attempted,
            "rows_completed": rows_completed,
            "rows_failed": rows_failed,
            "rows_with_candidate_signal": rows_with_candidate_signal,
            "rows_without_candidate_signal": rows_without_candidate_signal,
            "rows_flagged_for_manual_review": rows_flagged_for_manual_review,
            "rows_requiring_rescue_followup": rows_requiring_rescue_followup,
            "command_used": command_used,
            "runtime_seconds": runtime_seconds,
            "runtime_notes": runtime_notes,
            "failure_modes_encountered": failure_modes,
            "label_counts": label_counts_text,
            "dominant_label_reason": dominant_label_reason,
            "representative_for_batch_002": representative["representative_for_batch_002"],
            "representative_note": representative["representative_note"],
            "stage_e_plan_csv": str(Path(stage_e_plan_csv)),
            "stage_f_run_dir": str(run_dir),
            "pipeline_batch_results_csv": str(batch_results_csv),
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
            "command_used": command_used,
            "runtime_notes": runtime_notes,
            "failure_modes_encountered": failure_modes,
            "label_counts": label_counts_text,
            "dominant_label_reason": dominant_label_reason,
            "representative_for_batch_002": representative["representative_for_batch_002"],
            "representative_note": representative["representative_note"],
            "stage_f_run_dir": str(run_dir),
            "pipeline_batch_results_csv": str(batch_results_csv),
        }
