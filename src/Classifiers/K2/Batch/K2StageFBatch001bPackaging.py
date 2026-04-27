from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class K2StageFBatch001bPackaging:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_RERANK_PREVIEW_CSV = DEFAULT_OUT_DIR / "k2_stage_e1_high_priority_rerank_preview.csv"
    DEFAULT_INPUT_CSV_NAME = "k2_stage_f_batch_001b_input.csv"
    DEFAULT_PLAN_SUMMARY_CSV_NAME = "k2_stage_f_batch_001b_plan_summary.csv"
    DEFAULT_BATCH_SIZE = 100

    REQUIRED_COLUMNS = [
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
        "epic_id_norm",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Package the accepted Stage E.1 reranked first 100 rows into the patched calibration batch 001b."
        )
        p.add_argument("--rerank-preview-csv", type=Path, default=cls.DEFAULT_RERANK_PREVIEW_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(rerank_preview_csv=Path(args.rerank_preview_csv), out_dir=Path(args.out_dir))

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_preview(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Rerank preview CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["new_execution_order"] = pd.to_numeric(out["new_execution_order"], errors="coerce")
        out["old_execution_order"] = pd.to_numeric(out["old_execution_order"], errors="coerce")
        if out["new_execution_order"].isna().any():
            raise ValueError("Rerank preview CSV contains non-numeric new_execution_order values.")
        if out["old_execution_order"].isna().any():
            raise ValueError("Rerank preview CSV contains non-numeric old_execution_order values.")
        return out.sort_values(by=["new_execution_order", "epic_id_norm"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _median(series: pd.Series) -> float:
        numeric = pd.to_numeric(series, errors="coerce")
        return float(numeric.median()) if numeric.notna().any() else float("nan")

    def run(self, *, rerank_preview_csv: Path, out_dir: Path) -> Dict[str, Any]:
        preview = self._prepare_preview(Path(rerank_preview_csv))
        batch_df = preview.nsmallest(self.DEFAULT_BATCH_SIZE, "new_execution_order").copy().reset_index(drop=True)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_csv = out_dir / self.DEFAULT_INPUT_CSV_NAME
        summary_csv = out_dir / self.DEFAULT_PLAN_SUMMARY_CSV_NAME

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
            "epic_id_norm",
            "current_status",
            "stage_b_bucket",
            "data_availability",
            "scope_status",
            "routing_rule_id",
            "priority_rule_id",
            "period_source_reason",
            "period_terminal_reason",
            "n_events",
            "n_periods_proposed",
            "best_depth_snr",
            "saved_triage_usable",
            "saved_triage_whiteness_definition",
            "saved_triage_why_not_usable",
            "saved_triage_step_score",
            "saved_triage_score_global",
        ]
        front = [c for c in preferred_front if c in batch_df.columns]
        remaining = [c for c in batch_df.columns if c not in front]
        batch_df = batch_df[front + remaining]
        batch_df.to_csv(input_csv, index=False)

        rows_total = int(len(batch_df))
        first_10_epics = batch_df["epic_id"].astype(str).head(10).tolist()
        median_whiteness_proxy_value = self._median(batch_df["whiteness_proxy_value"])
        median_saved_triage_step_score = self._median(batch_df["saved_triage_step_score"])
        median_n_events = self._median(batch_df["n_events"])
        median_best_depth_snr = self._median(batch_df["best_depth_snr"])

        summary_row = {
            "batch_id": "high_priority_batch_001b",
            "rows_total": rows_total,
            "source_rerank_preview_csv": str(Path(rerank_preview_csv)),
            "source_selection_rule": "first 100 rows by new_execution_order from k2_stage_e1_high_priority_rerank_preview.csv",
            "first_10_epics": " | ".join(first_10_epics),
            "median_whiteness_proxy_value": median_whiteness_proxy_value,
            "median_saved_triage_step_score": median_saved_triage_step_score,
            "median_n_events": median_n_events,
            "median_best_depth_snr": median_best_depth_snr,
            "calibration_replacement_note": (
                "This patched batch 001b replaces the original batch 001 for calibration purposes and should be run "
                "before any continuation to later batches."
            ),
            "run_sequence_note": (
                "Do not continue to later batches until the patched batch 001b calibration rerun has been executed and reviewed."
            ),
        }
        pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

        return {
            "input_csv": str(input_csv),
            "plan_summary_csv": str(summary_csv),
            "rows_total": rows_total,
            "first_10_epics": first_10_epics,
            "median_whiteness_proxy_value": median_whiteness_proxy_value,
            "median_saved_triage_step_score": median_saved_triage_step_score,
            "median_n_events": median_n_events,
            "median_best_depth_snr": median_best_depth_snr,
        }
