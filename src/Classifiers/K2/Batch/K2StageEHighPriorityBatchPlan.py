from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class K2StageEHighPriorityBatchPlan:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_D_HIGH_PRIORITY_CSV = DEFAULT_OUT_DIR / "k2_stage_d_process_now_high_priority.csv"
    DEFAULT_BATCH_PLAN_CSV_NAME = "k2_stage_e_high_priority_batch_plan.csv"
    DEFAULT_BATCH_SIZE = 100

    REQUIRED_COLUMNS = [
        "epic_id",
        "query",
        "execution_order",
        "next_action",
        "priority",
        "best_depth_snr",
        "n_events",
        "n_periods_proposed",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Build the Stage E first-pass execution plan from the Stage D high-priority queue."
        )
        p.add_argument("--stage-d-high-priority-csv", type=Path, default=cls.DEFAULT_STAGE_D_HIGH_PRIORITY_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--batch-size", type=int, default=cls.DEFAULT_BATCH_SIZE)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            stage_d_high_priority_csv=Path(args.stage_d_high_priority_csv),
            out_dir=Path(args.out_dir),
            batch_size=int(args.batch_size),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_stage_d(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage D high-priority CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["execution_order"] = pd.to_numeric(out["execution_order"], errors="coerce")
        if out["execution_order"].isna().any():
            raise ValueError("Stage D high-priority CSV contains non-numeric execution_order values.")
        out = out.sort_values(by=["execution_order"], ascending=[True], kind="mergesort").reset_index(drop=True)
        return out

    @staticmethod
    def _format_batch_id(batch_number: int) -> str:
        return f"high_priority_batch_{batch_number:03d}"

    @staticmethod
    def _rows_per_batch_summary(plan_df: pd.DataFrame) -> str:
        counts = plan_df.groupby("batch_id", sort=False).size()
        return " | ".join([f"{batch_id}:{int(count)}" for batch_id, count in counts.items()])

    @staticmethod
    def _first_batch_epics(plan_df: pd.DataFrame, limit: int = 10) -> List[str]:
        batch_one = plan_df.loc[plan_df["batch_id"].eq("high_priority_batch_001"), "epic_id"].astype(str).tolist()
        return batch_one[:limit]

    def run(self, *, stage_d_high_priority_csv: Path, out_dir: Path, batch_size: int) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        df = self._prepare_stage_d(Path(stage_d_high_priority_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plan_df = df.copy()
        zero_based = range(len(plan_df))
        batch_numbers = [(i // batch_size) + 1 for i in zero_based]
        batch_positions = [(i % batch_size) + 1 for i in zero_based]
        plan_df["batch_id"] = [self._format_batch_id(n) for n in batch_numbers]
        plan_df["batch_position"] = batch_positions

        preferred_front = [
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
        remaining = [c for c in plan_df.columns if c not in preferred_front]
        plan_df = plan_df[preferred_front + remaining]

        output_csv = out_dir / self.DEFAULT_BATCH_PLAN_CSV_NAME
        plan_df.to_csv(output_csv, index=False)

        total_rows = int(len(plan_df))
        total_batches = int((total_rows + batch_size - 1) // batch_size)
        return {
            "stage_d_high_priority_csv": str(Path(stage_d_high_priority_csv)),
            "batch_plan_csv": str(output_csv),
            "total_rows": total_rows,
            "batch_size": int(batch_size),
            "total_batches": total_batches,
            "rows_per_batch_summary": self._rows_per_batch_summary(plan_df),
            "first_10_epics_batch_1": self._first_batch_epics(plan_df, limit=10),
            "recommended_batch_size": self.DEFAULT_BATCH_SIZE,
            "batch_size_rationale": (
                "A fixed batch size of 100 keeps the first operational pass large enough for useful throughput "
                "while remaining small enough to monitor, retry, and inspect between sequential runs."
            ),
        }
