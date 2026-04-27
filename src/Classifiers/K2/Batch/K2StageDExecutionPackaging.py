from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2StageCActionQueue import K2StageCActionQueue


class K2StageDExecutionPackaging:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_C_ACTION_QUEUE_CSV = DEFAULT_OUT_DIR / "k2_stage_c_action_queue.csv"

    HIGH_PRIORITY_CSV_NAME = "k2_stage_d_process_now_high_priority.csv"
    MEDIUM_PRIORITY_CSV_NAME = "k2_stage_d_process_now_medium_priority.csv"
    RESCUE_CSV_NAME = "k2_stage_d_rescue_candidates.csv"
    MANUAL_REVIEW_CSV_NAME = "k2_stage_d_manual_review.csv"
    DEFERRED_CSV_NAME = "k2_stage_d_deferred.csv"

    RANKING_LOGIC = "best_depth_snr desc, n_periods_proposed desc, n_events desc, epic_id_norm asc"
    NULL_HANDLING = (
        "best_depth_snr, n_periods_proposed, and n_events are coerced with pandas.to_numeric(errors='coerce'); "
        "missing numeric values sort after non-missing values for the descending signal keys. "
        "epic_id_norm is normalized to stripped text and used as the final ascending tie-breaker."
    )

    REQUIRED_COLUMNS = [
        "next_action",
        "priority",
        "epic_id_norm",
        "best_depth_snr",
        "n_periods_proposed",
        "n_events",
    ]
    NUMERIC_RANKING_COLUMNS = ["best_depth_snr", "n_periods_proposed", "n_events"]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Package the accepted Stage C action queue into Stage D execution CSVs."
        )
        p.add_argument("--stage-c-csv", type=Path, default=cls.DEFAULT_STAGE_C_ACTION_QUEUE_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(stage_c_csv=Path(args.stage_c_csv), out_dir=Path(args.out_dir))

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_stage_c(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage C action queue missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        return out

    @classmethod
    def _output_specs(cls, out_dir: Path) -> List[Dict[str, Any]]:
        return [
            {
                "label": "process_now_high_priority",
                "path": out_dir / cls.HIGH_PRIORITY_CSV_NAME,
                "next_action": K2StageCActionQueue.ACTION_PROCESS_NOW,
                "priority": K2StageCActionQueue.PRIORITY_HIGH,
            },
            {
                "label": "process_now_medium_priority",
                "path": out_dir / cls.MEDIUM_PRIORITY_CSV_NAME,
                "next_action": K2StageCActionQueue.ACTION_PROCESS_NOW,
                "priority": K2StageCActionQueue.PRIORITY_MEDIUM,
            },
            {
                "label": "rescue_candidates",
                "path": out_dir / cls.RESCUE_CSV_NAME,
                "next_action": K2StageCActionQueue.ACTION_RESCUE_PATH_CANDIDATE,
            },
            {
                "label": "manual_review",
                "path": out_dir / cls.MANUAL_REVIEW_CSV_NAME,
                "next_action": K2StageCActionQueue.ACTION_NEEDS_MANUAL_REVIEW,
            },
            {
                "label": "deferred",
                "path": out_dir / cls.DEFERRED_CSV_NAME,
                "next_action": K2StageCActionQueue.ACTION_LOW_PRIORITY_OR_DEFER,
            },
        ]

    def _sort_for_execution(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        helper_cols: List[str] = []
        for col in self.NUMERIC_RANKING_COLUMNS:
            numeric = pd.to_numeric(out[col], errors="coerce")
            missing_col = f"__{col}_missing"
            value_col = f"__{col}_value"
            out[missing_col] = numeric.isna()
            out[value_col] = numeric
            helper_cols.extend([missing_col, value_col])
        out["__epic_id_norm_value"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        helper_cols.append("__epic_id_norm_value")
        out = out.sort_values(
            by=[
                "__best_depth_snr_missing",
                "__best_depth_snr_value",
                "__n_periods_proposed_missing",
                "__n_periods_proposed_value",
                "__n_events_missing",
                "__n_events_value",
                "__epic_id_norm_value",
            ],
            ascending=[True, False, True, False, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        out["execution_order"] = range(1, len(out) + 1)
        return out.drop(columns=helper_cols)

    @staticmethod
    def _mask_for_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series:
        mask = df["next_action"].astype(str).eq(str(spec["next_action"]))
        if "priority" in spec:
            mask = mask & df["priority"].astype(str).eq(str(spec["priority"]))
        return mask

    @staticmethod
    def _missing_ranking_summary(df: pd.DataFrame) -> Dict[str, int]:
        best_depth_snr = pd.to_numeric(df["best_depth_snr"], errors="coerce")
        n_periods = pd.to_numeric(df["n_periods_proposed"], errors="coerce")
        n_events = pd.to_numeric(df["n_events"], errors="coerce")
        epic_id_norm = df["epic_id_norm"].fillna("").astype(str).str.strip()
        missing_epic_id_norm = epic_id_norm.eq("")
        missing_any = best_depth_snr.isna() | n_periods.isna() | n_events.isna() | missing_epic_id_norm
        return {
            "best_depth_snr": int(best_depth_snr.isna().sum()),
            "n_periods_proposed": int(n_periods.isna().sum()),
            "n_events": int(n_events.isna().sum()),
            "epic_id_norm": int(missing_epic_id_norm.sum()),
            "rows_with_any_missing_ranking_field": int(missing_any.sum()),
        }

    def run(self, *, stage_c_csv: Path, out_dir: Path) -> Dict[str, Any]:
        stage_c_df = self._prepare_stage_c(Path(stage_c_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        specs = self._output_specs(out_dir)
        assigned_mask = pd.Series(False, index=stage_c_df.index)
        outputs: Dict[str, str] = {}
        counts: Dict[str, int] = {}

        for spec in specs:
            mask = self._mask_for_spec(stage_c_df, spec)
            assigned_mask = assigned_mask | mask
            ranked = self._sort_for_execution(stage_c_df.loc[mask].copy())
            ranked.to_csv(spec["path"], index=False)
            outputs[spec["label"]] = str(spec["path"])
            counts[spec["label"]] = int(len(ranked))

        unexpected = stage_c_df.loc[~assigned_mask].copy()
        if len(unexpected) > 0:
            combos = (
                unexpected.assign(route_combo=unexpected["next_action"].astype(str) + "|" + unexpected["priority"].astype(str))[
                    "route_combo"
                ]
                .value_counts()
                .sort_index()
                .to_dict()
            )
            raise ValueError(f"Stage C contains unroutable rows for Stage D packaging: {combos}")

        total_rows = int(sum(counts.values()))
        if total_rows != int(len(stage_c_df)):
            raise ValueError(
                f"Stage D packaging row-count mismatch: stage_c_rows={len(stage_c_df)} packaged_rows={total_rows}"
            )

        missing_summary = self._missing_ranking_summary(stage_c_df)
        return {
            "stage_c_csv": str(Path(stage_c_csv)),
            "process_now_high_priority_csv": outputs["process_now_high_priority"],
            "process_now_medium_priority_csv": outputs["process_now_medium_priority"],
            "rescue_candidates_csv": outputs["rescue_candidates"],
            "manual_review_csv": outputs["manual_review"],
            "deferred_csv": outputs["deferred"],
            "process_now_high_priority_count": counts["process_now_high_priority"],
            "process_now_medium_priority_count": counts["process_now_medium_priority"],
            "rescue_candidates_count": counts["rescue_candidates"],
            "manual_review_count": counts["manual_review"],
            "deferred_count": counts["deferred"],
            "rows_total": total_rows,
            "ranking_logic": self.RANKING_LOGIC,
            "null_handling": self.NULL_HANDLING,
            "missing_ranking_fields": missing_summary,
        }
