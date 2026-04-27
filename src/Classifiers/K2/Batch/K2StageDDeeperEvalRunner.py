from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner


class K2StageDDeeperEvalRunner:
    DEFAULT_INPUT_CSV = Path("k2_stage_d_input_tier_a.csv")
    DEFAULT_OUTPUT_CSV = Path("k2_stage_d_tier_a_results.csv")
    DEFAULT_EPICS_DIR = Path(r"plots\k2_batch\epics")

    REQUIRED_INPUT_COLUMNS = [
        "epic_id",
        "promote_tier",
        "n_events_long_good",
        "n_events_ge_10_cadences",
        "max_shape_score",
        "spike_fraction_2cadence",
        "depth_ratio",
        "stage_r_reason",
    ]

    OUTPUT_COLUMNS = REQUIRED_INPUT_COLUMNS + [
        "events_csv",
        "events_csv_available",
        "n_events_raw",
        "n_events_after_filters",
        "best_period_days",
        "period_support_count",
        "event_family_count",
        "folded_depth_consistency",
        "duration_consistency",
        "odd_even_depth_delta",
        "period_search_reason",
        "cluster_center_phase",
        "n_predicted",
        "n_covered",
        "coverage_rate",
        "hit_rate_snr",
        "hit_rate_shape",
        "soft_hit_rate",
        "cache_hits",
        "cache_misses",
        "downloads_done",
        "validations_run",
        "stage_d_label",
        "stage_d_reason",
    ]

    def __init__(self, period_runner: Optional[K2ShortlistPeriodRunner] = None) -> None:
        self.period_runner = period_runner

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Run Stage D deeper evaluation for frozen Stage R promote-tier input rows."
        )
        p.add_argument("--input-csv", type=Path, default=cls.DEFAULT_INPUT_CSV)
        p.add_argument("--output-csv", type=Path, default=cls.DEFAULT_OUTPUT_CSV)
        p.add_argument("--epics-dir", type=Path, default=cls.DEFAULT_EPICS_DIR)
        p.add_argument("--min-cluster-count", type=int, default=2)
        p.add_argument("--max-workers", type=int, default=1)
        p.add_argument(
            "--disable-validation",
            action="store_true",
            help="Skip cache-only light-curve validation and use event-family period clustering only.",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
            epics_dir=Path(args.epics_dir),
            min_cluster_count=int(args.min_cluster_count),
            max_workers=int(args.max_workers),
            validation_enabled=not bool(args.disable_validation),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @classmethod
    def _prepare_input(cls, path: Path) -> pd.DataFrame:
        df = cls._read_required_csv(path)
        missing = [c for c in cls.REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage D input missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id"] = out["epic_id"].fillna("").astype(str).str.strip()
        if out["epic_id"].eq("").any():
            raise ValueError("Stage D input contains blank epic_id values.")
        return out.reset_index(drop=True)

    @staticmethod
    def _extract_epic_digits(epic_id: Any) -> str:
        text = "" if pd.isna(epic_id) else str(epic_id).strip()
        match = re.search(r"(\d+)", text)
        return match.group(1) if match is not None else ""

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            out = float("nan")
        return out if np.isfinite(out) else float("nan")

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return int(default)
            return int(float(value))
        except Exception:
            return int(default)

    @staticmethod
    def _circular_delta(a: np.ndarray, b: float) -> np.ndarray:
        return np.abs(((np.asarray(a, dtype=float) - float(b) + 0.5) % 1.0) - 0.5)

    @classmethod
    def _family_events(
        cls,
        *,
        events_df: pd.DataFrame,
        period: float,
        center_phase: float,
        tol_phase: float,
    ) -> pd.DataFrame:
        if len(events_df) == 0 or not np.isfinite(period) or period <= 0 or not np.isfinite(center_phase):
            return events_df.iloc[0:0].copy()
        t_mid = pd.to_numeric(events_df.get("t_mid", pd.Series(dtype=float)), errors="coerce")
        ok = np.isfinite(t_mid.to_numpy(dtype=float))
        if not bool(np.any(ok)):
            return events_df.iloc[0:0].copy()
        work = events_df.loc[ok].copy()
        work.loc[:, "t_mid"] = t_mid.loc[ok].to_numpy(dtype=float)
        phases = (np.mod(work["t_mid"].to_numpy(dtype=float), float(period)) / float(period))
        keep = cls._circular_delta(phases, float(center_phase)) <= (float(tol_phase) + 1e-12)
        return work.loc[keep].sort_values("t_mid").reset_index(drop=True)

    @classmethod
    def _robust_consistency(cls, values: pd.Series) -> float:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return float("nan")
        med = float(np.nanmedian(np.abs(arr)))
        if not np.isfinite(med) or med <= 0:
            return float("nan")
        mad = float(np.nanmedian(np.abs(arr - float(np.nanmedian(arr)))))
        scatter = mad / med
        if not np.isfinite(scatter):
            return float("nan")
        return float(np.clip(1.0 / (1.0 + scatter), 0.0, 1.0))

    @classmethod
    def _odd_even_depth_delta(cls, family_df: pd.DataFrame, period: float) -> float:
        if len(family_df) < 2 or "depth" not in family_df.columns or not np.isfinite(period) or period <= 0:
            return float("nan")
        work = family_df.copy().sort_values("t_mid").reset_index(drop=True)
        t_mid = pd.to_numeric(work.get("t_mid", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        depth = pd.to_numeric(work.get("depth", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t_mid) & np.isfinite(depth)
        t_mid = t_mid[ok]
        depth = depth[ok]
        if len(depth) < 2:
            return float("nan")
        epochs = np.rint((t_mid - float(t_mid[0])) / float(period)).astype(int)
        odd = depth[(epochs % 2) == 1]
        even = depth[(epochs % 2) == 0]
        if len(odd) == 0 or len(even) == 0:
            return float("nan")
        med = float(np.nanmedian(np.abs(depth)))
        if not np.isfinite(med) or med <= 0:
            return float("nan")
        return float(abs(float(np.nanmedian(odd)) - float(np.nanmedian(even))) / med)

    @classmethod
    def _event_family_metrics(
        cls,
        *,
        events_df: pd.DataFrame,
        period: float,
        center_phase: float,
        tol_phase: float,
    ) -> Dict[str, Any]:
        family = cls._family_events(
            events_df=events_df,
            period=float(period),
            center_phase=float(center_phase),
            tol_phase=float(tol_phase),
        )
        return {
            "event_family_count": int(len(family)),
            "folded_depth_consistency": cls._robust_consistency(family.get("depth", pd.Series(dtype=float))),
            "duration_consistency": cls._robust_consistency(
                family.get("duration_cadences", family.get("duration_days", pd.Series(dtype=float)))
            ),
            "odd_even_depth_delta": cls._odd_even_depth_delta(family, period=float(period)),
        }

    @staticmethod
    def _label_row(row: Dict[str, Any]) -> Tuple[str, str]:
        reason = str(row.get("period_search_reason", "")).strip().lower()
        support = K2StageDDeeperEvalRunner._as_int(row.get("period_support_count"), default=0)
        period = K2StageDDeeperEvalRunner._as_float(row.get("best_period_days"))
        depth_consistency = K2StageDDeeperEvalRunner._as_float(row.get("folded_depth_consistency"))
        duration_consistency = K2StageDDeeperEvalRunner._as_float(row.get("duration_consistency"))
        odd_even_delta = K2StageDDeeperEvalRunner._as_float(row.get("odd_even_depth_delta"))
        hit_shape = K2StageDDeeperEvalRunner._as_float(row.get("hit_rate_shape"))
        soft_hit = K2StageDDeeperEvalRunner._as_float(row.get("soft_hit_rate"))

        if not np.isfinite(period) or support < 2:
            return "fail_deeper_eval", f"no supported period family; reason={reason or 'unknown'}"

        consistency_ok = (
            np.isfinite(depth_consistency)
            and depth_consistency >= 0.50
            and np.isfinite(duration_consistency)
            and duration_consistency >= 0.50
        )
        odd_even_ok = (not np.isfinite(odd_even_delta)) or odd_even_delta <= 0.75
        validated_ok = reason == "validated" and (
            (not np.isfinite(hit_shape) or hit_shape >= 0.10)
            and (not np.isfinite(soft_hit) or soft_hit >= 0.10)
        )

        if support >= 3 and consistency_ok and odd_even_ok and (validated_ok or reason.startswith("cluster_only")):
            suffix = "validated cache-light-curve support" if validated_ok else "event-family support; light-curve validation unavailable or not decisive"
            return "pass_deeper_eval", (
                f"period_support_count={support}; depth_consistency={depth_consistency:.3f}; "
                f"duration_consistency={duration_consistency:.3f}; {suffix}"
            )

        return "hold_deeper_eval", (
            f"finite period with support={support}, but deeper-eval guardrails are incomplete or marginal; "
            f"reason={reason or 'unknown'}"
        )

    def _build_period_runner(
        self,
        *,
        epics_dir: Path,
        min_cluster_count: int,
        max_workers: int,
    ) -> K2ShortlistPeriodRunner:
        if self.period_runner is not None:
            return self.period_runner
        cfg = K2ShortlistPeriodConfig(
            EPICS_DIR=str(epics_dir),
            MIN_CLUSTER_COUNT=int(min_cluster_count),
            CACHE_ONLY_FIRST=True,
            DOWNLOAD_IF_CACHE_MISS=False,
            MAX_WORKERS=int(max_workers),
            PERIOD_STAGE_SELECTION_MODE="all",
            USE_RUN_SUBDIR=False,
            RUN_ID="stage_d_tier_a_deeper_eval",
        )
        return K2ShortlistPeriodRunner(config=cfg)

    def _evaluate_one(
        self,
        *,
        input_row: pd.Series,
        index_1based: int,
        total_rows: int,
        epics_dir: Path,
        runner: K2ShortlistPeriodRunner,
        validation_enabled: bool,
    ) -> Dict[str, Any]:
        epic_id = str(input_row["epic_id"]).strip()
        epic_digits = self._extract_epic_digits(epic_id)
        query = f"EPIC {epic_digits}" if epic_digits != "" else epic_id
        events_csv = epics_dir / f"EPIC_{epic_digits}" / "events.csv" if epic_digits != "" else Path("")
        events_available = bool(events_csv.exists())

        base = {c: input_row.get(c, np.nan) for c in self.REQUIRED_INPUT_COLUMNS}
        base.update(
            {
                "events_csv": str(events_csv) if str(events_csv) != "." else "",
                "events_csv_available": bool(events_available),
                "n_events_raw": 0,
                "n_events_after_filters": 0,
                "best_period_days": float("nan"),
                "period_support_count": 0,
                "event_family_count": 0,
                "folded_depth_consistency": float("nan"),
                "duration_consistency": float("nan"),
                "odd_even_depth_delta": float("nan"),
                "period_search_reason": "missing_events_csv" if not events_available else "",
                "cluster_center_phase": float("nan"),
                "n_predicted": 0,
                "n_covered": 0,
                "coverage_rate": float("nan"),
                "hit_rate_snr": float("nan"),
                "hit_rate_shape": float("nan"),
                "soft_hit_rate": float("nan"),
                "cache_hits": 0,
                "cache_misses": 0,
                "downloads_done": 0,
                "validations_run": 0,
            }
        )

        result = runner._process_selected_query(
            index_1based=int(index_1based),
            total_selected=int(total_rows),
            query=query,
            epic_dir_overrides={},
            validation_enabled=bool(validation_enabled),
        )
        for key in ["cache_hits", "cache_misses", "downloads_done", "validations_run"]:
            base[key] = int(result.get("run_counts", {}).get(key, 0))

        summary_rows = list(result.get("summary_rows", []))
        candidate_rows = [
            r for r in summary_rows
            if np.isfinite(self._as_float(r.get("P"))) and self._as_float(r.get("P")) > 0
        ]
        if len(candidate_rows) > 0:
            best = runner._select_best_row(candidate_rows)
            period = self._as_float(best.get("P"))
            center_phase = self._as_float(best.get("cluster_center_phase"))
            base.update(
                {
                    "n_events_raw": self._as_int(best.get("n_events_raw")),
                    "n_events_after_filters": self._as_int(best.get("n_events_after_filters")),
                    "best_period_days": period,
                    "period_support_count": self._as_int(best.get("cluster_count")),
                    "period_search_reason": str(best.get("reason", "")),
                    "cluster_center_phase": center_phase,
                    "n_predicted": self._as_int(best.get("n_predicted")),
                    "n_covered": self._as_int(best.get("n_covered")),
                    "coverage_rate": self._as_float(best.get("coverage_rate")),
                    "hit_rate_snr": self._as_float(best.get("hit_rate_snr")),
                    "hit_rate_shape": self._as_float(best.get("hit_rate_shape")),
                    "soft_hit_rate": self._as_float(best.get("soft_hit_rate")),
                }
            )
            if events_available:
                events_df = runner._filter_events_for_periods(pd.read_csv(events_csv))
                base.update(
                    self._event_family_metrics(
                        events_df=events_df,
                        period=float(period),
                        center_phase=float(center_phase),
                        tol_phase=float(runner.config.PERIOD_TOL_PHASE),
                    )
                )
        elif len(summary_rows) > 0:
            best = summary_rows[0]
            base.update(
                {
                    "n_events_raw": self._as_int(best.get("n_events_raw")),
                    "n_events_after_filters": self._as_int(best.get("n_events_after_filters")),
                    "period_search_reason": str(best.get("reason", "")),
                }
            )

        label, label_reason = self._label_row(base)
        base["stage_d_label"] = label
        base["stage_d_reason"] = label_reason
        return base

    def run(
        self,
        *,
        input_csv: Path = DEFAULT_INPUT_CSV,
        output_csv: Path = DEFAULT_OUTPUT_CSV,
        epics_dir: Path = DEFAULT_EPICS_DIR,
        min_cluster_count: int = 2,
        max_workers: int = 1,
        validation_enabled: bool = True,
    ) -> Dict[str, Any]:
        input_df = self._prepare_input(Path(input_csv))
        epics_dir = Path(epics_dir)
        if not epics_dir.exists():
            raise FileNotFoundError(f"EPICS directory not found: {epics_dir}")

        runner = self._build_period_runner(
            epics_dir=epics_dir,
            min_cluster_count=int(min_cluster_count),
            max_workers=int(max_workers),
        )
        if self.period_runner is None:
            runner.config = K2ShortlistPeriodConfig(
                **{
                    **runner.config.__dict__,
                    "ENABLE_VALIDATION": bool(validation_enabled),
                }
            )

        rows: List[Dict[str, Any]] = []
        total = int(len(input_df))
        for idx, row in input_df.iterrows():
            rows.append(
                self._evaluate_one(
                    input_row=row,
                    index_1based=int(idx) + 1,
                    total_rows=total,
                    epics_dir=epics_dir,
                    runner=runner,
                    validation_enabled=bool(validation_enabled),
                )
            )

        out_df = pd.DataFrame(rows).reindex(columns=self.OUTPUT_COLUMNS)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)

        labels = out_df["stage_d_label"].fillna("").astype(str)
        return {
            "input_csv": Path(input_csv),
            "output_csv": output_csv,
            "rows_input": int(len(input_df)),
            "rows_output": int(len(out_df)),
            "pass_count": int(labels.eq("pass_deeper_eval").sum()),
            "hold_count": int(labels.eq("hold_deeper_eval").sum()),
            "fail_count": int(labels.eq("fail_deeper_eval").sum()),
        }
