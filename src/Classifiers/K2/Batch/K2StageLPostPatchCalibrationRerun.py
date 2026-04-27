from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2StageFBatch001bExecution import K2StageFBatch001bExecution
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2StageLPostPatchCalibrationRerun:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_INPUT_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001b_input.csv"
    DEFAULT_RESULTS_CSV_NAME = "k2_stage_l_batch_001b_postpatch_results.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_l_batch_001b_postpatch_summary.csv"
    DEFAULT_AUDIT_CSV_NAME = "k2_stage_l_batch_001b_postpatch_audit.csv"
    DEFAULT_RUN_DIR_NAME = "k2_stage_l_batch_001b_postpatch_run"
    WHITENESS_ALPHA = 0.01
    FLOAT_TOL = 1e-12

    AUDIT_REQUIRED_COLUMNS = [
        "epic_id",
        "triage_whiteness_mode",
        "triage_whiteness_score",
        "triage_whiteness_pvalue",
        "triage_whiteness_log10_pvalue",
        "triage_whiteness_statistic_abs_rho",
        "triage_whiteness_z",
        "triage_whiteness_underflowed",
        "triage_whiteness_definition",
        "triage_usable",
        "triage_why_not_usable",
        "final_label",
        "final_label_reason",
    ]

    def __init__(self, runner_factory: Optional[Any] = None) -> None:
        self.runner_factory = runner_factory if runner_factory is not None else K2BatchRunner
        self.stage_f_helper = K2StageFBatch001bExecution(runner_factory=self.runner_factory)

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Run Stage L as a post-patch calibration rerun for batch 001b and emit "
                "results, summary, and audit artifacts without changing policy."
            )
        )
        p.add_argument("--input-csv", type=Path, default=cls.DEFAULT_INPUT_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(input_csv=Path(args.input_csv), out_dir=Path(args.out_dir))

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

    @staticmethod
    def _read_optional_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _single_row(
        section: str,
        metric: str,
        *,
        submetric: str = "",
        value_text: str = "",
        value_num: Any = "",
        count: Any = "",
        fraction: Any = "",
        note: str = "",
    ) -> Dict[str, Any]:
        return {
            "section": section,
            "metric": metric,
            "submetric": submetric,
            "value_text": value_text,
            "value_num": value_num,
            "count": count,
            "fraction": fraction,
            "note": note,
        }

    @staticmethod
    def _count_rows(section: str, metric: str, counts: pd.Series, total_rows: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for value, count in counts.items():
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "submetric": str(value),
                    "value_text": str(value),
                    "value_num": "",
                    "count": int(count),
                    "fraction": (float(count) / float(total_rows)) if total_rows > 0 else float("nan"),
                    "note": "",
                }
            )
        return rows

    @classmethod
    def _values_agree(cls, left: pd.Series, right: pd.Series) -> pd.Series:
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        both_missing = left_num.isna() & right_num.isna()
        both_finite = left_num.notna() & right_num.notna()
        agree = np.isclose(
            left_num.where(both_finite, 0.0).to_numpy(dtype=float),
            right_num.where(both_finite, 0.0).to_numpy(dtype=float),
            rtol=0.0,
            atol=cls.FLOAT_TOL,
        )
        return both_missing | pd.Series(agree, index=left.index) & both_finite

    @staticmethod
    def _coerce_whiteness_mode(mode: pd.Series, definition: pd.Series) -> pd.Series:
        raw_mode = mode.fillna("").astype(str).str.strip().str.lower()
        out = raw_mode.copy()
        missing = ~raw_mode.isin(["pvalue", "statistic"])
        defn = definition.fillna("").astype(str).str.strip().str.lower()
        out.loc[missing & defn.str.contains("pvalue", na=False)] = "pvalue"
        out.loc[missing & defn.str.contains("statistic", na=False)] = "statistic"
        out.loc[~out.isin(["pvalue", "statistic"])] = ""
        return out

    @classmethod
    def _build_results_output(cls, input_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        merged = cls.stage_f_results_merge(input_df=input_df, results_df=results_df)
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
            "saved_triage_whiteness_pvalue",
            "saved_triage_whiteness_definition",
            "saved_triage_step_score",
            "saved_triage_score_global",
            "triage_status_pipeline",
            "triage_usable_pipeline",
            "triage_why_not_usable_pipeline",
            "triage_whiteness_score",
            "triage_whiteness_pvalue",
            "triage_whiteness_log10_pvalue",
            "triage_whiteness_statistic_abs_rho",
            "triage_whiteness_z",
            "triage_whiteness_mode",
            "triage_whiteness_underflowed",
            "triage_whiteness_definition",
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
    def stage_f_results_merge(input_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        planned = input_df.copy().rename(
            columns={
                "best_depth_snr": "planned_best_depth_snr",
                "n_events": "planned_n_events",
                "n_periods_proposed": "planned_n_periods_proposed",
            }
        )
        return planned.merge(results_df, how="left", on=["epic_id", "query"], suffixes=("", "_pipeline"))

    def _build_audit(self, results_output: pd.DataFrame) -> pd.DataFrame:
        runtime_usable = results_output.get(
            "triage_usable_pipeline",
            results_output.get("triage_usable", pd.Series([False] * len(results_output), index=results_output.index)),
        )
        runtime_why_not = results_output.get(
            "triage_why_not_usable_pipeline",
            results_output.get("triage_why_not_usable", pd.Series([""] * len(results_output), index=results_output.index)),
        )
        mode = self._coerce_whiteness_mode(
            results_output.get("triage_whiteness_mode", pd.Series([""] * len(results_output), index=results_output.index)),
            results_output.get("triage_whiteness_definition", pd.Series([""] * len(results_output), index=results_output.index)),
        )
        pvalue = pd.to_numeric(
            results_output.get("triage_whiteness_pvalue", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        legacy = pd.to_numeric(
            results_output.get("triage_whiteness_score", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        log10_pvalue = pd.to_numeric(
            results_output.get("triage_whiteness_log10_pvalue", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        statistic = pd.to_numeric(
            results_output.get("triage_whiteness_statistic_abs_rho", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        z_value = pd.to_numeric(
            results_output.get("triage_whiteness_z", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        underflowed = self._bool_series(
            results_output.get("triage_whiteness_underflowed", pd.Series([False] * len(results_output), index=results_output.index))
        )
        saved_pvalue = pd.to_numeric(
            results_output.get("saved_triage_whiteness_pvalue", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
            errors="coerce",
        )
        pvalue_mode = mode.eq("pvalue")
        legacy_explicit_agree = self._values_agree(legacy, pvalue) & pvalue_mode
        zero_pvalue = pvalue.eq(0.0)
        finite_positive = pvalue.gt(0.0) & np.isfinite(pvalue)
        audit_df = pd.DataFrame(
            {
                "epic_id": results_output.get("epic_id", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str),
                "query": results_output.get("query", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str),
                "new_execution_order": pd.to_numeric(
                    results_output.get("new_execution_order", pd.Series([pd.NA] * len(results_output), index=results_output.index)),
                    errors="coerce",
                ),
                "saved_triage_whiteness_pvalue": saved_pvalue,
                "saved_triage_whiteness_definition": results_output.get(
                    "saved_triage_whiteness_definition", pd.Series([""] * len(results_output), index=results_output.index)
                ).fillna("").astype(str),
                "triage_whiteness_mode": mode,
                "triage_whiteness_score": legacy,
                "triage_whiteness_pvalue": pvalue,
                "triage_whiteness_log10_pvalue": log10_pvalue,
                "triage_whiteness_statistic_abs_rho": statistic,
                "triage_whiteness_z": z_value,
                "triage_whiteness_underflowed": underflowed,
                "triage_whiteness_definition": results_output.get(
                    "triage_whiteness_definition", pd.Series([""] * len(results_output), index=results_output.index)
                ).fillna("").astype(str),
                "triage_usable": self._bool_series(pd.Series(runtime_usable, index=results_output.index)),
                "triage_why_not_usable": pd.Series(runtime_why_not, index=results_output.index).fillna("").astype(str),
                "final_label": results_output.get("label", pd.Series([""] * len(results_output), index=results_output.index)).fillna("").astype(str),
                "final_label_reason": results_output.get(
                    "label_reason", pd.Series([""] * len(results_output), index=results_output.index)
                ).fillna("").astype(str),
                "runtime_whiteness_zero": zero_pvalue,
                "runtime_whiteness_finite_positive": finite_positive,
                "runtime_whiteness_missing": pvalue.isna(),
                "legacy_explicit_agree_pvalue_mode": legacy_explicit_agree,
                "saved_runtime_pvalue_gap": saved_pvalue - pvalue,
                "saved_runtime_same_threshold_side": (
                    saved_pvalue.ge(self.WHITENESS_ALPHA) & pvalue.ge(self.WHITENESS_ALPHA)
                ) | (
                    saved_pvalue.lt(self.WHITENESS_ALPHA) & pvalue.lt(self.WHITENESS_ALPHA)
                ),
            }
        )
        required_front = [c for c in self.AUDIT_REQUIRED_COLUMNS if c in audit_df.columns]
        remaining = [c for c in audit_df.columns if c not in required_front]
        return audit_df[required_front + remaining].sort_values(
            by=["new_execution_order", "epic_id"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)

    def _build_summary(
        self,
        *,
        input_csv: Path,
        run_dir: Path,
        pipeline_batch_results_csv: Path,
        runtime_seconds: float,
        results_output: pd.DataFrame,
        audit_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows_attempted = int(len(results_output))
        failed_mask = self.stage_f_helper._is_failed(results_output)
        rows_failed = int(failed_mask.sum())
        rows_completed = int(rows_attempted - rows_failed)
        total_rows = int(len(audit_df))

        label_counts = audit_df["final_label"].fillna("").astype(str).value_counts()
        label_reason_counts = audit_df["final_label_reason"].fillna("").astype(str).value_counts()
        mode_counts = audit_df["triage_whiteness_mode"].fillna("").astype(str).replace({"": "blank"}).value_counts()
        underflow_true = self._bool_series(audit_df["triage_whiteness_underflowed"]).sum()
        pvalue = pd.to_numeric(audit_df["triage_whiteness_pvalue"], errors="coerce")
        legacy = pd.to_numeric(audit_df["triage_whiteness_score"], errors="coerce")
        log10_pvalue = pd.to_numeric(audit_df["triage_whiteness_log10_pvalue"], errors="coerce")
        mode = audit_df["triage_whiteness_mode"].fillna("").astype(str)
        pvalue_mode = mode.eq("pvalue")
        legacy_explicit_agree = self._values_agree(legacy, pvalue) & pvalue_mode
        zero_pvalue = pvalue.eq(0.0)
        finite_positive = pvalue.gt(0.0) & np.isfinite(pvalue)
        pvalue_nan = pvalue.isna()
        log10_finite = np.isfinite(log10_pvalue)
        pvalue_mode_rows = int(pvalue_mode.sum())
        comparability_resolved = bool(
            pvalue_mode_rows > 0
            and int(legacy_explicit_agree.sum()) == pvalue_mode_rows
            and int((pvalue_mode & legacy.notna() & pvalue_nan).sum()) == 0
            and int((zero_pvalue & ~self._bool_series(audit_df["triage_whiteness_underflowed"]) & log10_finite).sum()) == 0
        )
        scientific_rejection_still_real = bool(
            total_rows > 0
            and audit_df["final_label"].fillna("").astype(str).eq("Noisy_trash").all()
            and audit_df["final_label_reason"].fillna("").astype(str).str.lower().str.contains("whiteness", na=False).all()
        )

        summary_rows: List[Dict[str, Any]] = [
            self._single_row("run", "rows_attempted", value_num=rows_attempted, count=rows_attempted),
            self._single_row("run", "rows_completed", value_num=rows_completed, count=rows_completed),
            self._single_row("run", "rows_failed", value_num=rows_failed, count=rows_failed),
            self._single_row(
                "run",
                "runtime_seconds",
                value_num=float(runtime_seconds),
                note=(
                    "Official default scientific policy unchanged; Stage L is a post-patch calibration rerun "
                    "of k2_stage_f_batch_001b_input.csv only."
                ),
            ),
            self._single_row("whiteness", "triage_whiteness_underflowed_true", count=int(underflow_true), value_num=int(underflow_true)),
            self._single_row("whiteness", "triage_whiteness_pvalue_eq_0_0", count=int(zero_pvalue.sum()), value_num=int(zero_pvalue.sum())),
            self._single_row(
                "whiteness",
                "triage_whiteness_pvalue_finite_positive",
                count=int(finite_positive.sum()),
                value_num=int(finite_positive.sum()),
            ),
            self._single_row("whiteness", "triage_whiteness_pvalue_nan", count=int(pvalue_nan.sum()), value_num=int(pvalue_nan.sum())),
            self._single_row(
                "whiteness",
                "triage_whiteness_log10_pvalue_finite",
                count=int(log10_finite.sum()),
                value_num=int(log10_finite.sum()),
            ),
            self._single_row(
                "comparability",
                "pvalue_mode_rows",
                count=pvalue_mode_rows,
                value_num=pvalue_mode_rows,
            ),
            self._single_row(
                "comparability",
                "legacy_score_explicit_pvalue_agree_in_pvalue_mode",
                count=int(legacy_explicit_agree.sum()),
                value_num=int(legacy_explicit_agree.sum()),
            ),
            self._single_row(
                "comparability",
                "runtime_pvalue_missing_despite_legacy_score_pvalue_mode",
                count=int((pvalue_mode & legacy.notna() & pvalue_nan).sum()),
                value_num=int((pvalue_mode & legacy.notna() & pvalue_nan).sum()),
            ),
            self._single_row(
                "comparability",
                "saved_runtime_both_pvalue_observed",
                count=int(
                    pd.to_numeric(audit_df["saved_triage_whiteness_pvalue"], errors="coerce").notna().sum()
                    and (pd.to_numeric(audit_df["saved_triage_whiteness_pvalue"], errors="coerce").notna() & pvalue.notna()).sum()
                ),
                value_num=int((pd.to_numeric(audit_df["saved_triage_whiteness_pvalue"], errors="coerce").notna() & pvalue.notna()).sum()),
            ),
            self._single_row(
                "conclusion",
                "patch_successfully_resolved_saved_runtime_comparability_problem",
                value_text="yes" if comparability_resolved else "no",
            ),
            self._single_row(
                "conclusion",
                "batch_still_scientifically_rejected_after_representation_fix",
                value_text="yes" if scientific_rejection_still_real else "no",
            ),
            self._single_row(
                "conclusion",
                "ready_for_policy_decision_next",
                value_text="yes" if comparability_resolved else "no",
                note=(
                    "Policy is untouched here; this flag only indicates whether representation and comparability "
                    "evidence is now clear enough to support the next policy discussion."
                ),
            ),
            self._single_row("artifacts", "source_input_csv", value_text=str(Path(input_csv))),
            self._single_row("artifacts", "stage_l_run_dir", value_text=str(run_dir)),
            self._single_row("artifacts", "pipeline_batch_results_csv", value_text=str(pipeline_batch_results_csv)),
        ]
        summary_rows.extend(self._count_rows("final_label", "final_label_counts", label_counts, total_rows))
        summary_rows.extend(self._count_rows("final_label_reason", "final_label_reason_counts", label_reason_counts, total_rows))
        summary_rows.extend(self._count_rows("whiteness", "triage_whiteness_mode_counts", mode_counts, total_rows))
        return pd.DataFrame(summary_rows)

    def run(self, *, input_csv: Path, out_dir: Path) -> Dict[str, Any]:
        input_df = self.stage_f_helper._prepare_input(Path(input_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir / self.DEFAULT_RUN_DIR_NAME
        results_csv = out_dir / self.DEFAULT_RESULTS_CSV_NAME
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME
        audit_csv = out_dir / self.DEFAULT_AUDIT_CSV_NAME

        started = time.perf_counter()
        runner = self.runner_factory(out_dir=run_dir, input_csv=input_csv, query_col="query")
        out = runner.run()
        runtime_seconds = float(time.perf_counter() - started)

        pipeline_batch_results_csv = Path(out["batch_results_csv"])
        results_df = (
            pd.read_csv(pipeline_batch_results_csv)
            if pipeline_batch_results_csv.exists()
            else pd.DataFrame(out.get("results_df", pd.DataFrame()))
        )
        results_output = self._build_results_output(input_df=input_df, results_df=results_df)
        results_output.to_csv(results_csv, index=False)

        audit_df = self._build_audit(results_output=results_output)
        audit_df.to_csv(audit_csv, index=False)

        summary_df = self._build_summary(
            input_csv=Path(input_csv),
            run_dir=run_dir,
            pipeline_batch_results_csv=pipeline_batch_results_csv,
            runtime_seconds=runtime_seconds,
            results_output=results_output,
            audit_df=audit_df,
        )
        summary_df.to_csv(summary_csv, index=False)

        return {
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "audit_csv": str(audit_csv),
            "rows_attempted": int(len(results_output)),
            "rows_completed": int(len(results_output) - int(self.stage_f_helper._is_failed(results_output).sum())),
            "rows_failed": int(self.stage_f_helper._is_failed(results_output).sum()),
            "runtime_seconds": float(runtime_seconds),
            "stage_l_run_dir": str(run_dir),
            "pipeline_batch_results_csv": str(pipeline_batch_results_csv),
        }


if __name__ == "__main__":
    K2StageLPostPatchCalibrationRerun.run_cli()
