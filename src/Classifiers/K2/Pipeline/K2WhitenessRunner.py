from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


@dataclass(frozen=True)
class K2WhitenessConfig:
    IN_CSV: str = r"plots\k2_batch\batch_results_retriaged.csv"
    OUT_CSV: str = r"plots\k2_batch\batch_results_whiteness.csv"
    WHITENESS_ALPHA: float = 0.01
    USE_RUN_SUBDIR: bool = False
    RUN_ID: Optional[str] = None
    MAX_ROWS: Optional[int] = None
    START_INDEX: int = 0
    END_INDEX: Optional[int] = None

    @property
    def in_csv_path(self) -> Path:
        return Path(self.IN_CSV)

    @property
    def out_csv_path(self) -> Path:
        return Path(self.OUT_CSV)


class K2WhitenessRunner:
    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Run K2 whiteness triage-only precompute from an existing batch_results_retriaged.csv.")
        p.add_argument("--input", type=Path, default=Path(K2WhitenessConfig.IN_CSV), help=f"Input batch CSV. Default: {K2WhitenessConfig.IN_CSV}")
        p.add_argument(
            "--output",
            "--out",
            dest="output",
            type=Path,
            default=Path(K2WhitenessConfig.OUT_CSV),
            help=f"Output CSV. Default: {K2WhitenessConfig.OUT_CSV}",
        )
        p.add_argument(
            "--whiteness-alpha",
            "--whiteness_alpha",
            dest="whiteness_alpha",
            type=float,
            default=K2WhitenessConfig.WHITENESS_ALPHA,
            help=f"P-value gate for whiteness triage. Default: {K2WhitenessConfig.WHITENESS_ALPHA}",
        )
        p.add_argument("--use-run-subdir", action="store_true", help="Write the output CSV under an auto-resolved run subdirectory.")
        p.add_argument("--run-id", default=None, help="Optional run identifier used with --use-run-subdir.")
        p.add_argument("--max-rows", "--max_rows", dest="max_rows", type=int, default=None, help="Optional head limit after slicing.")
        p.add_argument("--start-index", "--start_index", dest="start_index", type=int, default=0, help="Inclusive start row index.")
        p.add_argument("--end-index", "--end_index", dest="end_index", type=int, default=None, help="Exclusive end row index.")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        config = K2WhitenessConfig(
            IN_CSV=str(args.input),
            OUT_CSV=str(args.output),
            WHITENESS_ALPHA=float(args.whiteness_alpha),
            USE_RUN_SUBDIR=bool(args.use_run_subdir),
            RUN_ID=args.run_id,
            MAX_ROWS=args.max_rows,
            START_INDEX=int(args.start_index),
            END_INDEX=args.end_index,
        )
        return cls().run(config=config)

    @staticmethod
    def _sanitize_run_id(value: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return text.strip("._-") or "run"

    def _resolve_output_path(self, config: K2WhitenessConfig) -> Path:
        out_csv = config.out_csv_path
        if not bool(config.USE_RUN_SUBDIR):
            return out_csv

        run_id_cfg = config.RUN_ID
        run_id = (
            self._sanitize_run_id(str(run_id_cfg))
            if (run_id_cfg is not None and str(run_id_cfg).strip() != "")
            else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        run_dir = out_csv.parent / run_id
        if run_dir.exists():
            i = 1
            while True:
                candidate = out_csv.parent / f"{run_id}_{i:02d}"
                if not candidate.exists():
                    run_dir = candidate
                    break
                i += 1
        return run_dir / out_csv.name

    @staticmethod
    def _slice_df(df: pd.DataFrame, config: K2WhitenessConfig) -> pd.DataFrame:
        start = max(0, int(config.START_INDEX))
        end = None if config.END_INDEX is None else max(0, int(config.END_INDEX))
        work = df.iloc[start:end].copy().reset_index(drop=True)
        if config.MAX_ROWS is not None:
            work = work.head(max(0, int(config.MAX_ROWS))).copy().reset_index(drop=True)
        return work

    @staticmethod
    def _bool_count(series: pd.Series) -> int:
        if len(series) == 0:
            return 0
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any():
            return int(num.fillna(0).astype(int).astype(bool).sum())
        text = series.fillna("").astype(str).str.strip().str.lower()
        return int(text.isin({"true", "t", "yes", "y", "1"}).sum())

    @staticmethod
    def _extract_epic_id(query: Any) -> str:
        text = str(query).strip()
        if text == "" or text.lower() == "nan":
            return ""
        m = re.search(r"\d+", text)
        return m.group(0) if m is not None else text

    @staticmethod
    def _whiteness_is_pvalue_series(definition: pd.Series) -> pd.Series:
        low = definition.fillna("").astype(str).str.strip().str.lower()
        return low.str.contains("pvalue", regex=False)

    @staticmethod
    def _quantile_summary(series: pd.Series) -> Dict[str, float]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) == 0:
            nan = float("nan")
            return {"whiteness_min": nan, "whiteness_median": nan, "whiteness_max": nan}
        return {
            "whiteness_min": float(values.min()),
            "whiteness_median": float(values.median()),
            "whiteness_max": float(values.max()),
        }

    @staticmethod
    def _diagnostics(series: pd.Series) -> Dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce")
        finite = values.dropna()
        near_zero_eps = 1e-6
        out: Dict[str, Any] = {
            "whiteness_null_count": int(values.isna().sum()),
            "whiteness_near_zero_eps": float(near_zero_eps),
            "whiteness_near_zero_count": int((finite <= near_zero_eps).sum()) if len(finite) > 0 else 0,
            "whiteness_gt_0_95_count": int((finite > 0.95).sum()) if len(finite) > 0 else 0,
            "whiteness_lt_0_05_count": int((finite < 0.05).sum()) if len(finite) > 0 else 0,
        }
        if len(finite) == 0:
            out["whiteness_deciles"] = {}
            out["whiteness_histogram_bins"] = {}
            out["whiteness_outside_0_1_count"] = 0
            return out

        q = finite.quantile([x / 10.0 for x in range(1, 10)])
        out["whiteness_deciles"] = {f"p{int(100 * float(k))}": float(v) for k, v in q.to_dict().items()}

        edges = np.linspace(0.0, 1.0, 11)
        cats = pd.cut(finite, bins=edges, include_lowest=True, right=True)
        vc = cats.value_counts(sort=False)
        hist: Dict[str, int] = {}
        for interval, count in vc.items():
            left = float(interval.left)
            right = float(interval.right)
            hist[f"({left:.1f},{right:.1f}]"] = int(count)
        out["whiteness_histogram_bins"] = hist
        out["whiteness_outside_0_1_count"] = int(((finite < 0.0) | (finite > 1.0)).sum())
        return out

    @staticmethod
    def _to_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
        if len(series) == 0:
            return pd.Series([], dtype=bool)
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any():
            return num.fillna(0).astype(int).astype(bool)
        low = series.fillna("").astype(str).str.strip().str.lower()
        out = pd.Series([default] * len(series), index=series.index, dtype=bool)
        out.loc[low.isin({"true", "t", "yes", "y", "1"})] = True
        out.loc[low.isin({"false", "f", "no", "n", "0"})] = False
        return out

    @staticmethod
    def _value_counts_as_dict(series: pd.Series, *, top_n: Optional[int] = None, normalize_text: bool = False) -> Dict[str, int]:
        if len(series) == 0:
            return {}
        s = series.fillna("").astype(str).str.strip()
        if normalize_text:
            s = s.str.lower()
        s = s.replace("", "<empty>")
        vc = s.value_counts()
        if top_n is not None:
            vc = vc.head(max(0, int(top_n)))
        return {str(k): int(v) for k, v in vc.to_dict().items()}

    def _null_whiteness_breakdown(self, df: pd.DataFrame, whiteness_col: str) -> Dict[str, Any]:
        values = pd.to_numeric(df.get(whiteness_col, pd.Series([pd.NA] * len(df))), errors="coerce")
        null_mask = values.isna()
        sub = df.loc[null_mask].copy()
        if len(sub) == 0:
            return {
                "null_whiteness_by_triage_status": {},
                "null_whiteness_by_triage_usable": {},
                "null_whiteness_by_triage_why_not_usable_top": {},
                "null_whiteness_usable_true_count": 0,
                "null_whiteness_shortlist_attempt_count": 0,
                "null_whiteness_shortlist_topk_candidate_count": 0,
            }

        triage_status_counts = self._value_counts_as_dict(
            sub.get("triage_status", pd.Series([""] * len(sub))),
            normalize_text=True,
        )
        triage_usable_series = self._to_bool_series(sub.get("triage_usable", pd.Series([False] * len(sub), index=sub.index)))
        triage_usable_counts = {
            "true": int(triage_usable_series.sum()),
            "false": int((~triage_usable_series).sum()),
        }
        why_counts = self._value_counts_as_dict(
            sub.get("triage_why_not_usable", pd.Series([""] * len(sub))),
            top_n=20,
            normalize_text=False,
        )
        query_nonempty = sub.get("query", pd.Series([""] * len(sub))).fillna("").astype(str).str.strip() != ""
        epic_nonempty = sub.get("epic_id", pd.Series([""] * len(sub))).fillna("").astype(str).str.strip() != ""
        shortlist_attempt_mask = query_nonempty & epic_nonempty
        topk_candidate_mask = shortlist_attempt_mask & sub.get("triage_status", "").fillna("").astype(str).str.strip().str.lower().eq("ok")
        return {
            "null_whiteness_by_triage_status": triage_status_counts,
            "null_whiteness_by_triage_usable": triage_usable_counts,
            "null_whiteness_by_triage_why_not_usable_top": why_counts,
            "null_whiteness_usable_true_count": int(triage_usable_series.sum()),
            "null_whiteness_shortlist_attempt_count": int(shortlist_attempt_mask.sum()),
            "null_whiteness_shortlist_topk_candidate_count": int(topk_candidate_mask.sum()),
        }

    @staticmethod
    def _bucket_table(series: pd.Series) -> Dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce")
        total = int(len(values))
        if total <= 0:
            return {"rows_total": 0, "buckets": []}
        finite = values.dropna()
        masks = [
            ("null", values.isna()),
            ("[0,0.05)", values.ge(0.0) & values.lt(0.05)),
            ("[0.05,0.5)", values.ge(0.05) & values.lt(0.5)),
            ("[0.5,0.95]", values.ge(0.5) & values.le(0.95)),
            ("(0.95,1]", values.gt(0.95) & values.le(1.0)),
        ]
        rows: list[dict[str, Any]] = []
        for bucket, mask in masks:
            count = int(mask.sum())
            pct = (100.0 * float(count) / float(total))
            rows.append({"bucket": bucket, "count": count, "pct_total": float(pct)})
        outside_count = int(((finite < 0.0) | (finite > 1.0)).sum())
        return {"rows_total": total, "buckets": rows, "outside_0_1_count": outside_count}

    def _build_null_ok_anomaly_df(self, df: pd.DataFrame, whiteness_col: str) -> pd.DataFrame:
        work = df.copy()
        values = pd.to_numeric(work.get(whiteness_col, pd.Series([pd.NA] * len(work))), errors="coerce")
        status = work.get("triage_status", "").fillna("").astype(str).str.strip().str.lower()
        usable = self._to_bool_series(work.get("triage_usable", pd.Series([False] * len(work), index=work.index)))
        q = work.get("query", "").fillna("").astype(str).str.strip()
        epic = work.get("epic_id", "").fillna("").astype(str).str.strip()
        n_events = pd.to_numeric(
            work.get("n_events", pd.Series([np.nan] * len(work), index=work.index)),
            errors="coerce",
        )
        why = work.get("triage_why_not_usable", "").fillna("").astype(str)
        error_stage = work.get("error_stage", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
        error_type = work.get("error_type", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
        error_msg = work.get("error_msg", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
        null_mask = values.isna()
        mask = status.eq("ok") & null_mask
        sub = work.loc[mask].copy()
        if len(sub) == 0:
            return pd.DataFrame(
                columns=[
                    "query",
                    "epic_id",
                    "triage_status",
                    "triage_usable",
                    "triage_why_not_usable",
                    "error_stage",
                    "error_type",
                    "error_msg",
                    "whiteness_null_reason_category",
                    "shortlist_would_attempt_row",
                    "shortlist_would_rank_candidate_without_null_guard",
                    "shortlist_rejected_precheck",
                    "shortlist_rejection_reason",
                    "shortlist_considered_after_null_guard",
                ]
            )

        null_reason = K2ShortlistPeriodRunner._whiteness_null_reason_category(
            triage_status=status.loc[mask],
            triage_why_not_usable=why.loc[mask],
            error_stage=error_stage.loc[mask],
            error_type=error_type.loc[mask],
            error_msg=error_msg.loc[mask],
            whiteness_is_null=null_mask.loc[mask],
        )
        shortlist_would_attempt = (q.loc[mask] != "") & (epic.loc[mask] != "")
        shortlist_rank_without_guard = shortlist_would_attempt & n_events.loc[mask].gt(0)
        rejected_precheck = (~usable.loc[mask]) & null_mask.loc[mask]
        considered_after_guard = shortlist_rank_without_guard & (~rejected_precheck)

        sub["whiteness_null_reason_category"] = null_reason
        sub["shortlist_would_attempt_row"] = shortlist_would_attempt.astype(bool)
        sub["shortlist_would_rank_candidate_without_null_guard"] = shortlist_rank_without_guard.astype(bool)
        sub["shortlist_rejected_precheck"] = rejected_precheck.astype(bool)
        sub["shortlist_rejection_reason"] = np.where(
            rejected_precheck.to_numpy(dtype=bool),
            "whiteness_null_and_triage_unusable",
            "",
        )
        sub["shortlist_considered_after_null_guard"] = considered_after_guard.astype(bool)
        cols = [
            "query",
            "epic_id",
            "triage_status",
            "triage_usable",
            "triage_why_not_usable",
            "error_stage",
            "error_type",
            "error_msg",
            "whiteness_null_reason_category",
            "shortlist_would_attempt_row",
            "shortlist_would_rank_candidate_without_null_guard",
            "shortlist_rejected_precheck",
            "shortlist_rejection_reason",
            "shortlist_considered_after_null_guard",
        ]
        return sub.reindex(columns=cols).reset_index(drop=True)

    def _build_precompute_output_df(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        work = results_df.copy()
        if "query" not in work.columns:
            work["query"] = ""
        if "epic_id" not in work.columns:
            if "epic" in work.columns:
                work["epic_id"] = work["epic"].map(self._extract_epic_id)
            else:
                work["epic_id"] = work["query"].map(self._extract_epic_id)

        triage_defaults = {
            "triage_status": "",
            "triage_usable": False,
            "triage_whiteness_score": float("nan"),
            "triage_whiteness_definition": "",
            "triage_why_not_usable": "",
        }
        for col, default in triage_defaults.items():
            if col not in work.columns:
                work[col] = default

        work["query"] = work["query"].fillna("").astype(str)
        work["epic_id"] = work["epic_id"].fillna("").astype(str).str.strip()
        work["triage_whiteness_definition"] = work["triage_whiteness_definition"].fillna("").astype(str)
        work["triage_whiteness_score"] = pd.to_numeric(work["triage_whiteness_score"], errors="coerce")

        is_pvalue_series = self._whiteness_is_pvalue_series(work["triage_whiteness_definition"])
        non_empty_defs = work["triage_whiteness_definition"].str.strip() != ""
        has_definitions = bool(non_empty_defs.any())
        all_pvalue = bool(has_definitions and is_pvalue_series.loc[non_empty_defs].all())

        whiteness_value_col = "triage_whiteness_score"
        interpretation_col = ""
        if all_pvalue:
            work["triage_whiteness_pvalue"] = work["triage_whiteness_score"]
            work = work.drop(columns=["triage_whiteness_score"])
            whiteness_value_col = "triage_whiteness_pvalue"
            interpretation_col = "triage_whiteness_interpretation"
            work[interpretation_col] = (
                "Two-sided lag-1 autocorrelation p-value (normal approximation); higher means more white."
            )
            work["triage_whiteness_one_minus_pvalue"] = 1.0 - pd.to_numeric(work["triage_whiteness_pvalue"], errors="coerce")
        else:
            interpretation_col = "triage_whiteness_higher_is_better"
            rank_direction = pd.Series([pd.NA] * len(work), index=work.index, dtype="boolean")
            rank_direction.loc[is_pvalue_series] = True
            rank_direction.loc[~is_pvalue_series & non_empty_defs] = False
            work[interpretation_col] = rank_direction
            work["triage_whiteness_one_minus_score"] = 1.0 - pd.to_numeric(work["triage_whiteness_score"], errors="coerce")

        triage_cols = [c for c in work.columns if c.startswith("triage_")]
        preferred = [
            "query",
            "epic_id",
            "triage_status",
            "triage_usable",
            whiteness_value_col,
            "triage_whiteness_definition",
            "triage_why_not_usable",
            interpretation_col,
        ]
        remaining = [c for c in triage_cols if c not in preferred]
        out_cols = preferred + remaining
        out = work.reindex(columns=out_cols)
        return {
            "df": out,
            "whiteness_value_col": whiteness_value_col,
            "all_pvalue": all_pvalue,
            "interpretation_col": interpretation_col,
        }

    def run(self, config: K2WhitenessConfig) -> Dict[str, Any]:
        input_csv = config.in_csv_path
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        out_csv = self._resolve_output_path(config)
        df = self._slice_df(pd.read_csv(input_csv), config=config)

        runner = K2BatchRunner(
            out_dir=out_csv.parent,
            whiteness_alpha=float(config.WHITENESS_ALPHA),
            whiteness_score_definition="pvalue",
        )
        results_df = runner.retriage_results_df(df)
        output_pack = self._build_precompute_output_df(results_df)
        output_df = output_pack["df"]
        whiteness_value_col = str(output_pack["whiteness_value_col"])

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(out_csv, index=False)

        whiteness_series = output_df.get(whiteness_value_col, pd.Series([pd.NA] * len(output_df)))
        quantiles = self._quantile_summary(whiteness_series)
        diagnostics = self._diagnostics(whiteness_series)
        null_breakdown = self._null_whiteness_breakdown(output_df, whiteness_value_col)
        bucket_table = self._bucket_table(whiteness_series)
        anomaly_df = self._build_null_ok_anomaly_df(output_df, whiteness_value_col)
        anomaly_csv = out_csv.with_name(f"{out_csv.stem}_null_ok_rows.csv")
        anomaly_df.to_csv(anomaly_csv, index=False)

        return {
            "input_csv": input_csv,
            "out_csv": out_csv,
            "null_ok_rows_csv": anomaly_csv,
            "null_ok_rows_count": int(len(anomaly_df)),
            "total_rows": int(len(output_df)),
            "usable_rows": self._bool_count(output_df.get("triage_usable", pd.Series([], dtype=bool))),
            "whiteness_value_column": whiteness_value_col,
            "whiteness_is_pvalue": bool(output_pack["all_pvalue"]),
            "whiteness_interpretation_column": str(output_pack["interpretation_col"]),
            "whiteness_null_semantics": (
                "Null means upstream whiteness metric was unavailable: pipeline/fetch/clean/metrics error, "
                "or non-computable residual autocorrelation (all non-finite residuals or zero variance)."
            ),
            **quantiles,
            **diagnostics,
            **null_breakdown,
            "whiteness_bucket_table": bucket_table,
        }
