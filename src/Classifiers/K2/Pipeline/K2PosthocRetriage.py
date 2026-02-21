from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


class K2PosthocRetriage:
    DEFAULT_SHAPE_T = 0.75
    DEFAULT_MEAN_T = 0.50
    DEFAULT_PERIOD_HIT_SHAPE_T = 0.30
    DEFAULT_PERIOD_HIT_SNR_T = 0.30
    DEFAULT_PERIOD_COVERAGE_T = 0.85

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Posthoc K2 retriage from batch_results.csv only (no downloads).")
        p.add_argument("--input", type=Path, required=True, help="Input batch_results.csv")
        p.add_argument("--out", type=Path, required=True, help="Output CSV for retriaged rows")
        p.add_argument("--shape_t", type=float, default=cls.DEFAULT_SHAPE_T, help=f"best_shape_score threshold (default: {cls.DEFAULT_SHAPE_T})")
        p.add_argument("--mean_t", type=float, default=cls.DEFAULT_MEAN_T, help=f"mean_shape_score threshold (default: {cls.DEFAULT_MEAN_T})")
        p.add_argument(
            "--period_hit_shape_t",
            type=float,
            default=cls.DEFAULT_PERIOD_HIT_SHAPE_T,
            help=f"Periodic hit_rate_shape threshold (default: {cls.DEFAULT_PERIOD_HIT_SHAPE_T})",
        )
        p.add_argument(
            "--period_hit_snr_t",
            type=float,
            default=cls.DEFAULT_PERIOD_HIT_SNR_T,
            help=f"Periodic hit_rate_snr threshold (default: {cls.DEFAULT_PERIOD_HIT_SNR_T})",
        )
        p.add_argument(
            "--period_coverage_t",
            type=float,
            default=cls.DEFAULT_PERIOD_COVERAGE_T,
            help=f"Periodic coverage_rate threshold when available (default: {cls.DEFAULT_PERIOD_COVERAGE_T})",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        runner = cls()
        return runner.run(
            input_csv=Path(args.input),
            out_csv=Path(args.out),
            shape_t=float(args.shape_t),
            mean_t=float(args.mean_t),
            period_hit_shape_t=float(args.period_hit_shape_t),
            period_hit_snr_t=float(args.period_hit_snr_t),
            period_coverage_t=float(args.period_coverage_t),
        )

    @staticmethod
    def _as_float(value: Any, default: float = float("nan")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if np.isfinite(out) else float(default)

    @staticmethod
    def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
        if "label" not in df.columns:
            return {}
        vc = df["label"].fillna("").astype(str).value_counts(dropna=False)
        return {str(k): int(v) for k, v in vc.to_dict().items()}

    @staticmethod
    def _period_metric(row: pd.Series, primary: str, fallback: str) -> float:
        if primary in row.index:
            v = pd.to_numeric(pd.Series([row.get(primary)]), errors="coerce").iloc[0]
            if np.isfinite(v):
                return float(v)
        if fallback in row.index:
            v = pd.to_numeric(pd.Series([row.get(fallback)]), errors="coerce").iloc[0]
            if np.isfinite(v):
                return float(v)
        return float("nan")

    def run(
        self,
        input_csv: Path,
        out_csv: Path,
        shape_t: float,
        mean_t: float,
        period_hit_shape_t: float,
        period_hit_snr_t: float,
        period_coverage_t: float,
    ) -> Dict[str, Any]:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        required = ["triage_status"]
        missing = [c for c in required if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing required columns: {missing}")

        if "label" not in df.columns:
            df["label"] = ""
        if "label_reason" not in df.columns:
            df["label_reason"] = ""

        if "best_shape_score" not in df.columns:
            df["best_shape_score"] = np.nan
        if "mean_shape_score" not in df.columns:
            df["mean_shape_score"] = np.nan

        if "n_candidates" in df.columns:
            n_candidate_col = "n_candidates"
        elif "n_events" in df.columns:
            n_candidate_col = "n_events"
        else:
            n_candidate_col = None

        before_counts = self._label_counts(df)

        triage_ok = df["triage_status"].fillna("").astype(str).str.strip().str.lower().eq("ok")
        relabeled = 0

        for idx in df.index[triage_ok]:
            row = df.loc[idx]
            n_candidates = (
                int(max(0.0, self._as_float(row.get(n_candidate_col, 0), default=0.0)))
                if n_candidate_col is not None
                else 0
            )
            best_shape = self._as_float(row.get("best_shape_score", float("nan")))
            mean_shape = self._as_float(row.get("mean_shape_score", float("nan")))

            if n_candidates == 0:
                label = "No_events"
                reason = "posthoc_retriage:n_candidates==0"
            elif np.isfinite(best_shape) and np.isfinite(mean_shape) and (best_shape >= shape_t) and (mean_shape >= mean_t):
                hit_shape = self._period_metric(row, "hit_rate_shape", "best_period_hit_rate_shape")
                hit_snr = self._period_metric(row, "hit_rate_snr", "best_period_hit_rate_snr")
                coverage = self._period_metric(row, "coverage_rate", "best_period_coverage_rate")

                has_period_signal = np.isfinite(hit_shape) or np.isfinite(hit_snr)
                period_hit_ok = (np.isfinite(hit_shape) and (hit_shape >= period_hit_shape_t)) or (
                    np.isfinite(hit_snr) and (hit_snr >= period_hit_snr_t)
                )
                coverage_ok = True if (not np.isfinite(coverage)) else (coverage >= period_coverage_t)
                if has_period_signal and period_hit_ok and coverage_ok:
                    label = "Periodic_candidate"
                    reason = "posthoc_retriage:shape+mean+period"
                else:
                    label = "Sparse_or_mono"
                    reason = "posthoc_retriage:shape+mean"
            else:
                label = "Noisy_trash"
                reason = "posthoc_retriage:shape_or_mean_below_threshold"

            df.at[idx, "label"] = label
            df.at[idx, "label_reason"] = reason
            relabeled += 1

        after_counts = self._label_counts(df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        print(f"[k2_retriage] input={input_csv}")
        print(f"[k2_retriage] output={out_csv}")
        print(f"[k2_retriage] triage_status_ok_rows={int(triage_ok.sum())}")
        print(f"[k2_retriage] relabeled_rows={relabeled}")
        print(f"[k2_retriage] label_counts_before={before_counts}")
        print(f"[k2_retriage] label_counts_after={after_counts}")

        return {
            "input_csv": input_csv,
            "out_csv": out_csv,
            "triage_ok_rows": int(triage_ok.sum()),
            "relabeled_rows": int(relabeled),
            "label_counts_before": before_counts,
            "label_counts_after": after_counts,
        }
