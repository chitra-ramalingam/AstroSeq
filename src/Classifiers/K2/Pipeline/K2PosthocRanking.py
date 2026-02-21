from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


class K2PosthocRanking:
    DEFAULT_INPUT = Path("plots/k2_batch/batch_results.csv")

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Posthoc K2 ranking from batch CSV only (no downloads).")
        p.add_argument(
            "--input",
            type=Path,
            default=cls.DEFAULT_INPUT,
            help=f"Input CSV (batch_results.csv or batch_results_retriaged.csv). Default: {cls.DEFAULT_INPUT}",
        )
        p.add_argument(
            "--out_dir",
            "--out-dir",
            dest="out_dir",
            type=Path,
            default=None,
            help="Output directory for shortlist CSVs. Default: input CSV directory.",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.input).parent
        return cls().run(input_csv=Path(args.input), out_dir=out_dir)

    def run(self, input_csv: Path, out_dir: Path) -> Dict[str, Any]:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        required = ["query", "triage_status", "n_events", "best_shape_score", "best_depth_snr"]
        missing = [c for c in required if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing required columns in {input_csv}: {missing}")

        if "triage_usable" not in df.columns:
            df["triage_usable"] = np.nan
        if "triage_score_global" not in df.columns:
            df["triage_score_global"] = np.nan

        work = df.copy()
        work["triage_status"] = work["triage_status"].fillna("").astype(str).str.strip().str.lower()
        work["n_events"] = pd.to_numeric(work["n_events"], errors="coerce")
        work["best_shape_score"] = pd.to_numeric(work["best_shape_score"], errors="coerce")
        work["best_depth_snr"] = pd.to_numeric(work["best_depth_snr"], errors="coerce")
        work["triage_score_global"] = pd.to_numeric(work["triage_score_global"], errors="coerce")

        shortlist_top_shape = (
            work.loc[(work["triage_status"] == "ok") & (work["n_events"] > 0)]
            .sort_values(["best_shape_score", "best_depth_snr"], ascending=[False, False])
            .head(500)
        )
        out_cols = ["query", "best_shape_score", "best_depth_snr", "n_events", "triage_usable", "triage_score_global"]
        shortlist_top_shape = shortlist_top_shape.reindex(columns=out_cols)
        shortlist_top_shape_for_period = shortlist_top_shape.head(200).copy()

        out_dir.mkdir(parents=True, exist_ok=True)
        shortlist_top_shape_csv = out_dir / "shortlist_top_shape.csv"
        shortlist_period_csv = out_dir / "shortlist_top_shape_for_period.csv"
        shortlist_top_shape.to_csv(shortlist_top_shape_csv, index=False)
        shortlist_top_shape_for_period.to_csv(shortlist_period_csv, index=False)

        print(f"[k2_rank_posthoc] input={input_csv}")
        print(f"[k2_rank_posthoc] shortlist_top_shape_csv={shortlist_top_shape_csv} rows={len(shortlist_top_shape)}")
        print(f"[k2_rank_posthoc] shortlist_top_shape_for_period_csv={shortlist_period_csv} rows={len(shortlist_top_shape_for_period)}")

        return {
            "input_csv": input_csv,
            "shortlist_top_shape_csv": shortlist_top_shape_csv,
            "shortlist_top_shape_for_period_csv": shortlist_period_csv,
            "rows_shortlist_top_shape": int(len(shortlist_top_shape)),
            "rows_shortlist_top_shape_for_period": int(len(shortlist_top_shape_for_period)),
        }
