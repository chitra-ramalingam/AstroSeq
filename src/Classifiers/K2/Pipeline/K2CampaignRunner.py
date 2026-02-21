from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2CampaignRunner:
    DEFAULT_OUT_DIR = Path("plots/k2_batch")
    DEFAULT_INPUT_CSV = Path("data/k2_target_lists/K2Campaign5targets.csv")
    DEFAULT_QUERY_COL = "EPIC ID"
    DEFAULT_QUERIES = ["EPIC 211797674"]

    def build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="K2 batch pipeline runner")
        p.add_argument(
            "--out-dir",
            type=Path,
            default=self.DEFAULT_OUT_DIR,
            help=f"Output directory for batch artifacts (default: {self.DEFAULT_OUT_DIR})",
        )
        p.add_argument(
            "--input-csv",
            type=Path,
            default=self.DEFAULT_INPUT_CSV,
            help=f"CSV containing a query column (default: {self.DEFAULT_INPUT_CSV})",
        )
        p.add_argument("--query-col", type=str, default=self.DEFAULT_QUERY_COL, help="Query column name in --input-csv")
        p.add_argument(
            "--query",
            dest="queries",
            action="append",
            default=None,
            help="EPIC query; repeat flag or pass comma-separated values",
        )

        p.add_argument("--top-k-periods", type=int, default=3)
        p.add_argument("--period-candidate-pool", type=int, default=24)
        p.add_argument("--phase-tol", type=float, default=0.03)
        p.add_argument("--max-period-days", type=float, default=40.0)
        p.add_argument("--min-hits-for-period", type=int, default=3)
        p.add_argument("--period-tol-frac", type=float, default=0.01)

        p.add_argument("--noise-mode", type=str, default="strict", choices=["strict", "discovery"])
        p.add_argument("--limit", type=int, default=50)
        p.add_argument("--exptime", type=str, default=None)

        p.add_argument("--validator-tol-days", type=float, default=0.12)
        p.add_argument("--validator-outer-window-days", type=float, default=2.0)
        p.add_argument("--validator-min-duration-cadences", type=int, default=3)
        p.add_argument("--validator-shape-threshold", type=float, default=0.6)
        p.add_argument("--validator-snr-threshold", type=float, default=4.0)

        p.add_argument("--periodic-shape-threshold", type=float, default=0.75)
        p.add_argument("--periodic-hit-rate-shape-threshold", type=float, default=0.30)
        p.add_argument("--periodic-coverage-threshold", type=float, default=0.85)

        p.add_argument("--whiteness_alpha", type=float, default=None, help="Whiteness p-value alpha threshold.")
        p.add_argument("--noisy-whiteness-threshold", type=float, default=None)
        p.add_argument("--noisy-step-threshold", type=float, default=None)
        p.add_argument("--cache_only", action="store_true", help="Use local cache only; never download missing products.")
        p.add_argument(
            "--retriage_batch",
            action="store_true",
            help="Recompute triage/labels from stored batch_results metrics only (no downloads), then rebuild leaderboards.",
        )
        p.add_argument(
            "--rebuild_leaderboards",
            action="store_true",
            help="Rebuild leaderboard CSVs from an existing batch_results.csv and exit.",
        )
        p.add_argument(
            "--input",
            type=Path,
            default=None,
            help="Path to batch_results.csv used with --rebuild_leaderboards.",
        )
        return p

    def _build_batch_runner(self, args: argparse.Namespace, queries, input_csv) -> K2BatchRunner:
        return K2BatchRunner(
            out_dir=args.out_dir,
            queries=queries,
            input_csv=input_csv,
            query_col=args.query_col,
            top_k_periods=args.top_k_periods,
            period_candidate_pool=args.period_candidate_pool,
            phase_tol=args.phase_tol,
            max_period_days=args.max_period_days,
            min_hits_for_period=args.min_hits_for_period,
            period_tol_frac=args.period_tol_frac,
            noise_mode=args.noise_mode,
            limit=args.limit,
            exptime=args.exptime,
            validator_tol_days=args.validator_tol_days,
            validator_outer_window_days=args.validator_outer_window_days,
            validator_min_duration_cadences=args.validator_min_duration_cadences,
            validator_shape_threshold=args.validator_shape_threshold,
            validator_snr_threshold=args.validator_snr_threshold,
            periodic_shape_threshold=args.periodic_shape_threshold,
            periodic_hit_rate_shape_threshold=args.periodic_hit_rate_shape_threshold,
            periodic_coverage_threshold=args.periodic_coverage_threshold,
            whiteness_alpha=args.whiteness_alpha,
            noisy_whiteness_threshold=args.noisy_whiteness_threshold,
            noisy_step_threshold=args.noisy_step_threshold,
            cache_only=args.cache_only,
        )

    def run(self) -> None:
        args = self.build_parser().parse_args()
        if args.rebuild_leaderboards or args.retriage_batch:
            batch_csv = args.input if args.input is not None else (args.out_dir / "batch_results.csv")
            if (args.input is not None) and (not batch_csv.exists()) and (not batch_csv.is_absolute()):
                alt = args.out_dir / batch_csv
                if alt.exists():
                    batch_csv = alt
            runner = self._build_batch_runner(args=args, queries=[], input_csv=None)
            if args.retriage_batch:
                retriage = runner.retriage_batch_results(batch_csv=batch_csv, write=True)
                batch_csv = retriage["batch_results_csv"]
            out = runner.rebuild_leaderboards(batch_csv=batch_csv)
            runner._print_finalize_summary(results_df=out["results_df"])
            if args.retriage_batch:
                print(f"Retriaged from: {batch_csv}")
            print(f"Rebuilt from: {out['batch_results_csv']}")
            print(f"leaderboard_periodic.csv: {out['leaderboard_periodic_csv']}")
            print(f"leaderboard_sparse.csv: {out['leaderboard_sparse_csv']}")
            print(f"leaderboard_top_shape.csv: {out['leaderboard_top_shape_csv']}")
            print(f"leaderboard_top_snr.csv: {out['leaderboard_top_snr_csv']}")
            return

        input_csv = args.input_csv if (args.input_csv is not None and Path(args.input_csv).exists()) else None
        queries = args.queries if args.queries else (None if input_csv is not None else list(self.DEFAULT_QUERIES))
        if (not queries) and (input_csv is None):
            raise SystemExit("No queries available. Provide --query or create the default input csv.")

        exptime_value: Optional[str] = args.exptime if args.exptime not in {"", "none", "None", None} else None
        args.exptime = exptime_value

        runner = self._build_batch_runner(args=args, queries=queries, input_csv=input_csv)
        out = runner.run()
        print(f"Outputs saved under: {out['out_dir']}")
        print(f"batch_results.csv: {out['batch_results_csv']}")
        print(f"leaderboard_periodic.csv: {out['leaderboard_periodic_csv']}")
        print(f"leaderboard_sparse.csv: {out['leaderboard_sparse_csv']}")
        print(f"leaderboard_top_shape.csv: {out['leaderboard_top_shape_csv']}")
        print(f"leaderboard_top_snr.csv: {out['leaderboard_top_snr_csv']}")
