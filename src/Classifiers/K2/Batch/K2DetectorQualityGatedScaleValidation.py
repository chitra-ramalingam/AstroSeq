from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis import (
    K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis,
)
from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Pipeline.K2PosthocRanking import K2PosthocRanking
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2DetectorQualityGatedScaleValidation:
    DEFAULT_POPULATION_BATCH_CSV = Path(r"plots\k2_batch\batch_results.csv")
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_quality_gated_scale_validation")
    DEFAULT_INITIAL_SAMPLE_N = 600
    DEFAULT_EXPANDED_SAMPLE_N = 800
    DEFAULT_MAX_SAMPLE_N = 1000
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_MAX_WORKERS = 8
    DEFAULT_MIN_WINNERS_FOR_RELIABLE_CONVERSION = 15
    DEFAULT_CONVERSION_CI_WIDTH_THRESHOLD = 0.30
    DEFAULT_DOWNSTREAM_OPERATING_MODE = str(K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME)
    CACHED_FAILED_BASELINE_CONVERSION = 134.0 / 185.0
    STRATUM_ORDER = [
        "A_unusable_error",
        "B_unusable_quality_gate",
        "C_usable_n_events_0",
        "D_usable_n_events_1",
        "E_usable_n_events_ge2_no_valid_period",
        "F_usable_validated_period",
    ]
    KNOWN_FAILURE_CATEGORIES = {"insufficient_events", "empty_histogram", "candidate_filter_rejection"}

    def __init__(self) -> None:
        self.downstream_helper = K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Run the staged K2 detector quality-gated scale validation with paired stratified sampling, "
                "weighted summaries, and a final go/no-go report."
            )
        )
        p.add_argument("--population-batch-csv", type=Path, default=cls.DEFAULT_POPULATION_BATCH_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--initial-sample-n", type=int, default=cls.DEFAULT_INITIAL_SAMPLE_N)
        p.add_argument("--expanded-sample-n", type=int, default=cls.DEFAULT_EXPANDED_SAMPLE_N)
        p.add_argument("--max-sample-n", type=int, default=cls.DEFAULT_MAX_SAMPLE_N)
        p.add_argument("--random-seed", type=int, default=cls.DEFAULT_RANDOM_SEED)
        p.add_argument("--max-workers", type=int, default=cls.DEFAULT_MAX_WORKERS)
        p.add_argument(
            "--min-winners-for-reliable-conversion",
            type=int,
            default=cls.DEFAULT_MIN_WINNERS_FOR_RELIABLE_CONVERSION,
        )
        p.add_argument(
            "--conversion-ci-width-threshold",
            type=float,
            default=cls.DEFAULT_CONVERSION_CI_WIDTH_THRESHOLD,
        )
        p.add_argument(
            "--downstream-operating-mode",
            choices=K2ShortlistPeriodRunner._operating_mode_choices(),
            default=cls.DEFAULT_DOWNSTREAM_OPERATING_MODE,
        )
        p.add_argument("--cache-only", action="store_true", help="Use cache-only detector and downstream validation fetches.")
        p.add_argument(
            "--disable-validation",
            action="store_true",
            help="Run shortlist-period downstream in cluster-only mode without validation fetches.",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            population_batch_csv=Path(args.population_batch_csv),
            out_dir=Path(args.out_dir),
            initial_sample_n=int(args.initial_sample_n),
            expanded_sample_n=int(args.expanded_sample_n),
            max_sample_n=int(args.max_sample_n),
            random_seed=int(args.random_seed),
            max_workers=int(args.max_workers),
            min_winners_for_reliable_conversion=int(args.min_winners_for_reliable_conversion),
            conversion_ci_width_threshold=float(args.conversion_ci_width_threshold),
            downstream_operating_mode=str(args.downstream_operating_mode),
            cache_only=bool(args.cache_only),
            disable_validation=bool(args.disable_validation),
        )

    @staticmethod
    def _canonical_epic(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return ""
        match = re.search(r"(\d+)", text)
        if match is None:
            return ""
        return f"EPIC_{match.group(1)}"

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return int(default)

    @staticmethod
    def _first_nonempty_text(*values: Any) -> str:
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text != "" and text.lower() != "nan":
                return text
        return ""

    def _validate_population_columns(self, df: pd.DataFrame, source: Path) -> None:
        required = ["query", "epic_id", "triage_status", "triage_usable", "n_events", "n_periods_validated"]
        missing = [col for col in required if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"{source} missing required columns: {missing}")

    def _derive_stratum_label(self, row: pd.Series) -> str:
        triage_usable = self._as_bool(row.get("triage_usable", False))
        triage_status = str(row.get("triage_status", "")).strip().lower()
        n_events = self._as_int(row.get("n_events", 0), default=0)
        n_validated = self._as_int(row.get("n_periods_validated", 0), default=0)
        if not triage_usable:
            if triage_status != "ok":
                return "A_unusable_error"
            return "B_unusable_quality_gate"
        if n_events <= 0:
            return "C_usable_n_events_0"
        if n_events == 1:
            return "D_usable_n_events_1"
        if n_validated <= 0:
            return "E_usable_n_events_ge2_no_valid_period"
        return "F_usable_validated_period"

    def _load_population(self, population_batch_csv: Path) -> pd.DataFrame:
        if not population_batch_csv.exists():
            raise FileNotFoundError(f"Population batch CSV not found: {population_batch_csv}")
        df = pd.read_csv(population_batch_csv).copy()
        self._validate_population_columns(df=df, source=population_batch_csv)
        df["query"] = df["query"].fillna("").astype(str).str.strip()
        df["epic_id_canonical"] = df["epic_id"].map(self._canonical_epic)
        missing_epic_mask = df["epic_id_canonical"].eq("")
        df.loc[missing_epic_mask, "epic_id_canonical"] = df.loc[missing_epic_mask, "query"].map(self._canonical_epic)
        df = df.loc[(df["query"] != "") & (df["epic_id_canonical"] != "")].copy()
        df = df.drop_duplicates(subset=["epic_id_canonical"], keep="first").reset_index(drop=True)
        df["stratum_label"] = df.apply(self._derive_stratum_label, axis=1)
        df["is_usable_stratum"] = df["stratum_label"].astype(str).str.startswith(("C_", "D_", "E_", "F_"))
        return df

    def _base_floor_for_target(self, target_n: int, n_strata: int) -> int:
        if target_n >= 1000:
            base = 60
        elif target_n >= 800:
            base = 50
        elif target_n >= 600:
            base = 40
        else:
            base = max(1, int(target_n // max(1, n_strata)))
        return max(1, min(base, int(target_n // max(1, n_strata))))

    def _allocate_sample_sizes(self, population: pd.DataFrame, target_n: int) -> Dict[str, int]:
        counts = (
            population.groupby("stratum_label", observed=False)
            .size()
            .reindex(self.STRATUM_ORDER, fill_value=0)
            .astype(int)
        )
        if int(counts.sum()) < int(target_n):
            raise ValueError(f"Requested target_n={target_n} exceeds available population size {int(counts.sum())}.")
        base_floor = self._base_floor_for_target(target_n=target_n, n_strata=len(self.STRATUM_ORDER))
        allocation = {label: min(int(counts[label]), base_floor) for label in self.STRATUM_ORDER}
        remaining = int(target_n - sum(allocation.values()))
        if remaining < 0:
            raise ValueError("Base stratified allocation exceeded target sample size.")
        if remaining == 0:
            return allocation

        extras = counts.astype(int) - pd.Series(allocation)
        total_extra = int(extras.clip(lower=0).sum())
        if total_extra <= 0:
            return allocation

        raw_shares: List[Tuple[str, float, float]] = []
        for label in self.STRATUM_ORDER:
            extra_pool = max(0, int(extras[label]))
            if extra_pool <= 0:
                raw_shares.append((label, 0.0, 0.0))
                continue
            raw = remaining * (extra_pool / total_extra)
            raw_shares.append((label, raw, raw - math.floor(raw)))

        for label, raw, _ in raw_shares:
            add = min(max(0, int(math.floor(raw))), max(0, int(extras[label])))
            allocation[label] += add
        assigned = sum(allocation.values())
        leftover = int(target_n - assigned)
        if leftover <= 0:
            return allocation

        for label, _, frac in sorted(raw_shares, key=lambda item: (item[2], counts[item[0]]), reverse=True):
            if leftover <= 0:
                break
            if allocation[label] >= int(counts[label]):
                continue
            allocation[label] += 1
            leftover -= 1

        if leftover > 0:
            for label in self.STRATUM_ORDER:
                if leftover <= 0:
                    break
                while allocation[label] < int(counts[label]) and leftover > 0:
                    allocation[label] += 1
                    leftover -= 1
        return allocation

    def _build_sample_manifest_df(self, population: pd.DataFrame, target_n: int, random_seed: int) -> pd.DataFrame:
        allocation = self._allocate_sample_sizes(population=population, target_n=target_n)
        work = population.copy()
        stratum_counts = (
            work.groupby("stratum_label", observed=False)
            .size()
            .reindex(self.STRATUM_ORDER, fill_value=0)
            .astype(int)
        )
        work["stratum_population_n"] = work["stratum_label"].map(stratum_counts.to_dict()).astype(int)
        rng = np.random.default_rng(int(random_seed))
        work["_sample_u"] = rng.random(len(work))
        work = (
            work.sort_values(["stratum_label", "_sample_u", "query"], kind="mergesort")
            .reset_index(drop=True)
        )
        work["sample_rank_in_stratum"] = work.groupby("stratum_label", observed=False).cumcount() + 1
        work["stratum_sample_n"] = work["stratum_label"].map(allocation).astype(int)
        sampled = work.loc[work["sample_rank_in_stratum"] <= work["stratum_sample_n"]].copy()
        sampled["stage_target_n"] = int(target_n)
        sampled["sample_weight"] = (
            pd.to_numeric(sampled["stratum_population_n"], errors="coerce")
            / pd.to_numeric(sampled["stratum_sample_n"], errors="coerce")
        )
        sampled = sampled.sort_values(["stratum_label", "sample_rank_in_stratum", "query"], kind="mergesort").reset_index(drop=True)
        return sampled

    def _write_stage_manifest(self, manifest_df: pd.DataFrame, stage_dir: Path) -> Path:
        manifest_csv = stage_dir / "sampled_epic_manifest.csv"
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_csv, index=False)
        return manifest_csv

    @staticmethod
    def _candidate_batch_csvs(run_dir: Path) -> List[Path]:
        run_dir = Path(run_dir)
        exact = run_dir / "batch_results.csv"
        candidates: List[Path] = []
        if exact.exists():
            candidates.append(exact)
        for path in sorted(run_dir.glob("*batch_results*.csv")):
            if path == exact:
                continue
            if "detector_candidate_results" in path.name.lower():
                continue
            candidates.append(path)
        return candidates

    def _resolve_existing_detector_batch_csv(
        self,
        *,
        run_dir: Path,
        expected_mode: str,
    ) -> Optional[Path]:
        for path in self._candidate_batch_csvs(run_dir):
            try:
                df = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if "detector_operating_mode" not in df.columns:
                continue
            modes = sorted(df["detector_operating_mode"].dropna().astype(str).unique().tolist())
            if modes == [str(expected_mode)]:
                return path
        return None

    def _run_detector_mode(
        self,
        *,
        sample_manifest_csv: Path,
        run_dir: Path,
        detector_operating_mode: str,
        cache_only: bool,
        max_workers: int,
    ) -> Dict[str, Any]:
        existing_batch_csv = self._resolve_existing_detector_batch_csv(
            run_dir=run_dir,
            expected_mode=str(detector_operating_mode),
        )
        if existing_batch_csv is not None:
            print(
                f"[k2_detector_quality_gated_scale_validation] "
                f"reusing existing detector batch csv={existing_batch_csv}"
            )
            return {
                "out_dir": Path(run_dir),
                "batch_results_csv": Path(existing_batch_csv),
            }
        runner = K2BatchRunner(
            out_dir=run_dir,
            input_csv=sample_manifest_csv,
            query_col="query",
            detector_operating_mode=detector_operating_mode,
            detector_only_analysis=True,
            skip_existing_epics=True,
            cache_only=bool(cache_only),
            max_workers=int(max_workers),
        )
        out = runner.run()
        resolved_batch_csv = self._resolve_existing_detector_batch_csv(
            run_dir=run_dir,
            expected_mode=str(detector_operating_mode),
        )
        if resolved_batch_csv is not None:
            out["batch_results_csv"] = Path(resolved_batch_csv)
        return out

    def _load_detector_mode_batch(self, batch_csv: Path, expected_mode: str) -> pd.DataFrame:
        if not batch_csv.exists():
            raise FileNotFoundError(f"Detector batch CSV not found: {batch_csv}")
        df = pd.read_csv(batch_csv).copy()
        if "detector_operating_mode" not in df.columns:
            raise ValueError(f"{batch_csv} missing required column detector_operating_mode")
        modes = sorted(df["detector_operating_mode"].dropna().astype(str).unique().tolist())
        if modes != [str(expected_mode)]:
            raise ValueError(f"{batch_csv} expected detector_operating_mode={expected_mode!r}, found {modes!r}")
        df["epic_id_canonical"] = df["epic_id"].map(self._canonical_epic)
        missing_mask = df["epic_id_canonical"].eq("")
        df.loc[missing_mask, "epic_id_canonical"] = df.loc[missing_mask, "query"].map(self._canonical_epic)
        df = df.loc[df["epic_id_canonical"] != ""].drop_duplicates(subset=["epic_id_canonical"], keep="first").reset_index(drop=True)
        return df

    def _build_pairwise_detector_df(
        self,
        manifest_df: pd.DataFrame,
        default_batch_csv: Path,
        quality_gated_batch_csv: Path,
    ) -> pd.DataFrame:
        default_df = self._load_detector_mode_batch(
            batch_csv=default_batch_csv,
            expected_mode=str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
        )
        qg_df = self._load_detector_mode_batch(
            batch_csv=quality_gated_batch_csv,
            expected_mode=str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
        )
        left_cols = {
            "query": "default_query",
            "triage_status": "default_triage_status",
            "triage_usable": "default_triage_usable",
            "triage_why_not_usable": "default_triage_why_not_usable",
            "n_events": "default_n_events",
            "best_shape_score": "default_best_shape_score",
            "best_depth_snr": "default_best_depth_snr",
            "n_points_after_preprocess": "default_n_points_after_preprocess",
            "error_stage": "default_error_stage",
            "error_type": "default_error_type",
            "error_msg": "default_error_msg",
        }
        right_cols = {
            "query": "quality_gated_query",
            "triage_status": "quality_gated_triage_status",
            "triage_usable": "quality_gated_triage_usable",
            "triage_why_not_usable": "quality_gated_triage_why_not_usable",
            "n_events": "quality_gated_n_events",
            "best_shape_score": "quality_gated_best_shape_score",
            "best_depth_snr": "quality_gated_best_depth_snr",
            "n_points_after_preprocess": "quality_gated_n_points_after_preprocess",
            "error_stage": "quality_gated_error_stage",
            "error_type": "quality_gated_error_type",
            "error_msg": "quality_gated_error_msg",
        }
        paired = manifest_df.copy()
        paired = paired.merge(
            default_df.reindex(columns=["epic_id_canonical"] + list(left_cols)).rename(columns=left_cols),
            how="left",
            on="epic_id_canonical",
        )
        paired = paired.merge(
            qg_df.reindex(columns=["epic_id_canonical"] + list(right_cols)).rename(columns=right_cols),
            how="left",
            on="epic_id_canonical",
        )
        for col in [
            "default_n_events",
            "quality_gated_n_events",
            "default_best_shape_score",
            "quality_gated_best_shape_score",
            "default_best_depth_snr",
            "quality_gated_best_depth_snr",
        ]:
            paired[col] = pd.to_numeric(paired.get(col, np.nan), errors="coerce")
        paired["delta_n_events"] = paired["quality_gated_n_events"].fillna(0.0) - paired["default_n_events"].fillna(0.0)
        paired["delta_best_shape_score"] = paired["quality_gated_best_shape_score"] - paired["default_best_shape_score"]
        paired["delta_best_depth_snr"] = paired["quality_gated_best_depth_snr"] - paired["default_best_depth_snr"]
        paired["gained_extra_events"] = paired["delta_n_events"] > 0
        paired["improved_best_shape_score"] = (paired["delta_best_shape_score"] > 0).fillna(False)
        paired["improved_best_depth_snr"] = (paired["delta_best_depth_snr"] > 0).fillna(False)
        paired["default_triage_usable"] = paired["default_triage_usable"].map(self._as_bool)
        paired["quality_gated_triage_usable"] = paired["quality_gated_triage_usable"].map(self._as_bool)
        return paired

    def _write_pairwise_detector_outputs(
        self,
        pairwise_df: pd.DataFrame,
        stage_dir: Path,
    ) -> Tuple[Path, Path]:
        comparison_csv = stage_dir / "paired_detector_comparison.csv"
        summary_csv = stage_dir / "paired_detector_summary.csv"
        pairwise_df.to_csv(comparison_csv, index=False)
        summary_rows = [
            {"section": "summary", "metric": "sample_epics", "value": int(len(pairwise_df))},
            {"section": "summary", "metric": "detector_winners_observed", "value": int(pairwise_df["gained_extra_events"].sum())},
            {"section": "summary", "metric": "shape_improved_observed", "value": int(pairwise_df["improved_best_shape_score"].sum())},
            {"section": "summary", "metric": "depth_snr_improved_observed", "value": int(pairwise_df["improved_best_depth_snr"].sum())},
        ]
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        return comparison_csv, summary_csv

    def _run_downstream_stage(
        self,
        *,
        detector_run_dir: Path,
        detector_batch_csv: Path,
        stage_out_dir: Path,
        operating_mode: str,
        disable_validation: bool,
        cache_only: bool,
        max_workers: int,
    ) -> Dict[str, Any]:
        stage_out_dir.mkdir(parents=True, exist_ok=True)
        batch_csv = Path(detector_batch_csv)
        ranking_out = K2PosthocRanking().run(
            input_csv=batch_csv,
            out_dir=stage_out_dir,
            period_stage_max_epics=None,
        )
        config_kwargs: Dict[str, Any] = {
            **K2ShortlistPeriodRunner._operating_mode_overrides(str(operating_mode)),
            "RAW_EPIC_LIST_CSV": str(batch_csv),
            "SHORTLIST_CSV": str(ranking_out["shortlist_top_shape_for_period_csv"]),
            "EPICS_DIR": str(detector_run_dir / "epics"),
            "OUT_DIR": str(stage_out_dir),
            "USE_RUN_SUBDIR": False,
            "RUN_ID": "sampled_scale_validation",
            "PERIOD_STAGE_SELECTION_MODE": "all",
            "ENABLE_VALIDATION": not bool(disable_validation),
            "CACHE_ONLY_FIRST": True,
            "DOWNLOAD_IF_CACHE_MISS": not bool(cache_only),
            "MAX_WORKERS": int(max_workers),
        }
        downstream_out = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(**config_kwargs)).run()
        downstream_out["run_dir"] = str(stage_out_dir)
        return downstream_out

    def _build_downstream_pairwise_df(
        self,
        manifest_df: pd.DataFrame,
        detector_pairwise_df: pd.DataFrame,
        default_downstream_run_dir: Path,
        quality_gated_downstream_run_dir: Path,
    ) -> pd.DataFrame:
        default_state = self.downstream_helper._load_downstream_state(
            label="default",
            run_dir=default_downstream_run_dir,
            best_csv=None,
            quarantine_csv=None,
            diagnostics_csv=None,
            funnel_csv=None,
        )
        qg_state = self.downstream_helper._load_downstream_state(
            label="quality-gated",
            run_dir=quality_gated_downstream_run_dir,
            best_csv=None,
            quarantine_csv=None,
            diagnostics_csv=None,
            funnel_csv=None,
        )
        default_outcomes = self.downstream_helper._build_state_outcome_frame(default_state, prefix="default").copy()
        qg_outcomes = self.downstream_helper._build_state_outcome_frame(qg_state, prefix="quality_gated").copy()
        for frame in [default_outcomes, qg_outcomes]:
            frame["epic_id_canonical"] = frame["epic_id_canonical"].map(self._canonical_epic)
            frame.drop_duplicates(subset=["epic_id_canonical"], keep="first", inplace=True)
            frame.reset_index(drop=True, inplace=True)
        analysis = manifest_df.merge(
            detector_pairwise_df.drop(columns=[c for c in manifest_df.columns if c in detector_pairwise_df.columns and c != "epic_id_canonical"]),
            how="left",
            on="epic_id_canonical",
        )
        analysis = analysis.merge(
            default_outcomes,
            how="left",
            on="epic_id_canonical",
        )
        analysis = analysis.merge(
            qg_outcomes,
            how="left",
            on="epic_id_canonical",
        )
        analysis = self.downstream_helper._append_missing_outcome_defaults(analysis, prefix="default")
        analysis = self.downstream_helper._append_missing_outcome_defaults(analysis, prefix="quality_gated")
        analysis["detector_winner"] = analysis["gained_extra_events"].fillna(False).astype(bool)
        analysis["default_shortlisted"] = analysis["default_terminal_group"].astype(str).eq("shortlisted")
        analysis["quality_gated_shortlisted"] = analysis["quality_gated_terminal_group"].astype(str).eq("shortlisted")
        analysis["quality_gated_failed_downstream"] = analysis["quality_gated_terminal_group"].astype(str).eq("failed_downstream")
        analysis["quality_gated_no_downstream_record"] = analysis["quality_gated_terminal_group"].astype(str).eq("no_downstream_record")
        analysis["downstream_improved_vs_default"] = (
            pd.to_numeric(analysis["quality_gated_downstream_outcome_score"], errors="coerce").fillna(0)
            > pd.to_numeric(analysis["default_downstream_outcome_score"], errors="coerce").fillna(0)
        )
        analysis["downstream_regressed_vs_default"] = (
            pd.to_numeric(analysis["quality_gated_downstream_outcome_score"], errors="coerce").fillna(0)
            < pd.to_numeric(analysis["default_downstream_outcome_score"], errors="coerce").fillna(0)
        )
        analysis["winner_and_qg_shortlisted"] = analysis["detector_winner"] & analysis["quality_gated_shortlisted"]
        analysis["winner_and_qg_quarantined"] = analysis["detector_winner"] & analysis["quality_gated_failed_downstream"]
        analysis["winner_and_qg_no_record"] = analysis["detector_winner"] & analysis["quality_gated_no_downstream_record"]
        analysis["winner_in_usable_strata"] = analysis["detector_winner"] & analysis["is_usable_stratum"].astype(bool)
        analysis["default_shortlisted_but_qg_not_shortlisted"] = analysis["default_shortlisted"] & (~analysis["quality_gated_shortlisted"])
        analysis["quality_gated_failure_category_norm"] = analysis.apply(
            lambda row: self._first_nonempty_text(
                row.get("quality_gated_failure_category", ""),
                row.get("quality_gated_shortlist_rejection_reason", ""),
                row.get("quality_gated_terminal_reason", ""),
                "unknown_failure_reason",
            ),
            axis=1,
        )
        analysis["quality_gated_failure_reason_key"] = analysis.apply(
            lambda row: self.downstream_helper._failure_reason_key(row, prefix="quality_gated")
            if bool(row.get("winner_and_qg_quarantined", False))
            else "",
            axis=1,
        )
        analysis["known_residual_failure"] = analysis["winner_and_qg_quarantined"] & analysis["quality_gated_failure_category_norm"].isin(
            ["insufficient_events", "empty_histogram"]
        )
        return analysis

    def _write_downstream_pairwise_outputs(
        self,
        analysis_df: pd.DataFrame,
        stage_dir: Path,
    ) -> Path:
        out_csv = stage_dir / "paired_downstream_analysis.csv"
        analysis_df.to_csv(out_csv, index=False)
        return out_csv

    @staticmethod
    def _ratio_ci(
        df: pd.DataFrame,
        *,
        numerator_col: str,
        denominator_col: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        if len(df) == 0:
            return {
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "observed_numerator": 0,
                "observed_denominator": 0,
                "weighted_numerator": 0.0,
                "weighted_denominator": 0.0,
            }
        group_specs: List[Tuple[int, int, pd.Series, pd.Series]] = []
        weighted_num = 0.0
        weighted_den = 0.0
        for _, sub in df.groupby("stratum_label", observed=False):
            n_h = int(len(sub))
            if n_h <= 0:
                continue
            N_h = int(pd.to_numeric(sub["stratum_population_n"], errors="coerce").iloc[0])
            y = pd.to_numeric(sub[numerator_col], errors="coerce").fillna(0.0)
            x = pd.to_numeric(sub[denominator_col], errors="coerce").fillna(0.0)
            weighted_num += float(N_h * y.mean())
            weighted_den += float(N_h * x.mean())
            group_specs.append((N_h, n_h, y, x))
        observed_num = int(pd.to_numeric(df[numerator_col], errors="coerce").fillna(0.0).sum())
        observed_den = int(pd.to_numeric(df[denominator_col], errors="coerce").fillna(0.0).sum())
        if weighted_den <= 0:
            return {
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "observed_numerator": observed_num,
                "observed_denominator": observed_den,
                "weighted_numerator": float(weighted_num),
                "weighted_denominator": float(weighted_den),
            }
        ratio = float(weighted_num / weighted_den)
        var_ratio = 0.0
        for N_h, n_h, y, x in group_specs:
            if n_h <= 1:
                continue
            f_h = float(n_h / max(1, N_h))
            z = y - (ratio * x)
            s2 = float(z.var(ddof=1))
            var_ratio += float((N_h ** 2) * max(0.0, 1.0 - f_h) * s2 / n_h)
        se = math.sqrt(max(0.0, var_ratio)) / float(weighted_den)
        ci_low = float(ratio - 1.96 * se)
        ci_high = float(ratio + 1.96 * se)
        if lower_bound is not None:
            ci_low = max(float(lower_bound), ci_low)
        if upper_bound is not None:
            ci_high = min(float(upper_bound), ci_high)
        return {
            "estimate": float(ratio * scale),
            "ci_low": float(ci_low * scale),
            "ci_high": float(ci_high * scale),
            "observed_numerator": observed_num,
            "observed_denominator": observed_den,
            "weighted_numerator": float(weighted_num),
            "weighted_denominator": float(weighted_den),
        }

    @staticmethod
    def _mean_ci(
        df: pd.DataFrame,
        *,
        value_col: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        if len(df) == 0:
            return {
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "observed_numerator": 0,
                "observed_denominator": 0,
                "weighted_numerator": 0.0,
                "weighted_denominator": 0.0,
            }
        total_population = 0
        total_est = 0.0
        var_mean = 0.0
        for _, sub in df.groupby("stratum_label", observed=False):
            n_h = int(len(sub))
            if n_h <= 0:
                continue
            N_h = int(pd.to_numeric(sub["stratum_population_n"], errors="coerce").iloc[0])
            y = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0)
            ybar = float(y.mean())
            total_population += N_h
            total_est += float(N_h * ybar)
            if n_h > 1:
                f_h = float(n_h / max(1, N_h))
                s2 = float(y.var(ddof=1))
                var_mean += float((N_h ** 2) * max(0.0, 1.0 - f_h) * s2 / n_h)
        observed_num = int(pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).sum())
        observed_den = int(len(df))
        if total_population <= 0:
            return {
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "observed_numerator": observed_num,
                "observed_denominator": observed_den,
                "weighted_numerator": float(total_est),
                "weighted_denominator": float(total_population),
            }
        mean = float(total_est / total_population)
        se = math.sqrt(max(0.0, var_mean)) / float(total_population)
        ci_low = float(mean - 1.96 * se)
        ci_high = float(mean + 1.96 * se)
        if lower_bound is not None:
            ci_low = max(float(lower_bound), ci_low)
        if upper_bound is not None:
            ci_high = min(float(upper_bound), ci_high)
        return {
            "estimate": float(mean * scale),
            "ci_low": float(ci_low * scale),
            "ci_high": float(ci_high * scale),
            "observed_numerator": observed_num,
            "observed_denominator": observed_den,
            "weighted_numerator": float(total_est),
            "weighted_denominator": float(total_population),
        }

    def _summary_metric_rows_for_scope(self, analysis_df: pd.DataFrame, scope_label: str, stratum_label: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        metric_specs = [
            ("detector_winner_rate", self._mean_ci, {"value_col": "detector_winner", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            ("shape_improvement_rate", self._mean_ci, {"value_col": "improved_best_shape_score", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            ("depth_snr_improvement_rate", self._mean_ci, {"value_col": "improved_best_depth_snr", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            ("default_shortlisted_rate", self._mean_ci, {"value_col": "default_shortlisted", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            ("quality_gated_shortlisted_rate", self._mean_ci, {"value_col": "quality_gated_shortlisted", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            ("quality_gated_winner_best_per_1000", self._mean_ci, {"value_col": "winner_and_qg_shortlisted", "lower_bound": 0.0, "upper_bound": 1000.0, "scale": 1000.0}),
            ("quality_gated_winner_quarantine_per_1000", self._mean_ci, {"value_col": "winner_and_qg_quarantined", "lower_bound": 0.0, "upper_bound": 1000.0, "scale": 1000.0}),
            ("default_shortlisted_but_qg_not_shortlisted_rate", self._mean_ci, {"value_col": "default_shortlisted_but_qg_not_shortlisted", "lower_bound": 0.0, "upper_bound": 1.0, "scale": 1.0}),
            (
                "downstream_conversion_rate",
                self._ratio_ci,
                {
                    "numerator_col": "winner_and_qg_shortlisted",
                    "denominator_col": "detector_winner",
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                    "scale": 1.0,
                },
            ),
            (
                "quarantine_to_best_ratio",
                self._ratio_ci,
                {
                    "numerator_col": "winner_and_qg_quarantined",
                    "denominator_col": "winner_and_qg_shortlisted",
                    "lower_bound": 0.0,
                    "upper_bound": None,
                    "scale": 1.0,
                },
            ),
            (
                "winner_share_from_usable_strata",
                self._ratio_ci,
                {
                    "numerator_col": "winner_in_usable_strata",
                    "denominator_col": "detector_winner",
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                    "scale": 1.0,
                },
            ),
        ]
        for metric_name, fn, kwargs in metric_specs:
            metric = fn(analysis_df, **kwargs)
            rows.append(
                {
                    "scope": scope_label,
                    "stratum_label": stratum_label,
                    "metric": metric_name,
                    "estimate": metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                    "observed_numerator": metric["observed_numerator"],
                    "observed_denominator": metric["observed_denominator"],
                    "weighted_numerator": metric["weighted_numerator"],
                    "weighted_denominator": metric["weighted_denominator"],
                    "population_n": int(pd.to_numeric(analysis_df["stratum_population_n"], errors="coerce").iloc[0]) if scope_label == "by_stratum" else int(
                        analysis_df.groupby("stratum_label", observed=False)["stratum_population_n"].first().sum()
                    ),
                    "sample_n": int(len(analysis_df)),
                    "weighting": "stratified_design_weight",
                }
            )
        return rows

    def _build_downstream_summary_df(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._summary_metric_rows_for_scope(analysis_df=analysis_df, scope_label="overall", stratum_label="overall"))
        for stratum_label in self.STRATUM_ORDER:
            sub = analysis_df.loc[analysis_df["stratum_label"].astype(str) == str(stratum_label)].copy()
            if len(sub) == 0:
                continue
            rows.extend(self._summary_metric_rows_for_scope(analysis_df=sub, scope_label="by_stratum", stratum_label=str(stratum_label)))

        failure_pool = analysis_df.loc[analysis_df["winner_and_qg_quarantined"].astype(bool)].copy()
        if len(failure_pool) > 0:
            weighted_failure = (
                failure_pool.groupby("quality_gated_failure_category_norm", observed=False)["sample_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            top_failure_reasons = weighted_failure.head(5).index.tolist()
            for reason in top_failure_reasons:
                reason_col = f"_failure_reason__{reason}"
                analysis_df[reason_col] = (
                    analysis_df["winner_and_qg_quarantined"].astype(bool)
                    & analysis_df["quality_gated_failure_category_norm"].astype(str).eq(str(reason))
                ).astype(int)
                metric = self._ratio_ci(
                    analysis_df,
                    numerator_col=reason_col,
                    denominator_col="winner_and_qg_quarantined",
                    lower_bound=0.0,
                    upper_bound=1.0,
                )
                rows.append(
                    {
                        "scope": "overall",
                        "stratum_label": "overall",
                        "metric": f"qg_quarantined_failure_share__{reason}",
                        "estimate": metric["estimate"],
                        "ci_low": metric["ci_low"],
                        "ci_high": metric["ci_high"],
                        "observed_numerator": metric["observed_numerator"],
                        "observed_denominator": metric["observed_denominator"],
                        "weighted_numerator": metric["weighted_numerator"],
                        "weighted_denominator": metric["weighted_denominator"],
                        "population_n": int(analysis_df.groupby("stratum_label", observed=False)["stratum_population_n"].first().sum()),
                        "sample_n": int(len(analysis_df)),
                        "weighting": "stratified_design_weight",
                    }
                )
                for stratum_label in self.STRATUM_ORDER:
                    sub = analysis_df.loc[analysis_df["stratum_label"].astype(str) == str(stratum_label)].copy()
                    if len(sub) == 0:
                        continue
                    metric = self._ratio_ci(
                        sub,
                        numerator_col=reason_col,
                        denominator_col="winner_and_qg_quarantined",
                        lower_bound=0.0,
                        upper_bound=1.0,
                    )
                    rows.append(
                        {
                            "scope": "by_stratum",
                            "stratum_label": stratum_label,
                            "metric": f"qg_quarantined_failure_share__{reason}",
                            "estimate": metric["estimate"],
                            "ci_low": metric["ci_low"],
                            "ci_high": metric["ci_high"],
                            "observed_numerator": metric["observed_numerator"],
                            "observed_denominator": metric["observed_denominator"],
                            "weighted_numerator": metric["weighted_numerator"],
                            "weighted_denominator": metric["weighted_denominator"],
                            "population_n": int(pd.to_numeric(sub["stratum_population_n"], errors="coerce").iloc[0]),
                            "sample_n": int(len(sub)),
                            "weighting": "stratified_design_weight",
                        }
                    )
        return pd.DataFrame(rows)

    def _metric_lookup(self, summary_df: pd.DataFrame, metric: str, stratum_label: str = "overall") -> Dict[str, Any]:
        sub = summary_df.loc[
            summary_df["metric"].astype(str).eq(str(metric))
            & summary_df["stratum_label"].astype(str).eq(str(stratum_label))
        ]
        if len(sub) == 0:
            return {}
        return sub.iloc[0].to_dict()

    def _build_go_no_go_report_df(self, analysis_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        conversion = self._metric_lookup(summary_df, "downstream_conversion_rate")
        winner_share_usable = self._metric_lookup(summary_df, "winner_share_from_usable_strata")
        quarantine_ratio = self._metric_lookup(summary_df, "quarantine_to_best_ratio")
        observed_winners = int(analysis_df["detector_winner"].sum())
        weighted_quarantined = analysis_df.loc[analysis_df["winner_and_qg_quarantined"].astype(bool), "sample_weight"].sum()
        weighted_failures = (
            analysis_df.loc[analysis_df["winner_and_qg_quarantined"].astype(bool)]
            .groupby("quality_gated_failure_category_norm", observed=False)["sample_weight"]
            .sum()
            .sort_values(ascending=False)
        )
        if weighted_quarantined > 0:
            familiar_mix_share = float(
                weighted_failures.reindex(["insufficient_events", "empty_histogram"], fill_value=0.0).sum()
                / weighted_quarantined
            )
            max_new_failure_share = 0.0
            for key, value in weighted_failures.items():
                if str(key) in self.KNOWN_FAILURE_CATEGORIES:
                    continue
                max_new_failure_share = max(max_new_failure_share, float(value / weighted_quarantined))
        else:
            familiar_mix_share = float("nan")
            max_new_failure_share = 0.0
        default_shortlisted = self._mean_ci(
            analysis_df,
            value_col="default_shortlisted",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        qg_quarantine = self._mean_ci(
            analysis_df,
            value_col="winner_and_qg_quarantined",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        default_regression_rate = self._mean_ci(
            analysis_df,
            value_col="default_shortlisted_but_qg_not_shortlisted",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        default_ratio = 0.0
        if default_shortlisted["estimate"] > 0:
            default_ratio = float(qg_quarantine["estimate"] / default_shortlisted["estimate"])

        checks = [
            (
                "downstream_conversion_rate",
                float(conversion.get("estimate", float("nan"))),
                max(0.60, self.CACHED_FAILED_BASELINE_CONVERSION - 0.12),
                float("inf"),
            ),
            ("observed_winner_count", float(observed_winners), 15.0, float("inf")),
            (
                "winner_share_from_usable_strata",
                float(winner_share_usable.get("estimate", float("nan"))),
                0.20,
                float("inf"),
            ),
            ("known_failure_mix_share", float(familiar_mix_share), 0.75, float("inf")),
            ("max_new_failure_share", float(max_new_failure_share), float("-inf"), 0.15),
            (
                "quarantine_to_best_ratio",
                float(quarantine_ratio.get("estimate", float("nan"))),
                float("-inf"),
                max(0.50, default_ratio + 0.10),
            ),
            (
                "default_shortlisted_to_qg_not_shortlisted_rate",
                float(default_regression_rate["estimate"]),
                float("-inf"),
                0.02,
            ),
        ]
        for metric, value, min_allowed, max_allowed in checks:
            passed = True
            if math.isnan(value):
                passed = False
            if value < min_allowed:
                passed = False
            if value > max_allowed:
                passed = False
            rows.append(
                {
                    "stratum_label": "overall",
                    "weighting": "stratified_design_weight",
                    "metric": metric,
                    "observed_value": value,
                    "min_allowed": min_allowed,
                    "max_allowed": max_allowed,
                    "passed": bool(passed),
                }
            )
        overall_pass = bool(all(bool(row["passed"]) for row in rows))
        rows.append(
            {
                "stratum_label": "overall",
                "weighting": "stratified_design_weight",
                "metric": "final_recommendation",
                "observed_value": "go" if overall_pass else "hold",
                "min_allowed": "",
                "max_allowed": "",
                "passed": overall_pass,
            }
        )
        return pd.DataFrame(rows)

    def _write_go_no_go_report(self, report_df: pd.DataFrame, stage_dir: Path) -> Tuple[Path, Path]:
        report_csv = stage_dir / "go_no_go_report.csv"
        report_txt = stage_dir / "go_no_go_report.txt"
        report_df.to_csv(report_csv, index=False)
        final_row = report_df.loc[report_df["metric"].astype(str) == "final_recommendation"].iloc[0]
        failed = report_df.loc[(report_df["metric"].astype(str) != "final_recommendation") & (~report_df["passed"].astype(bool))]
        lines = [f"recommendation: {final_row['observed_value']}"]
        if len(failed) == 0:
            lines.append("all go/no-go checks passed")
        else:
            lines.append("failed checks:")
            for _, row in failed.iterrows():
                lines.append(f"- {row['metric']}: observed={row['observed_value']} min={row['min_allowed']} max={row['max_allowed']}")
        report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_csv, report_txt

    def _evaluate_expansion(
        self,
        *,
        stage_target_n: int,
        analysis_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        expanded_sample_n: int,
        max_sample_n: int,
        min_winners_for_reliable_conversion: int,
        conversion_ci_width_threshold: float,
    ) -> Dict[str, Any]:
        conversion = self._metric_lookup(summary_df, "downstream_conversion_rate")
        observed_winners = int(analysis_df["detector_winner"].sum())
        ci_low = float(conversion.get("ci_low", float("nan")))
        ci_high = float(conversion.get("ci_high", float("nan")))
        ci_width = float(ci_high - ci_low) if (not math.isnan(ci_low) and not math.isnan(ci_high)) else float("inf")
        expand = False
        next_target: Optional[int] = None
        reason = "sufficient"
        if int(stage_target_n) < int(expanded_sample_n):
            if observed_winners < int(min_winners_for_reliable_conversion):
                expand = True
                next_target = int(expanded_sample_n)
                reason = "winner_count_too_sparse"
            elif ci_width > float(conversion_ci_width_threshold):
                expand = True
                next_target = int(expanded_sample_n)
                reason = "conversion_ci_too_wide"
        elif int(stage_target_n) < int(max_sample_n):
            if observed_winners < int(min_winners_for_reliable_conversion):
                expand = True
                next_target = int(max_sample_n)
                reason = "winner_count_still_too_sparse"
        return {
            "expand": bool(expand),
            "next_target_n": next_target,
            "reason": reason,
            "observed_winners": observed_winners,
            "conversion_ci_width": ci_width,
        }

    def _run_stage(
        self,
        *,
        manifest_df: pd.DataFrame,
        sample_manifest_csv: Path,
        stage_dir: Path,
        detector_default_run_dir: Path,
        detector_quality_gated_run_dir: Path,
        downstream_operating_mode: str,
        cache_only: bool,
        disable_validation: bool,
        max_workers: int,
    ) -> Dict[str, Any]:
        default_out = self._run_detector_mode(
            sample_manifest_csv=sample_manifest_csv,
            run_dir=detector_default_run_dir,
            detector_operating_mode=str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
            cache_only=bool(cache_only),
            max_workers=int(max_workers),
        )
        qg_out = self._run_detector_mode(
            sample_manifest_csv=sample_manifest_csv,
            run_dir=detector_quality_gated_run_dir,
            detector_operating_mode=str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
            cache_only=bool(cache_only),
            max_workers=int(max_workers),
        )
        pairwise_df = self._build_pairwise_detector_df(
            manifest_df=manifest_df,
            default_batch_csv=Path(default_out["batch_results_csv"]),
            quality_gated_batch_csv=Path(qg_out["batch_results_csv"]),
        )
        pairwise_csv, detector_summary_csv = self._write_pairwise_detector_outputs(pairwise_df=pairwise_df, stage_dir=stage_dir)
        default_downstream_dir = stage_dir / "default_downstream"
        qg_downstream_dir = stage_dir / "quality_gated_downstream"
        self._run_downstream_stage(
            detector_run_dir=detector_default_run_dir,
            detector_batch_csv=Path(default_out["batch_results_csv"]),
            stage_out_dir=default_downstream_dir,
            operating_mode=str(downstream_operating_mode),
            disable_validation=bool(disable_validation),
            cache_only=bool(cache_only),
            max_workers=int(max_workers),
        )
        self._run_downstream_stage(
            detector_run_dir=detector_quality_gated_run_dir,
            detector_batch_csv=Path(qg_out["batch_results_csv"]),
            stage_out_dir=qg_downstream_dir,
            operating_mode=str(downstream_operating_mode),
            disable_validation=bool(disable_validation),
            cache_only=bool(cache_only),
            max_workers=int(max_workers),
        )
        downstream_pairwise_df = self._build_downstream_pairwise_df(
            manifest_df=manifest_df,
            detector_pairwise_df=pairwise_df,
            default_downstream_run_dir=default_downstream_dir,
            quality_gated_downstream_run_dir=qg_downstream_dir,
        )
        downstream_pairwise_csv = self._write_downstream_pairwise_outputs(analysis_df=downstream_pairwise_df, stage_dir=stage_dir)
        summary_df = self._build_downstream_summary_df(analysis_df=downstream_pairwise_df)
        summary_csv = stage_dir / "downstream_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        go_no_go_df = self._build_go_no_go_report_df(analysis_df=downstream_pairwise_df, summary_df=summary_df)
        go_no_go_csv, go_no_go_txt = self._write_go_no_go_report(report_df=go_no_go_df, stage_dir=stage_dir)
        return {
            "pairwise_df": pairwise_df,
            "pairwise_csv": pairwise_csv,
            "detector_summary_csv": detector_summary_csv,
            "downstream_pairwise_df": downstream_pairwise_df,
            "downstream_pairwise_csv": downstream_pairwise_csv,
            "summary_df": summary_df,
            "summary_csv": summary_csv,
            "go_no_go_df": go_no_go_df,
            "go_no_go_csv": go_no_go_csv,
            "go_no_go_txt": go_no_go_txt,
        }

    def _write_final_aliases(
        self,
        *,
        manifest_df: pd.DataFrame,
        stage_out: Dict[str, Any],
        out_dir: Path,
    ) -> Dict[str, str]:
        manifest_csv = out_dir / "sampled_epic_manifest.csv"
        manifest_df.to_csv(manifest_csv, index=False)
        pairwise_df = pd.read_csv(stage_out["pairwise_csv"])
        pairwise_csv = out_dir / "paired_detector_comparison.csv"
        pairwise_df.to_csv(pairwise_csv, index=False)
        summary_df = pd.read_csv(stage_out["summary_csv"])
        summary_csv = out_dir / "downstream_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        report_df = pd.read_csv(stage_out["go_no_go_csv"])
        report_csv = out_dir / "go_no_go_report.csv"
        report_df.to_csv(report_csv, index=False)
        report_txt = out_dir / "go_no_go_report.txt"
        report_txt.write_text(Path(stage_out["go_no_go_txt"]).read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "sample_manifest_csv": str(manifest_csv),
            "paired_detector_comparison_csv": str(pairwise_csv),
            "downstream_summary_csv": str(summary_csv),
            "go_no_go_report_csv": str(report_csv),
            "go_no_go_report_txt": str(report_txt),
        }

    def run(
        self,
        *,
        population_batch_csv: Path,
        out_dir: Path,
        initial_sample_n: int = DEFAULT_INITIAL_SAMPLE_N,
        expanded_sample_n: int = DEFAULT_EXPANDED_SAMPLE_N,
        max_sample_n: int = DEFAULT_MAX_SAMPLE_N,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_workers: int = DEFAULT_MAX_WORKERS,
        min_winners_for_reliable_conversion: int = DEFAULT_MIN_WINNERS_FOR_RELIABLE_CONVERSION,
        conversion_ci_width_threshold: float = DEFAULT_CONVERSION_CI_WIDTH_THRESHOLD,
        downstream_operating_mode: str = DEFAULT_DOWNSTREAM_OPERATING_MODE,
        cache_only: bool = False,
        disable_validation: bool = False,
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        population = self._load_population(population_batch_csv=Path(population_batch_csv))
        detector_default_run_dir = out_dir / "detector_default_run"
        detector_quality_gated_run_dir = out_dir / "detector_quality_gated_run"
        stage_targets = [int(initial_sample_n), int(expanded_sample_n), int(max_sample_n)]
        stage_targets = [n for i, n in enumerate(stage_targets) if n > 0 and n not in stage_targets[:i]]
        final_manifest_df = pd.DataFrame()
        final_stage_out: Dict[str, Any] = {}
        expansion_decision: Dict[str, Any] = {}

        for stage_target_n in stage_targets:
            stage_dir = out_dir / f"stage_n{stage_target_n}"
            manifest_df = self._build_sample_manifest_df(
                population=population,
                target_n=int(stage_target_n),
                random_seed=int(random_seed),
            )
            sample_manifest_csv = self._write_stage_manifest(manifest_df=manifest_df, stage_dir=stage_dir)
            stage_out = self._run_stage(
                manifest_df=manifest_df,
                sample_manifest_csv=sample_manifest_csv,
                stage_dir=stage_dir,
                detector_default_run_dir=detector_default_run_dir,
                detector_quality_gated_run_dir=detector_quality_gated_run_dir,
                downstream_operating_mode=str(downstream_operating_mode),
                cache_only=bool(cache_only),
                disable_validation=bool(disable_validation),
                max_workers=int(max_workers),
            )
            expansion_decision = self._evaluate_expansion(
                stage_target_n=int(stage_target_n),
                analysis_df=stage_out["downstream_pairwise_df"],
                summary_df=stage_out["summary_df"],
                expanded_sample_n=int(expanded_sample_n),
                max_sample_n=int(max_sample_n),
                min_winners_for_reliable_conversion=int(min_winners_for_reliable_conversion),
                conversion_ci_width_threshold=float(conversion_ci_width_threshold),
            )
            final_manifest_df = manifest_df
            final_stage_out = stage_out
            print(
                f"[k2_detector_quality_gated_scale_validation] "
                f"stage_n={stage_target_n} observed_winners={expansion_decision['observed_winners']} "
                f"conversion_ci_width={expansion_decision['conversion_ci_width']:.4f} "
                f"expand={expansion_decision['expand']} reason={expansion_decision['reason']}"
            )
            if not bool(expansion_decision["expand"]):
                break

        alias_paths = self._write_final_aliases(
            manifest_df=final_manifest_df,
            stage_out=final_stage_out,
            out_dir=out_dir,
        )
        report_df = pd.read_csv(alias_paths["go_no_go_report_csv"])
        recommendation_row = report_df.loc[report_df["metric"].astype(str) == "final_recommendation"].iloc[0]
        summary_df = pd.read_csv(alias_paths["downstream_summary_csv"])
        conversion = self._metric_lookup(summary_df, "downstream_conversion_rate")
        print(
            f"[k2_detector_quality_gated_scale_validation] "
            f"final_sample_n={len(final_manifest_df)} "
            f"observed_winners={int(final_stage_out['downstream_pairwise_df']['detector_winner'].sum())} "
            f"downstream_conversion={conversion.get('estimate', float('nan')):.6f} "
            f"ci=[{conversion.get('ci_low', float('nan')):.6f}, {conversion.get('ci_high', float('nan')):.6f}] "
            f"recommendation={recommendation_row['observed_value']}"
        )
        return {
            **alias_paths,
            "final_sample_n": int(len(final_manifest_df)),
            "observed_winners": int(final_stage_out["downstream_pairwise_df"]["detector_winner"].sum()),
            "downstream_conversion_rate": float(conversion.get("estimate", float("nan"))),
            "downstream_conversion_ci_low": float(conversion.get("ci_low", float("nan"))),
            "downstream_conversion_ci_high": float(conversion.get("ci_high", float("nan"))),
            "final_recommendation": str(recommendation_row["observed_value"]),
            "last_expansion_reason": str(expansion_decision.get("reason", "sufficient")),
        }
