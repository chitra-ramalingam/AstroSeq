from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2DetectorQualityGatedComparison:
    DEFAULT_BASELINE_RUN_DIR = Path(r"plots\k2_batch\detector_cached_5_baseline")
    DEFAULT_EXPERIMENTAL_RUN_DIR = Path(r"plots\k2_batch\detector_cached_5_exp")
    DEFAULT_QUALITY_GATED_RUN_DIR = Path(r"plots\k2_batch\detector_cached_5_quality_gated")
    DEFAULT_OUT_CSV = Path(r"plots\k2_batch\detector_quality_gated_comparison.csv")
    DEFAULT_EPIC_SUMMARY_CSV = Path(r"plots\k2_batch\detector_quality_gated_epic_summary.csv")
    DEFAULT_ROLLUP_CSV = Path(r"plots\k2_batch\detector_quality_gated_rollup.csv")
    OUTPUT_COLUMNS = [
        "epic_id",
        "query",
        "mode",
        "n_events",
        "best_shape_score",
        "best_depth_snr",
        "triage_usable",
        "triage_why_not_usable",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Compare detector default, high-recall experimental, and quality-gated experimental modes."
        )
        p.add_argument("--baseline-run-dir", type=Path, default=cls.DEFAULT_BASELINE_RUN_DIR)
        p.add_argument("--experimental-run-dir", type=Path, default=cls.DEFAULT_EXPERIMENTAL_RUN_DIR)
        p.add_argument("--quality-gated-run-dir", type=Path, default=cls.DEFAULT_QUALITY_GATED_RUN_DIR)
        p.add_argument("--out-csv", type=Path, default=cls.DEFAULT_OUT_CSV)
        p.add_argument("--epic-summary-csv", type=Path, default=cls.DEFAULT_EPIC_SUMMARY_CSV)
        p.add_argument("--rollup-csv", type=Path, default=cls.DEFAULT_ROLLUP_CSV)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            baseline_run_dir=Path(args.baseline_run_dir),
            experimental_run_dir=Path(args.experimental_run_dir),
            quality_gated_run_dir=Path(args.quality_gated_run_dir),
            out_csv=Path(args.out_csv),
            epic_summary_csv=Path(args.epic_summary_csv),
            rollup_csv=Path(args.rollup_csv),
        )

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return pd.read_csv(path)

    def _load_mode_table(self, run_dir: Path, expected_mode: str) -> pd.DataFrame:
        batch_csv = Path(run_dir) / "batch_results.csv"
        df = self._read_csv(batch_csv)
        missing_cols = [c for c in self.OUTPUT_COLUMNS if c != "mode" and c not in df.columns]
        if missing_cols:
            raise ValueError(f"{batch_csv} missing required columns: {missing_cols}")

        modes = sorted(set(df.get("detector_operating_mode", pd.Series(dtype=str)).dropna().astype(str)))
        if modes != [expected_mode]:
            raise ValueError(
                f"{batch_csv} expected detector_operating_mode={expected_mode!r}, found {modes!r}"
            )

        out = df.reindex(columns=[c for c in self.OUTPUT_COLUMNS if c != "mode"]).copy()
        out["mode"] = expected_mode
        out = out.reindex(columns=self.OUTPUT_COLUMNS).copy()
        out["epic_id"] = out["epic_id"].fillna("").astype(str)
        out["query"] = out["query"].fillna("").astype(str)
        return out.sort_values(["epic_id", "query"]).reset_index(drop=True)

    @staticmethod
    def _pair_index(df: pd.DataFrame) -> pd.Index:
        return pd.MultiIndex.from_frame(df.loc[:, ["epic_id", "query"]].copy())

    def _assert_same_epics(self, baseline: pd.DataFrame, experimental: pd.DataFrame, quality_gated: pd.DataFrame) -> None:
        baseline_idx = self._pair_index(baseline)
        experimental_idx = self._pair_index(experimental)
        quality_idx = self._pair_index(quality_gated)
        if not baseline_idx.equals(experimental_idx):
            raise ValueError("Baseline and experimental runs do not cover the same EPIC/query set.")
        if not baseline_idx.equals(quality_idx):
            raise ValueError("Baseline and quality-gated runs do not cover the same EPIC/query set.")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _recommendation(
        quality_gated_extra_events_vs_default_count: int,
        quality_gated_best_shape_score_improvement_vs_default_count: int,
        quality_gated_best_depth_snr_improvement_vs_default_count: int,
        plain_high_recall_regressed_quality_count: int,
        quality_gated_avoided_plain_regression_count: int,
    ) -> str:
        quality_improvement_count = (
            int(quality_gated_best_shape_score_improvement_vs_default_count)
            + int(quality_gated_best_depth_snr_improvement_vs_default_count)
        )
        if int(quality_gated_avoided_plain_regression_count) <= 0:
            return "stop"
        if int(quality_gated_extra_events_vs_default_count) <= 0 and quality_improvement_count <= 0:
            return "stop"
        if quality_improvement_count <= 0 and int(plain_high_recall_regressed_quality_count) > int(
            quality_gated_avoided_plain_regression_count
        ):
            return "stop"
        return "continue"

    def run(
        self,
        baseline_run_dir: Path,
        experimental_run_dir: Path,
        quality_gated_run_dir: Path,
        out_csv: Path,
        epic_summary_csv: Path,
        rollup_csv: Path,
    ) -> Dict[str, Any]:
        baseline = self._load_mode_table(
            run_dir=baseline_run_dir,
            expected_mode=str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
        )
        experimental = self._load_mode_table(
            run_dir=experimental_run_dir,
            expected_mode=str(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE),
        )
        quality_gated = self._load_mode_table(
            run_dir=quality_gated_run_dir,
            expected_mode=str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
        )
        self._assert_same_epics(baseline=baseline, experimental=experimental, quality_gated=quality_gated)

        combined = pd.concat([baseline, experimental, quality_gated], ignore_index=True)
        mode_order = {
            str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE): 0,
            str(K2BatchRunner.DETECTOR_HIGH_RECALL_EXPERIMENTAL_MODE): 1,
            str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE): 2,
        }
        combined["_mode_order"] = combined["mode"].map(mode_order).fillna(999).astype(int)
        combined = combined.sort_values(["epic_id", "_mode_order"]).drop(columns=["_mode_order"]).reset_index(drop=True)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv, index=False)

        baseline_keyed = baseline.set_index(["epic_id", "query"])
        experimental_keyed = experimental.set_index(["epic_id", "query"])
        quality_keyed = quality_gated.set_index(["epic_id", "query"])

        qg_event_delta_vs_default = (
            pd.to_numeric(quality_keyed["n_events"], errors="coerce").fillna(0.0)
            - pd.to_numeric(baseline_keyed["n_events"], errors="coerce").fillna(0.0)
        )
        qg_shape_delta_vs_exp = (
            pd.to_numeric(quality_keyed["best_shape_score"], errors="coerce")
            - pd.to_numeric(experimental_keyed["best_shape_score"], errors="coerce")
        )
        qg_depth_delta_vs_exp = (
            pd.to_numeric(quality_keyed["best_depth_snr"], errors="coerce")
            - pd.to_numeric(experimental_keyed["best_depth_snr"], errors="coerce")
        )
        qg_event_delta_vs_exp = (
            pd.to_numeric(quality_keyed["n_events"], errors="coerce").fillna(0.0)
            - pd.to_numeric(experimental_keyed["n_events"], errors="coerce").fillna(0.0)
        )
        exp_event_delta_vs_default = (
            pd.to_numeric(experimental_keyed["n_events"], errors="coerce").fillna(0.0)
            - pd.to_numeric(baseline_keyed["n_events"], errors="coerce").fillna(0.0)
        )
        exp_shape_delta_vs_default = (
            pd.to_numeric(experimental_keyed["best_shape_score"], errors="coerce")
            - pd.to_numeric(baseline_keyed["best_shape_score"], errors="coerce")
        )
        qg_shape_delta_vs_default = (
            pd.to_numeric(quality_keyed["best_shape_score"], errors="coerce")
            - pd.to_numeric(baseline_keyed["best_shape_score"], errors="coerce")
        )
        exp_depth_delta_vs_default = (
            pd.to_numeric(experimental_keyed["best_depth_snr"], errors="coerce")
            - pd.to_numeric(baseline_keyed["best_depth_snr"], errors="coerce")
        )
        qg_depth_delta_vs_default = (
            pd.to_numeric(quality_keyed["best_depth_snr"], errors="coerce")
            - pd.to_numeric(baseline_keyed["best_depth_snr"], errors="coerce")
        )

        exp_shape_regressed = (exp_shape_delta_vs_default < 0).fillna(False)
        exp_depth_regressed = (exp_depth_delta_vs_default < 0).fillna(False)
        plain_high_recall_regressed_quality = exp_shape_regressed | exp_depth_regressed
        qg_avoids_plain_regression = plain_high_recall_regressed_quality & (
            ((~exp_shape_regressed) | (qg_shape_delta_vs_default >= 0).fillna(False))
            & ((~exp_depth_regressed) | (qg_depth_delta_vs_default >= 0).fillna(False))
        )

        epic_summary = pd.DataFrame(
            {
                "epic_id": baseline_keyed.index.get_level_values("epic_id"),
                "query": baseline_keyed.index.get_level_values("query"),
                "default_n_events": pd.to_numeric(baseline_keyed["n_events"], errors="coerce"),
                "experimental_n_events": pd.to_numeric(experimental_keyed["n_events"], errors="coerce"),
                "quality_gated_n_events": pd.to_numeric(quality_keyed["n_events"], errors="coerce"),
                "experimental_delta_n_events_vs_default": exp_event_delta_vs_default,
                "quality_gated_delta_n_events_vs_default": qg_event_delta_vs_default,
                "quality_gated_delta_n_events_vs_experimental": qg_event_delta_vs_exp,
                "default_best_shape_score": pd.to_numeric(baseline_keyed["best_shape_score"], errors="coerce"),
                "experimental_best_shape_score": pd.to_numeric(experimental_keyed["best_shape_score"], errors="coerce"),
                "quality_gated_best_shape_score": pd.to_numeric(quality_keyed["best_shape_score"], errors="coerce"),
                "experimental_delta_best_shape_score_vs_default": exp_shape_delta_vs_default,
                "quality_gated_delta_best_shape_score_vs_default": qg_shape_delta_vs_default,
                "quality_gated_delta_best_shape_score_vs_experimental": qg_shape_delta_vs_exp,
                "default_best_depth_snr": pd.to_numeric(baseline_keyed["best_depth_snr"], errors="coerce"),
                "experimental_best_depth_snr": pd.to_numeric(experimental_keyed["best_depth_snr"], errors="coerce"),
                "quality_gated_best_depth_snr": pd.to_numeric(quality_keyed["best_depth_snr"], errors="coerce"),
                "experimental_delta_best_depth_snr_vs_default": exp_depth_delta_vs_default,
                "quality_gated_delta_best_depth_snr_vs_default": qg_depth_delta_vs_default,
                "quality_gated_delta_best_depth_snr_vs_experimental": qg_depth_delta_vs_exp,
                "default_triage_usable": baseline_keyed["triage_usable"].map(self._as_bool),
                "experimental_triage_usable": experimental_keyed["triage_usable"].map(self._as_bool),
                "quality_gated_triage_usable": quality_keyed["triage_usable"].map(self._as_bool),
                "default_triage_why_not_usable": baseline_keyed["triage_why_not_usable"].fillna("").astype(str),
                "experimental_triage_why_not_usable": experimental_keyed["triage_why_not_usable"].fillna("").astype(str),
                "quality_gated_triage_why_not_usable": quality_keyed["triage_why_not_usable"].fillna("").astype(str),
                "experimental_extra_events_vs_default": (exp_event_delta_vs_default > 0).fillna(False),
                "quality_gated_extra_events_vs_default": (qg_event_delta_vs_default > 0).fillna(False),
                "experimental_best_shape_improved_vs_default": (exp_shape_delta_vs_default > 0).fillna(False),
                "quality_gated_best_shape_improved_vs_default": (qg_shape_delta_vs_default > 0).fillna(False),
                "experimental_best_depth_snr_improved_vs_default": (exp_depth_delta_vs_default > 0).fillna(False),
                "quality_gated_best_depth_snr_improved_vs_default": (qg_depth_delta_vs_default > 0).fillna(False),
                "any_best_shape_score_improvement_vs_default": (
                    (exp_shape_delta_vs_default > 0).fillna(False) | (qg_shape_delta_vs_default > 0).fillna(False)
                ),
                "any_best_depth_snr_improvement_vs_default": (
                    (exp_depth_delta_vs_default > 0).fillna(False) | (qg_depth_delta_vs_default > 0).fillna(False)
                ),
                "plain_high_recall_regressed_quality": plain_high_recall_regressed_quality,
                "quality_gated_avoided_plain_regression": qg_avoids_plain_regression,
            }
        ).reset_index(drop=True)

        epic_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        epic_summary.to_csv(epic_summary_csv, index=False)

        rollup_rows = [
            {"metric": "epic_count", "value": int(len(epic_summary))},
            {"metric": "experimental_extra_events_vs_default_count", "value": int(epic_summary["experimental_extra_events_vs_default"].sum())},
            {"metric": "quality_gated_extra_events_vs_default_count", "value": int(epic_summary["quality_gated_extra_events_vs_default"].sum())},
            {"metric": "any_best_shape_score_improvement_vs_default_count", "value": int(epic_summary["any_best_shape_score_improvement_vs_default"].sum())},
            {"metric": "experimental_best_shape_score_improvement_vs_default_count", "value": int(epic_summary["experimental_best_shape_improved_vs_default"].sum())},
            {"metric": "quality_gated_best_shape_score_improvement_vs_default_count", "value": int(epic_summary["quality_gated_best_shape_improved_vs_default"].sum())},
            {"metric": "any_best_depth_snr_improvement_vs_default_count", "value": int(epic_summary["any_best_depth_snr_improvement_vs_default"].sum())},
            {"metric": "experimental_best_depth_snr_improvement_vs_default_count", "value": int(epic_summary["experimental_best_depth_snr_improved_vs_default"].sum())},
            {"metric": "quality_gated_best_depth_snr_improvement_vs_default_count", "value": int(epic_summary["quality_gated_best_depth_snr_improved_vs_default"].sum())},
            {"metric": "plain_high_recall_regressed_quality_count", "value": int(epic_summary["plain_high_recall_regressed_quality"].sum())},
            {"metric": "quality_gated_avoided_plain_regression_count", "value": int(epic_summary["quality_gated_avoided_plain_regression"].sum())},
            {"metric": "experimental_event_gain_total_vs_default", "value": float(exp_event_delta_vs_default.sum())},
            {"metric": "quality_gated_event_gain_total_vs_default", "value": float(qg_event_delta_vs_default.sum())},
            {"metric": "quality_gated_event_delta_total_vs_experimental", "value": float(qg_event_delta_vs_exp.sum())},
            {
                "metric": "recommendation",
                "value": self._recommendation(
                    quality_gated_extra_events_vs_default_count=int(epic_summary["quality_gated_extra_events_vs_default"].sum()),
                    quality_gated_best_shape_score_improvement_vs_default_count=int(
                        epic_summary["quality_gated_best_shape_improved_vs_default"].sum()
                    ),
                    quality_gated_best_depth_snr_improvement_vs_default_count=int(
                        epic_summary["quality_gated_best_depth_snr_improved_vs_default"].sum()
                    ),
                    plain_high_recall_regressed_quality_count=int(
                        epic_summary["plain_high_recall_regressed_quality"].sum()
                    ),
                    quality_gated_avoided_plain_regression_count=int(
                        epic_summary["quality_gated_avoided_plain_regression"].sum()
                    ),
                ),
            },
        ]
        rollup = pd.DataFrame(rollup_rows, columns=["metric", "value"])
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup.to_csv(rollup_csv, index=False)
        recommendation = str(rollup.loc[rollup["metric"].astype(str) == "recommendation", "value"].iloc[0])

        keeps_some_event_count_gain = bool((qg_event_delta_vs_default > 0).any())
        improves_best_shape_score_any = bool((qg_shape_delta_vs_exp > 0).fillna(False).any())
        improves_best_depth_snr_any = bool((qg_depth_delta_vs_exp > 0).fillna(False).any())
        looks_better_for_scaling = bool(
            keeps_some_event_count_gain and (improves_best_shape_score_any or improves_best_depth_snr_any)
        )

        triage_usable_true_count = int(sum(self._as_bool(v) for v in combined["triage_usable"].tolist()))
        return {
            "out_csv": str(out_csv),
            "epic_summary_csv": str(epic_summary_csv),
            "rollup_csv": str(rollup_csv),
            "row_count": int(len(combined)),
            "epic_count": int(len(baseline)),
            "keeps_some_event_count_gain": keeps_some_event_count_gain,
            "qg_event_gain_epic_count_vs_default": int((qg_event_delta_vs_default > 0).sum()),
            "qg_event_gain_total_vs_default": float(qg_event_delta_vs_default.sum()),
            "improves_best_shape_score_any": improves_best_shape_score_any,
            "qg_shape_improved_epic_count_vs_experimental": int((qg_shape_delta_vs_exp > 0).fillna(False).sum()),
            "improves_best_depth_snr_any": improves_best_depth_snr_any,
            "qg_depth_improved_epic_count_vs_experimental": int((qg_depth_delta_vs_exp > 0).fillna(False).sum()),
            "qg_event_delta_total_vs_experimental": float(qg_event_delta_vs_exp.sum()),
            "qg_event_loss_epic_count_vs_experimental": int((qg_event_delta_vs_exp < 0).sum()),
            "looks_better_for_scaling": looks_better_for_scaling,
            "triage_usable_true_count": triage_usable_true_count,
            "experimental_extra_events_vs_default_count": int(epic_summary["experimental_extra_events_vs_default"].sum()),
            "quality_gated_extra_events_vs_default_count": int(epic_summary["quality_gated_extra_events_vs_default"].sum()),
            "any_best_shape_score_improvement_vs_default_count": int(epic_summary["any_best_shape_score_improvement_vs_default"].sum()),
            "any_best_depth_snr_improvement_vs_default_count": int(epic_summary["any_best_depth_snr_improvement_vs_default"].sum()),
            "plain_high_recall_regressed_quality_count": int(epic_summary["plain_high_recall_regressed_quality"].sum()),
            "quality_gated_avoided_plain_regression_count": int(epic_summary["quality_gated_avoided_plain_regression"].sum()),
            "recommendation": recommendation,
        }
