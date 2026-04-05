from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.Classifiers.K2.Batch.K2CachedFailedBroaderDownstreamRunner import K2CachedFailedBroaderDownstreamRunner
from src.Classifiers.K2.Batch.K2DetectorQualityGatedBroaderCachedFailedDownstreamReport import (
    K2DetectorQualityGatedBroaderCachedFailedDownstreamReport,
)
from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig


class K2DetectorQualityGatedConditionalMcc2LimitedBroaderValidation:
    DEFAULT_OUT_DIR = Path(
        r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream_conditional_mcc2_experiment"
    )
    DEFAULT_PLAN_TXT_NAME = "limited_broader_validation_plan.txt"

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Prepare or run the next limited broader validation step using the guarded conditional MCC=2 "
                "experimental operating mode. This remains a limited experimental carve-out only; it does not "
                "change the default policy and should not be treated as an automatic scale-up path."
            )
        )
        p.add_argument(
            "--shards-root",
            type=Path,
            default=K2CachedFailedBroaderDownstreamRunner.DEFAULT_QUALITY_GATED_SHARDS_ROOT,
        )
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--winners-csv", type=Path, default=K2DetectorQualityGatedBroaderCachedFailedDownstreamReport.DEFAULT_WINNERS_CSV)
        p.add_argument("--max-workers", type=int, default=K2CachedFailedBroaderDownstreamRunner.DEFAULT_MAX_WORKERS)
        p.add_argument("--disable-validation", action="store_true")
        p.add_argument("--prepare-only", action="store_true")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            shards_root=Path(args.shards_root),
            out_dir=Path(args.out_dir),
            winners_csv=Path(args.winners_csv),
            max_workers=int(args.max_workers),
            disable_validation=bool(args.disable_validation),
            prepare_only=bool(args.prepare_only),
        )

    @staticmethod
    def _write_plan_txt(path: Path, *, shards_root: Path, out_dir: Path, winners_csv: Path, disable_validation: bool, max_workers: int) -> str:
        run_cmd = (
            ".\\.venv\\Scripts\\python.exe main.py k2_detector_quality_gated_conditional_mcc2_limited_broader_validation "
            f"--shards-root \"{shards_root}\" --out-dir \"{out_dir}\" --winners-csv \"{winners_csv}\" "
            f"--max-workers {int(max_workers)}"
        )
        if disable_validation:
            run_cmd += " --disable-validation"
        report_best = out_dir / "period_shortlist_best.csv"
        report_quarantine = out_dir / "period_shortlist_quarantine.csv"
        report_funnel = out_dir / "epic_funnel_reasons.csv"
        followup_report_cmd = (
            ".\\.venv\\Scripts\\python.exe main.py k2_detector_quality_gated_broader_cached_failed_downstream_report "
            f"--out-dir \"{out_dir}\" --winners-csv \"{winners_csv}\" "
            f"--best-csv \"{report_best}\" --quarantine-csv \"{report_quarantine}\" --funnel-csv \"{report_funnel}\""
        )
        lines = [
            "limited_broader_validation: conditional MCC=2 carve-out",
            f"operating_mode: {K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME}",
            "policy_status: experimental_flag_only",
            "default_global_policy_changed: false",
            "supported_experimental_policy: false",
            "automatic_scale_up_scheduled: false",
            "manual_invocation_only: true",
            f"prepare_only_supported: true",
            "commands:",
            f"1. {run_cmd}",
            f"2. {followup_report_cmd}",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return run_cmd

    def run(
        self,
        *,
        shards_root: Path,
        out_dir: Path,
        winners_csv: Path,
        max_workers: int,
        disable_validation: bool,
        prepare_only: bool,
    ) -> Dict[str, Any]:
        shards_root = Path(shards_root).resolve()
        out_dir = Path(out_dir).resolve()
        winners_csv = Path(winners_csv).resolve()
        plan_txt = out_dir / self.DEFAULT_PLAN_TXT_NAME
        run_command = self._write_plan_txt(
            plan_txt,
            shards_root=shards_root,
            out_dir=out_dir,
            winners_csv=winners_csv,
            disable_validation=bool(disable_validation),
            max_workers=int(max_workers),
        )
        if prepare_only:
            return {
                "prepare_only": True,
                "plan_txt": plan_txt,
                "run_command": run_command,
                "out_dir": out_dir,
                "operating_mode": K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME,
            }

        runner_out = K2CachedFailedBroaderDownstreamRunner().run(
            shards_root=shards_root,
            out_dir=out_dir,
            operating_mode=str(K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME),
            max_workers=int(max_workers),
            disable_validation=bool(disable_validation),
        )
        report_out = K2DetectorQualityGatedBroaderCachedFailedDownstreamReport().run(
            winners_csv=winners_csv,
            best_csv=Path(runner_out["period_shortlist_best_csv"]),
            quarantine_csv=Path(runner_out["period_shortlist_quarantine_csv"]),
            funnel_csv=Path(runner_out["epic_funnel_reasons_csv"]),
            summary_csv=out_dir / K2DetectorQualityGatedBroaderCachedFailedDownstreamReport.DEFAULT_SUMMARY_CSV_NAME,
            quarantined_winners_csv=out_dir / K2DetectorQualityGatedBroaderCachedFailedDownstreamReport.DEFAULT_QUARANTINED_WINNERS_CSV_NAME,
            best_winners_csv=out_dir / K2DetectorQualityGatedBroaderCachedFailedDownstreamReport.DEFAULT_BEST_WINNERS_CSV_NAME,
        )
        return {
            "prepare_only": False,
            "plan_txt": plan_txt,
            "run_command": run_command,
            "out_dir": out_dir,
            "operating_mode": K2ShortlistPeriodConfig.SCALE_VALIDATION_CONDITIONAL_MCC2_MODE_NAME,
            "runner_out": runner_out,
            "report_out": report_out,
        }
