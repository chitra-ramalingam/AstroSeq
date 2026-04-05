from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Pipeline.K2PosthocRanking import K2PosthocRanking


class K2CachedFailedBroaderDownstreamRunner:
    DEFAULT_QUALITY_GATED_SHARDS_ROOT = Path(r"plots\k2_batch\detector_cached_failed_broader_quality_gated_shards")
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream")
    DEFAULT_MERGED_BATCH_CSV = "merged_batch_results.csv"
    DEFAULT_INPUT_MANIFEST_CSV = "downstream_input_shards.csv"
    DEFAULT_OPERATING_MODE = str(K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME)
    DEFAULT_MAX_WORKERS = 8

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Build broader downstream shortlist-period outputs from cached-failed detector shard outputs "
                "without rerunning detector work."
            )
        )
        p.add_argument("--shards-root", type=Path, default=cls.DEFAULT_QUALITY_GATED_SHARDS_ROOT)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--operating-mode", type=str, default=cls.DEFAULT_OPERATING_MODE, choices=K2ShortlistPeriodRunner._operating_mode_choices())
        p.add_argument("--max-workers", type=int, default=cls.DEFAULT_MAX_WORKERS)
        p.add_argument(
            "--disable-validation",
            action="store_true",
            help="Skip cached lightcurve validation and run cluster-only downstream period inference.",
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            shards_root=Path(args.shards_root),
            out_dir=Path(args.out_dir),
            operating_mode=str(args.operating_mode),
            max_workers=int(args.max_workers),
            disable_validation=bool(args.disable_validation),
        )

    @staticmethod
    def _extract_epic(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return ""
        match = re.search(r"(\d+)", text)
        return match.group(1) if match is not None else ""

    @staticmethod
    def _discover_shard_dirs(shards_root: Path) -> List[Path]:
        if not shards_root.exists():
            raise FileNotFoundError(f"Shard root not found: {shards_root}")
        shard_dirs = [p for p in sorted(shards_root.iterdir()) if p.is_dir()]
        if len(shard_dirs) == 0:
            raise FileNotFoundError(f"No shard directories found under: {shards_root}")
        return shard_dirs

    def _load_and_annotate_shards(self, shard_dirs: List[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        merged_rows: List[pd.DataFrame] = []
        manifest_rows: List[Dict[str, Any]] = []
        for shard_dir in shard_dirs:
            batch_csv = shard_dir / "batch_results.csv"
            epics_dir = shard_dir / "epics"
            if not batch_csv.exists():
                raise FileNotFoundError(f"Missing shard batch_results.csv: {batch_csv}")
            if not epics_dir.exists():
                raise FileNotFoundError(f"Missing shard epics dir: {epics_dir}")

            df = pd.read_csv(batch_csv)
            if "query" not in df.columns:
                raise ValueError(f"{batch_csv} missing required column: query")
            epic_series = (
                df["epic_id"].map(self._extract_epic)
                if "epic_id" in df.columns
                else df["query"].map(self._extract_epic)
            )
            df = df.copy()
            df["epic_id"] = epic_series
            df["source_shard"] = shard_dir.name
            df["epic_dir"] = df["epic_id"].map(
                lambda epic: str(epics_dir / f"EPIC_{epic}") if str(epic).strip() != "" else ""
            )
            merged_rows.append(df)
            manifest_rows.append(
                {
                    "shard_dir": str(shard_dir),
                    "batch_results_csv": str(batch_csv),
                    "epics_dir": str(epics_dir),
                    "row_count": int(len(df)),
                    "query_count": int(df["query"].fillna("").astype(str).str.strip().ne("").sum()),
                    "unique_epic_count": int(df["epic_id"].fillna("").astype(str).str.strip().ne("").sum()),
                }
            )

        merged = pd.concat(merged_rows, ignore_index=True)
        manifest = pd.DataFrame(manifest_rows)
        dup_query = merged["query"].fillna("").astype(str).str.strip()
        dup_query = dup_query.loc[dup_query != ""]
        dup_counts = dup_query.value_counts()
        dup_values = dup_counts.loc[dup_counts > 1]
        if len(dup_values) > 0:
            sample = dup_values.head(10).to_dict()
            raise ValueError(f"Duplicate query rows detected across shard inputs: {sample}")
        return merged, manifest

    def run(
        self,
        *,
        shards_root: Path,
        out_dir: Path,
        operating_mode: str = DEFAULT_OPERATING_MODE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        disable_validation: bool = False,
    ) -> Dict[str, Any]:
        shard_dirs = self._discover_shard_dirs(shards_root=shards_root)
        merged_batch_df, manifest_df = self._load_and_annotate_shards(shard_dirs=shard_dirs)

        out_dir.mkdir(parents=True, exist_ok=True)
        merged_batch_csv = out_dir / self.DEFAULT_MERGED_BATCH_CSV
        manifest_csv = out_dir / self.DEFAULT_INPUT_MANIFEST_CSV
        merged_batch_df.to_csv(merged_batch_csv, index=False)
        manifest_df.to_csv(manifest_csv, index=False)

        ranking_out = K2PosthocRanking().run(
            input_csv=merged_batch_csv,
            out_dir=out_dir,
            period_stage_max_epics=None,
        )

        first_epics_dir = shard_dirs[0] / "epics"
        config_kwargs: Dict[str, Any] = {
            **K2ShortlistPeriodRunner._operating_mode_overrides(str(operating_mode)),
            "RAW_EPIC_LIST_CSV": str(merged_batch_csv),
            "SHORTLIST_CSV": str(ranking_out["shortlist_top_shape_for_period_csv"]),
            "EPICS_DIR": str(first_epics_dir),
            "OUT_DIR": str(out_dir),
            "USE_RUN_SUBDIR": False,
            "RUN_ID": "detector_cached_failed_broader_quality_gated_downstream",
            "PERIOD_STAGE_SELECTION_MODE": "all",
            "CACHE_ONLY_FIRST": True,
            "DOWNLOAD_IF_CACHE_MISS": False,
            "ENABLE_VALIDATION": not bool(disable_validation),
            "MAX_WORKERS": int(max_workers),
        }
        downstream_out = K2ShortlistPeriodRunner(config=K2ShortlistPeriodConfig(**config_kwargs)).run()
        return {
            "shards_root": shards_root,
            "shard_count": int(len(shard_dirs)),
            "out_dir": out_dir,
            "merged_batch_csv": merged_batch_csv,
            "input_manifest_csv": manifest_csv,
            "shortlist_top_shape_csv": ranking_out["shortlist_top_shape_csv"],
            "shortlist_top_shape_for_period_csv": ranking_out["shortlist_top_shape_for_period_csv"],
            "period_shortlist_best_csv": downstream_out["out_best_csv"],
            "period_shortlist_quarantine_csv": downstream_out["out_quarantine_csv"],
            "period_shortlist_diagnostics_csv": downstream_out["out_diagnostics_csv"],
            "epic_funnel_reasons_csv": downstream_out["out_epic_funnel_reasons_csv"],
            "period_shortlist_summary_csv": downstream_out["out_summary_csv"],
            "period_shortlist_summary_unique_epicp_csv": downstream_out["out_summary_unique_epicp_csv"],
            "period_shortlist_summary_validated_only_csv": downstream_out["out_summary_validated_only_csv"],
            "period_hist_counts_csv": downstream_out["out_period_hist_counts_csv"],
            "validation_enabled": not bool(disable_validation),
        }
