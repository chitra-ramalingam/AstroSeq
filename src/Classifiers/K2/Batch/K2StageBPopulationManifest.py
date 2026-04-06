from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2StageBPopulationManifest:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_BATCH_RESULTS_CSV = DEFAULT_OUT_DIR / "batch_results.csv"
    DEFAULT_BATCH_RESULTS_RETRIAGED_CSV = DEFAULT_OUT_DIR / "batch_results_retriaged.csv"
    DEFAULT_BATCH_RESULTS_WHITENESS_CSV = DEFAULT_OUT_DIR / "batch_results_whiteness.csv"
    DEFAULT_BEST_CSV = DEFAULT_OUT_DIR / "period_shortlist_best.csv"
    DEFAULT_QUARANTINE_CSV = DEFAULT_OUT_DIR / "period_shortlist_quarantine.csv"
    DEFAULT_FUNNEL_CSV = DEFAULT_OUT_DIR / "epic_funnel_reasons.csv"
    DEFAULT_REFERENCE_CSV = DEFAULT_OUT_DIR / "nasa_confirmed_k2_planets_reference.csv"
    DEFAULT_MASTER_CSV_NAME = "k2_stage_b_master_population_manifest.csv"
    DEFAULT_UNRESOLVED_CSV_NAME = "k2_stage_b_master_unresolved_manifest.csv"
    DEFAULT_ROLLUP_CSV_NAME = "k2_stage_b_population_rollup.csv"

    BUCKET_RESOLVED = "resolved / already classified"
    BUCKET_KNOWN_CONFIRMED = "known confirmed calibration cases"
    BUCKET_UNRESOLVED = "unresolved and still needing triage/classification"
    BUCKET_LOAD_FAILED = "load_failed / missing light curve"
    BUCKET_OUTSIDE_SCOPE = "outside current scope"
    UNRESOLVED_SOURCE_REASON = "not_in_period_stage_random_sample_n5000"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Build the Stage B K2 population manifest from saved CSV outputs and make the unresolved "
                "population explicit for the next production-style pass."
            )
        )
        p.add_argument("--batch-results-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_CSV)
        p.add_argument("--batch-results-retriaged-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_RETRIAGED_CSV)
        p.add_argument("--batch-results-whiteness-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_WHITENESS_CSV)
        p.add_argument("--best-csv", type=Path, default=cls.DEFAULT_BEST_CSV)
        p.add_argument("--quarantine-csv", type=Path, default=cls.DEFAULT_QUARANTINE_CSV)
        p.add_argument("--funnel-csv", type=Path, default=cls.DEFAULT_FUNNEL_CSV)
        p.add_argument("--reference-csv", type=Path, default=cls.DEFAULT_REFERENCE_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--master-csv", type=Path, default=None)
        p.add_argument("--unresolved-csv", type=Path, default=None)
        p.add_argument("--rollup-csv", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            batch_results_csv=Path(args.batch_results_csv),
            batch_results_retriaged_csv=Path(args.batch_results_retriaged_csv),
            batch_results_whiteness_csv=Path(args.batch_results_whiteness_csv),
            best_csv=Path(args.best_csv),
            quarantine_csv=Path(args.quarantine_csv),
            funnel_csv=Path(args.funnel_csv),
            reference_csv=Path(args.reference_csv),
            master_csv=Path(args.master_csv) if args.master_csv is not None else out_dir / cls.DEFAULT_MASTER_CSV_NAME,
            unresolved_csv=Path(args.unresolved_csv)
            if args.unresolved_csv is not None
            else out_dir / cls.DEFAULT_UNRESOLVED_CSV_NAME,
            rollup_csv=Path(args.rollup_csv) if args.rollup_csv is not None else out_dir / cls.DEFAULT_ROLLUP_CSV_NAME,
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _first_nonempty_text(*values: Any) -> str:
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text != "" and text.lower() != "nan":
                return text
        return ""

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _sample_examples(df: pd.DataFrame, limit: int = 5) -> str:
        if len(df) == 0:
            return ""
        return "|".join(df["epic_id"].astype(str).head(max(1, int(limit))).tolist())

    def _canonical_epic(self, value: Any) -> str:
        return self.helper._canonical_epic(value)

    def _prepare_epic_table(self, path: Path, *, label: str, epic_columns: Sequence[str]) -> pd.DataFrame:
        df = self._read_required_csv(path)
        epic_col = ""
        for candidate in epic_columns:
            if candidate in df.columns:
                epic_col = candidate
                break
        if epic_col == "":
            raise ValueError(f"{label} CSV missing required EPIC columns: {list(epic_columns)} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self._canonical_epic)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out

    def _prepare_funnel(self, path: Path) -> pd.DataFrame:
        raw = self._read_required_csv(path)
        expanded = self.helper._expand_funnel_details(raw)
        if "epic_id" not in expanded.columns and "query" not in expanded.columns:
            raise ValueError(f"funnel CSV missing required EPIC columns: {path}")
        epic_col = "epic_id" if "epic_id" in expanded.columns else "query"
        expanded["epic_id_norm"] = expanded[epic_col].map(self._canonical_epic)
        return (
            expanded.loc[expanded["epic_id_norm"] != ""]
            .drop_duplicates(subset=["epic_id_norm"], keep="first")
            .reset_index(drop=True)
        )

    def _prepare_reference(self, path: Path) -> pd.DataFrame:
        ref = self._read_required_csv(path)
        if "epic_id" not in ref.columns:
            raise ValueError(f"reference CSV missing required column: epic_id ({path})")
        ref["epic_id_norm"] = ref["epic_id"].map(self._canonical_epic)
        ref = ref.loc[ref["epic_id_norm"] != ""].copy()
        grouped = (
            ref.groupby("epic_id_norm", dropna=False)
            .agg(
                known_confirmed_planet_count=("pl_name", "nunique"),
                known_confirmed_planets=("pl_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                known_confirmed_hostnames=("hostname", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                known_confirmed_k2_names=("k2_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
            )
            .reset_index()
        )
        grouped["epic_id"] = grouped["epic_id_norm"].map(lambda x: f"EPIC_{x}")
        return grouped

    @staticmethod
    def _lookup_row(df: pd.DataFrame, epic_id_norm: str) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype=object)
        sub = df.loc[df["epic_id_norm"].astype(str).eq(str(epic_id_norm))]
        if len(sub) == 0:
            return pd.Series(dtype=object)
        return sub.iloc[0]

    def _resolved_reason(
        self,
        *,
        in_best: bool,
        in_quarantine: bool,
        best_row: pd.Series,
        quarantine_row: pd.Series,
        funnel_row: pd.Series,
    ) -> str:
        if in_best:
            return self._first_nonempty_text(
                best_row.get("reason", ""),
                funnel_row.get("source_reason", ""),
                "present in period_shortlist_best.csv",
            )
        if in_quarantine:
            return self._first_nonempty_text(
                quarantine_row.get("failure_category", ""),
                quarantine_row.get("failure_detail", ""),
                quarantine_row.get("source_reason", ""),
                funnel_row.get("source_reason", ""),
                "present in period_shortlist_quarantine.csv",
            )
        return self._first_nonempty_text(
            funnel_row.get("terminal_reason", ""),
            funnel_row.get("source_reason", ""),
            funnel_row.get("stage_reached", ""),
            "present in processed universe with a saved terminal downstream outcome",
        )

    def run(
        self,
        *,
        batch_results_csv: Path,
        batch_results_retriaged_csv: Path,
        batch_results_whiteness_csv: Path,
        best_csv: Path,
        quarantine_csv: Path,
        funnel_csv: Path,
        reference_csv: Path,
        master_csv: Path,
        unresolved_csv: Path,
        rollup_csv: Path,
    ) -> Dict[str, Any]:
        batch = self._prepare_epic_table(Path(batch_results_csv), label="batch results", epic_columns=["epic_id", "query"])
        retriaged = self._prepare_epic_table(
            Path(batch_results_retriaged_csv), label="batch results retriaged", epic_columns=["epic_id", "query"]
        )
        whiteness = self._prepare_epic_table(
            Path(batch_results_whiteness_csv), label="batch results whiteness", epic_columns=["epic_id", "query"]
        )
        best = self._prepare_epic_table(Path(best_csv), label="best", epic_columns=["epic", "epic_id", "query"])
        quarantine = self._prepare_epic_table(
            Path(quarantine_csv), label="quarantine", epic_columns=["epic_id", "epic", "query"]
        )
        funnel = self._prepare_funnel(Path(funnel_csv))
        reference = self._prepare_reference(Path(reference_csv))

        processed_ids = (
            set(batch["epic_id_norm"].astype(str))
            .union(set(retriaged["epic_id_norm"].astype(str)))
            .union(set(whiteness["epic_id_norm"].astype(str)))
            .union(set(funnel["epic_id_norm"].astype(str)))
        )
        processed_ids.discard("")
        confirmed_ids = set(reference["epic_id_norm"].astype(str))
        confirmed_ids.discard("")
        all_ids = sorted(processed_ids.union(confirmed_ids))

        rows: List[Dict[str, Any]] = []
        for epic_id_norm in all_ids:
            batch_row = self._lookup_row(batch, epic_id_norm)
            retriaged_row = self._lookup_row(retriaged, epic_id_norm)
            whiteness_row = self._lookup_row(whiteness, epic_id_norm)
            best_row = self._lookup_row(best, epic_id_norm)
            quarantine_row = self._lookup_row(quarantine, epic_id_norm)
            funnel_row = self._lookup_row(funnel, epic_id_norm)
            reference_row = self._lookup_row(reference, epic_id_norm)

            in_batch = len(batch_row) > 0
            in_retriaged = len(retriaged_row) > 0
            in_whiteness = len(whiteness_row) > 0
            in_funnel = len(funnel_row) > 0
            in_processed_universe = bool(in_batch or in_retriaged or in_whiteness or in_funnel)
            known_confirmed = bool(len(reference_row) > 0)
            already_shortlisted = bool(len(best_row) > 0)
            already_quarantined = bool(len(quarantine_row) > 0)
            terminal_reason = self._first_nonempty_text(funnel_row.get("terminal_reason", ""))
            source_reason = self._first_nonempty_text(funnel_row.get("source_reason", ""))
            stage_reached = self._first_nonempty_text(funnel_row.get("stage_reached", ""))
            selected_for_period_stage = self._to_bool(funnel_row.get("selected_for_period_stage", False))
            load_failed = terminal_reason.strip().lower() == "no_lightcurve/load_failed"
            outside_scope = not in_processed_universe
            unresolved = (
                in_processed_universe
                and (not known_confirmed)
                and (not load_failed)
                and (not already_shortlisted)
                and (not already_quarantined)
                and source_reason.strip().lower() == self.UNRESOLVED_SOURCE_REASON
            )

            if outside_scope:
                bucket = self.BUCKET_OUTSIDE_SCOPE
                bucket_reason = "Absent from the local processed K2 universe; present only in the confirmed-reference set."
            elif known_confirmed:
                bucket = self.BUCKET_KNOWN_CONFIRMED
                bucket_reason = "known confirmed K2 calibration case present in the local processed universe"
            elif load_failed:
                bucket = self.BUCKET_LOAD_FAILED
                bucket_reason = self._first_nonempty_text(
                    source_reason,
                    terminal_reason,
                    funnel_row.get("load_failed_exception_type", ""),
                    funnel_row.get("load_failed_exception_message", ""),
                    "no light curve / load failed",
                )
            elif unresolved:
                bucket = self.BUCKET_UNRESOLVED
                bucket_reason = self._first_nonempty_text(
                    source_reason,
                    "saved outputs show this EPIC was not yet sent through the sampled period-stage funnel",
                )
            else:
                bucket = self.BUCKET_RESOLVED
                bucket_reason = self._resolved_reason(
                    in_best=already_shortlisted,
                    in_quarantine=already_quarantined,
                    best_row=best_row,
                    quarantine_row=quarantine_row,
                    funnel_row=funnel_row,
                )

            display_epic_id = self._first_nonempty_text(
                reference_row.get("epic_id", ""),
                batch_row.get("epic_id", ""),
                retriaged_row.get("epic_id", ""),
                whiteness_row.get("epic_id", ""),
                best_row.get("epic", ""),
                quarantine_row.get("epic_id", ""),
                funnel_row.get("epic_id", ""),
                f"EPIC_{epic_id_norm}",
            )
            manifest_sources = []
            if in_batch:
                manifest_sources.append("batch_results")
            if in_retriaged:
                manifest_sources.append("batch_results_retriaged")
            if in_whiteness:
                manifest_sources.append("batch_results_whiteness")
            if in_funnel:
                manifest_sources.append("epic_funnel_reasons")
            if known_confirmed:
                manifest_sources.append("nasa_confirmed_k2_planets_reference")

            rows.append(
                {
                    "epic_id": display_epic_id,
                    "epic_id_norm": str(epic_id_norm),
                    "current_status_bucket": bucket,
                    "current_status_reason": bucket_reason,
                    "already_processed": bool(in_processed_universe),
                    "already_shortlisted": bool(already_shortlisted),
                    "already_quarantined": bool(already_quarantined),
                    "known_confirmed": bool(known_confirmed),
                    "unresolved": bool(unresolved),
                    "outside_current_scope": bool(outside_scope),
                    "load_failed_or_missing_light_curve": bool(load_failed),
                    "selected_for_period_stage": bool(selected_for_period_stage),
                    "in_processed_universe": bool(in_processed_universe),
                    "manifest_sources": "|".join(manifest_sources),
                    "batch_triage_status": self._first_nonempty_text(batch_row.get("triage_status", "")),
                    "batch_triage_usable": self._to_bool(batch_row.get("triage_usable", False)),
                    "batch_triage_why_not_usable": self._first_nonempty_text(batch_row.get("triage_why_not_usable", "")),
                    "period_terminal_reason": terminal_reason,
                    "period_source_reason": source_reason,
                    "period_stage_reached": stage_reached,
                    "best_reason": self._first_nonempty_text(best_row.get("reason", "")),
                    "quarantine_reason": self._first_nonempty_text(quarantine_row.get("reason", "")),
                    "failure_category": self._first_nonempty_text(
                        quarantine_row.get("failure_category", ""),
                        funnel_row.get("failure_category", ""),
                    ),
                    "failure_detail": self._first_nonempty_text(
                        quarantine_row.get("failure_detail", ""),
                        funnel_row.get("failure_detail", ""),
                    ),
                    "known_confirmed_planet_count": int(reference_row.get("known_confirmed_planet_count", 0))
                    if known_confirmed
                    else 0,
                    "known_confirmed_planets": self._first_nonempty_text(reference_row.get("known_confirmed_planets", "")),
                    "known_confirmed_hostnames": self._first_nonempty_text(
                        reference_row.get("known_confirmed_hostnames", "")
                    ),
                    "known_confirmed_k2_names": self._first_nonempty_text(reference_row.get("known_confirmed_k2_names", "")),
                }
            )

        master_df = pd.DataFrame(rows).sort_values(by=["current_status_bucket", "epic_id_norm"]).reset_index(drop=True)
        unresolved_df = master_df.loc[master_df["unresolved"].astype(bool)].copy().reset_index(drop=True)

        bucket_counts = master_df["current_status_bucket"].value_counts()
        rollup_rows: List[Dict[str, Any]] = [
            {"section": "summary", "metric": "total_relevant_epics", "value": int(len(master_df)), "notes": ""},
            {
                "section": "summary",
                "metric": "processed_universe_epics",
                "value": int(master_df["in_processed_universe"].astype(bool).sum()),
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "known_confirmed_unique_epics",
                "value": int(master_df["known_confirmed"].astype(bool).sum()),
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "already_shortlisted",
                "value": int(master_df["already_shortlisted"].astype(bool).sum()),
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "already_quarantined",
                "value": int(master_df["already_quarantined"].astype(bool).sum()),
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "unresolved_definition_source_reason",
                "value": self.UNRESOLVED_SOURCE_REASON,
                "notes": "Rows with this saved funnel source_reason remain the Stage B unresolved input population.",
            },
        ]
        for bucket in [
            self.BUCKET_RESOLVED,
            self.BUCKET_KNOWN_CONFIRMED,
            self.BUCKET_UNRESOLVED,
            self.BUCKET_LOAD_FAILED,
            self.BUCKET_OUTSIDE_SCOPE,
        ]:
            rollup_rows.append(
                {
                    "section": "bucket_counts",
                    "metric": bucket,
                    "value": int(bucket_counts.get(bucket, 0)),
                    "notes": "",
                }
            )
        rollup_rows.append(
            {
                "section": "examples",
                "metric": self.BUCKET_UNRESOLVED,
                "value": self._sample_examples(unresolved_df),
                "notes": "Representative unresolved EPICs for the next production-style pass.",
            }
        )
        rollup_df = pd.DataFrame(rollup_rows)

        master_csv = Path(master_csv)
        unresolved_csv = Path(unresolved_csv)
        rollup_csv = Path(rollup_csv)
        master_csv.parent.mkdir(parents=True, exist_ok=True)
        unresolved_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(master_csv, index=False)
        unresolved_df.to_csv(unresolved_csv, index=False)
        rollup_df.to_csv(rollup_csv, index=False)

        return {
            "master_csv": str(master_csv),
            "unresolved_csv": str(unresolved_csv),
            "rollup_csv": str(rollup_csv),
            "total_relevant_epics": int(len(master_df)),
            "processed_universe_epics": int(master_df["in_processed_universe"].astype(bool).sum()),
            "known_confirmed_unique_epics": int(master_df["known_confirmed"].astype(bool).sum()),
            "resolved_already_classified": int(bucket_counts.get(self.BUCKET_RESOLVED, 0)),
            "known_confirmed_calibration_cases": int(bucket_counts.get(self.BUCKET_KNOWN_CONFIRMED, 0)),
            "unresolved_needing_triage": int(bucket_counts.get(self.BUCKET_UNRESOLVED, 0)),
            "load_failed_missing_light_curve": int(bucket_counts.get(self.BUCKET_LOAD_FAILED, 0)),
            "outside_current_scope": int(bucket_counts.get(self.BUCKET_OUTSIDE_SCOPE, 0)),
        }
