from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2ConfirmedPlanetRecallAudit:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_BATCH_RESULTS_CSV = DEFAULT_OUT_DIR / "batch_results.csv"
    DEFAULT_BEST_CSV = DEFAULT_OUT_DIR / "period_shortlist_best.csv"
    DEFAULT_QUARANTINE_CSV = DEFAULT_OUT_DIR / "period_shortlist_quarantine.csv"
    DEFAULT_FUNNEL_CSV = DEFAULT_OUT_DIR / "epic_funnel_reasons.csv"
    DEFAULT_DIAGNOSTICS_CSV = DEFAULT_OUT_DIR / "period_shortlist_diagnostics.csv"
    DEFAULT_REFERENCE_CSV = DEFAULT_OUT_DIR / "nasa_confirmed_k2_planets_reference.csv"
    DEFAULT_AUDIT_CSV_NAME = "k2_confirmed_planet_recall_audit.csv"
    DEFAULT_ROLLUP_CSV_NAME = "k2_confirmed_planet_recall_rollup.csv"
    DEFAULT_FALSE_NEGATIVES_CSV_NAME = "k2_confirmed_false_negatives.csv"
    NASA_REFERENCE_URL = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+ps.pl_name,ps.hostname,ps.default_flag,ps.disc_facility,"
        "k2names.epic_id,k2names.k2_name+from+ps+left+join+k2names+on+ps.pl_name+=+k2names.pl_name+"
        "where+ps.disc_facility+like+%27%25K2%25%27+and+ps.default_flag+=+1+order+by+ps.pl_name&format=csv"
    )

    OUTCOME_RECOVERED_IN_BEST = "recovered_in_best"
    OUTCOME_RECOVERED_IN_QUARANTINE = "recovered_in_quarantine"
    OUTCOME_DETECTED_BUT_FAILED_DOWNSTREAM = "detected_but_failed_downstream"
    OUTCOME_NO_EVENTS_AFTER_FILTERS = "no_events_after_filters"
    OUTCOME_NOT_SEEN = "not_seen / not_matched"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Audit NASA-confirmed K2 planets against the current AstroSeq K2 outputs, distinguishing "
                "best recall, best-or-quarantine recall, and confirmed false negatives."
            )
        )
        p.add_argument("--batch-results-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_CSV)
        p.add_argument("--best-csv", type=Path, default=cls.DEFAULT_BEST_CSV)
        p.add_argument("--quarantine-csv", type=Path, default=cls.DEFAULT_QUARANTINE_CSV)
        p.add_argument("--funnel-csv", type=Path, default=cls.DEFAULT_FUNNEL_CSV)
        p.add_argument("--diagnostics-csv", type=Path, default=cls.DEFAULT_DIAGNOSTICS_CSV)
        p.add_argument("--reference-csv", type=Path, default=cls.DEFAULT_REFERENCE_CSV)
        p.add_argument("--refresh-reference", action="store_true", help="Refresh the NASA confirmed-K2 reference CSV.")
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--audit-csv", type=Path, default=None)
        p.add_argument("--rollup-csv", type=Path, default=None)
        p.add_argument("--false-negatives-csv", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            batch_results_csv=Path(args.batch_results_csv),
            best_csv=Path(args.best_csv),
            quarantine_csv=Path(args.quarantine_csv),
            funnel_csv=Path(args.funnel_csv),
            diagnostics_csv=Path(args.diagnostics_csv),
            reference_csv=Path(args.reference_csv),
            refresh_reference=bool(args.refresh_reference),
            audit_csv=Path(args.audit_csv) if args.audit_csv is not None else out_dir / cls.DEFAULT_AUDIT_CSV_NAME,
            rollup_csv=Path(args.rollup_csv) if args.rollup_csv is not None else out_dir / cls.DEFAULT_ROLLUP_CSV_NAME,
            false_negatives_csv=Path(args.false_negatives_csv)
            if args.false_negatives_csv is not None
            else out_dir / cls.DEFAULT_FALSE_NEGATIVES_CSV_NAME,
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
    def _first_numeric(*values: Any) -> float:
        for value in values:
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(num):
                return float(num)
        return float("nan")

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "t", "yes", "y"}

    def _canonical_epic(self, value: Any) -> str:
        return self.helper._canonical_epic(value)

    def _fetch_reference_df(self) -> pd.DataFrame:
        req = urllib.request.Request(self.NASA_REFERENCE_URL, headers={"User-Agent": "AstroSeq/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode("utf-8")
        return pd.read_csv(io.StringIO(text))

    def _load_reference_df(self, *, reference_csv: Path, refresh_reference: bool) -> pd.DataFrame:
        reference_csv = Path(reference_csv).resolve()
        if refresh_reference or (not reference_csv.exists()):
            reference_df = self._fetch_reference_df()
            reference_csv.parent.mkdir(parents=True, exist_ok=True)
            reference_df.to_csv(reference_csv, index=False)
            return reference_df
        return self._read_required_csv(reference_csv)

    def _prepare_reference(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        work = df.copy()
        if "epic_id" not in work.columns:
            raise ValueError("NASA confirmed reference is missing required column: epic_id")
        work["epic_id_norm"] = work["epic_id"].map(self._canonical_epic)
        unmatched_reference_rows = int(work["epic_id_norm"].eq("").sum())
        matched = work.loc[work["epic_id_norm"] != ""].copy()
        grouped = (
            matched.groupby("epic_id_norm", dropna=False)
            .agg(
                nasa_confirmed_planet_count=("pl_name", "nunique"),
                nasa_confirmed_planets=("pl_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                nasa_hostnames=("hostname", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                nasa_k2_names=("k2_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                nasa_disc_facility=("disc_facility", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
            )
            .reset_index()
        )
        grouped["epic_id"] = grouped["epic_id_norm"].map(lambda x: f"EPIC_{x}")
        return grouped, unmatched_reference_rows

    def _prepare_table(self, df: pd.DataFrame, *, epic_col: str, label: str) -> pd.DataFrame:
        if epic_col not in df.columns:
            raise ValueError(f"{label} CSV missing required column: {epic_col}")
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self._canonical_epic)
        out = out.loc[out["epic_id_norm"] != ""].reset_index(drop=True)
        return out

    def _prepare_best(self, path: Path) -> pd.DataFrame:
        best = self._prepare_table(self._read_required_csv(path), epic_col="epic", label="best")
        return best.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_quarantine(self, path: Path) -> pd.DataFrame:
        quarantine = self._prepare_table(self._read_required_csv(path), epic_col="epic_id", label="quarantine")
        return quarantine.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_funnel(self, path: Path) -> pd.DataFrame:
        funnel = self.helper._expand_funnel_details(self._read_required_csv(path))
        funnel = self._prepare_table(funnel, epic_col="epic_id", label="funnel")
        return funnel.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    def _prepare_batch_results(self, path: Path) -> pd.DataFrame:
        batch = self._read_required_csv(path)
        if "epic_id" in batch.columns:
            batch["epic_id_norm"] = batch["epic_id"].map(self._canonical_epic)
        elif "query" in batch.columns:
            batch["epic_id_norm"] = batch["query"].map(self._canonical_epic)
        else:
            raise ValueError(f"batch results CSV missing epic_id/query columns: {path}")
        batch = batch.loc[batch["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return batch

    @staticmethod
    def _lookup_row(df: pd.DataFrame, epic_id_norm: str) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype=object)
        sub = df.loc[df["epic_id_norm"].astype(str).eq(str(epic_id_norm))]
        if len(sub) == 0:
            return pd.Series(dtype=object)
        return sub.iloc[0]

    def _classify_outcome(
        self,
        *,
        in_best: bool,
        in_quarantine: bool,
        batch_present: bool,
        selected_for_period_stage: bool,
        failure_category: str,
        source_reason: str,
        terminal_reason: str,
        n_events_after_filters: float,
    ) -> str:
        if in_best:
            return self.OUTCOME_RECOVERED_IN_BEST
        failure = failure_category.strip().lower()
        source = source_reason.strip().lower()
        terminal = terminal_reason.strip().lower()
        if (
            failure == "events_filtered_to_zero"
            or source == "events_filtered_to_zero"
            or (pd.notna(n_events_after_filters) and n_events_after_filters <= 0.0)
        ):
            return self.OUTCOME_NO_EVENTS_AFTER_FILTERS
        if in_quarantine:
            return self.OUTCOME_RECOVERED_IN_QUARANTINE
        if batch_present or selected_for_period_stage or terminal != "":
            return self.OUTCOME_DETECTED_BUT_FAILED_DOWNSTREAM
        return self.OUTCOME_NOT_SEEN

    def _top_failure_reason(self, row: pd.Series) -> str:
        return self._first_nonempty_text(
            row.get("failure_category", ""),
            row.get("shortlist_rejection_reason", ""),
            row.get("terminal_reason", ""),
            row.get("source_reason", ""),
            row.get("outcome_label", ""),
        )

    @staticmethod
    def _sample_examples(df: pd.DataFrame, limit: int = 5) -> str:
        if len(df) == 0:
            return ""
        return "|".join(df["epic_id"].astype(str).head(max(1, int(limit))).tolist())

    def run(
        self,
        *,
        batch_results_csv: Path,
        best_csv: Path,
        quarantine_csv: Path,
        funnel_csv: Path,
        diagnostics_csv: Path,
        reference_csv: Path,
        refresh_reference: bool,
        audit_csv: Path,
        rollup_csv: Path,
        false_negatives_csv: Path,
    ) -> Dict[str, Any]:
        reference_raw = self._load_reference_df(reference_csv=reference_csv, refresh_reference=bool(refresh_reference))
        reference_df, unmatched_reference_rows = self._prepare_reference(reference_raw)

        batch = self._prepare_batch_results(Path(batch_results_csv))
        best = self._prepare_best(Path(best_csv))
        quarantine = self._prepare_quarantine(Path(quarantine_csv))
        funnel = self._prepare_funnel(Path(funnel_csv))
        diagnostics = self._read_required_csv(Path(diagnostics_csv))

        rows: List[Dict[str, Any]] = []
        for _, ref in reference_df.iterrows():
            epic_id_norm = str(ref["epic_id_norm"])
            best_row = self._lookup_row(best, epic_id_norm)
            quarantine_row = self._lookup_row(quarantine, epic_id_norm)
            funnel_row = self._lookup_row(funnel, epic_id_norm)
            batch_row = self._lookup_row(batch, epic_id_norm)

            in_best = len(best_row) > 0
            in_quarantine = len(quarantine_row) > 0
            batch_present = len(batch_row) > 0
            selected_for_period_stage = self._to_bool(funnel_row.get("selected_for_period_stage", False))

            failure_category = self._first_nonempty_text(
                quarantine_row.get("failure_category", ""),
                funnel_row.get("period_failure_category", ""),
            )
            shortlist_rejection_reason = self._first_nonempty_text(
                quarantine_row.get("shortlist_rejection_reason", ""),
                funnel_row.get("shortlist_rejection_reason", ""),
            )
            terminal_reason = self._first_nonempty_text(funnel_row.get("terminal_reason", ""))
            source_reason = self._first_nonempty_text(
                quarantine_row.get("source_reason", ""),
                funnel_row.get("source_reason", ""),
            )
            n_events_after_filters = self._first_numeric(
                best_row.get("n_events_after_filters", pd.NA),
                quarantine_row.get("n_events_after_filters", pd.NA),
                funnel_row.get("period_n_events_after_filters", pd.NA),
                funnel_row.get("n_events", pd.NA),
            )

            outcome_label = self._classify_outcome(
                in_best=in_best,
                in_quarantine=in_quarantine,
                batch_present=batch_present,
                selected_for_period_stage=selected_for_period_stage,
                failure_category=failure_category,
                source_reason=source_reason,
                terminal_reason=terminal_reason,
                n_events_after_filters=n_events_after_filters,
            )
            details_json = self._first_nonempty_text(funnel_row.get("details_json", ""))

            rows.append(
                {
                    "epic_id": ref["epic_id"],
                    "epic_id_norm": epic_id_norm,
                    "nasa_confirmed_planet_count": int(ref["nasa_confirmed_planet_count"]),
                    "nasa_confirmed_planets": ref["nasa_confirmed_planets"],
                    "nasa_hostnames": ref["nasa_hostnames"],
                    "nasa_k2_names": ref["nasa_k2_names"],
                    "nasa_disc_facility": ref["nasa_disc_facility"],
                    "batch_present": bool(batch_present),
                    "selected_for_period_stage": bool(selected_for_period_stage),
                    "in_best": bool(in_best),
                    "in_quarantine": bool(in_quarantine),
                    "outcome_label": outcome_label,
                    "detector_triage_status": self._first_nonempty_text(batch_row.get("triage_status", "")),
                    "detector_n_events": self._first_numeric(batch_row.get("n_events", pd.NA)),
                    "detector_best_shape_score": self._first_numeric(batch_row.get("best_shape_score", pd.NA)),
                    "detector_best_depth_snr": self._first_numeric(batch_row.get("best_depth_snr", pd.NA)),
                    "best_reason": self._first_nonempty_text(best_row.get("reason", "")),
                    "best_P": self._first_numeric(best_row.get("P", pd.NA)),
                    "failure_category": failure_category,
                    "shortlist_rejection_reason": shortlist_rejection_reason,
                    "terminal_reason": terminal_reason,
                    "source_reason": source_reason,
                    "stage_reached": self._first_nonempty_text(funnel_row.get("stage_reached", "")),
                    "n_events_after_filters": n_events_after_filters,
                    "details_json": details_json,
                }
            )

        audit_df = pd.DataFrame(rows).sort_values(by=["epic_id_norm"]).reset_index(drop=True)
        false_negatives_df = audit_df.loc[~audit_df["outcome_label"].astype(str).eq(self.OUTCOME_RECOVERED_IN_BEST)].copy()
        false_negatives_df["top_failure_reason"] = false_negatives_df.apply(self._top_failure_reason, axis=1)

        confirmed_total = int(len(audit_df))
        confirmed_in_best = int(audit_df["outcome_label"].astype(str).eq(self.OUTCOME_RECOVERED_IN_BEST).sum())
        confirmed_in_quarantine = int(audit_df["outcome_label"].astype(str).eq(self.OUTCOME_RECOVERED_IN_QUARANTINE).sum())
        confirmed_detected_but_failed_downstream = int(
            audit_df["outcome_label"].astype(str).eq(self.OUTCOME_DETECTED_BUT_FAILED_DOWNSTREAM).sum()
        )
        confirmed_no_events_after_filters = int(
            audit_df["outcome_label"].astype(str).eq(self.OUTCOME_NO_EVENTS_AFTER_FILTERS).sum()
        )
        confirmed_not_seen = int(audit_df["outcome_label"].astype(str).eq(self.OUTCOME_NOT_SEEN).sum())
        confirmed_recall_best_only = float(confirmed_in_best / confirmed_total) if confirmed_total > 0 else 0.0
        confirmed_recall_best_plus_quarantine = (
            float((confirmed_in_best + confirmed_in_quarantine) / confirmed_total) if confirmed_total > 0 else 0.0
        )

        top_failure_rows: List[Dict[str, Any]] = []
        if len(false_negatives_df) > 0:
            counts = false_negatives_df["top_failure_reason"].fillna("").astype(str).replace({"": "unknown"}).value_counts()
            for reason, count in counts.items():
                subset = false_negatives_df.loc[false_negatives_df["top_failure_reason"].fillna("").replace({"": "unknown"}).astype(str).eq(str(reason))]
                top_failure_rows.append(
                    {
                        "section": "top_failure_reasons_not_recovered_in_best",
                        "metric": str(reason),
                        "value": int(count),
                        "notes": self._sample_examples(subset),
                    }
                )

        period_stage_mode = ""
        if len(funnel) > 0 and "period_stage_mode" in funnel.columns:
            mode_counts = funnel["period_stage_mode"].fillna("").astype(str).replace({"nan": ""})
            mode_counts = mode_counts.loc[mode_counts != ""]
            if len(mode_counts) > 0:
                period_stage_mode = str(mode_counts.value_counts().index[0])

        rollup_rows: List[Dict[str, Any]] = [
            {"section": "summary", "metric": "confirmed_total", "value": confirmed_total, "notes": ""},
            {"section": "summary", "metric": "confirmed_in_best", "value": confirmed_in_best, "notes": ""},
            {"section": "summary", "metric": "confirmed_in_quarantine", "value": confirmed_in_quarantine, "notes": ""},
            {
                "section": "summary",
                "metric": "confirmed_detected_but_failed_downstream",
                "value": confirmed_detected_but_failed_downstream,
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "confirmed_no_events_after_filters",
                "value": confirmed_no_events_after_filters,
                "notes": "",
            },
            {"section": "summary", "metric": "confirmed_not_seen", "value": confirmed_not_seen, "notes": ""},
            {
                "section": "summary",
                "metric": "confirmed_recall_best_only",
                "value": confirmed_recall_best_only,
                "notes": f"{confirmed_in_best}/{confirmed_total}",
            },
            {
                "section": "summary",
                "metric": "confirmed_recall_best_plus_quarantine",
                "value": confirmed_recall_best_plus_quarantine,
                "notes": f"{confirmed_in_best + confirmed_in_quarantine}/{confirmed_total}",
            },
            {
                "section": "reference",
                "metric": "reference_rows_without_epic_mapping",
                "value": unmatched_reference_rows,
                "notes": "Excluded from the EPIC-based recall denominator.",
            },
            {
                "section": "pipeline_context",
                "metric": "period_stage_mode_detected_from_funnel",
                "value": period_stage_mode,
                "notes": "",
            },
            {
                "section": "pipeline_context",
                "metric": "current_operating_mode_requested",
                "value": self._first_nonempty_text(diagnostics.iloc[0].get("operating_mode_requested", "")) if len(diagnostics) > 0 else "",
                "notes": "",
            },
        ]
        if period_stage_mode == "randomN":
            rollup_rows.append(
                {
                    "section": "pipeline_context",
                    "metric": "sampling_note",
                    "value": "period-stage outputs appear sample-limited",
                    "notes": "Confirmed-planet recall is being measured against a sampled period-stage run, not a full-population downstream pass.",
                }
            )
        rollup_rows.extend(top_failure_rows)

        audit_csv = Path(audit_csv).resolve()
        rollup_csv = Path(rollup_csv).resolve()
        false_negatives_csv = Path(false_negatives_csv).resolve()
        for path in [audit_csv, rollup_csv, false_negatives_csv]:
            path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(audit_csv, index=False)
        pd.DataFrame(rollup_rows).to_csv(rollup_csv, index=False)
        false_negatives_df.to_csv(false_negatives_csv, index=False)

        top_failure_reason_map = {
            str(row["metric"]): int(row["value"])
            for row in top_failure_rows[:5]
        }
        representative_examples = self._sample_examples(false_negatives_df, limit=5)
        return {
            "reference_csv": Path(reference_csv).resolve(),
            "audit_csv": audit_csv,
            "rollup_csv": rollup_csv,
            "false_negatives_csv": false_negatives_csv,
            "confirmed_total": confirmed_total,
            "confirmed_in_best": confirmed_in_best,
            "confirmed_in_quarantine": confirmed_in_quarantine,
            "confirmed_detected_but_failed_downstream": confirmed_detected_but_failed_downstream,
            "confirmed_no_events_after_filters": confirmed_no_events_after_filters,
            "confirmed_not_seen": confirmed_not_seen,
            "confirmed_recall_best_only": confirmed_recall_best_only,
            "confirmed_recall_best_plus_quarantine": confirmed_recall_best_plus_quarantine,
            "top_failure_reasons": top_failure_reason_map,
            "representative_examples": representative_examples,
        }
