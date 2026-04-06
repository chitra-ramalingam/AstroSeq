from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2ConfirmedPlanetCoverageAudit:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_REFERENCE_CSV = DEFAULT_OUT_DIR / "nasa_confirmed_k2_planets_reference.csv"
    DEFAULT_RECALL_AUDIT_CSV = DEFAULT_OUT_DIR / "k2_confirmed_planet_recall_audit.csv"
    DEFAULT_BATCH_RESULTS_CSV = DEFAULT_OUT_DIR / "batch_results.csv"
    DEFAULT_BATCH_RESULTS_RETRIAGED_CSV = DEFAULT_OUT_DIR / "batch_results_retriaged.csv"
    DEFAULT_BATCH_RESULTS_WHITENESS_CSV = DEFAULT_OUT_DIR / "batch_results_whiteness.csv"
    DEFAULT_AUDIT_CSV_NAME = "k2_confirmed_planet_coverage_audit.csv"
    DEFAULT_ROLLUP_CSV_NAME = "k2_confirmed_planet_coverage_rollup.csv"

    BUCKET_MATCHED = "present in AstroSeq processed universe and matched"
    BUCKET_RAW_NEVER_PROCESSED = "present in raw K2 manifest but never processed"
    BUCKET_LOAD_FAILED = "processed but no light curve / load failed"
    BUCKET_ID_MISMATCH = "identifier mismatch / normalization issue"
    BUCKET_OUTSIDE_SCOPE = "outside current pipeline universe / campaign scope"
    BUCKET_OTHER = "other clearly supported reason"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Audit coverage of NASA-confirmed K2 planets against the local AstroSeq K2 processed universe "
                "and classify whether poor confirmed recall is driven by coverage or by scientific recall."
            )
        )
        p.add_argument("--reference-csv", type=Path, default=cls.DEFAULT_REFERENCE_CSV)
        p.add_argument("--recall-audit-csv", type=Path, default=cls.DEFAULT_RECALL_AUDIT_CSV)
        p.add_argument("--batch-results-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_CSV)
        p.add_argument("--batch-results-retriaged-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_RETRIAGED_CSV)
        p.add_argument("--batch-results-whiteness-csv", type=Path, default=cls.DEFAULT_BATCH_RESULTS_WHITENESS_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--audit-csv", type=Path, default=None)
        p.add_argument("--rollup-csv", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            reference_csv=Path(args.reference_csv),
            recall_audit_csv=Path(args.recall_audit_csv),
            batch_results_csv=Path(args.batch_results_csv),
            batch_results_retriaged_csv=Path(args.batch_results_retriaged_csv),
            batch_results_whiteness_csv=Path(args.batch_results_whiteness_csv),
            audit_csv=Path(args.audit_csv) if args.audit_csv is not None else out_dir / cls.DEFAULT_AUDIT_CSV_NAME,
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

    def _canonical_epic(self, value: Any) -> str:
        return self.helper._canonical_epic(value)

    def _prepare_reference(self, path: Path) -> pd.DataFrame:
        ref = self._read_required_csv(path)
        if "epic_id" not in ref.columns:
            raise ValueError(f"reference CSV missing required column: epic_id ({path})")
        ref["epic_id_norm"] = ref["epic_id"].map(self._canonical_epic)
        ref = ref.loc[ref["epic_id_norm"] != ""].copy()
        grouped = (
            ref.groupby("epic_id_norm", dropna=False)
            .agg(
                epic_id=("epic_id_norm", lambda s: f"EPIC_{list(s)[0]}"),
                nasa_confirmed_planets=("pl_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                nasa_hostnames=("hostname", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
                nasa_k2_names=("k2_name", lambda s: "|".join(sorted({str(x) for x in s if str(x).strip() != ""}))),
            )
            .reset_index()
        )
        return grouped

    def _prepare_batch(self, path: Path, *, epic_col_preference: str) -> pd.DataFrame:
        df = self._read_required_csv(path)
        if epic_col_preference in df.columns:
            df["epic_id_norm"] = df[epic_col_preference].map(self._canonical_epic)
        elif "query" in df.columns:
            df["epic_id_norm"] = df["query"].map(self._canonical_epic)
        else:
            raise ValueError(f"CSV missing expected EPIC columns: {path}")
        return df.loc[df["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

    @staticmethod
    def _lookup_row(df: pd.DataFrame, epic_id_norm: str) -> pd.Series:
        if len(df) == 0:
            return pd.Series(dtype=object)
        sub = df.loc[df["epic_id_norm"].astype(str).eq(str(epic_id_norm))]
        if len(sub) == 0:
            return pd.Series(dtype=object)
        return sub.iloc[0]

    @staticmethod
    def _sample_examples(df: pd.DataFrame, limit: int = 5) -> str:
        if len(df) == 0:
            return ""
        return "|".join(df["epic_id"].astype(str).head(max(1, int(limit))).tolist())

    def _classify_bucket(
        self,
        *,
        batch_row: pd.Series,
        retriaged_row: pd.Series,
        whiteness_row: pd.Series,
        recall_row: pd.Series,
    ) -> tuple[str, str]:
        in_batch = len(batch_row) > 0
        in_retriaged = len(retriaged_row) > 0
        in_whiteness = len(whiteness_row) > 0
        if in_batch:
            triage_status = self._first_nonempty_text(batch_row.get("triage_status", ""))
            error_stage = self._first_nonempty_text(batch_row.get("error_stage", ""))
            error_type = self._first_nonempty_text(batch_row.get("error_type", ""))
            error_msg = self._first_nonempty_text(batch_row.get("error_msg", ""))
            recall_outcome = self._first_nonempty_text(recall_row.get("outcome_label", ""))
            if triage_status.lower() == "error" or error_stage != "" or error_type != "" or error_msg != "":
                return (
                    self.BUCKET_LOAD_FAILED,
                    self._first_nonempty_text(error_stage, error_type, error_msg, "triage_status=error"),
                )
            return (
                self.BUCKET_MATCHED,
                self._first_nonempty_text(recall_outcome, "present in batch_results.csv"),
            )
        if in_retriaged or in_whiteness:
            return (
                self.BUCKET_RAW_NEVER_PROCESSED,
                "Present in a local manifest-like stage file but absent from batch_results.csv.",
            )
        query_text = self._first_nonempty_text(recall_row.get("epic_id", ""))
        if query_text != "" and self._canonical_epic(query_text) != "":
            return (
                self.BUCKET_OUTSIDE_SCOPE,
                "Absent from batch_results.csv, batch_results_retriaged.csv, and batch_results_whiteness.csv.",
            )
        return (self.BUCKET_ID_MISMATCH, "Reference EPIC id could not be normalized against local pipeline ids.")

    def run(
        self,
        *,
        reference_csv: Path,
        recall_audit_csv: Path,
        batch_results_csv: Path,
        batch_results_retriaged_csv: Path,
        batch_results_whiteness_csv: Path,
        audit_csv: Path,
        rollup_csv: Path,
    ) -> Dict[str, Any]:
        reference = self._prepare_reference(Path(reference_csv))
        recall = self._read_required_csv(Path(recall_audit_csv))
        if "epic_id_norm" not in recall.columns:
            raise ValueError("recall audit CSV missing required column: epic_id_norm")
        recall["epic_id_norm"] = recall["epic_id_norm"].astype(str)
        recall = recall.drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)

        batch = self._prepare_batch(Path(batch_results_csv), epic_col_preference="epic_id")
        retriaged = self._prepare_batch(Path(batch_results_retriaged_csv), epic_col_preference="epic_id")
        whiteness = self._prepare_batch(Path(batch_results_whiteness_csv), epic_col_preference="epic_id")

        rows: List[Dict[str, Any]] = []
        for _, ref in reference.iterrows():
            epic_id_norm = str(ref["epic_id_norm"])
            batch_row = self._lookup_row(batch, epic_id_norm)
            retriaged_row = self._lookup_row(retriaged, epic_id_norm)
            whiteness_row = self._lookup_row(whiteness, epic_id_norm)
            recall_row = self._lookup_row(recall, epic_id_norm)
            coverage_bucket, coverage_reason = self._classify_bucket(
                batch_row=batch_row,
                retriaged_row=retriaged_row,
                whiteness_row=whiteness_row,
                recall_row=recall_row,
            )
            rows.append(
                {
                    "epic_id": ref["epic_id"],
                    "epic_id_norm": epic_id_norm,
                    "nasa_confirmed_planets": ref["nasa_confirmed_planets"],
                    "nasa_hostnames": ref["nasa_hostnames"],
                    "nasa_k2_names": ref["nasa_k2_names"],
                    "in_batch_results": bool(len(batch_row) > 0),
                    "in_batch_results_retriaged": bool(len(retriaged_row) > 0),
                    "in_batch_results_whiteness": bool(len(whiteness_row) > 0),
                    "coverage_bucket": coverage_bucket,
                    "coverage_reason": coverage_reason,
                    "recall_outcome_label": self._first_nonempty_text(recall_row.get("outcome_label", "")),
                    "detector_triage_status": self._first_nonempty_text(batch_row.get("triage_status", "")),
                    "triage_why_not_usable": self._first_nonempty_text(
                        batch_row.get("triage_why_not_usable", ""),
                        retriaged_row.get("triage_why_not_usable", ""),
                        whiteness_row.get("triage_why_not_usable", ""),
                    ),
                    "error_stage": self._first_nonempty_text(batch_row.get("error_stage", "")),
                    "error_type": self._first_nonempty_text(batch_row.get("error_type", "")),
                    "error_msg": self._first_nonempty_text(batch_row.get("error_msg", "")),
                }
            )

        audit_df = pd.DataFrame(rows).sort_values(by=["epic_id_norm"]).reset_index(drop=True)
        confirmed_total = int(len(audit_df))

        bucket_counts = audit_df["coverage_bucket"].value_counts()
        matched_to_processed_universe = int(bucket_counts.get(self.BUCKET_MATCHED, 0))
        raw_never_processed = int(bucket_counts.get(self.BUCKET_RAW_NEVER_PROCESSED, 0))
        load_failed = int(bucket_counts.get(self.BUCKET_LOAD_FAILED, 0))
        id_mismatch = int(bucket_counts.get(self.BUCKET_ID_MISMATCH, 0))
        outside_scope = int(bucket_counts.get(self.BUCKET_OUTSIDE_SCOPE, 0))
        other = int(bucket_counts.get(self.BUCKET_OTHER, 0))

        not_seen = audit_df.loc[audit_df["recall_outcome_label"].astype(str).eq("not_seen / not_matched")].copy()
        not_seen_bucket_counts = not_seen["coverage_bucket"].value_counts()
        dominant_blocker = (
            str(not_seen_bucket_counts.index[0]) if len(not_seen_bucket_counts) > 0 else "none"
        )
        incomplete_population_coverage = outside_scope + raw_never_processed + id_mismatch + load_failed
        scientific_recall_limited_subset = matched_to_processed_universe
        limiting_conclusion = (
            "incomplete population coverage"
            if incomplete_population_coverage > max(0, scientific_recall_limited_subset - load_failed)
            else "poor scientific recall"
        )

        rollup_rows: List[Dict[str, Any]] = [
            {"section": "summary", "metric": "confirmed_total", "value": confirmed_total, "notes": ""},
            {
                "section": "summary",
                "metric": "matched_to_processed_universe",
                "value": matched_to_processed_universe,
                "notes": "Confirmed EPICs present in batch_results.csv and not triage-error rows.",
            },
            {"section": "summary", "metric": "not_processed", "value": raw_never_processed, "notes": ""},
            {"section": "summary", "metric": "id_mismatch", "value": id_mismatch, "notes": ""},
            {"section": "summary", "metric": "outside_scope", "value": outside_scope, "notes": ""},
            {"section": "summary", "metric": "load_failed", "value": load_failed, "notes": ""},
            {"section": "summary", "metric": "other", "value": other, "notes": ""},
            {
                "section": "summary",
                "metric": "final_dominant_coverage_blocker",
                "value": dominant_blocker,
                "notes": "",
            },
            {
                "section": "summary",
                "metric": "coverage_vs_science_conclusion",
                "value": limiting_conclusion,
                "notes": "Compares unmatched/out-of-scope/load-failed coverage losses against matched-universe scientific failures.",
            },
        ]
        for bucket in [
            self.BUCKET_MATCHED,
            self.BUCKET_RAW_NEVER_PROCESSED,
            self.BUCKET_LOAD_FAILED,
            self.BUCKET_ID_MISMATCH,
            self.BUCKET_OUTSIDE_SCOPE,
            self.BUCKET_OTHER,
        ]:
            subset = audit_df.loc[audit_df["coverage_bucket"].astype(str).eq(bucket)].copy()
            rollup_rows.append(
                {
                    "section": "bucket_counts",
                    "metric": bucket,
                    "value": int(len(subset)),
                    "notes": self._sample_examples(subset),
                }
            )
        for reason, count in not_seen["coverage_bucket"].value_counts().items():
            subset = not_seen.loc[not_seen["coverage_bucket"].astype(str).eq(str(reason))].copy()
            rollup_rows.append(
                {
                    "section": "not_seen_breakdown",
                    "metric": str(reason),
                    "value": int(count),
                    "notes": self._sample_examples(subset),
                }
            )

        audit_csv = Path(audit_csv).resolve()
        rollup_csv = Path(rollup_csv).resolve()
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(audit_csv, index=False)
        pd.DataFrame(rollup_rows).to_csv(rollup_csv, index=False)

        return {
            "audit_csv": audit_csv,
            "rollup_csv": rollup_csv,
            "confirmed_total": confirmed_total,
            "matched_to_processed_universe": matched_to_processed_universe,
            "not_processed": raw_never_processed,
            "id_mismatch": id_mismatch,
            "outside_scope": outside_scope,
            "load_failed": load_failed,
            "final_dominant_coverage_blocker": dominant_blocker,
            "coverage_vs_science_conclusion": limiting_conclusion,
        }
