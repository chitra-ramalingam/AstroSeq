from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis


class K2DetectorQualityGatedBroaderCachedFailedDownstreamReport:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\detector_cached_failed_broader_quality_gated_downstream")
    DEFAULT_WINNERS_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_winners.csv")
    DEFAULT_BEST_CSV = DEFAULT_OUT_DIR / "Apr1_period_shortlist_best.csv"
    DEFAULT_QUARANTINE_CSV = DEFAULT_OUT_DIR / "Apr1_period_shortlist_quarantine.csv"
    DEFAULT_FUNNEL_CSV = DEFAULT_OUT_DIR / "Apr1_epic_funnel_reasons.csv"
    DEFAULT_SUMMARY_CSV_NAME = "detector_quality_gated_broader_downstream_summary.csv"
    DEFAULT_QUARANTINED_WINNERS_CSV_NAME = "detector_quality_gated_broader_quarantined_winners.csv"
    DEFAULT_BEST_WINNERS_CSV_NAME = "detector_quality_gated_broader_best_winners.csv"

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Join broader detector-quality-gated winners to the cached-failed broader downstream "
                "April 1 outputs and summarize downstream conversion."
            )
        )
        p.add_argument("--winners-csv", type=Path, default=cls.DEFAULT_WINNERS_CSV)
        p.add_argument("--best-csv", type=Path, default=cls.DEFAULT_BEST_CSV)
        p.add_argument("--quarantine-csv", type=Path, default=cls.DEFAULT_QUARANTINE_CSV)
        p.add_argument("--funnel-csv", type=Path, default=cls.DEFAULT_FUNNEL_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--summary-csv", type=Path, default=None)
        p.add_argument("--quarantined-winners-csv", type=Path, default=None)
        p.add_argument("--best-winners-csv", type=Path, default=None)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir)
        return cls().run(
            winners_csv=Path(args.winners_csv),
            best_csv=Path(args.best_csv),
            quarantine_csv=Path(args.quarantine_csv),
            funnel_csv=Path(args.funnel_csv),
            summary_csv=Path(args.summary_csv) if args.summary_csv is not None else out_dir / cls.DEFAULT_SUMMARY_CSV_NAME,
            quarantined_winners_csv=(
                Path(args.quarantined_winners_csv)
                if args.quarantined_winners_csv is not None
                else out_dir / cls.DEFAULT_QUARANTINED_WINNERS_CSV_NAME
            ),
            best_winners_csv=(
                Path(args.best_winners_csv)
                if args.best_winners_csv is not None
                else out_dir / cls.DEFAULT_BEST_WINNERS_CSV_NAME
            ),
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

    def _normalize_epic(self, value: Any) -> str:
        return self.helper._canonical_epic(value)

    @staticmethod
    def _require_columns(df: pd.DataFrame, *, label: str, required_columns: Sequence[str]) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"{label} CSV missing required columns: {', '.join(missing)}")

    def _prepare_table(
        self,
        df: pd.DataFrame,
        *,
        epic_col: str,
        label: str,
        required_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        needed = [epic_col]
        if required_columns is not None:
            needed.extend(list(required_columns))
        self._require_columns(df, label=label, required_columns=needed)
        out = df.copy()
        out["epic_id_norm"] = out[epic_col].map(self._normalize_epic)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out

    @staticmethod
    def _count_reasons(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        if column not in df.columns or len(df) == 0:
            return []
        values = df[column].fillna("").astype(str).str.strip()
        values = values.loc[(values != "") & (values.str.lower() != "nan")]
        if len(values) == 0:
            return []
        counts = values.value_counts()
        return [
            {"reason_value": str(reason_value), "count": int(count)}
            for reason_value, count in counts.items()
        ]

    @staticmethod
    def _top_reason_map(reason_rows: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, int]:
        return {
            str(row["reason_value"]): int(row["count"])
            for row in reason_rows[: max(0, int(top_n))]
        }

    def run(
        self,
        *,
        winners_csv: Path,
        best_csv: Path,
        quarantine_csv: Path,
        funnel_csv: Path,
        summary_csv: Path,
        quarantined_winners_csv: Path,
        best_winners_csv: Path,
    ) -> Dict[str, Any]:
        winners_csv = Path(winners_csv).resolve()
        best_csv = Path(best_csv).resolve()
        quarantine_csv = Path(quarantine_csv).resolve()
        funnel_csv = Path(funnel_csv).resolve()
        summary_csv = Path(summary_csv).resolve()
        quarantined_winners_csv = Path(quarantined_winners_csv).resolve()
        best_winners_csv = Path(best_winners_csv).resolve()

        winners = self._prepare_table(
            self._read_required_csv(winners_csv),
            epic_col="epic_id",
            label="winners",
            required_columns=["epic_id"],
        )
        best = self._prepare_table(
            self._read_required_csv(best_csv),
            epic_col="epic",
            label="best",
            required_columns=["epic"],
        )
        quarantine = self._prepare_table(
            self._read_required_csv(quarantine_csv),
            epic_col="epic_id",
            label="quarantine",
            required_columns=["epic_id", "failure_category", "shortlist_rejection_reason"],
        )
        funnel = self.helper._expand_funnel_details(self._read_required_csv(funnel_csv))
        funnel = self._prepare_table(
            funnel,
            epic_col="epic_id",
            label="funnel",
            required_columns=["epic_id", "terminal_reason"],
        )

        best_prefixed = best.rename(columns={c: f"best_{c}" for c in best.columns if c != "epic_id_norm"})
        quarantine_prefixed = quarantine.rename(
            columns={c: f"quarantine_{c}" for c in quarantine.columns if c != "epic_id_norm"}
        )
        funnel_prefixed = funnel.rename(columns={c: f"funnel_{c}" for c in funnel.columns if c != "epic_id_norm"})

        best_winners = winners.merge(best_prefixed, on="epic_id_norm", how="inner").copy()
        quarantined_winners = (
            winners.merge(quarantine_prefixed, on="epic_id_norm", how="inner")
            .merge(funnel_prefixed, on="epic_id_norm", how="left")
            .copy()
        )

        if len(quarantined_winners) > 0:
            quarantined_winners["failure_category"] = quarantined_winners.apply(
                lambda row: self._first_nonempty_text(
                    row.get("quarantine_failure_category", ""),
                    row.get("funnel_period_failure_category", ""),
                ),
                axis=1,
            )
            quarantined_winners["shortlist_rejection_reason"] = quarantined_winners.apply(
                lambda row: self._first_nonempty_text(
                    row.get("quarantine_shortlist_rejection_reason", ""),
                    row.get("funnel_shortlist_rejection_reason", ""),
                ),
                axis=1,
            )
            quarantined_winners["terminal_reason"] = quarantined_winners.apply(
                lambda row: self._first_nonempty_text(
                    row.get("funnel_terminal_reason", ""),
                    row.get("quarantine_terminal_reason", ""),
                    row.get("quarantine_reason", ""),
                ),
                axis=1,
            )
        else:
            quarantined_winners["failure_category"] = pd.Series(dtype=str)
            quarantined_winners["shortlist_rejection_reason"] = pd.Series(dtype=str)
            quarantined_winners["terminal_reason"] = pd.Series(dtype=str)

        winners_total = int(len(winners))
        winners_in_best = int(len(best_winners))
        winners_in_quarantine = int(len(quarantined_winners))
        downstream_conversion_rate = float(winners_in_best / winners_total) if winners_total > 0 else 0.0

        reason_columns = [
            "failure_category",
            "shortlist_rejection_reason",
            "terminal_reason",
        ]
        summary_rows: List[Dict[str, Any]] = [
            {"section": "summary", "reason_column": "", "metric": "winners_total", "value": winners_total},
            {"section": "summary", "reason_column": "", "metric": "winners_in_best", "value": winners_in_best},
            {"section": "summary", "reason_column": "", "metric": "winners_in_quarantine", "value": winners_in_quarantine},
            {
                "section": "summary",
                "reason_column": "",
                "metric": "downstream_conversion_rate",
                "value": downstream_conversion_rate,
            },
        ]
        top_reasons: Dict[str, Dict[str, int]] = {}
        for column in reason_columns:
            reason_rows = self._count_reasons(quarantined_winners, column=column)
            top_reasons[column] = self._top_reason_map(reason_rows)
            for row in reason_rows:
                summary_rows.append(
                    {
                        "section": "top_failure_reasons",
                        "reason_column": column,
                        "metric": row["reason_value"],
                        "value": row["count"],
                    }
                )

        summary_df = pd.DataFrame(summary_rows)

        for out_path in [summary_csv, quarantined_winners_csv, best_winners_csv]:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        quarantined_winners.to_csv(quarantined_winners_csv, index=False)
        best_winners.to_csv(best_winners_csv, index=False)

        return {
            "winners_csv": winners_csv,
            "best_csv": best_csv,
            "quarantine_csv": quarantine_csv,
            "funnel_csv": funnel_csv,
            "summary_csv": summary_csv,
            "quarantined_winners_csv": quarantined_winners_csv,
            "best_winners_csv": best_winners_csv,
            "winners_total": winners_total,
            "winners_in_best": winners_in_best,
            "winners_in_quarantine": winners_in_quarantine,
            "downstream_conversion_rate": downstream_conversion_rate,
            "top_failure_reasons": top_reasons,
        }
