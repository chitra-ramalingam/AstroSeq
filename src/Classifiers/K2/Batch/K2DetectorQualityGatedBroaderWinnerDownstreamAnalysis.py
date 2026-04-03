from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistRecoveryModeAnalysis import K2ShortlistRecoveryModeAnalysis
from src.Classifiers.K2.Pipeline.K2_BatchRunner import K2BatchRunner


class K2DetectorQualityGatedBroaderWinnerDownstreamAnalysis:
    DEFAULT_WINNERS_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_winners.csv")
    DEFAULT_COMPARISON_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_comparison.csv")
    DEFAULT_ROLLUP_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_rollup.csv")
    DEFAULT_ANALYSIS_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_winner_downstream_analysis.csv")
    DEFAULT_ANALYSIS_ROLLUP_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_winner_downstream_rollup.csv")
    DEFAULT_REAL_RESCUES_CSV = Path(r"plots\k2_batch\detector_quality_gated_broader_real_rescues.csv")
    DEFAULT_FAILURE_REASON_TOP_N = 10
    PERIOD_BIN_ORDER = ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]", "no_P_available"]

    def __init__(self) -> None:
        self.helper = K2ShortlistRecoveryModeAnalysis()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Join broader detector-quality-gated winners to downstream default and quality-gated "
                "shortlist outputs and classify real rescues vs detector-only gains."
            )
        )
        p.add_argument("--winners-csv", type=Path, default=cls.DEFAULT_WINNERS_CSV)
        p.add_argument("--comparison-csv", type=Path, default=cls.DEFAULT_COMPARISON_CSV)
        p.add_argument("--rollup-csv", type=Path, default=cls.DEFAULT_ROLLUP_CSV)
        p.add_argument("--default-run-dir", type=Path, default=None)
        p.add_argument("--default-best-csv", type=Path, default=None)
        p.add_argument("--default-quarantine-csv", type=Path, default=None)
        p.add_argument("--default-diagnostics-csv", type=Path, default=None)
        p.add_argument("--default-funnel-csv", type=Path, default=None)
        p.add_argument("--quality-gated-run-dir", type=Path, default=None)
        p.add_argument("--quality-gated-best-csv", type=Path, default=None)
        p.add_argument("--quality-gated-quarantine-csv", type=Path, default=None)
        p.add_argument("--quality-gated-diagnostics-csv", type=Path, default=None)
        p.add_argument("--quality-gated-funnel-csv", type=Path, default=None)
        p.add_argument("--analysis-csv", type=Path, default=cls.DEFAULT_ANALYSIS_CSV)
        p.add_argument("--analysis-rollup-csv", type=Path, default=cls.DEFAULT_ANALYSIS_ROLLUP_CSV)
        p.add_argument("--real-rescues-csv", type=Path, default=cls.DEFAULT_REAL_RESCUES_CSV)
        p.add_argument("--failure-reason-top-n", type=int, default=cls.DEFAULT_FAILURE_REASON_TOP_N)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            winners_csv=Path(args.winners_csv),
            comparison_csv=Path(args.comparison_csv),
            rollup_csv=Path(args.rollup_csv),
            default_run_dir=Path(args.default_run_dir) if args.default_run_dir is not None else None,
            default_best_csv=Path(args.default_best_csv) if args.default_best_csv is not None else None,
            default_quarantine_csv=Path(args.default_quarantine_csv) if args.default_quarantine_csv is not None else None,
            default_diagnostics_csv=Path(args.default_diagnostics_csv) if args.default_diagnostics_csv is not None else None,
            default_funnel_csv=Path(args.default_funnel_csv) if args.default_funnel_csv is not None else None,
            quality_gated_run_dir=Path(args.quality_gated_run_dir) if args.quality_gated_run_dir is not None else None,
            quality_gated_best_csv=Path(args.quality_gated_best_csv) if args.quality_gated_best_csv is not None else None,
            quality_gated_quarantine_csv=Path(args.quality_gated_quarantine_csv)
            if args.quality_gated_quarantine_csv is not None
            else None,
            quality_gated_diagnostics_csv=Path(args.quality_gated_diagnostics_csv)
            if args.quality_gated_diagnostics_csv is not None
            else None,
            quality_gated_funnel_csv=Path(args.quality_gated_funnel_csv) if args.quality_gated_funnel_csv is not None else None,
            analysis_csv=Path(args.analysis_csv),
            analysis_rollup_csv=Path(args.analysis_rollup_csv),
            real_rescues_csv=Path(args.real_rescues_csv),
            failure_reason_top_n=int(args.failure_reason_top_n),
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
    def _require_downstream_source(label: str, run_dir: Optional[Path], best_csv: Optional[Path], quarantine_csv: Optional[Path]) -> None:
        if run_dir is not None:
            return
        if best_csv is not None and quarantine_csv is not None:
            return
        raise ValueError(
            f"{label} downstream inputs require either --{label}-run-dir or both "
            f"--{label}-best-csv and --{label}-quarantine-csv."
        )

    @staticmethod
    def _resolve_downstream_path(run_dir: Optional[Path], explicit_path: Optional[Path], filename: str) -> Optional[Path]:
        if explicit_path is not None:
            return Path(explicit_path)
        if run_dir is None:
            return None
        return Path(run_dir) / filename

    @staticmethod
    def _first_nonempty_text(*values: Any) -> str:
        for value in values:
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text != "" and text.lower() != "nan":
                return text
        return ""

    def _load_downstream_state(
        self,
        *,
        label: str,
        run_dir: Optional[Path],
        best_csv: Optional[Path],
        quarantine_csv: Optional[Path],
        diagnostics_csv: Optional[Path],
        funnel_csv: Optional[Path],
    ) -> Dict[str, Any]:
        self._require_downstream_source(label=label, run_dir=run_dir, best_csv=best_csv, quarantine_csv=quarantine_csv)
        best_path = self._resolve_downstream_path(run_dir=run_dir, explicit_path=best_csv, filename="period_shortlist_best.csv")
        quarantine_path = self._resolve_downstream_path(
            run_dir=run_dir,
            explicit_path=quarantine_csv,
            filename="period_shortlist_quarantine.csv",
        )
        diagnostics_path = self._resolve_downstream_path(
            run_dir=run_dir,
            explicit_path=diagnostics_csv,
            filename="period_shortlist_diagnostics.csv",
        )
        funnel_path = self._resolve_downstream_path(run_dir=run_dir, explicit_path=funnel_csv, filename="epic_funnel_reasons.csv")

        if best_path is None or quarantine_path is None:
            raise ValueError(f"{label} downstream inputs could not resolve best/quarantine CSV paths.")

        best = self.helper._read_csv(best_path).copy()
        quarantine = self.helper._read_csv(quarantine_path).copy()
        diagnostics = self.helper._read_csv(diagnostics_path).copy() if diagnostics_path is not None else pd.DataFrame()
        funnel = (
            self.helper._expand_funnel_details(self.helper._read_csv(funnel_path).copy())
            if funnel_path is not None
            else pd.DataFrame()
        )

        if "epic" in best.columns:
            best["epic"] = best["epic"].map(self.helper._canonical_epic)
            best = best.loc[best["epic"] != ""].drop_duplicates(subset=["epic"], keep="first").reset_index(drop=True)
        else:
            best["epic"] = pd.Series(dtype=str)

        if "epic_id" in quarantine.columns:
            quarantine["epic_id"] = quarantine["epic_id"].map(self.helper._canonical_epic)
            quarantine = quarantine.loc[quarantine["epic_id"] != ""].drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)
        else:
            quarantine["epic_id"] = pd.Series(dtype=str)

        if "epic_id" in funnel.columns:
            funnel["epic_id"] = funnel["epic_id"].map(self.helper._canonical_epic)
        else:
            funnel["epic_id"] = pd.Series(dtype=str)

        selected_epics = set(best["epic"].tolist()).union(set(quarantine["epic_id"].tolist()))
        if "selected_for_period_stage" in funnel.columns:
            selected_mask = funnel["selected_for_period_stage"].fillna(False).astype(bool)
            selected_from_funnel = {str(x) for x in funnel.loc[selected_mask, "epic_id"].tolist() if str(x) != ""}
            if len(selected_from_funnel) > 0:
                selected_epics = selected_from_funnel

        best_epics = set(best["epic"].tolist())
        failed_epics = selected_epics.difference(best_epics)
        failed_rows = quarantine.loc[quarantine["epic_id"].isin(failed_epics)].copy()
        if len(failed_rows) > 0:
            failed_rows["failure_reason_bucket"] = failed_rows.apply(
                lambda row: self.helper._failure_bucket(row.to_dict()),
                axis=1,
            )
        else:
            failed_rows = pd.DataFrame(columns=["epic_id", "failure_reason_bucket", "P"])

        missing_failed_epics = failed_epics.difference(set(failed_rows.get("epic_id", pd.Series(dtype=str)).astype(str).tolist()))
        if len(missing_failed_epics) > 0 and len(funnel) > 0:
            funnel_fail = funnel.loc[funnel["epic_id"].isin(missing_failed_epics)].copy()
            if len(funnel_fail) > 0:
                funnel_fail["failure_reason_bucket"] = funnel_fail.apply(
                    lambda row: self.helper._failure_bucket(row.to_dict()),
                    axis=1,
                )
                failed_rows = pd.concat(
                    [failed_rows, funnel_fail.reindex(columns=list(failed_rows.columns.union(funnel_fail.columns)))],
                    ignore_index=True,
                ).drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)

        if len(failed_rows) == 0:
            failed_rows = pd.DataFrame(columns=["epic_id", "failure_reason_bucket", "P"])

        if "P" not in failed_rows.columns:
            failed_rows["P"] = pd.NA
        failed_rows["period_bin"] = self.helper._period_bin_for_series(failed_rows.get("P", pd.Series(dtype=float)))

        return {
            "label": label,
            "run_dir": run_dir,
            "best_path": best_path,
            "quarantine_path": quarantine_path,
            "diagnostics_path": diagnostics_path,
            "funnel_path": funnel_path,
            "best": best,
            "quarantine": quarantine,
            "diagnostics": diagnostics,
            "funnel": funnel,
            "best_epics": best_epics,
            "failed_rows": failed_rows,
        }

    @staticmethod
    def _best_outcome_label(reason: str, manual_review_required: bool) -> str:
        reason_low = str(reason).strip().lower()
        if reason_low == "validated":
            return "validated"
        if manual_review_required or ("manual" in reason_low):
            return "manual_review"
        if reason_low != "":
            return f"shortlisted:{reason_low}"
        return "shortlisted"

    @staticmethod
    def _best_outcome_score(outcome_label: str) -> int:
        label = str(outcome_label).strip().lower()
        if label == "validated":
            return 5
        if label in {"manual_review", "shortlisted"} or label.startswith("shortlisted:"):
            return 4
        return 4

    @staticmethod
    def _failure_reason_key(row: pd.Series, prefix: str) -> str:
        terminal_group = str(row.get(f"{prefix}_terminal_group", "") or "").strip()
        if terminal_group == "no_downstream_record":
            return "no_downstream_record"
        if terminal_group == "shortlisted":
            return "already_shortlisted_under_default"
        bucket = str(row.get(f"{prefix}_failure_reason_bucket", "") or "").strip()
        if bucket != "":
            return bucket
        for col in [
            f"{prefix}_failure_category",
            f"{prefix}_source_reason",
            f"{prefix}_shortlist_rejection_reason",
            f"{prefix}_failure_detail",
            f"{prefix}_reason",
            f"{prefix}_terminal_reason",
        ]:
            text = str(row.get(col, "") or "").strip()
            if text != "":
                return text
        return "failed_downstream_other"

    @staticmethod
    def _failure_reason_text(reason_key: str) -> str:
        mapping = {
            "events_filtered_to_zero": "all downstream events were filtered out before period validation",
            "insufficient_events": "too few events remained after downstream filtering",
            "cluster_related_failures": "period clustering found no usable period candidate",
            "triage_unusable_or_quality_failures": "triage or quality checks blocked the EPIC before a usable period result",
            "other": "another downstream gate blocked the EPIC",
            "failed_downstream_other": "another downstream gate blocked the EPIC",
            "no_downstream_record": "no downstream shortlist or quarantine record was produced",
            "already_shortlisted_under_default": "the EPIC was already shortlisted under default, so the detector gain changed nothing downstream",
        }
        return mapping.get(reason_key, str(reason_key))

    def _build_best_outcome_frame(self, state: Dict[str, Any], prefix: str) -> pd.DataFrame:
        best = state["best"].copy()
        if len(best) == 0:
            return pd.DataFrame(
                columns=[
                    "epic_id_canonical",
                    f"{prefix}_terminal_group",
                    f"{prefix}_downstream_outcome",
                    f"{prefix}_downstream_outcome_score",
                    f"{prefix}_best_reason",
                    f"{prefix}_best_P",
                    f"{prefix}_best_period_bin",
                    f"{prefix}_manual_review_required",
                    f"{prefix}_best_query",
                ]
            )
        manual_review_series = best.get("manual_review_required", pd.Series([False] * len(best), index=best.index)).fillna(False).astype(bool)
        best_reason = best.get("reason", pd.Series([""] * len(best), index=best.index)).fillna("").astype(str)
        best_outcome = [self._best_outcome_label(reason=r, manual_review_required=bool(m)) for r, m in zip(best_reason, manual_review_series)]
        best_period = pd.to_numeric(best.get("P", pd.Series(dtype=float)), errors="coerce")
        return pd.DataFrame(
            {
                "epic_id_canonical": best["epic"].astype(str),
                f"{prefix}_terminal_group": "shortlisted",
                f"{prefix}_downstream_outcome": best_outcome,
                f"{prefix}_downstream_outcome_score": [self._best_outcome_score(label) for label in best_outcome],
                f"{prefix}_best_reason": best_reason,
                f"{prefix}_best_P": best_period,
                f"{prefix}_best_period_bin": self.helper._period_bin_for_series(best_period),
                f"{prefix}_manual_review_required": manual_review_series.astype(bool),
                f"{prefix}_best_query": best.get("query", pd.Series([""] * len(best), index=best.index)).fillna("").astype(str),
            }
        )

    def _build_failure_outcome_frame(self, state: Dict[str, Any], prefix: str) -> pd.DataFrame:
        failed = state["failed_rows"].copy()
        if len(failed) == 0:
            return pd.DataFrame(
                columns=[
                    "epic_id_canonical",
                    f"{prefix}_terminal_group",
                    f"{prefix}_downstream_outcome",
                    f"{prefix}_downstream_outcome_score",
                    f"{prefix}_failure_reason_bucket",
                    f"{prefix}_failure_category",
                    f"{prefix}_failure_detail",
                    f"{prefix}_reason",
                    f"{prefix}_source_reason",
                    f"{prefix}_shortlist_rejection_reason",
                    f"{prefix}_terminal_reason",
                    f"{prefix}_failure_P",
                    f"{prefix}_failure_period_bin",
                ]
            )
        failure_p = pd.to_numeric(failed.get("P", pd.Series(dtype=float)), errors="coerce")
        has_period = failure_p.notna()
        return pd.DataFrame(
            {
                "epic_id_canonical": failed["epic_id"].astype(str),
                f"{prefix}_terminal_group": "failed_downstream",
                f"{prefix}_downstream_outcome": has_period.map(
                    lambda ok: "failed_with_period" if bool(ok) else "failed_no_period"
                ),
                f"{prefix}_downstream_outcome_score": has_period.map(lambda ok: 2 if bool(ok) else 1).astype(int),
                f"{prefix}_failure_reason_bucket": failed.get(
                    "failure_reason_bucket",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_failure_category": failed.get(
                    "failure_category",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_failure_detail": failed.get(
                    "failure_detail",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_reason": failed.get(
                    "reason",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_source_reason": failed.get(
                    "source_reason",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_shortlist_rejection_reason": failed.get(
                    "shortlist_rejection_reason",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_terminal_reason": failed.get(
                    "terminal_reason",
                    pd.Series([""] * len(failed), index=failed.index),
                ).fillna("").astype(str),
                f"{prefix}_failure_P": failure_p,
                f"{prefix}_failure_period_bin": self.helper._period_bin_for_series(failure_p),
            }
        )

    def _build_state_outcome_frame(self, state: Dict[str, Any], prefix: str) -> pd.DataFrame:
        best_frame = self._build_best_outcome_frame(state, prefix=prefix)
        fail_frame = self._build_failure_outcome_frame(state, prefix=prefix)
        if len(best_frame) == 0:
            return fail_frame.copy()
        if len(fail_frame) == 0:
            return best_frame.copy()
        return (
            best_frame.set_index("epic_id_canonical")
            .combine_first(fail_frame.set_index("epic_id_canonical"))
            .reset_index()
        )

    @staticmethod
    def _append_missing_outcome_defaults(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = df.copy()
        string_defaults = {
            f"{prefix}_terminal_group": "no_downstream_record",
            f"{prefix}_downstream_outcome": "no_downstream_record",
            f"{prefix}_best_reason": "",
            f"{prefix}_best_period_bin": "no_P_available",
            f"{prefix}_best_query": "",
            f"{prefix}_failure_reason_bucket": "",
            f"{prefix}_failure_category": "",
            f"{prefix}_failure_detail": "",
            f"{prefix}_reason": "",
            f"{prefix}_source_reason": "",
            f"{prefix}_shortlist_rejection_reason": "",
            f"{prefix}_terminal_reason": "",
            f"{prefix}_failure_period_bin": "no_P_available",
        }
        for col, value in string_defaults.items():
            if col not in out.columns:
                out[col] = value
            else:
                out[col] = out[col].fillna(value)
        bool_col = f"{prefix}_manual_review_required"
        if bool_col not in out.columns:
            out[bool_col] = False
        else:
            out[bool_col] = out[bool_col].map(
                lambda value: False if pd.isna(value) else str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
            )
        numeric_defaults = {
            f"{prefix}_downstream_outcome_score": 0,
            f"{prefix}_best_P": pd.NA,
            f"{prefix}_failure_P": pd.NA,
        }
        for col, value in numeric_defaults.items():
            if col not in out.columns:
                out[col] = value
            elif col.endswith("_score"):
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value).astype(int)
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _build_comparison_mode_frame(self, comparison: pd.DataFrame, target_mode: str, prefix: str) -> pd.DataFrame:
        sub = comparison.loc[comparison["mode"].astype(str) == str(target_mode)].copy()
        if len(sub) == 0:
            return pd.DataFrame(columns=["epic_id_canonical", f"{prefix}_detector_mode", f"{prefix}_query"])
        sub = sub.sort_values(["epic_id_canonical", "query"]).drop_duplicates(subset=["epic_id_canonical"], keep="first")
        return pd.DataFrame(
            {
                "epic_id_canonical": sub["epic_id_canonical"].astype(str),
                f"{prefix}_detector_mode": sub["mode"].fillna("").astype(str),
                f"{prefix}_query": sub.get("query", pd.Series([""] * len(sub), index=sub.index)).fillna("").astype(str),
            }
        )

    def _non_rescue_explanation(self, row: pd.Series) -> str:
        bucket = str(row.get("winner_bucket", "") or "").strip()
        default_terminal = str(row.get("default_terminal_group", "") or "").strip()
        qg_terminal = str(row.get("quality_gated_terminal_group", "") or "").strip()
        reason_text = str(row.get("non_rescue_failure_reason_text", "") or "").strip()
        qg_detail = self._first_nonempty_text(
            row.get("quality_gated_failure_detail", ""),
            row.get("quality_gated_source_reason", ""),
            row.get("quality_gated_shortlist_rejection_reason", ""),
            row.get("quality_gated_reason", ""),
            row.get("quality_gated_terminal_reason", ""),
        )
        if bucket == "real_rescue":
            return ""
        if qg_terminal == "shortlisted":
            return "Quality-gated detection did not improve the terminal downstream outcome beyond what default already achieved."
        if bucket == "still_blocked":
            if qg_detail != "":
                return f"Quality-gated detection progressed further downstream but still failed because {reason_text} ({qg_detail})."
            return f"Quality-gated detection progressed further downstream but still failed because {reason_text}."
        if default_terminal == "shortlisted":
            return "Default already reached the downstream shortlist, so the detector gain did not change the downstream result."
        if qg_terminal == "no_downstream_record":
            return "Quality-gated detection added detector-side gain, but no downstream shortlist or quarantine record was produced."
        if qg_detail != "":
            return f"Quality-gated detection still failed downstream because {reason_text} ({qg_detail})."
        return f"Quality-gated detection still failed downstream because {reason_text}."

    def _ordered_count_map(self, series: pd.Series, allowed_order: Optional[List[str]] = None) -> Dict[str, int]:
        counts = series.fillna("").astype(str).value_counts()
        if allowed_order is None:
            return {str(k): int(v) for k, v in counts.items()}
        out: Dict[str, int] = {}
        for key in allowed_order:
            out[str(key)] = int(counts.get(key, 0))
        extras = [str(k) for k in counts.index.tolist() if str(k) not in out]
        for key in sorted(extras):
            out[key] = int(counts.get(key, 0))
        return out

    def run(
        self,
        *,
        winners_csv: Path,
        comparison_csv: Path,
        rollup_csv: Path,
        default_run_dir: Optional[Path],
        default_best_csv: Optional[Path],
        default_quarantine_csv: Optional[Path],
        default_diagnostics_csv: Optional[Path],
        default_funnel_csv: Optional[Path],
        quality_gated_run_dir: Optional[Path],
        quality_gated_best_csv: Optional[Path],
        quality_gated_quarantine_csv: Optional[Path],
        quality_gated_diagnostics_csv: Optional[Path],
        quality_gated_funnel_csv: Optional[Path],
        analysis_csv: Path,
        analysis_rollup_csv: Path,
        real_rescues_csv: Path,
        failure_reason_top_n: int = DEFAULT_FAILURE_REASON_TOP_N,
    ) -> Dict[str, Any]:
        winners = self._read_required_csv(winners_csv).copy()
        comparison = self._read_required_csv(comparison_csv).copy()
        rollup = self._read_required_csv(rollup_csv).copy()
        if "epic_id" not in winners.columns:
            raise ValueError(f"{winners_csv} missing required column: epic_id")
        if "epic_id" not in comparison.columns or "mode" not in comparison.columns:
            raise ValueError(f"{comparison_csv} missing required columns: epic_id/mode")

        winners["epic_id"] = winners["epic_id"].fillna("").astype(str)
        winners["epic_id_canonical"] = winners["epic_id"].map(self.helper._canonical_epic)
        winners = winners.loc[winners["epic_id_canonical"] != ""].drop_duplicates(subset=["epic_id_canonical"], keep="first").reset_index(drop=True)
        comparison["epic_id_canonical"] = comparison["epic_id"].map(self.helper._canonical_epic)

        expected_winners = pd.to_numeric(
            rollup.loc[rollup.get("metric", pd.Series(dtype=str)).astype(str) == "count_with_extra_events_vs_default", "value"],
            errors="coerce",
        )
        if len(expected_winners) > 0:
            expected_value = int(expected_winners.iloc[0])
            if expected_value != int(len(winners)):
                raise ValueError(
                    f"Winner count mismatch: winners CSV has {len(winners)} EPICs but rollup expects {expected_value}."
                )

        default_state = self._load_downstream_state(
            label="default",
            run_dir=default_run_dir,
            best_csv=default_best_csv,
            quarantine_csv=default_quarantine_csv,
            diagnostics_csv=default_diagnostics_csv,
            funnel_csv=default_funnel_csv,
        )
        quality_gated_state = self._load_downstream_state(
            label="quality-gated",
            run_dir=quality_gated_run_dir,
            best_csv=quality_gated_best_csv,
            quarantine_csv=quality_gated_quarantine_csv,
            diagnostics_csv=quality_gated_diagnostics_csv,
            funnel_csv=quality_gated_funnel_csv,
        )

        analysis = winners.copy()
        analysis = analysis.merge(
            self._build_comparison_mode_frame(
                comparison=comparison,
                target_mode=str(K2BatchRunner.DEFAULT_DETECTOR_OPERATING_MODE),
                prefix="default",
            ),
            how="left",
            on="epic_id_canonical",
        )
        analysis = analysis.merge(
            self._build_comparison_mode_frame(
                comparison=comparison,
                target_mode=str(K2BatchRunner.DETECTOR_HIGH_RECALL_QUALITY_GATED_EXPERIMENTAL_MODE),
                prefix="quality_gated",
            ),
            how="left",
            on="epic_id_canonical",
        )
        analysis = analysis.merge(
            self._build_state_outcome_frame(default_state, prefix="default"),
            how="left",
            on="epic_id_canonical",
        )
        analysis = analysis.merge(
            self._build_state_outcome_frame(quality_gated_state, prefix="quality_gated"),
            how="left",
            on="epic_id_canonical",
        )
        analysis = self._append_missing_outcome_defaults(analysis, prefix="default")
        analysis = self._append_missing_outcome_defaults(analysis, prefix="quality_gated")

        qg_is_shortlisted = analysis["quality_gated_terminal_group"].astype(str).eq("shortlisted")
        downstream_improved = analysis["quality_gated_downstream_outcome_score"] > analysis["default_downstream_outcome_score"]
        analysis["winner_bucket"] = "detector_only_gain"
        analysis.loc[downstream_improved & qg_is_shortlisted, "winner_bucket"] = "real_rescue"
        analysis.loc[downstream_improved & (~qg_is_shortlisted), "winner_bucket"] = "still_blocked"
        analysis["downstream_improved_vs_default"] = downstream_improved.astype(bool)
        analysis["downstream_improvement_delta"] = (
            analysis["quality_gated_downstream_outcome_score"] - analysis["default_downstream_outcome_score"]
        )
        analysis["non_rescue_failure_reason_key"] = analysis.apply(
            lambda row: self._failure_reason_key(
                row,
                prefix="quality_gated" if str(row.get("quality_gated_terminal_group", "")) != "shortlisted" else "default",
            )
            if str(row.get("winner_bucket", "")) != "real_rescue"
            else "",
            axis=1,
        )
        analysis["non_rescue_failure_reason_text"] = analysis["non_rescue_failure_reason_key"].map(
            lambda value: "" if str(value).strip() == "" else self._failure_reason_text(str(value))
        )
        analysis["non_rescue_explanation"] = analysis.apply(self._non_rescue_explanation, axis=1)

        column_order = [
            "epic_id",
            "epic_id_canonical",
            "winner_bucket",
            "downstream_improved_vs_default",
            "downstream_improvement_delta",
            "gained_extra_events",
            "improved_best_shape_score",
            "default_n_events",
            "quality_gated_n_events",
            "delta_n_events",
            "default_best_shape_score",
            "quality_gated_best_shape_score",
            "delta_best_shape_score",
            "default_best_depth_snr",
            "quality_gated_best_depth_snr",
            "delta_best_depth_snr",
            "default_detector_mode",
            "quality_gated_detector_mode",
            "default_query",
            "quality_gated_query",
            "default_terminal_group",
            "default_downstream_outcome",
            "default_downstream_outcome_score",
            "default_best_reason",
            "default_best_P",
            "default_best_period_bin",
            "default_manual_review_required",
            "default_best_query",
            "default_failure_reason_bucket",
            "default_failure_category",
            "default_failure_detail",
            "default_reason",
            "default_source_reason",
            "default_shortlist_rejection_reason",
            "default_terminal_reason",
            "default_failure_P",
            "default_failure_period_bin",
            "quality_gated_terminal_group",
            "quality_gated_downstream_outcome",
            "quality_gated_downstream_outcome_score",
            "quality_gated_best_reason",
            "quality_gated_best_P",
            "quality_gated_best_period_bin",
            "quality_gated_manual_review_required",
            "quality_gated_best_query",
            "quality_gated_failure_reason_bucket",
            "quality_gated_failure_category",
            "quality_gated_failure_detail",
            "quality_gated_reason",
            "quality_gated_source_reason",
            "quality_gated_shortlist_rejection_reason",
            "quality_gated_terminal_reason",
            "quality_gated_failure_P",
            "quality_gated_failure_period_bin",
            "non_rescue_failure_reason_key",
            "non_rescue_failure_reason_text",
            "non_rescue_explanation",
        ]
        ordered_existing = [col for col in column_order if col in analysis.columns]
        remaining_cols = [col for col in analysis.columns if col not in ordered_existing]
        analysis = analysis.reindex(columns=ordered_existing + remaining_cols).sort_values(["winner_bucket", "epic_id"]).reset_index(drop=True)

        real_rescues = analysis.loc[analysis["winner_bucket"].astype(str) == "real_rescue"].copy()
        real_rescues = real_rescues.sort_values(["quality_gated_best_P", "epic_id"], na_position="last").reset_index(drop=True)
        winners_total = int(len(analysis))
        real_rescues_count = int((analysis["winner_bucket"] == "real_rescue").sum())
        detector_only_gain_count = int((analysis["winner_bucket"] == "detector_only_gain").sum())
        still_blocked_count = int((analysis["winner_bucket"] == "still_blocked").sum())

        non_rescue = analysis.loc[analysis["winner_bucket"] != "real_rescue"].copy()
        failure_reason_counts = (
            non_rescue["non_rescue_failure_reason_text"].fillna("").astype(str).loc[lambda s: s != ""].value_counts()
        )
        top_failure_reason_counts = {
            str(key): int(value)
            for key, value in failure_reason_counts.head(max(0, int(failure_reason_top_n))).items()
        }
        rescue_bin_counts = self._ordered_count_map(
            real_rescues.get("quality_gated_best_period_bin", pd.Series(dtype=str)),
            allowed_order=self.PERIOD_BIN_ORDER,
        )

        rollup_rows: List[Dict[str, Any]] = [
            {"section": "summary", "metric": "winners_total", "value": winners_total},
            {"section": "summary", "metric": "real_rescues", "value": real_rescues_count},
            {"section": "summary", "metric": "detector_only_gains", "value": detector_only_gain_count},
            {"section": "summary", "metric": "still_blocked", "value": still_blocked_count},
            {"section": "audit", "metric": "detector_rollup_count_with_extra_events_vs_default", "value": winners_total},
            {"section": "input", "metric": "default_best_csv", "value": str(default_state["best_path"])},
            {"section": "input", "metric": "default_quarantine_csv", "value": str(default_state["quarantine_path"])},
            {"section": "input", "metric": "quality_gated_best_csv", "value": str(quality_gated_state["best_path"])},
            {"section": "input", "metric": "quality_gated_quarantine_csv", "value": str(quality_gated_state["quarantine_path"])},
        ]
        for reason_text, count in top_failure_reason_counts.items():
            rollup_rows.append({"section": "top_failure_reason", "metric": reason_text, "value": int(count)})
        for period_bin, count in rescue_bin_counts.items():
            rollup_rows.append({"section": "rescue_period_bin", "metric": period_bin, "value": int(count)})
        rollup_out = pd.DataFrame(rollup_rows)

        analysis_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis_rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        real_rescues_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(analysis_csv, index=False)
        rollup_out.to_csv(analysis_rollup_csv, index=False)
        real_rescues.to_csv(real_rescues_csv, index=False)

        return {
            "analysis_csv": analysis_csv,
            "analysis_rollup_csv": analysis_rollup_csv,
            "real_rescues_csv": real_rescues_csv,
            "winners_total": winners_total,
            "real_rescues": real_rescues_count,
            "detector_only_gains": detector_only_gain_count,
            "still_blocked": still_blocked_count,
            "top_failure_reasons": top_failure_reason_counts,
            "rescue_counts_by_period_bin": rescue_bin_counts,
        }
