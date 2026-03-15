from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig


class K2ShortlistPeriodCompare:
    DEFAULT_BASELINE_RUN_DIR = Path(r"plots\k2_batch\compare_mcc3")
    DEFAULT_TRIAL_RUN_DIR = Path(r"plots\k2_batch\compare_mcc2")
    DEFAULT_RESCUED_CSV = "period_shortlist_rescued_epics.csv"
    DEFAULT_REVIEW_CSV = "period_shortlist_rescued_review.csv"
    DEFAULT_CLUSTER2_RANKED_REVIEW_CSV = "period_shortlist_cluster2_review_ranked.csv"
    DEFAULT_QUALITY_CSV = "period_shortlist_quality_comparison.csv"
    DEFAULT_REPORT_CSV = "period_shortlist_compare_report.csv"
    DEFAULT_REPORT_JSON = "period_shortlist_compare_report.json"

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Compare two K2 shortlist period runs and export rescue artifacts.")
        p.add_argument("--baseline-run-dir", type=Path, default=cls.DEFAULT_BASELINE_RUN_DIR, help=f"Baseline run directory. Default: {cls.DEFAULT_BASELINE_RUN_DIR}")
        p.add_argument("--trial-run-dir", type=Path, default=cls.DEFAULT_TRIAL_RUN_DIR, help=f"Trial run directory. Default: {cls.DEFAULT_TRIAL_RUN_DIR}")
        p.add_argument("--out-dir", type=Path, default=None, help="Output directory for comparison artifacts. Default: trial run directory.")
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.trial_run_dir)
        return cls().run(
            baseline_run_dir=Path(args.baseline_run_dir),
            trial_run_dir=Path(args.trial_run_dir),
            out_dir=out_dir,
        )

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _value_counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
        if col not in df.columns or len(df) == 0:
            return {}
        return {
            str(k): int(v)
            for k, v in df[col].fillna("").astype(str).value_counts().to_dict().items()
        }

    @staticmethod
    def _period_bin_counts(best_df: pd.DataFrame) -> Dict[str, int]:
        if len(best_df) == 0 or "P" not in best_df.columns:
            return {}
        p = pd.to_numeric(best_df["P"], errors="coerce")
        bins = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]
        labels = ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]
        out = pd.cut(p, bins=bins, labels=labels, include_lowest=True, right=True)
        counts = out.value_counts(sort=False, dropna=False)
        return {str(k): int(v) for k, v in counts.items() if str(k) != "nan"}

    @staticmethod
    def _annotate_period_bin(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
        out = df.copy()
        p = pd.to_numeric(out.get(p_col, pd.Series(dtype=float)), errors="coerce")
        bins = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]
        labels = ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]
        out["period_bin"] = pd.cut(p, bins=bins, labels=labels, include_lowest=True, right=True)
        return out

    @staticmethod
    def _match_validated_rows(best_df: pd.DataFrame, validated_df: pd.DataFrame) -> pd.DataFrame:
        if len(best_df) == 0:
            return pd.DataFrame(columns=validated_df.columns)
        if len(validated_df) == 0:
            out = best_df[["epic"]].copy()
            return out.rename(columns={"epic": "epic"})

        best = best_df.copy()
        valid = validated_df.copy()
        best["epic"] = best["epic"].fillna("").astype(str)
        valid["epic"] = valid["epic"].fillna("").astype(str)
        best["P"] = pd.to_numeric(best["P"], errors="coerce")
        valid["P"] = pd.to_numeric(valid["P"], errors="coerce")

        rows: List[Dict[str, Any]] = []
        for _, brow in best.iterrows():
            epic = str(brow["epic"])
            sub = valid.loc[valid["epic"] == epic].copy()
            if len(sub) == 0:
                payload = {c: pd.NA for c in valid.columns}
                payload["epic"] = epic
                rows.append(payload)
                continue
            bp = float(brow["P"]) if pd.notna(brow["P"]) else float("nan")
            if pd.notna(bp):
                sub["_gap"] = (sub["P"] - bp).abs()
                sub = sub.sort_values(["_gap", "P"], ascending=[True, True], kind="mergesort")
            else:
                sub = sub.sort_values(["P"], ascending=[True], kind="mergesort")
            rows.append(sub.iloc[0].drop(labels=["_gap"], errors="ignore").to_dict())
        return pd.DataFrame(rows)

    @staticmethod
    def _quality_rows(group_name: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        work = df.copy()
        for col in ["P", "n_events_after_filters", "hit_rate_snr", "hit_rate_shape", "soft_hit_rate"]:
            work[col] = pd.to_numeric(work.get(col, pd.Series(dtype=float)), errors="coerce")
        scalar_metrics = ["P", "n_events_after_filters", "hit_rate_snr", "hit_rate_shape", "soft_hit_rate"]
        for metric in scalar_metrics:
            s = work[metric].dropna()
            rows.append(
                {
                    "section": "metric_summary",
                    "group": group_name,
                    "metric": metric,
                    "bin_label": "",
                    "count": int(len(s)),
                    "min": float(s.min()) if len(s) > 0 else pd.NA,
                    "median": float(s.median()) if len(s) > 0 else pd.NA,
                    "mean": float(s.mean()) if len(s) > 0 else pd.NA,
                    "max": float(s.max()) if len(s) > 0 else pd.NA,
                }
            )
        bins = K2ShortlistPeriodCompare._annotate_period_bin(work, "P")["period_bin"].value_counts(sort=False, dropna=False)
        total = int(len(work))
        for label, count in bins.items():
            if str(label) == "nan":
                continue
            rows.append(
                {
                    "section": "period_distribution",
                    "group": group_name,
                    "metric": "P",
                    "bin_label": str(label),
                    "count": int(count),
                    "min": pd.NA,
                    "median": pd.NA,
                    "mean": float(count / total) if total > 0 else pd.NA,
                    "max": pd.NA,
                }
            )
        return rows

    @staticmethod
    def _report_rows(section: str, metric: str, baseline_value: Any, trial_value: Any, delta: Any, note: str = "", bin_label: str = "") -> Dict[str, Any]:
        return {
            "section": section,
            "metric": metric,
            "bin_label": bin_label,
            "baseline_value": baseline_value,
            "trial_value": trial_value,
            "delta": delta,
            "note": note,
        }

    @staticmethod
    def _coalesce_series(df: pd.DataFrame, primary: str, secondary: str, default: Any = pd.NA) -> pd.Series:
        if primary in df.columns:
            out = df[primary].copy()
        elif secondary in df.columns:
            out = df[secondary].copy()
        else:
            return pd.Series([default] * len(df), index=df.index)
        if secondary in df.columns and primary in df.columns:
            out = out.where(out.notna(), df[secondary])
        return out

    @staticmethod
    def _bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
        if col not in df.columns:
            return pd.Series([default] * len(df), index=df.index, dtype=bool)
        return df[col].fillna(default).astype(bool)

    @staticmethod
    def _normalized_policy_mode(raw_value: Any, min_cluster_count_value: Any) -> str:
        raw = str(raw_value).strip()
        if raw in {
            K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME,
            K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME,
            "custom_threshold",
        }:
            return raw
        if raw == "default_scientific":
            return K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME
        if raw == "experimental_recovery":
            return K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME
        try:
            min_cluster_count = int(float(min_cluster_count_value))
        except Exception:
            return raw
        if min_cluster_count == int(K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT):
            return K2ShortlistPeriodConfig.PRECISION_FIRST_MODE_NAME
        if min_cluster_count == int(K2ShortlistPeriodConfig.MANUAL_REVIEW_CLUSTER_COUNT_EQ):
            return K2ShortlistPeriodConfig.HIGH_RECALL_MODE_NAME
        return "custom_threshold"

    def _annotate_cluster2_review(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ["P", "cluster_count", "n_events_after_filters", "hit_rate_snr", "hit_rate_shape", "soft_hit_rate"]:
            out[col] = pd.to_numeric(out.get(col, pd.Series(dtype=float)), errors="coerce")

        cluster_target = int(K2ShortlistPeriodConfig.MANUAL_REVIEW_CLUSTER_COUNT_EQ)
        shape_floor_default = getattr(K2ShortlistPeriodConfig, "CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE", None)
        soft_floor_default = getattr(K2ShortlistPeriodConfig, "CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE", None)
        very_short_max = float(getattr(K2ShortlistPeriodConfig, "CLUSTER2_REVIEW_VERY_SHORT_PERIOD_DAYS_MAX", 1.0))
        low_event_max = float(getattr(K2ShortlistPeriodConfig, "CLUSTER2_REVIEW_LOW_EVENT_SUPPORT_MAX", 2))
        near_zero_shape_max = float(getattr(K2ShortlistPeriodConfig, "CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SHAPE_MAX", 0.05))
        near_zero_snr_max = float(getattr(K2ShortlistPeriodConfig, "CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SNR_MAX", 0.05))

        cluster2_mask = out["cluster_count"].eq(float(cluster_target))
        if "manual_review_required" not in out.columns:
            out["manual_review_required"] = cluster2_mask.astype(bool)
        if "manual_review_reason" not in out.columns:
            out["manual_review_reason"] = ""
            out.loc[cluster2_mask, "manual_review_reason"] = (
                f"validated_cluster_count=={cluster_target}; experimental MCC recovery candidate"
            )

        if "cluster2_guardrail_hit_rate_shape_min" not in out.columns:
            out["cluster2_guardrail_hit_rate_shape_min"] = shape_floor_default if shape_floor_default is not None else pd.NA
        if "cluster2_guardrail_soft_hit_rate_min" not in out.columns:
            out["cluster2_guardrail_soft_hit_rate_min"] = soft_floor_default if soft_floor_default is not None else pd.NA
        out["cluster2_guardrail_hit_rate_shape_min"] = pd.to_numeric(out["cluster2_guardrail_hit_rate_shape_min"], errors="coerce")
        out["cluster2_guardrail_soft_hit_rate_min"] = pd.to_numeric(out["cluster2_guardrail_soft_hit_rate_min"], errors="coerce")
        out["cluster2_guardrail_hit_rate_shape_min"] = out["cluster2_guardrail_hit_rate_shape_min"].fillna(shape_floor_default)
        out["cluster2_guardrail_soft_hit_rate_min"] = out["cluster2_guardrail_soft_hit_rate_min"].fillna(soft_floor_default)

        shape_floor = out["cluster2_guardrail_hit_rate_shape_min"]
        soft_floor = out["cluster2_guardrail_soft_hit_rate_min"]
        shape_pass = (~cluster2_mask) | shape_floor.isna() | out["hit_rate_shape"].ge(shape_floor)
        soft_pass = (~cluster2_mask) | soft_floor.isna() | out["soft_hit_rate"].ge(soft_floor)
        out["cluster2_guardrail_hit_rate_shape_pass"] = self._bool_series(out, "cluster2_guardrail_hit_rate_shape_pass", default=True) & shape_pass
        out["cluster2_guardrail_soft_hit_rate_pass"] = self._bool_series(out, "cluster2_guardrail_soft_hit_rate_pass", default=True) & soft_pass
        out["cluster2_guardrail_pass"] = self._bool_series(out, "cluster2_guardrail_pass", default=True) & out["cluster2_guardrail_hit_rate_shape_pass"] & out["cluster2_guardrail_soft_hit_rate_pass"]

        reasons = out.get("cluster2_guardrail_reason", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        fail_shape = cluster2_mask & (~out["cluster2_guardrail_hit_rate_shape_pass"]) & shape_floor.notna()
        fail_soft = cluster2_mask & (~out["cluster2_guardrail_soft_hit_rate_pass"]) & soft_floor.notna()
        if bool(fail_shape.any()):
            reasons.loc[fail_shape] = reasons.loc[fail_shape] + shape_floor.loc[fail_shape].map(
                lambda x: f"hit_rate_shape<{float(x):.3f};"
            )
        if bool(fail_soft.any()):
            reasons.loc[fail_soft] = reasons.loc[fail_soft] + soft_floor.loc[fail_soft].map(
                lambda x: f"soft_hit_rate<{float(x):.3f};"
            )
        out["cluster2_guardrail_reason"] = reasons.str.strip(";")

        out["cluster2_watch_very_short_period"] = self._bool_series(out, "cluster2_watch_very_short_period", default=False) | (cluster2_mask & out["P"].le(very_short_max))
        out["cluster2_watch_low_event_support"] = self._bool_series(out, "cluster2_watch_low_event_support", default=False) | (cluster2_mask & out["n_events_after_filters"].le(low_event_max))
        out["cluster2_watch_near_zero_hit_rate_shape"] = self._bool_series(out, "cluster2_watch_near_zero_hit_rate_shape", default=False) | (cluster2_mask & out["hit_rate_shape"].lt(near_zero_shape_max))
        out["cluster2_watch_near_zero_hit_rate_snr"] = self._bool_series(out, "cluster2_watch_near_zero_hit_rate_snr", default=False) | (cluster2_mask & out["hit_rate_snr"].lt(near_zero_snr_max))

        notes = pd.Series([""] * len(out), index=out.index, dtype=str)
        notes.loc[out["cluster2_watch_very_short_period"]] = notes.loc[out["cluster2_watch_very_short_period"]] + "very_short_period;"
        notes.loc[out["cluster2_watch_low_event_support"]] = notes.loc[out["cluster2_watch_low_event_support"]] + "two_filtered_events_or_fewer;"
        notes.loc[out["cluster2_watch_near_zero_hit_rate_shape"]] = notes.loc[out["cluster2_watch_near_zero_hit_rate_shape"]] + "near_zero_hit_rate_shape;"
        notes.loc[out["cluster2_watch_near_zero_hit_rate_snr"]] = notes.loc[out["cluster2_watch_near_zero_hit_rate_snr"]] + "near_zero_hit_rate_snr;"
        existing_notes = out.get("cluster2_watch_notes", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["cluster2_watch_notes"] = (existing_notes + notes).str.strip(";")
        out["weak_evidence_score"] = (
            (~out["cluster2_guardrail_pass"]).astype(int) * 100
            + out["cluster2_watch_very_short_period"].astype(int) * 20
            + out["cluster2_watch_low_event_support"].astype(int) * 20
            + out["cluster2_watch_near_zero_hit_rate_shape"].astype(int) * 15
            + out["cluster2_watch_near_zero_hit_rate_snr"].astype(int) * 10
        )
        return out

    def run(self, baseline_run_dir: Path, trial_run_dir: Path, out_dir: Path) -> Dict[str, Any]:
        baseline_best = self._annotate_cluster2_review(self._read_csv(baseline_run_dir / "period_shortlist_best.csv"))
        trial_best = self._annotate_cluster2_review(self._read_csv(trial_run_dir / "period_shortlist_best.csv"))
        baseline_validated = self._read_csv(baseline_run_dir / "period_shortlist_summary_validated_only.csv")
        trial_validated = self._read_csv(trial_run_dir / "period_shortlist_summary_validated_only.csv")
        baseline_funnel = self._read_csv(baseline_run_dir / "epic_funnel_reasons.csv")
        trial_funnel = self._read_csv(trial_run_dir / "epic_funnel_reasons.csv")
        baseline_diag = self._read_csv(baseline_run_dir / "period_shortlist_diagnostics.csv")
        trial_diag = self._read_csv(trial_run_dir / "period_shortlist_diagnostics.csv")

        baseline_best_epics = set(baseline_best.get("epic", pd.Series(dtype=str)).fillna("").astype(str))
        trial_best_epics = set(trial_best.get("epic", pd.Series(dtype=str)).fillna("").astype(str))
        baseline_validated_epics = set(baseline_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str))
        trial_validated_epics = set(trial_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str))

        candidate_rescued_epics = sorted([x for x in (trial_best_epics - baseline_best_epics) if x != ""])
        validated_rescued_epics = sorted([x for x in (trial_validated_epics - baseline_validated_epics) if x != ""])

        candidate_rescued = trial_best.loc[trial_best.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()
        validated_rescued = trial_validated.loc[trial_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()
        baseline_best_rescued = baseline_best.loc[baseline_best.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()
        baseline_validated_rescued = baseline_validated.loc[baseline_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()
        matched_trial_validated = self._match_validated_rows(candidate_rescued, validated_rescued)
        matched_baseline_validated = self._match_validated_rows(candidate_rescued, baseline_validated_rescued)

        trial_funnel_sub = trial_funnel.loc[trial_funnel.get("epic_id", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()
        baseline_funnel_sub = baseline_funnel.loc[baseline_funnel.get("epic_id", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)].copy()

        for df, col in [
            (candidate_rescued, "epic"),
            (matched_trial_validated, "epic"),
            (baseline_best_rescued, "epic"),
            (matched_baseline_validated, "epic"),
            (trial_funnel_sub, "epic_id"),
            (baseline_funnel_sub, "epic_id"),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        candidate_rescued = candidate_rescued.add_prefix("trial_best_")
        matched_trial_validated = matched_trial_validated.add_prefix("trial_validated_")
        baseline_best_rescued = baseline_best_rescued.add_prefix("baseline_best_")
        matched_baseline_validated = matched_baseline_validated.add_prefix("baseline_validated_")
        trial_funnel_sub = trial_funnel_sub.add_prefix("trial_funnel_")
        baseline_funnel_sub = baseline_funnel_sub.add_prefix("baseline_funnel_")

        rescued = pd.DataFrame({"epic": candidate_rescued_epics})
        rescued = rescued.merge(candidate_rescued, how="left", left_on="epic", right_on="trial_best_epic")
        rescued = rescued.merge(matched_trial_validated, how="left", left_on="epic", right_on="trial_validated_epic")
        rescued = rescued.merge(baseline_best_rescued, how="left", left_on="epic", right_on="baseline_best_epic")
        rescued = rescued.merge(matched_baseline_validated, how="left", left_on="epic", right_on="baseline_validated_epic")
        rescued = rescued.merge(trial_funnel_sub, how="left", left_on="epic", right_on="trial_funnel_epic_id")
        rescued = rescued.merge(baseline_funnel_sub, how="left", left_on="epic", right_on="baseline_funnel_epic_id")
        rescued = rescued.drop_duplicates(subset=["epic"], keep="first").reset_index(drop=True)
        rescued["candidate_stage_rescued"] = True
        rescued["validated_stage_rescued"] = rescued["epic"].isin(validated_rescued_epics)

        trial_validated_counts = (
            trial_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
            if len(trial_validated) > 0
            else {}
        )
        baseline_validated_counts = (
            baseline_validated.get("epic", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
            if len(baseline_validated) > 0
            else {}
        )
        rescued["trial_validated_row_count"] = rescued["epic"].map(lambda x: int(trial_validated_counts.get(str(x), 0)))
        rescued["baseline_validated_row_count"] = rescued["epic"].map(lambda x: int(baseline_validated_counts.get(str(x), 0)))

        report_rows: List[Dict[str, Any]] = []
        report_rows.append(self._report_rows("candidate_stage", "best_rows", len(baseline_best), len(trial_best), len(trial_best) - len(baseline_best)))
        report_rows.append(self._report_rows("candidate_stage", "best_unique_epics", len(baseline_best_epics), len(trial_best_epics), len(trial_best_epics) - len(baseline_best_epics)))
        report_rows.append(self._report_rows("candidate_stage", "rescued_unique_epics", 0, len(candidate_rescued_epics), len(candidate_rescued_epics)))
        report_rows.append(self._report_rows("validated_stage", "validated_rows", len(baseline_validated), len(trial_validated), len(trial_validated) - len(baseline_validated)))
        report_rows.append(self._report_rows("validated_stage", "validated_unique_epics", len(baseline_validated_epics), len(trial_validated_epics), len(trial_validated_epics) - len(baseline_validated_epics)))
        report_rows.append(self._report_rows("validated_stage", "rescued_unique_epics", 0, len(validated_rescued_epics), len(validated_rescued_epics)))

        baseline_bins = self._period_bin_counts(baseline_best)
        trial_bins = self._period_bin_counts(trial_best)
        for bin_label in ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]:
            b = int(baseline_bins.get(bin_label, 0))
            t = int(trial_bins.get(bin_label, 0))
            report_rows.append(self._report_rows("period_bin_best", "best_count", b, t, t - b, bin_label=bin_label))

        baseline_quarantine = self._read_csv(baseline_run_dir / "period_shortlist_quarantine.csv")
        trial_quarantine = self._read_csv(trial_run_dir / "period_shortlist_quarantine.csv")
        base_fail = self._value_counts(baseline_quarantine, "failure_category")
        trial_fail = self._value_counts(trial_quarantine, "failure_category")
        for key in sorted(set(base_fail.keys()).union(trial_fail.keys())):
            b = int(base_fail.get(key, 0))
            t = int(trial_fail.get(key, 0))
            report_rows.append(self._report_rows("quarantine", key, b, t, t - b))

        if len(baseline_validated_epics) > 0 or len(trial_validated_epics) > 0:
            baseline_validation_status = "validated_outputs_present" if len(baseline_validated_epics) > 0 else "no_validated_outputs"
            trial_validation_status = "validated_outputs_present" if len(trial_validated_epics) > 0 else "no_validated_outputs"
            validation_note = (
                "Validation-capable comparison: baseline and/or trial runs produced validated outputs, "
                "so validated-stage rescue counts are based on actual validated rows."
            )
        else:
            baseline_validation_status = "cluster_only_validation_error"
            trial_validation_status = "cluster_only_validation_error"
            validation_note = (
                "Validation is enabled in the runner, but current runs fail in K2_NoiseHandler.fetch_best() "
                "at the MAST search/download stage before local validation metrics can be computed."
            )
        report_rows.append(
            self._report_rows(
                "validation_diagnosis",
                "environment_status",
                baseline_validation_status,
                trial_validation_status,
                "",
                note=validation_note,
            )
        )

        if len(baseline_diag) > 0 and len(trial_diag) > 0:
            bdiag = baseline_diag.iloc[0].to_dict()
            tdiag = trial_diag.iloc[0].to_dict()
            policy_pairs = [
                (
                    "mcc_policy_mode",
                    self._normalized_policy_mode(bdiag.get("mcc_policy_mode", ""), bdiag.get("min_cluster_count", "")),
                    self._normalized_policy_mode(tdiag.get("mcc_policy_mode", ""), tdiag.get("min_cluster_count", "")),
                ),
                ("min_cluster_count", bdiag.get("min_cluster_count", ""), tdiag.get("min_cluster_count", "")),
                ("default_min_cluster_count", bdiag.get("default_min_cluster_count", ""), tdiag.get("default_min_cluster_count", "")),
                ("manual_review_cluster_count_eq", bdiag.get("manual_review_cluster_count_eq", ""), tdiag.get("manual_review_cluster_count_eq", "")),
                (
                    "cluster2_guardrail_hit_rate_shape_min",
                    bdiag.get("cluster2_guardrail_hit_rate_shape_min", ""),
                    tdiag.get("cluster2_guardrail_hit_rate_shape_min", ""),
                ),
                (
                    "cluster2_guardrail_soft_hit_rate_min",
                    bdiag.get("cluster2_guardrail_soft_hit_rate_min", ""),
                    tdiag.get("cluster2_guardrail_soft_hit_rate_min", ""),
                ),
            ]
            for metric, bval, tval in policy_pairs:
                report_rows.append(
                    self._report_rows(
                        "policy",
                        metric,
                        bval,
                        tval,
                        "",
                    )
                )
            for metric in ["rows_best", "rows_validated_only", "n_validated_period", "n_quarantined_no_cluster_periods"]:
                b = int(float(bdiag.get(metric, 0)))
                t = int(float(tdiag.get(metric, 0)))
                report_rows.append(self._report_rows("diagnostics", metric, b, t, t - b))
        review_df = pd.DataFrame(
            {
                "epic": rescued["epic"],
                "query": rescued.get("trial_best_query"),
                "baseline_terminal_reason": rescued.get("baseline_funnel_terminal_reason"),
                "baseline_source_reason": rescued.get("baseline_funnel_source_reason"),
                "trial_reason": rescued.get("trial_best_reason"),
                "trial_P": pd.to_numeric(rescued.get("trial_best_P"), errors="coerce"),
                "trial_cluster_count": pd.to_numeric(rescued.get("trial_best_cluster_count"), errors="coerce"),
                "trial_n_events_after_filters": pd.to_numeric(rescued.get("trial_best_n_events_after_filters"), errors="coerce"),
                "trial_n_predicted": pd.to_numeric(rescued.get("trial_best_n_predicted"), errors="coerce"),
                "trial_n_covered": pd.to_numeric(rescued.get("trial_best_n_covered"), errors="coerce"),
                "trial_coverage_rate": pd.to_numeric(rescued.get("trial_best_coverage_rate"), errors="coerce"),
                "trial_hit_rate_snr": pd.to_numeric(rescued.get("trial_best_hit_rate_snr"), errors="coerce"),
                "trial_hit_rate_shape": pd.to_numeric(rescued.get("trial_best_hit_rate_shape"), errors="coerce"),
                "trial_soft_hit_rate": pd.to_numeric(rescued.get("trial_best_soft_hit_rate"), errors="coerce"),
                "trial_n_windows_with_no_candidates": pd.to_numeric(rescued.get("trial_best_n_windows_with_no_candidates"), errors="coerce"),
                "trial_manual_review_required": self._coalesce_series(rescued, "trial_best_manual_review_required", "trial_validated_manual_review_required"),
                "trial_manual_review_reason": self._coalesce_series(rescued, "trial_best_manual_review_reason", "trial_validated_manual_review_reason", default=""),
                "trial_cluster2_guardrail_pass": self._coalesce_series(rescued, "trial_best_cluster2_guardrail_pass", "trial_validated_cluster2_guardrail_pass"),
                "trial_cluster2_guardrail_reason": self._coalesce_series(rescued, "trial_best_cluster2_guardrail_reason", "trial_validated_cluster2_guardrail_reason", default=""),
                "trial_validated_row_count": rescued.get("trial_validated_row_count"),
                "validated_stage_rescued": rescued.get("validated_stage_rescued"),
            }
        )
        review_df = self._annotate_period_bin(review_df.rename(columns={"trial_P": "P"}), "P").rename(columns={"P": "trial_P"})
        trial_best_review = self._annotate_cluster2_review(trial_best.copy())
        cluster2_ranked_review = trial_best_review.loc[
            pd.to_numeric(trial_best_review.get("cluster_count", pd.Series(dtype=float)), errors="coerce").eq(
                float(K2ShortlistPeriodConfig.MANUAL_REVIEW_CLUSTER_COUNT_EQ)
            )
        ].copy()
        cluster2_ranked_review["candidate_stage_rescued"] = cluster2_ranked_review.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(candidate_rescued_epics)
        cluster2_ranked_review["validated_stage_rescued"] = cluster2_ranked_review.get("epic", pd.Series(dtype=str)).fillna("").astype(str).isin(validated_rescued_epics)
        cluster2_ranked_review["epic"] = cluster2_ranked_review.get("epic", pd.Series(dtype=str)).fillna("").astype(str)
        baseline_cluster2_context = baseline_funnel_sub.rename(
            columns={
                "baseline_funnel_epic_id": "epic_id",
                "baseline_funnel_terminal_reason": "baseline_terminal_reason",
                "baseline_funnel_source_reason": "baseline_source_reason",
            }
        )[["epic_id", "baseline_terminal_reason", "baseline_source_reason"]].copy()
        baseline_cluster2_context["epic_id"] = baseline_cluster2_context.get("epic_id", pd.Series(dtype=str)).fillna("").astype(str)
        cluster2_ranked_review = cluster2_ranked_review.merge(
            baseline_cluster2_context.drop_duplicates(subset=["epic_id"]),
            how="left",
            left_on="epic",
            right_on="epic_id",
        ).drop(columns=["epic_id"], errors="ignore")
        cluster2_ranked_review = self._annotate_period_bin(cluster2_ranked_review, "P")
        cluster2_ranked_review = cluster2_ranked_review.sort_values(
            by=[
                "weak_evidence_score",
                "hit_rate_shape",
                "soft_hit_rate",
                "hit_rate_snr",
                "n_events_after_filters",
                "P",
            ],
            ascending=[False, True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        cluster2_ranked_review.insert(0, "review_rank", range(1, len(cluster2_ranked_review) + 1))
        cluster2_ranked_review = cluster2_ranked_review[
            [
                "review_rank",
                "epic",
                "query",
                "reason",
                "P",
                "period_bin",
                "cluster_count",
                "n_events_after_filters",
                "coverage_rate",
                "hit_rate_snr",
                "hit_rate_shape",
                "soft_hit_rate",
                "manual_review_required",
                "manual_review_reason",
                "cluster2_guardrail_hit_rate_shape_min",
                "cluster2_guardrail_soft_hit_rate_min",
                "cluster2_guardrail_pass",
                "cluster2_guardrail_reason",
                "cluster2_watch_very_short_period",
                "cluster2_watch_low_event_support",
                "cluster2_watch_near_zero_hit_rate_shape",
                "cluster2_watch_near_zero_hit_rate_snr",
                "cluster2_watch_notes",
                "weak_evidence_score",
                "candidate_stage_rescued",
                "validated_stage_rescued",
                "baseline_terminal_reason",
                "baseline_source_reason",
            ]
        ]
        quality_rows = self._quality_rows("original_validated", baseline_best.copy()) + self._quality_rows("rescued_validated", candidate_rescued.rename(columns=lambda c: c.replace("trial_best_", "")).copy())
        quality_df = pd.DataFrame(quality_rows)
        for metric in ["n_events_after_filters", "hit_rate_snr", "hit_rate_shape", "soft_hit_rate"]:
            base_metric = quality_df.loc[
                (quality_df["section"] == "metric_summary")
                & (quality_df["group"] == "original_validated")
                & (quality_df["metric"] == metric)
            ]
            rescued_metric = quality_df.loc[
                (quality_df["section"] == "metric_summary")
                & (quality_df["group"] == "rescued_validated")
                & (quality_df["metric"] == metric)
            ]
            if len(base_metric) > 0 and len(rescued_metric) > 0:
                b_med = pd.to_numeric(pd.Series([base_metric["median"].iloc[0]]), errors="coerce").iloc[0]
                t_med = pd.to_numeric(pd.Series([rescued_metric["median"].iloc[0]]), errors="coerce").iloc[0]
                b_mean = pd.to_numeric(pd.Series([base_metric["mean"].iloc[0]]), errors="coerce").iloc[0]
                t_mean = pd.to_numeric(pd.Series([rescued_metric["mean"].iloc[0]]), errors="coerce").iloc[0]
                if pd.notna(b_med) and pd.notna(t_med):
                    report_rows.append(self._report_rows("quality_vs_baseline", f"{metric}_median", float(b_med), float(t_med), float(t_med - b_med)))
                if pd.notna(b_mean) and pd.notna(t_mean):
                    report_rows.append(self._report_rows("quality_vs_baseline", f"{metric}_mean", float(b_mean), float(t_mean), float(t_mean - b_mean)))
        for bin_label in ["(0,1]", "(1,5]", "(5,10]", "(10,15]", "(15,20]"]:
            base_bin = quality_df.loc[
                (quality_df["section"] == "period_distribution")
                & (quality_df["group"] == "original_validated")
                & (quality_df["bin_label"] == bin_label)
            ]
            rescued_bin = quality_df.loc[
                (quality_df["section"] == "period_distribution")
                & (quality_df["group"] == "rescued_validated")
                & (quality_df["bin_label"] == bin_label)
            ]
            b = int(base_bin["count"].iloc[0]) if len(base_bin) > 0 else 0
            t = int(rescued_bin["count"].iloc[0]) if len(rescued_bin) > 0 else 0
            report_rows.append(self._report_rows("quality_period_distribution", "validated_count", b, t, t - b, bin_label=bin_label))
        report_df = pd.DataFrame(report_rows)
        report_json = {
            "baseline_run_dir": str(baseline_run_dir),
            "trial_run_dir": str(trial_run_dir),
            "candidate_stage": {
                "baseline_best_unique_epics": int(len(baseline_best_epics)),
                "trial_best_unique_epics": int(len(trial_best_epics)),
                "rescued_epics": candidate_rescued_epics,
            },
            "validated_stage": {
                "baseline_validated_unique_epics": int(len(baseline_validated_epics)),
                "trial_validated_unique_epics": int(len(trial_validated_epics)),
                "rescued_epics": validated_rescued_epics,
            },
            "validation_diagnosis": validation_note,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        rescued_csv = out_dir / self.DEFAULT_RESCUED_CSV
        review_csv = out_dir / self.DEFAULT_REVIEW_CSV
        cluster2_review_csv = out_dir / self.DEFAULT_CLUSTER2_RANKED_REVIEW_CSV
        quality_csv = out_dir / self.DEFAULT_QUALITY_CSV
        report_csv = out_dir / self.DEFAULT_REPORT_CSV
        report_json_path = out_dir / self.DEFAULT_REPORT_JSON
        rescued.to_csv(rescued_csv, index=False)
        review_df.to_csv(review_csv, index=False)
        cluster2_ranked_review.to_csv(cluster2_review_csv, index=False)
        quality_df.to_csv(quality_csv, index=False)
        report_df.to_csv(report_csv, index=False)
        report_json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

        print(f"[K2ShortlistPeriodCompare] wrote rescued_csv: {rescued_csv}")
        print(f"[K2ShortlistPeriodCompare] wrote review_csv: {review_csv}")
        print(f"[K2ShortlistPeriodCompare] wrote cluster2_review_csv: {cluster2_review_csv}")
        print(f"[K2ShortlistPeriodCompare] wrote quality_csv: {quality_csv}")
        print(f"[K2ShortlistPeriodCompare] wrote report_csv: {report_csv}")
        print(f"[K2ShortlistPeriodCompare] wrote report_json: {report_json_path}")

        return {
            "rescued_csv": rescued_csv,
            "review_csv": review_csv,
            "cluster2_review_csv": cluster2_review_csv,
            "quality_csv": quality_csv,
            "report_csv": report_csv,
            "report_json": report_json_path,
            "candidate_rescued_unique_epics": int(len(candidate_rescued_epics)),
            "validated_rescued_unique_epics": int(len(validated_rescued_epics)),
        }
