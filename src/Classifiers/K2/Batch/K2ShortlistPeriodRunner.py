from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodConfig import K2ShortlistPeriodConfig
from src.Classifiers.K2.K2_TimeDomainTransitPipeline import infer_periods_from_events
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2PipelineStageError, K2_NoiseHandler
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR
from src.Classifiers.K2.Systematics.K2Validation_Prediction import K2Validation_Prediction


class K2ShortlistPeriodRunner:
    SUMMARY_COLUMNS = [
        "epic",
        "query",
        "reason",
        "n_events_raw",
        "n_events_after_filters",
        "P",
        "cluster_count",
        "cluster_center_phase",
        "n_predicted",
        "n_covered",
        "coverage_rate",
        "hit_rate_snr",
        "hit_rate_shape",
        "soft_hit_rate",
        "n_windows_with_no_candidates",
        "no_cand",
    ]
    QUARANTINE_COLUMNS = [
        "epic_id",
        "query",
        "reason",
        "missing_upstream_source",
        "source_reason",
        "whiteness_missing",
        "whiteness_null_reason_category",
        "shortlist_rejection_stage",
        "shortlist_rejection_reason",
        "rejected_before_candidate_scoring",
        "rejected_after_candidate_scoring",
        "failure_category",
        "failure_detail",
        "n_events_raw",
        "n_events_after_filters",
        "infer_max_period_days",
        "infer_min_hits",
        "infer_tol_frac",
        "min_cluster_count",
        "period_cap_days",
        "min_period_days",
        "top_k_periods",
        "hist_total",
        "hist_finite_period",
        "hist_in_period_range",
        "hist_pass_cluster_count",
        "hist_pass_all_filters",
        "P",
    ]
    RAW_EPIC_REQUIRED_COLUMNS = [
        "query",
        "triage_status",
        "triage_usable",
        "triage_whiteness_definition",
        "triage_why_not_usable",
    ]
    RAW_EPIC_REQUIRED_WHITENESS_VALUE_COLUMNS = ("triage_whiteness_pvalue", "triage_whiteness_score")

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Run K2 shortlist period analysis from a whiteness-precomputed batch CSV.")
        p.add_argument(
            "--min-cluster-count",
            "--min_cluster_count",
            dest="min_cluster_count",
            type=int,
            default=K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT,
            help=f"Minimum count_hits required for a candidate period cluster. Default: {K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT}",
        )
        p.add_argument(
            "--run-id",
            dest="run_id",
            default=None,
            help="Optional run identifier used for the output subdirectory.",
        )
        p.add_argument(
            "--period-stage-n",
            "--period_stage_n",
            dest="period_stage_n",
            type=int,
            default=None,
            help=(
                "Optional override for PERIOD_STAGE_N when PERIOD_STAGE_SELECTION_MODE='randomN'. "
                f"Default policy: {K2ShortlistPeriodConfig.PERIOD_STAGE_N}."
            ),
        )
        p.add_argument(
            "--cluster2-min-hit-rate-shape",
            "--cluster2_min_hit_rate_shape",
            dest="cluster2_min_hit_rate_shape",
            type=float,
            default=None,
            help=(
                "Validated-stage guardrail applied only when cluster_count==2. "
                f"Rejects rows below this hit_rate_shape floor. Default policy: "
                f"{K2ShortlistPeriodConfig.CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE}."
            ),
        )
        p.add_argument(
            "--cluster2-min-soft-hit-rate",
            "--cluster2_min_soft_hit_rate",
            dest="cluster2_min_soft_hit_rate",
            type=float,
            default=None,
            help=(
                "Validated-stage guardrail applied only when cluster_count==2. "
                f"Rejects rows below this soft_hit_rate floor. Default policy: "
                f"{K2ShortlistPeriodConfig.CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE}."
            ),
        )
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        config_kwargs: Dict[str, Any] = {
            "MIN_CLUSTER_COUNT": int(args.min_cluster_count),
            "RUN_ID": (str(args.run_id) if args.run_id is not None and str(args.run_id).strip() != "" else K2ShortlistPeriodConfig.RUN_ID),
        }
        if args.period_stage_n is not None:
            config_kwargs["PERIOD_STAGE_N"] = int(args.period_stage_n)
        if args.cluster2_min_hit_rate_shape is not None:
            config_kwargs["CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE"] = float(args.cluster2_min_hit_rate_shape)
        if args.cluster2_min_soft_hit_rate is not None:
            config_kwargs["CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE"] = float(args.cluster2_min_soft_hit_rate)
        config = K2ShortlistPeriodConfig(**config_kwargs)
        return cls(config=config).run()

    def __init__(self, config: Optional[K2ShortlistPeriodConfig] = None) -> None:
        self.config = config if config is not None else K2ShortlistPeriodConfig()
        self._period_file_re = re.compile(r"^period_([0-9]+(?:\.[0-9]+)?)_(hits|misses|uncovered)\.csv$", flags=re.IGNORECASE)

    def _mcc_policy_mode(self) -> str:
        default_mcc = int(K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT)
        current_mcc = int(getattr(self.config, "MIN_CLUSTER_COUNT", default_mcc))
        high_recall_mcc = int(getattr(self.config, "MANUAL_REVIEW_CLUSTER_COUNT_EQ", 2))
        if current_mcc == default_mcc:
            return str(getattr(self.config, "PRECISION_FIRST_MODE_NAME", "precision_first_default"))
        if current_mcc == high_recall_mcc:
            return str(getattr(self.config, "HIGH_RECALL_MODE_NAME", "supported_high_recall"))
        return "custom_threshold"

    def _mcc_policy_note(self) -> str:
        mode = self._mcc_policy_mode()
        if mode == str(getattr(self.config, "PRECISION_FIRST_MODE_NAME", "precision_first_default")):
            return (
                "MIN_CLUSTER_COUNT=3 is the conservative, precision-first default. "
                "MCC=2 is the supported high-recall mode and cluster_count==2 validated rows require guardrails and manual review."
            )
        if mode == str(getattr(self.config, "HIGH_RECALL_MODE_NAME", "supported_high_recall")):
            return (
                "MIN_CLUSTER_COUNT=2 is the supported high-recall mode. "
                "It increases validated yield, but preferentially admits lower-support, weaker-hit-rate candidates; keep cluster_count==2 guardrails and review enabled."
            )
        return (
            "Custom MIN_CLUSTER_COUNT in use; compare against the MCC=3 default before promoting any threshold change."
        )

    @staticmethod
    def _sanitize_run_id(value: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return text.strip("._-") or "run"

    def _resolve_run_output_paths(self) -> Dict[str, Path]:
        cfg = self.config
        base_out_dir = cfg.out_dir_path
        use_run_subdir = bool(getattr(cfg, "USE_RUN_SUBDIR", True))
        run_id_cfg = getattr(cfg, "RUN_ID", None)
        run_id = self._sanitize_run_id(str(run_id_cfg)) if (run_id_cfg is not None and str(run_id_cfg).strip() != "") else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        run_dir = base_out_dir
        if use_run_subdir:
            candidate = base_out_dir / run_id
            if candidate.exists():
                i = 1
                while True:
                    candidate_i = base_out_dir / f"{run_id}_{i:02d}"
                    if not candidate_i.exists():
                        candidate = candidate_i
                        break
                    i += 1
            run_dir = candidate

        def _name_only(path: Path) -> str:
            return Path(path).name

        return {
            "run_id": Path(run_dir).name if use_run_subdir else run_id,
            "run_dir": run_dir,
            "out_summary_csv": run_dir / _name_only(cfg.out_summary_csv_path),
            "out_summary_unique_epicp_csv": run_dir / _name_only(cfg.out_summary_unique_epicp_csv_path),
            "out_summary_validated_only_csv": run_dir / _name_only(cfg.out_summary_validated_only_csv_path),
            "out_best_csv": run_dir / _name_only(cfg.out_best_csv_path),
            "out_quarantine_csv": run_dir / _name_only(cfg.out_quarantine_csv_path),
            "out_diagnostics_csv": run_dir / _name_only(cfg.out_diagnostics_csv_path),
            "out_epic_funnel_reasons_csv": run_dir / _name_only(cfg.out_epic_funnel_reasons_csv_path),
            "out_period_hist_png": run_dir / _name_only(cfg.out_period_hist_png_path),
            "out_period_hist_counts_csv": run_dir / _name_only(cfg.out_period_hist_counts_csv_path),
        }

    @staticmethod
    def _extract_epic(query: str) -> Optional[str]:
        m = re.search(r"\d+", str(query))
        return m.group(0) if m is not None else None

    @staticmethod
    def _as_float(value: Any, default: float = float("nan")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if np.isfinite(out) else float(default)

    @staticmethod
    def _to_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
        if len(series) == 0:
            return pd.Series([], dtype=bool)
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any():
            return num.fillna(0).astype(int).astype(bool)
        low = series.fillna("").astype(str).str.strip().str.lower()
        true_set = {"true", "t", "yes", "y", "1"}
        false_set = {"false", "f", "no", "n", "0"}
        out = pd.Series([default] * len(series), index=series.index, dtype=bool)
        out.loc[low.isin(true_set)] = True
        out.loc[low.isin(false_set)] = False
        return out

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _phase_cluster_score_quiet(events_df: pd.DataFrame, period: float, tol_phase: float) -> Tuple[int, float]:
        if "t_mid" not in events_df.columns or (not np.isfinite(period)) or period <= 0:
            return 0, float("nan")
        t = pd.to_numeric(events_df["t_mid"], errors="coerce")
        ok = np.isfinite(t.to_numpy(dtype=float))
        if not np.any(ok):
            return 0, float("nan")
        work = events_df.loc[ok].copy()
        work = work.assign(t_mid=pd.to_numeric(work["t_mid"], errors="coerce"))
        work = work.assign(phase=(np.mod(work["t_mid"].to_numpy(dtype=float), float(period)) / float(period)))
        work = work.sort_values("phase")
        phases = work["phase"].to_numpy(dtype=float)
        n = len(phases)
        if n == 0:
            return 0, float("nan")

        phase2 = np.concatenate([phases, phases + 1.0])
        best_count, best_start, best_end = 0, 0, -1
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j + 1 < i + n and (phase2[j + 1] - phase2[i]) <= float(tol_phase) + 1e-12:
                j += 1
            count = int(j - i + 1)
            if count > best_count:
                best_count, best_start, best_end = count, i, j

        if best_count <= 0:
            return 0, float("nan")
        cluster_phases = np.asarray(phase2[best_start:best_end + 1], dtype=float)
        theta = 2.0 * np.pi * np.mod(cluster_phases, 1.0)
        mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
        if mean_angle < 0:
            mean_angle += 2.0 * np.pi
        return int(best_count), float(mean_angle / (2.0 * np.pi))

    def _discover_period_files(self, epic_dir: Path) -> List[Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        for p in epic_dir.glob("period_*_*.csv"):
            m = self._period_file_re.match(p.name)
            if m is None:
                continue
            tag = str(m.group(1))
            kind = str(m.group(2)).lower()
            entry = found.setdefault(tag, {"period": float(tag), "hits": None, "misses": None, "uncovered": None})
            entry[kind] = p

        out = sorted(found.values(), key=lambda d: float(d["period"]))
        return out

    @staticmethod
    def _filter_events_for_periods(events_df: pd.DataFrame) -> pd.DataFrame:
        if len(events_df) == 0 or ("t_mid" not in events_df.columns):
            return pd.DataFrame(columns=["t_mid"])
        t_mid = pd.to_numeric(events_df["t_mid"], errors="coerce")
        ok = np.isfinite(t_mid.to_numpy(dtype=float))
        if not np.any(ok):
            return pd.DataFrame(columns=events_df.columns)
        work = events_df.loc[ok].copy()
        work.loc[:, "t_mid"] = t_mid.loc[ok].to_numpy(dtype=float)
        return work.sort_values("t_mid")

    @staticmethod
    def _match_period_entry_idx(period: float, period_entries: List[Dict[str, Any]], used: set[int]) -> Optional[int]:
        best_idx: Optional[int] = None
        best_gap = float("inf")
        p = float(period)
        tol = max(1e-6, 1e-3 * max(abs(p), 1.0))
        for idx, entry in enumerate(period_entries):
            if idx in used:
                continue
            gap = abs(float(entry["period"]) - p)
            if gap <= tol and gap < best_gap:
                best_gap = gap
                best_idx = idx
        return best_idx

    @staticmethod
    def _summary_row(
        epic: str,
        query: str,
        reason: str,
        n_events_raw: int = 0,
        n_events_after_filters: int = 0,
        P: float = float("nan"),
        cluster_count: int = 0,
        cluster_center_phase: float = float("nan"),
        n_predicted: int = 0,
        n_covered: int = 0,
        coverage_rate: float = float("nan"),
        hit_rate_snr: float = float("nan"),
        hit_rate_shape: float = float("nan"),
        soft_hit_rate: float = float("nan"),
        n_windows_with_no_candidates: int = 0,
    ) -> Dict[str, Any]:
        no_cand = int(n_windows_with_no_candidates)
        return {
            "epic": str(epic),
            "query": str(query),
            "reason": str(reason),
            "n_events_raw": int(n_events_raw),
            "n_events_after_filters": int(n_events_after_filters),
            "P": float(P),
            "cluster_count": int(cluster_count),
            "cluster_center_phase": float(cluster_center_phase),
            "n_predicted": int(n_predicted),
            "n_covered": int(n_covered),
            "coverage_rate": float(coverage_rate),
            "hit_rate_snr": float(hit_rate_snr),
            "hit_rate_shape": float(hit_rate_shape),
            "soft_hit_rate": float(soft_hit_rate),
            "n_windows_with_no_candidates": no_cand,
            "no_cand": no_cand,
        }

    def _annotate_validated_review_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in [
            "P",
            "cluster_count",
            "n_events_after_filters",
            "hit_rate_snr",
            "hit_rate_shape",
            "soft_hit_rate",
        ]:
            out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        reason = out.get("reason", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.strip().str.lower()
        review_cluster = int(getattr(self.config, "MANUAL_REVIEW_CLUSTER_COUNT_EQ", 2))
        shape_floor = getattr(self.config, "CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE", None)
        soft_floor = getattr(self.config, "CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE", None)
        short_period_max = float(getattr(self.config, "CLUSTER2_REVIEW_VERY_SHORT_PERIOD_DAYS_MAX", 1.0))
        low_event_max = float(getattr(self.config, "CLUSTER2_REVIEW_LOW_EVENT_SUPPORT_MAX", 2))
        near_zero_shape_max = float(getattr(self.config, "CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SHAPE_MAX", 0.05))
        near_zero_snr_max = float(getattr(self.config, "CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SNR_MAX", 0.05))

        validated_mask = reason.eq("validated")
        cluster2_any_mask = out["cluster_count"].eq(float(review_cluster))
        cluster2_mask = validated_mask & cluster2_any_mask
        out["manual_review_required"] = cluster2_mask.astype(bool)
        out["manual_review_reason"] = ""
        out.loc[cluster2_mask, "manual_review_reason"] = (
            f"validated_cluster_count=={review_cluster}; supported high-recall mode candidate requires review"
        )
        out["cluster2_watch_very_short_period"] = (cluster2_any_mask & out["P"].le(short_period_max)).astype(bool)
        out["cluster2_watch_low_event_support"] = (cluster2_any_mask & out["n_events_after_filters"].le(low_event_max)).astype(bool)
        out["cluster2_watch_near_zero_hit_rate_shape"] = (cluster2_any_mask & out["hit_rate_shape"].lt(near_zero_shape_max)).astype(bool)
        out["cluster2_watch_near_zero_hit_rate_snr"] = (cluster2_any_mask & out["hit_rate_snr"].lt(near_zero_snr_max)).astype(bool)
        watch_notes = pd.Series([""] * len(out), index=out.index, dtype=str)
        watch_notes.loc[self._to_bool_series(out["cluster2_watch_very_short_period"], default=False)] = (
            watch_notes.loc[self._to_bool_series(out["cluster2_watch_very_short_period"], default=False)] + "very_short_period;"
        )
        watch_notes.loc[self._to_bool_series(out["cluster2_watch_low_event_support"], default=False)] = (
            watch_notes.loc[self._to_bool_series(out["cluster2_watch_low_event_support"], default=False)] + "two_filtered_events_or_fewer;"
        )
        watch_notes.loc[self._to_bool_series(out["cluster2_watch_near_zero_hit_rate_shape"], default=False)] = (
            watch_notes.loc[self._to_bool_series(out["cluster2_watch_near_zero_hit_rate_shape"], default=False)] + "near_zero_hit_rate_shape;"
        )
        watch_notes.loc[self._to_bool_series(out["cluster2_watch_near_zero_hit_rate_snr"], default=False)] = (
            watch_notes.loc[self._to_bool_series(out["cluster2_watch_near_zero_hit_rate_snr"], default=False)] + "near_zero_hit_rate_snr;"
        )
        out["cluster2_watch_notes"] = watch_notes.str.strip(";")
        out["cluster2_guardrail_hit_rate_shape_min"] = float(shape_floor) if shape_floor is not None else np.nan
        out["cluster2_guardrail_soft_hit_rate_min"] = float(soft_floor) if soft_floor is not None else np.nan
        out["cluster2_guardrail_hit_rate_shape_pass"] = True
        out["cluster2_guardrail_soft_hit_rate_pass"] = True
        out["cluster2_guardrail_pass"] = True
        out["cluster2_guardrail_reason"] = ""

        if shape_floor is not None:
            pass_shape = out["hit_rate_shape"].ge(float(shape_floor)) | (~cluster2_mask)
            out["cluster2_guardrail_hit_rate_shape_pass"] = pass_shape.astype(bool)
        if soft_floor is not None:
            pass_soft = out["soft_hit_rate"].ge(float(soft_floor)) | (~cluster2_mask)
            out["cluster2_guardrail_soft_hit_rate_pass"] = pass_soft.astype(bool)

        guardrail_pass = (
            self._to_bool_series(out["cluster2_guardrail_hit_rate_shape_pass"], default=True)
            & self._to_bool_series(out["cluster2_guardrail_soft_hit_rate_pass"], default=True)
        )
        out["cluster2_guardrail_pass"] = (~cluster2_mask) | guardrail_pass

        reasons = pd.Series([""] * len(out), index=out.index, dtype=str)
        if shape_floor is not None:
            fail_shape = cluster2_mask & (~self._to_bool_series(out["cluster2_guardrail_hit_rate_shape_pass"], default=True))
            reasons.loc[fail_shape] = reasons.loc[fail_shape] + (
                f"hit_rate_shape<{float(shape_floor):.3f};"
            )
        if soft_floor is not None:
            fail_soft = cluster2_mask & (~self._to_bool_series(out["cluster2_guardrail_soft_hit_rate_pass"], default=True))
            reasons.loc[fail_soft] = reasons.loc[fail_soft] + (
                f"soft_hit_rate<{float(soft_floor):.3f};"
            )
        out["cluster2_guardrail_reason"] = reasons.str.strip(";")
        return out

    def _apply_cluster2_validated_guardrails(
        self,
        df_summary_valid: pd.DataFrame,
        quarantine_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        annotated = self._annotate_validated_review_flags(df_summary_valid)
        fail_mask = (
            self._to_bool_series(annotated.get("manual_review_required", pd.Series([False] * len(annotated), index=annotated.index)))
            & (~self._to_bool_series(annotated.get("cluster2_guardrail_pass", pd.Series([True] * len(annotated), index=annotated.index)), default=True))
        )
        if not bool(fail_mask.any()):
            return annotated, quarantine_df.copy(), 0

        kept = annotated.loc[~fail_mask].copy()
        rejected = annotated.loc[fail_mask].copy()
        q_add = pd.DataFrame(
            {
                "epic_id": rejected.get("epic", "").fillna("").astype(str),
                "query": rejected.get("query", "").fillna("").astype(str),
                "reason": "validated_guardrail_reject",
                "missing_upstream_source": "",
                "source_reason": "cluster2_guardrail_rejection",
                "whiteness_missing": False,
                "whiteness_null_reason_category": "",
                "shortlist_rejection_stage": "validated_guardrail",
                "shortlist_rejection_reason": "cluster2_guardrail_rejection",
                "rejected_before_candidate_scoring": False,
                "rejected_after_candidate_scoring": True,
                "failure_category": "cluster2_guardrail_rejection",
                "failure_detail": rejected.get("cluster2_guardrail_reason", "").fillna("").astype(str),
                "n_events_raw": pd.to_numeric(rejected.get("n_events_raw", np.nan), errors="coerce"),
                "n_events_after_filters": pd.to_numeric(rejected.get("n_events_after_filters", np.nan), errors="coerce"),
                "infer_max_period_days": self._effective_max_period_days(),
                "infer_min_hits": 1.0,
                "infer_tol_frac": 0.01,
                "min_cluster_count": float(getattr(self.config, "MIN_CLUSTER_COUNT", 3)),
                "period_cap_days": self._effective_max_period_days(),
                "min_period_days": self._as_float(getattr(self.config, "MIN_PERIOD_DAYS", 0.5), default=0.5),
                "top_k_periods": float(getattr(self.config, "TOP_K_PERIODS", getattr(self.config, "VALIDATION_TOP_K", 3))),
                "hist_total": np.nan,
                "hist_finite_period": np.nan,
                "hist_in_period_range": np.nan,
                "hist_pass_cluster_count": np.nan,
                "hist_pass_all_filters": np.nan,
                "P": pd.to_numeric(rejected.get("P", np.nan), errors="coerce"),
            }
        ).reindex(columns=self.QUARANTINE_COLUMNS)

        combined_quarantine = pd.concat([quarantine_df.copy(), q_add], ignore_index=True)
        return kept, combined_quarantine, int(len(q_add))

    @staticmethod
    def _filter_candidate_period_rows(
        hist_df: pd.DataFrame,
        min_period_days: float,
        max_period_days: float,
        min_cluster_count: int,
        top_k: int,
    ) -> pd.DataFrame:
        if len(hist_df) == 0:
            return pd.DataFrame(columns=["period", "count_hits", "pair_count"])

        work = hist_df.copy()
        for c in ["period", "count_hits", "pair_count"]:
            if c not in work.columns:
                work[c] = np.nan
            work[c] = pd.to_numeric(work[c], errors="coerce")

        work = work[
            work["period"].notna()
            & work["count_hits"].notna()
            & (work["period"] >= float(min_period_days))
            & (work["period"] <= float(max_period_days))
            & (work["count_hits"] >= int(min_cluster_count))
        ].copy()
        if len(work) == 0:
            return work

        work = work.sort_values(["count_hits", "pair_count", "period"], ascending=[False, False, True])
        return work.head(max(0, int(top_k))).reset_index(drop=True)

    @staticmethod
    def _nearest_cluster_row(period: float, cluster_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(cluster_rows) == 0:
            return {}
        p = float(period)
        best_row = cluster_rows[0]
        best_gap = abs(float(best_row.get("P", float("nan"))) - p) if np.isfinite(float(best_row.get("P", float("nan")))) else float("inf")
        for row in cluster_rows[1:]:
            rp = float(row.get("P", float("nan")))
            gap = abs(rp - p) if np.isfinite(rp) else float("inf")
            if gap < best_gap:
                best_gap = gap
                best_row = row
        return best_row

    def _validate_top_periods_from_cache(
        self,
        query: str,
        events_df: pd.DataFrame,
        cluster_rows: List[Dict[str, Any]],
        top_periods: List[float],
        handler: K2_NoiseHandler,
        validator: K2Validation_Prediction,
        snr: K2SNR,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        cfg = self.config
        stats = {"cache_hits": 0, "cache_misses": 0, "downloads_done": 0, "validations_run": 0}
        cache_only_first = bool(getattr(cfg, "CACHE_ONLY_FIRST", True))

        fetched: Dict[str, Any]
        if cache_only_first:
            try:
                fetched = handler.fetch_best(query=str(query), cache_only=True)
            except K2PipelineStageError as exc:
                print(f"[K2ShortlistPeriodRunner] validation fetch failed query={query} stage={exc.stage}: {exc.error_msg}")
                return "cluster_only_validation_error", [], stats
            except Exception as exc:
                print(f"[K2ShortlistPeriodRunner] validation fetch failed query={query}: {type(exc).__name__}: {exc}")
                return "cluster_only_validation_error", [], stats
        else:
            try:
                fetched = handler.fetch_best(query=str(query), cache_only=False)
            except K2PipelineStageError as exc:
                print(f"[K2ShortlistPeriodRunner] validation download failed query={query} stage={exc.stage}: {exc.error_msg}")
                return "cluster_only_validation_error", [], stats
            except Exception as exc:
                print(f"[K2ShortlistPeriodRunner] validation download failed query={query}: {type(exc).__name__}: {exc}")
                return "cluster_only_validation_error", [], stats
            if str(fetched.get("status", "error")).lower() == "ok":
                stats["downloads_done"] += 1

        status = str(fetched.get("status", "error")).lower()
        if status == "ok":
            if cache_only_first:
                stats["cache_hits"] += 1
        elif status == "cache_miss":
            stats["cache_misses"] += 1
            if bool(getattr(cfg, "DOWNLOAD_IF_CACHE_MISS", True)):
                try:
                    fetched = handler.fetch_best(query=str(query), cache_only=False)
                except K2PipelineStageError as exc:
                    print(
                        f"[K2ShortlistPeriodRunner] validation download failed query={query} "
                        f"stage={exc.stage}: {exc.error_msg}"
                    )
                    return "cluster_only_cache_miss", [], stats
                except Exception as exc:
                    print(
                        f"[K2ShortlistPeriodRunner] validation download failed query={query}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return "cluster_only_cache_miss", [], stats

                if str(fetched.get("status", "error")).lower() == "ok":
                    stats["downloads_done"] += 1
                else:
                    return "cluster_only_cache_miss", [], stats
            else:
                return "cluster_only_cache_miss", [], stats
        else:
            return "cluster_only_validation_error", [], stats

        try:
            cleaned = handler.clean(
                fetched["lc"],
                normalize=False,
                remove_nans=True,
                quality_mask=True,
                sigma_clip=False,
                flatten=False,
            )
            t = np.asarray(cleaned["time"], dtype=float)
            f = np.asarray(cleaned["flux"], dtype=float)
            norm = snr.normalize(time=t, flux=f)
            resid = np.asarray(norm["resid"], dtype=float)
            local_sigma = np.asarray(norm["local_sigma"], dtype=float)
        except Exception as exc:
            print(f"[K2ShortlistPeriodRunner] validation preprocess failed query={query}: {type(exc).__name__}: {exc}")
            return "cluster_only_validation_error", [], stats

        event_times = pd.to_numeric(events_df.get("t_mid", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        event_times = event_times[np.isfinite(event_times)]
        tol_days = 0.08
        validated_rows: List[Dict[str, Any]] = []

        for p in top_periods:
            base = self._nearest_cluster_row(period=float(p), cluster_rows=cluster_rows)
            if len(base) == 0:
                continue
            out = validator.validate_period_by_prediction(
                time=t,
                resid=resid,
                local_sigma=local_sigma,
                events_df=events_df,
                P=float(p),
                tol_days=float(tol_days),
                do_plot=False,
            )
            stats["validations_run"] += 1
            n_predicted = int(out.get("n_predicted", 0))
            if n_predicted <= 0:
                raise RuntimeError(
                    f"[K2ShortlistPeriodRunner] validation bug: n_predicted<=0 for query={query} P={float(p):.8f}"
                )
            n_covered = int(out.get("n_covered", 0))
            coverage_rate = self._as_float(out.get("coverage_rate"))
            hit_rate_snr = self._as_float(out.get("hit_rate"))

            rows = list(out.get("rows", []))
            covered_mask = list(out.get("covered_mask", []))
            covered_snrs: List[float] = []
            covered_has_candidate: List[bool] = []
            for idx, row in enumerate(rows):
                try:
                    tk = float(row[0])
                    dip_snr = float(row[1])
                except Exception:
                    continue
                covered = bool(covered_mask[idx]) if idx < len(covered_mask) else np.isfinite(dip_snr)
                if not covered:
                    continue
                if np.isfinite(dip_snr):
                    covered_snrs.append(dip_snr)
                has_candidate = bool(np.any(np.abs(event_times - tk) <= float(tol_days))) if len(event_times) > 0 else False
                covered_has_candidate.append(has_candidate)

            soft_hit_rate = float(np.mean(np.asarray(covered_snrs, dtype=float) >= float(cfg.SOFT_SNR_T))) if len(covered_snrs) > 0 else float("nan")
            n_no_cand = int(np.sum(~np.asarray(covered_has_candidate, dtype=bool))) if len(covered_has_candidate) > 0 else int(max(0, n_covered))
            hit_rate_shape = float(np.mean(np.asarray(covered_has_candidate, dtype=bool))) if len(covered_has_candidate) > 0 else float("nan")

            validated_rows.append(
                self._summary_row(
                    epic=str(base.get("epic", "")),
                    query=str(base.get("query", query)),
                    reason="validated",
                    n_events_raw=int(base.get("n_events_raw", 0)),
                    n_events_after_filters=int(base.get("n_events_after_filters", 0)),
                    P=float(base.get("P", p)),
                    cluster_count=int(base.get("cluster_count", 0)),
                    cluster_center_phase=self._as_float(base.get("cluster_center_phase")),
                    n_predicted=n_predicted,
                    n_covered=n_covered,
                    coverage_rate=coverage_rate,
                    hit_rate_snr=hit_rate_snr,
                    hit_rate_shape=hit_rate_shape,
                    soft_hit_rate=soft_hit_rate,
                    n_windows_with_no_candidates=n_no_cand,
                )
            )

        return "cluster_only", validated_rows, stats

    def _summarize_period(
        self,
        epic: str,
        query: str,
        events_df: pd.DataFrame,
        period: float,
        hits_path: Optional[Path],
        misses_path: Optional[Path],
        uncovered_path: Optional[Path],
    ) -> Dict[str, Any]:
        hits_df = self._read_csv(hits_path) if hits_path is not None else pd.DataFrame()
        misses_df = self._read_csv(misses_path) if misses_path is not None else pd.DataFrame()
        uncovered_df = self._read_csv(uncovered_path) if uncovered_path is not None else pd.DataFrame()

        cluster_count, cluster_center_phase = self._phase_cluster_score_quiet(
            events_df=events_df,
            period=float(period),
            tol_phase=float(self.config.PERIOD_TOL_PHASE),
        )

        n_predicted = int(len(hits_df) + len(misses_df) + len(uncovered_df))
        n_covered = int(len(hits_df) + len(misses_df))
        coverage_rate = float(n_covered / n_predicted) if n_predicted > 0 else float("nan")

        if len(hits_df) > 0:
            hit_shape_hits = self._to_bool_series(hits_df.get("hit_shape", pd.Series([True] * len(hits_df))), default=True)
            hit_snr_hits = self._to_bool_series(hits_df.get("hit_snr", pd.Series([True] * len(hits_df))), default=True)
        else:
            hit_shape_hits = pd.Series([], dtype=bool)
            hit_snr_hits = pd.Series([], dtype=bool)

        if len(misses_df) > 0:
            hit_shape_miss = self._to_bool_series(misses_df.get("hit_shape", pd.Series([False] * len(misses_df))), default=False)
            hit_snr_miss = self._to_bool_series(misses_df.get("hit_snr", pd.Series([False] * len(misses_df))), default=False)
        else:
            hit_shape_miss = pd.Series([], dtype=bool)
            hit_snr_miss = pd.Series([], dtype=bool)

        hit_shape_all = pd.concat([hit_shape_hits, hit_shape_miss], ignore_index=True)
        hit_snr_all = pd.concat([hit_snr_hits, hit_snr_miss], ignore_index=True)
        hit_rate_shape = float(hit_shape_all.mean()) if len(hit_shape_all) > 0 else float("nan")
        hit_rate_snr = float(hit_snr_all.mean()) if len(hit_snr_all) > 0 else float("nan")

        has_candidate = self._to_bool_series(misses_df.get("has_candidate", pd.Series([True] * len(misses_df))), default=True)
        n_windows_with_no_candidates = int((~has_candidate).sum()) if len(has_candidate) > 0 else 0

        covered = pd.concat([hits_df, misses_df], ignore_index=True)
        dip_snr = pd.to_numeric(covered.get("dip_snr_at_min", np.nan), errors="coerce")
        run_len_at_min = pd.to_numeric(covered.get("duration_below_threshold", np.nan), errors="coerce")
        soft_hit = (dip_snr >= float(self.config.SOFT_SNR_T)) & (run_len_at_min >= int(self.config.SOFT_MIN_RUN))
        soft_hit_rate = float(soft_hit.mean()) if len(covered) > 0 else float("nan")

        return {
            "epic": str(epic),
            "query": str(query),
            "P": float(period),
            "cluster_count": int(cluster_count),
            "cluster_center_phase": float(cluster_center_phase),
            "n_predicted": int(n_predicted),
            "n_covered": int(n_covered),
            "coverage_rate": float(coverage_rate),
            "hit_rate_snr": float(hit_rate_snr),
            "hit_rate_shape": float(hit_rate_shape),
            "soft_hit_rate": float(soft_hit_rate),
            "n_windows_with_no_candidates": int(n_windows_with_no_candidates),
        }

    @staticmethod
    def _append_rows(csv_path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
        if len(rows) == 0:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).reindex(columns=list(columns))
        write_header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        rows.clear()

    @staticmethod
    def _select_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        work = pd.DataFrame(rows)
        for c in ["soft_hit_rate", "hit_rate_snr", "hit_rate_shape"]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.sort_values(["soft_hit_rate", "hit_rate_snr", "hit_rate_shape"], ascending=[False, False, False])
        return work.iloc[0].to_dict()

    def _effective_max_period_days(self) -> float:
        cfg = self.config
        cap = self._as_float(getattr(cfg, "PERIOD_CAP_DAYS", 20.0), default=20.0)
        if (not np.isfinite(cap)) or (cap <= 0):
            cap = 20.0
        return float(cap)

    def _period_bin_edges(self) -> List[float]:
        cfg = self.config
        configured_edges = list(getattr(cfg, "PERIOD_BIN_EDGES_DAYS", (1.0, 5.0, 10.0, 15.0, 20.0)))
        edges = sorted({float(x) for x in configured_edges if np.isfinite(float(x)) and float(x) >= 0})
        max_period = float(self._effective_max_period_days())
        if len(edges) == 0:
            edges = [0.0, 1.0, 5.0, 10.0, 15.0, max_period]
        if edges[0] > 0.0:
            edges = [0.0] + edges
        edges = [e for e in edges if e < max_period] + [max_period]
        edges = sorted({float(x) for x in edges if np.isfinite(float(x)) and float(x) >= 0})
        if len(edges) < 2:
            low = 0.0 if max_period > 0 else max(1e-6, max_period - 1.0)
            edges = [low, max_period]
        return edges

    def _best_selection_bin_mode(self) -> str:
        mode = str(getattr(self.config, "BEST_SELECTION_BIN_MODE", "match_summary_distribution")).strip().lower()
        if mode not in {"match_summary_distribution", "equal_per_bin"}:
            mode = "match_summary_distribution"
        return mode

    @staticmethod
    def _missing_upstream_source(source_reason: str) -> str:
        key = str(source_reason).strip().lower()
        mapping = {
            "cannot_parse_epic": "shortlist.query",
            "missing_events_csv": "epics/<EPIC>/events.csv",
            "events_filtered_to_zero": "events.csv:t_mid",
            "no_cluster_periods": "infer_periods_from_events(events_df)",
            "cluster_only_no_valid_period": "candidate_period_filter_or_validation",
            "cluster_only_cache_miss": "noise_handler_cache_or_download",
            "cluster_only_validation_error": "period_validation_pipeline",
        }
        return mapping.get(key, "")

    @staticmethod
    def _source_reason_from_missing_upstream_source(missing_upstream_source: str) -> str:
        key = str(missing_upstream_source).strip().lower()
        reverse_mapping = {
            "shortlist.query": "cannot_parse_epic",
            "epics/<epic>/events.csv": "missing_events_csv",
            "events.csv:t_mid": "events_filtered_to_zero",
            "infer_periods_from_events(events_df)": "no_cluster_periods",
            "candidate_period_filter_or_validation": "cluster_only_no_valid_period",
            "noise_handler_cache_or_download": "cluster_only_cache_miss",
            "period_validation_pipeline": "cluster_only_validation_error",
        }
        return reverse_mapping.get(key, "")

    @staticmethod
    def _canonical_epic_id(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return ""
        m = re.search(r"\d+", text)
        return m.group(0) if m is not None else text

    @staticmethod
    def _whiteness_null_reason_category(
        *,
        triage_status: pd.Series,
        triage_why_not_usable: pd.Series,
        error_stage: pd.Series,
        error_type: pd.Series,
        error_msg: pd.Series,
        whiteness_is_null: pd.Series,
    ) -> pd.Series:
        out = pd.Series([""] * len(triage_status), index=triage_status.index, dtype=str)
        why = triage_why_not_usable.fillna("").astype(str).str.strip().str.lower()
        has_error_detail = (
            error_stage.fillna("").astype(str).str.strip().ne("")
            | error_type.fillna("").astype(str).str.strip().ne("")
            | error_msg.fillna("").astype(str).str.strip().ne("")
        )
        out.loc[whiteness_is_null & (triage_status == "error")] = "upstream_error"
        out.loc[whiteness_is_null & has_error_detail & out.eq("")] = "upstream_error"
        out.loc[
            whiteness_is_null
            & why.str.contains("n_points<|baseline_days<|robust_sigma<|outlier_rate|all_flux_nan|insufficient", regex=True)
            & out.eq("")
        ] = "noncomputable_quality_metrics"
        out.loc[whiteness_is_null & out.eq("")] = "missing_or_noncomputable_whiteness"
        return out

    def _load_raw_epic_table(self, shortlist_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        raw_csv = cfg.raw_epic_list_csv_path
        query_col_cfg = str(getattr(cfg, "RAW_EPIC_QUERY_COL", "query"))

        if not raw_csv.exists():
            raise FileNotFoundError(
                f"[K2ShortlistPeriodRunner] raw EPIC list not found at {raw_csv}. "
                "Run `python main.py k2_whiteness` first to precompute whiteness fields."
            )
        raw = pd.read_csv(raw_csv)
        source = str(raw_csv)

        missing = [c for c in self.RAW_EPIC_REQUIRED_COLUMNS if c not in raw.columns]
        if len(missing) > 0:
            raise ValueError(
                f"[K2ShortlistPeriodRunner] raw EPIC list missing required columns: {missing}. "
                f"input={raw_csv}"
            )
        if not any(c in raw.columns for c in self.RAW_EPIC_REQUIRED_WHITENESS_VALUE_COLUMNS):
            raise ValueError(
                "[K2ShortlistPeriodRunner] raw EPIC list must contain one whiteness value column: "
                f"{list(self.RAW_EPIC_REQUIRED_WHITENESS_VALUE_COLUMNS)}. input={raw_csv}"
            )

        query_col = query_col_cfg if query_col_cfg in raw.columns else ("query" if "query" in raw.columns else "")
        epic_col = ""
        for c in ["epic_id", "epic", "EPIC ID"]:
            if c in raw.columns:
                epic_col = c
                break

        out = pd.DataFrame(index=raw.index)
        if epic_col != "":
            out["epic_id"] = raw[epic_col].map(self._canonical_epic_id)
        elif query_col != "":
            out["epic_id"] = raw[query_col].map(self._extract_epic).fillna("").astype(str)
        else:
            out["epic_id"] = ""

        if query_col != "":
            out["query"] = raw[query_col].fillna("").astype(str)
        else:
            out["query"] = out["epic_id"].map(lambda x: f"EPIC {x}" if x != "" else "")

        for c in [
            "triage_status",
            "triage_why_not_usable",
            "triage_usable",
            "triage_score_global",
            "triage_n_points",
            "triage_whiteness_pvalue",
            "triage_whiteness_score",
            "triage_whiteness_definition",
            "triage_whiteness_interpretation",
            "triage_whiteness_higher_is_better",
            "n_events",
            "n_periods_proposed",
            "n_periods_validated",
            "best_shape_score",
            "best_depth_snr",
            "error_stage",
            "error_type",
            "error_msg",
            "campaign_selected",
            "epic_dir",
        ]:
            out[c] = raw[c] if c in raw.columns else np.nan

        out["epic_id"] = out["epic_id"].fillna("").astype(str).str.strip()
        out = out.loc[out["epic_id"] != ""].copy()
        out = out.drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)

        whiteness_col = "triage_whiteness_pvalue" if "triage_whiteness_pvalue" in out.columns else "triage_whiteness_score"
        w = pd.to_numeric(out.get(whiteness_col, np.nan), errors="coerce")
        null_mask = w.isna()
        out["triage_whiteness_value"] = w
        out["triage_whiteness_is_null"] = null_mask
        out["whiteness_missing"] = null_mask.astype(bool)
        out["triage_usable"] = self._to_bool_series(
            out.get("triage_usable", pd.Series([False] * len(out), index=out.index))
        )
        triage_status_series = out.get("triage_status", "").fillna("").astype(str).str.strip().str.lower()
        why_not_series = out.get("triage_why_not_usable", "").fillna("").astype(str)
        error_stage_series = out.get("error_stage", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        error_type_series = out.get("error_type", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        error_msg_series = out.get("error_msg", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
        out["whiteness_null_reason_category"] = self._whiteness_null_reason_category(
            triage_status=triage_status_series,
            triage_why_not_usable=why_not_series,
            error_stage=error_stage_series,
            error_type=error_type_series,
            error_msg=error_msg_series,
            whiteness_is_null=null_mask,
        )
        out["shortlist_rejection_stage"] = ""
        out["shortlist_rejection_reason"] = ""
        out["rejected_before_candidate_scoring"] = False
        out["rejected_after_candidate_scoring"] = False
        null_unusable_mask = null_mask & (~out["triage_usable"].astype(bool))
        out.loc[null_unusable_mask, "shortlist_rejection_stage"] = "pre_candidate_scoring"
        out.loc[null_unusable_mask, "shortlist_rejection_reason"] = "whiteness_null_and_triage_unusable"
        out.loc[null_unusable_mask, "rejected_before_candidate_scoring"] = True
        q_nonempty = out.get("query", "").fillna("").astype(str).str.strip() != ""
        shortlist_attempt_mask = q_nonempty & out["epic_id"].ne("")
        usable_series = self._to_bool_series(out.get("triage_usable", pd.Series([False] * len(out), index=out.index)))
        null_usable_true = int((null_mask & usable_series).sum())
        null_shortlist_attempt = int((null_mask & shortlist_attempt_mask).sum())
        null_why = (
            out.loc[null_mask, "triage_why_not_usable"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<empty>")
            .value_counts()
            .head(10)
            .to_dict()
        )

        print(
            f"[K2ShortlistPeriodRunner] raw_epic_source={source} "
            f"n_total_epics={len(out)}"
        )
        print(
            f"[K2ShortlistPeriodRunner] whiteness_null_policy=reject_if_null_and_triage_unusable "
            f"whiteness_col={whiteness_col} "
            f"null_whiteness_rows={int(null_mask.sum())} "
            f"null_whiteness_usable_true={null_usable_true} "
            f"null_whiteness_shortlist_attempt={null_shortlist_attempt}"
        )
        if len(null_why) > 0:
            why_text = " | ".join([f"{k}:{int(v)}" for k, v in null_why.items()])
            print(f"[K2ShortlistPeriodRunner] null_whiteness_top_why_not_usable: {why_text}")
        return out

    def _period_stage_selection_mode(self) -> str:
        cfg = self.config
        raw_mode = getattr(cfg, "PERIOD_STAGE_SELECTION_MODE", "topk")
        mode = str(raw_mode).strip().lower()
        aliases = {
            "topk": "topk",
            "top_k": "topk",
            "random": "randomN",
            "randomn": "randomN",
            "all": "all",
        }
        resolved = aliases.get(mode, "")
        if resolved == "":
            raise ValueError(
                f"Unsupported PERIOD_STAGE_SELECTION_MODE={raw_mode!r}; "
                "expected one of {'topk','randomN','all'}."
            )
        return resolved

    def _rank_raw_epics_for_period_stage(self, raw_epics_df: pd.DataFrame) -> pd.DataFrame:
        work = raw_epics_df.copy()
        for c in ["query", "epic_id", "triage_status"]:
            if c not in work.columns:
                work[c] = ""
        for c in ["n_events", "best_shape_score", "best_depth_snr"]:
            if c not in work.columns:
                work[c] = np.nan

        work["query"] = work["query"].fillna("").astype(str)
        work["epic_id"] = work["epic_id"].fillna("").astype(str).str.strip()
        work["triage_status"] = work["triage_status"].fillna("").astype(str).str.strip().str.lower()
        work["n_events"] = pd.to_numeric(work["n_events"], errors="coerce")
        work["best_shape_score"] = pd.to_numeric(work["best_shape_score"], errors="coerce")
        work["best_depth_snr"] = pd.to_numeric(work["best_depth_snr"], errors="coerce")
        rejection_reason = work.get("shortlist_rejection_reason", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str).str.strip()
        eligible_mask = rejection_reason.eq("")

        ranked = (
            work.loc[eligible_mask & (work["triage_status"] == "ok") & (work["n_events"] > 0) & (work["query"] != "")]
            .sort_values(["best_shape_score", "best_depth_snr"], ascending=[False, False], kind="mergesort")
            .drop_duplicates(subset=["epic_id"], keep="first")
        )
        return ranked.reset_index(drop=True)

    def _select_period_stage_queries(
        self,
        raw_epics_df: pd.DataFrame,
        shortlist_df: pd.DataFrame,
    ) -> Tuple[List[str], Dict[str, Any]]:
        cfg = self.config
        mode = self._period_stage_selection_mode()
        selected_df = pd.DataFrame(columns=["query", "epic_id"])
        period_stage_k = getattr(cfg, "PERIOD_STAGE_K", None)
        period_stage_n = getattr(cfg, "PERIOD_STAGE_N", None)
        seed = int(getattr(cfg, "PERIOD_STAGE_RANDOM_SEED", 42))
        selection_meta: Dict[str, Any] = {
            "period_stage_selection_mode": mode,
            "period_stage_k": period_stage_k,
            "period_stage_n": period_stage_n,
            "period_stage_random_seed": seed,
            "ranking_basis": "best_shape_score desc, best_depth_snr desc",
            "n_ranked_candidates": 0,
            "n_population_for_gate": 0,
            "n_population_before_shortlist_precheck": 0,
            "n_excluded_by_shortlist_precheck": 0,
            "n_selected_for_period_stage": 0,
            "n_enter_period_stage_pre_slice": 0,
            "n_excluded_by_topk_gate": 0,
            "n_excluded_by_gate": 0,
        }

        if mode == "all":
            base_all = raw_epics_df.copy()
            base_all["query"] = base_all.get("query", "").fillna("").astype(str)
            base_all["epic_id"] = base_all.get("epic_id", "").fillna("").astype(str).str.strip()
            base_all["shortlist_rejection_reason"] = base_all.get(
                "shortlist_rejection_reason",
                pd.Series([""] * len(base_all), index=base_all.index),
            ).fillna("").astype(str).str.strip()
            base_all = base_all.loc[base_all["query"] != ""].drop_duplicates(subset=["epic_id"], keep="first")
            selected_df = base_all.loc[base_all["shortlist_rejection_reason"].eq("")].copy()
            selection_meta["n_population_before_shortlist_precheck"] = int(len(base_all))
            selection_meta["n_excluded_by_shortlist_precheck"] = int(len(base_all) - len(selected_df))
            selection_meta["n_population_for_gate"] = int(len(selected_df))
        elif mode == "randomN":
            if period_stage_n is None:
                raise ValueError("PERIOD_STAGE_SELECTION_MODE='randomN' requires PERIOD_STAGE_N to be set.")
            base_all = raw_epics_df.copy()
            base_all["query"] = base_all.get("query", "").fillna("").astype(str)
            base_all["epic_id"] = base_all.get("epic_id", "").fillna("").astype(str).str.strip()
            base_all["shortlist_rejection_reason"] = base_all.get(
                "shortlist_rejection_reason",
                pd.Series([""] * len(base_all), index=base_all.index),
            ).fillna("").astype(str).str.strip()
            base_all = base_all.loc[base_all["query"] != ""].drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)
            base = base_all.loc[base_all["shortlist_rejection_reason"].eq("")].copy().reset_index(drop=True)
            selection_meta["n_population_before_shortlist_precheck"] = int(len(base_all))
            selection_meta["n_excluded_by_shortlist_precheck"] = int(len(base_all) - len(base))
            selection_meta["n_population_for_gate"] = int(len(base))
            sample_n = max(0, int(period_stage_n))
            sample_n = min(sample_n, len(base))
            if sample_n >= len(base):
                selected_df = base
            else:
                rng = np.random.default_rng(seed)
                pick = np.sort(rng.choice(len(base), size=sample_n, replace=False))
                selected_df = base.iloc[pick].copy()
        else:
            if period_stage_k is None:
                raise ValueError("PERIOD_STAGE_SELECTION_MODE='topk' requires PERIOD_STAGE_K to be set.")
            population_before = raw_epics_df.copy()
            population_before["query"] = population_before.get("query", "").fillna("").astype(str)
            population_before["epic_id"] = population_before.get("epic_id", "").fillna("").astype(str).str.strip()
            population_before["shortlist_rejection_reason"] = population_before.get(
                "shortlist_rejection_reason",
                pd.Series([""] * len(population_before), index=population_before.index),
            ).fillna("").astype(str).str.strip()
            population_before = population_before.loc[population_before["query"] != ""].drop_duplicates(subset=["epic_id"], keep="first")
            selection_meta["n_population_before_shortlist_precheck"] = int(len(population_before))
            selection_meta["n_excluded_by_shortlist_precheck"] = int(
                len(population_before.loc[population_before["shortlist_rejection_reason"] != ""])
            )
            ranked = self._rank_raw_epics_for_period_stage(raw_epics_df=raw_epics_df)
            selection_meta["n_ranked_candidates"] = int(len(ranked))
            selection_meta["n_population_for_gate"] = int(len(ranked))
            topk = max(0, int(period_stage_k))
            if topk <= 0:
                selected_df = ranked.iloc[0:0].copy()
            else:
                selected_df = ranked.head(topk).copy()
            selection_meta["n_excluded_by_topk_gate"] = max(0, int(len(ranked) - len(selected_df)))

        selected_df = selected_df.copy()
        selected_df["query"] = selected_df.get("query", "").fillna("").astype(str)
        selected_df["epic_id"] = selected_df.get("epic_id", "").fillna("").astype(str).str.strip()
        selected_df = selected_df.loc[selected_df["query"] != ""].drop_duplicates(subset=["epic_id"], keep="first").reset_index(drop=True)
        selection_meta["n_selected_for_period_stage"] = int(len(selected_df))
        selection_meta["n_enter_period_stage_pre_slice"] = int(len(selected_df))
        selection_meta["n_excluded_by_gate"] = int(max(0, int(selection_meta.get("n_population_for_gate", 0)) - len(selected_df)))

        queries = selected_df["query"].tolist()
        start = max(0, int(cfg.START_INDEX))
        end = len(queries) - 1 if cfg.END_INDEX is None else min(int(cfg.END_INDEX), len(queries) - 1)
        selected = queries[start : end + 1] if end >= start else []
        if cfg.MAX_TARGETS is not None:
            selected = selected[: max(0, int(cfg.MAX_TARGETS))]
        selection_meta["n_enter_period_stage"] = int(len(selected))
        selection_meta["n_excluded_by_slice"] = int(max(0, len(queries) - len(selected)))
        if mode != "topk":
            selection_meta["n_excluded_by_topk_gate"] = 0
        return selected, selection_meta

    def _build_epic_funnel_and_reasons(
        self,
        raw_epics_df: pd.DataFrame,
        selected_queries: Sequence[str],
        selection_meta: Dict[str, Any],
        df_summary_raw: pd.DataFrame,
        df_summary_valid: pd.DataFrame,
        df_summary_unique: pd.DataFrame,
        df_summary_validated_only: pd.DataFrame,
        best_df: pd.DataFrame,
        quarantine_df: pd.DataFrame,
    ) -> Tuple[Dict[str, int], pd.DataFrame]:
        selected_epics = {
            str(self._extract_epic(q) or "").strip()
            for q in selected_queries
            if str(self._extract_epic(q) or "").strip() != ""
        }

        reasons = raw_epics_df.copy()
        reasons["epic_id"] = reasons["epic_id"].fillna("").astype(str).str.strip()
        reasons["query"] = reasons.get(
            "query", pd.Series([""] * len(reasons), index=reasons.index)
        ).fillna("").astype(str)
        for c in [
            "triage_status",
            "triage_why_not_usable",
            "n_events",
            "n_periods_proposed",
            "n_periods_validated",
            "whiteness_missing",
            "whiteness_null_reason_category",
            "shortlist_rejection_stage",
            "shortlist_rejection_reason",
            "rejected_before_candidate_scoring",
            "rejected_after_candidate_scoring",
        ]:
            if c not in reasons.columns:
                reasons[c] = np.nan
        reasons["selected_for_period_stage"] = reasons["epic_id"].isin(selected_epics)
        reasons["terminal_reason"] = "other"
        reasons["source_reason"] = ""
        reasons["no_events_breakdown"] = ""
        reasons["load_failed_exception_type"] = ""
        reasons["load_failed_exception_message"] = ""
        reasons["load_failed_campaign"] = ""
        reasons["load_failed_source"] = ""
        reasons["no_events_n_points_after_clean"] = np.nan
        reasons["no_events_baseline_days"] = np.nan
        reasons["no_events_thresholds_used"] = ""
        reasons["period_failure_category"] = ""
        reasons["period_failure_detail"] = ""
        reasons["period_n_events_raw"] = np.nan
        reasons["period_n_events_after_filters"] = np.nan
        reasons["period_infer_max_period_days"] = np.nan
        reasons["period_infer_min_hits"] = np.nan
        reasons["period_infer_tol_frac"] = np.nan
        reasons["period_min_cluster_count"] = np.nan
        reasons["period_cap_days"] = np.nan
        reasons["period_min_period_days"] = np.nan
        reasons["period_top_k_periods"] = np.nan
        reasons["period_hist_total"] = np.nan
        reasons["period_hist_finite_period"] = np.nan
        reasons["period_hist_in_period_range"] = np.nan
        reasons["period_hist_pass_cluster_count"] = np.nan
        reasons["period_hist_pass_all_filters"] = np.nan
        reasons["whiteness_missing"] = self._to_bool_series(
            reasons.get("whiteness_missing", pd.Series([False] * len(reasons), index=reasons.index))
        )
        reasons["whiteness_null_reason_category"] = reasons.get(
            "whiteness_null_reason_category", pd.Series([""] * len(reasons), index=reasons.index)
        ).fillna("").astype(str)
        reasons["shortlist_rejection_stage"] = reasons.get(
            "shortlist_rejection_stage", pd.Series([""] * len(reasons), index=reasons.index)
        ).fillna("").astype(str)
        reasons["shortlist_rejection_reason"] = reasons.get(
            "shortlist_rejection_reason", pd.Series([""] * len(reasons), index=reasons.index)
        ).fillna("").astype(str)
        reasons["rejected_before_candidate_scoring"] = self._to_bool_series(
            reasons.get("rejected_before_candidate_scoring", pd.Series([False] * len(reasons), index=reasons.index))
        )
        reasons["rejected_after_candidate_scoring"] = self._to_bool_series(
            reasons.get("rejected_after_candidate_scoring", pd.Series([False] * len(reasons), index=reasons.index))
        )
        empty_s = pd.Series([""] * len(reasons), index=reasons.index, dtype=str)

        status = reasons.get("triage_status", "").fillna("").astype(str).str.strip().str.lower()
        why_not = reasons.get("triage_why_not_usable", "").fillna("").astype(str).str.strip().str.lower()
        n_events = pd.to_numeric(reasons.get("n_events", np.nan), errors="coerce")
        error_type = reasons.get("error_type", empty_s).fillna("").astype(str).str.strip()
        error_msg = reasons.get("error_msg", empty_s).fillna("").astype(str).str.strip()
        error_stage = reasons.get("error_stage", empty_s).fillna("").astype(str).str.strip()
        campaign = reasons.get("campaign_selected", empty_s).fillna("").astype(str).str.strip()
        triage_n_points = pd.to_numeric(reasons.get("triage_n_points", pd.Series([np.nan] * len(reasons), index=reasons.index)), errors="coerce")
        whiteness_def = reasons.get("triage_whiteness_definition", empty_s).fillna("").astype(str).str.strip()

        mask_load_failed = status.eq("error") | why_not.str.contains("triage_status=error|no_lightcurve|load_failed", regex=True)
        mask_insufficient_points = why_not.str.contains("all_flux_nan|n_points<|insufficient_points|baseline_days<", regex=True)
        mask_no_events = n_events.fillna(0.0) <= 0.0

        reasons.loc[mask_load_failed, "terminal_reason"] = "no_lightcurve/load_failed"
        reasons.loc[mask_load_failed, "source_reason"] = why_not.loc[mask_load_failed].replace("", "triage_status=error")
        reasons.loc[mask_load_failed, "load_failed_exception_type"] = error_type.loc[mask_load_failed].replace("", "UnknownError")
        reasons.loc[mask_load_failed, "load_failed_exception_message"] = error_msg.loc[mask_load_failed].replace("", "triage_status=error")
        reasons.loc[mask_load_failed, "load_failed_campaign"] = campaign.loc[mask_load_failed].replace("", "unknown_campaign")
        reasons.loc[mask_load_failed, "load_failed_source"] = error_stage.loc[mask_load_failed].replace("", "batch_triage")

        mask_apply = (~mask_load_failed) & mask_insufficient_points
        reasons.loc[mask_apply, "terminal_reason"] = "all_flux_nan/insufficient_points"
        reasons.loc[mask_apply, "source_reason"] = why_not.loc[mask_apply].replace("", "insufficient_points")

        mask_apply = (~mask_load_failed) & (~mask_insufficient_points) & mask_no_events
        reasons.loc[mask_apply, "terminal_reason"] = "no_events"
        reasons.loc[mask_apply, "source_reason"] = "n_events<=0"
        no_events_sub = pd.Series(["other_no_events"] * len(reasons), index=reasons.index, dtype=str)
        no_events_sub.loc[
            why_not.str.contains("n_points<|baseline_days<|insufficient", regex=True)
        ] = "insufficient_baseline_or_points"
        no_events_sub.loc[
            why_not.str.contains("outlier_rate|robust_sigma|quality|noisy|whiten", regex=True)
        ] = "bad_quality_flags"
        no_events_sub.loc[(status == "ok") & (why_not == "")] = "too_strict_thresholds_or_no_signal"
        reasons.loc[mask_apply, "no_events_breakdown"] = no_events_sub.loc[mask_apply]
        reasons.loc[mask_apply, "no_events_n_points_after_clean"] = triage_n_points.loc[mask_apply]
        no_events_thresholds = why_not.copy()
        no_events_thresholds = no_events_thresholds.where(
            no_events_thresholds != "",
            whiteness_def.replace("", "unknown_thresholds"),
        )
        reasons.loc[mask_apply, "no_events_thresholds_used"] = no_events_thresholds.loc[mask_apply]
        baseline_extracted = why_not.str.extract(r"baseline_days<([0-9]+(?:\.[0-9]+)?)", expand=False)
        reasons.loc[mask_apply, "no_events_baseline_days"] = pd.to_numeric(
            baseline_extracted.loc[mask_apply], errors="coerce"
        )

        summary_work = df_summary_raw.copy()
        summary_work["epic"] = summary_work.get("epic", "").fillna("").astype(str).str.strip()
        summary_work["reason"] = summary_work.get("reason", "").fillna("").astype(str).str.strip().str.lower()
        n_after_by_epic = (
            pd.to_numeric(summary_work.get("n_events_after_filters", np.nan), errors="coerce")
            .groupby(summary_work["epic"])
            .max()
            .to_dict()
            if len(summary_work) > 0
            else {}
        )
        reason_set_by_epic: Dict[str, set[str]] = {}
        if len(summary_work) > 0:
            for epic, grp in summary_work.groupby("epic"):
                reason_set_by_epic[str(epic)] = set(grp["reason"].fillna("").astype(str).tolist())

        q = quarantine_df.copy()
        if len(q) > 0:
            q["epic_id"] = q.get("epic_id", "").fillna("").astype(str).str.strip()
            q["reason"] = q.get("reason", "").fillna("").astype(str).str.strip().str.lower()
            q["source_reason"] = q.get("source_reason", "").fillna("").astype(str).str.strip().str.lower()
            q = q.sort_values(["epic_id"], kind="mergesort")
            q_first = q.drop_duplicates(subset=["epic_id"], keep="first")
            quarantine_by_epic = {
                str(r["epic_id"]): {
                    "reason": str(r.get("reason", "")),
                    "source_reason": str(r.get("source_reason", "")),
                    "failure_category": str(r.get("failure_category", "")),
                    "failure_detail": str(r.get("failure_detail", "")),
                    "n_events_raw": self._as_float(r.get("n_events_raw", np.nan)),
                    "n_events_after_filters": self._as_float(r.get("n_events_after_filters", np.nan)),
                    "infer_max_period_days": self._as_float(r.get("infer_max_period_days", np.nan)),
                    "infer_min_hits": self._as_float(r.get("infer_min_hits", np.nan)),
                    "infer_tol_frac": self._as_float(r.get("infer_tol_frac", np.nan)),
                    "min_cluster_count": self._as_float(r.get("min_cluster_count", np.nan)),
                    "period_cap_days": self._as_float(r.get("period_cap_days", np.nan)),
                    "min_period_days": self._as_float(r.get("min_period_days", np.nan)),
                    "top_k_periods": self._as_float(r.get("top_k_periods", np.nan)),
                    "hist_total": self._as_float(r.get("hist_total", np.nan)),
                    "hist_finite_period": self._as_float(r.get("hist_finite_period", np.nan)),
                    "hist_in_period_range": self._as_float(r.get("hist_in_period_range", np.nan)),
                    "hist_pass_cluster_count": self._as_float(r.get("hist_pass_cluster_count", np.nan)),
                    "hist_pass_all_filters": self._as_float(r.get("hist_pass_all_filters", np.nan)),
                    "whiteness_missing": bool(self._to_bool_series(pd.Series([r.get("whiteness_missing", False)])).iloc[0]),
                    "whiteness_null_reason_category": str(r.get("whiteness_null_reason_category", "")),
                    "shortlist_rejection_stage": str(r.get("shortlist_rejection_stage", "")),
                    "shortlist_rejection_reason": str(r.get("shortlist_rejection_reason", "")),
                    "rejected_before_candidate_scoring": bool(
                        self._to_bool_series(pd.Series([r.get("rejected_before_candidate_scoring", False)])).iloc[0]
                    ),
                    "rejected_after_candidate_scoring": bool(
                        self._to_bool_series(pd.Series([r.get("rejected_after_candidate_scoring", False)])).iloc[0]
                    ),
                }
                for _, r in q_first.iterrows()
            }
        else:
            quarantine_by_epic = {}

        summary_valid_work = df_summary_valid.copy()
        summary_valid_work["epic"] = summary_valid_work.get("epic", "").fillna("").astype(str).str.strip()
        summary_valid_work["reason"] = summary_valid_work.get("reason", "").fillna("").astype(str).str.strip().str.lower()
        p_valid = pd.to_numeric(summary_valid_work.get("P", np.nan), errors="coerce")
        candidate_epics = set(summary_valid_work.loc[p_valid.notna(), "epic"].astype(str).tolist())
        validated_epics = set(
            df_summary_validated_only.get("epic", pd.Series(dtype=str)).fillna("").astype(str).str.strip().tolist()
        )
        best_epics = set(best_df.get("epic", pd.Series(dtype=str)).fillna("").astype(str).str.strip().tolist())
        quarantine_epics = set(q.get("epic_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip().tolist()) if len(q) > 0 else set()
        validation_fail_reasons = {"cluster_only_no_valid_period", "cluster_only_validation_error", "cluster_only_cache_miss"}

        reason_index = {str(ep): idx for idx, ep in zip(reasons.index, reasons["epic_id"].astype(str))}
        if len(selected_epics - set(reason_index.keys())) > 0:
            missing_rows = pd.DataFrame(
                {
                    "epic_id": sorted(selected_epics - set(reason_index.keys())),
                    "query": [f"EPIC {x}" for x in sorted(selected_epics - set(reason_index.keys()))],
                    "triage_status": "",
                    "triage_why_not_usable": "",
                    "n_events": np.nan,
                    "n_periods_proposed": np.nan,
                    "n_periods_validated": np.nan,
                    "selected_for_period_stage": True,
                    "terminal_reason": "other",
                    "source_reason": "selected_epic_missing_in_raw_epic_list",
                }
            )
            reasons = pd.concat([reasons, missing_rows], ignore_index=True)
            reason_index = {str(ep): idx for idx, ep in zip(reasons.index, reasons["epic_id"].astype(str))}

        for epic in selected_epics:
            idx = reason_index.get(epic)
            if idx is None:
                continue
            q_info = quarantine_by_epic.get(epic)
            if q_info is not None:
                q_reason = str(q_info.get("reason", "")).strip().lower()
                src_reason = str(q_info.get("source_reason", "")).strip().lower()
                reasons.at[idx, "period_failure_category"] = str(q_info.get("failure_category", ""))
                reasons.at[idx, "period_failure_detail"] = str(q_info.get("failure_detail", ""))
                reasons.at[idx, "period_n_events_raw"] = self._as_float(q_info.get("n_events_raw", np.nan))
                reasons.at[idx, "period_n_events_after_filters"] = self._as_float(q_info.get("n_events_after_filters", np.nan))
                reasons.at[idx, "period_infer_max_period_days"] = self._as_float(q_info.get("infer_max_period_days", np.nan))
                reasons.at[idx, "period_infer_min_hits"] = self._as_float(q_info.get("infer_min_hits", np.nan))
                reasons.at[idx, "period_infer_tol_frac"] = self._as_float(q_info.get("infer_tol_frac", np.nan))
                reasons.at[idx, "period_min_cluster_count"] = self._as_float(q_info.get("min_cluster_count", np.nan))
                reasons.at[idx, "period_cap_days"] = self._as_float(q_info.get("period_cap_days", np.nan))
                reasons.at[idx, "period_min_period_days"] = self._as_float(q_info.get("min_period_days", np.nan))
                reasons.at[idx, "period_top_k_periods"] = self._as_float(q_info.get("top_k_periods", np.nan))
                reasons.at[idx, "period_hist_total"] = self._as_float(q_info.get("hist_total", np.nan))
                reasons.at[idx, "period_hist_finite_period"] = self._as_float(q_info.get("hist_finite_period", np.nan))
                reasons.at[idx, "period_hist_in_period_range"] = self._as_float(q_info.get("hist_in_period_range", np.nan))
                reasons.at[idx, "period_hist_pass_cluster_count"] = self._as_float(q_info.get("hist_pass_cluster_count", np.nan))
                reasons.at[idx, "period_hist_pass_all_filters"] = self._as_float(q_info.get("hist_pass_all_filters", np.nan))
                reasons.at[idx, "whiteness_missing"] = bool(q_info.get("whiteness_missing", False))
                reasons.at[idx, "whiteness_null_reason_category"] = str(q_info.get("whiteness_null_reason_category", ""))
                reasons.at[idx, "shortlist_rejection_stage"] = str(
                    q_info.get("shortlist_rejection_stage", "") or "post_candidate_scoring"
                )
                reasons.at[idx, "shortlist_rejection_reason"] = str(
                    q_info.get("shortlist_rejection_reason", "") or src_reason or q_reason
                )
                reasons.at[idx, "rejected_before_candidate_scoring"] = bool(
                    q_info.get("rejected_before_candidate_scoring", False)
                )
                reasons.at[idx, "rejected_after_candidate_scoring"] = True
                if src_reason == "events_filtered_to_zero":
                    reasons.at[idx, "terminal_reason"] = "too_few_events_after_filters"
                    reasons.at[idx, "source_reason"] = str(q_info.get("failure_detail", "") or "events_filtered_to_zero")
                elif src_reason == "no_cluster_periods":
                    reasons.at[idx, "terminal_reason"] = "no_cluster_periods"
                    reasons.at[idx, "source_reason"] = str(
                        q_info.get("failure_category", "") or q_info.get("failure_detail", "") or "no_cluster_periods"
                    )
                elif q_reason == "p_above_max_period":
                    reasons.at[idx, "terminal_reason"] = "period_over_cap"
                    reasons.at[idx, "source_reason"] = "P_above_max_period"
                elif src_reason == "cluster2_guardrail_rejection":
                    reasons.at[idx, "terminal_reason"] = "validated_guardrail_reject"
                    reasons.at[idx, "source_reason"] = src_reason
                elif src_reason in validation_fail_reasons:
                    reasons.at[idx, "terminal_reason"] = "fails_validation"
                    reasons.at[idx, "source_reason"] = src_reason
                else:
                    reasons.at[idx, "terminal_reason"] = "other"
                    reasons.at[idx, "source_reason"] = f"quarantine:{src_reason or q_reason or 'unknown'}"
                continue

            epic_reasons = reason_set_by_epic.get(epic, set())
            if epic in validated_epics:
                reasons.at[idx, "terminal_reason"] = "other"
                reasons.at[idx, "source_reason"] = "validated_period"
            elif epic in candidate_epics:
                if len(epic_reasons.intersection(validation_fail_reasons)) > 0:
                    reasons.at[idx, "terminal_reason"] = "fails_validation"
                    reasons.at[idx, "source_reason"] = "cluster_only_validation_error"
                else:
                    reasons.at[idx, "terminal_reason"] = "other"
                    reasons.at[idx, "source_reason"] = "candidate_periods_generated"
            else:
                n_after = self._as_float(n_after_by_epic.get(epic, np.nan))
                if np.isfinite(n_after) and n_after < 2:
                    reasons.at[idx, "terminal_reason"] = "too_few_events_after_filters"
                    reasons.at[idx, "source_reason"] = f"n_events_after_filters={int(n_after)}"
                    reasons.at[idx, "shortlist_rejection_stage"] = "post_candidate_scoring"
                    reasons.at[idx, "shortlist_rejection_reason"] = "too_few_events_after_filters"
                    reasons.at[idx, "rejected_after_candidate_scoring"] = True
                elif "no_cluster_periods" in epic_reasons:
                    reasons.at[idx, "terminal_reason"] = "no_cluster_periods"
                    reasons.at[idx, "source_reason"] = "no_cluster_periods"
                    reasons.at[idx, "shortlist_rejection_stage"] = "post_candidate_scoring"
                    reasons.at[idx, "shortlist_rejection_reason"] = "no_cluster_periods"
                    reasons.at[idx, "rejected_after_candidate_scoring"] = True
                else:
                    reasons.at[idx, "terminal_reason"] = "other"
                    reasons.at[idx, "source_reason"] = "selected_but_no_period_rows"

        not_selected_mask = ~reasons["selected_for_period_stage"].astype(bool)
        other_mask = reasons["terminal_reason"].eq("other")
        mode = str(selection_meta.get("period_stage_selection_mode", self._period_stage_selection_mode()))
        topk = selection_meta.get("period_stage_k", getattr(self.config, "PERIOD_STAGE_K", None))
        if mode == "topk":
            gate_label = f"not_in_period_stage_topk_{int(topk)}_by_best_shape_score_then_best_depth_snr"
        elif mode == "randomN":
            gate_label = f"not_in_period_stage_random_sample_n{int(selection_meta.get('n_enter_period_stage_pre_slice', 0))}"
        elif mode == "all":
            gate_label = "not_in_period_stage_all_mode_slice_exclusion"
        else:
            gate_label = f"not_in_period_stage_mode_{mode}"
        explicit_precheck_mask = not_selected_mask & reasons["shortlist_rejection_reason"].fillna("").astype(str).str.strip().ne("")
        reasons.loc[explicit_precheck_mask, "terminal_reason"] = "shortlist_precheck_reject"
        reasons.loc[explicit_precheck_mask, "source_reason"] = reasons.loc[explicit_precheck_mask, "shortlist_rejection_reason"]
        reasons.loc[explicit_precheck_mask, "rejected_before_candidate_scoring"] = True
        reasons.loc[not_selected_mask & other_mask & ~explicit_precheck_mask, "source_reason"] = gate_label
        reasons["source_reason"] = reasons["source_reason"].fillna("").astype(str)

        n_total_epics = int(len(reasons))
        if "triage_status" in reasons.columns and reasons["triage_status"].notna().any():
            n_with_lightcurve_loaded = int((status != "error").sum())
        else:
            n_with_lightcurve_loaded = int((reasons["terminal_reason"] != "no_lightcurve/load_failed").sum())
        n_with_events_detected = int((pd.to_numeric(reasons.get("n_events", np.nan), errors="coerce").fillna(0.0) > 0.0).sum())
        n_with_candidate_periods_generated = int(len(candidate_epics))
        n_with_validated_periods = int(len(validated_epics))
        n_with_unique_epic_p = int(len(df_summary_unique))
        n_best_unique_epics = int(len(best_epics))
        n_quarantined_epics = int(len(quarantine_epics))
        n_enter_period_stage = int(selection_meta.get("n_enter_period_stage", len(selected_epics)))
        n_excluded_by_topk_gate = int(selection_meta.get("n_excluded_by_topk_gate", 0))
        n_excluded_by_gate = int(selection_meta.get("n_excluded_by_gate", n_excluded_by_topk_gate))
        n_excluded_by_shortlist_precheck = int(selection_meta.get("n_excluded_by_shortlist_precheck", 0))
        n_validated_period = int(len(validated_epics))
        no_events_breakdown_counts = (
            reasons.loc[reasons["terminal_reason"] == "no_events", "no_events_breakdown"]
            .fillna("other_no_events")
            .astype(str)
            .value_counts()
            .to_dict()
        )

        funnel = {
            "n_total_epics": n_total_epics,
            "n_enter_period_stage": n_enter_period_stage,
            "n_excluded_by_topk_gate": n_excluded_by_topk_gate,
            "n_excluded_by_gate": n_excluded_by_gate,
            "n_excluded_by_shortlist_precheck": n_excluded_by_shortlist_precheck,
            "n_validated_period": n_validated_period,
            "n_with_lightcurve_loaded": n_with_lightcurve_loaded,
            "n_with_events_detected": n_with_events_detected,
            "n_with_candidate_periods_generated": n_with_candidate_periods_generated,
            "n_with_validated_periods": n_with_validated_periods,
            "n_with_unique_epic_p": n_with_unique_epic_p,
            "n_best_unique_epics": n_best_unique_epics,
            "n_quarantined_epics": n_quarantined_epics,
            "n_selected_for_period_stage": int(selection_meta.get("n_selected_for_period_stage", len(selected_epics))),
            "best_count_rows": int(len(best_df)),
            "best_unique_epics": int(best_df.get("epic", pd.Series(dtype=str)).fillna("").astype(str).str.strip().nunique()),
            "summary_unique_epic_p": int(len(df_summary_unique)),
            "validated_only_unique_epic_p": int(len(df_summary_validated_only)),
            "period_stage_selection_mode": str(mode),
            "period_stage_ranking_basis": str(selection_meta.get("ranking_basis", "best_shape_score desc, best_depth_snr desc")),
            "no_events_breakdown_counts": "|".join([f"{k}:{int(v)}" for k, v in no_events_breakdown_counts.items()]),
        }

        reasons["stage_reached"] = "period_stage_other"
        selected_mask = reasons["selected_for_period_stage"].astype(bool)
        reasons.loc[~selected_mask, "stage_reached"] = "pre_period_gate"
        reasons.loc[selected_mask, "stage_reached"] = "period_stage_entered"
        reasons.loc[reasons["terminal_reason"] == "no_lightcurve/load_failed", "stage_reached"] = "lightcurve_load"
        reasons.loc[reasons["terminal_reason"] == "all_flux_nan/insufficient_points", "stage_reached"] = "event_precheck"
        reasons.loc[reasons["terminal_reason"] == "no_events", "stage_reached"] = "event_detection"
        reasons.loc[reasons["terminal_reason"] == "too_few_events_after_filters", "stage_reached"] = "event_filtering"
        reasons.loc[reasons["terminal_reason"] == "no_cluster_periods", "stage_reached"] = "period_inference"
        reasons.loc[reasons["terminal_reason"] == "period_over_cap", "stage_reached"] = "period_cap_filter"
        reasons.loc[reasons["terminal_reason"] == "fails_validation", "stage_reached"] = "period_validation"
        reasons.loc[reasons["terminal_reason"] == "validated_guardrail_reject", "stage_reached"] = "validated_guardrail"
        reasons.loc[reasons["terminal_reason"] == "shortlist_precheck_reject", "stage_reached"] = "pre_candidate_scoring"
        reasons.loc[reasons["source_reason"] == "validated_period", "stage_reached"] = "validated_period"
        reasons.loc[reasons["source_reason"] == "candidate_periods_generated", "stage_reached"] = "candidate_period_generation"

        details_keys = [
            "query",
            "triage_status",
            "triage_why_not_usable",
            "n_events",
            "n_periods_proposed",
            "n_periods_validated",
            "no_events_breakdown",
            "load_failed_exception_type",
            "load_failed_exception_message",
            "load_failed_campaign",
            "load_failed_source",
            "no_events_n_points_after_clean",
            "no_events_baseline_days",
            "no_events_thresholds_used",
            "period_failure_category",
            "period_failure_detail",
            "period_n_events_raw",
            "period_n_events_after_filters",
            "period_infer_max_period_days",
            "period_infer_min_hits",
            "period_infer_tol_frac",
            "period_min_cluster_count",
            "period_cap_days",
            "period_min_period_days",
            "period_top_k_periods",
            "period_hist_total",
            "period_hist_finite_period",
            "period_hist_in_period_range",
            "period_hist_pass_cluster_count",
            "period_hist_pass_all_filters",
            "whiteness_missing",
            "whiteness_null_reason_category",
            "shortlist_rejection_stage",
            "shortlist_rejection_reason",
            "rejected_before_candidate_scoring",
            "rejected_after_candidate_scoring",
        ]

        def _to_details_json(row: pd.Series) -> str:
            term = str(row.get("terminal_reason", "")).strip().lower()
            payload: Dict[str, Any] = {
                "selected_for_period_stage": bool(row.get("selected_for_period_stage", False)),
                "period_stage_mode": str(mode),
            }
            force_null_keys: set[str] = set()
            if term == "no_events":
                force_null_keys = {
                    "no_events_breakdown",
                    "no_events_n_points_after_clean",
                    "no_events_baseline_days",
                    "no_events_thresholds_used",
                }
            elif term == "no_lightcurve/load_failed":
                force_null_keys = {
                    "load_failed_exception_type",
                    "load_failed_exception_message",
                    "load_failed_campaign",
                    "load_failed_source",
                }
            for key in details_keys:
                value = row.get(key, None)
                if pd.isna(value):
                    if key in force_null_keys:
                        payload[key] = None
                    continue
                if isinstance(value, str):
                    v = value.strip()
                    if v == "":
                        if key in force_null_keys:
                            payload[key] = None
                        continue
                    payload[key] = v
                    continue
                if isinstance(value, (np.integer,)):
                    payload[key] = int(value)
                    continue
                if isinstance(value, (np.floating, float)):
                    fv = float(value)
                    if np.isfinite(fv):
                        payload[key] = fv
                    continue
                payload[key] = value
            return json.dumps(payload, sort_keys=True)

        reasons["details_json"] = reasons.apply(_to_details_json, axis=1)
        reasons = reasons[
            [
                "query",
                "epic_id",
                "terminal_reason",
                "source_reason",
                "stage_reached",
                "whiteness_missing",
                "whiteness_null_reason_category",
                "shortlist_rejection_stage",
                "shortlist_rejection_reason",
                "rejected_before_candidate_scoring",
                "rejected_after_candidate_scoring",
                "details_json",
            ]
        ].copy()
        return funnel, reasons

    def _validate_period_rows(
        self,
        df_summary: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        cfg = self.config
        work = df_summary.copy()
        if "P" not in work.columns:
            work["P"] = np.nan
        if "reason" not in work.columns:
            work["reason"] = ""

        work["P"] = pd.to_numeric(work["P"], errors="coerce")
        p = work["P"].to_numpy(dtype=float)
        min_period = float(self._as_float(getattr(cfg, "MIN_PERIOD_DAYS", 0.5), default=0.5))
        max_period = float(self._effective_max_period_days())

        mask_null = ~np.isfinite(p)
        mask_nonpositive = np.isfinite(p) & (p <= 0.0)
        mask_below_min = np.isfinite(p) & (p > 0.0) & (p < min_period)
        mask_above_max = np.isfinite(p) & (p > max_period)
        invalid_mask = mask_null | mask_nonpositive | mask_below_min | mask_above_max

        invalid_reason = np.select(
            [mask_null, mask_nonpositive, mask_below_min, mask_above_max],
            ["P_null_or_missing", "P_nonpositive", "P_below_min_period", "P_above_max_period"],
            default="",
        )

        quarantine = work.loc[invalid_mask].copy()
        quarantine["source_reason"] = quarantine["reason"].fillna("").astype(str)
        quarantine["reason"] = invalid_reason[invalid_mask]
        quarantine["missing_upstream_source"] = quarantine["source_reason"].map(self._missing_upstream_source)
        quarantine["epic_id"] = quarantine.get("epic", pd.Series([""] * len(quarantine), index=quarantine.index)).astype(str)
        quarantine["whiteness_missing"] = False
        quarantine["whiteness_null_reason_category"] = ""
        quarantine["shortlist_rejection_stage"] = "post_candidate_scoring"
        quarantine["shortlist_rejection_reason"] = quarantine["source_reason"].fillna("").astype(str)
        quarantine["rejected_before_candidate_scoring"] = False
        quarantine["rejected_after_candidate_scoring"] = True
        quarantine["failure_category"] = ""
        quarantine["failure_detail"] = ""
        quarantine["infer_max_period_days"] = np.nan
        quarantine["infer_min_hits"] = np.nan
        quarantine["infer_tol_frac"] = np.nan
        quarantine["min_cluster_count"] = np.nan
        quarantine["period_cap_days"] = np.nan
        quarantine["min_period_days"] = np.nan
        quarantine["top_k_periods"] = np.nan
        quarantine["hist_total"] = np.nan
        quarantine["hist_finite_period"] = np.nan
        quarantine["hist_in_period_range"] = np.nan
        quarantine["hist_pass_cluster_count"] = np.nan
        quarantine["hist_pass_all_filters"] = np.nan
        source_reason_low = quarantine["source_reason"].fillna("").astype(str).str.strip().str.lower()
        quarantine.loc[source_reason_low == "events_filtered_to_zero", "failure_category"] = "events_filtered_to_zero"
        quarantine.loc[source_reason_low == "events_filtered_to_zero", "failure_detail"] = (
            "all_events_removed_before_period_inference"
        )
        quarantine.loc[source_reason_low == "missing_events_csv", "failure_category"] = "missing_events_csv"
        quarantine.loc[source_reason_low == "missing_events_csv", "failure_detail"] = "events_csv_not_found"
        quarantine.loc[source_reason_low == "cannot_parse_epic", "failure_category"] = "cannot_parse_epic"
        quarantine.loc[source_reason_low == "cannot_parse_epic", "failure_detail"] = "query_to_epic_parse_failed"
        quarantine = quarantine.reindex(columns=self.QUARANTINE_COLUMNS)

        valid = work.loc[~invalid_mask].copy().reindex(columns=self.SUMMARY_COLUMNS)
        diagnostics = {
            "rows_total": int(len(work)),
            "rows_null_p": int(mask_null.sum()),
            "rows_invalid_p": int(invalid_mask.sum()),
            "rows_dropped": int(invalid_mask.sum()),
            "rows_valid": int((~invalid_mask).sum()),
        }
        return valid, quarantine, diagnostics

    def _assert_output_consistency(
        self,
        *,
        out_summary_csv: Path,
        out_best_csv: Path,
        out_diagnostics_csv: Path,
        expected_summary_unique_rows: int,
        expected_best_rows: int,
    ) -> None:
        written_summary = self._read_csv(out_summary_csv)
        written_best = self._read_csv(out_best_csv)
        if int(len(written_summary)) != int(expected_summary_unique_rows):
            raise RuntimeError(
                "[K2ShortlistPeriodRunner] consistency check failed: "
                f"rows_unique_epic_p expected={int(expected_summary_unique_rows)} "
                f"written_summary_rows={int(len(written_summary))}"
            )
        if int(len(written_best)) != int(expected_best_rows):
            raise RuntimeError(
                "[K2ShortlistPeriodRunner] consistency check failed: "
                f"rows_best expected={int(expected_best_rows)} "
                f"written_best_rows={int(len(written_best))}"
            )

        diag_df = self._read_csv(out_diagnostics_csv)
        if len(diag_df) == 0:
            raise RuntimeError("[K2ShortlistPeriodRunner] consistency check failed: diagnostics CSV is empty")
        row = diag_df.iloc[0]
        diag_unique = int(pd.to_numeric(pd.Series([row.get("rows_unique_epic_p", np.nan)]), errors="coerce").fillna(-1).iloc[0])
        diag_best = int(pd.to_numeric(pd.Series([row.get("rows_best", np.nan)]), errors="coerce").fillna(-1).iloc[0])
        if diag_unique != int(expected_summary_unique_rows):
            raise RuntimeError(
                "[K2ShortlistPeriodRunner] consistency check failed: diagnostics rows_unique_epic_p "
                f"expected={int(expected_summary_unique_rows)} diagnostics={diag_unique}"
            )
        if diag_best != int(expected_best_rows):
            raise RuntimeError(
                "[K2ShortlistPeriodRunner] consistency check failed: diagnostics rows_best "
                f"expected={int(expected_best_rows)} diagnostics={diag_best}"
            )

    def _enforce_null_p_rate_threshold(
        self,
        diagnostics: Dict[str, int],
        quarantine_df: pd.DataFrame,
    ) -> None:
        cfg = self.config
        rows_total = int(diagnostics.get("rows_total", 0))
        rows_null_p = int(diagnostics.get("rows_null_p", 0))
        if rows_total <= 0:
            return
        null_rate = float(rows_null_p / rows_total)
        max_rate = float(self._as_float(getattr(cfg, "NULL_P_RATE_MAX", 0.001), default=0.001))
        if (not np.isfinite(max_rate)) or (max_rate < 0):
            max_rate = 0.001
        if null_rate <= max_rate:
            return

        q = quarantine_df.copy() if len(quarantine_df) > 0 else pd.DataFrame()
        if "reason" not in q.columns:
            q["reason"] = ""
        if "source_reason" not in q.columns:
            q["source_reason"] = ""
        if "missing_upstream_source" not in q.columns:
            q["missing_upstream_source"] = ""
        q["reason"] = q["reason"].fillna("").astype(str)
        q["source_reason"] = q["source_reason"].fillna("").astype(str).str.strip().str.lower()
        q["missing_upstream_source"] = q["missing_upstream_source"].fillna("").astype(str).str.strip()
        infer_mask = (q["source_reason"] == "") & (q["missing_upstream_source"] != "")
        if bool(infer_mask.any()):
            q.loc[infer_mask, "source_reason"] = (
                q.loc[infer_mask, "missing_upstream_source"].map(self._source_reason_from_missing_upstream_source)
            )
        q_null = q.loc[q["reason"].str.strip().str.lower() == "p_null_or_missing"].copy()

        exempt_reasons_cfg = getattr(cfg, "NULL_P_RATE_EXEMPT_SOURCE_REASONS", ("no_cluster_periods",))
        exempt_reasons = {str(x).strip().lower() for x in exempt_reasons_cfg}
        q_unexpected = q_null.loc[~q_null["source_reason"].isin(exempt_reasons)].copy()

        if len(q_unexpected) == 0 and len(q_null) > 0:
            print(
                f"[K2ShortlistPeriodRunner] null-P rate {null_rate:.4%} exceeds threshold {max_rate:.4%}, "
                f"but all null-P rows are exempt reasons={sorted(exempt_reasons)}; continuing."
            )
            return

        top_epics: List[str] = []
        source_breakdown: Dict[str, int] = {}
        q_for_top = q_unexpected if len(q_unexpected) > 0 else q_null
        if "epic_id" in q_for_top.columns and len(q_for_top) > 0:
            vc = q_for_top["epic_id"].fillna("").astype(str).value_counts().head(20)
            top_epics = [str(x) for x in vc.index.tolist()]
        if len(q_for_top) > 0:
            source_breakdown = q_for_top["source_reason"].value_counts().to_dict()

        raise RuntimeError(
            f"[K2ShortlistPeriodRunner] null-P rate {null_rate:.4%} exceeds threshold {max_rate:.4%}; "
            f"top_20_epics={top_epics}; source_reason_counts={source_breakdown}"
        )

    def _augment_quarantine_with_failure_diagnostics(
        self,
        quarantine_df: pd.DataFrame,
        inference_failures_by_epic: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        if len(quarantine_df) == 0:
            return quarantine_df.copy().reindex(columns=self.QUARANTINE_COLUMNS)
        out = quarantine_df.copy()
        numeric_cols = {
            "P",
            "n_events_raw",
            "n_events_after_filters",
            "infer_max_period_days",
            "infer_min_hits",
            "infer_tol_frac",
            "min_cluster_count",
            "period_cap_days",
            "min_period_days",
            "top_k_periods",
            "hist_total",
            "hist_finite_period",
            "hist_in_period_range",
            "hist_pass_cluster_count",
            "hist_pass_all_filters",
        }
        for c in self.QUARANTINE_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan if c in numeric_cols else ""
        out["whiteness_missing"] = self._to_bool_series(
            out.get("whiteness_missing", pd.Series([False] * len(out), index=out.index))
        )
        out["rejected_before_candidate_scoring"] = self._to_bool_series(
            out.get("rejected_before_candidate_scoring", pd.Series([False] * len(out), index=out.index))
        )
        out["rejected_after_candidate_scoring"] = self._to_bool_series(
            out.get("rejected_after_candidate_scoring", pd.Series([True] * len(out), index=out.index))
        )
        out["shortlist_rejection_stage"] = out.get(
            "shortlist_rejection_stage", pd.Series(["post_candidate_scoring"] * len(out), index=out.index)
        ).fillna("").astype(str)
        out["shortlist_rejection_reason"] = out.get(
            "shortlist_rejection_reason", pd.Series([""] * len(out), index=out.index)
        ).fillna("").astype(str)
        out["whiteness_null_reason_category"] = out.get(
            "whiteness_null_reason_category", pd.Series([""] * len(out), index=out.index)
        ).fillna("").astype(str)

        no_cluster_mask = (
            out.get("source_reason", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.strip().str.lower()
            == "no_cluster_periods"
        )
        if not bool(no_cluster_mask.any()):
            return out.reindex(columns=self.QUARANTINE_COLUMNS)

        for idx in out.index[no_cluster_mask]:
            epic_id = str(out.at[idx, "epic_id"]).strip()
            payload = inference_failures_by_epic.get(epic_id)
            if payload is None:
                out.at[idx, "failure_category"] = "unknown_no_cluster_periods"
                continue

            params = payload.get("params", {}) if isinstance(payload.get("params", {}), dict) else {}
            out.at[idx, "failure_category"] = str(payload.get("failure_category", "no_cluster_periods"))
            out.at[idx, "failure_detail"] = str(payload.get("detail", ""))
            out.at[idx, "n_events_raw"] = int(payload.get("n_events_raw", out.at[idx, "n_events_raw"] if pd.notna(out.at[idx, "n_events_raw"]) else 0))
            out.at[idx, "n_events_after_filters"] = int(payload.get("n_events_after_filters", out.at[idx, "n_events_after_filters"] if pd.notna(out.at[idx, "n_events_after_filters"]) else 0))
            out.at[idx, "infer_max_period_days"] = self._as_float(params.get("infer_max_period_days", np.nan))
            out.at[idx, "infer_min_hits"] = self._as_float(params.get("infer_min_hits", np.nan))
            out.at[idx, "infer_tol_frac"] = self._as_float(params.get("infer_tol_frac", np.nan))
            out.at[idx, "min_cluster_count"] = self._as_float(params.get("min_cluster_count", np.nan))
            out.at[idx, "period_cap_days"] = self._as_float(params.get("period_cap_days", self._effective_max_period_days()))
            out.at[idx, "min_period_days"] = self._as_float(params.get("min_period_days", np.nan))
            out.at[idx, "top_k_periods"] = self._as_float(params.get("top_k_periods", np.nan))
            extra = payload.get("extra", {}) if isinstance(payload.get("extra", {}), dict) else {}
            out.at[idx, "hist_total"] = self._as_float(extra.get("hist_total", np.nan))
            out.at[idx, "hist_finite_period"] = self._as_float(extra.get("hist_finite_period", np.nan))
            out.at[idx, "hist_in_period_range"] = self._as_float(extra.get("hist_in_period_range", np.nan))
            out.at[idx, "hist_pass_cluster_count"] = self._as_float(extra.get("hist_pass_cluster_count", np.nan))
            out.at[idx, "hist_pass_all_filters"] = self._as_float(extra.get("hist_pass_all_filters", np.nan))
            out.at[idx, "shortlist_rejection_stage"] = "post_candidate_scoring"
            out.at[idx, "shortlist_rejection_reason"] = str(payload.get("failure_category", "") or payload.get("detail", ""))
            out.at[idx, "rejected_before_candidate_scoring"] = False
            out.at[idx, "rejected_after_candidate_scoring"] = True

        return out.reindex(columns=self.QUARANTINE_COLUMNS)

    def _dedupe_epic_period_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df.copy().reindex(columns=self.SUMMARY_COLUMNS)
        input_cols = list(df.columns)
        work = df.copy()
        for c in ["P", "soft_hit_rate", "hit_rate_snr", "hit_rate_shape", "cluster_count"]:
            work[c] = pd.to_numeric(work.get(c, np.nan), errors="coerce")
        work["epic"] = work.get("epic", "").fillna("").astype(str)
        work["reason"] = work.get("reason", "").fillna("").astype(str)
        work["_is_validated"] = work["reason"].str.lower().eq("validated").astype(int)
        work["_p_key"] = pd.to_numeric(work["P"], errors="coerce").round(8)
        work = work.sort_values(
            ["epic", "_p_key", "_is_validated", "soft_hit_rate", "hit_rate_snr", "hit_rate_shape", "cluster_count"],
            ascending=[True, True, False, False, False, False, False],
            kind="mergesort",
        )
        dedup = work.drop_duplicates(subset=["epic", "_p_key"], keep="first").copy()
        dedup = dedup.reindex(columns=input_cols).reset_index(drop=True)
        return dedup

    def _select_best_rows_stratified(
        self,
        work: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int]]:
        if len(work) == 0:
            return work.copy(), {}, {}, {}

        out = work.copy()
        out["P"] = pd.to_numeric(out.get("P", np.nan), errors="coerce")
        out["score_raw"] = pd.to_numeric(out.get("score_raw", 0.0), errors="coerce").fillna(0.0)
        edges = self._period_bin_edges()
        bins = [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(len(edges) - 1)]
        out["period_bin_label"] = pd.cut(
            out["P"],
            bins=edges,
            labels=bins,
            include_lowest=True,
            right=True,
        )
        out["period_bin_label"] = out["period_bin_label"].astype(str).replace("nan", "__out_of_bin__")
        out = out.sort_values(["score_raw", "_row_order"], ascending=[False, True], kind="mergesort")

        all_epics = out["epic"].astype(str).tolist()
        all_epics_unique = list(dict.fromkeys(all_epics))
        n_epics = len(all_epics_unique)
        if n_epics == 0:
            return pd.DataFrame(columns=work.columns), {}, {}, {}

        bin_df = out.loc[out["period_bin_label"].isin(bins)].copy()
        bin_best = (
            bin_df.sort_values(["score_raw", "_row_order"], ascending=[False, True], kind="mergesort")
            .drop_duplicates(subset=["epic", "period_bin_label"], keep="first")
        )
        summary_bin_counts = {
            b: int((bin_df["period_bin_label"] == b).sum())
            for b in bins
        }
        availability = {
            b: int(bin_best.loc[bin_best["period_bin_label"] == b, "epic"].astype(str).nunique())
            for b in bins
        }

        n_bins = max(1, len(bins))
        mode = self._best_selection_bin_mode()
        quota_targets = {b: 0 for b in bins}
        if mode == "equal_per_bin":
            base = n_epics // n_bins
            rem = n_epics % n_bins
            for i, b in enumerate(bins):
                quota_targets[b] = int(base + (1 if i < rem else 0))
        else:
            total_summary = int(sum(summary_bin_counts.values()))
            if total_summary <= 0:
                base = n_epics // n_bins
                rem = n_epics % n_bins
                for i, b in enumerate(bins):
                    quota_targets[b] = int(base + (1 if i < rem else 0))
            else:
                raw = {b: (float(n_epics) * float(summary_bin_counts[b]) / float(total_summary)) for b in bins}
                quota_targets = {b: int(np.floor(raw[b])) for b in bins}
                rem = int(n_epics - sum(quota_targets.values()))
                order = sorted(
                    bins,
                    key=lambda b: (-(raw[b] - np.floor(raw[b])), bins.index(b)),
                )
                for b in order:
                    if rem <= 0:
                        break
                    quota_targets[b] += 1
                    rem -= 1

        quotas = {b: min(int(quota_targets[b]), int(availability.get(b, 0))) for b in bins}
        rem = int(n_epics - sum(quotas.values()))
        while rem > 0:
            candidates = [b for b in bins if quotas[b] < availability.get(b, 0)]
            if len(candidates) == 0:
                break
            pick = sorted(
                candidates,
                key=lambda b: (
                    -(quota_targets.get(b, 0) - quotas.get(b, 0)),
                    -(availability.get(b, 0) - quotas.get(b, 0)),
                    bins.index(b),
                ),
            )[0]
            quotas[pick] += 1
            rem -= 1

        selected_rows: List[pd.Series] = []
        selected_epics: set[str] = set()
        achieved = {b: 0 for b in bins}

        for b in bins:
            candidates = bin_best.loc[bin_best["period_bin_label"] == b].sort_values(
                ["score_raw", "_row_order"],
                ascending=[False, True],
                kind="mergesort",
            )
            for _, row in candidates.iterrows():
                epic = str(row["epic"])
                if epic in selected_epics:
                    continue
                if achieved[b] >= quotas[b]:
                    break
                selected_rows.append(row)
                selected_epics.add(epic)
                achieved[b] += 1

        if len(selected_epics) < n_epics:
            fallback = out.sort_values(["score_raw", "_row_order"], ascending=[False, True], kind="mergesort")
            for _, row in fallback.iterrows():
                epic = str(row["epic"])
                if epic in selected_epics:
                    continue
                selected_rows.append(row)
                selected_epics.add(epic)
                if len(selected_epics) >= n_epics:
                    break

        best_df = pd.DataFrame(selected_rows)
        if len(best_df) == 0:
            best_df = pd.DataFrame(columns=work.columns)
            return best_df, quotas, achieved, summary_bin_counts

        best_df = best_df.sort_values("_row_order", kind="mergesort")
        achieved = {b: int((best_df["period_bin_label"] == b).sum()) for b in bins}
        best_df = best_df.reindex(columns=work.columns)
        return best_df, quotas, achieved, summary_bin_counts

    def _save_period_histograms(
        self,
        summary_df: pd.DataFrame,
        best_df: pd.DataFrame,
        out_png: Path,
        out_counts_csv: Path,
    ) -> Dict[str, Any]:
        edges = np.asarray(self._period_bin_edges(), dtype=float)
        if edges.size < 2:
            edges = np.asarray([0.0, float(self._effective_max_period_days())], dtype=float)
        labels = [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(edges.size - 1)]

        p_summary = pd.to_numeric(summary_df.get("P", np.nan), errors="coerce").to_numpy(dtype=float)
        p_best = pd.to_numeric(best_df.get("P", np.nan), errors="coerce").to_numpy(dtype=float)
        p_summary = p_summary[np.isfinite(p_summary)]
        p_best = p_best[np.isfinite(p_best)]

        hist_summary, _ = np.histogram(p_summary, bins=edges)
        hist_best, _ = np.histogram(p_best, bins=edges)
        accept_rate = np.divide(
            hist_best.astype(float),
            hist_summary.astype(float),
            out=np.zeros_like(hist_best, dtype=float),
            where=(hist_summary.astype(float) > 0),
        )
        low_acceptance_flag = accept_rate < 0.30
        counts_df = pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "bin_label": labels,
                "summary_count": hist_summary.astype(int),
                "best_count": hist_best.astype(int),
                "accept_rate": accept_rate.astype(float),
                "low_acceptance_flag": low_acceptance_flag.astype(int),
            }
        )
        out_counts_csv.parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(out_counts_csv, index=False)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.hist(
            p_summary,
            bins=edges,
            alpha=0.45,
            color="#4C78A8",
            label=f"Summary unique (epic,P) (n={len(p_summary)})",
        )
        ax.hist(
            p_best,
            bins=edges,
            alpha=0.45,
            color="#F58518",
            label=f"Best rows (n={len(p_best)})",
        )
        cap = float(self._effective_max_period_days())
        ax.axvline(cap, color="#B22222", linestyle="--", linewidth=1.2, label=f"cap P <= {cap:g} d")
        ax.set_xlabel("Period P [days]")
        ax.set_ylabel("Rows")
        ax.set_title(f"Period histogram (same bins): Summary vs Best; cap={cap:g} d")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        return {
            "hist_bin_edges": [float(x) for x in edges.tolist()],
            "summary_hist_total": int(hist_summary.sum()),
            "best_hist_total": int(hist_best.sum()),
            "accept_rates_by_bin": {str(labels[i]): float(accept_rate[i]) for i in range(len(labels))},
            "low_acceptance_bins": [str(labels[i]) for i in range(len(labels)) if bool(low_acceptance_flag[i])],
        }

    def _log_period_inference_failure(
        self,
        *,
        epic_id: str,
        query: str,
        n_events_raw: int,
        n_events_after_filters: int,
        max_period_days: float,
        min_hits: int,
        tol_frac: float,
        reason: str,
        failure_category: str,
        detail: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        payload = {
            "epic_id": str(epic_id),
            "query": str(query),
            "reason": str(reason),
            "failure_category": str(failure_category),
            "detail": str(detail),
            "n_events_raw": int(n_events_raw),
            "n_events_after_filters": int(n_events_after_filters),
            "params": {
                "min_period_days": float(self._as_float(getattr(cfg, "MIN_PERIOD_DAYS", 0.5), default=0.5)),
                "period_cap_days": float(self._effective_max_period_days()),
                "infer_max_period_days": float(max_period_days),
                "infer_min_hits": int(min_hits),
                "infer_tol_frac": float(tol_frac),
                "min_cluster_count": int(getattr(cfg, "MIN_CLUSTER_COUNT", 3)),
                "top_k_periods": int(getattr(cfg, "TOP_K_PERIODS", getattr(cfg, "VALIDATION_TOP_K", 3))),
            },
        }
        if extra is not None:
            payload["extra"] = extra
        print(f"[K2ShortlistPeriodRunner][period_inference_fail] {payload}")
        return payload

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        shortlist_csv = cfg.shortlist_csv_path
        epics_dir = cfg.epics_dir_path
        out_paths = self._resolve_run_output_paths()
        run_id = str(out_paths["run_id"])
        run_dir = Path(out_paths["run_dir"])
        out_summary_csv = Path(out_paths["out_summary_csv"])
        out_summary_unique_epicp_csv = Path(out_paths["out_summary_unique_epicp_csv"])
        out_summary_validated_only_csv = Path(out_paths["out_summary_validated_only_csv"])
        out_best_csv = Path(out_paths["out_best_csv"])
        out_quarantine_csv = Path(out_paths["out_quarantine_csv"])
        out_diagnostics_csv = Path(out_paths["out_diagnostics_csv"])
        out_epic_funnel_reasons_csv = Path(out_paths["out_epic_funnel_reasons_csv"])
        out_period_hist_png = Path(out_paths["out_period_hist_png"])
        out_period_hist_counts_csv = Path(out_paths["out_period_hist_counts_csv"])

        selection_mode = self._period_stage_selection_mode()
        if not epics_dir.exists():
            raise FileNotFoundError(f"EPICS directory not found: {epics_dir}")

        if shortlist_csv.exists():
            shortlist_df = pd.read_csv(shortlist_csv)
            if "query" not in shortlist_df.columns:
                raise ValueError(f"Column 'query' not found in {shortlist_csv}")
        else:
            shortlist_df = pd.DataFrame({"query": pd.Series([], dtype=str)})
        raw_epics_df = self._load_raw_epic_table(shortlist_df=shortlist_df)
        selected, selection_meta = self._select_period_stage_queries(
            raw_epics_df=raw_epics_df,
            shortlist_df=shortlist_df,
        )

        run_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        inference_failures_by_epic: Dict[str, Dict[str, Any]] = {}
        run_counts = {"cache_hits": 0, "cache_misses": 0, "downloads_done": 0, "validations_run": 0}
        validation_enabled = bool(getattr(cfg, "ENABLE_VALIDATION", True))
        validation_handler: Optional[K2_NoiseHandler] = None
        validation_engine: Optional[K2Validation_Prediction] = None
        validation_snr: Optional[K2SNR] = None
        if validation_enabled:
            try:
                validation_handler = K2_NoiseHandler(quality_strict=True)
                validation_engine = K2Validation_Prediction()
                validation_snr = K2SNR()
            except Exception as exc:
                validation_enabled = False
                print(
                    f"[K2ShortlistPeriodRunner] validation init failed: "
                    f"{type(exc).__name__}: {exc} -> cluster-only mode"
                )

        print(
            f"[K2ShortlistPeriodRunner] period_stage_selection_mode={selection_meta['period_stage_selection_mode']} "
            f"period_stage_K={selection_meta.get('period_stage_k', None)} "
            f"period_stage_N={selection_meta.get('period_stage_n', None)} "
            f"period_stage_seed={selection_meta.get('period_stage_random_seed', None)} "
            f"targets={len(selected)} shortlist={shortlist_csv} "
            f"total_epics={int(len(raw_epics_df))} "
            f"n_excluded_by_shortlist_precheck={int(selection_meta.get('n_excluded_by_shortlist_precheck', 0))} "
            f"n_ranked_candidates={int(selection_meta.get('n_ranked_candidates', 0))} "
            f"n_selected_for_period_stage={int(selection_meta.get('n_selected_for_period_stage', selection_meta.get('n_enter_period_stage_pre_slice', 0)))} "
            f"n_enter_period_stage={int(selection_meta.get('n_enter_period_stage', len(selected)))} "
            f"n_excluded_by_gate={int(selection_meta.get('n_excluded_by_gate', selection_meta.get('n_excluded_by_topk_gate', 0)))}"
        )
        print(f"[K2ShortlistPeriodRunner] mcc_policy_mode={self._mcc_policy_mode()}")
        print(f"[K2ShortlistPeriodRunner] note: {self._mcc_policy_note()}")
        if getattr(cfg, "CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE", None) is not None or getattr(cfg, "CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE", None) is not None:
            print(
                f"[K2ShortlistPeriodRunner] cluster2_guardrails "
                f"hit_rate_shape_min={getattr(cfg, 'CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE', None)} "
                f"soft_hit_rate_min={getattr(cfg, 'CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE', None)}"
            )
        for i, query in enumerate(selected, start=1):
            epic_id = self._extract_epic(query)
            epic_folder = f"EPIC_{epic_id}" if epic_id is not None else None
            events_path = (epics_dir / epic_folder / "events.csv") if epic_folder is not None else None
            events_exists = bool(events_path is not None and events_path.exists())

            if i <= 5:
                print(
                    f"[K2ShortlistPeriodRunner][debug] "
                    f"{query} -> {epic_id} -> {events_path} -> {events_exists}"
                )

            if epic_id is None:
                print(f"[{i}/{len(selected)}] EPIC ? query={query} -> skip (cannot parse EPIC)")
                summary_rows.append(self._summary_row(epic=str(query), query=str(query), reason="cannot_parse_epic"))
                continue

            epic_dir = epics_dir / epic_folder
            if not events_exists:
                print(f"[{i}/{len(selected)}] EPIC {epic_id} -> skip (missing {events_path})")
                summary_rows.append(self._summary_row(epic=str(epic_id), query=str(query), reason="missing_events_csv"))
                continue

            events_raw_df = self._read_csv(events_path)
            n_events_raw = int(len(events_raw_df))
            events_df = self._filter_events_for_periods(events_raw_df)
            n_events_after = int(len(events_df))
            print(
                f"[{i}/{len(selected)}] EPIC {epic_id} "
                f"n_events_raw={n_events_raw} n_events_after_filters={n_events_after}"
            )

            if n_events_after == 0:
                summary_rows.append(
                    self._summary_row(
                        epic=str(epic_id),
                        query=str(query),
                        reason="events_filtered_to_zero",
                        n_events_raw=int(n_events_raw),
                        n_events_after_filters=int(n_events_after),
                    )
                )
                continue

            max_period = float(self._effective_max_period_days())
            infer_min_hits = 1
            infer_tol_frac = 0.01
            _, hist_df = infer_periods_from_events(
                events_df=events_df,
                max_period=max(max_period, 1e-6),
                min_hits=infer_min_hits,
                tol_frac=infer_tol_frac,
            )

            if len(hist_df) == 0:
                if n_events_after < 2:
                    detail = "insufficient_events_for_period_inference"
                else:
                    detail = "infer_periods_from_events_returned_empty_hist"
                failure_payload = self._log_period_inference_failure(
                    epic_id=str(epic_id),
                    query=str(query),
                    n_events_raw=int(n_events_raw),
                    n_events_after_filters=int(n_events_after),
                    max_period_days=float(max_period),
                    min_hits=int(infer_min_hits),
                    tol_frac=float(infer_tol_frac),
                    reason="no_cluster_periods",
                    failure_category="empty_histogram" if n_events_after >= 2 else "insufficient_events",
                    detail=detail,
                )
                inference_failures_by_epic[str(epic_id)] = failure_payload
                summary_rows.append(
                    self._summary_row(
                        epic=str(epic_id),
                        query=str(query),
                        reason="no_cluster_periods",
                        n_events_raw=int(n_events_raw),
                        n_events_after_filters=int(n_events_after),
                    )
                )
                print(f"[{i}/{len(selected)}] EPIC {epic_id} -> no period candidates from t_mid clustering")
                continue

            candidate_hist = self._filter_candidate_period_rows(
                hist_df=hist_df,
                min_period_days=float(getattr(cfg, "MIN_PERIOD_DAYS", 0.5)),
                max_period_days=float(max_period),
                min_cluster_count=int(getattr(cfg, "MIN_CLUSTER_COUNT", 3)),
                top_k=int(getattr(cfg, "TOP_K_PERIODS", getattr(cfg, "VALIDATION_TOP_K", 3))),
            )
            if len(candidate_hist) == 0:
                hist_num = hist_df.copy()
                for c in ["period", "count_hits"]:
                    hist_num[c] = pd.to_numeric(hist_num.get(c, np.nan), errors="coerce")
                min_p = float(self._as_float(getattr(cfg, "MIN_PERIOD_DAYS", 0.5), default=0.5))
                max_p = float(max_period)
                total_hist = int(len(hist_num))
                finite_period = int(hist_num["period"].notna().sum())
                in_range = int(((hist_num["period"] >= min_p) & (hist_num["period"] <= max_p)).sum())
                pass_cluster = int((hist_num["count_hits"] >= int(getattr(cfg, "MIN_CLUSTER_COUNT", 3))).sum())
                pass_all = int(
                    (
                        hist_num["period"].notna()
                        & hist_num["count_hits"].notna()
                        & (hist_num["period"] >= min_p)
                        & (hist_num["period"] <= max_p)
                        & (hist_num["count_hits"] >= int(getattr(cfg, "MIN_CLUSTER_COUNT", 3)))
                    ).sum()
                )
                detail = "all_candidate_periods_failed_filters"
                if pass_cluster == 0:
                    detail = "all_candidate_periods_below_min_cluster_count"
                elif in_range == 0:
                    detail = "all_candidate_periods_outside_period_bounds"
                failure_payload = self._log_period_inference_failure(
                    epic_id=str(epic_id),
                    query=str(query),
                    n_events_raw=int(n_events_raw),
                    n_events_after_filters=int(n_events_after),
                    max_period_days=float(max_period),
                    min_hits=int(infer_min_hits),
                    tol_frac=float(infer_tol_frac),
                    reason="no_cluster_periods",
                    failure_category="candidate_filter_rejection",
                    detail=detail,
                    extra={
                        "hist_total": total_hist,
                        "hist_finite_period": finite_period,
                        "hist_in_period_range": in_range,
                        "hist_pass_cluster_count": pass_cluster,
                        "hist_pass_all_filters": pass_all,
                    },
                )
                inference_failures_by_epic[str(epic_id)] = failure_payload
                summary_rows.append(
                    self._summary_row(
                        epic=str(epic_id),
                        query=str(query),
                        reason="no_cluster_periods",
                        n_events_raw=int(n_events_raw),
                        n_events_after_filters=int(n_events_after),
                    )
                )
                print(
                    f"[{i}/{len(selected)}] EPIC {epic_id} -> no candidates after "
                    f"period/count filters (minP={cfg.MIN_PERIOD_DAYS}, maxP={max_period}, "
                    f"min_cluster={cfg.MIN_CLUSTER_COUNT})"
                )
                continue

            epic_rows: List[Dict[str, Any]] = []
            for _, h in candidate_hist.iterrows():
                p = self._as_float(h.get("period"))
                if not np.isfinite(p) or p <= 0:
                    continue
                cluster_count = int(pd.to_numeric(pd.Series([h.get("count_hits")]), errors="coerce").fillna(0).iloc[0])
                _cluster_count_exact, cluster_center_phase = self._phase_cluster_score_quiet(
                    events_df=events_df,
                    period=float(p),
                    tol_phase=float(cfg.PERIOD_TOL_PHASE),
                )
                epic_rows.append(
                    self._summary_row(
                        epic=str(epic_id),
                        query=str(query),
                        reason="cluster_only",
                        n_events_raw=int(n_events_raw),
                        n_events_after_filters=int(n_events_after),
                        P=float(p),
                        cluster_count=int(cluster_count),
                        cluster_center_phase=float(cluster_center_phase),
                        n_predicted=0,
                        n_covered=0,
                        coverage_rate=float("nan"),
                        hit_rate_snr=0.0,
                        hit_rate_shape=0.0,
                        soft_hit_rate=0.0,
                        n_windows_with_no_candidates=0,
                    )
                )

            top_periods = [float(x) for x in candidate_hist["period"].to_numpy(dtype=float) if np.isfinite(x) and (x > 0)]
            if validation_enabled and len(top_periods) > 0:
                cluster_reason, validated_rows, val_counts = self._validate_top_periods_from_cache(
                    query=str(query),
                    events_df=events_df,
                    cluster_rows=epic_rows,
                    top_periods=top_periods,
                    handler=validation_handler,  # type: ignore[arg-type]
                    validator=validation_engine,  # type: ignore[arg-type]
                    snr=validation_snr,  # type: ignore[arg-type]
                )
                for k in run_counts:
                    run_counts[k] += int(val_counts.get(k, 0))
                if cluster_reason != "cluster_only":
                    for row in epic_rows:
                        row["reason"] = cluster_reason
                epic_rows.extend(validated_rows)

            if len(epic_rows) == 0:
                epic_rows.append(
                    self._summary_row(
                        epic=str(epic_id),
                        query=str(query),
                        reason="cluster_only_no_valid_period",
                        n_events_raw=int(n_events_raw),
                        n_events_after_filters=int(n_events_after),
                    )
                )

            summary_rows.extend(epic_rows)
            best = self._select_best_row(epic_rows)
            print(
                f"[{i}/{len(selected)}] EPIC {epic_id} "
                f"candidate_period_rows={len(epic_rows)} "
                f"best_P={self._as_float(best.get('P')):.6f} "
                f"reason={best.get('reason', '')}"
            )

        df_summary_raw = pd.DataFrame(summary_rows).reindex(columns=self.SUMMARY_COLUMNS)
        print(f"[K2ShortlistPeriodRunner] df_summary_raw.shape={df_summary_raw.shape}")
        if len(df_summary_raw) == 0:
            raise RuntimeError("[K2ShortlistPeriodRunner] df_summary is empty before writing summary output")

        df_summary_valid, quarantine_df, diagnostics = self._validate_period_rows(df_summary_raw)
        quarantine_df = self._augment_quarantine_with_failure_diagnostics(
            quarantine_df=quarantine_df,
            inference_failures_by_epic=inference_failures_by_epic,
        )
        df_summary_valid, quarantine_df, n_cluster2_guardrail_rejected = self._apply_cluster2_validated_guardrails(
            df_summary_valid=df_summary_valid,
            quarantine_df=quarantine_df,
        )
        out_quarantine_csv.parent.mkdir(parents=True, exist_ok=True)
        quarantine_df.to_csv(out_quarantine_csv, index=False)
        self._enforce_null_p_rate_threshold(diagnostics=diagnostics, quarantine_df=quarantine_df)
        df_summary_unique = self._dedupe_epic_period_rows(df_summary_valid)
        validated_only = df_summary_valid.loc[
            df_summary_valid.get("reason", "").fillna("").astype(str).str.lower() == "validated"
        ].copy()
        df_summary_validated_only = self._dedupe_epic_period_rows(validated_only)
        out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary_unique.to_csv(out_summary_csv, index=False)
        out_summary_unique_epicp_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary_unique.to_csv(out_summary_unique_epicp_csv, index=False)
        out_summary_validated_only_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary_validated_only.to_csv(out_summary_validated_only_csv, index=False)

        metric_cols = ["soft_hit_rate", "hit_rate_snr", "hit_rate_shape"]
        for c in metric_cols:
            if c not in df_summary_unique.columns:
                df_summary_unique[c] = np.nan

        print(
            f"[K2ShortlistPeriodRunner] rows_total={diagnostics['rows_total']} "
            f"null_P={diagnostics['rows_null_p']} dropped_invalid_P={diagnostics['rows_dropped']} "
            f"rows_valid={diagnostics['rows_valid']} rows_unique_epic_p={len(df_summary_unique)} "
            f"rows_validated_only={len(df_summary_validated_only)}"
        )
        print(
            f"[K2ShortlistPeriodRunner] period_cap_days={self._effective_max_period_days()} "
            f"(config PERIOD_CAP_DAYS={self._as_float(getattr(cfg, 'PERIOD_CAP_DAYS', float('nan')))})"
        )
        if len(df_summary_unique) > 0:
            print(
                f"[K2ShortlistPeriodRunner] df_summary_unique[{metric_cols}].describe():\n"
                f"{df_summary_unique[metric_cols].describe()}"
            )
        if n_cluster2_guardrail_rejected > 0:
            print(f"[K2ShortlistPeriodRunner] cluster2_guardrail_rejected={n_cluster2_guardrail_rejected}")

        work = df_summary_unique.copy().reset_index(drop=False).rename(columns={"index": "_row_order"})
        if "epic" not in work.columns:
            work["epic"] = ""

        score_cols = ["soft_hit_rate", "hit_rate_snr", "hit_rate_shape", "cluster_count"]
        for c in score_cols:
            if c not in work.columns:
                work[c] = 0.0
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

        work["score_raw"] = (
            100.0 * work["soft_hit_rate"]
            + 10.0 * work["hit_rate_snr"]
            + 5.0 * work["hit_rate_shape"]
            + 0.1 * work["cluster_count"]
        )
        best_df_work, bin_quotas, bin_achieved, summary_bin_counts = self._select_best_rows_stratified(work=work)
        if len(best_df_work) > 0:
            best_df = best_df_work.reindex(columns=list(df_summary_unique.columns))
        else:
            best_df = pd.DataFrame(columns=list(df_summary_unique.columns))

        best_df.to_csv(out_best_csv, index=False)
        n_best_rows = int(len(best_df))
        hist_meta = self._save_period_histograms(
            summary_df=df_summary_unique,
            best_df=best_df,
            out_png=out_period_hist_png,
            out_counts_csv=out_period_hist_counts_csv,
        )
        funnel, funnel_reasons_df = self._build_epic_funnel_and_reasons(
            raw_epics_df=raw_epics_df,
            selected_queries=selected,
            selection_meta=selection_meta,
            df_summary_raw=df_summary_raw,
            df_summary_valid=df_summary_valid,
            df_summary_unique=df_summary_unique,
            df_summary_validated_only=df_summary_validated_only,
            best_df=best_df,
            quarantine_df=quarantine_df,
        )
        out_epic_funnel_reasons_csv.parent.mkdir(parents=True, exist_ok=True)
        funnel_reasons_df.to_csv(out_epic_funnel_reasons_csv, index=False)
        reason_counts = funnel_reasons_df["terminal_reason"].fillna("other").astype(str).value_counts()
        reason_total = max(1, int(reason_counts.sum()))
        print("[K2ShortlistPeriodRunner] terminal_reason summary (top 10):")
        for reason, count in reason_counts.head(10).items():
            pct = (100.0 * float(count) / float(reason_total))
            print(f"  {reason:32s} {int(count):7d} ({pct:6.2f}%)")
        print("[K2ShortlistPeriodRunner] no_events breakdown:")
        breakdown_str = str(funnel.get("no_events_breakdown_counts", "")).strip()
        if breakdown_str == "" or breakdown_str.lower() == "nan":
            print("  none")
        else:
            for token in breakdown_str.split("|"):
                token = token.strip()
                if token == "":
                    continue
                if ":" in token:
                    key, value = token.rsplit(":", 1)
                    print(f"  {key.strip()}: {value.strip()}")
                else:
                    print(f"  {token}")
        total_epics = max(1, int(funnel["n_total_epics"]))
        load_failed_count = int(
            (funnel_reasons_df["terminal_reason"].fillna("").astype(str) == "no_lightcurve/load_failed").sum()
        )
        no_events_count = int(
            (funnel_reasons_df["terminal_reason"].fillna("").astype(str) == "no_events").sum()
        )
        quarantined_no_cluster_periods = int(
            quarantine_df.loc[
                quarantine_df.get("source_reason", "").fillna("").astype(str).str.strip().str.lower() == "no_cluster_periods",
                "epic_id",
            ]
            .fillna("")
            .astype(str)
            .nunique()
        )
        summary_lines = [
            ("total_epics", int(funnel["n_total_epics"])),
            ("entered_period_stage", int(funnel["n_enter_period_stage"])),
            ("excluded_by_shortlist_precheck", int(funnel.get("n_excluded_by_shortlist_precheck", 0))),
            ("excluded_by_gate", int(funnel["n_excluded_by_gate"])),
            ("load_failed", load_failed_count),
            ("no_events", no_events_count),
            ("candidate_periods_generated", int(funnel["n_with_candidate_periods_generated"])),
            ("validated_period", int(funnel["n_validated_period"])),
            ("quarantined_no_cluster_periods", quarantined_no_cluster_periods),
        ]
        print("[K2ShortlistPeriodRunner] Funnel Summary")
        for key, value in summary_lines:
            pct = (100.0 * float(value) / float(total_epics))
            print(f"  {key:32s} {int(value):7d} ({pct:6.2f}%)")
        print(
            "[K2ShortlistPeriodRunner] dedup_counts "
            f"best_count_rows={int(funnel['best_count_rows'])} "
            f"best_unique_epics={int(funnel['best_unique_epics'])} "
            f"summary_unique_epic_p={int(funnel['summary_unique_epic_p'])} "
            f"validated_only_unique_epic_p={int(funnel['validated_only_unique_epic_p'])}"
        )

        diagnostics_row = {
            "mcc_policy_mode": str(self._mcc_policy_mode()),
            "mcc_policy_note": str(self._mcc_policy_note()),
            "min_cluster_count": int(getattr(cfg, "MIN_CLUSTER_COUNT", K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT)),
            "default_min_cluster_count": int(K2ShortlistPeriodConfig.MIN_CLUSTER_COUNT),
            "manual_review_cluster_count_eq": int(getattr(cfg, "MANUAL_REVIEW_CLUSTER_COUNT_EQ", 2)),
            "cluster2_guardrail_hit_rate_shape_min": self._as_float(
                getattr(cfg, "CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE", float("nan"))
            ),
            "cluster2_guardrail_soft_hit_rate_min": self._as_float(
                getattr(cfg, "CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE", float("nan"))
            ),
            "rows_total": int(diagnostics["rows_total"]),
            "rows_null_p": int(diagnostics["rows_null_p"]),
            "rows_invalid_p": int(diagnostics["rows_invalid_p"]),
            "rows_dropped": int(diagnostics["rows_dropped"]),
            "rows_valid": int(diagnostics["rows_valid"]),
            "rows_unique_epic_p": int(len(df_summary_unique)),
            "rows_validated_only": int(len(df_summary_validated_only)),
            "rows_best": int(n_best_rows),
            "period_cap_days": float(self._effective_max_period_days()),
            "hist_bin_edges": "|".join([f"{x:g}" for x in hist_meta["hist_bin_edges"]]),
            "best_bin_mode": str(self._best_selection_bin_mode()),
            "summary_bin_counts": "|".join([f"{k}:{int(v)}" for k, v in summary_bin_counts.items()]),
            "summary_hist_total": int(hist_meta["summary_hist_total"]),
            "best_hist_total": int(hist_meta["best_hist_total"]),
            "best_bin_quotas": "|".join([f"{k}:{int(v)}" for k, v in bin_quotas.items()]),
            "best_bin_achieved": "|".join([f"{k}:{int(v)}" for k, v in bin_achieved.items()]),
            "best_count_rows": int(funnel["best_count_rows"]),
            "best_unique_epics": int(funnel["best_unique_epics"]),
            "summary_unique_epic_p": int(funnel["summary_unique_epic_p"]),
            "validated_only_unique_epic_p": int(funnel["validated_only_unique_epic_p"]),
            "n_total_epics": int(funnel["n_total_epics"]),
            "n_with_lightcurve_loaded": int(funnel["n_with_lightcurve_loaded"]),
            "n_with_events_detected": int(funnel["n_with_events_detected"]),
            "n_with_candidate_periods_generated": int(funnel["n_with_candidate_periods_generated"]),
            "n_with_validated_periods": int(funnel["n_with_validated_periods"]),
            "n_with_unique_epic_p": int(funnel["n_with_unique_epic_p"]),
            "n_best_unique_epics": int(funnel["n_best_unique_epics"]),
            "n_quarantined_epics": int(funnel["n_quarantined_epics"]),
            "n_selected_for_period_stage": int(funnel["n_selected_for_period_stage"]),
            "n_enter_period_stage": int(funnel["n_enter_period_stage"]),
            "n_excluded_by_topk_gate": int(funnel["n_excluded_by_topk_gate"]),
            "n_excluded_by_gate": int(funnel["n_excluded_by_gate"]),
            "n_excluded_by_shortlist_precheck": int(funnel.get("n_excluded_by_shortlist_precheck", 0)),
            "n_validated_period": int(funnel["n_validated_period"]),
            "n_load_failed": int(load_failed_count),
            "n_no_events": int(no_events_count),
            "n_quarantined_no_cluster_periods": int(quarantined_no_cluster_periods),
            "period_stage_selection_mode": str(funnel["period_stage_selection_mode"]),
            "period_stage_ranking_basis": str(funnel["period_stage_ranking_basis"]),
            "no_events_breakdown_counts": str(funnel["no_events_breakdown_counts"]),
            "accept_rates_by_bin": "|".join([f"{k}:{float(v):.4f}" for k, v in hist_meta.get("accept_rates_by_bin", {}).items()]),
            "low_acceptance_bins": "|".join([str(x) for x in hist_meta.get("low_acceptance_bins", [])]),
        }
        out_diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([diagnostics_row]).to_csv(out_diagnostics_csv, index=False)
        self._assert_output_consistency(
            out_summary_csv=out_summary_csv,
            out_best_csv=out_best_csv,
            out_diagnostics_csv=out_diagnostics_csv,
            expected_summary_unique_rows=int(len(df_summary_unique)),
            expected_best_rows=int(n_best_rows),
        )

        print(
            f"[K2ShortlistPeriodRunner] validated={run_counts['validations_run']} "
            f"cache_hits={run_counts['cache_hits']} "
            f"downloads={run_counts['downloads_done']} "
            f"cache_misses={run_counts['cache_misses']}"
        )
        print(f"[K2ShortlistPeriodRunner] wrote summary: {out_summary_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote summary_unique_epicP: {out_summary_unique_epicp_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote summary_validated_only: {out_summary_validated_only_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote best: {out_best_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote quarantine: {out_quarantine_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote epic funnel reasons: {out_epic_funnel_reasons_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote diagnostics: {out_diagnostics_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote period histogram: {out_period_hist_png}")
        print(f"[K2ShortlistPeriodRunner] wrote period histogram counts: {out_period_hist_counts_csv}")
        print(f"[K2ShortlistPeriodRunner] run_id={run_id} run_dir={run_dir}")
        return {
            "shortlist_csv": shortlist_csv,
            "run_id": run_id,
            "run_dir": run_dir,
            "out_summary_csv": out_summary_csv,
            "out_summary_unique_epicp_csv": out_summary_unique_epicp_csv,
            "out_summary_validated_only_csv": out_summary_validated_only_csv,
            "out_best_csv": out_best_csv,
            "out_quarantine_csv": out_quarantine_csv,
            "out_epic_funnel_reasons_csv": out_epic_funnel_reasons_csv,
            "out_diagnostics_csv": out_diagnostics_csv,
            "out_period_hist_png": out_period_hist_png,
            "out_period_hist_counts_csv": out_period_hist_counts_csv,
            "n_targets": int(len(selected)),
            "n_best_rows": int(n_best_rows),
            "rows_total": int(diagnostics["rows_total"]),
            "rows_null_p": int(diagnostics["rows_null_p"]),
            "rows_dropped": int(diagnostics["rows_dropped"]),
            "rows_valid": int(diagnostics["rows_valid"]),
            "cache_hits": int(run_counts["cache_hits"]),
            "cache_misses": int(run_counts["cache_misses"]),
            "downloads_done": int(run_counts["downloads_done"]),
            "validations_run": int(run_counts["validations_run"]),
        }
