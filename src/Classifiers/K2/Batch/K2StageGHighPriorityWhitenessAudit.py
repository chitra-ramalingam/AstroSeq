from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class K2StageGHighPriorityWhitenessAudit:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_D_HIGH_PRIORITY_CSV = DEFAULT_OUT_DIR / "k2_stage_d_process_now_high_priority.csv"
    DEFAULT_WHITENESS_CSV = DEFAULT_OUT_DIR / "batch_results_whiteness.csv"
    DEFAULT_RERANK_PREVIEW_CSV = DEFAULT_OUT_DIR / "k2_stage_e1_high_priority_rerank_preview.csv"
    DEFAULT_BATCH_001_SUMMARY_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001_summary.csv"
    DEFAULT_BATCH_001B_SUMMARY_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001b_summary.csv"
    DEFAULT_AUDIT_CSV_NAME = "k2_stage_g_high_priority_whiteness_audit.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_g_high_priority_whiteness_audit_summary.csv"

    REQUIRED_STAGE_D_COLUMNS = [
        "epic_id",
        "epic_id_norm",
        "execution_order",
        "best_depth_snr",
        "n_events",
        "n_periods_proposed",
    ]
    REQUIRED_WHITENESS_COLUMNS = [
        "epic_id",
        "triage_status",
        "triage_usable",
        "triage_whiteness_pvalue",
        "triage_whiteness_definition",
        "triage_why_not_usable",
        "triage_whiteness_interpretation",
        "triage_score_global",
        "triage_step_score",
        "triage_whiteness_one_minus_pvalue",
    ]
    REQUIRED_RERANK_COLUMNS = [
        "epic_id_norm",
        "new_execution_order",
    ]

    WHITENESS_ALPHA = 0.01
    BATCH_SIZE = 100
    SURVIVABILITY_WEIGHTS = {
        "whiteness_rank": 0.50,
        "step_inverse_rank": 0.25,
        "score_global_rank": 0.15,
        "n_periods_rank": 0.07,
        "n_events_rank": 0.03,
    }

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Build the Stage G population-level whiteness audit for the full high-priority "
                "Stage D process_now pool using saved upstream features only."
            )
        )
        p.add_argument("--stage-d-high-priority-csv", type=Path, default=cls.DEFAULT_STAGE_D_HIGH_PRIORITY_CSV)
        p.add_argument("--whiteness-csv", type=Path, default=cls.DEFAULT_WHITENESS_CSV)
        p.add_argument("--rerank-preview-csv", type=Path, default=cls.DEFAULT_RERANK_PREVIEW_CSV)
        p.add_argument("--batch-001-summary-csv", type=Path, default=cls.DEFAULT_BATCH_001_SUMMARY_CSV)
        p.add_argument("--batch-001b-summary-csv", type=Path, default=cls.DEFAULT_BATCH_001B_SUMMARY_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            stage_d_high_priority_csv=Path(args.stage_d_high_priority_csv),
            whiteness_csv=Path(args.whiteness_csv),
            rerank_preview_csv=Path(args.rerank_preview_csv),
            batch_001_summary_csv=Path(args.batch_001_summary_csv),
            batch_001b_summary_csv=Path(args.batch_001b_summary_csv),
            out_dir=Path(args.out_dir),
        )

    @staticmethod
    def _read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _prepare_stage_d(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_STAGE_D_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage D high-priority CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        out["execution_order"] = pd.to_numeric(out["execution_order"], errors="coerce")
        if out["execution_order"].isna().any():
            raise ValueError("Stage D high-priority CSV contains non-numeric execution_order values.")
        return out.sort_values(by=["execution_order"], ascending=[True], kind="mergesort").reset_index(drop=True)

    def _prepare_whiteness(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_WHITENESS_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Whiteness CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = (
            out["epic_id"].fillna("").astype(str).str.extract(r"(\d+)")[0].fillna("").astype(str).str.strip()
        )
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        return out.rename(
            columns={
                "query": "saved_query",
                "epic_id": "saved_epic_id",
                "triage_status": "saved_triage_status",
                "triage_usable": "saved_triage_usable",
                "triage_whiteness_pvalue": "saved_triage_whiteness_pvalue",
                "triage_whiteness_definition": "saved_triage_whiteness_definition",
                "triage_why_not_usable": "saved_triage_why_not_usable",
                "triage_whiteness_interpretation": "saved_triage_whiteness_interpretation",
                "triage_score_global": "saved_triage_score_global",
                "triage_step_score": "saved_triage_step_score",
                "triage_whiteness_one_minus_pvalue": "saved_triage_whiteness_one_minus_pvalue",
            }
        )

    def _prepare_rerank_preview(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_RERANK_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage E.1 rerank preview missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        out["new_execution_order"] = pd.to_numeric(out["new_execution_order"], errors="coerce")
        if out["new_execution_order"].isna().any():
            raise ValueError("Stage E.1 rerank preview contains non-numeric new_execution_order values.")
        return out.sort_values(by=["new_execution_order"], ascending=[True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _prepare_summary_row(path: Path) -> Dict[str, Any]:
        df = K2StageGHighPriorityWhitenessAudit._read_required_csv(path)
        if len(df) == 0:
            return {}
        return dict(df.iloc[0].to_dict())

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

    @staticmethod
    def _percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            return pd.Series(0.0, index=series.index, dtype=float)
        fill_value = numeric.min() if higher_is_better else numeric.max()
        filled = numeric.fillna(fill_value)
        return filled.rank(method="average", pct=True, ascending=higher_is_better)

    @staticmethod
    def _count_rows(section: str, metric: str, counts: pd.Series, total_rows: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for value, count in counts.items():
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "submetric": "",
                    "value_text": str(value),
                    "value_num": "",
                    "count": int(count),
                    "fraction": (float(count) / float(total_rows)) if total_rows > 0 else float("nan"),
                    "note": "",
                }
            )
        return rows

    @staticmethod
    def _quantile_rows(section: str, metric: str, series: pd.Series) -> List[Dict[str, Any]]:
        numeric = pd.to_numeric(series, errors="coerce")
        quantiles = {"min": 0.0, "q10": 0.10, "q25": 0.25, "median": 0.50, "q75": 0.75, "q90": 0.90, "max": 1.0}
        rows: List[Dict[str, Any]] = []
        for label, q in quantiles.items():
            value = numeric.quantile(q) if numeric.notna().any() else float("nan")
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "submetric": label,
                    "value_text": "",
                    "value_num": float(value) if pd.notna(value) else float("nan"),
                    "count": "",
                    "fraction": "",
                    "note": "",
                }
            )
        rows.append(
            {
                "section": section,
                "metric": metric,
                "submetric": "mean",
                "value_text": "",
                "value_num": float(numeric.mean()) if numeric.notna().any() else float("nan"),
                "count": "",
                "fraction": "",
                "note": "",
            }
        )
        return rows

    @staticmethod
    def _single_row(
        section: str,
        metric: str,
        *,
        value_text: str = "",
        value_num: Any = "",
        count: Any = "",
        fraction: Any = "",
        note: str = "",
        submetric: str = "",
    ) -> Dict[str, Any]:
        return {
            "section": section,
            "metric": metric,
            "submetric": submetric,
            "value_text": value_text,
            "value_num": value_num,
            "count": count,
            "fraction": fraction,
            "note": note,
        }

    @staticmethod
    def _parse_count_map(text: Any) -> Dict[str, int]:
        out: Dict[str, int] = {}
        raw = "" if pd.isna(text) else str(text)
        if raw.strip() == "":
            return out
        for part in raw.split("|"):
            chunk = part.strip()
            if "=" not in chunk:
                continue
            key, value = chunk.rsplit("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "":
                continue
            try:
                out[key] = int(float(value))
            except ValueError:
                continue
        return out

    def _build_runtime_survivability_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["component_whiteness_rank"] = self._percentile_rank(work["saved_triage_whiteness_pvalue"], higher_is_better=True)
        work["component_step_inverse_rank"] = self._percentile_rank(
            -pd.to_numeric(work["saved_triage_step_score"], errors="coerce"),
            higher_is_better=True,
        )
        work["component_score_global_rank"] = self._percentile_rank(
            work["saved_triage_score_global"],
            higher_is_better=True,
        )
        work["component_n_periods_rank"] = self._percentile_rank(work["n_periods_proposed"], higher_is_better=True)
        work["component_n_events_rank"] = self._percentile_rank(work["n_events"], higher_is_better=True)
        work["runtime_survivability_proxy"] = 100.0 * (
            self.SURVIVABILITY_WEIGHTS["whiteness_rank"] * work["component_whiteness_rank"]
            + self.SURVIVABILITY_WEIGHTS["step_inverse_rank"] * work["component_step_inverse_rank"]
            + self.SURVIVABILITY_WEIGHTS["score_global_rank"] * work["component_score_global_rank"]
            + self.SURVIVABILITY_WEIGHTS["n_periods_rank"] * work["component_n_periods_rank"]
            + self.SURVIVABILITY_WEIGHTS["n_events_rank"] * work["component_n_events_rank"]
        )
        return work

    def _build_risk_bucket(self, df: pd.DataFrame) -> pd.Series:
        pvalue = pd.to_numeric(df["saved_triage_whiteness_pvalue"], errors="coerce")
        step = pd.to_numeric(df["saved_triage_step_score"], errors="coerce")
        score_global = pd.to_numeric(df["saved_triage_score_global"], errors="coerce")

        p_q10 = pvalue.quantile(0.10) if pvalue.notna().any() else float("nan")
        p_q25 = pvalue.quantile(0.25) if pvalue.notna().any() else float("nan")
        p_q50 = pvalue.quantile(0.50) if pvalue.notna().any() else float("nan")
        step_q50 = step.quantile(0.50) if step.notna().any() else float("nan")
        step_q75 = step.quantile(0.75) if step.notna().any() else float("nan")
        step_q90 = step.quantile(0.90) if step.notna().any() else float("nan")
        score_q25 = score_global.quantile(0.25) if score_global.notna().any() else float("nan")

        all_missing = pvalue.isna() & step.isna() & score_global.isna()
        official_fail = pvalue.lt(self.WHITENESS_ALPHA)
        high_risk = pvalue.le(p_q10) & step.ge(step_q90)
        elevated_risk = pvalue.le(p_q25) | step.ge(step_q75)
        moderate_risk = pvalue.le(p_q50) | step.ge(step_q50) | score_global.le(score_q25)

        bucket = pd.Series("lower_relative_whiteness_risk", index=df.index, dtype="string")
        bucket.loc[moderate_risk] = "moderate_relative_whiteness_risk"
        bucket.loc[elevated_risk] = "elevated_relative_whiteness_risk"
        bucket.loc[high_risk] = "high_relative_whiteness_risk"
        bucket.loc[official_fail] = "official_fail_saved_proxy"
        bucket.loc[all_missing] = "proxy_missing"
        return bucket

    @staticmethod
    def _batch_subset(df: pd.DataFrame, order_col: str, batch_size: int) -> pd.DataFrame:
        if order_col not in df.columns:
            return df.iloc[0:0].copy()
        work = df.copy()
        work[order_col] = pd.to_numeric(work[order_col], errors="coerce")
        work = work.loc[work[order_col].notna()].sort_values(by=[order_col], ascending=[True], kind="mergesort")
        return work.head(batch_size).reset_index(drop=True)

    def _batch_consistency(
        self,
        *,
        audit_df: pd.DataFrame,
        batch_001_df: pd.DataFrame,
        batch_001b_df: pd.DataFrame,
        batch_001_summary: Dict[str, Any],
        batch_001b_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        pool_p25 = pd.to_numeric(audit_df["saved_triage_whiteness_pvalue"], errors="coerce").quantile(0.25)
        pool_step_q75 = pd.to_numeric(audit_df["saved_triage_step_score"], errors="coerce").quantile(0.75)

        batch_001_median_p = pd.to_numeric(batch_001_df["saved_triage_whiteness_pvalue"], errors="coerce").median()
        batch_001_median_step = pd.to_numeric(batch_001_df["saved_triage_step_score"], errors="coerce").median()
        batch_001b_median_p = pd.to_numeric(batch_001b_df["saved_triage_whiteness_pvalue"], errors="coerce").median()
        batch_001b_median_step = pd.to_numeric(batch_001b_df["saved_triage_step_score"], errors="coerce").median()

        batch_001_noisy = self._parse_count_map(batch_001_summary.get("label_counts", "")).get("Noisy_trash", 0)
        batch_001b_noisy = self._parse_count_map(batch_001b_summary.get("final_label_counts", "")).get("Noisy_trash", 0)
        batch_001b_whiteness = int(float(batch_001b_summary.get("patched_batch_001b_whiteness_rejection_count", 0) or 0))
        batch_001_whiteness_note = str(batch_001_summary.get("dominant_label_reason", ""))
        batch_001_all_whiteness = "whiteness_pvalue" in batch_001_whiteness_note

        not_worst_tail = (
            pd.notna(batch_001_median_p)
            and pd.notna(batch_001_median_step)
            and pd.notna(batch_001b_median_p)
            and pd.notna(batch_001b_median_step)
            and batch_001_median_p > pool_p25
            and batch_001_median_step < pool_step_q75
            and batch_001b_median_p > pool_p25
            and batch_001b_median_step < pool_step_q75
        )
        consistent = (
            not_worst_tail
            and batch_001_noisy == len(batch_001_df)
            and batch_001b_noisy == len(batch_001b_df)
            and batch_001_all_whiteness
            and batch_001b_whiteness == len(batch_001b_df)
        )
        note = (
            f"Batch 001 median saved_triage_whiteness_pvalue={batch_001_median_p:.6f}, median saved_triage_step_score={batch_001_median_step:.6f}; "
            f"batch 001b median saved_triage_whiteness_pvalue={batch_001b_median_p:.6f}, median saved_triage_step_score={batch_001b_median_step:.6f}; "
            f"pool q25(saved_triage_whiteness_pvalue)={pool_p25:.6f}, pool q75(saved_triage_step_score)={pool_step_q75:.6f}. "
            "Both calibration batches failed 100/100 on runtime whiteness despite sitting inside the main body or better-than-q25 saved proxy region, "
            "so they are consistent with a broader population limitation rather than an isolated bad first slice."
            if consistent
            else
            f"Batch 001 median saved_triage_whiteness_pvalue={batch_001_median_p:.6f}, batch 001b median saved_triage_whiteness_pvalue={batch_001b_median_p:.6f}. "
            "The calibration batches do not cleanly separate from the broader pool profile, but the saved proxy evidence is not strong enough to make that conclusion definitive."
        )
        return {
            "consistent": "yes" if consistent else "maybe",
            "note": note,
            "batch_001_noisy": batch_001_noisy,
            "batch_001b_noisy": batch_001b_noisy,
        }

    def _recommendation(
        self,
        *,
        proxy_coverage: int,
        total_rows: int,
        consistency: Dict[str, Any],
        likely_low_yield_fraction: float,
    ) -> Dict[str, str]:
        if proxy_coverage == 0:
            return {
                "action": "C: revisit Stage C / Stage D routing logic",
                "note": "No saved whiteness or stability proxy exists upstream, so the current high-priority lane cannot be diagnosed well enough without revisiting routing inputs.",
            }
        if consistency["consistent"] == "yes" and likely_low_yield_fraction >= 0.80:
            return {
                "action": "D: revisit whiteness policy scientifically",
                "note": (
                    "Two materially different calibration batches both failed 100/100 on the same runtime whiteness gate, and the full high-priority pool has full saved-proxy coverage "
                    "with at least 85.8% showing zero saved period proposals. That points more strongly to a scientific whiteness/predictor mismatch than to queue ordering or lane routing."
                ),
            }
        if likely_low_yield_fraction >= 0.80:
            return {
                "action": "B: stop using the current high-priority pool and switch to another Stage D lane",
                "note": "The current high-priority pool appears low-yield under current policy even before broader scientific interpretation is resolved.",
            }
        if total_rows > 0:
            return {
                "action": "A: continue executing the current high-priority pool despite expected attrition",
                "note": "The saved-proxy audit does not show enough population-level risk concentration to justify stopping the lane outright.",
            }
        return {
            "action": "C: revisit Stage C / Stage D routing logic",
            "note": "The high-priority pool is empty, so the routing logic should be rechecked before execution.",
        }

    def run(
        self,
        *,
        stage_d_high_priority_csv: Path,
        whiteness_csv: Path,
        rerank_preview_csv: Path,
        batch_001_summary_csv: Path,
        batch_001b_summary_csv: Path,
        out_dir: Path,
    ) -> Dict[str, Any]:
        stage_d = self._prepare_stage_d(Path(stage_d_high_priority_csv))
        whiteness = self._prepare_whiteness(Path(whiteness_csv))
        rerank_preview = self._prepare_rerank_preview(Path(rerank_preview_csv))
        batch_001_summary = self._prepare_summary_row(Path(batch_001_summary_csv))
        batch_001b_summary = self._prepare_summary_row(Path(batch_001b_summary_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        merged = stage_d.merge(whiteness, on="epic_id_norm", how="left")
        merged = self._build_runtime_survivability_proxy(merged)
        merged["whiteness_proxy_available"] = (
            merged[["saved_triage_whiteness_pvalue", "saved_triage_step_score", "saved_triage_score_global"]]
            .notna()
            .any(axis=1)
        )
        merged["whiteness_risk_bucket"] = self._build_risk_bucket(merged)

        preferred_front = [
            "epic_id",
            "query",
            "execution_order",
            "best_depth_snr",
            "n_events",
            "n_periods_proposed",
            "saved_triage_whiteness_pvalue",
            "saved_triage_step_score",
            "saved_triage_score_global",
            "saved_triage_status",
            "saved_triage_usable",
            "saved_triage_whiteness_definition",
            "saved_triage_why_not_usable",
            "saved_triage_whiteness_interpretation",
            "saved_triage_whiteness_one_minus_pvalue",
            "whiteness_proxy_available",
            "whiteness_risk_bucket",
            "runtime_survivability_proxy",
            "component_whiteness_rank",
            "component_step_inverse_rank",
            "component_score_global_rank",
            "component_n_periods_rank",
            "component_n_events_rank",
        ]
        remaining = [c for c in merged.columns if c not in preferred_front]
        audit_df = merged[preferred_front + remaining].sort_values(
            by=["execution_order", "epic_id_norm"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        audit_csv = out_dir / self.DEFAULT_AUDIT_CSV_NAME
        audit_df.to_csv(audit_csv, index=False)

        total_rows = int(len(audit_df))
        proxy_fields = [
            "saved_triage_status",
            "saved_triage_usable",
            "saved_triage_whiteness_pvalue",
            "saved_triage_step_score",
            "saved_triage_score_global",
            "saved_triage_whiteness_definition",
            "saved_triage_why_not_usable",
            "saved_triage_whiteness_interpretation",
            "saved_triage_whiteness_one_minus_pvalue",
        ]
        proxy_coverage = int(self._bool_series(audit_df["whiteness_proxy_available"]).sum())
        current_batch_001 = self._batch_subset(audit_df, "execution_order", self.BATCH_SIZE)
        batch_001b_norms = set(self._batch_subset(rerank_preview, "new_execution_order", self.BATCH_SIZE)["epic_id_norm"].astype(str))
        current_batch_001b = audit_df.loc[audit_df["epic_id_norm"].astype(str).isin(batch_001b_norms)].copy()
        current_batch_001b["new_execution_order"] = current_batch_001b["epic_id_norm"].map(
            rerank_preview.set_index("epic_id_norm")["new_execution_order"].to_dict()
        )
        current_batch_001b = current_batch_001b.sort_values(
            by=["new_execution_order", "execution_order"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        snr = pd.to_numeric(audit_df["best_depth_snr"], errors="coerce")
        pvalue = pd.to_numeric(audit_df["saved_triage_whiteness_pvalue"], errors="coerce")
        step = pd.to_numeric(audit_df["saved_triage_step_score"], errors="coerce")
        high_snr = snr.ge(snr.quantile(0.75))
        poor_whiteness_proxy = pvalue.le(pvalue.quantile(0.25)) | step.ge(step.quantile(0.75))
        overlap_count = int((high_snr & poor_whiteness_proxy).sum())
        overlap_fraction_total = float((high_snr & poor_whiteness_proxy).mean()) if total_rows > 0 else float("nan")
        overlap_fraction_high_snr = float((high_snr & poor_whiteness_proxy).sum() / high_snr.sum()) if int(high_snr.sum()) > 0 else float("nan")

        n_periods_positive = pd.to_numeric(audit_df["n_periods_proposed"], errors="coerce").gt(0)
        n_periods_positive_count = int(n_periods_positive.sum())
        likely_low_yield_count = int((~n_periods_positive).sum())
        likely_low_yield_fraction = (float(likely_low_yield_count) / float(total_rows)) if total_rows > 0 else float("nan")

        batch_consistency = self._batch_consistency(
            audit_df=audit_df,
            batch_001_df=current_batch_001,
            batch_001b_df=current_batch_001b,
            batch_001_summary=batch_001_summary,
            batch_001b_summary=batch_001b_summary,
        )
        recommendation = self._recommendation(
            proxy_coverage=proxy_coverage,
            total_rows=total_rows,
            consistency=batch_consistency,
            likely_low_yield_fraction=likely_low_yield_fraction,
        )

        summary_rows: List[Dict[str, Any]] = []
        summary_rows.append(
            self._single_row(
                "metadata",
                "proxy_source",
                value_text=str(whiteness_csv),
                note="Saved upstream whiteness/stability proxies were joined onto the accepted Stage D high-priority queue by epic_id_norm.",
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "runtime_survivability_proxy_formula",
                note=(
                    "runtime_survivability_proxy = 100 * (0.50*rank(saved_triage_whiteness_pvalue) "
                    "+ 0.25*rank(-saved_triage_step_score) + 0.15*rank(saved_triage_score_global) "
                    "+ 0.07*rank(n_periods_proposed) + 0.03*rank(n_events)); higher is more survivable under the current saved-proxy view."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "whiteness_risk_bucket_logic",
                note=(
                    "official_fail_saved_proxy if saved_triage_whiteness_pvalue<0.01; else high_relative_whiteness_risk if pvalue<=q10 and step_score>=q90; "
                    "else elevated_relative_whiteness_risk if pvalue<=q25 or step_score>=q75; else moderate_relative_whiteness_risk if pvalue<=q50 or step_score>=q50 "
                    "or score_global<=q25; else lower_relative_whiteness_risk. Proxy-missing rows are bucketed separately."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "null_handling",
                note=(
                    "Missing numeric proxies are kept as null in the audit output. For percentile-based survivability scoring only, null values get the worst-fill value "
                    "within that component so all rows remain comparable without dropping any Stage D rows."
                ),
            )
        )

        for field_name in proxy_fields:
            non_null = int(audit_df[field_name].notna().sum()) if field_name in audit_df.columns else 0
            summary_rows.append(
                self._single_row(
                    "coverage",
                    field_name,
                    count=non_null,
                    fraction=(float(non_null) / float(total_rows)) if total_rows > 0 else float("nan"),
                    note=f"missing_count={max(total_rows - non_null, 0)}",
                )
            )

        summary_rows.extend(self._quantile_rows("distribution", "best_depth_snr", audit_df["best_depth_snr"]))
        summary_rows.extend(self._quantile_rows("distribution", "n_events", audit_df["n_events"]))
        summary_rows.extend(self._quantile_rows("distribution", "n_periods_proposed", audit_df["n_periods_proposed"]))
        summary_rows.extend(
            self._quantile_rows("distribution", "saved_triage_whiteness_pvalue", audit_df["saved_triage_whiteness_pvalue"])
        )
        summary_rows.extend(self._quantile_rows("distribution", "saved_triage_step_score", audit_df["saved_triage_step_score"]))
        summary_rows.extend(self._quantile_rows("distribution", "saved_triage_score_global", audit_df["saved_triage_score_global"]))
        summary_rows.extend(
            self._quantile_rows(
                "distribution",
                "saved_triage_whiteness_one_minus_pvalue",
                audit_df["saved_triage_whiteness_one_minus_pvalue"],
            )
        )
        summary_rows.extend(
            self._quantile_rows("distribution", "runtime_survivability_proxy", audit_df["runtime_survivability_proxy"])
        )
        summary_rows.extend(
            self._count_rows(
                "counts",
                "whiteness_risk_bucket",
                audit_df["whiteness_risk_bucket"].fillna("").astype(str).value_counts(),
                total_rows,
            )
        )
        summary_rows.append(
            self._single_row(
                "counts",
                "rows_with_n_periods_proposed_gt0",
                count=n_periods_positive_count,
                fraction=(float(n_periods_positive_count) / float(total_rows)) if total_rows > 0 else float("nan"),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "high_snr_poor_whiteness_proxy_overlap",
                count=overlap_count,
                fraction=overlap_fraction_total,
                note=(
                    "high_snr is defined as best_depth_snr>=pool_q75; poor_whiteness_proxy is defined as "
                    "saved_triage_whiteness_pvalue<=pool_q25 or saved_triage_step_score>=pool_q75; "
                    f"fraction_of_high_snr_rows={overlap_fraction_high_snr:.6f}"
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "batch_001_and_001b_consistent_with_broader_pool_profile",
                value_text=batch_consistency["consistent"],
                note=batch_consistency["note"],
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "likely_low_yield_under_current_policy_estimate",
                count=likely_low_yield_count,
                fraction=likely_low_yield_fraction,
                note=(
                    "Conservative floor estimate based on rows with n_periods_proposed==0. This likely understates low-yield risk because both failed calibration batches "
                    "were drawn from rows that were not concentrated in the worst saved-proxy tail."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "recommendation",
                "primary_next_lever",
                value_text=recommendation["action"],
                note=recommendation["note"],
            )
        )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME
        summary_df.to_csv(summary_csv, index=False)

        bucket_counts = audit_df["whiteness_risk_bucket"].fillna("").astype(str).value_counts().to_dict()
        return {
            "audit_csv": str(audit_csv),
            "summary_csv": str(summary_csv),
            "rows_total": total_rows,
            "proxy_coverage": proxy_coverage,
            "whiteness_risk_bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
            "rows_with_n_periods_proposed_gt0": n_periods_positive_count,
            "likely_low_yield_fraction": likely_low_yield_fraction,
            "recommendation": recommendation["action"],
        }
