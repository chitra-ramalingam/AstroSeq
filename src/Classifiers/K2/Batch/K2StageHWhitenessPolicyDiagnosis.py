from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class K2StageHWhitenessPolicyDiagnosis:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_ORIGINAL_RESULTS_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001_results.csv"
    DEFAULT_PATCHED_RESULTS_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001b_results.csv"
    DEFAULT_WHITENESS_CSV = DEFAULT_OUT_DIR / "batch_results_whiteness.csv"
    DEFAULT_DIAGNOSIS_CSV_NAME = "k2_stage_h_whiteness_policy_diagnosis.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_h_whiteness_policy_diagnosis_summary.csv"
    WHITENESS_ALPHA = 0.01

    REQUIRED_ORIGINAL_COLUMNS = [
        "epic_id",
        "query",
        "execution_order",
        "batch_id",
        "planned_best_depth_snr",
        "planned_n_events",
        "planned_n_periods_proposed",
        "triage_status_pipeline",
        "triage_usable_pipeline",
        "triage_score_global",
        "triage_step_score",
        "triage_whiteness_score",
        "triage_whiteness_definition",
        "triage_why_not_usable_pipeline",
        "label",
        "label_reason",
        "n_events",
        "n_periods_proposed",
        "best_depth_snr",
        "epic_id_norm",
    ]
    REQUIRED_PATCHED_COLUMNS = [
        "epic_id",
        "query",
        "old_execution_order",
        "new_execution_order",
        "planned_best_depth_snr",
        "planned_n_events",
        "planned_n_periods_proposed",
        "saved_triage_whiteness_pvalue",
        "saved_triage_step_score",
        "saved_triage_score_global",
        "saved_triage_whiteness_definition",
        "saved_triage_usable",
        "triage_status_pipeline",
        "triage_usable_pipeline",
        "triage_score_global",
        "triage_step_score",
        "triage_whiteness_score",
        "triage_whiteness_definition",
        "triage_why_not_usable_pipeline",
        "label",
        "label_reason",
        "n_events",
        "n_periods_proposed",
        "best_depth_snr",
        "epic_id_norm",
    ]
    REQUIRED_WHITENESS_COLUMNS = [
        "epic_id",
        "triage_usable",
        "triage_whiteness_pvalue",
        "triage_step_score",
        "triage_score_global",
        "triage_whiteness_definition",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Build the Stage H whiteness-policy diagnosis using saved proxies and the two executed calibration batches."
            )
        )
        p.add_argument("--original-results-csv", type=Path, default=cls.DEFAULT_ORIGINAL_RESULTS_CSV)
        p.add_argument("--patched-results-csv", type=Path, default=cls.DEFAULT_PATCHED_RESULTS_CSV)
        p.add_argument("--whiteness-csv", type=Path, default=cls.DEFAULT_WHITENESS_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            original_results_csv=Path(args.original_results_csv),
            patched_results_csv=Path(args.patched_results_csv),
            whiteness_csv=Path(args.whiteness_csv),
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

    def _prepare_original_results(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_ORIGINAL_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Original batch-001 results CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        out["execution_order"] = pd.to_numeric(out["execution_order"], errors="coerce")
        if out["execution_order"].isna().any():
            raise ValueError("Original batch-001 results contain non-numeric execution_order values.")
        return out

    def _prepare_patched_results(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_PATCHED_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Patched batch-001b results CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["epic_id_norm"] = out["epic_id_norm"].fillna("").astype(str).str.strip()
        out["new_execution_order"] = pd.to_numeric(out["new_execution_order"], errors="coerce")
        if out["new_execution_order"].isna().any():
            raise ValueError("Patched batch-001b results contain non-numeric new_execution_order values.")
        return out

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
                "epic_id": "saved_epic_id",
                "triage_usable": "saved_triage_usable",
                "triage_whiteness_pvalue": "saved_triage_whiteness_pvalue",
                "triage_step_score": "saved_triage_step_score",
                "triage_score_global": "saved_triage_score_global",
                "triage_whiteness_definition": "saved_triage_whiteness_definition",
            }
        )

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

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
        labels = {"min": 0.0, "q25": 0.25, "median": 0.50, "q75": 0.75, "max": 1.0}
        rows: List[Dict[str, Any]] = []
        for label, q in labels.items():
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

    def _normalize_original(self, results_df: pd.DataFrame, whiteness_df: pd.DataFrame) -> pd.DataFrame:
        merged = results_df.merge(
            whiteness_df[
                [
                    "epic_id_norm",
                    "saved_triage_usable",
                    "saved_triage_whiteness_pvalue",
                    "saved_triage_step_score",
                    "saved_triage_score_global",
                    "saved_triage_whiteness_definition",
                ]
            ],
            on="epic_id_norm",
            how="left",
        )
        runtime_pvalue = pd.to_numeric(
            merged.get("triage_whiteness_pvalue", merged.get("triage_whiteness_score", pd.Series([pd.NA] * len(merged), index=merged.index))),
            errors="coerce",
        )
        out = pd.DataFrame(
            {
                "batch_variant": "original_batch_001",
                "batch_id": merged["batch_id"].fillna("").astype(str),
                "epic_id": merged["epic_id"].fillna("").astype(str),
                "epic_id_norm": merged["epic_id_norm"].fillna("").astype(str),
                "query": merged["query"].fillna("").astype(str),
                "execution_order": pd.to_numeric(merged["execution_order"], errors="coerce"),
                "new_execution_order": pd.Series([pd.NA] * len(merged), index=merged.index),
                "planned_best_depth_snr": pd.to_numeric(merged["planned_best_depth_snr"], errors="coerce"),
                "planned_n_events": pd.to_numeric(merged["planned_n_events"], errors="coerce"),
                "planned_n_periods_proposed": pd.to_numeric(merged["planned_n_periods_proposed"], errors="coerce"),
                "saved_triage_usable": merged["saved_triage_usable"],
                "saved_triage_whiteness_pvalue": pd.to_numeric(merged["saved_triage_whiteness_pvalue"], errors="coerce"),
                "saved_triage_step_score": pd.to_numeric(merged["saved_triage_step_score"], errors="coerce"),
                "saved_triage_score_global": pd.to_numeric(merged["saved_triage_score_global"], errors="coerce"),
                "saved_triage_whiteness_definition": merged["saved_triage_whiteness_definition"].fillna("").astype(str),
                "runtime_triage_status": merged["triage_status_pipeline"].fillna("").astype(str),
                "runtime_triage_usable": merged["triage_usable_pipeline"],
                "runtime_triage_why_not_usable": merged["triage_why_not_usable_pipeline"].fillna("").astype(str),
                "runtime_triage_whiteness_pvalue": runtime_pvalue,
                "runtime_triage_whiteness_log10_pvalue": pd.to_numeric(
                    merged.get("triage_whiteness_log10_pvalue", pd.Series([pd.NA] * len(merged), index=merged.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_statistic_abs_rho": pd.to_numeric(
                    merged.get("triage_whiteness_statistic_abs_rho", pd.Series([pd.NA] * len(merged), index=merged.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_z": pd.to_numeric(
                    merged.get("triage_whiteness_z", pd.Series([pd.NA] * len(merged), index=merged.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_mode": merged.get(
                    "triage_whiteness_mode", pd.Series([""] * len(merged), index=merged.index)
                ).fillna("").astype(str),
                "runtime_triage_whiteness_underflowed": merged.get(
                    "triage_whiteness_underflowed", pd.Series([False] * len(merged), index=merged.index)
                ),
                "runtime_triage_whiteness_score": pd.to_numeric(merged["triage_whiteness_score"], errors="coerce"),
                "runtime_triage_whiteness_definition": merged["triage_whiteness_definition"].fillna("").astype(str),
                "runtime_triage_step_score": pd.to_numeric(merged["triage_step_score"], errors="coerce"),
                "runtime_triage_score_global": pd.to_numeric(merged["triage_score_global"], errors="coerce"),
                "final_label": merged["label"].fillna("").astype(str),
                "final_label_reason": merged["label_reason"].fillna("").astype(str),
                "runtime_n_events": pd.to_numeric(merged["n_events"], errors="coerce"),
                "runtime_n_periods_proposed": pd.to_numeric(merged["n_periods_proposed"], errors="coerce"),
                "runtime_best_depth_snr": pd.to_numeric(merged["best_depth_snr"], errors="coerce"),
            }
        )
        return out

    def _normalize_patched(self, results_df: pd.DataFrame) -> pd.DataFrame:
        runtime_pvalue = pd.to_numeric(
            results_df.get("triage_whiteness_pvalue", results_df.get("triage_whiteness_score", pd.Series([pd.NA] * len(results_df), index=results_df.index))),
            errors="coerce",
        )
        out = pd.DataFrame(
            {
                "batch_variant": "patched_batch_001b",
                "batch_id": results_df.get("batch_id", pd.Series(["high_priority_batch_001b"] * len(results_df), index=results_df.index)).fillna("").astype(str),
                "epic_id": results_df["epic_id"].fillna("").astype(str),
                "epic_id_norm": results_df["epic_id_norm"].fillna("").astype(str),
                "query": results_df["query"].fillna("").astype(str),
                "execution_order": pd.to_numeric(results_df.get("execution_order", pd.Series([pd.NA] * len(results_df), index=results_df.index)), errors="coerce"),
                "new_execution_order": pd.to_numeric(results_df["new_execution_order"], errors="coerce"),
                "planned_best_depth_snr": pd.to_numeric(results_df["planned_best_depth_snr"], errors="coerce"),
                "planned_n_events": pd.to_numeric(results_df["planned_n_events"], errors="coerce"),
                "planned_n_periods_proposed": pd.to_numeric(results_df["planned_n_periods_proposed"], errors="coerce"),
                "saved_triage_usable": results_df["saved_triage_usable"],
                "saved_triage_whiteness_pvalue": pd.to_numeric(results_df["saved_triage_whiteness_pvalue"], errors="coerce"),
                "saved_triage_step_score": pd.to_numeric(results_df["saved_triage_step_score"], errors="coerce"),
                "saved_triage_score_global": pd.to_numeric(results_df["saved_triage_score_global"], errors="coerce"),
                "saved_triage_whiteness_definition": results_df["saved_triage_whiteness_definition"].fillna("").astype(str),
                "runtime_triage_status": results_df["triage_status_pipeline"].fillna("").astype(str),
                "runtime_triage_usable": results_df["triage_usable_pipeline"],
                "runtime_triage_why_not_usable": results_df["triage_why_not_usable_pipeline"].fillna("").astype(str),
                "runtime_triage_whiteness_pvalue": runtime_pvalue,
                "runtime_triage_whiteness_log10_pvalue": pd.to_numeric(
                    results_df.get("triage_whiteness_log10_pvalue", pd.Series([pd.NA] * len(results_df), index=results_df.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_statistic_abs_rho": pd.to_numeric(
                    results_df.get("triage_whiteness_statistic_abs_rho", pd.Series([pd.NA] * len(results_df), index=results_df.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_z": pd.to_numeric(
                    results_df.get("triage_whiteness_z", pd.Series([pd.NA] * len(results_df), index=results_df.index)),
                    errors="coerce",
                ),
                "runtime_triage_whiteness_mode": results_df.get(
                    "triage_whiteness_mode", pd.Series([""] * len(results_df), index=results_df.index)
                ).fillna("").astype(str),
                "runtime_triage_whiteness_underflowed": results_df.get(
                    "triage_whiteness_underflowed", pd.Series([False] * len(results_df), index=results_df.index)
                ),
                "runtime_triage_whiteness_score": pd.to_numeric(results_df["triage_whiteness_score"], errors="coerce"),
                "runtime_triage_whiteness_definition": results_df["triage_whiteness_definition"].fillna("").astype(str),
                "runtime_triage_step_score": pd.to_numeric(results_df["triage_step_score"], errors="coerce"),
                "runtime_triage_score_global": pd.to_numeric(results_df["triage_score_global"], errors="coerce"),
                "final_label": results_df["label"].fillna("").astype(str),
                "final_label_reason": results_df["label_reason"].fillna("").astype(str),
                "runtime_n_events": pd.to_numeric(results_df["n_events"], errors="coerce"),
                "runtime_n_periods_proposed": pd.to_numeric(results_df["n_periods_proposed"], errors="coerce"),
                "runtime_best_depth_snr": pd.to_numeric(results_df["best_depth_snr"], errors="coerce"),
            }
        )
        return out

    def _diagnosis_frame(self, original_df: pd.DataFrame, patched_df: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([original_df, patched_df], ignore_index=True, sort=False)
        combined["saved_triage_usable_bool"] = self._bool_series(combined["saved_triage_usable"])
        combined["runtime_triage_usable_bool"] = self._bool_series(combined["runtime_triage_usable"])
        combined["whiteness_definition_same"] = (
            combined["saved_triage_whiteness_definition"].fillna("").astype(str)
            == combined["runtime_triage_whiteness_definition"].fillna("").astype(str)
        )
        combined["step_score_delta"] = (
            pd.to_numeric(combined["runtime_triage_step_score"], errors="coerce")
            - pd.to_numeric(combined["saved_triage_step_score"], errors="coerce")
        )
        combined["score_global_delta"] = (
            pd.to_numeric(combined["runtime_triage_score_global"], errors="coerce")
            - pd.to_numeric(combined["saved_triage_score_global"], errors="coerce")
        )
        combined["whiteness_gap"] = (
            pd.to_numeric(combined["saved_triage_whiteness_pvalue"], errors="coerce")
            - pd.to_numeric(combined["runtime_triage_whiteness_pvalue"], errors="coerce")
        )
        combined["saved_official_whiteness_pass"] = (
            pd.to_numeric(combined["saved_triage_whiteness_pvalue"], errors="coerce").ge(self.WHITENESS_ALPHA)
        )
        combined["runtime_official_whiteness_fail"] = (
            pd.to_numeric(combined["runtime_triage_whiteness_pvalue"], errors="coerce").lt(self.WHITENESS_ALPHA)
        )
        combined["runtime_whiteness_gate_failed"] = (
            combined["runtime_official_whiteness_fail"]
            | combined["runtime_triage_why_not_usable"].fillna("").astype(str).str.lower().str.contains("whiteness", na=False)
            | combined["final_label_reason"].fillna("").astype(str).str.lower().str.contains("whiteness", na=False)
        )
        combined["step_score_exact_match"] = combined["step_score_delta"].fillna(999.0).abs().le(1e-12)
        combined["strong_saved_proxy_runtime_fail"] = (
            pd.to_numeric(combined["saved_triage_whiteness_pvalue"], errors="coerce").ge(0.95)
            & combined["runtime_official_whiteness_fail"]
        )
        combined["proxy_non_equivalence_flag"] = (
            combined["whiteness_definition_same"]
            & combined["step_score_exact_match"]
            & pd.to_numeric(combined["saved_triage_whiteness_pvalue"], errors="coerce").ge(0.95)
            & combined["runtime_official_whiteness_fail"]
        )
        preferred_front = [
            "batch_variant",
            "batch_id",
            "epic_id",
            "epic_id_norm",
            "query",
            "execution_order",
            "new_execution_order",
            "planned_best_depth_snr",
            "planned_n_events",
            "planned_n_periods_proposed",
            "saved_triage_usable",
            "saved_triage_whiteness_pvalue",
            "saved_triage_step_score",
            "saved_triage_score_global",
            "saved_triage_whiteness_definition",
            "runtime_triage_status",
            "runtime_triage_usable",
            "runtime_triage_why_not_usable",
            "runtime_triage_whiteness_pvalue",
            "runtime_triage_whiteness_log10_pvalue",
            "runtime_triage_whiteness_statistic_abs_rho",
            "runtime_triage_whiteness_z",
            "runtime_triage_whiteness_mode",
            "runtime_triage_whiteness_underflowed",
            "runtime_triage_whiteness_score",
            "runtime_triage_whiteness_definition",
            "runtime_triage_step_score",
            "runtime_triage_score_global",
            "final_label",
            "final_label_reason",
            "runtime_n_events",
            "runtime_n_periods_proposed",
            "runtime_best_depth_snr",
            "whiteness_definition_same",
            "step_score_exact_match",
            "whiteness_gap",
            "step_score_delta",
            "score_global_delta",
            "saved_official_whiteness_pass",
            "runtime_official_whiteness_fail",
            "runtime_whiteness_gate_failed",
            "strong_saved_proxy_runtime_fail",
            "proxy_non_equivalence_flag",
        ]
        remaining = [c for c in combined.columns if c not in preferred_front]
        return combined[preferred_front + remaining].sort_values(
            by=["batch_variant", "new_execution_order", "execution_order", "epic_id"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    def _overlap_diagnostic(self, diagnosis_df: pd.DataFrame) -> Dict[str, Any]:
        overlap = diagnosis_df["epic_id"].value_counts()
        overlap_epics = set(overlap.loc[overlap.eq(2)].index.astype(str))
        subset = diagnosis_df.loc[diagnosis_df["epic_id"].astype(str).isin(overlap_epics)].copy()
        if len(overlap_epics) == 0 or len(subset) == 0:
            return {
                "overlap_epics": 0,
                "same_runtime_whiteness": 0,
                "same_final_label": 0,
            }
        whiteness_same = 0
        label_same = 0
        for _, group in subset.groupby("epic_id", sort=False):
            if len(group) != 2:
                continue
            runtime_values = pd.to_numeric(group["runtime_triage_whiteness_score"], errors="coerce")
            labels = group["final_label"].fillna("").astype(str)
            if runtime_values.nunique(dropna=False) == 1:
                whiteness_same += 1
            if labels.nunique(dropna=False) == 1:
                label_same += 1
        return {
            "overlap_epics": int(len(overlap_epics)),
            "same_runtime_whiteness": int(whiteness_same),
            "same_final_label": int(label_same),
        }

    def _recommendation(self, diagnosis_df: pd.DataFrame) -> Dict[str, str]:
        total_rows = int(len(diagnosis_df))
        same_definition_fraction = float(diagnosis_df["whiteness_definition_same"].mean()) if total_rows > 0 else 0.0
        exact_step_fraction = float(diagnosis_df["step_score_exact_match"].mean()) if total_rows > 0 else 0.0
        strong_proxy_fail_fraction = float(diagnosis_df["strong_saved_proxy_runtime_fail"].mean()) if total_rows > 0 else 0.0
        runtime_whiteness_fail_fraction = float(diagnosis_df["runtime_whiteness_gate_failed"].mean()) if total_rows > 0 else 0.0
        final_reason_whiteness_fraction = (
            float(diagnosis_df["final_label_reason"].fillna("").astype(str).str.lower().str.contains("whiteness", na=False).mean())
            if total_rows > 0
            else 0.0
        )

        if (
            same_definition_fraction >= 0.95
            and exact_step_fraction >= 0.95
            and strong_proxy_fail_fraction >= 0.95
            and runtime_whiteness_fail_fraction >= 0.95
            and final_reason_whiteness_fraction >= 0.95
        ):
            return {
                "action": "C: redefine how whiteness is computed/interpreted",
                "note": (
                    "The saved and runtime whiteness definition strings agree, and the runtime step score matches the saved step score, "
                    "but the runtime whiteness p-value collapses to 0.0 while the saved p-value remains near 1.0. That is a non-equivalence or interpretation problem, "
                    "not just a threshold problem."
                ),
            }
        if runtime_whiteness_fail_fraction >= 0.95 and strong_proxy_fail_fraction < 0.50:
            return {
                "action": "B: adjust whiteness threshold",
                "note": "The runtime gate is dominating outcomes, but the saved and runtime whiteness evidence still looks broadly aligned enough to test a threshold calibration change first.",
            }
        return {
            "action": "A: keep current whiteness policy unchanged",
            "note": "The executed calibration batches do not show enough mismatch between saved and runtime whiteness behavior to justify a policy change.",
        }

    def run(
        self,
        *,
        original_results_csv: Path,
        patched_results_csv: Path,
        whiteness_csv: Path,
        out_dir: Path,
    ) -> Dict[str, Any]:
        original_results = self._prepare_original_results(Path(original_results_csv))
        patched_results = self._prepare_patched_results(Path(patched_results_csv))
        whiteness = self._prepare_whiteness(Path(whiteness_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        original_norm = self._normalize_original(original_results, whiteness)
        patched_norm = self._normalize_patched(patched_results)
        diagnosis_df = self._diagnosis_frame(original_norm, patched_norm)

        diagnosis_csv = out_dir / self.DEFAULT_DIAGNOSIS_CSV_NAME
        diagnosis_df.to_csv(diagnosis_csv, index=False)

        total_rows = int(len(diagnosis_df))
        same_definition_count = int(diagnosis_df["whiteness_definition_same"].sum())
        step_score_exact_match_count = int(diagnosis_df["step_score_exact_match"].sum())
        runtime_whiteness_zero_count = int(
            pd.to_numeric(diagnosis_df["runtime_triage_whiteness_score"], errors="coerce").eq(0.0).sum()
        )
        saved_pass_runtime_fail_count = int(
            (diagnosis_df["saved_official_whiteness_pass"] & diagnosis_df["runtime_official_whiteness_fail"]).sum()
        )
        runtime_gate_dominates = (
            "yes"
            if (
                total_rows > 0
                and diagnosis_df["runtime_whiteness_gate_failed"].all()
                and diagnosis_df["final_label_reason"].fillna("").astype(str).str.lower().str.contains("whiteness", na=False).all()
            )
            else "no"
        )
        same_definition_text = "yes" if same_definition_count == total_rows and total_rows > 0 else "no"
        scale_mismatch_text = (
            "yes"
            if (
                total_rows > 0
                and same_definition_count == total_rows
                and runtime_whiteness_zero_count == total_rows
                and pd.to_numeric(diagnosis_df["saved_triage_whiteness_pvalue"], errors="coerce").median() > 0.95
            )
            else "no"
        )
        proxy_nonequivalence_text = (
            "yes"
            if (
                total_rows > 0
                and scale_mismatch_text == "yes"
                and step_score_exact_match_count == total_rows
            )
            else "no"
        )
        overlap_diag = self._overlap_diagnostic(diagnosis_df)
        recommendation = self._recommendation(diagnosis_df)

        summary_rows: List[Dict[str, Any]] = []
        summary_rows.append(
            self._single_row(
                "metadata",
                "source_files",
                note=(
                    f"original_batch_001={original_results_csv}; patched_batch_001b={patched_results_csv}; "
                    f"saved_whiteness_proxies={whiteness_csv}"
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "official_whiteness_threshold",
                value_num=self.WHITENESS_ALPHA,
                note="Runtime whiteness failure is defined against the official p-value threshold used in label_reason and triage_why_not_usable_pipeline.",
            )
        )
        summary_rows.extend(
            self._count_rows("counts", "rows_by_batch_variant", diagnosis_df["batch_variant"].value_counts().sort_index(), total_rows)
        )
        summary_rows.extend(
            self._count_rows("counts", "final_label", diagnosis_df["final_label"].value_counts().sort_index(), total_rows)
        )
        summary_rows.extend(
            self._count_rows("counts", "final_label_reason", diagnosis_df["final_label_reason"].value_counts(), total_rows)
        )
        summary_rows.extend(self._quantile_rows("distribution", "saved_triage_whiteness_pvalue", diagnosis_df["saved_triage_whiteness_pvalue"]))
        summary_rows.extend(self._quantile_rows("distribution", "runtime_triage_whiteness_score", diagnosis_df["runtime_triage_whiteness_score"]))
        summary_rows.extend(self._quantile_rows("distribution", "saved_triage_step_score", diagnosis_df["saved_triage_step_score"]))
        summary_rows.extend(self._quantile_rows("distribution", "runtime_triage_step_score", diagnosis_df["runtime_triage_step_score"]))
        summary_rows.extend(self._quantile_rows("distribution", "saved_triage_score_global", diagnosis_df["saved_triage_score_global"]))
        summary_rows.extend(self._quantile_rows("distribution", "runtime_triage_score_global", diagnosis_df["runtime_triage_score_global"]))
        summary_rows.extend(self._quantile_rows("distribution", "planned_best_depth_snr", diagnosis_df["planned_best_depth_snr"]))
        summary_rows.extend(self._quantile_rows("distribution", "planned_n_events", diagnosis_df["planned_n_events"]))
        summary_rows.extend(self._quantile_rows("distribution", "planned_n_periods_proposed", diagnosis_df["planned_n_periods_proposed"]))
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "saved_proxy_official_pass_but_runtime_official_fail",
                count=saved_pass_runtime_fail_count,
                fraction=(float(saved_pass_runtime_fail_count) / float(total_rows)) if total_rows > 0 else float("nan"),
                note="Rows whose saved whiteness p-value passes the official threshold while the runtime whiteness score fails it.",
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "saved_and_runtime_whiteness_definition_same",
                value_text=same_definition_text,
                count=same_definition_count,
                fraction=(float(same_definition_count) / float(total_rows)) if total_rows > 0 else float("nan"),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "runtime_step_score_exactly_matches_saved_step_score",
                count=step_score_exact_match_count,
                fraction=(float(step_score_exact_match_count) / float(total_rows)) if total_rows > 0 else float("nan"),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "runtime_whiteness_zero_count",
                count=runtime_whiteness_zero_count,
                fraction=(float(runtime_whiteness_zero_count) / float(total_rows)) if total_rows > 0 else float("nan"),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "runtime_whiteness_gate_dominates_all_other_evidence",
                value_text=runtime_gate_dominates,
                note=(
                    "The runtime gate is treated as dominating if all executed rows fail runtime whiteness and all final label reasons point to whiteness, "
                    "despite n_events>0 and operationally clean execution."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "saved_and_runtime_whiteness_on_different_scales_or_non_equivalent",
                value_text=scale_mismatch_text,
                note=(
                    "Definition text agreement with universal runtime p-value collapse to 0.0 indicates the saved and runtime whiteness values are not behaving as interchangeable measures."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "evidence_of_calibration_drift_or_proxy_non_equivalence",
                value_text=proxy_nonequivalence_text,
                note=(
                    "This flag is set when the same named whiteness definition is reported, the runtime step score exactly matches the saved step score, "
                    "yet the runtime whiteness p-value fails universally while the saved p-value remains strong."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "overlapping_epics_repeated_across_both_batches",
                count=overlap_diag["overlap_epics"],
                fraction=(float(overlap_diag["overlap_epics"]) / float(total_rows)) if total_rows > 0 else float("nan"),
                note=(
                    f"same_runtime_whiteness_score_for_overlap={overlap_diag['same_runtime_whiteness']}; "
                    f"same_final_label_for_overlap={overlap_diag['same_final_label']}"
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "recommendation",
                "primary_scientific_action",
                value_text=recommendation["action"],
                note=recommendation["note"],
            )
        )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME
        summary_df.to_csv(summary_csv, index=False)

        return {
            "diagnosis_csv": str(diagnosis_csv),
            "summary_csv": str(summary_csv),
            "rows_total": total_rows,
            "saved_proxy_pass_runtime_fail_count": saved_pass_runtime_fail_count,
            "same_definition_count": same_definition_count,
            "step_score_exact_match_count": step_score_exact_match_count,
            "runtime_whiteness_zero_count": runtime_whiteness_zero_count,
            "recommendation": recommendation["action"],
        }
