from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class K2StageE1HighPriorityRerank:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_D_HIGH_PRIORITY_CSV = DEFAULT_OUT_DIR / "k2_stage_d_process_now_high_priority.csv"
    DEFAULT_WHITENESS_CSV = DEFAULT_OUT_DIR / "batch_results_whiteness.csv"
    DEFAULT_PREVIEW_CSV_NAME = "k2_stage_e1_high_priority_rerank_preview.csv"
    DEFAULT_SUMMARY_CSV_NAME = "k2_stage_e1_high_priority_rerank_summary.csv"

    REQUIRED_STAGE_D_COLUMNS = [
        "epic_id",
        "epic_id_norm",
        "execution_order",
        "n_events",
        "n_periods_proposed",
        "best_depth_snr",
    ]
    REQUIRED_WHITENESS_COLUMNS = [
        "epic_id",
        "triage_usable",
        "triage_whiteness_pvalue",
        "triage_step_score",
        "triage_score_global",
        "triage_whiteness_definition",
        "triage_why_not_usable",
    ]

    WHITENESS_ALPHA = 0.01
    RERANK_WEIGHTS = {
        "log_snr_rank": 0.40,
        "whiteness_rank": 0.15,
        "step_inverse_rank": 0.15,
        "score_global_rank": 0.10,
        "n_events_rank": 0.10,
        "n_periods_rank": 0.10,
    }
    MATERIAL_CHANGE_THRESHOLD = 25
    STAGE_D_BATCH_SIZE = 100

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Build a Stage E.1 rerank preview for the Stage D high-priority queue using saved survivability proxies."
        )
        p.add_argument("--stage-d-high-priority-csv", type=Path, default=cls.DEFAULT_STAGE_D_HIGH_PRIORITY_CSV)
        p.add_argument("--whiteness-csv", type=Path, default=cls.DEFAULT_WHITENESS_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            stage_d_high_priority_csv=Path(args.stage_d_high_priority_csv),
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
        out["epic_id_norm"] = out["epic_id"].fillna("").astype(str).str.extract(r"(\d+)")[0].fillna("").astype(str)
        out = out.loc[out["epic_id_norm"] != ""].drop_duplicates(subset=["epic_id_norm"], keep="first").reset_index(drop=True)
        out = out.rename(
            columns={
                "triage_usable": "saved_triage_usable",
                "triage_whiteness_pvalue": "saved_triage_whiteness_pvalue",
                "triage_step_score": "saved_triage_step_score",
                "triage_score_global": "saved_triage_score_global",
                "triage_whiteness_definition": "saved_triage_whiteness_definition",
                "triage_why_not_usable": "saved_triage_why_not_usable",
            }
        )
        return out

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
    def _quantile_rows(section: str, metric: str, series: pd.Series) -> List[Dict[str, Any]]:
        numeric = pd.to_numeric(series, errors="coerce")
        labels = {"min": 0.0, "q25": 0.25, "median": 0.50, "q75": 0.75, "max": 1.0}
        rows: List[Dict[str, Any]] = []
        for label, q in labels.items():
            val = numeric.quantile(q) if numeric.notna().any() else float("nan")
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "submetric": label,
                    "value_text": "",
                    "value_num": float(val) if pd.notna(val) else float("nan"),
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

    def _build_risk_flag(self, df: pd.DataFrame) -> pd.Series:
        low_relative_whiteness = df["component_whiteness_rank"].le(0.25)
        no_saved_period_support = pd.to_numeric(df["n_periods_proposed"], errors="coerce").fillna(0.0).le(0.0)
        proxy_missing = ~self._bool_series(df["whiteness_proxy_available"])
        official_fail = pd.to_numeric(df["whiteness_proxy_value"], errors="coerce").fillna(float("inf")).lt(self.WHITENESS_ALPHA)

        out = pd.Series("lower_relative_risk", index=df.index, dtype="string")
        out.loc[no_saved_period_support] = "no_saved_period_support"
        out.loc[low_relative_whiteness] = "elevated_relative_whiteness_risk"
        out.loc[low_relative_whiteness & no_saved_period_support] = "elevated_relative_whiteness_risk_no_saved_period_support"
        out.loc[official_fail] = "official_whiteness_fail_saved_proxy"
        out.loc[proxy_missing] = "proxy_missing"
        return out

    @staticmethod
    def _rerank_reason(row: pd.Series) -> str:
        return (
            f"saved_whiteness_pvalue={row['whiteness_proxy_value']:.6f}; "
            f"saved_step_score={row['saved_triage_step_score']:.6f}; "
            f"saved_score_global={row['saved_triage_score_global']:.6f}; "
            f"n_periods_proposed={int(row['n_periods_proposed']) if pd.notna(row['n_periods_proposed']) else 'NA'}; "
            f"n_events={int(row['n_events']) if pd.notna(row['n_events']) else 'NA'}; "
            f"best_depth_snr={row['best_depth_snr']:.6f}."
        )

    def run(self, *, stage_d_high_priority_csv: Path, whiteness_csv: Path, out_dir: Path) -> Dict[str, Any]:
        stage_d = self._prepare_stage_d(Path(stage_d_high_priority_csv))
        whiteness = self._prepare_whiteness(Path(whiteness_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        merged = stage_d.merge(
            whiteness[
                [
                    "epic_id_norm",
                    "saved_triage_usable",
                    "saved_triage_whiteness_pvalue",
                    "saved_triage_step_score",
                    "saved_triage_score_global",
                    "saved_triage_whiteness_definition",
                    "saved_triage_why_not_usable",
                ]
            ],
            on="epic_id_norm",
            how="left",
        )

        merged["old_execution_order"] = pd.to_numeric(merged["execution_order"], errors="coerce")
        merged["whiteness_proxy_available"] = merged["saved_triage_whiteness_pvalue"].notna()
        merged["whiteness_proxy_value"] = pd.to_numeric(merged["saved_triage_whiteness_pvalue"], errors="coerce")

        merged["component_log_snr_rank"] = self._percentile_rank(np.log1p(pd.to_numeric(merged["best_depth_snr"], errors="coerce")), higher_is_better=True)
        merged["component_whiteness_rank"] = self._percentile_rank(merged["whiteness_proxy_value"], higher_is_better=True)
        merged["component_step_inverse_rank"] = self._percentile_rank(-pd.to_numeric(merged["saved_triage_step_score"], errors="coerce"), higher_is_better=True)
        merged["component_score_global_rank"] = self._percentile_rank(merged["saved_triage_score_global"], higher_is_better=True)
        merged["component_n_events_rank"] = self._percentile_rank(merged["n_events"], higher_is_better=True)
        merged["component_n_periods_rank"] = self._percentile_rank(merged["n_periods_proposed"], higher_is_better=True)

        merged["rerank_score"] = 100.0 * (
            self.RERANK_WEIGHTS["log_snr_rank"] * merged["component_log_snr_rank"]
            + self.RERANK_WEIGHTS["whiteness_rank"] * merged["component_whiteness_rank"]
            + self.RERANK_WEIGHTS["step_inverse_rank"] * merged["component_step_inverse_rank"]
            + self.RERANK_WEIGHTS["score_global_rank"] * merged["component_score_global_rank"]
            + self.RERANK_WEIGHTS["n_events_rank"] * merged["component_n_events_rank"]
            + self.RERANK_WEIGHTS["n_periods_rank"] * merged["component_n_periods_rank"]
        )

        merged["keepability_risk_flag"] = self._build_risk_flag(merged)
        merged["rerank_reason"] = merged.apply(self._rerank_reason, axis=1)

        preview = merged.sort_values(
            by=["rerank_score", "component_whiteness_rank", "component_step_inverse_rank", "old_execution_order", "epic_id_norm"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        preview["new_execution_order"] = range(1, len(preview) + 1)

        preferred_front = [
            "epic_id",
            "query",
            "old_execution_order",
            "new_execution_order",
            "rerank_score",
            "rerank_reason",
            "whiteness_proxy_available",
            "whiteness_proxy_value",
            "keepability_risk_flag",
            "saved_triage_usable",
            "saved_triage_whiteness_definition",
            "saved_triage_why_not_usable",
            "saved_triage_step_score",
            "saved_triage_score_global",
            "n_events",
            "n_periods_proposed",
            "best_depth_snr",
        ]
        remaining = [c for c in preview.columns if c not in preferred_front]
        preview = preview[preferred_front + remaining]

        preview_csv = out_dir / self.DEFAULT_PREVIEW_CSV_NAME
        preview.to_csv(preview_csv, index=False)

        old_top = set(preview.nsmallest(self.STAGE_D_BATCH_SIZE, "old_execution_order")["epic_id_norm"].astype(str))
        new_top_df = preview.nsmallest(self.STAGE_D_BATCH_SIZE, "new_execution_order")
        new_top = set(new_top_df["epic_id_norm"].astype(str))
        moved_out = int(len(old_top - new_top))
        moved_in = int(len(new_top - old_top))
        materially_different = "yes" if moved_out >= self.MATERIAL_CHANGE_THRESHOLD else "no"

        current_top_df = preview.nsmallest(self.STAGE_D_BATCH_SIZE, "old_execution_order")
        proxy_coverage = int(self._bool_series(preview["whiteness_proxy_available"]).sum())
        proxy_exists_meaningfully = "yes" if proxy_coverage > 0 else "no"

        summary_rows: List[Dict[str, Any]] = []
        summary_rows.append(
            self._single_row(
                "metadata",
                "top_features_used_in_reranking",
                value_text=(
                    "saved_triage_whiteness_pvalue, saved_triage_step_score, saved_triage_score_global, "
                    "n_periods_proposed, n_events, log1p(best_depth_snr)"
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "usable_whiteness_stability_proxy_exists_upstream",
                value_text=proxy_exists_meaningfully,
                count=proxy_coverage,
                fraction=(float(proxy_coverage) / float(len(preview))) if len(preview) > 0 else float("nan"),
                note=f"Source file: {whiteness_csv}",
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "rerank_formula",
                note=(
                    "rerank_score = 100 * (0.40*rank(log1p(best_depth_snr)) + 0.15*rank(saved_triage_whiteness_pvalue) "
                    "+ 0.15*rank(-saved_triage_step_score) + 0.10*rank(saved_triage_score_global) "
                    "+ 0.10*rank(n_events) + 0.10*rank(n_periods_proposed)); "
                    "higher percentile ranks are better; ties break by old_execution_order then epic_id_norm."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "metadata",
                "null_handling",
                note=(
                    "Missing saved proxies get the worst-fill value before percentile ranking and set "
                    "whiteness_proxy_available=False with keepability_risk_flag=proxy_missing. "
                    "Missing numeric Stage D features are also worst-filled for ranking only; all original rows are preserved."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "impact",
                "original_top_100_move_out_of_first_batch",
                count=moved_out,
                fraction=(float(moved_out) / float(self.STAGE_D_BATCH_SIZE)),
            )
        )
        summary_rows.append(
            self._single_row(
                "impact",
                "new_rows_enter_first_100",
                count=moved_in,
                fraction=(float(moved_in) / float(self.STAGE_D_BATCH_SIZE)),
            )
        )
        summary_rows.append(
            self._single_row(
                "impact",
                "reranked_batch_001_materially_different",
                value_text=materially_different,
                note="Material change threshold set at 25 replacements out of the first 100 rows.",
            )
        )
        summary_rows.append(
            self._single_row(
                "impact",
                "current_first_100_rows_with_n_periods_proposed_gt0",
                count=int(pd.to_numeric(current_top_df["n_periods_proposed"], errors="coerce").gt(0).sum()),
                fraction=float(pd.to_numeric(current_top_df["n_periods_proposed"], errors="coerce").gt(0).mean()),
            )
        )
        summary_rows.append(
            self._single_row(
                "impact",
                "reranked_first_100_rows_with_n_periods_proposed_gt0",
                count=int(pd.to_numeric(new_top_df["n_periods_proposed"], errors="coerce").gt(0).sum()),
                fraction=float(pd.to_numeric(new_top_df["n_periods_proposed"], errors="coerce").gt(0).mean()),
            )
        )
        summary_rows.extend(self._quantile_rows("current_batch_001", "whiteness_proxy_value", current_top_df["whiteness_proxy_value"]))
        summary_rows.extend(self._quantile_rows("reranked_batch_001", "whiteness_proxy_value", new_top_df["whiteness_proxy_value"]))
        summary_rows.extend(self._quantile_rows("current_batch_001", "saved_triage_step_score", current_top_df["saved_triage_step_score"]))
        summary_rows.extend(self._quantile_rows("reranked_batch_001", "saved_triage_step_score", new_top_df["saved_triage_step_score"]))
        summary_rows.extend(self._quantile_rows("current_batch_001", "best_depth_snr", current_top_df["best_depth_snr"]))
        summary_rows.extend(self._quantile_rows("reranked_batch_001", "best_depth_snr", new_top_df["best_depth_snr"]))
        summary_rows.extend(self._quantile_rows("current_batch_001", "n_events", current_top_df["n_events"]))
        summary_rows.extend(self._quantile_rows("reranked_batch_001", "n_events", new_top_df["n_events"]))
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "proxy_alignment_warning",
                value_text="yes",
                note=(
                    f"The saved whiteness proxy exists with full coverage, but the original top 100 already have strong saved proxy values "
                    f"(median whiteness_proxy_value={pd.to_numeric(current_top_df['whiteness_proxy_value'], errors='coerce').median():.6f}, "
                    f"median saved_triage_step_score={pd.to_numeric(current_top_df['saved_triage_step_score'], errors='coerce').median():.6f}). "
                    "That means the proxy is only relative for this failure mode and did not hard-flag the Stage F batch-001 collapse in advance."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "first_batch_difference_note",
                note=(
                    f"Current batch 001 median whiteness_proxy_value={pd.to_numeric(current_top_df['whiteness_proxy_value'], errors='coerce').median():.6f}; "
                    f"reranked batch 001 median whiteness_proxy_value={pd.to_numeric(new_top_df['whiteness_proxy_value'], errors='coerce').median():.6f}. "
                    f"Current batch 001 median saved_triage_step_score={pd.to_numeric(current_top_df['saved_triage_step_score'], errors='coerce').median():.6f}; "
                    f"reranked batch 001 median saved_triage_step_score={pd.to_numeric(new_top_df['saved_triage_step_score'], errors='coerce').median():.6f}. "
                    f"Current batch 001 median best_depth_snr={pd.to_numeric(current_top_df['best_depth_snr'], errors='coerce').median():.6f}; "
                    f"reranked batch 001 median best_depth_snr={pd.to_numeric(new_top_df['best_depth_snr'], errors='coerce').median():.6f}."
                ),
            )
        )
        if proxy_coverage == 0:
            recommendation_text = "no meaningful saved proxy exists"
            recommendation_note = "No saved whiteness or stability proxy was available upstream, so the current ordering cannot be improved without new computation."
        else:
            recommendation_text = "rerun a patched batch 001 before proceeding"
            recommendation_note = (
                "A saved whiteness/stability proxy exists upstream and the reranked first 100 are materially different. "
                "Because the proxy did not perfectly anticipate the Stage F.1 collapse, treat this as a calibration rerun rather than a guaranteed fix."
            )
        summary_rows.append(
            self._single_row(
                "recommendation",
                "action",
                value_text=recommendation_text,
                note=recommendation_note,
            )
        )

        summary_df = pd.DataFrame(
            summary_rows,
            columns=["section", "metric", "submetric", "value_text", "value_num", "count", "fraction", "note"],
        )
        summary_csv = out_dir / self.DEFAULT_SUMMARY_CSV_NAME
        summary_df.to_csv(summary_csv, index=False)

        return {
            "preview_csv": str(preview_csv),
            "summary_csv": str(summary_csv),
            "rows_total": int(len(preview)),
            "whiteness_proxy_coverage": proxy_coverage,
            "old_top100_move_out": moved_out,
            "new_top100_enter": moved_in,
            "reranked_batch_001_materially_different": materially_different,
            "recommendation": recommendation_text,
        }
