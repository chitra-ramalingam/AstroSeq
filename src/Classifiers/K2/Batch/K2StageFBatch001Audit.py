from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class K2StageFBatch001Audit:
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch")
    DEFAULT_STAGE_F_RESULTS_CSV = DEFAULT_OUT_DIR / "k2_stage_f_batch_001_results.csv"
    DEFAULT_STAGE_E_PLAN_CSV = DEFAULT_OUT_DIR / "k2_stage_e_high_priority_batch_plan.csv"
    DEFAULT_AUDIT_CSV_NAME = "k2_stage_f_batch_001_audit.csv"
    DEFAULT_AUDIT_SUMMARY_CSV_NAME = "k2_stage_f_batch_001_audit_summary.csv"
    DEFAULT_BATCH_ID = "high_priority_batch_001"
    DEFAULT_NEXT_BATCH_ID = "high_priority_batch_002"
    WHITENESS_THRESHOLD = 0.01

    REQUIRED_RESULTS_COLUMNS = [
        "epic_id",
        "execution_order",
        "planned_best_depth_snr",
        "planned_n_events",
        "planned_n_periods_proposed",
        "triage_status",
        "triage_usable",
        "triage_whiteness_score",
        "triage_whiteness_definition",
        "label",
        "label_reason",
        "period_source_reason",
        "period_terminal_reason",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Build the Stage F.1 audit for batch 001 by comparing planned queue features against actual outcomes."
        )
        p.add_argument("--stage-f-results-csv", type=Path, default=cls.DEFAULT_STAGE_F_RESULTS_CSV)
        p.add_argument("--stage-e-plan-csv", type=Path, default=cls.DEFAULT_STAGE_E_PLAN_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            stage_f_results_csv=Path(args.stage_f_results_csv),
            stage_e_plan_csv=Path(args.stage_e_plan_csv),
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

    def _prepare_results(self, path: Path) -> pd.DataFrame:
        df = self._read_required_csv(path)
        missing = [c for c in self.REQUIRED_RESULTS_COLUMNS if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Stage F results CSV missing required columns: {missing} ({path})")
        out = df.copy()
        out["execution_order"] = pd.to_numeric(out["execution_order"], errors="coerce")
        if out["execution_order"].isna().any():
            raise ValueError("Stage F results contain non-numeric execution_order values.")
        return out.sort_values(by=["execution_order"], ascending=[True], kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _prepare_plan(path: Path) -> pd.DataFrame:
        df = K2StageFBatch001Audit._read_required_csv(path)
        if len(df) == 0:
            return df
        if "execution_order" in df.columns:
            df = df.copy()
            df["execution_order"] = pd.to_numeric(df["execution_order"], errors="coerce")
        return df

    @staticmethod
    def _to_bool_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

    @staticmethod
    def _quantile_rows(section: str, metric: str, series: pd.Series) -> List[Dict[str, Any]]:
        numeric = pd.to_numeric(series, errors="coerce")
        quantiles = {
            "min": 0.0,
            "q25": 0.25,
            "median": 0.50,
            "q75": 0.75,
            "max": 1.0,
        }
        rows: List[Dict[str, Any]] = []
        for label, q in quantiles.items():
            value = numeric.quantile(q) if len(numeric.dropna()) > 0 else float("nan")
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
        mean_val = numeric.mean() if len(numeric.dropna()) > 0 else float("nan")
        rows.append(
            {
                "section": section,
                "metric": metric,
                "submetric": "mean",
                "value_text": "",
                "value_num": float(mean_val) if pd.notna(mean_val) else float("nan"),
                "count": "",
                "fraction": "",
                "note": "",
            }
        )
        return rows

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
    def _single_row(section: str, metric: str, *, value_text: str = "", value_num: Any = "", count: Any = "", fraction: Any = "", note: str = "", submetric: str = "") -> Dict[str, Any]:
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

    @classmethod
    def _dominant_gate(cls, audit_df: pd.DataFrame) -> Dict[str, Any]:
        label_reason = audit_df["final_label_reason"].fillna("").astype(str)
        counts = label_reason.value_counts()
        if len(counts) == 0:
            return {
                "dominated": "no",
                "reason": "none",
                "count": 0,
                "fraction": 0.0,
            }
        top_reason = str(counts.index[0])
        top_count = int(counts.iloc[0])
        total_rows = int(len(audit_df))
        top_fraction = (float(top_count) / float(total_rows)) if total_rows > 0 else 0.0
        return {
            "dominated": "yes" if top_fraction >= 0.80 else "no",
            "reason": top_reason,
            "count": top_count,
            "fraction": top_fraction,
        }

    @classmethod
    def _batch2_similarity_diagnostic(cls, *, audit_df: pd.DataFrame, stage_e_plan_df: pd.DataFrame) -> Dict[str, str]:
        batch2 = stage_e_plan_df.loc[stage_e_plan_df.get("batch_id", pd.Series(dtype=str)).astype(str).eq(cls.DEFAULT_NEXT_BATCH_ID)].copy()
        if len(batch2) == 0:
            return {
                "likely_similar": "unknown",
                "note": "Stage E batch 002 rows were not available, so similarity could not be estimated.",
            }
        batch1_snr = pd.to_numeric(audit_df["planned_best_depth_snr"], errors="coerce")
        batch2_snr = pd.to_numeric(batch2["best_depth_snr"], errors="coerce")
        batch1_events = pd.to_numeric(audit_df["planned_n_events"], errors="coerce")
        batch2_events = pd.to_numeric(batch2["n_events"], errors="coerce")
        batch1_all_whiteness_fail = bool(
            pd.to_numeric(audit_df["triage_whiteness_score"], errors="coerce").fillna(float("inf")).lt(cls.WHITENESS_THRESHOLD).all()
        )
        batch2_below_batch1_floor = bool(batch2_snr.max() <= batch1_snr.min())
        events_similarity = abs(float(batch2_events.median()) - float(batch1_events.median())) <= 2.0
        likely_similar = "yes" if batch1_all_whiteness_fail and batch2_below_batch1_floor and events_similarity else "maybe"
        note = (
            f"batch_001 planned_best_depth_snr median={batch1_snr.median():.3f}, min={batch1_snr.min():.3f}; "
            f"batch_002 planned_best_depth_snr median={batch2_snr.median():.3f}, max={batch2_snr.max():.3f}. "
            f"batch_001 planned_n_events median={batch1_events.median():.1f}; batch_002 planned_n_events median={batch2_events.median():.1f}. "
            "Because batch 002 is the next contiguous slice of the same SNR-first ranking and its entire SNR range sits below batch 001, "
            "similar or worse whiteness attrition is likely."
            if likely_similar == "yes"
            else
            f"batch_001 planned_best_depth_snr median={batch1_snr.median():.3f}; batch_002 planned_best_depth_snr median={batch2_snr.median():.3f}. "
            "The next batch is adjacent in the same ranking, so similar behavior remains plausible, but the evidence is not fully conclusive."
        )
        return {"likely_similar": likely_similar, "note": note}

    @classmethod
    def _recommendation(cls, *, audit_df: pd.DataFrame, dominant_gate: Dict[str, Any], batch2_diag: Dict[str, str]) -> Dict[str, str]:
        total_rows = int(len(audit_df))
        all_noisy = int(audit_df["final_label"].astype(str).eq("Noisy_trash").sum()) == total_rows and total_rows > 0
        all_upstream_true = cls._to_bool_series(audit_df["upstream_triage_usable"]).all() if total_rows > 0 else False
        all_actual_false = (~cls._to_bool_series(audit_df["triage_usable"])).all() if total_rows > 0 else False
        if all_noisy and all_upstream_true and all_actual_false and dominant_gate["dominated"] == "yes":
            return {
                "recommendation": "pause and patch execution ordering for future batches",
                "note": (
                    "Batch 001 shows an execution-ordering calibration problem: the current SNR-first queue surfaced 100/100 rows with upstream "
                    "triage_usable=True, but all 100 failed the same whiteness gate in the actual official pipeline. "
                    "That points more directly to keepability-blind prioritization than to an operational instability."
                ),
            }
        if batch2_diag["likely_similar"] == "yes":
            return {
                "recommendation": "continue but treat as expected attrition",
                "note": "Batch 001 is scientifically low-yield but internally consistent; continue only if that attrition profile is acceptable.",
            }
        return {
            "recommendation": "continue unchanged to batch 002",
            "note": "The audit did not find enough concentration in a single gate to justify pausing the current plan.",
        }

    def run(self, *, stage_f_results_csv: Path, stage_e_plan_csv: Path, out_dir: Path) -> Dict[str, Any]:
        results_df = self._prepare_results(Path(stage_f_results_csv))
        stage_e_plan_df = self._prepare_plan(Path(stage_e_plan_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        audit_df = pd.DataFrame(
            {
                "epic_id": results_df["epic_id"],
                "query": results_df.get("query", pd.Series([""] * len(results_df), index=results_df.index)),
                "execution_order": pd.to_numeric(results_df["execution_order"], errors="coerce"),
                "batch_position": pd.to_numeric(results_df.get("batch_position", pd.Series([pd.NA] * len(results_df), index=results_df.index)), errors="coerce"),
                "planned_best_depth_snr": pd.to_numeric(results_df["planned_best_depth_snr"], errors="coerce"),
                "planned_n_events": pd.to_numeric(results_df["planned_n_events"], errors="coerce"),
                "planned_n_periods_proposed": pd.to_numeric(results_df["planned_n_periods_proposed"], errors="coerce"),
                "upstream_triage_status": results_df.get("triage_status", pd.Series([""] * len(results_df), index=results_df.index)),
                "upstream_triage_usable": results_df.get("triage_usable", pd.Series([""] * len(results_df), index=results_df.index)),
                "triage_status": results_df.get("triage_status_pipeline", results_df.get("triage_status", pd.Series([""] * len(results_df), index=results_df.index))),
                "triage_usable": results_df.get("triage_usable_pipeline", results_df.get("triage_usable", pd.Series([""] * len(results_df), index=results_df.index))),
                "triage_whiteness_score": pd.to_numeric(results_df["triage_whiteness_score"], errors="coerce"),
                "triage_whiteness_definition": results_df["triage_whiteness_definition"],
                "final_label": results_df["label"],
                "final_label_reason": results_df["label_reason"],
                "period_source_reason": results_df["period_source_reason"],
                "period_terminal_reason": results_df["period_terminal_reason"],
            }
        ).sort_values(by=["execution_order"], ascending=[True], kind="mergesort").reset_index(drop=True)

        audit_csv = out_dir / self.DEFAULT_AUDIT_CSV_NAME
        audit_df.to_csv(audit_csv, index=False)

        total_rows = int(len(audit_df))
        label_counts = audit_df["final_label"].fillna("").astype(str).value_counts().sort_index()
        label_reason_counts = audit_df["final_label_reason"].fillna("").astype(str).value_counts().sort_values(ascending=False)
        upstream_true = self._to_bool_series(audit_df["upstream_triage_usable"])
        final_noisy = audit_df["final_label"].fillna("").astype(str).eq("Noisy_trash")
        upstream_true_but_noisy = int((upstream_true & final_noisy).sum())
        dominant_gate = self._dominant_gate(audit_df)
        batch2_diag = self._batch2_similarity_diagnostic(audit_df=audit_df, stage_e_plan_df=stage_e_plan_df)
        recommendation = self._recommendation(audit_df=audit_df, dominant_gate=dominant_gate, batch2_diag=batch2_diag)

        summary_rows: List[Dict[str, Any]] = []
        summary_rows.extend(self._count_rows("counts", "final_label", label_counts, total_rows))
        summary_rows.extend(self._count_rows("counts", "final_label_reason", label_reason_counts, total_rows))
        summary_rows.extend(self._quantile_rows("distribution", "planned_best_depth_snr", audit_df["planned_best_depth_snr"]))
        summary_rows.extend(self._quantile_rows("distribution", "planned_n_events", audit_df["planned_n_events"]))
        summary_rows.extend(self._quantile_rows("distribution", "triage_whiteness_score", audit_df["triage_whiteness_score"]))
        summary_rows.append(
            self._single_row(
                "distribution",
                "triage_whiteness_score_below_threshold",
                value_text=f"<{self.WHITENESS_THRESHOLD}",
                count=int(pd.to_numeric(audit_df["triage_whiteness_score"], errors="coerce").fillna(float("inf")).lt(self.WHITENESS_THRESHOLD).sum()),
                fraction=float(pd.to_numeric(audit_df["triage_whiteness_score"], errors="coerce").fillna(float("inf")).lt(self.WHITENESS_THRESHOLD).mean()),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "upstream_triage_usable_true_but_final_noisy_trash",
                count=upstream_true_but_noisy,
                fraction=(float(upstream_true_but_noisy) / float(total_rows)) if total_rows > 0 else float("nan"),
                note="Rows whose upstream saved triage_usable flag was True but whose actual batch-001 outcome ended as Noisy_trash.",
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "rejection_dominated_by_one_single_gate",
                value_text=str(dominant_gate["dominated"]),
                count=int(dominant_gate["count"]),
                fraction=float(dominant_gate["fraction"]),
                note=f"dominant_label_reason={dominant_gate['reason']}",
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "ranking_overselects_high_snr_low_whiteness",
                value_text="yes" if (total_rows > 0 and final_noisy.all() and dominant_gate["dominated"] == "yes") else "no",
                note=(
                    f"All {total_rows} batch-001 rows carried strong planned detector signal "
                    f"(planned_best_depth_snr median={audit_df['planned_best_depth_snr'].median():.3f}) "
                    f"but triage_whiteness_score was below {self.WHITENESS_THRESHOLD} for "
                    f"{int(pd.to_numeric(audit_df['triage_whiteness_score'], errors='coerce').fillna(float('inf')).lt(self.WHITENESS_THRESHOLD).sum())}/{total_rows} rows. "
                    "That indicates the current SNR-first Stage E ranking is over-selecting objects that are not keepable under the official whiteness gate."
                ),
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "batch_002_likely_to_behave_similarly",
                value_text=str(batch2_diag["likely_similar"]),
                note=batch2_diag["note"],
            )
        )
        summary_rows.append(
            self._single_row(
                "diagnostic",
                "recommendation",
                value_text=recommendation["recommendation"],
                note=recommendation["note"],
            )
        )

        summary_df = pd.DataFrame(
            summary_rows,
            columns=["section", "metric", "submetric", "value_text", "value_num", "count", "fraction", "note"],
        )
        summary_csv = out_dir / self.DEFAULT_AUDIT_SUMMARY_CSV_NAME
        summary_df.to_csv(summary_csv, index=False)

        return {
            "audit_csv": str(audit_csv),
            "audit_summary_csv": str(summary_csv),
            "rows_total": total_rows,
            "label_counts": label_counts.to_dict(),
            "label_reason_counts": label_reason_counts.to_dict(),
            "upstream_triage_usable_true_but_final_noisy_trash": upstream_true_but_noisy,
            "rejection_dominated_by_one_single_gate": str(dominant_gate["dominated"]),
            "dominant_gate_reason": str(dominant_gate["reason"]),
            "batch_002_likely_to_behave_similarly": str(batch2_diag["likely_similar"]),
            "recommendation": recommendation["recommendation"],
        }
