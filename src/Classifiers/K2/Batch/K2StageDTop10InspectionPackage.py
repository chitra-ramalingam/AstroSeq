from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2StageDDeeperEvalRunner import K2StageDDeeperEvalRunner
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2_NoiseHandler
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


class K2StageDTop10InspectionPackage:
    DEFAULT_INPUT_CSV = Path("k2_stage_d_pass_ranked.csv")
    DEFAULT_FULL_RESULTS_CSV = Path("k2_stage_d_tier_a_results.csv")
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\stage_d_top10")
    DEFAULT_INDEX_CSV = Path("k2_stage_d_top10_inspection_index.csv")
    DEFAULT_EPICS_DIR = Path(r"plots\k2_batch\epics")
    TOP_N = 10

    INDEX_COLUMNS = [
        "epic_id",
        "best_period_days",
        "period_support_count",
        "folded_depth_consistency",
        "duration_consistency",
        "odd_even_depth_delta",
        "folded_plot_path",
        "event_overlay_path",
        "odd_even_plot_path",
        "summary_json_path",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Create Stage D top-candidate inspection plots and metadata.")
        p.add_argument("--input-csv", type=Path, default=cls.DEFAULT_INPUT_CSV)
        p.add_argument("--full-results-csv", type=Path, default=cls.DEFAULT_FULL_RESULTS_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        p.add_argument("--index-csv", type=Path, default=cls.DEFAULT_INDEX_CSV)
        p.add_argument("--epics-dir", type=Path, default=cls.DEFAULT_EPICS_DIR)
        p.add_argument("--top-n", type=int, default=cls.TOP_N)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            input_csv=Path(args.input_csv),
            full_results_csv=Path(args.full_results_csv),
            out_dir=Path(args.out_dir),
            index_csv=Path(args.index_csv),
            epics_dir=Path(args.epics_dir),
            top_n=int(args.top_n),
        )

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    @classmethod
    def _row_json(cls, row: pd.Series) -> Dict[str, Any]:
        return {str(k): cls._json_safe(v) for k, v in row.to_dict().items()}

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            out = float("nan")
        return out if np.isfinite(out) else float("nan")

    @staticmethod
    def _phase_centered(time: np.ndarray, period: float, center_phase: float) -> np.ndarray:
        phase = np.mod(np.asarray(time, dtype=float) / float(period), 1.0)
        return ((phase - float(center_phase) + 0.5) % 1.0) - 0.5

    @staticmethod
    def _load_light_curve(query: str) -> Tuple[np.ndarray, np.ndarray, str]:
        handler = K2_NoiseHandler(quality_strict=True)
        fetched = handler.fetch_best(query=str(query), cache_only=True)
        if str(fetched.get("status", "")).lower() != "ok":
            raise RuntimeError(f"cache-only light curve unavailable for {query}: {fetched.get('status')}")
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
        norm = K2SNR().normalize(time=t, flux=f)
        resid = np.asarray(norm["resid"], dtype=float)
        ok = np.isfinite(t) & np.isfinite(resid)
        return t[ok], resid[ok], str(fetched.get("cache_path", ""))

    @classmethod
    def _prepare_events(
        cls,
        *,
        events_csv: Path,
        period: float,
        center_phase: float,
    ) -> pd.DataFrame:
        events = cls._read_csv(events_csv)
        if "t_mid" not in events.columns:
            raise ValueError(f"events.csv missing t_mid: {events_csv}")
        out = events.copy()
        out["t_mid"] = pd.to_numeric(out["t_mid"], errors="coerce")
        out = out.loc[out["t_mid"].notna()].sort_values("t_mid").reset_index(drop=True)
        out["folded_phase"] = cls._phase_centered(out["t_mid"].to_numpy(dtype=float), period, center_phase)
        family = K2StageDDeeperEvalRunner._family_events(
            events_df=out,
            period=float(period),
            center_phase=float(center_phase),
            tol_phase=0.03,
        )
        family_times = set(np.round(pd.to_numeric(family.get("t_mid", pd.Series(dtype=float)), errors="coerce"), 10).dropna())
        out["in_best_event_family"] = np.round(out["t_mid"], 10).isin(family_times)
        if len(family) > 0:
            t0 = float(pd.to_numeric(family["t_mid"], errors="coerce").dropna().min())
            out["family_epoch"] = np.rint((out["t_mid"] - t0) / float(period)).astype("Int64")
            out.loc[~out["in_best_event_family"], "family_epoch"] = pd.NA
            out["odd_even"] = out["family_epoch"].map(
                lambda x: "" if pd.isna(x) else ("odd" if int(x) % 2 else "even")
            )
        else:
            out["family_epoch"] = pd.NA
            out["odd_even"] = ""
        return out

    @staticmethod
    def _save_no_data_plot(path: Path, title: str, message: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def _plot_folded(
        cls,
        *,
        path: Path,
        epic_id: str,
        period: float,
        center_phase: float,
        time: np.ndarray,
        resid: np.ndarray,
        events: pd.DataFrame,
    ) -> None:
        phase = cls._phase_centered(time, period, center_phase)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(phase, resid, s=4, alpha=0.22, color="#3c6e71", linewidths=0)
        family = events.loc[events["in_best_event_family"].astype(bool)].copy()
        if len(family) > 0:
            ax.scatter(
                family["folded_phase"],
                -pd.to_numeric(family.get("depth", pd.Series([0] * len(family))), errors="coerce").abs(),
                s=36,
                color="#c1121f",
                label="event family",
                zorder=3,
            )
        ax.axvline(0.0, color="#111111", lw=1, alpha=0.7)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("Folded phase")
        ax.set_ylabel("Normalized residual flux")
        ax.set_title(f"{epic_id} folded at P={period:.6f} d")
        if len(family) > 0:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def _plot_overlay(
        cls,
        *,
        path: Path,
        epic_id: str,
        period: float,
        time: np.ndarray,
        resid: np.ndarray,
        events: pd.DataFrame,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time, resid, lw=0.6, color="#264653", alpha=0.75)
        for _, ev in events.iterrows():
            if not bool(ev.get("in_best_event_family", False)):
                continue
            t0 = cls._as_float(ev.get("t_start", np.nan))
            t1 = cls._as_float(ev.get("t_end", np.nan))
            tm = cls._as_float(ev.get("t_mid", np.nan))
            if np.isfinite(t0) and np.isfinite(t1) and t1 >= t0:
                ax.axvspan(t0, t1, color="#e76f51", alpha=0.22)
            if np.isfinite(tm):
                ax.axvline(tm, color="#c1121f", lw=0.8, alpha=0.65)
        ax.set_xlabel("Time [BKJD]")
        ax.set_ylabel("Normalized residual flux")
        ax.set_title(f"{epic_id} event overlay, P={period:.6f} d")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def _plot_odd_even(
        cls,
        *,
        path: Path,
        epic_id: str,
        period: float,
        center_phase: float,
        time: np.ndarray,
        resid: np.ndarray,
        events: pd.DataFrame,
    ) -> None:
        family = events.loc[events["in_best_event_family"].astype(bool)].copy()
        if len(family) < 2 or family["odd_even"].replace("", np.nan).dropna().nunique() < 2:
            cls._save_no_data_plot(path, f"{epic_id} odd/even comparison", "Odd/even split unavailable")
            return

        phase = cls._phase_centered(time, period, center_phase)
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, color in [("even", "#2a9d8f"), ("odd", "#e76f51")]:
            subset = family.loc[family["odd_even"].eq(label)]
            if len(subset) == 0:
                continue
            mask = np.zeros(len(time), dtype=bool)
            for _, ev in subset.iterrows():
                tm = cls._as_float(ev.get("t_mid", np.nan))
                if np.isfinite(tm):
                    mask |= np.abs(time - tm) <= min(0.35, max(0.08, 0.04 * float(period)))
            ax.scatter(phase[mask], resid[mask], s=8, alpha=0.35, color=color, label=label, linewidths=0)
            ax.scatter(
                subset["folded_phase"],
                -pd.to_numeric(subset.get("depth", pd.Series([0] * len(subset))), errors="coerce").abs(),
                s=44,
                color=color,
                edgecolor="black",
                linewidth=0.3,
            )
        ax.axvline(0.0, color="#111111", lw=1, alpha=0.7)
        ax.set_xlim(-0.18, 0.18)
        ax.set_xlabel("Folded phase near transit")
        ax.set_ylabel("Normalized residual flux")
        ax.set_title(f"{epic_id} odd/even transit comparison")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _build_one(
        self,
        *,
        row: pd.Series,
        full_row: pd.Series,
        out_dir: Path,
        epics_dir: Path,
    ) -> Dict[str, Any]:
        epic_id = str(row["epic_id"]).strip()
        epic_digits = K2StageDDeeperEvalRunner._extract_epic_digits(epic_id)
        query = f"EPIC {epic_digits}"
        period = self._as_float(row["best_period_days"])
        center_phase = self._as_float(full_row.get("cluster_center_phase", np.nan))
        if not np.isfinite(center_phase):
            center_phase = 0.0

        epic_dir = out_dir / epic_id
        epic_dir.mkdir(parents=True, exist_ok=True)
        folded_plot = epic_dir / f"{epic_id}_folded_period.png"
        overlay_plot = epic_dir / f"{epic_id}_event_overlay.png"
        odd_even_plot = epic_dir / f"{epic_id}_odd_even.png"
        events_table = epic_dir / f"{epic_id}_per_event_table.csv"
        summary_json = epic_dir / f"{epic_id}_summary.json"

        events_csv = epics_dir / f"EPIC_{epic_digits}" / "events.csv"
        events = self._prepare_events(events_csv=events_csv, period=period, center_phase=center_phase)
        events.to_csv(events_table, index=False)

        time, resid, cache_path = self._load_light_curve(query=query)
        self._plot_folded(
            path=folded_plot,
            epic_id=epic_id,
            period=period,
            center_phase=center_phase,
            time=time,
            resid=resid,
            events=events,
        )
        self._plot_overlay(
            path=overlay_plot,
            epic_id=epic_id,
            period=period,
            time=time,
            resid=resid,
            events=events,
        )
        self._plot_odd_even(
            path=odd_even_plot,
            epic_id=epic_id,
            period=period,
            center_phase=center_phase,
            time=time,
            resid=resid,
            events=events,
        )

        summary = {
            "epic_id": epic_id,
            "query": query,
            "source_ranked_row": self._row_json(row),
            "stage_r_and_stage_d_metrics": self._row_json(full_row),
            "artifacts": {
                "events_csv": str(events_csv),
                "light_curve_cache_path": cache_path,
                "folded_plot_path": str(folded_plot),
                "event_overlay_path": str(overlay_plot),
                "odd_even_plot_path": str(odd_even_plot),
                "per_event_table_path": str(events_table),
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "epic_id": epic_id,
            "best_period_days": row.get("best_period_days", np.nan),
            "period_support_count": row.get("period_support_count", np.nan),
            "folded_depth_consistency": row.get("folded_depth_consistency", np.nan),
            "duration_consistency": row.get("duration_consistency", np.nan),
            "odd_even_depth_delta": row.get("odd_even_depth_delta", np.nan),
            "folded_plot_path": str(folded_plot),
            "event_overlay_path": str(overlay_plot),
            "odd_even_plot_path": str(odd_even_plot),
            "summary_json_path": str(summary_json),
        }

    def run(
        self,
        *,
        input_csv: Path = DEFAULT_INPUT_CSV,
        full_results_csv: Path = DEFAULT_FULL_RESULTS_CSV,
        out_dir: Path = DEFAULT_OUT_DIR,
        index_csv: Path = DEFAULT_INDEX_CSV,
        epics_dir: Path = DEFAULT_EPICS_DIR,
        top_n: int = TOP_N,
    ) -> Dict[str, Any]:
        ranked = self._read_csv(Path(input_csv))
        if "stage_d_label" in ranked.columns:
            ranked = ranked.loc[ranked["stage_d_label"].fillna("").astype(str).eq("pass_deeper_eval")].copy()
        top = ranked.head(max(0, int(top_n))).reset_index(drop=True)
        full = self._read_csv(Path(full_results_csv))
        full_map = {
            str(r["epic_id"]).strip(): r
            for _, r in full.iterrows()
            if str(r.get("epic_id", "")).strip() != ""
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        index_rows: List[Dict[str, Any]] = []
        for _, row in top.iterrows():
            epic_id = str(row["epic_id"]).strip()
            full_row = full_map.get(epic_id, row)
            index_rows.append(
                self._build_one(
                    row=row,
                    full_row=full_row,
                    out_dir=out_dir,
                    epics_dir=Path(epics_dir),
                )
            )

        index_df = pd.DataFrame(index_rows).reindex(columns=self.INDEX_COLUMNS)
        index_csv = Path(index_csv)
        index_csv.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(index_csv, index=False)
        return {
            "input_csv": Path(input_csv),
            "out_dir": out_dir,
            "index_csv": index_csv,
            "rows_input": int(len(ranked)),
            "top_n": int(len(top)),
            "rows_output": int(len(index_df)),
            "top_epics": index_df["epic_id"].fillna("").astype(str).tolist(),
        }
