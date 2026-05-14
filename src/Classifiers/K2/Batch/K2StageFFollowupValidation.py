from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageDDeeperEvalRunner import K2StageDDeeperEvalRunner
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2_NoiseHandler
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


class K2StageFFollowupValidation:
    DEFAULT_INPUT_CSV = Path("k2_stage_e_followup_targets.csv")
    DEFAULT_OUTPUT_CSV = Path("k2_stage_f_followup_validation.csv")
    DEFAULT_OUT_DIR = Path(r"plots\k2_batch\stage_f_followup")

    OUTPUT_COLUMNS = [
        "epic_id",
        "best_period_days",
        "stage_e_visual_label",
        "stage_f_label",
        "stage_f_reason",
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_days",
        "transit_duration_hours",
        "radius_ratio_sqrt_depth",
        "secondary_depth_phase_05",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "odd_depth_median",
        "even_depth_median",
        "odd_even_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_amp",
        "oot_variability_to_depth",
        "alias_best_period_days",
        "alias_best_support_count",
        "alias_best_support_ratio",
        "half_period_support_count",
        "double_period_support_count",
        "alias_risk",
        "phase_0_folded_path",
        "phase_05_secondary_check_path",
        "alias_period_comparison_path",
        "odd_even_zoom_path",
        "validation_summary_json_path",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Run Stage F EB/alias validation on Stage E follow-up targets.")
        p.add_argument("--input-csv", type=Path, default=cls.DEFAULT_INPUT_CSV)
        p.add_argument("--output-csv", type=Path, default=cls.DEFAULT_OUTPUT_CSV)
        p.add_argument("--out-dir", type=Path, default=cls.DEFAULT_OUT_DIR)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
            out_dir=Path(args.out_dir),
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
    def _as_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            out = float("nan")
        return out if np.isfinite(out) else float("nan")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    @classmethod
    def _phase_centered(cls, time: np.ndarray, period: float, center_phase: float) -> np.ndarray:
        phase = np.mod(np.asarray(time, dtype=float) / float(period), 1.0)
        return ((phase - float(center_phase) + 0.5) % 1.0) - 0.5

    @classmethod
    def _phase_bin_median(cls, phase: np.ndarray, value: np.ndarray, bins: int = 120) -> pd.DataFrame:
        p = np.asarray(phase, dtype=float)
        y = np.asarray(value, dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        if not np.any(ok):
            return pd.DataFrame(columns=["phase", "median", "count"])
        p = p[ok]
        y = y[ok]
        edges = np.linspace(-0.5, 0.5, int(bins) + 1)
        idx = np.digitize(p, edges) - 1
        rows: List[Dict[str, Any]] = []
        for i in range(int(bins)):
            mask = idx == i
            if not np.any(mask):
                continue
            rows.append(
                {
                    "phase": float((edges[i] + edges[i + 1]) / 2.0),
                    "median": float(np.nanmedian(y[mask])),
                    "count": int(np.sum(mask)),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _robust_sigma(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float("nan")
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        sig = 1.4826 * mad
        return sig if np.isfinite(sig) and sig > 0 else float("nan")

    @classmethod
    def _folded_depth(
        cls,
        *,
        phase: np.ndarray,
        resid: np.ndarray,
        half_width_phase: float,
    ) -> Dict[str, Any]:
        p = np.asarray(phase, dtype=float)
        r = np.asarray(resid, dtype=float)
        in_transit = np.isfinite(p) & np.isfinite(r) & (np.abs(p) <= float(half_width_phase))
        oot = np.isfinite(p) & np.isfinite(r) & (np.abs(p) >= min(0.45, float(half_width_phase) * 3.0))
        if not np.any(in_transit) or not np.any(oot):
            return {"depth": float("nan"), "snr": float("nan"), "n_in": int(np.sum(in_transit))}
        oot_med = float(np.nanmedian(r[oot]))
        in_med = float(np.nanmedian(r[in_transit]))
        depth = oot_med - in_med
        sig = cls._robust_sigma(r[oot])
        snr = float(depth / sig) if np.isfinite(depth) and np.isfinite(sig) and sig > 0 else float("nan")
        return {"depth": float(depth), "snr": float(snr), "n_in": int(np.sum(in_transit))}

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
    def _load_summary(cls, row: pd.Series) -> Dict[str, Any]:
        path = Path(str(row.get("summary_json_path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"Missing summary_json_path for {row.get('epic_id')}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _prepare_family_events(cls, summary: Dict[str, Any], period: float, center_phase: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events_csv = Path(str(summary.get("artifacts", {}).get("events_csv", "")))
        if not events_csv.exists():
            metrics = summary.get("stage_r_and_stage_d_metrics", {})
            events_csv = Path(str(metrics.get("events_csv", "")))
        events = cls._read_csv(events_csv)
        filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
        family = K2StageDDeeperEvalRunner._family_events(
            events_df=filtered,
            period=float(period),
            center_phase=float(center_phase),
            tol_phase=0.03,
        )
        if len(family) > 0:
            t0 = float(pd.to_numeric(family["t_mid"], errors="coerce").dropna().min())
            family = family.copy().sort_values("t_mid").reset_index(drop=True)
            family["family_epoch"] = np.rint((pd.to_numeric(family["t_mid"], errors="coerce") - t0) / float(period)).astype(int)
            family["odd_even"] = family["family_epoch"].map(lambda x: "odd" if int(x) % 2 else "even")
        return filtered, family

    @classmethod
    def _epoch_depth_stats(cls, family: pd.DataFrame) -> Dict[str, Any]:
        if len(family) == 0 or "depth" not in family.columns or "family_epoch" not in family.columns:
            return {
                "odd_depth_median": float("nan"),
                "even_depth_median": float("nan"),
                "odd_even_depth_ratio": float("nan"),
                "odd_even_depth_delta_explicit": float("nan"),
            }
        work = family.copy()
        work["depth"] = pd.to_numeric(work["depth"], errors="coerce")
        work = work.loc[work["depth"].notna()].copy()
        if len(work) == 0:
            return {
                "odd_depth_median": float("nan"),
                "even_depth_median": float("nan"),
                "odd_even_depth_ratio": float("nan"),
                "odd_even_depth_delta_explicit": float("nan"),
            }
        epoch_depths = work.groupby("family_epoch", dropna=True)["depth"].median().reset_index()
        even = epoch_depths.loc[(epoch_depths["family_epoch"].astype(int) % 2) == 0, "depth"].to_numpy(dtype=float)
        odd = epoch_depths.loc[(epoch_depths["family_epoch"].astype(int) % 2) == 1, "depth"].to_numpy(dtype=float)
        if len(even) == 0 or len(odd) == 0:
            return {
                "odd_depth_median": float("nan"),
                "even_depth_median": float("nan"),
                "odd_even_depth_ratio": float("nan"),
                "odd_even_depth_delta_explicit": float("nan"),
            }
        odd_med = float(np.nanmedian(odd))
        even_med = float(np.nanmedian(even))
        hi = max(abs(odd_med), abs(even_med))
        lo = min(abs(odd_med), abs(even_med))
        ratio = float(lo / hi) if hi > 0 else float("nan")
        ref = float(np.nanmedian(np.abs(epoch_depths["depth"].to_numpy(dtype=float))))
        delta = float(abs(odd_med - even_med) / ref) if np.isfinite(ref) and ref > 0 else float("nan")
        return {
            "odd_depth_median": odd_med,
            "even_depth_median": even_med,
            "odd_even_depth_ratio": ratio,
            "odd_even_depth_delta_explicit": delta,
        }

    @classmethod
    def _alias_stats(cls, events: pd.DataFrame, period: float, primary_support_count: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        candidates = [
            ("P/2", float(period) / 2.0),
            ("2P/3", float(period) * 2.0 / 3.0),
            ("P", float(period)),
            ("3P/2", float(period) * 1.5),
            ("P*2", float(period) * 2.0),
        ]
        rows: List[Dict[str, Any]] = []
        for name, p in candidates:
            support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(events_df=events, period=float(p), tol_phase=0.03)
            rows.append({"alias_name": name, "period_days": float(p), "support_count": int(support), "cluster_center_phase": float(center)})
        df = pd.DataFrame(rows)
        p_support = int(max(0, int(primary_support_count)))
        if p_support <= 0 and len(df) > 0:
            p_support = int(df.loc[df["alias_name"].eq("P"), "support_count"].iloc[0])
        alternatives = df.loc[~df["alias_name"].eq("P")].copy()
        best_alt = alternatives.sort_values(["support_count", "period_days"], ascending=[False, True]).iloc[0].to_dict()
        ratio = float(best_alt["support_count"] / p_support) if p_support > 0 else float("nan")
        half_support = int(df.loc[df["alias_name"].eq("P/2"), "support_count"].iloc[0])
        double_support = int(df.loc[df["alias_name"].eq("P*2"), "support_count"].iloc[0])
        alias_risk = "low"
        if np.isfinite(ratio) and ratio >= 1.20:
            alias_risk = "high"
        elif np.isfinite(ratio) and ratio >= 0.85:
            alias_risk = "moderate"
        return df, {
            "alias_best_period_days": float(best_alt["period_days"]),
            "alias_best_support_count": int(best_alt["support_count"]),
            "alias_best_support_ratio": ratio,
            "half_period_support_count": half_support,
            "double_period_support_count": double_support,
            "alias_risk": alias_risk,
        }

    @classmethod
    def _oot_variability(cls, phase0: np.ndarray, resid: np.ndarray, primary_half_width: float) -> Dict[str, Any]:
        p = np.asarray(phase0, dtype=float)
        r = np.asarray(resid, dtype=float)
        secondary_phase = ((p - 0.5 + 0.5) % 1.0) - 0.5
        oot = (
            np.isfinite(p)
            & np.isfinite(r)
            & (np.abs(p) > max(0.08, 2.0 * float(primary_half_width)))
            & (np.abs(secondary_phase) > max(0.08, 2.0 * float(primary_half_width)))
        )
        binned = cls._phase_bin_median(p[oot], r[oot], bins=80)
        if len(binned) == 0:
            return {"oot_variability_amp": float("nan")}
        vals = binned["median"].to_numpy(dtype=float)
        amp = float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5))
        return {"oot_variability_amp": amp}

    @classmethod
    def _label(
        cls,
        *,
        visual_label: str,
        secondary_ratio: float,
        secondary_snr: float,
        odd_even_ratio: float,
        odd_even_delta: float,
        alias_risk: str,
        oot_to_depth: float,
        primary_snr: float,
    ) -> Tuple[str, str]:
        strong_secondary = np.isfinite(secondary_ratio) and secondary_ratio >= 0.35 and np.isfinite(secondary_snr) and secondary_snr >= 5.0
        strong_odd_even = (
            (np.isfinite(odd_even_ratio) and odd_even_ratio < 0.55)
            or (np.isfinite(odd_even_delta) and odd_even_delta >= 0.75)
        )
        high_oot = np.isfinite(oot_to_depth) and oot_to_depth >= 3.0
        if not np.isfinite(primary_snr) or primary_snr < 5.0:
            return "stage_f_reject", "primary transit depth is not significant in cached folded light curve"
        if strong_secondary:
            return "stage_f_likely_eb", f"secondary eclipse at phase 0.5: depth_ratio={secondary_ratio:.3f}, snr={secondary_snr:.2f}"
        if strong_odd_even and str(alias_risk) == "high":
            return "stage_f_likely_eb", f"odd/even mismatch plus high alias risk: ratio={odd_even_ratio:.3f}, alias_risk={alias_risk}"
        if str(visual_label) == "visual_planet_like_candidate" and str(alias_risk) != "high" and not strong_odd_even and not high_oot:
            return "stage_f_planet_like", "survives secondary-eclipse, alias, odd/even, and OOT variability checks"
        if strong_odd_even or str(alias_risk) in {"moderate", "high"} or high_oot:
            return "stage_f_hold", (
                f"ambiguous follow-up checks: odd_even_ratio={odd_even_ratio:.3f} "
                f"alias_risk={alias_risk} oot_to_depth={oot_to_depth:.3f}"
            )
        return "stage_f_hold", "no EB rejection, but visual/metric confidence remains secondary queue"

    @staticmethod
    def _plot_phase(
        *,
        path: Path,
        title: str,
        phase: np.ndarray,
        resid: np.ndarray,
        xlim: Tuple[float, float],
        marker_phase: float = 0.0,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(phase, resid, s=4, alpha=0.22, color="#3c6e71", linewidths=0)
        binned = K2StageFFollowupValidation._phase_bin_median(phase, resid, bins=140)
        if len(binned) > 0:
            ax.plot(binned["phase"], binned["median"], color="#c1121f", lw=1.5)
        ax.axvline(marker_phase, color="#111111", lw=1.0, alpha=0.75)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Folded phase")
        ax.set_ylabel("Normalized residual flux")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def _plot_aliases(
        cls,
        *,
        path: Path,
        epic_id: str,
        time: np.ndarray,
        resid: np.ndarray,
        alias_df: pd.DataFrame,
    ) -> None:
        fig, axes = plt.subplots(len(alias_df), 1, figsize=(10, 2.4 * len(alias_df)), sharey=True)
        if len(alias_df) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, alias_df.iterrows()):
            p = float(row["period_days"])
            center = float(row["cluster_center_phase"]) if np.isfinite(float(row["cluster_center_phase"])) else 0.0
            phase = cls._phase_centered(time, p, center)
            ax.scatter(phase, resid, s=3, alpha=0.16, color="#264653", linewidths=0)
            binned = cls._phase_bin_median(phase, resid, bins=100)
            if len(binned) > 0:
                ax.plot(binned["phase"], binned["median"], color="#e76f51", lw=1.2)
            ax.axvline(0, color="#111111", lw=0.8)
            ax.set_xlim(-0.25, 0.25)
            ax.set_title(f"{row['alias_name']} P={p:.6f} d support={int(row['support_count'])}")
        axes[-1].set_xlabel("Folded phase near best cluster")
        fig.supylabel("Normalized residual flux")
        fig.suptitle(f"{epic_id} alias period comparison", y=0.995)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def _plot_odd_even(
        cls,
        *,
        path: Path,
        epic_id: str,
        phase: np.ndarray,
        resid: np.ndarray,
        family: pd.DataFrame,
        time: np.ndarray,
        period: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, color in [("even", "#2a9d8f"), ("odd", "#e76f51")]:
            subset = family.loc[family.get("odd_even", pd.Series(dtype=str)).astype(str).eq(label)]
            mask = np.zeros(len(time), dtype=bool)
            for _, ev in subset.iterrows():
                tm = cls._as_float(ev.get("t_mid", np.nan))
                if np.isfinite(tm):
                    mask |= np.abs(time - tm) <= min(0.35, max(0.08, 0.04 * float(period)))
            ax.scatter(phase[mask], resid[mask], s=8, alpha=0.35, color=color, label=label, linewidths=0)
        ax.axvline(0.0, color="#111111", lw=1.0)
        ax.set_xlim(-0.16, 0.16)
        ax.set_xlabel("Folded phase near primary")
        ax.set_ylabel("Normalized residual flux")
        ax.set_title(f"{epic_id} odd/even primary zoom")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _validate_one(self, row: pd.Series, out_dir: Path) -> Dict[str, Any]:
        epic_id = str(row["epic_id"]).strip()
        period = self._as_float(row["best_period_days"])
        visual_label = str(row.get("visual_label", "")).strip()
        summary = self._load_summary(row)
        metrics = summary.get("stage_r_and_stage_d_metrics", {})
        center_phase = self._as_float(metrics.get("cluster_center_phase", np.nan))
        if not np.isfinite(center_phase):
            center_phase = 0.0

        epic_digits = K2StageDDeeperEvalRunner._extract_epic_digits(epic_id)
        query = f"EPIC {epic_digits}"
        time, resid, lc_cache_path = self._load_light_curve(query=query)
        events, family = self._prepare_family_events(summary=summary, period=period, center_phase=center_phase)

        duration_days = self._as_float(pd.to_numeric(family.get("duration_days", pd.Series(dtype=float)), errors="coerce").median())
        if not np.isfinite(duration_days) or duration_days <= 0:
            duration_days = max(0.15, 0.03 * float(period))
        half_width_phase = float(np.clip(1.6 * duration_days / float(period), 0.015, 0.08))

        phase0 = self._phase_centered(time, period, center_phase)
        folded_primary = self._folded_depth(phase=phase0, resid=resid, half_width_phase=half_width_phase)
        secondary_center = (float(center_phase) + 0.5) % 1.0
        phase05 = self._phase_centered(time, period, secondary_center)
        secondary = self._folded_depth(phase=phase05, resid=resid, half_width_phase=half_width_phase)

        family_depth = pd.to_numeric(family.get("depth", pd.Series(dtype=float)), errors="coerce")
        family_snr = pd.to_numeric(family.get("depth_snr", pd.Series(dtype=float)), errors="coerce")
        primary_depth = self._as_float(family_depth.median())
        primary_snr = self._as_float(family_snr.median())
        if not np.isfinite(primary_depth) or primary_depth <= 0:
            primary_depth = self._as_float(folded_primary["depth"])
        if not np.isfinite(primary_snr) or primary_snr <= 0:
            primary_snr = self._as_float(folded_primary["snr"])
        secondary_depth = self._as_float(secondary["depth"])
        secondary_ratio = float(secondary_depth / primary_depth) if np.isfinite(primary_depth) and primary_depth > 0 and np.isfinite(secondary_depth) else float("nan")
        radius_ratio = float(np.sqrt(primary_depth)) if np.isfinite(primary_depth) and primary_depth > 0 else float("nan")

        odd_even = self._epoch_depth_stats(family)
        primary_support_count = int(self._as_float(row.get("period_support_count", metrics.get("period_support_count", 0))))
        alias_df, alias = self._alias_stats(events, period=period, primary_support_count=primary_support_count)
        oot = self._oot_variability(phase0=phase0, resid=resid, primary_half_width=half_width_phase)
        oot_amp = self._as_float(oot["oot_variability_amp"])
        oot_to_depth = float(oot_amp / primary_depth) if np.isfinite(oot_amp) and np.isfinite(primary_depth) and primary_depth > 0 else float("nan")

        label, reason = self._label(
            visual_label=visual_label,
            secondary_ratio=secondary_ratio,
            secondary_snr=self._as_float(secondary["snr"]),
            odd_even_ratio=self._as_float(odd_even["odd_even_depth_ratio"]),
            odd_even_delta=self._as_float(odd_even["odd_even_depth_delta_explicit"]),
            alias_risk=str(alias["alias_risk"]),
            oot_to_depth=oot_to_depth,
            primary_snr=primary_snr,
        )

        epic_dir = out_dir / epic_id
        epic_dir.mkdir(parents=True, exist_ok=True)
        phase0_path = epic_dir / "phase_0_folded.png"
        phase05_path = epic_dir / "phase_05_secondary_check.png"
        alias_path = epic_dir / "alias_period_comparison.png"
        odd_even_path = epic_dir / "odd_even_zoom.png"
        summary_path = epic_dir / "validation_summary.json"

        self._plot_phase(
            path=phase0_path,
            title=f"{epic_id} primary folded check, P={period:.6f} d",
            phase=phase0,
            resid=resid,
            xlim=(-0.25, 0.25),
        )
        self._plot_phase(
            path=phase05_path,
            title=f"{epic_id} phase 0.5 secondary eclipse check",
            phase=phase05,
            resid=resid,
            xlim=(-0.25, 0.25),
        )
        self._plot_aliases(path=alias_path, epic_id=epic_id, time=time, resid=resid, alias_df=alias_df)
        self._plot_odd_even(path=odd_even_path, epic_id=epic_id, phase=phase0, resid=resid, family=family, time=time, period=period)

        out = {
            "epic_id": epic_id,
            "best_period_days": period,
            "stage_e_visual_label": visual_label,
            "stage_f_label": label,
            "stage_f_reason": reason,
            "primary_depth": primary_depth,
            "primary_depth_snr": primary_snr,
            "transit_duration_days": duration_days,
            "transit_duration_hours": float(duration_days * 24.0),
            "radius_ratio_sqrt_depth": radius_ratio,
            "secondary_depth_phase_05": secondary_depth,
            "secondary_depth_snr": self._as_float(secondary["snr"]),
            "secondary_to_primary_depth_ratio": secondary_ratio,
            "odd_depth_median": self._as_float(odd_even["odd_depth_median"]),
            "even_depth_median": self._as_float(odd_even["even_depth_median"]),
            "odd_even_depth_ratio": self._as_float(odd_even["odd_even_depth_ratio"]),
            "odd_even_depth_delta_explicit": self._as_float(odd_even["odd_even_depth_delta_explicit"]),
            "oot_variability_amp": oot_amp,
            "oot_variability_to_depth": oot_to_depth,
            "alias_best_period_days": self._as_float(alias["alias_best_period_days"]),
            "alias_best_support_count": int(alias["alias_best_support_count"]),
            "alias_best_support_ratio": self._as_float(alias["alias_best_support_ratio"]),
            "half_period_support_count": int(alias["half_period_support_count"]),
            "double_period_support_count": int(alias["double_period_support_count"]),
            "alias_risk": str(alias["alias_risk"]),
            "phase_0_folded_path": str(phase0_path),
            "phase_05_secondary_check_path": str(phase05_path),
            "alias_period_comparison_path": str(alias_path),
            "odd_even_zoom_path": str(odd_even_path),
            "validation_summary_json_path": str(summary_path),
        }

        summary_payload = {
            "validation": {k: self._json_safe(v) for k, v in out.items()},
            "stage_e_row": {str(k): self._json_safe(v) for k, v in row.to_dict().items()},
            "stage_d_summary": summary,
            "light_curve_cache_path": lc_cache_path,
            "alias_periods": [
                {str(k): self._json_safe(v) for k, v in r.items()}
                for r in alias_df.to_dict(orient="records")
            ],
            "family_event_count": int(len(family)),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return out

    def run(
        self,
        *,
        input_csv: Path = DEFAULT_INPUT_CSV,
        output_csv: Path = DEFAULT_OUTPUT_CSV,
        out_dir: Path = DEFAULT_OUT_DIR,
    ) -> Dict[str, Any]:
        df = self._read_csv(Path(input_csv))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [self._validate_one(row=row, out_dir=out_dir) for _, row in df.iterrows()]
        out_df = pd.DataFrame(rows).reindex(columns=self.OUTPUT_COLUMNS)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        labels = out_df["stage_f_label"].fillna("").astype(str)
        return {
            "input_csv": Path(input_csv),
            "output_csv": output_csv,
            "out_dir": out_dir,
            "rows_input": int(len(df)),
            "rows_output": int(len(out_df)),
            "label_counts": labels.value_counts().to_dict(),
            "planet_like_epics": out_df.loc[labels.eq("stage_f_planet_like"), "epic_id"].astype(str).tolist(),
            "hold_epics": out_df.loc[labels.eq("stage_f_hold"), "epic_id"].astype(str).tolist(),
            "likely_eb_epics": out_df.loc[labels.eq("stage_f_likely_eb"), "epic_id"].astype(str).tolist(),
            "reject_epics": out_df.loc[labels.eq("stage_f_reject"), "epic_id"].astype(str).tolist(),
        }
