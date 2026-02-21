from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.Classifiers.K2.K2_TimeDomainTransitPipeline import (
    K2TimeDomainTransitPipeline,
    infer_periods_from_events,
)
from src.Classifiers.K2.Systematics.K2NoiseLoader import (
    K2NoiseConfig,
    K2NoiseLoader,
    K2NoiseLoaderConfig,
)
from src.Classifiers.K2.Systematics.K2_PeriodValidator import K2PeriodValidator


class K2BatchRunner:
    def __init__(
        self,
        out_dir: Union[str, Path],
        queries: Optional[Sequence[str]] = None,
        input_csv: Optional[Union[str, Path]] = None,
        query_col: str = "query",
        top_k_periods: int = 3,
        period_candidate_pool: int = 24,
        phase_tol: float = 0.03,
        max_period_days: float = 40.0,
        min_hits_for_period: int = 3,
        period_tol_frac: float = 0.01,
        noise_mode: str = "strict",
        limit: int = 50,
        exptime: Optional[Union[str, float]] = None,
        validator_tol_days: float = 0.12,
        validator_outer_window_days: float = 2.0,
        validator_min_duration_cadences: int = 3,
        validator_shape_threshold: float = 0.6,
        validator_snr_threshold: float = 4.0,
        periodic_shape_threshold: float = 0.75,
        periodic_hit_rate_shape_threshold: float = 0.30,
        periodic_coverage_threshold: float = 0.85,
        whiteness_alpha: Optional[float] = None,
        whiteness_score_definition: str = "pvalue",
        noisy_whiteness_threshold: Optional[float] = None,
        noisy_step_threshold: Optional[float] = None,
        resume: bool = False,
        skip_existing_epics: bool = True,
        cache_only: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.epics_dir = self.out_dir / "epics"
        self.progress_path = self.out_dir / "progress.json"
        self.queries = list(queries) if queries is not None else None
        self.input_csv = Path(input_csv) if input_csv is not None else None
        self.query_col = str(query_col)
        self.resume = bool(resume)
        self.skip_existing_epics = bool(skip_existing_epics)
        self.cache_only = bool(cache_only)
        self.top_k_periods = int(max(1, top_k_periods))
        self.period_candidate_pool = int(max(self.top_k_periods, period_candidate_pool))
        self.phase_tol = float(phase_tol)
        self.max_period_days = float(max_period_days)
        self.min_hits_for_period = int(max(2, min_hits_for_period))
        self.period_tol_frac = float(period_tol_frac)
        self.limit = int(max(1, limit))
        self.exptime = exptime
        self.periodic_shape_threshold = float(periodic_shape_threshold)
        self.periodic_hit_rate_shape_threshold = float(periodic_hit_rate_shape_threshold)
        self.periodic_coverage_threshold = float(periodic_coverage_threshold)

        logging.getLogger("lightkurve").setLevel(logging.WARNING)

        self.loader_config = K2NoiseLoaderConfig(
            limit=self.limit,
            exptime=self.exptime,
            flatten=False,
            per_segment=True,
            mode=str(noise_mode).strip().lower(),
            cache_only=self.cache_only,
        )
        self.noise_config = K2NoiseConfig(
            mode=self.loader_config.mode,
            whiteness_alpha=whiteness_alpha,
            whiteness_score_definition=whiteness_score_definition,
        )
        self.noise_loader = K2NoiseLoader(loader_config=self.loader_config, noise_config=self.noise_config)
        self.detector = K2TimeDomainTransitPipeline(loader=self.noise_loader)
        self.validator = K2PeriodValidator(
            detector=self.detector,
            tol_days=float(validator_tol_days),
            outer_window_days=float(validator_outer_window_days),
            min_duration_cadences=int(validator_min_duration_cadences),
            shape_threshold=float(validator_shape_threshold),
            snr_threshold=float(validator_snr_threshold),
        )
        self.noisy_whiteness_threshold = (
            float(noisy_whiteness_threshold)
            if noisy_whiteness_threshold is not None
            else (
                float(self.noise_config.whiteness_alpha)
                if (
                    str(getattr(self.noise_config, "whiteness_score_definition", "statistic")).lower() == "pvalue"
                    and getattr(self.noise_config, "whiteness_alpha", None) is not None
                )
                else (
                    float(self.noise_config.max_whiteness_score)
                    if self.noise_config.max_whiteness_score is not None
                    else float("inf")
                )
            )
        )
        self.noisy_step_threshold = (
            float(noisy_step_threshold)
            if noisy_step_threshold is not None
            else float(self.noise_config.max_step_score)
        )
        self._whiteness_debug_limit = 20
        self._whiteness_debug_printed = 0

    @staticmethod
    def _as_float(value: Any, default: float = float("nan")) -> float:
        try:
            v = float(value)
        except Exception:
            return float(default)
        return v if np.isfinite(v) else float(default)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return default

    @staticmethod
    def _normalize_queries(raw_queries: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in raw_queries:
            if raw is None:
                continue
            for part in str(raw).split(","):
                q = part.strip()
                if q == "" or q in seen:
                    continue
                # Canonicalize numeric EPIC IDs to "EPIC <id>" queries.
                m_float = re.fullmatch(r"(\d{6,})\.0+", q)
                if m_float is not None:
                    q = m_float.group(1)
                m_plain = re.fullmatch(r"\d{6,}", q)
                if m_plain is not None:
                    q = f"EPIC {m_plain.group(0)}"
                else:
                    m_epic = re.fullmatch(r"EPIC[_\s-]*(\d{6,})", q, flags=re.IGNORECASE)
                    if m_epic is not None:
                        q = f"EPIC {m_epic.group(1)}"
                seen.add(q)
                out.append(q)
        return out

    def _load_queries(self) -> List[str]:
        if self.queries is not None and len(self.queries) > 0:
            return self._normalize_queries(self.queries)
        if self.input_csv is None:
            raise ValueError("Provide queries or input_csv.")
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        if self.query_col not in df.columns:
            raise ValueError(f"Column '{self.query_col}' not found in {self.input_csv}")
        return self._normalize_queries(df[self.query_col].dropna().astype(str).tolist())

    @staticmethod
    def _epic_slug(query: str, fallback_idx: int) -> str:
        text = str(query).strip()
        m = re.search(r"(\d{6,})", text)
        if m:
            return f"EPIC_{m.group(1)}"
        clean = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return clean if clean else f"QUERY_{int(fallback_idx):05d}"

    @staticmethod
    def _unique_slug(slug: str, used: set) -> str:
        base = slug
        n = 2
        while slug in used:
            slug = f"{base}_{n}"
            n += 1
        used.add(slug)
        return slug

    def _hard_fail_reasons(self, triage: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        status = str(triage.get("status", "")).strip().lower()
        usable = self._as_bool(triage.get("usable", False))
        why_not = str(triage.get("why_not_usable", "")).strip()
        whiteness_definition = str(
            triage.get("whiteness_definition", getattr(self.noise_loader.handler, "whiteness_definition", lambda: "")())
        ).lower()
        whiteness_is_pvalue = "pvalue" in whiteness_definition
        if status != "ok":
            reasons.append(f"triage_status={status or 'unknown'}")
        if (not usable) and why_not:
            reasons.append(f"usable=False:{why_not}")
        step = self._as_float(triage.get("step_score", float("nan")))
        if np.isfinite(step) and np.isfinite(self.noisy_step_threshold) and step > self.noisy_step_threshold:
            reasons.append(f"step_score>{self.noisy_step_threshold:.3f} ({step:.3f})")
        white = self._as_float(triage.get("whiteness_score", float("nan")))
        if np.isfinite(white) and np.isfinite(self.noisy_whiteness_threshold):
            if whiteness_is_pvalue:
                if white < self.noisy_whiteness_threshold:
                    reasons.append(f"whiteness_pvalue<{self.noisy_whiteness_threshold:.3f} ({white:.3f})")
            elif white > self.noisy_whiteness_threshold:
                reasons.append(f"whiteness_score>{self.noisy_whiteness_threshold:.3f} ({white:.3f})")
        return reasons

    def _debug_whiteness_gate(self, query: str, triage: Dict[str, Any]) -> None:
        if self._whiteness_debug_printed >= self._whiteness_debug_limit:
            return
        white = self._as_float(triage.get("whiteness_score", float("nan")))
        definition = str(
            triage.get("whiteness_definition", getattr(self.noise_loader.handler, "whiteness_definition", lambda: "unknown")())
        )
        is_pvalue = "pvalue" in definition.lower()
        if np.isfinite(white) and np.isfinite(self.noisy_whiteness_threshold):
            gate_pass = (white >= self.noisy_whiteness_threshold) if is_pvalue else (white <= self.noisy_whiteness_threshold)
        else:
            gate_pass = False
        status = "PASS" if gate_pass else "FAIL"
        n = self._whiteness_debug_printed + 1
        print(
            f"[whiteness debug {n}/{self._whiteness_debug_limit}] "
            f"{query} whiteness_score={white} definition={definition} triage={status}"
        )
        self._whiteness_debug_printed += 1

    @staticmethod
    def _best_event_metrics(events_df: pd.DataFrame) -> Tuple[float, float]:
        if len(events_df) == 0:
            return float("nan"), float("nan")
        shape = pd.to_numeric(events_df.get("shape_score", np.nan), errors="coerce")
        snr = pd.to_numeric(events_df.get("depth_snr", np.nan), errors="coerce")
        return (
            float(shape.max()) if shape.notna().any() else float("nan"),
            float(snr.max()) if snr.notna().any() else float("nan"),
        )

    def _phase_cluster_score_quiet(self, events_df: pd.DataFrame, period: float) -> Tuple[int, float, List[Any]]:
        if "t_mid" not in events_df.columns or (not np.isfinite(period)) or period <= 0:
            return 0, float("nan"), []
        t = pd.to_numeric(events_df["t_mid"], errors="coerce")
        ok = np.isfinite(t.to_numpy(dtype=float))
        if not np.any(ok):
            return 0, float("nan"), []
        work = events_df.loc[ok].copy()
        work = work.assign(t_mid=pd.to_numeric(work["t_mid"], errors="coerce"))
        work = work.assign(phase=(np.mod(work["t_mid"].to_numpy(dtype=float), float(period)) / float(period)))
        work = work.sort_values("phase")
        phases = work["phase"].to_numpy(dtype=float)
        ids = list(work.index)
        n = len(phases)
        if n == 0:
            return 0, float("nan"), []

        phase2 = np.concatenate([phases, phases + 1.0])
        id2 = ids + ids
        best_count, best_start, best_end = 0, 0, -1
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j + 1 < i + n and (phase2[j + 1] - phase2[i]) <= float(self.phase_tol) + 1e-12:
                j += 1
            count = int(j - i + 1)
            if count > best_count:
                best_count, best_start, best_end = count, i, j

        if best_count <= 0:
            return 0, float("nan"), []
        cluster_phases = np.asarray(phase2[best_start:best_end + 1], dtype=float)
        cluster_ids = id2[best_start:best_end + 1]
        seen = set()
        in_cluster: List[Any] = []
        for idx in cluster_ids:
            if idx in seen:
                continue
            seen.add(idx)
            in_cluster.append(idx)
        theta = 2.0 * np.pi * np.mod(cluster_phases, 1.0)
        mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
        if mean_angle < 0:
            mean_angle += 2.0 * np.pi
        return int(best_count), float(mean_angle / (2.0 * np.pi)), in_cluster

    def _propose_periods(self, events_df: pd.DataFrame) -> pd.DataFrame:
        cols = ["period", "count_hits", "cluster_count", "cluster_center_phase", "in_cluster_indices"]
        if len(events_df) < 2 or "t_mid" not in events_df.columns:
            return pd.DataFrame(columns=cols)
        ranked, _ = infer_periods_from_events(
            events_df=events_df,
            max_period=self.max_period_days,
            min_hits=self.min_hits_for_period,
            tol_frac=self.period_tol_frac,
        )
        if len(ranked) == 0:
            ranked, _ = infer_periods_from_events(
                events_df=events_df,
                max_period=self.max_period_days,
                min_hits=2,
                tol_frac=self.period_tol_frac,
            )
        if len(ranked) == 0:
            return pd.DataFrame(columns=cols)

        rows: List[Dict[str, Any]] = []
        for period, count_hits, _support in ranked[: self.period_candidate_pool]:
            cc, cphase, in_cluster = self._phase_cluster_score_quiet(events_df=events_df, period=float(period))
            rows.append(
                {
                    "period": float(period),
                    "count_hits": int(count_hits),
                    "cluster_count": int(cc),
                    "cluster_center_phase": float(cphase),
                    "in_cluster_indices": list(in_cluster),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["cluster_count", "count_hits", "period"], ascending=[False, False, True])
            .head(self.top_k_periods)
            .reset_index(drop=True)
        )

    @staticmethod
    def _guess_t0(events_df: pd.DataFrame, period: float, in_cluster: Sequence[Any], time: np.ndarray) -> Optional[float]:
        if len(events_df) == 0 or not np.any(np.isfinite(time)):
            return None
        sub = events_df.loc[events_df.index.intersection(list(in_cluster))].copy()
        if sub.empty:
            sub = events_df.copy()
        if "shape_score" not in sub.columns or "t_mid" not in sub.columns:
            return None
        shape = pd.to_numeric(sub["shape_score"], errors="coerce")
        tmid = pd.to_numeric(sub["t_mid"], errors="coerce")
        if not shape.notna().any() or not tmid.notna().any():
            return None
        best_idx = shape.idxmax()
        t_mid_ref = float(pd.to_numeric(sub.loc[best_idx, "t_mid"], errors="coerce"))
        t_min = float(np.nanmin(time))
        if not np.isfinite(t_mid_ref) or not np.isfinite(t_min):
            return None
        return float(t_mid_ref - np.round((t_mid_ref - t_min) / float(period)) * float(period))

    @staticmethod
    def _split_no_cand(misses_df: pd.DataFrame) -> pd.DataFrame:
        if len(misses_df) == 0 or "has_candidate" not in misses_df.columns:
            return misses_df.iloc[0:0].copy()
        raw = misses_df["has_candidate"]
        num = pd.to_numeric(raw, errors="coerce")
        has = num.fillna(0).astype(int).astype(bool) if num.notna().any() else raw.astype(str).str.lower().isin({"true", "t", "yes", "y", "1"})
        return misses_df.loc[~has].copy()

    @staticmethod
    def _print_soft_top10(scores_df: pd.DataFrame, period: float) -> None:
        if not isinstance(scores_df, pd.DataFrame) or len(scores_df) == 0:
            print(f"\nTop 10 windows by soft dip_snr_at_min for P={period:.6f}: no rows")
            return
        top = scores_df.copy()
        top["dip_snr_at_min"] = pd.to_numeric(top.get("dip_snr_at_min", np.nan), errors="coerce")
        top["duration_below_threshold"] = pd.to_numeric(top.get("duration_below_threshold", np.nan), errors="coerce")
        top = top.sort_values("dip_snr_at_min", ascending=False, na_position="last").head(10)
        cols = ["tk", "dip_snr_at_min", "duration_below_threshold", "has_candidate", "hit_shape", "hit_snr", "best_shape_score", "best_depth_snr"]
        cols = [c for c in cols if c in top.columns]
        print(f"\nTop 10 windows by soft dip_snr_at_min for P={period:.6f}:")
        print(top.reindex(columns=cols).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    def _label_row(self, row: Dict[str, Any], hard_reasons: List[str]) -> Tuple[str, str]:
        if len(hard_reasons) > 0:
            return "Noisy_trash", "; ".join(hard_reasons)
        n_events = int(self._as_float(row.get("n_events", 0), default=0.0))
        if n_events <= 0:
            return "No_events", "0 events detected"
        shape = self._as_float(row.get("best_shape_score", float("nan")))
        hit = self._as_float(
            row.get("hit_rate_shape", row.get("best_period_hit_rate_shape", float("nan")))
        )
        cov = self._as_float(
            row.get("coverage_rate", row.get("best_period_coverage_rate", float("nan")))
        )
        if np.isfinite(shape) and np.isfinite(hit) and np.isfinite(cov):
            if shape >= self.periodic_shape_threshold and hit >= self.periodic_hit_rate_shape_threshold and cov >= self.periodic_coverage_threshold:
                return "Periodic_candidate", "meets periodic thresholds"
        hit_sparse = hit if np.isfinite(hit) else float("-inf")
        if np.isfinite(shape) and shape >= self.periodic_shape_threshold and hit_sparse < self.periodic_hit_rate_shape_threshold:
            return "Sparse_or_mono", "strong shape but weak periodic hit rate"
        return "Unclassified", f"shape={shape:.3f} hit_rate_shape={hit:.3f} coverage={cov:.3f}"

    def _fetch_clean_time_flux(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        fetched = self.noise_loader.handler.fetch_best(
            query=query,
            limit=self.limit,
            exptime=self.exptime,
            cache_only=self.cache_only,
        )
        if str(fetched.get("status", "ok")).lower() != "ok":
            raise RuntimeError(f"fetch_status={fetched.get('status', 'error')}")
        cleaned = self.noise_loader.handler.clean(
            fetched["lc"],
            normalize=False,
            remove_nans=True,
            quality_mask=True,
            sigma_clip=False,
            flatten=False,
        )
        return np.asarray(cleaned["time"], dtype=float), np.asarray(cleaned["flux"], dtype=float)

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"{context}: missing required columns: {missing}")

    @staticmethod
    def _tokenize_reasons(raw: Any) -> List[str]:
        if raw is None:
            return []
        try:
            if pd.isna(raw):
                return []
        except Exception:
            pass
        out: List[str] = []
        for tok in str(raw).split(";"):
            t = tok.strip()
            if t != "" and t.lower() != "nan":
                out.append(t)
        return out

    def _skip_reason_counts(self, df: pd.DataFrame) -> pd.Series:
        reasons: List[str] = []
        if "triage_why_not_usable" in df.columns:
            for v in df["triage_why_not_usable"].tolist():
                reasons.extend(self._tokenize_reasons(v))
        if ("label" in df.columns) and ("label_reason" in df.columns):
            noisy = df["label"].astype(str).eq("Noisy_trash")
            for v in df.loc[noisy, "label_reason"].tolist():
                reasons.extend(self._tokenize_reasons(v))
        if len(reasons) == 0:
            return pd.Series(dtype=int)
        return pd.Series(reasons, dtype="string").value_counts()

    def _strip_retriage_managed_reasons(self, raw: Any) -> List[str]:
        kept: List[str] = []
        for r in self._tokenize_reasons(raw):
            rl = r.lower().strip()
            if rl.startswith("whiteness_score"):
                continue
            if rl.startswith("whiteness_pvalue"):
                continue
            if rl.startswith("step_score>"):
                continue
            if rl.startswith("usable=false:"):
                continue
            if rl.startswith("triage_status="):
                continue
            kept.append(r)
        return kept

    def retriage_batch_results(
        self,
        batch_csv: Optional[Union[str, Path]] = None,
        write: bool = True,
    ) -> Dict[str, Any]:
        batch_path = Path(batch_csv) if batch_csv is not None else (self.out_dir / "batch_results.csv")
        if not batch_path.exists():
            raise FileNotFoundError(f"batch_results.csv not found: {batch_path}")

        df = pd.read_csv(batch_path)
        self._require_columns(df, ["triage_status"], context="retriage")

        if "triage_step_score" not in df.columns:
            df["triage_step_score"] = np.nan
        if "triage_whiteness_score" not in df.columns:
            df["triage_whiteness_score"] = np.nan
        if "triage_why_not_usable" not in df.columns:
            df["triage_why_not_usable"] = ""
        if "triage_whiteness_definition" not in df.columns:
            df["triage_whiteness_definition"] = ""
        if "label" not in df.columns:
            df["label"] = ""
        if "label_reason" not in df.columns:
            df["label_reason"] = ""
        if "n_events" not in df.columns:
            df["n_events"] = 0

        for idx in df.index:
            status = str(df.at[idx, "triage_status"]).strip().lower()
            step = self._as_float(df.at[idx, "triage_step_score"], default=float("nan"))
            white = self._as_float(df.at[idx, "triage_whiteness_score"], default=float("nan"))

            wdef_raw = str(df.at[idx, "triage_whiteness_definition"]).strip()
            if (wdef_raw == "") or (wdef_raw.lower() == "nan"):
                wdef_raw = self.noise_loader.handler.whiteness_definition()
            whiteness_is_pvalue = "pvalue" in wdef_raw.lower()
            df.at[idx, "triage_whiteness_definition"] = wdef_raw

            reasons = self._strip_retriage_managed_reasons(df.at[idx, "triage_why_not_usable"])
            if status != "ok":
                reasons.append(f"triage_status={status or 'unknown'}")
            if np.isfinite(step) and np.isfinite(self.noisy_step_threshold) and step > self.noisy_step_threshold:
                reasons.append(f"step_score>{self.noisy_step_threshold:.3f} ({step:.3f})")
            if np.isfinite(white) and np.isfinite(self.noisy_whiteness_threshold):
                if whiteness_is_pvalue:
                    if white < self.noisy_whiteness_threshold:
                        reasons.append(f"whiteness_pvalue<{self.noisy_whiteness_threshold:.3f} ({white:.3f})")
                elif white > self.noisy_whiteness_threshold:
                    reasons.append(f"whiteness_score>{self.noisy_whiteness_threshold:.3f} ({white:.3f})")

            # Deduplicate while preserving order.
            reasons = list(dict.fromkeys([r for r in reasons if str(r).strip() != ""]))

            triage_usable = (status == "ok") and (len(reasons) == 0)
            triage_why = ";".join(reasons)
            df.at[idx, "triage_usable"] = bool(triage_usable)
            df.at[idx, "triage_why_not_usable"] = triage_why

            triage_dict = {
                "status": status,
                "usable": bool(triage_usable),
                "why_not_usable": triage_why,
                "step_score": step,
                "whiteness_score": white,
                "whiteness_definition": wdef_raw,
            }
            hard_reasons = self._hard_fail_reasons(triage_dict)
            row_for_label = df.loc[idx].to_dict()
            label, reason = self._label_row(row=row_for_label, hard_reasons=hard_reasons)
            df.at[idx, "label"] = label
            df.at[idx, "label_reason"] = reason

        if write:
            df.to_csv(batch_path, index=False)

        print(
            f"[retriage] rows={len(df)} "
            f"whiteness_threshold={self.noisy_whiteness_threshold} "
            f"definition_default={self.noise_loader.handler.whiteness_definition()}"
        )
        return {"batch_results_csv": batch_path, "results_df": df}

    def rebuild_leaderboards(
        self,
        batch_csv: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        batch_path = Path(batch_csv) if batch_csv is not None else (self.out_dir / "batch_results.csv")
        periodic_csv = self.out_dir / "leaderboard_periodic.csv"
        sparse_csv = self.out_dir / "leaderboard_sparse.csv"
        top_shape_csv = self.out_dir / "leaderboard_top_shape.csv"
        top_snr_csv = self.out_dir / "leaderboard_top_snr.csv"

        if not batch_path.exists():
            raise FileNotFoundError(f"batch_results.csv not found: {batch_path}")

        df = pd.read_csv(batch_path)
        self._require_columns(df, ["label"], context="leaderboard rebuild")
        self._require_columns(
            df,
            ["best_shape_score", "best_depth_snr", "hit_rate_shape", "coverage_rate", "hit_rate_snr"],
            context="leaderboard rebuild",
        )

        work = df.copy()
        for c in ["best_shape_score", "best_depth_snr", "hit_rate_shape", "coverage_rate", "hit_rate_snr"]:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        periodic_df = work.loc[work["label"].astype(str) == "Periodic_candidate"].copy()
        periodic_df = periodic_df.sort_values(["hit_rate_shape", "best_shape_score"], ascending=[False, False])
        periodic_df.to_csv(periodic_csv, index=False)

        sparse_df = work.loc[work["label"].astype(str) == "Sparse_or_mono"].copy()
        sparse_df = sparse_df.sort_values(["best_shape_score", "best_depth_snr"], ascending=[False, False])
        sparse_df.to_csv(sparse_csv, index=False)

        top_shape_df = work.sort_values(["best_shape_score", "best_depth_snr"], ascending=[False, False]).head(200)
        top_shape_df.to_csv(top_shape_csv, index=False)
        top_snr_df = work.sort_values(["best_depth_snr", "best_shape_score"], ascending=[False, False]).head(200)
        top_snr_df.to_csv(top_snr_csv, index=False)

        label_counts = work["label"].astype(str).value_counts(dropna=False)
        skip_counts = self._skip_reason_counts(work)
        if len(periodic_df) == 0:
            print("[leaderboard_periodic] leaderboard empty: reason=0 rows after filtering")
            print(f"[leaderboard_periodic] label_counts={label_counts.to_dict()}")
            print(f"[leaderboard_periodic] skip_reason_counts={skip_counts.to_dict()}")
        if len(sparse_df) == 0:
            print("[leaderboard_sparse] leaderboard empty: reason=0 rows after filtering")
            print(f"[leaderboard_sparse] label_counts={label_counts.to_dict()}")
            print(f"[leaderboard_sparse] skip_reason_counts={skip_counts.to_dict()}")

        return {
            "batch_results_csv": batch_path,
            "leaderboard_periodic_csv": periodic_csv,
            "leaderboard_sparse_csv": sparse_csv,
            "leaderboard_top_shape_csv": top_shape_csv,
            "leaderboard_top_snr_csv": top_snr_csv,
            "results_df": work,
        }

    def _print_finalize_summary(self, results_df: pd.DataFrame) -> None:
        if len(results_df) == 0:
            print("[summary] total_processed=0")
            print("[summary] total_skipped_by_whiteness=0")
            print("[summary] total_skipped_by_n_points=0")
            print("[summary] total_with_events=0")
            print("[summary] total_with_period_validation_run=0")
            print("[summary] label_counts={'Periodic_candidate': 0, 'Sparse_or_mono': 0, 'Noisy_trash': 0, 'No_events': 0}")
            print("[summary] triage_error_missing_error_stage=0")
            print("[summary] top_error_stage_type=[]")
            return

        why = results_df.get("triage_why_not_usable", pd.Series([""] * len(results_df))).astype(str)
        total_skipped_by_whiteness = int(why.str.contains("whiteness", case=False, na=False).sum())
        total_skipped_by_n_points = int(why.str.contains("n_points<", case=False, na=False).sum())
        n_events = pd.to_numeric(results_df.get("n_events", 0), errors="coerce").fillna(0)
        n_valid = pd.to_numeric(results_df.get("n_periods_validated", 0), errors="coerce").fillna(0)
        labels = results_df.get("label", pd.Series([], dtype="string")).astype(str).value_counts(dropna=False).to_dict()
        label_counts = {
            "Periodic_candidate": int(labels.get("Periodic_candidate", 0)),
            "Sparse_or_mono": int(labels.get("Sparse_or_mono", 0)),
            "Noisy_trash": int(labels.get("Noisy_trash", 0)),
            "No_events": int(labels.get("No_events", 0)),
        }
        print(f"[summary] total_processed={int(len(results_df))}")
        print(f"[summary] total_skipped_by_whiteness={total_skipped_by_whiteness}")
        print(f"[summary] total_skipped_by_n_points={total_skipped_by_n_points}")
        print(f"[summary] total_with_events={int((n_events > 0).sum())}")
        print(f"[summary] total_with_period_validation_run={int((n_valid > 0).sum())}")
        print(f"[summary] label_counts={label_counts}")
        if ("triage_status" not in results_df.columns) or ("error_stage" not in results_df.columns) or ("error_type" not in results_df.columns):
            print("[summary] triage_error_missing_error_stage=0")
            print("[summary] top_error_stage_type=[]")
            return

        err_rows = results_df.copy()
        triage_status = err_rows["triage_status"].fillna("").astype(str).str.strip().str.lower()
        err_rows = err_rows.loc[triage_status == "error"].copy()
        if len(err_rows) == 0:
            print("[summary] triage_error_missing_error_stage=0")
            print("[summary] top_error_stage_type=[]")
            return

        err_rows["error_stage"] = err_rows["error_stage"].fillna("").astype(str).str.strip()
        err_rows["error_type"] = err_rows["error_type"].fillna("").astype(str).str.strip()
        missing_stage_count = int((err_rows["error_stage"] == "").sum())
        print(f"[summary] triage_error_missing_error_stage={missing_stage_count}")

        count_df = err_rows.copy()
        count_df["error_stage"] = count_df["error_stage"].replace({"": "<missing>"})
        count_df["error_type"] = count_df["error_type"].replace({"": "<missing>"})
        counts = (
            count_df.groupby(["error_stage", "error_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        print("[summary] top_error_stage_type:")
        for r in counts.itertuples(index=False):
            print(f"[summary] error_count stage={r.error_stage} type={r.error_type} count={int(r.count)}")

    def _load_progress(self) -> Optional[Dict[str, Any]]:
        if not self.progress_path.exists():
            return None
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _write_progress(self, last_completed_index: int, last_completed_query: str) -> None:
        payload = {
            "last_completed_index": int(last_completed_index),
            "last_completed_query": str(last_completed_query),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _resolve_resume_start(self, queries: Sequence[str]) -> int:
        if not self.resume:
            return 0
        data = self._load_progress()
        if data is None:
            print(f"[resume] No readable progress file at {self.progress_path}; starting from first query")
            return 0
        try:
            last_idx = int(data.get("last_completed_index", -1))
        except Exception:
            last_idx = -1
        start_idx = max(0, last_idx + 1)
        if start_idx >= len(queries):
            print("[resume] Progress indicates all queries are already completed")
        else:
            print(f"[resume] Resuming from query index {start_idx} ({start_idx + 1}/{len(queries)})")
        return start_idx

    @staticmethod
    def _append_batch_row(
        batch_csv: Path,
        row: Dict[str, Any],
        batch_columns: Optional[List[str]] = None,
    ) -> List[str]:
        row_df = pd.DataFrame([row])
        if batch_columns is None:
            batch_columns = row_df.columns.tolist()
            row_df.to_csv(batch_csv, mode="a", header=not batch_csv.exists(), index=False)
            return batch_columns

        for col in batch_columns:
            if col not in row_df.columns:
                row_df[col] = np.nan
        row_df.reindex(columns=batch_columns).to_csv(batch_csv, mode="a", header=False, index=False)
        return batch_columns

    @staticmethod
    def _ensure_batch_csv_columns(batch_csv: Path, required_columns: Sequence[str]) -> Optional[List[str]]:
        if not batch_csv.exists():
            return None
        try:
            df = pd.read_csv(batch_csv)
        except Exception:
            try:
                return pd.read_csv(batch_csv, nrows=0).columns.tolist()
            except Exception:
                return None

        missing = [c for c in required_columns if c not in df.columns]
        if len(missing) == 0:
            return list(df.columns)

        for c in missing:
            df[c] = ""
        df.to_csv(batch_csv, index=False)
        print(f"[batch schema] Added columns to existing batch_results.csv: {missing}")
        return list(df.columns)

    def run(self) -> Dict[str, Any]:
        queries = self._load_queries()
        if len(queries) == 0:
            raise ValueError("No queries to run")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epics_dir.mkdir(parents=True, exist_ok=True)

        batch_csv = self.out_dir / "batch_results.csv"

        if (not self.resume) and batch_csv.exists():
            if self.skip_existing_epics:
                print(f"[resume-by-dir] Keeping existing batch CSV: {batch_csv}")
            else:
                batch_csv.unlink()

        batch_columns: Optional[List[str]] = None
        required_batch_cols = ["error_stage", "error_type", "error_msg", "author_selected", "campaign_selected"]
        if batch_csv.exists():
            batch_columns = self._ensure_batch_csv_columns(batch_csv=batch_csv, required_columns=required_batch_cols)

        start_idx = self._resolve_resume_start(queries)

        used_slugs: set = set()
        query_slugs: List[str] = []
        for i, q in enumerate(queries, start=1):
            query_slugs.append(self._unique_slug(self._epic_slug(query=q, fallback_idx=i), used=used_slugs))

        rows: List[Dict[str, Any]] = []
        print(f"[K2BatchRunner] Running {len(queries)} queries")
        if start_idx >= len(queries):
            print("[K2BatchRunner] No new queries to process")
        existing_epic_dirs: set = set()
        if self.skip_existing_epics:
            existing_epic_dirs = {p.name for p in self.epics_dir.iterdir() if p.is_dir()}
            print(f"[K2BatchRunner] Existing EPIC dirs detected: {len(existing_epic_dirs)}")

        for idx in range(start_idx, len(queries)):
            query = queries[idx]
            slug = query_slugs[idx]
            epic_dir = self.epics_dir / slug

            if self.skip_existing_epics and slug in existing_epic_dirs:
                print(f"\n[{idx + 1}/{len(queries)}] {query} -> {slug}")
                print(f"[skip] existing epic directory: {epic_dir}")
                completed = idx + 1
                if (completed % 50 == 0) or (idx == (len(queries) - 1)):
                    self._write_progress(last_completed_index=idx, last_completed_query=query)
                continue

            epic_dir.mkdir(parents=True, exist_ok=True)
            existing_epic_dirs.add(slug)
            print(f"\n[{idx + 1}/{len(queries)}] {query} -> {slug}")

            row: Dict[str, Any] = {
                "query": str(query),
                "epic_id": slug,
                "epic_dir": str(epic_dir),
                "triage_status": "error",
                "triage_usable": False,
                "triage_score_global": float("nan"),
                "triage_n_points": 0,
                "triage_step_score": float("nan"),
                "triage_whiteness_score": float("nan"),
                "triage_whiteness_definition": "",
                "triage_why_not_usable": "",
                "error_stage": "",
                "error_type": "",
                "error_msg": "",
                "author_selected": "",
                "campaign_selected": "",
                "n_events": 0,
                "best_shape_score": float("nan"),
                "best_depth_snr": float("nan"),
                "n_periods_proposed": 0,
                "n_periods_validated": 0,
                "best_period": float("nan"),
                "hit_rate_shape": float("nan"),
                "hit_rate_snr": float("nan"),
                "coverage_rate": float("nan"),
                "best_period_hit_rate_shape": float("nan"),
                "best_period_hit_rate_snr": float("nan"),
                "best_period_coverage_rate": float("nan"),
                "events_csv": str(epic_dir / "events.csv"),
                "best_hits_csv": "",
                "best_misses_csv": "",
                "best_uncovered_csv": "",
                "best_hitmap_png": "",
                "best_phase_offset_png": "",
                "all_hitmap_pngs": "",
                "all_phase_offset_pngs": "",
                "label": "",
                "label_reason": "",
            }
            period_rows: List[Dict[str, Any]] = []
            hard_reasons: List[str] = []
            current_error_stage = "detect"

            try:
                det = self.detector.run_one(
                    query=query,
                    limit=self.limit,
                    exptime=self.exptime,
                    cache_only=self.cache_only,
                )
                triage = dict(det.get("summary", {}))
                events_df = pd.DataFrame(det.get("candidates", []))
                events_df.to_csv(row["events_csv"], index=False)
                best_shape, best_depth_snr = self._best_event_metrics(events_df)

                row.update(
                    {
                        "triage_status": str(triage.get("status", "error")),
                        "triage_usable": self._as_bool(triage.get("usable", False), default=False),
                        "triage_score_global": self._as_float(triage.get("score_global", float("nan"))),
                        "triage_n_points": int(self._as_float(triage.get("n_points", 0.0), default=0.0)),
                        "triage_step_score": self._as_float(triage.get("step_score", float("nan"))),
                        "triage_whiteness_score": self._as_float(triage.get("whiteness_score", float("nan"))),
                        "triage_whiteness_definition": str(triage.get("whiteness_definition", "")),
                        "triage_why_not_usable": str(triage.get("why_not_usable", "")),
                        "error_stage": str(triage.get("error_stage", "")),
                        "error_type": str(triage.get("error_type", "")),
                        "error_msg": str(triage.get("error_msg", ""))[:200],
                        "author_selected": str(triage.get("author_selected", triage.get("author", ""))),
                        "campaign_selected": str(triage.get("campaign_selected", "")),
                        "n_events": int(len(events_df)),
                        "best_shape_score": float(best_shape),
                        "best_depth_snr": float(best_depth_snr),
                    }
                )
                if (str(row.get("triage_status", "")).strip().lower() == "error") and (str(row.get("error_stage", "")).strip() == ""):
                    row["error_stage"] = "detect"
                if (str(row.get("triage_status", "")).strip().lower() == "error") and (str(row.get("error_type", "")).strip() == ""):
                    row["error_type"] = "RuntimeError"
                if (str(row.get("triage_status", "")).strip().lower() == "error") and (str(row.get("error_msg", "")).strip() == ""):
                    row["error_msg"] = "triage_status=error"
                self._debug_whiteness_gate(query=query, triage=triage)

                hard_reasons = self._hard_fail_reasons(triage)
                if len(hard_reasons) == 0 and len(events_df) > 0:
                    current_error_stage = "period"
                    t, f = self._fetch_clean_time_flux(query=query)
                    proposals = self._propose_periods(events_df)
                    row["n_periods_proposed"] = int(len(proposals))

                    for proposal in proposals.itertuples(index=False):
                        p = float(proposal.period)
                        p_tag = f"{p:.6f}"
                        t0_guess = self._guess_t0(events_df=events_df, period=p, in_cluster=list(proposal.in_cluster_indices), time=t)
                        val = self.validator.validate(time=t, flux=f, P=p, t0=t0_guess, quality_mask=None)
                        self._print_soft_top10(scores_df=val.get("scores_df", pd.DataFrame()), period=p)

                        hits_csv = epic_dir / f"period_{p_tag}_hits.csv"
                        misses_csv = epic_dir / f"period_{p_tag}_misses.csv"
                        uncovered_csv = epic_dir / f"period_{p_tag}_uncovered.csv"
                        hitmap_png = epic_dir / f"period_{p_tag}_hitmap.png"
                        phase_png = epic_dir / f"period_{p_tag}_phase_offset.png"

                        hits_df = val["hits_df"]
                        misses_df = val["misses_df"]
                        uncovered_df = val["uncovered_df"]
                        no_cand_df = self._split_no_cand(misses_df)
                        print(
                            f"P={p:.6f} hits={len(hits_df)} misses={len(misses_df)} "
                            f"no_cand={len(no_cand_df)} uncovered={len(uncovered_df)}"
                        )

                        hits_df.to_csv(hits_csv, index=False)
                        misses_df.to_csv(misses_csv, index=False)
                        uncovered_df.to_csv(uncovered_csv, index=False)
                        self.validator.plot_hitmap(hits_csv=hits_csv, misses_csv=misses_csv, uncovered_csv=uncovered_csv, outpath=hitmap_png, P=p)
                        self.validator.plot_phase_offset(
                            hits_csv=hits_csv,
                            misses_csv=misses_csv,
                            uncovered_csv=uncovered_csv,
                            outpath=phase_png,
                            P=p,
                            score_col="best_shape_score",
                        )

                        period_rows.append(
                            {
                                "period": float(p),
                                "hit_rate_shape": self._as_float(val.get("hit_rate_shape", float("nan"))),
                                "hit_rate_snr": self._as_float(val.get("hit_rate_snr", float("nan"))),
                                "coverage_rate": self._as_float(val.get("coverage_rate", float("nan"))),
                                "hits_csv": str(hits_csv),
                                "misses_csv": str(misses_csv),
                                "uncovered_csv": str(uncovered_csv),
                                "hitmap_png": str(hitmap_png),
                                "phase_offset_png": str(phase_png),
                            }
                        )
                elif len(hard_reasons) > 0:
                    print(f"[skip] hard triage fail: {'; '.join(hard_reasons)}")
                else:
                    print("[skip] no events detected; skipping period workflow")

            except Exception as e:
                row["triage_status"] = "error"
                row["triage_usable"] = False
                row["error_stage"] = str(current_error_stage if str(current_error_stage).strip() != "" else "detect")
                row["error_type"] = type(e).__name__
                row["error_msg"] = str(e)[:200]
                hard_reasons.append(f"pipeline_error={type(e).__name__}:{e}")
                print(f"[error] {query}: {type(e).__name__}: {e}")

            if len(period_rows) > 0:
                p_df = pd.DataFrame(period_rows).sort_values(
                    ["hit_rate_shape", "coverage_rate", "hit_rate_snr"],
                    ascending=[False, False, False],
                )
                best = p_df.iloc[0]
                row.update(
                    {
                        "n_periods_validated": int(len(p_df)),
                        "best_period": self._as_float(best["period"]),
                        "hit_rate_shape": self._as_float(best["hit_rate_shape"]),
                        "hit_rate_snr": self._as_float(best["hit_rate_snr"]),
                        "coverage_rate": self._as_float(best["coverage_rate"]),
                        "best_period_hit_rate_shape": self._as_float(best["hit_rate_shape"]),
                        "best_period_hit_rate_snr": self._as_float(best["hit_rate_snr"]),
                        "best_period_coverage_rate": self._as_float(best["coverage_rate"]),
                        "best_hits_csv": str(best["hits_csv"]),
                        "best_misses_csv": str(best["misses_csv"]),
                        "best_uncovered_csv": str(best["uncovered_csv"]),
                        "best_hitmap_png": str(best["hitmap_png"]),
                        "best_phase_offset_png": str(best["phase_offset_png"]),
                        "all_hitmap_pngs": ";".join(p_df["hitmap_png"].astype(str).tolist()),
                        "all_phase_offset_pngs": ";".join(p_df["phase_offset_png"].astype(str).tolist()),
                    }
                )

            label, reason = self._label_row(row=row, hard_reasons=hard_reasons)
            row["label"] = label
            row["label_reason"] = reason
            rows.append(row)
            batch_columns = self._append_batch_row(batch_csv=batch_csv, row=row, batch_columns=batch_columns)
            completed = idx + 1
            if (completed % 50 == 0) or (idx == (len(queries) - 1)):
                self._write_progress(last_completed_index=idx, last_completed_query=query)

        rebuilt = self.rebuild_leaderboards(batch_csv=batch_csv)
        results_df = rebuilt["results_df"]
        self._print_finalize_summary(results_df=results_df)

        print(f"\n[K2BatchRunner] Wrote: {rebuilt['batch_results_csv']}")
        print(f"[K2BatchRunner] Wrote: {rebuilt['leaderboard_periodic_csv']}")
        print(f"[K2BatchRunner] Wrote: {rebuilt['leaderboard_sparse_csv']}")
        print(f"[K2BatchRunner] Wrote: {rebuilt['leaderboard_top_shape_csv']}")
        print(f"[K2BatchRunner] Wrote: {rebuilt['leaderboard_top_snr_csv']}")
        return {
            "out_dir": self.out_dir,
            "batch_results_csv": rebuilt["batch_results_csv"],
            "leaderboard_periodic_csv": rebuilt["leaderboard_periodic_csv"],
            "leaderboard_sparse_csv": rebuilt["leaderboard_sparse_csv"],
            "leaderboard_top_shape_csv": rebuilt["leaderboard_top_shape_csv"],
            "leaderboard_top_snr_csv": rebuilt["leaderboard_top_snr_csv"],
            "results_df": results_df,
        }


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run K2 batch scoring pipeline.")
    p.add_argument("--out-dir", required=True, help="Output directory for batch artifacts.")
    p.add_argument(
        "--rebuild-leaderboards",
        action="store_true",
        help="Rebuild leaderboard CSVs from an existing batch_results.csv without processing queries.",
    )
    p.add_argument("--input", default=None, help="Path to batch_results.csv used with --rebuild-leaderboards.")
    src_group = p.add_mutually_exclusive_group(required=False)
    src_group.add_argument("--queries", nargs="+", help="One or more target queries (supports comma-separated tokens).")
    src_group.add_argument("--input-csv", help="CSV path containing query column.")
    p.add_argument("--query-col", default="query", help="Query column name when using --input-csv.")
    p.add_argument("--limit", type=int, default=50, help="Lightkurve search result limit.")
    p.add_argument("--exptime", default=None, help="Optional cadence filter (e.g. 'long', 'short', or seconds).")
    p.add_argument("--noise-mode", default="strict", choices=["strict", "discovery"], help="Noise triage mode.")
    p.add_argument("--resume", action="store_true", help="Resume from out_dir/progress.json if present.")
    p.add_argument("--cache-only", action="store_true", help="Process using local cache only; never download missing products.")
    return p


def main() -> None:
    args = _build_cli_parser().parse_args()
    if (not args.rebuild_leaderboards) and (args.queries is None) and (args.input_csv is None):
        raise SystemExit("Provide --queries or --input-csv (or use --rebuild-leaderboards).")
    runner = K2BatchRunner(
        out_dir=args.out_dir,
        queries=args.queries,
        input_csv=args.input_csv,
        query_col=args.query_col,
        noise_mode=args.noise_mode,
        limit=args.limit,
        exptime=args.exptime,
        resume=args.resume,
        cache_only=args.cache_only,
    )
    if args.rebuild_leaderboards:
        batch_csv = Path(args.input) if args.input else (Path(args.out_dir) / "batch_results.csv")
        if (args.input is not None) and (not batch_csv.exists()) and (not batch_csv.is_absolute()):
            alt = Path(args.out_dir) / batch_csv
            if alt.exists():
                batch_csv = alt
        out = runner.rebuild_leaderboards(batch_csv=batch_csv)
        runner._print_finalize_summary(results_df=out["results_df"])
        print(f"[K2BatchRunner] Wrote: {out['leaderboard_periodic_csv']}")
        print(f"[K2BatchRunner] Wrote: {out['leaderboard_sparse_csv']}")
        print(f"[K2BatchRunner] Wrote: {out['leaderboard_top_shape_csv']}")
        print(f"[K2BatchRunner] Wrote: {out['leaderboard_top_snr_csv']}")
        return
    runner.run()


if __name__ == "__main__":
    main()
