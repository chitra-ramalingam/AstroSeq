from __future__ import annotations

import re
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
        "failure_category",
        "failure_detail",
        "n_events_raw",
        "n_events_after_filters",
        "infer_max_period_days",
        "infer_min_hits",
        "infer_tol_frac",
        "min_cluster_count",
        "period_cap_days",
        "P",
    ]

    def __init__(self, config: Optional[K2ShortlistPeriodConfig] = None) -> None:
        self.config = config if config is not None else K2ShortlistPeriodConfig()
        self._period_file_re = re.compile(r"^period_([0-9]+(?:\.[0-9]+)?)_(hits|misses|uncovered)\.csv$", flags=re.IGNORECASE)

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
        edges = sorted({float(x) for x in configured_edges if np.isfinite(float(x)) and float(x) > 0})
        max_period = float(self._effective_max_period_days())
        if len(edges) == 0:
            edges = [1.0, 5.0, 10.0, 15.0, max_period]
        edges = [e for e in edges if e < max_period] + [max_period]
        edges = sorted({float(x) for x in edges if np.isfinite(float(x)) and float(x) > 0})
        if len(edges) < 2:
            low = max(1e-6, max_period - 1.0)
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
        quarantine["failure_category"] = ""
        quarantine["failure_detail"] = ""
        quarantine["infer_max_period_days"] = np.nan
        quarantine["infer_min_hits"] = np.nan
        quarantine["infer_tol_frac"] = np.nan
        quarantine["min_cluster_count"] = np.nan
        quarantine["period_cap_days"] = np.nan
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
        for c in self.QUARANTINE_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan if c in {"P", "n_events_raw", "n_events_after_filters", "infer_max_period_days", "infer_min_hits", "infer_tol_frac", "min_cluster_count", "period_cap_days"} else ""

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

        return out.reindex(columns=self.QUARANTINE_COLUMNS)

    def _dedupe_epic_period_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df.copy().reindex(columns=self.SUMMARY_COLUMNS)
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
        dedup = dedup.reindex(columns=self.SUMMARY_COLUMNS).reset_index(drop=True)
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
            edges = np.asarray([1.0, float(self._effective_max_period_days())], dtype=float)
        labels = [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(edges.size - 1)]

        p_summary = pd.to_numeric(summary_df.get("P", np.nan), errors="coerce").to_numpy(dtype=float)
        p_best = pd.to_numeric(best_df.get("P", np.nan), errors="coerce").to_numpy(dtype=float)
        p_summary = p_summary[np.isfinite(p_summary)]
        p_best = p_best[np.isfinite(p_best)]

        hist_summary, _ = np.histogram(p_summary, bins=edges)
        hist_best, _ = np.histogram(p_best, bins=edges)
        counts_df = pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "bin_label": labels,
                "summary_count": hist_summary.astype(int),
                "best_count": hist_best.astype(int),
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
            label=f"Summary (n={len(p_summary)})",
        )
        ax.hist(
            p_best,
            bins=edges,
            alpha=0.45,
            color="#F58518",
            label=f"Best (n={len(p_best)})",
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
        out_summary_csv = cfg.out_summary_csv_path
        out_summary_unique_epicp_csv = cfg.out_summary_unique_epicp_csv_path
        out_summary_validated_only_csv = cfg.out_summary_validated_only_csv_path
        out_best_csv = cfg.out_best_csv_path
        out_quarantine_csv = cfg.out_quarantine_csv_path
        out_diagnostics_csv = cfg.out_diagnostics_csv_path
        out_period_hist_png = cfg.out_period_hist_png_path
        out_period_hist_counts_csv = cfg.out_period_hist_counts_csv_path

        if not shortlist_csv.exists():
            raise FileNotFoundError(f"Shortlist CSV not found: {shortlist_csv}")
        if not epics_dir.exists():
            raise FileNotFoundError(f"EPICS directory not found: {epics_dir}")

        shortlist_df = pd.read_csv(shortlist_csv)
        if "query" not in shortlist_df.columns:
            raise ValueError(f"Column 'query' not found in {shortlist_csv}")

        queries = shortlist_df["query"].dropna().astype(str).tolist()
        start = max(0, int(cfg.START_INDEX))
        end = len(queries) - 1 if cfg.END_INDEX is None else min(int(cfg.END_INDEX), len(queries) - 1)
        selected = queries[start : end + 1] if end >= start else []
        if cfg.MAX_TARGETS is not None:
            selected = selected[: max(0, int(cfg.MAX_TARGETS))]

        if out_summary_csv.exists():
            out_summary_csv.unlink()
        if out_summary_unique_epicp_csv.exists():
            out_summary_unique_epicp_csv.unlink()
        if out_summary_validated_only_csv.exists():
            out_summary_validated_only_csv.unlink()
        if out_best_csv.exists():
            out_best_csv.unlink()
        if out_quarantine_csv.exists():
            out_quarantine_csv.unlink()
        if out_diagnostics_csv.exists():
            out_diagnostics_csv.unlink()
        if out_period_hist_png.exists():
            out_period_hist_png.unlink()
        if out_period_hist_counts_csv.exists():
            out_period_hist_counts_csv.unlink()

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

        print(f"[K2ShortlistPeriodRunner] targets={len(selected)} shortlist={shortlist_csv}")
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

        diagnostics_row = {
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
        }
        out_diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([diagnostics_row]).to_csv(out_diagnostics_csv, index=False)

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
        print(f"[K2ShortlistPeriodRunner] wrote diagnostics: {out_diagnostics_csv}")
        print(f"[K2ShortlistPeriodRunner] wrote period histogram: {out_period_hist_png}")
        print(f"[K2ShortlistPeriodRunner] wrote period histogram counts: {out_period_hist_counts_csv}")
        return {
            "shortlist_csv": shortlist_csv,
            "out_summary_csv": out_summary_csv,
            "out_summary_unique_epicp_csv": out_summary_unique_epicp_csv,
            "out_summary_validated_only_csv": out_summary_validated_only_csv,
            "out_best_csv": out_best_csv,
            "out_quarantine_csv": out_quarantine_csv,
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
