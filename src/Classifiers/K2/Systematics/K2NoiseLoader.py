from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from src.Classifiers.K2.Systematics.K2_NoiseHandler import (
    K2PipelineStageError,
    K2NoiseConfig,
    K2NoiseMetrics,
    K2_NoiseHandler,
)


@dataclass
class K2NoiseLoaderConfig:
    limit: int = 50
    exptime: Optional[Union[str, float]] = None
    flatten: bool = False
    per_segment: bool = False
    mode: str = "strict"
    cache_only: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower().strip()
        if self.mode not in {"strict", "discovery"}:
            raise ValueError(f"Unsupported mode {self.mode!r}. Expected 'strict' or 'discovery'.")


class K2NoiseLoader:
    """
    K2NoiseLoader

    Thin orchestration layer around K2_NoiseHandler.
    It accepts star queries (e.g. 'EPIC 211797674'), runs fetch->clean->metrics,
    and returns serializable result rows ready for dataframe/csv export.
    """

    def __init__(
        self,
        loader_config: Optional[K2NoiseLoaderConfig] = None,
        noise_config: Optional[K2NoiseConfig] = None,
        handler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.loader_config = loader_config if loader_config is not None else K2NoiseLoaderConfig()
        kwargs: Dict[str, Any] = dict(handler_kwargs or {})
        if ("mode" not in kwargs) and (noise_config is None):
            kwargs["mode"] = self.loader_config.mode
        if noise_config is not None:
            kwargs["noise_config"] = noise_config
        self.handler = K2_NoiseHandler(**kwargs)

    @staticmethod
    def _metrics_to_dict(m: K2NoiseMetrics) -> Dict[str, Any]:
        return asdict(m)

    @staticmethod
    def _format_error(exc: Exception) -> Dict[str, str]:
        return {"error_type": type(exc).__name__, "error_msg": str(exc)[:200]}

    def _error_row(
        self,
        base: Dict[str, Any],
        stage: str,
        exc: Exception,
        author_selected: str = "",
        campaign_selected: str = "",
        status: str = "error",
        search_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(exc, K2PipelineStageError):
            err = {"error_type": str(exc.error_type), "error_msg": str(exc.error_msg)[:200]}
        else:
            err = self._format_error(exc)
        row = dict(base)
        row.update(
            {
                "status": str(status),
                "error_stage": str(stage),
                "error_type": err["error_type"],
                "error_msg": err["error_msg"],
                "author_selected": str(author_selected),
                "campaign_selected": str(campaign_selected),
                "author": str(author_selected),
                "search_result": search_result if search_result is not None else {},
                "why_not_usable": f"{stage}:{err['error_type']}",
                "notes": f"{stage}:{err['error_type']}",
                "n_points": 0,
                "baseline_days": 0.0,
                "duty_cycle": 0.0,
                "mad": np.nan,
                "robust_sigma": np.nan,
                "outlier_rate_6sigma": np.nan,
                "outlier_rate_global": np.nan,
                "step_score": np.nan,
                "whiteness_score": np.nan,
            }
        )
        return row

    def run_queries(
        self,
        queries: Iterable[str],
        limit: Optional[int] = None,
        exptime: Optional[Union[str, float]] = None,
        flatten: Optional[bool] = None,
        per_segment: Optional[bool] = None,
        cache_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        use_limit = int(self.loader_config.limit if limit is None else limit)
        use_exptime = self.loader_config.exptime if exptime is None else exptime
        use_flatten = bool(self.loader_config.flatten if flatten is None else flatten)
        use_per_segment = bool(self.loader_config.per_segment if per_segment is None else per_segment)
        use_cache_only = bool(self.loader_config.cache_only if cache_only is None else cache_only)

        for q in queries:
            query = str(q).strip()
            if query == "":
                continue
            rows.append(
                self.run_one(
                    query=query,
                    limit=use_limit,
                    exptime=use_exptime,
                    flatten=use_flatten,
                    per_segment=use_per_segment,
                    cache_only=use_cache_only,
                )
            )

        return rows

    def run_one(
        self,
        query: str,
        limit: Optional[int] = None,
        exptime: Optional[Union[str, float]] = None,
        flatten: Optional[bool] = None,
        per_segment: Optional[bool] = None,
        cache_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        use_limit = int(self.loader_config.limit if limit is None else limit)
        use_exptime = self.loader_config.exptime if exptime is None else exptime
        use_flatten = bool(self.loader_config.flatten if flatten is None else flatten)
        use_per_segment = bool(self.loader_config.per_segment if per_segment is None else per_segment)
        use_cache_only = bool(self.loader_config.cache_only if cache_only is None else cache_only)

        base: Dict[str, Any] = {
            "query": query,
            "status": "ok",
            "author": "",
            "author_selected": "",
            "campaign_selected": "",
            "whiteness_definition": self.handler.whiteness_definition(),
            "score": float("-inf"),
            "score_global": float("-inf"),
            "score_best_seg": float("-inf"),
            "score_median_seg": float("-inf"),
            "score_worst_seg": float("-inf"),
            "usable": False,
            "why_not_usable": "",
            "search_result": {},
            "segments": [],
            "n_segments": 0,
            "error_stage": "",
            "error_type": "",
            "error_msg": "",
        }

        fetched: Dict[str, Any]
        try:
            fetched = self.handler.fetch_best(
                query=query,
                limit=use_limit,
                exptime=use_exptime,
                cache_only=use_cache_only,
            )
        except K2PipelineStageError as e:
            return self._error_row(
                base=base,
                stage=e.stage,
                exc=e,
                author_selected=e.author_selected,
                campaign_selected=e.campaign_selected,
            )
        except Exception as e:
            return self._error_row(base=base, stage="search", exc=e)

        if str(fetched.get("status", "ok")).lower() != "ok":
            fetch_status = str(fetched.get("status", "error"))
            row = dict(base)
            row.update(
                {
                    "status": fetch_status,
                    "author": str(fetched.get("author", "")),
                    "author_selected": str(fetched.get("author_selected", fetched.get("author", ""))),
                    "campaign_selected": str(fetched.get("campaign_selected", "")),
                    "whiteness_definition": self.handler.whiteness_definition(),
                    "search_result": fetched.get("search_result", {}),
                    "why_not_usable": fetch_status,
                    "notes": f"fetch_status={fetch_status}",
                    "n_points": 0,
                    "baseline_days": 0.0,
                    "duty_cycle": 0.0,
                    "mad": np.nan,
                    "robust_sigma": np.nan,
                    "outlier_rate_6sigma": np.nan,
                    "outlier_rate_global": np.nan,
                    "step_score": np.nan,
                    "whiteness_score": np.nan,
                }
            )
            return row

        author_selected = str(fetched.get("author_selected", fetched.get("author", "")))
        campaign_selected = str(fetched.get("campaign_selected", ""))

        try:
            cleaned = self.handler.clean(fetched["lc"], flatten=use_flatten)
        except Exception as e:
            return self._error_row(
                base=base,
                stage="clean",
                exc=e,
                author_selected=author_selected,
                campaign_selected=campaign_selected,
                search_result=fetched.get("search_result", {}),
            )

        try:
            metric_obj = self.handler.metrics(
                cleaned["time"],
                cleaned["flux"],
                notes=cleaned["notes"],
                per_segment=use_per_segment,
            )
        except Exception as e:
            return self._error_row(
                base=base,
                stage="metrics",
                exc=e,
                author_selected=author_selected,
                campaign_selected=campaign_selected,
                search_result=fetched.get("search_result", {}),
            )

        try:
            if use_per_segment:
                global_m = metric_obj["global"]
                seg_metrics = metric_obj["segments"]
            else:
                global_m = metric_obj
                seg_metrics = []

            explain_global = self.handler.explain(global_m)
            fail_reasons = list(explain_global.get("fail_reasons", []))

            score_global = float(self.handler.score(global_m))
            score_best_seg = float(self.handler.score_segments(seg_metrics, "best")) if use_per_segment else float("-inf")
            score_median_seg = (
                float(self.handler.score_segments(seg_metrics, "median")) if use_per_segment else float("-inf")
            )
            score_worst_seg = float(self.handler.score_segments(seg_metrics, "worst")) if use_per_segment else float("-inf")

            catastrophic_limit = float(self.handler.noise_config.catastrophic_outlier_rate_6sigma)
            global_outlier = (
                float(global_m.outlier_rate_global)
                if np.isfinite(global_m.outlier_rate_global)
                else float(global_m.outlier_rate_6sigma)
            )
            catastrophic_outlier = (not np.isfinite(global_outlier)) or (global_outlier > catastrophic_limit)

            if use_per_segment:
                usable = bool((score_best_seg > 0.0) and (not catastrophic_outlier))
            else:
                usable = bool(score_global > 0.0)

            row = dict(base)
            row.update(self._metrics_to_dict(global_m))
            row["author"] = author_selected
            row["author_selected"] = author_selected
            row["campaign_selected"] = campaign_selected
            row["whiteness_definition"] = self.handler.whiteness_definition()
            row["search_result"] = fetched.get("search_result", {})
            row["score"] = score_global
            row["score_global"] = score_global
            row["score_best_seg"] = score_best_seg
            row["score_median_seg"] = score_median_seg
            row["score_worst_seg"] = score_worst_seg
            row["usable"] = usable
            why_not = [str(r) for r in fail_reasons if str(r).strip() != ""]
            if use_per_segment and catastrophic_outlier:
                why_not.append(f"outlier_rate_global>{catastrophic_limit}")
            row["why_not_usable"] = ";".join(why_not)
            row["n_segments"] = len(seg_metrics)
            row["segments"] = [self._metrics_to_dict(ms) for ms in seg_metrics]
            return row
        except Exception as e:
            return self._error_row(
                base=base,
                stage="metrics",
                exc=e,
                author_selected=author_selected,
                campaign_selected=campaign_selected,
                search_result=fetched.get("search_result", {}),
            )

    def run_from_csv(
        self,
        csv_path: Union[str, Path],
        query_col: str = "query",
        limit: Optional[int] = None,
        exptime: Optional[Union[str, float]] = None,
        flatten: Optional[bool] = None,
        per_segment: Optional[bool] = None,
        cache_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        df = pd.read_csv(csv_path)
        if query_col not in df.columns:
            raise ValueError(f"Column '{query_col}' not found in {csv_path}. Columns: {list(df.columns)}")
        queries = df[query_col].dropna().astype(str).tolist()
        return self.run_queries(
            queries=queries,
            limit=limit,
            exptime=exptime,
            flatten=flatten,
            per_segment=per_segment,
            cache_only=cache_only,
        )

    @staticmethod
    def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)
