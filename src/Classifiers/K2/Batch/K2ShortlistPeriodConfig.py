from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class K2ShortlistPeriodConfig:
    # Operating modes:
    # - MIN_CLUSTER_COUNT=3: precision-first default
    # - MIN_CLUSTER_COUNT=2: supported high-recall mode with cluster_count==2 review/guardrails
    RAW_EPIC_LIST_CSV: str = r"plots\k2_batch\batch_results_whiteness.csv"
    RAW_EPIC_QUERY_COL: str = "query"
    SHORTLIST_CSV: str = r"plots\k2_batch\shortlist_top_shape_for_period.csv"
    EPICS_DIR: str = r"plots\k2_batch\epics"
    OUT_DIR: str = r"plots\k2_batch"
    OUT_SUMMARY_CSV: str = r"period_shortlist_summary.csv"
    OUT_SUMMARY_UNIQUE_EPICP_CSV: str = r"period_shortlist_summary_unique_epicP.csv"
    OUT_SUMMARY_VALIDATED_ONLY_CSV: str = r"period_shortlist_summary_validated_only.csv"
    OUT_BEST_CSV: str = r"period_shortlist_best.csv"
    OUT_QUARANTINE_CSV: str = r"period_shortlist_quarantine.csv"
    OUT_DIAGNOSTICS_CSV: str = r"period_shortlist_diagnostics.csv"
    OUT_EPIC_FUNNEL_REASONS_CSV: str = r"epic_funnel_reasons.csv"
    OUT_PERIOD_HIST_PNG: str = r"period_hist_summary_vs_best.png"
    OUT_PERIOD_HIST_COUNTS_CSV: str = r"period_hist_summary_vs_best_counts.csv"
    RUN_ID: Optional[str] = r"slice_200"
    USE_RUN_SUBDIR: bool = True

    MAX_TARGETS: Optional[int] = None
    START_INDEX: int = 0
    END_INDEX: Optional[int] = None
    PERIOD_STAGE_MODE: Optional[str] = None
    PERIOD_STAGE_K: Optional[int] = 50 #200
    PERIOD_STAGE_N: Optional[int] = 50 #25771
    PERIOD_STAGE_SELECTION_MODE: str = "randomN"
    PERIOD_STAGE_MAX_EPICS: Optional[int] = None
    PERIOD_STAGE_RANDOM_SAMPLE_SIZE: Optional[int] = None
    PERIOD_STAGE_RANDOM_SEED: int = 42

    PERIOD_TOL_PHASE: float = 0.03
    SOFT_SNR_T: float = 3.0
    SOFT_MIN_RUN: int = 2
    MIN_PERIOD_DAYS: float = 0.5
    PERIOD_CAP_DAYS: float = 20.0
    MAX_PERIOD_DAYS: float = 20.0
    PERIOD_HARD_MAX_DAYS: float = 20.0
    PERIOD_BIN_EDGES_DAYS: Tuple[float, ...] = (1.0, 5.0, 10.0, 15.0, 20.0)
    BEST_SELECTION_BIN_MODE: str = "match_summary_distribution"
    NULL_P_RATE_MAX: float = 0.001
    NULL_P_RATE_EXEMPT_SOURCE_REASONS: Tuple[str, ...] = (
        "no_cluster_periods",
        "events_filtered_to_zero",
    )
    MIN_CLUSTER_COUNT: int = 3
    PRECISION_FIRST_MODE_NAME: str = "precision_first_default"
    HIGH_RECALL_MODE_NAME: str = "supported_high_recall"
    MANUAL_REVIEW_CLUSTER_COUNT_EQ: int = 2
    CLUSTER2_VALIDATED_MIN_HIT_RATE_SHAPE: Optional[float] = 0.10
    CLUSTER2_VALIDATED_MIN_SOFT_HIT_RATE: Optional[float] = 0.10
    CLUSTER2_REVIEW_VERY_SHORT_PERIOD_DAYS_MAX: float = 1.0
    CLUSTER2_REVIEW_LOW_EVENT_SUPPORT_MAX: int = 2
    CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SHAPE_MAX: float = 0.05
    CLUSTER2_REVIEW_NEAR_ZERO_HIT_RATE_SNR_MAX: float = 0.05
    ENABLE_VALIDATION: bool = True
    CACHE_ONLY_FIRST: bool = False
    DOWNLOAD_IF_CACHE_MISS: bool = True
    TOP_K_PERIODS: int = 3
    VALIDATION_TOP_K: int = 3

    @property
    def shortlist_csv_path(self) -> Path:
        return Path(self.SHORTLIST_CSV)

    @property
    def raw_epic_list_csv_path(self) -> Path:
        return Path(self.RAW_EPIC_LIST_CSV)

    @property
    def epics_dir_path(self) -> Path:
        return Path(self.EPICS_DIR)

    @property
    def out_dir_path(self) -> Path:
        return Path(self.OUT_DIR)

    @property
    def out_summary_csv_path(self) -> Path:
        return Path(self.OUT_SUMMARY_CSV)

    @property
    def out_best_csv_path(self) -> Path:
        return Path(self.OUT_BEST_CSV)

    @property
    def out_summary_unique_epicp_csv_path(self) -> Path:
        return Path(self.OUT_SUMMARY_UNIQUE_EPICP_CSV)

    @property
    def out_summary_validated_only_csv_path(self) -> Path:
        return Path(self.OUT_SUMMARY_VALIDATED_ONLY_CSV)

    @property
    def out_quarantine_csv_path(self) -> Path:
        return Path(self.OUT_QUARANTINE_CSV)

    @property
    def out_diagnostics_csv_path(self) -> Path:
        return Path(self.OUT_DIAGNOSTICS_CSV)

    @property
    def out_epic_funnel_reasons_csv_path(self) -> Path:
        return Path(self.OUT_EPIC_FUNNEL_REASONS_CSV)

    @property
    def out_period_hist_png_path(self) -> Path:
        return Path(self.OUT_PERIOD_HIST_PNG)

    @property
    def out_period_hist_counts_csv_path(self) -> Path:
        return Path(self.OUT_PERIOD_HIST_COUNTS_CSV)
