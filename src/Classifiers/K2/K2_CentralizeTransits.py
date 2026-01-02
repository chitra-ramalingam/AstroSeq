import numpy as np

class K2_CentralizeSegmentTransit:
    def __init__(self):
        pass

    def _overlap(self, a0, a1, b0, b1) -> float:
            """Overlap length between [a0,a1] and [b0,b1]."""
            return max(0.0, min(a1, b1) - max(a0, b0))

    def label_segment_centered(self,
        seg_t0: float,
        seg_t1: float,
        t0: float,
        period: float,
        dur_days: float,
        center_keep_frac: float = 0.2,     # keep central 50% of window
        min_dur_coverage: float = 0.6      # require 60% of transit duration inside window
    ) -> int:
        """
        Returns 1 if the segment is a positive transit window under:
        - a transit mid-time is inside the central region of the segment
        - the segment covers at least min_dur_coverage of the transit duration

        seg_t0/seg_t1 are segment start/end times in DAYS (same units as t0/period).
        t0 is reference transit mid-time (same units).
        period is orbital period in DAYS.
        dur_days is transit duration in DAYS.
        """
        if period <= 0 or dur_days <= 0:
            return 0

        # Segment center + central "keep" margin
        seg_span = seg_t1 - seg_t0
        if seg_span <= 0:
            return 0

        seg_center = 0.5 * (seg_t0 + seg_t1)
        center_margin = 0.5 * center_keep_frac * seg_span  # e.g. keep_frac=0.5 -> margin=0.25*span

        # Find all transit epochs that could overlap this segment (including edge overlap)
        # We expand the search window by dur/2 so we don't miss a transit that slightly overlaps.
        search_start = seg_t0 - 0.5 * dur_days
        search_end   = seg_t1 + 0.5 * dur_days

        n_min = int(np.floor((search_start - t0) / period))
        n_max = int(np.ceil((search_end   - t0) / period))
        if n_max < n_min:
            return 0

        # Check candidate transit mid-times for BOTH centrality + coverage
        for n in range(n_min, n_max + 1):
            t_mid = t0 + n * period

            # Centrality: transit mid-time must be near the segment center
            if abs(t_mid - seg_center) > center_margin:
                continue

            # Coverage: segment must include enough of the transit duration
            tr0 = t_mid - 0.5 * dur_days
            tr1 = t_mid + 0.5 * dur_days
            cov = self._overlap(seg_t0, seg_t1, tr0, tr1) / dur_days

            if cov >= min_dur_coverage:
                return 1

        return 0
