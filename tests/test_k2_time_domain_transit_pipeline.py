import unittest
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from src.Classifiers.K2.K2_TimeDomainTransitPipeline import (
    K2TimeDomainPreprocessConfig,
    K2TimeDomainPreprocessor,
    K2TimeDomainRankConfig,
    K2TimeDomainTransitRanker,
    infer_periods_from_events,
    phase_cluster_score,
)


class TestK2TimeDomainPreprocessor(unittest.TestCase):
    def test_asymmetric_outlier_handling(self) -> None:
        cfg = K2TimeDomainPreprocessConfig(
            positive_outlier_sigma=3.0,
            positive_clip_sigma=2.0,
            negative_outlier_sigma=3.0,
            min_negative_run_keep=2,
        )
        pre = K2TimeDomainPreprocessor(config=cfg)

        x = np.zeros(40, dtype=float)
        x[5] = 10.0
        x[10] = -7.0
        x[20:22] = -6.0
        sigma = np.ones_like(x)

        out = pre._asymmetric_outlier_handle(x, sigma)

        self.assertLessEqual(out[5], 2.0)
        self.assertEqual(out[10], 0.0)
        self.assertLess(out[20], -1.0)
        self.assertLess(out[21], -1.0)

    def test_local_normalization_removes_slow_trend(self) -> None:
        cfg = K2TimeDomainPreprocessConfig(
            local_window_days=0.4,
            local_min_window_cadences=21,
            thruster_step_sigma=999.0,
        )
        pre = K2TimeDomainPreprocessor(config=cfg)

        t = np.linspace(0.0, 2.0, 200)
        trend = 1.0 + 0.002 * np.arange(t.size)
        var = 0.01 * np.sin(2.0 * np.pi * t)
        flux = trend + var
        flux[90:93] -= 0.03

        out = pre.preprocess(t, flux)
        med = float(np.nanmedian(out["flux"]))
        self.assertLess(abs(med), 0.1)
        self.assertEqual(len(out["time"]), len(out["flux"]))


class TestK2TimeDomainRanker(unittest.TestCase):
    def test_min_duration_and_shape_ranking(self) -> None:
        cfg = K2TimeDomainRankConfig(
            detect_sigma=2.0,
            min_dip_cadences=2,
            max_dip_cadences=20,
            rank_window_cadences=32,
        )
        ranker = K2TimeDomainTransitRanker(config=cfg)

        t = np.arange(120, dtype=float)
        x = np.zeros(120, dtype=float)
        s = np.full(120, 1.0, dtype=float)

        # Single-cadence glitch (should be rejected by duration gate).
        x[15] = -3.0

        # Multi-cadence dip with coherent ingress/egress (should pass).
        x[50:53] = np.array([-2.4, -3.2, -2.1], dtype=float)

        candidates = ranker.rank_windows(
            query="EPIC 000000001",
            author="EVEREST",
            time=t,
            flux=x,
            sigma_local=s,
        )

        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertGreaterEqual(c.duration_cadences, 2)
        self.assertGreaterEqual(c.shape_score, 0.0)
        self.assertLessEqual(c.shape_score, 1.0)
        self.assertTrue(c.ingress_egress_ok)


class TestInferPeriodsFromEvents(unittest.TestCase):
    def test_infer_periods_returns_ranked_periods_and_hist(self) -> None:
        events_df = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)
        df = pd.DataFrame({"t_mid": events_df})

        ranked, hist_df = infer_periods_from_events(
            df,
            max_period=3.0,
            min_hits=3,
            tol_frac=0.01,
        )

        self.assertGreaterEqual(len(ranked), 1)
        p0, hits0, idx0 = ranked[0]
        self.assertAlmostEqual(p0, 2.0, places=6)
        self.assertEqual(hits0, 5)
        self.assertEqual(idx0, [0, 1, 2, 3, 4])

        self.assertIn("period", hist_df.columns)
        self.assertIn("pair_count", hist_df.columns)
        self.assertIn("count_hits", hist_df.columns)
        self.assertIn("supporting_event_indices", hist_df.columns)
        self.assertGreaterEqual(len(hist_df), 1)

    def test_infer_periods_min_hits_filters_ranked_output(self) -> None:
        df = pd.DataFrame({"t_mid": [0.0, 2.0, 4.0]})
        ranked, hist_df = infer_periods_from_events(
            df,
            max_period=2.2,
            min_hits=4,
            tol_frac=0.01,
        )

        self.assertEqual(ranked, [])
        self.assertTrue((hist_df["passes_min_hits"] == False).all())  # noqa: E712


class TestPhaseClusterScore(unittest.TestCase):
    def test_phase_cluster_wrap_around(self) -> None:
        # For P=1, phases ~0.98, 0.99, 0.01 should cluster together.
        df = pd.DataFrame({"t_mid": [0.98, 0.99, 1.01, 1.50]})
        buf = StringIO()
        with redirect_stdout(buf):
            cnt, center, idxs = phase_cluster_score(df, P=1.0, tol_phase=0.04)

        out = buf.getvalue()
        self.assertIn("phases sorted", out)
        self.assertEqual(cnt, 3)
        self.assertEqual(idxs, [0, 1, 2])
        self.assertTrue((center < 0.03) or (center > 0.97))


if __name__ == "__main__":
    unittest.main()
