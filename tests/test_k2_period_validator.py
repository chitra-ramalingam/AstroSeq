import unittest
from pathlib import Path
import shutil

import numpy as np

from src.Classifiers.K2.K2_TimeDomainTransitPipeline import (
    K2TimeDomainPreprocessConfig,
    K2TimeDomainPreprocessor,
    K2TimeDomainRankConfig,
    K2TimeDomainTransitRanker,
)
from src.Classifiers.K2.Systematics.K2_PeriodValidator import K2PeriodValidator


class _SyntheticDetector:
    def __init__(self) -> None:
        self.preprocessor = K2TimeDomainPreprocessor(
            config=K2TimeDomainPreprocessConfig(
                local_window_days=0.5,
                local_min_window_cadences=21,
                thruster_step_sigma=999.0,
                thruster_expand_cadences=0,
                positive_outlier_sigma=8.0,
                positive_clip_sigma=6.0,
                negative_outlier_sigma=8.0,
                min_negative_run_keep=2,
            )
        )
        self.ranker = K2TimeDomainTransitRanker(
            config=K2TimeDomainRankConfig(
                detect_sigma=2.2,
                min_dip_cadences=2,
                max_dip_cadences=40,
                rank_window_cadences=64,
                depth_snr_scale=3.0,
            )
        )


def _inject_box_transits(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    t0: float,
    depth: float,
    half_width_days: float,
) -> np.ndarray:
    out = np.asarray(flux, dtype=float).copy()
    t_min = float(np.min(time))
    t_max = float(np.max(time))
    k0 = int(np.ceil((t_min - t0) / P))
    k1 = int(np.floor((t_max - t0) / P))
    for k in range(k0, k1 + 1):
        tk = t0 + k * P
        in_dip = np.abs(time - tk) <= float(half_width_days)
        out[in_dip] -= float(depth)
    return out


class TestK2PeriodValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = _SyntheticDetector()
        self.validator = K2PeriodValidator(
            detector=self.detector,
            tol_days=0.12,
            min_duration_cadences=3,
            shape_threshold=0.6,
            snr_threshold=4.0,
        )

    def test_periodic_transits_yield_high_hit_rates(self) -> None:
        rng = np.random.default_rng(123)
        time = np.arange(0.0, 24.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
        flux = _inject_box_transits(time, flux, P=2.0, t0=0.35, depth=0.011, half_width_days=0.04)

        out = self.validator.validate(time=time, flux=flux, P=2.0, t0=0.35)

        self.assertGreater(out["n_predicted"], 8)
        self.assertGreater(out["coverage_rate"], 0.95)
        self.assertGreater(out["hit_rate_shape"], 0.6)
        self.assertGreater(out["hit_rate_snr"], 0.6)
        self.assertGreater(out["mean_best_snr"], 4.0)
        self.assertGreater(len(out["hits_df"]), 0)

    def test_random_dips_yield_low_hit_rates(self) -> None:
        rng = np.random.default_rng(456)
        time = np.arange(0.0, 24.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)

        random_tk = rng.uniform(float(np.min(time)) + 0.3, float(np.max(time)) - 0.3, size=16)
        for tk in random_tk:
            in_dip = np.abs(time - float(tk)) <= 0.02
            flux[in_dip] -= 0.0025

        out = self.validator.validate(time=time, flux=flux, P=2.0, t0=0.35)

        self.assertGreater(out["coverage_rate"], 0.95)
        self.assertLess(out["hit_rate_snr"], 0.35)
        self.assertLess(out["hit_rate_shape"], 0.5)
        for col in [
            "min_resid_inner",
            "dip_snr_at_min",
            "duration_below_threshold",
            "n_points_in_window",
            "has_candidate",
        ]:
            self.assertIn(col, out["misses_df"].columns)

    def test_uncovered_windows_are_not_counted_as_misses(self) -> None:
        rng = np.random.default_rng(789)
        time = np.arange(0.0, 24.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
        flux = _inject_box_transits(time, flux, P=2.0, t0=0.35, depth=0.009, half_width_days=0.04)

        quality_mask = np.ones_like(time, dtype=bool)
        gap_tk = 8.35
        quality_mask[np.abs(time - gap_tk) <= 0.14] = False

        out = self.validator.validate(time=time, flux=flux, P=2.0, t0=0.35, quality_mask=quality_mask)

        self.assertGreater(len(out["uncovered_df"]), 0)
        self.assertEqual(out["n_predicted"], out["n_covered"] + len(out["uncovered_df"]))
        self.assertEqual(out["n_covered"], len(out["hits_df"]) + len(out["misses_df"]))

    def test_detected_event_near_predicted_time_has_candidate(self) -> None:
        rng = np.random.default_rng(2468)
        time = np.arange(0.0, 24.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
        P = 2.0
        t0 = 0.35
        flux = _inject_box_transits(time, flux, P=P, t0=t0, depth=0.009, half_width_days=0.04)

        pre = self.detector.preprocessor.preprocess(time, flux)
        candidates = self.detector.ranker.rank_windows(
            query="SYNTH",
            author="TEST",
            time=pre["time"],
            flux=pre["flux"],
            sigma_local=pre["local_sigma"],
        )
        self.assertGreater(len(candidates), 0)
        event_tmid = float(getattr(candidates[0], "t_mid"))

        nearest_tk = float(t0 + np.round((event_tmid - t0) / P) * P)
        self.assertLessEqual(abs(event_tmid - nearest_tk), self.validator.tol_days)

        out = self.validator.validate(time=time, flux=flux, P=P, t0=t0)
        scores = out["scores_df"].copy()
        self.assertGreater(len(scores), 0)

        idx = int(np.argmin(np.abs(scores["tk"].to_numpy(dtype=float) - nearest_tk)))
        row = scores.iloc[idx]
        self.assertAlmostEqual(float(row["tk"]), nearest_tk, places=8)
        self.assertGreater(float(row["best_shape_score"]), 0.0)

    def test_no_candidate_windows_still_report_soft_dip_stats(self) -> None:
        rng = np.random.default_rng(97531)
        time = np.arange(0.0, 24.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
        P = 2.0
        t0 = 0.35
        flux = _inject_box_transits(time, flux, P=P, t0=t0, depth=0.010, half_width_days=0.04)

        strict_validator = K2PeriodValidator(
            detector=self.detector,
            tol_days=0.12,
            min_duration_cadences=10,  # force strict no-candidate in many windows
            shape_threshold=0.6,
            snr_threshold=4.0,
        )
        out = strict_validator.validate(time=time, flux=flux, P=P, t0=t0)

        self.assertGreater(out["n_windows_with_no_candidates"], 0)
        self.assertIn("dip_snr_at_min", out["scores_df"].columns)
        self.assertIn("duration_below_threshold", out["scores_df"].columns)
        self.assertIn("n_points_in_window", out["scores_df"].columns)
        self.assertIn("has_candidate", out["scores_df"].columns)
        self.assertIn("frac_no_cand_dip_snr_gt3", out)
        self.assertIn("duration_below_threshold_dist_no_cand", out)
        self.assertIsInstance(out["duration_below_threshold_dist_no_cand"], dict)

    def test_plot_outputs_are_saved(self) -> None:
        rng = np.random.default_rng(333)
        time = np.arange(0.0, 12.0, 0.02, dtype=float)
        sigma = 1e-3
        flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
        flux = _inject_box_transits(time, flux, P=2.0, t0=0.35, depth=0.009, half_width_days=0.04)
        out = self.validator.validate(time=time, flux=flux, P=2.0, t0=0.35)

        out_dir = Path("tmp_pycache") / "period_validator_plot_test"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hitmap = out_dir / "hitmap.png"
        phase = out_dir / "phase.png"
        self.validator.plot_validation_hitmap(out, hitmap)
        self.validator.plot_scores_vs_phase(out, phase, score_col="best_shape_score")
        self.assertTrue(hitmap.exists())
        self.assertTrue(phase.exists())
        self.assertGreater(hitmap.stat().st_size, 0)
        self.assertGreater(phase.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
