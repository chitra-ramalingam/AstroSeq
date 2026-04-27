import dataclasses
import math
import unittest
from types import MethodType
from unittest import mock
from pathlib import Path
import os
from uuid import uuid4
import shutil

import numpy as np

from src.Classifiers.K2.Systematics.K2NoiseLoader import K2NoiseLoader, K2NoiseLoaderConfig
from src.Classifiers.K2.Systematics.K2_NoiseHandler import K2NoiseConfig, K2NoiseMetrics, K2_NoiseHandler


def _make_handler(cfg: K2NoiseConfig | None = None) -> K2_NoiseHandler:
    h = K2_NoiseHandler.__new__(K2_NoiseHandler)
    h.noise_config = cfg if cfg is not None else K2NoiseConfig()
    h.max_gap_days = 0.5
    h.author_priority = K2_NoiseHandler.AUTHOR_PRIORITY
    h.verbose = False
    return h


class TestK2NoiseHandlerExplain(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = K2NoiseConfig(
            whiteness_score_definition="statistic",
            min_points=100,
            min_baseline_days=10.0,
            min_robust_sigma=0.1,
            max_outlier_rate_6sigma=0.02,
            catastrophic_outlier_rate_6sigma=0.10,
            max_step_score=1.5,
            max_whiteness_score=0.6,
        )
        self.handler = _make_handler(self.cfg)
        self.base_metric = K2NoiseMetrics(
            n_points=200,
            baseline_days=20.0,
            duty_cycle=0.9,
            mad=0.01,
            robust_sigma=2.0,
            outlier_rate_6sigma=0.01,
            step_score=0.5,
            whiteness_score=0.3,
        )

    def test_explain_threshold_fail_reasons(self) -> None:
        checks = [
            ({"n_points": 50}, "n_points<100"),
            ({"baseline_days": 5.0}, "baseline_days<10.0"),
            ({"robust_sigma": 0.05}, "robust_sigma<0.1"),
            ({"outlier_rate_6sigma": 0.5}, "outlier_rate_6sigma>0.02"),
            ({"step_score": 2.0}, "step_score>1.5"),
            ({"whiteness_score": 0.9}, "whiteness_score=0.9>0.6"),
        ]

        for updates, expected_reason in checks:
            m = dataclasses.replace(self.base_metric, **updates)
            ex = self.handler.explain(m)
            self.assertIn(expected_reason, ex["fail_reasons"])
            self.assertIn("thresholds", ex)
            self.assertIn("values", ex)
            self.assertIn("robust_sigma", ex["values"])
            self.assertIn("max_outlier_rate_6sigma", ex["thresholds"])
            self.assertEqual(ex["thresholds"]["mode"], "strict")


class TestK2NoiseConfigPresets(unittest.TestCase):
    def test_strict_and_discovery_presets(self) -> None:
        strict = K2NoiseConfig(mode="strict")
        discovery = K2NoiseConfig(mode="discovery")
        self.assertAlmostEqual(float(strict.max_outlier_rate_6sigma), 0.02, places=9)
        self.assertAlmostEqual(float(discovery.max_outlier_rate_6sigma), 0.08, places=9)
        self.assertAlmostEqual(float(strict.whiteness_alpha), 0.01, places=9)
        self.assertAlmostEqual(float(discovery.whiteness_alpha), 0.05, places=9)

    def test_loader_mode_passed_to_handler(self) -> None:
        loader = K2NoiseLoader(loader_config=K2NoiseLoaderConfig(mode="discovery"))
        self.assertEqual(loader.handler.noise_config.mode, "discovery")
        self.assertIsNotNone(loader.handler.noise_config.max_outlier_rate_6sigma)
        self.assertIsNotNone(loader.handler.noise_config.whiteness_alpha)


class TestK2NoiseSegmentScoring(unittest.TestCase):
    def test_score_segments_policies(self) -> None:
        h = _make_handler(K2NoiseConfig(min_points=100, min_baseline_days=10.0, min_robust_sigma=0.1))
        metrics = [
            K2NoiseMetrics(150, 20.0, 1.0, 0.01, 2.0, 0.0, 0.5, 0.1),
            K2NoiseMetrics(120, 20.0, 1.0, 0.01, 2.0, 0.0, 0.5, 0.1),
            K2NoiseMetrics(80, 20.0, 1.0, 0.01, 2.0, 0.0, 0.5, 0.1),
        ]

        best = h.score_segments(metrics, "best")
        median = h.score_segments(metrics, "median")
        worst = h.score_segments(metrics, "worst")

        self.assertAlmostEqual(best, 0.5, places=6)
        self.assertAlmostEqual(median, 0.2, places=6)
        self.assertAlmostEqual(worst, -0.2, places=6)


class TestLocalVsGlobalOutlierRate(unittest.TestCase):
    def test_noisy_segment_has_higher_global_than_local_outlier_rate(self) -> None:
        h = _make_handler()
        t = np.linspace(0.0, 30.0, 3000)
        rng = np.random.default_rng(123)
        f = rng.normal(0.0, 1e-3, size=t.size)
        noisy = (t > 12.0) & (t < 13.0)
        f[noisy] += rng.normal(0.0, 3e-2, size=int(noisy.sum()))

        m = h._metrics_single(t, f)

        self.assertGreater(m.outlier_rate_global, 0.02)
        self.assertLess(m.outlier_rate_6sigma, m.outlier_rate_global)
        self.assertLess(m.outlier_rate_6sigma, 0.01)


class TestK2WhitenessRepresentation(unittest.TestCase):
    def test_metrics_single_preserves_finite_pvalue_and_explicit_fields(self) -> None:
        handler = _make_handler(K2NoiseConfig(whiteness_score_definition="pvalue"))
        rng = np.random.default_rng(123)
        t = np.linspace(0.0, 30.0, 2000)
        f = rng.normal(0.0, 1e-3, size=t.size)

        m = handler._metrics_single(t, f)

        self.assertEqual(str(m.whiteness_mode), "pvalue")
        self.assertTrue(np.isfinite(m.whiteness_score))
        self.assertTrue(np.isfinite(m.whiteness_pvalue))
        self.assertGreater(float(m.whiteness_pvalue), 0.0)
        self.assertAlmostEqual(float(m.whiteness_score), float(m.whiteness_pvalue), places=12)
        self.assertTrue(np.isfinite(m.whiteness_log10_pvalue))
        self.assertTrue(np.isfinite(m.whiteness_statistic_abs_rho))
        self.assertTrue(np.isfinite(m.whiteness_z))
        self.assertFalse(bool(m.whiteness_underflowed))

    def test_metrics_single_marks_underflowed_pvalue_but_keeps_log_representation(self) -> None:
        handler = _make_handler(K2NoiseConfig(whiteness_score_definition="pvalue"))
        t = np.linspace(0.0, 30.0, 5000)
        f = np.linspace(0.0, 1.0, t.size)

        m = handler._metrics_single(t, f)

        self.assertEqual(str(m.whiteness_mode), "pvalue")
        self.assertEqual(float(m.whiteness_score), 0.0)
        self.assertEqual(float(m.whiteness_pvalue), 0.0)
        self.assertTrue(np.isfinite(m.whiteness_log10_pvalue))
        self.assertTrue(np.isfinite(m.whiteness_z))
        self.assertTrue(np.isfinite(m.whiteness_statistic_abs_rho))
        self.assertTrue(bool(m.whiteness_underflowed))

    def test_metrics_single_statistic_mode_keeps_explicit_statistic_and_nan_pvalues(self) -> None:
        handler = _make_handler(K2NoiseConfig(whiteness_score_definition="statistic", max_whiteness_score=0.6))
        rng = np.random.default_rng(456)
        t = np.linspace(0.0, 30.0, 2000)
        f = rng.normal(0.0, 1e-3, size=t.size)

        m = handler._metrics_single(t, f)

        self.assertEqual(str(m.whiteness_mode), "statistic")
        self.assertTrue(np.isfinite(m.whiteness_score))
        self.assertTrue(np.isfinite(m.whiteness_statistic_abs_rho))
        self.assertAlmostEqual(float(m.whiteness_score), float(m.whiteness_statistic_abs_rho), places=12)
        self.assertTrue(np.isnan(m.whiteness_pvalue))
        self.assertTrue(np.isnan(m.whiteness_log10_pvalue))
        self.assertTrue(np.isfinite(m.whiteness_z))
        self.assertFalse(bool(m.whiteness_underflowed))

    def test_metrics_single_noncomputable_whiteness_stays_nan_and_not_underflow(self) -> None:
        handler = _make_handler(K2NoiseConfig(whiteness_score_definition="pvalue"))
        t = np.linspace(0.0, 30.0, 2000)
        f = np.ones_like(t)

        m = handler._metrics_single(t, f)

        self.assertEqual(str(m.whiteness_mode), "pvalue")
        self.assertTrue(np.isnan(m.whiteness_score))
        self.assertTrue(np.isnan(m.whiteness_pvalue))
        self.assertTrue(np.isnan(m.whiteness_log10_pvalue))
        self.assertTrue(np.isnan(m.whiteness_statistic_abs_rho))
        self.assertTrue(np.isnan(m.whiteness_z))
        self.assertFalse(bool(m.whiteness_underflowed))


class TestK2NoiseLoaderRunOne(unittest.TestCase):
    def test_run_one_per_segment_includes_scores_and_reason(self) -> None:
        cfg = K2NoiseConfig(
            whiteness_score_definition="statistic",
            min_points=100,
            min_baseline_days=10.0,
            min_robust_sigma=0.1,
            max_outlier_rate_6sigma=0.02,
            catastrophic_outlier_rate_6sigma=0.10,
            max_step_score=1.5,
            max_whiteness_score=0.6,
        )
        handler = _make_handler(cfg)

        global_m = K2NoiseMetrics(
            n_points=250,
            baseline_days=30.0,
            duty_cycle=0.9,
            mad=0.01,
            robust_sigma=2.0,
            outlier_rate_6sigma=0.01,
            step_score=0.5,
            whiteness_score=0.2,
        )
        seg_good = K2NoiseMetrics(150, 20.0, 0.9, 0.01, 2.0, 0.0, 0.5, 0.1)
        seg_bad = K2NoiseMetrics(80, 20.0, 0.9, 0.01, 2.0, 0.0, 0.5, 0.1)

        def _fetch_best(self, query, limit=50, exptime=None, cache_only=None):  # noqa: ANN001
            return {"lc": object(), "author": "EVEREST", "search_result": {"query": query}}

        def _clean(self, lc, flatten=False):  # noqa: ANN001
            return {"time": [1.0, 2.0, 3.0], "flux": [0.0, 0.0, 0.0], "notes": ""}

        def _metrics(self, time, flux, notes="", per_segment=False):  # noqa: ANN001
            if per_segment:
                return {"global": global_m, "segments": [seg_good, seg_bad]}
            return global_m

        handler.fetch_best = MethodType(_fetch_best, handler)
        handler.clean = MethodType(_clean, handler)
        handler.metrics = MethodType(_metrics, handler)

        loader = K2NoiseLoader.__new__(K2NoiseLoader)
        loader.loader_config = K2NoiseLoaderConfig()
        loader.handler = handler

        row = loader.run_one("EPIC 211797674", per_segment=True)

        self.assertIn("score_global", row)
        self.assertIn("score_best_seg", row)
        self.assertIn("score_median_seg", row)
        self.assertIn("score_worst_seg", row)
        self.assertIn("why_not_usable", row)
        self.assertTrue(math.isfinite(row["score_global"]))
        self.assertTrue(math.isfinite(row["score_best_seg"]))

    def test_run_one_why_not_usable_includes_whiteness_value(self) -> None:
        cfg = K2NoiseConfig(
            mode="strict",
            whiteness_score_definition="statistic",
            min_points=100,
            min_baseline_days=10.0,
            min_robust_sigma=0.1,
            max_outlier_rate_6sigma=0.02,
            catastrophic_outlier_rate_6sigma=0.10,
            max_step_score=1.5,
            max_whiteness_score=0.6,
        )
        handler = _make_handler(cfg)

        global_m = K2NoiseMetrics(
            n_points=250,
            baseline_days=30.0,
            duty_cycle=0.9,
            mad=0.01,
            robust_sigma=2.0,
            outlier_rate_6sigma=0.01,
            step_score=0.5,
            whiteness_score=0.9,
            outlier_rate_global=0.01,
        )
        seg_good = K2NoiseMetrics(150, 20.0, 0.9, 0.01, 2.0, 0.0, 0.5, 0.1, outlier_rate_global=0.0)

        def _fetch_best(self, query, limit=50, exptime=None, cache_only=None):  # noqa: ANN001
            return {"lc": object(), "author": "EVEREST", "search_result": {"query": query}}

        def _clean(self, lc, flatten=False):  # noqa: ANN001
            return {"time": [1.0, 2.0, 3.0], "flux": [0.0, 0.0, 0.0], "notes": ""}

        def _metrics(self, time, flux, notes="", per_segment=False):  # noqa: ANN001
            if per_segment:
                return {"global": global_m, "segments": [seg_good]}
            return global_m

        handler.fetch_best = MethodType(_fetch_best, handler)
        handler.clean = MethodType(_clean, handler)
        handler.metrics = MethodType(_metrics, handler)

        loader = K2NoiseLoader.__new__(K2NoiseLoader)
        loader.loader_config = K2NoiseLoaderConfig(mode="strict")
        loader.handler = handler

        row = loader.run_one("EPIC 211797674", per_segment=True)
        self.assertIn("whiteness_score=0.9>0.6", row["why_not_usable"])


class TestK2NoiseHandlerDirectLocalFetch(unittest.TestCase):
    def test_fetch_best_uses_direct_local_cache_before_search(self) -> None:
        handler = K2_NoiseHandler()
        case_dir = Path("tmp_pycache") / f"k2_noise_handler_local_fetch_{uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
        try:
            cache_root = case_dir
            local_dir = cache_root / "mastDownload" / "HLSP" / "hlsp_everest_k2_llc_211301344-c05_kepler_v2.0_lc"
            local_dir.mkdir(parents=True, exist_ok=False)
            local_path = local_dir / "hlsp_everest_k2_llc_211301344-c05_kepler_v2.0_lc.fits"
            local_path.write_text("stub", encoding="ascii")

            with mock.patch.dict(os.environ, {"LIGHTKURVE_CACHE_DIR": str(cache_root)}), \
                mock.patch("src.Classifiers.K2.Systematics.K2_NoiseHandler.lk.read", return_value=object()) as read_mock, \
                mock.patch(
                    "src.Classifiers.K2.Systematics.K2_NoiseHandler.lk.search_lightcurve",
                    side_effect=AssertionError("search_lightcurve should not be called"),
                ):
                out = handler.fetch_best(query="EPIC 211301344", cache_only=True)

            self.assertEqual(str(out["status"]), "ok")
            self.assertEqual(str(out["author_selected"]), "EVEREST")
            self.assertEqual(str(out["campaign_selected"]), "c05")
            self.assertEqual(str(out["search_result"]["selection_reason"]), "direct_local_cache_lookup")
            self.assertEqual(str(out["cache_path"]), str(local_path))
            read_mock.assert_called_once()
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_fetch_best_cache_only_without_local_file_returns_cache_miss_without_search(self) -> None:
        handler = K2_NoiseHandler()
        with mock.patch.dict(os.environ, {"LIGHTKURVE_CACHE_DIR": str(Path("tmp_pycache") / f"missing_cache_{uuid4().hex}")}), \
            mock.patch(
                "src.Classifiers.K2.Systematics.K2_NoiseHandler.lk.search_lightcurve",
                side_effect=AssertionError("search_lightcurve should not be called"),
            ):
            out = handler.fetch_best(query="EPIC 299999999", cache_only=True)

        self.assertEqual(str(out["status"]), "cache_miss")
        self.assertEqual(str(out["cache_source"]), "local_direct_cache_miss")
        self.assertEqual(str(out["search_result"]["selection_reason"]), "direct_local_cache_miss_no_search")


if __name__ == "__main__":
    unittest.main()
