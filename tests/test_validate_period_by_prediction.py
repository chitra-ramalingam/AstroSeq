import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from src.Classifiers.K2.Systematics.K2Validation_Prediction import K2Validation_Prediction


class TestValidatePeriodByPrediction(unittest.TestCase):
    def test_synthetic_periodic_transits_snr_scale_and_coverage(self) -> None:
        rng = np.random.default_rng(123)

        t = np.arange(0.0, 20.0, 0.02, dtype=float)
        sigma = 1e-3
        local_sigma = np.full_like(t, sigma, dtype=float)
        resid = rng.normal(0.0, sigma, size=t.shape)

        P = 2.0
        t0_true = 0.5
        k0 = int(np.ceil((float(np.min(t)) - t0_true) / P))
        k1 = int(np.floor((float(np.max(t)) - t0_true) / P))
        tk_all = t0_true + np.arange(k0, k1 + 1, dtype=float) * P

        depth = 0.006  # expected dip SNR ~ depth/sigma ~= 6
        width = 0.010
        for tk in tk_all:
            resid -= depth * np.exp(-0.5 * ((t - tk) / width) ** 2)

        # Create one missing-coverage zone around a predicted transit.
        gap_tk = float(tk_all[len(tk_all) // 2])
        keep = np.abs(t - gap_tk) > 0.09
        t = t[keep]
        resid = resid[keep]
        local_sigma = local_sigma[keep]

        # Event mids spaced by P; shape_score chooses one anchor for t0 inference.
        events_df = pd.DataFrame(
            {
                "t_mid": tk_all[:6],
                "shape_score": [0.1, 0.2, 0.95, 0.4, 0.3, 0.25],
            }
        )
        events_df.attrs["in_cluster_indices"] = list(events_df.index)

        with redirect_stdout(StringIO()):
            out = K2Validation_Prediction().validate_period_by_prediction(
                time=t,
                resid=resid,
                local_sigma=local_sigma,
                events_df=events_df,
                P=P,
                t0=None,
                tol_days=0.08,
                snr_threshold=3.0,
                do_plot=False,
            )

        self.assertGreater(out["n_predicted"], 5)
        self.assertGreater(out["n_covered"], 0)
        self.assertLess(out["n_covered"], out["n_predicted"])
        self.assertGreater(out["coverage_rate"], 0.5)
        self.assertLess(out["coverage_rate"], 1.0)

        self.assertGreater(out["hit_rate_3"], 0.5)
        self.assertGreater(out["mean_hit_snr"], 1.0)
        self.assertLess(out["mean_hit_snr"], 20.0)
        self.assertGreater(len(out["uncovered_windows"]), 0)

        covered_snrs = [
            snr
            for (tk, snr, _), covered in zip(out["rows"], out["covered_mask"])
            if covered and np.isfinite(snr)
        ]
        self.assertGreater(len(covered_snrs), 0)
        expected = depth / sigma
        self.assertAlmostEqual(float(np.nanmedian(covered_snrs)), expected, delta=3.0)


if __name__ == "__main__":
    unittest.main()
