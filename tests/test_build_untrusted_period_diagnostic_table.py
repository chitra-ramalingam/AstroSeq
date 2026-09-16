from __future__ import annotations

import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch

from scripts import build_untrusted_period_diagnostic_table as diagnostic


class UntrustedPeriodDiagnosticTests(unittest.TestCase):
    def test_tested_periods_adds_historical_harmonics(self):
        candidates = pd.DataFrame(
            [
                {
                    "period_days": 3.0,
                    "candidate_origin": "event_spacing_fallback",
                    "event_support_count": 4,
                    "cluster_center_phase": 0.1,
                },
                {
                    "period_days": 9.0,
                    "candidate_origin": "historical_period_search",
                    "event_support_count": 3,
                    "cluster_center_phase": 0.2,
                },
            ]
        )
        row = pd.Series(
            {
                "epic_id": "EPIC_1",
                "packet_source": "test_packet",
                "event_spacing_fallback_period_days": 3.0,
                "saved_historical_period_days": np.nan,
                "historical_period_search_periods": "9.0",
            }
        )
        with (
            patch.object(diagnostic.Path, "exists", return_value=True),
            patch.object(diagnostic.pd, "read_csv", return_value=candidates),
        ):
                periods = diagnostic.tested_periods(row)

        by_period = {item["period_days"]: item for item in periods}
        self.assertIn(1.5, by_period)
        self.assertIn(4.5, by_period)
        self.assertIn(6.0, by_period)
        self.assertIn(18.0, by_period)
        self.assertIn("half_historical_period", by_period[4.5]["roles"])
        self.assertIn("double_historical_period", by_period[18.0]["roles"])

    def test_failure_reasons_do_not_mask_finite_untrusted_metrics(self):
        metrics = {
            "primary_depth": 0.001,
            "primary_depth_snr": 8.0,
            "odd_depth_median": 0.0011,
            "even_depth_median": 0.0009,
            "odd_even_depth_ratio": 0.82,
            "secondary_depth_phase_05": 0.0001,
            "secondary_depth_snr": 1.2,
            "secondary_to_primary_depth_ratio": 0.1,
            "oot_to_depth": 0.4,
        }

        self.assertEqual(diagnostic.failure_reasons(metrics, coverage=0.75), [])

    def test_normalize_selected_periods_marks_one_closest_row(self):
        table = pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_1",
                    "tested_period_days": 9.0,
                    "is_selected_period": True,
                    "is_trusted_period": False,
                    "metric_trust_level": "untrusted_period_dependent",
                },
                {
                    "epic_id": "EPIC_1",
                    "tested_period_days": 9.00001,
                    "is_selected_period": True,
                    "is_trusted_period": False,
                    "metric_trust_level": "untrusted_period_dependent",
                },
            ]
        )
        comparisons = pd.DataFrame(
            [
                {
                    "epic_id": "EPIC_1",
                    "validation_period_days": 9.000009,
                    "period_ambiguity_flag": True,
                    "period_comparison_status": "no_automatic_period_selection",
                }
            ]
        )

        normalized = diagnostic.normalize_selected_periods(table, comparisons)

        self.assertEqual(int(normalized["is_selected_period"].sum()), 1)
        self.assertTrue(bool(normalized.iloc[1]["is_selected_period"]))
        self.assertFalse(bool(normalized["is_trusted_period"].any()))


if __name__ == "__main__":
    unittest.main()
