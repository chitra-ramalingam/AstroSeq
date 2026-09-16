from __future__ import annotations

import pandas as pd
import unittest

import gatevetter_v0_2_rules as rules
import scripts.run_gatevetter_v0_2_unseen as unseen_runner


def base_row(**updates):
    row = {
        "epic_id": "EPIC_TEST",
        "cnn_score": 0.8,
        "primary_depth": 0.001,
        "primary_depth_snr": 10.0,
        "odd_even_depth_ratio": 0.95,
        "secondary_depth_snr": 0.0,
        "secondary_to_primary_depth_ratio": 0.0,
        "oot_to_depth": 0.1,
        "candidate_period_count": 10,
        "event_family_count": 5,
        "alias_risk": "low",
        "fallback_period_flag": "false",
        "duration_fraction_of_period": 0.05,
        "best_period_days": 5.0,
        "transit_duration_hours": 6.0,
        "stage_f_label": "",
        "stage_f_reason": "",
        "period_source": "saved_best_period",
        "period_ambiguity_flag": "false",
        "validation_period_source": "saved_best_period",
        "period_comparison_status": "trusted_saved_period",
        "trusted_period_validation": "true",
        "metric_trust_level": "trusted_period_dependent",
        "odd_even_depth_ratio_missing_reason": "",
    }
    row.update(updates)
    return pd.Series(row)


class GateVetterV02PolicyTests(unittest.TestCase):
    def test_missing_crosschecks_cannot_promote(self):
        result = rules.gatevet(
            base_row(
                cnn_score=0.70,
                primary_depth_snr=30.0,
                fallback_period_flag="true",
                primary_depth="",
                odd_even_depth_ratio="",
                oot_to_depth="",
                duration_fraction_of_period="",
                transit_duration_hours="",
            )
        )

        self.assertNotIn(
            result["gatevetter_prediction"],
            {"candidate_like_positive", "caveated_candidate_stage_g_review"},
        )
        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "missing_core_diagnostics_block_promotion",
        )

    def test_period_ambiguity_blocks_promotion(self):
        result = rules.gatevet(
            base_row(
                period_ambiguity_flag="true",
                trusted_period_validation="false",
                period_comparison_status="no_automatic_period_selection",
                validation_period_source="period_ambiguous",
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "period_ambiguity_blocks_stage_g_promotion",
        )

    def test_untrusted_period_blocks_promotion(self):
        result = rules.gatevet(
            base_row(
                trusted_period_validation="false",
                period_comparison_status="provisional_not_trusted",
                validation_period_source="saved_best_period",
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "trusted_period_validation_required",
        )

    def test_fallback_ge_500_with_missing_core_is_low_priority(self):
        result = rules.gatevet(
            base_row(
                fallback_period_flag="true",
                period_source="event_spacing_fallback",
                candidate_period_count=500,
                odd_even_depth_ratio="",
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "fallback_period_clutter_missing_core_diagnostics",
        )
        self.assertEqual(result["gatevetter_prediction"], "negative_low_priority")

    def test_duration_fraction_ge_point_20_is_hold(self):
        result = rules.gatevet(
            base_row(
                duration_fraction_of_period=0.20,
                transit_duration_hours=24.0,
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "duration_fraction_ge_0_20_hold",
        )
        self.assertEqual(
            result["gatevetter_prediction"],
            "excluded_uncertain_hold",
        )

    def test_duration_fraction_ge_point_35_is_hard_reject(self):
        result = rules.gatevet(
            base_row(
                duration_fraction_of_period=0.35,
                transit_duration_hours=42.0,
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "hard_duration_fraction_ge_0_35",
        )
        self.assertEqual(
            result["gatevetter_prediction"],
            "negative_reject_as_noise_or_artifact",
        )

    def test_validation_period_source_ambiguous_blocks_promotion(self):
        result = rules.gatevet(
            base_row(
                period_ambiguity_flag="false",
                validation_period_source="period_ambiguous",
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "period_ambiguity_blocks_stage_g_promotion",
        )

    def test_alias_period_ambiguous_blocks_promotion(self):
        result = rules.gatevet(
            base_row(alias_risk="period_ambiguous")
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "period_ambiguity_blocks_stage_g_promotion",
        )

    def test_event_spacing_fallback_only_blocks_promotion(self):
        result = rules.gatevet(
            base_row(
                period_source="event_spacing_fallback",
                validation_period_source="event_spacing_fallback_only",
                trusted_period_validation="false",
                period_comparison_status="provisional_not_trusted",
            )
        )

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "fallback_only_period_blocks_stage_g_promotion",
        )

    def test_primary_depth_must_be_positive_for_promotion(self):
        result = rules.gatevet(base_row(primary_depth=0.0))

        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "missing_core_diagnostics_block_promotion",
        )

    def test_justified_odd_even_unavailable_requires_trusted_period(self):
        trusted = base_row(
            odd_even_depth_ratio="",
            odd_even_depth_ratio_missing_reason="single_valid_event_family",
        )
        untrusted = base_row(
            odd_even_depth_ratio="",
            odd_even_depth_ratio_missing_reason="single_valid_event_family",
            trusted_period_validation="false",
            period_comparison_status="provisional_not_trusted",
        )

        self.assertNotIn(
            "odd_even_depth_ratio",
            rules.missing_core_diagnostics(trusted),
        )
        self.assertIn(
            "odd_even_depth_ratio",
            rules.missing_core_diagnostics(untrusted),
        )

    def test_complete_trusted_candidate_can_still_promote(self):
        result = rules.gatevet(base_row())

        self.assertEqual(
            result["gatevetter_prediction"],
            "candidate_like_positive",
        )
        self.assertEqual(
            result["gatevetter_v0_2_reason"],
            "passed_gates_candidate_survivor",
        )

    def test_all_manually_rejected_v0_1_stage_g_rows_are_blocked(self):
        source = unseen_runner.v0_1_runner.build_unseen_source_batch()
        features = unseen_runner.build_features(source)
        reviewed = features[
            features["epic_id"].isin(
                unseen_runner.MANUALLY_REVIEWED_V0_1_STAGE_G
            )
        ]
        predictions = pd.DataFrame(
            [rules.gatevet(row) for _, row in reviewed.iterrows()]
        )

        self.assertEqual(len(predictions), 11)
        self.assertFalse(
            predictions["gatevetter_prediction"]
            .isin(
                {
                    "candidate_like_positive",
                    "caveated_candidate_stage_g_review",
                }
            )
            .any()
        )
        self.assertEqual(
            set(predictions["epic_id"]),
            unseen_runner.MANUALLY_REVIEWED_V0_1_STAGE_G,
        )


if __name__ == "__main__":
    unittest.main()
