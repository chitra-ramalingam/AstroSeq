from __future__ import annotations

import numpy as np
import pandas as pd
import unittest
from pathlib import Path

from scripts.build_phase2_feature_table import LEAKAGE_EXCLUSIONS, resolve_numeric_measurements


class FeatureTableTests(unittest.TestCase):
    def test_leakage_exclusions_cover_decisions_labels_and_reasons(self) -> None:
        joined = "|".join(LEAKAGE_EXCLUSIONS)
        for required in [
            "prediction", "recommendation", "final_candidate_status", "manual_reason",
            "training_label_rule", "positive_evidence_tier", "external_disposition",
            "training_role", "correction_basis", "catalogue_normalized_target",
            "archive_disposition", "catalogue_class", "catalogue_reference",
            "gatevetter_decision", "gatevetter_recommendation", "normalized_target_label",
        ]:
            self.assertIn(required, joined)

    def test_conflicting_measurements_are_quarantined(self) -> None:
        resolved, conflicts = resolve_numeric_measurements([0.1, 0.2])
        self.assertTrue(np.isnan(resolved))
        self.assertEqual(conflicts, [0.1, 0.2])

    def test_audit_only_switch_is_mandatory(self) -> None:
        from scripts.build_phase2_feature_table import build_table
        with self.assertRaisesRegex(ValueError, "audit-only"):
            build_table(pd.NA, pd.NA, audit_only=False)

    def test_generated_table_contract_when_artifact_exists(self) -> None:
        path = Path("data/phase2/phase2_feature_table.parquet")
        if not path.exists():
            self.skipTest("generated audit artifact not present")
        table = pd.read_parquet(path)
        self.assertEqual(len(table), 268)
        self.assertEqual(table["epic_id"].nunique(), 268)
        self.assertEqual(len([c for c in table if c.startswith("cnn_embedding_") and c[-3:].isdigit()]), 128)
        self.assertTrue(all(f"missing__cnn_embedding_{i:03d}" in table for i in range(128)))
        self.assertFalse(set(LEAKAGE_EXCLUSIONS) & set(table.select_dtypes(include="number").columns))

    def test_positive_hypothesis_artifact_has_three_rows_per_epic(self) -> None:
        path = Path("data/phase2/phase2_positive_period_diagnostics.csv")
        if not path.exists():
            self.skipTest("generated audit artifact not present")
        frame = pd.read_csv(path)
        self.assertEqual(frame["epic_id"].nunique(), 9)
        self.assertTrue((frame.groupby("epic_id").size() == 3).all())


if __name__ == "__main__":
    unittest.main()
