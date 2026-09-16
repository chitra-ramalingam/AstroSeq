from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.build_phase2_conflict_review_packet import ALLOWED_DECISIONS, build_packet


class ConflictReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = build_packet()
        cls.host = pd.read_csv("docs/phase2/phase2_conflict_review_packet.csv", dtype=str, keep_default_na=False)
        cls.objects = pd.read_csv("docs/phase2/phase2_conflict_review_object_evidence.csv", dtype=str, keep_default_na=False)
        cls.expanded = pd.read_parquet("data/phase2/phase2_expanded_label_table.parquet")

    def test_exactly_11_unresolved_loss_ineligible_hosts(self) -> None:
        self.assertEqual(len(self.host), 11)
        self.assertEqual(self.host["epic_id"].nunique(), 11)
        self.assertTrue(self.host["adjudication_status"].eq("unresolved").all())
        self.assertTrue(self.host["current_physical_loss_eligible"].str.lower().eq("false").all())
        self.assertTrue(self.host["reviewer_decision"].eq("").all())

    def test_all_26_catalogue_rows_and_solutions_are_retained(self) -> None:
        self.assertEqual(len(self.objects), 26)
        self.assertEqual(set(self.objects["epic_id"]), set(self.host["epic_id"]))
        self.assertTrue((self.objects.groupby("epic_id").size() >= 1).all())
        self.assertIn("default_solution", self.objects)
        self.assertIn("source_record_locator", self.objects)

    def test_adjudication_vocabulary_is_explicit_but_blank(self) -> None:
        expected = "|".join(ALLOWED_DECISIONS)
        self.assertTrue(self.host["permitted_adjudication_values"].eq(expected).all())
        template = pd.read_csv("docs/phase2/phase2_conflict_adjudication_template.csv", dtype=str, keep_default_na=False)
        self.assertEqual(len(template), 11)
        self.assertTrue(template["reviewer_decision"].eq("").all())
        self.assertTrue(template["reviewer_reason"].eq("").all())

    def test_packet_does_not_include_gatevetter_feature_fields(self) -> None:
        forbidden = {"decision", "action", "recommendation", "prediction", "verdict"}
        for column in self.host.columns:
            lowered = column.lower()
            self.assertFalse("gatevetter" in lowered)
            self.assertFalse(any(token == lowered for token in forbidden))

    def test_expanded_label_state_remains_unchanged(self) -> None:
        rows = self.expanded[self.expanded["epic_id"].isin(self.host["epic_id"])]
        self.assertEqual(len(rows), 11)
        self.assertTrue(rows["cross_class_conflict"].all())
        self.assertFalse(rows["physical_loss_eligible"].any())
        self.assertTrue(rows["corrected_physical_class"].eq("cross_class_conflict").all())
        self.assertEqual(self.summary["physical_loss_eligible_host_count"], 0)

    def test_required_outputs_exist(self) -> None:
        for path in [
            "docs/phase2/phase2_conflict_review_packet.csv",
            "docs/phase2/phase2_conflict_review_object_evidence.csv",
            "docs/phase2/phase2_conflict_adjudication_template.csv",
            "docs/phase2/phase2_conflict_review_packet_summary.json",
            "docs/phase2/PHASE2_CONFLICT_REVIEW_PACKET_AUDIT.md",
        ]:
            self.assertTrue(Path(path).exists(), path)


if __name__ == "__main__":
    unittest.main()
