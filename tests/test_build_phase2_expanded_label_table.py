from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.build_phase2_expanded_label_table import (
    aggregate_catalogue,
    attach_existing_feature_coverage,
    build_expanded_hosts,
    feature_coverage,
    tensor_epics,
)


def catalogue_rows() -> pd.DataFrame:
    common = {
        "source_slug": "test", "source_table": "k2pandc", "source_version": "v1",
        "source_retrieved_at_utc": "2026-08-05T00:00:00Z", "source_sha256": "abc",
        "source_record_locator": "row", "source_row_number": 1, "raw_epic_hostname": "",
        "epic_identifier_status": "valid", "candidate_suffix": ".01", "k2_name": "",
        "raw_disposition_reference": "Ref 2026", "disposition_known": True,
        "has_disposition_reference": True, "raw_campaigns": "5", "campaigns_json": '["5"]',
        "raw_period_days": "2", "period_days": 2.0, "period_source": "test",
        "period_provenance": "test", "period_trusted": False, "row_update_date": "2026-01-01",
        "release_date": "2026-01-01", "eligibility_exclusion_reason": "",
        "exact_duplicate_source_row": False, "exact_duplicate_group_size": 1,
        "source_citation": "test", "ingestion_schema_version": "test",
    }
    rows = [
        {**common, "catalogue_row_id": "a", "epic_id": "EPIC_200000001", "object_id": "EPIC 200000001.01", "default_solution": True, "raw_disposition": "CONFIRMED", "normalized_label_proposal": "candidate_like", "evidence_kind": "archive_confirmed", "physical_loss_eligible_proposal": True, "cross_class_conflict": False},
        {**common, "catalogue_row_id": "b", "epic_id": "EPIC_200000001", "object_id": "EPIC 200000001.02", "candidate_suffix": ".02", "default_solution": False, "raw_disposition": "CANDIDATE", "normalized_label_proposal": "candidate_like", "evidence_kind": "archive_candidate", "physical_loss_eligible_proposal": False, "cross_class_conflict": False},
        {**common, "catalogue_row_id": "c", "epic_id": "EPIC_200000002", "object_id": "EPIC 200000002.01", "default_solution": True, "raw_disposition": "CANDIDATE", "normalized_label_proposal": "candidate_like", "evidence_kind": "archive_candidate", "physical_loss_eligible_proposal": False, "cross_class_conflict": True},
        {**common, "catalogue_row_id": "d", "epic_id": "EPIC_200000002", "object_id": "EPIC 200000002.02", "candidate_suffix": ".02", "default_solution": True, "raw_disposition": "FALSE POSITIVE", "normalized_label_proposal": "false_positive_eb_or_variable", "evidence_kind": "archive_false_positive", "physical_loss_eligible_proposal": False, "cross_class_conflict": True},
        {**common, "catalogue_row_id": "e", "epic_id": "", "epic_identifier_status": "missing", "object_id": "Unknown.01", "default_solution": True, "raw_disposition": "CANDIDATE", "normalized_label_proposal": "candidate_like", "evidence_kind": "archive_candidate", "physical_loss_eligible_proposal": False, "cross_class_conflict": False},
    ]
    return pd.DataFrame(rows)


def internal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.DataFrame([
        {"epic_id": "EPIC_200000001", "current_final_label": "candidate_like", "physical_loss_eligible": "true", "label_source": "manual", "training_role": "candidate", "positive_evidence_tier": "positive_silver", "k2_campaign": "5", "all_label_evidence": "manual=candidate_like"},
        {"epic_id": "EPIC_200000003", "current_final_label": "candidate_like", "physical_loss_eligible": "false", "label_source": "manual", "training_role": "candidate_trace", "positive_evidence_tier": "positive_bronze", "k2_campaign": "", "all_label_evidence": "manual=candidate_like"},
        {"epic_id": "EPIC_200000004", "current_final_label": "false_positive_eb_or_variable", "physical_loss_eligible": "true", "label_source": "manual", "training_role": "negative", "positive_evidence_tier": "", "k2_campaign": "5", "all_label_evidence": "manual=negative"},
    ])
    tiers = pd.DataFrame([
        {"epic_id": "EPIC_200000001", "period_feature_trust": "trusted_saved_period"},
        {"epic_id": "EPIC_200000003", "period_feature_trust": "fallback_untrusted_event_spacing"},
    ])
    corrections = pd.DataFrame(columns=["epic_id", "correction_basis"])
    return inventory, tiers, corrections


class ExpandedLabelTableTests(unittest.TestCase):
    def test_retains_every_object_row_and_preserves_multi_object_solutions(self) -> None:
        source = catalogue_rows()
        evidence, hosts = aggregate_catalogue(source)
        self.assertEqual(len(evidence), len(source))
        self.assertTrue(evidence["object_row_retained_before_host_aggregation"].all())
        row = hosts.set_index("epic_id").loc["EPIC_200000001"]
        self.assertEqual(row["catalogue_object_row_count"], 2)
        self.assertEqual(row["catalogue_unique_object_count"], 2)
        self.assertTrue(row["catalogue_multi_planet_or_object_system"])
        self.assertEqual(row["catalogue_default_solution_count"], 1)
        self.assertEqual(row["catalogue_non_default_solution_count"], 1)

    def test_one_host_per_epic_conflicts_quarantined_and_unmatched_candidate_retained(self) -> None:
        _, catalogue_hosts = aggregate_catalogue(catalogue_rows())
        inventory, tiers, corrections = internal_inputs()
        expanded = build_expanded_hosts(inventory, tiers, corrections, catalogue_hosts).set_index("epic_id")
        self.assertEqual(len(expanded), 4)
        self.assertFalse(expanded.index.duplicated().any())
        self.assertEqual(expanded.loc["EPIC_200000002", "corrected_physical_class"], "cross_class_conflict")
        self.assertFalse(expanded.loc["EPIC_200000002", "physical_loss_eligible"])
        self.assertEqual(expanded.loc["EPIC_200000003", "corrected_physical_class"], "candidate_like")
        self.assertFalse(expanded.loc["EPIC_200000003", "physical_loss_eligible"])
        self.assertEqual(expanded.loc["EPIC_200000003", "conflict_resolution"], "internal_status_retained_no_catalogue_match")
        self.assertTrue((expanded["split_assignment"] == "").all())

    def test_internal_archive_physical_disagreement_is_quarantined(self) -> None:
        _, catalogue_hosts = aggregate_catalogue(catalogue_rows())
        inventory, tiers, corrections = internal_inputs()
        extra = catalogue_hosts.loc[catalogue_hosts["epic_id"].eq("EPIC_200000001")].copy()
        extra["epic_id"] = "EPIC_200000004"
        expanded = build_expanded_hosts(inventory, tiers, corrections, pd.concat([catalogue_hosts, extra], ignore_index=True)).set_index("epic_id")
        self.assertTrue(expanded.loc["EPIC_200000004", "internal_archive_class_conflict"])
        self.assertFalse(expanded.loc["EPIC_200000004", "physical_loss_eligible"])

    def test_existing_feature_coverage_only_and_campaign_aggregation(self) -> None:
        _, catalogue_hosts = aggregate_catalogue(catalogue_rows())
        inventory, tiers, corrections = internal_inputs()
        expanded = build_expanded_hosts(inventory, tiers, corrections, catalogue_hosts)
        feature_row = {"epic_id": "EPIC_200000001", "cnn_probability": 0.9, "primary_depth": 0.1, "primary_depth_snr": 5.0, "validation_period_days": 2.0, "p_half_primary_depth": 0.1, "p_primary_depth": 0.1, "2p_primary_depth": 0.1, "nominal_period_trusted": True}
        feature_row.update({f"cnn_embedding_{i:03d}": float(i) for i in range(128)})
        covered = attach_existing_feature_coverage(expanded, pd.DataFrame([feature_row]), {"EPIC_200000001"}, {"EPIC_200000001"})
        coverage = feature_coverage(covered)
        row = coverage[(coverage["corrected_physical_class"] == "confirmed_planet") & (coverage["campaign"] == "5")].iloc[0]
        self.assertEqual(row["existing_128d_embeddings"], 1)
        self.assertEqual(row["p_half_p_2p_diagnostics"], 1)
        self.assertEqual(row["trusted_periods"], 1)

    def test_tensor_audit_requires_512_samples_and_build_is_deterministic(self) -> None:
        fake_tensor = np.zeros((1, 512, 1), dtype=np.float32)
        with patch.object(Path, "exists", return_value=True), patch("numpy.load", return_value=fake_tensor), patch("pandas.read_parquet", return_value=pd.DataFrame({"star_id": ["EPIC_200000001"]})):
            self.assertEqual(tensor_epics(Path("meta.parquet"), Path("X.npy")), {"EPIC_200000001"})
        evidence1, hosts1 = aggregate_catalogue(catalogue_rows())
        evidence2, hosts2 = aggregate_catalogue(catalogue_rows())
        assert_frame_equal(evidence1, evidence2)
        assert_frame_equal(hosts1, hosts2)


if __name__ == "__main__":
    unittest.main()
