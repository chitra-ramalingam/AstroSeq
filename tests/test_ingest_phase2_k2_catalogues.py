from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.ingest_phase2_k2_catalogues import (
    REQUIRED_RAW_COLUMNS,
    SCHEMA_VERSION,
    candidate_suffix,
    canonical_epic,
    load_and_verify_snapshot,
    map_disposition,
    normalize_rows,
    parse_campaigns,
    sha256,
    validate_raw_frame,
)


class K2PandCIngestionTests(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        return pd.DataFrame([
            ["EPIC 211915147.01", "", "EPIC 211915147", "1", "FALSE POSITIVE", "Yu et al. 2018", "5, 16", "1.810708", "2019-02-26", "2019-02-26"],
            ["EPIC 211915147.02", "", "EPIC 211915147", "1", "CANDIDATE", "Example 2020", "5", "2.5", "2020-01-01", "2020-01-01"],
            ["K2-TEST b", "K2-TEST b", "EPIC_201000001", "0", "MYSTERY", "", "1", "", "2020-01-02", "2020-01-02"],
        ], columns=REQUIRED_RAW_COLUMNS)

    def test_epic_normalization_is_exact(self) -> None:
        self.assertEqual(canonical_epic("EPIC 211915147"), ("EPIC_211915147", "valid"))
        self.assertEqual(canonical_epic("EPIC_20100001"), ("EPIC_20100001", "valid"))
        self.assertEqual(canonical_epic("211915147"), ("", "malformed"))
        self.assertEqual(canonical_epic(""), ("", "missing"))

    def test_candidate_suffix_remains_object_level(self) -> None:
        self.assertEqual(candidate_suffix("EPIC 211915147.01", "EPIC_211915147"), ".01")
        self.assertEqual(candidate_suffix("K2-108 b", "EPIC_211915147"), "")

    def test_null_campaign_marker_is_not_a_campaign(self) -> None:
        self.assertEqual(parse_campaigns("None"), [])
        self.assertEqual(parse_campaigns("5, 16, 18"), ["5", "16", "18"])

    def test_disposition_mapping_and_unknown_quarantine(self) -> None:
        self.assertEqual(map_disposition("CONFIRMED"), ("candidate_like", "archive_confirmed", True))
        self.assertEqual(map_disposition("CANDIDATE"), ("candidate_like", "archive_candidate", True))
        self.assertEqual(map_disposition("FALSE POSITIVE"), ("false_positive_eb_or_variable", "archive_false_positive", True))
        self.assertEqual(map_disposition("REFUTED"), ("false_positive_eb_or_variable", "archive_refuted", True))
        self.assertEqual(map_disposition("MYSTERY"), ("uncertain_hold", "unknown_or_missing_disposition", False))

    def test_rows_planets_solutions_and_conflicts_are_preserved(self) -> None:
        raw = self.fixture()
        manifest = {
            "source_version": "test-version",
            "retrieval_utc_timestamp": "2026-08-05T00:00:00Z",
            "source_citation": {"doi": "10.26133/NEA19"},
            "raw_file": {"sha256": "a" * 64},
        }
        first = normalize_rows(raw, manifest)
        second = normalize_rows(raw, manifest)
        self.assertEqual(len(first), len(raw))
        self.assertEqual(first["catalogue_row_id"].nunique(), len(raw))
        self.assertEqual(first.loc[0, "candidate_suffix"], ".01")
        self.assertEqual(first.loc[1, "candidate_suffix"], ".02")
        self.assertTrue(first.loc[0, "cross_class_conflict"])
        self.assertTrue(first.loc[1, "cross_class_conflict"])
        self.assertFalse(first.loc[0, "physical_loss_eligible_proposal"])
        self.assertIn("cross_class_conflict", first.loc[0, "eligibility_exclusion_reason"])
        self.assertEqual(first.loc[2, "normalized_label_proposal"], "uncertain_hold")
        self.assertIn("unknown_or_missing_disposition", first.loc[2, "eligibility_exclusion_reason"])
        pd.testing.assert_frame_equal(first, second)

    def test_exact_duplicate_source_rows_are_retained_and_flagged(self) -> None:
        raw = pd.concat([self.fixture().iloc[[0]], self.fixture().iloc[[0]]], ignore_index=True)
        manifest = {
            "source_version": "test-version",
            "retrieval_utc_timestamp": "2026-08-05T00:00:00Z",
            "source_citation": {"doi": "10.26133/NEA19"},
            "raw_file": {"sha256": "b" * 64},
        }
        rows = normalize_rows(raw, manifest)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows["catalogue_row_id"].nunique(), 2)
        self.assertTrue(rows["exact_duplicate_source_row"].all())
        self.assertTrue(rows["exact_duplicate_group_size"].eq(2).all())

    def test_hash_and_schema_are_verified_before_ingestion(self) -> None:
        manifest_path = Path("data/phase2/catalogues/nea_k2pandc/2026-08-05/source_manifest.json")
        manifest, loaded, raw_path = load_and_verify_snapshot(manifest_path)
        self.assertEqual(len(loaded), 4064)
        self.assertEqual(manifest["raw_file"]["sha256"], sha256(raw_path))
        with patch("scripts.ingest_phase2_k2_catalogues.sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_and_verify_snapshot(manifest_path)

    def test_missing_required_column_is_schema_drift(self) -> None:
        frame = self.fixture().drop(columns=["disp_refname"])
        raw_info = {"columns": list(frame.columns), "row_count": len(frame)}
        with self.assertRaisesRegex(ValueError, "missing columns.*disp_refname"):
            validate_raw_frame(frame, raw_info)

    def test_generated_snapshot_contract_when_present(self) -> None:
        root = Path("data/phase2/catalogues/nea_k2pandc/2026-08-05")
        if not (root / "source_manifest.json").exists():
            self.skipTest("authoritative snapshot not present")
        manifest, frame, _ = load_and_verify_snapshot(root / "source_manifest.json")
        self.assertEqual(manifest["raw_file"]["row_count"], 4064)
        self.assertEqual(list(frame.columns), REQUIRED_RAW_COLUMNS)


if __name__ == "__main__":
    unittest.main()
