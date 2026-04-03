from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.Classifiers.K2.Batch.K2CachedFailedBroaderDownstreamRunner import K2CachedFailedBroaderDownstreamRunner


class K2CachedFailedBroaderDownstreamRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = Path("tmp_pycache") / f"k2_cached_failed_broader_downstream_runner_{uuid4().hex}"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

    def _write_shard(self, shard_dir: Path, epic_id: str, shape_score: float) -> None:
        epics_dir = shard_dir / "epics" / f"EPIC_{epic_id}"
        epics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "query": f"EPIC {epic_id}",
                    "epic_id": epic_id,
                    "triage_status": "ok",
                    "triage_usable": True,
                    "triage_whiteness_definition": "pvalue",
                    "triage_whiteness_score": 0.5,
                    "triage_why_not_usable": "",
                    "n_events": 3,
                    "best_shape_score": shape_score,
                    "best_depth_snr": 4.5,
                }
            ]
        ).to_csv(shard_dir / "batch_results.csv", index=False)
        pd.DataFrame(columns=["t_mid"]).to_csv(epics_dir / "events.csv", index=False)
        pd.DataFrame([{"last_completed_index": 0}]).to_json(shard_dir / "progress.json", orient="records")

    def test_runner_builds_broader_downstream_outputs_from_shards(self) -> None:
        shards_root = self.case_dir / "detector_cached_failed_broader_quality_gated_shards"
        out_dir = self.case_dir / "detector_cached_failed_broader_quality_gated_downstream"
        shard_1 = shards_root / "detector_cached_failed_broader_queries_shard_001"
        shard_2 = shards_root / "detector_cached_failed_broader_queries_shard_002"
        self._write_shard(shard_1, epic_id="200000001", shape_score=0.81)
        self._write_shard(shard_2, epic_id="200000002", shape_score=0.79)

        out = K2CachedFailedBroaderDownstreamRunner().run(
            shards_root=shards_root,
            out_dir=out_dir,
            disable_validation=True,
        )

        self.assertEqual(int(out["shard_count"]), 2)
        self.assertTrue(Path(out["merged_batch_csv"]).exists())
        self.assertTrue(Path(out["input_manifest_csv"]).exists())
        self.assertTrue(Path(out["shortlist_top_shape_for_period_csv"]).exists())
        self.assertTrue(Path(out["period_shortlist_quarantine_csv"]).exists())
        self.assertTrue(Path(out["period_shortlist_diagnostics_csv"]).exists())
        self.assertTrue(Path(out["epic_funnel_reasons_csv"]).exists())
        self.assertFalse(bool(out["validation_enabled"]))

        merged_df = pd.read_csv(out["merged_batch_csv"])
        self.assertEqual(int(len(merged_df)), 2)
        self.assertIn("epic_dir", merged_df.columns)
        self.assertTrue(merged_df["epic_dir"].astype(str).str.contains("EPIC_200000001|EPIC_200000002").any())

        quarantine_df = pd.read_csv(out["period_shortlist_quarantine_csv"])
        self.assertEqual(set(quarantine_df["epic_id"].astype(str)), {"200000001", "200000002"})
        self.assertTrue(quarantine_df["failure_category"].astype(str).eq("events_filtered_to_zero").all())

        diag_df = pd.read_csv(out["period_shortlist_diagnostics_csv"])
        self.assertEqual(str(diag_df.iloc[0]["raw_epic_list_csv"]), str(out["merged_batch_csv"]))


if __name__ == "__main__":
    unittest.main()
