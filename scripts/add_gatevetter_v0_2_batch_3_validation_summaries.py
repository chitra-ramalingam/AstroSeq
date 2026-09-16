from __future__ import annotations

import scripts.add_gatevetter_v0_2_batch_next_validation_summaries as base


base.PLOT_ROOT = base.ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_batch_3"
base.MANIFEST = base.ROOT / "gatevetter_v0_2_batch_3_visual_manifest.csv"
base.RUN_SUMMARY = base.ROOT / "gatevetter_v0_2_batch_3_summary.txt"


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
