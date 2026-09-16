from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_14"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211890606", "variable_or_possible_eb", "Broad EB/variable-like folded morphology with raw modulation; not a clean candidate-like hold."),
    ("EPIC_211751400", "variable_or_possible_eb", "Period-ambiguous broad phase structure and odd/even unavailable context are variable-like rather than transit-like."),
    ("EPIC_211974687", "false_positive_eb_or_variable", "Strong coherent EB/variable morphology with broad primary/secondary phase structure; reject sanity remains valid."),
    ("EPIC_211927964", "reject_as_noise_or_artifact", "Systematics-dominated raw trend with nearly flat folded views and no reliable transit recovery."),
    ("EPIC_212066407", "false_positive_eb_or_variable", "Low-SNR dip in false-positive context with OOT/secondary concerns; no candidate-like rescue."),
    ("EPIC_211713182", "reject_as_noise_or_artifact", "Large raw variability and an isolated sparse dip without reliable repeatable event support."),
    ("EPIC_211958946", "reject_as_noise_or_artifact", "Trend/systematics-dominated packet with flat noisy folded evidence and no missed transit family."),
    ("EPIC_211527790", "false_positive_eb_or_variable", "Large raw stellar variability and a period-ambiguous narrow feature are EB/variable false-positive context."),
    ("EPIC_211949473", "reject_as_noise_or_artifact", "Noisy period-ambiguous folded feature with weak event-stack support; reject remains valid."),
    ("EPIC_211376339", "reject_as_noise_or_artifact", "Weak flat/noisy period-ambiguous packet with no coherent candidate-like signal."),
    ("EPIC_211506325", "reject_as_noise_or_artifact", "Noisy low-SNR folded structure and period ambiguity provide no candidate-like recovery."),
    ("EPIC_211373495", "reject_as_noise_or_artifact", "Sparse noisy dip in a trend-dominated packet with period ambiguity; reject sanity remains valid."),
]


if __name__ == "__main__":
    base.main()
