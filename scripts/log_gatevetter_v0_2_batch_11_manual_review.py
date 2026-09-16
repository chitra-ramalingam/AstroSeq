from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_11"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211311608", "variable_or_possible_eb", "Broad phase structure, secondary context, and odd/even scatter are EB/variable-like rather than a clean planet hold."),
    ("EPIC_212019726", "variable_or_possible_eb", "Strong stellar variability and broad deep folded morphology dominate; not a planet-like transit recovery."),
    ("EPIC_200008990", "reject_as_noise_or_artifact", "K2 packet is noise/systematics dominated with no reliable repeatable transit family."),
    ("EPIC_212075775", "variable_or_possible_eb", "Coherent raw variability plus primary/secondary phase structure make this EB/variable-like."),
    ("EPIC_211358886", "variable_or_possible_eb", "Strong periodic raw variability and broad folded/event-stack morphology are variable-like rather than candidate-like."),
    ("EPIC_211995639", "false_positive_eb_or_variable", "Broad phase modulation and period ambiguity indicate a variable/EB false positive rather than a clean transit."),
    ("EPIC_211915085", "reject_as_noise_or_artifact", "Dominated by a large instrumental/artifact excursion and sparse inconsistent event support."),
    ("EPIC_211972431", "false_positive_eb_or_variable", "Low-SNR periodic phase structure and broad event-stack trend are variable-like, not planet-like."),
    ("EPIC_211755452", "reject_as_noise_or_artifact", "Noisy high-CNN packet with isolated spike-like structure and no coherent repeatable transit."),
    ("EPIC_211698963", "false_positive_eb_or_variable", "Variable-like raw trend and broad phased structure outweigh the shallow ambiguous dip."),
    ("EPIC_211904513", "reject_as_noise_or_artifact", "Systematics-dominated raw trend with ambiguous noisy folded evidence and no candidate recovery."),
    ("EPIC_211895138", "reject_as_noise_or_artifact", "Period-ambiguous noisy dip with unavailable odd/even and weak event-stack support."),
    ("EPIC_211944929", "reject_as_noise_or_artifact", "Noisy folded feature in a strong trend/systematics packet; no reliable transit family."),
    ("EPIC_211933577", "reject_as_noise_or_artifact", "Weak ambiguous dip in noisy systematics-dominated views; reject sanity remains valid."),
    ("EPIC_211884720", "reject_as_noise_or_artifact", "Noisy period-ambiguous packet with scattered event-stack support and no missed candidate-like signal."),
]


if __name__ == "__main__":
    base.main()
