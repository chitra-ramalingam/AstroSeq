from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_5"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211351798", "false_positive_eb_or_variable", "Strong sinusoidal variability, OOT variability, and secondary-like structure; not a planet-candidate hold."),
    ("EPIC_211768304", "uncertain_hold_positive", "Sharp transit-like dip with good CNN support, but variability/secondary context keeps it as a positive hold rather than Stage G."),
    ("EPIC_212029934", "false_positive_eb_or_variable", "Large coherent modulation and broad EB-like phased structure; reject from planet-candidate queue."),
    ("EPIC_211485867", "uncertain_hold_period_ambiguous", "Transit-like dip is visible, but period ambiguity and missing/NA cross-checks prevent positive promotion."),
    ("EPIC_211775796", "variable_or_possible_eb", "Structured stellar variability and secondary/phase features outweigh the weak primary dip."),
    ("EPIC_211775229", "variable_or_possible_eb", "Variable/EB-like morphology with period ambiguity and secondary structure; not a planet candidate."),
    ("EPIC_211328173", "reject_as_noise_or_artifact", "No convincing transit-like signal and invalid/negative primary-depth evidence."),
    ("EPIC_211918830", "false_positive_eb_or_variable", "Deep coherent EB-like modulation and high OOT-to-depth; reject sanity remains valid."),
    ("EPIC_211308816", "reject_as_noise_or_artifact", "No coherent transit-like family in folded or event-stack views; high-CNN reject remains valid."),
    ("EPIC_212031287", "reject_as_noise_or_artifact", "Period-ambiguous spike/noise structure without repeatable transit evidence."),
    ("EPIC_211947216", "reject_as_noise_or_artifact", "Noisy folded/zoom views and period ambiguity; no missed candidate-like signal."),
    ("EPIC_211600288", "reject_as_noise_or_artifact", "Period-ambiguous spike/noise morphology with no reliable event family."),
    ("EPIC_212020821", "reject_as_noise_or_artifact", "Isolated narrow feature in noisy period-ambiguous context; reject remains valid."),
    ("EPIC_211796395", "reject_as_noise_or_artifact", "Noisy ambiguous folded structure with no convincing planet-like transit family."),
    ("EPIC_211496260", "reject_as_noise_or_artifact", "Weak/noisy period-ambiguous dip; no candidate-like recovery."),
    ("EPIC_212032390", "reject_as_noise_or_artifact", "Tiny period-ambiguous feature in otherwise noisy folded view; reject remains valid."),
    ("EPIC_211610944", "reject_as_noise_or_artifact", "Noisy/structured period-ambiguous signal without reliable transit evidence."),
]


if __name__ == "__main__":
    base.main()
