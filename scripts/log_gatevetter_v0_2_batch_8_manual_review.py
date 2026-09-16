from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_8"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211349119", "variable_or_possible_eb", "Trusted refreshed period but folded/zoom views are dominated by variable-like structure and high OOT-to-depth; not a clean transit candidate."),
    ("EPIC_200008725", "reject_as_noise_or_artifact", "No reliable repeatable transit signal in noisy K2 views; event stack is sparse and inconsistent."),
    ("EPIC_211889082", "uncertain_hold_period_ambiguous", "Broad transit-like dip is visible, but period ambiguity, strong raw variability, and missing cross-checks keep it excluded from promotion and training."),
    ("EPIC_211626213", "variable_or_possible_eb", "Strong raw variability and sharp period-ambiguous folded feature are more consistent with variable or EB contamination than a planet candidate."),
    ("EPIC_211377847", "variable_or_possible_eb", "Broad variable/EB-like phase structure and OOT variability dominate the candidate-looking dip."),
    ("EPIC_212015640", "reject_as_noise_or_artifact", "Duration-fraction hold with weak/noisy folded evidence and no convincing repeatable transit family."),
    ("EPIC_212034839", "reject_as_noise_or_artifact", "Period-ambiguous noisy folded feature with no reliable event-stack support; reject sanity remains valid."),
    ("EPIC_200008962", "reject_as_noise_or_artifact", "Noisy K2 signal with sparse inconsistent event stack and no candidate-like recovery."),
    ("EPIC_211825860", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and weak ambiguous folded feature; no missed candidate-like signal."),
    ("EPIC_211499454", "reject_as_noise_or_artifact", "Low-SNR ambiguous folded dip with no reliable supporting transit family."),
    ("EPIC_211507490", "false_positive_eb_or_variable", "Phase structure and secondary/odd-even context are EB/variable-like rather than planet-like."),
    ("EPIC_211499472", "reject_as_noise_or_artifact", "High-CNN weak reject remains noise/systematics dominated with ambiguous period support."),
    ("EPIC_200008785", "reject_as_noise_or_artifact", "Noisy K2 packet with sparse event stack and no stable transit-like recurrence."),
    ("EPIC_211566160", "reject_as_noise_or_artifact", "Period-ambiguous cadence-like banding and unavailable odd/even evidence; no clean transit signal."),
    ("EPIC_211977134", "reject_as_noise_or_artifact", "Weak ambiguous folded feature in a systematics-dominated light curve; reject remains valid."),
    ("EPIC_211780791", "reject_as_noise_or_artifact", "Long-period ambiguous spike/noise morphology without reliable candidate-like support."),
]


if __name__ == "__main__":
    base.main()
