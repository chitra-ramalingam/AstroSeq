from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_6"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211816343", "variable_or_possible_eb", "Deep coherent sinusoidal/EB-like modulation and high OOT-to-depth dominate the folded view; not candidate-like despite high CNN morphology score."),
    ("EPIC_211996306", "uncertain_hold_positive", "Shallow transit-like dip is repeatable enough to keep as a positive hold, but weak primary SNR and high OOT-to-depth prevent promotion."),
    ("EPIC_211387236", "uncertain_hold_period_ambiguous", "Narrow transit-like dip is visible, but saved-period ambiguity and variable-star context keep it excluded from promotion and training."),
    ("EPIC_211483149", "low_priority_negative", "Only a weak low-SNR dip in a noisy folded view with OOT variability; visual evidence is not candidate-like enough to keep positive."),
    ("EPIC_200008917", "reject_as_noise_or_artifact", "Sparse events and noisy folded/event-stack views do not show a reliable repeatable transit signal."),
    ("EPIC_211960228", "variable_or_possible_eb", "Variable-star morphology, period ambiguity, and secondary/phase structure outweigh the primary-dip evidence."),
    ("EPIC_211812028", "reject_as_noise_or_artifact", "Broad noisy period-ambiguous structure without a convincing event family; reject from candidate queue."),
    ("EPIC_211635232", "variable_or_possible_eb", "Broad coherent modulation with period ambiguity and no clean transit isolation; more consistent with variable/EB contamination than a planet candidate."),
    ("EPIC_211693443", "false_positive_eb_or_variable", "Very strong coherent EB-like modulation with deep primary/secondary structure; reject sanity remains valid."),
    ("EPIC_211923932", "false_positive_eb_or_variable", "Periodic sawtooth/variable morphology and period ambiguity dominate over any transit-like interpretation."),
    ("EPIC_211813818", "false_positive_eb_or_variable", "Variable raw light curve with period ambiguity and secondary/phase structure; not a missed planet candidate."),
    ("EPIC_211346827", "reject_as_noise_or_artifact", "Low-SNR ambiguous folded feature with no reliable transit family in the event stack."),
    ("EPIC_211578168", "reject_as_noise_or_artifact", "Isolated narrow feature in a period-ambiguous, systematics-dominated light curve; reject remains valid."),
    ("EPIC_211927125", "false_positive_eb_or_variable", "Large coherent sinusoidal modulation and broad eclipse-like folded shape are variable/EB-like, not planet-like."),
    ("EPIC_211940500", "reject_as_noise_or_artifact", "Tiny narrow feature with unavailable odd/even evidence and period ambiguity; no candidate-like recovery."),
    ("EPIC_211428897", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and isolated narrow dip without trustworthy support; reject remains valid."),
    ("EPIC_211599622", "reject_as_noise_or_artifact", "Weak period-ambiguous feature in noisy folded views with no convincing repeatable event stack."),
    ("EPIC_211700210", "reject_as_noise_or_artifact", "Noisy period-ambiguous folded signal and inconsistent event-stack support; no missed candidate-like signal."),
]


if __name__ == "__main__":
    base.main()
