from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_7"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211959909", "uncertain_hold_positive", "Repeatable transit-like dip is visible at the trusted refreshed period, but raw variability, OOT variability, and weak primary SNR keep it as a positive hold only."),
    ("EPIC_211769243", "variable_or_possible_eb", "Strong coherent stellar variability and period ambiguity dominate; the narrow folded feature is not clean candidate-like evidence."),
    ("EPIC_200008977", "reject_as_noise_or_artifact", "Sparse noisy events and inconsistent event-stack support do not show a reliable repeatable transit signal."),
    ("EPIC_211980815", "variable_or_possible_eb", "Coherent long-period variability with period ambiguity and only a tiny folded dip; not a planet-candidate hold."),
    ("EPIC_211485234", "variable_or_possible_eb", "Large sinusoidal variability, period ambiguity, and secondary/phase structure outweigh the primary-dip evidence."),
    ("EPIC_211811904", "variable_or_possible_eb", "Strong variable-star morphology with period ambiguity and weak transit isolation; exclude from candidate queue."),
    ("EPIC_211421801", "variable_or_possible_eb", "Broad EB/variable-like folded morphology and high OOT-to-depth keep this out of candidate-like promotion."),
    ("EPIC_211799289", "variable_or_possible_eb", "Broad variable/EB-like phased structure with OOT variability; not a clean transit candidate."),
    ("EPIC_211804680", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and weak ambiguous folded feature; reject sanity remains valid."),
    ("EPIC_211584718", "reject_as_noise_or_artifact", "Noisy period-ambiguous folded feature without reliable event-stack support."),
    ("EPIC_212019055", "false_positive_eb_or_variable", "Very deep coherent eclipsing/variable morphology with strong secondary structure; not planet-like."),
    ("EPIC_211790174", "reject_as_noise_or_artifact", "Low-SNR ambiguous folded feature with no candidate-like repeatable transit family."),
    ("EPIC_211790348", "reject_as_noise_or_artifact", "Period-ambiguous noisy spike structure and no convincing transit evidence."),
    ("EPIC_211990732", "reject_as_noise_or_artifact", "Systematics/noise-dominated folded view with weak, ambiguous features only."),
    ("EPIC_211957146", "variable_or_possible_eb", "Large coherent variability and EB-like folded/event-stack morphology; reject from planet-candidate queue."),
    ("EPIC_211486822", "variable_or_possible_eb", "Raw light curve shows coherent variability and the folded feature is too weak for candidate-like recovery."),
    ("EPIC_211899798", "reject_as_noise_or_artifact", "Noisy period-ambiguous narrow feature with no reliable supporting event stack."),
    ("EPIC_211378170", "reject_as_noise_or_artifact", "Weak ambiguous folded dip in a systematics-dominated light curve; no missed candidate-like signal."),
]


if __name__ == "__main__":
    base.main()
