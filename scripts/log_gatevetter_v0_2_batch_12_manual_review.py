from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_12"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211402632", "variable_or_possible_eb", "Broad deep folded morphology and raw variability are EB/variable-like rather than a clean planet hold."),
    ("EPIC_211683861", "variable_or_possible_eb", "Strong raw variability and broad deep phase structure dominate; not a candidate-like transit recovery."),
    ("EPIC_211793748", "variable_or_possible_eb", "Broad deep variable-like morphology with ambiguous period support; keep out of candidate queue."),
    ("EPIC_211678611", "variable_or_possible_eb", "Broad/deep folded structure and variable context are more consistent with EB or stellar variability than a clean transit."),
    ("EPIC_211313403", "false_positive_eb_or_variable", "Very broad periodic EB/variable-like folded morphology; reject sanity does not reveal a planet-like miss."),
    ("EPIC_211936869", "false_positive_eb_or_variable", "Broad V-shaped folded structure and variable raw context indicate an EB/variable false positive."),
    ("EPIC_211887839", "reject_as_noise_or_artifact", "High-CNN packet is essentially flat/noisy after folding with no reliable event-stack transit family."),
    ("EPIC_211412628", "reject_as_noise_or_artifact", "Weak noisy ambiguous dip with low SNR and no convincing repeatable transit support."),
    ("EPIC_211509129", "reject_as_noise_or_artifact", "Noisy low-SNR packet with ambiguous folded structure; no missed candidate-like signal."),
    ("EPIC_211891774", "false_positive_eb_or_variable", "Broad sinusoidal phase modulation and period ambiguity are variable-like rather than planet-like."),
    ("EPIC_211976838", "false_positive_eb_or_variable", "Broad deep folded morphology with structured raw variability is EB/variable-like, not a clean transit."),
    ("EPIC_211708484", "reject_as_noise_or_artifact", "Very shallow noisy folded feature on a strong trend with weak repeatable event support."),
    ("EPIC_211759880", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and weak folded dip do not support a candidate-like recovery."),
    ("EPIC_211982300", "reject_as_noise_or_artifact", "Weak noisy low-SNR dip in a trend-dominated packet; reject sanity remains valid."),
]


if __name__ == "__main__":
    base.main()
