from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_9"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211431812", "variable_or_possible_eb", "Large coherent raw variability and secondary/phase structure dominate the shallow folded dip; not a clean planet-like hold."),
    ("EPIC_212021367", "reject_as_noise_or_artifact", "Weak low-SNR folded dip in a variable raw light curve with sparse event-stack support; no reliable transit family."),
    ("EPIC_211593145", "variable_or_possible_eb", "Deep broad eclipse/variable-like morphology with secondary/phase structure; exclude from planet-candidate promotion."),
    ("EPIC_211610125", "false_positive_eb_or_variable", "Period-ambiguous short-period variability with primary/secondary-like structure and broad phased modulation; not planet-like."),
    ("EPIC_211716614", "reject_as_noise_or_artifact", "No coherent folded transit; event stack and zoom are noisy and period ambiguous despite high CNN score."),
    ("EPIC_211432167", "reject_as_noise_or_artifact", "Very shallow ambiguous structure with no convincing transit zoom or event-stack support."),
    ("EPIC_200008946", "reject_as_noise_or_artifact", "K2 noise/banding dominates the packet; folded and event-stack evidence are sparse and inconsistent."),
    ("EPIC_211685323", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and period-ambiguous weak folded feature; no missed candidate-like signal."),
    ("EPIC_211534290", "reject_as_noise_or_artifact", "Ambiguous low-SNR folded feature in a variable/systematics-dominated light curve without stable transit support."),
    ("EPIC_211766291", "false_positive_eb_or_variable", "Sharp eclipse-like dip occurs in strong stellar variability with period ambiguity and non-planet-like odd/even context."),
    ("EPIC_211383838", "reject_as_noise_or_artifact", "Low-amplitude ambiguous folded trend with no reliable event-stack recurrence; reject sanity remains valid."),
    ("EPIC_211946205", "reject_as_noise_or_artifact", "Period-ambiguous feature in a systematics-dominated light curve; event stack is not strong enough for a candidate-like hold."),
    ("EPIC_211976999", "reject_as_noise_or_artifact", "Noisy high-CNN packet with period ambiguity and inconsistent event-stack support; no candidate-like recovery."),
]


if __name__ == "__main__":
    base.main()
