from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_10"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211411882", "variable_or_possible_eb", "Broad primary-like phase structure, secondary/phase context, and raw variability make this EB/variable-like rather than a clean planet hold."),
    ("EPIC_211696209", "reject_as_noise_or_artifact", "Weak shallow folded feature in a strongly variable raw light curve with poor candidate isolation; no reliable transit family."),
    ("EPIC_211768258", "variable_or_possible_eb", "Coherent stellar variability and broad folded/secondary context dominate the shallow dip."),
    ("EPIC_211845034", "variable_or_possible_eb", "Deep broad EB/variable-like morphology with strong periodic raw variability and non-planet-like event structure."),
    ("EPIC_211905009", "reject_as_noise_or_artifact", "Low-SNR ambiguous folded feature with sparse event-stack support and no clean recurrence."),
    ("EPIC_211931266", "reject_as_noise_or_artifact", "Noisy period-ambiguous packet with weak folded evidence and no convincing transit family."),
    ("EPIC_211778092", "variable_or_possible_eb", "Raw light curve is dominated by coherent variability and the folded feature is broad/ambiguous rather than candidate-like."),
    ("EPIC_211950430", "variable_or_possible_eb", "Deep broad eclipse-like morphology, strong raw variability, and large secondary/phase structure are EB/variable-like."),
    ("EPIC_211396385", "reject_as_noise_or_artifact", "High-CNN period-ambiguous reject has a narrow noisy dip but inconsistent event-stack and secondary context; no candidate recovery."),
    ("EPIC_211953962", "reject_as_noise_or_artifact", "Very shallow noisy folded dip with weak event-stack support and no stable transit-like recurrence."),
    ("EPIC_211721876", "reject_as_noise_or_artifact", "High-CNN packet is dominated by systematics and isolated spikes; folded/zoom support is not candidate-like."),
    ("EPIC_211722736", "false_positive_eb_or_variable", "Broad variable-like folded morphology with OOT structure and weak odd/even support; reject remains valid."),
    ("EPIC_211492164", "false_positive_eb_or_variable", "Strong raw variability plus primary/secondary eclipse-like features indicate EB/variable contamination, not a planet candidate."),
    ("EPIC_211563488", "reject_as_noise_or_artifact", "Systematics-dominated raw trend and noisy period-ambiguous fold; no missed candidate-like signal."),
    ("EPIC_211753405", "reject_as_noise_or_artifact", "Noisy ambiguous folded feature with weak inconsistent event-stack support; reject sanity remains valid."),
    ("EPIC_212021968", "reject_as_noise_or_artifact", "High-CNN noise/systematics packet with broad scatter and no coherent repeatable transit signal."),
    ("EPIC_211978488", "reject_as_noise_or_artifact", "Weak period-ambiguous dip in a trend/systematics-dominated light curve; no candidate-like recovery."),
    ("EPIC_212009056", "reject_as_noise_or_artifact", "Noisy folded evidence and inconsistent event stack do not support a reliable candidate-like signal."),
]


if __name__ == "__main__":
    base.main()
