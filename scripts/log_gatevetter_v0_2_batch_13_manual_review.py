from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.log_gatevetter_v0_2_batch_4_manual_review as base


base.BATCH_PREFIX = "gatevetter_v0_2_batch_13"
base.REVIEW_BATCH = f"{base.BATCH_PREFIX}_manual_review"
base.STAGE_G_QUEUE = ROOT / f"{base.BATCH_PREFIX}_stage_g_queue.csv"
base.TOP_HOLDS = ROOT / f"{base.BATCH_PREFIX}_top_20_holds.csv"
base.REJECT_SANITY = ROOT / f"{base.BATCH_PREFIX}_reject_sanity_sample.csv"
base.VISUAL_MANIFEST = ROOT / f"{base.BATCH_PREFIX}_visual_manifest.csv"
base.DECISIONS_CSV = ROOT / f"{base.BATCH_PREFIX}_manual_review_decisions.csv"
base.SUMMARY_TXT = ROOT / f"{base.BATCH_PREFIX}_manual_review_summary.txt"

base.DECISIONS = [
    ("EPIC_211842777", "uncertain_hold_positive", "Narrow repeatable folded dip with good SNR and low alias risk, but strong raw variability keeps it as a conservative positive hold."),
    ("EPIC_211904163", "variable_or_possible_eb", "Large coherent raw variability and broad phase modulation dominate over the transit-like window."),
    ("EPIC_211939380", "reject_as_noise_or_artifact", "Noisy weak folded feature with poor event-stack support and no convincing repeatable transit family."),
    ("EPIC_211965141", "reject_as_noise_or_artifact", "Very low-SNR noisy packet with no coherent folded or event-stack transit signal."),
    ("EPIC_211753556", "variable_or_possible_eb", "Broad deep U-shaped phase structure and strong raw variability are EB/variable-like rather than planet-like."),
    ("EPIC_211910338", "uncertain_hold_positive", "Compact transit-like folded dip is visually plausible, but low SNR and moderate alias risk keep it as a hold."),
    ("EPIC_211944237", "variable_or_possible_eb", "Deep broad period-ambiguous morphology and strong raw variability are consistent with EB/variable behavior."),
    ("EPIC_211341364", "false_positive_eb_or_variable", "Broad period-ambiguous event-stack structure with weak folded evidence supports EB/variable false-positive handling."),
    ("EPIC_211828464", "reject_as_noise_or_artifact", "Extreme raw systematics and nearly flat folded views provide no reliable transit-like recovery despite high CNN score."),
    ("EPIC_211927492", "false_positive_eb_or_variable", "Broad variable-like phase structure with period ambiguity remains an EB/variable false positive, not a planet-like miss."),
    ("EPIC_211569427", "reject_as_noise_or_artifact", "Systematics-dominated raw trend with flat/noisy folded evidence and no credible repeatable transit."),
    ("EPIC_211963425", "reject_as_noise_or_artifact", "Low-SNR noisy period-ambiguous feature with weak event support; reject sanity remains valid."),
    ("EPIC_211874676", "reject_as_noise_or_artifact", "Raw trend/systematics dominate and the folded/event-stack views do not show a reliable transit family."),
    ("EPIC_211406539", "reject_as_noise_or_artifact", "Noisy period-ambiguous packet with isolated dip-like structure and inconsistent event support."),
    ("EPIC_211695616", "reject_as_noise_or_artifact", "Long-period ambiguous low-SNR packet with no convincing repeatable transit evidence."),
    ("EPIC_211746807", "reject_as_noise_or_artifact", "Noisy folded structure and ambiguous event-stack behavior do not recover a candidate-like signal."),
    ("EPIC_211685798", "reject_as_noise_or_artifact", "High-CNN but very shallow noisy period-ambiguous packet with no reliable candidate-like transit."),
]


if __name__ == "__main__":
    base.main()
