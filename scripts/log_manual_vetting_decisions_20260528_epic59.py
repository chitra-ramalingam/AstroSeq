from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260528_epic53"
RUN_ID = "20260528_epic59"

base.PREV_RUN = PREV_RUN
base.RUN_ID = RUN_ID
base.OUT_DIR = base.CATALOG_DIR / "manual_review_updates" / RUN_ID
base.SOURCE_CATALOG = (
    base.CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"master_vetted_catalog_manual_review_update_{PREV_RUN}.csv"
)
base.PREVIOUS_LEDGER = (
    base.CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"manual_vetting_decisions_cumulative_{PREV_RUN}.csv"
)
base.SOURCE_RECONCILED_LEDGER = (
    base.CATALOG_DIR
    / "manual_review_updates"
    / PREV_RUN
    / f"final_candidate_master_ledger_reconciled_manual_review_update_{PREV_RUN}.csv"
)
base.UPDATED_CATALOG = base.OUT_DIR / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
base.UPDATED_CONFLICTS = base.OUT_DIR / f"master_vetted_catalog_conflicts_manual_review_update_{RUN_ID}.csv"
base.MANUAL_LEDGER_INCREMENT = base.OUT_DIR / f"manual_vetting_decisions_increment_{RUN_ID}.csv"
base.MANUAL_LEDGER_CUMULATIVE = base.OUT_DIR / f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
base.UPDATED_RECONCILED_LEDGER = base.OUT_DIR / f"final_candidate_master_ledger_reconciled_manual_review_update_{RUN_ID}.csv"
base.LEDGER_AUDIT = base.OUT_DIR / f"final_candidate_master_ledger_manual_review_audit_{RUN_ID}.csv"
base.SUMMARY_TXT = base.OUT_DIR / f"manual_vetting_update_summary_{RUN_ID}.txt"
base.MANIFEST_JSON = base.OUT_DIR / f"manual_vetting_update_manifest_{RUN_ID}.json"
base.SOURCE_BACKUP = base.OUT_DIR / f"source_catalog_backup_before_manual_review_update_{RUN_ID}.csv"
base.SOURCE_RECONCILED_LEDGER_BACKUP = base.OUT_DIR / f"source_reconciled_ledger_backup_before_manual_review_update_{RUN_ID}.csv"
base.SOURCE_FINAL_LEDGER_BACKUP = base.OUT_DIR / f"source_final_candidate_master_ledger_backup_unmodified_{RUN_ID}.csv"
base.REVIEW_DATE = date(2026, 5, 28).isoformat()
base.LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)


def label_family_with_batch_false_positives(value: str) -> str:
    raw = value.lower()
    if "low_priority_negative" in raw or "variable_or_possible_eb" in raw:
        return "reject"
    return _ORIGINAL_LABEL_FAMILY(value)


def is_manual_reject_with_batch_false_positives(value: str) -> bool:
    return label_family_with_batch_false_positives(value) == "reject"


_ORIGINAL_LABEL_FAMILY = base.label_family
base.label_family = label_family_with_batch_false_positives
base.is_manual_reject = is_manual_reject_with_batch_false_positives

base.DECISIONS = [
    {
        "epic_id": "EPIC_211977407",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_low_priority_negative",
        "manual_reason": (
            "Manual review not convinced; noise-like despite some acceptable numbers. No secondary eclipse detected, "
            "but not enough to rescue it. Primary depth = 0.001346. SNR = 18.58. Odd/even imperfect: ratio = 0.828. "
            "OOT variability high: oot_variability_to_depth = 0.432. Long duration: 10.79 h at P = 4.62576 d. "
            "Period source = event_spacing_fallback. Candidate period count = 1574."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override manual noise/artifact rejection."
        ),
    },
    {
        "epic_id": "EPIC_211979334",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact",
        "manual_reason": (
            "Manual review classifies as poor-quality noise / artifact. Primary depth = 0.000730. SNR = 73.04, but "
            "context is weak. Odd/even poor: ratio = 0.601. Transit duration long: 12.26 h at P = 3.92696 d. "
            "Alias risk = moderate. Strong half-period alias support: support_ratio = 0.857. Period source = "
            "event_spacing_fallback. Candidate period count = 698."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override weak validation context or alias support."
        ),
    },
    {
        "epic_id": "EPIC_211980250",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_false_positive",
        "manual_reason": (
            "Manual review rejects despite strong-looking metrics. Primary depth = 0.014447. SNR = 306.85. Odd/even "
            "good: ratio = 0.928. OOT/depth low: 0.023. No secondary eclipse detected. But visual/manual confidence "
            "is not candidate-like. Long duration: 12.26 h at P = 2.97623 d. Large radius ratio estimate: 0.120. "
            "Period source = event_spacing_fallback. Candidate period count = 476."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override manual judgement that the object is not candidate-like."
        ),
    },
    {
        "epic_id": "EPIC_211980366",
        "manual_label": "variable_or_possible_eb",
        "manual_next_action": "reject_as_planet_candidate",
        "manual_confidence": "not_provided",
        "decision": "reject_as_planet_candidate",
        "stage_g_action": "do_not_promote_as_candidate",
        "training_update": "variable_or_possible_eb_false_positive",
        "manual_reason": (
            "Strong coherent signal, but morphology is not planet-like. Folded light curve shows broad sinusoidal / "
            "wave-like modulation. Transit-like depression is embedded in periodic variability. Raw and detrended "
            "curves show strong repetitive variability. Odd/even excellent: ratio = 0.988. Secondary is negative, "
            "not a clean positive EB secondary, but secondary region is structured. OOT variability high: "
            "oot_variability_to_depth = 0.468. Very short period: P = 0.86494 d. Event family count = 65. Candidate "
            "period count = 4537. Best interpretation: variable star / possible EB / contact-binary-like false "
            "positive."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only; strong coherent morphology is not promotion authority for variable/"
            "possible-EB false positives."
        ),
    },
    {
        "epic_id": "EPIC_211980688",
        "manual_label": "uncertain_hold_positive",
        "manual_next_action": "keep_hold",
        "manual_confidence": "not_provided",
        "decision": "keep_hold",
        "stage_g_action": "do_not_promote_yet",
        "training_update": "do_not_use_for_training_yet",
        "manual_reason": (
            "Manual review mildly positive; possible transit-like signal exists. Depth is shallow but detectable: "
            "0.000397. SNR weak: 5.63. Duration plausible: 1.96 h at P = 13.14578 d. CNN score = 0.811, "
            "morphology-only. Odd/even poor: ratio = 0.634. Possible phase-0.5 feature: "
            "secondary_to_primary_depth_ratio = 0.246. OOT variability high: oot_variability_to_depth = 0.472. "
            "Alias risk = moderate. Strong half-period support: support_ratio = 0.889. Period source = "
            "event_spacing_fallback. Candidate period count = 1119. Keep as positive hold, but do not promote yet."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override hold status from weak SNR, odd/even mismatch, "
            "possible phase-0.5 feature, OOT variability, and alias support."
        ),
    },
    {
        "epic_id": "EPIC_211990908",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_low_priority_negative",
        "manual_reason": (
            "Manual review rejects as unreliable / noise-like. Primary depth = 0.001270. SNR = 15.42. Odd/even poor: "
            "ratio = 0.673. OOT variability high: oot_variability_to_depth = 0.309. Long duration: 9.32 h at "
            "P = 2.6398 d. No secondary eclipse detected, but not enough to rescue. Period source = "
            "event_spacing_fallback. Candidate period count = 469."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and does not override fallback-period support, weak odd/even, high OOT "
            "variability, or manual noise/artifact rejection."
        ),
    },
]


if __name__ == "__main__":
    base.main()
