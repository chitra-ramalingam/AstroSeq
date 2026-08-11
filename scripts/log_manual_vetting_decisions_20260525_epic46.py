from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260525_epic40"
RUN_ID = "20260525_epic46"

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
base.REVIEW_DATE = date(2026, 5, 25).isoformat()
base.LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)


def label_family_with_low_priority_negative(value: str) -> str:
    raw = value.lower()
    if "low_priority_negative" in raw:
        return "reject"
    return _ORIGINAL_LABEL_FAMILY(value)


def is_manual_reject_with_low_priority_negative(value: str) -> bool:
    return label_family_with_low_priority_negative(value) == "reject"


_ORIGINAL_LABEL_FAMILY = base.label_family
base.label_family = label_family_with_low_priority_negative
base.is_manual_reject = is_manual_reject_with_low_priority_negative

base.DECISIONS = [
    {
        "epic_id": "EPIC_211920612",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_bad_fold",
        "manual_reason": (
            "Strong dip-like morphology, but physically broken. Period = 0.62108 d / 14.91 h. Transit duration = "
            "1.8695 d / 44.87 h, longer than the orbital period. Severe odd/even mismatch: odd_even_depth_ratio = "
            "0.304. CNN score = 0.823 is morphology-only and should not rescue it."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and should not override failed validation checks or impossible duration "
            "geometry."
        ),
    },
    {
        "epic_id": "EPIC_211933215",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_low_priority_negative",
        "manual_reason": (
            "Too weak and period-unstable. Primary depth is shallow: 0.000394. SNR is only moderate: 11.86. Period "
            "source = event_spacing_fallback. Autovet says no reliable saved period support. Candidate period count "
            "is very high: 1039. CNN score = 0.731 is morphology-only."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and should not override weak period support or period instability."
        ),
    },
    {
        "epic_id": "EPIC_211935518",
        "manual_label": "low_priority_negative",
        "manual_next_action": "reject_low_priority",
        "manual_confidence": "not_provided",
        "decision": "reject_low_priority",
        "stage_g_action": "do_not_promote",
        "training_update": "low_priority_negative_or_uncertain_hold",
        "manual_reason": (
            "Visible dip exists and metrics are not terrible, but too noisy / unreliable to promote. Primary depth = "
            "0.005235. SNR = 174.90. Odd/even agreement is good: ratio = 0.929. No meaningful secondary. However "
            "period source = event_spacing_fallback. Autovet says no reliable saved period support. Candidate period "
            "count is high: 751. CNN score = 0.769 is morphology-only."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and does not override fallback-period support and unreliable validation."
        ),
    },
    {
        "epic_id": "EPIC_211939206",
        "manual_label": "low_priority_negative",
        "manual_next_action": "reject_low_priority",
        "manual_confidence": "not_provided",
        "decision": "reject_low_priority",
        "stage_g_action": "do_not_promote",
        "training_update": "low_priority_negative",
        "manual_reason": (
            "Best-looking noise case. Some transit-like dip morphology exists, but validation is weak. Primary depth "
            "= 0.001054. SNR = 43.47. Odd/even agreement is imperfect: ratio = 0.712, delta = 0.310. Alias risk = "
            "moderate. Strong alias/double-period support: alias_best_period_days = 3.16692, support_ratio = 0.889. "
            "Period source = event_spacing_fallback. Candidate period count is extremely high: 1671. CNN score = "
            "0.736 is morphology-only."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and should not override weak validation, alias risk, or fallback-period "
            "support."
        ),
    },
    {
        "epic_id": "EPIC_211942823",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact",
        "manual_reason": (
            "Manual visual review sees no convincing dip; folded view appears essentially flat / line-like. Primary "
            "depth is shallow: 0.000423. SNR = 13.01. Period = 0.20432 d / 4.90 h. Transit duration = 0.36777 d / "
            "8.83 h, longer than the orbital period. Odd/even agreement is weak: ratio = 0.702. Alias risk = high. "
            "Period source = event_spacing_fallback. Candidate period count is high: 1049."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN morphology is not promotion authority; manual rejection is driven by no convincing dip, impossible "
            "duration geometry, high alias risk, and fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211944670",
        "manual_label": "low_priority_negative",
        "manual_next_action": "reject_low_priority",
        "manual_confidence": "not_provided",
        "decision": "reject_low_priority",
        "stage_g_action": "do_not_promote",
        "training_update": "low_priority_negative",
        "manual_reason": (
            "Manual review sees a dip, but not good enough for promotion. OOT behaviour is acceptable. Primary "
            "depth = 0.002908. SNR is very low: 3.02. Odd/even agreement is imperfect: ratio = 0.721. No secondary "
            "eclipse detected. Duration is physically plausible: 1.47 h at P = 1.55281 d. But period source = "
            "event_spacing_fallback. Alias risk = moderate. Strong double-period support: alias_best_period_days = "
            "3.10562, support_ratio = 1.0. Candidate period count is high: 865. CNN score = 0.981 is morphology-only "
            "and should not override weak validation."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is high but morphology-only; manual review rejects low priority because SNR is very low with "
            "fallback-period support and strong double-period alias support."
        ),
    },
]


if __name__ == "__main__":
    base.main()
