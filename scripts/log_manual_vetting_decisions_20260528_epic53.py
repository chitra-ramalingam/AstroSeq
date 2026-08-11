from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260525_epic46"
RUN_ID = "20260528_epic53"

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
        "epic_id": "EPIC_211945246",
        "manual_label": "low_priority_negative",
        "manual_next_action": "reject_low_priority",
        "manual_confidence": "not_provided",
        "decision": "reject_low_priority",
        "stage_g_action": "do_not_promote",
        "training_update": "low_priority_negative",
        "manual_reason": (
            "Folded light curve shows only a small weak possible dip. Secondary diagnostic is flat / line-like, "
            "with no meaningful secondary eclipse. Primary depth is shallow: 0.000907. SNR is weak: 6.65. "
            "Odd/even is acceptable but not clean: ratio = 0.786. OOT is good: oot_variability_to_depth = 0.0119. "
            "Period source = event_spacing_fallback. Candidate period count = 698. Not convincing enough for "
            "promotion."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override weak manual/validation context."
        ),
    },
    {
        "epic_id": "EPIC_211945592",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_possible_eb",
        "manual_reason": (
            "Manual review suggests heavy noise and/or possible EB-like contamination. Primary depth = 0.002213. "
            "SNR = 119.76, but likely driven by structured noise / non-planet morphology. OOT variability is high: "
            "oot_variability_to_depth = 0.249. Odd/even is only moderately acceptable: ratio = 0.845. Period source "
            "= event_spacing_fallback. Candidate period count = 1248. CNN score = 0.682 is morphology-only and does "
            "not rescue it."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and does not override structured-noise or possible EB-like context."
        ),
    },
    {
        "epic_id": "EPIC_211949114",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_low_priority_negative",
        "manual_reason": (
            "Manual review classifies this as noise despite some decent metrics. Primary depth = 0.001353. "
            "SNR = 87.44. Odd/even is decent: ratio = 0.885. No detected secondary. But OOT variability is high: "
            "oot_variability_to_depth = 0.288. Period source = event_spacing_fallback. Candidate period count = "
            "1258. Context is too unstable for promotion."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override unstable validation context."
        ),
    },
    {
        "epic_id": "EPIC_211952381",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_possible_eb",
        "manual_reason": (
            "Manual review suggests noise / possible EB-like morphology. Stage F already held it due to poor visual/"
            "metric confidence. Primary depth = 0.001463. SNR = 18.06. Odd/even is weak: ratio = 0.717. OOT "
            "variability is high: oot_variability_to_depth = 0.355. Period source = saved_validation_period, but "
            "metrics and visual confidence still fail. Reject despite saved period support."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override failed manual visual/metric confidence."
        ),
    },
    {
        "epic_id": "EPIC_211953866",
        "manual_label": "candidate_like",
        "manual_next_action": "promote_to_stage_g",
        "manual_confidence": "not_provided",
        "decision": "promote_to_stage_g",
        "stage_g_action": "promote_for_deeper_eval",
        "training_update": "candidate_like_with_caveat",
        "manual_reason": (
            "Manual visual review is very positive. Folded light curve and transit zoom show a clear coherent "
            "primary dip. Event stack shows multiple events contributing to the dip structure. Primary depth = "
            "0.024265. CNN score = 0.990 supports strong morphology, but remains morphology-only. Odd/even is "
            "acceptable: ratio = 0.883. Duration is plausible: 3.43 h at P = 1.78777 d. No convincing positive "
            "phase-0.5 secondary. Caveats: fallback period, candidate_period_count = 1013, OOT/depth = 0.400, low "
            "numeric SNR = 3.30, large radius ratio = 0.156. Promote only for deeper Stage G evaluation, not "
            "planet_like / confirmed."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN supports morphology only; manual review promotes for deeper Stage G evaluation with caveats, not as "
            "a confirmed planet."
        ),
    },
    {
        "epic_id": "EPIC_211959522",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold",
        "manual_confidence": "not_provided",
        "decision": "keep_hold",
        "stage_g_action": "do_not_promote_yet",
        "training_update": "do_not_use_for_training_yet",
        "manual_reason": (
            "Folded view shows a small possible primary dip, but not enough for candidate_like. Odd/even agreement "
            "is excellent: ratio = 0.974. No obvious phase-0.5 secondary. But depth is very shallow: 0.000614. SNR "
            "is modest: 10.62. OOT variability is very high relative to depth: oot_variability_to_depth = 0.859. "
            "Event stack does not show a clearly repeated aligned transit-like dip. Period source = "
            "event_spacing_fallback. Candidate period count = 1066. Keep on hold; do not use for training yet."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override hold status from shallow depth, high OOT variability, "
            "and unclear event-stack support."
        ),
    },
    {
        "epic_id": "EPIC_211975006",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact",
        "manual_reason": (
            "Manual review calls this a perfect noise case despite strong-looking numbers. Primary depth = 0.004142. "
            "SNR = 409.32, but not trusted due to noisy / artifact-like context. Odd/even is fairly good: ratio = "
            "0.911. No convincing positive secondary. Transit duration is long: 13.24 h at P = 3.08007 d. Period "
            "source = event_spacing_fallback. Candidate period count = 1448. CNN score = 0.761 is morphology-only "
            "and should not override manual rejection."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is morphology-only and should not override manual noise/artifact rejection."
        ),
    },
]


if __name__ == "__main__":
    base.main()
