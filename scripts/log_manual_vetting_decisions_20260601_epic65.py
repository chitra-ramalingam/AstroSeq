from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260528_epic59"
RUN_ID = "20260601_epic65"

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
base.REVIEW_DATE = date(2026, 6, 1).isoformat()
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
        "epic_id": "EPIC_211991001",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_false_positive",
        "manual_reason": (
            "Formidable noise / false-positive-like signal. Large depth = 0.016837, but SNR only 7.02. Very short "
            "period: 0.52685 d. Duration = 6.37 h, about half the orbital period. Odd/even imperfect: ratio = "
            "0.764. Alias risk = moderate. Half-period alias support is perfect: support_ratio = 1.0. Large radius "
            "ratio estimate = 0.130. Saved period exists, but Stage F already held due to ambiguous checks."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override manual visual review or ambiguous validation context."
        ),
    },
    {
        "epic_id": "EPIC_211995966",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_bad_fold",
        "manual_reason": (
            "Weird / ambiguous noise: something but nothing. Stage F already rejected it because primary transit "
            "depth is not significant in cached folded light curve. Primary SNR is very weak: 2.73. Event support "
            "is thin: event_family_count = 3. Alias risk = moderate. Odd/even is good: ratio = 0.952, but weak "
            "primary detection dominates. CNN score = 0.879 is morphology-only and does not override Stage F "
            "rejection."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override Stage F rejection or weak primary detection."
        ),
    },
    {
        "epic_id": "EPIC_212007527",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_low_priority_negative",
        "manual_reason": (
            "Manual review says no; not convincing as candidate. Primary depth is very shallow: 0.000408. SNR is "
            "modest: 12.24. Period source = event_spacing_fallback. Candidate period count is enormous: 7655. "
            "Half-period and double-period support are both strong. OOT is not terrible, but period ambiguity is too "
            "severe."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override manual rejection or severe period ambiguity."
        ),
    },
    {
        "epic_id": "EPIC_212009427",
        "manual_label": "variable_or_possible_eb",
        "manual_next_action": "reject_as_planet_candidate",
        "manual_confidence": "not_provided",
        "decision": "reject_as_planet_candidate",
        "stage_g_action": "do_not_promote_as_candidate",
        "training_update": "variable_or_possible_eb_false_positive",
        "manual_reason": (
            "Signal is real-looking, but morphology suggests EB / variable-star-like false positive. Folded light "
            "curve and transit zoom show a broad deep event embedded in variability. Raw and detrended curves show "
            "strong repetitive variability. Secondary check is structured rather than clean/flat. Secondary depth is "
            "negative, not a clean positive secondary, but still suspicious in context. Odd/even agreement is "
            "excellent: ratio = 0.975. OOT/depth is high: 0.505. Alias risk = moderate. Strong half-period support: "
            "support_ratio = 0.867. Candidate period count = 3383."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only; coherent morphology is not promotion authority for variable/possible-EB "
            "false positives."
        ),
    },
    {
        "epic_id": "EPIC_212020442",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact_or_period_failure",
        "manual_reason": (
            "Manual review indicates no convincing period / period solution not trustworthy. Primary depth is "
            "shallow: 0.000683. SNR is weak: 7.60. Odd/even only moderate: ratio = 0.815. Alias risk = moderate. "
            "Strong alternate-period support: alias_best_period_days = 11.768625, support_ratio = 0.875. Master next "
            "action was run_period_search_before_label_update. Treat as period failure / noise, not candidate."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override manual rejection or period-failure context."
        ),
    },
    {
        "epic_id": "EPIC_212025784",
        "manual_label": "reject_as_noise_or_artifact",
        "manual_next_action": "reject",
        "manual_confidence": "not_provided",
        "decision": "reject",
        "stage_g_action": "do_not_promote",
        "training_update": "noise_or_artifact",
        "manual_reason": (
            "Perfect noise / artifact. Primary depth = 0.002349. SNR = 76.34, but not trusted due to noisy context. "
            "Odd/even fairly good: ratio = 0.913. OOT variability is very high: oot_variability_to_depth = 0.718. "
            "Long duration: 12.01 h at P = 3.74308 d. Period source = event_spacing_fallback. Candidate period count "
            "= 1301. Half-period and double-period support both high."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN remains morphology-only and does not override noisy context, high OOT variability, or alias support."
        ),
    },
]


if __name__ == "__main__":
    base.main()
