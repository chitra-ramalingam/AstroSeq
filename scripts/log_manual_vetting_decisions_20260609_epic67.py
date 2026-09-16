from __future__ import annotations

from datetime import date
import shutil

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260601_epic65"
RUN_ID = "20260609_epic67"

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
base.UPDATED_CATALOG = (
    base.OUT_DIR
    / f"master_vetted_catalog_manual_review_update_{RUN_ID}.csv"
)
base.UPDATED_CONFLICTS = (
    base.OUT_DIR
    / f"master_vetted_catalog_conflicts_manual_review_update_{RUN_ID}.csv"
)
base.MANUAL_LEDGER_INCREMENT = (
    base.OUT_DIR / f"manual_vetting_decisions_increment_{RUN_ID}.csv"
)
base.MANUAL_LEDGER_CUMULATIVE = (
    base.OUT_DIR / f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)
base.UPDATED_RECONCILED_LEDGER = (
    base.OUT_DIR
    / f"final_candidate_master_ledger_reconciled_manual_review_update_{RUN_ID}.csv"
)
base.LEDGER_AUDIT = (
    base.OUT_DIR
    / f"final_candidate_master_ledger_manual_review_audit_{RUN_ID}.csv"
)
base.SUMMARY_TXT = (
    base.OUT_DIR / f"manual_vetting_update_summary_{RUN_ID}.txt"
)
base.MANIFEST_JSON = (
    base.OUT_DIR / f"manual_vetting_update_manifest_{RUN_ID}.json"
)
base.SOURCE_BACKUP = (
    base.OUT_DIR
    / f"source_catalog_backup_before_manual_review_update_{RUN_ID}.csv"
)
base.SOURCE_RECONCILED_LEDGER_BACKUP = (
    base.OUT_DIR
    / f"source_reconciled_ledger_backup_before_manual_review_update_{RUN_ID}.csv"
)
base.SOURCE_FINAL_LEDGER_BACKUP = (
    base.OUT_DIR
    / f"source_final_candidate_master_ledger_backup_unmodified_{RUN_ID}.csv"
)
base.REVIEW_DATE = date(2026, 6, 9).isoformat()
base.LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)


def label_family_with_variable_false_positives(value: str) -> str:
    if "variable_or_possible_eb" in value.lower():
        return "reject"
    return _ORIGINAL_LABEL_FAMILY(value)


def is_manual_reject_with_variable_false_positives(value: str) -> bool:
    return label_family_with_variable_false_positives(value) == "reject"


_ORIGINAL_LABEL_FAMILY = base.label_family
base.label_family = label_family_with_variable_false_positives
base.is_manual_reject = is_manual_reject_with_variable_false_positives


base.DECISIONS = [
    {
        "epic_id": "EPIC_211839462",
        "manual_label": "variable_or_possible_eb",
        "manual_next_action": "reject_as_planet_candidate",
        "manual_confidence": "high",
        "decision": "reject_as_planet_candidate",
        "stage_g_action": "do_not_promote_as_candidate",
        "training_update": "do_not_update_training_automatically",
        "manual_reason": (
            "Rejected and removed from the positive hold path after deeper "
            "saved-, half-, and double-period review plus variability "
            "diagnostics. The signal is coherent but dominated by strong "
            "periodic variability with deep repeating troughs. Saved- and "
            "half-period folds show broad structured events, high OOT/depth "
            "of approximately 8-12, strong half-period support, and non-flat "
            "out-of-transit morphology. The final manual interpretation is "
            "a variable/possible-EB astrophysical false positive rather than "
            "an isolated planet-like transit."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN morphology evidence does not override deeper manual "
            "variable/possible-EB adjudication."
        ),
    },
    {
        "epic_id": "EPIC_212019207",
        "manual_label": "uncertain_hold_period_ambiguous",
        "manual_next_action": "keep_hold_for_period_resolution",
        "manual_confidence": "medium",
        "decision": "keep_hold",
        "stage_g_action": "do_not_promote_yet",
        "training_update": "do_not_update_training_automatically",
        "manual_reason": (
            "Retain as an uncertain hold because the period remains "
            "ambiguous. Do not promote until a trusted period and required "
            "period-dependent cross-checks are available."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN morphology evidence is compatible with review priority but "
            "does not resolve period ambiguity."
        ),
    },
]


CANONICAL_LEDGER = (
    base.CATALOG_DIR / "manual_vetting_decisions_ledger.csv"
)
CURRENT_RECONCILED = (
    base.CATALOG_DIR
    / "final_candidate_master_ledger_reconciled_current.csv"
)
CANONICAL_LEDGER_BACKUP = (
    base.OUT_DIR
    / f"manual_vetting_decisions_ledger_before_{RUN_ID}.csv"
)
CURRENT_RECONCILED_BACKUP = (
    base.OUT_DIR
    / f"final_candidate_master_ledger_reconciled_current_before_{RUN_ID}.csv"
)


def publish_current_copies() -> None:
    shutil.copy2(CANONICAL_LEDGER, CANONICAL_LEDGER_BACKUP)
    shutil.copy2(CURRENT_RECONCILED, CURRENT_RECONCILED_BACKUP)
    shutil.copy2(base.MANUAL_LEDGER_CUMULATIVE, CANONICAL_LEDGER)
    shutil.copy2(base.UPDATED_RECONCILED_LEDGER, CURRENT_RECONCILED)


if __name__ == "__main__":
    base.main()
    publish_current_copies()
