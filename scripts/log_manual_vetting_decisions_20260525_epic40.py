from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260522_epic35"
RUN_ID = "20260525_epic40"

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

base.DECISIONS = [
    {
        "epic_id": "EPIC_211902535",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Manual review favors weak noise/artifact rather than a planet-like candidate. The signal is very "
            "shallow, with a long apparent duration of about 7.36 hours for P=4.32d, and odd/even consistency is "
            "poor with ratio about 0.680 and explicit delta about 0.381. There is no meaningful positive secondary "
            "eclipse, and OOT variability is low, but the period source is event-spacing fallback with many "
            "candidate periods. Reject as noise/artifact despite morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is moderately high but not extreme; manual rejection is driven by weak visual credibility, "
            "poor odd/even consistency, long duration, and fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211910237",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_stage_f_period_validation_and_eb_diagnostic_review",
        "manual_confidence": "medium",
        "manual_reason": (
            "Clear dip-like signal is present with strong primary SNR around 13.60, low OOT variability at about "
            "0.06x transit depth, many event-family members, and no meaningful positive secondary eclipse. However "
            "the primary depth is relatively large at about 0.015, radius ratio is about 0.123, apparent duration "
            "is long at about 9.32 hours for P=2.21d, and odd/even consistency is only mediocre with ratio about "
            "0.780 and explicit delta about 0.267. Period source is event-spacing fallback with many candidate "
            "periods. Keep as uncertain hold pending Stage F period validation and EB diagnostic review rather "
            "than promote directly."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and consistent with cautious hold rather than promotion.",
    },
    {
        "epic_id": "EPIC_211911273",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual review favors noise/artifact rather than a planet-like candidate. Although the reported primary "
            "SNR is very high, the apparent transit duration is extremely long at about 11.77 hours for P=3.32d, "
            "suggesting broad variability or folded systematic structure rather than a clean transit. Odd/even "
            "consistency is only mediocre with ratio about 0.776 and explicit delta about 0.242, and there is "
            "half-period alias support. Period source is event-spacing fallback with many candidate periods. Reject "
            "as noise/artifact despite morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and does not materially conflict with manual rejection.",
    },
    {
        "epic_id": "EPIC_211915147",
        "manual_label": "candidate_like",
        "manual_next_action": "promote_to_stage_f_validation_with_oot_duration_and_eb_caveats",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual review promotes this as a caveated candidate-like signal. A good primary dip is visible, "
            "odd/even consistency is excellent with ratio about 0.983 and explicit delta about 0.017, there is no "
            "meaningful positive secondary eclipse, event-family support is strong with 21 events, and CNN "
            "morphology score is high. However the promotion is caveated because primary SNR is weak around 4.62, "
            "apparent duration is very long at about 8.34 hours for P=1.81d, OOT variability is high at about "
            "0.69x transit depth, and the period source is event-spacing fallback with stronger double-period "
            "support. Promote to Stage F validation with OOT detrending, duration sanity check, period validation, "
            "and EB diagnostic review."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "High CNN morphology score is consistent with the visible primary dip, but CNN is not the promotion "
            "authority; manual review promotes with explicit caveats."
        ),
    },
    {
        "epic_id": "EPIC_211920612",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Manual review rejects this as noise/artifact. Although the reported primary SNR is extremely high, "
            "the fitted duration is physically implausible at about 44.87 hours for P=0.621d, longer than the "
            "orbital period itself. Odd/even consistency is also severely poor, with ratio about 0.304 and explicit "
            "delta about 0.982. Stage F only held it as ambiguous, and the signal is better explained by bad folded "
            "geometry, variability, or artifact behavior than a planet-like transit. Reject from candidate pipeline "
            "despite morphology-positive CNN score and no detected secondary eclipse."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is moderate-high but not extreme; manual rejection is driven by impossible duration geometry, "
            "severe odd/even mismatch, and artifact-like folded signal behavior."
        ),
    },
]


if __name__ == "__main__":
    base.main()
