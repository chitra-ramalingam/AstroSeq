from __future__ import annotations

from datetime import date

import log_manual_vetting_decisions_20260522_epic29 as base


PREV_RUN = "20260522_epic29"
RUN_ID = "20260522_epic35"

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
base.REVIEW_DATE = date(2026, 5, 22).isoformat()
base.LEDGER_REL = (
    f"plots/k2_batch/master_vetted_catalog/manual_review_updates/{RUN_ID}/"
    f"manual_vetting_decisions_cumulative_{RUN_ID}.csv"
)

base.DECISIONS = [
    {
        "epic_id": "EPIC_211770390",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_oot_detrending_and_stage_f_period_validation",
        "manual_confidence": "medium_low",
        "manual_reason": (
            "Some signal-like structure may be present and the basic duration is plausible for P=7.46d, with no "
            "detected secondary eclipse and high CNN morphology score. However the signal is tough to read visually, "
            "primary SNR is modest around 5.36, odd/even consistency is poor with ratio about 0.562 and explicit "
            "delta about 0.702, and OOT variability is high at about 1.26x the transit depth. Period source is "
            "event-spacing fallback with many candidate periods, so keep only as uncertain hold pending OOT "
            "detrending review and Stage F period validation rather than promote."
        ),
        "cnn_manual_conflict": "cnn_high_manual_hold",
        "cnn_manual_conflict_reason": (
            "CNN score is high at about 0.936, but manual review keeps it on hold because the signal is visually "
            "unclear with poor odd/even consistency, high OOT variability, and fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211781094",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Some signal-like features may be present, but the detection is not credible as a planet-like candidate. "
            "The measured primary depth is extremely large at about 0.217, implying radius ratio about 0.466, while "
            "primary SNR is weak around 3.34 and event support is limited. Odd/even consistency is acceptable and "
            "there is no detected secondary eclipse, but the period source is event-spacing fallback with many "
            "candidate periods and the signal appears unreliable. Reject as noise/artifact or bad folded-event fit "
            "rather than promote or hold."
        ),
        "cnn_manual_conflict": "cnn_high_manual_reject",
        "cnn_manual_conflict_reason": (
            "CNN score is high at about 0.917, but manual review rejects because the signal is not visually or "
            "physically credible as planet-like and appears dominated by bad folded-event geometry or artifact behavior."
        ),
    },
    {
        "epic_id": "EPIC_211800810",
        "manual_label": "binary_system",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "high",
        "manual_reason": (
            "Strong odd/even depth mismatch with ratio about 0.219 and explicit delta about 1.19, making the signal "
            "EB-like or unreliable rather than planet-like. The signal is also very shallow, with an unusually long "
            "apparent duration of about 7.11 hours for P=2.80d. Period source is event-spacing fallback with limited "
            "event support and many candidate periods. Reject from candidate pipeline despite no detected secondary "
            "eclipse and morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and does not materially conflict with manual rejection.",
    },
    {
        "epic_id": "EPIC_211805860",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_oot_detrending_and_stage_f_period_validation",
        "manual_confidence": "medium",
        "manual_reason": (
            "Manual review keeps this as a cautious hold despite Stage F rejection. Odd/even consistency is good with "
            "ratio about 0.957 and explicit delta about 0.044, there is no detected secondary eclipse, the duration "
            "is plausible at about 3.43 hours for P=4.14d, and the period source is saved_validation_period. However "
            "primary SNR is weak around 4.67, Stage F rejected because the primary transit depth is not significant "
            "in the cached folded light curve, and OOT variability is very high at about 1.48x transit depth. Keep "
            "as uncertain hold pending OOT detrending review and Stage F period/depth validation rather than promote."
        ),
        "cnn_manual_conflict": "cnn_high_manual_hold",
        "cnn_manual_conflict_reason": (
            "CNN score is very high at about 0.980, but manual review keeps it on hold due to weak primary "
            "significance, Stage F rejection, and high OOT variability."
        ),
    },
    {
        "epic_id": "EPIC_211844348",
        "manual_label": "noise_or_artifact",
        "manual_next_action": "reject_from_candidate_pipeline",
        "manual_confidence": "medium_high",
        "manual_reason": (
            "Manual review favors noise/artifact despite superficially acceptable diagnostics. The signal is very "
            "shallow, and the apparent duration is fairly long at about 5.88 hours. Odd/even consistency is good "
            "with ratio about 0.949 and no detected secondary eclipse, but the period source is event-spacing fallback "
            "with many candidate periods, and the visual signal is not convincing enough to justify holding. Reject "
            "as noise/artifact despite morphology-positive CNN score."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": (
            "CNN score is moderate-high but not extreme; manual rejection is driven by poor visual credibility and "
            "fallback period support."
        ),
    },
    {
        "epic_id": "EPIC_211851580",
        "manual_label": "uncertain_hold",
        "manual_next_action": "keep_hold_for_recheck_with_stage_f_period_validation_and_secondary_window_inspection",
        "manual_confidence": "medium",
        "manual_reason": (
            "Transit-like signal is present with acceptable primary SNR around 8.50, good odd/even consistency with "
            "ratio about 0.914 and explicit delta about 0.094, no numerically detected secondary eclipse, and OOT "
            "variability is moderate at about 0.27x transit depth. However the secondary-check panel appears visually "
            "odd despite zero/null secondary metrics, the apparent transit duration is long at about 8.09 hours, and "
            "the period source is event-spacing fallback with many candidate periods. Keep as uncertain hold pending "
            "Stage F period validation and manual inspection of the secondary window rather than promote directly."
        ),
        "cnn_manual_conflict": "none",
        "cnn_manual_conflict_reason": "CNN score is moderate and consistent with cautious hold rather than promotion.",
    },
]


if __name__ == "__main__":
    base.main()
