from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "gatevetter_v0_2_deep_review_shortlist.csv"
DECISIONS = ROOT / "gatevetter_v0_2_deep_review_manual_decisions.csv"
MASTER = ROOT / "k2_master_candidate_ledger.csv"
REPORT = ROOT / "gatevetter_v0_2_deep_review_reconciliation_report.csv"
SUMMARY = ROOT / "gatevetter_v0_2_deep_review_provisional_scientific_summary.txt"

REVIEW_DATE = "2026-07-31"
REVIEW_SOURCE = "user_supplied_gatevetter_v0_2_deep_review_update"

DECISION_DATA = {
    "EPIC_211624954": ("uncertain_hold_positive", "Best surviving positive hold, but period remains untrusted and 2P shows a possible near-equal secondary concern. Do not promote."),
    "EPIC_211687388": ("false_positive_eb_or_variable", "Competing-period review exposes EB/variable evidence."),
    "EPIC_211768304": ("false_positive_eb_or_variable", "At 2P, coherent near-equal secondary structure, odd/even concern, elevated OOT variability, and untrusted period indicate a weak EB or stellar false positive."),
    "EPIC_211959909": ("false_positive_eb_or_variable", "2P comparison exposes a strong secondary/EB interpretation."),
    "EPIC_211996306": ("reject_as_noise_or_artifact", "Weak and unreliable event evidence with excessive surrounding variability."),
    "EPIC_211351798": ("false_positive_eb_or_variable", "2P reveals coherent near-equal primary and secondary eclipses."),
    "EPIC_211912465": ("reject_as_noise_or_artifact", "Untrusted period, inconsistent P/2-P-2P behaviour, implausible competing-event selection, and insufficient reproducible transit evidence."),
    "EPIC_211485867": ("false_positive_eb_or_variable", "At 2P, primary and secondary depths are near equal and coherently repeated, consistent with an eclipsing binary folded at half-period."),
    "EPIC_211387236": ("false_positive_eb_or_variable", "Approximately 16.96-day solution reveals strong, coherent, near-equal primary and secondary eclipses."),
    "EPIC_211889082": ("false_positive_eb_or_variable", "Broad coherent stellar modulation, unresolved period, odd/even concern, and near-equal structure at 2P. Likely variable/contact binary."),
    "EPIC_211816343": ("false_positive_eb_or_variable", "Broad periodic modulation, very high OOT/depth, and near-equal structures at 2P. Not transit-like."),
    "EPIC_212029934": ("false_positive_eb_or_variable", "Coherent broad stellar modulation with near-equal primary and secondary features at 2P. Likely contact binary or ellipsoidal variable."),
    "EPIC_211431812": ("reject_as_noise_or_artifact", "Weak period-inconsistent signal, elevated OOT variability, unstable event depths, and no robust EB or planetary pattern."),
    "EPIC_211696209": ("false_positive_eb_or_variable", "Nominal-period SNR is numerically suspicious. At 2P there is a stronger secondary, severe odd/even mismatch, high OOT/depth, and coherent stellar/alias evidence."),
    # Earlier completed batch-10 review, normalized to the deep-review label taxonomy.
    "EPIC_211845034": ("false_positive_eb_or_variable", "Deep broad EB/variable-like morphology with strong periodic raw variability and non-planet-like event structure."),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def provenance(epic_id: str) -> tuple[str, str, str]:
    if epic_id == "EPIC_211845034":
        return (
            "2026-06-23",
            "codex_visual_review; gatevetter_v0_2_batch_10_manual_review_decisions.csv",
            "completed_manual_review_found_locally; manual_label=variable_or_possible_eb; normalized_to=false_positive_eb_or_variable",
        )
    return REVIEW_DATE, REVIEW_SOURCE, "manual_decision_supplied_in_update"


def main() -> None:
    shortlist_fields, shortlist_rows = read_csv(SHORTLIST)
    ids = [row["epic_id"] for row in shortlist_rows]
    if len(ids) != 15 or len(set(ids)) != 15:
        raise RuntimeError(f"Expected 15 unique shortlist EPICs; found {len(ids)} rows/{len(set(ids))} unique")
    if set(ids) != set(DECISION_DATA):
        raise RuntimeError("Decision data and shortlist EPIC sets differ")

    manual_fields = ["final_label", "manual_reason", "review_date", "review_source", "review_status"]
    for row in shortlist_rows:
        epic_id = row["epic_id"]
        label, reason = DECISION_DATA[epic_id]
        date, source, _ = provenance(epic_id)
        row.update(final_label=label, manual_reason=reason, review_date=date, review_source=source, review_status="complete")
    write_csv(SHORTLIST, shortlist_fields + [f for f in manual_fields if f not in shortlist_fields], shortlist_rows)

    decision_fields = shortlist_fields + [f for f in manual_fields if f not in shortlist_fields]
    report_rows = []
    decision_rows = []
    for row in shortlist_rows:
        epic_id = row["epic_id"]
        _, _, source_status = provenance(epic_id)
        # These targets were absent from the deep-review decision ledger and master
        # ledger at the start of this reconciliation. Keep that audit result stable
        # across later idempotent reruns.
        report_rows.append({"epic_id": epic_id, "reconciliation_status": "inserted", "source_status": source_status, "final_label": row["final_label"]})
        decision_rows.append(row)
    write_csv(DECISIONS, decision_fields, decision_rows)

    master_fields, master_rows = read_csv(MASTER)
    added_master_fields = [
        "gatevetter_v0_2_automated_prediction", "candidate_survivor_score", "trusted_period",
        "period_decision", "odd_even_depth_ratio", "oot_to_depth", "secondary_depth_snr",
        "secondary_to_primary_depth_ratio", "event_stack_coherence", "event_stack_coherence_score",
        "local_baseline_stability", "manual_final_label", "manual_reason", "review_source",
        "deep_review_panel_path",
    ]
    master_fields = master_fields + [f for f in added_master_fields if f not in master_fields]
    target_ids = set(ids)
    retained = [row for row in master_rows if row.get("epic_id") not in target_ids]
    for source in shortlist_rows:
        retained.append({
            "epic_id": source["epic_id"],
            "source_batch": f"gatevetter_v0_2_batch_{source['source_batch']}_deep_review",
            "best_period_days": source["validation_period_days"],
            "stage_f_label": source["review_category"],
            "visual_label": source["final_label"],
            "final_candidate_status": source["final_label"],
            "review_bin": "gatevetter_v0_2_deep_review",
            "visual_notes": source["manual_reason"],
            "reviewer": source["review_source"],
            "reviewed_at": source["review_date"],
            "phase_0_folded_path": source["one_page_visual_panel"],
            "validation_summary_json_path": source["validation_summary_json"],
            "gatevetter_v0_2_automated_prediction": source["review_category"],
            "candidate_survivor_score": source["candidate_survivor_score"],
            "trusted_period": source["trusted_period"],
            "period_decision": source["period_decision"],
            "odd_even_depth_ratio": source["odd_even_depth_ratio"],
            "oot_to_depth": source["oot_to_depth"],
            "secondary_depth_snr": source["secondary_depth_snr"],
            "secondary_to_primary_depth_ratio": source["secondary_to_primary_depth_ratio"],
            "event_stack_coherence": source["event_stack_coherence"],
            "event_stack_coherence_score": source["event_stack_coherence_score"],
            "local_baseline_stability": source["local_baseline_stability"],
            "manual_final_label": source["final_label"],
            "manual_reason": source["manual_reason"],
            "review_source": source["review_source"],
            "deep_review_panel_path": source["one_page_visual_panel"],
        })
    write_csv(MASTER, master_fields, retained)
    write_csv(REPORT, ["epic_id", "reconciliation_status", "source_status", "final_label"], report_rows)

    counts = Counter(row["final_label"] for row in shortlist_rows)
    summary = f"""GateVetter v0.2 deep-review provisional scientific summary
review_status=complete_15_of_15
unique_shortlist_targets=15
unique_reviewed_targets=15
uncertain_hold_positive={counts['uncertain_hold_positive']}
false_positive_eb_or_variable={counts['false_positive_eb_or_variable']}
reject_as_noise_or_artifact={counts['reject_as_noise_or_artifact']}
candidate_like={counts['candidate_like']}
pending_manual_review={counts['pending_manual_review']}
promoted_candidates=0

EPIC_211845034_status=completed_manual_review_found_locally
EPIC_211845034_original_review_date=2026-06-23
EPIC_211845034_original_review_source=gatevetter_v0_2_batch_10_manual_review_decisions.csv; codex_visual_review
EPIC_211845034_original_manual_label=variable_or_possible_eb
EPIC_211845034_reconciled_final_label=false_positive_eb_or_variable
EPIC_211845034_deep_review_panel=plots/k2_batch/gatevetter_v0_2_deep_review/EPIC_211845034/deep_review_panel.png
EPIC_211845034_validation_summary=plots/k2_batch/gatevetter_v0_2_deep_review/EPIC_211845034/validation_summary.json

scientific_interpretation=The shortlist yields one unpromoted uncertain positive hold, eleven EB/variable false positives, and three noise/artifact rejects. No candidate-like target is promoted.
batching_status=paused_no_new_batches_run
gatevetter_v0_2_status=unchanged_thresholds_unchanged
cnn_status=unchanged_not_retrained
gatevetter_v0_3_status=not_implemented
diagnostic_sources=unchanged
"""
    SUMMARY.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
