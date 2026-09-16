from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "phase2"
DATA = ROOT / "data" / "phase2"

CONFLICTS = DOCS / "phase2_expanded_label_conflicts.csv"
EXPANDED = DATA / "phase2_expanded_label_table.parquet"
CATALOGUE_EVIDENCE = DATA / "phase2_expanded_catalogue_evidence.parquet"
INVENTORY = DOCS / "phase2_label_inventory.csv"
FEATURES = DATA / "phase2_feature_table.parquet"
MANUAL_LEDGER = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "manual_vetting_decisions_ledger.csv"
FINAL_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"

HOST_PACKET = DOCS / "phase2_conflict_review_packet.csv"
OBJECT_PACKET = DOCS / "phase2_conflict_review_object_evidence.csv"
ADJUDICATION_TEMPLATE = DOCS / "phase2_conflict_adjudication_template.csv"
SUMMARY = DOCS / "phase2_conflict_review_packet_summary.json"
AUDIT = DOCS / "PHASE2_CONFLICT_REVIEW_PACKET_AUDIT.md"

ALLOWED_DECISIONS = [
    "confirmed_planet",
    "candidate_like",
    "false_positive_eb_or_variable",
    "reject_as_noise_or_artifact",
    "retain_cross_class_conflict",
]


def clean(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def as_bool(value: object) -> bool:
    return clean(value).lower() in {"true", "1", "yes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_manual_evidence(epic_id: str, manual: pd.DataFrame, final: pd.DataFrame) -> dict[str, str]:
    direct = manual.loc[manual["epic_id"].eq(epic_id)].copy()
    if len(direct):
        direct = direct.sort_values("reviewed_at", kind="stable")
        row = direct.iloc[-1]
        return {
            "manual_review_label": clean(row.get("manual_label")),
            "manual_review_reason": clean(row.get("manual_reason")),
            "manual_review_reviewer": clean(row.get("reviewer")),
            "manual_review_date": clean(row.get("reviewed_at")),
            "manual_review_source": "plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv",
        }
    ledger = final.loc[final["epic_id"].eq(epic_id)].copy()
    if len(ledger):
        ledger = ledger.sort_values("reviewed_at", kind="stable")
        row = ledger.iloc[-1]
        return {
            "manual_review_label": clean(row.get("final_candidate_status")),
            "manual_review_reason": clean(row.get("status_reason")) or clean(row.get("visual_notes")),
            "manual_review_reviewer": clean(row.get("reviewer")),
            "manual_review_date": clean(row.get("reviewed_at")),
            "manual_review_source": "plots/k2_batch/final_candidate_master_ledger.csv",
        }
    return {
        "manual_review_label": "",
        "manual_review_reason": "",
        "manual_review_reviewer": "",
        "manual_review_date": "",
        "manual_review_source": "",
    }


def conflict_category(row: pd.Series) -> tuple[str, str]:
    internal = clean(row.get("internal_normalized_class"))
    archive = clean(row.get("archive_normalized_class"))
    if as_bool(row.get("archive_cross_class_conflict")):
        return (
            "archive_object_mixed_candidate_false_positive",
            "Review object-level dispositions separately; a mixed-disposition multi-object host cannot receive one physical host class without an explicit policy decision.",
        )
    if archive == "candidate_like" and internal == "false_positive_eb_or_variable":
        return (
            "internal_eb_variable_vs_archive_candidate",
            "Compare the manual EB/variable morphology rationale with the archive candidate reference; decide whether archive candidate evidence supersedes, coexists with, or remains quarantined against the internal negative.",
        )
    if archive == "candidate_like" and internal == "reject_as_noise_or_artifact":
        return (
            "internal_noise_artifact_vs_archive_candidate",
            "Compare the manual noise/systematics rationale with the archive candidate evidence; do not infer a negative or positive from either source without adjudication.",
        )
    if archive == "false_positive_eb_or_variable" and internal == "reject_as_noise_or_artifact":
        return (
            "internal_noise_artifact_vs_archive_false_positive",
            "Both sources are negative at broad polarity but disagree on the physical subclass; choose a class only if the evidence supports that distinction, otherwise retain quarantine.",
        )
    return (
        "other_cross_class_conflict",
        "Review the retained internal and catalogue evidence and either assign one supported physical class or retain the conflict quarantine.",
    )


def build_packet() -> dict[str, object]:
    conflicts = pd.read_csv(CONFLICTS, dtype=str, keep_default_na=False)
    expanded = pd.read_parquet(EXPANDED)
    catalogue = pd.read_parquet(CATALOGUE_EVIDENCE)
    inventory = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False)
    features = pd.read_parquet(FEATURES)
    manual = pd.read_csv(MANUAL_LEDGER, dtype=str, keep_default_na=False)
    final = pd.read_csv(FINAL_LEDGER, dtype=str, keep_default_na=False)

    if len(conflicts) != 11 or conflicts["epic_id"].nunique() != 11:
        raise AssertionError("Expected exactly 11 unique unresolved conflict hosts")
    ids = set(conflicts["epic_id"])
    expanded_conflicts = expanded.loc[expanded["epic_id"].isin(ids)].copy()
    if len(expanded_conflicts) != 11 or expanded_conflicts["physical_loss_eligible"].map(as_bool).any():
        raise AssertionError("Conflict hosts must remain one-per-EPIC and loss-ineligible")

    inv = inventory.set_index("epic_id", drop=False)
    feat = features.set_index("epic_id", drop=False)
    host = expanded_conflicts.set_index("epic_id", drop=False)
    rows: list[dict[str, object]] = []
    for review_order, conflict in enumerate(conflicts.sort_values("epic_id", kind="stable").itertuples(index=False), start=1):
        epic_id = conflict.epic_id
        h = host.loc[epic_id]
        internal = inv.loc[epic_id] if epic_id in inv.index else pd.Series(dtype=object)
        feature = feat.loc[epic_id] if epic_id in feat.index else pd.Series(dtype=object)
        category, focus = conflict_category(pd.Series(conflict._asdict()))
        manual_evidence = latest_manual_evidence(epic_id, manual, final)
        rows.append({
            "review_order": review_order,
            "epic_id": epic_id,
            "adjudication_status": "unresolved",
            "current_corrected_physical_class": "cross_class_conflict",
            "current_physical_loss_eligible": False,
            "conflict_category": category,
            "review_focus": focus,
            "internal_normalized_class": clean(h.get("internal_normalized_class")),
            "internal_original_label": clean(internal.get("original_manual_label")),
            **manual_evidence,
            "internal_label_source": clean(internal.get("label_source")),
            "internal_all_label_evidence": clean(internal.get("all_label_evidence")),
            "archive_normalized_class": clean(h.get("archive_normalized_class")),
            "archive_evidence_kinds_json": clean(h.get("archive_evidence_kinds_json")),
            "archive_dispositions_json": clean(h.get("archive_dispositions_json")),
            "archive_references_json": clean(h.get("archive_references_json")),
            "catalogue_object_ids_json": clean(h.get("catalogue_object_ids_json")),
            "campaigns_json": clean(h.get("campaigns_json")),
            "archive_periods_json": clean(h.get("archive_periods_json")),
            "catalogue_object_row_count": int(h.get("catalogue_object_row_count", 0)),
            "catalogue_unique_object_count": int(h.get("catalogue_unique_object_count", 0)),
            "catalogue_default_solution_count": int(h.get("catalogue_default_solution_count", 0)),
            "catalogue_non_default_solution_count": int(h.get("catalogue_non_default_solution_count", 0)),
            "archive_cross_class_conflict": as_bool(h.get("archive_cross_class_conflict")),
            "internal_archive_class_conflict": as_bool(h.get("internal_archive_class_conflict")),
            "accessible_local_light_curve": as_bool(h.get("accessible_local_light_curve")),
            "accessible_512_sample_tensor": as_bool(h.get("accessible_512_sample_tensor")),
            "existing_cnn_probability": as_bool(h.get("existing_cnn_probability")),
            "existing_128d_embedding": as_bool(h.get("existing_128d_embedding")),
            "nominal_numerical_diagnostics": as_bool(h.get("nominal_numerical_diagnostics")),
            "p_half_p_2p_diagnostics": as_bool(h.get("p_half_p_2p_diagnostics")),
            "trusted_period": as_bool(h.get("trusted_period")),
            "cnn_probability": feature.get("cnn_probability", None),
            "validation_period_days": feature.get("validation_period_days", None),
            "primary_depth": feature.get("primary_depth", None),
            "primary_depth_snr": feature.get("primary_depth_snr", None),
            "odd_even_depth_ratio": feature.get("odd_even_depth_ratio", None),
            "secondary_to_primary_depth_ratio": feature.get("secondary_to_primary_depth_ratio", None),
            "oot_to_depth": feature.get("oot_to_depth", None),
            "permitted_adjudication_values": "|".join(ALLOWED_DECISIONS),
            "reviewer_decision": "",
            "reviewer_reason": "",
            "reviewer": "",
            "adjudicated_at_utc": "",
        })
    packet = pd.DataFrame(rows)

    object_columns = [
        "epic_id", "catalogue_row_id", "object_id", "candidate_suffix", "k2_name",
        "default_solution", "raw_disposition", "raw_disposition_reference", "evidence_kind",
        "normalized_label_proposal", "raw_campaigns", "campaigns_json", "period_days",
        "period_source", "period_provenance", "period_trusted", "physical_loss_eligible_proposal",
        "eligibility_exclusion_reason", "cross_class_conflict", "source_version",
        "source_retrieved_at_utc", "source_sha256", "source_record_locator", "source_citation",
    ]
    objects = catalogue.loc[catalogue["epic_id"].isin(ids), object_columns].sort_values(
        ["epic_id", "object_id", "default_solution", "catalogue_row_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    if len(objects) != 26 or set(objects["epic_id"]) != ids:
        raise AssertionError("Expected all 26 catalogue object/solution rows across the 11 conflicts")

    template = packet[["review_order", "epic_id", "conflict_category", "current_corrected_physical_class", "current_physical_loss_eligible", "permitted_adjudication_values", "reviewer_decision", "reviewer_reason", "reviewer", "adjudicated_at_utc"]].copy()
    packet.to_csv(HOST_PACKET, index=False)
    objects.to_csv(OBJECT_PACKET, index=False)
    template.to_csv(ADJUDICATION_TEMPLATE, index=False)

    category_counts = packet["conflict_category"].value_counts().sort_index().to_dict()
    coverage_columns = [
        "accessible_local_light_curve", "accessible_512_sample_tensor", "existing_cnn_probability",
        "existing_128d_embedding", "nominal_numerical_diagnostics", "p_half_p_2p_diagnostics", "trusted_period",
    ]
    coverage_counts = {column: int(packet[column].sum()) for column in coverage_columns}
    summary = {
        "schema_version": "phase2_conflict_review_packet_v1.0.0",
        "conflict_host_count": len(packet),
        "catalogue_object_solution_row_count": len(objects),
        "unresolved_host_count": int(packet["adjudication_status"].eq("unresolved").sum()),
        "physical_loss_eligible_host_count": int(packet["current_physical_loss_eligible"].sum()),
        "conflict_category_counts": category_counts,
        "existing_feature_coverage": coverage_counts,
        "allowed_decisions": ALLOWED_DECISIONS,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in [CONFLICTS, EXPANDED, CATALOGUE_EVIDENCE, INVENTORY, FEATURES, MANUAL_LEDGER, FINAL_LEDGER]
        },
        "non_actions": [
            "no_adjudication_applied", "no_label_change", "no_loss_eligibility_change",
            "no_training", "no_split_assignment", "no_cnn_operation", "no_feature_generation",
            "no_gatevetter_verdict_used_as_feature",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    category_lines = "\n".join(f"| `{key}` | {value} |" for key, value in category_counts.items())
    audit = f"""# Phase 2 Conflict Review Packet Audit

## Outcome

- Unresolved conflict hosts retained: **{len(packet)}**.
- Underlying catalogue object/solution rows retained: **{len(objects)}**.
- Physical-loss-eligible conflict hosts: **0**.
- Reviewer decisions populated: **0**.

| Conflict category | Hosts |
| --- | ---: |
{category_lines}

## Existing-only feature context

| Availability | Hosts |
| --- | ---: |
| Cached local light curve | {coverage_counts['accessible_local_light_curve']} |
| 512-sample tensor | {coverage_counts['accessible_512_sample_tensor']} |
| Existing CNN probability | {coverage_counts['existing_cnn_probability']} |
| Existing 128-D embedding | {coverage_counts['existing_128d_embedding']} |
| Nominal numerical diagnostics | {coverage_counts['nominal_numerical_diagnostics']} |
| P/2-P-2P diagnostics | {coverage_counts['p_half_p_2p_diagnostics']} |
| Trusted period | {coverage_counts['trusted_period']} |

The packet includes existing numerical measurements only. Archive dispositions, manual labels/reasons, conflict categories, review fields, and provenance are target/review metadata and are not scientific features. GateVetter decisions, actions, recommendations, and verdict-derived fields are not included as feature inputs.

## Read-only guarantee

This build did not alter `phase2_expanded_label_table.parquet`, the internal label inventory, positive tiers, corrections, baseline supersessions, feature table, or any loss-eligibility field. Every conflict remains `cross_class_conflict`, unresolved, and excluded from physical-class loss. No training, split assignment, CNN operation, feature generation, or catalogue download occurred.

The feature-generation manifest is intentionally blocked until a completed adjudication file supplies one permitted decision, reason, reviewer, and UTC timestamp for each of the 11 EPICs.
"""
    AUDIT.write_text(audit, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the read-only Phase 2 11-conflict review packet.")
    parser.add_argument("--audit-only", action="store_true", help="Required; this script cannot apply adjudications or generate features.")
    args = parser.parse_args()
    if not args.audit_only:
        parser.error("--audit-only is required")
    return args


def main() -> None:
    parse_args()
    print(json.dumps(build_packet(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
