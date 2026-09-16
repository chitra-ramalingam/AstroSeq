from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase2"
DOCS = ROOT / "docs" / "phase2"

DEFAULT_CATALOGUE = DATA / "catalogues" / "phase2_k2_catalogue_rows.parquet"
DEFAULT_INVENTORY = DOCS / "phase2_label_inventory.csv"
DEFAULT_TIERS = DOCS / "phase2_positive_training_tiers.csv"
DEFAULT_CORRECTIONS = DOCS / "phase2_positive_tier_corrections.csv"
DEFAULT_FEATURES = DATA / "phase2_feature_table.parquet"
DEFAULT_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"
DEFAULT_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"

PROTECTED_METADATA = {
    "archive_disposition", "catalogue_class", "catalogue_reference",
    "positive_evidence_tier", "training_role", "manual_labels", "manual_reasons",
    "correction_basis", "gatevetter_decision", "gatevetter_action",
    "gatevetter_recommendation", "normalized_target_label", "physical_loss_eligible",
}


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def as_bool(value: object) -> bool:
    return clean(value).lower() in {"true", "1", "yes"}


def canonical_epic(value: object) -> str:
    match = re.search(r"(\d{8,10})", clean(value))
    return f"EPIC_{match.group(1)}" if match else ""


def json_values(values: Iterable[object]) -> str:
    return json.dumps(sorted({clean(value) for value in values if clean(value)}), separators=(",", ":"))


def parse_campaigns(value: object) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return [clean(v) for v in parsed if clean(v)]
    except (TypeError, ValueError, json.JSONDecodeError):
        return re.findall(r"-?\d+", text)


def aggregate_catalogue(catalogue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain every object/solution row, then aggregate valid rows to one EPIC host."""
    evidence = catalogue.copy()
    evidence["epic_id"] = evidence["epic_id"].map(canonical_epic)
    evidence.insert(0, "evidence_record_type", "catalogue_object_solution")
    evidence["object_row_retained_before_host_aggregation"] = True

    valid = evidence[evidence["epic_id"].ne("")].copy()
    host_rows: list[dict[str, object]] = []
    for epic_id, group in valid.groupby("epic_id", sort=True):
        kinds = set(group["evidence_kind"].map(clean))
        has_confirmed = "archive_confirmed" in kinds
        has_candidate = "archive_candidate" in kinds
        has_false_positive = "archive_false_positive" in kinds
        has_refuted = "archive_refuted" in kinds
        archive_conflict = bool(group["cross_class_conflict"].map(as_bool).any())
        if archive_conflict:
            archive_class = ""
        elif has_confirmed:
            archive_class = "confirmed_planet"
        elif has_candidate:
            archive_class = "candidate_like"
        elif has_false_positive or has_refuted:
            archive_class = "false_positive_eb_or_variable"
        else:
            archive_class = ""

        campaigns = sorted({c for value in group["campaigns_json"] for c in parse_campaigns(value)}, key=lambda x: (int(x) if x.lstrip("-").isdigit() else 999, x))
        periods = pd.to_numeric(group["period_days"], errors="coerce")
        trusted = group["period_trusted"].map(as_bool) & periods.notna()
        solutions = []
        for row in group[["catalogue_row_id", "object_id", "candidate_suffix", "default_solution", "raw_disposition", "raw_disposition_reference", "period_days", "period_source"]].to_dict("records"):
            solutions.append({key: (None if pd.isna(value) else value) for key, value in row.items()})
        host_rows.append({
            "epic_id": epic_id,
            "catalogue_present": True,
            "catalogue_object_row_count": int(len(group)),
            "catalogue_unique_object_count": int(group["object_id"].map(clean).replace("", np.nan).nunique()),
            "catalogue_multi_planet_or_object_system": bool(group["object_id"].map(clean).replace("", np.nan).nunique() > 1),
            "catalogue_object_ids_json": json_values(group["object_id"]),
            "catalogue_candidate_suffixes_json": json_values(group["candidate_suffix"]),
            "catalogue_default_solution_count": int(group["default_solution"].map(as_bool).sum()),
            "catalogue_non_default_solution_count": int((~group["default_solution"].map(as_bool)).sum()),
            "catalogue_solution_provenance_json": json.dumps(solutions, sort_keys=True, separators=(",", ":")),
            "archive_confirmed_evidence": has_confirmed,
            "archive_candidate_evidence": has_candidate,
            "archive_false_positive_evidence": has_false_positive,
            "archive_refuted_evidence": has_refuted,
            "archive_dispositions_json": json_values(group["raw_disposition"]),
            "archive_references_json": json_values(group["raw_disposition_reference"]),
            "archive_evidence_kinds_json": json_values(group["evidence_kind"]),
            "archive_normalized_class": archive_class,
            "archive_cross_class_conflict": archive_conflict,
            "archive_physical_loss_eligible": bool(not archive_conflict and group["physical_loss_eligible_proposal"].map(as_bool).any()),
            "archive_campaigns_json": json.dumps(campaigns, separators=(",", ":")),
            "archive_periods_json": json.dumps(sorted({float(v) for v in periods.dropna()}), separators=(",", ":")),
            "archive_trusted_period_available": bool(trusted.any()),
            "archive_fallback_period_available": bool(periods.notna().any() and not trusted.any()),
        })
    return evidence, pd.DataFrame(host_rows)


def correction_summary(corrections: pd.DataFrame) -> pd.DataFrame:
    if corrections.empty:
        return pd.DataFrame(columns=["epic_id", "correction_record_count", "correction_history_json", "latest_correction_basis"])
    rows = []
    for epic_id, group in corrections.groupby("epic_id", sort=True):
        records = group.to_dict("records")
        rows.append({
            "epic_id": canonical_epic(epic_id),
            "correction_record_count": len(group),
            "correction_history_json": json.dumps(records, sort_keys=True, separators=(",", ":")),
            "latest_correction_basis": clean(group.iloc[-1].get("correction_basis")),
        })
    return pd.DataFrame(rows)


def classes_conflict(internal_class: str, archive_class: str) -> bool:
    if not internal_class or not archive_class or internal_class == "uncertain_hold":
        return False
    if internal_class == archive_class:
        return False
    return not ({internal_class, archive_class} <= {"candidate_like", "confirmed_planet"})


def choose_class(internal_class: str, archive_class: str, conflict: bool) -> str:
    if conflict:
        return "cross_class_conflict"
    if archive_class == "confirmed_planet":
        return archive_class
    if archive_class:
        return archive_class
    return internal_class or "unlabelled"


def build_expanded_hosts(
    inventory: pd.DataFrame,
    tiers: pd.DataFrame,
    corrections: pd.DataFrame,
    catalogue_hosts: pd.DataFrame,
) -> pd.DataFrame:
    internal = inventory.copy()
    internal["epic_id"] = internal["epic_id"].map(canonical_epic)
    internal = internal[internal["epic_id"].ne("")].drop_duplicates("epic_id", keep="last")
    internal = internal.rename(columns={
        "current_final_label": "internal_normalized_class",
        "physical_loss_eligible": "internal_physical_loss_eligible",
        "label_source": "internal_label_source",
        "training_role": "internal_training_role",
        "positive_evidence_tier": "internal_positive_evidence_tier",
    })
    internal["internal_present"] = True

    tier_cols = ["epic_id", "period_feature_trust"]
    available_tier_cols = [column for column in tier_cols if column in tiers]
    tier_frame = tiers[available_tier_cols].copy()
    tier_frame["epic_id"] = tier_frame["epic_id"].map(canonical_epic)
    internal = internal.merge(tier_frame.drop_duplicates("epic_id", keep="last"), on="epic_id", how="left", validate="one_to_one")

    all_epics = sorted(set(internal["epic_id"]) | set(catalogue_hosts["epic_id"]))
    expanded = pd.DataFrame({"epic_id": all_epics})
    expanded = expanded.merge(internal, on="epic_id", how="left", validate="one_to_one")
    expanded = expanded.merge(catalogue_hosts, on="epic_id", how="left", validate="one_to_one")
    expanded = expanded.merge(correction_summary(corrections), on="epic_id", how="left", validate="one_to_one")

    expanded["internal_present"] = expanded["internal_present"].map(as_bool)
    expanded["catalogue_present"] = expanded["catalogue_present"].map(as_bool)
    for column in ["archive_confirmed_evidence", "archive_candidate_evidence", "archive_false_positive_evidence", "archive_refuted_evidence", "archive_cross_class_conflict", "archive_physical_loss_eligible", "archive_trusted_period_available", "archive_fallback_period_available"]:
        expanded[column] = expanded[column].map(as_bool)
    expanded["internal_normalized_class"] = expanded["internal_normalized_class"].map(clean)
    expanded["archive_normalized_class"] = expanded["archive_normalized_class"].map(clean)
    expanded["internal_archive_class_conflict"] = expanded.apply(
        lambda row: classes_conflict(row["internal_normalized_class"], row["archive_normalized_class"]), axis=1
    )
    expanded["cross_class_conflict"] = expanded["archive_cross_class_conflict"] | expanded["internal_archive_class_conflict"]
    expanded["corrected_physical_class"] = expanded.apply(
        lambda row: choose_class(row["internal_normalized_class"], row["archive_normalized_class"], bool(row["cross_class_conflict"])), axis=1
    )
    internal_eligible = expanded["internal_physical_loss_eligible"].map(as_bool)
    expanded["physical_loss_eligible"] = np.where(
        expanded["cross_class_conflict"], False,
        np.where(expanded["catalogue_present"] & expanded["archive_normalized_class"].ne(""), expanded["archive_physical_loss_eligible"], internal_eligible),
    ).astype(bool)
    expanded["conflict_resolution"] = np.where(
        expanded["cross_class_conflict"], "quarantined_excluded_from_physical_class_loss",
        np.where(expanded["catalogue_present"], "archive_evidence_applied_with_internal_provenance_retained", "internal_status_retained_no_catalogue_match"),
    )
    expanded["catalogue_dispositions_are_scientific_features"] = False
    expanded["split_assignment"] = ""
    expanded["split_status"] = "not_assigned"

    def campaigns(row: pd.Series) -> str:
        values = set(parse_campaigns(row.get("archive_campaigns_json")))
        internal_campaign = clean(row.get("k2_campaign"))
        if internal_campaign:
            values.add(internal_campaign)
        ordered = sorted(values, key=lambda x: (int(x) if x.lstrip("-").isdigit() else 999, x))
        return json.dumps(ordered, separators=(",", ":"))

    expanded["campaigns_json"] = expanded.apply(campaigns, axis=1)
    expanded["catalogue_only_host"] = expanded["catalogue_present"] & ~expanded["internal_present"]
    expanded["internal_only_host"] = expanded["internal_present"] & ~expanded["catalogue_present"]
    return expanded.sort_values("epic_id").reset_index(drop=True)


def cached_light_curve_epics(cache_root: Path) -> set[str]:
    if not cache_root.exists():
        return set()
    epics: set[str] = set()
    try:
        paths = cache_root.rglob("*.fits")
        for path in paths:
            epic = canonical_epic(path.as_posix())
            if epic:
                epics.add(epic)
    except (OSError, PermissionError):
        return epics
    return epics


def tensor_epics(meta_path: Path, x_path: Path) -> set[str]:
    if not meta_path.exists() or not x_path.exists():
        return set()
    shape = np.load(x_path, mmap_mode="r").shape
    if len(shape) < 2 or shape[1] != 512:
        raise ValueError(f"Expected 512-sample inference tensors, found shape={shape}")
    meta = pd.read_parquet(meta_path, columns=["star_id"])
    return {canonical_epic(value) for value in meta["star_id"] if canonical_epic(value)}


def attach_existing_feature_coverage(
    expanded: pd.DataFrame,
    features: pd.DataFrame,
    light_curve_epics: set[str],
    tensors: set[str],
) -> pd.DataFrame:
    out = expanded.copy()
    feature = features.copy()
    feature["epic_id"] = feature["epic_id"].map(canonical_epic)
    feature = feature.drop_duplicates("epic_id", keep="last").set_index("epic_id")
    ids = out["epic_id"]
    out["accessible_local_light_curve"] = ids.isin(light_curve_epics)
    out["accessible_512_sample_tensor"] = ids.isin(tensors)
    out["existing_cnn_probability"] = ids.map(feature["cnn_probability"].notna()).map(as_bool)
    embedding_cols = [column for column in feature if re.fullmatch(r"cnn_embedding_\d{3}", column)]
    embedding_available = feature[embedding_cols].notna().all(axis=1) if len(embedding_cols) == 128 else pd.Series(False, index=feature.index)
    out["existing_128d_embedding"] = ids.map(embedding_available).map(as_bool)
    nominal_cols = [column for column in ["primary_depth", "primary_depth_snr", "validation_period_days"] if column in feature]
    nominal = feature[nominal_cols].notna().any(axis=1) if nominal_cols else pd.Series(False, index=feature.index)
    out["nominal_numerical_diagnostics"] = ids.map(nominal).map(as_bool)
    triple_cols = ["p_half_primary_depth", "p_primary_depth", "2p_primary_depth"]
    triple = feature[triple_cols].notna().all(axis=1) if all(column in feature for column in triple_cols) else pd.Series(False, index=feature.index)
    out["p_half_p_2p_diagnostics"] = ids.map(triple).map(as_bool)
    feature_trusted = feature["nominal_period_trusted"].map(as_bool) if "nominal_period_trusted" in feature else pd.Series(False, index=feature.index)
    tier_trusted = out["period_feature_trust"].map(lambda value: clean(value).startswith("trusted"))
    out["trusted_period"] = ids.map(feature_trusted).map(as_bool) | tier_trusted | out["archive_trusted_period_available"]
    out["fallback_or_missing_period"] = ~out["trusted_period"]
    return out


def feature_coverage(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    flags = [
        "accessible_local_light_curve", "accessible_512_sample_tensor", "existing_cnn_probability",
        "existing_128d_embedding", "nominal_numerical_diagnostics", "p_half_p_2p_diagnostics", "trusted_period",
    ]
    exploded = expanded.copy()
    exploded["campaign"] = exploded["campaigns_json"].map(lambda value: parse_campaigns(value) or ["unknown"])
    exploded = exploded.explode("campaign")
    for (physical_class, campaign), group in exploded.groupby(["corrected_physical_class", "campaign"], sort=True):
        patterns = Counter()
        for _, row in group.iterrows():
            missing = [flag for flag in flags if not bool(row[flag])]
            patterns["complete" if not missing else "+".join(missing)] += 1
        rows.append({
            "corrected_physical_class": physical_class,
            "campaign": campaign,
            "total_unique_epic_hosts": int(group["epic_id"].nunique()),
            "physical_loss_eligible_hosts": int(group.loc[group["physical_loss_eligible"], "epic_id"].nunique()),
            "accessible_local_light_curves": int(group.loc[group["accessible_local_light_curve"], "epic_id"].nunique()),
            "accessible_512_sample_tensors": int(group.loc[group["accessible_512_sample_tensor"], "epic_id"].nunique()),
            "existing_cnn_probabilities": int(group.loc[group["existing_cnn_probability"], "epic_id"].nunique()),
            "existing_128d_embeddings": int(group.loc[group["existing_128d_embedding"], "epic_id"].nunique()),
            "nominal_numerical_diagnostics": int(group.loc[group["nominal_numerical_diagnostics"], "epic_id"].nunique()),
            "p_half_p_2p_diagnostics": int(group.loc[group["p_half_p_2p_diagnostics"], "epic_id"].nunique()),
            "trusted_periods": int(group.loc[group["trusted_period"], "epic_id"].nunique()),
            "fallback_or_missing_periods": int(group.loc[group["fallback_or_missing_period"], "epic_id"].nunique()),
            "cross_class_conflicts": int(group.loc[group["cross_class_conflict"], "epic_id"].nunique()),
            "missing_feature_patterns_json": json.dumps(dict(sorted(patterns.items())), separators=(",", ":")),
        })
    return pd.DataFrame(rows)


def write_audits(expanded: pd.DataFrame, evidence: pd.DataFrame, conflicts: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, object]:
    class_counts = expanded["corrected_physical_class"].value_counts().sort_index().to_dict()
    eligible_counts = expanded.loc[expanded["physical_loss_eligible"], "corrected_physical_class"].value_counts().sort_index().to_dict()
    summary = {
        "schema_version": "phase2_expanded_label_summary_v1.0.0",
        "catalogue_object_solution_rows": int(len(evidence)),
        "expanded_unique_epic_hosts": int(len(expanded)),
        "catalogue_unique_epic_hosts": int(expanded["catalogue_present"].sum()),
        "internal_unique_epic_hosts": int(expanded["internal_present"].sum()),
        "corrected_class_counts": class_counts,
        "physical_loss_eligible_counts_by_class": eligible_counts,
        "confirmed_evidence_host_count": int(expanded["archive_confirmed_evidence"].sum()),
        "candidate_evidence_host_count": int(expanded["archive_candidate_evidence"].sum()),
        "false_positive_or_refuted_evidence_host_count": int((expanded["archive_false_positive_evidence"] | expanded["archive_refuted_evidence"]).sum()),
        "archive_cross_class_conflict_host_count": int(expanded["archive_cross_class_conflict"].sum()),
        "expanded_cross_class_conflict_host_count": int(expanded["cross_class_conflict"].sum()),
        "multi_planet_or_object_host_count": int(expanded["catalogue_multi_planet_or_object_system"].map(as_bool).sum()),
        "split_assigned_host_count": int(expanded["split_assignment"].ne("").sum()),
        "existing_feature_coverage": {
            "accessible_local_light_curves": int(expanded["accessible_local_light_curve"].sum()),
            "accessible_512_sample_tensors": int(expanded["accessible_512_sample_tensor"].sum()),
            "existing_cnn_probabilities": int(expanded["existing_cnn_probability"].sum()),
            "existing_128d_embeddings": int(expanded["existing_128d_embedding"].sum()),
            "nominal_numerical_diagnostics": int(expanded["nominal_numerical_diagnostics"].sum()),
            "p_half_p_2p_diagnostics": int(expanded["p_half_p_2p_diagnostics"].sum()),
            "trusted_periods": int(expanded["trusted_period"].sum()),
        },
        "scientific_feature_columns_added": [],
        "protected_metadata_excluded_from_scientific_matrix": sorted(PROTECTED_METADATA),
        "non_actions": ["no_training", "no_split_freeze", "no_cnn_load_or_change", "no_candidate_search", "no_new_embeddings", "no_new_diagnostics", "no_catalogue_download"],
    }
    (DOCS / "phase2_expanded_label_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    class_lines = "\n".join(f"| `{key}` | {value} | {eligible_counts.get(key, 0)} |" for key, value in class_counts.items())
    audit = f"""# Phase 2 Expanded Label Table Audit

## Outcome

- Retained catalogue object/solution rows before aggregation: **{len(evidence)}**.
- Canonical one-row-per-EPIC expanded hosts: **{len(expanded)}**.
- Archive evidence hosts: confirmed **{summary['confirmed_evidence_host_count']}**, candidate **{summary['candidate_evidence_host_count']}**, false-positive/refuted **{summary['false_positive_or_refuted_evidence_host_count']}**.
- Archive-only cross-class conflicts: **{summary['archive_cross_class_conflict_host_count']}**; expanded archive/internal cross-class conflicts: **{summary['expanded_cross_class_conflict_host_count']}**.
- Multi-object systems preserved: **{summary['multi_planet_or_object_host_count']}** hosts. Default and non-default solutions remain in the object-evidence parquet and per-host solution-provenance JSON.

| Corrected physical class | Hosts | Physical-loss eligible |
| --- | ---: | ---: |
{class_lines}

## Adjudication and aggregation policy

The accepted append-only correction makes `EPIC_212024647` an eligible `false_positive_eb_or_variable` / `negative_archive_false_positive` while preserving its internal promotion and Bronze evidence. Unmatched internal candidates retain their internal state; catalogue absence is never negative evidence. Confirmed archive evidence refines compatible internal candidate evidence, while unresolved physical-class disagreements are assigned `cross_class_conflict` and excluded from physical-class loss.

All 4,064 source rows are preserved before host aggregation. Object IDs, candidate suffixes, multi-object systems, references, periods, campaigns, and default/non-default solution provenance remain explicit. No train/validation/test split is assigned.

## Leakage protection and non-actions

This label table is target/provenance/training-control metadata, not a scientific feature matrix. Archive disposition/class/reference, evidence tier, training role, manual labels/reasons, correction basis, GateVetter outputs, normalized target, and loss eligibility were not added to the accepted scientific feature matrix. No training, CNN operation, candidate search, embedding generation, diagnostic generation, or download occurred.
"""
    (DOCS / "PHASE2_EXPANDED_LABEL_TABLE_AUDIT.md").write_text(audit, encoding="utf-8")

    coverage_class = coverage.groupby("corrected_physical_class", sort=True).agg({
        "total_unique_epic_hosts": "sum", "physical_loss_eligible_hosts": "sum",
        "accessible_local_light_curves": "sum", "accessible_512_sample_tensors": "sum",
        "existing_cnn_probabilities": "sum", "existing_128d_embeddings": "sum",
        "nominal_numerical_diagnostics": "sum", "p_half_p_2p_diagnostics": "sum",
        "trusted_periods": "sum", "fallback_or_missing_periods": "sum", "cross_class_conflicts": "sum",
    }).reset_index()
    # Campaign sums intentionally count multi-campaign hosts once in each campaign; recompute unique all-campaign totals below.
    overall_rows = []
    for physical_class, group in expanded.groupby("corrected_physical_class", sort=True):
        overall_rows.append({
            "corrected_physical_class": physical_class,
            "hosts": len(group), "eligible": int(group["physical_loss_eligible"].sum()),
            "light_curves": int(group["accessible_local_light_curve"].sum()),
            "tensors": int(group["accessible_512_sample_tensor"].sum()),
            "cnn": int(group["existing_cnn_probability"].sum()),
            "embeddings": int(group["existing_128d_embedding"].sum()),
            "nominal": int(group["nominal_numerical_diagnostics"].sum()),
            "triple": int(group["p_half_p_2p_diagnostics"].sum()),
            "trusted": int(group["trusted_period"].sum()),
            "fallback": int(group["fallback_or_missing_period"].sum()),
            "conflicts": int(group["cross_class_conflict"].sum()),
        })
    coverage_lines = "\n".join(
        f"| `{r['corrected_physical_class']}` | {r['hosts']} | {r['eligible']} | {r['light_curves']} | {r['tensors']} | {r['cnn']} | {r['embeddings']} | {r['nominal']} | {r['triple']} | {r['trusted']} | {r['fallback']} | {r['conflicts']} |"
        for r in overall_rows
    )
    total_hosts = len(expanded)
    existing = summary["existing_feature_coverage"]
    coverage_audit = f"""# Phase 2 Expanded Feature Coverage Audit

## Existing-only coverage

| Corrected class | Hosts | Eligible | Local light curves | 512 tensors | CNN probabilities | 128-D embeddings | Nominal diagnostics | P/2-P-2P diagnostics | Trusted periods | Fallback/missing periods | Conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{coverage_lines}

The companion CSV reports the same fields by corrected class and campaign. A host appearing in multiple archive campaigns is counted once in each applicable campaign, so campaign totals are intentionally non-additive. `missing_feature_patterns_json` reports exact combinations of missing existing artifacts per group.

This is coverage auditing only. No light curve was downloaded; no tensor, CNN probability, embedding, numerical diagnostic, or period diagnostic was generated.

The exact blocker before feature generation is: **{total_hosts - existing['accessible_local_light_curves']}** hosts lack an accessible cached light curve, **{total_hosts - existing['accessible_512_sample_tensors']}** lack a 512-sample tensor, **{total_hosts - existing['nominal_numerical_diagnostics']}** lack nominal diagnostics, **{total_hosts - existing['p_half_p_2p_diagnostics']}** lack P/2-P-2P diagnostics, and **{int(expanded['cross_class_conflict'].sum())}** unresolved cross-class conflicts are quarantined. A provenance-controlled multi-campaign light-curve/tensor generation plan and conflict adjudication are required before population feature generation or physical-class loss.
"""
    (DOCS / "PHASE2_EXPANDED_FEATURE_COVERAGE_AUDIT.md").write_text(coverage_audit, encoding="utf-8")
    return summary


def build(args: argparse.Namespace) -> dict[str, object]:
    catalogue = pd.read_parquet(args.catalogue)
    inventory = pd.read_csv(args.inventory, dtype=str, keep_default_na=False)
    tiers = pd.read_csv(args.tiers, dtype=str, keep_default_na=False)
    corrections = pd.read_csv(args.corrections, dtype=str, keep_default_na=False)
    features = pd.read_parquet(args.features)

    evidence, catalogue_hosts = aggregate_catalogue(catalogue)
    expanded = build_expanded_hosts(inventory, tiers, corrections, catalogue_hosts)
    light_curves = cached_light_curve_epics(args.lightkurve_cache)
    tensors = tensor_epics(args.tensor_meta, args.tensor_x)
    expanded = attach_existing_feature_coverage(expanded, features, light_curves, tensors)
    conflicts = expanded.loc[expanded["cross_class_conflict"], [
        "epic_id", "internal_normalized_class", "archive_normalized_class",
        "archive_dispositions_json", "archive_references_json", "archive_cross_class_conflict",
        "internal_archive_class_conflict", "conflict_resolution",
    ]].copy()
    coverage = feature_coverage(expanded)

    evidence.to_parquet(args.evidence_output, index=False)
    expanded.to_parquet(args.label_output, index=False)
    conflicts.to_csv(args.conflicts_output, index=False)
    coverage.to_csv(args.coverage_output, index=False)
    summary = write_audits(expanded, evidence, conflicts, coverage)

    if len(evidence) != len(catalogue):
        raise AssertionError("Catalogue object/solution rows were lost before host aggregation")
    if expanded["epic_id"].duplicated().any() or expanded["epic_id"].eq("").any():
        raise AssertionError("Expanded label table is not exactly one valid row per EPIC")
    if expanded.loc[expanded["cross_class_conflict"], "physical_loss_eligible"].any():
        raise AssertionError("Unresolved cross-class conflict entered physical-class loss")
    if expanded["split_assignment"].ne("").any():
        raise AssertionError("A train/validation/test split was assigned")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audit-only expanded Phase 2 EPIC label table and existing-feature coverage.")
    parser.add_argument("--audit-only", action="store_true", help="Required safety switch; no training/generation/network operations exist.")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--tiers", type=Path, default=DEFAULT_TIERS)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--tensor-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--tensor-x", type=Path, default=DEFAULT_X)
    parser.add_argument("--lightkurve-cache", type=Path, default=Path.home() / ".lightkurve" / "cache" / "mastDownload")
    parser.add_argument("--evidence-output", type=Path, default=DATA / "phase2_expanded_catalogue_evidence.parquet")
    parser.add_argument("--label-output", type=Path, default=DATA / "phase2_expanded_label_table.parquet")
    parser.add_argument("--conflicts-output", type=Path, default=DOCS / "phase2_expanded_label_conflicts.csv")
    parser.add_argument("--coverage-output", type=Path, default=DOCS / "phase2_expanded_feature_coverage.csv")
    args = parser.parse_args()
    if not args.audit_only:
        parser.error("--audit-only is required")
    return args


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
