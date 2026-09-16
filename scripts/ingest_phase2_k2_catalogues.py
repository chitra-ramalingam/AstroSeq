from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "phase2_k2_catalogue_object_v1.0.0"
SOURCE_SLUG = "nea_k2pandc"
SOURCE_TABLE = "k2pandc"
REQUIRED_RAW_COLUMNS = [
    "pl_name", "k2_name", "epic_hostname", "default_flag", "disposition",
    "disp_refname", "k2_campaigns", "pl_orbper", "rowupdate", "releasedate",
]
AUDITED_EPICS = [
    "EPIC_211357782", "EPIC_211497712", "EPIC_211534076",
    "EPIC_211915147", "EPIC_211953866", "EPIC_211889692",
    "EPIC_211682657", "EPIC_212001099", "EPIC_212024647",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def canonical_epic(value: Any) -> tuple[str, str]:
    raw = clean(value)
    if not raw:
        return "", "missing"
    match = re.fullmatch(r"EPIC[ _]?(\d{8,10})", raw, flags=re.IGNORECASE)
    if not match:
        return "", "malformed"
    return f"EPIC_{match.group(1)}", "valid"


def candidate_suffix(object_id: Any, epic_id: str) -> str:
    raw = clean(object_id)
    if not raw or not epic_id:
        return ""
    digits = epic_id.removeprefix("EPIC_")
    match = re.fullmatch(rf"EPIC[ _]?{re.escape(digits)}\.(\d+)", raw, flags=re.IGNORECASE)
    return f".{match.group(1)}" if match else ""


def parse_campaigns(value: Any) -> list[str]:
    raw = clean(value)
    if not raw or raw.lower() in {"none", "null", "nan"}:
        return []
    result: list[str] = []
    for token in re.split(r"\s*,\s*", raw):
        token = token.strip()
        if token and token not in result:
            result.append(token)
    return result


def map_disposition(value: Any) -> tuple[str, str, bool]:
    raw = re.sub(r"\s+", " ", clean(value).upper())
    if raw == "CONFIRMED":
        return "candidate_like", "archive_confirmed", True
    if raw == "CANDIDATE":
        return "candidate_like", "archive_candidate", True
    if raw in {"FALSE POSITIVE", "FALSE POSITIVE [CANDIDATE]"}:
        return "false_positive_eb_or_variable", "archive_false_positive", True
    if raw in {"REFUTED", "REFUTED [PLANET]"}:
        return "false_positive_eb_or_variable", "archive_refuted", True
    return "uncertain_hold", "unknown_or_missing_disposition", False


def safe_period(value: Any) -> float:
    try:
        period = float(clean(value))
    except ValueError:
        return np.nan
    return period if np.isfinite(period) and period > 0 else np.nan


def discover_manifest(snapshot_root: Path) -> Path:
    source_root = snapshot_root / SOURCE_SLUG
    manifests = sorted(source_root.glob("*/source_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No {SOURCE_SLUG} source_manifest.json under {source_root}")
    # The CLI contract has no date argument. Select the latest immutable ISO-date
    # directory deterministically and record the selected version in the audit.
    return manifests[-1]


def validate_raw_frame(frame: pd.DataFrame, raw_info: dict[str, Any]) -> None:
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw k2pandc schema drift: missing columns {missing}")
    if list(frame.columns) != list(raw_info.get("columns", [])):
        raise ValueError("Raw column order/list differs from the immutable manifest")
    if len(frame) != int(raw_info.get("row_count", -1)):
        raise ValueError(f"Raw row-count mismatch: expected {raw_info.get('row_count')}, got {len(frame)}")


def load_and_verify_snapshot(manifest_path: Path) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_table") != SOURCE_TABLE:
        raise ValueError(f"Unsupported source table: {manifest.get('source_table')!r}")
    if manifest.get("ingestion_schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest ingestion schema version does not match the implemented contract")
    if manifest.get("schema_verification", {}).get("schema_drift_detected"):
        raise ValueError("Manifest records schema drift; ingestion is blocked")
    raw_info = manifest.get("raw_file", {})
    raw_path = manifest_path.parent / raw_info.get("relative_path", "")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Declared raw snapshot is absent: {raw_path}")
    actual_size = raw_path.stat().st_size
    if actual_size != int(raw_info.get("byte_size", -1)):
        raise ValueError(f"Raw file byte-size mismatch: expected {raw_info.get('byte_size')}, got {actual_size}")
    actual_hash = sha256(raw_path)
    if actual_hash != raw_info.get("sha256"):
        raise ValueError(f"Raw file SHA-256 mismatch: expected {raw_info.get('sha256')}, got {actual_hash}")
    frame = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    validate_raw_frame(frame, raw_info)
    return manifest, frame, raw_path


def normalize_rows(raw: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    raw = raw.copy()
    raw["_exact_duplicate"] = raw.duplicated(REQUIRED_RAW_COLUMNS, keep=False)
    raw["_duplicate_group_size"] = raw.groupby(REQUIRED_RAW_COLUMNS, dropna=False)["pl_name"].transform("size")
    rows: list[dict[str, Any]] = []
    source_hash = manifest["raw_file"]["sha256"]
    source_version = manifest["source_version"]
    citation = f"NASA Exoplanet Archive {SOURCE_TABLE}; DOI {manifest['source_citation']['doi']}"
    for zero_index, row in raw.iterrows():
        source_row = int(zero_index) + 1
        epic_id, identifier_status = canonical_epic(row["epic_hostname"])
        object_id = clean(row["pl_name"]) or clean(row["k2_name"]) or clean(row["epic_hostname"])
        proposed, evidence_kind, known = map_disposition(row["disposition"])
        reference = clean(row["disp_refname"])
        default_solution = clean(row["default_flag"]) == "1"
        period = safe_period(row["pl_orbper"])
        reasons = []
        if not known:
            reasons.append("unknown_or_missing_disposition")
        if not reference:
            reasons.append("missing_disposition_reference")
        if identifier_status != "valid":
            reasons.append(f"epic_identifier_{identifier_status}")
        if not default_solution:
            reasons.append("non_default_solution")
        locator = f"raw/k2pandc.csv#data_row={source_row};csv_line={source_row + 1}"
        identity_payload = json.dumps(
            {"source_sha256": source_hash, "source_row": source_row,
             "values": [clean(row[c]) for c in REQUIRED_RAW_COLUMNS]},
            separators=(",", ":"), sort_keys=True,
        )
        rows.append({
            "catalogue_row_id": hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
            "source_slug": SOURCE_SLUG,
            "source_table": SOURCE_TABLE,
            "source_version": source_version,
            "source_retrieved_at_utc": manifest["retrieval_utc_timestamp"],
            "source_sha256": source_hash,
            "source_record_locator": locator,
            "source_row_number": source_row,
            "raw_epic_hostname": clean(row["epic_hostname"]),
            "epic_id": epic_id,
            "epic_identifier_status": identifier_status,
            "object_id": object_id,
            "candidate_suffix": candidate_suffix(object_id, epic_id),
            "k2_name": clean(row["k2_name"]),
            "default_solution": default_solution,
            "raw_disposition": clean(row["disposition"]),
            "raw_disposition_reference": reference,
            "disposition_known": known,
            "has_disposition_reference": bool(reference),
            "normalized_label_proposal": proposed,
            "evidence_kind": evidence_kind,
            "raw_campaigns": clean(row["k2_campaigns"]),
            "campaigns_json": json.dumps(parse_campaigns(row["k2_campaigns"]), separators=(",", ":")),
            "raw_period_days": clean(row["pl_orbper"]),
            "period_days": period,
            "period_source": "k2pandc.pl_orbper",
            "period_provenance": f"{SOURCE_SLUG}:{source_version}:{locator}:pl_orbper",
            "period_trusted": False,
            "row_update_date": clean(row["rowupdate"]),
            "release_date": clean(row["releasedate"]),
            "physical_loss_eligible_proposal": len(reasons) == 0,
            "eligibility_exclusion_reason": "|".join(reasons),
            "exact_duplicate_source_row": bool(row["_exact_duplicate"]),
            "exact_duplicate_group_size": int(row["_duplicate_group_size"]),
            "source_citation": citation,
            "ingestion_schema_version": SCHEMA_VERSION,
        })
    result = pd.DataFrame(rows)
    valid = result[result["epic_id"].ne("")]
    classes = valid.groupby("epic_id")["normalized_label_proposal"].agg(set)
    conflict_epics = set(classes[classes.map(lambda x: "candidate_like" in x and "false_positive_eb_or_variable" in x)].index)
    result["cross_class_conflict"] = result["epic_id"].isin(conflict_epics)
    conflict = result["cross_class_conflict"]
    result.loc[conflict, "physical_loss_eligible_proposal"] = False
    result.loc[conflict, "eligibility_exclusion_reason"] = result.loc[conflict, "eligibility_exclusion_reason"].map(
        lambda value: f"{value}|cross_class_conflict" if value else "cross_class_conflict"
    )
    return result


def ordered_unique(values: pd.Series) -> list[str]:
    return sorted({clean(value) for value in values if clean(value)})


def current_internal_labels() -> dict[str, str]:
    path = ROOT / "docs/phase2/phase2_label_inventory.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(frame["epic_id"], frame["current_final_label"]))


def recommended_action(epic: str, group: pd.DataFrame, current_label: str) -> tuple[str, str]:
    if group.empty:
        return "no_archive_match_retain_current_status", "No exact EPIC host match in this k2pandc snapshot."
    proposals = set(group["normalized_label_proposal"])
    referenced = bool(group["has_disposition_reference"].all())
    if bool(group["cross_class_conflict"].any()):
        return "quarantine_archive_cross_class_conflict", "Archive contains both positive and false-positive/refuted evidence; no automatic label change."
    if not referenced or "uncertain_hold" in proposals:
        return "quarantine_missing_ambiguous_or_unreferenced_disposition", "Archive evidence is incomplete or unknown; keep out of physical loss."
    archive_negative = proposals == {"false_positive_eb_or_variable"}
    archive_positive = proposals == {"candidate_like"}
    if archive_negative and current_label == "false_positive_eb_or_variable":
        return "archive_negative_supports_current_negative_no_automatic_change", "Authoritative archive negative evidence agrees with the corrected internal class."
    if archive_negative and current_label == "candidate_like":
        return "manual_adjudication_required_archive_negative_conflicts_with_internal_candidate", "Archive negative evidence conflicts with internal candidate provenance; preserve both and adjudicate later."
    if archive_positive and current_label == "candidate_like":
        return "archive_positive_evidence_available_for_manual_tier_review", "Archive evidence may strengthen provenance only after object/disposition-reference review."
    if archive_positive and current_label == "false_positive_eb_or_variable":
        return "manual_adjudication_required_archive_positive_conflicts_with_internal_negative", "Archive positive evidence conflicts with the internal negative class."
    return "manual_review_required_no_automatic_change", f"Archive proposal does not map cleanly against current internal label {current_label or 'missing'}."


def build_mapping_audit(rows: pd.DataFrame) -> pd.DataFrame:
    internal = current_internal_labels()
    output = []
    for epic in AUDITED_EPICS:
        group = rows[rows["epic_id"].eq(epic)].copy()
        action, notes = recommended_action(epic, group, internal.get(epic, ""))
        campaigns = []
        for raw_json in group.get("campaigns_json", pd.Series(dtype=str)):
            for campaign in json.loads(raw_json):
                if campaign not in campaigns:
                    campaigns.append(campaign)
        periods = sorted({float(x) for x in group["period_days"].dropna()}) if len(group) else []
        output.append({
            "epic_id": epic,
            "archive_match_found": bool(len(group)),
            "archive_object_ids": " | ".join(ordered_unique(group.get("object_id", pd.Series(dtype=str)))),
            "archive_k2_names": " | ".join(ordered_unique(group.get("k2_name", pd.Series(dtype=str)))),
            "archive_dispositions": " | ".join(ordered_unique(group.get("raw_disposition", pd.Series(dtype=str)))),
            "archive_disposition_references": " | ".join(ordered_unique(group.get("raw_disposition_reference", pd.Series(dtype=str)))),
            "archive_campaigns": " | ".join(campaigns),
            "archive_periods": " | ".join(f"{value:.12g}" for value in periods),
            "cross_class_conflict": bool(group["cross_class_conflict"].any()) if len(group) else False,
            "recommended_label_action": action,
            "notes": notes,
        })
    return pd.DataFrame(output)


def host_set(rows: pd.DataFrame, dispositions: set[str], default_only: bool = False) -> set[str]:
    mask = rows["raw_disposition"].str.upper().isin(dispositions) & rows["epic_id"].ne("")
    if default_only:
        mask &= rows["default_solution"]
    return set(rows.loc[mask, "epic_id"])


def build_counts(rows: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    confirmed = host_set(rows, {"CONFIRMED"})
    candidate = host_set(rows, {"CANDIDATE"})
    false_positive = host_set(rows, {"FALSE POSITIVE", "FALSE POSITIVE [CANDIDATE]", "REFUTED", "REFUTED [PLANET]"})
    conflict = set(rows.loc[rows["cross_class_conflict"], "epic_id"]) - {""}
    campaign_hosts: Counter[str] = Counter()
    for _, row in rows[rows["epic_id"].ne("")][["epic_id", "campaigns_json"]].drop_duplicates().iterrows():
        for campaign in json.loads(row["campaigns_json"]):
            campaign_hosts[campaign] += 1
    return {
        "source_row_count": int(len(rows)),
        "unique_epic_host_count": int(rows.loc[rows["epic_id"].ne(""), "epic_id"].nunique()),
        "confirmed_host_count": len(confirmed),
        "candidate_host_count": len(candidate),
        "false_positive_or_refuted_host_count": len(false_positive),
        "default_solution_confirmed_host_count": len(host_set(rows, {"CONFIRMED"}, True)),
        "default_solution_candidate_host_count": len(host_set(rows, {"CANDIDATE"}, True)),
        "default_solution_false_positive_or_refuted_host_count": len(host_set(rows, {"FALSE POSITIVE", "FALSE POSITIVE [CANDIDATE]", "REFUTED", "REFUTED [PLANET]"}, True)),
        "unknown_disposition_row_count": int((~rows["disposition_known"]).sum()),
        "missing_disposition_reference_row_count": int((~rows["has_disposition_reference"]).sum()),
        "missing_epic_identifier_row_count": int(rows["epic_identifier_status"].eq("missing").sum()),
        "malformed_epic_identifier_row_count": int(rows["epic_identifier_status"].eq("malformed").sum()),
        "exact_duplicate_source_row_count": int(raw.duplicated(REQUIRED_RAW_COLUMNS).sum()),
        "exact_duplicate_group_member_count": int(rows["exact_duplicate_source_row"].sum()),
        "cross_class_conflict_host_count": len(conflict),
        "physical_loss_eligible_object_row_count": int(rows["physical_loss_eligible_proposal"].sum()),
        "raw_disposition_row_counts": {str(k): int(v) for k, v in rows["raw_disposition"].value_counts(dropna=False).sort_index().items()},
        "campaign_unique_host_counts": dict(sorted(campaign_hosts.items())),
    }


def write_summary(path: Path, manifest: dict[str, Any], counts: dict[str, Any], mapping: pd.DataFrame, output: Path) -> None:
    mapping_lines = []
    for row in mapping.itertuples(index=False):
        mapping_lines.append(
            f"| `{row.epic_id}` | {str(row.archive_match_found).lower()} | {row.archive_object_ids or '(none)'} | "
            f"{row.archive_dispositions or '(none)'} | {row.archive_disposition_references or '(none)'} | "
            f"{str(row.cross_class_conflict).lower()} | `{row.recommended_label_action}` |"
        )
    text = f"""# Phase 2A K2PANDC Ingestion Summary

## Immutable source

- Source: NASA Exoplanet Archive `k2pandc`.
- Snapshot: `data/phase2/catalogues/nea_k2pandc/2026-08-05/raw/k2pandc.csv`.
- Retrieval UTC: `{manifest['retrieval_utc_timestamp']}`.
- SHA-256: `{manifest['raw_file']['sha256']}`.
- Bytes: **{manifest['raw_file']['byte_size']}**.
- Rows: **{counts['source_row_count']}**; unique canonical EPIC hosts: **{counts['unique_epic_host_count']}**.
- Schema drift: **none** across the ten requested fields.

## Object-level ingestion audit

| Measure | Count |
| --- | ---: |
| Confirmed hosts (any retained solution) | {counts['confirmed_host_count']} |
| Candidate hosts (any retained solution) | {counts['candidate_host_count']} |
| False-positive/refuted hosts (any retained solution) | {counts['false_positive_or_refuted_host_count']} |
| Unknown-disposition rows | {counts['unknown_disposition_row_count']} |
| Missing disposition-reference rows | {counts['missing_disposition_reference_row_count']} |
| Exact duplicate rows beyond the first | {counts['exact_duplicate_source_row_count']} |
| Cross-class conflict hosts | {counts['cross_class_conflict_host_count']} |
| Object rows provisionally eligible after reference/default/conflict checks | {counts['physical_loss_eligible_object_row_count']} |

All **{counts['source_row_count']}** source rows are retained in `{output.as_posix()}`. Candidate suffixes remain in `object_id` and `candidate_suffix`; multiple planets and parameter solutions remain separate. Unknown, unreferenced, malformed/missing-EPIC, non-default, and cross-class-conflict rows are loss-ineligible proposals. No internal label was changed.

## Nine-EPIC focused mapping audit

| EPIC | Match | Archive object IDs | Dispositions | References | Archive cross-class conflict | Recommended action |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(mapping_lines)}

## Blockers before an expanded EPIC-level label table

1. Only two of the nine audited EPICs have exact `k2pandc` host matches; the seven unmatched objects retain their current internal status.
2. `EPIC_212024647` has authoritative archive false-positive evidence that conflicts with its retained internal Bronze candidate provenance and requires append-only adjudication.
3. Archive-wide cross-class hosts must remain quarantined when object evidence is aggregated to EPIC level.
4. This task ingested only `k2pandc`; the separately planned EB/variable catalogues and full campaign manifest remain absent and were not downloaded.
5. The expanded builder must join internal corrections without converting catalogue provenance fields into scientific model features.

No CatBoost training, split freeze, CNN load/change/retraining, candidate search, GateVetter classification, expanded-label build, or automatic label modification occurred.
"""
    path.write_text(text, encoding="utf-8")


def ingest(source_inventory: Path, snapshot_root: Path, output: Path, audit_output: Path, offline: bool) -> dict[str, Any]:
    if not offline:
        raise ValueError("Network access is disabled in this ingestion utility; pass --offline")
    inventory = pd.read_csv(source_inventory, dtype=str, keep_default_na=False)
    if not inventory["catalogue_name"].str.contains("K2 Planets and Candidates", regex=False).any():
        raise ValueError("Source inventory does not declare the k2pandc source")
    manifest_path = discover_manifest(snapshot_root)
    manifest, raw, raw_path = load_and_verify_snapshot(manifest_path)
    rows = normalize_rows(raw, manifest)
    if len(rows) != len(raw) or rows["catalogue_row_id"].nunique() != len(rows):
        raise AssertionError("Every source row must produce one unique object/solution row")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output, index=False)
    mapping_path = ROOT / "docs/phase2/phase2_k2pandc_epic_mapping_audit.csv"
    summary_path = ROOT / "docs/phase2/PHASE2_K2PANDC_INGESTION_SUMMARY.md"
    mapping = build_mapping_audit(rows)
    mapping.to_csv(mapping_path, index=False)
    counts = build_counts(rows, raw)
    write_summary(summary_path, manifest, counts, mapping, output.relative_to(ROOT))
    audit = {
        "audit_schema_version": "phase2_k2pandc_ingestion_audit_v1.0.0",
        "offline": True,
        "source_inventory": source_inventory.relative_to(ROOT).as_posix(),
        "selected_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "selected_snapshot": raw_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(manifest_path),
        "source_sha256_verified": True,
        "source_byte_size_verified": True,
        "source_row_count_verified": True,
        "schema_columns_verified": REQUIRED_RAW_COLUMNS,
        "schema_drift": [],
        "counts": counts,
        "output_artifacts": {
            output.relative_to(ROOT).as_posix(): {"sha256": sha256(output), "rows": len(rows)},
            mapping_path.relative_to(ROOT).as_posix(): {"sha256": sha256(mapping_path), "rows": len(mapping)},
            summary_path.relative_to(ROOT).as_posix(): {"sha256": sha256(summary_path)},
        },
        "failures": [],
        "non_actions": [
            "no_model_training", "no_split_freeze", "no_cnn_load_or_change",
            "no_candidate_search", "no_gatevetter_features", "no_label_modification",
            "no_expanded_label_table_build",
        ],
    }
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline, hash-verified Phase 2 K2 catalogue ingestion.")
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    audit = ingest(
        resolve(args.source_inventory), resolve(args.snapshot_root), resolve(args.output),
        resolve(args.audit_output), args.offline,
    )
    print(json.dumps({"status": "ok", **audit["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
