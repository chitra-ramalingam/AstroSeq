from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "plots" / "k2_batch" / "stage_h_candidate_evidence_packets"
TARGETS = [
    "EPIC_212001099",
    "EPIC_211889692",
    "EPIC_211534076",
    "EPIC_211954033",
    "EPIC_211759361",
]

LABELS_LIVE = ROOT / "training_labels_v3.csv"
LABELS_FROZEN = ROOT / "freezes" / "training_labels_v3_stage_h_candidate_followup_45.csv"
LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"
STAGE_G = ROOT / "freezes" / "stage_g_v2_support_tier_review.csv"
FLUX = ROOT / "k2_hybrid_candidate_score_v2.csv"

VALIDATION_PATHS = {
    "EPIC_212001099": ROOT
    / "plots"
    / "k2_batch"
    / "stage_f_next10_batch2"
    / "EPIC_212001099"
    / "validation_summary.json",
    "EPIC_211889692": ROOT
    / "plots"
    / "k2_batch"
    / "stage_f_followup"
    / "EPIC_211889692"
    / "validation_summary.json",
    "EPIC_211534076": ROOT
    / "plots"
    / "k2_batch"
    / "stage_f_followup"
    / "EPIC_211534076"
    / "validation_summary.json",
    "EPIC_211954033": ROOT
    / "plots"
    / "k2_batch"
    / "stage_f_next10_batch2"
    / "EPIC_211954033"
    / "validation_summary.json",
    "EPIC_211759361": ROOT
    / "plots"
    / "k2_batch"
    / "stage_h_candidate_followup"
    / "EPIC_211759361"
    / "validation_summary.json",
}


def read_csv_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["epic_id"]: row for row in csv.DictReader(handle)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def nonblank(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def metric(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data
        found = True
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                found = False
                break
        if found and value is not None and str(value) != "":
            return value
    return ""


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def alias_notes(validation_json: dict[str, Any]) -> str:
    validation = validation_json.get("validation", {})
    parts = []
    alias_risk = validation.get("alias_risk")
    if alias_risk:
        parts.append(f"alias_risk={alias_risk}")
    best_period = validation.get("alias_best_period_days")
    best_support = validation.get("alias_best_support_count")
    best_ratio = validation.get("alias_best_support_ratio")
    if best_period not in (None, ""):
        parts.append(
            "best_alias="
            + fmt(best_period)
            + " d"
            + (f", support={fmt(best_support)}" if best_support not in (None, "") else "")
            + (f", ratio={fmt(best_ratio)}" if best_ratio not in (None, "") else "")
        )
    aliases = []
    for row in validation_json.get("alias_periods", []):
        aliases.append(
            f"{row.get('alias_name')}:{fmt(row.get('period_days'))}d/support={fmt(row.get('support_count'))}"
        )
    if aliases:
        parts.append("aliases=[" + "; ".join(aliases) + "]")
    return " | ".join(parts)


def copy_artifacts(packet_dir: Path, validation_json: dict[str, Any], validation_path: Path) -> list[str]:
    copied: list[str] = []
    artifact_dir = packet_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if validation_path.exists():
        dest = artifact_dir / "validation_summary.json"
        shutil.copy2(validation_path, dest)
        copied.append(rel(dest))

    validation = validation_json.get("validation", {})
    for key in [
        "phase_0_folded_path",
        "phase_05_secondary_check_path",
        "alias_period_comparison_path",
        "odd_even_zoom_path",
    ]:
        source_value = validation.get(key)
        if not source_value:
            continue
        source = Path(source_value)
        if not source.is_absolute():
            source = ROOT / source
        if source.exists():
            dest = artifact_dir / source.name
            shutil.copy2(source, dest)
            copied.append(rel(dest))
    return copied


def caveats_for(epic_id: str, ledger_row: dict[str, str], label_row: dict[str, str], validation_json: dict[str, Any]) -> str:
    explicit = nonblank(
        ledger_row.get("stage_h_reason"),
        label_row.get("stage_h_reason"),
        ledger_row.get("stage_f_closed_45_caveats"),
        ledger_row.get("status_reason"),
    )
    if explicit:
        return explicit
    validation = validation_json.get("validation", {})
    caveats = []
    if validation.get("alias_risk"):
        caveats.append(f"alias_risk={validation.get('alias_risk')}")
    ratio = validation.get("secondary_to_primary_depth_ratio")
    if ratio not in (None, "") and float(ratio) > 0:
        caveats.append(f"secondary_to_primary_depth_ratio={fmt(ratio)}")
    oot = validation.get("oot_variability_to_depth")
    if oot not in (None, "") and float(oot) >= 1:
        caveats.append(f"oot_variability_to_depth={fmt(oot)}")
    return "; ".join(caveats)


def packet_markdown(row: dict[str, str], copied: list[str]) -> str:
    lines = [
        f"# {row['epic_id']} Stage H Candidate Evidence Packet",
        "",
        "## Status",
        f"- training_label_v3: {row['training_label_v3']}",
        f"- science_binary_v3: {row['science_binary_v3']}",
        f"- ledger_status: {row['ledger_status']}",
        f"- recommended_candidate_status: {row['recommended_candidate_status']}",
        "",
        "## Scores",
        f"- Stage G support tier: {row['stage_g_support_tier']}",
        f"- Stage G calibrated score: {row['stage_g_calibrated_score']}",
        f"- flux_p_science_like: {row['flux_p_science_like']}",
        "",
        "## Transit Evidence",
        f"- best_period_days: {row['best_period_days']}",
        f"- primary_depth: {row['primary_depth']}",
        f"- primary_depth_snr: {row['primary_depth_snr']}",
        f"- transit_duration_hours: {row['transit_duration_hours']}",
        f"- odd_even_depth_ratio: {row['odd_even_depth_ratio']}",
        f"- secondary_to_primary_depth_ratio: {row['secondary_to_primary_depth_ratio']}",
        f"- oot_variability_to_depth: {row['oot_variability_to_depth']}",
        "",
        "## Alias Notes",
        row["alias_notes"] or "No alias notes found.",
        "",
        "## Key Caveats",
        row["key_caveats"] or "No additional caveats recorded.",
        "",
        "## Source Artifacts",
        f"- validation_summary_json: {row['validation_summary_json_path']}",
        f"- phase_0_folded_path: {row['phase_0_folded_path']}",
        f"- phase_05_secondary_check_path: {row['phase_05_secondary_check_path']}",
        f"- alias_period_comparison_path: {row['alias_period_comparison_path']}",
        f"- odd_even_zoom_path: {row['odd_even_zoom_path']}",
        "",
        "## Copied Packet Artifacts",
    ]
    lines.extend([f"- {path}" for path in copied] or ["- none"])
    lines.extend(
        [
            "",
            "## Policy Guardrail",
            "This packet is evidence-only. It does not update labels, the ledger, or any model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    labels_live = read_csv_index(LABELS_LIVE)
    labels_frozen = read_csv_index(LABELS_FROZEN)
    ledger = read_csv_index(LEDGER)
    stage_g = read_csv_index(STAGE_G)
    flux = read_csv_index(FLUX)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for epic_id in TARGETS:
        label = labels_frozen.get(epic_id) or labels_live[epic_id]
        live_label = labels_live.get(epic_id, {})
        ledger_row = ledger.get(epic_id, {})
        stage_g_row = stage_g.get(epic_id, {})
        flux_row = flux.get(epic_id, {})
        validation_path = VALIDATION_PATHS[epic_id]
        validation_json = load_json(validation_path)
        validation = validation_json.get("validation", {})

        packet_dir = OUT_ROOT / epic_id
        packet_dir.mkdir(parents=True, exist_ok=True)
        copied = copy_artifacts(packet_dir, validation_json, validation_path)

        ledger_status = nonblank(
            ledger_row.get("stage_h_ledger_status"),
            ledger_row.get("final_candidate_status"),
            label.get("final_candidate_status"),
        )
        recommended_status = nonblank(
            ledger_row.get("stage_h_ledger_status"),
            ledger_row.get("final_candidate_status"),
            label.get("final_candidate_status"),
        )

        row = {
            "epic_id": epic_id,
            "training_label_v3": nonblank(label.get("training_label_v3"), live_label.get("training_label_v3")),
            "science_binary_v3": nonblank(label.get("science_binary_v3"), live_label.get("science_binary_v3")),
            "ledger_status": ledger_status,
            "recommended_candidate_status": recommended_status,
            "stage_g_support_tier": nonblank(
                stage_g_row.get("stage_g_support_tier"), ledger_row.get("stage_g_v2_support_tier")
            ),
            "stage_g_calibrated_score": nonblank(
                stage_g_row.get("calibrated_score"), ledger_row.get("stage_g_v2_calibrated_score")
            ),
            "flux_p_science_like": nonblank(flux_row.get("flux_p_science_like")),
            "best_period_days": fmt(metric(validation_json, "validation.best_period_days", "stage_e_row.best_period_days")),
            "primary_depth": fmt(metric(validation_json, "validation.primary_depth")),
            "primary_depth_snr": fmt(
                metric(validation_json, "validation.primary_depth_snr", "stage_d_summary.stage_r_and_stage_d_metrics.primary_depth_snr")
            ),
            "transit_duration_hours": fmt(metric(validation_json, "validation.transit_duration_hours")),
            "odd_even_depth_ratio": fmt(metric(validation_json, "validation.odd_even_depth_ratio")),
            "secondary_to_primary_depth_ratio": fmt(metric(validation_json, "validation.secondary_to_primary_depth_ratio")),
            "oot_variability_to_depth": fmt(metric(validation_json, "validation.oot_variability_to_depth")),
            "alias_notes": alias_notes(validation_json),
            "key_caveats": caveats_for(epic_id, ledger_row, label, validation_json),
            "validation_summary_json_path": rel(validation_path),
            "phase_0_folded_path": rel(validation.get("phase_0_folded_path")),
            "phase_05_secondary_check_path": rel(validation.get("phase_05_secondary_check_path")),
            "alias_period_comparison_path": rel(validation.get("alias_period_comparison_path")),
            "odd_even_zoom_path": rel(validation.get("odd_even_zoom_path")),
            "packet_dir": rel(packet_dir),
            "packet_markdown": rel(packet_dir / "evidence_packet.md"),
            "packet_json": rel(packet_dir / "evidence_packet.json"),
            "copied_artifacts": ";".join(copied),
        }

        (packet_dir / "evidence_packet.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (packet_dir / "evidence_packet.md").write_text(packet_markdown(row, copied), encoding="utf-8")
        summary_rows.append(row)

    summary_path = OUT_ROOT / "stage_h_candidate_evidence_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} packets under {rel(OUT_ROOT)}")
    print(f"Wrote {rel(summary_path)}")


if __name__ == "__main__":
    main()
