from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PACKETS = [
    ("stage_g_review", ROOT / "unseen_stage_g_review_manual_packet.csv"),
    ("top_holds", ROOT / "unseen_top_holds_manual_packet.csv"),
    ("reject_sanity_sample", ROOT / "unseen_reject_sanity_sample.csv"),
]

VALIDATION_SOURCES = [
    ROOT
    / "plots"
    / "k2_batch"
    / "stage_i_autovet_v1_candidate_batch1"
    / "autovet_candidate_validation_ledger.csv",
    ROOT
    / "plots"
    / "k2_batch"
    / "stage_i_autovet_v1_hold_batch1"
    / "autovet_hold_validation_ledger.csv",
    ROOT
    / "plots"
    / "k2_batch"
    / "stage_i_autovet_v1_hold_batch1"
    / "post_repair_validation_ledger.csv",
]

STAGE_I_RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
FINAL_LEDGER = ROOT / "plots" / "k2_batch" / "final_candidate_master_ledger.csv"
OUT_CSV = ROOT / "unseen_null_metric_audit.csv"
OUT_SUMMARY = ROOT / "unseen_null_metric_audit_summary.txt"

METRICS = {
    "primary_depth": ["post_repair_primary_depth", "primary_depth"],
    "odd_even_depth_ratio": [
        "post_repair_odd_even_depth_ratio",
        "odd_even_ratio",
        "odd_even_depth_ratio",
    ],
    "oot_to_depth": [
        "post_repair_oot_variability_to_depth",
        "oot_variability_to_depth",
        "oot_to_depth",
    ],
    "secondary_to_primary_depth_ratio": [
        "post_repair_secondary_to_primary_depth_ratio",
        "secondary_to_primary_ratio",
        "secondary_to_primary_depth_ratio",
    ],
    "alias_risk": ["post_repair_alias_risk", "alias_risk"],
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_epic(value: Any) -> str:
    text = clean(value).replace("EPIC ", "EPIC_")
    if text.isdigit():
        return f"EPIC_{text}"
    return text


def load_by_epic(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    epic_column = next(
        (column for column in ("epic_id", "epic", "query") if column in frame.columns),
        None,
    )
    if epic_column is None:
        return {}
    return {
        normalize_epic(row[epic_column]): row.to_dict()
        for _, row in frame.iterrows()
        if normalize_epic(row[epic_column])
    }


def validation_rows() -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    rows: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in VALIDATION_SOURCES:
        for epic_id, row in load_by_epic(path).items():
            rows.setdefault(epic_id, []).append((path, row))
    return rows


def historical_period_sources(epic_ids: set[str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in (ROOT / "plots" / "k2_batch").rglob("period_shortlist_best.csv"):
        try:
            rows = load_by_epic(path)
        except Exception:
            continue
        for epic_id in epic_ids.intersection(rows):
            found.setdefault(epic_id, []).append(path)
    return found


def resolve_metric(
    epic_id: str,
    metric: str,
    packet_row: dict[str, Any],
    validations: dict[str, list[tuple[Path, dict[str, Any]]]],
    packet_path: Path,
    stage_i_rows: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    aliases = METRICS[metric]

    # Reverse order matches the unseen runner's update behavior: post-repair
    # validation supersedes the earlier hold/candidate validation rows.
    for path, row in reversed(validations.get(epic_id, [])):
        for column in aliases:
            value = clean(row.get(column))
            if value:
                return value, f"{rel(path)}::{column}"

    for column in aliases:
        value = clean(packet_row.get(column))
        if value:
            return value, f"{rel(packet_path)}::{column}"

    stage_i_row = stage_i_rows.get(epic_id, {})
    for column in aliases:
        value = clean(stage_i_row.get(column))
        if value:
            return value, f"{rel(STAGE_I_RESULTS)}::{column}"

    return "", ""


def main() -> None:
    packet_rows: list[tuple[str, Path, dict[str, Any]]] = []
    for packet_source, packet_path in PACKETS:
        frame = pd.read_csv(packet_path, dtype=str).fillna("")
        for _, row in frame.iterrows():
            packet_rows.append((packet_source, packet_path, row.to_dict()))

    epic_ids = {normalize_epic(row["epic_id"]) for _, _, row in packet_rows}
    validations = validation_rows()
    stage_i_rows = load_by_epic(STAGE_I_RESULTS)
    final_ledger_rows = load_by_epic(FINAL_LEDGER)
    period_sources = historical_period_sources(epic_ids)

    audit_rows: list[dict[str, Any]] = []
    for packet_source, packet_path, packet_row in packet_rows:
        epic_id = normalize_epic(packet_row["epic_id"])
        resolved: dict[str, tuple[str, str]] = {}
        for metric in METRICS:
            resolved[metric] = resolve_metric(
                epic_id,
                metric,
                packet_row,
                validations,
                packet_path,
                stage_i_rows,
            )

        missing = [metric for metric, (value, _) in resolved.items() if not value]
        stage_i_row = stage_i_rows.get(epic_id, {})
        saved_period = clean(packet_row.get("best_period_days")) or clean(
            stage_i_row.get("best_period_days")
        )
        has_validation = epic_id in validations
        has_historical_period = epic_id in period_sources

        reasons: list[str] = []
        if missing:
            reasons.append("missing:" + "|".join(missing))
            if not has_validation:
                reasons.append("no_full_stage_f_or_stage_i_validation_row")
            if saved_period:
                reasons.append("saved_period_present_but_aggregate_diagnostics_absent")
            else:
                reasons.append("unseen_source_has_no_saved_period_support")
            if has_historical_period:
                reasons.append("historical_period_search_exists_but_was_not_joined")
            reasons.append("validation_summary_copies_packet_metrics_without_recomputing")
        else:
            reasons.append("none")

        fix_needed = "yes" if missing else "no"

        audit_rows.append(
            {
                "epic_id": epic_id,
                "packet_source": packet_source,
                "primary_depth": resolved["primary_depth"][0],
                "primary_depth_source_file": resolved["primary_depth"][1],
                "odd_even_depth_ratio": resolved["odd_even_depth_ratio"][0],
                "odd_even_source_file": resolved["odd_even_depth_ratio"][1],
                "oot_to_depth": resolved["oot_to_depth"][0],
                "oot_source_file": resolved["oot_to_depth"][1],
                "secondary_to_primary_depth_ratio": resolved[
                    "secondary_to_primary_depth_ratio"
                ][0],
                "secondary_source_file": resolved[
                    "secondary_to_primary_depth_ratio"
                ][1],
                "alias_risk": resolved["alias_risk"][0],
                "alias_source_file": resolved["alias_risk"][1],
                "missing_reason": ";".join(reasons),
                "fix_needed": fix_needed,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_CSV, index=False)

    complete = audit[
        [
            "primary_depth",
            "odd_even_depth_ratio",
            "oot_to_depth",
            "secondary_to_primary_depth_ratio",
            "alias_risk",
        ]
    ].ne("").all(axis=1)
    stage_i_packet = {
        epic_id: row for epic_id, row in stage_i_rows.items() if epic_id in epic_ids
    }
    saved_period_count = sum(
        bool(clean(row.get("best_period_days"))) for row in stage_i_packet.values()
    )
    historical_period_count = len(period_sources)

    epic = "EPIC_211340132"
    epic_stage_i = stage_i_rows.get(epic, {})
    epic_validation_path = (
        ROOT
        / "plots"
        / "k2_batch"
        / "gatevetter_v0_1_unseen_manual_review"
        / "stage_g_review"
        / epic
        / "validation_summary.json"
    )
    epic_validation = json.loads(epic_validation_path.read_text(encoding="utf-8"))[
        "validation"
    ]
    epic_period_files = ", ".join(rel(path) for path in period_sources.get(epic, []))

    lines = [
        "GateVetter v0.1 unseen null-metric audit",
        "",
        "Root cause",
        "- validation_summary.json is built from the relevant unseen packet CSV row.",
        "- Its diagnostic fields are copied directly from packet_row; they are not recomputed by the visual-summary writer.",
        "- events.csv is used to choose a diagnostic period/family and to populate period_candidates/event_family, but those derived diagnostics are not written back to the metric fields.",
        "- The unseen runner only enriches rows from three Stage I validation ledgers. Most packet EPICs have no row in those ledgers.",
        "",
        "Audit counts",
        f"- packet EPICs: {len(audit)}",
        f"- all five persisted diagnostics available: {int(complete.sum())}",
        f"- one or more diagnostics missing: {int((~complete).sum())}",
        f"- packet EPICs with a full validation-ledger row: {sum(epic_id in validations for epic_id in epic_ids)}",
        f"- packet EPICs with saved best_period_days in Stage I results: {saved_period_count}",
        f"- packet EPICs with at least one historical period_shortlist_best.csv row: {historical_period_count}",
        f"- packet EPICs present in final_candidate_master_ledger.csv: {sum(epic_id in final_ledger_rows for epic_id in epic_ids)}",
        "",
        "Answers",
        "1. Source for validation_summary.json: packet CSV row selected by packet_source; events.csv supplies period/family context and plot inputs.",
        "2. For 3/59 packet EPICs the diagnostics are persisted in Stage I hold/post-repair validation files under recognized aliases. For 56/59 they are not persisted as aggregate metrics, although events/light curves make them calculable after a period is selected.",
        "3. primary_depth is not null because of a wrong alias. The unseen runner correctly checks post_repair_primary_depth and primary_depth. It is null when no validation row supplied the aggregate value. The visual JSON writer then copies that null.",
        "4. odd_even_depth_ratio was not dropped by the blind merge. The runner recognizes post_repair_odd_even_depth_ratio, odd_even_ratio, and odd_even_depth_ratio. For unvalidated rows it was never persisted for the selected period.",
        "5. oot_to_depth was not lost to a column rename. The runner explicitly maps oot_variability_to_depth to oot_to_depth. It is absent where Stage F-style OOT computation was not run/persisted.",
        "6. Blind-mode allowlisting did not remove these metrics. primary_depth, odd_even_depth_ratio, oot_to_depth, secondary metrics, and alias_risk are all in ALLOWED_FEATURES.",
        "7. Yes. 56/59 packet EPICs skipped the full validation step that produces these aggregate diagnostics. Three packet EPICs inherited persisted validation metrics.",
        "",
        f"{epic} trace",
        f"- gatevetter_v0_1_unseen_predictions.csv: primary_depth/odd_even/oot/secondary/alias are null; primary_depth_snr={clean(epic_stage_i.get('primary_depth_snr')) or clean(epic_stage_i.get('best_depth_snr'))}.",
        "- gatevetter_v0_1_unseen_rule_trace.csv: diagnostic values are not included; penalties explicitly report missing odd/even and OOT.",
        "- unseen_stage_g_review_manual_packet.csv: the same aggregate diagnostic fields remain null.",
        f"- validation_summary.json: metric fields remain null; visual period={epic_validation.get('best_period_days')} d, source={epic_validation.get('period_source')}.",
        "- final_candidate_master_ledger.csv: no EPIC_211340132 row.",
        "- Stage I validation metrics/ledgers: no EPIC_211340132 row.",
        f"- Stage I global result: n_periods_validated={clean(epic_stage_i.get('n_periods_validated'))}, best_period_days={clean(epic_stage_i.get('best_period_days')) or 'null'}, prefilter_reason={clean(epic_stage_i.get('prefilter_reason'))}.",
        f"- Historical period-search files: {epic_period_files or 'none'}",
        "- A historical 2026-02-27 period shortlist reports P=9.153408014513843 d and center phase=0.3300506872548972, but that result was not joined into the Stage I unseen source row.",
        "- plots/k2_batch/epics/EPIC_211340132/events.csv contains per-event depth/depth_snr values, not the period-family aggregate diagnostics.",
        "",
        "Diagnostic recomputation probe for EPIC_211340132 (not written into predictions or labels)",
        "- At visual fallback P=3.02388 d: primary_depth=0.007075912997282351; odd_even_depth_ratio=0.9592290389415332; oot_to_depth=0.8693363244485081; secondary_to_primary_depth_ratio=0.0; alias_risk=low.",
        "- At historical P=9.153408014513843 d: primary_depth=0.0065075228479963; odd_even_depth_ratio=0.8494855193943407; oot_to_depth=1.1813924528994741; secondary_to_primary_depth_ratio=0.0; alias_risk=moderate.",
        "- The differing diagnostics show why the period must be confirmed before any backfill.",
        "",
        "Required fix",
        "- Do not change GateVetter v0.1 predictions.",
        "- For missing rows, select or confirm the period used for review, run the existing Stage F-style diagnostic computation, persist the metrics with provenance, and rebuild validation_summary.json from those persisted diagnostics.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"audit_rows={len(audit)}")
    print(f"complete_metric_rows={int(complete.sum())}")
    print(f"missing_metric_rows={int((~complete).sum())}")
    print(f"validation_ledger_rows={sum(epic_id in validations for epic_id in epic_ids)}")


if __name__ == "__main__":
    main()
