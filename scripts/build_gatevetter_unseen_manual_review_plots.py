from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_manual_vetting_next64_plot_pack as plot_pack  # noqa: E402

PACKETS = [
    (
        "stage_g_review",
        ROOT / "unseen_stage_g_review_manual_packet.csv",
        ROOT / "plots" / "k2_batch" / "gatevetter_v0_1_unseen_manual_review" / "stage_g_review",
    ),
    (
        "top_holds",
        ROOT / "unseen_top_holds_manual_packet.csv",
        ROOT / "plots" / "k2_batch" / "gatevetter_v0_1_unseen_manual_review" / "top_holds",
    ),
    (
        "reject_sanity_sample",
        ROOT / "unseen_reject_sanity_sample.csv",
        ROOT / "plots" / "k2_batch" / "gatevetter_v0_1_unseen_manual_review" / "reject_sanity_sample",
    ),
]

MANIFEST_CSV = ROOT / "unseen_manual_vetting_plot_manifest.csv"
SUMMARY_TXT = ROOT / "unseen_manual_vetting_plot_summary.txt"

PLOT_COLUMNS = {
    "plot_full_lc_path": "raw_light_curve.png",
    "plot_folded_path": "folded_light_curve_best_period.png",
    "plot_transit_zoom_path": "transit_window_zoom.png",
    "plot_event_stack_path": "event_stack.png",
    "plot_secondary_path": "secondary_eclipse_check.png",
    "plot_odd_even_path": "odd_even_transits.png",
    "plot_periodogram_path": "periodogram_or_period_search.png",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def json_record(row: pd.Series) -> dict[str, Any]:
    return {str(key): json_value(value) for key, value in row.to_dict().items()}


def packet_rank(row: pd.Series, fallback: int) -> int:
    for column in ("manual_packet_rank", "sanity_sample_rank", "source_source_queue_rank", "source_prefilter_rank"):
        value = clean(row.get(column))
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    return fallback


def plot_row(packet_row: pd.Series, rank: int) -> pd.Series:
    return pd.Series(
        {
            "epic_id": clean(packet_row.get("epic_id")),
            "queue_rank": rank,
            "cnn_score": packet_row.get("cnn_score", ""),
            "morphology_positive": "",
            "autovet_label": clean(packet_row.get("source_autovet_label")),
            "explanation_short": clean(packet_row.get("source_prefilter_reason"))
            or clean(packet_row.get("gatevetter_v0_reason")),
            "best_period_days": packet_row.get("best_period_days", ""),
            "master_label": clean(packet_row.get("gatevetter_prediction")),
            "review_level": clean(packet_row.get("stage_g_action")) or "manual_packet_review",
            "decision_authority": "gatevetter_v0_1_visual_packet_only",
            "manual_vetted": clean(packet_row.get("posthoc_manual_vetted")),
        }
    )


def plot_paths(epic_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    missing: list[str] = []
    for column, filename in PLOT_COLUMNS.items():
        path = epic_dir / filename
        if path.exists() and path.stat().st_size > 0:
            out[column] = rel(path)
        else:
            out[column] = ""
            missing.append(column.removeprefix("plot_").removesuffix("_path"))
    out["missing_plot_types"] = "|".join(missing) if missing else "none"
    return out


def build_manifest_row(epic_id: str, packet_source: str, epic_dir: Path) -> dict[str, str]:
    row = {"epic_id": epic_id, "packet_source": packet_source}
    row.update(plot_paths(epic_dir))
    return row


def write_validation_summary(
    packet_source: str,
    packet_row: pd.Series,
    row: pd.Series,
    epic_dir: Path,
) -> Path:
    epic_id = str(row["epic_id"])
    events = plot_pack.load_events(epic_id)
    period, center_phase, candidates, period_source = plot_pack.choose_period(row, events)
    family = plot_pack.family_events(events, period, center_phase)
    duration = plot_pack.duration_days(row, family, period)

    artifact_paths = {
        "raw_light_curve_path": epic_dir / "raw_light_curve.png",
        "detrended_light_curve_path": epic_dir / "detrended_light_curve.png",
        "folded_light_curve_best_period_path": epic_dir / "folded_light_curve_best_period.png",
        "transit_window_zoom_path": epic_dir / "transit_window_zoom.png",
        "event_stack_path": epic_dir / "event_stack.png",
        "secondary_eclipse_check_path": epic_dir / "secondary_eclipse_check.png",
        "odd_even_transits_path": epic_dir / "odd_even_transits.png",
        "periodogram_or_period_search_path": epic_dir / "periodogram_or_period_search.png",
        "oot_variability_check_path": epic_dir / "oot_variability_check.png",
        "summary_panel_path": epic_dir / "summary_panel.png",
        "events_csv": plot_pack.EPICS_DIR / epic_id / "events.csv",
    }
    artifacts = {
        key: rel(path) if path.exists() and path.stat().st_size > 0 else ""
        for key, path in artifact_paths.items()
    }

    summary_path = epic_dir / "validation_summary.json"
    payload = {
        "validation": {
            "epic_id": epic_id,
            "packet_source": packet_source,
            "packet_rank": int(row["queue_rank"]),
            "gatevetter_version": "v0.1",
            "gatevetter_prediction": json_value(packet_row.get("gatevetter_prediction")),
            "gatevetter_score": json_value(packet_row.get("gatevetter_score")),
            "gatevetter_reason": json_value(packet_row.get("gatevetter_v0_reason")),
            "rule_trace": json_value(packet_row.get("rule_trace")),
            "best_period_days": json_value(period),
            "period_source": period_source,
            "cluster_center_phase": json_value(center_phase),
            "transit_duration_days": json_value(duration),
            "transit_duration_hours": json_value(duration * 24.0 if np.isfinite(duration) else None),
            "event_family_count": int(len(family)),
            "candidate_period_count": int(len(candidates)),
            "cnn_score": json_value(packet_row.get("cnn_score")),
            "primary_depth": json_value(packet_row.get("primary_depth")),
            "primary_depth_snr": json_value(packet_row.get("primary_depth_snr")),
            "odd_even_depth_ratio": json_value(packet_row.get("odd_even_depth_ratio")),
            "secondary_depth_snr": json_value(packet_row.get("secondary_depth_snr")),
            "secondary_to_primary_depth_ratio": json_value(
                packet_row.get("secondary_to_primary_depth_ratio")
            ),
            "oot_to_depth": json_value(packet_row.get("oot_to_depth")),
            "alias_risk": json_value(packet_row.get("alias_risk")),
            "hard_gate_fired": json_value(packet_row.get("hard_gate_fired")),
            "manual_label": None,
            "manual_notes": None,
            "validation_summary_json_path": rel(summary_path),
        },
        "packet_row": json_record(packet_row),
        "artifacts": artifacts,
        "period_candidates": [
            {str(key): json_value(value) for key, value in record.items()}
            for record in candidates.head(40).to_dict(orient="records")
        ],
        "event_family": [
            {str(key): json_value(value) for key, value in record.items()}
            for record in family.to_dict(orient="records")
        ],
        "notes": [
            "Generated for the GateVetter v0.1 unseen manual-inspection packet.",
            "Visual-review preparation only; GateVetter rules and predictions were not changed.",
            "No labels were created or updated.",
            "Saved GateVetter/best period is preferred; event-spacing fallback is diagnostic only.",
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main(summaries_only: bool = False) -> None:
    manifest_rows: list[dict[str, str]] = []
    packet_counts: dict[str, int] = {}
    success_count = 0
    failure_count = 0
    missing_by_epic: dict[str, str] = {}

    for packet_source, packet_csv, out_dir in PACKETS:
        if not packet_csv.exists():
            raise FileNotFoundError(packet_csv)

        out_dir.mkdir(parents=True, exist_ok=True)
        packet = pd.read_csv(packet_csv)
        packet_counts[packet_source] = len(packet)

        original_out_dir = plot_pack.OUT_DIR
        plot_pack.OUT_DIR = out_dir
        try:
            for idx, packet_row in packet.iterrows():
                rank = packet_rank(packet_row, idx + 1)
                row = plot_row(packet_row, rank)
                epic_id = str(row["epic_id"])
                epic_dir = out_dir / epic_id

                if not summaries_only:
                    try:
                        plot_pack.build_one(row)
                        success_count += 1
                    except Exception as exc:  # Keep the packet flowing and record the failed EPIC.
                        failure_count += 1
                        epic_dir.mkdir(parents=True, exist_ok=True)
                        missing_by_epic[f"{packet_source}:{epic_id}"] = str(exc)

                epic_dir.mkdir(parents=True, exist_ok=True)
                try:
                    write_validation_summary(packet_source, packet_row, row, epic_dir)
                except Exception as exc:
                    failure_count += 1
                    missing_by_epic[f"{packet_source}:{epic_id}"] = (
                        f"validation_summary.json: {exc}"
                    )

                manifest_row = build_manifest_row(epic_id, packet_source, epic_dir)
                manifest_rows.append(manifest_row)
        finally:
            plot_pack.OUT_DIR = original_out_dir

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "epic_id",
            "packet_source",
            "plot_full_lc_path",
            "plot_folded_path",
            "plot_transit_zoom_path",
            "plot_event_stack_path",
            "plot_secondary_path",
            "plot_odd_even_path",
            "plot_periodogram_path",
            "missing_plot_types",
        ],
    )
    manifest.to_csv(MANIFEST_CSV, index=False)

    complete_rows = int(manifest["missing_plot_types"].eq("none").sum()) if len(manifest) else 0
    incomplete = manifest.loc[~manifest["missing_plot_types"].eq("none")]

    lines = [
        "GateVetter v0.1 unseen manual-vetting plot summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "Scope",
        "- Visual review material preparation only.",
        "- GateVetter rules were not changed.",
        "- Predictions were not changed.",
        "- Labels were not updated.",
        "",
        "Packet row counts",
    ]
    for packet_source, count in packet_counts.items():
        lines.append(f"- {packet_source}: {count}")
    lines.extend(
        [
            "",
            f"manifest_rows: {len(manifest)}",
            f"complete_plot_rows: {complete_rows}",
            f"incomplete_plot_rows: {len(incomplete)}",
            f"plot_build_successes: {success_count if not summaries_only else 'not_rerun'}",
            f"plot_build_failures: {failure_count}",
            f"validation_summary_json_count: {sum(1 for _, _, out_dir in PACKETS for _ in out_dir.glob('*/validation_summary.json'))}",
            f"output_root: {rel(ROOT / 'plots' / 'k2_batch' / 'gatevetter_v0_1_unseen_manual_review')}",
            f"manifest_csv: {rel(MANIFEST_CSV)}",
            "",
            "Missing plot rows",
        ]
    )
    if len(incomplete) == 0:
        lines.append("- none")
    else:
        for _, row in incomplete.iterrows():
            key = f"{row['packet_source']}:{row['epic_id']}"
            reason = missing_by_epic.get(key, "")
            suffix = f"; build_error={reason}" if reason else ""
            lines.append(f"- {key}: {row['missing_plot_types']}{suffix}")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"manifest_rows={len(manifest)}")
    print(f"complete_plot_rows={complete_rows}")
    print(f"incomplete_plot_rows={len(incomplete)}")
    print(f"plot_build_successes={success_count if not summaries_only else 'not_rerun'}")
    print(f"plot_build_failures={failure_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Write validation_summary.json files and refresh manifests without regenerating plots.",
    )
    args = parser.parse_args()
    main(summaries_only=args.summaries_only)
