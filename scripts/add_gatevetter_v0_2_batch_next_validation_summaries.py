from __future__ import annotations

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

import scripts.build_manual_vetting_next64_plot_pack as plot_pack


PLOT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_batch_next"
MANIFEST = ROOT / "gatevetter_v0_2_batch_next_visual_manifest.csv"
RUN_SUMMARY = ROOT / "gatevetter_v0_2_batch_next_summary.txt"

ARTIFACT_FILES = {
    "raw_light_curve_path": "raw_light_curve.png",
    "detrended_light_curve_path": "detrended_light_curve.png",
    "folded_light_curve_best_period_path": "folded_light_curve_best_period.png",
    "transit_window_zoom_path": "transit_window_zoom.png",
    "event_stack_path": "event_stack.png",
    "secondary_eclipse_check_path": "secondary_eclipse_check.png",
    "odd_even_transits_path": "odd_even_transits.png",
    "periodogram_or_period_search_path": "periodogram_or_period_search.png",
    "oot_variability_check_path": "oot_variability_check.png",
    "summary_panel_path": "summary_panel.png",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
        if math.isfinite(number):
            return int(number) if number.is_integer() else number
    except ValueError:
        pass
    return value


def json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_value(value) for key, value in record.items()}


def validation_row(packet_row: dict[str, Any], rank: int) -> pd.Series:
    return pd.Series(
        {
            "epic_id": clean(packet_row.get("epic_id")),
            "queue_rank": rank,
            "cnn_score": packet_row.get("cnn_score", ""),
            "best_period_days": packet_row.get("best_period_days", ""),
            "transit_duration_hours": packet_row.get(
                "transit_duration_hours", ""
            ),
        }
    )


def artifacts(epic_id: str, epic_dir: Path) -> dict[str, str]:
    paths = {
        key: epic_dir / filename for key, filename in ARTIFACT_FILES.items()
    }
    paths["events_csv"] = plot_pack.EPICS_DIR / epic_id / "events.csv"
    return {
        key: rel(path) if path.exists() and path.stat().st_size > 0 else ""
        for key, path in paths.items()
    }


def write_one(packet: str, rank: int, epic_id: str) -> Path:
    epic_dir = PLOT_ROOT / packet / epic_id
    packet_path = epic_dir / "gatevetter_packet.json"
    packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_row = packet_payload["row"]
    row = validation_row(packet_row, rank)

    events = plot_pack.load_events(epic_id)
    period, center_phase, candidates, selected_source = plot_pack.choose_period(
        row, events
    )
    family = plot_pack.family_events(events, period, center_phase)
    duration = plot_pack.duration_days(row, family, period)
    summary_path = epic_dir / "validation_summary.json"

    validation = {
        "epic_id": epic_id,
        "packet_source": packet,
        "packet_rank": rank,
        "gatevetter_version": "v0.2",
        "gatevetter_prediction": json_value(
            packet_row.get("gatevetter_prediction")
        ),
        "gatevetter_score": json_value(packet_row.get("gatevetter_score")),
        "gatevetter_reason": json_value(
            packet_row.get("gatevetter_v0_2_reason")
            or packet_row.get("gatevetter_v0_reason")
        ),
        "rule_trace": json_value(packet_row.get("rule_trace")),
        "best_period_days": json_value(period),
        "period_source": json_value(
            packet_row.get("period_source") or selected_source
        ),
        "validation_period_source": json_value(
            packet_row.get("validation_period_source")
        ),
        "period_comparison_status": json_value(
            packet_row.get("period_comparison_status")
        ),
        "trusted_period_validation": json_value(
            packet_row.get("trusted_period_validation")
        ),
        "period_ambiguity_flag": json_value(
            packet_row.get("period_ambiguity_flag")
        ),
        "metric_trust_level": json_value(
            packet_row.get("metric_trust_level")
        ),
        "cluster_center_phase": json_value(center_phase),
        "transit_duration_days": json_value(duration),
        "transit_duration_hours": json_value(
            duration * 24.0 if np.isfinite(duration) else None
        ),
        "duration_fraction_of_period": json_value(
            packet_row.get("duration_fraction_of_period")
        ),
        "event_family_count": int(len(family)),
        "candidate_period_count": int(len(candidates)),
        "cnn_score": json_value(packet_row.get("cnn_score")),
        "primary_depth": json_value(packet_row.get("primary_depth")),
        "primary_depth_snr": json_value(
            packet_row.get("primary_depth_snr")
        ),
        "odd_even_depth_ratio": json_value(
            packet_row.get("odd_even_depth_ratio")
        ),
        "odd_even_depth_ratio_missing_reason": json_value(
            packet_row.get("odd_even_depth_ratio_missing_reason")
        ),
        "secondary_depth_snr": json_value(
            packet_row.get("secondary_depth_snr")
        ),
        "secondary_to_primary_depth_ratio": json_value(
            packet_row.get("secondary_to_primary_depth_ratio")
        ),
        "oot_to_depth": json_value(packet_row.get("oot_to_depth")),
        "alias_risk": json_value(packet_row.get("alias_risk")),
        "hard_gate_fired": json_value(packet_row.get("hard_gate_fired")),
        "primary_gate": json_value(packet_row.get("primary_gate")),
        "penalties_or_missing_evidence": json_value(
            packet_row.get("penalties_or_missing_evidence")
        ),
        "manual_label": None,
        "manual_notes": None,
        "validation_summary_json_path": rel(summary_path),
    }
    payload = {
        "validation": validation,
        "packet_row": json_record(packet_row),
        "artifacts": artifacts(epic_id, epic_dir),
        "period_candidates": [
            json_record(record)
            for record in candidates.head(40).to_dict(orient="records")
        ],
        "event_family": [
            json_record(record)
            for record in family.to_dict(orient="records")
        ],
        "notes": [
            "Generated for the GateVetter v0.2 batch-next visual packet.",
            "Visual-review preparation only; GateVetter rules and predictions were not changed.",
            "No manual labels were used or created.",
            "Trusted period-validation fields are preserved from the prediction diagnostics.",
            "Saved or refreshed GateVetter period is preferred; event-spacing fallback is diagnostic only.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    written: list[Path] = []
    summary_paths: dict[tuple[str, str], str] = {}
    for _, row in manifest.sort_values(["packet", "rank"]).iterrows():
        packet = clean(row["packet"])
        epic_id = clean(row["epic_id"])
        path = write_one(
            packet,
            int(float(row["rank"])),
            epic_id,
        )
        written.append(path)
        summary_paths[(packet, epic_id)] = rel(path)
        print(f"Wrote {rel(path)}")
    manifest["validation_summary_json"] = [
        summary_paths[(clean(row["packet"]), clean(row["epic_id"]))]
        for _, row in manifest.iterrows()
    ]
    manifest.to_csv(MANIFEST, index=False)
    if RUN_SUMMARY.exists():
        lines = RUN_SUMMARY.read_text(encoding="utf-8").splitlines()
        lines = [
            line
            for line in lines
            if not line.startswith("validation_summary_json_rows=")
        ]
        insert_at = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if line.startswith("visual_packet_rows=")
            ),
            len(lines),
        )
        lines.insert(insert_at, f"validation_summary_json_rows={len(written)}")
        RUN_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validation_summary_json_written={len(written)}")


if __name__ == "__main__":
    main()
