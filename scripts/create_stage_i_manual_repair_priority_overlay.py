from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"
INPUT_QUEUE = BATCH_DIR / "period_support_repair_queue.csv"
OUT_OVERLAY = BATCH_DIR / "manual_repair_priority_overlay.csv"
OUT_SUMMARY = BATCH_DIR / "manual_repair_priority_overlay_summary.txt"

OUTPUT_COLUMNS = [
    "epic_id",
    "manual_priority",
    "manual_note",
    "human_review_required",
    "label_update_allowed",
]

MANUAL_OVERLAY = {
    "EPIC_211839462": {
        "manual_priority": "high",
        "manual_note": "promising repair target; no EB/alias/artifact flags; good event count",
    },
    "EPIC_212024647": {
        "manual_priority": "high",
        "manual_note": "promising repair target; no EB/alias/artifact flags; good event count",
    },
    "EPIC_211682657": {
        "manual_priority": "medium_caution",
        "manual_note": "possible signal but EB/alias caution; secondary_ratio=0.5734 and alias_risk moderate",
    },
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    if not INPUT_QUEUE.exists():
        raise FileNotFoundError(f"Missing period-support repair queue: {INPUT_QUEUE}")

    queue = pd.read_csv(INPUT_QUEUE)
    if "epic_id" not in queue.columns:
        raise ValueError(f"Missing epic_id column in {INPUT_QUEUE}")

    overlay = pd.DataFrame({"epic_id": queue["epic_id"].astype(str)})
    overlay["manual_priority"] = overlay["epic_id"].map(
        lambda epic_id: MANUAL_OVERLAY.get(epic_id, {}).get("manual_priority", "normal")
    )
    overlay["manual_note"] = overlay["epic_id"].map(
        lambda epic_id: MANUAL_OVERLAY.get(epic_id, {}).get(
            "manual_note", "not manually prioritized in this pass"
        )
    )
    overlay["human_review_required"] = True
    overlay["label_update_allowed"] = False
    overlay = overlay[OUTPUT_COLUMNS]
    overlay.to_csv(OUT_OVERLAY, index=False)

    high_epics = overlay.loc[overlay["manual_priority"].eq("high"), "epic_id"].tolist()
    caution_epics = overlay.loc[
        overlay["manual_priority"].eq("medium_caution"), "epic_id"
    ].tolist()

    lines = [
        "Stage I manual repair priority overlay summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input repair queue: {rel(INPUT_QUEUE)}",
        f"Output overlay: {rel(OUT_OVERLAY)}",
        "",
        f"Total repair rows: {len(overlay)}",
        f"High priority count: {int(overlay['manual_priority'].eq('high').sum())}",
        f"medium_caution count: {int(overlay['manual_priority'].eq('medium_caution').sum())}",
        f"normal count: {int(overlay['manual_priority'].eq('normal').sum())}",
        f"High priority EPICs: {', '.join(high_epics) if high_epics else 'none'}",
        f"medium_caution EPICs: {', '.join(caution_epics) if caution_epics else 'none'}",
        "",
        "This is a manual overlay only.",
        "No rows were promoted or rejected.",
        "No recommended_training_label values were changed.",
        "No recommended_ledger_label values were changed.",
        "No training labels, final candidate ledger rows, or model artifacts were modified.",
        "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Manual repair priority overlay: {rel(OUT_OVERLAY)}")
    print(f"Manual repair priority overlay summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(overlay.to_string(index=False))


if __name__ == "__main__":
    main()
