from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_hold_batch1"
INPUT_LEDGER = BATCH_DIR / "autovet_hold_validation_ledger.csv"
OUT_QUEUE = BATCH_DIR / "period_support_repair_queue.csv"
OUT_SUMMARY = BATCH_DIR / "period_support_repair_queue_summary.txt"

REPAIR_ACTION = "run_period_search_before_label_update"

OUTPUT_COLUMNS = [
    "epic_id",
    "stage_f_validation_label",
    "recommended_ledger_label",
    "recommended_training_label",
    "training_safe",
    "promotion_safe",
    "rejection_safe",
    "human_review_required",
    "label_update_allowed",
    "primary_period",
    "primary_snr",
    "primary_depth",
    "num_events",
    "alias_risk_flag",
    "eb_risk_flag",
    "artifact_risk_flag",
    "recommended_next_action",
    "repair_reason",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    if not INPUT_LEDGER.exists():
        raise FileNotFoundError(f"Missing hold validation ledger: {INPUT_LEDGER}")

    ledger = pd.read_csv(INPUT_LEDGER)
    repair = ledger.loc[ledger["recommended_next_action"].astype(str).eq(REPAIR_ACTION)].copy()
    repair["human_review_required"] = True
    repair["label_update_allowed"] = False
    repair["promotion_safe"] = False
    repair["repair_reason"] = repair.get("reason", "").astype(str)

    for col in OUTPUT_COLUMNS:
        if col not in repair.columns:
            repair[col] = pd.NA
    repair = repair[OUTPUT_COLUMNS]
    repair.to_csv(OUT_QUEUE, index=False)

    total_rows = int(len(ledger))
    repair_rows = int(len(repair))
    nonrepair_rows = total_rows - repair_rows
    promotion_safe = int(repair["promotion_safe"].map(as_bool).sum())
    rejection_safe = int(repair["rejection_safe"].map(as_bool).sum())
    training_safe = int(repair["training_safe"].map(as_bool).sum())
    human_review_required = int(repair["human_review_required"].map(as_bool).sum())
    auto_updates_allowed = int(repair["label_update_allowed"].map(as_bool).sum())

    lines = [
        "Stage I period-support repair queue summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input: {rel(INPUT_LEDGER)}",
        f"Output: {rel(OUT_QUEUE)}",
        "",
        f"Total rows in hold audit ledger: {total_rows}",
        f"Rows needing period-support repair: {repair_rows}",
        f"Rows not needing period-support repair: {nonrepair_rows}",
        f"promotion_safe count: {promotion_safe}",
        f"rejection_safe count: {rejection_safe}",
        f"training_safe count: {training_safe}",
        f"human_review_required count: {human_review_required}",
        f"automatic label updates allowed count: {auto_updates_allowed}",
        "",
        "No training labels, final candidate ledger rows, or model artifacts were modified.",
        "The frozen K2 model remains unchanged: models/k2_nocrop_flux_seed46_split303.best.keras",
        "This is only a repair queue.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Period-support repair queue: {rel(OUT_QUEUE)}")
    print(f"Period-support repair queue summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(repair.to_string(index=False))


if __name__ == "__main__":
    main()
