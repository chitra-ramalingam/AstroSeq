from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_candidate_batch1"
INPUT_LEDGER = BATCH_DIR / "autovet_candidate_validation_ledger.csv"
OUT_CSV = BATCH_DIR / "pending_candidate_label_updates.csv"
OUT_SUMMARY = BATCH_DIR / "pending_candidate_label_updates_summary.txt"

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
    "recommended_next_action",
    "reason",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    if not INPUT_LEDGER.exists():
        raise FileNotFoundError(f"Missing candidate validation ledger: {INPUT_LEDGER}")

    ledger = pd.read_csv(INPUT_LEDGER)
    pending = ledger.copy()
    pending["human_review_required"] = True
    pending["label_update_allowed"] = False

    for col in OUTPUT_COLUMNS:
        if col not in pending.columns:
            pending[col] = pd.NA
    pending = pending[OUTPUT_COLUMNS]
    pending.to_csv(OUT_CSV, index=False)

    promotion_safe = int(pending["promotion_safe"].map(as_bool).sum())
    human_review = int(pending["human_review_required"].map(as_bool).sum())
    auto_allowed = int(pending["label_update_allowed"].map(as_bool).sum())

    lines = [
        "Stage I pending candidate label updates summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input: {rel(INPUT_LEDGER)}",
        f"Output: {rel(OUT_CSV)}",
        "",
        f"Candidate-with-caveat rows processed: {len(pending)}",
        f"Promotion-safe rows: {promotion_safe}",
        f"Rows requiring human review: {human_review}",
        f"Automatic label updates allowed: {auto_allowed}",
        "",
        "No training labels, final candidate ledger rows, or model artifacts were modified.",
        "This file is a pending human-review handoff only.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Pending candidate label updates: {rel(OUT_CSV)}")
    print(f"Pending candidate label updates summary: {rel(OUT_SUMMARY)}")
    print()
    print(OUT_SUMMARY.read_text(encoding="utf-8"))
    print(pending.to_string(index=False))


if __name__ == "__main__":
    main()
