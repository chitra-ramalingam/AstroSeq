from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_gatevetter_v0_2_batch2 as base


BATCH_SIZE = 100
BASE_PLOTTING_ROW = base.plotting_row
BASE_COMPUTE_DIAGNOSTICS = base.compute_diagnostics
FULL_RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
PUBLISHED_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
PREVIOUS_PREDICTIONS = [
    ROOT / "gatevetter_v0_2_unseen_predictions.csv",
    ROOT / "gatevetter_v0_2_batch2_predictions.csv",
]

OUT_PREDICTIONS = ROOT / "gatevetter_v0_2_batch_next_predictions.csv"
OUT_STAGE_G = ROOT / "gatevetter_v0_2_batch_next_stage_g_queue.csv"
OUT_HOLDS = ROOT / "gatevetter_v0_2_batch_next_hold_queue.csv"
OUT_REJECTS = ROOT / "gatevetter_v0_2_batch_next_reject_summary.csv"
OUT_SUMMARY = ROOT / "gatevetter_v0_2_batch_next_summary.txt"
OUT_SOURCE = ROOT / "gatevetter_v0_2_batch_next_source.csv"
OUT_DIAGNOSTICS = ROOT / "gatevetter_v0_2_batch_next_diagnostics.csv"
OUT_VISUAL_MANIFEST = ROOT / "gatevetter_v0_2_batch_next_visual_manifest.csv"
PLOT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_batch_next"


def extended_review_queue() -> pd.DataFrame:
    results = base.source_runner.load_csv(FULL_RESULTS)
    direct = results.loc[
        results["autovet_label"].isin(
            {"auto_high_priority_candidate", "auto_candidate_with_caveat"}
        )
    ].copy()
    holds = results.loc[
        results["autovet_label"].eq("auto_hold_needs_review")
    ].copy()
    holds = holds.sort_values(
        ["review_priority_score", "autovet_rank_score"],
        ascending=[False, False],
    )
    queue = pd.concat([direct, holds], ignore_index=True)
    order = pd.Categorical(
        queue["autovet_label"],
        categories=[
            "auto_high_priority_candidate",
            "auto_candidate_with_caveat",
            "auto_hold_needs_review",
        ],
        ordered=True,
    )
    queue = (
        queue.assign(_label_order=order)
        .sort_values(
            ["_label_order", "review_priority_score", "autovet_rank_score"],
            ascending=[True, False, False],
        )
        .drop(columns=["_label_order"])
        .reset_index(drop=True)
    )
    queue.insert(0, "source_queue_rank", range(1, len(queue) + 1))

    published = base.source_runner.load_csv(PUBLISHED_QUEUE)
    prefix = queue.head(len(published))["epic_id"].map(base.clean).tolist()
    expected = published["epic_id"].map(base.clean).tolist()
    if prefix != expected:
        raise RuntimeError(
            "Extended review ordering does not reproduce the published queue"
        )
    return queue


def build_source_batch() -> pd.DataFrame:
    queue = extended_review_queue()
    excluded: set[str] = set()
    excluded.update(base.load_ids(base.source_runner.MANUAL_64_DECISIONS))
    excluded.update(base.load_ids(base.source_runner.MANUAL_REVIEW_LEDGER))
    for path in PREVIOUS_PREDICTIONS:
        excluded.update(base.load_ids(path))

    source = queue.loc[
        ~queue["epic_id"].map(base.clean).isin(excluded)
    ].head(BATCH_SIZE).copy()
    if len(source) != BATCH_SIZE:
        raise RuntimeError(
            f"Expected {BATCH_SIZE} fresh unseen EPICs, found {len(source)}"
        )
    if source["epic_id"].map(base.clean).isin(excluded).any():
        raise RuntimeError("Previously reviewed or predicted EPIC leaked into batch")
    if source["epic_id"].duplicated().any():
        raise RuntimeError("Duplicate EPIC in batch-next source")
    return source


def compute_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    if OUT_SOURCE.exists() and OUT_DIAGNOSTICS.exists():
        prior_source = base.source_runner.load_csv(OUT_SOURCE)
        prior_ids = prior_source["epic_id"].map(base.clean).tolist()
        source_ids = source["epic_id"].map(base.clean).tolist()
        if prior_ids == source_ids:
            diagnostics = pd.read_csv(OUT_DIAGNOSTICS, dtype=str).fillna("")
            if diagnostics["epic_id"].map(base.clean).tolist() == source_ids:
                print("Reusing completed batch-next diagnostics", flush=True)
                return diagnostics
    return BASE_COMPUTE_DIAGNOSTICS(source)


def plotting_row(row: pd.Series, rank: int, packet: str) -> pd.Series:
    out = BASE_PLOTTING_ROW(row, rank, packet)
    out["decision_authority"] = "gatevetter_v0_2_batch_next_visual_only"
    return out


def write_summary(
    scored: pd.DataFrame,
    stage_g: pd.DataFrame,
    holds: pd.DataFrame,
    rejects: pd.DataFrame,
    sanity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    visual_manifest: pd.DataFrame,
    rules_hash: str,
) -> None:
    lines = [
        "GateVetter v0.2 batch-next fresh unseen run",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"batch_size={BATCH_SIZE}",
        (
            "source_selection=remaining published review-queue rows followed "
            "by the same ranked hold population after excluding all prior "
            "v0.2 predictions and manual-review ledger EPICs"
        ),
        f"source_queue_rank_first={int(scored['source_source_queue_rank'].min())}",
        f"source_queue_rank_last={int(scored['source_source_queue_rank'].max())}",
        f"rules_file={base.RULES_PATH.name}",
        f"rules_sha256={rules_hash}",
        "rules_changed=false",
        "cnn_retrained=false",
        "gatevetter_thresholds_changed=false",
        "manual_labels_used_during_prediction=false",
        "missing_diagnostics_treated_as_pass=false",
        "trusted_period_required_for_stage_g=true",
        (
            "diagnostic_failures="
            f"{int(diagnostics['diagnostic_error'].fillna('').ne('').sum())}"
        ),
        f"stage_g_queue_rows={len(stage_g)}",
        f"hold_queue_rows={len(holds)}",
        f"reject_summary_rows={len(rejects)}",
        f"reject_sanity_sample_rows={len(sanity)}",
        f"visual_packet_rows={len(visual_manifest)}",
        (
            "visual_packet_failures="
            f"{int((~visual_manifest['success']).sum()) if len(visual_manifest) else 0}"
        ),
        "",
        "Prediction counts",
    ]
    lines.extend(
        f"{label}={count}"
        for label, count in scored["gatevetter_prediction"].value_counts().items()
    )
    lines.extend(["", "Period validation status"])
    lines.extend(
        f"{label}={count}"
        for label, count in scored["period_comparison_status"]
        .value_counts(dropna=False)
        .items()
    )
    lines.extend(
        [
            "",
            "Outputs",
            f"- {OUT_PREDICTIONS.name}",
            f"- {OUT_STAGE_G.name}",
            f"- {OUT_HOLDS.name}",
            f"- {OUT_REJECTS.name}",
            f"- {OUT_SUMMARY.name}",
            f"- {OUT_VISUAL_MANIFEST.name}",
            f"- {PLOT_ROOT.relative_to(ROOT).as_posix()}",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_base() -> None:
    base.BATCH_SIZE = BATCH_SIZE
    base.OUT_PREDICTIONS = OUT_PREDICTIONS
    base.OUT_STAGE_G = OUT_STAGE_G
    base.OUT_HOLDS = OUT_HOLDS
    base.OUT_REJECTS = OUT_REJECTS
    base.OUT_SUMMARY = OUT_SUMMARY
    base.OUT_SOURCE = OUT_SOURCE
    base.OUT_DIAGNOSTICS = OUT_DIAGNOSTICS
    base.OUT_VISUAL_MANIFEST = OUT_VISUAL_MANIFEST
    base.PLOT_ROOT = PLOT_ROOT
    base.build_source_batch = build_source_batch
    base.compute_diagnostics = compute_diagnostics
    base.plotting_row = plotting_row
    base.write_summary = write_summary


def main() -> None:
    configure_base()
    base.main()
    from scripts.add_gatevetter_v0_2_batch_next_validation_summaries import (
        main as add_validation_summaries,
    )

    add_validation_summaries()


if __name__ == "__main__":
    main()
