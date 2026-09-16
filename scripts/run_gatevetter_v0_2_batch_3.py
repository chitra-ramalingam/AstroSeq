from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gatevetter_v0_2_rules as rules
import scripts.run_gatevetter_v0_2_batch2 as base


BATCH_SIZE = 100
BASE_PLOTTING_ROW = base.plotting_row
BASE_COMPUTE_DIAGNOSTICS = base.compute_diagnostics
FULL_RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
PUBLISHED_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
PREVIOUS_PREDICTIONS = [
    ROOT / "gatevetter_v0_2_unseen_predictions.csv",
    ROOT / "gatevetter_v0_2_batch2_predictions.csv",
    ROOT / "gatevetter_v0_2_batch_next_predictions.csv",
]

OUT_PREDICTIONS = ROOT / "gatevetter_v0_2_batch_3_predictions.csv"
OUT_STAGE_G = ROOT / "gatevetter_v0_2_batch_3_stage_g_queue.csv"
OUT_TOP_HOLDS = ROOT / "gatevetter_v0_2_batch_3_top_20_holds.csv"
OUT_REJECTS = ROOT / "gatevetter_v0_2_batch_3_reject_summary.csv"
OUT_SUMMARY = ROOT / "gatevetter_v0_2_batch_3_summary.txt"
OUT_SOURCE = ROOT / "gatevetter_v0_2_batch_3_source.csv"
OUT_DIAGNOSTICS = ROOT / "gatevetter_v0_2_batch_3_diagnostics.csv"
OUT_VISUAL_MANIFEST = ROOT / "gatevetter_v0_2_batch_3_visual_manifest.csv"
OUT_REJECT_SANITY = ROOT / "gatevetter_v0_2_batch_3_reject_sanity_sample.csv"
PLOT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_batch_3"

PACKET_STAGE_G = "stage_g_queue"
PACKET_HOLDS = "top_20_holds"
PACKET_REJECTS = "reject_sanity_high_cnn_high_snr"
REJECT_GROUP_SIZE = 5


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
        raise RuntimeError("Previously reviewed or predicted EPIC leaked into batch 3")
    if source["epic_id"].duplicated().any():
        raise RuntimeError("Duplicate EPIC in batch-3 source")
    return source


def compute_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    if OUT_SOURCE.exists() and OUT_DIAGNOSTICS.exists():
        prior_source = base.source_runner.load_csv(OUT_SOURCE)
        prior_ids = prior_source["epic_id"].map(base.clean).tolist()
        source_ids = source["epic_id"].map(base.clean).tolist()
        if prior_ids == source_ids:
            diagnostics = pd.read_csv(OUT_DIAGNOSTICS, dtype=str).fillna("")
            if diagnostics["epic_id"].map(base.clean).tolist() == source_ids:
                print("Reusing completed batch-3 diagnostics", flush=True)
                return diagnostics
    return BASE_COMPUTE_DIAGNOSTICS(source)


def plotting_row(row: pd.Series, rank: int, packet: str) -> pd.Series:
    out = BASE_PLOTTING_ROW(row, rank, packet)
    out["decision_authority"] = "gatevetter_v0_2_batch_3_visual_only"
    return out


def top_holds(scored: pd.DataFrame) -> pd.DataFrame:
    return base.build_holds(scored).head(20).reset_index(drop=True)


def reject_sanity_sample(scored: pd.DataFrame) -> pd.DataFrame:
    rejects = scored.loc[
        ~scored["gatevetter_prediction"].isin(
            rules.STAGE_G_PREDICTIONS
            | {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
        )
    ].copy()
    rejects["cnn_numeric"] = pd.to_numeric(
        rejects["cnn_score"], errors="coerce"
    )
    rejects["snr_numeric"] = pd.to_numeric(
        rejects["primary_depth_snr"], errors="coerce"
    )
    eligible = rejects.loc[
        rejects["cnn_numeric"].notna() & rejects["snr_numeric"].notna()
    ].copy()
    if eligible.empty:
        return eligible

    eligible["cnn_percentile"] = eligible["cnn_numeric"].rank(pct=True)
    eligible["snr_percentile"] = eligible["snr_numeric"].rank(pct=True)
    eligible["reject_sanity_score"] = (
        eligible["cnn_percentile"] + eligible["snr_percentile"]
    )

    eb = eligible.loc[
        eligible["gatevetter_v0_2_reason"].eq("eb_or_variable_gate")
    ].copy()
    eb = eb.sort_values(
        ["reject_sanity_score", "cnn_numeric", "snr_numeric"],
        ascending=[False, False, False],
    ).head(REJECT_GROUP_SIZE)
    eb.insert(0, "packet_group", "eb_variable_rejects_all")
    eb.insert(1, "group_rank", range(1, len(eb) + 1))

    weak = eligible.loc[
        eligible["gatevetter_v0_2_reason"].eq(
            "weak_or_incomplete_signal_after_gates"
        )
    ].copy()
    weak = weak.sort_values(
        ["reject_sanity_score", "cnn_numeric", "snr_numeric"],
        ascending=[False, False, False],
    ).head(REJECT_GROUP_SIZE)
    weak.insert(0, "packet_group", "weak_incomplete_rejects_top5_by_cnn_snr")
    weak.insert(1, "group_rank", range(1, len(weak) + 1))

    sample = pd.concat([eb, weak], ignore_index=True)
    sample.insert(0, "manual_packet_rank", range(1, len(sample) + 1))
    sample.insert(3, "sanity_sample_rank", range(1, len(sample) + 1))
    return sample


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
        "GateVetter v0.2 batch-3 fresh unseen run",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"batch_size={BATCH_SIZE}",
        (
            "source_selection=remaining published review-queue rows followed "
            "by the same ranked hold population after excluding all prior "
            "v0.2 unseen/batch predictions and manual-review ledger EPICs"
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
        f"top_20_holds_rows={len(holds)}",
        f"reject_summary_rows={len(rejects)}",
        f"reject_sanity_sample_rows={len(sanity)}",
        (
            "reject_sanity_eb_variable_rows="
            f"{int(sanity['packet_group'].eq('eb_variable_rejects_all').sum()) if len(sanity) else 0}"
        ),
        (
            "reject_sanity_weak_high_cnn_high_snr_rows="
            f"{int(sanity['packet_group'].eq('weak_incomplete_rejects_top5_by_cnn_snr').sum()) if len(sanity) else 0}"
        ),
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
            f"- {OUT_TOP_HOLDS.name}",
            f"- {OUT_REJECTS.name}",
            f"- {OUT_SUMMARY.name}",
            f"- {OUT_VISUAL_MANIFEST.name}",
            f"- {OUT_REJECT_SANITY.name}",
            f"- {PLOT_ROOT.relative_to(ROOT).as_posix()}",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rules_hash = base.sha256(base.RULES_PATH)
    source = build_source_batch()
    source.to_csv(OUT_SOURCE, index=False)

    diagnostics = compute_diagnostics(source)
    diagnostics.to_csv(OUT_DIAGNOSTICS, index=False)
    features = base.build_features(source, diagnostics)
    predictions = pd.DataFrame(
        [rules.gatevet(row) for _, row in features.iterrows()]
    )
    scored = base.source_runner.add_source_context(predictions, source)

    stage_g = base.build_stage_g(scored)
    holds = top_holds(scored)
    rejects = base.build_reject_summary(scored)
    sanity = reject_sanity_sample(scored)
    base.validate(source, features, scored, stage_g, rules_hash)

    scored.to_csv(OUT_PREDICTIONS, index=False)
    stage_g.to_csv(OUT_STAGE_G, index=False)
    holds.to_csv(OUT_TOP_HOLDS, index=False)
    rejects.to_csv(OUT_REJECTS, index=False)
    sanity.to_csv(OUT_REJECT_SANITY, index=False)

    original_plot_root = base.PLOT_ROOT
    original_plotting_row = base.plotting_row
    base.PLOT_ROOT = PLOT_ROOT
    base.plotting_row = plotting_row
    try:
        visual_rows: list[dict[str, Any]] = []
        visual_rows.extend(base.generate_visual_packet(PACKET_STAGE_G, stage_g))
        visual_rows.extend(base.generate_visual_packet(PACKET_HOLDS, holds))
        visual_rows.extend(base.generate_visual_packet(PACKET_REJECTS, sanity))
    finally:
        base.PLOT_ROOT = original_plot_root
        base.plotting_row = original_plotting_row

    visual_manifest = pd.DataFrame(visual_rows)
    visual_manifest.to_csv(OUT_VISUAL_MANIFEST, index=False)

    write_summary(
        scored,
        stage_g,
        holds,
        rejects,
        sanity,
        diagnostics,
        visual_manifest,
        rules_hash,
    )

    from scripts.add_gatevetter_v0_2_batch_3_validation_summaries import (
        main as add_validation_summaries,
    )

    add_validation_summaries()

    print(f"Wrote {OUT_PREDICTIONS.name} ({len(scored)} rows)")
    print(f"Wrote {OUT_STAGE_G.name} ({len(stage_g)} rows)")
    print(f"Wrote {OUT_TOP_HOLDS.name} ({len(holds)} rows)")
    print(f"Wrote {OUT_REJECTS.name} ({len(rejects)} rows)")
    print(f"Wrote {OUT_REJECT_SANITY.name} ({len(sanity)} rows)")
    print(f"Wrote {OUT_SUMMARY.name}")


if __name__ == "__main__":
    main()
