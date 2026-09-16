from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gatevetter_v0_2_rules as rules
import scripts.add_gatevetter_v0_2_batch_next_validation_summaries as validation_summaries
import scripts.run_gatevetter_v0_2_batch2 as base


BATCH_SIZE = 100
FULL_RESULTS = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_results.csv"
PUBLISHED_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
ROLLING_SUMMARY = ROOT / "gatevetter_v0_2_rolling_batch_summary.csv"
BATCH_STOP_MARKER = ROOT / "freezes" / "GATEVETTER_V0_2_BATCHES_STOPPED.txt"
FINAL_LEDGER_DIRS = [ROOT / "plots" / "k2_batch", ROOT / "freezes"]
MANUAL_DIRS = [ROOT, ROOT / "freezes", ROOT / "plots" / "k2_batch"]
PACKET_STAGE_G = "stage_g_queue"
PACKET_HOLDS = "top_20_holds"
PACKET_REJECTS = "reject_sanity_high_cnn_high_snr"
REJECT_GROUP_SIZE = 5


def existing_numeric_batch_ids() -> list[int]:
    out: list[int] = []
    for path in ROOT.glob("gatevetter_v0_2_batch*_predictions.csv"):
        name = path.name
        match = re.fullmatch(r"gatevetter_v0_2_batch_?(\d+)_predictions\.csv", name)
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def default_next_batch_id() -> int:
    existing = existing_numeric_batch_ids()
    return (max(existing) + 1) if existing else 1


def clean(value: Any) -> str:
    return base.clean(value)


def load_ids(path: Path) -> set[str]:
    return base.load_ids(path)


def csv_ids_from_paths(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        try:
            ids.update(load_ids(path))
        except Exception as exc:
            print(f"Skipping exclusion file {path}: {exc}", flush=True)
    return ids


def prior_prediction_paths(out_predictions: Path) -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob("gatevetter_v0_2*predictions.csv")
        if path.resolve() != out_predictions.resolve()
    )


def manual_review_paths() -> list[Path]:
    patterns = [
        "*manual*decision*.csv",
        "*manual*reviewed*.csv",
        "*manual_review_outcomes*.csv",
        "*manual_vetting_decisions*.csv",
        "*validation_ledger.csv",
    ]
    paths: set[Path] = {
        base.source_runner.MANUAL_64_DECISIONS,
        base.source_runner.MANUAL_REVIEW_LEDGER,
    }
    for root in MANUAL_DIRS:
        if not root.exists():
            continue
        for pattern in patterns:
            paths.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(paths)


def final_ledger_paths() -> list[Path]:
    paths: set[Path] = set()
    for root in FINAL_LEDGER_DIRS:
        if not root.exists():
            continue
        paths.update(root.glob("final_candidate_master_ledger*.csv"))
    paths.update(ROOT.glob("k2_master_candidate_ledger*.csv"))
    return sorted(paths)


def exclusion_ids(out_predictions: Path) -> tuple[set[str], dict[str, int]]:
    prior_predictions = prior_prediction_paths(out_predictions)
    manual_paths = manual_review_paths()
    final_paths = final_ledger_paths()
    groups = {
        "prior_prediction": csv_ids_from_paths(prior_predictions),
        "manual_review": csv_ids_from_paths(manual_paths),
        "final_ledger": csv_ids_from_paths(final_paths),
    }
    combined: set[str] = set()
    for ids in groups.values():
        combined.update(ids)
    return combined, {key: len(value) for key, value in groups.items()}


def extended_review_queue() -> pd.DataFrame:
    results = base.source_runner.load_csv(FULL_RESULTS)
    direct = results.loc[
        results["autovet_label"].isin(
            {"auto_high_priority_candidate", "auto_candidate_with_caveat"}
        )
    ].copy()
    holds = results.loc[results["autovet_label"].eq("auto_hold_needs_review")].copy()
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
    prefix = queue.head(len(published))["epic_id"].map(clean).tolist()
    expected = published["epic_id"].map(clean).tolist()
    if prefix != expected:
        raise RuntimeError("Extended review ordering does not reproduce the published queue")
    return queue


class BatchPaths:
    def __init__(self, batch_id: int) -> None:
        self.batch_id = batch_id
        self.prefix = f"gatevetter_v0_2_batch_{batch_id}"
        self.predictions = ROOT / f"{self.prefix}_predictions.csv"
        self.stage_g = ROOT / f"{self.prefix}_stage_g_queue.csv"
        self.top_holds = ROOT / f"{self.prefix}_top_20_holds.csv"
        self.rejects = ROOT / f"{self.prefix}_reject_summary.csv"
        self.summary = ROOT / f"{self.prefix}_summary.txt"
        self.source = ROOT / f"{self.prefix}_source.csv"
        self.diagnostics = ROOT / f"{self.prefix}_diagnostics.csv"
        self.visual_manifest = ROOT / f"{self.prefix}_visual_manifest.csv"
        self.reject_sanity = ROOT / f"{self.prefix}_reject_sanity_sample.csv"
        self.plot_root = ROOT / "plots" / "k2_batch" / self.prefix


def build_source_batch(paths: BatchPaths) -> tuple[pd.DataFrame, dict[str, int]]:
    queue = extended_review_queue()
    excluded, exclusion_counts = exclusion_ids(paths.predictions)
    source = queue.loc[~queue["epic_id"].map(clean).isin(excluded)].head(BATCH_SIZE).copy()
    if len(source) != BATCH_SIZE:
        raise RuntimeError(f"Expected {BATCH_SIZE} fresh unseen EPICs, found {len(source)}")
    if source["epic_id"].map(clean).isin(excluded).any():
        raise RuntimeError(f"Excluded EPIC leaked into batch {paths.batch_id}")
    if source["epic_id"].duplicated().any():
        raise RuntimeError(f"Duplicate EPIC in batch {paths.batch_id} source")
    return source, exclusion_counts


def compute_diagnostics(paths: BatchPaths, source: pd.DataFrame) -> pd.DataFrame:
    if paths.source.exists() and paths.diagnostics.exists():
        prior_source = base.source_runner.load_csv(paths.source)
        source_ids = source["epic_id"].map(clean).tolist()
        prior_ids = prior_source["epic_id"].map(clean).tolist()
        if prior_ids == source_ids:
            diagnostics = pd.read_csv(paths.diagnostics, dtype=str).fillna("")
            if diagnostics["epic_id"].map(clean).tolist() == source_ids:
                print(f"Reusing completed batch-{paths.batch_id} diagnostics", flush=True)
                return diagnostics
    return base.compute_diagnostics(source)


def top_holds(scored: pd.DataFrame) -> pd.DataFrame:
    return base.build_holds(scored).head(20).reset_index(drop=True)


def reject_sanity_sample(scored: pd.DataFrame) -> pd.DataFrame:
    rejects = scored.loc[
        ~scored["gatevetter_prediction"].isin(
            rules.STAGE_G_PREDICTIONS
            | {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
        )
    ].copy()
    rejects["cnn_numeric"] = pd.to_numeric(rejects["cnn_score"], errors="coerce")
    rejects["snr_numeric"] = pd.to_numeric(rejects["primary_depth_snr"], errors="coerce")
    eligible = rejects.loc[
        rejects["cnn_numeric"].notna() & rejects["snr_numeric"].notna()
    ].copy()
    if eligible.empty:
        return eligible
    eligible["cnn_percentile"] = eligible["cnn_numeric"].rank(pct=True)
    eligible["snr_percentile"] = eligible["snr_numeric"].rank(pct=True)
    eligible["reject_sanity_score"] = eligible["cnn_percentile"] + eligible["snr_percentile"]

    samples: list[pd.DataFrame] = []
    groups = [
        ("eb_variable_rejects_all", "eb_or_variable_gate"),
        ("weak_incomplete_rejects_top5_by_cnn_snr", "weak_or_incomplete_signal_after_gates"),
    ]
    for group_name, reason in groups:
        frame = eligible.loc[eligible["gatevetter_v0_2_reason"].eq(reason)].copy()
        frame = frame.sort_values(
            ["reject_sanity_score", "cnn_numeric", "snr_numeric"],
            ascending=[False, False, False],
        ).head(REJECT_GROUP_SIZE)
        frame.insert(0, "packet_group", group_name)
        frame.insert(1, "group_rank", range(1, len(frame) + 1))
        samples.append(frame)
    sample = pd.concat(samples, ignore_index=True)
    sample.insert(0, "manual_packet_rank", range(1, len(sample) + 1))
    sample.insert(3, "sanity_sample_rank", range(1, len(sample) + 1))
    return sample


def plotting_row(
    original_plotting_row: Any,
    batch_id: int,
    row: pd.Series,
    rank: int,
    packet: str,
) -> pd.Series:
    out = original_plotting_row(row, rank, packet)
    out["decision_authority"] = f"gatevetter_v0_2_batch_{batch_id}_visual_only"
    return out


def generate_visual_manifest(
    paths: BatchPaths,
    stage_g: pd.DataFrame,
    holds: pd.DataFrame,
    sanity: pd.DataFrame,
) -> pd.DataFrame:
    original_plot_root = base.PLOT_ROOT
    original_plotting_row = base.plotting_row
    base.PLOT_ROOT = paths.plot_root
    base.plotting_row = lambda row, rank, packet: plotting_row(
        original_plotting_row, paths.batch_id, row, rank, packet
    )
    try:
        visual_rows: list[dict[str, Any]] = []
        visual_rows.extend(base.generate_visual_packet(PACKET_STAGE_G, stage_g))
        visual_rows.extend(base.generate_visual_packet(PACKET_HOLDS, holds))
        visual_rows.extend(base.generate_visual_packet(PACKET_REJECTS, sanity))
        return pd.DataFrame(visual_rows)
    finally:
        base.PLOT_ROOT = original_plot_root
        base.plotting_row = original_plotting_row


def write_summary(
    paths: BatchPaths,
    scored: pd.DataFrame,
    stage_g: pd.DataFrame,
    holds: pd.DataFrame,
    rejects: pd.DataFrame,
    sanity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    visual_manifest: pd.DataFrame,
    rules_hash: str,
    exclusion_counts: dict[str, int],
) -> None:
    lines = [
        f"GateVetter v0.2 batch-{paths.batch_id} fresh unseen run",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"batch_size={BATCH_SIZE}",
        (
            "source_selection=remaining published review-queue rows followed by "
            "the same ranked hold population after excluding all prior v0.2 "
            "unseen/batch predictions, manually reviewed EPICs, and final ledger EPICs"
        ),
        f"source_queue_rank_first={int(scored['source_source_queue_rank'].min())}",
        f"source_queue_rank_last={int(scored['source_source_queue_rank'].max())}",
        f"excluded_prior_prediction_epics={exclusion_counts['prior_prediction']}",
        f"excluded_manual_review_epics={exclusion_counts['manual_review']}",
        f"excluded_final_ledger_epics={exclusion_counts['final_ledger']}",
        f"rules_file={base.RULES_PATH.name}",
        f"rules_sha256={rules_hash}",
        "rules_changed=false",
        "cnn_retrained=false",
        "gatevetter_thresholds_changed=false",
        "manual_labels_used_during_prediction=false",
        "missing_diagnostics_treated_as_pass=false",
        "trusted_period_required_for_stage_g=true",
        f"diagnostic_failures={int(diagnostics['diagnostic_error'].fillna('').ne('').sum())}",
        f"stage_g_queue_rows={len(stage_g)}",
        f"top_20_holds_rows={len(holds)}",
        f"reject_epic_rows={reject_count(scored)}",
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
        f"visual_packet_failures={int((~visual_manifest['success']).sum()) if len(visual_manifest) else 0}",
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
        for label, count in scored["period_comparison_status"].value_counts(dropna=False).items()
    )
    lines.extend(
        [
            "",
            "Outputs",
            f"- {paths.predictions.name}",
            f"- {paths.stage_g.name}",
            f"- {paths.top_holds.name}",
            f"- {paths.rejects.name}",
            f"- {paths.reject_sanity.name}",
            f"- {paths.summary.name}",
            f"- {paths.visual_manifest.name}",
            f"- {paths.plot_root.relative_to(ROOT).as_posix()}",
        ]
    )
    paths.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reject_mask(scored: pd.DataFrame) -> pd.Series:
    return ~scored["gatevetter_prediction"].isin(
        rules.STAGE_G_PREDICTIONS
        | {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
    )


def reject_count(scored: pd.DataFrame) -> int:
    return int(reject_mask(scored).sum())


def add_validation_summaries(paths: BatchPaths) -> None:
    validation_summaries.PLOT_ROOT = paths.plot_root
    validation_summaries.MANIFEST = paths.visual_manifest
    validation_summaries.RUN_SUMMARY = paths.summary
    validation_summaries.main()


def manual_decision_counts(prefix: str) -> dict[str, int]:
    paths = sorted(ROOT.glob(f"{prefix}_manual*review*decisions.csv"))
    if not paths:
        return {
            "candidate_like": 0,
            "uncertain_hold_positive": 0,
            "period_ambiguous_hold": 0,
            "reject": 0,
        }
    labels_by_epic: dict[str, str] = {}
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "epic_id" in frame.columns and "manual_label" in frame.columns:
            for _, row in frame.iterrows():
                epic_id = clean(row.get("epic_id"))
                label = clean(row.get("manual_label"))
                if epic_id and label:
                    labels_by_epic[epic_id] = label
    labels = list(labels_by_epic.values())
    return {
        "candidate_like": sum(label == "candidate_like" for label in labels),
        "uncertain_hold_positive": sum(label == "uncertain_hold_positive" for label in labels),
        "period_ambiguous_hold": sum(label == "uncertain_hold_period_ambiguous" for label in labels),
        "reject": sum(
            bool(label)
            and label
            not in {
                "candidate_like",
                "uncertain_hold_positive",
                "uncertain_hold_period_ambiguous",
            }
            for label in labels
        ),
    }


def rolling_row(paths: BatchPaths) -> dict[str, Any]:
    scored = pd.read_csv(paths.predictions, dtype=str).fillna("")
    stage_g = pd.read_csv(paths.stage_g, dtype=str).fillna("")
    holds = pd.read_csv(paths.top_holds, dtype=str).fillna("")
    manual_counts = manual_decision_counts(paths.prefix)
    trusted = scored["trusted_period_validation"].map(lambda value: clean(value).lower() == "true")
    ambiguous = (
        scored["period_ambiguity_flag"].map(lambda value: clean(value).lower() in {"true", "1", "yes"})
        if "period_ambiguity_flag" in scored.columns
        else pd.Series([False] * len(scored))
    )
    notes = []
    if not any(manual_counts.values()):
        notes.append("manual_review_pending")
    return {
        "batch_id": paths.prefix,
        "batch_size": len(scored),
        "stage_g_count": len(stage_g),
        "hold_count": len(holds),
        "reject_count": reject_count(scored),
        "eb_variable_count": int(scored["gatevetter_prediction"].eq("false_positive_eb_or_variable").sum()),
        "noise_count": int(scored["gatevetter_prediction"].isin(["negative_noise_or_artifact", "negative_reject_as_noise_or_artifact"]).sum()),
        "trusted_period_count": int(trusted.sum()),
        "period_ambiguous_count": int(ambiguous.sum()),
        "manual_candidate_like_count": manual_counts["candidate_like"],
        "manual_uncertain_hold_positive_count": manual_counts["uncertain_hold_positive"],
        "manual_period_ambiguous_hold_count": manual_counts["period_ambiguous_hold"],
        "manual_reject_count": manual_counts["reject"],
        "notes": "; ".join(notes),
    }


def update_rolling_summary() -> None:
    rows: list[dict[str, Any]] = []
    for batch_id in existing_numeric_batch_ids():
        paths = BatchPaths(batch_id)
        if paths.predictions.exists() and paths.stage_g.exists() and paths.top_holds.exists():
            rows.append(rolling_row(paths))
    columns = [
        "batch_id",
        "batch_size",
        "stage_g_count",
        "hold_count",
        "reject_count",
        "eb_variable_count",
        "noise_count",
        "trusted_period_count",
        "period_ambiguous_count",
        "manual_candidate_like_count",
        "manual_uncertain_hold_positive_count",
        "manual_period_ambiguous_hold_count",
        "manual_reject_count",
        "notes",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(ROLLING_SUMMARY, index=False)


def run_batch(batch_id: int) -> BatchPaths:
    if BATCH_STOP_MARKER.exists():
        raise RuntimeError(
            "New GateVetter v0.2 batches are stopped pending deep review. "
            f"See {BATCH_STOP_MARKER}."
        )
    paths = BatchPaths(batch_id)
    rules_hash = base.sha256(base.RULES_PATH)
    core_paths = [
        paths.source,
        paths.diagnostics,
        paths.predictions,
        paths.stage_g,
        paths.top_holds,
        paths.rejects,
        paths.reject_sanity,
    ]
    if all(path.exists() for path in core_paths):
        print(f"Resuming batch-{batch_id} from existing core CSVs", flush=True)
        source = pd.read_csv(paths.source, dtype=str).fillna("")
        diagnostics = pd.read_csv(paths.diagnostics, dtype=str).fillna("")
        features = base.build_features(source, diagnostics)
        scored = pd.read_csv(paths.predictions, dtype=str).fillna("")
        stage_g = pd.read_csv(paths.stage_g, dtype=str).fillna("")
        holds = pd.read_csv(paths.top_holds, dtype=str).fillna("")
        rejects = pd.read_csv(paths.rejects, dtype=str).fillna("")
        sanity = pd.read_csv(paths.reject_sanity, dtype=str).fillna("")
        _, exclusion_counts = exclusion_ids(paths.predictions)
        base.validate(source, features, scored, stage_g, rules_hash)
    elif any(path.exists() for path in core_paths):
        existing = [path.name for path in core_paths if path.exists()]
        raise FileExistsError(
            f"Refusing partial overwrite for batch-{batch_id}; existing core files: {existing}"
        )
    else:
        source, exclusion_counts = build_source_batch(paths)
        source.to_csv(paths.source, index=False)

        diagnostics = compute_diagnostics(paths, source)
        diagnostics.to_csv(paths.diagnostics, index=False)
        features = base.build_features(source, diagnostics)
        predictions = pd.DataFrame([rules.gatevet(row) for _, row in features.iterrows()])
        scored = base.source_runner.add_source_context(predictions, source)

        stage_g = base.build_stage_g(scored)
        holds = top_holds(scored)
        rejects = base.build_reject_summary(scored)
        sanity = reject_sanity_sample(scored)
        base.validate(source, features, scored, stage_g, rules_hash)

        scored.to_csv(paths.predictions, index=False)
        stage_g.to_csv(paths.stage_g, index=False)
        holds.to_csv(paths.top_holds, index=False)
        rejects.to_csv(paths.rejects, index=False)
        sanity.to_csv(paths.reject_sanity, index=False)

    visual_manifest = generate_visual_manifest(paths, stage_g, holds, sanity)
    visual_manifest.to_csv(paths.visual_manifest, index=False)
    write_summary(
        paths,
        scored,
        stage_g,
        holds,
        rejects,
        sanity,
        diagnostics,
        visual_manifest,
        rules_hash,
        exclusion_counts,
    )
    add_validation_summaries(paths)
    update_rolling_summary()
    print(f"Wrote {paths.predictions.name} ({len(scored)} rows)")
    print(f"Wrote {paths.stage_g.name} ({len(stage_g)} rows)")
    print(f"Wrote {paths.top_holds.name} ({len(holds)} rows)")
    print(f"Wrote {paths.rejects.name} ({len(rejects)} summary rows)")
    print(f"Wrote {paths.reject_sanity.name} ({len(sanity)} rows)")
    print(f"Wrote {paths.summary.name}")
    print(f"Wrote {ROLLING_SUMMARY.name}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", type=int, default=default_next_batch_id())
    args = parser.parse_args()
    run_batch(args.batch_id)


if __name__ == "__main__":
    main()
