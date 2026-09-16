from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gatevetter_v0_2_rules as rules
import scripts.build_manual_vetting_next64_plot_pack as plot_pack
import scripts.refresh_gatevetter_unseen_full_validation as refresh
import scripts.run_gatevetter_v0_1_unseen as source_runner


BATCH_SIZE = 100
SOURCE_QUEUE = source_runner.SOURCE_QUEUE
PREVIOUS_PREDICTIONS = ROOT / "gatevetter_v0_2_unseen_predictions.csv"
RULES_PATH = ROOT / "gatevetter_v0_2_rules.py"

OUT_PREDICTIONS = ROOT / "gatevetter_v0_2_batch2_predictions.csv"
OUT_STAGE_G = ROOT / "gatevetter_v0_2_batch2_stage_g_queue.csv"
OUT_HOLDS = ROOT / "gatevetter_v0_2_batch2_hold_queue.csv"
OUT_REJECTS = ROOT / "gatevetter_v0_2_batch2_reject_summary.csv"
OUT_SUMMARY = ROOT / "gatevetter_v0_2_batch2_summary.txt"
OUT_SOURCE = ROOT / "gatevetter_v0_2_batch2_source.csv"
OUT_DIAGNOSTICS = ROOT / "gatevetter_v0_2_batch2_diagnostics.csv"
OUT_VISUAL_MANIFEST = ROOT / "gatevetter_v0_2_batch2_visual_manifest.csv"
PLOT_ROOT = ROOT / "plots" / "k2_batch" / "gatevetter_v0_2_batch2"

PACKET_STAGE_G = "stage_g_queue"
PACKET_HOLDS = "top_20_holds"
PACKET_REJECTS = "reject_sanity_high_cnn_high_snr"
REJECT_SANITY_COUNT = 10


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "epic_id" not in frame.columns:
        return set()
    return {clean(value) for value in frame["epic_id"] if clean(value)}


def build_source_batch() -> pd.DataFrame:
    queue = source_runner.load_csv(SOURCE_QUEUE).copy()
    queue.insert(0, "source_queue_rank", range(1, len(queue) + 1))

    excluded = set()
    excluded.update(load_ids(source_runner.MANUAL_64_DECISIONS))
    excluded.update(load_ids(source_runner.MANUAL_REVIEW_LEDGER))
    excluded.update(load_ids(PREVIOUS_PREDICTIONS))

    source = queue.loc[
        ~queue["epic_id"].map(clean).isin(excluded)
    ].head(BATCH_SIZE).copy()
    if len(source) != BATCH_SIZE:
        raise RuntimeError(
            f"Expected {BATCH_SIZE} fresh unseen EPICs, found {len(source)}"
        )
    if source["epic_id"].map(clean).isin(excluded).any():
        raise RuntimeError("Previously reviewed or predicted EPIC leaked into batch 2")
    if source["epic_id"].duplicated().any():
        raise RuntimeError("Duplicate EPIC in batch-2 source")
    return source


def period_status(info: dict[str, Any]) -> tuple[str, str]:
    source = clean(info.get("validation_period_source"))
    ambiguous = truthy(info.get("period_ambiguity_flag"))
    if ambiguous or source == "period_ambiguous":
        return "no_automatic_period_selection", "false"
    if source == "saved_best_period":
        return "trusted_saved_period", "true"
    if source == "refreshed_period_search":
        return "trusted_refreshed_period", "true"
    return "provisional_not_trusted", "false"


def failed_diagnostics(epic_id: str, error: Exception) -> dict[str, Any]:
    return {
        "epic_id": epic_id,
        "validation_period_days": "",
        "validation_period_source": "",
        "period_ambiguity_flag": "true",
        "period_comparison_status": "diagnostic_refresh_failed",
        "trusted_period_validation": "false",
        "primary_depth": "",
        "primary_depth_snr": "",
        "transit_duration_hours": "",
        "duration_fraction_of_period": "",
        "odd_even_depth_ratio": "",
        "odd_even_depth_ratio_missing_reason": "diagnostic_refresh_failed",
        "secondary_depth_snr": "",
        "secondary_to_primary_depth_ratio": "",
        "oot_to_depth": "",
        "alias_risk": "period_ambiguous",
        "event_family_count": "",
        "candidate_period_count": "",
        "missing_reason": "diagnostic_refresh_failed",
        "diagnostic_error": str(error),
    }


def compute_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(source)
    for index, source_row in source.reset_index(drop=True).iterrows():
        epic_id = clean(source_row["epic_id"])
        print(f"[diagnostics {index + 1:03d}/{total}] {epic_id}", flush=True)
        try:
            lc = refresh.load_cached_light_curve(epic_id)
            events = refresh.load_events(epic_id)
            info, _ = refresh.period_confirmation(source_row, events, lc)
            status, trusted = period_status(info)
            info["period_comparison_status"] = status
            metrics, _ = refresh.recompute_metrics(epic_id, info, events, lc)
            metrics.update(
                {
                    "period_comparison_status": status,
                    "trusted_period_validation": trusted,
                    "odd_even_depth_ratio_missing_reason": clean(
                        metrics.get("odd_even_depth_ratio_missing_reason")
                    ),
                    "diagnostic_error": "",
                }
            )
            rows.append(metrics)
        except Exception as exc:
            rows.append(failed_diagnostics(epic_id, exc))
    return pd.DataFrame(rows)


def build_features(
    source: pd.DataFrame, diagnostics: pd.DataFrame
) -> pd.DataFrame:
    base = source_runner.build_features(source).set_index("epic_id", drop=False)
    diagnostic_map = diagnostics.set_index("epic_id", drop=False)
    rows: list[dict[str, Any]] = []
    overlay = [
        "primary_depth",
        "primary_depth_snr",
        "transit_duration_hours",
        "duration_fraction_of_period",
        "odd_even_depth_ratio",
        "secondary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "oot_to_depth",
        "alias_risk",
        "event_family_count",
        "candidate_period_count",
    ]
    for epic_id, base_row in base.iterrows():
        row = base_row.to_dict()
        diagnostic = diagnostic_map.loc[epic_id]
        for column in overlay:
            row[column] = clean(diagnostic.get(column))
        row.update(
            {
                "best_period_days": clean(
                    diagnostic.get("validation_period_days")
                ),
                "period_source": clean(
                    diagnostic.get("validation_period_source")
                ),
                "period_ambiguity_flag": clean(
                    diagnostic.get("period_ambiguity_flag")
                ),
                "validation_period_source": clean(
                    diagnostic.get("validation_period_source")
                ),
                "period_comparison_status": clean(
                    diagnostic.get("period_comparison_status")
                ),
                "trusted_period_validation": clean(
                    diagnostic.get("trusted_period_validation")
                ),
                "metric_trust_level": (
                    "trusted_period_dependent"
                    if truthy(diagnostic.get("trusted_period_validation"))
                    else "untrusted_period_dependent"
                ),
                "odd_even_depth_ratio_missing_reason": clean(
                    diagnostic.get("odd_even_depth_ratio_missing_reason")
                ),
            }
        )
        rows.append(row)
    features = pd.DataFrame(rows, columns=["epic_id", *rules.ALLOWED_FEATURES])
    rules.assert_no_forbidden_prediction_columns(features)
    return features


def build_stage_g(scored: pd.DataFrame) -> pd.DataFrame:
    queue = scored.loc[
        scored["gatevetter_prediction"].isin(rules.STAGE_G_PREDICTIONS)
    ].copy()
    queue = queue.sort_values(
        ["gatevetter_score", "epic_id"], ascending=[False, True]
    ).reset_index(drop=True)
    queue.insert(0, "stage_g_rank", range(1, len(queue) + 1))
    return queue


def build_holds(scored: pd.DataFrame) -> pd.DataFrame:
    queue = scored.loc[
        scored["gatevetter_prediction"].isin(
            {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
        )
    ].copy()
    queue = queue.sort_values(
        ["gatevetter_score", "epic_id"], ascending=[False, True]
    ).reset_index(drop=True)
    queue.insert(0, "hold_rank", range(1, len(queue) + 1))
    return queue


def build_reject_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rejects = scored.loc[
        ~scored["gatevetter_prediction"].isin(
            rules.STAGE_G_PREDICTIONS
            | {"excluded_uncertain_hold", "excluded_uncertain_hold_positive"}
        )
    ].copy()
    if rejects.empty:
        return pd.DataFrame(
            columns=[
                "gatevetter_prediction",
                "gatevetter_v0_2_reason",
                "reject_count",
                "epic_ids",
            ]
        )
    return (
        rejects.groupby(
            ["gatevetter_prediction", "gatevetter_v0_2_reason"],
            dropna=False,
        )["epic_id"]
        .agg(
            reject_count="size",
            epic_ids=lambda values: "|".join(sorted(values)),
        )
        .reset_index()
        .sort_values(
            ["reject_count", "gatevetter_prediction"],
            ascending=[False, True],
        )
    )


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
    eligible = eligible.sort_values(
        ["reject_sanity_score", "cnn_numeric", "snr_numeric"],
        ascending=[False, False, False],
    ).head(REJECT_SANITY_COUNT)
    eligible = eligible.reset_index(drop=True)
    eligible.insert(0, "sanity_sample_rank", range(1, len(eligible) + 1))
    return eligible


def plotting_row(row: pd.Series, rank: int, packet: str) -> pd.Series:
    return pd.Series(
        {
            "epic_id": clean(row.get("epic_id")),
            "queue_rank": rank,
            "cnn_score": row.get("cnn_score", ""),
            "morphology_positive": "",
            "autovet_label": clean(row.get("source_autovet_label")),
            "explanation_short": clean(row.get("gatevetter_v0_2_reason")),
            "best_period_days": row.get("best_period_days", ""),
            "validation_period_days": row.get("best_period_days", ""),
            "validation_period_source": row.get(
                "validation_period_source", ""
            ),
            "period_ambiguity_flag": row.get("period_ambiguity_flag", ""),
            "primary_depth": row.get("primary_depth", ""),
            "primary_depth_snr": row.get("primary_depth_snr", ""),
            "transit_duration_hours": row.get(
                "transit_duration_hours", ""
            ),
            "odd_even_depth_ratio": row.get("odd_even_depth_ratio", ""),
            "secondary_depth_snr": row.get("secondary_depth_snr", ""),
            "secondary_to_primary_depth_ratio": row.get(
                "secondary_to_primary_depth_ratio", ""
            ),
            "oot_to_depth": row.get("oot_to_depth", ""),
            "alias_risk": row.get("alias_risk", ""),
            "event_family_count": row.get("event_family_count", ""),
            "candidate_period_count": row.get("candidate_period_count", ""),
            "master_label": row.get("gatevetter_prediction", ""),
            "review_level": packet,
            "decision_authority": "gatevetter_v0_2_batch2_visual_only",
        }
    )


def generate_visual_packet(
    packet: str, rows: pd.DataFrame
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    packet_dir = PLOT_ROOT / packet
    packet_dir.mkdir(parents=True, exist_ok=True)
    original_out = plot_pack.OUT_DIR
    plot_pack.OUT_DIR = packet_dir
    try:
        for index, row in rows.reset_index(drop=True).iterrows():
            epic_id = clean(row.get("epic_id"))
            print(
                f"[visual {packet} {index + 1:02d}/{len(rows):02d}] {epic_id}",
                flush=True,
            )
            rank = index + 1
            plot_row = plotting_row(row, rank, packet)
            result = plot_pack.build_one(plot_row)
            epic_dir = packet_dir / epic_id
            summary_path = epic_dir / "gatevetter_packet.json"
            payload = {
                "packet": packet,
                "rank": rank,
                "gatevetter_version": "v0.2",
                "row": {
                    str(key): (
                        None
                        if pd.isna(value)
                        else value.item()
                        if isinstance(value, np.generic)
                        else value
                    )
                    for key, value in row.to_dict().items()
                },
                "manual_label": None,
                "manual_notes": None,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            summary_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            manifests.append(
                {
                    "packet": packet,
                    "rank": rank,
                    "epic_id": epic_id,
                    "success": bool(result["success"]),
                    "missing_light_curve": bool(
                        result["missing_light_curve"]
                    ),
                    "packet_json": summary_path.relative_to(ROOT).as_posix(),
                    **{
                        f"plot_{key}": path.relative_to(ROOT).as_posix()
                        for key, path in result["paths"].items()
                    },
                }
            )
    finally:
        plot_pack.OUT_DIR = original_out
    return manifests


def validate(
    source: pd.DataFrame,
    features: pd.DataFrame,
    scored: pd.DataFrame,
    stage_g: pd.DataFrame,
    rules_hash_before: str,
) -> None:
    if len(source) != BATCH_SIZE or len(scored) != BATCH_SIZE:
        raise RuntimeError("Batch-2 row count changed")
    if scored["epic_id"].duplicated().any():
        raise RuntimeError("Duplicate prediction EPIC")
    if rules_hash_before != sha256(RULES_PATH):
        raise RuntimeError("Frozen GateVetter v0.2 rules changed during run")
    forbidden = rules.FORBIDDEN_DURING_PREDICTION.intersection(features.columns)
    if forbidden:
        raise RuntimeError(
            f"Manual/forbidden prediction columns present: {sorted(forbidden)}"
        )
    for _, row in stage_g.iterrows():
        failures = rules.stage_g_prerequisite_failures(row)
        if not rules.trusted_period(row):
            failures.append("trusted_period_validation_required")
        if rules.period_is_ambiguous(row):
            failures.append("period_ambiguous")
        if rules.period_source_is_fallback_only(row):
            failures.append("fallback_only_period")
        if failures:
            raise RuntimeError(
                f"Invalid Stage G promotion for {row['epic_id']}: {failures}"
            )


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
        "GateVetter v0.2 batch-2 fresh unseen run",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"batch_size={BATCH_SIZE}",
        "source_selection=next fresh source-queue rows after excluding all prior v0.2 unseen predictions and manual-review ledger EPICs",
        f"rules_file={RULES_PATH.name}",
        f"rules_sha256={rules_hash}",
        "rules_changed=false",
        "manual_labels_used_during_prediction=false",
        "missing_diagnostics_treated_as_pass=false",
        "trusted_period_required_for_stage_g=true",
        f"diagnostic_failures={int(diagnostics['diagnostic_error'].fillna('').ne('').sum())}",
        f"stage_g_queue_rows={len(stage_g)}",
        f"hold_queue_rows={len(holds)}",
        f"reject_summary_rows={len(rejects)}",
        f"reject_sanity_sample_rows={len(sanity)}",
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
        for label, count in scored["period_comparison_status"].value_counts(
            dropna=False
        ).items()
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


def main() -> None:
    rules_hash = sha256(RULES_PATH)
    source = build_source_batch()
    source.to_csv(OUT_SOURCE, index=False)

    diagnostics = compute_diagnostics(source)
    diagnostics.to_csv(OUT_DIAGNOSTICS, index=False)
    features = build_features(source, diagnostics)
    predictions = pd.DataFrame(
        [rules.gatevet(row) for _, row in features.iterrows()]
    )
    scored = source_runner.add_source_context(predictions, source)

    stage_g = build_stage_g(scored)
    holds = build_holds(scored)
    rejects = build_reject_summary(scored)
    sanity = reject_sanity_sample(scored)
    validate(source, features, scored, stage_g, rules_hash)

    scored.to_csv(OUT_PREDICTIONS, index=False)
    stage_g.to_csv(OUT_STAGE_G, index=False)
    holds.to_csv(OUT_HOLDS, index=False)
    rejects.to_csv(OUT_REJECTS, index=False)

    visual_rows: list[dict[str, Any]] = []
    visual_rows.extend(generate_visual_packet(PACKET_STAGE_G, stage_g))
    visual_rows.extend(generate_visual_packet(PACKET_HOLDS, holds.head(20)))
    visual_rows.extend(generate_visual_packet(PACKET_REJECTS, sanity))
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
    print(f"Wrote {OUT_PREDICTIONS.name} ({len(scored)} rows)")
    print(f"Wrote {OUT_STAGE_G.name} ({len(stage_g)} rows)")
    print(f"Wrote {OUT_HOLDS.name} ({len(holds)} rows)")
    print(f"Wrote {OUT_REJECTS.name} ({len(rejects)} rows)")
    print(f"Wrote {OUT_SUMMARY.name}")


if __name__ == "__main__":
    main()
