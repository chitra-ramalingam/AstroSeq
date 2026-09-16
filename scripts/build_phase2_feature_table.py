from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DATA = ROOT / "data/phase2"
OUT_DOCS = ROOT / "docs/phase2"
POSITIVE_EPICS = [
    "EPIC_211357782", "EPIC_211497712", "EPIC_211534076",
    "EPIC_211682657", "EPIC_211889692", "EPIC_211915147",
    "EPIC_211953866", "EPIC_212001099", "EPIC_212024647",
]
LEAKAGE_EXCLUSIONS = [
    "gatevetter_v0_2_prediction", "gatevetter_v0_2_reason", "gatevetter_v0_2_action",
    "gatevetter_prediction", "gatevetter_reason", "gatevetter_action",
    "stage_g_final_recommendation", "final_recommendation", "final_candidate_status",
    "stage_h_training_label_v3", "stage_h_ledger_status", "manual_reason", "status_reason",
    "master_label", "master_reason", "manual_label", "training_label_rule",
    "positive_evidence_tier", "external_disposition", "external_confirmation_status",
    "training_role", "correction_basis", "physical_loss_eligible", "normalized_label",
    "catalogue_normalized_label", "catalogue_normalized_target", "catalogue_label_proposal",
    "archive_disposition", "catalogue_class", "catalogue_reference", "normalized_target_label",
    "gatevetter_decision", "gatevetter_recommendation",
    "odd_even_assessment", "secondary_assessment", "alias_risk",
    "event_stack_coherence", "local_baseline_stability",
]
NOMINAL_FEATURES = [
    "validation_period_days", "primary_depth", "primary_depth_snr",
    "transit_duration_days", "transit_duration_hours", "duration_fraction_of_period",
    "odd_depth_median", "even_depth_median", "odd_even_depth_ratio",
    "secondary_depth_phase_05", "secondary_depth_snr", "secondary_to_primary_depth_ratio",
    "oot_to_depth", "event_family_count", "candidate_period_count",
    "alias_best_support_count", "alias_best_support_ratio",
    "half_period_support_count", "double_period_support_count",
]
HYPOTHESIS_FEATURES = [
    "period_days", "cluster_center_phase", "event_support_count", "event_family_count",
    "primary_depth", "primary_depth_snr", "duration_days", "odd_depth_median",
    "even_depth_median", "odd_even_depth_ratio", "oot_to_depth", "secondary_depth",
    "secondary_depth_snr", "secondary_to_primary_depth_ratio",
    "event_stack_coherence_score", "event_timing_rms_phase", "event_epoch_coverage",
    "event_depth_mad_fraction", "local_baseline_event_count",
    "median_abs_baseline_slope_per_day", "median_abs_left_right_offset",
    "both_sidebands_fraction", "baseline_slope_depth_fraction", "baseline_offset_depth_fraction",
]


def clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def safe_float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def resolve_numeric_measurements(values: list[float]) -> tuple[float, list[float]]:
    """Return one stable value, or NaN plus distinct values for quarantine."""
    unique: list[float] = []
    for value in values:
        if np.isfinite(value) and not any(np.isclose(value, prior, rtol=1e-8, atol=1e-12) for prior in unique):
            unique.append(float(value))
    return (unique[0] if len(unique) == 1 else np.nan), unique


def verify_positive_labels(labels: pd.DataFrame) -> pd.DataFrame:
    final = pd.read_csv(ROOT / "plots/k2_batch/final_candidate_master_ledger.csv", dtype=str)
    manual = pd.read_csv(ROOT / "plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv", dtype=str)
    rows = []
    for epic in POSITIVE_EPICS:
        inv = labels.loc[labels["epic_id"].eq(epic)].iloc[0]
        fin = final.loc[final["epic_id"].eq(epic)].tail(1)
        man = manual.loc[manual["epic_id"].eq(epic)].tail(1)
        original = clean(inv["original_manual_label"])
        manual_source = ""
        if len(man):
            r = man.iloc[0]
            manual_source = f"manual_vetting_decisions_ledger.csv; reviewer={clean(r.get('reviewer'))}; reviewed_at={clean(r.get('reviewed_at'))}"
        elif len(fin):
            r = fin.iloc[0]
            manual_source = f"final_candidate_master_ledger.csv; reviewer={clean(r.get('reviewer'))}; reviewed_at={clean(r.get('reviewed_at'))}"
        confirmed = ""
        candidate = ""
        role = ""
        note = ""
        if epic == "EPIC_211889692":
            confirmed = "final ledger identifies K2-108 b as externally confirmed/validated"
            candidate = "recovered known positive control"
            role = "confirmed_positive_benchmark"
            note = "Confirmed evidence is explicit in the final ledger; retain external provenance."
        elif epic == "EPIC_211534076":
            candidate = "known unconfirmed candidate recovered as a positive control"
            role = "recovered_known_candidate_positive"
            note = "Known candidate, not a confirmed planet."
        elif original == "candidate_like":
            candidate = "manual candidate_like review"
            role = "manually_candidate_like_positive"
            note = "Manual morphology/scientific review positive; not confirmed."
        elif original == "candidate_ready_for_ledger":
            candidate = "manual Stage F promoted candidate / candidate_ready_for_ledger"
            role = "manually_promoted_candidate_positive"
            note = "Candidate promotion is not confirmation."
        elif original == "promote_to_stage_g":
            candidate = "manual promote_to_stage_g outcome"
            role = "promoted_hold_or_candidate_with_caveat"
            note = "Retain as a caveated promoted candidate; not confirmed."
        rows.append({
            "epic_id": epic,
            "original_label": original,
            "normalized_label": clean(inv["current_final_label"]),
            "label_source": clean(inv["label_source"]),
            "manual_review_source": manual_source,
            "confirmed_planet_evidence": confirmed,
            "candidate_evidence": candidate,
            "conflicting_evidence": clean(inv["duplicate_conflicting_labels"]),
            "safe_for_supervised_training": bool(inv["safe_for_supervised_training"]),
            "recommended_training_role": role,
            "verification_notes": note,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DOCS / "phase2_positive_label_verification.csv", index=False)
    return out


def existing_nominal_diagnostics(epics: set[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    for path in sorted(ROOT.glob("gatevetter_v0_2*_diagnostics.csv")):
        frame = pd.read_csv(path, dtype=str)
        if "epic_id" not in frame:
            continue
        frame = frame[frame["epic_id"].isin(epics)].copy()
        frame["_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(index=sorted(epics)), []
    all_rows = pd.concat(frames, ignore_index=True)
    output: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for epic, group in all_rows.groupby("epic_id", sort=True):
        row: dict[str, Any] = {"epic_id": epic}
        sources = sorted(group["_source_file"].unique())
        for feature in NOMINAL_FEATURES:
            if feature not in group:
                row[feature] = np.nan
                continue
            vals = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy(float)
            resolved, unique = resolve_numeric_measurements(vals.tolist())
            if len(unique) <= 1:
                row[feature] = resolved
            else:
                row[feature] = np.nan  # quarantine instead of selecting silently
                conflicts.append({
                    "epic_id": epic, "conflict_type": "diagnostic_measurement_conflict",
                    "feature": feature, "values": "|".join(map(str, unique)),
                    "sources": "|".join(sources), "resolution": "quarantined_as_missing",
                })
        row["nominal_diagnostic_sources"] = "|".join(sources)
        output[epic] = row
    return pd.DataFrame(output.values()), conflicts


def positive_period_diagnostics(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Reuse the exact measurement implementation used for the negative deep-review packet.
    import scripts.build_manual_vetting_next64_plot_pack as plot_pack
    import scripts.prepare_gatevetter_v0_2_deep_review as deep
    import scripts.refresh_gatevetter_unseen_full_validation as refresh

    final = pd.read_csv(ROOT / "plots/k2_batch/final_candidate_master_ledger.csv", dtype=str)
    period_map = {
        r["epic_id"]: safe_float(r.get("best_period_days"))
        for _, r in final[final["epic_id"].isin(POSITIVE_EPICS)].drop_duplicates("epic_id", keep="last").iterrows()
    }
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    version = sha256(ROOT / "scripts/prepare_gatevetter_v0_2_deep_review.py")
    for epic in POSITIVE_EPICS:
        lc = refresh.load_cached_light_curve(epic)
        events = refresh.load_events(epic)
        period = period_map.get(epic, np.nan)
        if np.isfinite(period) and period > 0:
            source = "saved_final_ledger_period"
            trusted = True
        else:
            row = pd.Series({"epic_id": epic, "best_period_days": np.nan})
            period, _, _, source = plot_pack.choose_period(row, events)
            trusted = source in {"saved_best_period", "saved_validation_period"}
        out: dict[str, Any] = {
            "epic_id": epic, "nominal_period_source": source,
            "nominal_period_trusted": trusted,
            "positive_diagnostic_provenance": "scripts/prepare_gatevetter_v0_2_deep_review.py:evaluate_period",
            "positive_diagnostic_script_sha256": version,
        }
        if not np.isfinite(period) or period <= 0:
            out["positive_diagnostic_missing_reason"] = "no_technical_period_available"
            wide_rows.append(out)
            continue
        for role, value in (("p_half", period / 2.0), ("p", period), ("2p", period * 2.0)):
            measured, _, _ = deep.evaluate_period(epic, role, value, events, lc)
            measurement_only = {k: measured.get(k, np.nan) for k in HYPOTHESIS_FEATURES}
            long_rows.append({
                "epic_id": epic, "period_role": role, "nominal_period_source": source,
                "nominal_period_trusted": trusted, "measurement_provenance": out["positive_diagnostic_provenance"],
                "producing_script_sha256": version, **measurement_only,
            })
            for feature, feature_value in measurement_only.items():
                out[f"{role}_{feature}"] = feature_value
        # Nominal aliases permit the same scalar schema as existing negative diagnostics.
        out.update({
            "validation_period_days": out.get("p_period_days"),
            "primary_depth": out.get("p_primary_depth"),
            "primary_depth_snr": out.get("p_primary_depth_snr"),
            "transit_duration_days": out.get("p_duration_days"),
            "transit_duration_hours": safe_float(out.get("p_duration_days")) * 24.0,
            "duration_fraction_of_period": safe_float(out.get("p_duration_days")) / period,
            "odd_depth_median": out.get("p_odd_depth_median"),
            "even_depth_median": out.get("p_even_depth_median"),
            "odd_even_depth_ratio": out.get("p_odd_even_depth_ratio"),
            "secondary_depth_phase_05": out.get("p_secondary_depth"),
            "secondary_depth_snr": out.get("p_secondary_depth_snr"),
            "secondary_to_primary_depth_ratio": out.get("p_secondary_to_primary_depth_ratio"),
            "oot_to_depth": out.get("p_oot_to_depth"),
            "event_family_count": out.get("p_event_family_count"),
            "candidate_period_count": np.nan,
            "nominal_diagnostic_sources": "phase2_positive_measurement_only",
        })
        wide_rows.append(out)
    long = pd.DataFrame(long_rows)
    long.to_csv(OUT_DATA / "phase2_positive_period_diagnostics.csv", index=False)
    return pd.DataFrame(wide_rows), long


def build_table(labels_path: Path, output: Path, audit_only: bool) -> dict[str, int]:
    if not audit_only:
        raise ValueError("Phase 2A feature infrastructure is audit-only; pass --audit-only")
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path)
    safe = labels[labels["safe_for_supervised_training"].astype(str).str.lower().eq("true")].copy()
    if safe["epic_id"].duplicated().any():
        raise ValueError("Label inventory is not one row per EPIC")
    verification = verify_positive_labels(labels)
    nominal, conflicts = existing_nominal_diagnostics(set(safe["epic_id"]))
    positive_wide, positive_long = positive_period_diagnostics(labels)
    if len(nominal):
        nominal = nominal[~nominal["epic_id"].isin(POSITIVE_EPICS)]
    diagnostics = pd.concat([nominal, positive_wide], ignore_index=True, sort=False)

    scores = pd.read_csv(OUT_DATA / "phase2_cnn_scores.csv")
    bundle = np.load(OUT_DATA / "phase2_cnn_embeddings.npz")
    embedding_ids = bundle["epic_id"].astype(str)
    embeddings = bundle["embedding"].astype(np.float32)
    embedding_frame = pd.DataFrame(embeddings, columns=[f"cnn_embedding_{i:03d}" for i in range(128)])
    embedding_frame.insert(0, "epic_id", embedding_ids)

    target = safe[["epic_id", "original_manual_label", "current_final_label", "label_source", "safe_for_supervised_training", "duplicate_conflicting_labels"]].rename(columns={
        "original_manual_label": "original_label", "current_final_label": "normalized_label",
        "safe_for_supervised_training": "target_eligible", "duplicate_conflicting_labels": "label_conflict_evidence",
    })
    table = target.merge(diagnostics, on="epic_id", how="left", validate="one_to_one")
    table = table.merge(scores[["epic_id", "cnn_probability", "cnn_model_path", "cnn_model_sha256", "cnn_embedding_layer", "cnn_embedding_dim", "cnn_embedding_aggregation", "cnn_segment_count"]], on="epic_id", how="left", validate="one_to_one")
    table = table.merge(embedding_frame, on="epic_id", how="left", validate="one_to_one")
    table["feature_provenance"] = table.apply(lambda r: json.dumps({
        "nominal_diagnostics": clean(r.get("nominal_diagnostic_sources")),
        "period_hypotheses": clean(r.get("positive_diagnostic_provenance")),
        "cnn": clean(r.get("cnn_model_path")),
        "embedding": clean(r.get("cnn_embedding_layer")),
    }, sort_keys=True), axis=1)

    protected = {"epic_id", "original_label", "normalized_label", "label_source", "target_eligible", "label_conflict_evidence", "feature_provenance", "nominal_period_source", "nominal_period_trusted", "positive_diagnostic_provenance", "positive_diagnostic_script_sha256", "positive_diagnostic_missing_reason", "nominal_diagnostic_sources", "cnn_model_path", "cnn_model_sha256", "cnn_embedding_layer", "cnn_embedding_dim", "cnn_embedding_aggregation"}
    numerical_features = [c for c in table.columns if c not in protected and pd.api.types.is_numeric_dtype(table[c])]
    missing_frame = table[numerical_features].isna().rename(columns=lambda c: f"missing__{c}")
    table = pd.concat([table, missing_frame], axis=1)
    table = table.sort_values("epic_id").reset_index(drop=True)
    if len(table) != len(safe) or table["epic_id"].nunique() != len(table):
        raise AssertionError("Feature table must contain exactly one row per safe EPIC")
    if set(map(str.lower, numerical_features)) & set(map(str.lower, LEAKAGE_EXCLUSIONS)):
        raise AssertionError("Leakage field entered numerical features")

    for _, row in target[target["label_conflict_evidence"].fillna("").ne("")].iterrows():
        # Safe rows have one normalized physical class; these are vocabulary/provenance
        # differences, not silent confirmations or physical-class conflicts.
        conflicts.append({"epic_id": row["epic_id"], "conflict_type": "label_alias_difference", "feature": "target", "values": row["label_conflict_evidence"], "sources": row["label_source"], "resolution": "preserved_for_traceability_normalized_class_consistent"})
    pd.DataFrame(conflicts, columns=["epic_id", "conflict_type", "feature", "values", "sources", "resolution"]).to_csv(OUT_DOCS / "phase2_feature_table_conflicts.csv", index=False)

    missing_rows = []
    for feature in numerical_features:
        for label, group in table.groupby("normalized_label"):
            count = int(group[feature].isna().sum())
            missing_rows.append({"feature": feature, "normalized_label": label, "rows": len(group), "missing_count": count, "missing_fraction": count / len(group)})
    pd.DataFrame(missing_rows).to_csv(OUT_DOCS / "phase2_feature_table_missingness.csv", index=False)
    table.to_parquet(output, index=False)
    preview = pd.concat([table[table["normalized_label"].eq("candidate_like")], table[~table["normalized_label"].eq("candidate_like")].head(41)], ignore_index=True)
    preview.to_csv(OUT_DATA / "phase2_feature_table_preview.csv", index=False)

    diag_cols = ["primary_depth", "primary_depth_snr", "validation_period_days"]
    has_diag = table[diag_cols].notna().any(axis=1)
    has_triple = table[[f"{role}_primary_depth" for role in ("p_half", "p", "2p")]].notna().all(axis=1)
    metrics = {
        "safe_rows": len(table),
        "audited_tier_rows_retained": len(verification),
        "physical_loss_eligible_candidate_positives": int((table["normalized_label"].eq("candidate_like") & table["target_eligible"].astype(bool)).sum()),
        "reliable_positives_with_diagnostics": int((table["normalized_label"].eq("candidate_like") & has_diag).sum()),
        "reliable_positives_with_p_half_p_2p": int((table["normalized_label"].eq("candidate_like") & has_triple).sum()),
        "safe_negatives_with_diagnostics": int((~table["normalized_label"].eq("candidate_like") & has_diag).sum()),
        "hard_conflicts": sum(c["conflict_type"] != "label_alias_difference" for c in conflicts),
        "label_alias_differences": sum(c["conflict_type"] == "label_alias_difference" for c in conflicts),
        "leakage_features_excluded": len(LEAKAGE_EXCLUSIONS),
    }
    write_audit(metrics, table, positive_long, verification)
    return metrics


def write_audit(metrics: dict[str, int], table: pd.DataFrame, positive_long: pd.DataFrame, verification: pd.DataFrame) -> None:
    emb_counts = table.groupby("normalized_label")["cnn_embedding_000"].apply(lambda x: int(x.notna().sum())).to_dict()
    score_counts = table.groupby("normalized_label")["cnn_probability"].apply(lambda x: int(x.notna().sum())).to_dict()
    text = f"""# Phase 2 Positive Coverage Audit

## Outcome

- Audited tier rows retained for lineage: **{metrics['audited_tier_rows_retained']}**.
- Physical-loss-eligible candidate positives: **{metrics['physical_loss_eligible_candidate_positives']}** (Gold 0; Silver 4; Bronze excluded 3; external-confirmation-pending 1).
- Candidate-labelled retained rows with nominal diagnostics: **{metrics['reliable_positives_with_diagnostics']}**.
- Candidate-labelled retained rows with P/2-P-2P measurements: **{metrics['reliable_positives_with_p_half_p_2p']}**.
- Safe negatives with aligned nominal diagnostics: **{metrics['safe_negatives_with_diagnostics']}**.
- Hard physical-class or measurement conflicts: **{metrics['hard_conflicts']}**. Label-vocabulary/provenance differences preserved for traceability: **{metrics['label_alias_differences']}**. Any conflicting measurement is quarantined as missing.
- Explicit leakage fields excluded: **{metrics['leakage_features_excluded']}** field families.

Frozen-CNN embeddings exported by class: {emb_counts}. CNN probabilities generated by class: {score_counts}. The model is `models/k2_nocrop_flux_seed46_split303.best.keras`; inference used `training=False`, `global_average_pooling1d_2`, and no weight changes.

## Positive-label distinctions

`EPIC_211889692` is external-confirmation-pending and quarantined, not Gold. `EPIC_211915147` is an eligible EB/variable hard negative while its earlier manual candidate-like review remains historical evidence. `EPIC_211534076` is a recovered known but unconfirmed Silver candidate. The other three Silver rows retain direct manual candidate-like provenance. All three Bronze rows remain lineage-only and loss-ineligible.

## Diagnostic policy and provenance

The positive measurements call the same `evaluate_period` implementation used for the Phase-1 negative deep-review packet in `scripts/prepare_gatevetter_v0_2_deep_review.py`. Manual labels are not passed into calculation. Numerical measurements are retained; categorical/rule verdicts (`odd_even_assessment`, `secondary_assessment`, `alias_risk`, event-stack/local-baseline verdicts) are excluded. Five positives use saved ledger periods; four use explicitly untrusted event-spacing fallback periods. Missing values remain missing.

The feature table contains exactly one row per safe EPIC, target eligibility, original/normalized labels, feature provenance, 128 embedding columns, and a missingness indicator for every numerical feature. GateVetter predictions/actions/reasons/recommendations, manual reasons, and target-derived rule outcomes are absent from model features.

Explicitly excluded field families: `{', '.join(LEAKAGE_EXCLUSIONS)}`.

## Remaining blockers before CatBoost

1. Review the caveated positive roles and decide whether promoted Stage-G objects belong in the physical-class loss or only validation/traceability.
2. Review fallback periods for the four manual candidates; their three-hypothesis measurements are technically complete but period trust is false.
3. Resolve/quarantine the recorded duplicate diagnostic conflicts.
4. Expand verified positives and multi-campaign coverage; nine positives remain too small for a stable four-class split.
5. Approve a fixed feature schema after missingness and provenance review, then create—but do not yet infer—the group/campaign split.

No training, candidate search, CNN modification, or GateVetter scientific decision was performed.
"""
    (OUT_DOCS / "PHASE2_POSITIVE_COVERAGE_AUDIT.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the leakage-safe Phase 2 feature table.")
    parser.add_argument("--labels", type=Path, default=OUT_DOCS / "phase2_label_inventory.csv")
    parser.add_argument("--output", type=Path, default=OUT_DATA / "phase2_feature_table.parquet")
    parser.add_argument("--audit-only", action="store_true", help="Required safety switch; this script never trains a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_table(args.labels, args.output, args.audit_only)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
