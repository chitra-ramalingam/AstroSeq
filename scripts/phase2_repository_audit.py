from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "phase2"


def canon(value: object) -> str:
    match = re.search(r"(\d{8,10})", str(value))
    return f"EPIC_{match.group(1)}" if match else ""


def sval(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)


def normalize(label: str) -> str:
    x = label.strip().lower()
    if not x:
        return ""
    if x in {"candidate_like", "planet_like", "recovered_known_confirmed_planet", "recovered_known_unconfirmed_candidate", "candidate_ready_for_ledger", "promote_primary_candidate", "promote_to_stage_g", "promote_to_deeper_eval", "strong_candidate_like"}:
        return "candidate_like"
    if x in {"false_positive_eb_or_variable", "binary_system", "variable_or_possible_eb", "variable_or_artifact", "reject_as_planet_binary_system", "reject_likely_eclipsing_binary", "reject_likely_variable_or_binary", "likely_eb_or_variable_hold", "noise_or_variable", "reject_as_likely_false_positive_or_variable"}:
        return "false_positive_eb_or_variable"
    if x in {"reject_as_noise_or_artifact", "noise_or_artifact", "reject_manual_false_positive", "low_priority_negative", "deprioritize_after_manual_visual_review", "reject_low_confidence_shallow_signal"}:
        return "reject_as_noise_or_artifact"
    if "uncertain" in x or "hold" in x or x in {"candidate_with_caveat", "candidate_needs_manual_followup", "keep_hold_for_variability_review", "secondary_hold", "candidate_demoted_after_stage_h", "hold_borderline_candidate"}:
        return "uncertain_hold"
    return "other_unmapped"


def build_label_inventory() -> tuple[pd.DataFrame, dict[str, int]]:
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add(frame: pd.DataFrame, id_col: str, label_col: str, source: str, date_col: str = "", reason_col: str = "") -> None:
        for _, row in frame.iterrows():
            epic = canon(row.get(id_col, ""))
            label = sval(row.get(label_col, ""))
            if epic and label:
                evidence[epic].append({"label": label, "source": source, "date": sval(row.get(date_col, "")) if date_col else "", "reason": sval(row.get(reason_col, "")) if reason_col else ""})

    final = read("plots/k2_batch/final_candidate_master_ledger.csv")
    add(final, "epic_id", "final_candidate_status", "plots/k2_batch/final_candidate_master_ledger.csv", "reviewed_at", "status_reason")
    manual = read("plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv")
    add(manual, "epic_id", "manual_label", "plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv", "reviewed_at", "manual_reason")
    deep = read("gatevetter_v0_2_deep_review_manual_decisions.csv")
    add(deep, "epic_id", "final_label", "gatevetter_v0_2_deep_review_manual_decisions.csv", "review_date", "manual_reason")
    training = read("training_labels_v3.csv")
    add(training, "epic_id", "training_label_v3", "training_labels_v3.csv", "reviewed_at", "training_label_rule")
    eph = read("K2_ephemerides.csv")
    eph["epic_id"] = eph["star_id"].map(canon)
    for epic in sorted(set(eph["epic_id"]) - {""}):
        evidence[epic].append({"label": "confirmed_planet", "source": "K2_ephemerides.csv (NASA Exoplanet Archive ephemerides; confirmation disposition not retained in columns)", "date": "", "reason": "Transit ephemeris catalogue row; verify disposition before supervised use."})

    master = read("plots/k2_batch/master_vetted_catalog/master_vetted_catalog.csv")
    master_ids = set(master["epic_id"].map(canon))
    score_map = {canon(r.epic_id): sval(r.cnn_score) for r in master.itertuples()}
    c5_targets = set(read("data/k2_target_lists/K2Campaign5targets.csv").iloc[:, 0].map(canon))
    infer_meta = pd.read_parquet(ROOT / "splits/infer_c5/meta_infer.parquet", columns=["star_id"])
    lightcurves = set(infer_meta["star_id"].map(canon))
    diagnostics = set()
    for path in ROOT.glob("gatevetter_v0_2*_diagnostics.csv"):
        diagnostics.update(read(path.name)["epic_id"].map(canon))
    pcomp = set(read("gatevetter_v0_2_deep_review_period_comparison.csv")["epic_id"].map(canon))

    rows = []
    precedence = [
        "gatevetter_v0_2_deep_review_manual_decisions.csv",
        "plots/k2_batch/master_vetted_catalog/manual_vetting_decisions_ledger.csv",
        "plots/k2_batch/final_candidate_master_ledger.csv",
        "training_labels_v3.csv",
    ]
    for epic in sorted(evidence):
        items = evidence[epic]
        chosen = None
        for src in precedence:
            found = [x for x in items if x["source"] == src]
            if found:
                chosen = found[-1]
                break
        if chosen is None:
            chosen = items[0]
        original = chosen["label"]
        norm = "candidate_like" if original == "confirmed_planet" else normalize(original)
        distinct = sorted(set(x["label"] for x in items))
        norm_distinct = sorted(set(("candidate_like" if x == "confirmed_planet" else normalize(x)) for x in distinct))
        is_catalog_planet = any(x["label"] == "confirmed_planet" for x in items)
        known_eb = any(normalize(x["label"]) == "false_positive_eb_or_variable" for x in items)
        is_noise = any(normalize(x["label"]) == "reject_as_noise_or_artifact" for x in items)
        ambiguous = norm == "uncertain_hold" or len(norm_distinct) > 1
        direct_manual = any("manual" in x["source"] or x["source"].endswith("final_candidate_master_ledger.csv") for x in items)
        # The stripped ephemeris table does not retain disposition. Catalogue-only
        # rows are therefore useful for matching/traceability, not target truth.
        safe = norm in {"candidate_like", "false_positive_eb_or_variable", "reject_as_noise_or_artifact"} and direct_manual and not ambiguous
        validation_only = is_catalog_planet and not direct_manual
        leakage = []
        if epic in master_ids:
            leakage.append("EPIC appears in downstream pipeline/master-ledger products; split by EPIC and exclude decision fields")
        if is_catalog_planet:
            leakage.append("Catalogue ephemeris may have been used in legacy segment labelling; verify provenance/disposition")
        if len(norm_distinct) > 1:
            leakage.append("conflicting normalized labels; adjudicate before training")
        rows.append({
            "epic_id": epic,
            "current_final_label": norm,
            "original_manual_label": original,
            "label_source": chosen["source"],
            "review_date": chosen["date"],
            "k2_campaign": "5" if epic in c5_targets or epic in master_ids else "",
            "confirmed_planet_status": "catalogued_ephemeris_unverified_disposition" if is_catalog_planet else "",
            "known_eb_variable_status": "yes" if known_eb else "",
            "noise_artifact_status": "yes" if is_noise else "",
            "ambiguity_status": "uncertain_or_conflicting" if ambiguous else "none_seen",
            "safe_for_supervised_training": str(safe).lower(),
            "validation_or_traceability_only": str(validation_only or ambiguous).lower(),
            "duplicate_conflicting_labels": " | ".join(distinct) if len(distinct) > 1 else "",
            "possible_leakage_concerns": "; ".join(leakage),
            "all_label_evidence": " || ".join(f"{x['source']}={x['label']}" for x in items),
            "has_diagnostics": str(epic in diagnostics).lower(),
            "has_p_half_p_2p_diagnostics": str(epic in pcomp).lower(),
            "has_accessible_light_curve_tensor": str(epic in lightcurves).lower(),
            "has_current_cnn_score": str(bool(score_map.get(epic, ""))).lower(),
            "embedding_exportable": str(epic in lightcurves).lower(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase2_label_inventory.csv", index=False)
    # Evidence-level counts answer "how many EPICs have this label anywhere"
    # without losing labels hidden by source precedence. They can overlap.
    label_epics: dict[str, set[str]] = defaultdict(set)
    for epic, items in evidence.items():
        for item in items:
            label_epics[item["label"]].add(epic)
    counts = {label: len(epics) for label, epics in label_epics.items()}
    norm_counts = Counter(out["current_final_label"])
    metrics = {
        "unique_labelled_epics": len(out),
        "safe_supervised": int(out["safe_for_supervised_training"].eq("true").sum()),
        "complete_diagnostics": int(out["has_diagnostics"].eq("true").sum()),
        "p_half_p_2p": int(out["has_p_half_p_2p_diagnostics"].eq("true").sum()),
        "lightcurves": int(out["has_accessible_light_curve_tensor"].eq("true").sum()),
        "cnn_scores": int(out["has_current_cnn_score"].eq("true").sum()),
        "embeddings": int(out["embedding_exportable"].eq("true").sum()),
        "reliable_positives": int(((out["current_final_label"] == "candidate_like") & (out["safe_for_supervised_training"] == "true") & (~out["confirmed_planet_status"].eq("catalogued_ephemeris_unverified_disposition"))).sum()),
    }
    # High-score hard negatives are manually grounded negative rows with score >= .5.
    score_num = out["epic_id"].map(lambda x: pd.to_numeric(score_map.get(x, ""), errors="coerce"))
    metrics["hard_negatives"] = int(((out["current_final_label"].isin(["false_positive_eb_or_variable", "reject_as_noise_or_artifact"])) & (out["safe_for_supervised_training"] == "true") & (score_num >= 0.5)).sum())
    lines = [
        "Phase 2 label inventory summary", "================================", "",
        *(f"{k}={v}" for k, v in metrics.items()), "",
        "Normalized class counts (proposed mapping; originals retained in CSV)",
        *(f"{k}={v}" for k, v in sorted(norm_counts.items())), "",
        "Required exact evidence-level original-label counts (unique EPICs; categories may overlap)",
    ]
    requested = ["candidate_like", "planet_like", "confirmed_planet", "false_positive_eb_or_variable", "binary_system", "variable_or_artifact", "reject_as_noise_or_artifact", "noise_or_artifact", "uncertain_hold"]
    for key in requested:
        lines.append(f"{key}={counts.get(key, 0)}")
    lines += ["", "All other original labels"]
    for key in sorted(set(counts) - set(requested)):
        lines.append(f"{key}={counts[key]}")
    lines += ["", "Notes", "- Normalized counts use one explicit source-precedence label per EPIC.", "- Evidence-level original-label counts count unique EPICs carrying each label in any source and can overlap; all evidence remains in all_label_evidence.", "- confirmed_planet means presence in K2_ephemerides.csv; its stripped schema does not retain disposition, so these are validation/traceability-only pending catalogue verification.", "- uncertain labels are not safe supervised physical-class targets."]
    (OUT / "phase2_label_inventory_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, metrics


FEATURE_ROWS = [
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "validation_period_days", "Selected validation period", "days", "blank when unresolved", "P", "conditional", "safe if period provenance retained"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "validation_period_source", "Period-selection provenance", "category", "explicit period_ambiguous", "P", "conditional", "safe; not the final class"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "primary_depth", "Median primary-event depth", "relative flux", "0 or blank with missing-reason companion", "P", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "primary_depth_snr", "Primary depth signal-to-noise", "dimensionless", "0 or blank with missing reason", "P", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "transit_duration_days|transit_duration_hours|duration_fraction_of_period", "Event duration and period fraction", "days; hours; fraction", "blank with missing-reason companion", "P", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "odd_depth_median|even_depth_median|odd_even_depth_ratio", "Odd/even event depths and ratio", "relative flux; ratio", "blank for untrusted periods", "P", "yes", "safe diagnostic; retain missingness"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "secondary_depth_phase_05|secondary_depth_snr|secondary_to_primary_depth_ratio", "Phase-0.5 secondary evidence", "relative flux; SNR; ratio", "blank/0 with missing reason", "P", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "oot_to_depth", "Out-of-event variability relative to depth", "ratio", "blank for unusable depth/period", "P", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "event_family_count|candidate_period_count", "Event-family and period-candidate multiplicities", "count", "blank with reason", "period search", "yes", "safe diagnostic"),
    ("gatevetter_v0_2*_diagnostics.csv", "scripts/run_gatevetter_v0_2_batch_n.py", "period_ambiguity_flag|alias_*|half_period_support_count|double_period_support_count", "Period ambiguity and alias support", "boolean/count/ratio/days", "explicit missing reasons", "P/2,P,2P", "yes", "safe except any final alias gate/decision"),
    ("gatevetter_v0_2_deep_review_period_comparison.csv", "scripts/prepare_gatevetter_v0_2_deep_review.py", "period_role|period_days", "Hypothesis identity and duration", "category; days", "row absent if view unavailable", "P/2,P,2P", "yes", "safe"),
    ("gatevetter_v0_2_deep_review_period_comparison.csv", "scripts/prepare_gatevetter_v0_2_deep_review.py", "event_support_count|event_family_count|event_epoch_coverage", "Event support and coverage", "count; fraction", "row/field may be blank", "P/2,P,2P", "yes", "safe"),
    ("gatevetter_v0_2_deep_review_period_comparison.csv", "scripts/prepare_gatevetter_v0_2_deep_review.py", "event_stack_coherence_score|event_depth_mad_fraction|event_timing_rms_phase", "Stack coherence, depth stability, timing stability", "score; fraction; phase", "blank if unavailable", "P/2,P,2P", "yes", "safe numerical fields; exclude categorical verdict if rule-derived"),
    ("gatevetter_v0_2_deep_review_period_comparison.csv", "scripts/prepare_gatevetter_v0_2_deep_review.py", "local_baseline_event_count|median_abs_baseline_slope_per_day|median_abs_left_right_offset|both_sidebands_fraction|baseline_*_depth_fraction", "Local baseline stability diagnostics", "count; relative flux/day; relative flux; fractions", "blank if windows unavailable", "P/2,P,2P", "yes", "safe numerical fields"),
    ("plots/k2_batch/master_vetted_catalog/master_vetted_catalog.csv", "scripts/backfill_master_catalog_cnn_and_manual_queue.py", "cnn_score", "Frozen CNN max segment morphology probability", "probability", "blank if no tensor", "not folded; 512-cadence segments", "yes", "safe; never use morphology_positive policy label"),
    ("splits/infer_c5/X_infer.npy + meta_infer.parquet", "src/Classifiers/K2/K2_Dataset_builder.py", "flux channel + start/end/seg_mid_time", "Robust-standardized 512-cadence flux segments and coordinates", "robust scale; cadence index; time", "EPIC absent when fetch/preprocess failed", "not period dependent", "yes", "safe if split by EPIC"),
    ("final ledgers / prediction CSVs", "multiple", "final_candidate_status|stage_h_training_label_v3|GateVetter prediction/recommendation/reason|manual_reason", "Decision or target-derived fields", "category/text", "varies", "n/a", "NO", "EXCLUDE: target leakage"),
]


def write_feature_inventory() -> None:
    cols = ["source_file", "producing_script", "column_name", "meaning", "units", "missing_value_behaviour", "period_dependence", "safe_for_training", "leakage_risk_or_policy"]
    pd.DataFrame(FEATURE_ROWS, columns=cols).to_csv(OUT / "phase2_feature_inventory.csv", index=False)


def write_docs(inv: pd.DataFrame, m: dict[str, int]) -> None:
    norm = inv["current_final_label"].value_counts().to_dict()
    model_audit = f"""# Phase 2 Current Model Audit

## Selected frozen morphology encoder

Freeze `models/k2_nocrop_flux_seed46_split303.best.keras`. This is the active model named by `freezes/k2_flux_model_official_policy_note.txt:8`, `freezes/stage_f_closed_manifest.txt:45`, and the scoring constant in `scripts/backfill_master_catalog_cnn_and_manual_queue.py:14`. Historical root and mission models are competing artifacts, but repository policy does not designate them as current.

Direct load inspection (TensorFlow, `compile=False`) gives input `(None, 512, 1)` and output `(None, 1)`. The file SHA-256 is `547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443`. Layers are Conv1D(64,k=11) → LayerNorm → max pool → Conv1D(128,k=7) → LayerNorm → max pool → Conv1D(128,k=5) → LayerNorm → max pool → Conv1D(128,k=3) → global average pooling → dropout(0.2) → Dense(1,sigmoid). The final layer is `dense_2`; the penultimate layer is `dropout_2`, dimension 128. Prefer the deterministic pre-dropout `global_average_pooling1d_2` at inference (also 128-D); with `training=False`, dropout output is identical but the pooling layer is semantically cleaner.

## Input, preprocessing, and target

`K2_training_main.py:17-18,45-78` fixes split seed 303, train seed 46, Campaign 5, 512-sample windows, stride 256, default preprocessing and synthetic injection. `src/Classifiers/K2/K2_Dataset_builder.py:15-47,170-222` flattens (window 401, polynomial order 2), converts to relative flux, injects box transits, robust-centres/MAD-scales, clips at 10 sigma, and fills non-finite values with zero. `src/Classifiers/K2/K2_trainer.py:42-50` selects flux channel 0 and does not crop. It is therefore an unfurled, fixed-length time-series segment model—not a P-fold, local-transit, or global-fold model.

The target is segment `meta['label']`, with positives created from injected box transits on about half of Campaign-5 stars (`K2_Dataset_builder.py:52-65,128-170`); the trainer balances positive and negative streams (`K2_trainer.py:69-94`). Binary cross-entropy and a sigmoid output are declared by `src/Classifiers/Builders/BuilderHelper.py`. Output is `flux_p_science_like` historically and `transit_morphology_score` in the master-catalog backfill; policy says it is dip/transit morphology only, never auto-promotion (`freezes/k2_flux_model_official_policy_note.txt:12-27`; `backfill_master_catalog_cnn_and_manual_queue.py:21-25`).

## Weight-preserving embedding export

```python
import numpy as np
import tensorflow as tf

path = "models/k2_nocrop_flux_seed46_split303.best.keras"
model = tf.keras.models.load_model(path, compile=False)
model.trainable = False
encoder = tf.keras.Model(model.input, model.get_layer("global_average_pooling1d_2").output)
x = np.asarray(x_512_flux[:, :, :1], dtype=np.float32)
p = model.predict(x, batch_size=256, verbose=0).reshape(-1)
z = encoder.predict(x, batch_size=256, verbose=0)  # (N, 128)
```

This creates a read-only view of existing weights. Aggregate multiple segment embeddings per EPIC explicitly (for example score-weighted mean plus max-score segment), retaining segment count and aggregation policy.

## Competing models

The repository contains numerous `.keras` files (root mission baselines, hard-negative variants, `best.keras`, `best_ref.keras`, and `models/*`). `K2_training_main.py` and three freeze/policy files resolve the ambiguity in favour of the exact `.best.keras` path above. Do not substitute the similarly named non-best checkpoint.
"""
    (OUT / "PHASE2_CURRENT_MODEL_AUDIT.md").write_text(model_audit, encoding="utf-8")

    gap = f"""# Phase 2 Data Gap Analysis

## Observed availability

- Unique labelled EPICs in the union inventory: **{m['unique_labelled_epics']}**.
- Proposed normalized distribution: {', '.join(f'`{k}`={v}' for k,v in sorted(norm.items()))}.
- Rows currently safe under the conservative supervised flag: **{m['safe_supervised']}**.
- With any GateVetter batch diagnostics: **{m['complete_diagnostics']}**; with the explicit P/2–P–2P comparison: **{m['p_half_p_2p']}**.
- With Campaign-5 inference tensors / accessible cached light-curve representation: **{m['lightcurves']}**; current CNN scores: **{m['cnn_scores']}**; therefore embedding-exportable without new downloads: **{m['embeddings']}**.
- High-CNN-score (≥0.5), manually grounded negative EPICs: **{m['hard_negatives']}**.
- Reliable internal positive examples excluding unverified stripped ephemeris-only rows: **{m['reliable_positives']}**.

The 15-object deep-review shortlist is hard-negative evidence and traceability/validation material, not a training population.

## Sufficiency

**Tabular baseline:** not yet training-ready as a four-class model. There are many labels, but only a subset has consistently aligned diagnostics, and positives are sparse once catalogue-only entries requiring disposition verification are excluded. A binary/three-way feasibility experiment could become viable after deterministic feature joining and catalogue verification.

**Multi-input neural vetter:** insufficient. Only {m['p_half_p_2p']} EPICs have explicit three-hypothesis diagnostics, and equivalent view tensors are not stored at population scale. The label volume and class balance cannot support a new multi-branch neural model without public catalogue ingestion and view generation.

## Campaign and coverage gaps

The active processed universe and inference tensors are Campaign 5. `K2_ephemerides.csv` spans K2 but lacks campaign and disposition columns. There is no repository-wide, provenance-rich campaign mapping for every labelled EPIC, so campaign distribution is `C5 where evidenced; unknown otherwise`; metadata was not invented. A leakage-safe held-out campaign cannot be instantiated until other campaigns are ingested and processed consistently.

## Local catalogues and missing datasets

Available: `K2_ephemerides.csv` (NASA Exoplanet Archive-derived transit ephemerides; 512 rows, stripped disposition), `plots/k2_batch/confirmed_planet_audit/nasa_confirmed_k2_planets_reference.csv` if regenerated/located by the existing audit class, `k2_recovered_known_planets.csv`, `k2_recovered_positive_controls.csv`, `k2_real_nonplanet_systems.csv`, the manual ledger, and final ledger. The current checkout does not contain a population-scale, authoritative K2 EB catalogue, variable catalogue, or published false-positive table joined to EPIC.

Before training, ingest snapshots (do not download in this audit) of: NASA Exoplanet Archive K2 confirmed planets with disposition/provenance; ExoFOP-K2 or equivalent candidate/false-positive dispositions; the Villanova/K2 eclipsing-binary catalogue; and a published K2 variability catalogue with stable EPIC identifiers. Preserve catalogue version/date and object-level disposition.

Match by canonical digits (`EPIC 211...`, `EPIC_211...`, and integer → `EPIC_#########`). Collapse multiple planets to one host only after retaining planet rows. Precedence: confirmed planet evidence overrides candidate status but never silently overrides an EB/variable conflict; contradictory host labels go to adjudication and are excluded from training. Catalogue-only positives stay validation/traceability-only until disposition and campaign provenance are verified.

## Largest blockers

1. No normalized, provenance-rich public four-class label population.
2. Severe reliable-positive shortage in the internally reviewed set.
3. Sparse population-wide diagnostics and only {m['p_half_p_2p']} explicit P/2–P–2P cases.
4. No generated P/2–P–2P view tensor dataset.
5. Single-campaign processing prevents a genuine campaign holdout.
6. Candidate-period rows and host labels need a deterministic EPIC-level join/conflict policy.
"""
    (OUT / "PHASE2_DATA_GAP_ANALYSIS.md").write_text(gap, encoding="utf-8")

    arch = """# Phase 2 Model Architecture Proposal

## Baseline A — learned tabular vetter (train first)

Use CatBoost first: native missing values/categoricals, strong small-data behaviour, class weighting, and SHAP diagnostics. Inputs are leakage-screened numerical diagnostics, missingness indicators, the frozen CNN probability, and optionally a compact PCA projection or regularized subset of the 128-D embedding. Advantages are low complexity, inspectability, fast ablations, and graceful missingness. Limitations are loss of detailed phase morphology and dependence on consistent diagnostics. Expect at least hundreds of reliable examples per major class for a credible four-class result; current effective positives are below that bar. First experiment: class-weighted CatBoost on diagnostics + CNN probability, with uncertain rows excluded from physical loss, EPIC grouping, campaign-aware validation, and feature-family ablations. Add embeddings only after the scalar baseline is stable.

## Baseline B — multi-input neural vetter

Freeze the 128-D CNN encoder; create shared-weight view encoders for P/2, P, and 2P global/local/odd/even/secondary/stack/baseline tensors; concatenate with a masked scalar branch. Heads: four-way physical class, four-way period hypothesis, and review confidence. Advantages are direct learned morphology comparisons and multi-task sharing. Limitations are much higher implementation/QA cost, calibration difficulty, opacity, sensitivity to missing views, and substantially larger balanced data needs—preferably thousands per physical class plus broad campaigns and SNR/period coverage. It is not suitable for the current explicit three-hypothesis volume. First experiment comes only after Phase 2B: frozen encoder, small shared view tower, masked diagnostics, no CNN fine-tuning.

## Required exclusions and uncertain policy

Never input GateVetter final prediction/decision/recommendation, human/master/final label, manual reason, training label/rule, decision authority, or fields derived from those targets. Diagnostic measurements are allowed; rule verdicts should be omitted where their construction encodes an outcome.

Do not force `uncertain_hold` positive or negative. Initially exclude it from the physical-class loss and use it only to develop/evaluate `requires_manual_review` (after enough examples exist). Preserve uncertain cases for later soft-label/semi-supervised work; do not use pseudo-labels in the first baseline.

## Recommendation

Train CatBoost first, after Phase 2A review and data-gap closure. Its first job is to establish whether measurements add scientific separation beyond the frozen morphology score, with calibrated probabilities and transparent error analysis—not to automate promotion.
"""
    (OUT / "PHASE2_MODEL_ARCHITECTURE_PROPOSAL.md").write_text(arch, encoding="utf-8")

    split_eval = """# Phase 2 Split and Evaluation Plan

## Leakage-safe split policy

Create a canonical `epic_id` group key before any row expansion. All candidate periods, detections, segments, planets in multi-planet systems, aliases, and P/2–P–2P views for one EPIC stay in one split. Deduplicate/resolve label evidence before splitting. Confirmed systems repeated across files become one host group with a provenance list; conflicting physical labels are quarantined.

Preferred campaign split after multi-campaign ingestion: train on C5 plus several nonadjacent campaigns (for example C1–C4 and C6–C8), validate on C9–C11, and keep C12–C18 as an untouched blind campaign group, subject to class coverage. Do not hard-code these assignments until campaign counts are audited. With the current C5-only processed data, use grouped repeated stratified cross-validation for development and reserve a deterministic 20% EPIC blind set, stratified by class/SNR/period—but recognize that this tests within-campaign generalization only. Never tune on the blind set.

Near-duplicate periods are not independent examples: assign by EPIC first. If external catalogues reveal blends/duplicate targets for the same physical source, add a sky-position/source-system group and keep the entire group together.

## Evaluation

Report per-class precision/recall and confusion matrix, candidate recall and precision, candidate one-vs-rest PR-AUC, EB/variable recall, noise/artifact recall, and false-promotion rate = negative EPICs predicted candidate_like / all negative EPICs. For the period head report exact P/2/P/2P/unresolved accuracy, macro-F1, and confusion matrix. Calibration: multiclass log loss, Brier score, expected calibration error, and reliability plots. Select a confidence threshold on validation only; report manual-review fraction and selective risk/coverage.

Slice every metric by SNR bins, log-period bins, campaign, event count/coverage, missing-feature pattern, and the predeclared high-CNN-score hard-negative cohort. Bootstrap confidence intervals by EPIC, not row.

## First-baseline success criterion

Before seeing the blind set, define operating points. A useful first baseline should improve candidate PR-AUC over CNN-score-only and majority/class-prior baselines; retain ≥90% candidate recall on validation (with confidence intervals reported); reduce false promotions among high-CNN hard negatives by at least 30% relative to the CNN-only operating point; achieve ≥80% recall for each negative superclass where sample size supports estimation; show no catastrophic campaign/SNR slice; and produce calibrated or post-calibratable probabilities. Any threshold must route low-confidence cases to manual review. Exact deployment thresholds require more reliable positives and are not claimed by this audit.
"""
    (OUT / "PHASE2_SPLIT_AND_EVALUATION_PLAN.md").write_text(split_eval, encoding="utf-8")

    sequence = """# Phase 2 Implementation Sequence

## Phase 2A — audit and dataset contract (current gate)

1. Review this package and approve the label-source precedence and normalization mapping.
2. Ingest/version authoritative positive, EB/variable, and false-positive catalogues; do not overwrite ledgers.
3. Build one EPIC-level label table with conflicts quarantined and uncertain labels preserved.
4. Build a leakage-screened feature table plus explicit missingness/provenance flags.
5. Export frozen 128-D embeddings from the official CNN without weight changes, retaining per-segment embeddings and a documented EPIC aggregation.
6. Audit campaign/class/SNR/period coverage, then materialize the immutable group split and hash its manifests.
7. Obtain Phase 2A approval. No training before this gate.

## Phase 2B — tabular baseline

Train CatBoost with diagnostics + CNN probability first; run scalar/embedding feature-family ablations; tune only on validation; calibrate; evaluate once on blind data; publish SHAP importance, confusion/error cohorts, and hard-negative analysis. Do not promote candidates automatically.

## Phase 2C — hypothesis views and neural model

Generate deterministic masked tensors for P/2, P, 2P global/local/odd/even/secondary/event-stack/baseline views. Validate tensor coverage and invariants. Implement the frozen-CNN, shared-view, scalar multi-task model and compare against Phase 2B under the identical split.

## Phase 2D — active learning and possible fine-tuning

Prioritize model disagreements, high-uncertainty cases, candidate-like predictions, and high-CNN negatives for blinded manual review. Add labels via append-only provenance. Consider CNN fine-tuning only after the learned-vetter benefit, dataset scale, and leakage audit justify it.

## Exact next implementation task (after approval)

`python scripts/build_phase2_feature_table.py --labels docs/phase2/phase2_label_inventory.csv --output data/phase2/phase2_feature_table.parquet --audit-only`

That script does not yet exist; creating it (with tests and no training) is the proposed next task.
"""
    (OUT / "PHASE2_IMPLEMENTATION_SEQUENCE.md").write_text(sequence, encoding="utf-8")

    summary = f"""# Phase 2 Audit Summary

Phase 2 should freeze `models/k2_nocrop_flux_seed46_split303.best.keras` as a 128-D morphology encoder and move scientific classification into a learned downstream model. GateVetter remains a diagnostic/view generator; its decisions are excluded from inputs.

The union inventory contains **{m['unique_labelled_epics']} unique EPICs** with proposed normalized counts: {', '.join(f'**{k}={v}**' for k,v in sorted(norm.items()))}. Only **{m['safe_supervised']}** meet the current conservative supervised flag; **{m['reliable_positives']}** are reliable internally reviewed positives excluding stripped ephemeris-only catalogue rows. There are **{m['hard_negatives']}** manually grounded negatives with CNN score ≥0.5. Availability is uneven: diagnostics **{m['complete_diagnostics']}**, explicit P/2–P–2P comparisons **{m['p_half_p_2p']}**, accessible C5 tensors/embedding-exportable **{m['embeddings']}**, CNN scores **{m['cnn_scores']}**.

The biggest gaps are verified positive/EB/variable/public-false-positive catalogues with dispositions and campaigns, population-wide consistent diagnostics, multi-campaign coverage, and P/2–P–2P view tensors. The 15-object deep review is valuable hard-negative/traceability evidence, not sufficient training data.

Recommendation: after Phase 2A review, build the audited EPIC-level feature table and train a class-weighted CatBoost baseline before attempting a multi-input neural model. `uncertain_hold` is excluded from the physical loss and reserved for review-head work. No model was trained, no CNN or labels were modified, and no candidate batch was run.
"""
    (OUT / "PHASE2_AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv, metrics = build_label_inventory()
    write_feature_inventory()
    write_docs(inv, metrics)
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
    print(metrics)


if __name__ == "__main__":
    main()
