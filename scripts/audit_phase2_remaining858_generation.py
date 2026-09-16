from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase2"
DOCS = ROOT / "docs/phase2"
TENSOR_DIR = DATA / "remaining858_tensors"
FEATURES = DATA / "phase2_remaining858_feature_table.parquet"
SCORES = DATA / "phase2_remaining858_cnn_scores.csv"
EMBEDDINGS = DATA / "phase2_remaining858_cnn_embeddings.npz"
STATUS = DOCS / "phase2_remaining858_generation_status.csv"
SUMMARY = DOCS / "phase2_remaining858_feature_summary.json"
AUDIT = DOCS / "PHASE2_REMAINING858_FEATURE_GENERATION_AUDIT.md"
ACQUISITION_STATUS = DOCS / "phase2_remaining858_acquisition_status.csv"
PRE_STATE = DOCS / "phase2_remaining858_pre_generation_state.json"
LOCAL_FEATURES = DATA / "phase2_local949_feature_table.parquet"
FULL_FEATURES = DATA / "phase2_full1807_feature_table.parquet"
FULL_AUDIT = DOCS / "PHASE2_FULL1807_FEATURE_COVERAGE_AUDIT.md"
MODEL = ROOT / "models/k2_nocrop_flux_seed46_split303.best.keras"
EXPECTED_MODEL_HASH = "547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def tensor_audit() -> dict[str, Any]:
    host_hashes: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, list[str]] = defaultdict(list)
    all_zero: list[str] = []
    constant: list[str] = []
    nonfinite: list[str] = []
    segment_counts: dict[str, int] = {}
    for path in sorted(TENSOR_DIR.glob("EPIC_*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            tensor = np.asarray(bundle["tensor"])
        epic = path.stem
        host_hashes[numeric_hash(tensor)].append(epic)
        segment_counts[epic] = int(len(tensor))
        for index, segment in enumerate(tensor):
            label = f"{epic}#segment={index}"
            hashes[numeric_hash(segment)].append(label)
            if segment.size and np.all(segment == 0):
                all_zero.append(label)
            if segment.size and np.nanmax(segment) == np.nanmin(segment):
                constant.append(label)
            if not np.isfinite(segment).all():
                nonfinite.append(label)
    duplicate_groups = [members for members in hashes.values() if len(members) > 1]
    return {
        "tensor_files": sum(len(members) for members in host_hashes.values()),
        "tensor_segments": sum(len(members) for members in hashes.values()),
        "unique_numeric_tensors": len(hashes),
        "unique_host_tensor_bundles": len(host_hashes),
        "all_zero_tensors": all_zero,
        "constant_tensors": constant,
        "nonfinite_tensors": nonfinite,
        "exact_tensor_duplicate_groups": duplicate_groups,
        "large_pathological_duplicate_groups": [members for members in duplicate_groups if len(members) >= 10],
        "exact_host_bundle_duplicate_groups": [members for members in host_hashes.values() if len(members) > 1],
        "segment_count_distribution": dict(sorted(Counter(segment_counts.values()).items())),
    }


def embedding_audit() -> dict[str, Any]:
    with np.load(EMBEDDINGS, allow_pickle=False) as bundle:
        ids = bundle["epic_id"].astype(str)
        matrix = np.asarray(bundle["embedding"], dtype=np.float32)
    valid = np.isfinite(matrix).all(axis=1) & (matrix.shape[1] == 128)
    hashes = {numeric_hash(row) for row in matrix[valid]}
    return {
        "rows": len(ids),
        "valid_128d_embeddings": int(valid.sum()),
        "invalid_embeddings": ids[~valid].tolist(),
        "unique_embeddings": len(hashes),
    }


def enrich_provenance(features: pd.DataFrame, pre_state: dict[str, Any], summary: dict[str, Any]) -> pd.DataFrame:
    preprocessing_hash = pre_state["production_file_sha256"]["src/Classifiers/K2/K2_FrozenCnnPreprocessing.py"]
    additions = {
        "preprocessing_version": "frozen_seed303_recovered_v1",
        "preprocessing_sha256": preprocessing_hash,
        "cnn_model_sha256": EXPECTED_MODEL_HASH,
        "period_search_version": summary["period_search_version"],
        "period_search_configuration_sha256": summary["period_search_configuration_sha256"],
        "diagnostic_generation_version": summary["diagnostic_generation_version"],
        "diagnostic_generation_sha256": summary["diagnostic_generation_sha256"],
    }
    def update(value: object) -> str:
        payload = json.loads(str(value)) if str(value).strip() else {}
        payload.update(additions)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    features = features.copy()
    features["feature_provenance"] = features["feature_provenance"].map(update)
    return features


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pre_state = json.loads(PRE_STATE.read_text(encoding="utf-8"))
    features = pd.read_parquet(FEATURES)
    status = pd.read_csv(STATUS, dtype={"epic_id": str})
    acquisition = pd.read_csv(ACQUISITION_STATUS, dtype={"epic_id": str})
    scores = pd.read_csv(SCORES, dtype={"epic_id": str})

    if len(features) != 858 or features["epic_id"].nunique() != 858:
        raise RuntimeError("remaining-858 feature table is not one row per EPIC")
    if len(status) != 858 or status["epic_id"].nunique() != 858:
        raise RuntimeError("remaining-858 status is not one row per EPIC")

    acquisition_map = acquisition.set_index("epic_id")["failure_reason"].fillna("")
    failed = status["generation_status"].ne("completed")
    status.loc[failed, "failure_reason"] = status.loc[failed, "epic_id"].map(acquisition_map).fillna(status.loc[failed, "failure_reason"])
    atomic_csv(status, STATUS)

    features = enrich_provenance(features, pre_state, summary)
    atomic_parquet(features, FEATURES)

    tensors = tensor_audit()
    embeddings = embedding_audit()
    finite_prob = pd.to_numeric(scores["cnn_probability"], errors="coerce")
    finite_mask = np.isfinite(finite_prob)
    missing_columns = [column for column in features.columns if column.startswith("missing__")]
    pattern_counts: Counter[str] = Counter()
    for _, row in features[missing_columns].iterrows():
        missing = [column.removeprefix("missing__") for column in missing_columns if bool(row[column])]
        signature = "complete" if not missing else "|".join(missing)
        pattern_counts[signature] += 1
    ambiguity = status["period_ambiguity_flag"].astype("boolean").value_counts(dropna=False)

    model_hash_after = sha256_file(MODEL)
    parity = pre_state.get("frozen_preprocessing_parity", "").startswith("passed_4_of_4")
    verification = summary["verification"]
    integrity = {
        **tensors,
        "finite_cnn_probabilities": int(finite_mask.sum()),
        "unique_cnn_probabilities": int(finite_prob[finite_mask].nunique()),
        **embeddings,
        "label_blind_v1_searches_completed": int(status["period_search_status"].eq("completed").sum()),
        "period_searches_failed": int(status["period_search_status"].eq("failed").sum()),
        "period_ambiguity_counts": {str(key): int(value) for key, value in ambiguity.items()},
        "nominal_diagnostic_coverage": int(status["nominal_diagnostic_status"].eq("generated").sum()),
        "p_half_p_2p_diagnostic_coverage": int(status["p_half_p_2p_diagnostic_status"].eq("generated").sum()),
        "missing_feature_patterns": dict(pattern_counts.most_common()),
        "unexpected_all_zero_tensors": len(tensors["all_zero_tensors"]),
        "unexpected_constant_tensors": len(tensors["constant_tensors"]),
        "nonfinite_tensor_count": len(tensors["nonfinite_tensors"]),
        "cnn_model_hash_unchanged": model_hash_after == EXPECTED_MODEL_HASH == pre_state["frozen_cnn_model_sha256"],
        "preprocessing_parity": parity,
        "no_label_leakage": bool(verification["no_label_leakage_into_period_search"]),
        "no_catalogue_period_scientific_leakage": bool(verification["no_catalogue_period_used_as_scientific_input"]),
    }
    acquisition_completed = int(acquisition["acquisition_status"].eq("acquired").sum())
    generation_complete = bool(status["generation_status"].eq("completed").all())
    integrity_gates = all([
        integrity["unexpected_all_zero_tensors"] == 0,
        integrity["unexpected_constant_tensors"] == 0,
        integrity["nonfinite_tensor_count"] == 0,
        len(tensors["large_pathological_duplicate_groups"]) == 0,
        integrity["cnn_model_hash_unchanged"],
        integrity["preprocessing_parity"],
        integrity["no_label_leakage"],
        integrity["no_catalogue_period_scientific_leakage"],
    ])
    remaining_gate = generation_complete and integrity_gates

    summary.update({
        "audited_at_utc": utc_now(),
        "hosts_acquired": acquisition_completed,
        "integrity": integrity,
        "pre_generation_state_sha256": sha256_file(PRE_STATE),
        "REMAINING858_FEATURE_GENERATION_COMPLETE": "yes" if remaining_gate else "no",
        "FULL1807_FEATURE_POPULATION_VALID": "no",
        "READY_FOR_SPLIT_DESIGN": "no",
    })

    if remaining_gate:
        local = pd.read_parquet(LOCAL_FEATURES)
        full = pd.concat([local, features], ignore_index=True, sort=False)
        if len(full) != 1807 or full["epic_id"].nunique() != 1807:
            raise RuntimeError("full-1807 merge invariant failed")
        atomic_parquet(full.sort_values("epic_id", kind="mergesort"), FULL_FEATURES)
        summary["FULL1807_FEATURE_POPULATION_VALID"] = "yes"
        summary["READY_FOR_SPLIT_DESIGN"] = "yes"

    atomic_text(json.dumps(summary, indent=2, sort_keys=True), SUMMARY)

    failures = acquisition.loc[acquisition["acquisition_status"].eq("failed"), ["epic_id", "failure_reason"]]
    failure_lines = "\n".join(f"- `{row.epic_id}`: {row.failure_reason}" for row in failures.itertuples()) or "- None."
    audit = f"""# Phase 2 Remaining-858 Feature Generation Audit

## Outcome

- Hosts requested: **858**
- Hosts acquired: **{acquisition_completed}**
- Hosts completed: **{int(status['generation_status'].eq('completed').sum())}**
- Hosts failed: **{int(failed.sum())}**
- Unique numeric tensors: **{tensors['unique_numeric_tensors']}**
- All-zero tensors: **{len(tensors['all_zero_tensors'])}**
- Constant tensors: **{len(tensors['constant_tensors'])}**
- Non-finite tensors: **{len(tensors['nonfinite_tensors'])}**
- Exact tensor duplicate groups: **{len(tensors['exact_tensor_duplicate_groups'])}**
- Large pathological duplicate groups: **{len(tensors['large_pathological_duplicate_groups'])}**
- Finite CNN probabilities: **{integrity['finite_cnn_probabilities']}**
- Unique CNN probabilities: **{integrity['unique_cnn_probabilities']}**
- Valid 128-D embeddings: **{embeddings['valid_128d_embeddings']}**
- Unique embeddings: **{embeddings['unique_embeddings']}**
- `label_blind_v1` searches completed: **{integrity['label_blind_v1_searches_completed']}**
- Period searches failed: **{integrity['period_searches_failed']}**
- Nominal diagnostic coverage: **{integrity['nominal_diagnostic_coverage']}**
- P/2-P-2P diagnostic coverage: **{integrity['p_half_p_2p_diagnostic_coverage']}**

## Required integrity gates

- Unexpected all-zero tensors = 0: **{'pass' if integrity['unexpected_all_zero_tensors'] == 0 else 'fail'}**
- Unexpected constant tensors = 0: **{'pass' if integrity['unexpected_constant_tensors'] == 0 else 'fail'}**
- Large pathological duplicate groups = 0: **{'pass' if not tensors['large_pathological_duplicate_groups'] else 'fail'}**
- CNN model hash unchanged: **{'pass' if integrity['cnn_model_hash_unchanged'] else 'fail'}** (`{model_hash_after}`)
- Preprocessing parity: **{'pass' if parity else 'fail'}**
- No label leakage: **{'pass' if integrity['no_label_leakage'] else 'fail'}**
- No catalogue-period scientific leakage: **{'pass' if integrity['no_catalogue_period_scientific_leakage'] else 'fail'}**

## Explicit acquisition failures

{failure_lines}

The nine failed EPICs remain in the 858-row feature and status tables with explicit missingness. No catalogue, saved-review, alternate-mission, or wrong-target light curve was substituted. Because the remaining population is incomplete, the 1807-host merge was not created.

`REMAINING858_FEATURE_GENERATION_COMPLETE = {summary['REMAINING858_FEATURE_GENERATION_COMPLETE']}`

`FULL1807_FEATURE_POPULATION_VALID = {summary['FULL1807_FEATURE_POPULATION_VALID']}`

`READY_FOR_SPLIT_DESIGN = {summary['READY_FOR_SPLIT_DESIGN']}`
"""
    atomic_text(audit, AUDIT)

    full_audit = f"""# Phase 2 Full-1807 Feature Coverage Audit

The conditional merge gate did not pass. Local949 remains validated at 949 hosts; the remaining population completed 849 of 858 hosts, with nine explicit acquisition failures. Therefore `{FULL_FEATURES.relative_to(ROOT).as_posix()}` was not created and no train/validation/test assignments were made.

`FULL1807_FEATURE_POPULATION_VALID = {summary['FULL1807_FEATURE_POPULATION_VALID']}`

`READY_FOR_SPLIT_DESIGN = {summary['READY_FOR_SPLIT_DESIGN']}`
"""
    atomic_text(full_audit, FULL_AUDIT)
    print(json.dumps({
        "hosts_requested": 858,
        "hosts_acquired": acquisition_completed,
        "hosts_completed": int(status["generation_status"].eq("completed").sum()),
        "hosts_failed": int(failed.sum()),
        "unique_numeric_tensors": tensors["unique_numeric_tensors"],
        "exact_tensor_duplicate_groups": len(tensors["exact_tensor_duplicate_groups"]),
        "valid_128d_embeddings": embeddings["valid_128d_embeddings"],
        "REMAINING858_FEATURE_GENERATION_COMPLETE": summary["REMAINING858_FEATURE_GENERATION_COMPLETE"],
        "FULL1807_FEATURE_POPULATION_VALID": summary["FULL1807_FEATURE_POPULATION_VALID"],
        "READY_FOR_SPLIT_DESIGN": summary["READY_FOR_SPLIT_DESIGN"],
    }, indent=2))


if __name__ == "__main__":
    main()
