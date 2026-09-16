from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OLD_ALL_ZERO_TENSOR_HASHES = {
    "4f81904a9b06c58572a0e5769b3b4ffb99e7bd4be88ee8c2b64a804f483d9dc6",
    "2aae7dc846aaf25f1cadf55f1666862046c6db9d65d84bdc07fa039dac405606",
}
OLD_PROBABILITY_PILEUP = 0.8580068945884705


def numeric_sha256(array: np.ndarray) -> str:
    """Hash contiguous numeric bytes, never NPZ/container metadata."""
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def tensor_quality(epic_id: str, tensor: np.ndarray) -> dict[str, Any]:
    value = np.asarray(tensor)
    finite = np.isfinite(value)
    finite_values = value[finite]
    unique_values = np.unique(finite_values) if finite_values.size else np.empty(0, dtype=value.dtype)
    all_zero = bool(value.size and finite.all() and np.all(value == 0))
    constant = bool(value.size and finite.all() and unique_values.size == 1)
    nonfinite = bool(not finite.all())
    unexpected_shape = bool(value.ndim != 3 or tuple(value.shape[1:]) != (512, 2) or value.shape[0] < 1)
    clipping_fraction = float(np.mean(np.abs(value[finite]) >= 10.0)) if finite_values.size else 0.0
    flags = []
    if all_zero:
        flags.append("all_zero_tensor")
    if constant:
        flags.append("constant_tensor")
    if nonfinite:
        flags.append("non_finite_tensor")
    if unexpected_shape:
        flags.append("unexpected_shape")
    if clipping_fraction > 0.25:
        flags.append("extreme_clipping_fraction")
    return {
        "epic_id": epic_id,
        "tensor_shape": json.dumps(list(value.shape), separators=(",", ":")),
        "tensor_dtype": str(value.dtype),
        "tensor_sha256_numeric_bytes": numeric_sha256(value),
        "tensor_min": _finite_float(np.min(finite_values)) if finite_values.size else None,
        "tensor_max": _finite_float(np.max(finite_values)) if finite_values.size else None,
        "tensor_std": _finite_float(np.std(finite_values)) if finite_values.size else None,
        "tensor_unique_values": int(unique_values.size),
        "tensor_fraction_zero": float(np.mean(value == 0)) if value.size else 0.0,
        "tensor_fraction_nan": float(np.mean(np.isnan(value))) if value.size else 0.0,
        "tensor_fraction_nonfinite": float(np.mean(~finite)) if value.size else 0.0,
        "tensor_clipping_fraction": clipping_fraction,
        "all_zero_tensor": all_zero,
        "constant_tensor": constant,
        "non_finite_tensor": nonfinite,
        "unexpected_tensor_shape": unexpected_shape,
        "quality_flags": "|".join(flags),
    }


def audit_generated470(
    tensor_paths: Mapping[str, Path],
    probabilities: Mapping[str, float],
    embeddings: Mapping[str, np.ndarray],
    *,
    target_classes: Mapping[str, str] | None = None,
    old_shared_embedding_hash: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """General post-repair audit with no expected-bad-group assertions."""
    epics = sorted(tensor_paths)
    if set(epics) != set(probabilities) or set(epics) != set(embeddings):
        raise AssertionError("tensor/probability/embedding host sets differ")

    records: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    hashes_to_epics: dict[str, list[str]] = defaultdict(list)
    for epic in epics:
        path = Path(tensor_paths[epic])
        with np.load(path, allow_pickle=False) as bundle:
            tensor = np.asarray(bundle["tensor"])
        record = tensor_quality(epic, tensor)
        record["model_physical_target"] = (target_classes or {}).get(epic, "")
        records.append(record)
        tensors[epic] = tensor
        hashes_to_epics[record["tensor_sha256_numeric_bytes"]].append(epic)

    duplicate_groups: list[dict[str, Any]] = []
    for digest, group_epics in sorted(hashes_to_epics.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group_epics) < 2:
            continue
        exemplar = tensors[group_epics[0]]
        if not all(np.array_equal(exemplar, tensors[epic], equal_nan=True) for epic in group_epics[1:]):
            raise AssertionError(f"numeric SHA-256 collision for {digest}")
        duplicate_groups.append({
            "tensor_sha256_numeric_bytes": digest,
            "host_count": len(group_epics),
            "all_zero": bool(np.all(exemplar == 0)),
            "constant": bool(np.isfinite(exemplar).all() and np.unique(exemplar).size == 1),
            "epic_ids": group_epics,
        })

    probability_values = np.asarray([probabilities[epic] for epic in epics], dtype=np.float64)
    embedding_values = [np.asarray(embeddings[epic], dtype=np.float32) for epic in epics]
    embedding_shape_valid = all(value.shape == (128,) for value in embedding_values)
    embeddings_finite = all(np.isfinite(value).all() for value in embedding_values)
    embedding_hashes = [numeric_sha256(value) for value in embedding_values]
    old_hash_count = embedding_hashes.count(old_shared_embedding_hash) if old_shared_embedding_hash else 0

    all_zero_count = sum(bool(row["all_zero_tensor"]) for row in records)
    constant_count = sum(bool(row["constant_tensor"]) for row in records)
    nonfinite_count = sum(bool(row["non_finite_tensor"]) for row in records)
    unexpected_shape_count = sum(bool(row["unexpected_tensor_shape"]) for row in records)
    large_groups = [group for group in duplicate_groups if int(group["host_count"]) >= 10]
    pileup_count = int(np.sum(probability_values == OLD_PROBABILITY_PILEUP))
    old_tensor_hash_count = sum(
        len(group_epics)
        for digest, group_epics in hashes_to_epics.items()
        if digest in OLD_ALL_ZERO_TENSOR_HASHES
    )
    flagged_target_counts: dict[str, int] = defaultdict(int)
    for row in records:
        if row["quality_flags"]:
            flagged_target_counts[str(row["model_physical_target"])] += 1

    summary = {
        "hosts_audited": len(epics),
        "unique_numeric_tensors": len(hashes_to_epics),
        "exact_duplicate_group_count": len(duplicate_groups),
        "hosts_in_duplicate_groups": int(sum(group["host_count"] for group in duplicate_groups)),
        "duplicate_groups": duplicate_groups,
        "unexpected_all_zero_tensors": int(all_zero_count),
        "unexpected_constant_tensors": int(constant_count),
        "non_finite_tensors": int(nonfinite_count),
        "unexpected_tensor_shapes": int(unexpected_shape_count),
        "large_pathological_duplicate_groups": len(large_groups),
        "large_duplicate_groups": large_groups,
        "old_230_118_tensor_hash_hosts_remaining": int(old_tensor_hash_count),
        "old_probability_pileup_count": pileup_count,
        "old_shared_all_zero_embedding_hash_count": int(old_hash_count),
        "unique_cnn_probabilities": int(np.unique(probability_values).size),
        "unique_128d_embeddings": len(set(embedding_hashes)),
        "all_CNN_probabilities_finite": bool(np.isfinite(probability_values).all()),
        "all_embeddings_shape_128": bool(embedding_shape_valid),
        "all_embeddings_finite": bool(embeddings_finite),
        "flagged_host_target_class_counts": dict(sorted(flagged_target_counts.items())),
        "tensor_hash_definition": "SHA-256 of np.ascontiguousarray(tensor).tobytes(order='C')",
    }
    return summary, records, duplicate_groups


def integrity_gates_pass(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary["hosts_audited"] == 470
        and summary["unexpected_all_zero_tensors"] == 0
        and summary["unexpected_constant_tensors"] == 0
        and summary["non_finite_tensors"] == 0
        and summary["unexpected_tensor_shapes"] == 0
        and summary["large_pathological_duplicate_groups"] == 0
        and summary["old_230_118_tensor_hash_hosts_remaining"] == 0
        and summary["old_probability_pileup_count"] == 0
        and summary["old_shared_all_zero_embedding_hash_count"] == 0
        and summary["all_CNN_probabilities_finite"]
        and summary["all_embeddings_shape_128"]
        and summary["all_embeddings_finite"]
    )
