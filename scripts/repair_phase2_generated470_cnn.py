from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_phase2_generated470_repair import audit_generated470, integrity_gates_pass, numeric_sha256
from src.Classifiers.K2.K2_FrozenCnnPreprocessing import FrozenK2CNNTensorBuilder


DATA = ROOT / "data/phase2"
DOCS = ROOT / "docs/phase2"
TENSOR_DIR = DATA / "local949_tensors"
MODEL = ROOT / "models/k2_nocrop_flux_seed46_split303.best.keras"
PREPROCESSOR = ROOT / "src/Classifiers/K2/K2_FrozenCnnPreprocessing.py"
STATUS_SOURCE = DOCS / "phase2_local949_generation_status.csv"
FEATURE_TABLE = DATA / "phase2_local949_feature_table.parquet"
SCORES = DATA / "phase2_local949_cnn_scores.csv"
EMBEDDINGS = DATA / "phase2_local949_cnn_embeddings.npz"
REPAIR_STATUS = DOCS / "phase2_generated470_repair_status.csv"
REPAIR_SUMMARY = DOCS / "phase2_generated470_repair_summary.json"
REPAIR_AUDIT = DOCS / "PHASE2_GENERATED470_CNN_REPAIR_AUDIT.md"
EXPECTED_MODEL_SHA256 = "547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443"
PREPROCESSING_VERSION = "frozen_k2_cnn_seed303_v1"
EMBEDDING_LAYER = "global_average_pooling1d_2"

PROTECTED_ARTIFACTS = [
    DATA / "phase2_local949_label_blind_period_candidates.parquet",
    DATA / "phase2_local949_diagnostics.parquet",
    DOCS / "phase2_feature_generation_manifest_preview.csv",
    DOCS / "phase2_local949_generation_status.csv",
    DOCS / "phase2_label_inventory.csv",
    DOCS / "phase2_target_contract.json",
]
TRUSTED_TENSOR_SOURCES = [
    ROOT / "splits/infer_c5/X_infer.npy",
    ROOT / "splits/infer_c5/meta_infer.parquet",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hashes(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required protected artifacts missing: {missing}")
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


@contextmanager
def workspace_staging(stamp: str):
    """Use an inheriting workspace directory; tempfile's Windows ACL is unsuitable here."""
    path = DATA / f"generated470_repair_staging_{stamp}"
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def fits_time_flux(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names or [])
        flux_column = next((name for name in ("FLUX", "PDCSAP_FLUX", "SAP_FLUX") if name in names), None)
        if flux_column is None:
            raise ValueError(f"no supported flux column in {path}")
        return (
            np.asarray(data["TIME"], dtype=np.float64).reshape(-1),
            np.asarray(data[flux_column], dtype=np.float32).reshape(-1),
        )


def parse_input_paths(value: str) -> list[Path]:
    paths = [Path(item) for item in json.loads(value)]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"registered local light curve is unavailable: {paths}")
    return paths


def input_hash_json(paths: list[Path]) -> str:
    return json.dumps({str(path): file_sha256(path) for path in paths}, sort_keys=True, separators=(",", ":"))


def build_tensor(builder: FrozenK2CNNTensorBuilder, epic: str, paths: list[Path]) -> tuple[np.ndarray, pd.DataFrame]:
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        time, flux = fits_time_flux(path)
        tensor, starts, ends, mid_times = builder.build_tensor(time, flux)
        for index in range(len(tensor)):
            arrays.append(tensor[index])
            metadata.append({
                "epic_id": epic,
                "start": int(starts[index]),
                "end": int(ends[index]),
                "seg_mid_time": float(mid_times[index]),
                "input_light_curve_path": str(path),
            })
    if not arrays:
        raise ValueError(f"{epic}: no complete 512-sample window")
    return np.stack(arrays).astype(np.float32), pd.DataFrame(metadata)


def save_tensor(path: Path, tensor: np.ndarray, metadata: pd.DataFrame, generated_at: str, code_hash: str) -> None:
    np.savez_compressed(
        path,
        tensor=np.asarray(tensor, dtype=np.float32),
        start=metadata["start"].to_numpy(np.int64),
        end=metadata["end"].to_numpy(np.int64),
        seg_mid_time=metadata["seg_mid_time"].to_numpy(float),
        input_light_curve_path=metadata["input_light_curve_path"].to_numpy(str),
        preprocessing_version=np.array(PREPROCESSING_VERSION),
        preprocessing_code_sha256=np.array(code_hash),
        tensor_generated_at=np.array(generated_at),
    )


def raw_score_lines(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != 950:
        raise AssertionError(f"expected score header plus 949 records, found {len(lines)} lines")
    result: dict[str, str] = {}
    for line in lines[1:]:
        fields = next(csv.reader([line]))
        result[fields[0]] = line
    if len(result) != 949:
        raise AssertionError("score CSV does not contain 949 unique EPIC rows")
    return lines, result


def merge_score_lines(
    source: Path,
    destination: Path,
    repair_details: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    lines, before_by_epic = raw_score_lines(source)
    reader = csv.DictReader(io.StringIO("".join(lines)))
    rows = {str(row["epic_id"]): row for row in reader}
    fieldnames = list(reader.fieldnames or [])
    output = [lines[0]]
    for original_line in lines[1:]:
        epic = next(csv.reader([original_line]))[0]
        if epic not in repair_details:
            output.append(original_line)
            continue
        detail = repair_details[epic]
        row = rows[epic]
        row.update({
            "cnn_probability": repr(float(detail["cnn_probability"])),
            "cnn_probability_action": "repaired_frozen_seed303",
            "cnn_model_path": MODEL.relative_to(ROOT).as_posix(),
            "cnn_model_sha256": EXPECTED_MODEL_SHA256,
            "cnn_inference_training": "False",
            "cnn_weights_modified": "False",
            "cnn_embedding_layer": EMBEDDING_LAYER,
            "cnn_embedding_dim": "128",
            "cnn_embedding_aggregation": "max_probability_segment",
            "cnn_segment_count": str(detail["cnn_segment_count"]),
            "cnn_best_segment_start": str(detail["cnn_best_segment_start"]),
            "cnn_best_segment_end": str(detail["cnn_best_segment_end"]),
            "cnn_best_segment_mid_time": repr(float(detail["cnn_best_segment_mid_time"])),
            "tensor_sha256": detail["tensor_sha256_numeric_bytes"],
        })
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writerow(row)
        output.append(buffer.getvalue())
    destination.write_text("".join(output), encoding="utf-8", newline="")
    _, after_by_epic = raw_score_lines(destination)
    return before_by_epic, after_by_epic


def git_state(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    return {
        "path": relative,
        "tracked": tracked,
        "git_state": "tracked" if tracked else "untracked",
        "sha256": file_sha256(path),
    }


def markdown_audit(summary: dict[str, Any]) -> str:
    integrity = summary["generated470_integrity"]
    merge = summary["local949_merge_integrity"]
    protected = summary["protected_artifact_hashes"]
    git_rows = summary["git_tracking_state"]
    backups = summary["versioned_pre_repair_artifacts"]
    duplicate_rows = integrity["duplicate_groups"]
    protected_lines = [
        f"- `{path}`: `{values['before']}` → `{values['after']}`; unchanged={str(values['unchanged']).lower()}"
        for path, values in protected.items()
    ]
    git_lines = [
        f"- `{row['path']}`: {row['git_state']}; SHA-256 `{row['sha256']}`"
        for row in git_rows
    ]
    backup_lines = [f"- `{source}` → `{backup}`" for source, backup in backups.items()]
    duplicate_lines = [
        f"- {group['host_count']} hosts, tensor `{group['tensor_sha256_numeric_bytes']}`, EPICs: {', '.join(group['epic_ids'])}"
        for group in duplicate_rows
    ] or ["- None."]
    return f"""# Phase 2 generated-470 CNN repair audit

## Outcome

- Repair hosts regenerated: **{summary['repair_population']['repair_hosts']}**
- Trusted reused hosts preserved: **{summary['repair_population']['trusted_reused_hosts']}**
- Frozen preprocessing parity passed: **{str(summary['parity_gate']['passed']).lower()}**
- Frozen model hash before/after: `{summary['cnn_model_sha256_before']}` / `{summary['cnn_model_sha256_after']}`

## Generated-470 tensor and CNN integrity

- Unique numeric tensors: **{integrity['unique_numeric_tensors']}**
- Exact duplicate groups: **{integrity['exact_duplicate_group_count']}**
- Hosts in duplicate groups: **{integrity['hosts_in_duplicate_groups']}**
- Unexpected all-zero tensors: **{integrity['unexpected_all_zero_tensors']}**
- Unexpected constant tensors: **{integrity['unexpected_constant_tensors']}**
- Non-finite tensors: **{integrity['non_finite_tensors']}**
- Unexpected tensor shapes: **{integrity['unexpected_tensor_shapes']}**
- Unique CNN probabilities: **{integrity['unique_cnn_probabilities']}**
- Unique 128-D embeddings: **{integrity['unique_128d_embeddings']}**
- Large pathological duplicate groups: **{integrity['large_pathological_duplicate_groups']}**
- Old 230/118 tensor hashes remaining: **{integrity['old_230_118_tensor_hash_hosts_remaining']}**
- Old `0.8580068945884705` pile-up count: **{integrity['old_probability_pileup_count']}**
- Old shared all-zero embedding remaining: **{integrity['old_shared_all_zero_embedding_hash_count']}**

Tensor hashes are SHA-256 over `np.ascontiguousarray(tensor).tobytes(order="C")` only.

## Exact duplicate groups

{chr(10).join(duplicate_lines)}

## Local949 merge integrity

- Total hosts: **{merge['total_hosts']}**
- Valid CNN probabilities: **{merge['valid_cnn_probabilities']}**
- Valid 128-D embeddings: **{merge['valid_128d_embeddings']}**
- Valid tensor provenance records: **{merge['valid_tensor_provenance_records']}**
- Trusted tensor source files byte-identical: **{str(merge['trusted_reused_tensors_byte_identical']).lower()}**
- Trusted score records byte-identical: **{str(merge['trusted_reused_score_records_byte_identical']).lower()}**
- Trusted probability/embedding bytes identical: **{str(merge['trusted_reused_cnn_arrays_byte_identical']).lower()}**
- Trusted feature-table CNN rows identical: **{str(merge['trusted_reused_feature_rows_identical']).lower()}**

## Protected artifacts

{chr(10).join(protected_lines)}

The Local949 period-candidate and diagnostic files were hash-checked before and after the repair. The generation status, label inventory, target contract, eligibility, and target mappings were not rewritten.

## Versioned pre-repair artifacts

{chr(10).join(backup_lines)}

## Code provenance and Git state

{chr(10).join(git_lines)}

The repaired tensor status records contain the frozen preprocessing version and exact preprocessing code SHA-256. The repair and generalized-audit code hashes are recorded in the JSON summary.

## Non-actions

- No light curves were acquired.
- The remaining 858 hosts were not accessed.
- Period searches and numerical diagnostics were not rerun.
- Labels, eligibility, and target mappings were not changed.
- The CNN and CatBoost were not trained.
- No split assignments were made.

`GENERATED470_CNN_REPAIR_COMPLETE = {summary['GENERATED470_CNN_REPAIR_COMPLETE']}`
`LOCAL949_CNN_FEATURES_VALID = {summary['LOCAL949_CNN_FEATURES_VALID']}`
`READY_TO_ACQUIRE_REMAINING_858 = {summary['READY_TO_ACQUIRE_REMAINING_858']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair only the generated-470 Local949 CNN branch.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    run_started = utc_now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    required = [MODEL, PREPROCESSOR, STATUS_SOURCE, FEATURE_TABLE, SCORES, EMBEDDINGS, *PROTECTED_ARTIFACTS, *TRUSTED_TENSOR_SOURCES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required repair inputs missing: {missing}")

    parity = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_phase2_frozen_cnn_preprocessing_parity", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if parity.returncode != 0:
        raise RuntimeError(f"frozen preprocessing parity gate failed:\n{parity.stdout}\n{parity.stderr}")

    source_status = pd.read_csv(STATUS_SOURCE, dtype={"epic_id": str})
    repair_frame = source_status.loc[source_status["tensor_action"].eq("generate")].copy()
    trusted_frame = source_status.loc[source_status["tensor_action"].eq("reuse_existing")].copy()
    repair_ids = set(repair_frame["epic_id"])
    trusted_ids = set(trusted_frame["epic_id"])
    if not (
        len(repair_frame) == len(repair_ids) == 470
        and len(trusted_frame) == len(trusted_ids) == 479
        and not repair_ids.intersection(trusted_ids)
        and len(repair_ids.union(trusted_ids)) == 949
    ):
        raise AssertionError("repair population gate failed: expected disjoint 470 + 479 = 949")

    protected_before = stable_hashes(PROTECTED_ARTIFACTS)
    trusted_tensor_sources_before = stable_hashes(TRUSTED_TENSOR_SOURCES)
    model_hash_before = file_sha256(MODEL)
    if model_hash_before != EXPECTED_MODEL_SHA256:
        raise AssertionError(f"frozen model hash mismatch before repair: {model_hash_before}")
    preprocess_hash = file_sha256(PREPROCESSOR)
    repair_script_hash = file_sha256(Path(__file__))
    generalized_audit_path = ROOT / "scripts/audit_phase2_generated470_repair.py"
    generalized_audit_hash = file_sha256(generalized_audit_path)

    with np.load(EMBEDDINGS, allow_pickle=False) as bundle:
        old_embedding_bundle = {key: np.array(value, copy=True) for key, value in bundle.items()}
    old_ids = old_embedding_bundle["epic_id"].astype(str)
    if len(old_ids) != 949 or len(set(old_ids)) != 949:
        raise AssertionError("pre-repair embedding bundle is not one row per Local949 host")
    old_index = {epic: index for index, epic in enumerate(old_ids)}
    old_embedding_matrix = np.asarray(old_embedding_bundle["embedding"], dtype=np.float32)
    old_probability_array = np.asarray(old_embedding_bundle["cnn_probability"], dtype=np.float32)
    old_feature_table = pd.read_parquet(FEATURE_TABLE)
    old_feature_table["epic_id"] = old_feature_table["epic_id"].astype(str)
    if set(old_feature_table["epic_id"]) != repair_ids.union(trusted_ids):
        raise AssertionError("feature table host set does not equal Local949 host set")

    old_shared_embedding_hash = None
    old_duplicate_summary = DOCS / "phase2_generated470_tensor_duplicate_summary.json"
    if old_duplicate_summary.exists():
        old_shared_embedding_hash = json.loads(old_duplicate_summary.read_text(encoding="utf-8")).get("large_groups_shared_embedding_hash")

    builder = FrozenK2CNNTensorBuilder()
    generated_at = utc_now()
    detail_by_epic: dict[str, dict[str, Any]] = {}
    staged_tensor_paths: dict[str, Path] = {}
    tensor_metadata: dict[str, pd.DataFrame] = {}
    input_arrays: list[np.ndarray] = []
    segment_slices: dict[str, slice] = {}

    with workspace_staging(stamp) as staging:
        staged_tensors = staging / "tensors"
        staged_tensors.mkdir(parents=True)
        offset = 0
        for number, row in enumerate(repair_frame.sort_values("epic_id").itertuples(index=False), start=1):
            epic = str(row.epic_id)
            paths = parse_input_paths(str(row.input_light_curve_path))
            current_input_hashes = input_hash_json(paths)
            registered_hashes = json.dumps(json.loads(str(row.input_light_curve_sha256)), sort_keys=True, separators=(",", ":"))
            if current_input_hashes != registered_hashes:
                raise AssertionError(f"{epic}: registered input light-curve hash changed")
            tensor, metadata = build_tensor(builder, epic, paths)
            tensor_path = staged_tensors / f"{epic}.npz"
            save_tensor(tensor_path, tensor, metadata, generated_at, preprocess_hash)
            staged_tensor_paths[epic] = tensor_path
            tensor_metadata[epic] = metadata
            input_arrays.append(tensor[:, :, :1])
            segment_slices[epic] = slice(offset, offset + len(tensor))
            offset += len(tensor)
            detail_by_epic[epic] = {
                "epic_id": epic,
                "input_light_curve_path": str(row.input_light_curve_path),
                "input_light_curve_sha256": current_input_hashes,
                "preprocessing_version": PREPROCESSING_VERSION,
                "preprocessing_code_sha256": preprocess_hash,
                "tensor_shape": json.dumps(list(tensor.shape), separators=(",", ":")),
                "tensor_sha256_numeric_bytes": numeric_sha256(tensor),
                "tensor_generated_at": generated_at,
                "cnn_segment_count": len(tensor),
                "model_physical_target": str(row.model_physical_target),
            }
            if number % args.progress_every == 0 or number == 470:
                print(f"stage=tensor_regeneration processed={number}/470", flush=True)

        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL, compile=False)
        model.trainable = False
        if tuple(model.input_shape[1:]) != (512, 1):
            raise AssertionError(f"unexpected frozen CNN input shape: {model.input_shape}")
        encoder = tf.keras.Model(model.input, model.get_layer(EMBEDDING_LAYER).output)
        encoder.trainable = False
        if int(encoder.output_shape[-1]) != 128:
            raise AssertionError(f"unexpected embedding shape: {encoder.output_shape}")

        cnn_input = np.concatenate(input_arrays, axis=0).astype(np.float32)
        segment_probabilities = np.asarray(model.predict(cnn_input, batch_size=512, verbose=0), dtype=np.float32).reshape(-1)
        segment_embeddings = np.asarray(encoder.predict(cnn_input, batch_size=512, verbose=0), dtype=np.float32)
        probabilities: dict[str, float] = {}
        embeddings: dict[str, np.ndarray] = {}
        for epic in sorted(repair_ids):
            selection = segment_slices[epic]
            local_probabilities = segment_probabilities[selection]
            local_embeddings = segment_embeddings[selection]
            if not np.isfinite(local_probabilities).all() or local_embeddings.shape[1:] != (128,) or not np.isfinite(local_embeddings).all():
                raise AssertionError(f"{epic}: invalid frozen CNN output")
            best = int(np.argmax(local_probabilities))
            metadata = tensor_metadata[epic].iloc[best]
            probabilities[epic] = float(local_probabilities[best])
            embeddings[epic] = np.asarray(local_embeddings[best], dtype=np.float32)
            detail_by_epic[epic].update({
                "cnn_probability": probabilities[epic],
                "cnn_embedding_sha256_numeric_bytes": numeric_sha256(embeddings[epic]),
                "cnn_embedding_shape": "[128]",
                "cnn_segment_count": int(selection.stop - selection.start),
                "cnn_best_segment_start": int(metadata["start"]),
                "cnn_best_segment_end": int(metadata["end"]),
                "cnn_best_segment_mid_time": float(metadata["seg_mid_time"]),
            })

        generated_integrity, quality_records, duplicate_groups = audit_generated470(
            staged_tensor_paths,
            probabilities,
            embeddings,
            target_classes={str(row.epic_id): str(row.model_physical_target) for row in repair_frame.itertuples(index=False)},
            old_shared_embedding_hash=old_shared_embedding_hash,
        )
        if not integrity_gates_pass(generated_integrity):
            raise AssertionError(f"generated-470 integrity gates failed: {json.dumps(generated_integrity, sort_keys=True)}")

        for quality in quality_records:
            detail_by_epic[str(quality["epic_id"])].update(quality)

        staged_scores = staging / SCORES.name
        score_before, score_after = merge_score_lines(SCORES, staged_scores, detail_by_epic)
        trusted_score_identical = all(score_before[epic] == score_after[epic] for epic in trusted_ids)

        new_embedding_bundle = {key: np.array(value, copy=True) for key, value in old_embedding_bundle.items()}
        new_embedding_matrix = np.asarray(new_embedding_bundle["embedding"], dtype=np.float32).copy()
        new_probability_array = np.asarray(new_embedding_bundle["cnn_probability"], dtype=np.float32).copy()
        provenance_values = new_embedding_bundle["embedding_provenance"].astype(str).tolist()
        for epic in repair_ids:
            index = old_index[epic]
            new_embedding_matrix[index] = embeddings[epic]
            new_probability_array[index] = probabilities[epic]
            provenance_values[index] = (
                f"data/phase2/local949_tensors/{epic}.npz->{MODEL.relative_to(ROOT).as_posix()}"
                f"#{EMBEDDING_LAYER};training=False;preprocessing={PREPROCESSING_VERSION}"
            )
        new_embedding_bundle["embedding"] = new_embedding_matrix
        new_embedding_bundle["cnn_probability"] = new_probability_array
        new_embedding_bundle["embedding_provenance"] = np.asarray(provenance_values)
        staged_embeddings = staging / EMBEDDINGS.name
        np.savez_compressed(staged_embeddings, **new_embedding_bundle)

        trusted_indices = np.asarray([old_index[epic] for epic in sorted(trusted_ids)], dtype=int)
        trusted_cnn_arrays_identical = bool(
            np.array_equal(old_embedding_matrix[trusted_indices], new_embedding_matrix[trusted_indices])
            and np.array_equal(old_probability_array[trusted_indices], new_probability_array[trusted_indices])
        )

        new_feature_table = old_feature_table.copy(deep=True)
        new_feature_table = new_feature_table.set_index("epic_id", drop=False)
        old_feature_indexed = old_feature_table.set_index("epic_id", drop=False)
        embedding_columns = [f"cnn_embedding_{index:03d}" for index in range(128)]
        for epic in repair_ids:
            detail = detail_by_epic[epic]
            new_feature_table.at[epic, "cnn_probability"] = detail["cnn_probability"]
            new_feature_table.at[epic, "cnn_segment_count"] = detail["cnn_segment_count"]
            for index, column in enumerate(embedding_columns):
                new_feature_table.at[epic, column] = embeddings[epic][index]
                missing_column = f"missing__{column}"
                if missing_column in new_feature_table.columns:
                    new_feature_table.at[epic, missing_column] = False
            if "missing__cnn_probability" in new_feature_table.columns:
                new_feature_table.at[epic, "missing__cnn_probability"] = False
            if "missing__cnn_segment_count" in new_feature_table.columns:
                new_feature_table.at[epic, "missing__cnn_segment_count"] = False
            provenance = json.loads(str(new_feature_table.at[epic, "feature_provenance"]))
            provenance.update({
                "tensor_path": f"data/phase2/local949_tensors/{epic}.npz",
                "tensor_sha256": detail["tensor_sha256_numeric_bytes"],
                "tensor_action": "repair_generated470_frozen_seed303",
                "preprocessing_version": PREPROCESSING_VERSION,
                "preprocessing_code_sha256": preprocess_hash,
                "tensor_generated_at": generated_at,
                "embedding_provenance": provenance_values[old_index[epic]],
            })
            new_feature_table.at[epic, "feature_provenance"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        new_feature_table = new_feature_table.reset_index(drop=True)
        staged_feature_table = staging / FEATURE_TABLE.name
        new_feature_table.to_parquet(staged_feature_table, index=False)
        verified_feature_table = pd.read_parquet(staged_feature_table).set_index("epic_id", drop=False)
        trusted_feature_columns = [
            "cnn_probability", "cnn_model_path", "cnn_model_sha256", "cnn_embedding_layer",
            "cnn_embedding_dim", "cnn_embedding_aggregation", "cnn_segment_count",
            *embedding_columns, "feature_provenance",
        ]
        trusted_feature_rows_identical = old_feature_indexed.loc[sorted(trusted_ids), trusted_feature_columns].equals(
            verified_feature_table.loc[sorted(trusted_ids), trusted_feature_columns]
        )
        untouched_columns = [
            column for column in old_feature_table.columns
            if column not in {"cnn_probability", "cnn_segment_count", "feature_provenance", *embedding_columns}
            and not column.startswith("missing__cnn_")
        ]
        if not old_feature_indexed.loc[:, untouched_columns].equals(verified_feature_table.loc[:, untouched_columns]):
            raise AssertionError("a non-CNN feature-table column changed during staged merge")

        merged_scores = pd.read_csv(staged_scores, dtype={"epic_id": str})
        with np.load(staged_embeddings, allow_pickle=False) as merged_embedding_check:
            merged_ids = merged_embedding_check["epic_id"].astype(str)
            merged_probability = np.asarray(merged_embedding_check["cnn_probability"], dtype=np.float32).copy()
            merged_embedding = np.asarray(merged_embedding_check["embedding"], dtype=np.float32).copy()
        valid_cnn_probabilities = int(np.isfinite(merged_scores["cnn_probability"].to_numpy(float)).sum())
        valid_embeddings = int(sum(value.shape == (128,) and np.isfinite(value).all() for value in merged_embedding))
        if not (
            len(merged_scores) == merged_scores["epic_id"].nunique() == 949
            and len(merged_ids) == len(set(merged_ids)) == 949
            and valid_cnn_probabilities == 949
            and valid_embeddings == 949
            and np.isfinite(merged_probability).all()
        ):
            raise AssertionError("staged Local949 CNN merge is incomplete or non-finite")

        tensor_archive = DATA / f"local949_tensors.invalid_pre_generated470_repair_{stamp}"
        if tensor_archive.exists():
            raise FileExistsError(tensor_archive)
        tensor_archive.mkdir(parents=True)
        backup_map: dict[str, str] = {}
        for source in (FEATURE_TABLE, SCORES, EMBEDDINGS):
            backup = source.with_name(source.name + f".pre_generated470_repair_{stamp}")
            shutil.copy2(source, backup)
            backup_map[source.relative_to(ROOT).as_posix()] = backup.relative_to(ROOT).as_posix()
        backup_map[TENSOR_DIR.relative_to(ROOT).as_posix()] = tensor_archive.relative_to(ROOT).as_posix()

        for epic in sorted(repair_ids):
            old_tensor = TENSOR_DIR / f"{epic}.npz"
            if not old_tensor.exists():
                raise FileNotFoundError(old_tensor)
            archived = tensor_archive / old_tensor.name
            shutil.copy2(old_tensor, archived)
            detail_by_epic[epic]["invalid_tensor_archive_path"] = archived.relative_to(ROOT).as_posix()
            with np.load(archived, allow_pickle=False) as bundle:
                detail_by_epic[epic]["invalid_tensor_sha256_numeric_bytes"] = numeric_sha256(bundle["tensor"])

        for epic in sorted(repair_ids):
            temporary = TENSOR_DIR / f".{epic}.{stamp}.tmp.npz"
            shutil.copy2(staged_tensor_paths[epic], temporary)
            atomic_replace(temporary, TENSOR_DIR / f"{epic}.npz")
        for staged, destination in (
            (staged_feature_table, FEATURE_TABLE),
            (staged_scores, SCORES),
            (staged_embeddings, EMBEDDINGS),
        ):
            temporary = destination.with_name(f".{destination.name}.{stamp}.tmp")
            shutil.copy2(staged, temporary)
            atomic_replace(temporary, destination)

    model_hash_after = file_sha256(MODEL)
    protected_after = stable_hashes(PROTECTED_ARTIFACTS)
    trusted_tensor_sources_after = stable_hashes(TRUSTED_TENSOR_SOURCES)
    protected_comparison = {
        path: {"before": digest, "after": protected_after[path], "unchanged": digest == protected_after[path]}
        for path, digest in protected_before.items()
    }
    all_protected_unchanged = all(value["unchanged"] for value in protected_comparison.values())
    trusted_tensors_identical = trusted_tensor_sources_before == trusted_tensor_sources_after

    final_scores = pd.read_csv(SCORES, dtype={"epic_id": str})
    with np.load(EMBEDDINGS, allow_pickle=False) as final_embeddings:
        final_embedding_values = np.asarray(final_embeddings["embedding"], dtype=np.float32).copy()
        final_ids = final_embeddings["epic_id"].astype(str)
    final_valid_probabilities = int(np.isfinite(final_scores["cnn_probability"].to_numpy(float)).sum())
    final_valid_embeddings = int(sum(value.shape == (128,) and np.isfinite(value).all() for value in final_embedding_values))
    valid_provenance = int(
        trusted_frame["tensor_path"].notna().sum()
        + sum(bool(detail_by_epic[epic]["tensor_sha256_numeric_bytes"]) for epic in repair_ids)
    )

    status_frame = pd.DataFrame([detail_by_epic[epic] for epic in sorted(repair_ids)])
    status_frame["cnn_model_path"] = MODEL.relative_to(ROOT).as_posix()
    status_frame["cnn_model_sha256"] = model_hash_after
    status_frame["cnn_embedding_layer"] = EMBEDDING_LAYER
    status_frame["repair_status"] = "completed"
    status_frame["tensor_path"] = status_frame["epic_id"].map(lambda epic: f"data/phase2/local949_tensors/{epic}.npz")

    files_for_git_state = [
        PREPROCESSOR,
        ROOT / "scripts/generate_phase2_local949_features.py",
        ROOT / "tests/test_phase2_frozen_cnn_preprocessing_parity.py",
        ROOT / "scripts/audit_phase2_generated_tensor_duplicates.py",
    ]
    git_tracking = [git_state(path) for path in files_for_git_state]
    git_tracking.append(git_state(Path(__file__)))
    git_tracking.append(git_state(generalized_audit_path))

    merge_integrity = {
        "total_hosts": int(len(final_ids)),
        "valid_cnn_probabilities": final_valid_probabilities,
        "valid_128d_embeddings": final_valid_embeddings,
        "valid_tensor_provenance_records": valid_provenance,
        "trusted_reused_tensors_byte_identical": trusted_tensors_identical,
        "trusted_reused_score_records_byte_identical": trusted_score_identical,
        "trusted_reused_cnn_arrays_byte_identical": trusted_cnn_arrays_identical,
        "trusted_reused_feature_rows_identical": bool(trusted_feature_rows_identical),
        "protected_artifacts_unchanged": all_protected_unchanged,
    }
    complete = bool(
        integrity_gates_pass(generated_integrity)
        and model_hash_before == model_hash_after == EXPECTED_MODEL_SHA256
        and all(value for value in merge_integrity.values() if isinstance(value, bool))
        and merge_integrity["total_hosts"] == 949
        and merge_integrity["valid_cnn_probabilities"] == 949
        and merge_integrity["valid_128d_embeddings"] == 949
        and merge_integrity["valid_tensor_provenance_records"] == 949
    )
    summary = {
        "generated_at_utc": utc_now(),
        "run_started_at_utc": run_started,
        "scope": "generated-470 tensors, CNN probabilities, and 128-D embeddings only",
        "repair_population": {
            "repair_hosts": len(repair_ids),
            "trusted_reused_hosts": len(trusted_ids),
            "intersection": len(repair_ids.intersection(trusted_ids)),
            "union": len(repair_ids.union(trusted_ids)),
            "source": STATUS_SOURCE.relative_to(ROOT).as_posix(),
        },
        "parity_gate": {
            "passed": parity.returncode == 0,
            "command": f"{sys.executable} -m unittest tests.test_phase2_frozen_cnn_preprocessing_parity -v",
            "stdout": parity.stdout,
            "stderr": parity.stderr,
        },
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_code_sha256": preprocess_hash,
        "repair_script_sha256": repair_script_hash,
        "generalized_audit_script_sha256": generalized_audit_hash,
        "cnn_model_sha256_before": model_hash_before,
        "cnn_model_sha256_after": model_hash_after,
        "generated470_integrity": generated_integrity,
        "local949_merge_integrity": merge_integrity,
        "protected_artifact_hashes": protected_comparison,
        "trusted_tensor_source_hashes_before": trusted_tensor_sources_before,
        "trusted_tensor_source_hashes_after": trusted_tensor_sources_after,
        "versioned_pre_repair_artifacts": backup_map,
        "git_tracking_state": git_tracking,
        "non_actions": [
            "no light-curve acquisition",
            "no remaining-858 acquisition",
            "no period-search regeneration",
            "no diagnostic regeneration",
            "no label, eligibility, or target-mapping changes",
            "no CNN or CatBoost training",
            "no split assignment",
        ],
        "GENERATED470_CNN_REPAIR_COMPLETE": "yes" if complete else "no",
        "LOCAL949_CNN_FEATURES_VALID": "yes" if complete else "no",
        "READY_TO_ACQUIRE_REMAINING_858": "yes" if complete else "no",
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    status_temp = REPAIR_STATUS.with_name(REPAIR_STATUS.name + ".tmp")
    summary_temp = REPAIR_SUMMARY.with_name(REPAIR_SUMMARY.name + ".tmp")
    audit_temp = REPAIR_AUDIT.with_name(REPAIR_AUDIT.name + ".tmp")
    status_frame.to_csv(status_temp, index=False)
    summary_temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_temp.write_text(markdown_audit(summary), encoding="utf-8")
    atomic_replace(status_temp, REPAIR_STATUS)
    atomic_replace(summary_temp, REPAIR_SUMMARY)
    atomic_replace(audit_temp, REPAIR_AUDIT)

    if not complete:
        raise AssertionError("repair completed writes but final readiness gates did not pass")
    print(json.dumps({
        "repair_hosts": len(repair_ids),
        "unique_numeric_tensors": generated_integrity["unique_numeric_tensors"],
        "exact_duplicate_groups": generated_integrity["exact_duplicate_group_count"],
        "unexpected_all_zero_tensors": generated_integrity["unexpected_all_zero_tensors"],
        "unique_cnn_probabilities": generated_integrity["unique_cnn_probabilities"],
        "unique_128d_embeddings": generated_integrity["unique_128d_embeddings"],
        "GENERATED470_CNN_REPAIR_COMPLETE": summary["GENERATED470_CNN_REPAIR_COMPLETE"],
        "LOCAL949_CNN_FEATURES_VALID": summary["LOCAL949_CNN_FEATURES_VALID"],
        "READY_TO_ACQUIRE_REMAINING_858": summary["READY_TO_ACQUIRE_REMAINING_858"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
