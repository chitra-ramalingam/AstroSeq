from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase2"
DOCS = ROOT / "docs" / "phase2"

EXPANDED_PATH = DATA / "phase2_expanded_label_table.parquet"
FEATURE_PATH = DATA / "phase2_feature_table.parquet"
CNN_SCORES_PATH = DATA / "phase2_cnn_scores.csv"
CNN_EMBEDDINGS_PATH = DATA / "phase2_cnn_embeddings.npz"
PERIOD_DIAGNOSTICS_PATH = DATA / "phase2_positive_period_diagnostics.csv"
REGISTRY_PATH = DOCS / "phase2_period_diagnostic_registry.csv"
PERIOD_CONTRACT_PATH = DOCS / "phase2_period_source_contract.json"
TARGET_CONTRACT_PATH = DOCS / "phase2_target_contract.json"
MODEL_PATH = ROOT / "models" / "k2_nocrop_flux_seed46_split303.best.keras"
TENSOR_PATH = ROOT / "splits" / "infer_c5" / "X_infer.npy"
TENSOR_META_PATH = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"

MANIFEST_PATH = DATA / "phase2_feature_generation_manifest.parquet"
PREVIEW_PATH = DOCS / "phase2_feature_generation_manifest_preview.csv"
BATCH_PATH = DOCS / "phase2_feature_generation_batch_summary.csv"
SUMMARY_PATH = DOCS / "phase2_feature_generation_manifest_summary.json"
AUDIT_PATH = DOCS / "PHASE2_FEATURE_GENERATION_MANIFEST_AUDIT.md"

MODEL_REL = "models/k2_nocrop_flux_seed46_split303.best.keras"
EXPECTED_MODEL_SHA256 = "547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443"
EMBEDDING_LAYER = "global_average_pooling1d_2"
PERIOD_SEARCH_MODE = "label_blind_v1"
PERIOD_SEARCH_VERSION = "astroseq_period_search_label_blind_v1.0.0"

TARGET_MAPPING = {
    "confirmed_planet": "candidate_like",
    "candidate_like": "candidate_like",
    "false_positive_eb_or_variable": "false_positive_eb_or_variable",
    "reject_as_noise_or_artifact": "reject_as_noise_or_artifact",
    "uncertain_hold": "",
    "cross_class_conflict": "",
}


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def as_bool(value: object) -> bool:
    return clean(value).lower() in {"true", "1", "yes"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical_epic(value: object) -> str:
    match = re.search(r"(\d{8,10})", clean(value))
    return f"EPIC_{match.group(1)}" if match else ""


def json_list(value: object) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return [clean(item) for item in parsed if clean(item)] if isinstance(parsed, list) else [clean(parsed)]


def json_compact(values: Iterable[object]) -> str:
    return json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)


def campaign_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.lstrip("-").isdigit() else (999, value)


def campaign_from_fits(path: Path) -> str:
    # K2 product names use c05, c16, or release suffixes such as c102 (campaign 10).
    match = re.search(r"-c(\d{2})\d?(?:_|-)", path.name.lower())
    if not match:
        match = re.search(r"-c(\d{2})\d?_", path.parent.name.lower())
    return str(int(match.group(1))) if match else ""


def scan_fits_root(root: Path, canonical_ids: set[str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return found
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        epic = canonical_epic(entry.name)
        if epic not in canonical_ids:
            continue
        for path in sorted(Path(entry.path).glob("*.fits")):
            found[epic].append(path)
    return found


def scan_hlsp_fits_root(root: Path, canonical_ids: set[str]) -> dict[str, list[Path]]:
    """Index allowed downloaded HLSP light curves by EPIC encoded in the product filename."""
    found: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return found
    for path in sorted(root.rglob("*.fits")):
        epic = canonical_epic(path.name)
        if epic not in canonical_ids and epic.startswith("EPIC_"):
            epic = f"EPIC_{int(epic.removeprefix('EPIC_'))}"
        if epic in canonical_ids:
            found[epic].append(path)
    return found


def validate_fits(path: Path) -> tuple[bool, str]:
    try:
        with fits.open(path, memmap=True, mode="readonly") as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                return False, "missing_binary_table"
            data = hdul[1].data
            names = {str(name).upper() for name in (data.names or [])}
            if "TIME" not in names:
                return False, "missing_TIME"
            flux_name = next((name for name in ("FLUX", "PDCSAP_FLUX", "SAP_FLUX") if name in names), "")
            if not flux_name:
                return False, "missing_flux_column"
            time = np.asarray(data["TIME"], dtype=float)
            flux = np.asarray(data[flux_name], dtype=float)
            finite = np.isfinite(time) & np.isfinite(flux)
            if int(finite.sum()) < 512:
                return False, "fewer_than_512_finite_cadences"
    except Exception as exc:  # integrity reporting must retain the concrete failure
        return False, f"{type(exc).__name__}:{exc}"
    return True, "validated_k2_fits"


def inspect_npz_cache(canonical_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "lc_cache").glob("EPIC_*.npz")):
        epic = canonical_epic(path.stem)
        if epic not in canonical_ids:
            continue
        record: dict[str, Any] = {"path": relative_or_absolute(path), "mission": "", "status": ""}
        try:
            with np.load(path, allow_pickle=False) as bundle:
                mission = str(bundle["mission"]) if "mission" in bundle.files else ""
                empty = bool(bundle["empty"]) if "empty" in bundle.files else False
                time = bundle["time"] if "time" in bundle.files else np.array([])
                flux = bundle["flux"] if "flux" in bundle.files else np.array([])
                record["mission"] = mission
                if empty:
                    record["status"] = "cached_empty_download_failure"
                elif mission.upper() != "K2":
                    record["status"] = f"incompatible_mission_{mission or 'missing'}"
                elif len(time) != len(flux) or int(np.isfinite(time).sum()) < 512 or not np.isfinite(flux).any():
                    record["status"] = "invalid_k2_npz_shape_or_finite_coverage"
                else:
                    record["status"] = "validated_k2_npz"
        except Exception as exc:
            record["status"] = f"npz_read_failure:{type(exc).__name__}:{exc}"
        records[epic] = record
    return records


def light_curve_inventory(expanded: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    canonical_ids = set(expanded["epic_id"])
    roots = [
        (Path.home() / ".lightkurve" / "cache" / "mastDownload" / "K2", "lightkurve_mastDownload_K2"),
        (ROOT / "k2_cache" / "mastDownload" / "K2", "repository_k2_cache"),
    ]
    path_map: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for root, source in roots:
        for epic, paths in scan_fits_root(root, canonical_ids).items():
            path_map[epic].extend((path, source) for path in paths)
    hlsp_roots = [
        (Path.home() / ".lightkurve" / "cache" / "mastDownload" / "HLSP", "lightkurve_mastDownload_HLSP"),
        (ROOT / "k2_cache" / "mastDownload" / "HLSP", "repository_k2_cache_HLSP"),
    ]
    for root, source in hlsp_roots:
        for epic, paths in scan_hlsp_fits_root(root, canonical_ids).items():
            path_map[epic].extend((path, source) for path in paths)
    npz_records = inspect_npz_cache(canonical_ids)

    inventory: dict[str, dict[str, Any]] = {}
    fit_failures: list[dict[str, str]] = []
    for epic in sorted(canonical_ids):
        valid: list[tuple[Path, str]] = []
        invalid: list[dict[str, str]] = []
        for path, source in sorted(path_map.get(epic, []), key=lambda pair: str(pair[0])):
            ok, reason = validate_fits(path)
            if ok:
                valid.append((path, source))
            else:
                failure = {"path": relative_or_absolute(path), "source": source, "reason": reason}
                invalid.append(failure)
                fit_failures.append({"epic_id": epic, **failure})

        npz = npz_records.get(epic)
        npz_valid = bool(npz and npz["status"] == "validated_k2_npz")
        usable = bool(valid or npz_valid)
        sources = sorted({source for _, source in valid} | ({"repository_lc_cache_npz"} if npz_valid else set()))
        paths = [relative_or_absolute(path) for path, _ in valid]
        if npz_valid and npz:
            paths.append(str(npz["path"]))
        campaigns = sorted({campaign_from_fits(path) for path, _ in valid if campaign_from_fits(path)}, key=campaign_sort_key)

        invalid_k2_artifact = bool(invalid) or bool(
            npz and str(npz["status"]).startswith("invalid_k2_npz")
        )
        incompatible: list[dict[str, str]] = invalid
        if npz and not npz_valid:
            incompatible.append({"path": str(npz["path"]), "source": "repository_lc_cache_npz", "reason": str(npz["status"])})

        if usable:
            status = "local_light_curve_immediately_usable"
            status_reason = "at_least_one_structurally_valid_K2_light_curve"
        elif any(item["reason"] == "cached_empty_download_failure" for item in incompatible):
            status = "light_curve_unavailable_or_blocked"
            status_reason = "cached_empty_download_failure"
        elif invalid_k2_artifact:
            status = "local_light_curve_exists_requires_validation_or_preprocessing"
            status_reason = "local_K2_artifact_failed_structural_validation"
        elif incompatible:
            status = "light_curve_acquisition_required"
            status_reason = "only_wrong_mission_or_other_incompatible_local_artifact_exists"
        else:
            status = "light_curve_acquisition_required"
            status_reason = "no_local_K2_artifact"

        inventory[epic] = {
            "available": usable,
            "paths": sorted(set(paths)),
            "sources": sources,
            "campaigns": campaigns,
            "status": status,
            "status_reason": status_reason,
            "incompatible": incompatible,
        }

    details = {
        "fits_integrity_failures": fit_failures,
        "npz_status_counts": dict(sorted(Counter(record["status"] for record in npz_records.values()).items())),
        "canonical_prior_accessible_count": int(expanded["accessible_local_light_curve"].sum()),
        "validated_current_accessible_count": sum(record["available"] for record in inventory.values()),
    }
    return inventory, details


def validate_tensor_and_cnn(
    expanded: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[set[str], dict[str, int], pd.DataFrame, dict[str, Any]]:
    model_sha = sha256_file(MODEL_PATH)
    assert model_sha == EXPECTED_MODEL_SHA256, f"Frozen CNN hash changed: {model_sha}"

    tensor = np.load(TENSOR_PATH, mmap_mode="r")
    assert tensor.ndim == 3 and tensor.shape[1] == 512 and tensor.shape[2] >= 1
    assert str(tensor.dtype) == "float32"
    tensor_meta = pd.read_parquet(TENSOR_META_PATH)
    assert len(tensor_meta) == tensor.shape[0]
    tensor_meta["epic_id"] = tensor_meta["star_id"].map(canonical_epic)
    tensor_counts = tensor_meta.groupby("epic_id").size().astype(int)
    tensor_ids = set(tensor_counts.index) & set(expanded["epic_id"])

    scores = pd.read_csv(CNN_SCORES_PATH)
    scores["epic_id"] = scores["epic_id"].map(canonical_epic)
    assert scores["epic_id"].is_unique
    assert set(scores["epic_id"]) <= tensor_ids
    assert set(scores["cnn_model_path"].map(clean)) == {MODEL_REL}
    assert set(scores["cnn_model_sha256"].map(clean)) == {model_sha}
    assert set(scores["cnn_embedding_layer"].map(clean)) == {EMBEDDING_LAYER}
    assert set(pd.to_numeric(scores["cnn_embedding_dim"], errors="raise").astype(int)) == {128}
    assert not scores["cnn_inference_training"].map(as_bool).any()
    assert not scores["cnn_weights_modified"].map(as_bool).any()

    with np.load(CNN_EMBEDDINGS_PATH, allow_pickle=False) as bundle:
        npz_ids = [canonical_epic(value) for value in bundle["epic_id"]]
        assert len(npz_ids) == len(set(npz_ids)) == len(scores)
        assert set(npz_ids) == set(scores["epic_id"])
        assert bundle["embedding"].shape == (len(npz_ids), 128)
        assert str(bundle["embedding_layer"]) == EMBEDDING_LAYER
        assert str(bundle["model_sha256"]) == model_sha
        np.testing.assert_allclose(
            bundle["cnn_probability"],
            scores.set_index("epic_id").loc[npz_ids, "cnn_probability"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-7,
        )

    feature = features.copy()
    feature["epic_id"] = feature["epic_id"].map(canonical_epic)
    assert feature["epic_id"].is_unique
    embedding_columns = [f"cnn_embedding_{index:03d}" for index in range(128)]
    assert all(column in feature for column in embedding_columns)
    compatible = (
        feature["cnn_probability"].notna()
        & feature[embedding_columns].notna().all(axis=1)
        & feature["cnn_model_path"].eq(MODEL_REL)
        & feature["cnn_model_sha256"].eq(model_sha)
        & feature["cnn_embedding_layer"].eq(EMBEDDING_LAYER)
        & pd.to_numeric(feature["cnn_embedding_dim"], errors="coerce").eq(128)
    )
    compatible_ids = set(feature.loc[compatible, "epic_id"])
    assert compatible_ids == set(scores["epic_id"])

    scores = scores.set_index("epic_id")
    scores["tensor_segment_count"] = pd.Series(tensor_counts)
    provenance = {
        "model_path": MODEL_REL,
        "model_sha256": model_sha,
        "tensor_path": relative_or_absolute(TENSOR_PATH),
        "tensor_sha256": sha256_file(TENSOR_PATH),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_meta_path": relative_or_absolute(TENSOR_META_PATH),
        "tensor_meta_sha256": sha256_file(TENSOR_META_PATH),
        "tensor_preprocessing_contract": "src/Classifiers/K2/K2_Dataset_builder.py; 512-cadence flux channel 0; existing infer_c5 contract",
        "cnn_scores_sha256": sha256_file(CNN_SCORES_PATH),
        "cnn_embeddings_sha256": sha256_file(CNN_EMBEDDINGS_PATH),
        "embedding_layer": EMBEDDING_LAYER,
        "embedding_dim": 128,
        "embedding_aggregation": "max_probability_segment",
    }
    return tensor_ids, {key: int(value) for key, value in tensor_counts.items()}, scores, provenance


def validate_period_registry(features: pd.DataFrame, canonical_ids: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    registry = pd.read_csv(REGISTRY_PATH)
    registry["epic_id"] = registry["EPIC"].map(canonical_epic)
    assert len(registry) == registry["epic_id"].nunique() == 24
    assert set(registry["epic_id"]) <= canonical_ids
    assert registry["artifact_usable"].map(as_bool).all()
    assert registry["registered_without_recomputation"].map(as_bool).all()
    assert registry["artifact_form"].value_counts().to_dict() == {
        "legacy_long_form": 15,
        "wide_feature_table": 9,
    }

    feature = features.set_index("epic_id")
    hashes: dict[str, str] = {}
    for _, row in registry.iterrows():
        artifact_ref = clean(row["artifact_path"])
        base = artifact_ref.split("#", 1)[0]
        path = ROOT / base
        assert path.exists(), f"Registered period diagnostic missing: {path}"
        hashes[base] = sha256_file(path)
        if row["artifact_form"] == "legacy_long_form":
            data = pd.read_csv(path)
            role_column = "period_role"
            assert role_column in data and set(data[role_column].map(clean)) == {"P/2", "P", "2P"}
            assert len(data) == 3
        else:
            epic = row["epic_id"]
            assert epic in feature.index
            assert feature.loc[epic, ["p_half_primary_depth", "p_primary_depth", "2p_primary_depth"]].notna().all()
    return registry.set_index("epic_id"), dict(sorted(hashes.items()))


def training_role(row: pd.Series) -> str:
    existing = clean(row.get("internal_training_role"))
    if as_bool(row["cross_class_conflict"]):
        return "quarantined_cross_class_conflict"
    if as_bool(row["physical_loss_eligible"]):
        return existing or "first_model_physical_loss"
    if clean(row["corrected_physical_class"]) == "uncertain_hold":
        return "provenance_or_evaluation_only_uncertain_hold"
    return existing or "provenance_or_evaluation_only_excluded"


def exclusion_state(row: pd.Series) -> str:
    if as_bool(row["cross_class_conflict"]):
        return "quarantined"
    if not as_bool(row["physical_loss_eligible"]):
        return "excluded_from_loss"
    return ""


def build_manifest(
    expanded: pd.DataFrame,
    features: pd.DataFrame,
    light_curves: dict[str, dict[str, Any]],
    tensor_ids: set[str],
    tensor_counts: dict[str, int],
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    cnn_provenance: dict[str, Any],
) -> pd.DataFrame:
    feature = features.copy()
    feature["epic_id"] = feature["epic_id"].map(canonical_epic)
    feature = feature.set_index("epic_id")
    nominal_fields = ["validation_period_days", "primary_depth", "primary_depth_snr"]
    nominal_available = feature[nominal_fields].notna().any(axis=1)
    rows: list[dict[str, Any]] = []

    for _, canonical in expanded.sort_values("epic_id").iterrows():
        epic = clean(canonical["epic_id"])
        eligible = as_bool(canonical["physical_loss_eligible"])
        cross_conflict = as_bool(canonical["cross_class_conflict"])
        excluded_action = exclusion_state(canonical)
        lc = light_curves[epic]
        lc_available = bool(lc["available"])
        tensor_available = epic in tensor_ids
        cnn_available = epic in scores.index
        embedding_available = cnn_available
        historical_nominal = bool(nominal_available.get(epic, False))
        registry_available = epic in registry.index

        feature_row = feature.loc[epic] if epic in feature.index else None
        nominal_source = clean(feature_row.get("nominal_diagnostic_sources")) if feature_row is not None else ""
        nominal_ref = f"{relative_or_absolute(FEATURE_PATH)}#epic_id={epic}" if historical_nominal else ""
        nominal_source = " | ".join(value for value in [nominal_source, nominal_ref] if value)
        legacy_period_source = clean(feature_row.get("nominal_period_source")) if feature_row is not None else ""

        if registry_available:
            period_diag_source = clean(registry.loc[epic, "artifact_path"])
            period_diag_format = clean(registry.loc[epic, "artifact_form"])
        else:
            period_diag_source = ""
            period_diag_format = "not_available"

        if excluded_action:
            default_missing_action = excluded_action
        elif not lc_available:
            default_missing_action = "blocked_missing_input"
        else:
            default_missing_action = "generate"

        if excluded_action:
            light_curve_action = excluded_action
        elif lc_available:
            light_curve_action = "reuse_existing"
        elif lc["status"] == "light_curve_unavailable_or_blocked":
            light_curve_action = "blocked_missing_input"
        else:
            light_curve_action = "acquire_light_curve"

        tensor_action = "reuse_existing" if tensor_available else default_missing_action
        cnn_action = "reuse_existing" if cnn_available else default_missing_action
        embedding_action = "reuse_existing" if embedding_available else default_missing_action
        nominal_action = excluded_action or ("generate" if lc_available else "blocked_missing_input")
        period_diag_action = "reuse_legacy_registered" if registry_available else (excluded_action or ("generate" if lc_available else "blocked_missing_input"))
        period_search_action = excluded_action or ("run_label_blind_period_search" if lc_available else "blocked_missing_input")

        if cross_conflict:
            overall = "quarantined"
            blocking = "not_applicable_quarantined_from_physical_loss"
        elif not eligible:
            overall = "excluded_from_loss"
            blocking = "not_applicable_excluded_from_physical_loss"
        elif not lc_available:
            overall = "blocked_missing_input"
            if lc["status"] == "light_curve_unavailable_or_blocked":
                blocking = "cached_empty_or_failed_K2_light_curve_requires_acquisition_recovery"
            elif lc["status"] == "local_light_curve_exists_requires_validation_or_preprocessing":
                blocking = "existing_K2_light_curve_requires_validation_or_preprocessing_before_generation"
            elif lc["status_reason"] == "only_wrong_mission_or_other_incompatible_local_artifact_exists":
                blocking = "compatible_K2_light_curve_acquisition_required; local artifact is wrong-mission_or_invalid"
            else:
                blocking = "compatible_K2_light_curve_acquisition_required"
        else:
            overall = "generate"
            blocking = "not_blocked"

        score = scores.loc[epic] if cnn_available else None
        tensor_ref = (
            f"{cnn_provenance['tensor_path']}#star_id={epic};meta={cnn_provenance['tensor_meta_path']};input=flux_channel_0"
            if tensor_available
            else ""
        )
        score_ref = f"{relative_or_absolute(CNN_SCORES_PATH)}#epic_id={epic}" if cnn_available else ""
        embedding_ref = f"{relative_or_absolute(CNN_EMBEDDINGS_PATH)}#epic_id={epic}" if embedding_available else ""
        registry_hash = ""
        if registry_available:
            base = period_diag_source.split("#", 1)[0]
            registry_hash = sha256_file(ROOT / base)

        rows.append(
            {
                "epic_id": epic,
                "campaigns": clean(canonical["campaigns_json"]),
                "canonical_evidence_class": clean(canonical["corrected_physical_class"]),
                "model_physical_target": TARGET_MAPPING[clean(canonical["corrected_physical_class"])],
                "physical_loss_eligible": eligible,
                "cross_class_conflict": cross_conflict,
                "training_role": training_role(canonical),
                "host_group_policy": "all_campaigns_objects_aliases_and_views_grouped_by_EPIC",
                "campaign_selection_policy": "retain_all_available_K2_campaign_products_sorted_by_campaign_and_path; no_arbitrary_single_campaign_selection",
                "local_light_curve_available": lc_available,
                "local_light_curve_path": json_compact(lc["paths"]),
                "local_light_curve_source": "|".join(lc["sources"]),
                "local_light_curve_campaigns": json_compact(lc["campaigns"]),
                "local_light_curve_status": lc["status"],
                "local_light_curve_status_reason": lc["status_reason"],
                "local_light_curve_incompatible_artifacts": json.dumps(lc["incompatible"], sort_keys=True, separators=(",", ":")),
                "light_curve_action": light_curve_action,
                "tensor_512_available": tensor_available,
                "tensor_512_path": tensor_ref,
                "tensor_512_segment_count": tensor_counts.get(epic, 0),
                "tensor_512_sha256": cnn_provenance["tensor_sha256"] if tensor_available else "",
                "tensor_preprocessing_provenance": cnn_provenance["tensor_preprocessing_contract"] if tensor_available else "",
                "tensor_action": tensor_action,
                "cnn_probability_available": cnn_available,
                "cnn_probability": float(score["cnn_probability"]) if cnn_available else np.nan,
                "cnn_probability_source": score_ref,
                "cnn_model_path": MODEL_REL,
                "cnn_model_sha256": cnn_provenance["model_sha256"],
                "cnn_probability_action": cnn_action,
                "embedding_128_available": embedding_available,
                "embedding_128_path_or_source": embedding_ref,
                "embedding_layer": EMBEDDING_LAYER,
                "embedding_action": embedding_action,
                "nominal_diagnostics_available": historical_nominal,
                "nominal_diagnostics_reusable_under_label_blind_v1": False,
                "nominal_diagnostics_source": nominal_source,
                "nominal_diagnostics_action": nominal_action,
                "period_diagnostics_available": registry_available,
                "period_diagnostics_format": period_diag_format,
                "period_diagnostics_source": period_diag_source,
                "period_diagnostics_sha256": registry_hash,
                "period_diagnostics_action": period_diag_action,
                "scientific_period_available": False,
                "scientific_period_source": (
                    f"legacy_non_label_blind:{legacy_period_source}" if legacy_period_source else "no_label_blind_v1_artifact"
                ),
                "legacy_period_dependent_features_available": historical_nominal,
                "legacy_period_dependent_features_require_regeneration": historical_nominal,
                "period_search_mode": PERIOD_SEARCH_MODE,
                "period_search_version": PERIOD_SEARCH_VERSION,
                "period_search_action": period_search_action,
                "overall_feature_action": overall,
                "blocking_reason": blocking,
            }
        )

    manifest = pd.DataFrame(rows).sort_values("epic_id").reset_index(drop=True)
    return manifest


def metric_counts(frame: pd.DataFrame) -> dict[str, int]:
    eligible = frame["physical_loss_eligible"]
    local = frame["local_light_curve_available"]
    tensor = frame["tensor_512_available"]
    cnn = frame["cnn_probability_available"]
    embedding = frame["embedding_128_available"]
    nominal_reusable = frame["nominal_diagnostics_reusable_under_label_blind_v1"]
    period_diag = frame["period_diagnostics_available"]
    period = frame["scientific_period_available"]
    return {
        "hosts": int(len(frame)),
        "physical_loss_eligible_hosts": int(eligible.sum()),
        "local_light_curves_available": int(local.sum()),
        "light_curves_requiring_acquisition_or_recovery": int((~local).sum()),
        "light_curve_acquisition_required": int(frame["local_light_curve_status"].eq("light_curve_acquisition_required").sum()),
        "local_light_curve_exists_requires_validation_or_preprocessing": int(frame["local_light_curve_status"].eq("local_light_curve_exists_requires_validation_or_preprocessing").sum()),
        "light_curve_unavailable_or_blocked": int(frame["local_light_curve_status"].eq("light_curve_unavailable_or_blocked").sum()),
        "tensor_512_reusable": int(tensor.sum()),
        "tensor_512_requiring_generation": int((~tensor).sum()),
        "cnn_probabilities_reusable": int(cnn.sum()),
        "cnn_probabilities_requiring_generation": int((~cnn).sum()),
        "embedding_128_reusable": int(embedding.sum()),
        "embedding_128_requiring_generation": int((~embedding).sum()),
        "nominal_diagnostics_reusable": int(nominal_reusable.sum()),
        "nominal_diagnostics_requiring_generation": int((~nominal_reusable).sum()),
        "historical_nominal_diagnostics_available": int(frame["nominal_diagnostics_available"].sum()),
        "period_diagnostics_reusable": int(period_diag.sum()),
        "period_diagnostics_requiring_generation": int((~period_diag).sum()),
        "label_blind_v1_scientific_periods_reusable": int(period.sum()),
        "label_blind_v1_searches_required": int((~period).sum()),
        "legacy_period_dependent_features_requiring_regeneration": int(frame["legacy_period_dependent_features_require_regeneration"].sum()),
        "hosts_blocked_by_missing_prerequisite": int(frame["overall_feature_action"].eq("blocked_missing_input").sum()),
        "hosts_excluded_from_physical_loss": int((~eligible).sum()),
        "hosts_quarantined": int(frame["cross_class_conflict"].sum()),
        "hosts_completely_feature_ready_today": int(
            (
                local
                & tensor
                & cnn
                & embedding
                & nominal_reusable
                & period_diag
                & period
            ).sum()
        ),
    }


def build_batches(manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest["physical_loss_eligible"]
    local = manifest["local_light_curve_available"]
    full_cnn = manifest["tensor_512_available"] & manifest["cnn_probability_available"] & manifest["embedding_128_available"]
    masks = {
        "Batch A": eligible & local & full_cnn,
        "Batch B": eligible & local & ~full_cnn,
        "Batch C": eligible & ~local,
        "Batch D": manifest["period_diagnostics_available"],
        "Batch E": ~eligible,
    }
    descriptions = {
        "Batch A": "existing compatible K2 light curve + existing 512 tensor + reusable frozen-CNN probability/embedding; run label_blind_v1 and regenerate nominal diagnostics; generate only missing P/2-P-2P diagnostics",
        "Batch B": "existing compatible K2 light curve but CNN chain incomplete; generate missing tensor, frozen-CNN probability, embedding, label_blind_v1 period, and diagnostics",
        "Batch C": "compatible K2 light curve missing; acquisition or cached-failure recovery is prerequisite, then downstream generation",
        "Batch D": "registered P/2-P-2P diagnostic reuse overlay; overlaps primary batches and schedules no recomputation",
        "Batch E": "excluded or quarantined; retain provenance/evaluation artifacts and do not enter physical loss",
    }
    rows = []
    for batch, mask in masks.items():
        group = manifest.loc[mask]
        primary = batch != "Batch D"
        is_reuse_overlay = batch == "Batch D"
        rows.append(
            {
                "batch_id": batch,
                "batch_type": "primary_mutually_exclusive" if primary else "reuse_overlay_non_additive",
                "host_count": int(len(group)),
                "physical_loss_eligible_hosts": int(group["physical_loss_eligible"].sum()),
                "light_curve_acquisitions_or_recoveries": 0 if is_reuse_overlay else int((~group["local_light_curve_available"] & group["physical_loss_eligible"]).sum()),
                "tensor_512_generations": 0 if is_reuse_overlay else int((~group["tensor_512_available"] & group["physical_loss_eligible"]).sum()),
                "frozen_cnn_inferences": 0 if is_reuse_overlay else int((~group["cnn_probability_available"] & group["physical_loss_eligible"]).sum()),
                "embedding_128_generations": 0 if is_reuse_overlay else int((~group["embedding_128_available"] & group["physical_loss_eligible"]).sum()),
                "label_blind_v1_searches": 0 if is_reuse_overlay else int((~group["scientific_period_available"] & group["physical_loss_eligible"]).sum()),
                "nominal_diagnostic_generations": 0 if is_reuse_overlay else int((~group["nominal_diagnostics_reusable_under_label_blind_v1"] & group["physical_loss_eligible"]).sum()),
                "period_diagnostic_generations": 0 if is_reuse_overlay else int((~group["period_diagnostics_available"] & group["physical_loss_eligible"]).sum()),
                "period_diagnostic_reuses": int(group["period_diagnostics_available"].sum()),
                "excluded_or_quarantined_hosts": int((~group["physical_loss_eligible"]).sum()),
                "description": descriptions[batch],
            }
        )
    batches = pd.DataFrame(rows)
    primary_count = int(batches.loc[batches["batch_type"].eq("primary_mutually_exclusive"), "host_count"].sum())
    assert primary_count == len(manifest)
    return batches


def exploded_campaign_summary(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    exploded = manifest.copy()
    exploded["campaign"] = exploded["campaigns"].map(lambda value: json_list(value) or ["unknown"])
    exploded = exploded.explode("campaign")
    rows = []
    for campaign, group in exploded.groupby("campaign", sort=False):
        record = {"campaign": campaign, **metric_counts(group)}
        record["counts_are_non_additive_for_multi_campaign_hosts"] = True
        rows.append(record)
    return sorted(rows, key=lambda row: campaign_sort_key(str(row["campaign"])))


def integrity_checks(
    manifest: pd.DataFrame,
    expanded: pd.DataFrame,
    registry: pd.DataFrame,
    cnn_provenance: dict[str, Any],
) -> dict[str, bool]:
    checks = {
        "one_manifest_row_per_epic": len(manifest) == manifest["epic_id"].nunique(),
        "no_duplicated_epic_host": not manifest["epic_id"].duplicated().any(),
        "canonical_population_exactly_accounted_for": set(manifest["epic_id"]) == set(expanded["epic_id"]),
        "expanded_host_count_is_1872": len(manifest) == 1872,
        "eligible_host_count_is_1807": int(manifest["physical_loss_eligible"].sum()) == 1807,
        "all_24_registered_period_diagnostic_hosts_recognised": set(registry.index) == set(manifest.loc[manifest["period_diagnostics_available"], "epic_id"]),
        "all_cross_class_conflicts_quarantined": manifest.loc[manifest["cross_class_conflict"], "overall_feature_action"].eq("quarantined").all(),
        "all_uncertain_holds_excluded": (~manifest.loc[manifest["canonical_evidence_class"].eq("uncertain_hold"), "physical_loss_eligible"]).all(),
        "only_label_blind_v1_scheduled_for_new_scientific_periods": set(manifest["period_search_mode"]) == {PERIOD_SEARCH_MODE},
        "frozen_cnn_model_path": set(manifest["cnn_model_path"]) == {MODEL_REL},
        "frozen_cnn_model_hash": set(manifest["cnn_model_sha256"]) == {cnn_provenance["model_sha256"]} == {EXPECTED_MODEL_SHA256},
        "cnn_embedding_layer_frozen": set(manifest["embedding_layer"]) == {EMBEDDING_LAYER},
        "cnn_and_embedding_reuse_are_tensor_compatible": (
            (~manifest["cnn_probability_available"] | manifest["tensor_512_available"])
            & (~manifest["embedding_128_available"] | manifest["tensor_512_available"])
        ).all(),
        "all_existing_period_diagnostics_are_reused": manifest.loc[manifest["period_diagnostics_available"], "period_diagnostics_action"].eq("reuse_legacy_registered").all(),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Manifest integrity failure: {failed}")
    return checks


def build_summary(
    manifest: pd.DataFrame,
    batches: pd.DataFrame,
    expanded: pd.DataFrame,
    registry: pd.DataFrame,
    input_hashes: dict[str, str],
    registry_hashes: dict[str, str],
    light_curve_details: dict[str, Any],
    checks: dict[str, bool],
) -> dict[str, Any]:
    eligible = manifest.loc[manifest["physical_loss_eligible"]]
    by_target: dict[str, Any] = {}
    for target in ["candidate_like", "false_positive_eb_or_variable", "reject_as_noise_or_artifact"]:
        by_target[target] = {
            "all_hosts_with_target_mapping": metric_counts(manifest.loc[manifest["model_physical_target"].eq(target)]),
            "physical_loss_eligible_hosts": metric_counts(eligible.loc[eligible["model_physical_target"].eq(target)]),
        }
    by_target["excluded_no_model_target"] = {
        "all_hosts_with_target_mapping": metric_counts(manifest.loc[manifest["model_physical_target"].eq("")]),
        "physical_loss_eligible_hosts": metric_counts(eligible.loc[eligible["model_physical_target"].eq("")]),
    }

    status_categories = [
        "local_light_curve_immediately_usable",
        "local_light_curve_exists_requires_validation_or_preprocessing",
        "light_curve_acquisition_required",
        "light_curve_unavailable_or_blocked",
    ]
    observed_status = manifest["local_light_curve_status"].value_counts().to_dict()
    status_counts = {category: int(observed_status.get(category, 0)) for category in status_categories}
    blockers = eligible.loc[eligible["overall_feature_action"].eq("blocked_missing_input"), "blocking_reason"].value_counts().sort_index().to_dict()
    return {
        "schema_version": "phase2_feature_generation_manifest_v1.0.0",
        "dry_run_only": True,
        "mass_feature_generation_performed": False,
        "canonical_population_path": relative_or_absolute(EXPANDED_PATH),
        "canonical_population_discrepancy": len(expanded) - 1872,
        "period_search_contract": {
            "mode": PERIOD_SEARCH_MODE,
            "version": PERIOD_SEARCH_VERSION,
            "saved_catalogue_and_target_metadata_use": "validation_and_provenance_only; prohibited from candidate generation, seeding, scoring, tie-breaking, selection, and fallback",
            "existing_label_blind_v1_artifacts_found": 0,
        },
        "counts": {
            "all_expanded_hosts": metric_counts(manifest),
            "physical_loss_eligible_hosts": metric_counts(eligible),
            "by_model_physical_target": by_target,
            "by_campaign_non_additive": exploded_campaign_summary(manifest),
            "local_light_curve_status": status_counts,
            "eligible_blocking_prerequisites": {key: int(value) for key, value in blockers.items()},
        },
        "important_distinction_eligible_1807": {
            "can_enter_feature_generation_immediately_with_accessible_local_K2_light_curve": int(eligible["local_light_curve_available"].sum()),
            "require_light_curve_acquisition_or_cached_failure_recovery_first": int((~eligible["local_light_curve_available"]).sum()),
            "already_have_reusable_cnn_probability_and_embedding": int((eligible["cnn_probability_available"] & eligible["embedding_128_available"]).sum()),
            "require_entirely_new_cnn_and_embedding_computation": int((~eligible["cnn_probability_available"] & ~eligible["embedding_128_available"]).sum()),
            "already_have_reusable_nominal_diagnostics_under_label_blind_v1": int(eligible["nominal_diagnostics_reusable_under_label_blind_v1"].sum()),
            "already_have_reusable_registered_period_diagnostics": int(eligible["period_diagnostics_available"].sum()),
            "require_new_label_blind_v1_period_search": int((~eligible["scientific_period_available"]).sum()),
            "existing_period_dependent_features_created_without_label_blind_v1_and_eventually_requiring_regeneration": int(eligible["legacy_period_dependent_features_require_regeneration"].sum()),
            "completely_feature_ready_today": metric_counts(eligible)["hosts_completely_feature_ready_today"],
            "blocked_hosts": int(eligible["overall_feature_action"].eq("blocked_missing_input").sum()),
            "blocked_by_exact_prerequisite": {key: int(value) for key, value in blockers.items()},
        },
        "batches": batches.to_dict("records"),
        "light_curve_inventory_notes": {
            **light_curve_details,
            "availability_rule": "at least one structurally valid K2 FITS or validated mission=K2 NPZ; wrong-mission TESS NPZ and cached-empty markers are not K2 inputs",
            "future_multi_campaign_rule": "retain all canonical campaigns and all matching local K2 products; de-duplicate deterministically and record the chosen product/hash before generation",
        },
        "registered_period_diagnostics": {
            "host_count": int(len(registry)),
            "artifact_form_counts": {key: int(value) for key, value in registry["artifact_form"].value_counts().to_dict().items()},
            "artifact_sha256": registry_hashes,
        },
        "input_artifact_sha256": input_hashes,
        "integrity_checks": checks,
        "non_actions_verified": [
            "no_light_curve_download",
            "no_tensor_generation",
            "no_cnn_inference_or_retraining",
            "no_embedding_generation",
            "no_numerical_or_period_diagnostic_generation",
            "no_period_or_candidate_search",
            "no_label_target_eligibility_or_split_change",
            "no_catboost_training",
        ],
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_audit(summary: dict[str, Any], manifest: pd.DataFrame, batches: pd.DataFrame) -> None:
    all_counts = summary["counts"]["all_expanded_hosts"]
    eligible_counts = summary["counts"]["physical_loss_eligible_hosts"]
    distinction = summary["important_distinction_eligible_1807"]
    checks = summary["integrity_checks"]
    blockers = summary["counts"]["eligible_blocking_prerequisites"]
    target_rows = []
    for target, payload in summary["counts"]["by_model_physical_target"].items():
        all_target = payload["all_hosts_with_target_mapping"]
        eligible_target = payload["physical_loss_eligible_hosts"]
        target_rows.append([
            f"`{target}`", all_target["hosts"], eligible_target["hosts"], eligible_target["local_light_curves_available"],
            eligible_target["tensor_512_reusable"], eligible_target["cnn_probabilities_reusable"],
            eligible_target["period_diagnostics_reusable"], eligible_target["label_blind_v1_searches_required"],
        ])
    batch_rows = []
    for row in batches.to_dict("records"):
        batch_rows.append([
            row["batch_id"], row["batch_type"], row["host_count"], row["light_curve_acquisitions_or_recoveries"],
            row["tensor_512_generations"], row["frozen_cnn_inferences"], row["embedding_128_generations"],
            row["label_blind_v1_searches"], row["nominal_diagnostic_generations"],
            row["period_diagnostic_generations"], row["period_diagnostic_reuses"],
        ])
    check_lines = "\n".join(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items())
    blocker_lines = "\n".join(f"- `{reason}`: **{count}**" for reason, count in blockers.items()) or "- None"
    historical_all = all_counts["historical_nominal_diagnostics_available"]
    historical_eligible = eligible_counts["historical_nominal_diagnostics_available"]
    ready = "no" if distinction["blocked_hosts"] else "yes"

    text = f"""# Phase 2 Feature Generation Manifest Audit

This is a dry run over the current canonical `{relative_or_absolute(EXPANDED_PATH)}`. It created no light curve, tensor, CNN output, embedding, diagnostic, scientific period, split, label, target, or trained model.

## Outcome

The canonical table contains **{all_counts['hosts']}** unique EPIC hosts, with **{eligible_counts['hosts']}** in the first-model physical loss. The expected population counts match exactly. Current validated K2 inputs make **{distinction['can_enter_feature_generation_immediately_with_accessible_local_K2_light_curve']}** eligible hosts immediately executable; **{distinction['require_light_curve_acquisition_or_cached_failure_recovery_first']}** eligible hosts require a compatible K2 light curve or recovery from a cached download failure first.

The prior canonical coverage flag reported **{summary['light_curve_inventory_notes']['canonical_prior_accessible_count']}** local light curves. This current scan finds **{summary['light_curve_inventory_notes']['validated_current_accessible_count']}** usable K2 light curves because it also inventories the repository `k2_cache`. EPIC-named NPZ files whose stored mission is TESS, plus cached-empty records, are not treated as K2 inputs.

## Required counts

{markdown_table(
    ['Metric', 'All 1,872', 'Eligible 1,807'],
    [
        ['Local light curves available', all_counts['local_light_curves_available'], eligible_counts['local_light_curves_available']],
        ['Light curves requiring acquisition/recovery', all_counts['light_curves_requiring_acquisition_or_recovery'], eligible_counts['light_curves_requiring_acquisition_or_recovery']],
        ['512 tensors reusable', all_counts['tensor_512_reusable'], eligible_counts['tensor_512_reusable']],
        ['512 tensors requiring generation', all_counts['tensor_512_requiring_generation'], eligible_counts['tensor_512_requiring_generation']],
        ['CNN probabilities reusable', all_counts['cnn_probabilities_reusable'], eligible_counts['cnn_probabilities_reusable']],
        ['CNN probabilities requiring generation', all_counts['cnn_probabilities_requiring_generation'], eligible_counts['cnn_probabilities_requiring_generation']],
        ['128-D embeddings reusable', all_counts['embedding_128_reusable'], eligible_counts['embedding_128_reusable']],
        ['128-D embeddings requiring generation', all_counts['embedding_128_requiring_generation'], eligible_counts['embedding_128_requiring_generation']],
        ['Nominal diagnostics reusable under label_blind_v1', all_counts['nominal_diagnostics_reusable'], eligible_counts['nominal_diagnostics_reusable']],
        ['Nominal diagnostics requiring generation', all_counts['nominal_diagnostics_requiring_generation'], eligible_counts['nominal_diagnostics_requiring_generation']],
        ['P/2-P-2P diagnostics reusable', all_counts['period_diagnostics_reusable'], eligible_counts['period_diagnostics_reusable']],
        ['P/2-P-2P diagnostics requiring generation', all_counts['period_diagnostics_requiring_generation'], eligible_counts['period_diagnostics_requiring_generation']],
        ['label_blind_v1 periods reusable', all_counts['label_blind_v1_scientific_periods_reusable'], eligible_counts['label_blind_v1_scientific_periods_reusable']],
        ['label_blind_v1 searches required', all_counts['label_blind_v1_searches_required'], eligible_counts['label_blind_v1_searches_required']],
        ['Blocked by missing prerequisite', all_counts['hosts_blocked_by_missing_prerequisite'], eligible_counts['hosts_blocked_by_missing_prerequisite']],
        ['Excluded from physical loss', all_counts['hosts_excluded_from_physical_loss'], eligible_counts['hosts_excluded_from_physical_loss']],
        ['Quarantined', all_counts['hosts_quarantined'], eligible_counts['hosts_quarantined']],
    ],
)}

The repository contains historical nominal diagnostics for **{historical_all}** hosts (**{historical_eligible}** eligible), but none carries `label_blind_v1` provenance. They remain visible as historical artifacts and are not silently considered reusable scientific diagnostics; all eventually require regeneration after a compliant search. The 24 registered P/2-P-2P artifacts are a deliberate exception required by the corrected registry: all **15** legacy long-form and **9** wide-form hosts are reusable without recomputation.

## Eligible-host answers

1. Immediately executable from a compatible local K2 light curve: **{distinction['can_enter_feature_generation_immediately_with_accessible_local_K2_light_curve']}**.
2. Require K2 light-curve acquisition or cached-failure recovery first: **{distinction['require_light_curve_acquisition_or_cached_failure_recovery_first']}**.
3. Reusable CNN probability plus 128-D embedding: **{distinction['already_have_reusable_cnn_probability_and_embedding']}**.
4. Entirely new CNN probability plus embedding: **{distinction['require_entirely_new_cnn_and_embedding_computation']}**.
5. Reusable nominal diagnostics under `label_blind_v1`: **{distinction['already_have_reusable_nominal_diagnostics_under_label_blind_v1']}**.
6. Reusable registered P/2-P-2P diagnostics: **{distinction['already_have_reusable_registered_period_diagnostics']}**.
7. New `label_blind_v1` searches required: **{distinction['require_new_label_blind_v1_period_search']}**.
8. Existing non-label-blind period-dependent feature hosts requiring eventual regeneration: **{distinction['existing_period_dependent_features_created_without_label_blind_v1_and_eventually_requiring_regeneration']}**.
9. Completely feature-ready today: **{distinction['completely_feature_ready_today']}**.
10. Blocked by a missing prerequisite: **{distinction['blocked_hosts']}**.

Exact blockers:

{blocker_lines}

## Target breakdown

{markdown_table(
    ['Model target', 'All hosts', 'Eligible hosts', 'Eligible local K2 LC', 'Eligible tensor', 'Eligible CNN+embedding', 'Eligible P/2-P-2P', 'Eligible searches required'],
    target_rows,
)}

Campaign-level non-additive counts are retained in `{relative_or_absolute(SUMMARY_PATH)}`. Multi-campaign hosts appear once in every applicable campaign, and the manifest retains all campaign provenance rather than selecting one campaign.

## Dry-run batches

Primary batches A, B, C, and E are mutually exclusive and sum to {len(manifest)} hosts. Batch D is an explicitly non-additive 24-host reuse overlay.

{markdown_table(
    ['Batch', 'Type', 'Hosts', 'LC acquire/recover', 'Tensor', 'CNN', 'Embedding', 'Search', 'Nominal', 'P/2-P-2P generate', 'P/2-P-2P reuse'],
    batch_rows,
)}

No batch was executed.

## Period and CNN contracts

Every new scientific search is scheduled as `{PERIOD_SEARCH_MODE}` / `{PERIOD_SEARCH_VERSION}`. NASA/archive disposition, NASA/catalogue period, saved review period, manual label, evidence tier, training role, and model target remain validation/provenance only and are prohibited from candidate generation, seeding, scoring, bonuses, tie-breaking, selection, and fallback.

The frozen CNN remains `{MODEL_REL}` at SHA-256 `{EXPECTED_MODEL_SHA256}`. Reused probabilities and embeddings match that hash, use a compatible 512-sample `infer_c5` tensor, and use `{EMBEDDING_LAYER}` with 128 dimensions. No inference or weight change occurred.

## Integrity checks

{check_lines}

## Concrete readiness blockers

The complete 1,807-host physical-loss population is not ready for one uninterrupted mass feature-generation run because {distinction['blocked_hosts']} eligible hosts lack a usable K2 light curve. The immediately executable A/B cohorts may be launched separately after approval, but the complete population requires the acquisition/recovery prerequisites listed above.

READY_FOR_MASS_FEATURE_GENERATION = {ready}
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    expanded = pd.read_parquet(EXPANDED_PATH)
    features = pd.read_parquet(FEATURE_PATH)
    assert len(expanded) == expanded["epic_id"].nunique()
    assert set(expanded["corrected_physical_class"]) == set(TARGET_MAPPING)

    light_curves, light_curve_details = light_curve_inventory(expanded)
    tensor_ids, tensor_counts, scores, cnn_provenance = validate_tensor_and_cnn(expanded, features)
    registry, registry_hashes = validate_period_registry(features, set(expanded["epic_id"]))
    manifest = build_manifest(expanded, features, light_curves, tensor_ids, tensor_counts, scores, registry, cnn_provenance)
    batches = build_batches(manifest)
    checks = integrity_checks(manifest, expanded, registry, cnn_provenance)

    input_hashes = {
        relative_or_absolute(path): sha256_file(path)
        for path in [
            EXPANDED_PATH,
            FEATURE_PATH,
            CNN_SCORES_PATH,
            CNN_EMBEDDINGS_PATH,
            REGISTRY_PATH,
            PERIOD_CONTRACT_PATH,
            TARGET_CONTRACT_PATH,
            MODEL_PATH,
            TENSOR_PATH,
            TENSOR_META_PATH,
        ]
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    manifest.to_csv(PREVIEW_PATH, index=False)
    batches.to_csv(BATCH_PATH, index=False)

    summary = build_summary(
        manifest,
        batches,
        expanded,
        registry,
        input_hashes,
        registry_hashes,
        light_curve_details,
        checks,
    )
    summary["output_artifact_sha256"] = {
        relative_or_absolute(MANIFEST_PATH): sha256_file(MANIFEST_PATH),
        relative_or_absolute(PREVIEW_PATH): sha256_file(PREVIEW_PATH),
        relative_or_absolute(BATCH_PATH): sha256_file(BATCH_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit(summary, manifest, batches)
    print(json.dumps(summary["important_distinction_eligible_1807"], indent=2, sort_keys=True))
    print(f"manifest_rows={len(manifest)} eligible={int(manifest['physical_loss_eligible'].sum())}")
    print(f"period_registry_hosts={int(manifest['period_diagnostics_available'].sum())}")
    print(f"outputs={MANIFEST_PATH},{PREVIEW_PATH},{BATCH_PATH},{SUMMARY_PATH},{AUDIT_PATH}")


if __name__ == "__main__":
    main()
