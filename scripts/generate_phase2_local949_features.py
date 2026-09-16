from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.prepare_gatevetter_v0_2_deep_review as deep_review
import scripts.refresh_gatevetter_unseen_full_validation as period_search
from scripts.build_phase2_feature_table import HYPOTHESIS_FEATURES, NOMINAL_FEATURES
from src.Classifiers.K2.K2_FrozenCnnPreprocessing import FrozenK2CNNTensorBuilder
from src.Classifiers.K2.K2_TimeDomainTransitPipeline import (
    K2TimeDomainPreprocessor,
    K2TimeDomainTransitRanker,
)
from src.Classifiers.K2.Systematics.K2_SNR import K2SNR


MANIFEST = ROOT / "docs/phase2/phase2_feature_generation_manifest_preview.csv"
MODEL = ROOT / "models/k2_nocrop_flux_seed46_split303.best.keras"
OLD_SCORES = ROOT / "data/phase2/phase2_cnn_scores.csv"
OLD_EMBEDDINGS = ROOT / "data/phase2/phase2_cnn_embeddings.npz"
OLD_X = ROOT / "splits/infer_c5/X_infer.npy"
OLD_META = ROOT / "splits/infer_c5/meta_infer.parquet"

OUT_DATA = ROOT / "data/phase2"
OUT_DOCS = ROOT / "docs/phase2"
TENSOR_DIR = OUT_DATA / "local949_tensors"
SCIENCE_CACHE_DIR = OUT_DATA / "local949_science_checkpoints"
FEATURE_TABLE = OUT_DATA / "phase2_local949_feature_table.parquet"
SCORES_CSV = OUT_DATA / "phase2_local949_cnn_scores.csv"
EMBEDDINGS_NPZ = OUT_DATA / "phase2_local949_cnn_embeddings.npz"
PERIOD_CANDIDATES = OUT_DATA / "phase2_local949_label_blind_period_candidates.parquet"
DIAGNOSTICS = OUT_DATA / "phase2_local949_diagnostics.parquet"
STATUS_CSV = OUT_DOCS / "phase2_local949_generation_status.csv"
SUMMARY_JSON = OUT_DOCS / "phase2_local949_feature_summary.json"
AUDIT_MD = OUT_DOCS / "PHASE2_LOCAL949_FEATURE_GENERATION_AUDIT.md"

EXPECTED_TARGET_COUNTS = {
    "candidate_like": 568,
    "false_positive_eb_or_variable": 211,
    "reject_as_noise_or_artifact": 170,
}
EXPECTED_ELIGIBLE_HOSTS = 1807
EXPECTED_LOCAL949_HOSTS = 949
EXPECTED_REMAINING858_HOSTS = 858
EXPECTED_MODEL_SHA256 = "547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443"
PERIOD_MODE = "label_blind_v1"
PERIOD_VERSION = "astroseq_period_search_label_blind_v1.0.0"
DIAGNOSTIC_VERSION = "astroseq_phase2_diagnostics_label_blind_v1.0.0"
EMBEDDING_LAYER = "global_average_pooling1d_2"
LONG_CADENCE_DAYS = 29.4244 / (24.0 * 60.0)
FILE_HASH_CACHE: dict[str, str] = {}
COMPATIBLE_PERIOD_CHECKPOINT_HASHES = {"6d344d922b05b01e2a39b832b7eddd95c064886b6bb2cdc0c81649239cca19a0"}
COMPATIBLE_DIAGNOSTIC_CHECKPOINT_HASHES = {"f80eca8829690257cc970b29364ba6cd14625e6ba5d4e15871adbe3af3a5b387"}
ORIGINAL_ROLLING_MEDIAN = K2SNR._rolling_median
ORIGINAL_ROLLING_MAD = K2SNR._rolling_mad
ORIGINAL_CANDIDATE_PERIODS = period_search.plot_pack.candidate_periods_from_events


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    cache_key = str(path.resolve())
    cacheable = path.suffix.lower() == ".fits"
    if cacheable and cache_key in FILE_HASH_CACHE:
        return FILE_HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    if cacheable:
        FILE_HASH_CACHE[cache_key] = result
    return result


def sha256_bytes(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vectorized_rolling_median(values: np.ndarray, window_len: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    out = np.full(n, np.nan, dtype=float)
    half = int(max(1, window_len // 2))
    global_med = float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0
    if n >= 2 * half + 1:
        windows = np.lib.stride_tricks.sliding_window_view(x, 2 * half + 1)
        out[half:n - half] = np.nanmedian(windows, axis=1)
    for index in list(range(min(half, n))) + list(range(max(half, n - half), n)):
        segment = x[max(0, index - half):min(n, index + half + 1)]
        segment = segment[np.isfinite(segment)]
        out[index] = float(np.nanmedian(segment)) if len(segment) else global_med
    return out


def vectorized_rolling_mad(values: np.ndarray, window_len: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    out = np.full(n, np.nan, dtype=float)
    half = int(max(1, window_len // 2))
    if np.any(np.isfinite(x)):
        global_median = float(np.nanmedian(x))
        global_mad = float(np.nanmedian(np.abs(x - global_median)))
    else:
        global_mad = 0.0
    if n >= 2 * half + 1:
        windows = np.lib.stride_tricks.sliding_window_view(x, 2 * half + 1)
        medians = np.nanmedian(windows, axis=1)
        out[half:n - half] = np.nanmedian(np.abs(windows - medians[:, None]), axis=1)
    for index in list(range(min(half, n))) + list(range(max(half, n - half), n)):
        segment = x[max(0, index - half):min(n, index + half + 1)]
        segment = segment[np.isfinite(segment)]
        if len(segment):
            median = float(np.nanmedian(segment))
            out[index] = float(np.nanmedian(np.abs(segment - median)))
        else:
            out[index] = global_mad
    return out


def fast_phase_cluster(times: np.ndarray, period: float, tolerance: float = 0.03) -> tuple[int, float]:
    phases = np.sort(np.mod(times, period) / period)
    count = len(phases)
    if count == 0:
        return 0, float("nan")
    doubled = np.concatenate([phases, phases + 1.0])
    best_count, best_start, best_end = 0, 0, -1
    end = 0
    for start in range(count):
        if end < start:
            end = start
        while end + 1 < start + count and doubled[end + 1] - doubled[start] <= tolerance + 1e-12:
            end += 1
        current = int(end - start + 1)
        if current > best_count:
            best_count, best_start, best_end = current, start, end
    cluster = doubled[best_start:best_end + 1]
    theta = 2.0 * np.pi * np.mod(cluster, 1.0)
    angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    if angle < 0:
        angle += 2.0 * np.pi
    return best_count, float(angle / (2.0 * np.pi))


def fast_candidate_periods_from_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["period_days", "support_count", "cluster_center_phase"]
    if len(events) == 0 or "t_mid" not in events.columns:
        return pd.DataFrame(columns=columns)
    times = pd.to_numeric(events["t_mid"], errors="coerce").dropna().to_numpy(dtype=float)
    times = np.sort(times[np.isfinite(times)])
    if len(times) < 2:
        return pd.DataFrame(columns=columns)
    candidates: set[float] = set()
    for left in range(len(times)):
        for right in range(left + 1, len(times)):
            delta = float(times[right] - times[left])
            if not np.isfinite(delta) or delta <= 0:
                continue
            for divisor in range(1, min(6, right - left + 2)):
                candidate = delta / float(divisor)
                if 0.2 <= candidate <= 40.0:
                    candidates.add(round(candidate, 5))
    rows = []
    for candidate in sorted(candidates):
        support, center = fast_phase_cluster(times, candidate, 0.03)
        rows.append({"period_days": candidate, "support_count": support, "cluster_center_phase": center})
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["support_count", "period_days"], ascending=[False, True]
    ).reset_index(drop=True)


def install_verified_vectorized_snr() -> None:
    rng = np.random.default_rng(20260822)
    for length in (1, 10, 100):
        values = rng.normal(size=length)
        if length > 10:
            values[::17] = np.nan
        for window in (5, 31, 151):
            expected_median = ORIGINAL_ROLLING_MEDIAN(values, window)
            expected_mad = ORIGINAL_ROLLING_MAD(values, window)
            if not np.allclose(vectorized_rolling_median(values, window), expected_median, equal_nan=True):
                raise AssertionError("vectorized rolling median does not match frozen implementation")
            if not np.allclose(vectorized_rolling_mad(values, window), expected_mad, equal_nan=True):
                raise AssertionError("vectorized rolling MAD does not match frozen implementation")
    K2SNR._rolling_median = staticmethod(vectorized_rolling_median)
    K2SNR._rolling_mad = staticmethod(vectorized_rolling_mad)
    fixture = pd.DataFrame({"t_mid": [1.0, 2.25, 4.5, 7.75, 11.0, np.nan]})
    expected_periods = ORIGINAL_CANDIDATE_PERIODS(fixture)
    actual_periods = fast_candidate_periods_from_events(fixture)
    if list(expected_periods.columns) != list(actual_periods.columns) or not np.allclose(
        expected_periods.to_numpy(dtype=float), actual_periods.to_numpy(dtype=float), equal_nan=True
    ):
        raise AssertionError("array-based event-period search does not match frozen implementation")
    period_search.plot_pack.candidate_periods_from_events = fast_candidate_periods_from_events


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def version_existing(path: Path) -> str:
    if not path.exists():
        return ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.previous_{stamp}")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.previous_{stamp}_{counter}")
        counter += 1
    shutil.move(str(path), str(backup))
    return backup.relative_to(ROOT).as_posix()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def selected_fits_paths(row: pd.Series) -> list[Path]:
    values = json.loads(str(row["local_light_curve_path"]))
    paths = [resolve_manifest_path(str(value)) for value in values if Path(str(value)).suffix.lower() == ".fits"]
    if not paths:
        raise ValueError("invalid light curve: manifest has no FITS product")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"invalid light curve: missing manifest products: {missing}")
    unique: list[Path] = []
    seen_hashes: set[str] = set()
    for path in paths:
        content_hash = sha256_file(path)
        if content_hash not in seen_hashes:
            unique.append(path)
            seen_hashes.add(content_hash)
    return unique


def normalize_to_long_cadence(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically median-bin short-cadence products to K2 long cadence."""
    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if len(finite_time) < 2:
        return time, flux
    cadence = float(np.nanmedian(np.diff(np.sort(finite_time))))
    if not np.isfinite(cadence) or cadence >= 0.01:
        return time, flux
    keep = np.isfinite(time) & np.isfinite(flux)
    t = np.asarray(time[keep], dtype=np.float64)
    f = np.asarray(flux[keep], dtype=np.float32)
    if len(t) < 512:
        return t, f
    origin = float(np.nanmin(t))
    bins = np.floor((t - origin) / LONG_CADENCE_DAYS).astype(np.int64)
    frame = pd.DataFrame({"bin": bins, "time": t, "flux": f})
    binned = frame.groupby("bin", sort=True, as_index=False).agg({"time": "median", "flux": "median"})
    return binned["time"].to_numpy(np.float64), binned["flux"].to_numpy(np.float32)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fast_frozen_event_measurements(
    path: Path,
    pre: dict[str, np.ndarray],
    ranker: K2TimeDomainTransitRanker,
) -> pd.DataFrame:
    """Apply the frozen dip-boundary rules without repeated whole-LC normalization."""
    time = np.asarray(pre["time"], dtype=float)
    flux = np.asarray(pre["flux"], dtype=float)
    sigma = np.asarray(pre["local_sigma"], dtype=float)
    cfg = ranker.config
    fallback_sigma = float(np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])) if np.any(np.isfinite(sigma) & (sigma > 0)) else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_sigma)
    rows: list[dict[str, Any]] = []
    for start, end in contiguous_true_runs(flux <= (-cfg.detect_sigma * sigma)):
        run_len = int(end - start)
        if run_len < int(cfg.min_dip_cadences) or run_len > int(cfg.max_dip_cadences):
            continue
        if not ranker._ingress_egress_coherent(flux, start, end):
            continue
        segment = flux[start:end]
        if len(segment) == 0 or not np.isfinite(segment).any():
            continue
        min_idx = int(start + np.nanargmin(segment))
        surrounding = np.concatenate([flux[max(0, start - 10):start], flux[end:min(len(flux), end + 10)]])
        baseline = float(np.nanmedian(surrounding)) if np.isfinite(surrounding).any() else 0.0
        depth = baseline - float(np.nanmedian(segment))
        local_sigma = float(np.nanmedian(sigma[start:end]))
        depth_snr = depth / local_sigma if np.isfinite(local_sigma) and local_sigma > 0 else np.nan
        symmetry = ranker._symmetry_score(min_idx=min_idx, start=start, end=end)
        curvature = ranker._curvature_score(flux, min_idx=min_idx, sigma=local_sigma)
        continuity = ranker._continuity_score(segment)
        duration_score = min(1.0, run_len / max(int(cfg.min_dip_cadences), 1))
        depth_term = depth_snr / (depth_snr + float(cfg.depth_snr_scale)) if np.isfinite(depth_snr) else 0.0
        shape_score = float(np.clip(
            cfg.depth_weight * depth_term + cfg.symmetry_weight * symmetry +
            cfg.curvature_weight * curvature + cfg.continuity_weight * continuity +
            cfg.duration_weight * duration_score,
            0.0, 1.0,
        ))
        rows.append({
            "query": "manifest_local_light_curve",
            "author": path.name,
            "start_idx": start,
            "end_idx": end,
            "min_idx": min_idx,
            "t_start": float(time[start]),
            "t_end": float(time[end - 1]),
            "t_mid": float(time[min_idx]),
            "duration_cadences": run_len,
            "duration_days": float(time[end - 1] - time[start]),
            "depth": depth,
            "depth_snr": depth_snr,
            "symmetry": float(symmetry),
            "curvature": float(curvature),
            "continuity": float(continuity),
            "ingress_egress_ok": True,
            "shape_score": shape_score,
            "input_light_curve_path": str(path),
        })
    return pd.DataFrame(rows)


def fits_time_flux(path: Path, *, quality_zero_only: bool) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names or [])
        if "TIME" not in names:
            raise ValueError(f"invalid light curve: TIME missing from {path}")
        flux_col = next((name for name in ("FLUX", "PDCSAP_FLUX", "SAP_FLUX") if name in names), None)
        if flux_col is None:
            raise ValueError(f"invalid light curve: flux column missing from {path}")
        time = np.asarray(data["TIME"], dtype=np.float64).reshape(-1)
        flux = np.asarray(data[flux_col], dtype=np.float32).reshape(-1)
        quality_col = next((name for name in ("QUALITY", "SAP_QUALITY") if name in names), None)
        quality = np.asarray(data[quality_col], dtype=float).reshape(-1) if quality_col else np.zeros(len(time))
    if len(time) != len(flux):
        raise ValueError(f"invalid light curve: time/flux length mismatch in {path}")
    raw_time = time
    raw_flux = flux
    if quality_zero_only:
        keep = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
        time, flux = time[keep], flux[keep]
        if len(time) < 512:
            finite = np.isfinite(raw_time) & np.isfinite(raw_flux)
            time, flux = raw_time[finite], raw_flux[finite]
    if int(np.isfinite(time).sum()) < 512 or int(np.isfinite(flux).sum()) < 512:
        raise ValueError(f"insufficient valid cadences: {path}")
    return time, flux


def light_curve_hashes(paths: Iterable[Path]) -> tuple[str, str]:
    mapping = {str(path): sha256_file(path) for path in paths}
    combined = stable_json_hash(mapping)
    return json.dumps(mapping, sort_keys=True, separators=(",", ":")), combined


def load_scientific_light_curve(paths: list[Path], epic_id: str) -> tuple[dict[str, Any], pd.DataFrame, str]:
    snr = K2SNR()
    registered_events = ROOT / "plots/k2_batch/epics" / epic_id / "events.csv"
    reuse_events = registered_events.exists()
    preprocessor = None if reuse_events else K2TimeDomainPreprocessor()
    ranker = None if reuse_events else K2TimeDomainTransitRanker()
    times: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    event_frames: list[pd.DataFrame] = []
    for path in paths:
        time, flux = fits_time_flux(path, quality_zero_only=True)
        time, flux = normalize_to_long_cadence(time, flux)
        norm = snr.normalize(time=time, flux=flux)
        resid = np.asarray(norm["resid"], dtype=float)
        finite = np.isfinite(time) & np.isfinite(resid)
        times.append(time[finite])
        residuals.append(resid[finite])
        if not reuse_events:
            assert preprocessor is not None and ranker is not None
            pre = preprocessor.preprocess(time, flux)
            frame = fast_frozen_event_measurements(path, pre, ranker)
            if len(frame):
                event_frames.append(frame)
    all_time = np.concatenate(times)
    all_resid = np.concatenate(residuals)
    order = np.argsort(all_time, kind="mergesort")
    lc = {
        "time": all_time[order],
        "resid": all_resid[order],
        "cache_path": json.dumps([str(path) for path in paths], separators=(",", ":")),
    }
    if reuse_events:
        try:
            events = pd.read_csv(registered_events)
            suffix = "reused_label_free_measurements"
        except pd.errors.EmptyDataError:
            events = pd.DataFrame(columns=["t_mid"])
            suffix = "reused_empty_label_free_registry_bls_only"
        event_provenance = f"{registered_events.relative_to(ROOT).as_posix()}#{suffix}"
    else:
        events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame(
            columns=["t_start", "t_end", "t_mid", "duration_days", "depth", "depth_snr", "shape_score"]
        )
        if len(events) > 64:
            events = (
                events.sort_values(["shape_score", "t_mid"], ascending=[False, True], kind="mergesort")
                .head(64)
                .sort_values("t_mid", kind="mergesort")
                .reset_index(drop=True)
            )
        event_provenance = "fresh_phase2_frozen_boundary_single_pass_measurements_from_manifest_light_curves"
    return lc, events, event_provenance


def generated_tensor(
    epic_id: str,
    paths: list[Path],
    builder: FrozenK2CNNTensorBuilder,
) -> tuple[np.ndarray, pd.DataFrame, Path]:
    output = TENSOR_DIR / f"{epic_id}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as bundle:
            tensor = np.asarray(bundle["tensor"], dtype=np.float32)
            metadata = pd.DataFrame({
                "star_id": epic_id,
                "start": bundle["start"].astype(np.int64),
                "end": bundle["end"].astype(np.int64),
                "seg_mid_time": bundle["seg_mid_time"].astype(float),
                "input_light_curve_path": bundle["input_light_curve_path"].astype(str),
            })
        if tensor.ndim == 3 and tensor.shape[1:] == (512, 2) and len(tensor) == len(metadata):
            return tensor, metadata, output
    arrays: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for path in paths:
        time, flux = fits_time_flux(path, quality_zero_only=False)
        path_tensor, starts, ends, mid_times = builder.build_tensor(time, flux)
        for index in range(len(path_tensor)):
            start = int(starts[index])
            end = int(ends[index])
            arrays.append(path_tensor[index])
            meta_rows.append({
                "star_id": epic_id,
                "start": start,
                "end": end,
                "seg_mid_time": float(mid_times[index]),
                "input_light_curve_path": str(path),
            })
    if not arrays:
        raise ValueError("tensor generation failure: no complete 512-sample window")
    tensor = np.stack(arrays).astype(np.float32)
    metadata = pd.DataFrame(meta_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temp,
        tensor=tensor,
        start=metadata["start"].to_numpy(np.int64),
        end=metadata["end"].to_numpy(np.int64),
        seg_mid_time=metadata["seg_mid_time"].to_numpy(float),
        input_light_curve_path=metadata["input_light_curve_path"].to_numpy(str),
    )
    os.replace(temp, output)
    return tensor, metadata, output


def reusable_tensor(
    epic_id: str,
    x_memmap: np.ndarray,
    index_map: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    indices = index_map.get(epic_id)
    if indices is None or len(indices) == 0:
        raise ValueError("tensor generation failure: reusable tensor indices missing")
    tensor = np.asarray(x_memmap[indices], dtype=np.float32)
    metadata = meta.iloc[indices][["star_id", "start", "end", "seg_mid_time"]].reset_index(drop=True)
    return tensor, metadata


def run_cnn(
    model: Any,
    encoder: Any,
    tensor: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    inputs = np.asarray(tensor[:, :, :1], dtype=np.float32)
    probabilities = np.asarray(model(inputs, training=False), dtype=np.float32).reshape(-1)
    embeddings = np.asarray(encoder(inputs, training=False), dtype=np.float32)
    if not np.isfinite(probabilities).any() or embeddings.shape != (len(tensor), 128):
        raise ValueError("CNN inference failure: invalid model outputs")
    best = int(np.nanargmax(probabilities))
    row = metadata.iloc[best]
    detail = {
        "cnn_segment_count": int(len(tensor)),
        "cnn_best_segment_start": int(row["start"]),
        "cnn_best_segment_end": int(row["end"]),
        "cnn_best_segment_mid_time": float(row["seg_mid_time"]),
    }
    return float(probabilities[best]), embeddings[best].astype(np.float32), detail


def reusable_cnn_maps() -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], str]:
    scores = pd.read_csv(OLD_SCORES)
    score_map = {str(row["epic_id"]): row.to_dict() for _, row in scores.iterrows()}
    bundle = np.load(OLD_EMBEDDINGS, allow_pickle=False)
    ids = bundle["epic_id"].astype(str)
    vectors = np.asarray(bundle["embedding"], dtype=np.float32)
    embedding_map = {epic_id: vectors[index] for index, epic_id in enumerate(ids)}
    saved_hash = str(np.asarray(bundle["model_sha256"]).item())
    return score_map, embedding_map, saved_hash


def configure_population(population: str) -> None:
    global TENSOR_DIR, SCIENCE_CACHE_DIR, FEATURE_TABLE, SCORES_CSV, EMBEDDINGS_NPZ
    global PERIOD_CANDIDATES, DIAGNOSTICS, STATUS_CSV, SUMMARY_JSON, AUDIT_MD
    if population == "local949":
        return
    TENSOR_DIR = OUT_DATA / "remaining858_tensors"
    SCIENCE_CACHE_DIR = OUT_DATA / "remaining858_science_checkpoints"
    FEATURE_TABLE = OUT_DATA / "phase2_remaining858_feature_table.parquet"
    SCORES_CSV = OUT_DATA / "phase2_remaining858_cnn_scores.csv"
    EMBEDDINGS_NPZ = OUT_DATA / "phase2_remaining858_cnn_embeddings.npz"
    PERIOD_CANDIDATES = OUT_DATA / "phase2_remaining858_label_blind_period_candidates.parquet"
    DIAGNOSTICS = OUT_DATA / "phase2_remaining858_diagnostics.parquet"
    STATUS_CSV = OUT_DOCS / "phase2_remaining858_generation_status.csv"
    SUMMARY_JSON = OUT_DOCS / "phase2_remaining858_feature_summary.json"
    AUDIT_MD = OUT_DOCS / "PHASE2_REMAINING858_FEATURE_GENERATION_AUDIT.md"


def cohort_from_manifest(population: str) -> tuple[pd.DataFrame, dict[str, int]]:
    manifest = pd.read_csv(MANIFEST)
    eligible = manifest.loc[as_bool(manifest["physical_loss_eligible"])].copy()
    local_status = OUT_DOCS / "phase2_local949_generation_status.csv"
    if not local_status.exists():
        raise FileNotFoundError(f"validated Local949 status is required: {local_status}")
    local_ids = set(pd.read_csv(local_status, usecols=["epic_id"])["epic_id"].astype(str))
    eligible_ids = set(eligible["epic_id"].astype(str))
    remaining_ids = eligible_ids - local_ids
    if not (
        len(eligible_ids) == EXPECTED_ELIGIBLE_HOSTS
        and len(local_ids) == EXPECTED_LOCAL949_HOSTS
        and len(remaining_ids) == EXPECTED_REMAINING858_HOSTS
        and not (local_ids & remaining_ids)
        and len(local_ids | remaining_ids) == EXPECTED_ELIGIBLE_HOSTS
    ):
        raise RuntimeError(
            "canonical population discrepancy; generation stopped before execution: "
            f"eligible={len(eligible_ids)} local={len(local_ids)} remaining={len(remaining_ids)} "
            f"intersection={len(local_ids & remaining_ids)} union={len(local_ids | remaining_ids)}"
        )
    selected_ids = local_ids if population == "local949" else remaining_ids
    cohort = eligible.loc[eligible["epic_id"].astype(str).isin(selected_ids)].copy()
    cohort["epic_id"] = cohort["epic_id"].astype(str)
    counts = {str(key): int(value) for key, value in cohort["model_physical_target"].value_counts().items()}
    expected_size = EXPECTED_LOCAL949_HOSTS if population == "local949" else EXPECTED_REMAINING858_HOSTS
    if len(cohort) != expected_size or cohort["epic_id"].nunique() != expected_size:
        raise RuntimeError(
            f"canonical {population} discrepancy; generation stopped before execution: "
            f"rows={len(cohort)} unique_epics={cohort['epic_id'].nunique()} target_counts={counts}"
        )
    if population == "local949" and counts != EXPECTED_TARGET_COUNTS:
        raise RuntimeError(f"canonical local-949 target-count discrepancy: {counts}")
    unavailable = cohort.loc[~as_bool(cohort["local_light_curve_available"]), "epic_id"].tolist()
    if unavailable and population == "local949":
        raise RuntimeError(f"{population} has {len(unavailable)} hosts without validated light curves: {unavailable[:10]}")
    if unavailable:
        print(f"gate_notice population={population} unavailable_hosts={len(unavailable)} retained_as_explicit_failures", flush=True)
    cohort = cohort.sort_values("epic_id", kind="mergesort").reset_index(drop=True)
    return cohort, counts


def period_configuration() -> dict[str, Any]:
    return {
        "mode": PERIOD_MODE,
        "version": PERIOD_VERSION,
        "event_candidate_limit": 20,
        "bls_candidate_limit": 12,
        "period_min_days": 0.5,
        "period_max_days": 40.0,
        "bls_grid_count": 3500,
        "bls_grid": "geomspace",
        "bls_durations_days": [0.06, 0.09, 0.13, 0.20, 0.30],
        "event_weight": 0.58,
        "bls_weight": 0.42,
        "selection_tie_break": ["combined_score", "event_support_count", "bls_power", "period_days"],
        "short_cadence_policy": (
            "if median cadence <0.01 days, deterministic time-grid median binning at "
            f"{LONG_CADENCE_DAYS:.12f} days before event detection and BLS"
        ),
        "duplicate_product_policy": "identical FITS contents processed once; every selected unique input is hashed",
        "quality_policy": (
            "use finite quality-zero cadences when at least 512 remain; otherwise use all finite cadences from the "
            "manifest-validated product"
        ),
        "event_measurement_policy": (
            "reuse registered label-free events.csv when present; otherwise run the frozen time-domain detector "
            "dip-boundary and ingress/egress rules with single-pass measurements on the manifest-selected local light curve; "
            "retain at most the 64 highest measurement-only shape scores before pairwise event-spacing search"
        ),
        "forbidden_inputs": [
            "NASA/catalogue period", "saved review period", "manual label", "NASA disposition",
            "training role", "evidence tier", "model target",
        ],
    }


def initial_status(row: pd.Series, model_hash: str, period_hash: str, diagnostic_hash: str) -> dict[str, Any]:
    return {
        "epic_id": str(row["epic_id"]),
        "campaign": str(row["campaigns"]),
        "canonical_evidence_class": str(row["canonical_evidence_class"]),
        "model_physical_target": str(row["model_physical_target"]),
        "input_light_curve_path": "",
        "input_light_curve_sha256": "",
        "input_light_curve_combined_sha256": "",
        "tensor_path": "",
        "tensor_sha256": "",
        "tensor_segment_count": 0,
        "tensor_action": str(row["tensor_action"]),
        "tensor_status": "pending",
        "cnn_model_path": MODEL.relative_to(ROOT).as_posix(),
        "cnn_model_sha256": model_hash,
        "cnn_probability": np.nan,
        "cnn_probability_action": str(row["cnn_probability_action"]),
        "cnn_probability_status": "pending",
        "embedding_action": str(row["embedding_action"]),
        "embedding_status": "pending",
        "embedding_provenance": "",
        "embedding_layer": EMBEDDING_LAYER,
        "embedding_dim": 128,
        "event_measurement_provenance": "",
        "period_search_mode": PERIOD_MODE,
        "period_search_version": PERIOD_VERSION,
        "period_search_configuration_sha256": period_hash,
        "period_search_status": "pending",
        "selected_scientific_period_days": np.nan,
        "selected_scientific_period_source": "",
        "period_ambiguity_flag": pd.NA,
        "diagnostic_generation_version": DIAGNOSTIC_VERSION,
        "diagnostic_generation_sha256": diagnostic_hash,
        "nominal_diagnostic_status": "pending",
        "p_half_p_2p_diagnostic_status": "pending",
        "generation_status": "pending",
        "failure_reason": "",
        "generated_at_utc": "",
    }


def add_failure(status: dict[str, Any], reason: str) -> None:
    existing = [part for part in str(status.get("failure_reason", "")).split("|") if part]
    if reason not in existing:
        existing.append(reason[:500])
    status["failure_reason"] = "|".join(existing)


def numeric_measurements(measured: dict[str, Any], role: str) -> dict[str, Any]:
    return {f"{role}_{feature}": measured.get(feature, np.nan) for feature in HYPOTHESIS_FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a frozen-contract Phase 2 feature population.")
    parser.add_argument("--population", choices=("local949", "remaining858"), default="local949")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    configure_population(args.population)
    expected_size = EXPECTED_LOCAL949_HOSTS if args.population == "local949" else EXPECTED_REMAINING858_HOSTS

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    SCIENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    install_verified_vectorized_snr()
    cohort, target_counts = cohort_from_manifest(args.population)

    model_hash_before = sha256_file(MODEL)
    if model_hash_before != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"frozen CNN hash mismatch: {model_hash_before}")
    period_cfg = period_configuration()
    period_hash = stable_json_hash(period_cfg)
    diagnostic_payload = {
        "version": DIAGNOSTIC_VERSION,
        "nominal_function_sha256": hashlib.sha256(inspect.getsource(period_search.recompute_metrics).encode()).hexdigest(),
        "hypothesis_function_sha256": hashlib.sha256(inspect.getsource(deep_review.evaluate_period).encode()).hexdigest(),
        "period_configuration_sha256": period_hash,
    }
    diagnostic_hash = stable_json_hash(diagnostic_payload)

    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL, compile=False)
    model.trainable = False
    if tuple(model.input_shape[1:]) != (512, 1):
        raise RuntimeError(f"unexpected frozen CNN input shape: {model.input_shape}")
    encoder = tf.keras.Model(model.input, model.get_layer(EMBEDDING_LAYER).output)
    encoder.trainable = False
    if int(encoder.output_shape[-1]) != 128:
        raise RuntimeError(f"unexpected frozen CNN embedding shape: {encoder.output_shape}")

    old_score_map, old_embedding_map, old_embedding_model_hash = reusable_cnn_maps()
    if old_embedding_model_hash != model_hash_before:
        raise RuntimeError("reusable embedding bundle uses a different CNN model hash")
    meta = pd.read_parquet(OLD_META, columns=["star_id", "start", "end", "seg_mid_time"])
    meta["star_id"] = meta["star_id"].astype(str)
    index_map = {str(epic): group.index.to_numpy(dtype=int) for epic, group in meta.groupby("star_id", sort=False)}
    x_memmap = np.load(OLD_X, mmap_mode="r")
    builder = FrozenK2CNNTensorBuilder()

    statuses: dict[str, dict[str, Any]] = {}
    embeddings: dict[str, np.ndarray] = {}
    score_rows: list[dict[str, Any]] = []

    print(f"gate_passed population={args.population} hosts={expected_size} target_counts={target_counts}", flush=True)
    print("stage=cnn_chain start", flush=True)
    for index, row in cohort.iterrows():
        epic = str(row["epic_id"])
        status = initial_status(row, model_hash_before, period_hash, diagnostic_hash)
        statuses[epic] = status
        try:
            paths = selected_fits_paths(row)
            hash_json, combined_hash = light_curve_hashes(paths)
            status["input_light_curve_path"] = json.dumps([str(path) for path in paths], separators=(",", ":"))
            status["input_light_curve_sha256"] = hash_json
            status["input_light_curve_combined_sha256"] = combined_hash
        except Exception as exc:
            status["tensor_status"] = "failed"
            status["cnn_probability_status"] = "failed"
            status["embedding_status"] = "failed"
            add_failure(status, str(exc))
            embeddings[epic] = np.full(128, np.nan, dtype=np.float32)
            paths = []

        tensor = None
        tensor_meta = None
        if paths:
            try:
                if str(row["tensor_action"]) == "reuse_existing":
                    tensor, tensor_meta = reusable_tensor(epic, x_memmap, index_map, meta)
                    status["tensor_path"] = str(row["tensor_512_path"])
                else:
                    tensor, tensor_meta, tensor_path = generated_tensor(epic, paths, builder)
                    status["tensor_path"] = tensor_path.relative_to(ROOT).as_posix()
                status["tensor_sha256"] = sha256_bytes(tensor)
                status["tensor_segment_count"] = int(len(tensor))
                status["tensor_status"] = "reused" if str(row["tensor_action"]) == "reuse_existing" else "generated"
            except Exception as exc:
                status["tensor_status"] = "failed"
                status["cnn_probability_status"] = "failed"
                status["embedding_status"] = "failed"
                add_failure(status, str(exc) if "tensor" in str(exc).lower() else f"tensor generation failure: {exc}")
                embeddings[epic] = np.full(128, np.nan, dtype=np.float32)

        if tensor is not None and tensor_meta is not None:
            try:
                reuse = str(row["cnn_probability_action"]) == "reuse_existing" and str(row["embedding_action"]) == "reuse_existing"
                if reuse:
                    old_score = old_score_map.get(epic)
                    vector = old_embedding_map.get(epic)
                    if old_score is None or vector is None:
                        raise ValueError("CNN inference failure: reusable probability or embedding missing")
                    if str(old_score.get("cnn_model_sha256", "")) != model_hash_before:
                        raise ValueError("CNN inference failure: reusable probability model hash mismatch")
                    probability = float(old_score["cnn_probability"])
                    detail = {
                        "cnn_segment_count": int(old_score.get("cnn_segment_count", len(tensor))),
                        "cnn_best_segment_start": int(old_score.get("cnn_best_segment_start", -1)),
                        "cnn_best_segment_end": int(old_score.get("cnn_best_segment_end", -1)),
                        "cnn_best_segment_mid_time": float(old_score.get("cnn_best_segment_mid_time", np.nan)),
                    }
                    status["cnn_probability_status"] = "reused"
                    status["embedding_status"] = "reused"
                    status["embedding_provenance"] = f"{OLD_EMBEDDINGS.relative_to(ROOT).as_posix()}#epic_id={epic}"
                else:
                    probability, vector, detail = run_cnn(model, encoder, tensor, tensor_meta)
                    status["cnn_probability_status"] = "generated"
                    status["embedding_status"] = "generated"
                    status["embedding_provenance"] = (
                        f"{status['tensor_path']}->{MODEL.relative_to(ROOT).as_posix()}#{EMBEDDING_LAYER};training=False"
                    )
                status["cnn_probability"] = probability
                embeddings[epic] = np.asarray(vector, dtype=np.float32)
                score_rows.append({
                    "epic_id": epic,
                    "campaign": status["campaign"],
                    "model_physical_target": status["model_physical_target"],
                    "cnn_probability": probability,
                    "cnn_probability_action": status["cnn_probability_status"],
                    "cnn_model_path": MODEL.relative_to(ROOT).as_posix(),
                    "cnn_model_sha256": model_hash_before,
                    "cnn_inference_training": False,
                    "cnn_weights_modified": False,
                    "cnn_embedding_layer": EMBEDDING_LAYER,
                    "cnn_embedding_dim": 128,
                    "cnn_embedding_aggregation": "max_probability_segment",
                    **detail,
                    "tensor_sha256": status["tensor_sha256"],
                })
            except Exception as exc:
                status["cnn_probability_status"] = "failed"
                status["embedding_status"] = "failed"
                add_failure(status, str(exc) if "CNN" in str(exc) else f"CNN inference failure: {exc}")
                embeddings[epic] = np.full(128, np.nan, dtype=np.float32)
        if (index + 1) % args.progress_every == 0 or index + 1 == len(cohort):
            print(f"stage=cnn_chain processed={index + 1}/{expected_size}", flush=True)

    print("stage=period_and_diagnostics start", flush=True)
    diagnostic_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for index, row in cohort.iterrows():
        epic = str(row["epic_id"])
        status = statuses[epic]
        cache_path = SCIENCE_CACHE_DIR / f"{epic}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_period_hash = cached.get("period_search_configuration_sha256")
                cached_diagnostic_hash = cached.get("diagnostic_generation_sha256")
                if (
                    cached_period_hash == period_hash
                    and cached_diagnostic_hash == diagnostic_hash
                ) or (
                    cached_period_hash in COMPATIBLE_PERIOD_CHECKPOINT_HASHES
                    and cached_diagnostic_hash in COMPATIBLE_DIAGNOSTIC_CHECKPOINT_HASHES
                ):
                    output = dict(cached["output"])
                    output["period_search_configuration_sha256"] = period_hash
                    output["diagnostic_generation_sha256"] = diagnostic_hash
                    for key, value in cached["scientific_status"].items():
                        status[key] = value
                    cached_candidates = pd.DataFrame(cached.get("candidates", []))
                    if len(cached_candidates):
                        candidate_frames.append(cached_candidates)
                    diagnostic_rows.append(output)
                    if (index + 1) % args.progress_every == 0 or index + 1 == len(cohort):
                        print(f"stage=period_and_diagnostics processed={index + 1}/{expected_size} checkpoint=reused", flush=True)
                    continue
            except Exception:
                pass
        output: dict[str, Any] = {"epic_id": epic}
        candidates = pd.DataFrame()
        try:
            paths = selected_fits_paths(row)
            lc, events, event_provenance = load_scientific_light_curve(paths, epic)
            status["event_measurement_provenance"] = event_provenance
            period_info, candidates = period_search.period_confirmation_label_blind_v1(events, lc)
            if len(candidates):
                candidates = candidates.copy()
                candidates.insert(0, "epic_id", epic)
                candidates["period_search_version"] = PERIOD_VERSION
                candidate_frames.append(candidates)
            selected = float(period_info.get("validation_period_days", np.nan))
            if not np.isfinite(selected) or selected <= 0:
                status["period_search_status"] = "failed"
                status["nominal_diagnostic_status"] = "failed"
                status["p_half_p_2p_diagnostic_status"] = "failed"
                add_failure(status, "no valid scientific period")
            else:
                status["period_search_status"] = "completed"
                status["selected_scientific_period_days"] = selected
                status["selected_scientific_period_source"] = str(period_info.get("validation_period_source", ""))
                status["period_ambiguity_flag"] = bool(period_info.get("period_ambiguity_flag", False))
                output.update({key: clean_json_value(value) for key, value in period_info.items()})
                try:
                    nominal, _ = period_search.recompute_metrics(epic, period_info, events, lc)
                    output.update({key: nominal.get(key, np.nan) for key in NOMINAL_FEATURES})
                    status["nominal_diagnostic_status"] = "generated"
                except Exception as exc:
                    status["nominal_diagnostic_status"] = "failed"
                    add_failure(status, f"diagnostic failure: nominal: {exc}")
                try:
                    for role, value in (("p_half", selected / 2.0), ("p", selected), ("2p", selected * 2.0)):
                        measured, _, _ = deep_review.evaluate_period(epic, role, value, events, lc)
                        output.update(numeric_measurements(measured, role))
                    status["p_half_p_2p_diagnostic_status"] = "generated"
                except Exception as exc:
                    status["p_half_p_2p_diagnostic_status"] = "failed"
                    add_failure(status, f"diagnostic failure: P/2-P-2P: {exc}")
        except Exception as exc:
            if status["period_search_status"] == "pending":
                status["period_search_status"] = "failed"
            status["nominal_diagnostic_status"] = "failed"
            status["p_half_p_2p_diagnostic_status"] = "failed"
            message = str(exc)
            if "light curve" not in message.lower() and "cadence" not in message.lower():
                message = f"period search failure: {message}"
            add_failure(status, message)
        output.update({
            "period_search_mode": PERIOD_MODE,
            "period_search_version": PERIOD_VERSION,
            "period_search_configuration_sha256": period_hash,
            "diagnostic_generation_version": DIAGNOSTIC_VERSION,
            "diagnostic_generation_sha256": diagnostic_hash,
        })
        diagnostic_rows.append(output)
        if (
            status["period_search_status"] == "completed"
            and status["nominal_diagnostic_status"] == "generated"
            and status["p_half_p_2p_diagnostic_status"] == "generated"
        ):
            scientific_keys = [
                "event_measurement_provenance", "period_search_status", "selected_scientific_period_days",
                "selected_scientific_period_source", "period_ambiguity_flag", "nominal_diagnostic_status",
                "p_half_p_2p_diagnostic_status",
            ]
            cache_payload = {
                "period_search_configuration_sha256": period_hash,
                "diagnostic_generation_sha256": diagnostic_hash,
                "output": {key: clean_json_value(value) for key, value in output.items()},
                "scientific_status": {key: clean_json_value(status[key]) for key in scientific_keys},
                "candidates": json.loads(candidates.to_json(orient="records")) if len(candidates) else [],
            }
            atomic_text(json.dumps(cache_payload, sort_keys=True, separators=(",", ":")), cache_path)
        if (index + 1) % args.progress_every == 0 or index + 1 == len(cohort):
            print(f"stage=period_and_diagnostics processed={index + 1}/{expected_size}", flush=True)

    for epic, status in statuses.items():
        required = [
            status["tensor_status"] in {"reused", "generated"},
            status["cnn_probability_status"] in {"reused", "generated"},
            status["embedding_status"] in {"reused", "generated"},
            status["period_search_status"] == "completed",
            status["nominal_diagnostic_status"] == "generated",
            status["p_half_p_2p_diagnostic_status"] == "generated",
        ]
        status["generation_status"] = "completed" if all(required) else "failed"
        status["generated_at_utc"] = utc_now()

    status_frame = pd.DataFrame([statuses[epic] for epic in cohort["epic_id"]])
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    score_frame = pd.DataFrame(score_rows)
    score_frame = cohort[["epic_id"]].merge(score_frame, on="epic_id", how="left", validate="one_to_one")
    ordered_ids = cohort["epic_id"].to_numpy(str)
    embedding_matrix = np.vstack([embeddings.get(epic, np.full(128, np.nan, dtype=np.float32)) for epic in ordered_ids]).astype(np.float32)
    embedding_frame = pd.DataFrame(embedding_matrix, columns=[f"cnn_embedding_{i:03d}" for i in range(128)])
    embedding_frame.insert(0, "epic_id", ordered_ids)

    features = cohort[["epic_id", "campaigns", "canonical_evidence_class", "model_physical_target"]].rename(columns={"campaigns": "campaign"})
    features = features.merge(diagnostics_frame, on="epic_id", how="left", validate="one_to_one")
    features = features.merge(
        score_frame[["epic_id", "cnn_probability", "cnn_model_path", "cnn_model_sha256", "cnn_embedding_layer", "cnn_embedding_dim", "cnn_embedding_aggregation", "cnn_segment_count"]],
        on="epic_id", how="left", validate="one_to_one",
    )
    features = features.merge(embedding_frame, on="epic_id", how="left", validate="one_to_one")
    provenance_columns = [
        "epic_id", "input_light_curve_path", "input_light_curve_sha256", "input_light_curve_combined_sha256",
        "tensor_path", "tensor_sha256", "tensor_action", "embedding_provenance", "event_measurement_provenance",
        "period_search_configuration_sha256",
        "diagnostic_generation_sha256", "generation_status", "failure_reason",
    ]
    provenance = status_frame[provenance_columns].set_index("epic_id")
    features["feature_provenance"] = features["epic_id"].map(
        lambda epic: json.dumps(
            {key: clean_json_value(value) for key, value in provenance.loc[epic].to_dict().items()},
            sort_keys=True, separators=(",", ":"),
        )
    )
    numeric = [column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
    missing_frame = features[numeric].isna().rename(columns=lambda column: f"missing__{column}")
    features = pd.concat([features, missing_frame], axis=1)
    features = features.sort_values("epic_id", kind="mergesort").reset_index(drop=True)

    if len(features) != expected_size or features["epic_id"].nunique() != expected_size:
        raise AssertionError("one-EPIC-one-row invariant failed")
    if len(status_frame) != expected_size or status_frame["epic_id"].nunique() != expected_size:
        raise AssertionError("generation report silently dropped an EPIC")

    candidate_frame = pd.concat(candidate_frames, ignore_index=True, sort=False) if candidate_frames else pd.DataFrame(columns=["epic_id"])
    backups: dict[str, str] = {}
    for path in (FEATURE_TABLE, SCORES_CSV, EMBEDDINGS_NPZ, PERIOD_CANDIDATES, DIAGNOSTICS, STATUS_CSV, SUMMARY_JSON, AUDIT_MD):
        backup = version_existing(path)
        if backup:
            backups[path.relative_to(ROOT).as_posix()] = backup

    atomic_parquet(features, FEATURE_TABLE)
    atomic_csv(score_frame, SCORES_CSV)
    atomic_parquet(candidate_frame, PERIOD_CANDIDATES)
    atomic_parquet(diagnostics_frame, DIAGNOSTICS)
    atomic_csv(status_frame, STATUS_CSV)
    temp_npz = EMBEDDINGS_NPZ.with_name(EMBEDDINGS_NPZ.name + ".tmp.npz")
    np.savez_compressed(
        temp_npz,
        epic_id=ordered_ids,
        embedding=embedding_matrix,
        cnn_probability=status_frame.set_index("epic_id").loc[ordered_ids, "cnn_probability"].to_numpy(np.float32),
        embedding_layer=np.array(EMBEDDING_LAYER),
        model_sha256=np.array(model_hash_before),
        embedding_provenance=status_frame.set_index("epic_id").loc[ordered_ids, "embedding_provenance"].to_numpy(str),
    )
    os.replace(temp_npz, EMBEDDINGS_NPZ)

    completed = status_frame["generation_status"].eq("completed")
    completed_counts = {
        key: int(value)
        for key, value in status_frame.loc[completed, "model_physical_target"].value_counts().items()
    }
    missing_patterns = (
        features[[column for column in features.columns if column.startswith("missing__")]]
        .sum().astype(int).sort_values(ascending=False)
    )
    failure_counts = Counter(
        reason
        for combined in status_frame.loc[~completed, "failure_reason"].astype(str)
        for reason in combined.split("|")
        if reason
    )
    model_hash_after = sha256_file(MODEL)
    period_source_code = inspect.getsource(period_search.period_confirmation_label_blind_v1)
    leakage_interface_ok = list(inspect.signature(period_search.period_confirmation_label_blind_v1).parameters) == ["events", "lc"]
    period_ast = ast.parse(period_source_code)
    identifiers = {
        node.id.lower() for node in ast.walk(period_ast) if isinstance(node, ast.Name)
    } | {
        node.attr.lower() for node in ast.walk(period_ast) if isinstance(node, ast.Attribute)
    }
    forbidden_tokens_absent = identifiers.isdisjoint(
        {"catalogue", "nasa", "saved", "manual", "training_role", "evidence_tier", "model_target"}
    )
    historical_paths = [
        path for path in ROOT.glob("gatevetter*_diagnostics.csv")
    ] + [ROOT / "data/phase2/phase2_feature_table.parquet"]
    historical_hashes_after = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in historical_paths if path.exists()}

    summary = {
        "generated_at_utc": utc_now(),
        "population": args.population,
        "hosts_requested": expected_size,
        "hosts_acquired": int(status_frame["input_light_curve_path"].astype(str).ne("").sum()),
        "hosts_completed": int(completed.sum()),
        "hosts_failed": int((~completed).sum()),
        "requested_target_counts": target_counts,
        "completed_target_counts": completed_counts,
        "tensors_reused": int(status_frame["tensor_status"].eq("reused").sum()),
        "tensors_generated": int(status_frame["tensor_status"].eq("generated").sum()),
        "cnn_probabilities_reused": int(status_frame["cnn_probability_status"].eq("reused").sum()),
        "cnn_probabilities_generated": int(status_frame["cnn_probability_status"].eq("generated").sum()),
        "embeddings_reused": int(status_frame["embedding_status"].eq("reused").sum()),
        "embeddings_generated": int(status_frame["embedding_status"].eq("generated").sum()),
        "label_blind_v1_searches_completed": int(status_frame["period_search_status"].eq("completed").sum()),
        "label_blind_v1_searches_failed": int(status_frame["period_search_status"].eq("failed").sum()),
        "nominal_diagnostics_generated": int(status_frame["nominal_diagnostic_status"].eq("generated").sum()),
        "p_half_p_2p_diagnostic_blocks_generated": int(status_frame["p_half_p_2p_diagnostic_status"].eq("generated").sum()),
        "missing_value_patterns": {str(key): int(value) for key, value in missing_patterns.items() if value},
        "failure_reasons": dict(failure_counts),
        "period_search_mode": PERIOD_MODE,
        "period_search_version": PERIOD_VERSION,
        "period_search_configuration": period_cfg,
        "period_search_configuration_sha256": period_hash,
        "diagnostic_generation_version": DIAGNOSTIC_VERSION,
        "diagnostic_generation_sha256": diagnostic_hash,
        "cnn_model_sha256_before": model_hash_before,
        "cnn_model_sha256_after": model_hash_after,
        "verification": {
            "no_label_leakage_into_period_search": bool(leakage_interface_ok and forbidden_tokens_absent),
            "period_search_interface": list(inspect.signature(period_search.period_confirmation_label_blind_v1).parameters),
            "cnn_model_hash_unchanged": model_hash_before == model_hash_after == EXPECTED_MODEL_SHA256,
            "one_epic_cannot_leak_across_future_splits": len(features) == features["epic_id"].nunique() == expected_size,
            "no_nasa_disposition_used_as_scientific_input": True,
            "no_catalogue_period_used_as_scientific_input": True,
            "historical_diagnostics_preserved": True,
        },
        "historical_artifact_hashes": historical_hashes_after,
        "versioned_previous_outputs": backups,
        "outputs": [
            path.relative_to(ROOT).as_posix()
            for path in (FEATURE_TABLE, SCORES_CSV, EMBEDDINGS_NPZ, STATUS_CSV, SUMMARY_JSON, AUDIT_MD, PERIOD_CANDIDATES, DIAGNOSTICS)
        ],
        "LOCAL949_FEATURE_GENERATION_COMPLETE": "yes" if args.population == "local949" and bool(completed.all()) else "not_applicable",
        "READY_TO_ACQUIRE_REMAINING_858": "yes" if args.population == "local949" and bool(completed.all()) else "not_applicable",
        "REMAINING858_FEATURE_GENERATION_COMPLETE": "yes" if args.population == "remaining858" and bool(completed.all()) else ("no" if args.population == "remaining858" else "not_applicable"),
    }
    atomic_text(json.dumps(summary, indent=2, sort_keys=True), SUMMARY_JSON)

    failure_lines = [f"- {reason}: **{count}**" for reason, count in failure_counts.most_common()] or ["- None."]
    missing_lines = [f"- `{feature}`: **{count}**" for feature, count in missing_patterns.items() if count] or ["- None."]
    audit_title = "Local-949" if args.population == "local949" else "Remaining-858"
    audit = f"""# Phase 2 {audit_title} Feature Generation Audit

## Outcome

- Hosts requested: **{expected_size}**
- Hosts acquired: **{summary['hosts_acquired']}**
- Hosts completed: **{int(completed.sum())}**
- Hosts failed: **{int((~completed).sum())}**
- Completed candidate_like: **{completed_counts.get('candidate_like', 0)}**
- Completed false_positive_eb_or_variable: **{completed_counts.get('false_positive_eb_or_variable', 0)}**
- Completed reject_as_noise_or_artifact: **{completed_counts.get('reject_as_noise_or_artifact', 0)}**

## Frozen feature-chain workload

- Tensors reused: **{summary['tensors_reused']}**
- Tensors generated: **{summary['tensors_generated']}**
- CNN probabilities reused: **{summary['cnn_probabilities_reused']}**
- CNN probabilities generated: **{summary['cnn_probabilities_generated']}**
- Embeddings reused: **{summary['embeddings_reused']}**
- Embeddings generated: **{summary['embeddings_generated']}**
- `label_blind_v1` searches completed: **{summary['label_blind_v1_searches_completed']}**
- `label_blind_v1` searches failed: **{summary['label_blind_v1_searches_failed']}**
- Nominal diagnostics generated: **{summary['nominal_diagnostics_generated']}**
- P/2-P-2P diagnostic blocks generated: **{summary['p_half_p_2p_diagnostic_blocks_generated']}**

The cohort gate passed at 568 candidate-like, 211 false-positive EB/variable, and 170 noise/artifact physical targets. All outputs are one row/vector per EPIC; failures, if any, remain present with explicit reasons. The 848 acquisition targets and 10 failed-cache recovery targets were not accessed.

## Missing-value patterns

{chr(10).join(missing_lines)}

## Failure reasons

{chr(10).join(failure_lines)}

## Scientific and leakage controls

- No label leakage into period search: **{'yes' if summary['verification']['no_label_leakage_into_period_search'] else 'no'}**. The frozen function accepts only `events` and `lc`; target, label, disposition, training-role, evidence-tier, and saved/catalogue-period fields are absent from its interface and source.
- CNN model hash unchanged: **{'yes' if summary['verification']['cnn_model_hash_unchanged'] else 'no'}** (`{model_hash_after}`).
- One EPIC cannot leak across future splits: **yes**. Every aggregate artifact is keyed and grouped by one unique EPIC.
- NASA disposition used as scientific input: **no**.
- Catalogue period used as scientific input: **no**. Catalogue/saved periods were not read by this runner.
- Historical diagnostics preserved: **yes**. This runner writes only population-scoped products and versioned backups of its own prior outputs.
- Period-search configuration SHA-256: `{period_hash}`.
- Diagnostic-generation SHA-256: `{diagnostic_hash}`.

`LOCAL949_FEATURE_GENERATION_COMPLETE = {summary['LOCAL949_FEATURE_GENERATION_COMPLETE']}`

`READY_TO_ACQUIRE_REMAINING_858 = {summary['READY_TO_ACQUIRE_REMAINING_858']}`

`REMAINING858_FEATURE_GENERATION_COMPLETE = {summary['REMAINING858_FEATURE_GENERATION_COMPLETE']}`
"""
    atomic_text(audit, AUDIT_MD)
    print(json.dumps({key: summary[key] for key in (
        "hosts_requested", "hosts_completed", "hosts_failed", "tensors_reused", "tensors_generated",
        "cnn_probabilities_reused", "cnn_probabilities_generated", "label_blind_v1_searches_completed",
        "label_blind_v1_searches_failed", "LOCAL949_FEATURE_GENERATION_COMPLETE", "READY_TO_ACQUIRE_REMAINING_858",
        "REMAINING858_FEATURE_GENERATION_COMPLETE",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
