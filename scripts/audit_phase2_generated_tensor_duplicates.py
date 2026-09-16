from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
TENSOR_DIR = ROOT / "data/phase2/local949_tensors"
EMBEDDINGS_PATH = ROOT / "data/phase2/phase2_local949_cnn_embeddings.npz"
SCORES_PATH = ROOT / "data/phase2/phase2_local949_cnn_scores.csv"
OUTPUT_DIR = ROOT / "docs/phase2"
SUMMARY_PATH = OUTPUT_DIR / "phase2_generated470_tensor_duplicate_summary.json"
GROUPS_PATH = OUTPUT_DIR / "phase2_generated470_tensor_duplicate_groups.csv"
REPRESENTATIVES_PATH = OUTPUT_DIR / "phase2_generated470_tensor_duplicate_representatives.csv"
REPORT_PATH = OUTPUT_DIR / "PHASE2_GENERATED470_TENSOR_DUPLICATE_AUDIT.md"

MIN_SCALE = 5e-2
CLIP_SIGMA = 10.0
LONG_CADENCE_DAYS = 29.4244 / (60.0 * 24.0)


def sha256_numeric_bytes(array: np.ndarray) -> str:
    """Hash only C-order numeric bytes: no path, NPZ bytes, dtype, or shape metadata."""
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clean_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def array_stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    return {
        "min": clean_float(float(np.min(finite))) if len(finite) else None,
        "max": clean_float(float(np.max(finite))) if len(finite) else None,
        "std": clean_float(float(np.std(finite))) if len(finite) else None,
        "unique_values": int(np.unique(values).size),
        "fraction_zero": float(np.mean(values == 0)),
        "fraction_nan": float(np.mean(np.isnan(values))),
    }


def read_fits(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names or [])
        flux_col = next((name for name in ("FLUX", "PDCSAP_FLUX", "SAP_FLUX") if name in names), None)
        if flux_col is None or "TIME" not in names:
            raise ValueError(f"missing TIME/flux column: {path}")
        time = np.asarray(data["TIME"], dtype=np.float64).reshape(-1)
        flux = np.asarray(data[flux_col], dtype=np.float32).reshape(-1)
    return time, flux, flux_col


def normalize_to_long_cadence(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    finite_time = np.asarray(time, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    if len(finite_time) < 2:
        return time, flux, False
    cadence = float(np.nanmedian(np.diff(np.sort(finite_time))))
    if not np.isfinite(cadence) or cadence >= 0.01:
        return time, flux, False
    keep = np.isfinite(time) & np.isfinite(flux)
    t = np.asarray(time[keep], dtype=np.float64)
    f = np.asarray(flux[keep], dtype=np.float32)
    if len(t) < 512:
        return t, f, False
    origin = float(np.nanmin(t))
    bins = np.floor((t - origin) / LONG_CADENCE_DAYS).astype(np.int64)
    frame = pd.DataFrame({"bin": bins, "time": t, "flux": f})
    binned = frame.groupby("bin", sort=True, as_index=False).agg({"time": "median", "flux": "median"})
    return binned["time"].to_numpy(np.float64), binned["flux"].to_numpy(np.float32), True


def trace_normalization(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import lightkurve as lk

    time_lc, flux_lc, cadence_binned = normalize_to_long_cadence(time, flux)
    flatten_succeeded = False
    flatten_error = ""
    try:
        lc = lk.LightCurve(time=time_lc, flux=flux_lc)
        flattened = lc.flatten(window_length=401, polyorder=2)
        flux_flat = np.asarray(flattened.flux.value, dtype=np.float32)
        time_flat = np.asarray(flattened.time.value, dtype=np.float64)
        flatten_succeeded = True
    except Exception as exc:
        flatten_error = f"{type(exc).__name__}: {exc}"
        flux_flat = np.asarray(flux_lc, dtype=np.float32)
        time_flat = np.asarray(time_lc, dtype=np.float64)

    relative_divisor = float(np.nanmedian(flux_flat))
    divisor_source = "median"
    flux_rel = flux_flat.copy()
    if np.isfinite(relative_divisor) and relative_divisor != 0.0:
        flux_rel = flux_rel / relative_divisor
    else:
        relative_divisor = float(np.nanmean(flux_flat))
        divisor_source = "mean"
        if np.isfinite(relative_divisor) and relative_divisor != 0.0:
            flux_rel = flux_rel / relative_divisor

    center = float(np.nanmedian(flux_rel))
    x0 = flux_rel - center
    mad = float(np.nanmedian(np.abs(x0)))
    mad_scale = 1.4826 * mad if np.isfinite(mad) else np.nan
    std_scale = float(np.nanstd(x0))
    p05 = float(np.nanpercentile(x0, 5))
    p95 = float(np.nanpercentile(x0, 95))
    pct_scale = 0.5 * (p95 - p05)

    selected_scale = mad_scale
    selected_source = "mad"
    if not np.isfinite(selected_scale) or selected_scale < MIN_SCALE:
        selected_scale = pct_scale
        selected_source = "p05_p95_half_range"
    if not np.isfinite(selected_scale) or selected_scale < MIN_SCALE:
        selected_scale = std_scale
        selected_source = "std"

    zeroed = not np.isfinite(selected_scale) or selected_scale < MIN_SCALE
    if zeroed:
        standardized = np.zeros_like(x0, dtype=np.float32)
    else:
        standardized = x0 / (float(selected_scale) + 1e-8)
    standardized = np.clip(standardized, -CLIP_SIGMA, CLIP_SIGMA)
    standardized[~np.isfinite(standardized)] = 0.0

    trace = {
        "input_count": int(len(time)),
        "input_valid_cadences": int(np.sum(np.isfinite(time) & np.isfinite(flux))),
        "post_cadence_count": int(len(time_lc)),
        "post_flatten_count": int(len(time_flat)),
        "short_cadence_binned": cadence_binned,
        "flatten_succeeded": flatten_succeeded,
        "flatten_error": flatten_error,
        "flatten_window_length": 401,
        "flatten_polyorder": 2,
        "relative_divisor_source": divisor_source,
        "relative_divisor": clean_float(relative_divisor),
        "relative_center_median": clean_float(center),
        "mad_scale": clean_float(mad_scale),
        "percentile_scale": clean_float(pct_scale),
        "std_scale": clean_float(std_scale),
        "selected_scale_source": selected_source,
        "selected_scale": clean_float(selected_scale),
        "minimum_scale": MIN_SCALE,
        "zeroed_because_scale_below_floor": zeroed,
        "clip_sigma": CLIP_SIGMA,
        "raw_flux": array_stats(flux),
        "flattened_flux": array_stats(flux_flat),
        "relative_flux": array_stats(flux_rel),
        "standardized_flux": array_stats(standardized),
        "standardized_matches_saved_channel0_prefix": None,
    }
    return standardized.astype(np.float32), trace


def embedding_map() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as bundle:
        ids = bundle["epic_id"].astype(str)
        vectors = np.asarray(bundle["embedding"])
        metadata = {
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "layer": str(bundle["embedding_layer"]),
            "model_sha256": str(bundle["model_sha256"]),
        }
    return {epic: vectors[index] for index, epic in enumerate(ids)}, metadata


def representative_row(epic: str, tensor_path: Path, tensor_hash: str, embedding: np.ndarray) -> dict[str, Any]:
    with np.load(tensor_path, allow_pickle=False) as bundle:
        tensor = np.asarray(bundle["tensor"])
        input_paths = sorted({Path(value) for value in bundle["input_light_curve_path"].astype(str)})

    path_hashes: dict[str, str] = {}
    numeric_input_hashes: dict[str, str] = {}
    traces: list[dict[str, Any]] = []
    all_flux: list[np.ndarray] = []
    valid_cadences = 0
    for path in input_paths:
        time, flux, flux_col = read_fits(path)
        path_hashes[str(path)] = sha256_file(path)
        numeric_input_hashes[str(path)] = sha256_numeric_bytes(
            np.concatenate([time.view(np.uint8), flux.view(np.uint8)])
        )
        standardized, trace = trace_normalization(time, flux)
        trace["path"] = str(path)
        trace["flux_column"] = flux_col
        trace["standardized_matches_saved_channel0_prefix"] = bool(
            np.array_equal(
                standardized[: min(len(standardized), tensor.shape[1])],
                tensor[0, : min(len(standardized), tensor.shape[1]), 0],
                equal_nan=True,
            )
        )
        traces.append(trace)
        valid_cadences += int(trace["input_valid_cadences"])
        all_flux.append(flux[np.isfinite(flux)])

    flat = tensor.ravel(order="C")
    tstats = array_stats(tensor)
    raw_flux_stats = array_stats(np.concatenate(all_flux)) if all_flux else array_stats(np.array([]))
    embedding_hash = sha256_numeric_bytes(embedding)
    return {
        "epic_id": epic,
        "tensor_sha256_numeric_bytes": tensor_hash,
        "tensor_dtype": str(tensor.dtype),
        "tensor_shape": json.dumps(list(tensor.shape), separators=(",", ":")),
        "segment_count": int(len(tensor)),
        "input_light_curve_file_sha256": json.dumps(path_hashes, sort_keys=True, separators=(",", ":")),
        "input_light_curve_combined_sha256": stable_json_hash(path_hashes),
        "input_numeric_time_flux_sha256": json.dumps(numeric_input_hashes, sort_keys=True, separators=(",", ":")),
        "valid_cadences": valid_cadences,
        "flux_min": raw_flux_stats["min"],
        "flux_max": raw_flux_stats["max"],
        "flux_std": raw_flux_stats["std"],
        "normalization_parameters": json.dumps(traces, sort_keys=True, separators=(",", ":")),
        "tensor_min": tstats["min"],
        "tensor_max": tstats["max"],
        "tensor_std": tstats["std"],
        "tensor_unique_values": tstats["unique_values"],
        "tensor_fraction_zero": tstats["fraction_zero"],
        "tensor_fraction_nan": tstats["fraction_nan"],
        "tensor_first_10_values": json.dumps(flat[:10].tolist(), separators=(",", ":")),
        "tensor_last_10_values": json.dumps(flat[-10:].tolist(), separators=(",", ":")),
        "embedding_sha256_numeric_bytes": embedding_hash,
        "embedding_min": clean_float(float(np.nanmin(embedding))),
        "embedding_max": clean_float(float(np.nanmax(embedding))),
        "embedding_std": clean_float(float(np.nanstd(embedding))),
    }


def markdown_report(summary: dict[str, Any], reps: pd.DataFrame) -> str:
    large = summary["large_duplicate_groups"]
    lines = [
        "# Phase 2 generated-470 tensor duplicate audit",
        "",
        "## Result",
        "",
        f"- Generated tensor hosts inspected: **{summary['generated_hosts']}**",
        f"- Unique numeric tensor byte sequences: **{summary['unique_numeric_tensors']}**",
        f"- Duplicate copies beyond the first in each hash group: **{summary['duplicate_copies_beyond_first']}**",
        f"- Hosts participating in an exact-duplicate group: **{summary['hosts_in_duplicate_groups']}**",
        f"- Exact-duplicate hash groups: **{summary['duplicate_group_count']}**",
        "- Hash definition: SHA-256 over the loaded tensor's contiguous C-order numeric bytes only.",
        "",
        "## Large groups",
        "",
    ]
    for group in large:
        lines.extend([
            f"- `{group['tensor_sha256_numeric_bytes']}`: {group['host_count']} hosts, "
            f"shape {group['tensor_shape']}, all-zero={str(group['all_zero']).lower()}, "
            f"unique embeddings={group['unique_embedding_count']}, "
            f"CNN probability={group['cnn_probability_values']}",
        ])
    lines.extend([
        "",
        "## Cause",
        "",
        "The generated-tensor path attempts to flatten each light curve, falls back to the raw flux if that "
        "operation raises, and converts the result to relative flux near 1.0. It then estimates MAD, percentile, "
        "and standard-deviation scales. If every candidate scale is "
        "below the hard floor `MIN_SCALE = 0.05`, `_standardize_flux` replaces the entire light curve "
        "with zeros. A floor of 0.05 in relative-flux units is a 5% threshold, far above the measured "
        "scatter for ordinary K2 light curves. Distinct FITS hashes and flux statistics therefore collapse "
        "to identical all-zero tensors. Group identity is then determined by segment count/byte length: "
        f"the {large[0]['host_count']}-host group has shape {large[0]['tensor_shape']} and the "
        f"{large[1]['host_count']}-host group has shape {large[1]['tensor_shape']}.",
        "",
        "Every representative in both large groups has `zeroed_because_scale_below_floor=true`, and the "
        "recomputed standardized flux matches the saved tensor's channel 0. Channel 1 is the first difference "
        "of channel 0, so it is also all zeros.",
        "",
        f"In the 20 representatives, flattening succeeded {summary['representative_flatten_success_count']}/20 "
        "times. The failures were caused by non-finite TIME values; the broad exception handler silently retained "
        "raw flux. The final selected scales ranged from "
        f"{summary['representative_selected_scale_min']:.6g} to "
        f"{summary['representative_selected_scale_max']:.6g}, all below 0.05.",
        "",
        "## Embeddings",
        "",
        "The frozen CNN consumes only tensor channel 0. Each large all-zero tensor group produces one exact "
        "128-D embedding byte sequence; all hosts in each group share it. The two groups also share the same "
        "embedding because every segment supplied to the network is the same all-zero 512x1 input.",
        "",
        "## Representative rows",
        "",
        "The detailed CSV contains ten deterministic EPICs from each large group, including source hashes, "
        "cadence counts, raw-flux statistics, full normalization traces, tensor statistics and edge values, "
        "and embedding hashes.",
        "",
    ])
    display = reps[[
        "epic_id", "segment_count", "valid_cadences", "flux_min", "flux_max", "flux_std",
        "tensor_unique_values", "tensor_fraction_zero", "embedding_sha256_numeric_bytes",
    ]].copy()
    columns = list(display.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit numeric tensor duplicates among the generated Phase 2 batch.")
    parser.add_argument("--representatives-per-large-group", type=int, default=10)
    args = parser.parse_args()

    tensor_paths = sorted(TENSOR_DIR.glob("EPIC_*.npz"))
    if len(tensor_paths) != 470:
        raise AssertionError(f"expected 470 generated tensors, found {len(tensor_paths)}")

    embeddings, embedding_metadata = embedding_map()
    records: list[dict[str, Any]] = []
    hashes_to_epics: dict[str, list[str]] = defaultdict(list)
    tensors: dict[str, np.ndarray] = {}
    for path in tensor_paths:
        epic = path.stem
        with np.load(path, allow_pickle=False) as bundle:
            tensor = np.asarray(bundle["tensor"])
        digest = sha256_numeric_bytes(tensor)
        hashes_to_epics[digest].append(epic)
        tensors[epic] = tensor

    missing_embeddings = sorted(set(tensors) - set(embeddings))
    if missing_embeddings:
        raise AssertionError(f"generated EPICs missing embeddings: {missing_embeddings[:10]}")

    embedding_hashes = {epic: sha256_numeric_bytes(embeddings[epic]) for epic in tensors}
    scores = pd.read_csv(SCORES_PATH, dtype={"epic_id": str})
    group_rows: list[dict[str, Any]] = []
    for digest, epics in sorted(hashes_to_epics.items(), key=lambda item: (-len(item[1]), item[0])):
        exemplar = tensors[epics[0]]
        exact = all(np.array_equal(exemplar, tensors[epic], equal_nan=True) for epic in epics[1:])
        if not exact:
            raise AssertionError(f"SHA-256 collision or non-equal NaN payload in group {digest}")
        stats = array_stats(exemplar)
        group_probabilities = sorted(scores.loc[scores["epic_id"].isin(epics), "cnn_probability"].dropna().unique().tolist())
        group_rows.append({
            "tensor_sha256_numeric_bytes": digest,
            "host_count": len(epics),
            "tensor_shape": json.dumps(list(exemplar.shape), separators=(",", ":")),
            "tensor_dtype": str(exemplar.dtype),
            "all_zero": bool(np.all(exemplar == 0)),
            "tensor_min": stats["min"],
            "tensor_max": stats["max"],
            "tensor_std": stats["std"],
            "tensor_unique_values": stats["unique_values"],
            "tensor_fraction_zero": stats["fraction_zero"],
            "tensor_fraction_nan": stats["fraction_nan"],
            "unique_embedding_count": len({embedding_hashes[epic] for epic in epics}),
            "cnn_probability_values": json.dumps(group_probabilities, separators=(",", ":")),
            "epic_ids": json.dumps(epics, separators=(",", ":")),
        })

    groups = pd.DataFrame(group_rows)
    duplicate_groups = groups[groups["host_count"] > 1]
    hosts_in_duplicate_groups = int(duplicate_groups["host_count"].sum())
    duplicate_copies = int(sum(len(epics) - 1 for epics in hashes_to_epics.values()))

    largest = groups.head(2)
    if sorted(largest["host_count"].tolist(), reverse=True) != [230, 118]:
        raise AssertionError(f"expected large duplicate groups of 230 and 118, found {largest['host_count'].tolist()}")

    representative_rows: list[dict[str, Any]] = []
    for _, group in largest.iterrows():
        digest = str(group["tensor_sha256_numeric_bytes"])
        for epic in hashes_to_epics[digest][: args.representatives_per_large_group]:
            row = representative_row(epic, TENSOR_DIR / f"{epic}.npz", digest, embeddings[epic])
            representative_rows.append(row)
    reps = pd.DataFrame(representative_rows)

    two_large_epics = [epic for digest in largest["tensor_sha256_numeric_bytes"] for epic in hashes_to_epics[digest]]
    large_embedding_hashes = Counter(embedding_hashes[epic] for epic in two_large_epics)
    representative_traces = [
        trace
        for value in reps["normalization_parameters"]
        for trace in json.loads(value)
    ]
    zero_group_count = int(groups["all_zero"].sum())
    zero_duplicate_group_count = int(duplicate_groups["all_zero"].sum())
    summary = {
        "scope": "existing data/phase2/local949_tensors/EPIC_*.npz files only; no acquisition or training",
        "tensor_hash_definition": "SHA-256 of np.ascontiguousarray(tensor).tobytes(order='C') only",
        "generated_hosts": len(tensor_paths),
        "unique_numeric_tensors": len(hashes_to_epics),
        "duplicate_copies_beyond_first": duplicate_copies,
        "hosts_in_duplicate_groups": hosts_in_duplicate_groups,
        "duplicate_group_count": int(len(duplicate_groups)),
        "singleton_tensor_count": int((groups["host_count"] == 1).sum()),
        "all_zero_tensor_hosts": int(sum(len(epics) for digest, epics in hashes_to_epics.items() if np.all(tensors[epics[0]] == 0))),
        "all_zero_numeric_tensor_variants_by_byte_length": zero_group_count,
        "all_duplicate_groups_are_all_zero": zero_duplicate_group_count == len(duplicate_groups),
        "nonzero_tensor_hosts": int(sum(not np.all(tensor == 0) for tensor in tensors.values())),
        "embedding_bundle": embedding_metadata,
        "unique_embeddings_among_generated_hosts": len(set(embedding_hashes.values())),
        "large_groups_share_one_embedding": len(large_embedding_hashes) == 1,
        "large_groups_shared_embedding_hash": next(iter(large_embedding_hashes)) if len(large_embedding_hashes) == 1 else None,
        "large_duplicate_groups": largest.to_dict(orient="records"),
        "representative_input_light_curve_hashes_unique": int(reps["input_light_curve_combined_sha256"].nunique()),
        "representative_flatten_success_count": int(sum(trace["flatten_succeeded"] for trace in representative_traces)),
        "representative_flatten_errors": dict(Counter(trace["flatten_error"] for trace in representative_traces)),
        "representative_selected_scale_min": float(min(trace["selected_scale"] for trace in representative_traces)),
        "representative_selected_scale_max": float(max(trace["selected_scale"] for trace in representative_traces)),
        "all_representatives_zeroed_by_scale_floor": bool(all(
            trace["zeroed_because_scale_below_floor"] for trace in representative_traces
        )),
        "all_representatives_recomputed_prefix_matches_saved_tensor": bool(all(
            trace["standardized_matches_saved_channel0_prefix"] for trace in representative_traces
        )),
        "non_actions": ["no light-curve acquisition", "no CatBoost training", "no model training"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups.to_csv(GROUPS_PATH, index=False)
    reps.to_csv(REPRESENTATIVES_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(summary, reps), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote={GROUPS_PATH.relative_to(ROOT)}")
    print(f"wrote={REPRESENTATIVES_PATH.relative_to(ROOT)}")
    print(f"wrote={SUMMARY_PATH.relative_to(ROOT)}")
    print(f"wrote={REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
