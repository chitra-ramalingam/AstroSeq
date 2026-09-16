from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/phase2/phase2_feature_generation_manifest.parquet"
LOCAL_STATUS = ROOT / "docs/phase2/phase2_local949_generation_status.csv"
DOWNLOAD_ROOT = ROOT / "k2_cache"
PROVENANCE = ROOT / "docs/phase2/phase2_remaining858_light_curve_provenance.csv"
ACQUISITION_STATUS = ROOT / "docs/phase2/phase2_remaining858_acquisition_status.csv"
ACQUISITION_SUMMARY = ROOT / "docs/phase2/phase2_remaining858_acquisition_summary.json"

EXPECTED_ELIGIBLE = 1807
EXPECTED_LOCAL = 949
EXPECTED_REMAINING = 858
QUALITY_HANDLING = "quality_bitmask=none; raw FITS retained; scientific loader applies its recorded deterministic quality policy"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def derive_population() -> pd.DataFrame:
    manifest = pd.read_parquet(MANIFEST)
    eligible = manifest.loc[as_bool(manifest["physical_loss_eligible"])].copy()
    local_ids = set(pd.read_csv(LOCAL_STATUS, usecols=["epic_id"])["epic_id"].astype(str))
    eligible_ids = set(eligible["epic_id"].astype(str))
    remaining_ids = eligible_ids - local_ids
    checks = {
        "eligible": len(eligible_ids),
        "local": len(local_ids),
        "remaining": len(remaining_ids),
        "intersection": len(local_ids & remaining_ids),
        "union": len(local_ids | remaining_ids),
    }
    if checks != {
        "eligible": EXPECTED_ELIGIBLE,
        "local": EXPECTED_LOCAL,
        "remaining": EXPECTED_REMAINING,
        "intersection": 0,
        "union": EXPECTED_ELIGIBLE,
    }:
        raise RuntimeError(f"remaining-858 population gate failed: {checks}")
    cohort = eligible.loc[eligible["epic_id"].astype(str).isin(remaining_ids)].copy()
    cohort["epic_id"] = cohort["epic_id"].astype(str)
    return cohort.sort_values("epic_id", kind="mergesort").reset_index(drop=True)


def clean(value: Any) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "--"} else text


def campaign_from_record(record: dict[str, Any]) -> str:
    sequence = clean(record.get("sequence_number"))
    if sequence and re.fullmatch(r"-?\d+", sequence):
        return str(int(sequence))
    text = " ".join(clean(record.get(key)) for key in ("mission", "obs_id", "productFilename"))
    match = re.search(r"(?:Campaign\s*|[-_]c)(\d{1,3})", text, flags=re.IGNORECASE)
    if not match:
        return ""
    digits = match.group(1)
    return str(int(digits[:2])) if len(digits) == 3 else str(int(digits))


def cadence_type(record: dict[str, Any]) -> str:
    filename = clean(record.get("productFilename")).lower()
    if "_llc.fits" in filename:
        return "long"
    if "_slc.fits" in filename:
        return "short"
    exposure = clean(record.get("exptime") or record.get("t_exptime"))
    try:
        return "short" if float(exposure) < 600 else "long"
    except ValueError:
        return "unknown"


def product_matches_epic(product_filename: str, epic: str) -> bool:
    match = re.search(r"(?:ktwo|lightcurve_)(\d{8,10})", product_filename.lower())
    if not match:
        return False
    return match.group(1).lstrip("0") == epic.removeprefix("EPIC_").lstrip("0")


def validate_fits(path: Path) -> tuple[bool, str]:
    try:
        with fits.open(path, memmap=True, mode="readonly") as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                return False, "missing_binary_table"
            names = {str(name).upper() for name in (hdul[1].data.names or [])}
            if "TIME" not in names or not names.intersection({"FLUX", "PDCSAP_FLUX", "SAP_FLUX"}):
                return False, "missing_TIME_or_flux"
            if len(hdul[1].data) < 512:
                return False, "fewer_than_512_rows"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    return True, "validated_k2_fits"


def locate_download(product_filename: str, light_curve: Any) -> Path:
    candidates: list[Path] = []
    filename_attr = getattr(light_curve, "filename", None)
    if filename_attr:
        candidates.append(Path(filename_attr))
    meta_filename = getattr(light_curve, "meta", {}).get("FILENAME") if hasattr(light_curve, "meta") else None
    if meta_filename:
        candidates.append(Path(str(meta_filename)))
    candidates.extend(DOWNLOAD_ROOT.rglob(product_filename))
    for candidate in candidates:
        if candidate.exists() and candidate.name == product_filename:
            return candidate.resolve()
    raise FileNotFoundError(f"download completed but product cannot be located: {product_filename}")


def selected_records(
    search_result: Any,
    expected_campaigns: set[str],
    allowed_provenance: tuple[str, ...] = ("K2",),
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index in range(len(search_result)):
        record = {name: search_result.table[index][name] for name in search_result.table.colnames}
        filename = clean(record.get("productFilename"))
        provenance = clean(record.get("provenance_name") or record.get("author"))
        campaign = campaign_from_record(record)
        if provenance.upper() not in {value.upper() for value in allowed_provenance}:
            continue
        if not filename.lower().endswith(("_llc.fits", "_slc.fits")):
            continue
        if expected_campaigns and campaign and campaign not in expected_campaigns:
            continue
        selected.append((index, record))
    selected.sort(
        key=lambda pair: (
            int(campaign_from_record(pair[1])) if campaign_from_record(pair[1]).lstrip("-").isdigit() else 999,
            0 if cadence_type(pair[1]) == "long" else 1,
            clean(pair[1].get("productFilename")),
        )
    )
    return selected


def recover_fallbacks(retries: int) -> None:
    import lightkurve as lk

    status = pd.read_csv(ACQUISITION_STATUS, dtype={"epic_id": str})
    products = pd.read_csv(PROVENANCE, dtype={"epic_id": str})
    identity_ok = products.apply(lambda row: product_matches_epic(str(row["product_filename"]), str(row["epic_id"])), axis=1)
    products.loc[~identity_ok, "validation_status"] = "invalid"
    products.loc[~identity_ok, "validation_reason"] = "product_target_epic_mismatch"
    valid_counts = products.loc[products["validation_status"].eq("valid")].groupby("epic_id").size()
    status["valid_products_acquired"] = status["epic_id"].map(valid_counts).fillna(0).astype(int)
    status["acquisition_status"] = np.where(status["valid_products_acquired"].gt(0), "acquired", "failed")
    mismatch_ids = set(products.loc[~identity_ok, "epic_id"].astype(str))
    status.loc[status["epic_id"].isin(mismatch_ids), "failure_reason"] = "product_target_epic_mismatch; no_exact_EPIC_product"
    failed_ids = status.loc[status["acquisition_status"].eq("failed"), "epic_id"].astype(str).tolist()
    cohort = derive_population().set_index("epic_id")
    for epic in failed_ids:
        epic_number = epic.replace("EPIC_", "")
        expected = {
            str(value) for value in json.loads(str(cohort.loc[epic, "campaigns"]))
            if str(value).lstrip("-").isdigit() and int(value) >= 0
        }
        errors: list[str] = []
        search_result = None
        for attempt in range(1, retries + 1):
            try:
                search_result = lk.search_lightcurve(
                    f"EPIC {epic_number}", mission="K2", author=("EVEREST", "K2SFF")
                )
                break
            except Exception as exc:
                errors.append(f"fallback_search_attempt_{attempt}:{type(exc).__name__}:{exc}")
        recovered: list[dict[str, Any]] = []
        if search_result is not None:
            selected = selected_records(search_result, expected, ("EVEREST", "K2SFF"))
            for result_index, record in selected:
                filename = clean(record.get("productFilename"))
                if not product_matches_epic(filename, epic):
                    errors.append(f"{filename}:product_target_epic_mismatch")
                    continue
                downloaded_path = None
                for attempt in range(1, retries + 1):
                    try:
                        lc = search_result[result_index].download(
                            download_dir=str(DOWNLOAD_ROOT), cache=True, quality_bitmask="none"
                        )
                        downloaded_path = locate_download(filename, lc)
                        break
                    except Exception as exc:
                        errors.append(f"{filename}:fallback_download_attempt_{attempt}:{type(exc).__name__}:{exc}")
                if downloaded_path is None:
                    continue
                valid, validation_reason = validate_fits(downloaded_path)
                provenance_name = clean(record.get("provenance_name") or record.get("author"))
                data_uri = clean(record.get("dataURI"))
                recovered.append({
                    "epic_id": epic,
                    "campaign": campaign_from_record(record),
                    "source": f"MAST_{provenance_name}",
                    "source_path": downloaded_path.as_posix(),
                    "source_url_provenance": "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(data_uri, safe=":/") if data_uri else "",
                    "data_uri": data_uri,
                    "product_identifier": clean(record.get("obs_id") or record.get("obsID")),
                    "product_filename": filename,
                    "cadence_type": cadence_type(record),
                    "file_sha256": sha256_file(downloaded_path),
                    "quality_handling": QUALITY_HANDLING,
                    "selected_product_reason": "official_K2_unavailable; allowed_fallback_priority_EVEREST_then_K2SFF; deterministic_campaign_cadence_filename_order",
                    "validation_status": "valid" if valid else "invalid",
                    "validation_reason": validation_reason,
                    "acquired_at_utc": utc_now(),
                })
        valid_recovered = [item for item in recovered if item["validation_status"] == "valid"]
        if recovered:
            products = pd.concat([products, pd.DataFrame(recovered)], ignore_index=True)
        mask = status["epic_id"].eq(epic)
        status.loc[mask, "products_selected"] = len(recovered)
        status.loc[mask, "valid_products_acquired"] = len(valid_recovered)
        status.loc[mask, "acquisition_status"] = "acquired" if valid_recovered else "failed"
        if valid_recovered:
            status.loc[mask, "failure_reason"] = "official_K2_unavailable; recovered_allowed_fallback"
        else:
            status.loc[mask, "failure_reason"] = "no_K2_EVEREST_or_K2SFF_light_curve_product|" + "|".join(errors)[:3500]
    products = products.drop_duplicates(["epic_id", "product_filename"], keep="last")
    products = products.sort_values(["epic_id", "campaign", "cadence_type", "product_filename"], kind="mergesort")
    status = status.sort_values("epic_id", kind="mergesort")
    atomic_csv(products, PROVENANCE)
    atomic_csv(status, ACQUISITION_STATUS)
    summary = json.loads(ACQUISITION_SUMMARY.read_text(encoding="utf-8"))
    acquired = int(status["acquisition_status"].eq("acquired").sum())
    summary.update({
        "generated_at_utc": utc_now(),
        "hosts_acquired": acquired,
        "hosts_failed": EXPECTED_REMAINING - acquired,
        "selected_products": len(products),
        "valid_products": int(products["validation_status"].eq("valid").sum()),
        "fallback_policy": "after official K2 absence only: EVEREST, then K2SFF; K2SC and K2VARCAT banned",
    })
    atomic_json(summary, ACQUISITION_SUMMARY)
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge_shards() -> None:
    shard_dir = ROOT / "docs/phase2/acquisition_shards"
    status_paths = sorted(shard_dir.glob("*_status.csv"))
    provenance_paths = sorted(shard_dir.glob("*_provenance.csv"))
    if not status_paths or not provenance_paths:
        raise FileNotFoundError("acquisition shard outputs are missing")
    status = pd.concat([pd.read_csv(path, dtype={"epic_id": str}) for path in status_paths], ignore_index=True)
    products = pd.concat([pd.read_csv(path, dtype={"epic_id": str}) for path in provenance_paths], ignore_index=True)
    if len(status) != EXPECTED_REMAINING or status["epic_id"].nunique() != EXPECTED_REMAINING:
        raise RuntimeError(f"shard host coverage failure: rows={len(status)} unique={status['epic_id'].nunique()}")
    if status["epic_id"].duplicated().any():
        raise RuntimeError("shard merge found duplicated EPIC hosts")
    if products.duplicated(["epic_id", "product_filename"]).any():
        raise RuntimeError("shard merge found duplicated host/product records")
    status = status.sort_values("epic_id", kind="mergesort").reset_index(drop=True)
    products = products.sort_values(["epic_id", "campaign", "cadence_type", "product_filename"], kind="mergesort").reset_index(drop=True)
    atomic_csv(products, PROVENANCE)
    atomic_csv(status, ACQUISITION_STATUS)
    acquired = int(status["acquisition_status"].eq("acquired").sum())
    summary = {
        "generated_at_utc": utc_now(),
        "hosts_requested": EXPECTED_REMAINING,
        "hosts_acquired": acquired,
        "hosts_failed": EXPECTED_REMAINING - acquired,
        "selected_products": len(products),
        "valid_products": int(products["validation_status"].eq("valid").sum()),
        "population_assertions": {"local": 949, "remaining": 858, "intersection": 0, "union": 1807},
        "quality_handling": QUALITY_HANDLING,
        "selection_policy": "official K2 author only; all products matching canonical campaigns; stable campaign/cadence/filename order",
        "shards_merged": len(status_paths),
        "outputs": [PROVENANCE.relative_to(ROOT).as_posix(), ACQUISITION_STATUS.relative_to(ROOT).as_posix()],
    }
    atomic_json(summary, ACQUISITION_SUMMARY)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire provenance-controlled K2 products for the Phase 2 remaining 858.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=EXPECTED_REMAINING)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--recover-fallbacks", action="store_true")
    args = parser.parse_args()

    if args.merge_shards:
        merge_shards()
        return
    if args.recover_fallbacks:
        recover_fallbacks(args.retries)
        return

    import lightkurve as lk

    full_cohort = derive_population()
    start = max(0, args.start_index)
    stop = min(EXPECTED_REMAINING, args.stop_index)
    if start >= stop:
        raise ValueError(f"invalid shard bounds: start={start} stop={stop}")
    cohort = full_cohort.iloc[start:stop].copy().reset_index(drop=True)
    if args.shard_name:
        shard_dir = ROOT / "docs/phase2/acquisition_shards"
        provenance_path = shard_dir / f"{args.shard_name}_provenance.csv"
        status_path = shard_dir / f"{args.shard_name}_status.csv"
        summary_path = shard_dir / f"{args.shard_name}_summary.json"
    else:
        provenance_path = PROVENANCE
        status_path = ACQUISITION_STATUS
        summary_path = ACQUISITION_SUMMARY
    product_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for host_index, row in cohort.iterrows():
        epic = str(row["epic_id"])
        epic_number = epic.replace("EPIC_", "")
        campaigns = {str(value) for value in json.loads(str(row["campaigns"])) if str(value).lstrip("-").isdigit() and int(value) >= 0}
        host_products: list[dict[str, Any]] = []
        errors: list[str] = []
        search_result = None
        for attempt in range(1, args.retries + 1):
            try:
                search_result = lk.search_lightcurve(f"EPIC {epic_number}", mission="K2", author="K2")
                break
            except Exception as exc:
                errors.append(f"search_attempt_{attempt}:{type(exc).__name__}:{exc}")
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 8))
        if search_result is not None:
            selected = selected_records(search_result, campaigns)
            if not selected:
                errors.append("no_selected_official_K2_light_curve_product")
            for result_index, record in selected:
                filename = clean(record.get("productFilename"))
                if not product_matches_epic(filename, epic):
                    errors.append(f"{filename}:product_target_epic_mismatch")
                    continue
                download_error = ""
                downloaded_path: Path | None = None
                for attempt in range(1, args.retries + 1):
                    try:
                        lc = search_result[result_index].download(
                            download_dir=str(DOWNLOAD_ROOT), cache=True, quality_bitmask="none"
                        )
                        if lc is None:
                            raise RuntimeError("Lightkurve returned no product")
                        downloaded_path = locate_download(filename, lc)
                        break
                    except Exception as exc:
                        download_error = f"download_attempt_{attempt}:{type(exc).__name__}:{exc}"
                        if attempt < args.retries:
                            time.sleep(min(2 ** attempt, 8))
                if downloaded_path is None:
                    errors.append(f"{filename}:{download_error}")
                    continue
                valid, validation_reason = validate_fits(downloaded_path)
                data_uri = clean(record.get("dataURI"))
                provenance_url = (
                    "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(data_uri, safe=":/")
                    if data_uri else ""
                )
                product = {
                    "epic_id": epic,
                    "campaign": campaign_from_record(record),
                    "source": "MAST_official_K2",
                    "source_path": downloaded_path.as_posix(),
                    "source_url_provenance": provenance_url,
                    "data_uri": data_uri,
                    "product_identifier": clean(record.get("obs_id") or record.get("obsID")),
                    "product_filename": filename,
                    "cadence_type": cadence_type(record),
                    "file_sha256": sha256_file(downloaded_path),
                    "quality_handling": QUALITY_HANDLING,
                    "selected_product_reason": "official_K2_light_curve; retain_all_canonical_campaign_products; deterministic_campaign_cadence_filename_order",
                    "validation_status": "valid" if valid else "invalid",
                    "validation_reason": validation_reason,
                    "acquired_at_utc": utc_now(),
                }
                host_products.append(product)
                product_rows.append(product)
                if not valid:
                    errors.append(f"{filename}:{validation_reason}")

        valid_products = [item for item in host_products if item["validation_status"] == "valid"]
        status_rows.append({
            "epic_id": epic,
            "canonical_campaigns": json.dumps(sorted(campaigns), separators=(",", ":")),
            "products_selected": len(host_products),
            "valid_products_acquired": len(valid_products),
            "acquisition_status": "acquired" if valid_products else "failed",
            "failure_reason": "|".join(errors)[:4000],
        })
        atomic_csv(pd.DataFrame(product_rows), provenance_path)
        atomic_csv(pd.DataFrame(status_rows), status_path)
        if (host_index + 1) % args.progress_every == 0 or host_index + 1 == len(cohort):
            acquired = sum(item["acquisition_status"] == "acquired" for item in status_rows)
            print(f"acquisition processed={host_index + 1}/{len(cohort)} acquired={acquired}", flush=True)

    status = pd.DataFrame(status_rows)
    acquired = int(status["acquisition_status"].eq("acquired").sum())
    summary = {
        "generated_at_utc": utc_now(),
        "hosts_requested": len(cohort),
        "hosts_acquired": acquired,
        "hosts_failed": len(cohort) - acquired,
        "shard": {"name": args.shard_name, "start_index": start, "stop_index": stop},
        "selected_products": len(product_rows),
        "valid_products": sum(item["validation_status"] == "valid" for item in product_rows),
        "population_assertions": {"local": 949, "remaining": 858, "intersection": 0, "union": 1807},
        "quality_handling": QUALITY_HANDLING,
        "selection_policy": "official K2 author only; all products matching canonical campaigns; stable campaign/cadence/filename order",
        "outputs": [provenance_path.relative_to(ROOT).as_posix(), status_path.relative_to(ROOT).as_posix()],
    }
    atomic_json(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
