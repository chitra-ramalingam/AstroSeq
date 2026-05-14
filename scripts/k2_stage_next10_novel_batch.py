from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Classifiers.K2.Batch.K2ShortlistPeriodRunner import K2ShortlistPeriodRunner
from src.Classifiers.K2.Batch.K2StageFFollowupValidation import K2StageFFollowupValidation


ROOT = Path(".")
STAGE_D_PASS_RANKED = ROOT / "k2_stage_d_pass_ranked.csv"
STAGE_H_STATUS = ROOT / "k2_stage_h_final_candidate_status_v3.csv"
STAGE_D_TOP10_INDEX = ROOT / "k2_stage_d_top10_inspection_index.csv"
STAGE_F_INPUT = ROOT / "k2_stage_f_next10_input.csv"
STAGE_F_OUTPUT = ROOT / "k2_stage_f_next10_validation.csv"
STAGE_F_OUT_DIR = ROOT / r"plots\k2_batch\stage_f_next10"
SUMMARY_DIR = ROOT / r"plots\k2_batch\stage_f_next10_input_summaries"
STAGE_H_OUTPUT = ROOT / "k2_stage_h_next10_external_vetting.csv"
NOVEL_QUEUE_OUTPUT = ROOT / "k2_next_novel_candidate_queue.csv"

EXPLICIT_EXCLUDES = {"EPIC_211889692", "EPIC_211534076"}
KNOWN_EXCLUDE_LABELS = {
    "known_confirmed_planet",
    "known_unconfirmed_candidate",
    "known_eb_or_variable",
}

CONFIRMED_PLANET_FILES = [
    ROOT / r"plots\k2_batch\nasa_confirmed_k2_planets_reference.csv",
    ROOT / "CombinedExoplanetData.csv",
    ROOT / "Cut_CombinedExoplanetData.csv",
]
CANDIDATE_FILES = [
    ROOT / "K2_ephemerides.csv",
]
EB_VARIABLE_FILES = [
    ROOT / "binary_star_classification_results.csv",
]
TARGET_LIST = ROOT / r"data\k2_target_lists\K2Campaign5targets.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _epic_digits(epic_id: str) -> str:
    return "".join(ch for ch in str(epic_id) if ch.isdigit())


def _is_known_status(row: pd.Series) -> bool:
    return str(row.get("stage_h_label", "")).strip() in KNOWN_EXCLUDE_LABELS


def select_next10() -> pd.DataFrame:
    ranked = _read_csv(STAGE_D_PASS_RANKED)
    status = _read_csv(STAGE_H_STATUS)
    known_epics = set()
    if len(status) > 0:
        for _, row in status.iterrows():
            if _is_known_status(row):
                known_epics.add(str(row.get("epic_id", "")).strip())
    exclude = EXPLICIT_EXCLUDES | known_epics
    selected_rows = []
    for _, row in ranked.loc[ranked["stage_d_label"].astype(str).eq("pass_deeper_eval")].iterrows():
        epic_id = str(row.get("epic_id", "")).strip()
        if epic_id in exclude:
            continue
        pc, _ = _count_confirmed_matches(epic_id)
        cand, _ = _count_candidate_matches(epic_id)
        eb, _ = _count_eb_variable_matches(epic_id)
        if pc > 0 or cand > 0 or eb > 0:
            continue
        selected_rows.append(row)
        if len(selected_rows) == 10:
            break
    selected = pd.DataFrame(selected_rows)
    if len(selected) != 10:
        raise RuntimeError(f"Expected 10 selected Stage D pass candidates; got {len(selected)}")
    return selected.reset_index(drop=True)


def _event_path(epic_id: str) -> Path:
    return ROOT / "plots" / "k2_batch" / "epics" / f"EPIC_{_epic_digits(epic_id)}" / "events.csv"


def _cluster_center(events_csv: Path, period: float) -> Tuple[int, float]:
    events = _read_csv(events_csv)
    if len(events) == 0:
        return 0, 0.0
    try:
        filtered = K2ShortlistPeriodRunner._filter_events_for_periods(events)
        support, center = K2ShortlistPeriodRunner._phase_cluster_score_quiet(
            events_df=filtered,
            period=float(period),
            tol_phase=0.03,
        )
        if not math.isfinite(float(center)):
            center = 0.0
        return int(support), float(center)
    except Exception:
        return 0, 0.0


def write_stage_f_input(selected: pd.DataFrame) -> pd.DataFrame:
    top10 = _read_csv(STAGE_D_TOP10_INDEX)
    summary_by_epic = {}
    if len(top10) > 0:
        summary_by_epic = dict(zip(top10["epic_id"].astype(str), top10["summary_json_path"].astype(str)))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        epic_id = str(row["epic_id"])
        period = float(row["best_period_days"])
        summary_path_raw = summary_by_epic.get(epic_id, "")
        summary_path = Path(summary_path_raw) if str(summary_path_raw).strip() else Path()
        if not str(summary_path_raw).strip() or not summary_path.is_file():
            events_csv = _event_path(epic_id)
            support, center = _cluster_center(events_csv, period)
            summary_path = SUMMARY_DIR / epic_id / f"{epic_id}_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = {k: _json_scalar(row.get(k)) for k in row.index}
            metrics["cluster_center_phase"] = center
            metrics["period_support_count"] = int(row.get("period_support_count", support) or support)
            summary = {
                "epic_id": epic_id,
                "query": f"EPIC {_epic_digits(epic_id)}",
                "stage_r_and_stage_d_metrics": metrics,
                "artifacts": {
                    "events_csv": str(events_csv),
                    "light_curve_cache_path": "",
                },
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(
            {
                "epic_id": epic_id,
                "best_period_days": row["best_period_days"],
                "period_support_count": row["period_support_count"],
                "visual_label": "visual_planet_like_candidate",
                "summary_json_path": str(summary_path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(STAGE_F_INPUT, index=False)
    return out


def _json_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def run_stage_f() -> Dict[str, Any]:
    return K2StageFFollowupValidation().run(
        input_csv=STAGE_F_INPUT,
        output_csv=STAGE_F_OUTPUT,
        out_dir=STAGE_F_OUT_DIR,
    )


def _row_contains_epic(row: pd.Series, epic_digits: str) -> bool:
    haystack = " ".join(str(v) for v in row.to_dict().values())
    return f"EPIC {epic_digits}" in haystack or f"EPIC_{epic_digits}" in haystack or epic_digits in haystack


def _count_confirmed_matches(epic_id: str) -> Tuple[int, List[Dict[str, str]]]:
    digits = _epic_digits(epic_id)
    hits: List[Dict[str, str]] = []
    for path in CONFIRMED_PLANET_FILES:
        df = _read_csv(path)
        if len(df) == 0:
            continue
        for idx, row in df.iterrows():
            if not _row_contains_epic(row, digits):
                continue
            disposition = " ".join(
                str(row.get(col, ""))
                for col in ["disposition", "koi_disposition", "koi_pdisposition", "tfopwg_disp"]
                if col in row.index
            ).lower()
            is_confirmed = (
                "confirmed" in disposition
                or path.name == "nasa_confirmed_k2_planets_reference.csv"
            )
            if is_confirmed:
                hits.append({"file": str(path), "row": str(idx + 2)})
    return len(hits), hits


def _count_candidate_matches(epic_id: str) -> Tuple[int, List[Dict[str, str]]]:
    digits = _epic_digits(epic_id)
    hits: List[Dict[str, str]] = []
    for path in CANDIDATE_FILES + CONFIRMED_PLANET_FILES:
        df = _read_csv(path)
        if len(df) == 0:
            continue
        for idx, row in df.iterrows():
            if not _row_contains_epic(row, digits):
                continue
            text = " ".join(str(v) for v in row.to_dict().values()).lower()
            if "candidate" in text and "false positive" not in text:
                hits.append({"file": str(path), "row": str(idx + 2)})
    return len(hits), hits


def _count_eb_variable_matches(epic_id: str) -> Tuple[int, List[Dict[str, str]]]:
    digits = _epic_digits(epic_id)
    hits: List[Dict[str, str]] = []
    for path in EB_VARIABLE_FILES:
        df = _read_csv(path)
        if len(df) == 0:
            continue
        for idx, row in df.iterrows():
            if _row_contains_epic(row, digits):
                hits.append({"file": str(path), "row": str(idx + 2)})
    return len(hits), hits


def _target_table() -> pd.DataFrame:
    df = _read_csv(TARGET_LIST)
    if len(df) == 0:
        return df
    rename = {
        "EPIC ID": "epic_digits",
        "RA (J2000) [deg]": "ra_deg",
        "Dec (J2000) [deg]": "dec_deg",
        "magnitude": "k2_magnitude",
        "Investigation IDs": "investigation_ids",
    }
    df = df.rename(columns=rename)
    for col in ["ra_deg", "dec_deg", "k2_magnitude"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["epic_id"] = "EPIC_" + df["epic_digits"].astype(str)
    return df


def _sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    if not all(math.isfinite(x) for x in [ra1, dec1, ra2, dec2]):
        return float("nan")
    dra = math.radians(ra2 - ra1) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = math.radians(dec2 - dec1)
    return math.degrees(math.hypot(dra, ddec)) * 3600.0


def _contamination(epic_id: str, targets: pd.DataFrame) -> Dict[str, Any]:
    if len(targets) == 0:
        return {
            "ra_deg": "",
            "dec_deg": "",
            "k2_magnitude": "",
            "investigation_ids": "",
            "nearby_sources_within_60_arcsec": 0,
            "nearby_sources_within_120_arcsec": 0,
            "nearest_neighbor_epic_id": "",
            "nearest_neighbor_sep_arcsec": "",
            "nearest_neighbor_mag": "",
            "nearest_neighbor_delta_mag": "",
            "contamination_risk": "unknown",
            "contamination_notes": "Campaign target list unavailable.",
        }
    current = targets.loc[targets["epic_id"].astype(str).eq(epic_id)]
    if len(current) == 0:
        return {
            "ra_deg": "",
            "dec_deg": "",
            "k2_magnitude": "",
            "investigation_ids": "",
            "nearby_sources_within_60_arcsec": 0,
            "nearby_sources_within_120_arcsec": 0,
            "nearest_neighbor_epic_id": "",
            "nearest_neighbor_sep_arcsec": "",
            "nearest_neighbor_mag": "",
            "nearest_neighbor_delta_mag": "",
            "contamination_risk": "unknown",
            "contamination_notes": "EPIC not found in Campaign 5 target list.",
        }
    r = current.iloc[0]
    ra = float(r["ra_deg"]) if pd.notna(r["ra_deg"]) else float("nan")
    dec = float(r["dec_deg"]) if pd.notna(r["dec_deg"]) else float("nan")
    mag = float(r["k2_magnitude"]) if pd.notna(r["k2_magnitude"]) else float("nan")
    neigh = []
    for _, other in targets.loc[~targets["epic_id"].astype(str).eq(epic_id)].iterrows():
        sep = _sep_arcsec(ra, dec, float(other["ra_deg"]), float(other["dec_deg"]))
        if math.isfinite(sep) and sep <= 120:
            omag = float(other["k2_magnitude"]) if pd.notna(other["k2_magnitude"]) else float("nan")
            dmag = omag - mag if math.isfinite(omag) and math.isfinite(mag) else float("nan")
            neigh.append(
                {
                    "epic_id": str(other["epic_id"]),
                    "sep_arcsec": sep,
                    "magnitude": omag,
                    "delta_mag": dmag,
                    "investigation_ids": str(other.get("investigation_ids", "")),
                }
            )
    neigh.sort(key=lambda x: x["sep_arcsec"])
    within60 = [n for n in neigh if n["sep_arcsec"] <= 60]
    risk = "low"
    notes = "No Campaign 5 target within 60 arcsec in local target list."
    if neigh:
        nearest = neigh[0]
        if nearest["sep_arcsec"] <= 30 and math.isfinite(nearest["delta_mag"]) and abs(nearest["delta_mag"]) <= 2.0:
            risk = "high"
        elif within60:
            risk = "medium"
        notes = (
            f"Nearest C5 target {nearest['epic_id']} at {nearest['sep_arcsec']:.1f} arcsec; "
            f"delta_mag={nearest['delta_mag']:.3f}."
        )
    nearest = neigh[0] if neigh else {}
    return {
        "ra_deg": ra if math.isfinite(ra) else "",
        "dec_deg": dec if math.isfinite(dec) else "",
        "k2_magnitude": mag if math.isfinite(mag) else "",
        "investigation_ids": str(r.get("investigation_ids", "")),
        "nearby_sources_within_60_arcsec": len(within60),
        "nearby_sources_within_120_arcsec": len(neigh),
        "nearest_neighbor_epic_id": nearest.get("epic_id", ""),
        "nearest_neighbor_sep_arcsec": nearest.get("sep_arcsec", ""),
        "nearest_neighbor_mag": nearest.get("magnitude", ""),
        "nearest_neighbor_delta_mag": nearest.get("delta_mag", ""),
        "contamination_risk": risk,
        "contamination_notes": notes,
    }


def run_stage_h(selected: pd.DataFrame) -> pd.DataFrame:
    f = _read_csv(STAGE_F_OUTPUT)
    selected_by_epic = {str(r["epic_id"]): r for _, r in selected.iterrows()}
    targets = _target_table()
    rows = []
    for _, fr in f.iterrows():
        epic_id = str(fr["epic_id"])
        pc, pc_hits = _count_confirmed_matches(epic_id)
        cand, cand_hits = _count_candidate_matches(epic_id)
        eb, eb_hits = _count_eb_variable_matches(epic_id)
        contam = _contamination(epic_id, targets)
        if pc > 0:
            label = "known_confirmed_planet"
            status = "recovered_known_confirmed_planet"
            action = "use as benchmark positive / recovered known planet"
            notes = "Matched local confirmed planet catalog."
        elif cand > 0:
            label = "known_unconfirmed_candidate"
            status = "recovered_known_unconfirmed_candidate"
            action = "use as candidate-positive recovery benchmark; literature follow-up / monitor for confirmation"
            notes = "Matched local candidate/ephemeris catalog."
        elif eb > 0:
            label = "known_eb_or_variable"
            status = "recovered_known_eb_or_variable"
            action = "exclude from novel planet queue; use as false-positive benchmark"
            notes = "Matched local EB/variable catalog."
        else:
            label = "new_candidate_needs_external_check"
            status = "new_candidate_needs_external_check"
            action = "external archive/name-resolution check required before claiming new discovery"
            notes = "No local known planet/candidate/EB match; manual external archive check still required."
        sd = selected_by_epic.get(epic_id, {})
        rows.append(
            {
                "epic_id": epic_id,
                "stage_d_rank_order": list(selected_by_epic).index(epic_id) + 1 if epic_id in selected_by_epic else "",
                "best_period_days": fr.get("best_period_days", ""),
                "period_support_count": sd.get("period_support_count", ""),
                "folded_depth_consistency": sd.get("folded_depth_consistency", ""),
                "duration_consistency": sd.get("duration_consistency", ""),
                "stage_f_label": fr.get("stage_f_label", ""),
                "stage_f_reason": fr.get("stage_f_reason", ""),
                "stage_h_label": label,
                "final_candidate_status": status,
                "known_planet_catalog_match_count": pc,
                "known_planet_candidate_match_count": cand,
                "known_eb_or_variable_match_count": eb,
                **contam,
                "recommended_next_action": action,
                "stage_h_notes": notes,
                "local_catalogs_checked": ";".join(str(p) for p in CONFIRMED_PLANET_FILES + CANDIDATE_FILES + EB_VARIABLE_FILES + [TARGET_LIST]),
                "evidence_json": json.dumps(
                    {
                        "known_planet_hits": pc_hits,
                        "known_candidate_hits": cand_hits,
                        "known_eb_variable_hits": eb_hits,
                    },
                    sort_keys=True,
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(STAGE_H_OUTPUT, index=False)
    return out


def write_novel_queue(selected: pd.DataFrame, stage_h: pd.DataFrame) -> pd.DataFrame:
    f = _read_csv(STAGE_F_OUTPUT)
    merged = f.merge(stage_h, on=["epic_id", "best_period_days", "stage_f_label", "stage_f_reason"], how="inner")
    keep = merged.loc[
        merged["stage_f_label"].astype(str).eq("stage_f_planet_like")
        & (pd.to_numeric(merged["known_planet_catalog_match_count"], errors="coerce").fillna(0).eq(0))
        & (pd.to_numeric(merged["known_planet_candidate_match_count"], errors="coerce").fillna(0).eq(0))
        & (pd.to_numeric(merged["known_eb_or_variable_match_count"], errors="coerce").fillna(0).eq(0))
        & ~merged["contamination_risk"].astype(str).eq("high")
    ].copy()
    risk_rank = {"low": 0, "medium": 1, "unknown": 2, "": 2}
    keep["_risk_rank"] = keep["contamination_risk"].map(lambda x: risk_rank.get(str(x), 2))
    keep = keep.sort_values(
        [
            "_risk_rank",
            "period_support_count",
            "folded_depth_consistency",
            "duration_consistency",
            "odd_even_depth_delta_explicit",
        ],
        ascending=[True, False, False, False, True],
    )
    columns = [
        "epic_id",
        "best_period_days",
        "stage_f_label",
        "stage_h_label",
        "final_candidate_status",
        "contamination_risk",
        "known_planet_catalog_match_count",
        "known_planet_candidate_match_count",
        "known_eb_or_variable_match_count",
        "period_support_count",
        "folded_depth_consistency",
        "duration_consistency",
        "primary_depth",
        "radius_ratio_sqrt_depth",
        "secondary_to_primary_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_risk",
        "primary_depth_snr",
        "recommended_next_action",
        "contamination_notes",
        "phase_0_folded_path",
        "phase_05_secondary_check_path",
        "alias_period_comparison_path",
        "odd_even_zoom_path",
        "validation_summary_json_path",
    ]
    out = keep.reindex(columns=columns)
    out.to_csv(NOVEL_QUEUE_OUTPUT, index=False)
    return out


def main() -> None:
    selected = select_next10()
    write_stage_f_input(selected)
    stage_f_result = run_stage_f()
    stage_h = run_stage_h(selected)
    queue = write_novel_queue(selected, stage_h)
    labels = _read_csv(STAGE_F_OUTPUT)["stage_f_label"].value_counts().to_dict()
    print(f"selected_next10={','.join(selected['epic_id'].astype(str).tolist())}")
    print(f"stage_f_rows={stage_f_result['rows_output']}")
    print(f"stage_f_label_counts={labels}")
    print(f"stage_h_rows={len(stage_h)}")
    print(f"novel_queue_rows={len(queue)}")
    print(f"novel_queue_epics={','.join(queue['epic_id'].astype(str).tolist())}")


if __name__ == "__main__":
    main()
