from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "k2_nocrop_flux_seed46_split303.best.keras"
STAGE_D_CSV = ROOT / "k2_stage_d_tier_a_results.csv"
LABELS_CSV = ROOT / "training_labels_v2.csv"
INFER_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"
INFER_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"
STAGE_F_CSVS = [
    ROOT / "k2_stage_f_followup_validation.csv",
    ROOT / "k2_stage_f_next10_validation.csv",
    ROOT / "k2_stage_f_next10_batch2_validation.csv",
]

OUT_ALL = ROOT / "k2_hybrid_candidate_score_v1.csv"
OUT_TOP50 = ROOT / "k2_hybrid_top50_unresolved.csv"


def as_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def score_up(s: pd.Series, good_at: float) -> pd.Series:
    return pd.Series(clip01(as_num(s) / good_at), index=s.index)


def score_down(s: pd.Series, bad_at: float) -> pd.Series:
    return pd.Series(1.0 - clip01(as_num(s) / bad_at), index=s.index)


def penalty_up(s: pd.Series, bad_at: float) -> pd.Series:
    return pd.Series(clip01(as_num(s) / bad_at), index=s.index)


def load_flux_scores(epic_ids: set[str]) -> pd.DataFrame:
    meta = pd.read_parquet(INFER_META).reset_index(drop=True)
    mask = meta["star_id"].astype(str).isin(epic_ids).to_numpy()
    idx = np.flatnonzero(mask)
    matched = set(meta.loc[mask, "star_id"].astype(str))
    missing = sorted(epic_ids - matched)
    if missing:
        raise ValueError(f"Missing Stage D EPICs in {INFER_META}: {missing}")

    x_mem = np.load(INFER_X, mmap_mode="r")
    model = tf.keras.models.load_model(MODEL_PATH)
    x = np.asarray(x_mem[idx, :, :1], dtype=np.float32)
    p = model.predict(x, batch_size=256, verbose=0).reshape(-1).astype(float)

    seg = meta.loc[mask].copy().reset_index(drop=True)
    seg["flux_segment_probability"] = p

    rows: list[dict[str, Any]] = []
    for epic_id, grp in seg.groupby("star_id", sort=True):
        probs = grp["flux_segment_probability"].to_numpy(float)
        best = grp.iloc[int(np.argmax(probs))]
        rows.append(
            {
                "epic_id": epic_id,
                "flux_num_segments": int(len(grp)),
                "flux_p_science_like": float(np.max(probs)),
                "flux_p_top3_mean": float(np.mean(np.sort(probs)[-min(3, len(probs)) :])),
                "flux_p_top10_mean": float(np.mean(np.sort(probs)[-min(10, len(probs)) :])),
                "flux_best_segment_start": int(best["start"]),
                "flux_best_segment_end": int(best["end"]),
                "flux_best_segment_mid_time": float(best["seg_mid_time"]),
            }
        )
    return pd.DataFrame(rows)


def load_stage_f() -> pd.DataFrame:
    frames = []
    for csv_path in STAGE_F_CSVS:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["stage_f_source_csv"] = csv_path.name
            frames.append(df)
    if not frames:
        return pd.DataFrame({"epic_id": []})

    raw = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "primary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "secondary_depth_snr",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "half_period_support_count",
        "double_period_support_count",
    ]
    for col in numeric_cols:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    label_score = {
        "stage_f_planet_like": 1.0,
        "stage_f_hold": 0.55,
        "stage_f_reject": 0.10,
    }
    raw["stage_f_label_score"] = raw.get("stage_f_label", "").map(label_score).fillna(0.5)
    raw["stage_f_depth_snr_score"] = clip01(np.log1p(raw["primary_depth_snr"].fillna(0.0)) / np.log1p(20.0))
    raw["stage_f_secondary_clean"] = 1.0 - clip01(raw["secondary_to_primary_depth_ratio"].fillna(0.0) / 0.50)
    raw["stage_f_odd_even_clean"] = 1.0 - clip01(raw["odd_even_depth_delta_explicit"].fillna(0.0) / 0.50)
    raw["stage_f_oot_clean"] = 1.0 - clip01(raw["oot_variability_to_depth"].fillna(0.0) / 2.00)
    raw["stage_f_alias_clean"] = 1.0 - clip01((raw["alias_best_support_ratio"].fillna(0.0) - 0.35) / 0.50)
    raw["stage_f_row_quality_score"] = (
        0.25 * raw["stage_f_depth_snr_score"]
        + 0.25 * raw["stage_f_secondary_clean"]
        + 0.20 * raw["stage_f_odd_even_clean"]
        + 0.15 * raw["stage_f_oot_clean"]
        + 0.10 * raw["stage_f_alias_clean"]
        + 0.05 * raw["stage_f_label_score"]
    )
    raw["stage_f_odd_even_penalty"] = penalty_up(raw["odd_even_depth_delta_explicit"], 0.50)
    raw["stage_f_secondary_penalty"] = np.maximum(
        penalty_up(raw["secondary_to_primary_depth_ratio"], 0.50),
        penalty_up(raw["secondary_depth_snr"].fillna(0.0), 7.0),
    )
    raw["stage_f_oot_penalty"] = penalty_up(raw["oot_variability_to_depth"], 2.00)
    raw["stage_f_alias_penalty"] = clip01((raw["alias_best_support_ratio"].fillna(0.0) - 0.35) / 0.50)

    agg_rows = []
    for epic_id, grp in raw.groupby("epic_id", sort=True):
        worst = grp.loc[grp["stage_f_row_quality_score"].idxmin()]
        agg_rows.append(
            {
                "epic_id": epic_id,
                "stage_f_available": True,
                "stage_f_quality_score": float(grp["stage_f_row_quality_score"].min()),
                "stage_f_best_quality_score": float(grp["stage_f_row_quality_score"].max()),
                "stage_f_rows": int(len(grp)),
                "stage_f_conservative_label": worst.get("stage_f_label", ""),
                "stage_f_source_csv": worst.get("stage_f_source_csv", ""),
                "primary_depth_snr": float(grp["primary_depth_snr"].max(skipna=True)),
                "secondary_to_primary_depth_ratio": float(grp["secondary_to_primary_depth_ratio"].max(skipna=True)),
                "secondary_depth_snr": float(grp["secondary_depth_snr"].max(skipna=True)),
                "odd_even_depth_delta_explicit": float(grp["odd_even_depth_delta_explicit"].max(skipna=True)),
                "oot_variability_to_depth": float(grp["oot_variability_to_depth"].max(skipna=True)),
                "alias_best_support_ratio": float(grp["alias_best_support_ratio"].max(skipna=True)),
                "stage_f_odd_even_penalty": float(grp["stage_f_odd_even_penalty"].max()),
                "stage_f_secondary_penalty": float(grp["stage_f_secondary_penalty"].max()),
                "stage_f_oot_penalty": float(grp["stage_f_oot_penalty"].max()),
                "stage_f_alias_penalty": float(grp["stage_f_alias_penalty"].max()),
            }
        )
    return pd.DataFrame(agg_rows)


def manual_penalty(label: str | float, status: str | float, review_bin: str | float) -> float:
    label_s = "" if pd.isna(label) else str(label)
    status_s = "" if pd.isna(status) else str(status)
    review_s = "" if pd.isna(review_bin) else str(review_bin)
    if label_s == "noise_or_artifact":
        return 1.0
    if label_s == "binary_system":
        return 0.9
    if label_s == "uncertain_hold":
        return 0.45
    if label_s == "candidate_like":
        return 0.05
    if label_s == "planet_like":
        return 0.0
    if status_s.startswith("reject_") or review_s.startswith("manual_false_positive"):
        return 1.0
    return 0.0


def main() -> None:
    stage_d = pd.read_csv(STAGE_D_CSV)
    labels = pd.read_csv(LABELS_CSV)
    flux = load_flux_scores(set(stage_d["epic_id"].astype(str)))
    stage_f = load_stage_f()

    df = stage_d.merge(flux, on="epic_id", how="left", validate="one_to_one")
    df = df.merge(stage_f, on="epic_id", how="left")
    df = df.merge(
        labels[
            [
                "epic_id",
                "training_label_v2",
                "final_candidate_status",
                "review_bin",
                "training_label_rule",
            ]
        ],
        on="epic_id",
        how="left",
        validate="one_to_one",
    )

    df["stage_f_available"] = df["stage_f_available"].fillna(False).astype(bool)
    df["stage_f_quality_score"] = df["stage_f_quality_score"].fillna(0.5)
    for col in [
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
    ]:
        df[col] = df[col].fillna(0.0)

    stage_d_positive_terms = pd.DataFrame(
        {
            "n_events_long_good": score_up(df["n_events_long_good"], 15.0),
            "n_events_ge_10_cadences": score_up(df["n_events_ge_10_cadences"], 20.0),
            "max_shape_score": score_up(df["max_shape_score"], 0.88),
            "period_support_count": score_up(df["period_support_count"], 20.0),
            "event_family_count": score_up(df["event_family_count"], 8.0),
            "folded_depth_consistency": score_up(df["folded_depth_consistency"], 1.0),
            "duration_consistency": score_up(df["duration_consistency"], 1.0),
            "coverage_rate": score_up(df["coverage_rate"], 1.0),
            "hit_rate_snr": score_up(df["hit_rate_snr"], 1.0),
            "hit_rate_shape": score_up(df["hit_rate_shape"], 1.0),
            "soft_hit_rate": score_up(df["soft_hit_rate"], 1.0),
        }
    )
    df["stage_d_positive_score"] = stage_d_positive_terms.mean(axis=1)
    df["stage_d_spike_clean"] = score_down(df["spike_fraction_2cadence"], 0.35)
    df["stage_d_odd_even_clean"] = score_down(df["odd_even_depth_delta"], 0.50)
    df["stage_d_quality_score"] = clip01(
        0.82 * df["stage_d_positive_score"]
        + 0.09 * df["stage_d_spike_clean"]
        + 0.09 * df["stage_d_odd_even_clean"]
        - np.where(df["stage_d_label"].eq("hold_deeper_eval"), 0.12, 0.0)
    )

    df["manual_label_penalty"] = [
        manual_penalty(label, status, review_bin)
        for label, status, review_bin in zip(
            df["training_label_v2"],
            df["final_candidate_status"],
            df["review_bin"],
        )
    ]
    df["stage_d_odd_even_penalty"] = penalty_up(df["odd_even_depth_delta"], 0.50)
    df["false_positive_penalty"] = clip01(
        0.22 * np.maximum(df["stage_d_odd_even_penalty"], df["stage_f_odd_even_penalty"])
        + 0.20 * df["stage_f_secondary_penalty"]
        + 0.18 * df["stage_f_oot_penalty"]
        + 0.15 * df["stage_f_alias_penalty"]
        + 0.25 * df["manual_label_penalty"]
    )
    df["flux_recall_filter_pass"] = df["flux_p_science_like"] >= 0.5
    df["candidate_score"] = clip01(
        0.35 * df["flux_p_science_like"]
        + 0.25 * df["stage_d_quality_score"]
        + 0.25 * df["stage_f_quality_score"]
        - 0.15 * df["false_positive_penalty"]
    )
    df["is_labeled_in_ledger"] = df["training_label_v2"].notna()
    df["is_unresolved"] = ~df["is_labeled_in_ledger"]

    df = df.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    df.insert(0, "hybrid_rank", np.arange(1, len(df) + 1))

    preferred_cols = [
        "hybrid_rank",
        "epic_id",
        "candidate_score",
        "flux_p_science_like",
        "flux_recall_filter_pass",
        "stage_d_quality_score",
        "stage_f_quality_score",
        "false_positive_penalty",
        "manual_label_penalty",
        "is_unresolved",
        "training_label_v2",
        "final_candidate_status",
        "review_bin",
        "stage_d_label",
        "stage_f_available",
        "stage_f_conservative_label",
        "best_period_days",
        "period_support_count",
        "event_family_count",
        "folded_depth_consistency",
        "duration_consistency",
        "odd_even_depth_delta",
        "primary_depth_snr",
        "secondary_to_primary_depth_ratio",
        "odd_even_depth_delta_explicit",
        "oot_variability_to_depth",
        "alias_best_support_ratio",
        "stage_f_odd_even_penalty",
        "stage_f_secondary_penalty",
        "stage_f_oot_penalty",
        "stage_f_alias_penalty",
        "flux_num_segments",
        "flux_p_top3_mean",
        "flux_p_top10_mean",
        "flux_best_segment_start",
        "flux_best_segment_end",
        "flux_best_segment_mid_time",
        "events_csv",
        "stage_f_source_csv",
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    df[[c for c in preferred_cols if c in df.columns] + remaining].to_csv(OUT_ALL, index=False)

    top50 = df[df["is_unresolved"] & df["flux_recall_filter_pass"]].head(50).copy()
    top50.insert(0, "unresolved_hybrid_rank", np.arange(1, len(top50) + 1))
    top50[[c for c in ["unresolved_hybrid_rank"] + preferred_cols if c in top50.columns] + remaining].to_csv(
        OUT_TOP50,
        index=False,
    )

    print(f"wrote {OUT_ALL} rows={len(df)}")
    print(f"wrote {OUT_TOP50} rows={len(top50)}")
    print(
        df[
            [
                "hybrid_rank",
                "epic_id",
                "candidate_score",
                "flux_p_science_like",
                "stage_d_quality_score",
                "stage_f_quality_score",
                "false_positive_penalty",
                "training_label_v2",
                "is_unresolved",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
