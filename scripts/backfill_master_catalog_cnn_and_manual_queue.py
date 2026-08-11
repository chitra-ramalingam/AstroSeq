from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MASTER_CATALOG = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "master_vetted_catalog.csv"
MODEL_PATH = ROOT / "models" / "k2_nocrop_flux_seed46_split303.best.keras"
INFER_X = ROOT / "splits" / "infer_c5" / "X_infer.npy"
INFER_META = ROOT / "splits" / "infer_c5" / "meta_infer.parquet"
AUTOVET_REVIEW_QUEUE = ROOT / "plots" / "k2_batch" / "stage_i_autovet_v1_review_queue.csv"
OUT_DIR = ROOT / "plots" / "k2_batch" / "master_vetted_catalog" / "cnn_backfill"

MODEL_PATH_REL = "models/k2_nocrop_flux_seed46_split303.best.keras"
CNN_SCORE_NAME = "transit_morphology_score"
CNN_ROLE = "morphology_scorer_only"
CNN_POLICY_VERSION = "frozen_batch3_transit_morphology_policy_2026-05-15"
NEXT_MANUAL_QUEUE_SIZE = 64

BACKFILL_FIELDS = [
    "cnn_model_path",
    "cnn_score",
    "cnn_score_name",
    "cnn_role",
    "morphology_positive",
    "cnn_policy_version",
]

PROTECTED_FIELDS = [
    "manual_label",
    "manual_reason",
    "manual_reviewer",
    "manual_review_date",
    "manual_next_action",
    "manual_confidence",
    "manual_vetted",
    "master_label",
    "master_reason",
    "master_next_action",
    "decision_authority",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def text_is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def load_missing_targets(catalog: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    missing = text_is_blank(catalog["cnn_score"])
    targets = catalog.loc[missing, ["epic_id"]].copy()
    targets["epic_id"] = targets["epic_id"].astype(str)
    return targets, set(targets["epic_id"])


def infer_missing_scores(target_ids: set[str]) -> pd.DataFrame:
    meta = pd.read_parquet(INFER_META, columns=["star_id", "start", "end", "seg_mid_time"]).reset_index(drop=True)
    meta["star_id"] = meta["star_id"].astype(str)
    mask = meta["star_id"].isin(target_ids).to_numpy()
    matched = set(meta.loc[mask, "star_id"])
    missing_light_curves = sorted(target_ids - matched)

    x_mem = np.load(INFER_X, mmap_mode="r")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    expected = tuple(model.input_shape[1:])
    flux_shape = (int(x_mem.shape[1]), 1)
    if expected != flux_shape:
        raise ValueError(f"Model input {model.input_shape} does not match flux-only tensor shape {flux_shape}")

    idx = np.flatnonzero(mask)
    preds = np.empty(len(idx), dtype="float32")
    batch_size = 2048
    for start in range(0, len(idx), batch_size):
        stop = min(start + batch_size, len(idx))
        x = np.asarray(x_mem[idx[start:stop], :, :1], dtype=np.float32)
        preds[start:stop] = np.asarray(model.predict(x, batch_size=256, verbose=0)).reshape(-1).astype("float32")

    seg = meta.loc[mask].copy().reset_index(drop=True)
    seg["segment_model_score"] = preds

    rows: list[dict[str, Any]] = []
    for epic_id, grp in seg.groupby("star_id", sort=True):
        probs = grp["segment_model_score"].to_numpy(float)
        best = grp.iloc[int(np.argmax(probs))]
        score = float(np.max(probs))
        rows.append(
            {
                "epic_id": epic_id,
                "cnn_model_path": MODEL_PATH_REL,
                "cnn_score": score,
                "cnn_score_name": CNN_SCORE_NAME,
                "cnn_role": CNN_ROLE,
                "morphology_positive": score > 0.5,
                "cnn_policy_version": CNN_POLICY_VERSION,
                "flux_num_segments": int(len(grp)),
                "best_segment_start": int(best["start"]),
                "best_segment_end": int(best["end"]),
                "best_segment_mid_time": float(best["seg_mid_time"]),
            }
        )

    scored = pd.DataFrame(rows).sort_values("epic_id").reset_index(drop=True)
    missing_df = pd.DataFrame({"epic_id": missing_light_curves})
    return scored, missing_df


def backfill_catalog(catalog: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = catalog.copy()
    score_map = scores.set_index("epic_id")
    for field in BACKFILL_FIELDS:
        out[field] = out[field].astype("object")

    missing = text_is_blank(out["cnn_score"])
    for idx, row in out.loc[missing, ["epic_id"]].iterrows():
        epic_id = str(row["epic_id"])
        if epic_id not in score_map.index:
            continue
        for field in BACKFILL_FIELDS:
            value = score_map.at[epic_id, field]
            if field == "morphology_positive":
                value = str(bool(value)).lower()
            out.at[idx, field] = value
    return out


def build_next_manual_queue(backfilled: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    review = pd.read_csv(AUTOVET_REVIEW_QUEUE)
    review["epic_id"] = review["epic_id"].astype(str)
    cat_cols = [
        "epic_id",
        "manual_vetted",
        "review_level",
        "master_label",
        "master_reason",
        "master_next_action",
        "decision_authority",
        "cnn_score",
        "morphology_positive",
    ]
    merged = review.merge(backfilled[cat_cols], on="epic_id", how="left", validate="one_to_one")
    eligible = merged[
        merged["manual_vetted"].astype(str).str.lower().eq("false")
        & merged["review_level"].astype(str).eq("auto_only")
    ].copy()

    label_order = {
        "auto_high_priority_candidate": 0,
        "auto_candidate_with_caveat": 1,
        "auto_hold_needs_review": 2,
    }
    eligible["_label_order"] = eligible["autovet_label"].map(label_order).fillna(99)
    eligible = eligible.sort_values(
        ["_label_order", "review_priority_score", "autovet_rank_score", "epic_id"],
        ascending=[True, False, False, True],
    ).head(NEXT_MANUAL_QUEUE_SIZE)
    eligible = eligible.drop(columns=["_label_order"]).reset_index(drop=True)
    eligible.insert(0, "queue_rank", np.arange(1, len(eligible) + 1))
    eligible["queue_goal"] = "manual_vetted_count_36_to_100"
    eligible["queue_policy"] = "existing_stage_i_review_queue_order_excluding_already_manual_vetted"
    eligible["cnn_score_name"] = CNN_SCORE_NAME
    eligible["cnn_role"] = CNN_ROLE
    eligible["cnn_policy_version"] = CNN_POLICY_VERSION
    return eligible


def assert_protected_fields_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    left = before[["epic_id", *PROTECTED_FIELDS]].copy().sort_values("epic_id").reset_index(drop=True)
    right = after[["epic_id", *PROTECTED_FIELDS]].copy().sort_values("epic_id").reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("Protected manual/master/authority fields changed during CNN backfill")


def write_summary(
    catalog: pd.DataFrame,
    backfilled: pd.DataFrame,
    scores: pd.DataFrame,
    missing_light_curves: pd.DataFrame,
    queue: pd.DataFrame,
) -> None:
    before_missing = int(text_is_blank(catalog["cnn_score"]).sum())
    after_missing = int(text_is_blank(backfilled["cnn_score"]).sum())
    before_manual = int(catalog["manual_vetted"].astype(str).str.lower().eq("true").sum())
    lines = [
        "Master vetted catalog CNN backfill summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"source_catalog={rel(MASTER_CATALOG)}",
        f"frozen_model={MODEL_PATH_REL}",
        f"cnn_score_name={CNN_SCORE_NAME}",
        f"cnn_role={CNN_ROLE}",
        f"cnn_policy_version={CNN_POLICY_VERSION}",
        "",
        f"catalog_rows={len(catalog)}",
        f"cnn_missing_before={before_missing}",
        f"cnn_backfilled_rows={len(scores)}",
        f"cnn_missing_light_curve_unavailable={len(missing_light_curves)}",
        f"cnn_missing_after={after_missing}",
        f"morphology_positive_backfilled_rows={int(scores['morphology_positive'].sum()) if len(scores) else 0}",
        "",
        "Safety confirmations",
        "- No retraining was run.",
        "- The frozen CNN was used as morphology_scorer_only.",
        "- cnn_score is transit_morphology_score / dip_likeness only.",
        "- cnn_score > 0.5 means morphology_positive only.",
        "- Protected manual labels, master_label, and decision_authority fields were unchanged.",
        f"- Original catalog not overwritten: {rel(MASTER_CATALOG)}",
        "- final_candidate_master_ledger.csv was not overwritten.",
        "",
        "Next manual-vetting queue",
        f"- manual_vetted_before={before_manual}",
        f"- requested_increment={NEXT_MANUAL_QUEUE_SIZE}",
        f"- target_manual_vetted_after_completion={before_manual + NEXT_MANUAL_QUEUE_SIZE}",
        f"- queue_rows={len(queue)}",
        "- queue policy: preserve the existing Stage I review queue ordering, remove already manual-vetted rows, keep auto_only rows only.",
    ]
    (OUT_DIR / "cnn_backfill_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [MASTER_CATALOG, MODEL_PATH, INFER_X, INFER_META, AUTOVET_REVIEW_QUEUE]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(MASTER_CATALOG)
    catalog["epic_id"] = catalog["epic_id"].astype(str)
    _, target_ids = load_missing_targets(catalog)
    scores, missing_light_curves = infer_missing_scores(target_ids)

    backfilled = backfill_catalog(catalog, scores)
    assert_protected_fields_unchanged(catalog, backfilled)
    queue = build_next_manual_queue(backfilled, scores)

    scores.to_csv(OUT_DIR / "frozen_cnn_morphology_backfill_inference.csv", index=False)
    missing_light_curves.to_csv(OUT_DIR / "cnn_backfill_missing_light_curves.csv", index=False)
    backfilled.to_csv(OUT_DIR / "master_vetted_catalog_cnn_backfilled.csv", index=False)
    queue.to_csv(OUT_DIR / "manual_vetting_priority_queue_next64.csv", index=False)
    write_summary(catalog, backfilled, scores, missing_light_curves, queue)

    print(f"cnn_backfilled_rows={len(scores)}")
    print(f"cnn_missing_light_curve_unavailable={len(missing_light_curves)}")
    print(f"manual_queue_rows={len(queue)}")


if __name__ == "__main__":
    main()
