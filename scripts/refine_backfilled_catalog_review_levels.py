from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    ROOT
    / "plots"
    / "k2_batch"
    / "master_vetted_catalog"
    / "cnn_backfill"
    / "master_vetted_catalog_cnn_backfilled.csv"
)
OUT_CSV = (
    ROOT
    / "plots"
    / "k2_batch"
    / "master_vetted_catalog"
    / "cnn_backfill"
    / "master_vetted_catalog_cnn_backfilled_review_level_refined.csv"
)
OUT_SUMMARY = (
    ROOT
    / "plots"
    / "k2_batch"
    / "master_vetted_catalog"
    / "cnn_backfill"
    / "master_vetted_catalog_cnn_backfilled_review_level_refined_summary.txt"
)

PROTECTED_FIELDS = [
    "master_label",
    "decision_authority",
    "cnn_model_path",
    "cnn_score",
    "cnn_score_name",
    "cnn_role",
    "morphology_positive",
    "cnn_policy_version",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def nonblank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def review_level(df: pd.DataFrame) -> pd.Series:
    manual_vetted = df["manual_vetted"].astype(str).str.lower().eq("true")
    has_manual_label = nonblank(df["manual_label"])
    has_stage_g = nonblank(df["stage_g_label"])
    has_stage_f = nonblank(df["stage_f_label"])
    has_autovet = nonblank(df["autovet_label"])

    out = pd.Series("unresolved", index=df.index, dtype="object")
    out.loc[has_autovet] = "auto_only"
    out.loc[has_stage_f & ~has_manual_label & ~has_stage_g] = "stage_f_reviewed"
    out.loc[has_stage_g] = "stage_g_reviewed"
    out.loc[manual_vetted] = "manually_reviewed"
    return out


def assert_protected_fields_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    left = before[["epic_id", *PROTECTED_FIELDS]].copy().sort_values("epic_id").reset_index(drop=True)
    right = after[["epic_id", *PROTECTED_FIELDS]].copy().sort_values("epic_id").reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("Protected master/authority/CNN fields changed during review-level refinement")


def write_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    changed = int(before["review_level"].ne(after["review_level"]).sum())
    stage_f_only = (
        nonblank(after["stage_f_label"])
        & ~nonblank(after["manual_label"])
        & ~nonblank(after["stage_g_label"])
    )
    lines = [
        "Master vetted catalog review-level refinement summary",
        f"generated_at={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        f"source_catalog={rel(INPUT_CSV)}",
        f"refined_catalog={rel(OUT_CSV)}",
        "",
        "Priority applied",
        "1. manually_reviewed when manual_vetted=true",
        "2. stage_g_reviewed when stage_g_label exists",
        "3. stage_f_reviewed when stage_f_label exists and no manual_label or stage_g_label exists",
        "4. auto_only when autovet_label exists",
        "5. unresolved otherwise",
        "",
        f"rows={len(after)}",
        f"review_level_rows_changed={changed}",
        f"stage_f_only_rows={int(stage_f_only.sum())}",
        "",
        "before_counts",
    ]
    lines.extend(f"- {k}={v}" for k, v in before["review_level"].value_counts().items())
    lines.append("")
    lines.append("after_counts")
    lines.extend(f"- {k}={v}" for k, v in after["review_level"].value_counts().items())
    lines.extend(
        [
            "",
            "Safety confirmations",
            "- master_label unchanged.",
            "- decision_authority unchanged.",
            "- CNN score fields unchanged.",
            f"- Original derived input not overwritten: {rel(INPUT_CSV)}",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    before = pd.read_csv(INPUT_CSV)
    after = before.copy()
    after["review_level"] = review_level(after)
    assert_protected_fields_unchanged(before, after)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    after.to_csv(OUT_CSV, index=False)
    write_summary(before, after)

    print(f"review_level_rows_changed={int(before['review_level'].ne(after['review_level']).sum())}")
    print(f"stage_f_reviewed_rows={int(after['review_level'].eq('stage_f_reviewed').sum())}")


if __name__ == "__main__":
    main()
