from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class K2StageRAutoLabeler:
    POLICY_VERSION = "stage_r_manual_calibration_v2"

    LABEL_PROMOTE = "promote_to_deeper_eval"
    LABEL_HOLD = "hold_for_review"
    LABEL_REJECT = "reject_as_noise_or_artifact"

    PROMOTE_DURATION_CADENCES = 10.0
    PROMOTE_SHAPE_SCORE = 0.74
    HOLD_DURATION_CADENCES = 5.0
    BORDERLINE_DURATION_CADENCES = 10.0
    BORDERLINE_SHAPE_SCORE = 0.70
    SHAPE_GE_071 = 0.71
    SHAPE_GE_074 = 0.74
    SPIKE_2CADENCE_MAX_PROMOTE_FRACTION = 0.50
    SPIKE_2CADENCE_DOMINANT_FRACTION = 0.50
    SPIKE_3CADENCE_DOMINANT_FRACTION = 0.70
    DEPTH_RATIO_INCONSISTENT = 10.0
    SINGLE_STRONG_DEPTH_RATIO_MAX = 15.0
    SINGLE_STRONG_NON_SPIKE_DEPTH_RATIO_MAX = 15.0
    SINGLE_STRONG_MANY_EVENTS_MAX = 35

    OUTPUT_COLUMNS = [
        "epic_id",
        "stage_r_label",
        "stage_r_reason",
        "n_events_total",
        "n_events_ge_5_cadences",
        "n_events_ge_10_cadences",
        "n_events_shape_ge_071",
        "n_events_shape_ge_074",
        "n_events_long_good",
        "max_shape_score",
        "median_shape_score",
        "max_duration_cadences",
        "median_duration_cadences",
        "depth_min",
        "depth_max",
        "depth_ratio",
        "spike_fraction_2cadence",
        "stage_r_policy_version",
        "stage_r_needs_manual_review",
        "stage_r_debug_json",
    ]

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=(
                "Apply Stage R manual-calibration labels to event-level K2 CSV rows. "
                "This is CSV-in, CSV-out only; it does not download data or run detection."
            )
        )
        p.add_argument("--input", "--input-csv", dest="input_csv", type=Path, required=True)
        p.add_argument("--output", "--output-csv", dest="output_csv", type=Path, required=True)
        return p

    @classmethod
    def run_cli(cls, argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        args = cls.build_parser().parse_args(list(argv) if argv is not None else None)
        return cls().run(input_csv=Path(args.input_csv), output_csv=Path(args.output_csv))

    @classmethod
    def run(cls, *, input_csv: Path, output_csv: Path) -> Dict[str, Any]:
        df = pd.read_csv(input_csv)
        label_df = cls.label_events(df)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        label_df.to_csv(output_csv, index=False)
        return {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "rows_input": int(len(df)),
            "rows_output": int(len(label_df)),
            "label_counts": label_df["stage_r_label"].value_counts(dropna=False).to_dict(),
        }

    @classmethod
    def label_events(cls, df: pd.DataFrame) -> pd.DataFrame:
        required = {"query", "duration_cadences", "depth", "shape_score"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required Stage R event columns: {', '.join(missing)}")

        work = df.copy()
        work["_stage_r_epic_id"] = cls._resolve_epic_ids(work)
        if work["_stage_r_epic_id"].isna().any() or (work["_stage_r_epic_id"].astype(str).str.len() == 0).any():
            bad_queries = work.loc[
                work["_stage_r_epic_id"].isna() | (work["_stage_r_epic_id"].astype(str).str.len() == 0),
                "query",
            ].head(5)
            raise ValueError(f"Unable to resolve EPIC id for Stage R rows with queries: {bad_queries.tolist()}")

        rows: List[Dict[str, Any]] = []
        for epic_id, group in work.groupby("_stage_r_epic_id", sort=True, dropna=False):
            rows.append(cls._label_group(str(epic_id), group))

        return pd.DataFrame(rows, columns=cls.OUTPUT_COLUMNS)

    @classmethod
    def _resolve_epic_ids(cls, df: pd.DataFrame) -> pd.Series:
        if "epic_id" in df.columns:
            from_epic = df["epic_id"].map(cls._normalize_epic_id)
        else:
            from_epic = pd.Series([None] * len(df), index=df.index, dtype=object)
        from_query = df["query"].map(cls._parse_epic_id)
        return from_epic.where(from_epic.notna() & (from_epic.astype(str).str.len() > 0), from_query)

    @staticmethod
    def _normalize_epic_id(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = K2StageRAutoLabeler._parse_epic_id(text)
        if parsed:
            return parsed
        if re.fullmatch(r"\d{6,12}", text):
            return f"EPIC_{text}"
        return text

    @staticmethod
    def _parse_epic_id(query: Any) -> Optional[str]:
        if query is None or (isinstance(query, float) and math.isnan(query)):
            return None
        match = re.search(r"\bEPIC[\s_-]*(\d{6,12})\b", str(query), flags=re.IGNORECASE)
        if not match:
            return None
        return f"EPIC_{match.group(1)}"

    @classmethod
    def _label_group(cls, epic_id: str, group: pd.DataFrame) -> Dict[str, Any]:
        duration = pd.to_numeric(group.get("duration_cadences"), errors="coerce")
        shape = pd.to_numeric(group.get("shape_score"), errors="coerce")
        depth = pd.to_numeric(group.get("depth"), errors="coerce")

        valid_event_mask = duration.notna() | shape.notna()
        n_events_total = int(len(group))
        n_events_ge_5 = int((duration >= cls.HOLD_DURATION_CADENCES).sum())
        n_events_ge_10 = int((duration >= cls.PROMOTE_DURATION_CADENCES).sum())
        n_shape_ge_071 = int((shape >= cls.SHAPE_GE_071).sum())
        n_shape_ge_074 = int((shape >= cls.SHAPE_GE_074).sum())
        strong_long_mask = (duration >= cls.PROMOTE_DURATION_CADENCES) & (shape >= cls.PROMOTE_SHAPE_SCORE)
        strong_event_mask = (duration >= cls.HOLD_DURATION_CADENCES) & (shape >= cls.PROMOTE_SHAPE_SCORE)
        borderline_long_mask = (duration >= cls.BORDERLINE_DURATION_CADENCES) & (shape >= cls.BORDERLINE_SHAPE_SCORE)
        n_long_good = int(strong_long_mask.sum())
        n_strong_events = int(strong_event_mask.sum())
        n_borderline_long = int(borderline_long_mask.sum())

        spike_2_count = int((duration <= 2.0).sum())
        spike_3_count = int((duration <= 3.0).sum())
        spike_fraction_2 = cls._fraction(spike_2_count, n_events_total)
        spike_fraction_3 = cls._fraction(spike_3_count, n_events_total)

        max_shape = cls._clean_float(shape.max(skipna=True))
        median_shape = cls._clean_float(shape.median(skipna=True))
        max_duration = cls._clean_float(duration.max(skipna=True))
        median_duration = cls._clean_float(duration.median(skipna=True))
        depth_min = cls._clean_float(depth.min(skipna=True))
        depth_max = cls._clean_float(depth.max(skipna=True))
        depth_ratio = cls._depth_ratio(depth)
        non_spike_depth_ratio = cls._depth_ratio(depth[duration > 2.0])

        spike_2_dominant = bool(spike_fraction_2 > cls.SPIKE_2CADENCE_DOMINANT_FRACTION)
        spike_3_dominant = bool(spike_fraction_3 > cls.SPIKE_3CADENCE_DOMINANT_FRACTION)
        spike_not_dominant_for_promote = bool(spike_fraction_2 <= cls.SPIKE_2CADENCE_MAX_PROMOTE_FRACTION)
        depth_highly_inconsistent = bool(
            not math.isnan(depth_ratio) and depth_ratio >= cls.DEPTH_RATIO_INCONSISTENT
        )
        single_strong_depth_ok = bool(
            (not math.isnan(depth_ratio) and depth_ratio <= cls.SINGLE_STRONG_DEPTH_RATIO_MAX)
            or (
                not math.isnan(non_spike_depth_ratio)
                and non_spike_depth_ratio <= cls.SINGLE_STRONG_NON_SPIKE_DEPTH_RATIO_MAX
            )
        )
        single_strong_spike_ok = bool(
            spike_fraction_2 < cls.SPIKE_2CADENCE_DOMINANT_FRACTION or n_events_ge_5 >= 2
        )
        single_strong_field_size_ok = bool(
            n_events_total <= cls.SINGLE_STRONG_MANY_EVENTS_MAX or n_long_good >= 2
        )
        single_strong_hold_allowed = bool(
            n_strong_events == 1
            and single_strong_spike_ok
            and single_strong_depth_ok
            and single_strong_field_size_ok
        )
        has_coherent_long_family = bool(n_borderline_long >= 2)

        promote = bool(n_long_good >= 2 and spike_not_dominant_for_promote and (max_duration >= 10.0))
        hold_single_strong = bool(single_strong_hold_allowed)
        hold_borderline_family = bool((not promote) and n_borderline_long >= 2)

        if promote:
            label = cls.LABEL_PROMOTE
            reason = (
                f"promote: {n_long_good} events have duration_cadences>=10 and shape_score>=0.74; "
                f"max_duration_cadences={max_duration:g}; spike_fraction_2cadence={spike_fraction_2:.3f}"
            )
        elif hold_single_strong:
            label = cls.LABEL_HOLD
            reason = (
                f"hold: exactly 1 strong event has duration_cadences>=5 and shape_score>=0.74; "
                f"single-strong guardrails passed; promote long-good count={n_long_good}"
            )
        elif hold_borderline_family:
            label = cls.LABEL_HOLD
            reason = (
                f"hold: borderline long-family case with {n_borderline_long} events "
                "duration_cadences>=10 and shape_score>=0.70, but promote criteria not met"
            )
        else:
            label = cls.LABEL_REJECT
            reason = cls._reject_reason(
                spike_2_dominant=spike_2_dominant,
                spike_3_dominant=spike_3_dominant,
                has_coherent_long_family=has_coherent_long_family,
                depth_highly_inconsistent=depth_highly_inconsistent,
                n_events_total=n_events_total,
                n_borderline_long=n_borderline_long,
                depth_ratio=depth_ratio,
            )

        debug = {
            "policy_version": cls.POLICY_VERSION,
            "thresholds": {
                "promote_duration_cadences": cls.PROMOTE_DURATION_CADENCES,
                "promote_shape_score": cls.PROMOTE_SHAPE_SCORE,
                "hold_duration_cadences": cls.HOLD_DURATION_CADENCES,
                "borderline_duration_cadences": cls.BORDERLINE_DURATION_CADENCES,
                "borderline_shape_score": cls.BORDERLINE_SHAPE_SCORE,
                "spike_2cadence_max_promote_fraction": cls.SPIKE_2CADENCE_MAX_PROMOTE_FRACTION,
                "spike_2cadence_dominant_fraction": cls.SPIKE_2CADENCE_DOMINANT_FRACTION,
                "spike_3cadence_dominant_fraction": cls.SPIKE_3CADENCE_DOMINANT_FRACTION,
                "depth_ratio_inconsistent": cls.DEPTH_RATIO_INCONSISTENT,
                "single_strong_depth_ratio_max": cls.SINGLE_STRONG_DEPTH_RATIO_MAX,
                "single_strong_non_spike_depth_ratio_max": cls.SINGLE_STRONG_NON_SPIKE_DEPTH_RATIO_MAX,
                "single_strong_many_events_max": cls.SINGLE_STRONG_MANY_EVENTS_MAX,
            },
            "counts": {
                "n_events_total": n_events_total,
                "n_events_with_duration_or_shape": int(valid_event_mask.sum()),
                "n_events_ge_5_cadences": n_events_ge_5,
                "n_events_ge_10_cadences": n_events_ge_10,
                "n_events_shape_ge_071": n_shape_ge_071,
                "n_events_shape_ge_074": n_shape_ge_074,
                "n_events_long_good": n_long_good,
                "n_events_strong_ge5_shape_ge074": n_strong_events,
                "n_events_borderline_long_ge10_shape_ge070": n_borderline_long,
                "n_events_spike_le_2_cadences": spike_2_count,
                "n_events_spike_le_3_cadences": spike_3_count,
            },
            "flags": {
                "promote_rule_passed": promote,
                "hold_single_strong_rule_passed": hold_single_strong,
                "hold_borderline_long_family_rule_passed": hold_borderline_family,
                "single_strong_spike_ok": single_strong_spike_ok,
                "single_strong_depth_ok": single_strong_depth_ok,
                "single_strong_field_size_ok": single_strong_field_size_ok,
                "single_strong_hold_allowed": single_strong_hold_allowed,
                "spike_2cadence_dominant": spike_2_dominant,
                "spike_3cadence_dominant": spike_3_dominant,
                "has_coherent_long_family": has_coherent_long_family,
                "depth_highly_inconsistent": depth_highly_inconsistent,
            },
            "metrics": {
                "max_shape_score": max_shape,
                "median_shape_score": median_shape,
                "max_duration_cadences": max_duration,
                "median_duration_cadences": median_duration,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "depth_ratio": depth_ratio,
                "non_spike_depth_ratio": non_spike_depth_ratio,
                "spike_fraction_2cadence": spike_fraction_2,
                "spike_fraction_3cadence": spike_fraction_3,
            },
        }

        return {
            "epic_id": epic_id,
            "stage_r_label": label,
            "stage_r_reason": reason,
            "n_events_total": n_events_total,
            "n_events_ge_5_cadences": n_events_ge_5,
            "n_events_ge_10_cadences": n_events_ge_10,
            "n_events_shape_ge_071": n_shape_ge_071,
            "n_events_shape_ge_074": n_shape_ge_074,
            "n_events_long_good": n_long_good,
            "max_shape_score": max_shape,
            "median_shape_score": median_shape,
            "max_duration_cadences": max_duration,
            "median_duration_cadences": median_duration,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_ratio": depth_ratio,
            "spike_fraction_2cadence": spike_fraction_2,
            "stage_r_policy_version": cls.POLICY_VERSION,
            "stage_r_needs_manual_review": label == cls.LABEL_HOLD,
            "stage_r_debug_json": json.dumps(debug, sort_keys=True, separators=(",", ":")),
        }

    @staticmethod
    def _reject_reason(
        *,
        spike_2_dominant: bool,
        spike_3_dominant: bool,
        has_coherent_long_family: bool,
        depth_highly_inconsistent: bool,
        n_events_total: int,
        n_borderline_long: int,
        depth_ratio: float,
    ) -> str:
        reasons: List[str] = []
        if spike_2_dominant or spike_3_dominant:
            reasons.append("dominated by 2-3 cadence spike events")
        if not has_coherent_long_family:
            reasons.append(
                f"no coherent long-duration event family ({n_borderline_long} events meet duration>=10 and shape>=0.70)"
            )
        if depth_highly_inconsistent:
            reasons.append(f"depth/duration are highly inconsistent (depth_ratio={depth_ratio:.3g})")
        if not reasons:
            reasons.append("fails promote and hold rules")
        return f"reject: {'; '.join(reasons)}; n_events_total={n_events_total}"

    @staticmethod
    def _fraction(count: int, total: int) -> float:
        if total <= 0:
            return float("nan")
        return float(count) / float(total)

    @staticmethod
    def _clean_float(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if math.isfinite(out) else float("nan")

    @classmethod
    def _depth_ratio(cls, depth: pd.Series) -> float:
        finite_abs = depth.dropna().astype(float).abs()
        finite_abs = finite_abs[np.isfinite(finite_abs)]
        finite_abs = finite_abs[finite_abs > 0.0]
        if finite_abs.empty:
            return float("nan")
        min_depth = float(finite_abs.min())
        if min_depth <= 0.0:
            return float("nan")
        return float(finite_abs.max()) / min_depth
