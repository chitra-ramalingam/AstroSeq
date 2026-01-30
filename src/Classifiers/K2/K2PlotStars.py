from __future__ import annotations
import re

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd


class K2PlotStars:
    """Rank candidate stars using ensemble predictions and export the top subsets."""

    def __init__(
        self,
        source_csv: Union[str, Path] = "candidates_c5_top3_windows_per_star.csv",
        output_csv: Union[str, Path] = "c5_star_candidates_top500.csv",
        top_k: int = 3,
        top_rows_to_export: int = 500,
        rows_to_print: int = 20,
        plot_dir: Union[str, Path] = "plots",
    ):
        self.source_csv = Path(source_csv)
        self.output_csv = Path(output_csv)
        self.top_k = top_k
        self.top_rows_to_export = top_rows_to_export
        self.rows_to_print = rows_to_print
        self.plot_dir = Path(plot_dir)
        self._cached_source_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _topk_stats(series: pd.Series, k: int) -> pd.Series:
        """Compute summary statistics for the k highest ensemble scores per star."""
        values = np.sort(series.to_numpy())[::-1]
        values = np.pad(values, (0, max(0, k - len(values))), constant_values=np.nan)[:k]
        p1, p2, p3 = values
        score = np.nanmean(values)
        gap12 = p1 - p2 if np.isfinite(p2) else np.nan
        return pd.Series({"star_score": score, "p1": p1, "p2": p2, "p3": p3, "gap12": gap12})

    def create_ranking(self) -> pd.DataFrame:
        """Build ranked star statistics from the source CSV.

        star_score = mean(top_k p_ens) per star (anti-haunting).
        Also stores best_p_ens and p1/p2/p3 as evidence.
        """
        df = pd.read_csv(self.source_csv)

        # Ensure top windows come first (important if source contains all windows per star)
        df_sorted = df.sort_values("p_ens", ascending=False)

        k = int(self.top_k)

        # --- star_score: mean of top-k window scores per star ---
        star_score = (
            df_sorted.groupby("star_id")["p_ens"]
            .apply(lambda s: float(np.mean(s.head(k))))
            .reset_index()
            .rename(columns={"p_ens": "star_score"})
        )

        # --- evidence: best window score per star ---
        best_per_star = (
            df_sorted.groupby("star_id", as_index=False)["p_ens"]
            .first()
            .rename(columns={"p_ens": "best_p_ens"})
        )

        # --- evidence: p1/p2/p3 and gap ---
        def _p123(s):
            v = s.head(max(k, 3)).to_numpy()
            # v is already sorted by p_ens desc because df_sorted is sorted before groupby
            p1 = float(v[0]) if len(v) > 0 else float("nan")
            p2 = float(v[1]) if len(v) > 1 else float("nan")
            p3 = float(v[2]) if len(v) > 2 else float("nan")
            gap12 = p1 - p2 if np.isfinite(p1) and np.isfinite(p2) else float("nan")
            return pd.Series({"p1": p1, "p2": p2, "p3": p3, "gap12": gap12})

        evidence = (
            df_sorted.groupby("star_id")["p_ens"]
            .apply(_p123)
            .reset_index()
        )

        star_df = star_score.merge(best_per_star, on="star_id", how="left").merge(evidence, on="star_id", how="left")

        # Rank by star_score (NOT best_p_ens)
        star_df = star_df.sort_values("star_score", ascending=False).reset_index(drop=True)

        self._cached_source_df = df
        star_df.insert(0, "rank", range(1, len(star_df) + 1))
        return star_df

    def save_top_candidates(self, star_df: pd.DataFrame) -> None:
        """Export the highest ranking stars to a CSV."""
        star_df.head(self.top_rows_to_export).to_csv(self.output_csv, index=False)

    def run(self) -> pd.DataFrame:
        """Generate the ranked stars, persist the top slice, and log a preview."""
        star_df = self.create_ranking()
        self.save_top_candidates(star_df)
        print(star_df.head(self.rows_to_print))
        return star_df
    
    
    def plot_star_windows(
        self,
        star_id: Union[str, int],
        *,
        save_path: Union[str, Path, None] = None,
        highlight_top: int | None = None,
    ) -> Path | None:
        """Visualize all ensemble scores for a given star and save the figure."""
        import re

        def canon(x: Union[str, int]) -> str:
            s = str(x).strip()
            digits = re.sub(r"\D+", "", s)
            return f"EPIC_{digits}" if digits else s

        df = (
            self._cached_source_df
            if self._cached_source_df is not None
            else pd.read_csv(self.source_csv, dtype={"star_id": str})
        )

        target = canon(star_id)
        df["star_id"] = df["star_id"].astype(str).str.strip()

        star_df = df.loc[df["star_id"] == target]
        if star_df.empty:
            print(f"No rows found for star_id={target} in {self.source_csv}")
            return None

        star_df = star_df.sort_values("p_ens", ascending=False)
        highlight_top = int(highlight_top or self.top_k)

        if "seg_mid_time" in star_df.columns:
            x_values = star_df["seg_mid_time"]
            x_label = "Segment mid time"
        else:
            x_values = list(range(1, len(star_df) + 1))
            x_label = "Window rank"

        plt.figure(figsize=(8, 4))
        plt.plot(x_values, star_df["p_ens"], "-o", label="ensemble score")

        top_df = star_df.head(highlight_top)
        if not top_df.empty:
            if "seg_mid_time" in top_df.columns:
                top_x = top_df["seg_mid_time"]
            else:
                top_x = list(range(1, len(top_df) + 1))
            plt.scatter(top_x, top_df["p_ens"], c="C1", label=f"top {len(top_df)}")

        plt.title(f"Star {target} ensemble window scores")
        plt.xlabel(x_label)
        plt.ylabel("p_ens")
        plt.grid(True)
        plt.legend()

        if save_path is None:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.plot_dir / f"star_{target}_p_ens.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Wrote star plot to {save_path}")
        return save_path
    
    
    def plot_star_flux_from_cache(
        self,
        star_id: Union[str, int],
        *,
        cache_root: Union[str, Path] = "splits/infer_c5/_cache/infer",
        save_dir: Union[str, Path] = "plots",
        zoom_pad: int = 256,
    ) -> None:
        import re
        from pathlib import Path

        def canon(x):
            s = str(x).strip()
            digits = re.sub(r"\D+", "", s)
            return f"EPIC_{digits}" if digits else s

        sid = canon(star_id)
        cache_root = Path(cache_root)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = cache_root / f"{sid}.npz"
        z = np.load(npz_path, allow_pickle=True)
        time = z["time_p"].astype(float)
        flux = z["flux_p"].astype(float)

        # Load the star’s top windows from your candidate CSV (top3 file)
        df = pd.read_csv(self.source_csv, dtype={"star_id": str})
        df["star_id"] = df["star_id"].astype(str).str.strip()
        star_df = df[df["star_id"] == sid].sort_values("p_ens", ascending=False)

        if star_df.empty:
            print(f"No candidate windows found for {sid} in {self.source_csv}")
            return

        # 1) Full LC + highlight windows
        plt.figure(figsize=(10, 4))
        plt.plot(time, flux, linewidth=0.6)
        for _, r in star_df.iterrows():
            s, e = int(r["start"]), int(r["end"])
            plt.axvspan(time[s], time[e - 1], alpha=0.15)
        plt.title(f"{sid} — full light curve (standardized flux) — top windows highlighted")
        plt.xlabel("Time (days)")
        plt.ylabel("Flux (standardized)")
        plt.tight_layout()
        out_full = save_dir / f"{sid}_full.png"
        plt.savefig(out_full)
        plt.close()
        print("Wrote:", out_full)

        # 2) Zoom each top window (+padding)
        for i, (_, r) in enumerate(star_df.iterrows(), start=1):
            s0, e0 = int(r["start"]), int(r["end"])
            s = max(0, s0 - zoom_pad)
            e = min(len(flux), e0 + zoom_pad)

            plt.figure(figsize=(10, 4))
            plt.plot(time[s:e], flux[s:e], linewidth=0.8)
            plt.axvspan(time[s0], time[e0 - 1], alpha=0.2)
            plt.title(f"{sid} — window {i} — p_ens={float(r['p_ens']):.6f} — idx[{s0}:{e0}]")
            plt.xlabel("Time (days)")
            plt.ylabel("Flux (standardized)")
            plt.tight_layout()
            out_zoom = save_dir / f"{sid}_zoom{i}.png"
            plt.savefig(out_zoom)
            plt.close()
            print("Wrote:", out_zoom)

    def _cache_path_for_star(self, star_id: str, cache_root: Union[str, Path]) -> Path:
        cache_root = Path(cache_root)
        return cache_root / f"{star_id}.npz"


    def _max_flux_from_cache(self, star_id: str, cache_root: Union[str, Path]) -> float:
        npz = self._cache_path_for_star(star_id, cache_root)
        with np.load(npz, allow_pickle=True) as z:
            flux = z["flux_p"]
        return float(np.max(flux))

    def vet_top_candidates(
        self,
        *,
        cache_root: Union[str, Path],
        in_star_csv: Union[str, Path] = "candidates_c5_star_ranking.csv",
        out_star_csv: Union[str, Path] = "candidates_c5_star_ranking_vetted.csv",
        top_n: int = 2000,
        deep_cut: float = -7.0,
        clip_sigma: float = 15.0,     # <-- match your PreprocessConfig clip_sigma
        clip_eps: float = 1e-3,       # <-- tolerance around the rail
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Vet top-ranked stars using cached standardized light curves.

        Buckets:
        - keep: not deep, not clipped, not insane
        - deep: min_flux < deep_cut AND not clipped (more "planet-like")
        - clipped: touches +/- clip_sigma rails (likely scaling issues or extreme events)
        - insane: NaN/inf stats or missing cache

        Writes:
        out_star_csv
        out_star_csv stem + "_deep.csv"
        out_star_csv stem + "_clipped.csv"
        out_star_csv stem + "_insane.csv"
        """
        cache_root = Path(cache_root)
        in_star_csv = Path(in_star_csv)
        out_star_csv = Path(out_star_csv)
        out_star_csv.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(in_star_csv, dtype={"star_id": str})

        # Sort by score column if present
        if "star_score" in df.columns:
            df = df.sort_values("star_score", ascending=False).reset_index(drop=True)
        elif "p_ens" in df.columns:
            df = df.sort_values("p_ens", ascending=False).reset_index(drop=True)

        df_top = df.head(int(top_n)).copy() if top_n is not None else df.copy()

        mins, maxs = [], []
        missing_cache = 0

        for sid in df_top["star_id"].tolist():
            try:
                mins.append(self._min_flux_from_cache(sid, cache_root))
                maxs.append(self._max_flux_from_cache(sid, cache_root))
            except FileNotFoundError:
                mins.append(np.nan)
                maxs.append(np.nan)
                missing_cache += 1

        df_top["min_flux"] = mins
        df_top["max_flux"] = maxs

        # Insane = missing or non-finite stats (should be rare now)
        df_top["is_insane"] = (~np.isfinite(df_top["min_flux"])) | (~np.isfinite(df_top["max_flux"]))

        # Clipped = touches the rails at +/- clip_sigma
        CLIP = float(clip_sigma)
        EPS = float(clip_eps)
        df_top["is_clipped"] = (
            (df_top["min_flux"] <= -CLIP + EPS) |
            (df_top["max_flux"] >=  CLIP - EPS)
        ) & (~df_top["is_insane"])

        # Deep = below threshold but NOT just because of clipping
        df_top["is_deep"] = (
            (df_top["min_flux"] < float(deep_cut)) &
            (~df_top["is_clipped"]) &
            (~df_top["is_insane"])
        )

        keep_df = df_top[(~df_top["is_deep"]) & (~df_top["is_clipped"]) & (~df_top["is_insane"])].copy()
        deep_df = df_top[df_top["is_deep"]].copy()
        clipped_df = df_top[df_top["is_clipped"]].copy()
        insane_df = df_top[df_top["is_insane"]].copy()

        deep_path = out_star_csv.with_name(out_star_csv.stem + "_deep.csv")
        clipped_path = out_star_csv.with_name(out_star_csv.stem + "_clipped.csv")
        insane_path = out_star_csv.with_name(out_star_csv.stem + "_insane.csv")

        keep_df.to_csv(out_star_csv, index=False)
        deep_df.to_csv(deep_path, index=False)
        clipped_df.to_csv(clipped_path, index=False)
        insane_df.to_csv(insane_path, index=False)

        if verbose:
            print(f"[vet] checked={len(df_top)} missing_cache={missing_cache}")
            print(f"[vet] wrote keep   : {out_star_csv} rows={len(keep_df)}")
            print(f"[vet] wrote deep   : {deep_path} rows={len(deep_df)}  (min_flux<{deep_cut} and not clipped)")
            print(f"[vet] wrote clipped: {clipped_path} rows={len(clipped_df)} (touches ±{clip_sigma})")
            print(f"[vet] wrote insane : {insane_path} rows={len(insane_df)}")

        return keep_df

    
    def create_star_ranking_from_top3(
    self,
    *,
    top3_csv: Union[str, Path] = "candidates_c5_top3_windows_per_star.csv",
    out_csv: Union[str, Path] = "candidates_c5_star_ranking.csv",
    verbose: bool = True,
    ) -> pd.DataFrame:
        """Build star-level ranking from a top3-windows-per-star CSV."""
        top3_csv = Path(top3_csv)
        out_csv = Path(out_csv)

        df = pd.read_csv(top3_csv, dtype={"star_id": str})
        df = df.sort_values(["star_id", "p_ens"], ascending=[True, False])

        def agg(g: pd.DataFrame) -> pd.Series:
            p = g["p_ens"].to_numpy(dtype=np.float32)
            p1 = float(p[0]) if len(p) > 0 else float("nan")
            p2 = float(p[1]) if len(p) > 1 else float("nan")
            p3 = float(p[2]) if len(p) > 2 else float("nan")

            star_score = float(np.mean(p)) if len(p) else float("nan")
            best_p = float(np.max(p)) if len(p) else float("nan")
            gap12 = p1 - p2 if np.isfinite(p1) and np.isfinite(p2) else float("nan")

            # Optional: model disagreement (average std across the top windows)
            disagree = float("nan")
            if {"p_m0", "p_m1", "p_m2"}.issubset(g.columns):
                m = g[["p_m0", "p_m1", "p_m2"]].to_numpy(dtype=np.float32)
                disagree = float(np.mean(np.std(m, axis=1)))

            return pd.Series(
                {
                    "star_score": star_score,
                    "best_p_ens": best_p,
                    "p1": p1,
                    "p2": p2,
                    "p3": p3,
                    "gap12": gap12,
                    "ens_disagree": disagree,
                    "n_windows_used": int(len(g)),
                    "provenance": str(g["provenance"].iloc[0]) if "provenance" in g.columns else "UNKNOWN",
                }
            )

        star_df = df.groupby("star_id", as_index=False).apply(agg).reset_index(drop=True)
        star_df = star_df.sort_values("star_score", ascending=False).reset_index(drop=True)
        star_df.insert(0, "rank", range(1, len(star_df) + 1))

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        star_df.to_csv(out_csv, index=False)

        if verbose:
            print("Wrote:", out_csv)
            print(star_df.head(20))

        return star_df

    def inspect_cached_star(
        self,
        star_id: str,
        cache_root: str | Path,
        *,
        print_head: int = 10,
    ) -> dict:
        """Inspect cached standardized flux for a star—no re-download, no rebuild."""
        cache_root = Path(cache_root)
        sid = str(star_id)
        if not sid.startswith("EPIC_"):
            sid = "EPIC_" + sid

        npz_path = cache_root / f"{sid}.npz"
        z = np.load(npz_path, allow_pickle=True)

        flux = z["flux_p"].astype(np.float32, copy=False)
        time = z["time_p"].astype(np.float64, copy=False)

        finite = np.isfinite(flux)
        xf = flux[finite]

        out = {
            "star_id": sid,
            "npz": str(npz_path),
            "n": int(flux.size),
            "finite": int(finite.sum()),
            "nonfinite": int((~finite).sum()),
            "min": float(np.nanmin(flux)),
            "max": float(np.nanmax(flux)),
            "p01": float(np.nanpercentile(flux, 1)),
            "p50": float(np.nanpercentile(flux, 50)),
            "p99": float(np.nanpercentile(flux, 99)),
            "std_finite": float(np.std(xf)) if xf.size else float("nan"),
            "mad_finite": float(np.median(np.abs(xf - np.median(xf)))) if xf.size else float("nan"),
            "near_zero_mad": bool(xf.size and (np.median(np.abs(xf - np.median(xf))) < 1e-6)),
            "count_below_-50": int(np.sum(flux < -50.0)),
            "count_below_-200": int(np.sum(flux < -200.0)),
            "count_below_-1000": int(np.sum(flux < -1000.0)),
        }

        print(pd.Series(out).head(print_head))
        return out
