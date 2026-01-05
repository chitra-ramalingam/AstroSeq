from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import time
import numpy as np
import pandas as pd
import re
from src.Classifiers.K2.K2_CentralizeTransits import K2_CentralizeSegmentTransit

@dataclass
class InjectionConfig:
    """
    Simple box-transit injection config (self-contained, no batman needed).
    """
    enabled: bool = True
    rng_seed: int = 42

    # Choose random transit parameters per star
    period_days_range: Tuple[float, float] = (0.7, 20.0)
    duration_hours_range: Tuple[float, float] = (1.0, 6.0)
    depth_ppm_range: Tuple[float, float] = (200.0, 4000.0)  # 200 ppm to 4000 ppm
    # How many injected stars (positives) to create within a split
    positive_star_fraction: float = 0.50


@dataclass
class PreprocessConfig:
    """
    Keep length stable. Do NOT drop NaNs.
    """
    use_flatten: bool = True
    flatten_window_length: int = 401
    flatten_polyorder: int = 2

    # After preprocessing, normalize to robust scale
    robust_center: bool = True
    fill_nonfinite_with_zero: bool = True


class K2SegmentDatasetBuilder:
    """
    Builds a K2 dataset from scratch and SAVES REAL WINDOW VECTORS.

    Output per split:
      - X_<split>.npy          float32 array shape (N, 1024, 2)
      - meta_<split>.parquet   dataframe with star_id, seg_mid_time, label, etc.

    Labeling:
      - If you don't have a reliable catalog, use injection positives.
      - This creates real "transit-like dips" and gives you ground-truth labels.

    Notes:
      - Uses Lightkurve for download.
      - Uses quality_bitmask="none" so cadence grid doesn't get silently shortened.
    """

    def __init__(
        self,
        out_dir: str | Path,
        window_len: int = 512,
        stride: int = 256,
        quality_bitmask: str = "none",
        provenance_priority: Tuple[str, ...] = ("K2", "EVEREST", "K2SFF"),
        banned_provenance: Tuple[str, ...] = ("K2SC", "K2VARCAT"),
        preprocess_cfg: Optional[PreprocessConfig] = None,
        inject_cfg: Optional[InjectionConfig] = None,
        verbose: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.window_len = int(window_len)
        self.stride = int(stride)
        self.quality_bitmask = str(quality_bitmask)

        self.provenance_priority = tuple(provenance_priority)
        self.banned_provenance = tuple(banned_provenance)

        self.pre_cfg = preprocess_cfg or PreprocessConfig()
        self.inj_cfg = inject_cfg or InjectionConfig()
        self.verbose = bool(verbose)

        self._rng = np.random.default_rng(self.inj_cfg.rng_seed)
        self.center_keep_frac = 0.15
        self.min_dur_coverage = 0.7
        self.k2CentralizedTransit = K2_CentralizeSegmentTransit()

    # -----------------------------
    # Public API
    # -----------------------------
    def build_split(self, epic_ids: List[str], split_name: str) -> Tuple[Path, Path]:
        """
        Builds and saves one split.
        Returns (X_path, meta_path).

        KEY CHANGE:
        - Download+preprocess ONCE per star, cached to disk.
        - Second phase reads cached arrays (fast, no network).
        - Positive-star selection is persisted so reruns are deterministic.
        """
        split_name = str(split_name)
        split_l = split_name.lower()

        X_path = self.out_dir / f"X_{split_name}.npy"
        meta_path = self.out_dir / f"meta_{split_name}.parquet"

        force_at_least_one = split_l.startswith(("train", "val"))
        cache_folder = Path("k2_dataset_centered_v2")
        # --- cache dirs
        cache_dir =  cache_folder / "_cache" / split_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # --- persist positive-star selection so reruns don’t reshuffle injected stars
        posmap_path = self.out_dir / f"posstars_{split_name}.parquet"

        epic_ids = list(epic_ids)
        epic_ids_can = [self._canon_epic(s) for s in epic_ids]

        if self.inj_cfg.enabled:
            if posmap_path.exists():
                dfp = pd.read_parquet(posmap_path)
                is_pos_star = {str(r["sid_can"]): bool(r["is_pos"]) for _, r in dfp.iterrows()}
            else:
                is_pos_star = self._choose_positive_stars(epic_ids, force_at_least_one=force_at_least_one)
                dfp = pd.DataFrame(
                    {"sid_can": list(is_pos_star.keys()), "is_pos": [int(v) for v in is_pos_star.values()]}
                )
                dfp.to_parquet(posmap_path, index=False)
        else:
            is_pos_star = {sid_can: False for sid_can in epic_ids_can}

        if self.verbose:
            npos = sum(1 for v in is_pos_star.values() if v)
            print(f"[{split_name}] pos_stars={npos}/{len(epic_ids_can)} (injection_enabled={self.inj_cfg.enabled})")
            print(f"[{split_name}] caching to: {cache_dir}")

        # -----------------------------
        # PASS 1: cache per-star arrays + compute total windows
        # -----------------------------
        total = 0
        plan_rows: List[Dict[str, object]] = []

        for sid_in in epic_ids:
            sid_can = self._canon_epic(sid_in)          # digits only
            star_id_out = f"EPIC_{sid_can}"

            cache_npz = cache_dir / f"{star_id_out}.npz"

            if cache_npz.exists():
                # Use cached shapes (no download)
                z = np.load(cache_npz, allow_pickle=True)
                n_points = int(z["n_points"])
                n_windows = self._count_windows(n_points)
                prov = str(z["prov"])
                label_star = int(z["label_star"])
                has_inj = int(z["has_inj"])
                total += n_windows
                plan_rows.append(
                    {
                        "star_id": star_id_out,
                        "cache": str(cache_npz),
                        "n_windows": int(n_windows),
                        "provenance": prov,
                        "label_star": int(label_star),
                        "has_inj": int(has_inj),
                    }
                )
                continue

            # Download + preprocess ONCE
            t, f, prov = self._fetch_time_flux(sid_can)
            flux_p, time_p = self._preprocess(t, f)

            n_windows = self._count_windows(len(flux_p))
            label_star = int(is_pos_star.get(sid_can, False))

            has_inj = 0
            t0 = np.nan
            period = np.nan
            dur_days = np.nan

            if self.inj_cfg.enabled and label_star == 1:
                flux_p, inj = self._inject_box_transits(time_p, flux_p)
                has_inj = 1
                t0 = float(inj["t0"])
                period = float(inj["period"])
                dur_days = float(inj["dur_days"])

            # Cache (fast to reuse after timeout)
            # NOTE: use np.savez (no compression) to keep it fast
            np.savez(
                cache_npz,
                flux_p=flux_p.astype(np.float32, copy=False),
                time_p=time_p.astype(np.float64, copy=False),
                n_points=np.int64(len(flux_p)),
                prov=np.array(prov, dtype=object),
                label_star=np.int32(label_star),
                has_inj=np.int32(has_inj),
                t0=np.float64(t0),
                period=np.float64(period),
                dur_days=np.float64(dur_days),
            )

            total += n_windows
            plan_rows.append(
                {
                    "star_id": star_id_out,
                    "cache": str(cache_npz),
                    "n_windows": int(n_windows),
                    "provenance": prov,
                    "label_star": int(label_star),
                    "has_inj": int(has_inj),
                }
            )

        if self.verbose:
            print(f"[{split_name}] Planned windows: {total} across {len(epic_ids)} stars")

        # If nothing planned, write empties (your existing behavior)
        if total == 0:
            X_empty = np.zeros((0, self.window_len, 2), dtype=np.float32)
            X_written_path = self._safe_save_npy_overwrite_or_version(
                target_path=X_path, arr=X_empty, split_name=split_name, retries=10, sleep_s=0.25
            )
            self._write_latest_pointer("X", split_name, X_written_path)

            df_meta = pd.DataFrame(
                columns=["star_id","mission","provenance","split","start","end","seg_mid_time","label","label_star"]
            )
            df_meta.to_parquet(meta_path, index=False)
            self._write_latest_pointer("meta", split_name, meta_path)
            return X_written_path, meta_path

        # -----------------------------
        # PASS 2: allocate X once + write from cached arrays (no network)
        # -----------------------------
        X = np.lib.format.open_memmap(
            X_path, mode="w+", dtype=np.float32, shape=(total, self.window_len, 2)
        )

        meta_records: List[Dict[str, object]] = []
        write_idx = 0

        for row in plan_rows:
            n_windows = int(row["n_windows"])
            if n_windows <= 0:
                continue

            star_id_out = str(row["star_id"])
            prov = str(row["provenance"])
            label_star = int(row["label_star"])
            has_inj = int(row["has_inj"]) == 1

            z = np.load(str(row["cache"]), allow_pickle=True)
            flux_p = z["flux_p"].astype(np.float32, copy=False)
            time_p = z["time_p"].astype(np.float64, copy=False)

            t0 = float(z["t0"])
            period = float(z["period"])
            dur_days = float(z["dur_days"])

            for w in range(n_windows):
                start = w * self.stride
                end = start + self.window_len

                seg_flux = flux_p[start:end]
                seg_time = time_p[start:end]

                ch0 = seg_flux.astype(np.float32, copy=False)
                ch1 = np.diff(ch0, prepend=ch0[:1]).astype(np.float32, copy=False)

                X[write_idx, :, 0] = ch0
                X[write_idx, :, 1] = ch1

                seg_mid_time = float(np.nanmedian(seg_time)) if np.isfinite(seg_time).any() else np.nan

                if has_inj:
                    seg_t0 = float(seg_time[0]) if np.isfinite(seg_time[0]) else float(np.nanmin(seg_time))
                    seg_t1 = float(seg_time[-1]) if np.isfinite(seg_time[-1]) else float(np.nanmax(seg_time))

                    label = self.k2CentralizedTransit.label_segment_centered(
                        seg_t0=seg_t0,
                        seg_t1=seg_t1,
                        t0=t0,
                        period=period,
                        dur_days=dur_days,
                        center_keep_frac=self.center_keep_frac,
                        min_dur_coverage=self.min_dur_coverage,
                    )
                else:
                    label = 0

                meta_records.append(
                    {
                        "star_id": star_id_out,
                        "mission": "k2",
                        "provenance": prov,
                        "split": split_name,
                        "start": int(start),
                        "end": int(end),
                        "seg_mid_time": seg_mid_time,
                        "label": int(label),
                        "label_star": int(label_star),
                    }
                )
                write_idx += 1

            if self.verbose:
                print(f"[{split_name}] {star_id_out}: windows={n_windows} pos_star={label_star}")

        # Safety truncate (shouldn't happen, but keep it)
        if write_idx != total:
            if self.verbose:
                print(f"[{split_name}] Truncating X: wrote {write_idx} of planned {total}")
            X.flush()
            X = np.asarray(X[:write_idx], dtype=np.float32)
            np.save(X_path, X)

        df_meta = pd.DataFrame(meta_records)
        df_meta.to_parquet(meta_path, index=False)

        if self.verbose:
            pos_win = int(df_meta["label"].sum()) if len(df_meta) else 0
            # Avoid np.load(mmap_mode='r') here to reduce Windows file locking pain
            print(f"[{split_name}] Saved: {X_path} shape=({write_idx}, {self.window_len}, 2)")
            print(f"[{split_name}] Saved: {meta_path} rows={len(df_meta)} pos_win={pos_win}")

        return X_path, meta_path

    # -----------------------------
    # Internal helpers
    # -----------------------------from typing import List, Dict


    def _canon_epic(self, x: str) -> str:
        """Return EPIC ID as digits only, e.g. 'EPIC_211822797' -> '211822797'."""
        return re.sub(r"\D+", "", str(x))

    def _choose_positive_stars(self, epic_ids, force_at_least_one=False):
        epic_ids_can = [self._canon_epic(x) for x in epic_ids]
        n = len(epic_ids_can)
        frac = float(self.inj_cfg.positive_star_fraction)
        k = int(frac * n + 0.5)
        if force_at_least_one and n > 0 and frac > 0:
            k = max(1, k)
        k = min(k, n)

        chosen = set(self._rng.choice(epic_ids_can, size=k, replace=False)) if (n > 0 and k > 0) else set()
        print(f"Chosen {len(chosen)} positive stars out of {n} for injection. (frac={frac}, k={k})")

        return {sid_can: (sid_can in chosen) for sid_can in epic_ids_can}

    def _fetch_time_flux(self, star_id: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Downloads a supported K2 light curve (avoids banned pipelines).
        Returns (time_days, flux, provenance_name).
        """
        import lightkurve as lk
        from astropy.utils.data import conf as astropy_conf

        # Helps prevent random MAST network stalls from killing you
        astropy_conf.remote_timeout = getattr(self, "remote_timeout", 120)

        # Normalize EPIC id
        sid_raw = str(star_id)
        sid = sid_raw.replace("EPIC_", "").replace("EPIC ", "").strip()
        query = f"EPIC {sid}"
        print("Querying:", query)

        banned = {b.upper() for b in getattr(self, "banned_provenance", [])}

        # Prefer these pipelines/authors if present (and not banned)
        priority = [p for p in getattr(self, "provenance_priority", []) if p.upper() not in banned]
        allowed_authors = tuple(priority) if len(priority) > 0 else None

        # 1) Try filtered search first (reduces 25K EPIC -> 140K products explosion)
        try:
            sr = lk.search_lightcurve(query, mission="K2", author=allowed_authors, limit=50)
        except TypeError:
            # Older lightkurve versions may not support limit= or tuple author
            sr = lk.search_lightcurve(query, mission="K2", author=allowed_authors)

        # 2) Fallback to unfiltered d if filtered yields nothing
        if len(sr) == 0:
            sr = lk.search_lightcurve(query, mission="K2")
            if len(sr) == 0:
                raise RuntimeError(f"No K2 lightcurve found for {sid} (query='{query}')")

        tbl = sr.table

        prov = np.asarray(tbl["provenance_name"]).astype(str) if "provenance_name" in tbl.colnames else None

        # Build candidate order (priority first, then other non-banned)
        idxs: List[int] = []
        if prov is not None:
            prov_u = np.char.upper(prov)

            for p in priority:
                idxs.extend([i for i in range(len(sr)) if prov_u[i] == p.upper()])

            idxs.extend([i for i in range(len(sr)) if prov_u[i] not in banned and i not in idxs])
        else:
            idxs = list(range(len(sr)))

        # Hard cap so a single EPIC can’t try 40+ products
        max_products = int(getattr(self, "max_products_per_star", 6))
        idxs = idxs[:max_products]

        last_err: Optional[Exception] = None

        # Ensure we always download into a stable folder (so cache reuse is reliable)
        download_dir = getattr(self, "download_dir", None)

        for i in idxs:
            try:
                if prov is not None and str(prov[i]).upper() in banned:
                    continue

                obj = sr[i].download(
                    quality_bitmask=self.quality_bitmask,
                    download_dir=download_dir,
                    cache=True,
                )
                if obj is None:
                    continue

                # If we ever get a collection, stitch it; otherwise keep single LC
                lc = obj
                if hasattr(obj, "stitch"):
                    try:
                        lc = obj.stitch()
                    except Exception:
                        lc = obj
                time = np.asarray(lc.time.value, dtype=np.float64)  # days
                flux = np.asarray(lc.flux.value, dtype=np.float32)  # keep NaNs

                prov_name = str(prov[i]) if prov is not None else "UNKNOWN"

                if time.size == 0 or flux.size == 0:
                    continue
                if time.size != flux.size:
                    continue

                return time, flux, prov_name

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to download supported LC for {sid}. Last error: {last_err}")

    def _preprocess(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Length-preserving preprocessing.
        """
        time = np.asarray(time, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float32)

        # Optional flatten (keeps same length; Lightkurve may introduce NaNs in masked areas)
        if self.pre_cfg.use_flatten:
            try:
                import lightkurve as lk
                lc = lk.LightCurve(time=time, flux=flux)
                lc2 = lc.flatten(window_length=self.pre_cfg.flatten_window_length, polyorder=self.pre_cfg.flatten_polyorder)
                flux = np.asarray(lc2.flux.value, dtype=np.float32)
                time = np.asarray(lc2.time.value, dtype=np.float64)
            except Exception:
                # If flatten fails, continue with raw flux
                pass

        # Robust normalize (nan-safe), keep length
        if self.pre_cfg.robust_center:
            med = np.nanmedian(flux)
            std = np.nanstd(flux) + 1e-8
            flux = (flux - med) / std

        if self.pre_cfg.fill_nonfinite_with_zero:
            flux = flux.astype(np.float32, copy=False)
            flux[~np.isfinite(flux)] = 0.0

        return flux.astype(np.float32, copy=False), time

    def _count_windows(self, n_points: int) -> int:
        n = int(n_points)
        if n < self.window_len:
            return 0
        return 1 + (n - self.window_len) // self.stride

    def _inject_box_transits(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Injects a simple box-shaped transit pattern into flux.

        Returns:
          flux_injected, info dict with in_transit boolean mask
        """
        t = np.asarray(time, dtype=np.float64)
        y = np.asarray(flux, dtype=np.float32).copy()

        # Random params
        P = self._rng.uniform(*self.inj_cfg.period_days_range)
        dur_hr = self._rng.uniform(*self.inj_cfg.duration_hours_range)
        depth_ppm = self._rng.uniform(*self.inj_cfg.depth_ppm_range)

        dur_days = dur_hr / 24.0
        depth = depth_ppm * 1e-6  # ppm -> relative

        # Choose t0 inside span
        tmin, tmax = np.nanmin(t), np.nanmax(t)
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            return y, {"in_transit": np.zeros_like(y, dtype=bool)}

        t0 = self._rng.uniform(tmin, min(tmax, tmin + P))

        # Compute phase distance to nearest transit center
        phase = ((t - t0) % P)
        dist = np.minimum(phase, P - phase)

        in_transit = dist <= (dur_days / 2.0)

        # Apply dip (subtract depth in normalized units; your flux is normalized so depth is relative-ish)
        # We inject as a negative offset scaled by depth.
        y[in_transit] = y[in_transit] - float(depth)

        return y.astype(np.float32), {
            "in_transit": in_transit,
            "t0": float(t0),
            "period": float(P),
            "dur_days": float(dur_days),
            "depth": float(depth),
        }

    
    def _write_latest_pointer(self, kind: str, split_name: str, path: Path) -> None:
        """
        Writes a small pointer file so you can always find the newest artifact even
        if the stable filename was locked and we had to fall back to a versioned name.
        """
        ptr = (self.out_dir / f"{kind}_{split_name}_LATEST.txt").resolve()
        ptr.write_text(str(Path(path).resolve()), encoding="utf-8")

    def _safe_save_npy_overwrite_or_version(
        self,
        target_path: Path,
        arr: np.ndarray,
        split_name: str,
        retries: int = 10,
        sleep_s: float = 0.25,
    ) -> Path:
        """
        Save arr to target_path safely on Windows.

        - writes to <stem>.tmp.npy first (prevents np.save adding .npy unexpectedly)
        - tries os.replace(tmp, target) a few times
        - if still locked, moves tmp to a versioned filename and returns that path

        Never hangs (bounded retries).
        """
        target_path = Path(target_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = target_path.with_name(f"{target_path.stem}.tmp.npy").resolve()
        np.save(tmp_path, arr)

        if not tmp_path.exists():
            raise RuntimeError(f"Temp file was not created: {tmp_path}")

        # Try to replace stable path (may fail if locked)
        for _ in range(int(retries)):
            try:
                os.replace(str(tmp_path), str(target_path))
                return target_path
            except PermissionError:
                time.sleep(float(sleep_s))

        # Fallback: write a versioned file (always succeeds unless directory perms are broken)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fallback_path = target_path.with_name(f"{target_path.stem}_{stamp}.npy").resolve()
        os.replace(str(tmp_path), str(fallback_path))

        if self.verbose:
            print(f"[{split_name}] WARNING: {target_path} locked. Wrote: {fallback_path}")

        return fallback_path

    def split_epics_min(self, ids):
        ids = list(ids)
        n = len(ids)
        if n <= 2:
            return ids, [], []

        # for n=3: 2/1/0
        if n == 3:
            return ids[:2], ids[2:], []

        # n>=4: 80/10/10 but guarantee val>=1 and test>=1
        n_train = max(1, int(round(0.8 * n)))
        n_val = max(1, int(round(0.1 * n)))
        if n_train + n_val >= n:
            n_val = 1
            n_train = n - n_val - 1

        train = ids[:n_train]
        val = ids[n_train:n_train + n_val]
        test = ids[n_train + n_val:]
        return train, val, test
