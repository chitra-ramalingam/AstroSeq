import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, models

class Astro1DCNN:
    def __init__(self, window=200, mission="TESS", author="SPOC"):
        self.window = window
        self.mission = mission
        self.author = author

    # ---------- helpers ----------
    def _intish(self, x):
        try:
            return int(float(x))
        except Exception:
            return None

    def _row_to_target_and_mission(self, row):
        # Kepler (KIC)
        for k in ["kepid", "KIC", "kic"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"KIC {n}", "Kepler"
        # K2 (EPIC)
        for k in ["epic", "EPIC"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"EPIC {n}", "K2"
        # TESS (TIC)
        for k in ["tid", "TIC ID", "tic_id", "tic"]:
            if k in row and pd.notna(row[k]):
                n = self._intish(row[k])
                if n is not None:
                    return f"TIC {n}", "TESS"
        # TOI / names
        for k in ["toi", "TOI"]:
            if k in row and pd.notna(row[k]):
                return f"TOI {str(row[k]).strip()}", "TESS"
        if "hostname" in row and pd.notna(row["hostname"]):
            return str(row["hostname"]).strip(), None
        return None, None

    def _get_ephemeris(self, row):
        """Return (t0_series_units, period_days, dur_hours, time_system) or (None,...)."""
        # Kepler/K2 KOI ephemeris (already BKJD)
        t0_koi = row.get("koi_time0bk", np.nan)
        per_koi = row.get("koi_period", np.nan)
        dur_koi = row.get("koi_duration", np.nan)  # hours
        if pd.notna(t0_koi) and pd.notna(per_koi):
            dur_h = float(dur_koi) if pd.notna(dur_koi) else 6.0
            return float(t0_koi), float(per_koi), dur_h, "BKJD"

        # NASA Exoplanet Archive 'pl_tranmid' (BJD_TDB), convert to BTJD for TESS
        t0_pl = row.get("pl_tranmid", np.nan)
        per_pl = row.get("pl_orbper", np.nan)
        dur_pl = row.get("pl_trandurh", np.nan)  # hours
        if pd.notna(t0_pl) and pd.notna(per_pl):
            dur_h = float(dur_pl) if pd.notna(dur_pl) else 6.0
            # convert BJD -> BTJD (BJD - 2457000)
            t0_btjd = float(t0_pl) - 2457000.0
            return t0_btjd, float(per_pl), dur_h, "BTJD"

        return None, None, None, None

    # ---------- data IO ----------
    def fetch_flux_row(self, row, use_all=True, max_files=4, any_author=True):
        """Return (time, flux, mission) on success; else (None, None, None)."""
        try:
            target, pref = self._row_to_target_and_mission(row)
            if not target:
                return None, None, None

            missions_try = [pref] if pref else []
            for m in ["TESS", "Kepler", "K2"]:
                if m and m not in missions_try:
                    missions_try.append(m)

            for m in missions_try:
                sr = lk.search_lightcurve(target, mission=m, author=None if any_author else self.author)
                if len(sr) == 0:
                    continue
                if use_all:
                    sr = sr[:max_files] if max_files else sr
                    lcc = sr.download_all()
                    lc = lcc.stitch().remove_nans().normalize()
                else:
                    lc = sr[0].download().remove_nans().normalize()
                time = lc.time.value.astype(np.float64)   # BTJD for TESS, BKJD for Kepler/K2
                flux = lc.flux.value.astype(np.float32)
                return time, flux, m
            return None, None, None
        except Exception:
            return None, None, None

    # ---------- preprocessing ----------
    def segment_with_idx(self, flux, w=None, stride=None):
        """Overlapping segments + per-segment z-score. Returns (segs, spans)."""
        if w is None: w = self.window
        if stride is None: stride = w // 4
        if w <= 0 or len(flux) < w:
            return np.empty((0, w), np.float32), np.empty((0, 2), int)
        segs, spans = [], []
        for i in range(0, len(flux) - w + 1, stride):
            seg = flux[i:i+w].astype(np.float32)
            mu = seg.mean(); sd = seg.std() + 1e-6
            segs.append((seg - mu) / sd)
            spans.append((i, i+w))
        return np.asarray(segs, np.float32), np.asarray(spans, int)

    def label_by_ephem(self, time, spans, t0, period, dur_h=6.0, time_system="BTJD"):
        """Label 1 if span overlaps a predicted transit window; else 0."""
        if t0 is None or period is None or len(spans) == 0:
            return np.zeros(len(spans), dtype=int)
        dur_d = (dur_h or 6.0) / 24.0
        y = []
        for i0, i1 in spans:
            t_start, t_end = time[i0], time[i1-1]
            mid = 0.5*(t_start + t_end)
            k = np.round((mid - t0) / period)
            t_transit = t0 + k*period
            overlaps = (t_transit >= t_start - 1.5*dur_d) and (t_transit <= t_end + 1.5*dur_d)
            y.append(1 if overlaps else 0)
        return np.array(y, dtype=int)

    # ---------- dataset builder ----------
    def build_from_csv(self, csv_path,
                       min_pos_groups=10, min_neg_groups=10,
                       any_author=True, use_all=True, max_files=4,
                       stride=None, per_star_cap=40):
        df = pd.read_csv(csv_path)

        # dedupe by any usable ID
        subset_cols = [c for c in ["tid","hostname","toi","kepid","kepler_name","koi_name","koi","epic"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)
        df = df.sample(frac=1.0, random_state=42)  # shuffle

        segs_all, labels_all, groups_all = [], [], []
        pos_groups, neg_groups = set(), set()
        skipped = dict(no_id=0, dl_fail=0, empty=0, no_ephem=0)

        for _, row in df.iterrows():
            time, flux, mission = self.fetch_flux_row(row, use_all=use_all, max_files=max_files, any_author=any_author)
            if time is None:
                # figure out if target missing or download failed
                target, _ = self._row_to_target_and_mission(row)
                if not target: skipped["no_id"] += 1
                else:          skipped["dl_fail"] += 1
                continue

            # segment
            segs, spans = self.segment_with_idx(flux, w=self.window, stride=stride)
            if len(segs) == 0:
                skipped["empty"] += 1
                continue

            # ephemeris
            t0, period, dur_h, timesys = self._get_ephemeris(row)
            # Ensure ephemeris on same time system as LC
            if t0 is not None:
                if mission in ["Kepler","K2"] and timesys != "BKJD":
                    # If we somehow got BTJD ephemeris for Kepler (rare), convert: BTJD->BJD add 2457000, then -2454833
                    t0 = t0 + 2457000.0 - 2454833.0
                if mission == "TESS" and timesys != "BTJD":
                    # If we somehow got BKJD ephemeris for TESS (rare), convert: BKJD->BJD add 2454833, then -2457000
                    t0 = t0 + 2454833.0 - 2457000.0

            if t0 is None or period is None:
                # No ephemeris: treat as NEGATIVE star (all 0s)
                y_seg = np.zeros(len(segs), dtype=int)
                star_label = 0
                neg_groups.add(self._row_to_target_and_mission(row)[0])
            else:
                # Label by overlap
                y_seg = self.label_by_ephem(time, spans, t0, period, dur_h=dur_h, time_system=("BTJD" if mission=="TESS" else "BKJD"))
                # Decide group label: positive if any positive segment
                star_label = int((y_seg == 1).any())
                if star_label == 1:
                    pos_groups.add(self._row_to_target_and_mission(row)[0])
                else:
                    neg_groups.add(self._row_to_target_and_mission(row)[0])

            # cap per star
            target_id = self._row_to_target_and_mission(row)[0]
            if per_star_cap is not None and len(segs) > per_star_cap:
                # keep up to cap, but preserve the class mix
                pos_idx = np.where(y_seg == 1)[0]
                neg_idx = np.where(y_seg == 0)[0]
                take_pos = min(len(pos_idx), per_star_cap // 2)
                take_neg = min(len(neg_idx), per_star_cap - take_pos)
                choose = np.concatenate([
                    np.random.default_rng(42).choice(pos_idx, size=take_pos, replace=False) if take_pos>0 else np.array([], int),
                    np.random.default_rng(43).choice(neg_idx, size=take_neg, replace=False) if take_neg>0 else np.array([], int),
                ])
                segs = segs[choose]
                y_seg = y_seg[choose]
                spans = spans[choose]

            # append
            segs_all.append(segs)
            labels_all.append(y_seg.astype(np.int32))
            groups_all.append(np.array([target_id]*len(segs), dtype=object))

            # stop when we have enough groups on both sides
            if len(pos_groups) >= min_pos_groups and len(neg_groups) >= min_neg_groups:
                break

        # final arrays
        if not segs_all:
            raise RuntimeError("No usable data collected.")
        X = np.vstack(segs_all).astype(np.float32).reshape(-1, self.window, 1)
        y = np.concatenate(labels_all).astype(np.int32)
        groups = np.concatenate(groups_all)

        # shuffle consistently
        idx = np.arange(len(y)); np.random.shuffle(idx)
        X, y, groups = X[idx], y[idx], groups[idx]

        print("Segments per class:", dict(zip(*np.unique(y, return_counts=True))))
        print("Stars with positives:", len(pos_groups), "Stars w/o positives:", len(neg_groups))
        print("Skipped:", skipped)
        print("Total segments:", X.shape[0], "Unique stars:", len(np.unique(groups)))
        return X, y, groups

    def declareModel(self):
        w = self.window
        model = models.Sequential([
            layers.Conv1D(32, 7, padding='same', activation='relu', input_shape=(w,1)),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, padding='same', activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, padding='same', activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # ---------- splitting ----------
    def stratified_group_split(self, y, groups, test_size=0.2, seed=42):
        rng = np.random.default_rng(seed)
        y = np.asarray(y); groups = np.asarray(groups)
        uniq = np.unique(groups)
        # group label = 1 if ANY positive in that group
        grp_lbl = np.array([int((y[groups == g] == 1).any()) for g in uniq], dtype=int)

        train_groups, test_groups = [], []
        for lbl in [0, 1]:
            g = uniq[grp_lbl == lbl]
            if len(g) == 0:
                continue
            rng.shuffle(g)
            n_test = max(1, int(len(g) * test_size))
            test_groups += list(g[:n_test])
            train_groups += list(g[n_test:])

        test_mask = np.isin(groups, test_groups)
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where(test_mask)[0]
        return train_idx, test_idx

    # ---------- train/eval ----------
    def trainModel(self, model, X, y, groups, epochs=20, batch_size=32):
        train_idx, test_idx = self.stratified_group_split(y, groups, test_size=0.2, seed=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=callbacks, verbose=1)
        return history, X_test, y_test

    def evaluateModel(self, model, X_test, y_test):
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.3f}")
        return acc
