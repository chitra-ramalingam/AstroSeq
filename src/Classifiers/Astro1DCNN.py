import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, models
from src.Classifiers.isolation_forest import IsolationForestScorer
from src.Classifiers.CnnNNet import CnnNNet


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
            print("Fetching from lightcurve:", target)
            missions_try = [pref] if pref else []
            for m in ["TESS", "Kepler", "K2"]:
                if m and m not in missions_try:
                    missions_try.append(m)

            for m in missions_try:
                sr = lk.search_lightcurve(target, mission=m, author=None if any_author else self.author)
                if len(sr) == 0:
                    continue
                if use_all:
                    lcc = sr[:max_files].download_all()
                    lc = lcc.stitch().remove_nans().normalize()
                else:
                    lc = sr[0].download().remove_nans().normalize()
                try:
                    lc_flat = lc.flatten(window_length=201, polyorder=2)
                except Exception:
                    lc_flat = lc.copy()

                time = lc.time.value.astype(np.float64)
                fl_raw = lc.flux.value.astype(np.float32)
                flux_flat = lc_flat.flux.value.astype(np.float32)
                flux = np.stack([fl_raw, flux_flat], axis=-1)  # (T, 2)
                return time, flux, m

            return None, None, None
        except Exception:
            return None, None, None

    # ---------- preprocessing ----------
    # ---- segment_with_idx ----
    def segment_with_idx(self, flux, w=None, stride=None):
        if flux.ndim == 1: flux = flux[:, None]
        T, C = flux.shape
        if w is None: w = self.window
        if stride is None: stride = max(1, w // 4)
        if w <= 0 or T < w:
            return np.empty((0, w, C), np.float32), np.empty((0, 2), int)

        segs, spans = [], []
        for i in range(0, T - w + 1, stride):
            seg = flux[i:i+w].astype(np.float32)          # (w, C)
            seg = seg - np.median(seg, axis=0, keepdims=True)  # center only; keep amplitude
            segs.append(seg)
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
        ##Some stars (many sectors/cadences) yield hundreds of segments; others yield only a few. Loss is summed over segments, so without a cap one prolific star can contribute 100× more gradient than another → the model overfits to that star’s quirks.

        subset_cols = [c for c in ["tid","hostname","toi","kepid","kepler_name","koi_name","koi","epic"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)
        df = df.sample(frac=1.0, random_state=42)  # shuffle

        segs_all, labels_all, groups_all = [], [], []
        pos_groups, neg_groups = set(), set()
        skipped = dict(no_id=0, dl_fail=0, empty=0, no_ephem=0)

        for _, row in df.iterrows():
            time, flux, mission = self.fetch_flux_row(row, use_all=use_all, max_files=max_files, any_author=any_author)
            flux = np.asarray(flux, dtype=np.float32)
            if flux.ndim == 1: 
                flux = flux[:, None]  # (T, C)
            try:
                if flux.size == 0 or flux.shape[0] < self.window:
                    skipped["empty"] += 1
                    continue
            except Exception:
                skipped["empty"] += 1
                continue

            med = np.median(flux, axis=0, keepdims=True)
            mad = np.median(np.abs(flux - med), axis=0, keepdims=True) + 1e-6
            flux = (flux - med) / mad

            if time is None:
                # figure out if target missing or download failed
                target, _ = self._row_to_target_and_mission(row)
                if not target: skipped["no_id"] += 1
                else:          skipped["dl_fail"] += 1
                continue

           # scorer = IsolationForestScorer(window=200, n_components=20, contamination=0.01, threshold_quantile=0.99)

            # Option A: per-star, fit on that star's segments
           # out = scorer.score_from_flux(flux, segment_with_idx_fn= self.segment_with_idx, fit_on_self=True)
           # scores, flags, spans,segs = out["scores"], out["flags"], out["spans"], out["segs"]


            # segment
            segs, spans = self.segment_with_idx(flux, w=self.window, stride=stride)
            if len(segs) == 0:
                skipped["empty"] += 1
                continue

            tt = self._row_to_target_and_mission(row)[0]
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
                # segs: all segments for this star,y_seg: labels for those segments (1 = transit overlap, 0 = off-transit)
                # spans: index ranges of each segment in the original light curve

                pos_idx = np.where(y_seg == 1)[0]
                neg_idx = np.where(y_seg == 0)[0]
                take_pos = min(len(pos_idx), per_star_cap // 2)
                take_neg = min(len(neg_idx), per_star_cap - take_pos)
                # Randomly sample those indices without replacement
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
        X = np.vstack(segs_all).astype(np.float32)   # shape: (N, window, C)
        y = np.concatenate(labels_all).astype(np.int32)
        groups = np.concatenate(groups_all)
        idx = np.arange(len(y)); np.random.shuffle(idx)
        X, y, groups = X[idx], y[idx], groups[idx]
        #X.shape == (5, 200, 1), y.shape == (5,) (e.g., [1,0,0,1,0])
        # #groups.shape == (5,) (e.g., ['TIC A','TIC A','TIC A','TIC B','TIC B'])
        print("Segments per class:", dict(zip(*np.unique(y, return_counts=True))))
        print("Stars with positives:", len(pos_groups), "Stars w/o positives:", len(neg_groups))
        print("Skipped:", skipped)
        print("Total segments:", X.shape[0], "Unique stars:", len(np.unique(groups)))
        return X, y, groups

    # ---- declareModel: accept channels ----
    def declareModel(self, channels=1):
        w = self.window
        inputs = layers.Input(shape=(w, channels))
        x = layers.Conv1D(32, 11, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, padding='same', activation='relu', dilation_rate=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding='same', activation='relu', dilation_rate=4)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                    tf.keras.metrics.AUC(curve="PR",  name="pr_auc")],
        )
        return model


    # def declareModel(self, channels=1):
    #     cnn_nnet = CnnNNet(window=self.window, channels=channels)
    #     return cnn_nnet.build_inception_resnet_1d(self.window, channels)

    # ---------- splitting ----------
    def stratified_group_train_val_test(self, y, groups, val_size=0.2, test_size=0.2, seed=42):
        rng = np.random.default_rng(seed)
        y = np.asarray(y); groups = np.asarray(groups)
        uniq = np.unique(groups)
        grp_lbl = np.array([int((y[groups == g] == 1).any()) for g in uniq], dtype=int)

        def split_groups(gs, ts):
            gs0 = gs[grp_lbl[np.isin(uniq, gs)] == 0]
            gs1 = gs[grp_lbl[np.isin(uniq, gs)] == 1]
            rng.shuffle(gs0); rng.shuffle(gs1)
            n = int(len(gs) * ts)
            take0 = int(len(gs0) * ts); take1 = n - take0
            return np.concatenate([gs0[:take0], gs1[:take1]]), np.concatenate([gs0[take0:], gs1[take1:]])

        # first carve out test
        test_groups, remain_groups = split_groups(uniq, test_size)
        # then from remain carve out val
        val_groups, train_groups = split_groups(remain_groups, val_size/(1.0 - test_size))

        def idx_for(gs): return np.where(np.isin(groups, gs))[0]
        return idx_for(train_groups), idx_for(val_groups), idx_for(test_groups)

    
    # ---------- train/eval ----------
    # def trainModel(self, model, X, y, groups, epochs=20, batch_size=32):
    #     train_idx, val_idx, test_idx = self.stratified_group_train_val_test(y, groups)
    #     X_train, y_train = X[train_idx], y[train_idx]
    #     X_val,   y_val   = X[val_idx],   y[val_idx]
    #     X_test,  y_test  = X[test_idx],  y[test_idx]

    #     cbs = [
    #         tf.keras.callbacks.ReduceLROnPlateau(
    #             monitor="val_pr_auc", mode="max", factor=0.5, patience=4, min_lr=1e-5, verbose=1
    #         ),
    #         tf.keras.callbacks.EarlyStopping(
    #             monitor="val_pr_auc", mode="max", patience=10, restore_best_weights=True
    #         ),
    #         tf.keras.callbacks.ModelCheckpoint(
    #             "best.keras", monitor="val_pr_auc", mode="max", save_best_only=True
    #         ),
    #     ]
    #     history = model.fit(
    #         X_train, y_train,
    #         validation_data=(X_val, y_val),
    #         epochs=60, batch_size=128,
    #         class_weight={0:1.0, 1: max(1.0, (y_train==0).sum()/(y_train==1).sum()+1e-9)},
    #         callbacks=cbs, verbose=1
    #     )

    #     return history, X_test, y_test
    def trainModel(self, model, X, y, groups, epochs=20, batch_size=32):
        train_idx, val_idx, test_idx = self.stratified_group_train_val_test(y, groups)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
       
        cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_ref.keras", monitor="val_pr_auc", mode="max", save_best_only=True)
        ]
        #model = model(window=X.shape[1], channels=X.shape[2] if X.ndim==3 else 1)
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=128, verbose=1, callbacks=cbs)
        return hist, X_test, y_test,X_val, y_val

    def evaluateModel(self, model, X_test, y_test):
        res = model.evaluate(X_test, y_test, verbose=0)
        names = model.metrics_names                 # e.g., ['loss','roc_auc','pr_auc','accuracy']
        if not isinstance(res, (list, tuple)):
            res = [res]
        report = dict(zip(names, res))
        print("Test -> " + " | ".join(f"{k}: {v:.4f}" for k, v in report.items()))
        return report
