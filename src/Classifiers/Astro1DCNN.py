import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, models
from src.Classifiers.isolation_forest import IsolationForestScorer
from src.Classifiers.CnnNNet import CnnNNet
from src.Classifiers.CommonHelper import CommonHelper
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
)

class Astro1DCNN:
    def __init__(self, window=200, mission="TESS", author="SPOC"):
        self.window = window
        self.mission = mission
        self.author = author
        self.commonHelper = CommonHelper()

        self.op_threshold = 0.5  # will overwrite after evaluateModel

    def set_threshold(self, thr: float):
        """Set deployment threshold chosen on VAL."""
        self.op_threshold = float(thr)

    def _aggregate_probs(self, probs, mode="topk_mean", k=3):
        """
        Turn per-seg probabilities -> one star score.
        mode='topk_mean' (default): mean of top-k segment probs.
        Alternatives: 'max', 'mean'.
        """
        p = np.asarray(probs).ravel()
        if p.size == 0: return float("nan")
        if mode == "max":
            return float(p.max())
        if mode == "mean":
            return float(p.mean())
        k = max(1, min(k, p.size))
        topk = np.partition(p, -k)[-k:]
        return float(topk.mean())
    # ---------- helpers ----------
    
    
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
    
    # ---------- preprocessing ----------
    # ---- segment_with_idx ----
    
    def label_by_ephem(self, time, spans, t0, period, dur_h=6.0, time_system="BTJD"):
        """Label 1 if span overlaps a predicted transit window; else 0."""
        if t0 is None or period is None or len(spans) == 0:
            return np.zeros(len(spans), dtype=int)
        dur_d = (dur_h or 6.0) / 24.0
        y = []
        for i0, i1 in spans:
            t_start, t_end = time[i0], time[i1-1]
            # search all transits that could intersect this span
            k0 = int(np.floor((t_start - t0) / period)) - 1
            k1 = int(np.ceil ((t_end   - t0) / period)) + 1
            hit = False
            for k in range(k0, k1 + 1):
                t_ctr = t0 + k * period
                # mark positive if the +/- 1.5*duration window touches [t_start, t_end]
                if (t_ctr + 1.5*dur_d) >= t_start and (t_ctr - 1.5*dur_d) <= t_end:
                    hit = True
                    break
            y.append(1 if hit else 0)
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
        weights_all = []
        pos_groups, neg_groups = set(), set()
        skipped = dict(no_id=0, dl_fail=0, empty=0, no_ephem=0)
        cache = {}
        for _, row in df.iterrows():
            time, flux, mission,t0 = self.commonHelper.fetch_flux_row(row, use_all=use_all, max_files=max_files, any_author=any_author)
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
                target, _ = self.commonHelper.row_to_target_and_mission(row)
                if not target: skipped["no_id"] += 1
                else:          skipped["dl_fail"] += 1
                continue

           # scorer = IsolationForestScorer(window=200, n_components=20, contamination=0.01, threshold_quantile=0.99)

            # Option A: per-star, fit on that star's segments
           # out = scorer.score_from_flux(flux, segment_with_idx_fn= self.segment_with_idx, fit_on_self=True)
           # scores, flags, spans,segs = out["scores"], out["flags"], out["spans"], out["segs"]


            # segment
            segs, spans = self.commonHelper.segment_with_idx(flux, w=self.window, stride=stride)
            if len(segs) == 0:
                skipped["empty"] += 1
                continue
            cache[t0] = {"segs": segs, "mission": mission, "n_segments": int(segs.shape[0])}
        
              # ephemeris
            t0, period, dur_h, timesys = self._get_ephemeris(row)
            # Ensure ephemeris on same time system as LC
            if t0 is None or period is None:
                # keep as negatives but mark them LOW CONFIDENCE
                y_seg = np.zeros(len(segs), dtype=int)
                star_label = 0
                neg_groups.add(self.commonHelper.row_to_target_and_mission(row)[0])
                w_seg = np.full(len(segs), 0.3, dtype=np.float32)   # <— NEW: weak weight
            else:
                y_seg = self.label_by_ephem(time, spans, t0, period, dur_h=dur_h,
                                            time_system=("BTJD" if mission=="TESS" else "BKJD"))
                star_label = int((y_seg == 1).any())
                (pos_groups if star_label==1 else neg_groups).add(self.commonHelper.row_to_target_and_mission(row)[0])
                # confident labels: full weight
                w_seg = np.ones(len(segs), dtype=np.float32)        # <— NEW

            # cap per star (keep weights in sync!)
            if per_star_cap is not None and len(segs) > per_star_cap:
                pos_idx = np.where(y_seg == 1)[0];  neg_idx = np.where(y_seg == 0)[0]
                take_pos = min(len(pos_idx), per_star_cap // 2)
                take_neg = min(len(neg_idx), per_star_cap - take_pos)
                choose = np.concatenate([
                    np.random.default_rng(42).choice(pos_idx, size=take_pos, replace=False) if take_pos>0 else np.array([], int),
                    np.random.default_rng(43).choice(neg_idx, size=take_neg, replace=False) if take_neg>0 else np.array([], int),
                ])
                segs   = segs[choose]
                y_seg  = y_seg[choose]
                w_seg  = w_seg[choose]                                 # <— NEW
                spans  = spans[choose]

            # append (keep channels!)
                segs_all.append(segs)
                labels_all.append(y_seg.astype(np.int32))
                groups_all.append(np.array([self.commonHelper.row_to_target_and_mission(row)[0]]*len(segs), dtype=object))
                weights_all.append(w_seg)                                   # <— NEW

                if len(pos_groups) >= min_pos_groups and len(neg_groups) >= min_neg_groups:
                    break

            if not segs_all:
                raise RuntimeError("No usable data collected.")

        X = np.vstack(segs_all).astype(np.float32)          # (N, window, C)
        y = np.concatenate(labels_all).astype(np.int32)
        groups = np.concatenate(groups_all)
        sample_w = np.concatenate(weights_all).astype(np.float32)       # <— NEW

        idx = np.arange(len(y)); np.random.shuffle(idx)
        X, y, groups, sample_w = X[idx], y[idx], groups[idx], sample_w[idx]

        print("Segments per class:", dict(zip(*np.unique(y, return_counts=True))))
        print("Stars with positives:", len(pos_groups), "Stars w/o positives:", len(neg_groups))
        print("Skipped:", skipped)
        print("Total segments:", X.shape[0], "Unique stars:", len(np.unique(groups)))
        return X, y, groups, sample_w , cache    
        
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

    def trainModel(self, model, X, y, groups, sample_w, epochs=20, batch_size=32):
        train_idx, val_idx, test_idx = self.stratified_group_train_val_test(y, groups)
        X_train, y_train, w_train = X[train_idx], y[train_idx], sample_w[train_idx]
        X_val,   y_val,   w_val   = X[val_idx],   y[val_idx],   sample_w[val_idx]
        X_test,  y_test           = X[test_idx],  y[test_idx]

        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("best_ref.keras", monitor="val_pr_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5),
        ]

        history = model.fit(
            X_train, y_train,
            sample_weight=w_train,                      
            validation_data=(X_val, y_val, w_val),      
            epochs=epochs, batch_size=batch_size,
            # drop class_weight when using sample_weight, or multiply into w_train if you want both
            callbacks=cbs, verbose=1
        )
        return history, X_test, y_test, X_val, y_val


    def _choose_thr_from_val(self,y_val, p_val, mode="balanced_accuracy", target_recall=0.65):
        pr, rc, th = precision_recall_curve(y_val, p_val)  # th aligns with pr[1:], rc[1:]
        if th.size == 0: 
            return 0.5
        if mode == "f1":
            f1s = (2*pr[1:]*rc[1:])/(pr[1:]+rc[1:]+1e-9)
            return float(th[np.argmax(f1s)])
        if mode == "recall":
            i = int(np.argmin(np.abs(rc[1:] - target_recall)))
            return float(th[i])
        # default: balanced accuracy
        scores = [balanced_accuracy_score(y_val, (p_val>=t).astype(int)) for t in th]
        return float(th[int(np.argmax(scores))])

    def evaluateModel(self, model, X_val, y_val, X_test, y_test, mode="balanced_accuracy"):
        # ----- 1) Keras-compiled metrics on TEST (threshold-free) -----
        try:
            compile_report = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        except TypeError:
            # TF versions without return_dict
            res = model.evaluate(X_test, y_test, verbose=0)
            names = model.metrics_names
            if not isinstance(res, (list, tuple)):
                res = [res]
            compile_report = {k: float(v) for k, v in zip(names, res)}
        print("Test -> " + " | ".join(f"{k}: {v:.4f}" for k, v in compile_report.items()))

        # ----- 2) Thresholded metrics (pick thr on VAL, apply to TEST) -----
        p_val  = model.predict(X_val,  verbose=0).ravel()
        p_test = model.predict(X_test, verbose=0).ravel()

        thr = self._choose_thr_from_val(y_val, p_val, mode=mode)
        yhat_val  = (p_val  >= thr).astype(int)
        yhat_test = (p_test >= thr).astype(int)

        # threshold-free summaries (nice to log)
        val_roc = roc_auc_score(y_val, p_val)
        val_ap  = average_precision_score(y_val, p_val)
        tst_roc = roc_auc_score(y_test, p_test)
        tst_ap  = average_precision_score(y_test, p_test)
        print(f"[VAL]  ROC-AUC={val_roc:.3f} | PR-AUC={val_ap:.3f}")
        print(f"[TEST] ROC-AUC={tst_roc:.3f} | PR-AUC={tst_ap:.3f}")

        # thresholded summaries
        print(f"Chosen threshold ({mode} on VAL): {thr:.4f}")
        print("[VAL]  acc:", accuracy_score(y_val,yhat_val),
            "bal_acc:", balanced_accuracy_score(y_val,yhat_val),
            "f1:", f1_score(y_val,yhat_val))
        print(confusion_matrix(y_val,yhat_val))
        print(classification_report(y_val,yhat_val, digits=3))

        print("[TEST] acc:", accuracy_score(y_test,yhat_test),
            "bal_acc:", balanced_accuracy_score(y_test,yhat_test),
            "f1:", f1_score(y_test,yhat_test))
        print(confusion_matrix(y_test,yhat_test))
        print(classification_report(y_test,yhat_test, digits=3))

        return {
            "compile_report": compile_report,
            "threshold": thr,
            "val":  {"roc_auc": float(val_roc), "pr_auc": float(val_ap)},
            "test": {"roc_auc": float(tst_roc), "pr_auc": float(tst_ap),
                    "acc": float(accuracy_score(y_test,yhat_test)),
                    "bal_acc": float(balanced_accuracy_score(y_test,yhat_test)),
                    "f1": float(f1_score(y_test,yhat_test))}
        }
    
    def predict_from_cache(self, model, cache, agg="topk_mean", k=3, threshold=None):
        out = []
        thr = self.op_threshold if threshold is None else float(threshold)
        for tid, entry in cache.items():
            segs = entry["segs"]
            probs = model.predict(segs, verbose=0).ravel()
            score = self._aggregate_probs(probs, mode=agg, k=k)
            label = "TRANSIT" if score >= thr else "NO_TRANSIT"
            out.append({"target_id": tid, "mission": entry.get("mission"),
                        "label": label, "score": score, "n_segments": entry["n_segments"]})
            print(f"{tid}\t{label}\tscore={score:.3f}\tsegs={entry['n_segments']}")
        return pd.DataFrame(out)
    
    def star_level_from_dataset(self, model, X_subset, groups_subset, agg="topk_mean", k=3, threshold=None):
        thr = self.op_threshold if threshold is None else float(threshold)
        out = []
        uniq = np.unique(groups_subset)
        for g in uniq:
            mask = (groups_subset == g)
            segs = X_subset[mask]
            probs = model.predict(segs, verbose=0).ravel()
            score = self._aggregate_probs(probs, mode=agg, k=k)
            label = "TRANSIT" if score >= thr else "NO_TRANSIT"
            out.append({"target_id": g, "label": label, "score": score, "n_segments": int(mask.sum())})
            print(f"{g}\t{label}\tscore={score:.3f}\tsegs={int(mask.sum())}")
        return pd.DataFrame(out)

