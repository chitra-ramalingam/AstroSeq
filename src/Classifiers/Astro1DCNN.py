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

    # ---------- data IO ----------
    def _row_to_target(self, row):
        """
        Prefer TIC id if present; fallback to TOI or hostname.
        """
        # many catalogs store TIC id under 'tid' (int) or 'TIC ID' (str)
        for key in ["tid", "TIC ID", "tic_id"]:
            if key in row and pd.notna(row[key]):
                try:
                    tid = int(row[key])
                    return f"TIC {tid}"
                except Exception:
                    pass
        # try TOI (e.g. 'TOI-1234')
        for key in ["toi", "TOI"]:
            if key in row and pd.notna(row[key]):
                return f"{row[key]}"
        # fallback to hostname (e.g. 'BD+20 594')
        if "hostname" in row and pd.notna(row["hostname"]):
            return row["hostname"]
        return None

    def fetch_flux(self, target):
        """
        Download, remove NaNs, and normalize flux (no plotting).
        Returns np.array or None on failure.
        """
        try:
            print(f"Fetching light curve for {target}...")
            sr = lk.search_lightcurve(target, mission=self.mission, author=self.author)
            if len(sr) == 0:
                return None
            lc = sr.download().remove_nans().normalize()
            return lc.flux.value.astype(np.float32)
        except Exception:
            return None

    # ---------- preprocessing ----------
    def segment_lightcurve(self, flux):
        """
        Chop into non-overlapping windows; z-score per-segment for stability.
        """
        w = self.window
        n = len(flux) // w
        if n == 0:
            return np.empty((0, w), dtype=np.float32)
        segs = flux[: n*w].reshape(n, w)
        # per-segment standardization (avoid div by 0)
        mu = segs.mean(axis=1, keepdims=True)
        sd = segs.std(axis=1, keepdims=True) + 1e-6
        segs = (segs - mu) / sd
        return segs.astype(np.float32)

    def build_from_csv(self, csv_path, min_groups_per_class=8, max_groups_per_class=20):
        df = pd.read_csv(csv_path)

        # --- Label by discovery method (NOT by TESS facility) ---
        def label_from_method(m):
            m = str(m).lower()
            return 1 if ("transit" in m) else 0

        df["label"] = df["discoverymethod"].apply(label_from_method)

        # Drop dups by target identity
        subset_cols = [c for c in ["tid","hostname","toi"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)

        # Split rows by label (groups = targets/stars)
        pos_rows = df[df["label"] == 1].copy()
        neg_rows = df[df["label"] == 0].copy()

        # Shuffle
        pos_rows = pos_rows.sample(frac=1, random_state=42)
        neg_rows = neg_rows.sample(frac=1, random_state=42)

        # Cap number of distinct stars per class (groups)
        pos_rows = pos_rows.head(max_groups_per_class)
        neg_rows = neg_rows.head(max_groups_per_class)

        # Fetch + segment per target, build groups
        segs_all, labels_all, groups_all = [], [], []
        got_pos_groups = set()
        got_neg_groups = set()

        def add_target(row):
            target = self._row_to_target(row)
            if not target: 
                return False
            flux = self.fetch_flux(target)
            if flux is None:
                return False
            segs = self.segment_lightcurve(flux)
            if len(segs) == 0:
                return False
            lbl = int(row["label"])
            segs_all.append(segs)
            labels_all.append(np.full((len(segs),), lbl, dtype=np.int32))
            groups_all.append(np.array([target]*len(segs), dtype=object))
            if lbl == 1: got_pos_groups.add(target)
            else:        got_neg_groups.add(target)
            return True

        # Add positives then negatives until we meet the min per class
        for _, r in pos_rows.iterrows():
            if len(got_pos_groups) >= min_groups_per_class: break
            add_target(r)
        for _, r in neg_rows.iterrows():
            if len(got_neg_groups) >= min_groups_per_class: break
            add_target(r)

        # If still short, keep filling from remaining rows
        if len(got_pos_groups) < min_groups_per_class:
            for _, r in pos_rows.iterrows():
                if r.get("hostname") in got_pos_groups: continue
                if len(got_pos_groups) >= min_groups_per_class: break
                add_target(r)
        if len(got_neg_groups) < min_groups_per_class:
            for _, r in neg_rows.iterrows():
                if r.get("hostname") in got_neg_groups: continue
                if len(got_neg_groups) >= min_groups_per_class: break
                add_target(r)

        # Safety check
        if not segs_all or len(got_pos_groups) == 0 or len(got_neg_groups) == 0:
            raise RuntimeError("Could not build both classes. Relax caps or check catalog.")

        X = np.vstack(segs_all).astype(np.float32).reshape(-1, self.window, 1)
        y = np.concatenate(labels_all).astype(np.int32)
        groups = np.concatenate(groups_all)

        # Shuffle consistently
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        X, y, groups = X[idx], y[idx], groups[idx]

        # Diagnostics
        u, c = np.unique(y, return_counts=True)
        print("Class counts:", dict(zip(u, c)))
        print("Unique stars per class (pos, neg):", len(got_pos_groups), len(got_neg_groups))
        return X, y, groups

    # ---------- model ----------
    def declareModel(self):
        w = self.window
        model = models.Sequential([
            layers.Conv1D(16, 5, activation='relu', input_shape=(w, 1)),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def trainModel(self, model, X, y, groups , epochs=20, batch_size=32):
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = self.stratified_group_split(y, groups, test_size=0.2, seed=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42, stratify=y
        # )
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=callbacks, verbose=1)
        return history, X_test, y_test

    def groupSplit(self, X, y, groups, test_size=0.2):
        from sklearn.model_selection import GroupShuffleSplit

# Suppose you built these:
# X.shape == (N, window, 1)
# y.shape == (N,)
# groups.shape == (N,)  # e.g., "TIC 141914082" repeated for all its segments

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    def stratified_group_split(self, y, groups, test_size=0.2, seed=42):
        rng = np.random.default_rng(seed)
        y = np.asarray(y); groups = np.asarray(groups)

        # one label per group (we constructed data that way)
        uniq = np.unique(groups)
        # label of each group = label of any of its segments
        grp_lbl = np.array([y[groups == g][0] for g in uniq])

        train_groups, test_groups = [], []
        for lbl in [0, 1]:
            g = uniq[grp_lbl == lbl]
            rng.shuffle(g)
            n_test = max(1, int(len(g) * test_size))
            test_groups += list(g[:n_test])
            train_groups += list(g[n_test:])

        test_mask = np.isin(groups, test_groups)
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where(test_mask)[0]
        return train_idx, test_idx



    def evaluateModel(self, model, X_test, y_test):
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.3f}")
        return acc
