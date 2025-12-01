import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.ensemble import IsolationForest


class RegressorNet:
    def __init__(self):
        pass


    def runModel(self, embedding_path, star_scores_csv):
        # 1) Load embeddings
        data = np.load(embedding_path, allow_pickle=True)
        star_ids  = data["star_ids"]
        star_vecs = data["star_vecs"]

        # 2) Read scores CSV with NO assumptions about header or columns
        # sep=None + engine="python" lets pandas guess delimiters (comma, tab, spaces)
        raw = pd.read_csv(star_scores_csv, header=None, sep=None, engine="python")

        # First column = target, last column = raw score
        targets = raw.iloc[:, 0].astype(str).str.strip()
        raw_scores = raw.iloc[:, -1]

        # Convert scores to numeric, drop non-numeric (e.g. "all segments non-finite")
        scores = pd.to_numeric(raw_scores, errors="coerce")
        mask = scores.notna()

        targets = targets[mask].reset_index(drop=True)
        scores  = scores[mask].reset_index(drop=True)

        # Now we have one Series of clean targets and one Series of numeric scores
        print("After cleaning scores file:")
        print("  rows:", len(scores))
        print("  example:", list(zip(targets.head(3), scores.head(3))))

        # If there are multiple rows per target, collapse to one score (mean)
        scores_df = pd.DataFrame({"target": targets, "star_score": scores})
        scores_series = (
            scores_df
            .groupby("target")["star_score"]
            .mean()  
        )

        # 3) X, y by matching star_ids vector embeds to this scores_series
        X, y = [], []
        for sid, vec in zip(star_ids, star_vecs):
            sid_str = str(sid).strip()
            if sid_str in scores_series.index:
                X.append(vec)
                y.append(scores_series.loc[sid_str])

        X = np.array(X)
        y = np.array(y, dtype=np.float32)

        print("Final training shapes:")
        print("  X:", X.shape)
        print("  y:", y.shape)


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=500,
            random_state=42,
        )
        reg.fit(X_train, y_train)

        pred = reg.predict(X_test)
        print("R^2:", r2_score(y_test, pred))
        smooth_scores = reg.predict(star_vecs)  # shape (num_stars,)
        np.savez(
                "star_embeddings_with_scores.npz",
            star_ids=star_ids,
            star_vecs=star_vecs,
            smooth_scores=smooth_scores
        )
        plt.hist(smooth_scores, bins=50)
        plt.xlabel("smooth star score")
        plt.ylabel("count")
        plt.title("Distribution of star scores from regressor")
        plt.show()
        thr = np.quantile(smooth_scores, 0.99)
        high_mask = smooth_scores >= thr
        print("Num high-score stars:", high_mask.sum())
        return smooth_scores, star_ids, star_vecs

    def calculate_cosinedistance(self, smooth_scores, star_ids, star_vecs):
        # find an index of a high-score star
        idx0 = np.argmax(smooth_scores)  # absolute highest
        target_id = star_ids[idx0]
        print("Using star:", target_id, "score:", smooth_scores[idx0])

        # compute neighbours in embedding space
        dists = cosine_distances(star_vecs[idx0:idx0+1], star_vecs)[0]
        order = np.argsort(dists)

        # show top neighbours that also have high scores
        k = 15
        print(f"Nearest neighbours to {target_id}:")
        for i in order[1:k+1]:
            print(f"  {star_ids[i]}  dist={dists[i]:.4f}  score={smooth_scores[i]:.3f}")

    def find_outliers(self, smooth_scores, star_ids, star_vecs) -> pd.DataFrame: 
        N = 200 
        top_idx = np.argsort(smooth_scores)[-N:]      # indices of top N stars
        top_idx = top_idx[::-1]                       

        top_ids    = star_ids[top_idx]
        top_scores = smooth_scores[top_idx]

        for sid, sc in zip(top_ids[:20], top_scores[:20]):
            print(sid, sc) 
        pass

        H = star_vecs[top_idx]      
        ids_top = star_ids[top_idx]

        iso = IsolationForest(
            n_estimators=200,
            contamination=0.1,   
            random_state=42
        )
        iso.fit(H)

        scores_iso = iso.decision_function(H)  # higher = more normal
        outlier_score = -scores_iso            # higher = more outlier

        order = np.argsort(outlier_score)[::-1]

        top_weird = 20
        print(f"Top {top_weird} weird high-score stars:")
        for i in order[:top_weird]:
            print(ids_top[i], "  score=", top_scores[i], "  outlier_score=", outlier_score[i])
        return self.save_wierd_stars()

    def save_wierd_stars(self) -> pd.DataFrame:
        weird_df = pd.DataFrame({
            "target": [
                "TIC 85459171","EPIC 201182911","TIC 412828754","TIC 119131709",
                "EPIC 205489894","TIC 50575306","HIP 54766","TIC 177612321",
                "EPIC 247281516","TIC 5592720","KIC 10015516","EPIC 211738534",
                "TIC 22560356","KIC 2708156","EPIC 203042994","EPIC 249649721",
                "KIC 9641031","TIC 302477411","KIC 10156064","KIC 10920813"
            ],
            "smooth_score": [
                1.0019772,0.95643747,1.0973542,0.9455979,0.9171698,0.9752532,
                0.9403577,1.0729016,0.94558495,0.9366224,0.91600704,1.1271552,
                1.0095346,0.9223468,0.96217906,0.926942,0.9379699,0.9168802,
                0.92517453,0.9158894
            ],
            "outlier_score": [
                0.1115715,0.08792022,0.06805908,0.0442868,0.03545162,0.03325745,
                0.03284683,0.03170089,0.03081968,0.03030619,0.02649286,0.0264573,
                0.02230827,0.0140081,0.01276873,0.01256233,0.01180361,0.00869482,
                0.00665302,0.00298898
            ]
        })
        print(weird_df)
        weird_df.to_csv("weird_high_score_stars.csv", index=False)
        return weird_df
