import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BinaryEmbeddingsClassifier:
    def __init__(self):
        self.thereshold = 0.5
        pass

    def classify(self, star_embedding_with_scores_path: str,
                 threshold: float = 0.5) -> pd.DataFrame:
        data = np.load(star_embedding_with_scores_path, allow_pickle=True)
        star_ids = data["star_ids"]       # (N,)
        star_vecs = data["star_vecs"]     # (N, emb_dim)
        star_scores = data["smooth_scores"] # (N,)
        threshold = self.thereshold
        y_cls_pred = (star_scores >= threshold).astype(int)

        X = star_vecs
        y = y_cls_pred
        ids = star_ids

        X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
            X, y, ids,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
            X_temp, y_temp, ids_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        star_clf, star_probs= self._build_and_train_model_(star_vecs,
                                 X_train,y_train,
                                X_test,y_test,  
                                 X_val,y_val)
        star_table = pd.DataFrame({
            "star_id": star_ids,
            "score_raw": star_scores,
            "p_highclass": star_probs,
        })
        star_table_sorted = star_table.sort_values("p_highclass", ascending=False)
        print("Top 10 stars by p_highclass:")
        print(star_table_sorted.head(100))
        np.savetxt("binary_star_classification_results.csv",
                   star_table_sorted,
                   delimiter=",",
                   header="star_id,score_raw,p_highclass",
                   fmt=["%s", "%.6f", "%d"],
                   comments=""
                   )


    def _build_and_train_model_(self, star_vecs,
                               X_train, 
                               y_train, 
                               X_test, y_test, X_val, y_val):

        D = star_vecs.shape[1]

        inputs = keras.Input(shape=(D,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)  # star probability

        star_clf = keras.Model(inputs, outputs)

        star_clf.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="AUC"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        )

        history = star_clf.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=128,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_AUC",
                    mode="max",
                    patience=5,
                    restore_best_weights=True,
                )
            ]
        )
        test_metrics = star_clf.evaluate(X_test, y_test, verbose=0)
        print("Test metrics:", dict(zip(star_clf.metrics_names, test_metrics)))
        star_probs = star_clf.predict(star_vecs).ravel()  # shape (N,)
        return star_clf, star_probs
