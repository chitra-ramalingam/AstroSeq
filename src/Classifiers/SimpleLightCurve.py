import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

class SimpleLightCurve:
    def __init__(self):
        pass

    def normalize(self):
        lc = lk.search_lightcurve("TIC 141914082", mission="TESS", author="SPOC").download().remove_nans()

        # Normalize flux
        flux = lc.normalize().flux.value
        time = lc.time.value

       
        return flux

    def segmentLightCurve(self, flux):
        window = 200  # number of time steps per segment
        segments = []
        for i in range(0, len(flux) - window, window):
            segment = flux[i:i+window]
            segments.append(segment)
        segments = np.array(segments)
        return segments

    def declareModel(self, window):
        model = models.Sequential([
        layers.Conv1D(16, 5, activation='relu', input_shape=(window, 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    

    def trainModel(self, model, segments, labels):
        X = segments.reshape((segments.shape[0], segments.shape[1], 1))
        y = np.array(labels)  # your 0/1 labels

        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
        return history, X_test, y_test

    def evaluateModel(self, model, X_test, y_test):
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('Test Accuracy:', test_acc)

    def plotAll(self, history):
       plt.plot(history.history['accuracy'], label='train')
       plt.plot(history.history['val_accuracy'], label='val')
       plt.legend()
       plt.show()
