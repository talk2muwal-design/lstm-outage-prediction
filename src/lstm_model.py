"""LSTMModel: stacked two-layer LSTM classifier for outage-occurrence
prediction. Corresponds to the LSTMModel class in Chapter 3's Class
Diagram (3.2.3.2: layers, threshold, build(), train(), predict(),
evaluate()) and Sections 3.3.5 (architecture) and 3.4.4 (training
configuration).

Architecture (3.3.5):
  Input(24, n_features) -> LSTM(64, return_sequences=True) -> Dropout(0.2)
  -> LSTM(32) -> Dense(16, ReLU) -> Dense(1, sigmoid)

Training (3.4.4): Adam optimizer, binary cross-entropy loss, AUC as
primary training metric, max 50 epochs, batch size 64, early stopping
(patience 10, monitors validation loss). NFR2: fixed random seed 42.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers as klayers, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

RANDOM_STATE = 42
MODEL_NAME = "AdaptGuard"
MODEL_VERSION = "1.0"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

MAX_EPOCHS = 50
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 10


class LSTMModel:
    name = MODEL_NAME
    version = MODEL_VERSION

    def __init__(self, n_features, window=24, classification_threshold=0.5):
        self.n_features = n_features
        self.window = window
        self.threshold = classification_threshold
        self.layers = ["LSTM(64, return_sequences=True)", "Dropout(0.2)",
                        "LSTM(32)", "Dense(16, relu)", "Dense(1, sigmoid)"]
        self.model = self.build()
        self.history_ = None

    def build(self):
        model = Sequential([
            klayers.Input(shape=(self.window, self.n_features)),
            klayers.LSTM(64, return_sequences=True),
            klayers.Dropout(0.2),
            klayers.LSTM(32),
            klayers.Dense(16, activation="relu"),
            klayers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=MAX_EPOCHS, verbose=0):
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
        self.history_ = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=BATCH_SIZE,
            callbacks=[es], verbose=verbose,
        )
        return self.history_

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def select_threshold(self, X_val, y_val, thresholds=None):
        """Sets self.threshold to whichever value maximizes F1 on a
        validation set, rather than leaving it fixed at the default 0.5
        -- standard practice under class imbalance, where 0.5 can push
        precision/recall to zero even when AUC-ROC (threshold-independent)
        is healthy."""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)
        y_prob = self.predict_proba(X_val)
        best_t, best_f1 = self.threshold, -1.0
        for t in thresholds:
            f1 = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = float(best_t)
        return self.threshold

    def evaluate(self, X, y):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= self.threshold).astype(int)
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1-Score": f1_score(y, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan"),
        }
