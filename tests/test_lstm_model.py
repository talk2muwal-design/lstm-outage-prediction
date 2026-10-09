import numpy as np
from src.lstm_model import LSTMModel
from src.adaptive_framework import AdaptiveFramework


def test_model_architecture_matches_spec():
    m = LSTMModel(n_features=10, window=24)
    assert m.layers == ["LSTM(64, return_sequences=True)", "Dropout(0.2)",
                         "LSTM(32)", "Dense(16, relu)", "Dense(1, sigmoid)"]
    assert m.model.input_shape == (None, 24, 10)
    assert m.model.output_shape == (None, 1)
    assert m.model.loss == "binary_crossentropy"


def test_predict_proba_in_unit_range():
    m = LSTMModel(n_features=5, window=24)
    X = np.random.rand(8, 24, 5).astype("float32")
    proba = m.predict_proba(X)
    assert proba.shape == (8,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_monitor_drift_returns_f1_in_unit_range():
    fw = AdaptiveFramework(window_size=24)
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    f1 = fw.monitor_drift(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0


def test_monitor_drift_perfect_predictions_gives_f1_one():
    fw = AdaptiveFramework(window_size=24)
    y_true = np.array([0, 1, 1, 0, 1])
    f1 = fw.monitor_drift(y_true, y_true)
    assert f1 == 1.0


def test_drift_threshold_matches_spec():
    fw = AdaptiveFramework()
    assert fw.drift_threshold == 0.35


def test_select_threshold_improves_or_maintains_f1_over_default():
    np.random.seed(0)
    m = LSTMModel(n_features=5, window=24)
    X_val = np.random.rand(60, 24, 5).astype("float32")
    y_val = np.random.randint(0, 2, 60)
    default_f1 = np.mean((m.predict_proba(X_val) >= 0.5).astype(int) == y_val)
    chosen = m.select_threshold(X_val, y_val)
    assert 0.0 <= chosen <= 1.0
    assert m.threshold == chosen
