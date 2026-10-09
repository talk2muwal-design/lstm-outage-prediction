"""AdaptiveFramework: orchestrates 5-fold time-series cross-validation
(3.4.4), the deployment simulation over sequential data batches, rolling
F1 drift monitoring, and retrain-triggering. Corresponds to the
AdaptiveFramework class in Chapter 3's Class Diagram (3.2.3.2:
window_size, drift_threshold, run_simulation(), monitor_drift(),
retrain()), the Activity Diagram (3.2.3.3), and the Sequence Diagram
(3.2.3.4: predict -> monitor_drift -> retrain if rolling F1 < 0.35).
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

from .lstm_model import LSTMModel, RANDOM_STATE
from .sequence_builder import SequenceBuilder

N_CV_SPLITS = 5
DRIFT_F1_THRESHOLD = 0.35  # Sequence Diagram 3.2.3.4 / Activity Diagram 3.2.3.3
N_DEPLOYMENT_BATCHES = 5


class AdaptiveFramework:
    drift_threshold = DRIFT_F1_THRESHOLD

    def __init__(self, window_size=24, drift_threshold=DRIFT_F1_THRESHOLD,
                 n_cv_splits=N_CV_SPLITS, epochs=None):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.n_cv_splits = n_cv_splits
        self.epochs = epochs  # None -> LSTMModel's default (spec: 50)

    # ------------------------------------------------------------------
    # 3.4.4: 5-fold time-series cross-validation.
    # X, y are the SCALED, PRE-SMOTE training pool -- SMOTE is applied
    # fresh to each fold's training slice only, so every fold's
    # validation slice stays at the real (imbalanced) class distribution
    # the model will actually be evaluated and deployed on.
    # ------------------------------------------------------------------
    def time_series_cross_validate(self, X, y, n_features):
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr_res, y_tr_res = SequenceBuilder.apply_smote(X[train_idx], y[train_idx])
            model = LSTMModel(n_features=n_features, window=self.window_size)
            train_kwargs = {} if self.epochs is None else {"epochs": self.epochs}
            model.train(X_tr_res, y_tr_res, X[val_idx], y[val_idx], **train_kwargs)
            model.select_threshold(X[val_idx], y[val_idx])
            metrics = model.evaluate(X[val_idx], y[val_idx])
            metrics["fold"] = fold
            fold_metrics.append(metrics)
        return fold_metrics

    # ------------------------------------------------------------------
    # Deployment simulation: sequential batches, rolling F1, retraining.
    # X_train/y_train is the SCALED, PRE-SMOTE training pool -- SMOTE is
    # applied inside retrain() to the training slice only, each time.
    # ------------------------------------------------------------------
    def run_simulation(self, model, X_test, y_test, X_train, y_train,
                        n_batches=N_DEPLOYMENT_BATCHES):
        batch_bounds = np.array_split(np.arange(len(X_test)), n_batches)
        rolling_f1_history = []
        retrain_events = []
        X_pool, y_pool = X_train.copy(), y_train.copy()

        for b, idx in enumerate(batch_bounds, start=1):
            X_batch, y_batch = X_test[idx], y_test[idx]
            y_pred = model.predict(X_batch)
            rolling_f1 = self.monitor_drift(y_batch, y_pred)
            rolling_f1_history.append(rolling_f1)

            triggered = rolling_f1 < self.drift_threshold
            if triggered:
                X_pool = np.concatenate([X_pool, X_batch])
                y_pool = np.concatenate([y_pool, y_batch])
                model = self.retrain(model, X_pool, y_pool)
                retrain_events.append(b)

        return {
            "rolling_f1_history": rolling_f1_history,
            "retrain_batches": retrain_events,
            "n_batches": n_batches,
            "final_model": model,
        }

    def monitor_drift(self, y_true_batch, y_pred_batch):
        """Rolling F1-score for the current deployment batch."""
        return f1_score(y_true_batch, y_pred_batch, zero_division=0)

    def retrain(self, model, X_pool, y_pool):
        """Retrains on the combined (original train + newly seen) pool,
        which is kept in its natural (pre-SMOTE) class distribution.
        SMOTE is applied only to the slice actually used for gradient
        updates; the tail slice carved off for early-stopping/threshold
        selection is left at the real imbalanced distribution."""
        n_val = max(1, int(0.1 * len(X_pool)))
        X_tr_raw, y_tr_raw = X_pool[:-n_val], y_pool[:-n_val]
        X_val, y_val = X_pool[-n_val:], y_pool[-n_val:]
        X_tr_res, y_tr_res = SequenceBuilder.apply_smote(X_tr_raw, y_tr_raw)

        new_model = LSTMModel(n_features=model.n_features, window=model.window,
                               classification_threshold=model.threshold)
        train_kwargs = {} if self.epochs is None else {"epochs": self.epochs}
        new_model.train(X_tr_res, y_tr_res, X_val, y_val, **train_kwargs)
        new_model.select_threshold(X_val, y_val)
        return new_model
