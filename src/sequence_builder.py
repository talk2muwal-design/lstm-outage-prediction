"""SequenceBuilder: transforms the cleaned, feature-engineered DataFrame
into fixed-length overlapping (window, n_features) input sequences for
LSTM ingestion, per Chapter 3 Section 3.4.3. Adapted to DAILY resolution
(see feature_engineer.py's module docstring for why): a sliding window
of 7 days (1-week lookback) generates each input/target pair; the
chronologically latest 20% of sequences are reserved as the held-out
test set; SMOTE is applied to the training partition only, to address
class imbalance.
"""

import numpy as np
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
WINDOW = 7
TEST_FRACTION = 0.20


class SequenceBuilder:
    def __init__(self, window=WINDOW, test_fraction=TEST_FRACTION):
        self.window = window
        self.test_fraction = test_fraction

    def build_sequences(self, df, feature_cols, target_col="target"):
        """Per-feeder sliding windows, sorted chronologically by each
        sequence's final (most recent) timestamp."""
        X_list, y_list, ts_list, seg_list = [], [], [], []
        for seg_id, grp in df.groupby("segment_id"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            feats = grp[feature_cols].to_numpy(dtype=float)
            targets = grp[target_col].to_numpy(dtype=int)
            timestamps = grp["timestamp"].to_numpy()

            n = len(grp)
            for end in range(self.window - 1, n):
                start = end - self.window + 1
                X_list.append(feats[start:end + 1])
                y_list.append(targets[end])
                ts_list.append(timestamps[end])
                seg_list.append(seg_id)

        X = np.stack(X_list)
        y = np.array(y_list)
        ts = np.array(ts_list)
        seg = np.array(seg_list)

        order = np.argsort(ts)
        return X[order], y[order], ts[order], seg[order]

    def chronological_split(self, X, y, ts, seg):
        n_test = int(round(len(X) * self.test_fraction))
        n_train = len(X) - n_test
        return (X[:n_train], y[:n_train], ts[:n_train], seg[:n_train],
                X[n_train:], y[n_train:], ts[n_train:], seg[n_train:])

    @staticmethod
    def apply_smote(X_train, y_train):
        """SMOTE requires 2D input -- flatten each (window, n_features)
        sequence, resample, then reshape back. k_neighbors is capped to
        what the minority class actually has available, since SMOTE's
        default (5) errors out on small folds with fewer minority
        samples than that."""
        n, w, f = X_train.shape
        X_flat = X_train.reshape(n, w * f)
        n_minority = int(min(np.bincount(y_train.astype(int))))
        if len(np.unique(y_train)) < 2 or n_minority < 2:
            return X_train, y_train  # SMOTE undefined; nothing to do
        k_neighbors = min(5, n_minority - 1)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X_flat, y_train)
        return X_res.reshape(-1, w, f), y_res
