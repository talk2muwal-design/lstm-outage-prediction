"""Preprocessor: IQR outlier treatment and Min-Max scaling, matching the
Preprocessor class in Chapter 3's Class Diagram (3.2.3.2: scaler,
params, process(), save_params()) and Sections 3.3.1 / 3.3.3.

Scaling is applied via a separate `scale()` call made AFTER the
chronological train/test split (see main.py), so that scaling
parameters are computed exclusively from the training partition (3.3.3)
even though `clean()` runs on the full daily panel beforehand.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.params = {}
        self.iqr_bounds = {}

    def clean(self, df, numeric_cols, fit=True):
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for c in numeric_cols:
            if df[c].isna().any():
                if fit:
                    self.params[f"{c}_fallback_median"] = float(df[c].median())
                df[c] = df[c].fillna(self.params.get(f"{c}_fallback_median", 0.0))

            if fit:
                q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                iqr = q3 - q1
                self.iqr_bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            lower, upper = self.iqr_bounds[c]
            df[c] = df[c].clip(lower=lower, upper=upper)
        return df

    def scale(self, df, feature_cols, fit=True):
        df = df.copy()
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def save_params(self, path="outputs/preprocessor_params.joblib"):
        dump({
            "scaler": self.scaler,
            "params": self.params,
            "iqr_bounds": self.iqr_bounds,
        }, path)
