"""FeatureEngineer: aggregates the real event-level interruption log
into a DAILY-RESOLUTION per-feeder time series, then computes rolling
statistical aggregates, lag features, and calendar variables -- the
daily-resolution counterpart of FR3 / Chapter 3's FeatureEngineer class.

*** ADAPTATION FROM CHAPTER 3'S SPEC, DOCUMENTED HONESTLY ***
Chapter 3 specifies 5-minute-resolution SCADA telemetry with a 1-hour
prediction horizon (Section 3.3.2, 3.4.3). The real YEDC event log has
no continuous SCADA readings at all -- only discrete interruption
records with exact start/end timestamps. The finest resolution
honestly derivable from this real data is DAILY: for each feeder and
calendar day, aggregate that day's real interruption events into
features (event count, total downtime minutes, load loss, etc.), then
predict whether an outage will occur on the following day. This keeps
every number in the pipeline grounded in real, non-synthetic records,
at the cost of a coarser resolution and horizon than Chapter 3's
literal specification. All architectural elements (LSTM design,
training procedure, adaptive framework) remain exactly as specified;
only the time resolution and horizon are adapted to what the real data
actually supports.
"""

import numpy as np
import pandas as pd

ROLL_WINDOWS = (3, 7, 14)
LAGS = (1, 2, 3)

INTERRUPTION_TYPES = ["O/C and E/F", "Inst. E/F", "B/F YEDC", "E/F", "O/C", "Inst. O/C"]


class FeatureEngineer:
    def __init__(self, window=7):
        self.window = window  # sequence length used downstream (days)

    def build_daily_panel(self, events):
        """One row per (feeder, calendar day) across the full observed
        span, with zero-filled days where no event occurred."""
        feeders = sorted(events["Affected Feeder"].unique())
        all_days = pd.date_range(
            events["Date Out"].min().normalize(),
            events["Date Out"].max().normalize(),
            freq="D",
        )
        events = events.copy()
        events["day"] = events["Date Out"].dt.normalize()

        panels = []
        for feeder in feeders:
            fe_events = events[events["Affected Feeder"] == feeder]
            daily = pd.DataFrame({"day": all_days})
            daily["feeder"] = feeder

            grp = fe_events.groupby("day")
            daily["event_count"] = daily["day"].map(grp.size()).fillna(0).astype(int)
            daily["total_downtime_min"] = daily["day"].map(
                grp["Duration_Minutes"].sum()
            ).fillna(0.0)
            daily["mean_load_loss_mw"] = daily["day"].map(
                grp["Load Loss (MW)"].mean()
            ).fillna(0.0)
            daily["max_load_loss_mw"] = daily["day"].map(
                grp["Load Loss (MW)"].max()
            ).fillna(0.0)

            for itype in INTERRUPTION_TYPES:
                col = f"type_{itype.replace(' ', '_').replace('/', '')}"
                counts = fe_events[fe_events["Interruption Type (*)"] == itype].groupby("day").size()
                daily[col] = daily["day"].map(counts).fillna(0).astype(int)

            panels.append(daily)

        panel = pd.concat(panels, ignore_index=True).sort_values(
            ["feeder", "day"]
        ).reset_index(drop=True)
        return panel

    def engineer(self, panel):
        df = panel.copy()
        g = df.groupby("feeder")

        base_cols = ["event_count", "total_downtime_min", "mean_load_loss_mw"]

        for c in base_cols:
            for w in ROLL_WINDOWS:
                df[f"{c}_roll_sum_{w}"] = g[c].transform(
                    lambda s: s.rolling(w, min_periods=1).sum()
                )
                df[f"{c}_roll_mean_{w}"] = g[c].transform(
                    lambda s: s.rolling(w, min_periods=1).mean()
                )

        for c in base_cols:
            for lag in LAGS:
                df[f"{c}_lag_{lag}"] = g[c].shift(lag).fillna(0.0)

        # Days since last outage (per feeder), causal (only past info)
        def days_since_last(s):
            out = np.zeros(len(s))
            last_idx = -np.inf
            for i, v in enumerate(s.to_numpy()):
                out[i] = i - last_idx if np.isfinite(last_idx) else 999
                if v > 0:
                    last_idx = i
            return out
        df["days_since_last_outage"] = g["event_count"].transform(days_since_last)

        # Calendar variables
        df["day_of_week"] = df["day"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["day_of_month"] = df["day"].dt.day

        # Forward-looking target (3.3.2 adapted to 1-day horizon): did an
        # outage occur on the FOLLOWING calendar day for this feeder?
        df["target"] = g["event_count"].shift(-1).fillna(0).gt(0).astype(int)

        # Drop the last day per feeder (no next-day label available)
        df = df.sort_values(["feeder", "day"]).reset_index(drop=True)
        is_last_day = df.groupby("feeder")["day"].transform("max") == df["day"]
        df = df[~is_last_day].reset_index(drop=True)

        # Align column names with SequenceBuilder's expectations
        df = df.rename(columns={"feeder": "segment_id", "day": "timestamp"})
        return df

    @staticmethod
    def feature_columns(df):
        exclude = {"timestamp", "segment_id", "target"}
        return [c for c in df.columns if c not in exclude
                and pd.api.types.is_numeric_dtype(df[c])]
