import pandas as pd
import pytest


@pytest.fixture
def tiny_events_df():
    """A small real-shaped event log: 2 feeders, a handful of events
    spanning 20 days."""
    rows = [
        {"Affected Feeder": "FeederA", "Interruption Type (*)": "O/C and E/F",
         "Load Loss (MW)": 2.5, "Date Out": pd.Timestamp("2026-01-02"),
         "Duration_Minutes": 30.0},
        {"Affected Feeder": "FeederA", "Interruption Type (*)": "E/F",
         "Load Loss (MW)": 1.2, "Date Out": pd.Timestamp("2026-01-05"),
         "Duration_Minutes": 600.0},
        {"Affected Feeder": "FeederA", "Interruption Type (*)": "O/C and E/F",
         "Load Loss (MW)": 3.0, "Date Out": pd.Timestamp("2026-01-06"),
         "Duration_Minutes": 15.0},
        {"Affected Feeder": "FeederB", "Interruption Type (*)": "B/F YEDC",
         "Load Loss (MW)": 0.5, "Date Out": pd.Timestamp("2026-01-10"),
         "Duration_Minutes": 45.0},
        {"Affected Feeder": "FeederB", "Interruption Type (*)": "O/C",
         "Load Loss (MW)": 1.8, "Date Out": pd.Timestamp("2026-01-15"),
         "Duration_Minutes": 90.0},
        {"Affected Feeder": "FeederB", "Interruption Type (*)": "O/C",
         "Load Loss (MW)": 2.0, "Date Out": pd.Timestamp("2026-01-18"),
         "Duration_Minutes": 20.0},
    ]
    return pd.DataFrame(rows)
