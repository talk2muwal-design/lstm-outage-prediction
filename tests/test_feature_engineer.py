import numpy as np
import pandas as pd
from src.feature_engineer import FeatureEngineer
from src.preprocessor import Preprocessor


def _build(tiny_events_df):
    fe = FeatureEngineer(window=7)
    panel = fe.build_daily_panel(tiny_events_df)
    pre = Preprocessor()
    clean = pre.clean(panel, ["mean_load_loss_mw", "max_load_loss_mw"], fit=True)
    eng = fe.engineer(clean)
    return fe, eng


def test_daily_panel_has_no_gaps(tiny_events_df):
    fe = FeatureEngineer(window=7)
    panel = fe.build_daily_panel(tiny_events_df)
    for feeder, grp in panel.groupby("feeder"):
        days = grp["day"].sort_values()
        assert (days.diff().dropna() == pd.Timedelta(days=1)).all() if len(days) > 1 else True


def test_iqr_does_not_zero_out_event_counts(tiny_events_df):
    """Regression test for the bug where blanket IQR clipping crushed
    sparse event_count values to zero."""
    fe, eng = _build(tiny_events_df)
    assert eng["event_count"].sum() > 0


def test_target_positive_rate_reasonable(tiny_events_df):
    fe, eng = _build(tiny_events_df)
    assert 0.0 < eng["target"].mean() < 1.0


def test_target_is_next_day_occurrence(tiny_events_df):
    fe, eng = _build(tiny_events_df)
    feeder_a = eng[eng["segment_id"] == "FeederA"].sort_values("timestamp").reset_index(drop=True)
    # event on 2026-01-05 -> the row for 2026-01-04 should have target=1
    row = feeder_a[feeder_a["timestamp"] == pd.Timestamp("2026-01-04")]
    assert len(row) == 1
    assert row["target"].iloc[0] == 1


def test_no_missing_values_in_features(tiny_events_df):
    fe, eng = _build(tiny_events_df)
    feat_cols = FeatureEngineer.feature_columns(eng)
    assert eng[feat_cols].isna().sum().sum() == 0


def test_feature_columns_excludes_identifiers(tiny_events_df):
    fe, eng = _build(tiny_events_df)
    feat_cols = FeatureEngineer.feature_columns(eng)
    for banned in ("timestamp", "segment_id", "target"):
        assert banned not in feat_cols
