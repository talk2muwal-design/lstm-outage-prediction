import numpy as np
from src.feature_engineer import FeatureEngineer
from src.preprocessor import Preprocessor
from src.sequence_builder import SequenceBuilder


def _build(tiny_events_df, window=7):
    fe = FeatureEngineer(window=window)
    panel = fe.build_daily_panel(tiny_events_df)
    pre = Preprocessor()
    clean = pre.clean(panel, ["mean_load_loss_mw", "max_load_loss_mw"], fit=True)
    eng = fe.engineer(clean)
    feat_cols = FeatureEngineer.feature_columns(eng)
    sb = SequenceBuilder(window=window)
    return sb, eng, feat_cols


def test_sequences_have_expected_shape(tiny_events_df):
    sb, eng, feat_cols = _build(tiny_events_df, window=5)
    X, y, ts, seg = sb.build_sequences(eng, feat_cols)
    assert X.shape[1] == 5
    assert X.shape[2] == len(feat_cols)
    assert X.shape[0] == len(y) == len(ts) == len(seg)


def test_sequences_sorted_chronologically(tiny_events_df):
    sb, eng, feat_cols = _build(tiny_events_df, window=5)
    X, y, ts, seg = sb.build_sequences(eng, feat_cols)
    assert np.all(ts[:-1] <= ts[1:])


def test_chronological_split_preserves_order_and_size(tiny_events_df):
    sb, eng, feat_cols = _build(tiny_events_df, window=5)
    X, y, ts, seg = sb.build_sequences(eng, feat_cols)
    (X_tr, y_tr, ts_tr, seg_tr,
     X_te, y_te, ts_te, seg_te) = sb.chronological_split(X, y, ts, seg)
    assert len(X_tr) + len(X_te) == len(X)
    assert ts_tr.max() <= ts_te.min()


def test_smote_balances_training_classes(tiny_events_df):
    sb, eng, feat_cols = _build(tiny_events_df, window=5)
    X, y, ts, seg = sb.build_sequences(eng, feat_cols)
    if y.sum() < 2:
        return
    X_res, y_res = SequenceBuilder.apply_smote(X, y)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
