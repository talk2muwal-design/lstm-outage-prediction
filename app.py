"""AdaptGuard interactive dashboard -- a defense-panel-facing front end
for the adaptive LSTM outage-occurrence prediction system.

This app does NOT retrain or re-fit anything. It loads the already
trained model (outputs/adaptguard_model.keras) and the already-fitted
preprocessor/scaler (outputs/preprocessor_params.joblib) exactly as
main.py saved them, then replays steps 1-5 of the real pipeline
(Data Acquisition -> Daily Aggregation -> Preprocessing -> Feature
Engineering -> Sequence Construction) on the real YEDC event log so
every number shown here is genuine and pipeline-consistent -- not
mocked or hand-typed.

Run locally with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.preprocessor import Preprocessor
from src.sequence_builder import SequenceBuilder, WINDOW

OUT_DIR = Path("outputs")
DATA_PATH = "data/yedc_interruption_log.xlsx"
IQR_COLS = ["mean_load_loss_mw", "max_load_loss_mw"]
CORE_COLS = ["event_count", "total_downtime_min", "mean_load_loss_mw", "max_load_loss_mw"]

st.set_page_config(page_title="AdaptGuard — YEDC Outage Prediction",
                    page_icon="⚡", layout="wide")


# ---------------------------------------------------------------------
# Cached loaders -- run once per app session, not on every interaction
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(OUT_DIR / "adaptguard_model.keras")


@st.cache_resource
def load_preprocessor_params():
    return joblib.load(OUT_DIR / "preprocessor_params.joblib")


@st.cache_data
def load_results():
    with open(OUT_DIR / "results.json") as f:
        return json.load(f)


@st.cache_data
def build_pipeline():
    """Replays Chapter 3's pipeline (Steps 1-5) on the real YEDC event
    log, using the ALREADY-FITTED preprocessor/scaler (fit=False
    throughout) so this exactly reproduces what main.py produced --
    nothing is refit or re-estimated here."""
    loader = DataLoader(DATA_PATH)
    events = loader.load()

    fe = FeatureEngineer(window=WINDOW)
    panel = fe.build_daily_panel(events)

    saved = load_preprocessor_params()
    pre = Preprocessor()
    pre.params = saved["params"]
    pre.iqr_bounds = saved["iqr_bounds"]
    pre.scaler = saved["scaler"]

    clean = pre.clean(panel, IQR_COLS, fit=False)
    engineered = fe.engineer(clean)
    feature_cols = FeatureEngineer.feature_columns(engineered)

    sb = SequenceBuilder(window=WINDOW)
    X_all, y_all, ts_all, seg_all = sb.build_sequences(engineered, feature_cols)

    n, w, f = X_all.shape
    X_scaled = pre.scaler.transform(X_all.reshape(-1, f)).reshape(n, w, f)

    (X_train_raw, y_train, ts_train, seg_train,
     X_test_raw, y_test, ts_test, seg_test) = sb.chronological_split(
        X_all, y_all, ts_all, seg_all
    )

    return {
        "events": events,
        "engineered": engineered,
        "feature_cols": feature_cols,
        "X_all": X_all,
        "X_scaled": X_scaled,
        "y_all": y_all,
        "ts_all": ts_all,
        "seg_all": seg_all,
        "n_train": len(X_train_raw),
    }


def select_threshold(_model, X_val, y_val):
    """Same max-F1-on-validation procedure as LSTMModel.select_threshold
    in src/lstm_model.py, applied here to the loaded model so the app
    shows the genuine operating threshold rather than a fixed 0.5."""
    y_prob = _model.predict(X_val, verbose=0).ravel()
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def predict_one(model, scaler, raw_window, feature_cols):
    """raw_window: (window, n_features) UNSCALED array -> scaled -> prob."""
    w, f = raw_window.shape
    scaled = scaler.transform(raw_window).reshape(1, w, f)
    return float(model.predict(scaled, verbose=0).ravel()[0])


model = load_model()
pipe = build_pipeline()
results = load_results()
scaler = load_preprocessor_params()["scaler"]

n_train = pipe["n_train"]
n_val = max(1, int(0.15 * n_train))
X_val = pipe["X_scaled"][n_train - n_val:n_train]
y_val = pipe["y_all"][n_train - n_val:n_train]
default_threshold = select_threshold(model, X_val, y_val)

X_test = pipe["X_scaled"][n_train:]
y_test = pipe["y_all"][n_train:]
y_prob_test = model.predict(X_test, verbose=0).ravel()

st.title("⚡ AdaptGuard — Adaptive LSTM Outage Prediction Dashboard")
st.caption(
    "YEDC Power Distribution Network · Feeder-level, daily-resolution, "
    "1-day-ahead outage prediction · Trained and evaluated on 104 real "
    "logged interruption events (Jan–Apr 2026)"
)

tab_live, tab_scenario, tab_perf, tab_about = st.tabs(
    ["🔴 Live Prediction", "🧪 Scenario Explorer", "📊 Model Performance", "ℹ️ About the Model"]
)

# =======================================================================
# TAB 1 — LIVE PREDICTION
# =======================================================================
with tab_live:
    feeders = sorted(np.unique(pipe["seg_all"]))

    st.subheader("Fleet snapshot — most recent available window per feeder")
    snap_rows = []
    for feeder in feeders:
        idx_f = np.where(pipe["seg_all"] == feeder)[0]
        latest_idx = idx_f[np.argmax(pipe["ts_all"][idx_f])]
        prob = float(model.predict(pipe["X_scaled"][latest_idx:latest_idx + 1],
                                    verbose=0).ravel()[0])
        risk = "HIGH" if prob >= default_threshold else "LOW"
        snap_rows.append({
            "Feeder": feeder,
            "Window ending": pd.Timestamp(pipe["ts_all"][latest_idx]).date(),
            "Predicted outage probability (next day)": f"{prob:.1%}",
            "Risk": risk,
        })
    snap_df = pd.DataFrame(snap_rows)

    def _risk_color(val):
        return "background-color:#ffcccc" if val == "HIGH" else "background-color:#d4f7d4"

    styler = snap_df.style
    style_fn = getattr(styler, "map", None) or styler.applymap
    st.dataframe(style_fn(_risk_color, subset=["Risk"]),
                 width='stretch', hide_index=True)

    st.divider()
    st.subheader("Inspect a specific feeder and window")

    col1, col2 = st.columns([1, 2])
    with col1:
        feeder_sel = st.selectbox("Feeder", feeders, key="live_feeder")

    idx_f = np.where(pipe["seg_all"] == feeder_sel)[0]
    order_f = idx_f[np.argsort(pipe["ts_all"][idx_f])][::-1]  # most recent first
    date_options = [pd.Timestamp(pipe["ts_all"][i]).date() for i in order_f]

    with col2:
        date_sel = st.selectbox(
            "Window end date (prediction is for the day AFTER this date)",
            date_options, key="live_date",
        )

    sel_pos = date_options.index(date_sel)
    idx = order_f[sel_pos]

    prob = float(model.predict(pipe["X_scaled"][idx:idx + 1], verbose=0).ravel()[0])
    pred_label = "Outage predicted" if prob >= default_threshold else "No outage predicted"
    actual = int(pipe["y_all"][idx])
    in_test = idx >= n_train

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted probability", f"{prob:.1%}")
    m2.metric("Classification (threshold = %.2f)" % default_threshold, pred_label)
    m3.metric("Actual outcome (ground truth)", "Outage occurred" if actual else "No outage")
    m4.metric("Data split", "Held-out test" if in_test else "Used in training")

    st.caption(
        "The model never sees 'Actual outcome' — it is shown only so the "
        "panel can compare the prediction against what really happened. "
        "'Held-out test' windows were not used to fit the model or choose "
        "its threshold; they are the basis for the performance figures in "
        "the Model Performance tab."
    )

    st.markdown(f"**7-day feature window for {feeder_sel}, ending {date_sel}:**")
    raw_window = pipe["X_all"][idx].copy()  # (7, n_features), UNSCALED
    window_dates = pd.date_range(end=date_sel, periods=WINDOW, freq="D")

    core_cols = [c for c in CORE_COLS + ["days_since_last_outage"] if c in pipe["feature_cols"]]
    core_idx = [pipe["feature_cols"].index(c) for c in core_cols]

    core_df = pd.DataFrame(raw_window[:, core_idx], columns=core_cols)
    core_df.insert(0, "Date", window_dates.date)
    st.dataframe(core_df, width='stretch', hide_index=True)

    with st.expander("Show all engineered features for this window "
                      f"({len(pipe['feature_cols'])} columns)"):
        full_df = pd.DataFrame(raw_window, columns=pipe["feature_cols"])
        full_df.insert(0, "Date", window_dates.date)
        st.dataframe(full_df, width='stretch', hide_index=True)

# =======================================================================
# TAB 2 — SCENARIO EXPLORER (what-if, interactive sensitivity analysis)
# =======================================================================
with tab_scenario:
    st.subheader("What happens to the prediction if tomorrow's conditions change?")
    st.caption(
        "Pick a real historical window as a starting point, then drag the "
        "sliders to change the MOST RECENT day's raw readings and watch the "
        "model's predicted probability update live. This edits the literal "
        "day-level features the LSTM consumes for that day; the rolling/lag "
        "statistics from earlier days in the window are left at their real "
        "historical values, so treat this as a sensitivity check on "
        "today's signal rather than a fully re-simulated history."
    )

    colf, cold = st.columns([1, 2])
    with colf:
        sc_feeder = st.selectbox("Feeder", feeders, key="sc_feeder")
    idx_f = np.where(pipe["seg_all"] == sc_feeder)[0]
    order_f = idx_f[np.argsort(pipe["ts_all"][idx_f])][::-1]
    sc_date_options = [pd.Timestamp(pipe["ts_all"][i]).date() for i in order_f]
    with cold:
        sc_date = st.selectbox("Baseline window (most recent day of the 7-day window)",
                                sc_date_options, key="sc_date")

    sc_idx = order_f[sc_date_options.index(sc_date)]
    base_window = pipe["X_all"][sc_idx].copy()  # (7, n_features)
    core_idx_map = {c: pipe["feature_cols"].index(c) for c in CORE_COLS
                     if c in pipe["feature_cols"]}

    # Slider ranges drawn from the real observed data, so sliders can't be
    # pushed into physically meaningless territory.
    all_core = pipe["X_all"][:, :, [core_idx_map[c] for c in CORE_COLS]]
    ranges = {
        c: (float(all_core[:, :, i].min()), float(max(all_core[:, :, i].max(), 1.0)))
        for i, c in enumerate(CORE_COLS)
    }
    baseline_vals = {c: float(base_window[-1, core_idx_map[c]]) for c in CORE_COLS}

    st.markdown(f"**Adjusting the last day of the window ({sc_date}) for {sc_feeder}:**")
    s1, s2 = st.columns(2)
    with s1:
        v_event_count = st.slider("Event count (today)", 0, 10,
                                   int(round(baseline_vals["event_count"])), 1,
                                   key="sc_event_count")
        v_downtime = st.slider("Total downtime (minutes, today)", 0, 1440,
                                int(round(baseline_vals["total_downtime_min"])), 10,
                                key="sc_downtime")
    with s2:
        v_mean_loss = st.slider("Mean load loss (MW, today)",
                                 0.0, round(ranges["mean_load_loss_mw"][1], 2),
                                 float(round(baseline_vals["mean_load_loss_mw"], 2)), 0.1,
                                 key="sc_mean_loss")
        v_max_loss = st.slider("Max load loss (MW, today)",
                                0.0, round(ranges["max_load_loss_mw"][1], 2),
                                float(round(baseline_vals["max_load_loss_mw"], 2)), 0.1,
                                key="sc_max_loss")

    scenario_window = base_window.copy()
    overrides = {"event_count": v_event_count, "total_downtime_min": v_downtime,
                 "mean_load_loss_mw": v_mean_loss, "max_load_loss_mw": v_max_loss}
    for c, v in overrides.items():
        scenario_window[-1, core_idx_map[c]] = v

    base_prob = predict_one(model, scaler, base_window, pipe["feature_cols"])
    scenario_prob = predict_one(model, scaler, scenario_window, pipe["feature_cols"])

    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline probability (real data)", f"{base_prob:.1%}")
    r2.metric("Scenario probability (your inputs)", f"{scenario_prob:.1%}",
              delta=f"{(scenario_prob - base_prob):+.1%}")
    r3.metric("Scenario classification (threshold = %.2f)" % default_threshold,
              "Outage predicted" if scenario_prob >= default_threshold else "No outage predicted")

    st.divider()
    st.markdown("**Sensitivity curve — sweep one input across its full observed range**")
    sweep_feature = st.selectbox("Feature to sweep", CORE_COLS, key="sc_sweep_feature")
    lo, hi = ranges[sweep_feature]
    sweep_vals = np.linspace(lo, hi, 25)
    sweep_probs = []
    for v in sweep_vals:
        w = base_window.copy()
        for c, val in overrides.items():
            w[-1, core_idx_map[c]] = val
        w[-1, core_idx_map[sweep_feature]] = v
        sweep_probs.append(predict_one(model, scaler, w, pipe["feature_cols"]))

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_vals, y=sweep_probs, mode="lines+markers",
                                    name="Predicted probability"))
    fig_sweep.add_hline(y=default_threshold, line_dash="dash", line_color="red",
                         annotation_text=f"Classification threshold ({default_threshold:.2f})")
    fig_sweep.add_vline(x=overrides[sweep_feature], line_dash="dot", line_color="gray",
                         annotation_text="Current slider value")
    fig_sweep.update_layout(
        xaxis_title=sweep_feature, yaxis_title="Predicted outage probability",
        yaxis_range=[0, 1], height=400, margin=dict(t=30),
    )
    st.plotly_chart(fig_sweep, width='stretch')
    st.caption(
        "All other sliders are held at their current values while "
        f"**{sweep_feature}** is swept across the full range observed in "
        "the real dataset. A steep curve means the model is highly "
        "sensitive to that input; a flat curve means it is not."
    )

# =======================================================================
# TAB 3 — MODEL PERFORMANCE
# =======================================================================
with tab_perf:
    st.subheader("Held-out test set performance")

    thr = st.slider(
        "Classification threshold (drag to explore the precision / recall trade-off — "
        f"the model's own max-F1 choice is {default_threshold:.2f})",
        0.05, 0.95, default_threshold, 0.01, key="perf_threshold",
    )
    y_pred_at_thr = (y_prob_test >= thr).astype(int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred_at_thr):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred_at_thr, zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred_at_thr, zero_division=0):.3f}")
    c4.metric("F1-Score", f"{f1_score(y_test, y_pred_at_thr, zero_division=0):.3f}")
    c5.metric("AUC-ROC (threshold-independent)",
              f"{roc_auc_score(y_test, y_prob_test):.3f}")
    nfr1_met = roc_auc_score(y_test, y_prob_test) >= 0.70
    st.success("NFR1 (AUC-ROC ≥ 0.70) — MET" if nfr1_met
               else "NFR1 (AUC-ROC ≥ 0.70) — NOT MET")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Confusion matrix at threshold = {thr:.2f}**")
        cm = confusion_matrix(y_test, y_pred_at_thr)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: No outage", "Predicted: Outage"],
            y=["Actual: No outage", "Actual: Outage"],
            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
        ))
        fig_cm.update_layout(height=380, margin=dict(t=20))
        st.plotly_chart(fig_cm, width='stretch')

    with col_b:
        st.markdown("**ROC curve — current threshold marked**")
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_test)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="gray"), name="Chance"))
        close_i = int(np.argmin(np.abs(roc_thresholds - thr)))
        fig_roc.add_trace(go.Scatter(
            x=[fpr[close_i]], y=[tpr[close_i]], mode="markers",
            marker=dict(size=12, color="red"), name=f"Threshold = {thr:.2f}",
        ))
        fig_roc.update_layout(xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate",
                               height=380, margin=dict(t=20))
        st.plotly_chart(fig_roc, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Permutation feature importance**")
        img = OUT_DIR / "04_feature_importance.png"
        if img.exists():
            st.image(str(img), width='stretch')
    with col_d:
        st.markdown("**SHAP summary**")
        img = OUT_DIR / "05_shap_summary.png"
        if img.exists():
            st.image(str(img), width='stretch')

    st.divider()
    st.markdown("**5-fold time-series cross-validation**")
    cv_df = pd.DataFrame(results["cv_fold_metrics"]).set_index("fold")
    st.dataframe(cv_df.style.format("{:.3f}"), width='stretch')
    st.caption(f"Mean CV AUC-ROC: {results['cv_auc_mean']:.3f}")

    st.divider()
    st.markdown("**Adaptive deployment simulation — rolling F1 per batch**")
    rf1 = results["rolling_f1_history"]
    batches = list(range(1, len(rf1) + 1))
    fig_rf1 = go.Figure()
    fig_rf1.add_trace(go.Scatter(x=batches, y=rf1, mode="lines+markers",
                                  name="Rolling F1 (per batch)"))
    fig_rf1.add_hline(y=results["drift_threshold"], line_dash="dash", line_color="red",
                       annotation_text=f"Drift threshold ({results['drift_threshold']})")
    for b in results["retrain_batches"]:
        fig_rf1.add_vline(x=b, line_dash="dot", line_color="orange")
    fig_rf1.update_layout(xaxis_title="Deployment batch", yaxis_title="F1-score",
                           yaxis_range=[0, 1], height=380, margin=dict(t=20))
    st.plotly_chart(fig_rf1, width='stretch')
    retrain_batches = results["retrain_batches"]
    st.caption(
        f"Retraining triggered on batches: {retrain_batches if retrain_batches else 'none'}"
    )

    st.divider()
    st.markdown("**Top 10 features by mutual information with the target**")
    mi_df = pd.DataFrame(results["mutual_information_top10"],
                          columns=["Feature", "Mutual information"])
    fig_mi = go.Figure(go.Bar(x=mi_df["Mutual information"][::-1],
                               y=mi_df["Feature"][::-1], orientation="h"))
    fig_mi.update_layout(height=400, margin=dict(t=20))
    st.plotly_chart(fig_mi, width='stretch')

# =======================================================================
# TAB 4 — ABOUT THE MODEL
# =======================================================================
with tab_about:
    st.markdown(f"""
### What this dashboard is showing
Every figure and prediction here comes from the trained **{results['model_name']}
v{results['model_version']}** model running on the real YEDC interruption log
(**{results['n_events']} logged events**, **{results['n_feeders']} feeders**,
{results['date_range'][0]} to {results['date_range'][1]}). Nothing is
simulated or hand-entered for this interface — the pipeline below runs
live against the same model file (`outputs/adaptguard_model.keras`) and
fitted preprocessor (`outputs/preprocessor_params.joblib`) that produced
the results reported in Chapter 4. The Scenario Explorer tab is the one
place you can type in hypothetical numbers; every other tab is driven
entirely by real logged data.

### Pipeline
1. **Data acquisition** — real multi-sheet Excel interruption log (`DataLoader`)
2. **Daily aggregation** — per-feeder, per-day event counts, downtime, and load-loss aggregates
3. **Preprocessing** — IQR clipping (continuous columns only) + Min-Max scaling, fitted once on the training partition only
4. **Feature engineering** — rolling (3/7/14-day) and lag (1-3 day) statistics, days-since-last-outage, calendar variables
5. **Sequence construction** — 7-day sliding windows, chronological 80/20 train/test split
6. **LSTM model** — `LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dense(16, relu) → Dense(1, sigmoid)`,
   Adam optimizer, binary cross-entropy loss, early stopping (patience 10), fixed seed 42
7. **Adaptive framework** — 5-fold time-series cross-validation, deployment-batch simulation with rolling-F1
   drift monitoring and automatic retraining below an F1 threshold of {results['drift_threshold']}

### An honest adaptation, stated plainly
The original design (Chapter 3) specified continuous 5-minute SCADA
telemetry with a 1-hour prediction horizon. The real YEDC data supplied
for this project is a discrete interruption event log, not continuous
telemetry, so the finest resolution that can honestly be derived from it
is **daily**: each feeder-day is aggregated from real events, and the
model predicts whether an outage will occur on the **following day**. The
LSTM architecture, training procedure, and adaptive retraining framework
are unchanged from the original specification — only the time resolution
and prediction horizon were adapted to match what the real data actually
supports.

### Limitations (for panel Q&A)
- Sample size: 104 events over a 4-month window is modest for deep learning
- Feeder imbalance: three feeders are not equally represented in the log
- No weather variation was captured in the observed period
- The deployment simulation is a retrospective replay of held-out test
  batches, not a live production deployment
""")
    st.caption(f"Selected classification threshold (max-F1 on validation set): {default_threshold:.2f}")
