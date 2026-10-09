"""Main pipeline for AdaptGuard v2: the adaptive LSTM outage-occurrence
prediction system, rebuilt on REAL YEDC data (104 logged interruption
events, January-April 2026) at DAILY resolution -- see
src/feature_engineer.py's module docstring for why the resolution and
horizon are adapted from Chapter 3's literal 5-minute/1-hour spec.
Follows the same Activity Diagram structure as before (3.2.3.3):

  1. Data Acquisition        -> DataLoader (real multi-sheet Excel log)
  2. Daily Aggregation        -> FeatureEngineer.build_daily_panel()
  3. Preprocessing            -> Preprocessor.clean() (IQR on continuous cols only)
  4. Feature Engineering      -> FeatureEngineer.engineer() (rolling/lag/calendar/target)
  5. Sequence Construction    -> SequenceBuilder (window=7 days, chronological split)
  6. LSTM Model Training      -> LSTMModel (Adam, binary CE, early stopping)
  7. Performance Evaluation   -> Accuracy/Precision/Recall/F1/AUC-ROC (+5-fold TS-CV)
  8. AUC-ROC >= 0.70 check    -> NFR1
  9. Deployment Simulation    -> AdaptiveFramework.run_simulation()
  10. Rolling F1 < 0.35 check -> AdaptiveFramework.retrain()
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.sequence_builder import SequenceBuilder, WINDOW
from src.lstm_model import LSTMModel, MODEL_NAME, MODEL_VERSION, RANDOM_STATE
from src.adaptive_framework import AdaptiveFramework
from src import utils

OUT_DIR = "outputs"
MODEL_OUT_PATH = f"{OUT_DIR}/adaptguard_model.keras"

# Only genuinely continuous measurements get IQR outlier treatment.
# event_count/total_downtime_min/type_* are sparse occurrence counts --
# clipping them with IQR would crush the real outage signal to zero
# (Q1=Q3=0 for a column that's ~80% zeros), which was caught during
# development and is deliberately avoided here.
IQR_COLS = ["mean_load_loss_mw", "max_load_loss_mw"]


def main(quick=False):
    epochs = 5 if quick else None
    n_cv_splits = 2 if quick else 5
    n_deploy_batches = 2 if quick else 5

    print(f"=== {MODEL_NAME} v{MODEL_VERSION} — Adaptive LSTM Outage-Occurrence "
          f"Prediction Pipeline (REAL DATA, daily resolution) ===")
    if quick:
        print("*** --quick mode: reduced epochs/CV-folds for a fast demo run. ***")
    print()

    # --- 1. Data Acquisition (real event log) ---
    loader = DataLoader("data/yedc_interruption_log.xlsx")
    events = loader.load()
    print(f"Loaded {len(events)} real interruption events across "
          f"{events['Affected Feeder'].nunique()} feeders "
          f"({events['Date Out'].min().date()} to {events['Date Out'].max().date()}).")

    # --- 2. Daily aggregation ---
    fe = FeatureEngineer(window=WINDOW)
    panel = fe.build_daily_panel(events)
    print(f"Built daily panel: {len(panel)} feeder-days.")

    # --- 3. Preprocessing (IQR on continuous cols only, no label leakage) ---
    pre = Preprocessor()
    clean = pre.clean(panel, IQR_COLS, fit=True)

    # --- 4. Feature engineering (rolling/lag/calendar + 1-day-ahead target) ---
    engineered = fe.engineer(clean)
    feature_cols = FeatureEngineer.feature_columns(engineered)
    print(f"Engineered {len(feature_cols)} feature columns. "
          f"Target (outage next day) positive rate: {engineered['target'].mean():.4f}")

    utils.plot_target_distribution(engineered["target"].to_numpy(),
                                    f"{OUT_DIR}/01_target_distribution.png")

    # --- Mutual information ranking, on window-mean aggregated features ---
    sb = SequenceBuilder(window=WINDOW)
    X_all, y_all, ts_all, seg_all = sb.build_sequences(engineered, feature_cols)
    X_agg = X_all.mean(axis=1)
    mi = mutual_info_classif(X_agg, y_all, random_state=RANDOM_STATE)
    mi_ranking = sorted(zip(feature_cols, mi), key=lambda kv: -kv[1])
    print("\nMUTUAL INFORMATION FEATURE RANKING (top 10)")
    for feat, score in mi_ranking[:10]:
        print(f"  {feat}: {score:.4f}")

    # --- 5. Sequence construction: chronological split (latest 20% test) ---
    (X_train_raw, y_train, ts_train, seg_train,
     X_test_raw, y_test, ts_test, seg_test) = sb.chronological_split(
        X_all, y_all, ts_all, seg_all
    )
    print(f"\nSequence construction: {len(X_all)} total sequences "
          f"(window={WINDOW} days). Train={len(X_train_raw)}, Test={len(X_test_raw)}.")

    # --- Scaling: fit exclusively on the training partition (3.3.3) ---
    n_train, w, n_feat = X_train_raw.shape
    pre.scaler.fit(X_train_raw.reshape(-1, n_feat))
    X_train = pre.scaler.transform(X_train_raw.reshape(-1, n_feat)).reshape(n_train, w, n_feat)
    X_test = pre.scaler.transform(X_test_raw.reshape(-1, n_feat)).reshape(len(X_test_raw), w, n_feat)
    pre.save_params(f"{OUT_DIR}/preprocessor_params.joblib")

    print(f"\nTraining pool: {len(X_train)} sequences "
          f"({int(y_train.sum())} positive / {int(len(y_train) - y_train.sum())} negative). "
          "SMOTE is applied only inside each fit's own training slice -- never to "
          "validation/test data (see README for why this matters).")

    # --- 5-fold time-series cross-validation (3.4.4) ---
    framework = AdaptiveFramework(window_size=WINDOW, n_cv_splits=n_cv_splits, epochs=epochs)
    print(f"\nRunning {n_cv_splits}-fold time-series cross-validation...")
    cv_metrics = framework.time_series_cross_validate(X_train, y_train, n_feat)
    for m in cv_metrics:
        print(f"  Fold {m['fold']}: AUC={m['AUC-ROC']:.3f}  F1={m['F1-Score']:.3f}")
    cv_auc_mean = float(np.nanmean([m["AUC-ROC"] for m in cv_metrics]))

    # --- 6. Final LSTM training: chronological fit/val split, SMOTE on fit slice only ---
    n_val = max(1, int(0.15 * len(X_train)))
    X_fit_raw, y_fit_raw = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_fit, y_fit = SequenceBuilder.apply_smote(X_fit_raw, y_fit_raw)
    print(f"\nAfter SMOTE (fit slice only): {len(X_fit)} training sequences "
          f"({int(y_fit.sum())} positive / {int(len(y_fit) - y_fit.sum())} negative).")

    model = LSTMModel(n_features=n_feat, window=WINDOW)
    print(f"\nTraining final {MODEL_NAME} v{MODEL_VERSION} "
          f"(max {epochs or 50} epochs, batch 64, early stopping patience 10)...")
    train_kwargs = {} if epochs is None else {"epochs": epochs}
    history = model.train(X_fit, y_fit, X_val, y_val, verbose=2, **train_kwargs)
    chosen_threshold = model.select_threshold(X_val, y_val)
    print(f"Selected classification threshold (max F1 on validation set): "
          f"{chosen_threshold:.2f}")
    utils.plot_training_history(history, f"{OUT_DIR}/02_training_history.png")

    # --- 7/8. Performance evaluation on the held-out chronological test set ---
    test_metrics = model.evaluate(X_test, y_test)
    print("\nHELD-OUT TEST SET PERFORMANCE")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")
    nfr1_met = test_metrics["AUC-ROC"] >= 0.70
    print(f"  NFR1 (AUC-ROC >= 0.70): {'MET' if nfr1_met else 'NOT MET'}")

    y_prob_test = model.predict_proba(X_test)
    y_pred_test = model.predict(X_test)
    utils.plot_confusion_and_roc(y_test, y_pred_test, y_prob_test,
                                  f"{OUT_DIR}/03_confusion_roc.png")

    names, imps = utils.permutation_importance(model, X_test, y_test, feature_cols)
    utils.plot_feature_importance(names, imps, f"{OUT_DIR}/04_feature_importance.png")
    utils.plot_shap_summary(model, X_test, feature_cols, f"{OUT_DIR}/05_shap_summary.png")

    model.model.save(MODEL_OUT_PATH)
    print(f"\nSaved {MODEL_NAME} v{MODEL_VERSION} to {MODEL_OUT_PATH}")

    # --- 9/10. Deployment simulation + adaptive retraining ---
    print("\nRunning deployment simulation (sequential batches, rolling F1 "
          "drift monitoring)...")
    sim_result = framework.run_simulation(model, X_test, y_test, X_train, y_train,
                                           n_batches=n_deploy_batches)
    print(f"  Rolling F1 per batch: "
          f"{[round(f, 3) for f in sim_result['rolling_f1_history']]}")
    print(f"  Retrain triggered on batches: {sim_result['retrain_batches'] or 'none'}")
    utils.plot_rolling_f1(sim_result["rolling_f1_history"], framework.drift_threshold,
                           sim_result["retrain_batches"], f"{OUT_DIR}/06_rolling_f1.png")

    results = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_events": int(len(events)),
        "n_feeders": int(events["Affected Feeder"].nunique()),
        "date_range": [str(events["Date Out"].min().date()), str(events["Date Out"].max().date())],
        "n_feeder_days": int(len(panel)),
        "n_sequences_total": int(len(X_all)),
        "n_sequences_train": int(len(X_train_raw)),
        "n_sequences_test": int(len(X_test_raw)),
        "n_sequences_train_after_smote": int(len(X_fit)),
        "window": WINDOW,
        "target_positive_rate": float(engineered["target"].mean()),
        "mutual_information_top10": [[f, float(s)] for f, s in mi_ranking[:10]],
        "cv_auc_mean": cv_auc_mean,
        "cv_fold_metrics": [{k: (float(v) if k != "fold" else int(v)) for k, v in m.items()}
                             for m in cv_metrics],
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "nfr1_met": bool(nfr1_met),
        "rolling_f1_history": [float(f) for f in sim_result["rolling_f1_history"]],
        "retrain_batches": [int(b) for b in sim_result["retrain_batches"]],
        "drift_threshold": framework.drift_threshold,
    }
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results summary to {OUT_DIR}/results.json")
    print("\nPipeline complete. Outputs saved to outputs/.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Fast demo run (5 epochs, 2-fold CV) instead of "
                              "the full spec (50 epochs, 5-fold CV).")
    args = parser.parse_args()
    main(quick=args.quick)
