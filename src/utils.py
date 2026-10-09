"""Plotting and explainability utilities: dataset/target distribution,
training curves, confusion matrix + ROC, rolling-F1 drift monitor plot,
permutation feature importance, and a SHAP summary plot (Chapter 2.1.2 /
Baembitov et al., 2025 -- explainability improves operator trust)."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

sns.set_style("whitegrid")


def plot_target_distribution(y, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    vals, counts = np.unique(y, return_counts=True)
    ax1.bar(vals.astype(str), counts, color=["#2ca02c", "#d62728"])
    ax1.set_xlabel("Outage within 1h (0/1)")
    ax1.set_ylabel("Count")
    ax1.set_title("Prediction Target Distribution")

    ax2.pie(counts, labels=vals.astype(str), autopct="%1.1f%%",
            colors=["#2ca02c", "#d62728"])
    ax2.set_title("Target Share")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Binary Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["auc"], label="train")
    ax2.plot(history.history["val_auc"], label="val")
    ax2.set_title("AUC")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_and_roc(y_true, y_pred, y_prob, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=False)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="gray")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_f1(rolling_f1_history, drift_threshold, retrain_batches, outpath):
    fig, ax = plt.subplots(figsize=(9, 4))
    batches = np.arange(1, len(rolling_f1_history) + 1)
    ax.plot(batches, rolling_f1_history, marker="o", color="#1f77b4",
            label="Rolling F1 (per batch)")
    ax.axhline(drift_threshold, color="#d62728", linestyle="--",
               label=f"Drift threshold ({drift_threshold})")
    for b in retrain_batches:
        ax.axvline(b, color="#ff7f0e", alpha=0.6, linestyle=":")
    ax.set_xlabel("Deployment batch")
    ax.set_ylabel("F1-score")
    ax.set_title("Adaptive Deployment Simulation: Rolling F1 & Retrain Triggers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def permutation_importance(model, X, y, feature_names, n_repeats=3,
                            random_state=42, top_n=15):
    """Feature importance via permutation: shuffle one feature across
    the window+sample axes, measure AUC drop. Neural nets have no
    built-in feature_importances_ (unlike Random Forest), so this is
    the model-agnostic substitute."""
    rng = np.random.default_rng(random_state)
    baseline_auc = roc_auc_score(y, model.predict_proba(X))
    importances = np.zeros(len(feature_names))

    for i in range(len(feature_names)):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            flat = X_perm[:, :, i].reshape(-1)
            rng.shuffle(flat)
            X_perm[:, :, i] = flat.reshape(X_perm.shape[0], X_perm.shape[1])
            perm_auc = roc_auc_score(y, model.predict_proba(X_perm))
            drops.append(baseline_auc - perm_auc)
        importances[i] = np.mean(drops)

    order = np.argsort(importances)[::-1][:top_n]
    return [feature_names[i] for i in order], importances[order]


def plot_feature_importance(names, importances, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), importances[::-1], color="#4c72b0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean AUC drop when permuted")
    ax.set_title("Permutation Feature Importance (LSTM)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(model, X_sample, feature_names, outpath,
                       n_background=20, n_explain=30):
    try:
        import shap
    except ImportError:
        print("shap not installed -- skipping SHAP summary plot.")
        return

    window = X_sample.shape[1]
    n_features = X_sample.shape[2]

    def predict_fn(flat):
        return model.predict_proba(flat.reshape(-1, window, n_features))

    background = X_sample[:n_background].reshape(n_background, -1)
    explain_set = X_sample[n_background:n_background + n_explain].reshape(n_explain, -1)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_set, nsamples=100, silent=True)

    # Aggregate SHAP values across the time window -> one value per feature
    shap_arr = np.asarray(shap_values).reshape(n_explain, window, n_features)
    shap_agg = shap_arr.mean(axis=1)
    X_agg = explain_set.reshape(n_explain, window, n_features).mean(axis=1)

    plt.figure()
    shap.summary_plot(shap_agg, X_agg, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
