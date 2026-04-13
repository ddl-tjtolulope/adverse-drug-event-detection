# generic_trainer.py
import os
import time
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix,
    balanced_accuracy_score, log_loss, brier_score_loss
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from domino_short_id import domino_short_id
from flytekitplugins.domino.artifact import Artifact, DATA, MODEL, REPORT


# Directories
experiment_name      = f"ADE Classifier Training {domino_short_id()}"
domino_working_dir   = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name  = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")
domino_artifact_dir  = domino_working_dir.replace('code', 'artifacts')
domino_dataset_dir   = f"{domino_working_dir.replace('code', 'data')}/{domino_project_name}"

ModelArtifact = Artifact(name="ADE Detection Models", type=MODEL)
DataArtifact  = Artifact(name="Training Data",         type=DATA)
ReportArtifact = Artifact(name="Model Reports",        type=REPORT)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
sns.set_context("poster")

COLORS = {
    'primary':   '#1B5E82',   # Pharma blue
    'secondary': '#6B3FA0',   # Deep violet
    'accent':    '#00897B',   # Teal
    'success':   '#2E7D32',   # Green
    'warning':   '#F57F17',   # Amber
    'danger':    '#C62828',   # Red
    'neutral':   '#546E7A',   # Blue-gray
    'light':     '#F8F9FA',
    'grid':      '#E9ECEF'
}


def plot_calibration_curve(y_val, proba, save_path, name):
    """Clean, professional calibration plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    fig.patch.set_facecolor('white')

    fraction_pos, mean_pred = calibration_curve(y_val, proba, n_bins=10)

    ax1.plot([0, 1], [0, 1], '--', linewidth=2, alpha=0.7, color=COLORS['neutral'],
             label='Perfect Calibration')
    ax1.plot(mean_pred, fraction_pos, 'o-', linewidth=3, markersize=8,
             color=COLORS['primary'], markerfacecolor=COLORS['accent'],
             markeredgecolor=COLORS['primary'], markeredgewidth=2,
             label='Model Calibration', alpha=0.9)
    ax1.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.15, color=COLORS['primary'])

    ax1.set_facecolor('white')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='600', color='#2C3E50')
    ax1.set_ylabel('Fraction of Positives',      fontsize=13, fontweight='600', color='#2C3E50')
    ax1.set_title('Model Calibration Assessment', fontsize=16, fontweight='700', color='#2C3E50', pad=20)
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-')
    ax1.tick_params(colors='#2C3E50', labelsize=10)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    n, bins, patches = ax2.hist(proba, bins=40, alpha=0.7, color=COLORS['success'],
                                edgecolor='white', linewidth=1.5)
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.viridis(i / len(patches)))
        p.set_alpha(0.8)

    mean_prob   = np.mean(proba)
    median_prob = np.median(proba)
    ax2.axvline(mean_prob,   color=COLORS['danger'],  linestyle='--', linewidth=3, alpha=0.8,
                label=f'Mean: {mean_prob:.3f}')
    ax2.axvline(median_prob, color=COLORS['warning'], linestyle='--', linewidth=3, alpha=0.8,
                label=f'Median: {median_prob:.3f}')

    ax2.set_facecolor('white')
    ax2.set_xlabel('Predicted Probability', fontsize=13, fontweight='600', color='#2C3E50')
    ax2.set_ylabel('Frequency',             fontsize=13, fontweight='600', color='#2C3E50')
    ax2.set_title('Prediction Distribution Analysis', fontsize=16, fontweight='700',
                  color='#2C3E50', pad=20)
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.tick_params(colors='#2C3E50', labelsize=10)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-')

    plt.suptitle(f'{name} — Calibration', fontsize=18, fontweight='700', color='#2C3E50')
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_learning_curves(model, X, y, save_path, name):
    """Professional learning curve analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
    fig.patch.set_facecolor('white')

    train_sizes = np.linspace(0.1, 1.0, 10)

    # 1. Accuracy learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores,  axis=1)
    val_mean   = np.mean(val_scores,   axis=1)
    val_std    = np.std(val_scores,    axis=1)

    ax1.plot(train_sizes_abs, train_mean, 'o-', linewidth=3, markersize=6,
             color=COLORS['primary'],  label='Training Performance', alpha=0.9)
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color=COLORS['primary'])
    ax1.plot(train_sizes_abs, val_mean, 's-', linewidth=3, markersize=6,
             color=COLORS['success'], label='Validation Performance', alpha=0.9)
    ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color=COLORS['success'])
    ax1.set_xlabel('Training Set Size', fontsize=12, fontweight='600', color='#2C3E50')
    ax1.set_ylabel('Accuracy Score',    fontsize=12, fontweight='600', color='#2C3E50')
    ax1.set_title('Model Accuracy Learning Curve', fontsize=14, fontweight='700', color='#2C3E50', pad=15)
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    # 2. ROC AUC learning curve
    _, train_scores_auc, val_scores_auc = learning_curve(
        model, X, y, cv=5, train_sizes=train_sizes, scoring='roc_auc', n_jobs=-1
    )
    train_mean_auc = np.mean(train_scores_auc, axis=1)
    train_std_auc  = np.std(train_scores_auc,  axis=1)
    val_mean_auc   = np.mean(val_scores_auc,   axis=1)
    val_std_auc    = np.std(val_scores_auc,    axis=1)

    ax2.plot(train_sizes_abs, train_mean_auc, 'o-', linewidth=3, markersize=6,
             color=COLORS['accent'],    label='Training AUC', alpha=0.9)
    ax2.fill_between(train_sizes_abs, train_mean_auc - train_std_auc,
                     train_mean_auc + train_std_auc, alpha=0.2, color=COLORS['accent'])
    ax2.plot(train_sizes_abs, val_mean_auc, 's-', linewidth=3, markersize=6,
             color=COLORS['secondary'], label='Validation AUC', alpha=0.9)
    ax2.fill_between(train_sizes_abs, val_mean_auc - val_std_auc,
                     val_mean_auc + val_std_auc, alpha=0.2, color=COLORS['secondary'])
    ax2.set_xlabel('Training Set Size', fontsize=12, fontweight='600', color='#2C3E50')
    ax2.set_ylabel('ROC AUC Score',     fontsize=12, fontweight='600', color='#2C3E50')
    ax2.set_title('Model AUC Learning Curve', fontsize=14, fontweight='700', color='#2C3E50', pad=15)
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)

    # 3. Training time complexity
    fit_times = []
    for train_size in train_sizes_abs:
        import time
        X_subset = X.iloc[:int(train_size)]
        y_subset = y.iloc[:int(train_size)]
        start = time.time()
        model.fit(X_subset, y_subset)
        fit_times.append(time.time() - start)

    ax3.plot(train_sizes_abs, fit_times, 'o-', linewidth=3, markersize=8,
             color=COLORS['warning'],  markerfacecolor=COLORS['danger'],
             markeredgecolor=COLORS['warning'], markeredgewidth=2, alpha=0.9)
    if len(fit_times) > 2:
        z = np.polyfit(train_sizes_abs, fit_times, 2)
        p = np.poly1d(z)
        ax3.plot(train_sizes_abs, p(train_sizes_abs), "--", linewidth=2,
                 alpha=0.7, color=COLORS['neutral'], label='Complexity Trend')
    ax3.set_xlabel('Training Set Size',     fontsize=12, fontweight='600', color='#2C3E50')
    ax3.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='600', color='#2C3E50')
    ax3.set_title('Training Time Complexity', fontsize=14, fontweight='700', color='#2C3E50', pad=15)
    ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)

    # 4. Overfitting diagnostic
    gap = train_mean - val_mean
    ax4.plot(train_sizes_abs, gap, 'o-', linewidth=3, markersize=6,
             color=COLORS['danger'], label='Performance Gap', alpha=0.9)
    ax4.fill_between(train_sizes_abs, 0, gap, alpha=0.3, color=COLORS['danger'])
    ax4.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.7, linewidth=1)
    if len(gap) >= 3:
        gap_smooth = np.convolve(gap, np.ones(3) / 3, mode='valid')
        ax4.plot(train_sizes_abs[1:-1], gap_smooth, '--', linewidth=2,
                 color=COLORS['primary'], alpha=0.8, label='Smoothed Trend')
    ax4.set_xlabel('Training Set Size',        fontsize=12, fontweight='600', color='#2C3E50')
    ax4.set_ylabel('Training − Validation Gap', fontsize=12, fontweight='600', color='#2C3E50')
    ax4.set_title('Overfitting Diagnostic',    fontsize=14, fontweight='700', color='#2C3E50', pad=15)
    ax4.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, color=COLORS['grid'])
    for spine in ['top', 'right']:
        ax4.spines[spine].set_visible(False)

    plt.suptitle(f'{name} — Training Analysis', fontsize=18, fontweight='700', color='#2C3E50', y=0.98)
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_model_performance_quad(y_val, proba, pred, name, save_path):
    """Comprehensive model performance analysis in quad layout"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc_score   = roc_auc_score(y_val, proba)
    ax1.plot(fpr, tpr, linewidth=3, color=COLORS['primary'], label=f'AUC = {auc_score:.3f}', alpha=0.9)
    ax1.plot([0, 1], [0, 1], '--', linewidth=2, color=COLORS['neutral'], alpha=0.7, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='600', color='#2C3E50')
    ax1.set_ylabel('True Positive Rate',  fontsize=12, fontweight='600', color='#2C3E50')
    ax1.set_title('ROC Curve Analysis',   fontsize=14, fontweight='700', color='#2C3E50')
    ax1.legend(fontsize=10, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_val, proba)
    pr_auc   = average_precision_score(y_val, proba)
    baseline = sum(y_val) / len(y_val)
    ax2.plot(recall, precision, linewidth=3, color=COLORS['accent'], label=f'AP = {pr_auc:.3f}', alpha=0.9)
    ax2.axhline(y=baseline, color=COLORS['neutral'], linestyle='--', linewidth=2, alpha=0.7,
                label=f'Baseline = {baseline:.3f}')
    ax2.fill_between(recall, precision, alpha=0.15, color=COLORS['accent'])
    ax2.set_xlabel('Recall',    fontsize=12, fontweight='600', color='#2C3E50')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='600', color='#2C3E50')
    ax2.set_title('Precision-Recall Analysis', fontsize=14, fontweight='700', color='#2C3E50')
    ax2.legend(fontsize=10, frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

    # 3. Confusion Matrix
    cm = confusion_matrix(y_val, pred, normalize='true')
    ax3.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, f'{cm[i, j]:.3f}', ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight='700')
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(['Non-Serious', 'Serious'], fontsize=11, fontweight='600')
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(['Non-Serious', 'Serious'], fontsize=11, fontweight='600')
    ax3.set_xlabel('Predicted', fontsize=12, fontweight='600', color='#2C3E50')
    ax3.set_ylabel('Actual',    fontsize=12, fontweight='600', color='#2C3E50')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='700', color='#2C3E50')

    # 4. Threshold sweep
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []
    for threshold in thresholds:
        pred_t = (proba >= threshold).astype(int)
        if len(np.unique(pred_t)) == 2:
            precisions.append(float(precision_score(y_val, pred_t, zero_division=0)))
            recalls.append(float(recall_score(y_val, pred_t, zero_division=0)))
            f1s.append(float(f1_score(y_val, pred_t, zero_division=0)))
        else:
            precisions.append(0); recalls.append(0); f1s.append(0)

    ax4.plot(thresholds, precisions, linewidth=3, color=COLORS['accent'],    label='Precision', alpha=0.9)
    ax4.plot(thresholds, recalls,    linewidth=3, color=COLORS['secondary'],  label='Recall',    alpha=0.9)
    ax4.plot(thresholds, f1s,        linewidth=3, color=COLORS['warning'],    label='F1-Score',  alpha=0.9)
    optimal_idx       = int(np.argmax(f1s))
    optimal_threshold = thresholds[optimal_idx]
    ax4.axvline(x=optimal_threshold, color=COLORS['danger'], linestyle='--', linewidth=2,
                alpha=0.7, label=f'Optimal = {optimal_threshold:.3f}')
    ax4.set_xlabel('Threshold', fontsize=12, fontweight='600', color='#2C3E50')
    ax4.set_ylabel('Score',     fontsize=12, fontweight='600', color='#2C3E50')
    ax4.set_title('Threshold Optimisation', fontsize=14, fontweight='700', color='#2C3E50')
    ax4.legend(fontsize=10, frameon=True, fancybox=True)
    ax4.grid(True, alpha=0.3, color=COLORS['grid'])
    ax4.set_xlim([0, 1]); ax4.set_ylim([0, 1.05])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'{name} — Performance Analysis', fontsize=18, fontweight='700', color='#2C3E50', y=0.98)
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def add_plots_to_training(model, name, X_train, X_val, y_train, y_val, features, df, proba):
    pred = model.predict(X_val)

    calibration_path = os.path.join(domino_artifact_dir, "calibration_plot.png")
    plot_calibration_curve(y_val, proba, calibration_path, name)
    mlflow.log_artifact(calibration_path)

    learning_path = os.path.join(domino_artifact_dir, "learning_curves_plot.png")
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    plot_learning_curves(model, X_combined, y_combined, learning_path, name)
    mlflow.log_artifact(learning_path)

    performance_path = os.path.join(domino_artifact_dir, "performance_dashboard.png")
    plot_model_performance_quad(y_val, proba, pred, name, performance_path)
    mlflow.log_artifact(performance_path)


def train_and_log(
    model, name: str,
    df: pd.DataFrame,
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series,    y_val: pd.Series,
    features: list
):
    """
    Train the model, log parameters, metrics, plots, and model artifact to MLflow.
    Returns a rich dict for downstream model comparison.
    """
    Path(domino_artifact_dir).mkdir(exist_ok=True, parents=True)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_name",   model.__class__.__name__)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("num_rows",     len(df))

        params_yaml = {
            "model_name":   model.__class__.__name__,
            "num_features": len(features),
            "num_rows":     len(df),
            "features":     features,
        }
        yaml_path = os.path.join(domino_artifact_dir, f"{name.lower().replace(' ', '_')}_params.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(params_yaml, f, default_flow_style=False)
        mlflow.log_artifact(yaml_path)

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        # Predict
        start_pred   = time.time()
        proba        = model.predict_proba(X_val)[:, 1]
        pred         = model.predict(X_val)
        predict_time = time.time() - start_pred
        inf_ms_row   = (predict_time / len(y_val)) * 1000

        # Core metrics
        roc  = roc_auc_score(y_val, proba)
        pr   = average_precision_score(y_val, proba)
        acc  = accuracy_score(y_val, pred)
        prec = precision_score(y_val, pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_val, pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_val, pred, pos_label=1, zero_division=0)
        ll   = log_loss(y_val, proba)
        brier = brier_score_loss(y_val, proba)

        fpr, tpr, roc_thr = roc_curve(y_val, proba)
        ks = float(np.max(tpr - fpr))

        eps   = 1e-15
        logit = np.log(np.clip(proba, eps, 1-eps) / np.clip(1-proba, eps, 1-eps))
        lr    = LogisticRegression(solver="lbfgs")
        lr.fit(logit.reshape(-1, 1), y_val)
        calib_slope     = float(lr.coef_[0][0])
        calib_intercept = float(lr.intercept_[0])
        frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=20, strategy="quantile")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))

        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()

        # Threshold scan
        taus = np.linspace(0.01, 0.99, 99)
        scan = []
        for t in taus:
            y_hat = (proba >= t).astype(int)
            if y_hat.max() != y_hat.min():
                scan.append({
                    "tau":       float(t),
                    "precision": float(precision_score(y_val, y_hat, zero_division=0)),
                    "recall":    float(recall_score(y_val, y_hat, zero_division=0)),
                    "f1":        float(f1_score(y_val, y_hat, zero_division=0))
                })
        best_f1_row = max(scan, key=lambda r: r["f1"]) if scan else {"tau": 0.5, "f1": f1}

        metrics = {
            "roc_auc": roc, "pr_auc": pr, "accuracy": acc,
            "precision_serious": prec, "recall_serious": rec, "f1_serious": f1,
            "fit_time_sec": fit_time, "predict_time_sec": predict_time,
            "inf_ms_row": inf_ms_row, "log_loss": ll, "brier": brier,
            "ks": ks, "calib_slope": calib_slope, "calib_intercept": calib_intercept,
            "ece": ece, "tau_best_f1": best_f1_row["tau"], "f1_best": best_f1_row["f1"]
        }
        mlflow.log_metrics(metrics)

        summary_metrics = {'model_name': name, 'validation_samples': int(len(y_val)),
                           'serious_samples': int(sum(y_val)), **metrics}
        metrics_df = pd.DataFrame([summary_metrics])
        metrics_csv_path = os.path.join(domino_artifact_dir,
                                        f"{name.lower().replace(' ', '_')}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path)

        model_pkl_path = os.path.join(domino_artifact_dir,
                                      f"{name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_pkl_path)
        model_size_kb = os.path.getsize(model_pkl_path) / 1024

        signature     = infer_signature(X_val, proba)
        input_example = X_val.iloc[:5]
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{name.lower().replace(' ', '_')}_model",
            signature=signature,
            input_example=input_example
        )

        mlflow.set_tag("pipeline", "ade_classifier_training")
        mlflow.set_tag("model", name)

        add_plots_to_training(model, name, X_train, X_val, y_train, y_val, features, df, proba)

        ret = {
            "schema_version":    "1.0",
            "model_name":        name,
            "model_type":        type(model).__name__,
            "run_id":            run.info.run_id,
            "artifact_uri":      mlflow.get_artifact_uri(),
            "n_test":            int(len(y_val)),
            "pos_rate_test":     float(np.mean(y_val)),
            "roc_auc":           float(roc),
            "pr_auc":            float(pr),
            "log_loss":          float(ll),
            "brier":             float(brier),
            "ks":                float(ks),
            "ece":               float(ece),
            "calib_slope":       calib_slope,
            "calib_intercept":   calib_intercept,
            "accuracy":          float(acc),
            "precision_serious": float(prec),
            "recall_serious":    float(rec),
            "f1_serious":        float(f1),
            "tau_default":       0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fit_time_sec":      float(fit_time),
            "predict_time_sec":  float(predict_time),
            "inf_ms_row":        float(inf_ms_row),
            "model_size_kb":     float(model_size_kb),
            "tau_best_f1":       float(best_f1_row["tau"]),
            "f1_best":           float(best_f1_row["f1"]),
            "threshold_scan":    scan,
            "curves": {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": roc_thr.tolist()},
                "pr":  {
                    "precision": precision_recall_curve(y_val, proba)[0].tolist(),
                    "recall":    precision_recall_curve(y_val, proba)[1].tolist()
                },
                "calibration": {
                    "bin_mean_pred": mean_pred.tolist(),
                    "bin_frac_pos":  frac_pos.tolist()
                }
            },
            "notes": None
        }

        results_path = os.path.join(domino_artifact_dir,
                                    f"{name.lower().replace(' ', '_')}_result.json")
        ret['results_path'] = results_path
        with open(results_path, 'w') as f:
            json.dump(ret, f, indent=2)
        mlflow.log_artifact(results_path)

    mlflow.end_run()
    return ret


def train_ade(model_obj, model_name, transformed_df_filename, random_state=None):
    mlflow.set_experiment(experiment_name)

    transformed_df_path = f"{domino_dataset_dir}/{transformed_df_filename}"
    transformed_df = pd.read_csv(transformed_df_path)

    labels   = "Serious"
    df       = transformed_df.dropna(subset=[labels]).copy()
    features = [c for c in df.columns if c != labels]
    X = df[features]
    y = df[labels]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    data_summary = {
        'total_samples':   len(df),
        'serious_samples': int(sum(y)),
        'serious_rate':    float(sum(y) / len(y)),
        'features':        features,
        'train_samples':   len(X_train),
        'val_samples':     len(X_val)
    }

    domino_artifacts_path = Path("/workflow/outputs")
    domino_artifacts_path.mkdir(exist_ok=True, parents=True)
    with open(domino_artifacts_path / "data_summary.json", 'w') as f:
        json.dump(data_summary, f, indent=2)

    print(f'Training model: {model_name}')
    res = train_and_log(
        model_obj, model_name,
        df, X_train, X_val, y_train, y_val,
        features
    )
    return res
