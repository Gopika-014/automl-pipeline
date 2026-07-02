import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = "reports/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Confusion matrix shows exactly where your model
    is making mistakes.

    For binary classification:
    - True Positive (TP): predicted positive, actually positive
    - True Negative (TN): predicted negative, actually negative
    - False Positive (FP): predicted positive, actually negative
      (Type 1 error)
    - False Negative (FN): predicted negative, actually positive
      (Type 2 error)

    In medical diagnosis:
    - FN is dangerous — missed a sick patient
    - FP is costly — unnecessary treatment

    In fraud detection:
    - FN is dangerous — missed a fraud transaction
    - FP is annoying — blocked a valid transaction

    Which error is worse depends on your domain.
    That's why you look at the confusion matrix,
    not just accuracy.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_roc_curve(best_model, X_test, y_test, model_name):
    """
    ROC Curve — Receiver Operating Characteristic.

    X axis: False Positive Rate (FPR)
    Y axis: True Positive Rate (TPR) = Recall

    AUC — Area Under the Curve:
    - AUC = 1.0: perfect model
    - AUC = 0.5: random guessing (diagonal line)
    - AUC < 0.5: worse than random

    ROC curve shows the tradeoff between
    sensitivity (catching positives) and
    specificity (avoiding false alarms)
    at different classification thresholds.

    Only works for binary classification.
    """
    if not hasattr(best_model, "predict_proba"):
        print("Model does not support probability — skipping ROC curve.")
        return None

    if len(np.unique(y_test)) != 2:
        print("ROC curve only for binary classification — skipping.")
        return None

    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='gray',
             linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = f"{PLOTS_DIR}/roc_curve.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path, roc_auc


def plot_prediction_vs_actual(y_test, y_pred, model_name):
    """
    For regression — scatter plot of predicted vs actual values.

    Perfect model: all points on diagonal line y=x.
    Points far from diagonal = large errors.
    Systematic bias shows up as points above or below diagonal.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual — {model_name}")
    plt.legend()
    plt.tight_layout()

    path = f"{PLOTS_DIR}/pred_vs_actual.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def compute_classification_metrics(y_test, y_pred,
                                   best_model, X_test,
                                   model_name):
    """
    Computes full classification evaluation.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(
            y_test, y_pred, average='weighted'), 4),
        "recall": round(recall_score(
            y_test, y_pred, average='weighted'), 4),
        "f1": round(f1_score(
            y_test, y_pred, average='weighted'), 4)
    }

    # ROC AUC
    roc_result = plot_roc_curve(
        best_model, X_test, y_test, model_name)
    if roc_result:
        _, roc_auc = roc_result
        metrics["roc_auc"] = round(roc_auc, 4)

    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    return metrics


def compute_regression_metrics(y_test, y_pred):
    """
    Computes full regression evaluation.
    """
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "r2": round(r2_score(y_test, y_pred), 4)
    }
    return metrics


def run_evaluation(best_model, best_model_name,
                   X_test, y_test, problem_type):
    """
    Master evaluation function.
    Call this from app.py
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    y_pred = best_model.predict(X_test)
    plot_paths = {}

    if problem_type == 'classification':
        # Confusion matrix
        cm_path = plot_confusion_matrix(
            y_test, y_pred, best_model_name)
        plot_paths['confusion_matrix'] = cm_path

        # Metrics + ROC
        metrics = compute_classification_metrics(
            y_test, y_pred, best_model,
            X_test, best_model_name)

        if 'roc_auc' in metrics:
            plot_paths['roc_curve'] = f"{PLOTS_DIR}/roc_curve.png"

    else:
        # Predicted vs actual
        pred_path = plot_prediction_vs_actual(
            y_test, y_pred, best_model_name)
        plot_paths['pred_vs_actual'] = pred_path

        # Metrics
        metrics = compute_regression_metrics(y_test, y_pred)

    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)

    return metrics, plot_paths