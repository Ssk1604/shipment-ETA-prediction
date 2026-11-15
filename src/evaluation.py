"""
Evaluation Metrics & Visualization Module for ETA Prediction and Delay Classification.

Provides standard evaluation computations and generates publication-grade
performance plots saved to reports/figures/.
"""

import json
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics: MAE, RMSE, R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "model": model_name,
        "mae_hours": round(mae, 3),
        "rmse_hours": round(rmse, 3),
        "r2_score": round(r2, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
    }


def plot_regression_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    output_path: str = "reports/figures/regression_actual_vs_predicted.png",
) -> str:
    """Generates an Actual vs Predicted delay scatter plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_true, y_pred, color="#1f77b4", alpha=0.75, edgecolors="k", label="Random Forest Regressor")
    plt.axhline(baseline_pred[0], color="#d62728", linestyle="--", linewidth=1.5, label=f"Baseline Mean ({baseline_pred[0]:.1f}h)")
    
    # 45-degree line for perfect fit
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6, label="Ideal Fit (y = x)")

    plt.xlabel("Actual Final Delay (Hours)", fontsize=11)
    plt.ylabel("Predicted Final Delay (Hours)", fontsize=11)
    plt.title("Model A: Actual vs. Predicted Final Delay", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_confusion_matrix(
    cm: List[List[int]],
    output_path: str = "reports/figures/confusion_matrix.png",
) -> str:
    """Generates a styled confusion matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["On-Time / Minor (<24h)", "Delayed (≥24h)"], fontsize=10)
    ax.set_yticklabels(["On-Time / Minor (<24h)", "Delayed (≥24h)"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Model B: Delay Risk Confusion Matrix", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_metrics(metrics: Dict[str, Any], output_path: str = "models/metrics.json") -> str:
    """Serializes evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return output_path
