"""
Model Evaluation and Metrics Module.

Calculates comprehensive classification metrics, confusion matrices, ROC curves,
and Precision-Recall curves on held-out test partitions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_model_performance(
    model_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates a fitted pipeline against held-out test data.

    Parameters
    ----------
    model_name : str
        Human-readable name of the model.
    pipeline : Any
        Fitted Scikit-Learn Pipeline object.
    X_test : pd.DataFrame
        Held-out feature records.
    y_test : pd.Series
        Ground-truth binary labels.
    decision_threshold : float
        Probability threshold above which a customer is classified as Churn (default 0.50).

    Returns
    -------
    Dict[str, Any]
        Dictionary of comprehensive evaluation metrics.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    # Downsample curve points for efficient JSON serialization and plotting
    stride_roc = max(1, len(fpr) // 100)
    roc_data = {
        "fpr": [round(float(v), 5) for v in fpr[::stride_roc]],
        "tpr": [round(float(v), 5) for v in tpr[::stride_roc]],
        "thresholds": [round(float(v), 5) for v in roc_thresh[::stride_roc]],
    }

    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, y_proba)
    stride_pr = max(1, len(prec_curve) // 100)
    pr_data = {
        "precision": [round(float(v), 5) for v in prec_curve[::stride_pr]],
        "recall": [round(float(v), 5) for v in rec_curve[::stride_pr]],
        "thresholds": [round(float(v), 5) for v in pr_thresh[::stride_pr]],
    }

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "decision_threshold": decision_threshold,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "business_impact": {
            "retention_cost_fp": f"{fp} loyal customers received unnecessary retention incentives",
            "lost_revenue_fn": f"{fn} churning customers went undetected and were lost",
            "successful_saves_tp": f"{tp} at-risk customers proactively flagged for intervention",
        },
        "roc_curve": roc_data,
        "pr_curve": pr_data,
    }

    logger.info(
        f"[{model_name}] ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | "
        f"Precision: {prec:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}"
    )
    return metrics
