from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate binary classification probabilities at a chosen threshold.
    Returns healthcare-friendly metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def find_threshold_recall_precision(
    y_true,
    y_proba,
    target_recall: float = 0.80,
    min_precision: float = 0.10,
) -> float:
    """
    Choose the highest threshold that satisfies:
    recall >= target_recall and precision >= min_precision.

    If no threshold satisfies both, fall back to the threshold with
    recall closest to target_recall while keeping precision as high as possible.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns:
    # len(precisions) = len(recalls) = len(thresholds) + 1
    # so align precision/recall with thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    candidates = []
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= target_recall and p >= min_precision:
            candidates.append((t, p, r))

    if candidates:
        # choose highest threshold to reduce false positives while satisfying constraints
        best_t, _, _ = max(candidates, key=lambda x: x[0])
        return float(best_t)

    # fallback: prioritize closeness to target recall, then precision, then threshold
    fallback = sorted(
        zip(thresholds, precisions, recalls),
        key=lambda x: (abs(x[2] - target_recall), -x[1], -x[0]),
    )[0]

    return float(fallback[0])