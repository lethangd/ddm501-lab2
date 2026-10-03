"""
Evaluation stage.

Two kinds of number come out of here, and the second is the one people forget.

  Aggregate metrics   ROC AUC, PR AUC, precision/recall at the decision threshold.
  Sliced metrics      the same numbers computed separately per group.

TODO: Complete compute_metrics, compute_group_metrics and fairness_gap.
"""

import logging
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline.config import REVIEW_THRESHOLD, SENSITIVE_ATTRIBUTE
from pipeline.preprocessing import prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: Implement compute_metrics
# =============================================================================
# Aggregate metrics at a fixed decision threshold. Return a dict with:
#   roc_auc, pr_auc, precision, recall, f1, brier,
#   true_positives, false_positives, false_negatives, true_negatives, threshold
#
# Hint: tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float = REVIEW_THRESHOLD
) -> Dict[str, float]:
    """Aggregate metrics at a fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "threshold": float(threshold),
    }


# =============================================================================
# TODO 2: Implement compute_group_metrics
# =============================================================================
# The same metrics, computed separately for each value of `groups`.
#
# Requirements:
#   - return {str(group_value): {...metrics..., "n": int, "selection_rate": float}}
#   - SKIP a group with fewer than 50 rows, or with only one class present.
#     Metrics on a handful of rows are noise, and reporting them as fairness
#     evidence is worse than reporting nothing.
#   - selection_rate = share of that group with y_proba >= threshold, i.e. the
#     fraction the model sends to review or decline.
#
# Hint: mask = (groups == value).to_numpy(); y_true[mask]; np.unique(y_true[mask])

def compute_group_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    groups: pd.Series,
    threshold: float = REVIEW_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """The same metrics, one set per group value."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    groups = pd.Series(np.asarray(groups))
    out: Dict[str, Dict[str, float]] = {}
    for value in groups.unique():
        mask = (groups == value).to_numpy()
        if mask.sum() < 50 or len(np.unique(y_true[mask])) < 2:
            continue
        m = compute_metrics(y_true[mask], y_proba[mask], threshold)
        m["n"] = int(mask.sum())
        m["selection_rate"] = float((y_proba[mask] >= threshold).mean())
        out[str(value)] = m
    return out


# =============================================================================
# TODO 3: Implement fairness_gap
# =============================================================================
# Largest difference in `key` between any two groups. Return 0.0 when there are
# fewer than two groups.

def fairness_gap(group_metrics: Dict[str, Dict[str, float]], key: str = "selection_rate") -> float:
    """Largest difference in `key` between any two groups."""
    values = [g[key] for g in group_metrics.values() if key in g]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: Optional[str] = None,
    threshold: float = REVIEW_THRESHOLD,
) -> Dict[str, Any]:
    """Score the test set, compute everything, and log it to the run."""
    sensitive = X_test[SENSITIVE_ATTRIBUTE].copy() if SENSITIVE_ATTRIBUTE in X_test else None
    X_prepared = prepare_features(X_test)

    y_proba = model.predict_proba(X_prepared)[:, 1]
    metrics = compute_metrics(y_test, y_proba, threshold)

    group_metrics: Dict[str, Dict[str, float]] = {}
    gap = 0.0
    if sensitive is not None:
        group_metrics = compute_group_metrics(y_test, y_proba, sensitive, threshold)
        gap = fairness_gap(group_metrics)

    result = {**metrics, "fairness_gap": gap, "group_metrics": group_metrics}

    logger.info(
        "ROC AUC %.4f | PR AUC %.4f | recall %.3f | fairness gap %.3f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["recall"], gap,
    )

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metric("fairness_gap", gap)
            for value, gm in group_metrics.items():
                for k, v in gm.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"group_{SENSITIVE_ATTRIBUTE}_{value}_{k}", v)
            mlflow.log_dict(result, "evaluation.json")
            mlflow.set_tag("stage", "evaluated")

    return result
