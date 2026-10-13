# losses.py
"""
Loss and evaluation metrics for Extra Trees (Extremely Randomized Trees).

Supported metrics
-----------------
Classification
~~~~~~~~~~~~~~
accuracy          : Fraction of correctly classified samples.
log_loss          : Negative log-likelihood (cross-entropy) of class probabilities.
precision         : Macro-averaged precision across all classes.
recall            : Macro-averaged recall across all classes.
f1_score          : Macro-averaged F1 score (harmonic mean of precision/recall).
gini_coefficient  : Weighted mean Gini impurity of predicted class distributions.

Regression
~~~~~~~~~~
mse               : Mean Squared Error between predictions and targets.
mae               : Mean Absolute Error between predictions and targets.
rmse              : Root Mean Squared Error.
r2                : Coefficient of determination R^2.
mape              : Mean Absolute Percentage Error (avoids division by zero).

Notes
-----
All metric functions accept plain NumPy arrays.  ``compute_loss`` dispatches
by name so the evaluation loop can mix classification and regression metrics
without conditional logic.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Fraction of correctly classified samples.

    Args:
        y_true : Integer ground-truth labels of shape ``(n,)``.
        y_pred : Integer predicted labels of shape ``(n,)``.

    Returns:
        Scalar accuracy in [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def log_loss(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    eps:        float = 1e-12,
    **kwargs,
) -> float:
    """
    Negative log-likelihood (cross-entropy) loss for multi-class predictions.

    L = -(1/n) * sum_i sum_k y_{ik} * log(p_{ik} + eps)

    Args:
        y_true : Integer class labels of shape ``(n,)``.
        y_prob : Class probability matrix of shape ``(n, K)``, rows summing
                 to 1.  If 1-D, interpreted as the probability of class 1
                 for binary classification.
        eps    : Small constant for numerical stability (default: ``1e-12``).

    Returns:
        Scalar non-negative log-loss (lower is better).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
    n, K   = y_prob.shape
    ll     = 0.0
    for i in range(n):
        ll -= math.log(float(y_prob[i, y_true[i]]) + eps)
    return ll / n


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple:
    """
    Compute macro-averaged precision, recall, and F1 score.

    Macro averaging computes the metric independently for each class
    and returns the unweighted mean, giving equal weight to each class
    regardless of its support.

    Args:
        y_true : Integer ground-truth labels of shape ``(n,)``.
        y_pred : Integer predicted labels of shape ``(n,)``.

    Returns:
        Tuple ``(precision, recall, f1)`` of scalar floats.
    """
    y_true   = np.asarray(y_true).ravel()
    y_pred   = np.asarray(y_pred).ravel()
    classes  = np.unique(np.concatenate([y_true, y_pred]))
    precs, recs, f1s = [], [], []

    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f  = 2 * p * r / max(p + r, 1e-12)
        precs.append(p); recs.append(r); f1s.append(f)

    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged precision.  See ``precision_recall_f1``."""
    return precision_recall_f1(y_true, y_pred)[0]


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged recall.  See ``precision_recall_f1``."""
    return precision_recall_f1(y_true, y_pred)[1]


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged F1 score.  See ``precision_recall_f1``."""
    return precision_recall_f1(y_true, y_pred)[2]


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean Squared Error: mean((y - y_hat)^2)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean Absolute Error: mean(|y - y_hat|)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Root Mean Squared Error: sqrt(MSE)."""
    return float(np.sqrt(mse(y_true, y_pred) + 1e-12))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot

    where SS_res = sum((y - y_hat)^2) and SS_tot = sum((y - mean(y))^2).
    R^2 = 1 for a perfect fit; R^2 <= 0 for a model no better than the mean.

    Args:
        y_true : Ground-truth values of shape ``(n,)``.
        y_pred : Predicted values of shape ``(n,)``.

    Returns:
        Scalar R^2 score (higher is better).
    """
    y_true  = np.asarray(y_true).ravel()
    y_pred  = np.asarray(y_pred).ravel()
    ss_res  = float(np.sum((y_true - y_pred) ** 2))
    ss_tot  = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mape(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE = (100 / n) * sum_i |y_i - y_hat_i| / max(|y_i|, eps)

    Samples where |y_true| < 1e-8 are excluded to avoid division by zero.

    Args:
        y_true : Ground-truth values of shape ``(n,)``.
        y_pred : Predicted values of shape ``(n,)``.

    Returns:
        Scalar MAPE in percentage points (lower is better).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask   = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(
        100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    )


# ---------------------------------------------------------------------------
# math import needed by log_loss
# ---------------------------------------------------------------------------
import math


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRIC_FN: dict = {
    # Classification
    'accuracy':  accuracy,
    'log_loss':  log_loss,
    'precision': precision_score,
    'recall':    recall_score,
    'f1':        f1_score,
    # Regression
    'mse':       mse,
    'mae':       mae,
    'rmse':      rmse,
    'r2':        r2_score,
    'mape':      mape,
}

LOSS_NAMES: list = list(_METRIC_FN.keys())


def compute_loss(
    metric_name: str,
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    **kwargs,
) -> float:
    """
    Compute a named evaluation metric.

    Args:
        metric_name : One of the keys in ``LOSS_NAMES``.
        y_true      : Ground-truth labels or values of shape ``(n,)``.
        y_pred      : Predicted labels, values, or probabilities.
        **kwargs    : Extra arguments forwarded to the metric function.

    Returns:
        Scalar metric value.

    Raises:
        ValueError: if *metric_name* is not registered.
    """
    key = metric_name.lower()
    if key not in _METRIC_FN:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {LOSS_NAMES}"
        )
    return _METRIC_FN[key](y_true, y_pred, **kwargs)
