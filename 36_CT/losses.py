# losses.py
"""
Loss / evaluation metrics for Causal Tree (CT) models.

Three task families are supported:

Regression
----------
mse       : Mean Squared Error on τ̂
mae       : Mean Absolute Error on τ̂
rmse      : Root Mean Squared Error on τ̂
tau_risk  : IPW pseudo-outcome MSE (no oracle τ* needed)
ate_bias  : |mean(τ̂) − Difference-in-Means ATE|

Binary classification
---------------------
bce       : Binary Cross-Entropy on the risk-difference τ̂ (via sigmoid)
accuracy  : Fraction of units assigned to correct arm by policy τ̂ > 0
risk_diff : Absolute error in estimated risk difference
tau_risk  : Shared with regression (works for binary Y too)

Multiclass
----------
scce      : Sparse Categorical Cross-Entropy on predicted class label
accuracy  : Top-1 accuracy of predicted class label
macro_ate : Mean absolute error of per-class ATE estimates

All metrics are accessed through the unified ``compute_metric()`` dispatcher.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def mse_loss(tau_hat: np.ndarray, tau_true: np.ndarray, **kw) -> float:
    """MSE between estimated and oracle CATE (requires tau_true)."""
    return float(np.mean((tau_hat - tau_true) ** 2))


def mae_loss(tau_hat: np.ndarray, tau_true: np.ndarray, **kw) -> float:
    """MAE between estimated and oracle CATE (requires tau_true)."""
    return float(np.mean(np.abs(tau_hat - tau_true)))


def rmse_loss(tau_hat: np.ndarray, tau_true: np.ndarray, **kw) -> float:
    """Root-MSE between estimated and oracle CATE (requires tau_true)."""
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2) + 1e-12))


def tau_risk(tau_hat: np.ndarray, y: np.ndarray, w: np.ndarray, **kw) -> float:
    """
    Tau-risk: IPW pseudo-outcome MSE.  No oracle τ* required.

    Pseudo-outcome:  Ỹ_i = W_i·Y_i/p̂ − (1−W_i)·Y_i/(1−p̂)

    score = mean((τ̂(x_i) − Ỹ_i)²)

    Consistent estimator of PEHE² under unconfoundedness.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    p = float(w.mean())
    if p <= 0 or p >= 1:
        raise ValueError("Propensity is 0 or 1 — tau_risk undefined.")
    pseudo = w * y / p - (1.0 - w) * y / (1.0 - p)
    return float(np.mean((tau_hat - pseudo) ** 2))


def ate_bias(tau_hat: np.ndarray, y: np.ndarray, w: np.ndarray, **kw) -> float:
    """
    Absolute bias in the ATE estimate.

    ATE_DIM = mean(Y|W=1) − mean(Y|W=0);   ate_bias = |mean(τ̂) − ATE_DIM|
    """
    y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    if (w == 1).sum() == 0 or (w == 0).sum() == 0:
        raise ValueError("Both treated and control units required.")
    ate_dim = float(y[w == 1].mean() - y[w == 0].mean())
    return float(abs(float(tau_hat.mean()) - ate_dim))


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def bce_loss(tau_hat: np.ndarray, y: np.ndarray, w: np.ndarray, **kw) -> float:
    """
    Binary Cross-Entropy on the binary outcome, using τ̂ as a logit-proxy
    for the conditional risk difference.

    Treats sigmoid(τ̂) as the estimated P(benefit|x) and computes
    BCE against binary targets derived from the treated arm.
    """
    y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    t_mask = w == 1
    if t_mask.sum() == 0:
        raise ValueError("No treated units — BCE undefined.")
    tau_t = torch.tensor(tau_hat[t_mask], dtype=torch.float32)
    y_t   = torch.tensor(y[t_mask],     dtype=torch.float32)
    return float(F.binary_cross_entropy_with_logits(tau_t, y_t))


def accuracy_binary(tau_hat: np.ndarray, y: np.ndarray, w: np.ndarray, **kw) -> float:
    """
    Policy accuracy for binary outcome.

    Policy: assign to treatment if τ̂ > 0.
    Accuracy: fraction of units where the assigned arm matches actual W.
    """
    pi = (tau_hat > 0).astype(int)
    w  = np.asarray(w, dtype=int)
    return float((pi == w).mean())


def risk_diff_error(tau_hat: np.ndarray, y: np.ndarray, w: np.ndarray, **kw) -> float:
    """
    Absolute error of the estimated risk difference.

    True risk diff = P(Y=1|W=1) − P(Y=1|W=0).
    """
    y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    t_mask, c_mask = w == 1, w == 0
    if t_mask.sum() == 0 or c_mask.sum() == 0:
        raise ValueError("Both arms required for risk_diff_error.")
    true_rd = float(y[t_mask].mean() - y[c_mask].mean())
    return float(abs(float(tau_hat.mean()) - true_rd))


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------

def scce_loss(
    tau_hat_labels: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    **kw,
) -> float:
    """
    Sparse Categorical Cross-Entropy.

    ``tau_hat_labels`` must be integer predicted class labels (from
    ``CausalTreeLayer.predict()`` with task='multiclass').
    Evaluated on the treated arm only (where the outcome is observed
    under treatment).
    """
    y = np.asarray(y, dtype=np.int64); w = np.asarray(w, dtype=np.int64)
    t_mask = w == 1
    if t_mask.sum() == 0:
        raise ValueError("No treated units — SCCE undefined.")
    pred_t = torch.tensor(tau_hat_labels[t_mask], dtype=torch.long)
    true_t = torch.tensor(y[t_mask],             dtype=torch.long)
    K = int(true_t.max().item()) + 1
    # one-hot logits from predicted labels
    logits = F.one_hot(pred_t, num_classes=K).float()
    return float(F.cross_entropy(logits, true_t))


def accuracy_multiclass(
    tau_hat_labels: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    **kw,
) -> float:
    """
    Top-1 accuracy of predicted class labels versus true labels
    on the treated arm (where Y(1) is observed).
    """
    y = np.asarray(y, dtype=np.int64); w = np.asarray(w, dtype=np.int64)
    t_mask = w == 1
    if t_mask.sum() == 0:
        return 0.0
    return float((tau_hat_labels[t_mask] == y[t_mask]).mean())


def macro_ate_error(
    tau_proba: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    **kw,
) -> float:
    """
    Mean Absolute Error of per-class ATE estimates.

    ``tau_proba`` : array of shape ``(n, K)`` from ``predict_proba()``.
    True per-class ATE = P(Y=k|W=1) − P(Y=k|W=0) for k=0,…,K-1.
    """
    y = np.asarray(y, dtype=np.int64); w = np.asarray(w, dtype=np.int64)
    t_mask, c_mask = w == 1, w == 0
    if t_mask.sum() == 0 or c_mask.sum() == 0:
        raise ValueError("Both arms required for macro_ate_error.")
    K = tau_proba.shape[1]
    errors = []
    for k in range(K):
        p_t = float((y[t_mask] == k).mean())
        p_c = float((y[c_mask] == k).mean())
        true_ate_k = p_t - p_c
        est_ate_k  = float(tau_proba[:, k].mean())
        errors.append(abs(est_ate_k - true_ate_k))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRICS: dict = {
    # regression
    'mse':              mse_loss,
    'mae':              mae_loss,
    'rmse':             rmse_loss,
    'tau_risk':         tau_risk,
    'ate_bias':         ate_bias,
    # binary
    'bce':              bce_loss,
    'accuracy':         accuracy_binary,
    'risk_diff_error':  risk_diff_error,
    # multiclass
    'scce':             scce_loss,
    'accuracy_multi':   accuracy_multiclass,
    'macro_ate_error':  macro_ate_error,
}

METRIC_NAMES: list[str] = list(_METRICS.keys())


def compute_metric(
    metric_name: str,
    tau_hat:     np.ndarray,
    **kwargs,
) -> float:
    """
    Compute a named evaluation metric.

    Args:
        metric_name : Metric key (see ``METRIC_NAMES``).
        tau_hat     : Estimated CATE / predicted labels of shape ``(n,)``
                      (or ``(n, K)`` for ``macro_ate_error``).
        **kwargs    : Keyword arguments forwarded to the metric function.
                      Common: ``y=``, ``w=``, ``tau_true=``.

    Returns:
        Scalar metric value.

    Raises:
        ValueError: if *metric_name* is not registered.
    """
    key = metric_name.lower()
    if key not in _METRICS:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {METRIC_NAMES}"
        )
    return _METRICS[key](tau_hat, **kwargs)
