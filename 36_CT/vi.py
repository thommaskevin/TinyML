# vi.py
"""
Variational and policy inference for Causal Tree (CT) models.

Mirrors the role of ``vi.py`` in the LNN framework: advanced estimation
and decision-making objectives that go beyond the raw leaf-level CATE.

Three task families are covered:

Regression
----------
elbo_reg          : AIPW-augmented MSE loss (doubly-robust ATE).
ltc_reg_loss      : τ̂-regularised task loss; penalises extreme leaf effects.

Binary classification
---------------------
elbo_binary       : AIPW-augmented BCE loss (doubly-robust risk difference).
policy_loss_bin   : IPW policy-value loss for binary treatment decisions.

Multiclass
----------
elbo_multi        : Per-class AIPW-augmented cross-entropy.
policy_loss_multi : IPW policy-value loss for K-class treatment decisions.

Shared utilities
----------------
doubly_robust_ate : AIPW estimator for the population ATE (any task).
compute_vi        : Unified dispatcher (mirrors ``compute_metric`` in losses.py).

References
----------
Robins, J., Rotnitzky, A., & Zhao, L. (1994). JASA 89(427), 846–866.
Athey, S., & Imbens, G. (2016). PNAS 113(27), 7353–7360.
Nie, X., & Wager, S. (2021). Biometrika 108(2), 299–319.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


# =============================================================================
# Shared — Doubly-Robust ATE (AIPW)
# =============================================================================

def doubly_robust_ate(
    tau_hat:  np.ndarray,
    y:        np.ndarray,
    w:        np.ndarray,
    mu_1_hat: Optional[np.ndarray] = None,
    mu_0_hat: Optional[np.ndarray] = None,
    e_hat:    Optional[np.ndarray] = None,
    alpha:    float = 0.05,
) -> dict:
    """
    Augmented IPW (AIPW) estimator for the population ATE.

    The AIPW pseudo-outcome for unit i is:

        φ_i = (μ̂₁(xᵢ) − μ̂₀(xᵢ))
              + (Wᵢ − ê(xᵢ)) / [ê(xᵢ)(1 − ê(xᵢ))] · (Yᵢ − μ̂_{Wᵢ}(xᵢ))

    ATE_AIPW = mean(φ_i).

    Doubly robust: consistent if *either* the outcome model or the
    propensity model is correctly specified.

    Args:
        tau_hat  : Estimated CATE of shape ``(n,)``.
        y        : Observed outcomes of shape ``(n,)``.
        w        : Binary treatment indicators of shape ``(n,)``.
        mu_1_hat : Predicted Y(1) of shape ``(n,)``; defaults to Y[W=1] mean.
        mu_0_hat : Predicted Y(0) of shape ``(n,)``; defaults to Y[W=0] mean.
        e_hat    : Propensity scores of shape ``(n,)``; defaults to w.mean().
        alpha    : Significance level for the CI (default: 0.05 → 95% CI).

    Returns:
        Dict with keys ``'ate'``, ``'std_err'``, ``'ci_lower'``,
        ``'ci_upper'``, ``'pvalue'``, ``'pseudo'``.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(y)

    if mu_1_hat is None:
        mu_1_hat = np.full(n, y[w == 1].mean() if (w == 1).sum() > 0 else 0.0)
    if mu_0_hat is None:
        mu_0_hat = np.full(n, y[w == 0].mean() if (w == 0).sum() > 0 else 0.0)
    if e_hat is None:
        e_hat = np.full(n, float(w.mean()))

    mu_1 = np.asarray(mu_1_hat, dtype=np.float64)
    mu_0 = np.asarray(mu_0_hat, dtype=np.float64)
    e    = np.clip(np.asarray(e_hat, dtype=np.float64), 1e-6, 1 - 1e-6)

    mu_w   = w * mu_1 + (1.0 - w) * mu_0
    aug    = (w - e) / (e * (1.0 - e)) * (y - mu_w)
    pseudo = (mu_1 - mu_0) + aug

    ate     = float(pseudo.mean())
    se      = float(pseudo.std(ddof=1) / np.sqrt(n))
    z       = float(stats.norm.ppf(1.0 - alpha / 2.0))
    pvalue  = float(2.0 * stats.norm.sf(abs(ate / se))) if se > 0 else 1.0

    return {
        'ate':      ate,
        'std_err':  se,
        'ci_lower': ate - z * se,
        'ci_upper': ate + z * se,
        'pvalue':   pvalue,
        'pseudo':   pseudo,
    }


# =============================================================================
# Regression
# =============================================================================

def elbo_reg(
    tau_hat:  np.ndarray,
    y:        np.ndarray,
    w:        np.ndarray,
    e_hat:    Optional[np.ndarray] = None,
    kl_weight: float = 1e-3,
    **kw,
) -> float:
    """
    AIPW-augmented MSE loss for regression.

    Combines the standard MSE task loss with a KL-proxy regularisation
    term that penalises the mean squared leaf effect (analogous to the
    KL term in a Bayesian ELBO):

        L = MSE(τ̂, Ỹ) + kl_weight · mean(τ̂²)

    where Ỹ is the IPW pseudo-outcome.

    Args:
        tau_hat   : Estimated CATE of shape ``(n,)``.
        y         : Outcomes of shape ``(n,)``.
        w         : Treatment indicators of shape ``(n,)``.
        e_hat     : Propensity scores (optional).
        kl_weight : Weight of the regularisation term (default: ``1e-3``).

    Returns:
        Scalar ELBO-style loss.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    e = np.clip(np.asarray(e_hat if e_hat is not None else np.full(len(y), w.mean()),
                           dtype=np.float64), 1e-6, 1 - 1e-6)
    pseudo = w * y / e - (1.0 - w) * y / (1.0 - e)
    mse    = float(np.mean((tau_hat - pseudo) ** 2))
    reg    = float(np.mean(tau_hat ** 2))
    return mse + kl_weight * reg


def ltc_reg_loss(
    tau_hat:   np.ndarray,
    y:         np.ndarray,
    w:         np.ndarray,
    reg_weight: float = 1e-3,
    **kw,
) -> float:
    """
    τ̂-regularised task loss for regression (no propensity needed).

    L = MAE(Y, W·mean(Y|T) + (1-W)·mean(Y|C)) + reg_weight · mean(|τ̂|)

    Analogous to ``ltc_reg_loss`` in the LNN vi.py: a lightweight
    weight-decay-style regulariser on the estimated effects.

    Args:
        tau_hat    : Estimated CATE of shape ``(n,)``.
        y          : Outcomes of shape ``(n,)``.
        w          : Treatment indicators of shape ``(n,)``.
        reg_weight : Effect regularisation weight (default: ``1e-3``).

    Returns:
        Scalar combined loss.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    t_mask, c_mask = w == 1, w == 0
    mu_1 = y[t_mask].mean() if t_mask.sum() > 0 else 0.0
    mu_0 = y[c_mask].mean() if c_mask.sum() > 0 else 0.0
    y_hat = w * mu_1 + (1.0 - w) * mu_0
    task  = float(np.mean(np.abs(y - y_hat)))
    reg   = float(np.mean(np.abs(tau_hat)))
    return task + reg_weight * reg


# =============================================================================
# Binary classification
# =============================================================================

def elbo_binary(
    tau_hat:   np.ndarray,
    y:         np.ndarray,
    w:         np.ndarray,
    e_hat:     Optional[np.ndarray] = None,
    kl_weight: float = 1e-3,
    **kw,
) -> float:
    """
    AIPW-augmented BCE loss for binary classification.

    ELBO = BCE(sigmoid(τ̂), Ỹ_binary) + kl_weight · mean(τ̂²)

    where Ỹ_binary = W·Y/ê − (1−W)·Y/(1−ê) is the IPW pseudo-label
    (clipped to [0,1]) used as a soft binary target.

    Args:
        tau_hat   : Risk-difference estimates of shape ``(n,)``.
        y         : Binary outcomes 0/1 of shape ``(n,)``.
        w         : Treatment indicators of shape ``(n,)``.
        e_hat     : Propensity scores (optional).
        kl_weight : Weight of the regularisation term (default: ``1e-3``).

    Returns:
        Scalar ELBO-style loss.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    e = np.clip(np.asarray(e_hat if e_hat is not None else np.full(len(y), w.mean()),
                           dtype=np.float64), 1e-6, 1 - 1e-6)
    pseudo  = np.clip(w * y / e - (1.0 - w) * y / (1.0 - e), 0.0, 1.0)
    tau_t   = torch.tensor(tau_hat, dtype=torch.float32)
    pseudo_t = torch.tensor(pseudo, dtype=torch.float32)
    bce     = float(F.binary_cross_entropy_with_logits(tau_t, pseudo_t))
    reg     = float(np.mean(tau_hat ** 2))
    return bce + kl_weight * reg


def policy_loss_bin(
    tau_hat:   np.ndarray,
    y:         np.ndarray,
    w:         np.ndarray,
    e_hat:     Optional[np.ndarray] = None,
    **kw,
) -> float:
    """
    IPW policy-value loss for binary treatment decisions.

    Policy: π(x) = 1[τ̂(x) > 0] (treat if estimated risk difference > 0).

    Policy value (IPW):
        PV = mean[ π·W·Y/ê + (1−π)·(1−W)·Y/(1−ê) ]

    Returns the *negative* policy value (loss to minimise).

    Args:
        tau_hat : Risk-difference estimates of shape ``(n,)``.
        y       : Binary outcomes of shape ``(n,)``.
        w       : Treatment indicators of shape ``(n,)``.
        e_hat   : Propensity scores (optional).

    Returns:
        Scalar policy loss (lower = better policy).
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    e = np.clip(np.asarray(e_hat if e_hat is not None else np.full(len(y), w.mean()),
                           dtype=np.float64), 1e-6, 1 - 1e-6)
    pi  = (tau_hat > 0).astype(np.float64)
    ipw = pi * w * y / e + (1.0 - pi) * (1.0 - w) * y / (1.0 - e)
    return float(-ipw.mean())


# =============================================================================
# Multiclass
# =============================================================================

def elbo_multi(
    tau_proba:  np.ndarray,
    y:          np.ndarray,
    w:          np.ndarray,
    e_hat:      Optional[np.ndarray] = None,
    kl_weight:  float = 1e-3,
    **kw,
) -> float:
    """
    Per-class AIPW-augmented cross-entropy for multiclass tasks.

    For each class k, constructs an IPW pseudo-outcome:

        Ỹ_ik = W_i · 1[Y_i=k] / ê_i − (1−W_i) · 1[Y_i=k] / (1−ê_i)

    and computes the categorical cross-entropy of τ̂_k against soft
    targets derived from the clipped pseudo-outcomes.

    ELBO = CCE(τ̂, Ỹ_soft) + kl_weight · mean(‖τ̂‖²_F)

    Args:
        tau_proba  : Per-class CATE differences of shape ``(n, K)``.
        y          : Integer class labels of shape ``(n,)``.
        w          : Treatment indicators of shape ``(n,)``.
        e_hat      : Propensity scores (optional).
        kl_weight  : Regularisation weight (default: ``1e-3``).

    Returns:
        Scalar ELBO-style loss.
    """
    y = np.asarray(y, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    K = tau_proba.shape[1]
    e = np.clip(np.asarray(e_hat if e_hat is not None else np.full(len(y), w.mean()),
                           dtype=np.float64), 1e-6, 1 - 1e-6)

    # Build soft pseudo-targets (n, K)
    one_hot = np.zeros((len(y), K), dtype=np.float64)
    for k in range(K):
        yk = (y == k).astype(np.float64)
        one_hot[:, k] = w * yk / e - (1.0 - w) * yk / (1.0 - e)
    # Shift and normalise to [0,1] per row for stable CE
    oh_min = one_hot.min(axis=1, keepdims=True)
    oh_max = one_hot.max(axis=1, keepdims=True)
    denom  = np.where(oh_max - oh_min > 1e-8, oh_max - oh_min, 1.0)
    soft   = (one_hot - oh_min) / denom
    soft   = soft / soft.sum(axis=1, keepdims=True).clip(1e-8)

    logits_t = torch.tensor(tau_proba, dtype=torch.float32)
    soft_t   = torch.tensor(soft,       dtype=torch.float32)
    cce      = float(-(soft_t * F.log_softmax(logits_t, dim=-1)).sum(dim=-1).mean())
    reg      = float(np.mean(tau_proba ** 2))
    return cce + kl_weight * reg


def policy_loss_multi(
    tau_proba:  np.ndarray,
    y:          np.ndarray,
    w:          np.ndarray,
    e_hat:      Optional[np.ndarray] = None,
    **kw,
) -> float:
    """
    IPW policy-value loss for K-class treatment decisions.

    Policy: π(x) = argmax_k τ̂_k(x) (assign to the class with largest
    positive treatment-effect shift).

    Policy value (IPW):
        PV = mean[ 1[W_i = π(x_i)] · Y_i^{binary} / ê(W_i | x_i) ]

    Here Y^binary = 1[Y = π(x)] and the propensity is computed as the
    marginal probability of receiving the recommended arm.

    Returns the *negative* policy value (loss to minimise).

    Args:
        tau_proba : Per-class CATE differences of shape ``(n, K)``
                    from ``predict_proba()``.
        y         : Integer class labels of shape ``(n,)``.
        w         : Treatment indicators of shape ``(n,)``.
        e_hat     : Propensity scores (optional).

    Returns:
        Scalar policy loss (lower = better).
    """
    y   = np.asarray(y, dtype=np.int64)
    w   = np.asarray(w, dtype=np.float64)
    e   = np.clip(np.asarray(e_hat if e_hat is not None else np.full(len(y), w.mean()),
                             dtype=np.float64), 1e-6, 1 - 1e-6)
    pi  = np.argmax(tau_proba, axis=1).astype(int)          # recommended class
    # reward = 1 if y == π(x) AND W==1 (treated and got predicted best outcome)
    reward    = ((y == pi) & (w == 1)).astype(np.float64)
    ipw_vals  = reward / e
    return float(-ipw_vals.mean())


# =============================================================================
# Registry
# =============================================================================

_VI_FN: dict = {
    # shared
    'dr_ate':            doubly_robust_ate,
    # regression
    'elbo_reg':          elbo_reg,
    'ltc_reg_loss':      ltc_reg_loss,
    # binary
    'elbo_binary':       elbo_binary,
    'policy_loss_bin':   policy_loss_bin,
    # multiclass
    'elbo_multi':        elbo_multi,
    'policy_loss_multi': policy_loss_multi,
}

VI_NAMES: list[str] = list(_VI_FN.keys())


def compute_vi(
    method:  str,
    tau_hat: np.ndarray,
    **kwargs,
):
    """
    Dispatch a VI / policy computation by name.

    Args:
        method  : One of the keys in ``VI_NAMES``.
        tau_hat : Estimated CATE / per-class probabilities.
        **kwargs: Forwarded keyword arguments (``y=``, ``w=``, etc.).

    Returns:
        Result of the selected function (scalar float or dict).

    Raises:
        ValueError: if *method* is not registered.
    """
    key = method.lower()
    if key not in _VI_FN:
        raise ValueError(
            f"Unknown VI method '{method}'. Available: {VI_NAMES}"
        )
    return _VI_FN[key](tau_hat, **kwargs)
