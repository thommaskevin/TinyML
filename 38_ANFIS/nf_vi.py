# nf_vi.py
"""
Variational Inference loss for Adaptive Neuro-Fuzzy Inference System (ANFIS).

The ELBO (Evidence Lower BOund) loss combines a negative log-likelihood
term with a KL-divergence regularisation term.  For standard (non-Bayesian)
ANFIS models, a lightweight regulariser based on the Frobenius norm of the
consequent coefficients is provided as a surrogate for KL.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ELBO loss (requires a Bayesian ANFIS exposing kl_loss())
# ---------------------------------------------------------------------------

def elbo_loss(
    output:     torch.Tensor,
    target:     torch.Tensor,
    model,
    kl_weight:  float = 1.0,
    likelihood: str   = 'mse',
    **kwargs,
) -> torch.Tensor:
    """
    Compute the ELBO loss for a variational ANFIS.

    Args:
        output     : Model predictions.
        target     : Ground-truth labels.
        model      : Model instance exposing a ``kl_loss() -> Tensor`` method.
        kl_weight  : Scalar multiplier applied to the KL term (default: ``1.0``).
        likelihood : Reconstruction loss: ``'mse'``, ``'mae'``, ``'rmse'``,
                     ``'bce'``, ``'cce'``, ``'scce'``, ``'huber'``.
        **kwargs   : Extra keyword arguments forwarded to the likelihood.

    Returns:
        Scalar ELBO loss tensor.
    """
    key = likelihood.lower()

    if key == 'mse':
        nll = F.mse_loss(output, target, reduction='mean')
    elif key == 'mae':
        nll = F.l1_loss(output, target, reduction='mean')
    elif key == 'rmse':
        nll = torch.sqrt(F.mse_loss(output, target, reduction='mean') + 1e-8)
    elif key == 'bce':
        out = output.squeeze()
        tgt = target.view_as(out).float()
        nll = F.binary_cross_entropy_with_logits(out, tgt, reduction='mean')
    elif key == 'cce':
        log_probs = F.log_softmax(output, dim=-1)
        nll = -(target.float() * log_probs).sum(dim=-1).mean()
    elif key == 'scce':
        nll = F.cross_entropy(output, target.long(), reduction='mean')
    elif key == 'huber':
        delta = float(kwargs.get('huber_delta', 1.0))
        diff  = output - target
        nll   = torch.where(
            diff.abs() <= delta,
            0.5 * diff ** 2,
            delta * diff.abs() - 0.5 * delta ** 2,
        ).mean()
    else:
        raise ValueError(f"Unknown likelihood '{likelihood}'.")

    kl = model.kl_loss() / output.size(0)
    return nll + kl_weight * kl


# ---------------------------------------------------------------------------
# ANFIS-specific regulariser (no Bayesian model required)
# ---------------------------------------------------------------------------

def anfis_reg_loss(
    output:       torch.Tensor,
    target:       torch.Tensor,
    model,
    reg_weight:   float = 1e-3,
    likelihood:   str   = 'mse',
    **kwargs,
) -> torch.Tensor:
    """
    Combined task loss + ANFIS regularisation for standard (non-Bayesian) models.

    The regularisation term penalises the L2 norm of the consequent
    coefficients, encouraging small, interpretable rule outputs and
    preventing over-fitting on high-dimensional rule bases.

    Regularisation term:

        L_reg = reg_weight * ‖coeff‖²_F / n_rules

    Args:
        output      : Model predictions.
        target      : Ground-truth labels.
        model       : ``ANFISModel`` instance.
        reg_weight  : Weighting coefficient (default: ``1e-3``).
        likelihood  : Name of the task loss.
        **kwargs    : Extra keyword arguments forwarded to the likelihood.

    Returns:
        Scalar combined loss tensor.
    """
    from losses import compute_loss

    task_loss = compute_loss(likelihood, output, target, **kwargs)

    reg = torch.tensor(0.0, device=output.device)
    if hasattr(model, 'consequent') and hasattr(model.consequent, 'coeff'):
        reg = reg + (model.consequent.coeff ** 2).mean()

    return task_loss + reg_weight * reg
