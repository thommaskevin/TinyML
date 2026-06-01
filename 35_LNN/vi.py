# vi.py
"""
Variational Inference loss for Liquid Neural Network (LNN / LTC) models.

The ELBO (Evidence Lower BOund) loss combines a negative log-likelihood
term (reconstruction loss) with a KL-divergence regularisation term:

    ELBO = NLL(output, target) + kl_weight * KL / batch_size

The model must expose a ``kl_loss()`` method that returns the summed
KL divergence across all stochastic parameters (e.g. Bayesian linear
layers with weight distributions).

For standard (non-Bayesian) LNNs, a lightweight weight-decay regulariser
based on the Frobenius norm of the learnable A parameters is provided as
a surrogate for KL in ``ltc_reg_loss``.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ELBO loss (requires a Bayesian LNN exposing kl_loss())
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
    Compute the ELBO loss for a variational Liquid Neural Network.

    Args:
        output     : Model predictions (raw logits for classification tasks).
        target     : Ground-truth labels.
        model      : Model instance exposing a ``kl_loss() -> Tensor`` method.
        kl_weight  : Scalar multiplier applied to the KL term (default: ``1.0``).
                     Common practice is to anneal this from 0 → 1 during
                     training (KL annealing / warm-up).
        likelihood : Reconstruction loss to use.  One of:

                     - ``'mse'``   — Mean Squared Error
                     - ``'mae'``   — Mean Absolute Error
                     - ``'rmse'``  — Root Mean Squared Error
                     - ``'bce'``   — Binary Cross-Entropy (with logits)
                     - ``'cce'``   — Categorical Cross-Entropy (one-hot)
                     - ``'scce'``  — Sparse Categorical Cross-Entropy (int)
                     - ``'huber'`` — Huber / Smooth-L1

                     Default: ``'mse'``.
        **kwargs   : Extra keyword arguments forwarded to the likelihood
                     function (e.g. ``huber_delta=0.5``).

    Returns:
        Scalar ELBO loss tensor.

    Raises:
        ValueError: if *likelihood* is not a recognised option.
    """
    key = likelihood.lower()

    # --- Negative log-likelihood (reconstruction term) ---
    if key == 'mse':
        nll = F.mse_loss(output, target, reduction='mean')

    elif key == 'mae':
        nll = F.l1_loss(output, target, reduction='mean')

    elif key == 'rmse':
        nll = torch.sqrt(
            F.mse_loss(output, target, reduction='mean') + 1e-8
        )

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
        raise ValueError(
            f"Unknown likelihood '{likelihood}'. "
            "Available options: 'mse', 'mae', 'rmse', 'bce', "
            "'cce', 'scce', 'huber'."
        )

    # --- KL divergence term (normalised by batch size) ---
    kl = model.kl_loss() / output.size(0)

    return nll + kl_weight * kl


# ---------------------------------------------------------------------------
# LTC-specific regulariser (no Bayesian model required)
# ---------------------------------------------------------------------------

def ltc_reg_loss(
    output:       torch.Tensor,
    target:       torch.Tensor,
    model,
    reg_weight:   float = 1e-3,
    likelihood:   str   = 'mse',
    **kwargs,
) -> torch.Tensor:
    """
    Combined task loss + LTC regularisation for standard (non-Bayesian) LNNs.

    The regularisation term penalises the Frobenius norm of the learnable
    amplitude parameters **A** across all LTC cells, encouraging the
    time constants to stay close to their initial values and preventing
    pathological collapse or explosion.

    Regularisation term:

        L_reg = reg_weight * Σ_i  ‖A_i‖²_F / hidden_size_i

    Args:
        output      : Model predictions.
        target      : Ground-truth labels.
        model       : ``LNNModel`` instance (must have ``recurrent_cells``
                      with learnable ``A`` parameters).
        reg_weight  : Weighting coefficient for the regularisation term
                      (default: ``1e-3``).
        likelihood  : Name of the task loss (same options as ``elbo_loss``).
                      Default: ``'mse'``.
        **kwargs    : Extra keyword arguments forwarded to the likelihood
                      function (e.g. ``huber_delta=0.5``).

    Returns:
        Scalar combined loss tensor.
    """
    from losses import compute_loss

    task_loss = compute_loss(likelihood, output, target, **kwargs)

    # Sum squared A norms, normalised per cell
    reg = torch.tensor(0.0, device=output.device)
    for cell in model.recurrent_cells:
        if hasattr(cell, 'A'):
            reg = reg + (cell.A ** 2).mean()

    return task_loss + reg_weight * reg
