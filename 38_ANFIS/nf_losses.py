# nf_losses.py
"""
Loss functions for Neuro-Fuzzy (ANFIS) training.

This module re-exports the same loss functions used for LNN models,
since ANFIS uses standard supervised learning (MSE, MAE, BCE, etc.).

Additional regularisers specific to fuzzy systems are also provided:
- ``fuzzy_reg_loss`` : penalises overlapping membership functions
                       to encourage interpretability.
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Re-export standard losses (identical API to losses.py)
# ---------------------------------------------------------------------------

from losses import (
    mse_loss,
    mae_loss,
    rmse_loss,
    bce_loss,
    cce_loss,
    scce_loss,
    huber_loss,
    compute_loss,
    LOSS_NAMES,
)


# ---------------------------------------------------------------------------
# Fuzzy-specific regulariser
# ---------------------------------------------------------------------------

def fuzzy_reg_loss(
    output:      torch.Tensor,
    target:      torch.Tensor,
    model,
    reg_weight:  float = 1e-3,
    overlap_pen: float = 1e-2,
    loss_name:   str   = 'mse',
    **loss_kwargs,
) -> torch.Tensor:
    """
    Combined task loss + fuzzy interpretability regularisation.

    The regularisation term has two components:

    1. **Width penalty** — encourages membership functions to stay
       compact (small σ / width), improving linguistic interpretability.
    2. **Overlap penalty** — discourages excessive overlap between
       adjacent membership functions for the same input variable.

    Args:
        output      : Model predictions.
        target      : Ground-truth labels.
        model       : ``ANFISModel`` instance.
        reg_weight  : Weight for width penalty (default: ``1e-3``).
        overlap_pen : Weight for overlap penalty (default: ``1e-2``).
        loss_name   : Name of the task loss (same options as ``compute_loss``).
        **loss_kwargs : Extra keyword arguments forwarded to the task loss.

    Returns:
        Scalar combined loss tensor.
    """
    task_loss = compute_loss(loss_name, output, target, **loss_kwargs)

    reg = torch.tensor(0.0, device=output.device)

    # Width penalty (Gaussian sigma)
    if hasattr(model, 'fuzzify') and hasattr(model.fuzzify, 'sigma'):
        reg = reg + reg_weight * (model.fuzzify.sigma ** 2).mean()

    # Overlap penalty: for each input, penalise close centres
    if hasattr(model, 'fuzzify') and hasattr(model.fuzzify, 'center'):
        c = model.fuzzify.center  # (n_inputs, n_terms)
        # Sort centres per input
        c_sorted, _ = torch.sort(c, dim=1)
        # Penalise small gaps between adjacent centres
        gaps = c_sorted[:, 1:] - c_sorted[:, :-1]
        reg = reg + overlap_pen * torch.exp(-10.0 * gaps).mean()

    return task_loss + reg
