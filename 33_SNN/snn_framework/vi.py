# vi.py
"""
Loss functions for Spiking Neural Network training.

Because SNNs use surrogate gradients rather than variational inference,
the analogy to the BNN's ELBO is a standard empirical loss applied to the
time-averaged membrane potential of the readout layer.

An optional activity regularisation term is included to prevent silent
(dead neuron) or saturated networks, which is the SNN equivalent of the
KL divergence regulariser in BNNs: both penalise degenerate representations.

Functions
---------
snn_loss   : unified loss dispatcher (regression / binary / multiclass).
activity_reg : L2 penalty on per-neuron mean firing rates.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from model import SpikingModel


# ---------------------------------------------------------------------------
# Activity regularisation
# ---------------------------------------------------------------------------

def activity_reg(model: SpikingModel, x: torch.Tensor,
                 target_rate: float = 0.1,
                 weight: float = 1e-3) -> torch.Tensor:
    """
    Penalise deviations of the mean spike rate from `target_rate`.

    This encourages sparse, information-rich representations and prevents
    the network from collapsing to an all-silent or all-firing state.

    Parameters
    ----------
    model : SpikingModel
    x : Tensor
        A batch of inputs used to estimate firing rates.
    target_rate : float
        Desired mean firing rate per neuron (default 0.1 = 10 %).
    weight : float
        Scaling coefficient λ for the regularisation term.

    Returns
    -------
    reg : scalar Tensor
    """
    rates = model.spike_rates(x)
    if not rates:
        return torch.tensor(0.0, requires_grad=False)
    mean_rate = torch.tensor(list(rates.values())).mean()
    return weight * (mean_rate - target_rate) ** 2


# ---------------------------------------------------------------------------
# Unified SNN loss
# ---------------------------------------------------------------------------

def snn_loss(output: torch.Tensor, target: torch.Tensor,
             model: Optional[SpikingModel] = None,
             x: Optional[torch.Tensor] = None,
             likelihood: str = 'classification',
             activity_weight: float = 0.0,
             target_rate: float = 0.1) -> torch.Tensor:
    """
    Compute the training loss for an SNN.

    The output is the time-averaged membrane potential returned by
    ``SpikingModel.forward()``.

    Parameters
    ----------
    output : Tensor of shape (batch, out_features) or (batch, 1)
        Network output (logits or regression values).
    target : Tensor
        Ground-truth labels or values.
    model : SpikingModel or None
        Required only when ``activity_weight > 0``.
    x : Tensor or None
        Input batch, required for activity regularisation.
    likelihood : str
        'classification' — cross-entropy (multiclass).
        'binary'         — binary cross-entropy with logits.
        'regression'     — mean-squared error.
    activity_weight : float
        Weight λ for activity regularisation (0 = disabled).
    target_rate : float
        Target firing rate for regularisation.

    Returns
    -------
    loss : scalar Tensor
    """
    if likelihood == 'classification':
        loss = F.cross_entropy(output, target, reduction='mean')

    elif likelihood == 'binary':
        loss = F.binary_cross_entropy_with_logits(
            output.squeeze(-1), target.float(), reduction='mean'
        )

    elif likelihood == 'regression':
        loss = F.mse_loss(output.squeeze(-1), target.float(),
                          reduction='mean')

    else:
        raise ValueError(
            "likelihood must be 'classification', 'binary', or 'regression'."
        )

    if activity_weight > 0.0 and model is not None and x is not None:
        loss = loss + activity_reg(model, x, target_rate, activity_weight)

    return loss