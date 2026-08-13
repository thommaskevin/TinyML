# losses.py
"""
Loss functions for Liquid Neural Network (LNN / LTC) training.

Supported losses
----------------
mse   : Mean Squared Error
mae   : Mean Absolute Error
rmse  : Root Mean Squared Error
bce   : Binary Cross-Entropy (with logits, numerically stable)
cce   : Categorical Cross-Entropy  (one-hot float targets)
scce  : Sparse Categorical Cross-Entropy (integer class targets)
huber : Huber / Smooth-L1 loss
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error: mean((ŷ − y)²)."""
    return F.mse_loss(output.squeeze(), target.squeeze())


def mae_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error: mean(|ŷ − y|)."""
    return F.l1_loss(output.squeeze(), target.squeeze())


def rmse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error: √MSE.  An epsilon is added for numerical stability."""
    return torch.sqrt(F.mse_loss(output.squeeze(), target.squeeze()) + 1e-8)


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross-Entropy computed from raw logits (numerically stable).

    Args:
        output : Raw logit predictions, any shape.
        target : Binary targets (0 or 1), same shape as *output*.
    """
    return F.binary_cross_entropy_with_logits(
        output.squeeze(), target.float().squeeze()
    )


def cce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Categorical Cross-Entropy with one-hot float targets.

    Args:
        output : Raw logits of shape ``(batch, num_classes)``.
        target : One-hot float targets of shape ``(batch, num_classes)``.
    """
    log_probs = F.log_softmax(output, dim=-1)
    return -(target.float() * log_probs).sum(dim=-1).mean()


def scce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Sparse Categorical Cross-Entropy with integer class targets.

    Args:
        output : Raw logits of shape ``(batch, num_classes)``.
        target : Integer class indices of shape ``(batch,)``.
    """
    return F.cross_entropy(output, target.long())


def huber_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Huber (Smooth-L1) loss.

    Behaves like MSE for |ŷ − y| ≤ *delta* and like MAE for larger errors.

    Args:
        output : Predictions, any shape.
        target : Ground-truth values, same shape as *output*.
        delta  : Threshold between the quadratic and linear regions
                 (default: ``1.0``).
    """
    return F.huber_loss(output.squeeze(), target.squeeze(), delta=delta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOSS_FN: dict = {
    'mse':   mse_loss,
    'mae':   mae_loss,
    'rmse':  rmse_loss,
    'bce':   bce_loss,
    'cce':   cce_loss,
    'scce':  scce_loss,
    'huber': huber_loss,
}

LOSS_NAMES: list[str] = list(_LOSS_FN.keys())


def compute_loss(
    loss_name: str,
    output:    torch.Tensor,
    target:    torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Compute a named loss between *output* and *target*.

    Args:
        loss_name : One of ``'mse'``, ``'mae'``, ``'rmse'``, ``'bce'``,
                    ``'cce'``, ``'scce'``, or ``'huber'``.
        output    : Model predictions.
        target    : Ground-truth labels.
        **kwargs  : Extra keyword arguments forwarded to the loss function
                    (e.g. ``delta=0.5`` for Huber).

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: if *loss_name* is not registered.
    """
    key = loss_name.lower()
    if key not in _LOSS_FN:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available losses: {LOSS_NAMES}"
        )
    return _LOSS_FN[key](output, target, **kwargs)
