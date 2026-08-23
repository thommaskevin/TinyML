# layers.py
"""
Layer definitions for the Quantile Regression (QR) framework.

Contents
--------
- Activation functions  (registry and factory helper)
- PinballLoss           : differentiable pinball / check loss per quantile
- QuantileHead          : multi-quantile output head with crossing prevention
- DenseLayer            : fully-connected layer with optional activation

Background
----------
Quantile regression, introduced by Koenker & Bassett (1978), estimates
conditional quantiles Q_τ(Y | X) of the response distribution rather than
the conditional mean.  For a target quantile τ ∈ (0, 1), the *pinball*
(check) loss function is:

    L_τ(y, ŷ) = (y − ŷ) · τ           if  y − ŷ ≥ 0
                (ŷ − y) · (1 − τ)       if  y − ŷ <  0

Minimising the expected pinball loss over τ yields an estimate of Q_τ(Y|X).
A neural network trained simultaneously on K quantile levels
τ₁ < τ₂ < … < τ_K provides a full distributional forecast: mean, median,
prediction intervals, and tail risk — all in a single forward pass.

A key practical challenge is *quantile crossing*: the raw network outputs
for different τ may violate monotonicity (ŷ_τ₁ > ŷ_τ₂ for τ₁ < τ₂).
The QuantileHead layer enforces monotonicity via a sorted cumulative-sum
reparameterisation of the output deltas.

References
----------
Koenker, R., & Bassett, G. (1978).
    Regression Quantiles. *Econometrica*, 46(1), 33–50.
Taylor, J. W. (2000).
    A Quantile Regression Neural Network Approach to Estimating the
    Conditional Density of Multiperiod Returns.  *Journal of Forecasting*.
Cannon, A. J. (2011).
    Quantile Regression Neural Networks: Implementation in R and Application
    to Precipitation Downscaling.  *Computers & Geosciences*, 37(9).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Activation Functions
# =============================================================================

class Tanh(nn.Module):
    """Hyperbolic tangent activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Sigmoid(nn.Module):
    """Logistic sigmoid activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class ReLU(nn.Module):
    """Rectified Linear Unit activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class Softmax(nn.Module):
    """Softmax activation along a specified dimension."""
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class LeakyReLU(nn.Module):
    """Leaky ReLU activation."""
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class Swish(nn.Module):
    """Swish (SiLU) activation: x · sigmoid(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Linear(nn.Module):
    """Identity (linear) activation — no-op."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Registry mapping lowercase name → class
ACTIVATIONS: dict = {
    'tanh':       Tanh,
    'sigmoid':    Sigmoid,
    'relu':       ReLU,
    'softmax':    Softmax,
    'leaky_relu': LeakyReLU,
    'gelu':       GELU,
    'swish':      Swish,
    'linear':     Linear,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Instantiate and return an activation module by name (case-insensitive).

    Args:
        name    : Activation name, e.g. ``'relu'``, ``'tanh'``, ``'gelu'``.
        **kwargs: Keyword arguments forwarded to the activation constructor
                  (e.g. ``dim=-1`` for Softmax).

    Returns:
        An ``nn.Module`` implementing the requested activation.

    Raises:
        ValueError: if *name* is not found in the registry.
    """
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available activations: {sorted(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[key](**kwargs)


# =============================================================================
# Pinball Loss Layer
# =============================================================================

class PinballLoss(nn.Module):
    """
    Differentiable pinball (check) loss for a single quantile level τ.

    The pinball loss is the canonical training objective for quantile
    regression.  For a batch of predictions ``ŷ`` and targets ``y``:

        L_τ(y, ŷ) = mean(max(τ·(y − ŷ),  (τ − 1)·(y − ŷ)))

    which is equivalent to:

        L_τ(y, ŷ) = mean((y − ŷ) · τ)              for residuals ≥ 0
                    mean((ŷ − y) · (1 − τ))          for residuals <  0

    When τ = 0.5, the pinball loss reduces to the Mean Absolute Error (MAE),
    so the median is the solution that minimises the MAE loss.

    Args:
        quantile : The target quantile level τ ∈ (0, 1).  Default is 0.5
                   (median regression).
        reduction: One of ``'mean'`` or ``'sum'``.  Default is ``'mean'``.

    Raises:
        ValueError: if *quantile* is not strictly in (0, 1).
    """

    def __init__(self, quantile: float = 0.5, reduction: str = 'mean') -> None:
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError(
                f"Quantile must be strictly in (0, 1); got {quantile}."
            )
        self.quantile  = quantile
        self.reduction = reduction.lower()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the pinball loss.

        Args:
            output : Predicted quantile values of shape ``(batch,)`` or
                     ``(batch, 1)``.
            target : Ground-truth outcomes of the same shape as *output*.

        Returns:
            Scalar pinball loss tensor.
        """
        output = output.squeeze()
        target = target.squeeze().float()
        residual = target - output
        loss = torch.where(
            residual >= 0,
            self.quantile * residual,
            (self.quantile - 1.0) * residual,
        )
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()


# =============================================================================
# Dense (hidden) Layer
# =============================================================================

class DenseLayer(nn.Module):
    """
    Fully-connected layer with an optional activation function.

    This is the standard building block for the hidden layers of the
    quantile regression network.  Each ``DenseLayer`` applies a linear
    transformation followed by a nonlinear activation:

        y = activation(W · x + b)

    Args:
        in_features  : Number of input features.
        out_features : Number of output features.
        activation   : Activation function name (default: ``'relu'``).
        bias         : Whether to include a bias term (default: ``True``).
        dropout      : Dropout probability applied *before* the activation
                       during training.  Set to 0.0 to disable (default).
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   str   = 'relu',
        bias:         bool  = True,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features     = in_features
        self.out_features    = out_features
        self.activation_name = activation.lower()

        self.linear  = nn.Linear(in_features, out_features, bias=bias)
        self.act     = get_activation(self.activation_name)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for ReLU-family activations;
        Xavier uniform for tanh / sigmoid."""
        act = self.activation_name
        if act in ('relu', 'leaky_relu', 'gelu', 'swish'):
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        out = self.linear(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return self.act(out)


# =============================================================================
# Quantile Head — multi-quantile output with crossing prevention
# =============================================================================

class QuantileHead(nn.Module):
    """
    Multi-quantile output head with monotonicity enforcement.

    Maps the penultimate hidden representation to ``K`` quantile estimates
    simultaneously.  Quantile crossing — where the raw network output
    violates ŷ_{τ_i} ≤ ŷ_{τ_{i+1}} — is prevented by a sorted
    cumulative-sum reparameterisation:

    1. Predict a *base* quantile ŷ₁ (the lowest requested quantile) and
       K−1 non-negative *deltas* δ₂, …, δ_K via softplus(·).
    2. Reconstruct the ordered quantile estimates as cumulative sums:

            ŷ_{τ_k} = ŷ₁ + Σ_{j=2}^{k} softplus(δ_j)

    This guarantees ŷ_{τ₁} ≤ ŷ_{τ₂} ≤ … ≤ ŷ_{τ_K} for any input,
    without imposing any constraint on the loss or the optimiser.

    Args:
        in_features : Number of input features from the penultimate layer.
        quantiles   : Sorted list or tuple of quantile levels,
                      e.g. ``[0.1, 0.5, 0.9]``.  Must be strictly
                      increasing and all values must be in (0, 1).
        bias        : Whether to include a bias term (default: ``True``).

    Raises:
        ValueError: if *quantiles* is not strictly increasing or contains
                    values outside (0, 1).
    """

    def __init__(
        self,
        in_features: int,
        quantiles:   list,
        bias:        bool = True,
    ) -> None:
        super().__init__()
        quantiles = sorted(quantiles)
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(
                    f"All quantile levels must be in (0, 1); got {q}."
                )
        self.in_features = in_features
        self.quantiles   = quantiles
        self.K           = len(quantiles)

        # One linear layer: first output = base quantile, remaining = raw deltas
        self.linear = nn.Linear(in_features, self.K, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute K crossing-free quantile estimates.

        Args:
            x : Penultimate feature tensor of shape ``(batch, in_features)``.

        Returns:
            Quantile estimates of shape ``(batch, K)`` in ascending order,
            with column *k* corresponding to ``self.quantiles[k]``.
        """
        raw = self.linear(x)                         # (batch, K)

        if self.K == 1:
            return raw                               # no crossing possible

        # Base quantile (unconstrained) + non-negative increments
        base   = raw[:, :1]                          # (batch, 1)
        deltas = F.softplus(raw[:, 1:])              # (batch, K-1), all ≥ 0
        increments = torch.cat([base, deltas], dim=1)# (batch, K)
        return torch.cumsum(increments, dim=1)       # (batch, K), monotone
