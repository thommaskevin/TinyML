# layers.py
"""
Layer definitions for the Elman RNN framework.

Contents
--------
- Activation functions (with registry and factory helper)
- RNNCell   : single Elman recurrent cell
- DenseLayer: fully-connected output layer with optional activation
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
    """Swish (SiLU) activation: x * sigmoid(x)."""
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
        name   : activation name, e.g. ``'tanh'``, ``'relu'``, ``'softmax'``.
        **kwargs: keyword arguments forwarded to the activation constructor
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
# RNN Cell (Elman)
# =============================================================================

class RNNCell(nn.Module):
    """
    Single-step Elman RNN cell.

    Computes the new hidden state as:

        h_t = activation(W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh)

    Args:
        input_size  : Dimensionality of the input vector ``x_t``.
        hidden_size : Dimensionality of the hidden state ``h_t``.
        activation  : Name of the activation function (default: ``'tanh'``).
        bias        : Whether to include bias terms (default: ``True``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = 'tanh',
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation.lower()

        self.W_ih = nn.Linear(input_size,  hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.act  = get_activation(self.activation_name)

        self._init_weights()

    def _init_weights(self) -> None:
        """Uniform initialisation in [-1/√H, 1/√H] for all parameters."""
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform one recurrent step.

        Args:
            x      : Input tensor of shape ``(batch, input_size)``.
            h_prev : Previous hidden state of shape ``(batch, hidden_size)``,
                     or ``None`` to use an all-zero initial state.

        Returns:
            h_t : New hidden state of shape ``(batch, hidden_size)``.
        """
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.hidden_size, device=x.device, dtype=x.dtype
            )
        return self.act(self.W_ih(x) + self.W_hh(h_prev))


# =============================================================================
# Dense (output) Layer
# =============================================================================

class DenseLayer(nn.Module):
    """
    Fully-connected layer with an optional activation function.

    Args:
        in_features  : Number of input features.
        out_features : Number of output features.
        activation   : Name of the activation function (default: ``'linear'``).
        bias         : Whether to include a bias term (default: ``True``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'linear',
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act    = get_activation(self.activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        return self.act(self.linear(x))