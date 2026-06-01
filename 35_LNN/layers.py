# layers.py
"""
Layer definitions for the Liquid Neural Network (LNN / LTC) framework.

Contents
--------
- Activation functions (with registry and factory helper)
- LTCCell    : single Liquid Time-Constant (LTC) cell with ODE integration
- DenseLayer : fully-connected output layer with optional activation

Background
----------
Liquid Time-Constant Networks are continuous-time recurrent neural networks
whose hidden-state dynamics are governed by a system of ordinary differential
equations (ODEs).  At each time step the state is updated by numerically
integrating the ODE

    τ(x,h) · dh/dt = -h + f(W_ih·x + W_hh·h + b)

where τ(x,h) > 0 is an *input-dependent* time constant.  The network is
called "liquid" because its effective time constant — and therefore its
temporal sensitivity — changes with the input signal.

References
----------
Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
    Liquid Time-constant Networks. *AAAI 2021*.
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
        name    : Activation name, e.g. ``'tanh'``, ``'relu'``, ``'softmax'``.
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
# LTC Cell (Liquid Time-Constant)
# =============================================================================

class LTCCell(nn.Module):
    """
    Single-step Liquid Time-Constant (LTC) cell.

    The hidden-state ODE is solved with a fixed-step Euler method over
    *ode_unfolds* sub-steps per input time step.  The continuous-time update
    equation is:

        dh/dt = (-h + f(W_ih·x + W_hh·h + b_ih + b_hh)) / τ(x, h)

    where the input-dependent time constant τ(x, h) is computed as:

        τ(x, h) = τ_min + softplus(A · sigmoid(W_τ·x + W_τh·h + b_τ))

    with τ_min a small positive floor that prevents τ from collapsing to zero.

    Euler integration over *ode_unfolds* micro-steps of size ``dt / ode_unfolds``:

        h_{t+δ} = h_t + δ · dh/dt

    Args:
        input_size   : Dimensionality of the input vector ``x_t``.
        hidden_size  : Dimensionality of the hidden (liquid) state ``h_t``.
        activation   : Name of the backbone activation (default: ``'tanh'``).
        ode_unfolds  : Number of Euler sub-steps per input time step
                       (default: ``6``).
        dt           : Nominal time-step size in continuous time
                       (default: ``1.0``).
        tau_min      : Minimum time constant to ensure numerical stability
                       (default: ``0.1``).
        bias         : Whether to include bias terms (default: ``True``).
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        activation:  str   = 'tanh',
        ode_unfolds: int   = 6,
        dt:          float = 1.0,
        tau_min:     float = 0.1,
        bias:        bool  = True,
    ) -> None:
        super().__init__()
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.activation_name = activation.lower()
        self.ode_unfolds     = ode_unfolds
        self.dt              = dt
        self.tau_min         = tau_min

        # Backbone synaptic weights (input-to-hidden and hidden-to-hidden)
        self.W_ih = nn.Linear(input_size,  hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Time-constant modulation weights
        self.W_tau  = nn.Linear(input_size,  hidden_size, bias=bias)
        self.W_tauh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable amplitude for the softplus term; initialised to 1
        self.A = nn.Parameter(torch.ones(hidden_size))

        # Backbone activation
        self.act = get_activation(self.activation_name)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Uniform initialisation in [-1/√H, 1/√H] for all parameters.

        The amplitude parameter A is initialised to 1 (log-scale ≈ 0.5413
        before softplus) which gives an initial extra time-constant of ≈ 1.
        """
        std = 1.0 / math.sqrt(self.hidden_size)
        for name, p in self.named_parameters():
            if 'A' not in name:
                nn.init.uniform_(p, -std, std)
        nn.init.ones_(self.A)

    # ------------------------------------------------------------------
    def _time_constant(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the input-dependent time constant τ(x, h).

        τ(x, h) = τ_min + softplus(A · σ(W_τ · x + W_τh · h))

        Returns:
            τ tensor of shape ``(batch, hidden_size)``, strictly > τ_min.
        """
        gate = torch.sigmoid(self.W_tau(x) + self.W_tauh(h))
        return self.tau_min + F.softplus(self.A * gate)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:      torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform one recurrent step with Euler ODE integration.

        Args:
            x      : Input tensor of shape ``(batch, input_size)``.
            h_prev : Previous hidden state of shape ``(batch, hidden_size)``,
                     or ``None`` to use an all-zero initial state.

        Returns:
            h_t : New hidden state of shape ``(batch, hidden_size)``.
        """
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0), self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h_prev
        delta = self.dt / self.ode_unfolds          # micro-step size

        for _ in range(self.ode_unfolds):
            # Synaptic input
            f_val = self.act(self.W_ih(x) + self.W_hh(h))
            # Input-dependent time constant
            tau   = self._time_constant(x, h)
            # Euler step:  dh = (-h + f) / tau
            dh    = (-h + f_val) / tau
            h     = h + delta * dh

        return h


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
        in_features:  int,
        out_features: int,
        activation:   str  = 'linear',
        bias:         bool = True,
    ) -> None:
        super().__init__()
        self.in_features      = in_features
        self.out_features     = out_features
        self.activation_name  = activation.lower()

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
