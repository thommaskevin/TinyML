# layers.py
"""
Spiking Neural Network layer definitions.

This module provides:
  - SpikingLinear  : a fully-connected layer whose neurons follow the
                     Leaky Integrate-and-Fire (LIF) dynamics.
  - SpikingConv2d  : a 2-D convolutional variant of the same dynamics.
  - SurrogateSpike : the non-differentiable Heaviside spike function
                     with an arc-tangent surrogate gradient.

All layers keep track of membrane potential across time steps and emit
binary spike tensors (0 or 1).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate gradient
# ---------------------------------------------------------------------------

class _SurrogateGradient(torch.autograd.Function):
    """
    Heaviside spike function with arc-tangent surrogate gradient.

    Forward  : H(u - threshold)  ∈ {0, 1}
    Backward : dL/du ≈ dL/dspike * (1 / (1 + (π * scale * u)²)) * scale

    The arc-tangent surrogate (Fang et al., 2021) approximates the
    derivative of the Heaviside more smoothly than the rectangular
    approximation, improving gradient flow in deep networks.
    """
    scale: float = 2.0   # sharpness of the surrogate

    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u,) = ctx.saved_tensors
        threshold = ctx.threshold
        # arc-tangent surrogate: derivative at (u - threshold)
        x = u - threshold
        sg = 1.0 / (1.0 + (math.pi * _SurrogateGradient.scale * x) ** 2)
        sg = sg * _SurrogateGradient.scale
        return grad_output * sg, None


def surrogate_spike(u: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Apply spike function with surrogate gradient."""
    return _SurrogateGradient.apply(u, threshold)


# ---------------------------------------------------------------------------
# Leaky Integrate-and-Fire (LIF) dynamics
# ---------------------------------------------------------------------------

class LIFDynamics(nn.Module):
    """
    Stateless helper that encapsulates LIF membrane update equations.

    Given pre-synaptic input current `I` and the previous membrane
    potential `u_prev`, it computes the new membrane potential and the
    corresponding spike tensor.

    Membrane update (discrete time):
        u[t] = beta * u[t-1] * (1 - spike[t-1]) + I[t]

    The reset-by-subtraction strategy is also supported:
        u[t] = beta * (u[t-1] - spike[t-1] * threshold) + I[t]

    Parameters
    ----------
    beta : float
        Membrane decay factor (leak), in (0, 1). Equivalent to exp(-dt/tau).
    threshold : float
        Firing threshold.
    reset_mode : str
        'zero'        — hard reset: membrane set to 0 after spike.
        'subtraction' — soft reset: threshold subtracted after spike.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0,
                 reset_mode: str = 'zero'):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset_mode = reset_mode

    def forward(self, I: torch.Tensor,
                u_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spike_prev = surrogate_spike(u_prev, self.threshold)

        if self.reset_mode == 'zero':
            u = self.beta * u_prev * (1.0 - spike_prev) + I
        else:  # subtraction
            u = self.beta * (u_prev - spike_prev * self.threshold) + I

        spike = surrogate_spike(u, self.threshold)
        return u, spike


# ---------------------------------------------------------------------------
# Spiking Linear layer
# ---------------------------------------------------------------------------

class SpikingLinear(nn.Module):
    """
    Fully-connected LIF layer.

    At each time step `t`, the layer:
      1. Computes the synaptic input current:  I[t] = W * spike_in[t] + b
      2. Updates the LIF membrane potential.
      3. Emits output spikes.

    The membrane state is maintained internally and must be reset between
    independent input sequences by calling `reset_state()`.

    Parameters
    ----------
    in_features : int
    out_features : int
    beta : float
        Membrane decay factor.
    threshold : float
        Spike threshold.
    reset_mode : str
        'zero' or 'subtraction'.
    bias : bool
        Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int,
                 beta: float = 0.9, threshold: float = 1.0,
                 reset_mode: str = 'zero', bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.lif = LIFDynamics(beta=beta, threshold=threshold,
                               reset_mode=reset_mode)

        self._u: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming uniform initialisation (He et al., 2015)."""
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def reset_state(self, batch_size: int = 1,
                    device: Optional[torch.device] = None):
        """Zero the membrane potential for a new sequence."""
        device = device or next(self.parameters()).device
        self._u = torch.zeros(batch_size, self.out_features, device=device)

    def forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        Single time-step forward pass.

        Parameters
        ----------
        spike_in : Tensor of shape (batch, in_features)
            Pre-synaptic spikes (binary, but gradients flow via surrogate).

        Returns
        -------
        spike_out : Tensor of shape (batch, out_features)
        """
        if self._u is None or self._u.shape[0] != spike_in.shape[0]:
            self.reset_state(spike_in.shape[0], spike_in.device)

        I = self.fc(spike_in)
        self._u, spike_out = self.lif(I, self._u)
        return spike_out

    @property
    def membrane_potential(self) -> Optional[torch.Tensor]:
        """Expose current membrane potential (read-only)."""
        return self._u


# ---------------------------------------------------------------------------
# Spiking Conv2d layer
# ---------------------------------------------------------------------------

class SpikingConv2d(nn.Module):
    """
    2-D convolutional LIF layer.

    Operates identically to SpikingLinear but applies a convolution in
    place of a linear projection.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding : standard
        PyTorch Conv2d arguments.
    beta : float
    threshold : float
    reset_mode : str
    bias : bool
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple, stride: int = 1, padding: int = 0,
                 beta: float = 0.9, threshold: float = 1.0,
                 reset_mode: str = 'zero', bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.lif = LIFDynamics(beta=beta, threshold=threshold,
                               reset_mode=reset_mode)

        self._u: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def reset_state(self, batch_size: int = 1, spatial_shape: tuple = (1, 1),
                    device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        self._u = torch.zeros(
            batch_size, self.out_channels, *spatial_shape, device=device
        )

    def forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        Single time-step forward pass.

        Parameters
        ----------
        spike_in : Tensor of shape (batch, in_channels, H, W)

        Returns
        -------
        spike_out : Tensor of shape (batch, out_channels, H', W')
        """
        I = self.conv(spike_in)
        if self._u is None or self._u.shape != I.shape:
            self.reset_state(I.shape[0], I.shape[2:], spike_in.device)
        self._u, spike_out = self.lif(I, self._u)
        return spike_out

    @property
    def membrane_potential(self) -> Optional[torch.Tensor]:
        return self._u


# ---------------------------------------------------------------------------
# Leaky readout (non-spiking output neuron)
# ---------------------------------------------------------------------------

class LeakyReadout(nn.Module):
    """
    Non-spiking integrator used as the output layer.

    Instead of emitting spikes, this layer accumulates incoming spikes into
    a leaky membrane potential over time, which is then used directly as the
    network's output (logit or regression value). This is consistent with
    the rate-coded readout strategy.

    Parameters
    ----------
    in_features : int
    out_features : int
    beta : float
        Membrane decay factor.
    bias : bool
    """

    def __init__(self, in_features: int, out_features: int,
                 beta: float = 0.9, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._u: Optional[torch.Tensor] = None

        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def reset_state(self, batch_size: int = 1,
                    device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        self._u = torch.zeros(batch_size, self.out_features, device=device)

    def forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spike_in : Tensor of shape (batch, in_features)

        Returns
        -------
        u : Tensor of shape (batch, out_features)
            Membrane potential (no spike emitted).
        """
        if self._u is None or self._u.shape[0] != spike_in.shape[0]:
            self.reset_state(spike_in.shape[0], spike_in.device)

        I = self.fc(spike_in)
        self._u = self.beta * self._u + I
        return self._u

    @property
    def membrane_potential(self) -> Optional[torch.Tensor]:
        return self._u