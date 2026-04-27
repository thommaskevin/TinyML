# model.py
"""
Spiking Neural Network model container.

SpikingModel wraps an ordered sequence of layers (SpikingLinear,
SpikingConv2d, LeakyReadout, nn.Flatten, etc.) and handles the
temporal unrolling required by spiking dynamics.

Key design decisions that mirror the BNN codebase
--------------------------------------------------
* The constructor receives a plain Python list of layer instances,
  which is registered as an nn.ModuleList for proper parameter tracking.
* A single `forward()` call unrolls the network for `num_steps` time
  steps and aggregates spike counts (for hidden layers) or membrane
  potential traces (for the readout).
* Input encoding strategies (rate coding, latency coding, direct
  current injection) are supported through the `encoding` argument.
"""

import torch
import torch.nn as nn
from typing import Dict
from layers import SpikingLinear, SpikingConv2d, LeakyReadout


class SpikingModel(nn.Module):
    """
    Container for a sequence of spiking (and non-spiking) layers.

    Parameters
    ----------
    layers : list[nn.Module]
        Ordered list of layers. The final element must be a LeakyReadout
        for the model to produce a meaningful scalar or vector output.
    num_steps : int
        Number of simulation time steps T.
    encoding : str
        Input encoding strategy applied before temporal unrolling:
          'rate'    — Bernoulli rate coding: spike_in[t] ~ Bernoulli(x).
                      Requires x ∈ [0, 1].
          'repeat'  — Repeat the input tensor unchanged at every step.
                      Suitable for normalised continuous inputs injected
                      as a constant current.
          'latency' — Not implemented here; provided as a placeholder.
    """

    def __init__(self, layers: list, num_steps: int = 25,
                 encoding: str = 'repeat'):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_steps = num_steps
        self.encoding = encoding

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(self, batch_size: int, device: torch.device):
        """Reset all stateful (spiking) layers before a new sequence."""
        for layer in self.layers:
            if isinstance(layer, (SpikingLinear, LeakyReadout)):
                layer.reset_state(batch_size, device)
            elif isinstance(layer, SpikingConv2d):
                # Spatial shape unknown until first forward; handled lazily.
                layer._u = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Return the input spike tensor for time step t."""
        if self.encoding == 'rate':
            return torch.bernoulli(x.clamp(0.0, 1.0))
        elif self.encoding == 'repeat':
            return x           # constant current injection
        else:
            raise ValueError(f"Unknown encoding '{self.encoding}'. "
                             "Choose 'rate' or 'repeat'.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                return_membrane: bool = False) -> torch.Tensor:
        """
        Unroll the network for `num_steps` time steps.

        Parameters
        ----------
        x : Tensor of shape (batch, features) or (batch, C, H, W)
            Static input (encoded independently at each step).
        return_membrane : bool
            If True, return the full membrane trace of the readout layer
            as a Tensor of shape (T, batch, out_features). Otherwise,
            return the time-averaged membrane potential.

        Returns
        -------
        output : Tensor of shape (batch, out_features)
            Time-averaged membrane potential of the final LeakyReadout,
            suitable for use as a logit or regression value.
        """
        batch_size = x.shape[0]
        device = x.device
        self.reset_states(batch_size, device)

        membrane_trace = []

        for t in range(self.num_steps):
            spike = self._encode(x, t)

            for layer in self.layers:
                if isinstance(layer, (SpikingLinear, SpikingConv2d,
                                      LeakyReadout)):
                    spike = layer(spike)
                else:
                    # Deterministic pass-through layers (nn.Flatten, etc.)
                    spike = layer(spike)

            membrane_trace.append(spike)          # spike here is u from readout

        # Stack: (T, batch, out_features)
        trace = torch.stack(membrane_trace, dim=0)

        if return_membrane:
            return trace

        # Time-averaged membrane potential as the network output
        return trace.mean(dim=0)

    # ------------------------------------------------------------------
    # Deterministic inference — matches C++ generated code exactly
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic inference pass that exactly mirrors the C++ predict().

        Key differences from forward():
          - Always uses 'repeat' encoding (constant current injection),
            regardless of self.encoding. This matches the C++ behaviour,
            where input[i] is copied unchanged at every time step.
          - Decorated with @torch.no_grad() for efficiency.
          - Use this method when comparing Python output to Arduino output.

        Parameters
        ----------
        x : Tensor of shape (batch, features)

        Returns
        -------
        output : Tensor of shape (batch, out_features)
            Time-averaged membrane potential — identical to C++ readout_acc / T.
        """
        batch_size = x.shape[0]
        device = x.device
        self.reset_states(batch_size, device)

        membrane_trace = []

        for _ in range(self.num_steps):
            spike = x                   # 'repeat' encoding: constant injection
            for layer in self.layers:
                spike = layer(spike)
            membrane_trace.append(spike)

        trace = torch.stack(membrane_trace, dim=0)   # (T, batch, out_features)
        return trace.mean(dim=0)

    # ------------------------------------------------------------------
    # Spike rate analysis (diagnostic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def spike_rates(self, x: torch.Tensor) -> dict:
        """
        Compute the mean firing rate of each spiking layer.

        Returns a dictionary mapping layer index → mean spike rate (float).
        This is useful for verifying that the network is neither silent
        nor saturated.
        """
        batch_size = x.shape[0]
        device = x.device
        self.reset_states(batch_size, device)

        rates: Dict[int, list] = {
            i: [] for i, l in enumerate(self.layers)
            if isinstance(l, SpikingLinear)
        }

        for t in range(self.num_steps):
            current = self._encode(x, t)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, (SpikingLinear, SpikingConv2d,
                                      LeakyReadout)):
                    current = layer(current)
                else:
                    current = layer(current)
                if i in rates:
                    rates[i].append(current.mean().item())

        return {i: float(torch.tensor(v).mean()) for i, v in rates.items()}