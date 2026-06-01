# model.py
"""
Liquid Neural Network (LNN) model with an arbitrary number of stacked
Liquid Time-Constant (LTC) recurrent layers followed by a fully-connected
dense head.
"""

import torch
import torch.nn as nn

from layers import LTCCell, DenseLayer


class LNNModel(nn.Module):
    """
    Stacked Liquid Time-Constant (LTC) neural network model.

    Architecture
    ------------
    The network consists of one or more ``LTCCell`` layers arranged in
    sequence (the output hidden state of layer *i* is fed as the input to
    layer *i+1*), followed by one or more ``DenseLayer`` modules that map
    the final hidden state to the desired output.

    Each LTC cell integrates an ODE over ``ode_unfolds`` Euler sub-steps per
    input time step, using an input-dependent time constant τ(x, h) that
    allows the network's effective temporal sensitivity to adapt to the
    signal.

    Configuration format
    --------------------
    ``recurrent_layers`` — list of dicts, one per LTC layer:

    .. code-block:: python

        [
            {
                'input_size':  1,
                'hidden_size': 64,
                'activation':  'tanh',
                'ode_unfolds': 6,
                'dt':          1.0,
                'tau_min':     0.1,
            },
            {
                'input_size':  64,
                'hidden_size': 32,
            },
        ]

    Required keys per dict:

    - ``input_size``  : Dimensionality of the input at that layer.
    - ``hidden_size`` : Dimensionality of the hidden / liquid state.

    Optional keys per dict (all have sensible defaults):

    - ``activation``  : Backbone activation name (default: ``'tanh'``).
    - ``ode_unfolds`` : Euler micro-steps per input step (default: ``6``).
    - ``dt``          : Nominal continuous-time step (default: ``1.0``).
    - ``tau_min``     : Minimum time constant floor (default: ``0.1``).
    - ``bias``        : Whether to use bias terms (default: ``True``).

    ``dense_layers`` — list of dicts, one per dense layer:

    .. code-block:: python

        [
            {'out_features': 32, 'activation': 'relu'},
            {'out_features': 1,  'activation': 'linear'},
        ]

    Required keys per dict:

    - ``out_features`` : Number of output neurons.

    Optional keys per dict:

    - ``activation``   : Activation name (default: ``'linear'``).
    - ``bias``         : Whether to use bias terms (default: ``True``).

    The ``in_features`` of the first dense layer is inferred automatically
    from the ``hidden_size`` of the last recurrent layer.

    Input / output shapes
    ---------------------
    - Input  : ``(batch, seq_len, input_size)``
    - Output : ``(batch, output_size)``          — last time step only (``forward``)
    - Output : ``(batch, seq_len, output_size)`` — every time step (``forward_sequence``)
    """

    def __init__(
        self,
        recurrent_layers: list[dict],
        dense_layers:     list[dict],
    ) -> None:
        super().__init__()

        if not recurrent_layers:
            raise ValueError("'recurrent_layers' must contain at least one entry.")
        if not dense_layers:
            raise ValueError("'dense_layers' must contain at least one entry.")

        # ----- Liquid (recurrent) stack -----
        self.recurrent_cells = nn.ModuleList()

        for cfg in recurrent_layers:
            cell = LTCCell(
                input_size=cfg['input_size'],
                hidden_size=cfg['hidden_size'],
                activation=cfg.get('activation', 'tanh'),
                ode_unfolds=cfg.get('ode_unfolds', 6),
                dt=cfg.get('dt', 1.0),
                tau_min=cfg.get('tau_min', 0.1),
                bias=cfg.get('bias', True),
            )
            self.recurrent_cells.append(cell)

        # ----- Dense head -----
        self.dense_head = nn.ModuleList()
        prev_size: int = recurrent_layers[-1]['hidden_size']

        for cfg in dense_layers:
            layer = DenseLayer(
                in_features=prev_size,
                out_features=cfg['out_features'],
                activation=cfg.get('activation', 'linear'),
                bias=cfg.get('bias', True),
            )
            self.dense_head.append(layer)
            prev_size = cfg['out_features']

        # Store architecture configurations for serialisation
        self.recurrent_configs: list[dict] = recurrent_layers
        self.dense_configs:     list[dict] = dense_layers

    # ------------------------------------------------------------------
    def _recurrent_pass(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Run the full recurrent pass over the input sequence.

        Args:
            x : Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tuple ``(step_outputs, final_states)`` where:

            - ``step_outputs`` is a list of length *seq_len*, each element
              being a tensor of shape ``(batch, hidden_size_last_layer)``
              representing the output of the last LTC layer at that time step.
            - ``final_states`` is a list of the last hidden-state tensors,
              one per LTC layer.
        """
        batch, seq_len, _ = x.shape
        # Initialise hidden states to None (LTCCell handles zero init)
        states: list[torch.Tensor | None] = [None] * len(self.recurrent_cells)
        step_outputs: list[torch.Tensor] = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            for i, cell in enumerate(self.recurrent_cells):
                h_t = cell(x_t, states[i])
                states[i] = h_t
                x_t = h_t  # pass this layer's output as next layer's input

            step_outputs.append(x_t)  # x_t is the last layer's h_t

        return step_outputs, states  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def _apply_dense_head(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every dense layer in sequence."""
        for layer in self.dense_head:
            x = layer(x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Many-to-one forward pass.

        Processes the full input sequence and returns the output computed
        from the hidden state at the **last** time step only.

        Args:
            x : Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Output tensor of shape ``(batch, output_size)``.
        """
        step_outputs, _ = self._recurrent_pass(x)
        last_hidden = step_outputs[-1]          # (batch, hidden_size_last)
        return self._apply_dense_head(last_hidden)

    # ------------------------------------------------------------------
    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Many-to-many forward pass.

        Processes the full input sequence and returns the output at
        **every** time step.

        Args:
            x : Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, output_size)``.
        """
        step_outputs, _ = self._recurrent_pass(x)
        outputs = [self._apply_dense_head(h).unsqueeze(1) for h in step_outputs]
        return torch.cat(outputs, dim=1)        # (batch, seq_len, output_size)
