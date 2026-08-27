# model.py
"""
Quantile Regression Neural Network (QRNN) model.

Contents
--------
- QRModel : Feed-forward network for single or simultaneous multi-quantile
            estimation with optional crossing-prevention head.

Architecture
------------
The QRModel stacks an arbitrary number of ``DenseLayer`` hidden blocks
followed by a ``QuantileHead`` output layer:

    Input x ∈ ℝ^{d_in}
       ↓  DenseLayer 1   (in_features → hidden_size, activation, dropout)
       ↓  DenseLayer 2
       ⋮
       ↓  DenseLayer L
       ↓  QuantileHead   (hidden_size → K quantiles, monotonicity enforced)
    Output  ŷ ∈ ℝ^K

For a single quantile (K = 1), the QuantileHead reduces to a plain linear
layer and the model is equivalent to a standard quantile regression network
as proposed by Taylor (2000).

For K > 1, the QuantileHead applies a cumulative-sum reparameterisation that
guarantees ŷ_{τ_1} ≤ … ≤ ŷ_{τ_K} for every input, eliminating quantile
crossing without any constraint on the optimiser.

Configuration format
--------------------
``hidden_layers`` — list of dicts, one per hidden block:

.. code-block:: python

    [
        {'in_features': 4, 'out_features': 64, 'activation': 'relu',
         'dropout': 0.1},
        {'in_features': 64, 'out_features': 64, 'activation': 'relu'},
    ]

Required keys:

- ``in_features``  : Input dimension.
- ``out_features`` : Output dimension of the block.

Optional keys (all have sensible defaults):

- ``activation``   : Activation name (default: ``'relu'``).
- ``bias``         : Whether to include bias (default: ``True``).
- ``dropout``      : Dropout rate ∈ [0, 1) during training (default: ``0.0``).

The ``in_features`` of the ``QuantileHead`` is inferred automatically from
the ``out_features`` of the last hidden layer.

Input / output shapes
---------------------
- Input  : ``(batch, in_features)``  — standard tabular input.
- Output : ``(batch, K)``            — K quantile estimates per sample.
           ``(batch, 1)``            — single quantile (K = 1), squeezable
                                       to ``(batch,)``.
"""

import torch
import torch.nn as nn

from layers import DenseLayer, QuantileHead


class QRModel(nn.Module):
    """
    Feed-forward Quantile Regression Neural Network.

    Args
    ----
    hidden_layers : List of dicts configuring each ``DenseLayer`` block.
                    See the module docstring for the expected keys.
    quantiles     : Sorted list of target quantile levels in (0, 1).
                    Pass a single-element list for single-quantile models.
                    Default: ``[0.1, 0.5, 0.9]``.
    bias          : Whether the ``QuantileHead`` uses a bias term.
                    Default: ``True``.

    Example
    -------
    .. code-block:: python

        model = QRModel(
            hidden_layers=[
                {'in_features': 1, 'out_features': 64, 'activation': 'relu'},
                {'in_features': 64, 'out_features': 64, 'activation': 'relu'},
            ],
            quantiles=[0.1, 0.5, 0.9],
        )
        x = torch.randn(32, 1)
        q_hat = model(x)          # shape (32, 3)
    """

    def __init__(
        self,
        hidden_layers: list,
        quantiles:     list = None,
        bias:          bool = True,
    ) -> None:
        super().__init__()

        if not hidden_layers:
            raise ValueError("'hidden_layers' must contain at least one entry.")

        self.quantiles = sorted(quantiles or [0.1, 0.5, 0.9])
        self.K         = len(self.quantiles)

        # ----- Hidden backbone -----
        self.backbone = nn.ModuleList()
        for cfg in hidden_layers:
            layer = DenseLayer(
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
                activation=cfg.get('activation', 'relu'),
                bias=cfg.get('bias', True),
                dropout=cfg.get('dropout', 0.0),
            )
            self.backbone.append(layer)

        # ----- Quantile output head -----
        last_size = hidden_layers[-1]['out_features']
        self.head  = QuantileHead(
            in_features=last_size,
            quantiles=self.quantiles,
            bias=bias,
        )

        # Store architecture for serialisation
        self.hidden_configs  = hidden_layers
        self.quantile_levels = self.quantiles

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x : Input tensor of shape ``(batch, in_features)``.

        Returns:
            Quantile predictions of shape ``(batch, K)``.  Column *k*
            estimates the conditional quantile at level ``self.quantiles[k]``.
        """
        out = x
        for layer in self.backbone:
            out = layer(out)
        return self.head(out)

    # ------------------------------------------------------------------
    def predict_quantile(
        self,
        x:        torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """
        Return the prediction for a specific quantile level.

        Args:
            x        : Input tensor of shape ``(batch, in_features)``.
            quantile : One of the quantile levels the model was trained on.

        Returns:
            Tensor of shape ``(batch,)`` with predictions at the requested
            quantile level.

        Raises:
            ValueError: if *quantile* was not included during training.
        """
        if quantile not in self.quantiles:
            raise ValueError(
                f"Quantile {quantile} not in model quantiles {self.quantiles}. "
                "Re-train the model including this level."
            )
        idx = self.quantiles.index(quantile)
        with torch.no_grad():
            return self(x)[:, idx]

    # ------------------------------------------------------------------
    def predict_interval(
        self,
        x:     torch.Tensor,
        lower: float = 0.1,
        upper: float = 0.9,
    ) -> tuple:
        """
        Return the prediction interval defined by two quantile levels.

        Args:
            x     : Input tensor of shape ``(batch, in_features)``.
            lower : Lower quantile level (must be in ``self.quantiles``).
            upper : Upper quantile level (must be in ``self.quantiles``).

        Returns:
            Tuple ``(lower_bound, median, upper_bound)`` of shape
            ``(batch,)`` each.  ``median`` is returned only if 0.5 is
            in ``self.quantiles``, otherwise ``None``.
        """
        with torch.no_grad():
            q_hat = self(x)                          # (batch, K)

        lo = q_hat[:, self.quantiles.index(lower)]
        hi = q_hat[:, self.quantiles.index(upper)]
        med = (
            q_hat[:, self.quantiles.index(0.5)]
            if 0.5 in self.quantiles else None
        )
        return lo, med, hi

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
