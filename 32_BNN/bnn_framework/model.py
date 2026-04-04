# model.py
import torch.nn as nn
from layers import BayesianLinear, BayesianConv2d

class BayesianModel(nn.Module):
    """
    Container for a sequence of layers (both Bayesian and deterministic).
    Provides a method to compute the total KL loss.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True):
        """Forward pass through all layers."""
        for layer in self.layers:
            if isinstance(layer, (BayesianLinear, BayesianConv2d)):
                x = layer(x, sample)
            else:
                x = layer(x)   # deterministic layers (ReLU, etc.)
        return x

    def kl_loss(self):
        """Sum KL divergences of all Bayesian layers."""
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'kl_divergence'):
                kl += layer.kl_divergence()
        return kl