# layers.py
"""
Layer definitions for the Generative Adversarial Network (GAN) framework.

Contents
--------
- Activation functions (with registry and factory helper)
- SpectralLinear  : linear layer with optional spectral normalization
- ResidualBlock   : pre-activation residual block for deep generators
- GeneratorBlock  : upsampling or flat block used in Generator stacks
- DiscriminatorBlock : downsampling or flat block used in Discriminator stacks
- ConditionalBatchNorm : batch normalization conditioned on a class label
  (used in Conditional GANs)

Background
----------
A GAN consists of two networks trained in opposition:

  Generator G(z; θ_G)   : maps a latent noise vector z ~ p_z(z) to a
                           synthetic sample x̃ in the data space.

  Discriminator D(x; θ_D): maps a sample x (real or synthetic) to a
                            scalar in [0, 1] (probability of being real)
                            or a raw logit (WGAN / WGAN-GP variant).

All building blocks in this file are designed to be composed into the
Generator and Discriminator through the GANModel class in model.py.

References
----------
Goodfellow, I., et al. (2014). Generative Adversarial Nets. *NeurIPS 2014*.
He, K., et al. (2016). Identity Mappings in Deep Residual Networks. *ECCV*.
Miyato, T., et al. (2018). Spectral Normalization for GANs. *ICLR 2018*.
"""

import math
from typing import Optional

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
    """Leaky ReLU activation (default negative slope: 0.2, standard for GANs)."""
    def __init__(self, negative_slope: float = 0.2) -> None:
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
        name    : Activation name, e.g. ``'relu'``, ``'leaky_relu'``,
                  ``'tanh'``.
        **kwargs: Keyword arguments forwarded to the activation constructor
                  (e.g. ``negative_slope=0.1`` for LeakyReLU).

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
# Spectral Normalization wrapper
# =============================================================================

class SpectralLinear(nn.Module):
    """
    Fully-connected linear layer with optional spectral normalization.

    Spectral normalization (Miyato et al., 2018) constrains the Lipschitz
    constant of the Discriminator by dividing the weight matrix by its
    largest singular value at each forward pass.  This stabilizes training
    without requiring gradient penalty, and is the standard choice for the
    WGAN-SN variant.

    Args:
        in_features  : Number of input features.
        out_features : Number of output features.
        bias         : Whether to include a bias term (default: ``True``).
        spectral_norm: Whether to apply spectral normalization
                       (default: ``False``).
    """

    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        bias:          bool = True,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias=bias)
        if spectral_norm:
            self.layer = nn.utils.spectral_norm(linear)
        else:
            self.layer = linear
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# =============================================================================
# Conditional Batch Normalization
# =============================================================================

class ConditionalBatchNorm(nn.Module):
    """
    Batch normalization whose scale (γ) and shift (β) parameters are
    predicted from a conditioning label embedding rather than being fixed.

    Used in Conditional GANs (cGANs) to inject class information into
    the Generator without concatenating it at every layer.

    For a batch of feature vectors x ∈ R^(B × D) and class labels c ∈ {0,...,C-1}:

        x_norm = (x - μ_B) / √(σ²_B + ε)
        γ(c)   = W_γ · embed(c)   ∈ R^D
        β(c)   = W_β · embed(c)   ∈ R^D
        out    = γ(c) ⊙ x_norm + β(c)

    Args:
        num_features   : Number of feature dimensions D.
        num_classes    : Total number of conditioning classes C.
        embed_dim      : Dimensionality of the class embedding
                         (default: ``64``).
    """

    def __init__(
        self,
        num_features: int,
        num_classes:  int,
        embed_dim:    int = 64,
    ) -> None:
        super().__init__()
        self.bn      = nn.BatchNorm1d(num_features, affine=False)
        self.embed   = nn.Embedding(num_classes, embed_dim)
        self.gamma_fc = nn.Linear(embed_dim, num_features)
        self.beta_fc  = nn.Linear(embed_dim, num_features)

        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x      : Feature tensor of shape ``(batch, num_features)``.
            labels : Integer class labels of shape ``(batch,)``.

        Returns:
            Normalized and conditionally rescaled tensor, same shape as x.
        """
        e     = self.embed(labels)          # (B, embed_dim)
        gamma = self.gamma_fc(e)            # (B, num_features)
        beta  = self.beta_fc(e)             # (B, num_features)
        return gamma * self.bn(x) + beta


# =============================================================================
# Generator Block
# =============================================================================

class GeneratorBlock(nn.Module):
    """
    A single fully-connected block for the Generator.

    Each block applies:

        Linear(in → out) → [BatchNorm | ConditionalBatchNorm] → Activation

    Batch normalization is standard in the Generator (Radford et al., 2015).
    Conditional batch normalization is used when class labels are available.

    Args:
        in_features  : Number of input features.
        out_features : Number of output features.
        activation   : Activation name (default: ``'relu'``).
        use_bn       : Whether to apply batch normalization (default: ``True``).
        num_classes  : If > 0, use ConditionalBatchNorm conditioned on labels;
                       otherwise use standard BatchNorm1d (default: ``0``).
        embed_dim    : Embedding dimension for ConditionalBatchNorm
                       (default: ``64``).
        bias         : Whether to include bias in the linear layer.
                       Typically ``False`` when ``use_bn=True``.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   str = 'relu',
        use_bn:       bool = True,
        num_classes:  int  = 0,
        embed_dim:    int  = 64,
        bias:         bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()
        self.conditional = (num_classes > 0)

        # Bias is redundant when BN follows; disable to reduce parameter count
        use_bias = bias and (not use_bn)
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        if use_bn:
            if self.conditional:
                self.norm = ConditionalBatchNorm(out_features, num_classes, embed_dim)
            else:
                self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = None

        self.act = get_activation(self.activation_name)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.linear.weight, 0.0, 0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x:      torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x      : Input tensor of shape ``(batch, in_features)``.
            labels : Integer labels of shape ``(batch,)`` — required when
                     ``conditional=True``, ignored otherwise.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        out = self.linear(x)
        if self.norm is not None:
            if self.conditional and labels is not None:
                out = self.norm(out, labels)
            else:
                out = self.norm(out)
        return self.act(out)


# =============================================================================
# Discriminator Block
# =============================================================================

class DiscriminatorBlock(nn.Module):
    """
    A single fully-connected block for the Discriminator.

    Each block applies:

        Linear(in → out) → [LayerNorm (optional)] → Activation

    .. note::
        The Discriminator should **not** use BatchNorm on its input layer
        (Radford et al., 2015) and should use LayerNorm or no normalization
        when training with gradient penalty (WGAN-GP), because BN introduces
        correlations across the batch that invalidate the penalty.

    Args:
        in_features   : Number of input features.
        out_features  : Number of output features.
        activation    : Activation name (default: ``'leaky_relu'``).
        use_ln        : Whether to apply LayerNorm after the linear layer
                        (default: ``False``).
        spectral_norm : Whether to apply spectral normalization to the linear
                        layer (default: ``False``).
        dropout       : Dropout probability (0 = disabled, default: ``0.0``).
        bias          : Whether to include a bias term (default: ``True``).
    """

    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        activation:    str   = 'leaky_relu',
        use_ln:        bool  = False,
        spectral_norm: bool  = False,
        dropout:       float = 0.0,
        bias:          bool  = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()

        self.linear = SpectralLinear(in_features, out_features, bias=bias,
                                     spectral_norm=spectral_norm)
        self.norm    = nn.LayerNorm(out_features) if use_ln else None
        self.act     = get_activation(self.activation_name)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self._init_weights()

    def _init_weights(self) -> None:
        inner = self.linear.layer
        if hasattr(inner, 'weight_orig'):   # spectral norm wraps the linear
            nn.init.normal_(inner.weight_orig, 0.0, 0.02)
        elif hasattr(inner, 'weight'):
            nn.init.normal_(inner.weight, 0.0, 0.02)
        if hasattr(inner, 'bias') and inner.bias is not None:
            nn.init.zeros_(inner.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
