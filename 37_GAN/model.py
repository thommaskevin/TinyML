# model.py
"""
Generative Adversarial Network (GAN) models.

Contents
--------
Generator      : Fully-connected generator that maps latent vectors z to
                 synthetic samples in the data space.
Discriminator  : Fully-connected discriminator / critic that maps samples
                 to a real/fake score or logit.
ConditionalGAN : Container that holds a class-conditional Generator and
                 Discriminator (cGAN, Mirza & Osindero 2014).

Configuration format
--------------------
``generator_layers`` — list of dicts, one per Generator hidden block:

.. code-block:: python

    [
        {'out_features': 128, 'activation': 'relu',  'use_bn': True},
        {'out_features': 256, 'activation': 'relu',  'use_bn': True},
        {'out_features': 784, 'activation': 'tanh',  'use_bn': False},
    ]

Required keys per dict:

- ``out_features`` : Number of output neurons.

Optional keys per dict:

- ``activation``  : Activation name (default: ``'relu'``).
- ``use_bn``      : Whether to apply batch normalization (default: ``True``
                    for hidden layers, ``False`` for the output layer).
- ``num_classes`` : Number of conditioning classes for cGAN
                    (default: ``0`` = unconditional).
- ``embed_dim``   : Embedding dimension for ConditionalBatchNorm
                    (default: ``64``).

``discriminator_layers`` — list of dicts, one per Discriminator hidden block:

.. code-block:: python

    [
        {'out_features': 256, 'activation': 'leaky_relu', 'dropout': 0.3},
        {'out_features': 128, 'activation': 'leaky_relu', 'dropout': 0.3},
        {'out_features': 1,   'activation': 'linear'},
    ]

Required keys per dict:

- ``out_features`` : Number of output neurons.

Optional keys per dict:

- ``activation``    : Activation name (default: ``'leaky_relu'``).
- ``dropout``       : Dropout probability (default: ``0.0``).
- ``use_ln``        : Whether to apply LayerNorm (default: ``False``).
- ``spectral_norm`` : Whether to apply spectral normalization
                      (default: ``False``).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from layers import GeneratorBlock, DiscriminatorBlock, get_activation


# =============================================================================
# Generator
# =============================================================================

class Generator(nn.Module):
    """
    Fully-connected Generator network.

    Maps a latent noise vector z ∈ R^{latent_dim} through a stack of
    ``GeneratorBlock`` layers to produce a synthetic sample
    x̃ ∈ R^{output_dim}.

    For class-conditional generation (cGAN), set ``num_classes > 0``.
    The class label is embedded and concatenated to z before the first
    layer, allowing the Generator to produce class-specific samples.

    Args:
        latent_dim       : Dimensionality of the input noise vector z.
        generator_layers : List of dicts configuring each Generator block.
                           See module docstring for the key specification.
        num_classes      : Number of conditioning classes for cGAN.
                           Set to 0 for unconditional generation.
        embed_dim        : Label embedding dimension for cGAN
                           (default: ``64``).
    """

    def __init__(
        self,
        latent_dim:       int,
        generator_layers: List[dict],
        num_classes:      int = 0,
        embed_dim:        int = 64,
    ) -> None:
        super().__init__()

        if not generator_layers:
            raise ValueError("'generator_layers' must contain at least one entry.")

        self.latent_dim    = latent_dim
        self.num_classes   = num_classes
        self.embed_dim     = embed_dim
        self.conditional   = (num_classes > 0)
        self.layer_configs = generator_layers

        # For cGAN: embed the label and concatenate to z
        if self.conditional:
            self.label_embed = nn.Embedding(num_classes, embed_dim)
            in_dim = latent_dim + embed_dim
        else:
            self.label_embed = None
            in_dim = latent_dim

        # Build the block stack
        self.blocks = nn.ModuleList()
        prev = in_dim

        for cfg in generator_layers:
            out  = cfg['out_features']
            act  = cfg.get('activation', 'relu')
            use_bn = cfg.get('use_bn', True)
            n_cls  = cfg.get('num_classes', num_classes if self.conditional else 0)
            e_dim  = cfg.get('embed_dim', embed_dim)
            block  = GeneratorBlock(
                in_features=prev,
                out_features=out,
                activation=act,
                use_bn=use_bn,
                num_classes=n_cls,
                embed_dim=e_dim,
            )
            self.blocks.append(block)
            prev = out

        self.output_dim = prev

    # ------------------------------------------------------------------
    def forward(
        self,
        z:      torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate synthetic samples from a latent vector.

        Args:
            z      : Latent noise tensor of shape ``(batch, latent_dim)``.
            labels : Integer class labels of shape ``(batch,)`` — required
                     when ``conditional=True``, ignored otherwise.

        Returns:
            Synthetic sample tensor of shape ``(batch, output_dim)``.
        """
        if self.conditional:
            if labels is None:
                raise ValueError(
                    "Generator is conditional but no labels were provided."
                )
            emb = self.label_embed(labels)      # (batch, embed_dim)
            x   = torch.cat([z, emb], dim=1)   # (batch, latent_dim + embed_dim)
        else:
            x = z

        for block in self.blocks:
            x = block(x, labels if self.conditional else None)
        return x

    # ------------------------------------------------------------------
    def sample(
        self,
        n:      int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Draw n samples from the Generator.

        Args:
            n      : Number of samples to generate.
            device : Target device.
            labels : Integer class labels of shape ``(n,)`` for cGAN.
                     If ``None`` and ``conditional=True``, labels are drawn
                     uniformly at random.

        Returns:
            Synthetic samples of shape ``(n, output_dim)``.
        """
        z = torch.randn(n, self.latent_dim, device=device)
        if self.conditional and labels is None:
            labels = torch.randint(0, self.num_classes, (n,), device=device)
        return self.forward(z, labels)


# =============================================================================
# Discriminator
# =============================================================================

class Discriminator(nn.Module):
    """
    Fully-connected Discriminator / Critic network.

    Maps a data sample x ∈ R^{input_dim} through a stack of
    ``DiscriminatorBlock`` layers to produce a scalar score (logit).

    For class-conditional discrimination (cGAN), the label is embedded and
    concatenated to x before the first layer.

    Args:
        input_dim             : Dimensionality of the input sample x.
        discriminator_layers  : List of dicts configuring each Discriminator
                                block. See module docstring for the key spec.
        num_classes           : Number of conditioning classes for cGAN.
                                Set to 0 for unconditional discrimination.
        embed_dim             : Label embedding dimension for cGAN
                                (default: ``64``).
    """

    def __init__(
        self,
        input_dim:            int,
        discriminator_layers: List[dict],
        num_classes:          int = 0,
        embed_dim:            int = 64,
    ) -> None:
        super().__init__()

        if not discriminator_layers:
            raise ValueError("'discriminator_layers' must contain at least one entry.")

        self.input_dim     = input_dim
        self.num_classes   = num_classes
        self.embed_dim     = embed_dim
        self.conditional   = (num_classes > 0)
        self.layer_configs = discriminator_layers

        # For cGAN: embed the label and concatenate to x
        if self.conditional:
            self.label_embed = nn.Embedding(num_classes, embed_dim)
            in_dim = input_dim + embed_dim
        else:
            self.label_embed = None
            in_dim = input_dim

        # Build the block stack
        self.blocks = nn.ModuleList()
        prev = in_dim

        for cfg in discriminator_layers:
            out  = cfg['out_features']
            act  = cfg.get('activation', 'leaky_relu')
            use_ln = cfg.get('use_ln', False)
            sn     = cfg.get('spectral_norm', False)
            drop   = cfg.get('dropout', 0.0)
            block  = DiscriminatorBlock(
                in_features=prev,
                out_features=out,
                activation=act,
                use_ln=use_ln,
                spectral_norm=sn,
                dropout=drop,
            )
            self.blocks.append(block)
            prev = out

    # ------------------------------------------------------------------
    def forward(
        self,
        x:      torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score a batch of samples.

        Args:
            x      : Input tensor of shape ``(batch, input_dim)``.
            labels : Integer class labels of shape ``(batch,)`` — required
                     when ``conditional=True``, ignored otherwise.

        Returns:
            Score tensor of shape ``(batch, 1)`` or ``(batch,)`` depending
            on the last layer's ``out_features``.
        """
        if self.conditional:
            if labels is None:
                raise ValueError(
                    "Discriminator is conditional but no labels were provided."
                )
            emb = self.label_embed(labels)     # (batch, embed_dim)
            x   = torch.cat([x, emb], dim=1)  # (batch, input_dim + embed_dim)

        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# Conditional GAN container
# =============================================================================

class ConditionalGAN(nn.Module):
    """
    Convenience container that holds a Generator and a Discriminator
    configured for class-conditional generation (cGAN).

    The container does **not** implement a combined forward pass — the
    Generator and Discriminator are accessed as attributes and updated
    separately via ``GANTrainer``.

    Args:
        generator     : A ``Generator`` instance.
        discriminator : A ``Discriminator`` instance.
    """

    def __init__(
        self,
        generator:     Generator,
        discriminator: Discriminator,
    ) -> None:
        super().__init__()
        self.generator     = generator
        self.discriminator = discriminator

    def generate(
        self,
        n:      int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper around ``Generator.sample``."""
        return self.generator.sample(n, device, labels)
