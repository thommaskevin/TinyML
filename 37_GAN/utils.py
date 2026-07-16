# utils.py
"""
Utility functions for GAN workflows.

Contents
--------
1. ``export_to_json``            — Serialise a trained Generator to JSON.
2. ``GANTrainer``                — Full adversarial training loop with
                                   support for all loss variants.
3. ``plot_training_history``     — Generator and Discriminator loss curves.
4. ``plot_generated_samples``    — 2D scatter of real vs generated samples.
5. ``plot_latent_interpolation`` — Linear interpolation in latent space.
6. ``plot_loss_landscape``       — D and G loss as a function of training step.
7. ``evaluate_fid_proxy``        — Lightweight FID proxy using mean/cov distance.
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from losses import (
    compute_discriminator_loss,
    compute_generator_loss,
    wgan_clip_weights,
    LOSS_TYPES,
)


# =============================================================================
# 1.  Export Generator to JSON
# =============================================================================

def export_to_json(generator: nn.Module, filepath: str) -> None:
    """
    Serialise a trained ``Generator`` to a JSON file.

    The JSON contains three top-level keys:

    - ``architecture`` : latent_dim, output_dim, layer metadata.
    - ``layers``       : per-block architecture metadata.
    - ``parameters``   : all weight and bias arrays as nested Python lists.

    Args:
        generator : A trained ``Generator`` instance.
        filepath  : Destination file path (parent dirs created automatically).
    """
    arch   = []
    params = {}

    def _t(tensor): return tensor.detach().cpu().tolist()

    # Optional label embedding (cGAN)
    if generator.conditional and generator.label_embed is not None:
        params['label_embed_weight'] = _t(generator.label_embed.weight)

    # Generator blocks
    for i, (block, cfg) in enumerate(
        zip(generator.blocks, generator.layer_configs)
    ):
        info = {
            'index':        i,
            'in_features':  block.linear.in_features,
            'out_features': block.linear.out_features,
            'activation':   block.activation_name,
            'use_bn':       block.norm is not None,
            'conditional':  block.conditional,
        }
        params[f'gen_{i}_weight'] = _t(block.linear.weight)
        if block.linear.bias is not None:
            params[f'gen_{i}_bias'] = _t(block.linear.bias)
        else:
            params[f'gen_{i}_bias'] = None

        if block.norm is not None:
            if block.conditional:
                params[f'gen_{i}_bn_gamma_w'] = _t(block.norm.gamma_fc.weight)
                params[f'gen_{i}_bn_gamma_b'] = _t(block.norm.gamma_fc.bias)
                params[f'gen_{i}_bn_beta_w']  = _t(block.norm.beta_fc.weight)
                params[f'gen_{i}_bn_beta_b']  = _t(block.norm.beta_fc.bias)
            else:
                params[f'gen_{i}_bn_weight'] = _t(block.norm.weight)
                params[f'gen_{i}_bn_bias']   = _t(block.norm.bias)
        arch.append(info)

    data = {
        'architecture': {
            'latent_dim':   generator.latent_dim,
            'output_dim':   generator.output_dim,
            'num_classes':  generator.num_classes,
            'embed_dim':    generator.embed_dim,
            'conditional':  generator.conditional,
        },
        'layers':     arch,
        'parameters': params,
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generator exported → {filepath}")


# =============================================================================
# 2.  GAN Trainer
# =============================================================================

class GANTrainer:
    """
    Adversarial training loop supporting all loss variants.

    The trainer implements the standard GAN alternating update:

        For each training step:
          1. Sample real batch x ~ p_data.
          2. Sample latent vectors z ~ p_z.
          3. Generate fake batch x̃ = G(z).
          4. Update D (or Critic) for ``n_critic`` steps.
          5. Update G for 1 step.

    For WGAN, weight clipping is applied after each Critic step.
    For WGAN-GP, the gradient penalty is computed on interpolated samples.

    Args:
        generator      : ``Generator`` instance.
        discriminator  : ``Discriminator`` instance.
        g_optimizer    : Optimizer for the Generator.
        d_optimizer    : Optimizer for the Discriminator / Critic.
        loss_type      : One of ``'vanilla'``, ``'lsgan'``, ``'wgan'``,
                         ``'wgan_gp'``, ``'hinge'``.
        device         : Training device.
        n_critic       : Number of Discriminator updates per Generator update
                         (default: ``1``; set to ``5`` for WGAN variants).
        clip_value     : Weight clipping bound for WGAN (default: ``0.01``).
        lambda_gp      : Gradient penalty coefficient for WGAN-GP
                         (default: ``10.0``).
        label_smoothing: One-sided label smoothing for vanilla GAN
                         (default: ``0.0``).
    """

    def __init__(
        self,
        generator:       nn.Module,
        discriminator:   nn.Module,
        g_optimizer:     torch.optim.Optimizer,
        d_optimizer:     torch.optim.Optimizer,
        loss_type:       str   = 'vanilla',
        device:          torch.device = torch.device('cpu'),
        n_critic:        int   = 1,
        clip_value:      float = 0.01,
        lambda_gp:       float = 10.0,
        label_smoothing: float = 0.0,
    ) -> None:
        if loss_type.lower() not in LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. "
                f"Available: {LOSS_TYPES}"
            )
        self.G               = generator
        self.D               = discriminator
        self.g_opt           = g_optimizer
        self.d_opt           = d_optimizer
        self.loss_type       = loss_type.lower()
        self.device          = device
        self.n_critic        = n_critic
        self.clip_value      = clip_value
        self.lambda_gp       = lambda_gp
        self.label_smoothing = label_smoothing

        self.G.to(device)
        self.D.to(device)

        self.g_losses: List[float] = []
        self.d_losses: List[float] = []

    # ------------------------------------------------------------------
    def _sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.G.latent_dim, device=self.device)

    # ------------------------------------------------------------------
    def train_step(
        self,
        real_batch: torch.Tensor,
        labels:     Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Perform one full GAN training step on a single real batch.

        Args:
            real_batch : Real data tensor of shape ``(batch, features)``.
            labels     : Integer class labels for cGAN, shape ``(batch,)``.
                         Pass ``None`` for unconditional GANs.

        Returns:
            Tuple ``(d_loss, g_loss)`` as Python floats.
        """
        batch  = real_batch.size(0)
        real   = real_batch.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # ---- Discriminator / Critic updates ----
        d_loss_accum = 0.0
        for _ in range(self.n_critic):
            self.d_opt.zero_grad()

            z    = self._sample_latent(batch)
            fake = self.G(z, labels).detach()   # no G gradients needed here

            real_logits = self.D(real, labels).squeeze()
            fake_logits = self.D(fake, labels).squeeze()

            if self.loss_type == 'wgan_gp':
                d_loss = compute_discriminator_loss(
                    self.loss_type, real_logits, fake_logits,
                    discriminator=self.D,
                    real_samples=real,
                    fake_samples=fake,
                    lambda_gp=self.lambda_gp,
                )
            elif self.loss_type == 'vanilla':
                d_loss = compute_discriminator_loss(
                    self.loss_type, real_logits, fake_logits,
                    label_smoothing=self.label_smoothing,
                )
            else:
                d_loss = compute_discriminator_loss(
                    self.loss_type, real_logits, fake_logits
                )

            d_loss.backward()
            self.d_opt.step()

            # WGAN weight clipping
            if self.loss_type == 'wgan':
                wgan_clip_weights(self.D, self.clip_value)

            d_loss_accum += d_loss.item()

        d_loss_avg = d_loss_accum / self.n_critic

        # ---- Generator update ----
        self.g_opt.zero_grad()

        z         = self._sample_latent(batch)
        fake      = self.G(z, labels)
        fake_logits = self.D(fake, labels).squeeze()
        g_loss    = compute_generator_loss(self.loss_type, fake_logits)
        g_loss.backward()
        self.g_opt.step()

        self.d_losses.append(d_loss_avg)
        self.g_losses.append(g_loss.item())

        return d_loss_avg, g_loss.item()

    # ------------------------------------------------------------------
    def train(
        self,
        data_loader,
        epochs:      int = 100,
        print_every: int = 10,
        labels_fn=None,
    ) -> Tuple[List[float], List[float]]:
        """
        Train the GAN for a fixed number of epochs over a data loader.

        Args:
            data_loader : Iterable of batches.  Each batch is either a
                          ``torch.Tensor`` of shape ``(B, features)`` or a
                          tuple ``(tensor, labels)``.
            epochs      : Number of full passes over the data loader.
            print_every : Print average losses every N epochs.
            labels_fn   : Optional callable that extracts labels from a batch
                          tuple.  Default behaviour handles both tensor-only
                          and ``(tensor, labels)`` tuples automatically.

        Returns:
            Tuple ``(d_losses, g_losses)`` — per-step loss lists.
        """
        self.G.train()
        self.D.train()

        step = 0
        for epoch in range(1, epochs + 1):
            epoch_d, epoch_g = [], []

            for batch in data_loader:
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                if labels_fn is not None:
                    y = labels_fn(batch)

                d_l, g_l = self.train_step(x, y)
                epoch_d.append(d_l)
                epoch_g.append(g_l)
                step += 1

            if epoch % print_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4}/{epochs}"
                    f"  |  D loss = {np.mean(epoch_d):.5f}"
                    f"  |  G loss = {np.mean(epoch_g):.5f}"
                )

        return self.d_losses, self.g_losses


# =============================================================================
# 3.  Training history plot
# =============================================================================

def plot_training_history(
    d_losses:  List[float],
    g_losses:  List[float],
    loss_type: str = 'vanilla',
    smooth_k:  int = 20,
) -> None:
    """
    Plot Generator and Discriminator loss curves with optional smoothing.

    Args:
        d_losses  : Per-step Discriminator losses (from ``GANTrainer``).
        g_losses  : Per-step Generator losses.
        loss_type : GAN variant name (used for the title).
        smooth_k  : Moving-average window size for smoothing (default: 20).
    """
    def _smooth(arr, k):
        if k <= 1:
            return arr
        kernel = np.ones(k) / k
        return np.convolve(arr, kernel, mode='valid')

    steps   = np.arange(len(d_losses))
    d_arr   = np.array(d_losses)
    g_arr   = np.array(g_losses)
    d_sm    = _smooth(d_arr, smooth_k)
    g_sm    = _smooth(g_arr, smooth_k)
    steps_sm = np.arange(len(d_sm))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'GAN Training History — {loss_type.upper()}',
                 fontsize=14, fontweight='bold')

    ax1.plot(steps,    d_arr, alpha=0.2, color='steelblue', linewidth=0.8)
    ax1.plot(steps_sm, d_sm,  color='steelblue', linewidth=2,
             label=f'D loss (MA-{smooth_k})')
    ax1.set_title('Discriminator Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps,    g_arr, alpha=0.2, color='tomato', linewidth=0.8)
    ax2.plot(steps_sm, g_sm,  color='tomato', linewidth=2,
             label=f'G loss (MA-{smooth_k})')
    ax2.set_title('Generator Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  Real vs generated 2D scatter
# =============================================================================

def plot_generated_samples(
    generator:   nn.Module,
    real_data:   np.ndarray,
    n_samples:   int = 1000,
    device:      torch.device = torch.device('cpu'),
    title:       str = 'Real vs Generated Samples',
    labels:      Optional[torch.Tensor] = None,
    num_classes: int = 0,
) -> None:
    """
    Scatter plot of real versus generated samples (2-D feature space).

    For data with more than 2 features, only the first two dimensions are
    plotted.  For class-conditional GANs, generated samples are colored by
    class label.

    Args:
        generator   : Trained ``Generator`` instance.
        real_data   : Real data array of shape ``(N, features)``.
        n_samples   : Number of samples to generate (default: 1000).
        device      : Inference device.
        title       : Figure title.
        labels      : Integer labels for conditional generation, shape ``(n,)``.
        num_classes : Number of classes (for coloring, 0 = unconditional).
    """
    generator.eval()
    with torch.no_grad():
        fake = generator.sample(n_samples, device, labels).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.scatter(real_data[:, 0], real_data[:, 1],
                alpha=0.4, s=8, c='steelblue', label='Real')
    ax1.set_title('Real Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if num_classes > 1 and labels is not None:
        lbl_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        scatter = ax2.scatter(fake[:, 0], fake[:, 1],
                              alpha=0.4, s=8, c=lbl_np[:n_samples],
                              cmap='tab10', label='Generated')
        plt.colorbar(scatter, ax=ax2, label='Class')
    else:
        ax2.scatter(fake[:, 0], fake[:, 1],
                    alpha=0.4, s=8, c='tomato', label='Generated')
    ax2.set_title('Generated Data')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Latent space interpolation
# =============================================================================

def plot_latent_interpolation(
    generator:  nn.Module,
    n_steps:    int = 10,
    n_pairs:    int = 4,
    device:     torch.device = torch.device('cpu'),
    title:      str = 'Latent Space Interpolation',
    labels:     Optional[torch.Tensor] = None,
) -> None:
    """
    Visualize linear interpolation between pairs of latent vectors.

    For each pair (z_A, z_B), generates n_steps intermediate samples
    z_t = (1 - t) * z_A + t * z_B, t ∈ [0, 1], and plots their first
    two output dimensions on a line.

    Args:
        generator : Trained ``Generator`` instance.
        n_steps   : Number of interpolation steps (default: 10).
        n_pairs   : Number of random z pairs (default: 4).
        device    : Inference device.
        title     : Figure title.
        labels    : Fixed class labels for conditional generation.
    """
    generator.eval()
    latent_dim = generator.latent_dim
    cmap       = plt.cm.viridis(np.linspace(0, 1, n_steps))

    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if n_pairs == 1:
        axes = [axes]

    with torch.no_grad():
        for ax in axes:
            zA = torch.randn(1, latent_dim, device=device)
            zB = torch.randn(1, latent_dim, device=device)
            xs, ys = [], []
            for k, t in enumerate(np.linspace(0, 1, n_steps)):
                z_t  = (1 - t) * zA + t * zB
                lbl  = labels[:1] if labels is not None else None
                out  = generator(z_t, lbl).cpu().numpy().squeeze()
                xs.append(float(out[0]))
                ys.append(float(out[1]) if len(out) > 1 else 0.0)

            for k in range(n_steps - 1):
                ax.plot([xs[k], xs[k+1]], [ys[k], ys[k+1]],
                        '-o', color=cmap[k], markersize=5)
            ax.scatter([xs[0]], [ys[0]], c='green', s=60, zorder=5,
                       label='z_A')
            ax.scatter([xs[-1]], [ys[-1]], c='red', s=60, zorder=5,
                       label='z_B')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  Loss landscape over steps
# =============================================================================

def plot_loss_landscape(
    d_losses: List[float],
    g_losses: List[float],
    title:    str = 'GAN Loss Landscape',
) -> None:
    """
    Overlay D and G losses on the same axes to reveal training dynamics.

    Useful for diagnosing mode collapse (G loss collapses) or discriminator
    saturation (D loss → 0 early).

    Args:
        d_losses : Per-step Discriminator losses.
        g_losses : Per-step Generator losses.
        title    : Figure title.
    """
    steps = np.arange(len(d_losses))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, d_losses, color='steelblue', alpha=0.7,
             linewidth=0.9, label='D loss')
    plt.plot(steps, g_losses, color='tomato', alpha=0.7,
             linewidth=0.9, label='G loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7.  FID proxy (mean + covariance distance)
# =============================================================================

def evaluate_fid_proxy(
    generator:  nn.Module,
    real_data:  np.ndarray,
    n_samples:  int = 2000,
    device:     torch.device = torch.device('cpu'),
    labels:     Optional[torch.Tensor] = None,
) -> float:
    """
    Compute a lightweight proxy for the Fréchet Inception Distance (FID)
    directly in the data space (not feature space).

    The proxy is the Fréchet distance between the Gaussian approximations
    of the real and generated distributions:

        FID_proxy = ‖μ_r - μ_g‖² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})

    **Note:** This is NOT the standard FID (which uses Inception features)
    but is a useful sanity check for low-dimensional tabular data.

    Args:
        generator  : Trained ``Generator`` instance.
        real_data  : Real data array of shape ``(N, features)``.
        n_samples  : Number of generated samples (default: 2000).
        device     : Inference device.
        labels     : Labels for conditional generation.

    Returns:
        Scalar FID proxy value (lower is better).
    """
    generator.eval()
    with torch.no_grad():
        fake = generator.sample(n_samples, device, labels).cpu().numpy()

    # Truncate to match minimum count
    n = min(len(real_data), len(fake))
    r = real_data[:n].astype(np.float64)
    f = fake[:n].astype(np.float64)

    mu_r, mu_f = r.mean(axis=0), f.mean(axis=0)
    sig_r = np.cov(r, rowvar=False)
    sig_f = np.cov(f, rowvar=False)

    diff    = mu_r - mu_f
    mean_sq = float(diff @ diff)

    # Matrix square root via eigendecomposition
    vals, vecs = np.linalg.eigh(sig_r @ sig_f)
    vals       = np.maximum(vals, 0.0)   # clamp numerical negatives
    sqrt_prod  = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    fid = mean_sq + np.trace(sig_r + sig_f - 2.0 * sqrt_prod)
    return float(fid)
