# vi.py
"""
Variational and regularization objectives for GAN training.

Contents
--------
``info_nce_loss``         — Mutual-information maximization (InfoGAN-style)
                            using a Noise Contrastive Estimation objective.
``feature_matching_loss`` — Generator regularizer that matches intermediate
                            Discriminator feature statistics between real and
                            fake batches (Salimans et al., 2016).
``r1_gradient_penalty``   — Real-data gradient penalty that regularizes the
                            Discriminator without interpolation
                            (Mescheder et al., 2018).
``mode_seeking_loss``     — Mode-seeking regularizer that encourages the
                            Generator to produce diverse outputs for different
                            latent inputs (Mao et al., 2019).
``elbo_gan_loss``         — VAE-GAN hybrid that combines a reconstruction
                            ELBO term with the GAN adversarial loss.

References
----------
Chen, X., et al. (2016). InfoGAN. NeurIPS.
Salimans, T., et al. (2016). Improved Techniques for Training GANs. NeurIPS.
Mescheder, L., et al. (2018). Which Training Methods for GANs Do Actually
    Converge? ICML.
Mao, Q., et al. (2019). Mode Seeking Generative Adversarial Networks
    for Diverse Image Synthesis. CVPR.
Larsen, A. B. L., et al. (2016). Autoencoding beyond pixels using a learned
    similarity metric (VAE-GAN). ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# =============================================================================
# InfoGAN mutual-information term (InfoNCE)
# =============================================================================

def info_nce_loss(
    predicted_codes: torch.Tensor,
    true_codes:      torch.Tensor,
    temperature:     float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE mutual-information lower bound used as the Generator auxiliary
    loss in InfoGAN.

    Encourages the Generator to preserve structured latent codes c in its
    output x̃ = G(z, c), so that a recognition network Q(c | x̃) can
    recover c from the generated sample.

    L_I = -E[ log Q(c | G(z,c)) ]

    This implementation uses a contrastive formulation where predicted_codes
    are the Q-network outputs and true_codes are the input latent codes.

    Args:
        predicted_codes : Estimated codes Q(c|G(z,c)), shape
                          ``(batch, code_dim)``.
        true_codes      : True latent codes c, shape ``(batch, code_dim)``.
        temperature     : Softmax temperature (default: ``0.07``).

    Returns:
        Scalar InfoNCE loss (lower = more mutual information preserved).
    """
    pred_norm = F.normalize(predicted_codes, dim=1)
    true_norm = F.normalize(true_codes,      dim=1)

    logits = (pred_norm @ true_norm.T) / temperature   # (batch, batch)
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels)


# =============================================================================
# Feature-matching loss (Salimans et al., 2016)
# =============================================================================

def feature_matching_loss(
    real_features: List[torch.Tensor],
    fake_features: List[torch.Tensor],
) -> torch.Tensor:
    """
    Generator regularizer that matches intermediate Discriminator activation
    statistics between real and generated batches.

    Instead of maximizing D(G(z)) directly, the Generator is trained to
    produce outputs whose Discriminator features have the same mean as those
    of real data:

        L_FM = Σ_l ‖E[f_l(x)] - E[f_l(G(z))]‖²

    where f_l(·) denotes the activation of the l-th Discriminator layer.

    This stabilizes training and reduces mode collapse, particularly in
    class-conditional settings.

    Args:
        real_features : List of real-batch activation tensors, one per
                        Discriminator layer, each of shape ``(batch, d)``.
        fake_features : List of generated-batch activation tensors, same
                        shapes as ``real_features``.

    Returns:
        Scalar feature-matching loss.
    """
    if len(real_features) != len(fake_features):
        raise ValueError(
            "real_features and fake_features must have the same number of layers."
        )

    loss = torch.tensor(0.0, device=real_features[0].device)
    for rf, ff in zip(real_features, fake_features):
        loss = loss + F.mse_loss(ff.mean(dim=0), rf.mean(dim=0).detach())
    return loss


# =============================================================================
# R1 gradient penalty (Mescheder et al., 2018)
# =============================================================================

def r1_gradient_penalty(
    discriminator: nn.Module,
    real_samples:  torch.Tensor,
    gamma:         float = 10.0,
    labels:        Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    R1 gradient penalty: penalizes the gradient of the Discriminator with
    respect to **real** data only.

        L_R1 = (γ/2) · E[‖∇_x D(x)‖²]

    Unlike WGAN-GP, R1 does not require interpolated samples and converges
    for a broader class of GAN objectives.

    Args:
        discriminator : Discriminator ``nn.Module``.
        real_samples  : Real data batch of shape ``(batch, features)``.
        gamma         : Regularization coefficient (default: ``10.0``).
        labels        : Integer labels for conditional Discriminators.

    Returns:
        Scalar R1 penalty term.
    """
    real = real_samples.requires_grad_(True)
    d_real = discriminator(real, labels).squeeze()

    gradients = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]   # (batch, features)

    penalty = (gamma / 2.0) * (gradients.view(real.size(0), -1).norm(2, dim=1) ** 2).mean()
    return penalty


# =============================================================================
# Mode-seeking regularizer (Mao et al., 2019)
# =============================================================================

def mode_seeking_loss(
    fake_1: torch.Tensor,
    fake_2: torch.Tensor,
    z_1:    torch.Tensor,
    z_2:    torch.Tensor,
    eps:    float = 1e-8,
) -> torch.Tensor:
    """
    Mode-seeking regularizer (MS-GAN) that encourages the Generator to
    produce **diverse** outputs for different latent inputs.

    Maximizes the ratio of output distance to latent distance:

        L_MS = -E[ ‖G(z₁) - G(z₂)‖ / (‖z₁ - z₂‖ + ε) ]

    The Generator is trained to minimize this (negative of the ratio),
    which discourages it from mapping different z values to the same output
    (mode collapse).

    Args:
        fake_1 : Generated samples G(z₁), shape ``(batch, features)``.
        fake_2 : Generated samples G(z₂), shape ``(batch, features)``.
        z_1    : Latent vectors z₁, shape ``(batch, latent_dim)``.
        z_2    : Latent vectors z₂, shape ``(batch, latent_dim)``.
        eps    : Small constant for numerical stability (default: ``1e-8``).

    Returns:
        Scalar mode-seeking loss (add to Generator loss with a positive
        weight to reduce mode collapse).
    """
    output_dist = (fake_1 - fake_2).abs().mean(dim=1)     # (batch,)
    latent_dist = (z_1    - z_2).abs().mean(dim=1) + eps  # (batch,)
    return -(output_dist / latent_dist).mean()


# =============================================================================
# VAE-GAN hybrid ELBO term (Larsen et al., 2016)
# =============================================================================

def elbo_gan_loss(
    x_real:       torch.Tensor,
    x_recon:      torch.Tensor,
    mu:           torch.Tensor,
    log_var:      torch.Tensor,
    real_features: List[torch.Tensor],
    recon_features: List[torch.Tensor],
    beta:          float = 1.0,
    fm_weight:     float = 1.0,
) -> torch.Tensor:
    """
    VAE-GAN hybrid loss: combines the VAE ELBO (reconstruction + KL)
    with a feature-matching loss using Discriminator features as a
    perceptual similarity measure.

    L_VAE-GAN = Recon + β · KL + λ_fm · L_FM

    Recon = ‖x - x̂‖²   (pixel / feature MSE)
    KL    = -½ Σ(1 + log σ² - μ² - σ²)
    L_FM  = feature-matching loss between D(x) and D(x̂)

    Args:
        x_real          : Real data, shape ``(batch, features)``.
        x_recon         : Reconstructed data, shape ``(batch, features)``.
        mu              : VAE encoder mean, shape ``(batch, latent_dim)``.
        log_var         : VAE encoder log-variance, same shape as ``mu``.
        real_features   : List of D activation tensors for x_real.
        recon_features  : List of D activation tensors for x_recon.
        beta            : KL weight (default: ``1.0``).
        fm_weight       : Feature-matching weight (default: ``1.0``).

    Returns:
        Scalar VAE-GAN loss tensor.
    """
    recon_loss = F.mse_loss(x_recon, x_real, reduction='mean')
    kl_loss    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
    fm_loss    = feature_matching_loss(real_features, recon_features)

    return recon_loss + beta * kl_loss + fm_weight * fm_loss
