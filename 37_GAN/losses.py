# losses.py
"""
Loss functions for Generative Adversarial Network (GAN) training.

Supported GAN objectives
------------------------
vanilla   : Original minimax loss (Goodfellow et al., 2014)
            Uses binary cross-entropy on real/fake labels.

lsgan     : Least-Squares GAN (Mao et al., 2017)
            Replaces BCE with MSE; stabilizes training and reduces
            mode collapse by penalizing samples far from the decision
            boundary.

wgan      : Wasserstein GAN (Arjovsky et al., 2017)
            Maximizes the Earth Mover's distance between real and fake
            distributions.  Requires weight clipping on the Critic.

wgan_gp   : Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017)
            Enforces the 1-Lipschitz constraint via a gradient penalty on
            interpolated samples instead of weight clipping.  The most
            stable variant in practice.

hinge     : Hinge loss GAN (Lim & Ye, 2017; Miyato et al., 2018)
            Used in combination with spectral normalization.

Notes
-----
All functions follow the same calling convention:

    d_loss = <variant>_discriminator_loss(real_logits, fake_logits, ...)
    g_loss = <variant>_generator_loss(fake_logits)

so they can be swapped by changing the ``loss_type`` string in
``GANTrainer``.

References
----------
Goodfellow et al. (2014). Generative Adversarial Nets. NeurIPS.
Mao et al. (2017). Least Squares Generative Adversarial Networks. ICCV.
Arjovsky et al. (2017). Wasserstein GAN. ICML.
Gulrajani et al. (2017). Improved Training of Wasserstein GANs. NeurIPS.
Lim & Ye (2017). Geometric GAN. arXiv:1705.02894.
Miyato et al. (2018). Spectral Normalization for GANs. ICLR.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Vanilla GAN (Goodfellow et al., 2014)
# =============================================================================

def vanilla_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Discriminator loss for the vanilla GAN (non-saturating formulation).

    L_D = -E[log D(x)] - E[log(1 - D(G(z)))]

    Args:
        real_logits     : Raw logits D(x_real), shape ``(batch,)``.
        fake_logits     : Raw logits D(G(z)),   shape ``(batch,)``.
        label_smoothing : One-sided label smoothing for real targets.
                          Real label = 1 − label_smoothing (default: 0.0).

    Returns:
        Scalar Discriminator loss tensor.
    """
    real_labels = torch.ones_like(real_logits) * (1.0 - label_smoothing)
    fake_labels = torch.zeros_like(fake_logits)
    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    return loss_real + loss_fake


def vanilla_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Generator loss for the vanilla GAN (non-saturating formulation).

    Instead of minimizing E[log(1 - D(G(z)))], which saturates early in
    training, the Generator maximizes E[log D(G(z))]:

        L_G = -E[log D(G(z))]  ≡  BCE(D(G(z)), 1)

    Args:
        fake_logits : Raw logits D(G(z)), shape ``(batch,)``.

    Returns:
        Scalar Generator loss tensor.
    """
    return F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )


# =============================================================================
# Least-Squares GAN (Mao et al., 2017)
# =============================================================================

def lsgan_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    a: float = 0.0,
    b: float = 1.0,
) -> torch.Tensor:
    """
    Discriminator loss for LSGAN.

    L_D = 0.5 * E[(D(x) - b)²] + 0.5 * E[(D(G(z)) - a)²]

    Args:
        real_logits : Raw scores D(x_real), shape ``(batch,)``.
        fake_logits : Raw scores D(G(z)),   shape ``(batch,)``.
        a           : Target label for fake samples (default: 0.0).
        b           : Target label for real samples (default: 1.0).

    Returns:
        Scalar Discriminator loss tensor.
    """
    loss_real = 0.5 * F.mse_loss(real_logits, torch.full_like(real_logits, b))
    loss_fake = 0.5 * F.mse_loss(fake_logits, torch.full_like(fake_logits, a))
    return loss_real + loss_fake


def lsgan_generator_loss(
    fake_logits: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Generator loss for LSGAN.

    L_G = 0.5 * E[(D(G(z)) - c)²]

    Args:
        fake_logits : Raw scores D(G(z)), shape ``(batch,)``.
        c           : Target label for generated samples (default: 1.0).

    Returns:
        Scalar Generator loss tensor.
    """
    return 0.5 * F.mse_loss(fake_logits, torch.full_like(fake_logits, c))


# =============================================================================
# Wasserstein GAN (Arjovsky et al., 2017)
# =============================================================================

def wgan_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Critic loss for WGAN (Wasserstein distance estimate).

    L_C = E[D(G(z))] - E[D(x)]

    Minimizing L_C maximizes the estimated Wasserstein distance between
    the real and generated distributions.  Requires weight clipping on the
    Critic weights after each update (use ``wgan_clip_weights``).

    Args:
        real_logits : Critic scores D(x_real), shape ``(batch,)``.
        fake_logits : Critic scores D(G(z)),   shape ``(batch,)``.

    Returns:
        Scalar Critic loss tensor.
    """
    return fake_logits.mean() - real_logits.mean()


def wgan_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Generator loss for WGAN.

    L_G = -E[D(G(z))]

    Args:
        fake_logits : Critic scores D(G(z)), shape ``(batch,)``.

    Returns:
        Scalar Generator loss tensor.
    """
    return -fake_logits.mean()


def wgan_clip_weights(discriminator: torch.nn.Module, clip_value: float = 0.01) -> None:
    """
    Enforce the Lipschitz constraint for WGAN by clipping all Critic
    parameters to [-clip_value, clip_value] after each Critic update.

    Args:
        discriminator : The Critic / Discriminator ``nn.Module``.
        clip_value    : Absolute clipping bound (default: ``0.01``).
    """
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value, clip_value)


# =============================================================================
# Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017)
# =============================================================================

def wgan_gp_discriminator_loss(
    real_logits:   torch.Tensor,
    fake_logits:   torch.Tensor,
    discriminator: torch.nn.Module,
    real_samples:  torch.Tensor,
    fake_samples:  torch.Tensor,
    lambda_gp:     float = 10.0,
) -> torch.Tensor:
    """
    Critic loss for WGAN-GP, including the gradient penalty term.

    L_C = E[D(G(z))] - E[D(x)] + λ · E[(‖∇_x̂ D(x̂)‖₂ - 1)²]

    where x̂ = ε·x + (1-ε)·G(z),  ε ~ Uniform(0, 1).

    Args:
        real_logits   : Critic scores D(x_real), shape ``(batch,)``.
        fake_logits   : Critic scores D(G(z)),   shape ``(batch,)``.
        discriminator : The Critic ``nn.Module``, called on interpolated
                        samples to compute the gradient penalty.
        real_samples  : Real data batch, shape ``(batch, features)``.
        fake_samples  : Generated batch, shape ``(batch, features)``.
        lambda_gp     : Gradient penalty coefficient (default: ``10.0``).

    Returns:
        Scalar Critic loss tensor (Wasserstein distance + penalty).
    """
    wasserstein = fake_logits.mean() - real_logits.mean()
    gp          = _gradient_penalty(discriminator, real_samples, fake_samples,
                                    lambda_gp)
    return wasserstein + gp


def _gradient_penalty(
    discriminator: torch.nn.Module,
    real_samples:  torch.Tensor,
    fake_samples:  torch.Tensor,
    lambda_gp:     float,
) -> torch.Tensor:
    """Compute the WGAN-GP gradient penalty on interpolated samples."""
    batch = real_samples.size(0)
    device = real_samples.device

    # Random interpolation coefficient ε ~ U(0,1) per sample
    eps = torch.rand(batch, 1, device=device).expand_as(real_samples)
    interpolates = (eps * real_samples + (1 - eps) * fake_samples).requires_grad_(True)

    d_interp = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (batch, features)

    grad_norm = gradients.view(batch, -1).norm(2, dim=1)   # (batch,)
    penalty   = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return penalty


def wgan_gp_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Generator loss for WGAN-GP.

    L_G = -E[D(G(z))]

    Args:
        fake_logits : Critic scores D(G(z)), shape ``(batch,)``.

    Returns:
        Scalar Generator loss tensor.
    """
    return -fake_logits.mean()


# =============================================================================
# Hinge Loss GAN (Lim & Ye, 2017; Miyato et al., 2018)
# =============================================================================

def hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator loss for the Hinge GAN.

    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]

    Args:
        real_logits : Raw scores D(x_real), shape ``(batch,)``.
        fake_logits : Raw scores D(G(z)),   shape ``(batch,)``.

    Returns:
        Scalar Discriminator loss tensor.
    """
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Generator loss for the Hinge GAN.

    L_G = -E[D(G(z))]

    Args:
        fake_logits : Raw scores D(G(z)), shape ``(batch,)``.

    Returns:
        Scalar Generator loss tensor.
    """
    return -fake_logits.mean()


# =============================================================================
# Unified dispatch
# =============================================================================

LOSS_TYPES = ['vanilla', 'lsgan', 'wgan', 'wgan_gp', 'hinge']


def compute_discriminator_loss(
    loss_type:     str,
    real_logits:   torch.Tensor,
    fake_logits:   torch.Tensor,
    discriminator: torch.nn.Module = None,
    real_samples:  torch.Tensor    = None,
    fake_samples:  torch.Tensor    = None,
    **kwargs,
) -> torch.Tensor:
    """
    Dispatch to the appropriate Discriminator / Critic loss.

    Args:
        loss_type     : One of ``'vanilla'``, ``'lsgan'``, ``'wgan'``,
                        ``'wgan_gp'``, or ``'hinge'``.
        real_logits   : Discriminator output on real samples.
        fake_logits   : Discriminator output on fake samples.
        discriminator : Required for ``'wgan_gp'`` to compute gradient penalty.
        real_samples  : Required for ``'wgan_gp'``.
        fake_samples  : Required for ``'wgan_gp'``.
        **kwargs      : Extra arguments forwarded to the loss function
                        (e.g. ``lambda_gp=10.0``, ``label_smoothing=0.1``).

    Returns:
        Scalar Discriminator loss tensor.

    Raises:
        ValueError: if *loss_type* is not registered.
    """
    key = loss_type.lower()
    if key == 'vanilla':
        return vanilla_discriminator_loss(real_logits, fake_logits, **kwargs)
    elif key == 'lsgan':
        return lsgan_discriminator_loss(real_logits, fake_logits, **kwargs)
    elif key == 'wgan':
        return wgan_discriminator_loss(real_logits, fake_logits)
    elif key == 'wgan_gp':
        return wgan_gp_discriminator_loss(
            real_logits, fake_logits,
            discriminator, real_samples, fake_samples, **kwargs
        )
    elif key == 'hinge':
        return hinge_discriminator_loss(real_logits, fake_logits)
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Available types: {LOSS_TYPES}"
        )


def compute_generator_loss(
    loss_type:   str,
    fake_logits: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Dispatch to the appropriate Generator loss.

    Args:
        loss_type   : One of ``'vanilla'``, ``'lsgan'``, ``'wgan'``,
                      ``'wgan_gp'``, or ``'hinge'``.
        fake_logits : Discriminator output on generated samples.
        **kwargs    : Extra arguments (e.g. ``c=1.0`` for LSGAN).

    Returns:
        Scalar Generator loss tensor.

    Raises:
        ValueError: if *loss_type* is not registered.
    """
    key = loss_type.lower()
    if key == 'vanilla':
        return vanilla_generator_loss(fake_logits)
    elif key == 'lsgan':
        return lsgan_generator_loss(fake_logits, **kwargs)
    elif key in ('wgan', 'wgan_gp'):
        return wgan_gp_generator_loss(fake_logits)
    elif key == 'hinge':
        return hinge_generator_loss(fake_logits)
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Available types: {LOSS_TYPES}"
        )
