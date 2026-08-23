# vi.py
"""
Variational and regularisation losses for Quantile Regression (QR) models.

Contents
--------
- ``quantile_elbo_loss`` : ELBO loss for Bayesian quantile regression networks
                           that expose a ``kl_loss()`` method.
- ``qr_reg_loss``        : Task loss + weight-decay regularisation on the
                           QuantileHead delta parameters, for standard
                           (non-Bayesian) QR networks.
- ``crossing_penalty``   : Auxiliary loss that explicitly penalises any
                           residual quantile crossing in the network output,
                           useful as an additional training signal when
                           the QuantileHead monotonicity reparameterisation
                           is disabled.

Background
----------
In Bayesian quantile regression the posterior predictive distribution over
the quantile function is approximated by a variational distribution q(θ).
Training maximises the Evidence Lower BOund (ELBO):

    ELBO = E_q[log p(y | x, θ)] − KL[q(θ) ‖ p(θ)]

where the first term is the expected negative pinball loss and the second
term is the KL divergence between the approximate posterior and the prior.

For standard (non-Bayesian) QR networks, a simpler regulariser penalises
the L2 norm of the QuantileHead parameters to prevent the spacing between
adjacent quantile outputs from collapsing to zero (degenerate crossing).

References
----------
Thompson, S., Koop, G., & Myles, J. (2010).
    Bayesian Quantile Regression.  Unpublished manuscript.
Yu, K., & Moyeed, R. A. (2001).
    Bayesian Quantile Regression.  *Statistics & Probability Letters*,
    54(4), 437–447.
"""

import torch
import torch.nn.functional as F

from losses import compute_loss


# ---------------------------------------------------------------------------
# ELBO loss (requires a Bayesian QR model exposing kl_loss())
# ---------------------------------------------------------------------------

def quantile_elbo_loss(
    output:     torch.Tensor,
    target:     torch.Tensor,
    model,
    quantiles:  list  = None,
    kl_weight:  float = 1.0,
    likelihood: str   = 'multi_pinball',
    **kwargs,
) -> torch.Tensor:
    """
    Compute the ELBO loss for a variational Quantile Regression network.

    The ELBO decomposes into two terms:

        ELBO = NLL(output, target) + kl_weight · KL / batch_size

    where NLL is the expected negative pinball log-likelihood (computed as
    the multi-pinball loss) and KL is the KL divergence returned by the
    model's ``kl_loss()`` method.

    Args:
        output     : Multi-quantile predictions of shape ``(batch, K)``.
        target     : Ground-truth outcomes of shape ``(batch,)``.
        model      : Model instance exposing a ``kl_loss() -> Tensor`` method.
        quantiles  : List of K quantile levels; default ``[0.1, 0.5, 0.9]``.
        kl_weight  : Scalar multiplier for the KL term (default: ``1.0``).
                     Common practice is to anneal this from 0 → 1 over the
                     first few training epochs (KL warm-up).
        likelihood : Name of the reconstruction loss.  Currently supports
                     ``'multi_pinball'`` and ``'combined'``.
                     Default: ``'multi_pinball'``.
        **kwargs   : Extra keyword arguments forwarded to the likelihood
                     function (e.g. ``lambda_is=0.1`` for ``'combined'``).

    Returns:
        Scalar ELBO loss tensor.

    Raises:
        ValueError: if *likelihood* is not a recognised option.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    key = likelihood.lower()
    if key not in ('multi_pinball', 'combined'):
        raise ValueError(
            f"Unsupported likelihood '{likelihood}' for ELBO. "
            "Use 'multi_pinball' or 'combined'."
        )

    nll = compute_loss(key, output, target, quantiles=quantiles, **kwargs)
    kl  = model.kl_loss() / output.size(0)
    return nll + kl_weight * kl


# ---------------------------------------------------------------------------
# QR-specific regulariser (no Bayesian model required)
# ---------------------------------------------------------------------------

def qr_reg_loss(
    output:     torch.Tensor,
    target:     torch.Tensor,
    model,
    quantiles:  list  = None,
    reg_weight: float = 1e-3,
    likelihood: str   = 'multi_pinball',
    **kwargs,
) -> torch.Tensor:
    """
    Combined task loss + QuantileHead weight regularisation.

    The regularisation term penalises the Frobenius norm of the
    ``QuantileHead`` linear layer weights, discouraging the inter-quantile
    spacing (encoded by the raw delta outputs) from collapsing to very small
    or very large values:

        L_reg = reg_weight · ‖W_head‖²_F / K

    This acts as a form of smoothness prior on the conditional quantile
    function, preventing degenerate solutions where adjacent quantile
    estimates are artificially close (near-crossing).

    Args:
        output     : Multi-quantile predictions of shape ``(batch, K)``.
        target     : Ground-truth outcomes of shape ``(batch,)``.
        model      : ``QRModel`` instance (must have a ``head`` attribute
                     of type ``QuantileHead``).
        quantiles  : List of K quantile levels; default ``[0.1, 0.5, 0.9]``.
        reg_weight : Weighting coefficient for the regularisation term
                     (default: ``1e-3``).
        likelihood : Name of the task loss (default: ``'multi_pinball'``).
        **kwargs   : Extra keyword arguments forwarded to the task loss.

    Returns:
        Scalar combined loss tensor.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    task_loss = compute_loss(
        likelihood, output, target, quantiles=quantiles, **kwargs
    )

    # Regularise the QuantileHead weight matrix
    reg = torch.tensor(0.0, device=output.device)
    if hasattr(model, 'head') and hasattr(model.head, 'linear'):
        W = model.head.linear.weight
        reg = (W ** 2).mean()

    return task_loss + reg_weight * reg


# ---------------------------------------------------------------------------
# Explicit crossing penalty (optional auxiliary loss)
# ---------------------------------------------------------------------------

def crossing_penalty(
    output: torch.Tensor,
) -> torch.Tensor:
    """
    Explicit penalty for quantile crossing in the network output.

    Even when the ``QuantileHead`` reparameterisation is active, this
    penalty can be used as an *additional* diagnostic signal during training.
    It computes the mean magnitude of any negative inter-quantile gaps:

        L_cross = mean(max(0,  ŷ_{k} − ŷ_{k+1}))   for k = 1, …, K−1

    When the output is fully monotone (no crossing), this term equals zero.

    Args:
        output : Multi-quantile predictions of shape ``(batch, K)``.

    Returns:
        Scalar non-negative crossing penalty.
    """
    if output.size(1) < 2:
        return torch.tensor(0.0, device=output.device)

    # gaps[k] = ŷ_{k+1} - ŷ_{k}  (should be ≥ 0 when properly ordered)
    gaps    = output[:, 1:] - output[:, :-1]   # (batch, K-1)
    penalty = F.relu(-gaps)                     # violation magnitude
    return penalty.mean()
