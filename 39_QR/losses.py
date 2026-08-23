# losses.py
"""
Loss functions for Quantile Regression (QR) neural network training.

Supported losses
----------------
pinball        : Single-quantile pinball (check) loss
multi_pinball  : Sum of pinball losses across K quantiles simultaneously
huber_pinball  : Huber-smoothed pinball loss (reduced gradient noise near zero)
interval_score : Winkler interval score for a (lower, upper) prediction band
coverage_loss  : Differentiable penalty for empirical coverage below nominal
combined       : Multi-pinball + interval_score + coverage_loss (composite)

Notes
-----
All loss functions accept raw tensor predictions and return a scalar.
The ``compute_loss`` dispatcher normalises the API across all variants so that
the training loop only needs to call ``compute_loss(name, output, target, ...)``.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def pinball_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    quantile:  float = 0.5,
    quantiles: list  = None,
) -> torch.Tensor:
    """
    Pinball (check) loss for a single quantile level τ.

    The loss asymmetrically penalises positive and negative residuals:

        L_τ(y, ŷ) = max(τ·(y − ŷ),  (τ−1)·(y − ŷ))

    Args:
        output    : Predicted quantile of shape ``(batch,)`` or ``(batch, 1)``.
                    When the model outputs ``(batch, K)`` (multi-quantile head),
                    pass *quantiles* so the correct column is selected.
        target    : Ground-truth outcomes, same shape as *output* after
                    column selection.
        quantile  : Target quantile τ ∈ (0, 1).  Default is 0.5 (median).
        quantiles : Optional list of K quantile levels the model was trained
                    on (e.g. ``[0.1, 0.5, 0.9]``).  When supplied and
                    *output* has K > 1 columns, the column matching
                    *quantile* is extracted before computing the loss.

    Returns:
        Scalar pinball loss.

    Raises:
        ValueError: if *quantiles* is given, *quantile* is not in the list,
                    and *output* has more than one column.
    """
    # --- Column extraction for multi-quantile output ---
    if output.dim() == 2 and output.shape[1] > 1:
        if quantiles is None:
            raise ValueError(
                f"output has shape {tuple(output.shape)} (multi-quantile). "
                "Pass quantiles=[...] so pinball_loss can select the right column, "
                "or use multi_pinball_loss for all quantiles simultaneously."
            )
        if quantile not in quantiles:
            raise ValueError(
                f"quantile={quantile} not found in quantiles={quantiles}."
            )
        output = output[:, quantiles.index(quantile)]   # (batch,)

    output   = output.squeeze()
    target   = target.squeeze().float()
    residual = target - output
    return torch.where(
        residual >= 0,
        quantile       * residual,
        (quantile - 1) * residual,
    ).mean()


def multi_pinball_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    quantiles: list = None,
) -> torch.Tensor:
    """
    Sum of pinball losses across K quantile levels simultaneously.

    This is the standard training objective for a simultaneous multi-quantile
    network.  Given predictions of shape ``(batch, K)`` and a corresponding
    list of K quantile levels, each column is penalised by its own pinball
    loss and the losses are averaged.

    Args:
        output    : Predicted quantiles of shape ``(batch, K)`` — column *k*
                    corresponds to ``quantiles[k]``.
        target    : Ground-truth outcomes of shape ``(batch,)`` or
                    ``(batch, 1)``.
        quantiles : List of K quantile levels in (0, 1), e.g.
                    ``[0.1, 0.5, 0.9]``.  Defaults to ``[0.1, 0.5, 0.9]``
                    when ``None``.

    Returns:
        Scalar combined pinball loss averaged over all quantile levels.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    target = target.squeeze().float()                # (batch,)
    total  = torch.tensor(0.0, device=output.device)

    for k, q in enumerate(quantiles):
        col      = output[:, k]                      # (batch,)
        residual = target - col
        loss_k   = torch.where(
            residual >= 0,
            q       * residual,
            (q - 1) * residual,
        ).mean()
        total = total + loss_k

    return total / len(quantiles)


def huber_pinball_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    quantile:  float = 0.5,
    delta:     float = 1.0,
    quantiles: list  = None,
) -> torch.Tensor:
    """
    Huber-smoothed pinball loss (also called the *expectile Huber* loss).

    Replaces the kink at zero in the standard pinball loss with a smooth
    quadratic region of half-width *delta*, reducing gradient noise for
    small residuals while preserving asymmetric penalisation in the tails.

    The loss function is:

        L_τ^δ(r) = τ · H_δ(r)          if r ≥ 0
                   (1−τ) · H_δ(−r)      if r < 0

    where H_δ(r) = r² / (2δ) if |r| ≤ δ, else |r| − δ/2.

    Args:
        output    : Predicted quantile of shape ``(batch,)`` or ``(batch, 1)``.
                    When the model outputs ``(batch, K)`` (multi-quantile head),
                    pass *quantiles* so the correct column is selected.
        target    : Ground-truth outcomes, same shape as *output* after
                    column selection.
        quantile  : Target quantile τ ∈ (0, 1).  Default is 0.5.
        delta     : Huber smoothing threshold.  Default is 1.0.
        quantiles : Optional list of K quantile levels (e.g. ``[0.1, 0.5, 0.9]``).
                    Required when *output* has K > 1 columns so the column
                    matching *quantile* can be extracted before the loss
                    computation.  Use ``multi_huber_pinball_loss`` (below)
                    to penalise all K quantiles simultaneously.

    Returns:
        Scalar Huber-pinball loss.

    Raises:
        ValueError: if *output* has K > 1 columns and *quantiles* is not given.
    """
    # --- Column extraction for multi-quantile output ---
    if output.dim() == 2 and output.shape[1] > 1:
        if quantiles is None:
            raise ValueError(
                f"output has shape {tuple(output.shape)} (multi-quantile). "
                "Pass quantiles=[...] so huber_pinball_loss can select the right "
                "column, or use multi_pinball_loss / combined for all quantiles."
            )
        if quantile not in quantiles:
            raise ValueError(
                f"quantile={quantile} not found in quantiles={quantiles}."
            )
        output = output[:, quantiles.index(quantile)]   # (batch,)

    output   = output.squeeze()
    target   = target.squeeze().float()
    residual = target - output

    abs_r = residual.abs()
    huber = torch.where(abs_r <= delta,
                        0.5 * residual ** 2 / delta,
                        abs_r - 0.5 * delta)

    loss = torch.where(residual >= 0,
                       quantile       * huber,
                       (quantile - 1) * huber)
    return loss.mean()


def multi_huber_pinball_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    quantiles: list  = None,
    delta:     float = 1.0,
) -> torch.Tensor:
    """
    Huber-smoothed pinball loss summed across K quantile levels simultaneously.

    This is the Huber-pinball analogue of ``multi_pinball_loss``: it applies
    the Huber-smoothed check function to each of the K output columns and
    averages the results.  Use this when training a multi-quantile model with
    the smoother gradient properties of the Huber-pinball.

    Args:
        output    : Multi-quantile predictions of shape ``(batch, K)``.
        target    : Ground-truth outcomes of shape ``(batch,)``.
        quantiles : List of K quantile levels.  Defaults to ``[0.1, 0.5, 0.9]``.
        delta     : Huber smoothing threshold applied to every quantile.
                    Default is 1.0.

    Returns:
        Scalar averaged Huber-pinball loss.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    target = target.squeeze().float()
    total  = torch.tensor(0.0, device=output.device)

    for k, q in enumerate(quantiles):
        col      = output[:, k]
        residual = target - col
        abs_r    = residual.abs()
        huber    = torch.where(abs_r <= delta,
                               0.5 * residual ** 2 / delta,
                               abs_r - 0.5 * delta)
        loss_k   = torch.where(residual >= 0,
                               q       * huber,
                               (q - 1) * huber).mean()
        total = total + loss_k

    return total / len(quantiles)


def interval_score_loss(
    lower:  torch.Tensor,
    upper:  torch.Tensor,
    target: torch.Tensor,
    alpha:  float = 0.1,
) -> torch.Tensor:
    """
    Winkler interval score for a prediction interval at level (1 − α).

    Proposed by Winkler (1972) and widely used for evaluating probabilistic
    forecasts, this score penalises both wide intervals and coverage failures:

        IS_α(l, u, y) = (u − l)
                        + (2/α) · (l − y)   if y < l
                        + (2/α) · (y − u)   if y > u

    A lower score is better: it rewards narrow intervals that still contain
    the true outcome.

    Args:
        lower  : Lower quantile predictions of shape ``(batch,)``,
                 corresponding to quantile level α/2.
        upper  : Upper quantile predictions of shape ``(batch,)``,
                 corresponding to quantile level 1 − α/2.
        target : Ground-truth outcomes of shape ``(batch,)``.
        alpha  : Nominal miscoverage level; default 0.10 for a 90% PI.

    Returns:
        Scalar interval score (lower is better).
    """
    lower  = lower.squeeze()
    upper  = upper.squeeze()
    target = target.squeeze().float()

    width          = upper - lower
    below_penalty  = F.relu(lower - target)
    above_penalty  = F.relu(target - upper)
    return (width + (2.0 / alpha) * (below_penalty + above_penalty)).mean()


def coverage_loss(
    lower:    torch.Tensor,
    upper:    torch.Tensor,
    target:   torch.Tensor,
    nominal:  float = 0.9,
) -> torch.Tensor:
    """
    Differentiable soft penalty for empirical coverage below the nominal level.

    The standard coverage metric (proportion of targets within the interval)
    is not differentiable.  This loss approximates it with a smooth sigmoid
    indicator and penalises only under-coverage:

        coverage = mean(σ((u − y) · S) · σ((y − l) · S))   [S = 20]
        loss     = max(nominal − coverage, 0)²

    Args:
        lower   : Lower bound predictions of shape ``(batch,)``.
        upper   : Upper bound predictions of shape ``(batch,)``.
        target  : Ground-truth outcomes of shape ``(batch,)``.
        nominal : Desired coverage probability.  Default is 0.90 (90% PI).

    Returns:
        Scalar non-negative penalty.
    """
    lower  = lower.squeeze()
    upper  = upper.squeeze()
    target = target.squeeze().float()

    S = 20.0   # sigmoid sharpness
    in_interval = (
        torch.sigmoid(S * (upper - target)) *
        torch.sigmoid(S * (target - lower))
    )
    empirical_coverage = in_interval.mean()
    under = F.relu(nominal - empirical_coverage)
    return under ** 2


def combined_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    quantiles: list  = None,
    alpha:     float = 0.1,
    lambda_is: float = 0.1,
    lambda_cv: float = 0.1,
) -> torch.Tensor:
    """
    Composite training loss combining multi-pinball, interval score, and
    coverage penalty.

    Total loss = multi_pinball
                 + λ_is · interval_score(lower, upper)
                 + λ_cv · coverage_loss(lower, upper)

    The lower and upper bound columns are extracted automatically from
    *output* based on the first and last quantile levels in *quantiles*.

    Args:
        output    : Multi-quantile predictions of shape ``(batch, K)``.
        target    : Ground-truth outcomes of shape ``(batch,)``.
        quantiles : K quantile levels; default ``[0.1, 0.5, 0.9]``.
        alpha     : Nominal miscoverage for the interval score.
                    Should equal ``1 − (quantiles[-1] − quantiles[0])``.
                    Default 0.1 (for a 90% PI).
        lambda_is : Weight for the interval score term.  Default 0.1.
        lambda_cv : Weight for the coverage penalty term.  Default 0.1.

    Returns:
        Scalar combined loss.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    pb    = multi_pinball_loss(output, target, quantiles)
    lower = output[:, 0]
    upper = output[:, -1]
    is_   = interval_score_loss(lower, upper, target, alpha=alpha)
    cv    = coverage_loss(lower, upper, target, nominal=1.0 - alpha)
    return pb + lambda_is * is_ + lambda_cv * cv


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOSS_FN: dict = {
    'pinball':            pinball_loss,
    'multi_pinball':      multi_pinball_loss,
    'huber_pinball':      huber_pinball_loss,
    'multi_huber_pinball':multi_huber_pinball_loss,
    'interval_score':     interval_score_loss,
    'coverage':           coverage_loss,
    'combined':      combined_loss,
}

LOSS_NAMES: list = list(_LOSS_FN.keys())


def compute_loss(
    loss_name: str,
    output:    torch.Tensor,
    target:    torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Compute a named loss between *output* and *target*.

    Args:
        loss_name : One of ``'pinball'``, ``'multi_pinball'``,
                    ``'huber_pinball'``, ``'interval_score'``,
                    ``'coverage'``, or ``'combined'``.
        output    : Model predictions.
        target    : Ground-truth labels.
        **kwargs  : Extra keyword arguments forwarded to the loss function
                    (e.g. ``quantile=0.9``, ``quantiles=[0.1,0.5,0.9]``).

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: if *loss_name* is not registered.
    """
    key = loss_name.lower()
    if key not in _LOSS_FN:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available losses: {LOSS_NAMES}"
        )
    return _LOSS_FN[key](output, target, **kwargs)