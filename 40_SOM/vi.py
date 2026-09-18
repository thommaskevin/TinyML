# vi.py
"""
Variational and regularisation extensions for Self-Organizing Map (SOM) models.

Contents
--------
- ``weight_smoothness_loss`` : Penalises discontinuities in the weight lattice
                               (promotes topographic ordering).
- ``coverage_loss``          : Penalises under-utilised neurons (dead units).
- ``entropy_reg_loss``       : Combined task metric + entropy regularisation
                               to encourage uniform map coverage.
- ``probabilistic_bmu``      : Soft BMU assignment via Gaussian kernel over
                               distances (differentiable approximation).
- ``som_kl_loss``            : KL divergence loss between the soft BMU
                               distribution and a target distribution
                               (inspired by DEC / Student-T clustering).

Background
----------
Standard SOM training is a winner-take-all competitive algorithm that does
not minimise a global differentiable objective.  However, several soft or
probabilistic extensions have been proposed that reframe SOM learning as
approximate posterior inference:

  - The **Generative Topographic Mapping** (GTM, Bishop et al. 1998) replaces
    the hard BMU assignment with a mixture-of-Gaussians model trained by EM.

  - **Deep Embedded Clustering** (DEC, Xie et al. 2016) and its SOM variant
    **SOM-VAE** (Fortuin et al. 2019) use a KL divergence between a soft
    assignment distribution and a sharpened target distribution to refine
    cluster assignments jointly with a feature encoder.

The functions in this module provide lightweight, framework-agnostic
approximations to these ideas.  They operate on NumPy arrays and can be
used as auxiliary monitoring metrics or as penalty terms in hybrid
encoder-SOM pipelines.

References
----------
Bishop, C. M., Svensen, M., & Williams, C. K. I. (1998).
    GTM: The Generative Topographic Mapping. *Neural Computation*, 10(1), 215-234.
Xie, J., Girshick, R., & Farhadi, A. (2016).
    Unsupervised Deep Embedding for Clustering Analysis. *ICML 2016*.
Fortuin, V., Huber, M., Rios, F., Zimmermann, T., & Ratsch, G. (2019).
    SOM-VAE: Interpretable Discrete Representation Learning on Time Series.
    *ICLR 2019*.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_finite(a: np.ndarray) -> bool:
    """Return True if every element of ``a`` is finite (no inf/nan)."""
    return bool(np.isfinite(a).all())


# ---------------------------------------------------------------------------
# Weight smoothness regulariser
# ---------------------------------------------------------------------------

def weight_smoothness_loss(w: np.ndarray) -> float:
    """
    Penalise discontinuities in the SOM weight lattice.

    Computes the mean squared difference between each neuron's weight
    vector and the weight vectors of its four cardinal neighbors:

        L_smooth = (1 / (4 * K * d)) *
                   sum_{i,j} sum_{(di,dj) in neighbors}
                   ||w_{i,j} - w_{i+di, j+dj}||_2^2

    A low smoothness loss indicates that neighboring neurons in the grid
    represent similar regions of input space — a hallmark of a well-ordered
    SOM.  This loss can be used as a stopping criterion during training or
    as a hyperparameter selection signal.

    Args:
        w : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Scalar smoothness loss (lower = smoother map, better topology).
        Returns ``nan`` if ``w`` contains non-finite values (i.e. the
        upstream SOM training has already diverged).
    """
    # Guard: if weights already contain inf/nan, upstream training diverged.
    if not _is_finite(w):
        return float("nan")

    total = 0.0
    count = 0
    n_rows, n_cols, _ = w.shape

    with np.errstate(over="ignore", invalid="ignore"):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r_src = slice(max(0, -dr), n_rows + min(0, -dr))
            c_src = slice(max(0, -dc), n_cols + min(0, -dc))
            r_dst = slice(max(0,  dr), n_rows + min(0,  dr))
            c_dst = slice(max(0,  dc), n_cols + min(0,  dc))
            diff   = w[r_src, c_src] - w[r_dst, c_dst]
            total += float((diff ** 2).sum())
            count += diff.shape[0] * diff.shape[1]

    return total / max(count * w.shape[-1], 1)


# ---------------------------------------------------------------------------
# Coverage (dead-unit) penalty
# ---------------------------------------------------------------------------

def coverage_loss(
    bmus:        np.ndarray,
    n_rows:      int,
    n_cols:      int,
    min_fraction: float = 0.01,
) -> float:
    """
    Penalise dead neurons (neurons that are never selected as BMU).

    A neuron is considered dead if its activation frequency falls below
    ``min_fraction * (1 / K)`` where K = n_rows * n_cols.  The penalty
    is the fraction of neurons that are dead:

        L_coverage = (number of dead neurons) / K

    A high coverage loss indicates that the map is under-utilised and
    some neurons are never contributing to representations.  This can
    be addressed by reducing the neighborhood radius more slowly or by
    using the PCA initialisation.

    Args:
        bmus         : BMU index array of shape ``(N, 2)`` with (row, col) pairs.
        n_rows       : Number of map rows.
        n_cols       : Number of map columns.
        min_fraction : Fraction of the uniform frequency below which a neuron
                       is considered dead (default: ``0.01``).

    Returns:
        Dead-neuron fraction in [0, 1] (lower is better).
    """
    K      = n_rows * n_cols
    counts = np.zeros(K)
    for r, c in bmus:
        counts[int(r) * n_cols + int(c)] += 1
    threshold = min_fraction / K * len(bmus)
    dead      = (counts < threshold).sum()
    return float(dead / K)


# ---------------------------------------------------------------------------
# Combined task metric + entropy regularisation
# ---------------------------------------------------------------------------

def entropy_reg_loss(
    X:           np.ndarray,
    bmus:        np.ndarray,
    w:           np.ndarray,
    reg_weight:  float = 0.1,
) -> float:
    """
    Combined quantization error + entropy regularisation.

    Adds a negative-entropy penalty to the quantization error to encourage
    uniform coverage of the map:

        L_total = QE - reg_weight * H(p_bmu)

    where H is the Shannon entropy of the BMU activation distribution.
    Maximising entropy (minimising -H) pushes the training distribution
    to activate all neurons equally, preventing dead units without changing
    the competitive BMU selection rule.

    Args:
        X          : Input data matrix of shape ``(N, d)``.
        bmus       : BMU index array of shape ``(N, 2)``.
        w          : Weight matrix of shape ``(n_rows, n_cols, d)``.
        reg_weight : Weight on the negative entropy term (default: ``0.1``).

    Returns:
        Scalar combined loss (lower is better).
    """
    from losses import quantization_error, bmu_entropy
    n_rows, n_cols = w.shape[:2]
    qe = quantization_error(X, bmus, w)
    H  = bmu_entropy(bmus, n_rows, n_cols)
    return qe - reg_weight * H


# ---------------------------------------------------------------------------
# Soft (probabilistic) BMU assignment
# ---------------------------------------------------------------------------

def probabilistic_bmu(
    X:     np.ndarray,
    w:     np.ndarray,
    beta:  float = 1.0,
) -> np.ndarray:
    """
    Compute soft BMU assignment probabilities via a Boltzmann distribution.

    Instead of the hard winner-take-all assignment, each input x_i is
    assigned to each neuron k with probability proportional to:

        q_{ik} = exp(-beta * ||x_i - w_k||^2) / Z_i

    where Z_i is the normalisation constant.  As beta -> infinity, this
    approaches the hard BMU assignment.  As beta -> 0, all neurons
    receive equal assignment probability.

    This soft assignment is the key ingredient of the Generative
    Topographic Mapping (Bishop et al., 1998) and the SOM-VAE
    reconstruction loss (Fortuin et al., 2019).

    Args:
        X    : Input data matrix of shape ``(N, d)``.
        w    : Weight matrix of shape ``(n_rows, n_cols, d)``.
        beta : Inverse temperature (sharpness) parameter (default: ``1.0``).

    Returns:
        Soft assignment matrix of shape ``(N, n_rows * n_cols)``,
        each row sums to 1.  Returns a matrix of ``nan`` if ``X`` or ``w``
        contains non-finite values.
    """
    N, d = X.shape
    K    = w.shape[0] * w.shape[1]

    # Guard: refuse to compute on divergent inputs.
    if not (_is_finite(X) and _is_finite(w)):
        return np.full((N, K), np.nan, dtype=float)

    w_flat = w.reshape(K, d)                       # (K, d)

    with np.errstate(over="ignore", invalid="ignore"):
        X_sq    = (X ** 2).sum(axis=1, keepdims=True)  # (N, 1)
        W_sq    = (w_flat ** 2).sum(axis=1)            # (K,)
        XW      = X @ w_flat.T                         # (N, K)
        dist_sq = X_sq - 2 * XW + W_sq                 # (N, K)

        log_q   = -beta * dist_sq
        log_q  -= log_q.max(axis=1, keepdims=True)     # numerical stability
        q       = np.exp(log_q)
        q      /= q.sum(axis=1, keepdims=True)

    return q                                           # (N, K)


# ---------------------------------------------------------------------------
# SOM KL divergence loss (DEC-style)
# ---------------------------------------------------------------------------

def som_kl_loss(
    X:    np.ndarray,
    w:    np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    KL divergence between soft BMU assignments and a sharpened target
    distribution (DEC-style clustering loss).

    Following Xie et al. (2016), define the soft assignment distribution:

        q_{ik} = softmax(-beta * ||x_i - w_k||^2)

    and the target distribution (sharpened, cluster-frequency normalised):

        p_{ik} = (q_{ik}^2 / f_k) / sum_j (q_{ij}^2 / f_j)

    where f_k = sum_i q_{ik} is the soft cluster frequency.

    The loss is:

        L_KL = sum_i sum_k p_{ik} * log(p_{ik} / q_{ik})

    Minimising this loss drives the soft assignments to become sharper
    (more confident) and more uniformly distributed across clusters.

    Args:
        X    : Input data matrix of shape ``(N, d)``.
        w    : Weight matrix of shape ``(n_rows, n_cols, d)``.
        beta : Inverse temperature for soft BMU assignment (default: ``1.0``).

    Returns:
        Scalar KL divergence loss (lower = more confident clustering).
        Returns ``nan`` if the soft assignments could not be computed.
    """
    q = probabilistic_bmu(X, w, beta=beta)         # (N, K)

    if not np.isfinite(q).all():
        return float("nan")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        f = q.sum(axis=0, keepdims=True)           # (1, K) — soft frequency
        p = (q ** 2) / (f + 1e-12)
        p /= p.sum(axis=1, keepdims=True)          # normalise rows

        kl = (p * np.log((p + 1e-12) / (q + 1e-12))).sum(axis=1)

    return float(kl.mean())