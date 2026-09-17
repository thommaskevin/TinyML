# losses.py
"""
Loss and evaluation metrics for Self-Organizing Map (SOM) training.

Supported metrics
-----------------
quantization_error   : Mean distance between each input and its BMU weight vector.
topographic_error    : Fraction of inputs whose two best BMUs are non-adjacent.
reconstruction_error : Mean squared reconstruction error via BMU weight vectors.
entropy              : Shannon entropy of the BMU activation distribution.
silhouette           : Mean silhouette coefficient of the learned clustering.
kl_divergence        : KL divergence between input and BMU frequency distributions.

Notes
-----
SOM training does not use gradient-based loss minimisation.  These metrics
are used exclusively to *monitor* learning quality and to provide stopping
criteria or hyperparameter selection signals.  They are computed after each
epoch (or after training) on the full dataset.

``compute_loss`` maps metric names to callables so that the training loop
can log any combination of metrics without conditional logic.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def quantization_error(
    X:    np.ndarray,
    bmus: np.ndarray,
    w:    np.ndarray,
) -> float:
    """
    Mean Euclidean distance between each input vector and its BMU weight.

    This is the primary quality metric for a SOM.  A lower value indicates
    that the weight vectors more accurately represent the input distribution.
    It decreases monotonically with training epochs when the learning rate
    and neighborhood radius are properly annealed.

    Formally:

        QE = (1/N) * sum_i  ||x_i - w_{bmu(i)}||_2

    Args:
        X    : Input data matrix of shape ``(N, d)``.
        bmus : BMU index array of shape ``(N, 2)`` with (row, col) pairs.
        w    : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Scalar quantization error (lower is better).
    """
    errors = []
    for i in range(len(X)):
        r, c = int(bmus[i, 0]), int(bmus[i, 1])
        errors.append(np.linalg.norm(X[i] - w[r, c]))
    return float(np.mean(errors))


def topographic_error(
    X:         np.ndarray,
    w:         np.ndarray,
    topology:  str = 'rectangular',
) -> float:
    """
    Fraction of input samples whose two best matching units (BMUs) are
    non-adjacent on the grid.

    Topographic error (Kiviluoto, 1996) measures how well the map preserves
    the topology of the input manifold.  A value of 0 means every sample's
    second-best BMU is always a direct neighbor of its best BMU; a value
    of 1 indicates complete topological disorder.

    Two neurons are adjacent if their Chebyshev (Chessboard) grid distance
    is exactly 1 (rectangular) or their hex-offset distance is <= 1 (hex).

    Args:
        X        : Input data matrix of shape ``(N, d)``.
        w        : Weight matrix of shape ``(n_rows, n_cols, d)``.
        topology : ``'rectangular'`` or ``'hexagonal'`` (default: ``'rectangular'``).

    Returns:
        Topographic error in [0, 1] (lower is better).
    """
    n_rows, n_cols, d = w.shape
    w_flat = w.reshape(-1, d)                      # (n_rows*n_cols, d)
    errors = 0

    for x in X:
        dists = np.linalg.norm(w_flat - x, axis=1)
        order = np.argsort(dists)
        bmu1_r, bmu1_c = divmod(int(order[0]), n_cols)
        bmu2_r, bmu2_c = divmod(int(order[1]), n_cols)
        dr = abs(bmu1_r - bmu2_r)
        dc = abs(bmu1_c - bmu2_c)
        if topology == 'hexagonal':
            adjacent = (dr <= 1) and (dc <= 1) and (dr + dc <= 1 + (bmu1_r % 2))
        else:
            adjacent = (dr <= 1) and (dc <= 1) and (dr + dc <= 2)
        if not adjacent:
            errors += 1

    return float(errors / max(len(X), 1))


def reconstruction_error(
    X:    np.ndarray,
    bmus: np.ndarray,
    w:    np.ndarray,
) -> float:
    """
    Mean squared reconstruction error: MSE between each input and its BMU
    weight vector.

    Unlike ``quantization_error`` (which uses L2 norm), this metric uses
    the squared L2 norm, making it more sensitive to large deviations.

        RE = (1/N) * sum_i  ||x_i - w_{bmu(i)}||_2^2

    Args:
        X    : Input data matrix of shape ``(N, d)``.
        bmus : BMU index array of shape ``(N, 2)``.
        w    : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Scalar reconstruction error (lower is better).
    """
    errors = []
    for i in range(len(X)):
        r, c = int(bmus[i, 0]), int(bmus[i, 1])
        errors.append(float(np.sum((X[i] - w[r, c]) ** 2)))
    return float(np.mean(errors))


def bmu_entropy(
    bmus:   np.ndarray,
    n_rows: int,
    n_cols: int,
) -> float:
    """
    Shannon entropy of the BMU activation frequency distribution.

    A perfectly uniform map (every neuron activated equally often) achieves
    the maximum entropy of log(n_rows * n_cols).  Low entropy indicates
    that only a small fraction of neurons are ever activated, signaling
    under-utilization of the map capacity.

    Args:
        bmus   : BMU index array of shape ``(N, 2)`` with (row, col) pairs.
        n_rows : Number of map rows.
        n_cols : Number of map columns.

    Returns:
        Entropy in nats (higher = more uniform coverage, better).
    """
    n_neurons = n_rows * n_cols
    counts = np.zeros(n_neurons)
    for r, c in bmus:
        counts[int(r) * n_cols + int(c)] += 1
    probs = counts / max(counts.sum(), 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def kl_divergence(
    bmus:   np.ndarray,
    n_rows: int,
    n_cols: int,
) -> float:
    """
    KL divergence from the uniform distribution to the empirical BMU
    activation frequency distribution.

    KL(uniform || p_bmu) = log(K) - H(p_bmu)

    where K = n_rows * n_cols and H is the Shannon entropy.  This is zero
    when the map coverage is perfectly uniform and positive otherwise.

    Args:
        bmus   : BMU index array of shape ``(N, 2)``.
        n_rows : Number of map rows.
        n_cols : Number of map columns.

    Returns:
        Scalar KL divergence >= 0 (lower = more uniform, better).
    """
    K   = n_rows * n_cols
    H   = bmu_entropy(bmus, n_rows, n_cols)
    return float(math.log(K) - H) if K > 1 else 0.0


def silhouette_score(
    X:    np.ndarray,
    bmus: np.ndarray,
) -> float:
    """
    Mean silhouette coefficient of the SOM clustering.

    The silhouette score (Rousseeuw, 1987) measures how well each point
    fits its assigned cluster (BMU) relative to the nearest alternative
    cluster:

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    where a(i) is the mean intra-cluster distance and b(i) is the mean
    distance to the nearest other cluster.  Values in [-1, 1]; higher
    is better, > 0.5 is considered good clustering.

    Args:
        X    : Input data matrix of shape ``(N, d)``.
        bmus : BMU flat index array of shape ``(N,)`` (use
               ``bmu_row * n_cols + bmu_col``).

    Returns:
        Mean silhouette score in [-1, 1].
    """
    labels = bmus.astype(int)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    s_vals = []
    for i in range(len(X)):
        lbl = labels[i]
        same = X[labels == lbl]
        if len(same) > 1:
            a = float(np.mean(np.linalg.norm(same - X[i], axis=1)))
        else:
            a = 0.0

        b = np.inf
        for other_lbl in unique:
            if other_lbl == lbl:
                continue
            other = X[labels == other_lbl]
            mean_dist = float(np.mean(np.linalg.norm(other - X[i], axis=1)))
            if mean_dist < b:
                b = mean_dist

        denom = max(a, b)
        s_vals.append((b - a) / denom if denom > 0 else 0.0)

    return float(np.mean(s_vals))


# ---------------------------------------------------------------------------
# math import needed by kl_divergence
# ---------------------------------------------------------------------------
import math


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRIC_FN: dict = {
    'quantization_error':  quantization_error,
    'topographic_error':   topographic_error,
    'reconstruction_error':reconstruction_error,
    'entropy':             bmu_entropy,
    'kl_divergence':       kl_divergence,
    'silhouette':          silhouette_score,
}

LOSS_NAMES: list = list(_METRIC_FN.keys())


def compute_loss(
    metric_name: str,
    X:           np.ndarray,
    bmus:        np.ndarray,
    w:           np.ndarray,
    **kwargs,
) -> float:
    """
    Compute a named SOM quality metric.

    Args:
        metric_name : One of ``'quantization_error'``, ``'topographic_error'``,
                      ``'reconstruction_error'``, ``'entropy'``,
                      ``'kl_divergence'``, or ``'silhouette'``.
        X           : Input data matrix of shape ``(N, d)``.
        bmus        : BMU index array of shape ``(N, 2)`` or ``(N,)`` depending
                      on the metric (see individual function signatures).
        w           : SOM weight matrix of shape ``(n_rows, n_cols, d)``.
        **kwargs    : Extra arguments forwarded to the metric function
                      (e.g. ``topology='hexagonal'``, ``n_rows=10``).

    Returns:
        Scalar metric value.

    Raises:
        ValueError: if *metric_name* is not registered.
    """
    key = metric_name.lower()
    if key not in _METRIC_FN:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: {LOSS_NAMES}"
        )
    return _METRIC_FN[key](X, bmus, w, **kwargs)
