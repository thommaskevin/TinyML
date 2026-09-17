# layers.py
"""
Layer definitions for the Self-Organizing Map (SOM) framework.

Contents
--------
- Distance metrics          : Euclidean, Manhattan, Cosine (with registry)
- NeighborhoodKernel        : Gaussian, Mexican Hat, Bubble, Epanechnikov
- LearningRateSchedule      : Linear, Exponential, Inverse-time, Cyclical
- GridTopology              : 2-D rectangular / hexagonal lattice coordinates

Background
----------
A Self-Organizing Map (Kohonen, 1982) is an unsupervised competitive neural
network that projects a high-dimensional input space onto a low-dimensional
(typically 2-D) discrete lattice while preserving topological structure.

At each training step a single input vector x is presented.  The neuron
whose weight vector w_i is closest to x (the Best Matching Unit, BMU) is
identified by a competitive search, and then a neighborhood of neurons
centered on the BMU is updated:

    w_i(t+1) = w_i(t) + alpha(t) * h(i, bmu, t) * (x - w_i(t))

where alpha(t) is the learning rate and h(i, bmu, t) is the neighborhood
kernel.  Over time, both alpha and the kernel radius sigma decrease, causing
the map to converge from a coarse global ordering to a fine local refinement.

The topology-preserving property arises because nearby neurons on the grid
are updated together during early training, pulling geographically adjacent
prototypes toward the same region of input space.

References
----------
Kohonen, T. (1982).
    Self-Organized Formation of Topologically Correct Feature Maps.
    *Biological Cybernetics*, 43(1), 59-69.
Kohonen, T. (2001).
    *Self-Organizing Maps* (3rd ed.). Springer.
Vesanto, J., & Alhoniemi, E. (2000).
    Clustering of the Self-Organizing Map.
    *IEEE Transactions on Neural Networks*, 11(3), 586-600.
"""

import math
from typing import Tuple

import numpy as np
import torch


# =============================================================================
# Distance Metrics
# =============================================================================

def euclidean_distance(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between an input vector and all weight vectors.

    Args:
        x : Input vector of shape ``(d,)``.
        w : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Distance matrix of shape ``(n_rows, n_cols)``.
    """
    diff = w - x.unsqueeze(0).unsqueeze(0)        # (n_rows, n_cols, d)
    return (diff ** 2).sum(dim=-1)                 # (n_rows, n_cols)


def manhattan_distance(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Manhattan (L1) distance between an input vector and all weight vectors.

    Args:
        x : Input vector of shape ``(d,)``.
        w : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Distance matrix of shape ``(n_rows, n_cols)``.
    """
    diff = w - x.unsqueeze(0).unsqueeze(0)
    return diff.abs().sum(dim=-1)


def cosine_distance(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Cosine dissimilarity (1 - cosine similarity) between an input vector
    and all weight vectors.

    Args:
        x : Input vector of shape ``(d,)``.
        w : Weight matrix of shape ``(n_rows, n_cols, d)``.

    Returns:
        Distance matrix of shape ``(n_rows, n_cols)``, values in [0, 2].
    """
    x_norm = x / (x.norm() + 1e-8)
    w_flat = w.reshape(-1, w.shape[-1])
    w_norm = w_flat / (w_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (w_norm @ x_norm).reshape(w.shape[:2])
    return 1.0 - cos_sim


DISTANCES: dict = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine':    cosine_distance,
}


def get_distance(name: str):
    """
    Return a distance function by name (case-insensitive).

    Args:
        name : One of ``'euclidean'``, ``'manhattan'``, ``'cosine'``.

    Returns:
        Callable ``(x, w) -> distance_matrix``.

    Raises:
        ValueError: if *name* is not registered.
    """
    key = name.lower()
    if key not in DISTANCES:
        raise ValueError(
            f"Unknown distance '{name}'. "
            f"Available: {sorted(DISTANCES.keys())}"
        )
    return DISTANCES[key]


# =============================================================================
# Grid Topology
# =============================================================================

class GridTopology:
    """
    2-D rectangular or hexagonal lattice coordinate system.

    Stores the (row, col) position of every neuron and provides the
    squared Euclidean distance on the grid between any pair of neurons.
    The distance is used by neighborhood kernels to determine how strongly
    a neighbor of the BMU is updated.

    Args:
        n_rows   : Number of rows in the lattice.
        n_cols   : Number of columns in the lattice.
        topology : ``'rectangular'`` (default) or ``'hexagonal'``.
    """

    def __init__(
        self,
        n_rows:   int,
        n_cols:   int,
        topology: str = 'rectangular',
    ) -> None:
        self.n_rows   = n_rows
        self.n_cols   = n_cols
        self.topology = topology.lower()
        self.coords   = self._build_coords()   # (n_rows, n_cols, 2)

    def _build_coords(self) -> np.ndarray:
        """
        Build the (row, col) coordinate array for every neuron.

        For hexagonal grids, odd rows are offset by 0.5 in the column
        direction, creating the classic honeycomb layout.
        """
        rows, cols = np.meshgrid(
            np.arange(self.n_rows, dtype=np.float32),
            np.arange(self.n_cols, dtype=np.float32),
            indexing='ij',
        )
        if self.topology == 'hexagonal':
            # Offset odd rows by 0.5 columns
            rows_idx = np.arange(self.n_rows)
            offset   = (rows_idx % 2) * 0.5
            cols += offset[:, np.newaxis]

        return np.stack([rows, cols], axis=-1)   # (n_rows, n_cols, 2)

    def grid_distance_sq(
        self,
        bmu_row: int,
        bmu_col: int,
    ) -> np.ndarray:
        """
        Compute the squared Euclidean distance on the grid from every
        neuron to the BMU located at ``(bmu_row, bmu_col)``.

        Args:
            bmu_row : Row index of the Best Matching Unit.
            bmu_col : Column index of the Best Matching Unit.

        Returns:
            Array of shape ``(n_rows, n_cols)`` with squared grid distances.
        """
        bmu_pos = self.coords[bmu_row, bmu_col]          # (2,)
        diff    = self.coords - bmu_pos                  # (n_rows, n_cols, 2)
        return (diff ** 2).sum(axis=-1)                  # (n_rows, n_cols)


# =============================================================================
# Neighborhood Kernels
# =============================================================================

class NeighborhoodKernel:
    """
    Neighborhood function h(i, bmu, sigma) for SOM weight updates.

    The kernel determines how strongly each neuron i is updated given
    that the BMU is at a grid distance of d(i, bmu).  As training
    progresses, the kernel width sigma decreases, restricting updates
    to neurons increasingly close to the BMU.

    Supported kernels
    -----------------
    gaussian
        h(d, sigma) = exp(-d^2 / (2 * sigma^2))
        The standard Kohonen kernel; smooth and infinitely supported.

    mexican_hat
        h(d, sigma) = (1 - 2*(d/sigma)^2) * exp(-d^2 / sigma^2)
        Adds a negative lateral inhibition ring, producing sharper
        feature maps at the cost of occasional instability.

    bubble
        h(d, sigma) = 1 if d <= sigma, else 0
        Tophat kernel; produces more uniform weight distributions
        within the bubble radius.

    epanechnikov
        h(d, sigma) = max(0, 1 - d^2 / sigma^2)
        Compact support like the bubble but with a smooth quadratic
        falloff; optimal in a mean-squared-error sense.

    Args:
        kernel : Kernel type (default: ``'gaussian'``).
    """

    def __init__(self, kernel: str = 'gaussian') -> None:
        key = kernel.lower()
        if key not in KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Available: {sorted(KERNELS.keys())}"
            )
        self.kernel = key

    def __call__(
        self,
        dist_sq: np.ndarray,
        sigma:   float,
    ) -> np.ndarray:
        """
        Evaluate the neighborhood kernel.

        Args:
            dist_sq : Squared grid distances of shape ``(n_rows, n_cols)``.
            sigma   : Current neighborhood radius (> 0).

        Returns:
            Neighborhood influence values of shape ``(n_rows, n_cols)``,
            all in [0, 1] for gaussian / bubble / epanechnikov, and
            potentially negative for mexican_hat.
        """
        return KERNELS[self.kernel](dist_sq, sigma)


def _gaussian(dist_sq: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-dist_sq / (2.0 * sigma ** 2 + 1e-12))


def _mexican_hat(dist_sq: np.ndarray, sigma: float) -> np.ndarray:
    norm = dist_sq / (sigma ** 2 + 1e-12)
    return (1.0 - 2.0 * norm) * np.exp(-norm)


def _bubble(dist_sq: np.ndarray, sigma: float) -> np.ndarray:
    return (dist_sq <= sigma ** 2).astype(np.float32)


def _epanechnikov(dist_sq: np.ndarray, sigma: float) -> np.ndarray:
    norm = dist_sq / (sigma ** 2 + 1e-12)
    return np.maximum(0.0, 1.0 - norm)


KERNELS: dict = {
    'gaussian':      _gaussian,
    'mexican_hat':   _mexican_hat,
    'bubble':        _bubble,
    'epanechnikov':  _epanechnikov,
}

KERNEL_NAMES: list = list(KERNELS.keys())


# =============================================================================
# Learning Rate Schedules
# =============================================================================

class LearningRateSchedule:
    """
    Annealing schedule for the SOM learning rate alpha(t) and
    neighborhood radius sigma(t).

    Both the learning rate and the neighborhood radius must decrease
    monotonically during training to guarantee convergence of the SOM
    algorithm.  This class provides four standard decay schedules.

    Supported schedules
    -------------------
    linear
        value(t) = value_0 * (1 - t / T)
        Simple linear decay from *value_0* to zero over *n_epochs*.

    exponential
        value(t) = value_0 * exp(-t / tau)
        where tau = n_epochs / log(value_0 / value_min).
        Rapid early decay, slow convergence; widely used in practice.

    inverse_time
        value(t) = value_0 / (1 + decay * t)
        Slower decay than exponential; useful for large maps.

    cyclical
        value(t) = value_min + 0.5 * (value_0 - value_min)
                               * (1 + cos(pi * t / half_period))
        Warm-restart cosine schedule.  Useful for avoiding local minima
        during map ordering.

    Args:
        schedule  : Schedule name (default: ``'exponential'``).
        value_0   : Initial value (learning rate or sigma).
        value_min : Minimum value at the end of training (default: ``0.01``).
        n_epochs  : Total number of training epochs.
        **kwargs  : Extra keyword arguments forwarded to the schedule:
                    ``decay`` (inverse_time), ``half_period`` (cyclical).
    """

    def __init__(
        self,
        schedule:  str   = 'exponential',
        value_0:   float = 0.5,
        value_min: float = 0.01,
        n_epochs:  int   = 100,
        **kwargs,
    ) -> None:
        key = schedule.lower()
        if key not in SCHEDULES:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                f"Available: {sorted(SCHEDULES.keys())}"
            )
        self.schedule  = key
        self.value_0   = value_0
        self.value_min = value_min
        self.n_epochs  = n_epochs
        self.kwargs    = kwargs

    def __call__(self, epoch: int) -> float:
        """
        Compute the schedule value at training epoch *epoch*.

        Args:
            epoch : Current epoch index (0-based).

        Returns:
            Scalar schedule value >= ``self.value_min``.
        """
        v = SCHEDULES[self.schedule](
            epoch, self.value_0, self.value_min, self.n_epochs,
            **self.kwargs,
        )
        return float(max(v, self.value_min))


def _linear_schedule(t, v0, vmin, T, **kw):
    return v0 * (1.0 - t / max(T - 1, 1))


def _exp_schedule(t, v0, vmin, T, **kw):
    tau = (T - 1) / (math.log(v0 / max(vmin, 1e-8)) + 1e-12)
    return v0 * math.exp(-t / max(tau, 1e-8))


def _inv_schedule(t, v0, vmin, T, **kw):
    decay = kw.get('decay', 1.0 / max(T, 1))
    return v0 / (1.0 + decay * t)


def _cyclical_schedule(t, v0, vmin, T, **kw):
    half = kw.get('half_period', T // 2)
    half = max(half, 1)
    return vmin + 0.5 * (v0 - vmin) * (1.0 + math.cos(math.pi * t / half))


SCHEDULES: dict = {
    'linear':      _linear_schedule,
    'exponential': _exp_schedule,
    'inverse_time':_inv_schedule,
    'cyclical':    _cyclical_schedule,
}

SCHEDULE_NAMES: list = list(SCHEDULES.keys())
