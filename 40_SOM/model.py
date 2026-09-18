# model.py
"""
Self-Organizing Map (SOM) model.

Contents
--------
- SOMModel : Full SOM with rectangular or hexagonal topology, configurable
             neighborhood kernel and learning-rate schedule, online and
             batch training modes, and a rich inference API.

Architecture
------------
The SOM consists of a 2-D lattice of ``n_rows * n_cols`` prototype vectors
(neurons), each of dimension ``d`` (equal to the input dimensionality).
The weight tensor W has shape ``(n_rows, n_cols, d)``.

Training algorithm (online mode)
---------------------------------
For each epoch, each input x_i is presented once in random order:

    1. BMU search:
           bmu = argmin_{i,j}  ||x - w_{ij}||

    2. Neighborhood update:
           w_{ij} += alpha(t) * h(d_grid(bmu, ij), sigma(t)) * (x - w_{ij})

    where alpha(t) is the learning rate at epoch t and sigma(t) is the
    neighborhood radius, both decaying according to the chosen schedule.

Batch training mode
-------------------
In batch mode the weight update for each neuron is computed as the
weighted mean of *all* training inputs in a single pass:

    w_{ij} = sum_n h(d(bmu_n, ij)) * x_n  /  sum_n h(d(bmu_n, ij))

This is equivalent to one step of the generalised Lloyd algorithm and
converges faster than online mode but requires all data in memory.

Inference API
-------------
- ``predict(X)``      : Return BMU (row, col) for each sample.
- ``transform(X)``    : Return SOM-space coordinates (continuous 2-D projection).
- ``quantize(X)``     : Return the BMU weight vector for each sample.
- ``winner_map(X)``   : Return the activation count per neuron (hit map).
- ``u_matrix()``      : Unified distance matrix for cluster boundary detection.
- ``component_planes``: Per-feature weight slice for visual interpretation.
"""

import math
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from layers import (
    GridTopology,
    NeighborhoodKernel,
    LearningRateSchedule,
    get_distance,
    KERNEL_NAMES,
    SCHEDULE_NAMES,
)


# Safety cap on individual weight magnitudes (well within float32 range).
_WEIGHT_CLIP = 1e30


class SOMModel:
    """
    Self-Organizing Map with configurable topology, kernel, and schedule.

    Args
    ----
    n_rows       : Number of rows in the map lattice.
    n_cols       : Number of columns in the map lattice.
    input_dim    : Dimensionality of the input vectors.
    topology     : Grid topology: ``'rectangular'`` (default) or
                   ``'hexagonal'``.
    kernel       : Neighborhood kernel: ``'gaussian'``, ``'mexican_hat'``,
                   ``'bubble'``, or ``'epanechnikov'`` (default: ``'gaussian'``).
    distance     : Distance metric for BMU search: ``'euclidean'``,
                   ``'manhattan'``, or ``'cosine'`` (default: ``'euclidean'``).
    lr_schedule  : Learning rate decay schedule (default: ``'exponential'``).
    sigma_schedule: Neighborhood radius decay schedule (default: ``'exponential'``).
    lr_0         : Initial learning rate (default: ``0.5``).
    lr_min       : Minimum learning rate (default: ``0.01``).
    sigma_0      : Initial neighborhood radius (default: ``max(n_rows, n_cols) / 2``).
    sigma_min    : Minimum neighborhood radius (default: ``1.0``).
    init         : Weight initialisation: ``'random'`` (uniform over data range),
                   ``'pca'`` (first two principal components), or
                   ``'data'`` (random sample from training data).
    random_state : Random seed for reproducibility.
    clip_weights : If True (default), clip weights after each update to
                   ``[-_WEIGHT_CLIP, _WEIGHT_CLIP]`` to prevent runaway values.
    max_step     : Upper bound on the per-update effective gain ``|alpha * h|``
                   (default: ``0.99``).  The online update
                   ``w <- w + alpha*h*(x - w)`` is only stable when
                   ``|alpha*h| < 1`` for every neuron.

    Example
    -------
    .. code-block:: python

        som = SOMModel(n_rows=10, n_cols=10, input_dim=4)
        som.fit(X_train, n_epochs=200)
        bmus   = som.predict(X_test)
        coords = som.transform(X_test)
    """

    def __init__(
        self,
        n_rows:         int,
        n_cols:         int,
        input_dim:      int,
        topology:       str   = 'rectangular',
        kernel:         str   = 'gaussian',
        distance:       str   = 'euclidean',
        lr_schedule:    str   = 'exponential',
        sigma_schedule: str   = 'exponential',
        lr_0:           float = 0.5,
        lr_min:         float = 0.01,
        sigma_0:        Optional[float] = None,
        sigma_min:      float = 1.0,
        init:           str   = 'random',
        random_state:   Optional[int] = None,
        clip_weights:   bool  = True,
        max_step:       float = 0.99,
    ) -> None:
        self.n_rows       = n_rows
        self.n_cols       = n_cols
        self.input_dim    = input_dim
        self.topology     = topology
        self.kernel_name  = kernel
        self.distance_name= distance
        self.lr_schedule_name    = lr_schedule
        self.sigma_schedule_name = sigma_schedule
        self.lr_0         = lr_0
        self.lr_min       = lr_min
        self.sigma_0      = sigma_0 if sigma_0 is not None else max(n_rows, n_cols) / 2.0
        self.sigma_min    = sigma_min
        self.init         = init
        self.random_state = random_state
        self.clip_weights = clip_weights
        self.max_step     = max_step

        self._rng = np.random.RandomState(random_state)

        # Weight matrix — initialised lazily in fit()
        self.weights: Optional[np.ndarray] = None   # (n_rows, n_cols, d)

        # Internal components
        self._grid     = GridTopology(n_rows, n_cols, topology)
        self._kernel   = NeighborhoodKernel(kernel)
        self._dist_fn  = get_distance(distance)

        # Training statistics
        self.quantization_errors_: list = []
        self.topographic_errors_:  list = []
        self.n_epochs_:            int  = 0
        self.diverged_:            bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_finite(name: str, arr: np.ndarray) -> None:
        """Raise a clear error if ``arr`` contains non-finite entries."""
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"{name} contains {n_bad} non-finite value(s) "
                f"(inf/nan). Clean the input before training."
            )

    def _safe_h(self, h: np.ndarray) -> np.ndarray:
        """
        Sanitise a neighborhood kernel response.

        - Replaces non-finite entries with 0.
        - Clips to [-1, 1] (no valid SOM kernel exceeds this range).
        - Applies the stability bound |alpha * h| <= max_step.
        """
        h = np.where(np.isfinite(h), h, 0.0)
        h = np.clip(h, -1.0, 1.0)
        return h

    def _clip_weights_inplace(self) -> None:
        """Clamp weights to a safe magnitude and replace non-finite entries."""
        if not np.isfinite(self.weights).all():
            # Extremely aggressive: replace non-finite entries with column means.
            bad = ~np.isfinite(self.weights)
            if bad.any():
                col_means = np.nanmean(
                    np.where(np.isfinite(self.weights), self.weights, np.nan),
                    axis=(0, 1),
                )
                col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                self.weights[bad] = np.broadcast_to(
                    col_means, self.weights.shape
                )[bad]
        if self.clip_weights:
            np.clip(self.weights, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=self.weights)

    # ------------------------------------------------------------------
    def _init_weights(self, X: np.ndarray) -> None:
        """Initialise the weight matrix according to ``self.init``."""
        d = self.input_dim

        if self.init == 'pca':
            X_centered = X - X.mean(axis=0)
            cov        = np.cov(X_centered.T)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            pc1 = vecs[:, order[0]]
            pc2 = vecs[:, order[1]] if d > 1 else np.zeros(d)

            self.weights = np.zeros((self.n_rows, self.n_cols, d), dtype=np.float32)
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    t1 = (r / max(self.n_rows - 1, 1)) * 2.0 - 1.0
                    t2 = (c / max(self.n_cols - 1, 1)) * 2.0 - 1.0
                    self.weights[r, c] = X.mean(axis=0) + t1 * pc1 + t2 * pc2

        elif self.init == 'data':
            idx = self._rng.choice(
                len(X),
                size=self.n_rows * self.n_cols,
                replace=len(X) < self.n_rows * self.n_cols,
            )
            self.weights = X[idx].reshape(self.n_rows, self.n_cols, d).astype(np.float32)

        else:  # 'random'
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            self.weights = (
                self._rng.rand(self.n_rows, self.n_cols, d).astype(np.float32)
                * (hi - lo) + lo
            ).astype(np.float32)

        self._clip_weights_inplace()

    # ------------------------------------------------------------------
    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit for a single input vector.

        Args:
            x : Input vector of shape ``(d,)``.

        Returns:
            Tuple ``(bmu_row, bmu_col)``.
        """
        x_t = torch.from_numpy(x.astype(np.float32))
        w_t = torch.from_numpy(self.weights.astype(np.float32))
        dist = self._dist_fn(x_t, w_t)            # (n_rows, n_cols)
        idx  = int(dist.argmin())
        return divmod(idx, self.n_cols)

    # ------------------------------------------------------------------
    def _find_bmu_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Find the BMU for every sample in ``X`` in a vectorised pass.

        Uses a numerically stable computation of squared Euclidean distance:
        we centre both ``X`` and ``w`` by their mean before squaring, which
        keeps the intermediate values well below the float32 overflow range.

        Args:
            X : Data matrix of shape ``(N, d)``.

        Returns:
            Array of shape ``(N, 2)`` with (row, col) BMU indices.
        """
        N, d    = X.shape
        w_flat  = self.weights.reshape(-1, d)        # (K, d)

        # Centre both matrices to keep intermediate squares small.
        mu      = X.mean(axis=0, keepdims=True)
        X_c     = X - mu
        W_c     = w_flat - mu

        # Use float64 for the squared-distance computation regardless of
        # the input dtype, then cast back for argmin.
        X_c64   = X_c.astype(np.float64, copy=False)
        W_c64   = W_c.astype(np.float64, copy=False)

        X_sq    = (X_c64 ** 2).sum(axis=1, keepdims=True)       # (N, 1)
        W_sq    = (W_c64 ** 2).sum(axis=1, keepdims=True).T     # (1, K)
        XW      = X_c64 @ W_c64.T                               # (N, K)
        dists   = X_sq - 2.0 * XW + W_sq                        # (N, K)

        # Guard: squared distances must be non-negative.
        np.maximum(dists, 0.0, out=dists)

        bmu_idx = dists.argmin(axis=1)               # (N,)
        rows, cols = np.divmod(bmu_idx, self.n_cols)
        return np.stack([rows, cols], axis=1)        # (N, 2)

    # ------------------------------------------------------------------
    def fit(
        self,
        X:        np.ndarray,
        n_epochs: int  = 100,
        mode:     str  = 'online',
        shuffle:  bool = True,
        verbose:  bool = True,
        log_every:int  = 10,
    ) -> 'SOMModel':
        """
        Train the SOM.

        Args:
            X         : Training data of shape ``(N, d)``.
            n_epochs  : Number of full passes over the data (default: ``100``).
            mode      : ``'online'`` (sequential, one sample at a time) or
                        ``'batch'`` (batch Kohonen update).  Default: ``'online'``.
            shuffle   : Whether to shuffle the data at each epoch (default: ``True``).
            verbose   : Whether to print epoch logs (default: ``True``).
            log_every : Log frequency in epochs (default: ``10``).

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError : If ``X`` contains inf/nan, or if training diverges
                         (weights become non-finite despite the guards).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (N, {self.input_dim}); got {X.shape}."
            )
        self._check_finite("X", X)

        if self.weights is None:
            self._init_weights(X)

        lr_sched    = LearningRateSchedule(
            self.lr_schedule_name,    self.lr_0,    self.lr_min,    n_epochs)
        sigma_sched = LearningRateSchedule(
            self.sigma_schedule_name, self.sigma_0, self.sigma_min, n_epochs)

        self.diverged_ = False

        for epoch in range(n_epochs):
            alpha = lr_sched(epoch)
            sigma = max(float(sigma_sched(epoch)), 1e-6)

            if mode == 'batch':
                self._batch_step(X, alpha, sigma)
            else:
                indices = (
                    self._rng.permutation(len(X)) if shuffle else np.arange(len(X))
                )
                for idx in indices:
                    self._online_step(X[idx], alpha, sigma)
                    if self.diverged_:
                        break

            # Post-epoch safety net.
            if not np.isfinite(self.weights).all():
                self.diverged_ = True

            if self.diverged_:
                warnings.warn(
                    f"SOM training diverged at epoch {epoch + 1} "
                    f"(alpha={alpha:.5f}, sigma={sigma:.4f}). "
                    f"Consider: (a) standardising X, (b) lowering lr_0, "
                    f"(c) using init='pca', or (d) increasing sigma_min.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            self.n_epochs_ += 1

            if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
                bmus  = self._find_bmu_batch(X)
                w_np  = self.weights
                qe    = float(np.mean([
                    np.linalg.norm(X[i] - w_np[int(bmus[i, 0]), int(bmus[i, 1])])
                    for i in range(len(X))
                ]))
                self.quantization_errors_.append((epoch, qe))
                print(
                    f"  Epoch {epoch + 1:>4}/{n_epochs}"
                    f"  |  alpha = {alpha:.5f}"
                    f"  |  sigma = {sigma:.4f}"
                    f"  |  QE = {qe:.6f}"
                )

        return self

    # ------------------------------------------------------------------
    def _online_step(
        self, x: np.ndarray, alpha: float, sigma: float
    ) -> None:
        """
        Update weights for a single training sample (online mode).

        The update rule
            w <- w + alpha * h * (x - w)
        is stable only when ``|alpha * h| < 1``.  We enforce that here and
        guard against non-finite kernels, weights, or deltas.
        """
        if not np.isfinite(x).all() or not np.isfinite(self.weights).all():
            self.diverged_ = True
            return

        bmu_r, bmu_c = self._find_bmu(x)
        dist_sq      = self._grid.grid_distance_sq(bmu_r, bmu_c)

        # Guard: squared grid distances must be non-negative.
        dist_sq = np.maximum(dist_sq, 0.0)
        if not np.isfinite(dist_sq).all():
            return

        h = self._kernel(dist_sq, sigma)

        # Sanitise kernel response.
        h = self._safe_h(h)

        # Enforce stability: |alpha * h| <= max_step.
        if abs(alpha) > 0.0:
            h_max = float(np.abs(h).max())
            gain  = abs(alpha) * h_max
            if gain > self.max_step:
                h = h * (self.max_step / gain)

        delta = x.astype(np.float64) - self.weights.astype(np.float64)
        if not np.isfinite(delta).all():
            self.diverged_ = True
            return

        update = alpha * h[:, :, np.newaxis] * delta

        # Accumulate in float64, then cast back.
        new_w = self.weights.astype(np.float64) + update
        if not np.isfinite(new_w).all():
            self.diverged_ = True
            return

        if self.clip_weights:
            np.clip(new_w, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=new_w)

        self.weights = new_w.astype(np.float32)

    # ------------------------------------------------------------------
    def _batch_step(
        self, X: np.ndarray, alpha: float, sigma: float
    ) -> None:
        """
        Update weights using the batch Kohonen rule.

        The new weight for neuron (i, j) is the kernel-weighted mean of
        all training inputs:

            w_{ij} = sum_n h(d(bmu_n, ij)) * x_n  /  sum_n h(d(bmu_n, ij))
        """
        numerator   = np.zeros_like(self.weights, dtype=np.float64)
        denominator = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)

        for x in X:
            if not np.isfinite(x).all():
                continue
            bmu_r, bmu_c = self._find_bmu(x)
            dist_sq      = np.maximum(
                self._grid.grid_distance_sq(bmu_r, bmu_c), 0.0
            )
            h            = self._safe_h(self._kernel(dist_sq, sigma))
            h            = np.abs(h)              # batch rule requires non-negative weights
            numerator   += h[:, :, np.newaxis] * x.astype(np.float64)
            denominator += h

        mask = denominator > 1e-12
        new_w = self.weights.astype(np.float64)
        new_w[mask] = numerator[mask] / denominator[mask, np.newaxis]

        if not np.isfinite(new_w).all():
            self.diverged_ = True
            return

        if self.clip_weights:
            np.clip(new_w, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=new_w)

        self.weights = new_w.astype(np.float32)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the BMU (row, col) for each input sample.

        Args:
            X : Data matrix of shape ``(N, d)``.

        Returns:
            Array of shape ``(N, 2)`` with (row, col) indices.
        """
        if self.weights is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        self._check_finite("X", X)
        return self._find_bmu_batch(X)

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return the continuous 2-D SOM-space coordinates for each sample.

        The coordinates are the (row, col) position of the BMU, normalised
        to [0, 1] x [0, 1], providing a 2-D embedding of the input data
        that preserves the topological structure discovered during training.

        Args:
            X : Data matrix of shape ``(N, d)``.

        Returns:
            Array of shape ``(N, 2)`` with normalised (row, col) coordinates.
        """
        bmus = self.predict(X)
        rows = bmus[:, 0] / max(self.n_rows - 1, 1)
        cols = bmus[:, 1] / max(self.n_cols - 1, 1)
        return np.stack([rows, cols], axis=1)

    # ------------------------------------------------------------------
    def quantize(self, X: np.ndarray) -> np.ndarray:
        """
        Return the BMU weight vector (prototype) for each input sample.

        Args:
            X : Data matrix of shape ``(N, d)``.

        Returns:
            Reconstructed data of shape ``(N, d)`` — each row is the weight
            vector of the BMU for the corresponding input.
        """
        bmus = self.predict(X)
        return np.array([
            self.weights[int(r), int(c)]
            for r, c in bmus
        ])

    # ------------------------------------------------------------------
    def winner_map(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the hit map: number of times each neuron wins as BMU.

        Args:
            X : Data matrix of shape ``(N, d)``.

        Returns:
            Integer array of shape ``(n_rows, n_cols)`` with activation counts.
        """
        bmus = self.predict(X)
        hmap = np.zeros((self.n_rows, self.n_cols), dtype=int)
        for r, c in bmus:
            hmap[int(r), int(c)] += 1
        return hmap

    # ------------------------------------------------------------------
    def u_matrix(self) -> np.ndarray:
        """
        Compute the Unified Distance Matrix (U-matrix).

        The U-matrix entry for neuron (i, j) is the mean Euclidean distance
        between its weight vector and the weight vectors of its direct
        neighbors on the grid.  High U-matrix values indicate cluster
        boundaries; low values indicate cluster interiors.

        Returns:
            Array of shape ``(n_rows, n_cols)`` with mean neighbor distances.
        """
        U = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                        neighbors.append(
                            np.linalg.norm(self.weights[r, c] - self.weights[nr, nc])
                        )
                U[r, c] = float(np.mean(neighbors)) if neighbors else 0.0
        return U

    # ------------------------------------------------------------------
    def component_planes(self) -> np.ndarray:
        """
        Return the component planes of the SOM weight matrix.

        The k-th component plane is the 2-D slice of the weight matrix
        along the k-th feature dimension, showing how strongly each neuron
        responds to that feature across the map.

        Returns:
            Array of shape ``(d, n_rows, n_cols)``.
        """
        return self.weights.transpose(2, 0, 1)   # (d, n_rows, n_cols)

    # ------------------------------------------------------------------
    def count_neurons(self) -> int:
        """Return the total number of neurons in the map."""
        return self.n_rows * self.n_cols