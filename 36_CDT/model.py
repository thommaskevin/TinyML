# model.py
"""
Causal Tree Model for regression, binary, and multiclass tasks.

Architecture
------------
A single honest Causal Tree is grown on a random 50/50 split of the
training data:

  - *Structure half*  → determines split rules (feature, threshold at each node)
  - *Estimation half* → populates leaf-level CATE estimates

The Conditional Average Treatment Effect (CATE) at a leaf ℓ is:

    Regression / Binary:
        τ̂(ℓ) = mean(Y | W=1, ℓ) − mean(Y | W=0, ℓ)

    Multiclass (K classes):
        τ̂_k(ℓ) = P(Y=k | W=1, ℓ) − P(Y=k | W=0, ℓ)   k = 0, …, K−1
        predicted class = argmax_k τ̂_k(ℓ)

Task configuration
------------------
Set ``task`` when constructing the model:

.. code-block:: python

    model = CausalTreeModel(task='regression',  ...)  # continuous Y
    model = CausalTreeModel(task='binary',      ...)  # binary Y ∈ {0,1}
    model = CausalTreeModel(task='multiclass',  n_classes=3, ...)

Configuration dict format
--------------------------
``tree_config`` — dict controlling the single CT layer:

.. code-block:: python

    {
        'max_depth':         5,
        'min_samples_leaf':  20,
        'min_samples_treat': 5,
        'criterion':         'variance',   # 'variance' | 'mse' | 'tau_risk'
        'honest':            True,
        'n_features':        None,         # None | 'sqrt' | 'log2' | int
    }

Input / output shapes
---------------------
- Input  : X of shape ``(n, p)``  (covariates)
           w of shape ``(n,)``    (binary treatment indicator)
           y of shape ``(n,)``    (outcome)
- Output : ``predict(X)``   → shape ``(n,)``
           ``predict_proba(X)`` → shape ``(n, K)`` (multiclass only)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from layers import CausalTreeLayer


class CausalTreeModel:
    """
    Single honest Causal Tree for heterogeneous treatment effect estimation.

    Supports three task types:

    - ``'regression'``  — continuous outcome; predicts τ̂(x) ∈ ℝ.
    - ``'binary'``      — binary outcome; predicts risk difference τ̂(x) ∈ [-1,1].
    - ``'multiclass'``  — multi-class outcome; predicts treatment-induced
                          class-probability shifts and returns the class with
                          the largest positive shift.

    Args:
        task          : Task type (default: ``'regression'``).
        tree_config   : Dict of ``CausalTreeLayer`` hyperparameters.
                        Unspecified keys fall back to layer defaults.
        n_classes     : Number of classes (required for multiclass).
        random_state  : Master random seed (default: ``42``).
    """

    def __init__(
        self,
        task:         str  = 'regression',
        tree_config:  Optional[dict] = None,
        n_classes:    Optional[int]  = None,
        random_state: int = 42,
    ) -> None:
        self.task         = task.lower()
        self.tree_config  = tree_config or {}
        self.n_classes    = n_classes
        self.random_state = random_state

        cfg = self.tree_config
        self._tree = CausalTreeLayer(
            task              = self.task,
            max_depth         = cfg.get('max_depth',         5),
            min_samples_leaf  = cfg.get('min_samples_leaf',  20),
            min_samples_treat = cfg.get('min_samples_treat', 5),
            criterion         = cfg.get('criterion',         'variance'),
            honest            = cfg.get('honest',            True),
            n_features        = cfg.get('n_features',        None),
            n_classes         = n_classes,
            random_state      = random_state,
        )

    # ------------------------------------------------------------------
    @property
    def tree_(self) -> CausalTreeLayer:
        """Underlying ``CausalTreeLayer`` (for direct inspection)."""
        return self._tree

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> 'CausalTreeModel':
        """
        Fit the Causal Tree.

        Args:
            X : Covariate matrix of shape ``(n, p)``.
            y : Outcome vector of shape ``(n,)``.
            w : Binary treatment indicator of shape ``(n,)`` (1 = treated).

        Returns:
            ``self`` (for method chaining).
        """
        self._tree.fit(
            np.asarray(X, dtype=np.float64),
            y,
            np.asarray(w, dtype=np.int32),
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE / class labels.

        Args:
            X : Covariate matrix of shape ``(n, p)``.

        Returns:
            - Regression / binary  : array of shape ``(n,)`` with τ̂(x).
            - Multiclass           : integer class labels, shape ``(n,)``.
        """
        return self._tree.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Per-class CATE differences (multiclass only).

        Args:
            X : Covariate matrix of shape ``(n, p)``.

        Returns:
            Array of shape ``(n, K)`` with τ̂_k(x) for each class k.
        """
        return self._tree.predict_proba(np.asarray(X, dtype=np.float64))

    # ------------------------------------------------------------------
    def get_depth(self) -> int:
        """Return the actual depth of the fitted tree."""
        return self._tree.get_depth()

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes."""
        return self._tree.get_n_leaves()

    def get_leaf_nodes(self):
        """Return all leaf nodes in depth-first order."""
        return self._tree.get_leaf_nodes()

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """
        Return a human-readable summary of the fitted tree.

        Returns:
            Multi-line string with task, depth, leaf count, and
            per-leaf statistics.
        """
        lines = [
            f"CausalTreeModel",
            f"  task        : {self.task}",
            f"  criterion   : {self._tree.criterion_name}",
            f"  honest      : {self._tree.honest}",
            f"  max_depth   : {self._tree.max_depth}",
            f"  depth (fit) : {self.get_depth()}",
            f"  n_leaves    : {self.get_n_leaves()}",
        ]
        if self.task == 'multiclass':
            lines.append(f"  n_classes   : {self._tree.n_classes}")
        lines.append("")
        leaves = self.get_leaf_nodes()
        lines.append(f"  Leaf estimates ({len(leaves)} leaves):")
        for i, lf in enumerate(leaves):
            if self.task == 'multiclass':
                tau_str = np.array2string(lf.tau, precision=3, separator=',')
            else:
                tau_str = f"{lf.tau:+.4f}"
            lines.append(
                f"    Leaf {i+1:2d}: τ̂={tau_str}  "
                f"T={lf.n_treated}  C={lf.n_control}"
            )
        return "\n".join(lines)
