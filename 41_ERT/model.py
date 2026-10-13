# model.py
"""
Extra Trees (Extremely Randomized Trees) model.

Contents
--------
- ExtraTreeModel   : A single Extremely Randomized Tree (regression or
                     classification).
- ExtraTreesModel  : An ensemble of ExtraTreeModel instances (the full
                     Extra Trees algorithm of Geurts et al., 2006).

Architecture
------------
Each ExtraTreeModel is a binary decision tree where:

  - At every internal node, ``max_features`` candidate features are drawn
    uniformly at random (without replacement) from all p input features.
  - For each candidate feature, exactly **one** split threshold is drawn
    uniformly from the range [min(x_feat), max(x_feat)] in the current node.
  - The (feature, threshold) pair that maximises the chosen impurity
    criterion is selected.  No search over candidate thresholds is performed.
  - Leaf predictions are:
      - Regression   : mean of target values reaching the leaf.
      - Classification: class probability vector (normalised class counts).

The ExtraTreesModel trains ``n_estimators`` trees on the **full** training
set (no bootstrap sampling) and aggregates predictions by:
  - Regression   : arithmetic mean of leaf means across all trees.
  - Classification: averaging class probability vectors, then argmax.

Input / output shapes
---------------------
- ``fit(X, y)``     : X (N, p), y (N,) integer labels or floats
- ``predict(X)``    : (N,) — class labels (classification) or means (regression)
- ``predict_proba(X)``: (N, K) — class probabilities (classification only)
- ``feature_importances_``: (p,) — MDI importances normalised to sum 1
"""

import math
from copy import deepcopy
from typing import Optional, List, Tuple

import numpy as np

from layers import ExtraTreeNode, SplitCriterion, ImportanceAccumulator


# =============================================================================
# Single Extra Tree
# =============================================================================

class ExtraTreeModel:
    """
    A single Extremely Randomized Tree.

    Supports both regression (criterion ``'mse'`` or ``'mae'``) and
    classification (criterion ``'gini'`` or ``'entropy'``).

    Args
    ----
    task             : ``'classification'`` or ``'regression'``.
    criterion        : Impurity function name (default: ``'gini'`` for
                       classification, ``'mse'`` for regression).
    max_depth        : Maximum tree depth (default: ``None`` = unlimited).
    min_samples_split: Minimum samples required to split an internal node
                       (default: ``2``).
    min_samples_leaf : Minimum samples required in each leaf (default: ``1``).
    max_features     : Number of candidate features per split.  ``'sqrt'``,
                       ``'log2'``, an integer, or ``None`` for all features.
    random_state     : Seed for reproducibility.
    """

    def __init__(
        self,
        task:              str  = 'classification',
        criterion:         str  = 'gini',
        max_depth:         Optional[int] = None,
        min_samples_split: int  = 2,
        min_samples_leaf:  int  = 1,
        max_features:      object = 'sqrt',
        random_state:      Optional[int] = None,
    ) -> None:
        self.task              = task
        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state

        self.root:        Optional[ExtraTreeNode] = None
        self.n_features_: int                     = 0
        self.n_classes_:  int                     = 0
        self.classes_:    Optional[np.ndarray]    = None
        self._node_id:    int                     = 0
        self._n_train:    int                     = 0

    # ------------------------------------------------------------------
    def _new_id(self) -> int:
        nid = self._node_id
        self._node_id += 1
        return nid

    # ------------------------------------------------------------------
    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        """Compute the leaf prediction value."""
        if self.task == 'classification':
            counts = np.bincount(y.astype(int), minlength=self.n_classes_)
            return counts / max(counts.sum(), 1)
        return np.array([float(y.mean())])

    # ------------------------------------------------------------------
    def _build(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        depth:    int,
        splitter: SplitCriterion,
    ) -> ExtraTreeNode:
        """Recursively build the tree from node at the given depth."""
        n = len(y)
        node_impurity = splitter.criterion_fn(y)
        node = ExtraTreeNode(
            n_samples=n,
            impurity=node_impurity,
            depth=depth,
            node_id=self._new_id(),
            is_leaf=True,
            value=self._leaf_value(y),
        )

        # Stopping conditions
        if n < self.min_samples_split:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if node_impurity == 0.0:
            return node

        # Find best random split
        feat, thresh, gain = splitter.find_best_split(X, y)

        if feat is None or gain <= 0:
            return node

        # Partition data
        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        node.is_leaf        = False
        node.feature_idx    = feat
        node.threshold      = float(thresh)
        node.impurity_gain  = float(gain)
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1, splitter)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1, splitter)
        return node

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreeModel':
        """
        Train the Extra Tree.

        Args:
            X : Feature matrix of shape ``(N, p)``.
            y : Target array of shape ``(N,)``.  Integer labels for
                classification; floats for regression.

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        self._n_train    = len(y)
        self._node_id    = 0

        if self.task == 'classification':
            self.classes_  = np.unique(y.astype(int))
            self.n_classes_ = int(self.classes_.max()) + 1
            y_enc = y.astype(int)
        else:
            self.n_classes_ = 1
            y_enc = y.astype(np.float64)

        splitter = SplitCriterion(
            criterion=self.criterion,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.root = self._build(X, y_enc, depth=0, splitter=splitter)
        return self

    # ------------------------------------------------------------------
    def _traverse(self, x: np.ndarray, node: ExtraTreeNode) -> np.ndarray:
        """Traverse the tree for a single sample; return leaf value."""
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability vectors for each sample.

        Args:
            X : Feature matrix of shape ``(M, p)``.

        Returns:
            Array of shape ``(M, K)`` with class probabilities.

        Raises:
            RuntimeError: if the tree has not been fitted.
            ValueError:   if the task is not ``'classification'``.
        """
        if self.root is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if self.task != 'classification':
            raise ValueError("predict_proba() is only available for classification.")
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x, self.root) for x in X])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (classification) or target means (regression).

        Args:
            X : Feature matrix of shape ``(M, p)``.

        Returns:
            Array of shape ``(M,)`` with predictions.
        """
        if self.root is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        preds = [self._traverse(x, self.root) for x in X]
        if self.task == 'classification':
            return np.array([int(p.argmax()) for p in preds])
        return np.array([float(p[0]) for p in preds])

    # ------------------------------------------------------------------
    def get_depth(self) -> int:
        """Return the actual maximum depth of the fitted tree."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)

    # ------------------------------------------------------------------
    def get_n_leaves(self) -> int:
        """Return the total number of leaf nodes."""
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root)


# =============================================================================
# Extra Trees Ensemble
# =============================================================================

class ExtraTreesModel:
    """
    Ensemble of Extremely Randomized Trees (Extra Trees).

    Trains ``n_estimators`` ``ExtraTreeModel`` instances on the **full**
    training set (no bootstrap resampling) and aggregates their predictions.

    Args
    ----
    n_estimators     : Number of trees in the ensemble (default: ``100``).
    task             : ``'classification'`` or ``'regression'``.
    criterion        : Impurity criterion (default: ``'gini'`` /  ``'mse'``).
    max_depth        : Maximum depth per tree (default: ``None``).
    min_samples_split: Minimum samples to split a node (default: ``2``).
    min_samples_leaf : Minimum samples per leaf (default: ``1``).
    max_features     : Features per split (default: ``'sqrt'``).
    random_state     : Global random seed.

    Attributes
    ----------
    estimators_             : List of fitted ``ExtraTreeModel`` instances.
    feature_importances_    : MDI importances of shape ``(p,)``.
    classes_                : Unique class labels (classification only).
    oob_score_              : Out-of-bag score (not applicable; always None,
                              included for API symmetry with Random Forest).
    """

    def __init__(
        self,
        n_estimators:      int  = 100,
        task:              str  = 'classification',
        criterion:         str  = 'gini',
        max_depth:         Optional[int] = None,
        min_samples_split: int  = 2,
        min_samples_leaf:  int  = 1,
        max_features:      object = 'sqrt',
        random_state:      Optional[int] = None,
    ) -> None:
        self.n_estimators      = n_estimators
        self.task              = task
        self.criterion         = criterion
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state

        self.estimators_:          List[ExtraTreeModel] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_:             Optional[np.ndarray] = None
        self.n_features_:          int                  = 0
        self.n_classes_:           int                  = 0
        self.oob_score_:           None                 = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreesModel':
        """
        Fit the Extra Trees ensemble.

        Args:
            X : Feature matrix of shape ``(N, p)``.
            y : Target array of shape ``(N,)``.

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        if self.task == 'classification':
            self.classes_   = np.unique(y.astype(int))
            self.n_classes_ = int(y.astype(int).max()) + 1

        master_rng = np.random.RandomState(self.random_state)
        accumulator = ImportanceAccumulator(self.n_features_)
        self.estimators_ = []

        for b in range(self.n_estimators):
            seed = int(master_rng.randint(0, 2 ** 31))
            tree = ExtraTreeModel(
                task=self.task,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X, y)
            self.estimators_.append(tree)
            accumulator.update(tree.root, len(y))

        self.feature_importances_ = accumulator.result()
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities averaged over all trees.

        Args:
            X : Feature matrix of shape ``(M, p)``.

        Returns:
            Array of shape ``(M, K)`` with averaged class probabilities.
        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict_proba().")
        X     = np.asarray(X, dtype=np.float64)
        proba = np.mean(
            [t.predict_proba(X) for t in self.estimators_], axis=0
        )
        return proba

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (classification) or regression means.

        Args:
            X : Feature matrix of shape ``(M, p)``.

        Returns:
            Array of shape ``(M,)``.
        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if self.task == 'classification':
            return self.predict_proba(X).argmax(axis=1).astype(int)
        preds = np.mean(
            [t.predict(X) for t in self.estimators_], axis=0
        )
        return preds

    # ------------------------------------------------------------------
    def feature_importances(self) -> np.ndarray:
        """
        Return normalised MDI feature importances.

        Returns:
            Array of shape ``(p,)`` summing to 1.
        """
        if self.feature_importances_ is None:
            raise RuntimeError("Call fit() first.")
        return self.feature_importances_
