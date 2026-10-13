# layers.py
"""
Layer definitions for the Extra Trees (Extremely Randomized Trees) framework.

Contents
--------
- Impurity / criterion functions  (with registry and factory helper)
- ExtraTreeNode                   : a single node in an Extra Tree
- SplitCriterion                  : evaluates candidate splits for regression
                                    and classification tasks
- ImportanceAccumulator           : aggregates feature importances across trees

Background
----------
Extra Trees (Extremely Randomized Trees), introduced by Geurts, Ernst, and
Wehenkel (2006), are an ensemble of decision trees with two key sources of
additional randomisation compared to Random Forests:

  1. **Random split thresholds.**  At each node, the split threshold for each
     candidate feature is drawn uniformly at random from the observed range of
     that feature in the current node's sample, rather than being chosen as the
     optimal threshold (as in CART or Random Forest).  This eliminates the
     expensive threshold search and dramatically reduces the computational cost
     of tree building.

  2. **No bootstrap resampling.**  Unlike Random Forests, every Extra Tree in
     the ensemble is trained on the full training set (no bagging).  The
     variance reduction of the ensemble comes entirely from the random split
     thresholds, not from sample diversity.

The combination of these two randomisations produces trees with higher bias
than Random Forest trees but lower variance, and the ensemble average of many
such trees achieves competitive or superior generalisation on most benchmark
datasets.

References
----------
Geurts, P., Ernst, D., & Wehenkel, L. (2006).
    Extremely Randomized Trees. *Machine Learning*, 63(1), 3-42.
Breiman, L. (2001).
    Random Forests. *Machine Learning*, 45(1), 5-32.
Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
    *Classification and Regression Trees*. Wadsworth.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np


# =============================================================================
# Impurity / Criterion Functions
# =============================================================================

def gini_impurity(y: np.ndarray) -> float:
    """
    Gini impurity of a discrete label array.

    Gini(y) = 1 - sum_k p_k^2

    where p_k is the fraction of samples with class label k.  A perfectly
    pure node (all samples from one class) achieves Gini = 0; a maximally
    impure node with K equal classes achieves Gini = 1 - 1/K.

    Args:
        y : Integer class label array of shape ``(n,)``.

    Returns:
        Scalar Gini impurity in [0, 1).
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return float(1.0 - np.sum(probs ** 2))


def entropy_impurity(y: np.ndarray) -> float:
    """
    Shannon entropy impurity of a discrete label array.

    H(y) = -sum_k p_k * log2(p_k)

    A pure node achieves H = 0; a uniform distribution over K classes
    achieves H = log2(K).

    Args:
        y : Integer class label array of shape ``(n,)``.

    Returns:
        Scalar entropy in [0, log2(K)].
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def mse_impurity(y: np.ndarray) -> float:
    """
    Mean Squared Error impurity for regression nodes.

    MSE(y) = Var(y) = (1/n) * sum_i (y_i - mean(y))^2

    A node containing a single unique value achieves MSE = 0.

    Args:
        y : Continuous outcome array of shape ``(n,)``.

    Returns:
        Scalar variance (MSE from the node mean).
    """
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


def mae_impurity(y: np.ndarray) -> float:
    """
    Mean Absolute Deviation impurity for regression nodes.

    MAD(y) = (1/n) * sum_i |y_i - median(y)|

    More robust to outliers than the MSE impurity; appropriate when
    the target distribution has heavy tails.

    Args:
        y : Continuous outcome array of shape ``(n,)``.

    Returns:
        Scalar mean absolute deviation from the median.
    """
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - np.median(y))))


# Registry mapping lowercase name -> callable
CRITERIA: dict = {
    'gini':    gini_impurity,
    'entropy': entropy_impurity,
    'mse':     mse_impurity,
    'mae':     mae_impurity,
}

CRITERION_NAMES: list = list(CRITERIA.keys())


def get_criterion(name: str):
    """
    Return an impurity function by name (case-insensitive).

    Args:
        name : One of ``'gini'``, ``'entropy'``, ``'mse'``, ``'mae'``.

    Returns:
        Callable ``(y: np.ndarray) -> float``.

    Raises:
        ValueError: if *name* is not registered.
    """
    key = name.lower()
    if key not in CRITERIA:
        raise ValueError(
            f"Unknown criterion '{name}'. "
            f"Available: {CRITERION_NAMES}"
        )
    return CRITERIA[key]


# =============================================================================
# Tree Node
# =============================================================================

@dataclass
class ExtraTreeNode:
    """
    A single node in an Extra Tree.

    Internal nodes store a split rule (feature index and threshold) and
    point to left and right children.  Leaf nodes store the prediction
    value: a class probability vector for classification or a scalar mean
    for regression.

    Attributes
    ----------
    feature_idx   : Feature column used for splitting (None at leaves).
    threshold     : Split threshold value (None at leaves).
    left          : Left child node (samples with x[feature] <= threshold).
    right         : Right child node (samples with x[feature] > threshold).
    value         : Leaf prediction.  For regression: scalar float (mean of
                    target values in this leaf).  For classification: array
                    of shape ``(n_classes,)`` with class probabilities.
    n_samples     : Number of training samples that reached this node.
    impurity      : Impurity of this node before splitting.
    impurity_gain : Weighted impurity reduction achieved by the split.
    depth         : Depth of this node in the tree (root = 0).
    node_id       : Unique integer identifier within the tree.
    is_leaf       : Whether this node is a terminal leaf.
    """
    feature_idx:   Optional[int]                  = None
    threshold:     Optional[float]                = None
    left:          Optional['ExtraTreeNode']      = field(default=None, repr=False)
    right:         Optional['ExtraTreeNode']      = field(default=None, repr=False)
    value:         Optional[np.ndarray]           = None
    n_samples:     int                            = 0
    impurity:      float                          = 0.0
    impurity_gain: float                          = 0.0
    depth:         int                            = 0
    node_id:       int                            = -1
    is_leaf:       bool                           = True


# =============================================================================
# Split Criterion
# =============================================================================

class SplitCriterion:
    """
    Evaluates and selects the best random split at a tree node.

    This class implements the core Extra Trees split logic:

      1. Draw ``k`` candidate features uniformly at random (without replacement).
      2. For each candidate feature, draw one threshold uniformly at random
         from the interval [min(x_feat), max(x_feat)] observed in the current
         node's sample.
      3. Evaluate the impurity reduction (information gain) of each
         (feature, threshold) pair.
      4. Return the pair with the highest impurity reduction.

    The key difference from Random Forest is step 2: no search over
    candidate thresholds is performed.  A single random threshold is drawn
    per feature, eliminating the O(n log n) sort-and-scan inner loop of
    CART and reducing split selection from O(k * n * log n) to O(k * n).

    Args:
        criterion       : Impurity function name (default: ``'gini'``).
        max_features    : Number of candidate features to draw at each node.
                          ``None`` uses all features; ``'sqrt'`` uses
                          floor(sqrt(p)); ``'log2'`` uses floor(log2(p)).
                          An integer uses that exact count.
        min_samples_leaf: Minimum samples required in each child node
                          (default: ``1``).
        random_state    : Seed for the random number generator.
    """

    def __init__(
        self,
        criterion:        str = 'gini',
        max_features:     object = 'sqrt',
        min_samples_leaf: int = 1,
        random_state:     Optional[int] = None,
    ) -> None:
        self.criterion_fn     = get_criterion(criterion)
        self.criterion_name   = criterion.lower()
        self.max_features     = max_features
        self.min_samples_leaf = min_samples_leaf
        self._rng             = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    def _n_features(self, p: int) -> int:
        """Resolve the number of candidate features for a given total p."""
        if self.max_features is None:
            return p
        if self.max_features == 'sqrt':
            return max(1, int(math.floor(math.sqrt(p))))
        if self.max_features == 'log2':
            return max(1, int(math.floor(math.log2(p))))
        return max(1, min(int(self.max_features), p))

    # ------------------------------------------------------------------
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best random (feature, threshold) split for this node.

        For each of the ``k`` randomly drawn candidate features, exactly one
        threshold is drawn uniformly from [min(x_feat), max(x_feat)].  The
        split with the highest weighted impurity reduction is returned.

        Args:
            X : Feature matrix of shape ``(n, p)`` for the current node.
            y : Target array of shape ``(n,)``.

        Returns:
            Tuple ``(best_feature, best_threshold, best_gain)`` where
            ``best_feature`` and ``best_threshold`` are ``None`` if no valid
            split exists (e.g. all feature values are constant).
        """
        n, p      = X.shape
        n_feats   = self._n_features(p)
        feat_ids  = self._rng.choice(p, size=n_feats, replace=False)

        parent_impurity = self.criterion_fn(y)
        best_feat       = None
        best_thresh     = None
        best_gain       = -np.inf

        for feat in feat_ids:
            x_col = X[:, feat]
            x_min, x_max = x_col.min(), x_col.max()
            if x_min == x_max:
                continue   # constant feature in this node — skip

            # Draw one uniformly random threshold
            thresh = self._rng.uniform(x_min, x_max)

            left_mask  = x_col <= thresh
            right_mask = ~left_mask
            n_left     = left_mask.sum()
            n_right    = right_mask.sum()

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # Weighted impurity reduction
            imp_left  = self.criterion_fn(y[left_mask])
            imp_right = self.criterion_fn(y[right_mask])
            gain = parent_impurity - (
                (n_left  / n) * imp_left +
                (n_right / n) * imp_right
            )

            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

        return best_feat, best_thresh, best_gain


# =============================================================================
# Feature Importance Accumulator
# =============================================================================

class ImportanceAccumulator:
    """
    Accumulates Mean Decrease in Impurity (MDI) feature importances
    across all trees in an Extra Trees ensemble.

    At each internal node, the impurity reduction weighted by the fraction
    of training samples reaching that node is added to the importance of
    the splitting feature:

        importance[feat] += (n_node / N) * impurity_gain_node

    After all trees are processed, importances are normalised to sum to 1.

    Args:
        n_features : Total number of input features.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self._raw       = np.zeros(n_features, dtype=np.float64)

    def update(self, node: ExtraTreeNode, n_total: int) -> None:
        """
        Recursively walk a tree and accumulate MDI importances.

        Args:
            node    : Root node of the tree (or any subtree).
            n_total : Total number of training samples in the full dataset.
        """
        if node is None or node.is_leaf:
            return
        weight = node.n_samples / max(n_total, 1)
        self._raw[node.feature_idx] += weight * node.impurity_gain
        self.update(node.left,  n_total)
        self.update(node.right, n_total)

    def result(self) -> np.ndarray:
        """
        Return normalised feature importances summing to 1.

        Returns:
            Array of shape ``(n_features,)``.
        """
        total = self._raw.sum()
        if total == 0.0:
            return np.ones(self.n_features) / self.n_features
        return self._raw / total
