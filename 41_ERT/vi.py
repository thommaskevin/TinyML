# vi.py
"""
Regularisation and post-hoc analysis extensions for Extra Trees models.

Contents
--------
- ``cost_complexity_prune``   : Post-hoc cost-complexity (alpha) pruning of a
                                single ExtraTreeModel.
- ``min_impurity_prune``      : Prune subtrees whose root impurity gain is
                                below a threshold.
- ``prediction_variance``     : Tree-level prediction variance (epistemic
                                uncertainty proxy via ensemble disagreement).
- ``permutation_importance``  : Model-agnostic permutation-based feature
                                importance.
- ``partial_dependence``      : Marginalised partial dependence of predictions
                                on a single feature.

Background
----------
Extra Trees do not have an explicit probabilistic or Bayesian interpretation,
but two forms of uncertainty quantification are commonly applied:

  1. **Ensemble variance** (``prediction_variance``).  The variance of
     individual tree predictions around the ensemble mean is a proxy for
     epistemic uncertainty: regions of input space where the trees disagree
     strongly are regions where the model is uncertain due to limited data.

  2. **Permutation feature importance** (``permutation_importance``).  Unlike
     MDI (Mean Decrease in Impurity), which is biased toward high-cardinality
     continuous features, permutation importance evaluates each feature by
     measuring the drop in test-set performance when that feature's values are
     randomly shuffled, breaking its statistical association with the target.

References
----------
Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
Breiman, L., et al. (1984). *Classification and Regression Trees*. Wadsworth.
Strobl, C., et al. (2007). Bias in Random Forest Variable Importance Measures.
    *BMC Bioinformatics*, 8(1), 25.
Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many
    are Useful. *Journal of Machine Learning Research*, 20(177).
"""

import numpy as np
from layers import ExtraTreeNode


# ---------------------------------------------------------------------------
# Cost-complexity pruning (reduced-error / alpha pruning)
# ---------------------------------------------------------------------------

def cost_complexity_prune(
    tree,
    X_val:  np.ndarray,
    y_val:  np.ndarray,
    alpha:  float = 0.01,
) -> None:
    """
    Post-hoc cost-complexity pruning of a fitted ExtraTreeModel.

    A subtree is collapsed to a leaf if the impurity gain at its root,
    normalised by the number of leaves in the subtree, is below *alpha*:

        effective_alpha = impurity_gain / n_leaves_subtree

    Pruning is applied bottom-up (leaves first) so that the tree is always
    reduced to the simplest subtree consistent with the given alpha.

    Args:
        tree  : Fitted ``ExtraTreeModel`` instance.  Modified in place.
        X_val : Validation feature matrix of shape ``(M, p)`` (unused in this
                implementation; included for API symmetry with sklearn).
        y_val : Validation targets of shape ``(M,)`` (unused).
        alpha : Complexity parameter.  Higher alpha prunes more aggressively
                (default: ``0.01``).
    """
    def _n_leaves(node: ExtraTreeNode) -> int:
        if node is None or node.is_leaf:
            return 1
        return _n_leaves(node.left) + _n_leaves(node.right)

    def _prune(node: ExtraTreeNode) -> None:
        if node is None or node.is_leaf:
            return
        _prune(node.left)
        _prune(node.right)
        n_sub = _n_leaves(node.left) + _n_leaves(node.right)
        eff_alpha = node.impurity_gain / max(n_sub, 1)
        if eff_alpha < alpha:
            # Collapse to leaf
            node.is_leaf       = True
            node.left          = None
            node.right         = None
            node.feature_idx   = None
            node.threshold     = None
            node.impurity_gain = 0.0

    if tree.root is not None:
        _prune(tree.root)


# ---------------------------------------------------------------------------
# Minimum impurity gain pruning
# ---------------------------------------------------------------------------

def min_impurity_prune(
    tree,
    min_gain: float = 0.001,
) -> None:
    """
    Prune all subtrees whose root node impurity gain falls below *min_gain*.

    This is a simpler, faster alternative to cost-complexity pruning that
    removes splits that contribute less than *min_gain* units of impurity
    reduction, regardless of the subtree depth or size.

    Args:
        tree     : Fitted ``ExtraTreeModel`` instance.  Modified in place.
        min_gain : Minimum impurity gain threshold (default: ``0.001``).
    """
    def _prune(node: ExtraTreeNode) -> None:
        if node is None or node.is_leaf:
            return
        if node.impurity_gain < min_gain:
            node.is_leaf       = True
            node.left          = None
            node.right         = None
            node.feature_idx   = None
            node.threshold     = None
        else:
            _prune(node.left)
            _prune(node.right)

    if tree.root is not None:
        _prune(tree.root)


# ---------------------------------------------------------------------------
# Prediction variance (epistemic uncertainty proxy)
# ---------------------------------------------------------------------------

def prediction_variance(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute the variance of individual tree predictions across the ensemble.

    For regression, this is the variance of the leaf means:

        Var[y_hat(x)] = (1 / B) * sum_b (y_hat_b(x) - mean_b)^2

    For classification, this is the mean variance of the class-1 probability
    (binary) or the mean entropy of the per-tree probability vectors
    (multi-class), both averaged over classes.

    High variance indicates high epistemic uncertainty: the trees disagree
    about the correct prediction in this region of input space.

    Args:
        model : Fitted ``ExtraTreesModel`` instance.
        X     : Feature matrix of shape ``(M, p)``.

    Returns:
        Array of shape ``(M,)`` with per-sample prediction variance.
    """
    X = np.asarray(X, dtype=np.float64)
    if model.task == 'regression':
        preds = np.stack(
            [t.predict(X) for t in model.estimators_], axis=0
        )   # (B, M)
        return preds.var(axis=0)
    else:
        probas = np.stack(
            [t.predict_proba(X) for t in model.estimators_], axis=0
        )   # (B, M, K)
        return probas.var(axis=0).mean(axis=1)   # mean over classes -> (M,)


# ---------------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------------

def permutation_importance(
    model,
    X:          np.ndarray,
    y:          np.ndarray,
    metric:     str = 'accuracy',
    n_repeats:  int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """
    Compute permutation-based feature importance on a held-out dataset.

    For each feature j, the function randomly permutes the values in column j
    ``n_repeats`` times and measures the average drop in the chosen metric
    relative to the baseline (unpermuted) score.  Features whose permutation
    causes a large performance drop are more important.

    Args:
        model        : Fitted ``ExtraTreesModel`` instance.
        X            : Feature matrix of shape ``(M, p)``.
        y            : Target array of shape ``(M,)``.
        metric       : Name of the evaluation metric (default: ``'accuracy'``
                       for classification; ``'mse'`` for regression).
        n_repeats    : Number of permutation repeats per feature (default: ``10``).
        random_state : Seed for reproducibility (default: ``0``).

    Returns:
        Array of shape ``(p,)`` with mean importance scores (higher = more
        important).  Negative values indicate that permuting the feature
        *improves* performance, suggesting the feature carries noise.
    """
    from losses import compute_loss

    X   = np.asarray(X, dtype=np.float64)
    y   = np.asarray(y)
    rng = np.random.RandomState(random_state)
    p   = X.shape[1]

    baseline_pred  = model.predict(X)
    baseline_score = compute_loss(metric, y, baseline_pred)

    importances = np.zeros((p, n_repeats))
    for j in range(p):
        for r in range(n_repeats):
            X_perm       = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred    = model.predict(X_perm)
            perm_score   = compute_loss(metric, y, perm_pred)
            # For accuracy / R2 (higher=better): drop = baseline - perm
            # For mse / mae (lower=better):      drop = perm - baseline
            if metric in ('accuracy', 'r2', 'f1', 'precision', 'recall'):
                importances[j, r] = baseline_score - perm_score
            else:
                importances[j, r] = perm_score - baseline_score

    return importances.mean(axis=1)


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

def partial_dependence(
    model,
    X:           np.ndarray,
    feature_idx: int,
    n_grid:      int = 50,
) -> tuple:
    """
    Compute the partial dependence of predictions on a single feature.

    The partial dependence function (Friedman, 2001) marginalises out all
    other features by averaging predictions over the empirical distribution
    of the training data:

        PD(x_j) = (1/N) * sum_i  f(x_j, X_{-j}^{(i)})

    where X_{-j}^{(i)} is the i-th training row with feature j replaced by
    the grid value x_j.

    Args:
        model       : Fitted ``ExtraTreesModel`` instance.
        X           : Reference data matrix of shape ``(N, p)``.  Used to
                      define the integration distribution over other features.
        feature_idx : Index of the feature to profile.
        n_grid      : Number of evenly-spaced grid values for the profiled
                      feature (default: ``50``).

    Returns:
        Tuple ``(grid_values, pd_values)`` where both arrays have shape
        ``(n_grid,)``.
    """
    X      = np.asarray(X, dtype=np.float64)
    x_col  = X[:, feature_idx]
    grid   = np.linspace(x_col.min(), x_col.max(), n_grid)
    pd_vals = np.zeros(n_grid)

    for k, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        preds  = model.predict(X_mod)
        pd_vals[k] = float(preds.mean())

    return grid, pd_vals
