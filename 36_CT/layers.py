# layers.py
"""
Layer definitions for the Causal Tree (CT) framework.

Contents
--------
- Split criteria (with registry and factory helper)
- TreeNode         : recursive data structure for a fitted tree
- CausalTreeLayer  : single Causal Tree supporting three task types:
                       'regression'  — continuous outcome Y ∈ ℝ
                       'binary'      — binary outcome Y ∈ {0, 1}
                       'multiclass'  — integer class labels Y ∈ {0,…,K-1}

Background
----------
A Causal Tree (Athey & Imbens, 2016) adapts CART to the causal-inference
setting.  Instead of minimising prediction error on Y, the tree maximises
the *variance of estimated treatment effects* across leaves — i.e. it
partitions X so that units within each leaf have similar treatment effects
while units in different leaves have heterogeneous effects.

The key idea is **honest estimation**: the sample is randomly halved into a
*structure* subset (determines split rules) and an *estimation* subset
(computes leaf-level ATEs).  This prevents over-fitting and enables valid
asymptotic inference on τ(x).

Task adaptations
----------------
Regression   : τ̂(ℓ) = Ȳ(T,ℓ) − Ȳ(C,ℓ)   (difference in means)
Binary       : τ̂(ℓ) = p̂(Y=1|T,ℓ) − p̂(Y=1|C,ℓ)  (risk difference)
Multiclass   : τ̂_k(ℓ) = p̂(Y=k|T,ℓ) − p̂(Y=k|C,ℓ) for k=0,…,K-1
               → predicted class = argmax_k τ̂_k(ℓ)

References
----------
Athey, S., & Imbens, G. (2016).
    Recursive partitioning for heterogeneous causal effects.
    *PNAS*, 113(27), 7353–7360.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# Split Criteria
# =============================================================================

def variance_criterion(
    y: np.ndarray,
    w: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    """
    Causal-tree variance criterion (Athey & Imbens, 2016).

    Scores a split by the weighted sum of squared within-child ATE estimates:

        score = (n_L / n) · τ̂_L² + (n_R / n) · τ̂_R²

    Works for continuous and binary outcomes (τ̂ = difference in means /
    difference in proportions).

    Args:
        y         : Outcome vector of shape ``(n,)``.
        w         : Binary treatment indicator (1 = treated, 0 = control).
        left_mask : Boolean mask selecting the left child.

    Returns:
        Score (higher is better). Returns ``-inf`` when a child has no
        treated or no control unit.
    """
    right_mask = ~left_mask
    n = len(y)
    n_L, n_R = int(left_mask.sum()), int(right_mask.sum())
    if n_L == 0 or n_R == 0:
        return -math.inf

    def _ate(mask):
        yt = y[mask & (w == 1)]; yc = y[mask & (w == 0)]
        if len(yt) == 0 or len(yc) == 0:
            return math.nan
        return float(yt.mean() - yc.mean())

    tau_L, tau_R = _ate(left_mask), _ate(right_mask)
    if math.isnan(tau_L) or math.isnan(tau_R):
        return -math.inf
    return (n_L / n) * tau_L ** 2 + (n_R / n) * tau_R ** 2


def mse_criterion(
    y: np.ndarray,
    w: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    """
    MSE-reduction criterion — weighted within-child outcome variance reduction.

        score = − (n_L · Var(Y_L) + n_R · Var(Y_R))

    Useful as a CART-style baseline; ignores treatment structure.

    Args:
        y         : Outcome vector of shape ``(n,)``.
        w         : Treatment indicator (unused; kept for API consistency).
        left_mask : Boolean mask selecting the left child.

    Returns:
        Negative weighted variance (higher / less negative is better).
    """
    right_mask = ~left_mask
    n_L, n_R = int(left_mask.sum()), int(right_mask.sum())
    if n_L == 0 or n_R == 0:
        return -math.inf
    var_L = float(y[left_mask].var())  if n_L > 1 else 0.0
    var_R = float(y[right_mask].var()) if n_R > 1 else 0.0
    return -(n_L * var_L + n_R * var_R)


def tau_risk_criterion(
    y: np.ndarray,
    w: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    """
    Tau-risk (IPW pseudo-outcome) criterion.

    Constructs the IPW pseudo-outcome:

        Ỹ_i = W_i · Y_i / p̂ − (1 − W_i) · Y_i / (1 − p̂)

    and scores the split by the reduction in Var(Ỹ) across children:

        score = − (n_L · Var(Ỹ_L) + n_R · Var(Ỹ_R))

    Provides a consistent proxy for the oracle CATE MSE under unconfoundedness.

    Args:
        y         : Outcome vector of shape ``(n,)``.
        w         : Binary treatment indicator of shape ``(n,)``.
        left_mask : Boolean mask selecting the left child.

    Returns:
        Negative weighted pseudo-outcome variance (higher is better).
        Returns ``-inf`` when propensity is 0 or 1 in either child.
    """
    right_mask = ~left_mask
    n_L, n_R = int(left_mask.sum()), int(right_mask.sum())
    if n_L == 0 or n_R == 0:
        return -math.inf

    def _var_pseudo(mask):
        y_c, w_c = y[mask], w[mask]
        p = float(w_c.mean())
        if p <= 0.0 or p >= 1.0:
            return math.nan
        pseudo = w_c * y_c / p - (1.0 - w_c) * y_c / (1.0 - p)
        return float(pseudo.var()) if len(pseudo) > 1 else 0.0

    v_L, v_R = _var_pseudo(left_mask), _var_pseudo(right_mask)
    if math.isnan(v_L) or math.isnan(v_R):
        return -math.inf
    return -(n_L * v_L + n_R * v_R)


# Registry
CRITERIA: dict = {
    'variance': variance_criterion,
    'mse':      mse_criterion,
    'tau_risk': tau_risk_criterion,
}


def get_criterion(name: str):
    """
    Return a split-criterion function by name (case-insensitive).

    Args:
        name : One of ``'variance'``, ``'mse'``, or ``'tau_risk'``.

    Returns:
        Callable ``(y, w, left_mask) → float``.

    Raises:
        ValueError: if *name* is not in the registry.
    """
    key = name.lower()
    if key not in CRITERIA:
        raise ValueError(
            f"Unknown criterion '{name}'. "
            f"Available: {sorted(CRITERIA)}"
        )
    return CRITERIA[key]


# =============================================================================
# Tree Node
# =============================================================================

@dataclass
class TreeNode:
    """
    Single node in a fitted Causal Tree.

    Leaf nodes store the treatment-effect estimate ``tau`` (scalar for
    regression/binary, array of shape ``(K,)`` for multiclass) and
    sample counts.  Internal nodes store the split rule and child pointers.

    Attributes:
        feature   : Column index of the splitting feature (internal only).
        threshold : Split threshold; left child receives X[:, feature] ≤ threshold.
        left      : Left child ``TreeNode`` (internal only).
        right     : Right child ``TreeNode`` (internal only).
        tau       : Leaf-level CATE estimate.
                    - Regression / binary : float  (τ̂ = Ȳ_T − Ȳ_C)
                    - Multiclass          : ndarray of shape ``(K,)``
                      (τ̂_k = p̂(Y=k|T) − p̂(Y=k|C) for each class k;
                       predict = argmax_k τ̂_k)
        n_treated : Number of treated estimation-sample units in this leaf.
        n_control : Number of control estimation-sample units in this leaf.
        depth     : Depth of this node (root = 0).
        node_id   : Integer identifier assigned during fitting.
        impurity  : Split score at this internal node.
    """
    feature:   Optional[int]               = None
    threshold: Optional[float]             = None
    left:      Optional['TreeNode']        = None
    right:     Optional['TreeNode']        = None
    tau:       Optional[object]            = None   # float or ndarray
    n_treated: int                         = 0
    n_control: int                         = 0
    depth:     int                         = 0
    node_id:   int                         = 0
    impurity:  float                       = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def n_samples(self) -> int:
        return self.n_treated + self.n_control


# =============================================================================
# Causal Tree Layer
# =============================================================================

class CausalTreeLayer:
    """
    Single Causal Tree supporting regression, binary, and multiclass tasks.

    The tree uses **honest estimation**: the training data is randomly
    split 50/50 into a *structure* half (determines splits) and an
    *estimation* half (computes leaf-level CATE estimates).

    Task-specific leaf estimates
    ----------------------------
    - ``'regression'``  : τ̂ = mean(Y | T, leaf) − mean(Y | C, leaf)
    - ``'binary'``      : τ̂ = P(Y=1 | T, leaf) − P(Y=1 | C, leaf)
    - ``'multiclass'``  : τ̂_k = P(Y=k | T, leaf) − P(Y=k | C, leaf)
                          for k = 0, …, K−1.  Prediction = argmax_k τ̂_k.

    Args:
        task               : One of ``'regression'``, ``'binary'``, or
                             ``'multiclass'`` (default: ``'regression'``).
        max_depth          : Maximum tree depth (default: ``5``).
        min_samples_leaf   : Minimum samples per leaf on the structure half
                             (default: ``20``).
        min_samples_treat  : Minimum treated units per child for a valid
                             split (default: ``5``).
        criterion          : Split-criterion name — ``'variance'``,
                             ``'mse'``, or ``'tau_risk'``
                             (default: ``'variance'``).
        honest             : Use honest sample splitting (default: ``True``).
        n_features         : Features to consider per split.
                             ``None`` = all; ``'sqrt'`` = √p;
                             ``'log2'`` = log₂p; integer = exact count
                             (default: ``None``).
        n_classes          : Number of classes (required only when
                             ``task='multiclass'``; default: ``None``).
        random_state       : Random seed (default: ``42``).
    """

    def __init__(
        self,
        task:              str   = 'regression',
        max_depth:         int   = 5,
        min_samples_leaf:  int   = 20,
        min_samples_treat: int   = 5,
        criterion:         str   = 'variance',
        honest:            bool  = True,
        n_features               = None,
        n_classes:         Optional[int] = None,
        random_state:      int   = 42,
    ) -> None:
        self.task              = task.lower()
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.min_samples_treat = min_samples_treat
        self.criterion_name    = criterion.lower()
        self.honest            = honest
        self.n_features        = n_features
        self.n_classes         = n_classes
        self.random_state      = random_state

        self._criterion_fn     = get_criterion(criterion)
        self.root_: Optional[TreeNode] = None
        self.n_features_in_: Optional[int] = None
        self._node_counter: int = 0

        if self.task not in ('regression', 'binary', 'multiclass'):
            raise ValueError(
                f"Unknown task '{task}'. "
                "Choose 'regression', 'binary', or 'multiclass'."
            )

    # ------------------------------------------------------------------
    def _resolve_n_features(self, p: int) -> int:
        if self.n_features is None:
            return p
        if self.n_features == 'sqrt':
            return max(1, int(math.sqrt(p)))
        if self.n_features == 'log2':
            return max(1, int(math.log2(p)))
        return int(self.n_features)

    # ------------------------------------------------------------------
    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, float, float]:
        """Find the best (feature, threshold) split on the structure sample."""
        n, p = X.shape
        n_try = self._resolve_n_features(p)
        feats = rng.choice(p, size=min(n_try, p), replace=False)

        best_feat, best_thr, best_score = -1, 0.0, -math.inf

        # For multiclass tasks the criterion operates on class-0 probability
        # or on the raw integer labels — both work for variance/tau_risk.
        y_score = y if self.task != 'multiclass' else y.astype(float)

        for feat in feats:
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thr in thresholds:
                lm = X[:, feat] <= thr
                score = self._criterion_fn(y_score, w, lm)
                if score > best_score:
                    best_score = score
                    best_feat  = feat
                    best_thr   = thr

        return best_feat, best_thr, best_score

    # ------------------------------------------------------------------
    def _leaf_estimate(
        self,
        y: np.ndarray,
        w: np.ndarray,
    ) -> TreeNode:
        """
        Compute the honest CATE estimate for a leaf.

        Returns a ``TreeNode`` with ``tau``, ``n_treated``, ``n_control`` set.
        """
        t_mask = w == 1
        c_mask = w == 0
        n_t = int(t_mask.sum())
        n_c = int(c_mask.sum())

        if self.task == 'regression' or self.task == 'binary':
            if n_t == 0 or n_c == 0:
                tau = float(y.mean()) if len(y) > 0 else 0.0
            else:
                tau = float(y[t_mask].mean() - y[c_mask].mean())

        else:  # multiclass
            K = self.n_classes or int(y.max()) + 1
            if n_t == 0 or n_c == 0:
                # fallback: empirical class distribution
                counts = np.bincount(y.astype(int), minlength=K)
                tau = counts / counts.sum() if counts.sum() > 0 else np.ones(K) / K
            else:
                p_t = np.bincount(y[t_mask].astype(int), minlength=K) / n_t
                p_c = np.bincount(y[c_mask].astype(int), minlength=K) / n_c
                tau = p_t - p_c  # shape (K,)

        node = TreeNode(tau=tau, n_treated=n_t, n_control=n_c)
        node.node_id = self._node_counter
        self._node_counter += 1
        return node

    # ------------------------------------------------------------------
    def _grow(
        self,
        X_s: np.ndarray, y_s: np.ndarray, w_s: np.ndarray,
        X_e: np.ndarray, y_e: np.ndarray, w_e: np.ndarray,
        depth: int,
        rng:   np.random.Generator,
    ) -> TreeNode:
        """Recursively grow the tree (structure sample drives splits)."""
        n_s = len(y_s)
        stop = (
            depth >= self.max_depth
            or n_s < self.min_samples_leaf
            or (w_s == 1).sum() < self.min_samples_treat
            or (w_s == 0).sum() < self.min_samples_treat
        )
        if stop:
            node = self._leaf_estimate(y_e, w_e)
            node.depth = depth
            return node

        feat, thr, score = self._best_split(X_s, y_s, w_s, rng)
        if feat == -1:
            node = self._leaf_estimate(y_e, w_e)
            node.depth = depth
            return node

        lm_s = X_s[:, feat] <= thr
        rm_s = ~lm_s

        # validate treatment balance in each child
        for mask in (lm_s, rm_s):
            if (mask.sum() < self.min_samples_leaf
                    or (w_s[mask] == 1).sum() < 1
                    or (w_s[mask] == 0).sum() < 1):
                node = self._leaf_estimate(y_e, w_e)
                node.depth = depth
                return node

        lm_e = X_e[:, feat] <= thr
        rm_e = ~lm_e

        left  = self._grow(X_s[lm_s], y_s[lm_s], w_s[lm_s],
                           X_e[lm_e], y_e[lm_e], w_e[lm_e],
                           depth + 1, rng)
        right = self._grow(X_s[rm_s], y_s[rm_s], w_s[rm_s],
                           X_e[rm_e], y_e[rm_e], w_e[rm_e],
                           depth + 1, rng)

        node = TreeNode(feature=feat, threshold=thr,
                        left=left, right=right,
                        depth=depth, impurity=score)
        node.node_id = self._node_counter
        self._node_counter += 1
        return node

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> 'CausalTreeLayer':
        """
        Fit the Causal Tree.

        Args:
            X : Covariate matrix of shape ``(n, p)``.
            y : Outcome vector of shape ``(n,)``.
                - Regression  : continuous float
                - Binary      : 0 or 1
                - Multiclass  : integer class labels 0 … K−1
            w : Binary treatment indicator of shape ``(n,)`` (1=treated).

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64 if self.task == 'regression'
                       else np.int32)
        w = np.asarray(w, dtype=np.int32)

        n, p = X.shape
        self.n_features_in_ = p
        self._node_counter  = 0

        if self.task == 'multiclass' and self.n_classes is None:
            self.n_classes = int(y.max()) + 1

        rng = np.random.default_rng(self.random_state)
        if self.honest:
            idx = rng.permutation(n)
            mid = n // 2
            s_idx, e_idx = idx[:mid], idx[mid:]
        else:
            s_idx = e_idx = np.arange(n)

        self.root_ = self._grow(
            X[s_idx], y[s_idx], w[s_idx],
            X[e_idx], y[e_idx], w[e_idx],
            depth=0, rng=rng,
        )
        return self

    # ------------------------------------------------------------------
    def _route(self, x: np.ndarray, node: TreeNode):
        """Route a single sample to a leaf and return its tau."""
        if node.is_leaf:
            return node.tau
        return (self._route(x, node.left)
                if x[node.feature] <= node.threshold
                else self._route(x, node.right))

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict treatment effects (or class labels for multiclass).

        Args:
            X : Covariate matrix of shape ``(n, p)``.

        Returns:
            - Regression / binary : array of shape ``(n,)`` with τ̂(x).
            - Multiclass          : array of shape ``(n,)`` with predicted
                                    class labels (argmax of τ̂_k).
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float64)

        if self.task == 'multiclass':
            taus = np.array([self._route(x, self.root_) for x in X])
            return np.argmax(taus, axis=1).astype(int)
        return np.array([self._route(x, self.root_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-class treatment-effect differences for multiclass tasks.

        Args:
            X : Covariate matrix of shape ``(n, p)``.

        Returns:
            Array of shape ``(n, K)`` with τ̂_k(x) for each class k.

        Raises:
            RuntimeError: if ``task`` is not ``'multiclass'``.
        """
        if self.task != 'multiclass':
            raise RuntimeError("predict_proba() is only available for 'multiclass'.")
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._route(x, self.root_) for x in X])

    # ------------------------------------------------------------------
    def get_leaf_nodes(self) -> list[TreeNode]:
        """Return all leaf nodes in depth-first order."""
        leaves: list[TreeNode] = []
        def _collect(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                _collect(node.left); _collect(node.right)
        if self.root_ is not None:
            _collect(self.root_)
        return leaves

    def get_depth(self) -> int:
        """Return the actual depth of the fitted tree."""
        def _d(node):
            if node is None or node.is_leaf: return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root_)

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes."""
        return len(self.get_leaf_nodes())
