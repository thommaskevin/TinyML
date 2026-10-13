# utils.py
"""
Utility functions for Extra Trees (Extremely Randomized Trees) workflows.

Contents
--------
1.  ``export_to_json``           -- Serialise a trained ExtraTreesModel to JSON.
2.  ``train_model``              -- Convenience wrapper around ExtraTreesModel.fit().
3.  ``plot_training_history``    -- Metric vs. number of estimators curve.
4.  ``plot_tree``                -- ASCII / Matplotlib single-tree visualisation.
5.  ``plot_feature_importance``  -- MDI and permutation importance bar charts.
6.  ``plot_decision_boundary``   -- 2-D decision boundary for classification.
7.  ``plot_regression_fit``      -- Predictions vs. true values for regression.
8.  ``plot_partial_dependence``  -- Per-feature partial dependence profiles.
9.  ``plot_prediction_variance`` -- Epistemic uncertainty map over the feature space.
10. ``evaluate_metrics``         -- Full metric report for classification or regression.
"""

import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


# =============================================================================
# 1.  Export
# =============================================================================

def _node_to_dict(node) -> dict:
    """Recursively serialise an ExtraTreeNode to a plain dict."""
    if node is None:
        return None
    d = {
        'node_id':       int(node.node_id),
        'is_leaf':       bool(node.is_leaf),
        'n_samples':     int(node.n_samples),
        'impurity':      round(float(node.impurity), 8),
        'impurity_gain': round(float(node.impurity_gain), 8),
        'depth':         int(node.depth),
    }
    if node.is_leaf:
        d['value'] = [float(v) for v in node.value] if node.value is not None else None
    else:
        d['feature_idx'] = int(node.feature_idx)
        d['threshold']   = round(float(node.threshold), 8)
        d['left']        = _node_to_dict(node.left)
        d['right']       = _node_to_dict(node.right)
    return d


def export_to_json(model, filepath: str) -> None:
    """
    Serialise a trained ``ExtraTreesModel`` to a JSON file.

    The produced JSON contains three top-level keys:

    - ``architecture``  : Hyperparameters (n_estimators, task, criterion, etc.).
    - ``metadata``      : Training statistics (n_features, n_classes, classes).
    - ``trees``         : List of serialised tree structures, one per estimator.
                          Each tree is a nested dict of ExtraTreeNode fields.

    Args:
        model    : A trained ``ExtraTreesModel`` instance.
        filepath : Destination file path.  Parent directories are created
                   automatically.
    """
    arch = {
        'n_estimators':      model.n_estimators,
        'task':              model.task,
        'criterion':         model.criterion,
        'max_depth':         model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf':  model.min_samples_leaf,
        'max_features':      str(model.max_features),
    }
    meta = {
        'n_features': model.n_features_,
        'n_classes':  model.n_classes_,
        'classes':    model.classes_.tolist() if model.classes_ is not None else None,
        'feature_importances': (
            model.feature_importances_.tolist()
            if model.feature_importances_ is not None else None
        ),
    }
    trees = [_node_to_dict(t.root) for t in model.estimators_]

    data = {'architecture': arch, 'metadata': meta, 'trees': trees}

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Model exported -> {filepath}")


# =============================================================================
# 2.  Training wrapper
# =============================================================================

def train_model(
    model,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     Optional[np.ndarray] = None,
    y_val:     Optional[np.ndarray] = None,
    metric:    str = 'accuracy',
    log_every: int = 10,
) -> list:
    """
    Fit an ``ExtraTreesModel`` and optionally track a validation metric
    as the ensemble grows tree by tree.

    Args:
        model     : Unfitted ``ExtraTreesModel`` instance.
        X_train   : Training features of shape ``(N, p)``.
        y_train   : Training targets of shape ``(N,)``.
        X_val     : Optional validation features of shape ``(M, p)``.
        y_val     : Optional validation targets of shape ``(M,)``.
        metric    : Metric name for validation tracking (default: ``'accuracy'``).
        log_every : Logging frequency in number of trees (default: ``10``).

    Returns:
        List of ``(n_trees, metric_value)`` tuples.  Empty when no
        validation data is provided.
    """
    from losses import compute_loss
    from model  import ExtraTreeModel
    from layers import ImportanceAccumulator

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    model.n_features_ = X_train.shape[1]
    if model.task == 'classification':
        model.classes_   = np.unique(y_train.astype(int))
        model.n_classes_ = int(y_train.astype(int).max()) + 1

    accumulator       = ImportanceAccumulator(model.n_features_)
    master_rng        = np.random.RandomState(model.random_state)
    model.estimators_ = []
    history           = []

    for b in range(model.n_estimators):
        seed = int(master_rng.randint(0, 2 ** 31))
        tree = ExtraTreeModel(
            task=model.task,
            criterion=model.criterion,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            max_features=model.max_features,
            random_state=seed,
        )
        tree.fit(X_train, y_train)
        model.estimators_.append(tree)
        accumulator.update(tree.root, len(y_train))

        n_so_far = b + 1
        if X_val is not None and y_val is not None:
            if n_so_far % log_every == 0 or n_so_far == model.n_estimators:
                val_pred   = model.predict(np.asarray(X_val, dtype=np.float64))
                val_metric = compute_loss(metric, y_val, val_pred)
                print(
                    f"  Trees {n_so_far:>4}/{model.n_estimators}"
                    f"  |  {metric.upper()} = {val_metric:.6f}"
                )
                history.append((n_so_far, val_metric))

    model.feature_importances_ = accumulator.result()
    return history


# =============================================================================
# 3.  Training history
# =============================================================================

def plot_training_history(
    history:     list,
    metric_name: str = 'metric',
    title:       str = 'Extra Trees -- Validation Metric vs. Number of Trees',
) -> None:
    """
    Plot a validation metric as a function of the number of trees.

    Args:
        history     : List of ``(n_trees, metric_value)`` tuples from
                      ``train_model``.
        metric_name : Y-axis label (default: ``'metric'``).
        title       : Figure title.
    """
    n_trees, vals = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(n_trees, vals, color='darkviolet', marker='o',
             markersize=4, linewidth=1.5)
    plt.xlabel('Number of Trees')
    plt.ylabel(metric_name.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  Tree visualisation
# =============================================================================

def plot_tree(
    model,
    tree_idx:      int = 0,
    feature_names: Optional[list] = None,
    class_names:   Optional[list] = None,
    max_depth:     int = 4,
    title:         str = 'Extra Tree Structure',
) -> None:
    """
    Visualise a single tree from the Extra Trees ensemble using Matplotlib.

    Renders internal nodes as rectangles showing the split rule and impurity,
    and leaf nodes showing the prediction.  The visualisation is a
    left-to-right binary tree layout with depth limited to *max_depth* for
    readability.

    Args:
        model         : Trained ``ExtraTreesModel`` instance.
        tree_idx      : Index of the tree to display (default: ``0``).
        feature_names : Optional list of feature name strings.
        class_names   : Optional list of class name strings.
        max_depth     : Maximum depth to display (default: ``4``).
        title         : Figure title.
    """
    from layers import ExtraTreeNode

    tree  = model.estimators_[tree_idx]
    root  = tree.root
    names = feature_names or [f'x[{i}]' for i in range(model.n_features_)]

    fig, ax = plt.subplots(figsize=(18, max_depth * 2.5 + 2))
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

    positions: dict = {}   # node_id -> (x, y)

    def _assign_pos(node, depth, x_lo, x_hi):
        if node is None or depth > max_depth:
            return
        x_mid = (x_lo + x_hi) / 2.0
        y     = -depth
        positions[node.node_id] = (x_mid, y)
        if not node.is_leaf and depth < max_depth:
            _assign_pos(node.left,  depth + 1, x_lo, x_mid)
            _assign_pos(node.right, depth + 1, x_mid, x_hi)

    _assign_pos(root, 0, 0.0, 1.0)

    def _draw_node(node, depth):
        if node is None or depth > max_depth:
            return
        pos = positions.get(node.node_id)
        if pos is None:
            return
        x, y = pos

        if node.is_leaf:
            if model.task == 'classification' and node.value is not None:
                cls_idx = int(node.value.argmax())
                lbl_cls = class_names[cls_idx] if class_names else str(cls_idx)
                label = f"Leaf\nclass={lbl_cls}\nn={node.n_samples}"
            else:
                val = node.value[0] if node.value is not None else 0.0
                label = f"Leaf\nval={val:.3f}\nn={node.n_samples}"
            box_color = '#d4edda'
        else:
            feat_name = names[node.feature_idx]
            label = (
                f"{feat_name} <= {node.threshold:.3f}\n"
                f"impurity={node.impurity:.3f}\n"
                f"gain={node.impurity_gain:.3f}  n={node.n_samples}"
            )
            box_color = '#cce5ff'

        ax.text(
            x, y, label,
            ha='center', va='center',
            fontsize=7.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color,
                      edgecolor='#555', linewidth=0.8),
        )

        # Draw edges to children
        if not node.is_leaf and depth < max_depth:
            for child, side in [(node.left, 'True'), (node.right, 'False')]:
                if child is not None and child.node_id in positions:
                    cx, cy = positions[child.node_id]
                    ax.annotate('', xy=(cx, cy + 0.12), xytext=(x, y - 0.12),
                                arrowprops=dict(arrowstyle='->', color='#333',
                                                lw=0.9))
                    mid_x = (x + cx) / 2
                    mid_y = (y + cy) / 2
                    ax.text(mid_x, mid_y, side, fontsize=6.5,
                            color='#555', ha='center')
            _draw_node(node.left,  depth + 1)
            _draw_node(node.right, depth + 1)

    _draw_node(root, 0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-(max_depth + 0.5), 0.5)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Feature importance
# =============================================================================

def plot_feature_importance(
    model,
    X_val:         Optional[np.ndarray] = None,
    y_val:         Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    metric:        str = 'accuracy',
    n_repeats:     int = 10,
    title:         str = 'Extra Trees -- Feature Importance',
) -> None:
    """
    Plot MDI (Mean Decrease in Impurity) and, optionally, permutation
    importance side by side.

    Args:
        model         : Trained ``ExtraTreesModel`` instance.
        X_val         : Validation features for permutation importance.
        y_val         : Validation targets for permutation importance.
        feature_names : Optional list of feature name strings.
        metric        : Metric used for permutation importance.
        n_repeats     : Number of permutation repeats (default: ``10``).
        title         : Figure title.
    """
    from vi import permutation_importance

    p     = model.n_features_
    names = feature_names or [f'Feature {i}' for i in range(p)]
    mdi   = model.feature_importances_
    order = np.argsort(mdi)[::-1]

    do_perm = (X_val is not None and y_val is not None)
    n_panels = 2 if do_perm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels + 1, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    if n_panels == 1:
        axes = [axes]

    # ---- MDI ----
    ax = axes[0]
    ax.bar(range(p), mdi[order], color='darkviolet', alpha=0.75,
           edgecolor='white')
    ax.set_xticks(range(p))
    ax.set_xticklabels([names[i] for i in order], rotation=40, ha='right')
    ax.set_ylabel('MDI Importance')
    ax.set_title('Mean Decrease in Impurity (MDI)')
    ax.grid(True, alpha=0.25, axis='y')

    # ---- Permutation ----
    if do_perm:
        perm = permutation_importance(model, X_val, y_val,
                                      metric=metric, n_repeats=n_repeats)
        perm_order = np.argsort(perm)[::-1]
        ax2 = axes[1]
        colors = ['darkviolet' if v >= 0 else 'tomato' for v in perm[perm_order]]
        ax2.bar(range(p), perm[perm_order], color=colors, alpha=0.75,
                edgecolor='white')
        ax2.set_xticks(range(p))
        ax2.set_xticklabels([names[i] for i in perm_order],
                            rotation=40, ha='right')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel(f'Mean {metric.upper()} drop')
        ax2.set_title('Permutation Importance')
        ax2.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  Decision boundary
# =============================================================================

def plot_decision_boundary(
    model,
    X:      np.ndarray,
    y:      np.ndarray,
    feature_names: Optional[list] = None,
    feat_x: int = 0,
    feat_y: int = 1,
    title:  str = 'Extra Trees -- Decision Boundary',
) -> None:
    """
    Plot the 2-D decision boundary of a classification Extra Trees model.

    When the feature matrix has more than 2 columns, two features are
    selected via *feat_x* and *feat_y*; the remaining features are fixed
    at their training-set medians.

    A dual-panel figure is produced:

    - Left  : Decision boundary + training scatter.
    - Right : Prediction confidence (max class probability) heatmap.

    Args:
        model         : Trained ``ExtraTreesModel`` instance.
        X             : Feature matrix of shape ``(N, p)``.
        y             : Integer class labels of shape ``(N,)``.
        feature_names : Optional list of feature name strings.
        feat_x        : Column index for the x-axis (default: ``0``).
        feat_y        : Column index for the y-axis (default: ``1``).
        title         : Figure title.
    """
    X    = np.asarray(X, dtype=np.float64)
    p    = X.shape[1]
    names = feature_names or [f'x[{i}]' for i in range(p)]

    h      = 0.05
    x_min  = X[:, feat_x].min() - 0.5
    x_max  = X[:, feat_x].max() + 0.5
    y_min  = X[:, feat_y].min() - 0.5
    y_max  = X[:, feat_y].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Build grid with medians for other features
    medians = np.median(X, axis=0)
    grid    = np.tile(medians, (xx.ravel().shape[0], 1))
    grid[:, feat_x] = xx.ravel()
    grid[:, feat_y] = yy.ravel()

    proba = model.predict_proba(grid)
    Z_cls = proba.argmax(axis=1).reshape(xx.shape)
    Z_conf= proba.max(axis=1).reshape(xx.shape)

    unique = np.unique(y.astype(int))
    palette = cm.get_cmap('tab10', len(unique))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left: decision regions
    ax = axes[0]
    ax.contourf(xx, yy, Z_cls, alpha=0.35, cmap='tab10',
                levels=np.arange(-0.5, len(unique) + 0.5))
    ax.contour(xx, yy, Z_cls, colors='k', linewidths=0.4,
               levels=np.arange(-0.5, len(unique) + 0.5))
    for k, lbl in enumerate(unique):
        mask = y.astype(int) == lbl
        ax.scatter(X[mask, feat_x], X[mask, feat_y],
                   c=[palette(k)], s=20, alpha=0.8,
                   edgecolors='k', linewidths=0.3,
                   label=f'Class {lbl}')
    ax.set_xlabel(names[feat_x])
    ax.set_ylabel(names[feat_y])
    ax.set_title('Decision Regions')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Right: confidence
    ax2 = axes[1]
    im = ax2.contourf(xx, yy, Z_conf, levels=20, cmap='plasma', alpha=0.85)
    ax2.scatter(X[:, feat_x], X[:, feat_y], c='white',
                edgecolors='black', s=12, alpha=0.5, linewidths=0.3)
    fig.colorbar(im, ax=ax2, label='Max class probability')
    ax2.set_xlabel(names[feat_x])
    ax2.set_ylabel(names[feat_y])
    ax2.set_title('Prediction Confidence')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 7.  Regression fit
# =============================================================================

def plot_regression_fit(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_idx: int = 0,
    title:   str = 'Extra Trees -- Regression Fit',
) -> None:
    """
    Plot true vs. predicted values and the prediction variance band.

    Args:
        model       : Trained ``ExtraTreesModel`` instance (task='regression').
        X_train     : Training features of shape ``(N_tr, p)``.
        y_train     : Training targets of shape ``(N_tr,)``.
        X_test      : Test features of shape ``(N_te, p)``.
        y_test      : Test targets of shape ``(N_te,)``.
        feature_idx : Feature to use as x-axis (default: ``0``).
        title       : Figure title.
    """
    from vi import prediction_variance

    sort_idx  = np.argsort(X_test[:, feature_idx])
    x_sorted  = X_test[sort_idx, feature_idx]
    y_sorted  = y_test[sort_idx] if hasattr(y_test, '__len__') else y_test
    y_pred    = model.predict(X_test)[sort_idx]
    var_pred  = prediction_variance(model, X_test)[sort_idx]
    std_pred  = np.sqrt(var_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left: fit with uncertainty
    ax = axes[0]
    ax.fill_between(x_sorted,
                    y_pred - 2 * std_pred,
                    y_pred + 2 * std_pred,
                    alpha=0.20, color='mediumorchid', label='+-2 std')
    ax.fill_between(x_sorted,
                    y_pred - std_pred,
                    y_pred + std_pred,
                    alpha=0.40, color='mediumorchid', label='+-1 std')
    ax.plot(x_sorted, y_pred, color='darkviolet', linewidth=2,
            label='Ensemble mean')
    ax.scatter(X_train[:, feature_idx], y_train,
               c='steelblue', s=12, alpha=0.5, label='Train', zorder=5)
    ax.scatter(x_sorted, y_sorted,
               c='tomato', s=12, alpha=0.6, label='Test', zorder=5)
    ax.set_xlabel(f'Feature {feature_idx}')
    ax.set_ylabel('y')
    ax.set_title('Fit with Prediction Variance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Right: predicted vs. actual
    ax2 = axes[1]
    all_y_pred = model.predict(np.vstack([X_train, X_test]))
    all_y_true = np.concatenate([y_train, y_test])
    lo = min(all_y_true.min(), all_y_pred.min()) - 0.5
    hi = max(all_y_true.max(), all_y_pred.max()) + 0.5
    ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=1.0, label='y = y_hat')
    ax2.scatter(y_train, model.predict(X_train),
                c='steelblue', s=14, alpha=0.5, label='Train')
    ax2.scatter(y_test, model.predict(X_test),
                c='tomato', s=14, alpha=0.7, label='Test')
    ax2.set_xlabel('True y')
    ax2.set_ylabel('Predicted y')
    ax2.set_title('Predicted vs. Actual')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 8.  Partial dependence
# =============================================================================

def plot_partial_dependence(
    model,
    X:             np.ndarray,
    feature_names: Optional[list] = None,
    n_grid:        int  = 50,
    max_features:  int  = 6,
    title:         str  = 'Extra Trees -- Partial Dependence Plots',
) -> None:
    """
    Plot the partial dependence of the ensemble predictions on each feature.

    Args:
        model         : Trained ``ExtraTreesModel`` instance.
        X             : Reference data of shape ``(N, p)``.
        feature_names : Optional list of feature name strings.
        n_grid        : Number of grid points per feature (default: ``50``).
        max_features  : Maximum number of features to plot (default: ``6``).
        title         : Figure title.
    """
    from vi import partial_dependence

    p     = model.n_features_
    d     = min(p, max_features)
    names = feature_names or [f'Feature {i}' for i in range(p)]
    cols  = min(d, 3)
    rows  = (d + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    axes_flat = np.array(axes).flatten() if d > 1 else [axes]

    for k in range(d):
        ax   = axes_flat[k]
        grid, pdp = partial_dependence(model, X, feature_idx=k, n_grid=n_grid)
        ax.plot(grid, pdp, color='darkviolet', linewidth=2)
        ax.fill_between(grid, pdp.min(), pdp, alpha=0.15,
                        color='mediumorchid')
        ax.set_xlabel(names[k])
        ax.set_ylabel('Average prediction')
        ax.set_title(f'PDP: {names[k]}')
        ax.grid(True, alpha=0.25)

    for idx in range(d, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 9.  Prediction variance map
# =============================================================================

def plot_prediction_variance(
    model,
    X:      np.ndarray,
    y:      np.ndarray,
    feat_x: int = 0,
    feat_y: int = 1,
    feature_names: Optional[list] = None,
    title:  str = 'Extra Trees -- Epistemic Uncertainty Map',
) -> None:
    """
    Plot ensemble prediction variance as a 2-D heatmap over the feature space.

    High variance regions indicate where the individual trees disagree most,
    i.e. where the model is epistemically uncertain due to sparse training data
    or high feature space complexity.

    Args:
        model         : Trained ``ExtraTreesModel`` instance.
        X             : Feature matrix of shape ``(N, p)``.
        y             : Target array of shape ``(N,)``.
        feat_x        : Feature index for x-axis (default: ``0``).
        feat_y        : Feature index for y-axis (default: ``1``).
        feature_names : Optional list of feature name strings.
        title         : Figure title.
    """
    from vi import prediction_variance

    X     = np.asarray(X, dtype=np.float64)
    names = feature_names or [f'x[{i}]' for i in range(X.shape[1])]
    h     = 0.05
    x_min, x_max = X[:, feat_x].min() - 0.5, X[:, feat_x].max() + 0.5
    y_min, y_max = X[:, feat_y].min() - 0.5, X[:, feat_y].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    medians = np.median(X, axis=0)
    grid    = np.tile(medians, (xx.ravel().shape[0], 1))
    grid[:, feat_x] = xx.ravel()
    grid[:, feat_y] = yy.ravel()

    var = prediction_variance(model, grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(xx, yy, var, levels=25, cmap='plasma', alpha=0.85)
    ax.scatter(X[:, feat_x], X[:, feat_y], c='white',
               edgecolors='black', s=14, alpha=0.6, linewidths=0.4,
               zorder=5, label='Training data')
    fig.colorbar(im, ax=ax, label='Prediction variance')
    ax.set_xlabel(names[feat_x])
    ax.set_ylabel(names[feat_y])
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 10.  Evaluation metrics
# =============================================================================

def evaluate_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task:   Optional[str] = None,
) -> dict:
    """
    Compute a full battery of evaluation metrics for the given task.

    Classification metrics
    ~~~~~~~~~~~~~~~~~~~~~~
    accuracy, log_loss, precision, recall, f1

    Regression metrics
    ~~~~~~~~~~~~~~~~~~
    mse, mae, rmse, r2, mape

    Args:
        model  : Trained ``ExtraTreesModel`` instance.
        X_test : Feature matrix of shape ``(M, p)``.
        y_test : Target array of shape ``(M,)``.
        task   : ``'classification'`` or ``'regression'``.  Inferred from
                 ``model.task`` when ``None`` (default).

    Returns:
        Dictionary mapping metric names to scalar float values.
    """
    from losses import compute_loss

    task    = task or model.task
    X_test  = np.asarray(X_test, dtype=np.float64)
    y_pred  = model.predict(X_test)
    metrics = {}

    if task == 'classification':
        metrics['accuracy']  = compute_loss('accuracy',  y_test, y_pred)
        metrics['precision'] = compute_loss('precision', y_test, y_pred)
        metrics['recall']    = compute_loss('recall',    y_test, y_pred)
        metrics['f1']        = compute_loss('f1',        y_test, y_pred)
        try:
            y_prob = model.predict_proba(X_test)
            metrics['log_loss'] = compute_loss('log_loss', y_test, y_prob)
        except Exception:
            pass
    else:
        metrics['mse']  = compute_loss('mse',  y_test, y_pred)
        metrics['mae']  = compute_loss('mae',  y_test, y_pred)
        metrics['rmse'] = compute_loss('rmse', y_test, y_pred)
        metrics['r2']   = compute_loss('r2',   y_test, y_pred)
        metrics['mape'] = compute_loss('mape', y_test, y_pred)

    return metrics
