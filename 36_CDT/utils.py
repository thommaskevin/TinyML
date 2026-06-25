# utils.py
"""
Utility functions for Causal Tree (CT) training and analysis.

Contents
--------
1.  ``train_causal_tree``           — Fit + evaluate for any task.
2.  ``export_to_json``              — Serialise fitted tree to JSON.
3.  ``plot_tree``                   — Matplotlib tree diagram.
4.  ``print_tree``                  — ASCII tree (console).
5.  ``plot_cate_distribution``      — CATE histogram (regression/binary).
6.  ``plot_treatment_effect_by_feature`` — CATE vs one covariate.
7.  ``plot_decision_boundary``      — 2-D CATE heatmap.
8.  ``plot_leaf_effects``           — Per-leaf τ̂ dot-plot with bootstrap CIs.
9.  ``plot_multiclass_effects``     — Per-class per-leaf τ̂_k heatmap.
10. ``plot_training_history``       — Metric curve over a hyperparameter sweep.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from layers import CausalTreeLayer, TreeNode
from losses import compute_metric


# =============================================================================
# 1. Training loop
# =============================================================================

def train_causal_tree(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val:   Optional[np.ndarray] = None,
    y_val:   Optional[np.ndarray] = None,
    w_val:   Optional[np.ndarray] = None,
    metric:  str   = 'tau_risk',
    verbose: bool  = True,
) -> list[tuple[str, float]]:
    """
    Fit a ``CausalTreeModel`` and evaluate on train (and optionally val) sets.

    Args:
        model   : Unfitted ``CausalTreeModel``.
        X_train : Training covariates ``(n, p)``.
        y_train : Training outcomes ``(n,)``.
        w_train : Training treatment indicators ``(n,)``.
        X_val   : Validation covariates (optional).
        y_val   : Validation outcomes (optional).
        w_val   : Validation treatment indicators (optional).
        metric  : Metric name from ``losses.METRIC_NAMES``.
        verbose : Print results if ``True``.

    Returns:
        List of ``('train' | 'val', metric_value)`` tuples.
    """
    model.fit(X_train, y_train, w_train)

    def _eval(X, y, w, split_name):
        tau = model.predict(X)
        val = compute_metric(metric, tau, y=y, w=w)
        if verbose:
            print(f"  [{split_name:5s}] {metric.upper()} = {val:.6f}")
        return (split_name, val)

    history = [_eval(X_train, y_train, w_train, 'train')]
    if X_val is not None and y_val is not None and w_val is not None:
        history.append(_eval(X_val, y_val, w_val, 'val'))

    if verbose:
        print(f"  depth={model.get_depth()}  leaves={model.get_n_leaves()}")
    return history


# =============================================================================
# 2. Export
# =============================================================================

def export_to_json(model, filepath: str) -> None:
    """
    Serialise a fitted ``CausalTreeModel`` to JSON.

    The JSON contains ``model_type``, ``task``, ``hyperparams``, and a
    recursive ``tree`` node structure.  Multiclass leaf ``tau`` arrays
    are stored as lists.

    Args:
        model    : Fitted ``CausalTreeModel``.
        filepath : Destination path (parent dirs created automatically).
    """
    def _node(node: Optional[TreeNode]) -> Optional[dict]:
        if node is None:
            return None
        d = {
            'node_id':   int(node.node_id),
            'depth':     int(node.depth),
            'is_leaf':   bool(node.is_leaf),
            'impurity':  float(node.impurity),
            'n_treated': int(node.n_treated),
            'n_control': int(node.n_control),
        }
        if node.is_leaf:
            tau = node.tau
            d['tau'] = (tau.tolist() if isinstance(tau, np.ndarray)
                        else float(tau) if tau is not None else None)
        else:
            d['feature']   = int(node.feature)
            d['threshold'] = float(node.threshold)
            d['left']      = _node(node.left)
            d['right']     = _node(node.right)
        return d

    cfg = model.tree_config
    data = {
        'model_type': 'CausalTree',
        'task':       model.task,
        'hyperparams': {
            'max_depth':         int(model._tree.max_depth),
            'min_samples_leaf':  int(model._tree.min_samples_leaf),
            'min_samples_treat': int(model._tree.min_samples_treat),
            'criterion':         model._tree.criterion_name,
            'honest':            bool(model._tree.honest),
            'n_features':        str(model._tree.n_features),
            'n_classes':         model._tree.n_classes,
        },
        'tree': _node(model._tree.root_),
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Model exported → {filepath}")


# =============================================================================
# 3. Matplotlib Tree Diagram
# =============================================================================

def plot_tree(
    model,
    feature_names: Optional[list] = None,
    class_names:   Optional[list] = None,
    title:         str   = 'Causal Tree',
    figsize:       tuple = (16, 9),
    fontsize:      int   = 8,
) -> None:
    """
    Matplotlib top-down diagram of a fitted Causal Tree.

    - **Internal nodes** — rounded rectangles showing the split rule.
    - **Leaf nodes** — coloured ellipses showing τ̂ (diverging red/blue
      for regression/binary; categorical colour for multiclass) plus
      sample counts T / C.

    Args:
        model         : Fitted ``CausalTreeModel``.
        feature_names : Feature name list; defaults to ``['X0','X1',…]``.
        class_names   : Class name list (multiclass only).
        title         : Figure title.
        figsize       : Figure size.
        fontsize      : Base font size.
    """
    tree_layer = model._tree
    root = tree_layer.root_
    if root is None:
        raise RuntimeError("Tree not fitted.")

    task = model.task
    p    = tree_layer.n_features_in_ or 10
    feat_names  = feature_names or [f'X{i}' for i in range(p)]

    # colour setup
    if task == 'multiclass':
        K = tree_layer.n_classes or 3
        c_names = class_names or [f'C{k}' for k in range(K)]
        cmap_leaf = plt.cm.get_cmap('tab10', K)
    else:
        leaves   = tree_layer.get_leaf_nodes()
        tau_vals = [float(lf.tau) for lf in leaves if lf.tau is not None]
        vmax     = max(abs(min(tau_vals)), abs(max(tau_vals))) if tau_vals else 1.0
        norm_tau = Normalize(vmin=-vmax, vmax=vmax)
        cmap_leaf = plt.cm.RdBu_r

    # ── layout: assign x by leaf order, y by depth ──
    positions: dict[int, tuple[float, float]] = {}
    _lc = [0]

    def _layout(node: TreeNode) -> float:
        if node.is_leaf:
            x = float(_lc[0]); _lc[0] += 1
            positions[node.node_id] = (x, -float(node.depth))
            return x
        xl = _layout(node.left); xr = _layout(node.right)
        x  = (xl + xr) / 2.0
        positions[node.node_id] = (x, -float(node.depth))
        return x

    _layout(root)

    xs = [v[0] for v in positions.values()]
    ys = [v[1] for v in positions.values()]
    xr = max(xs) - min(xs) or 1.0
    yr = max(ys) - min(ys) or 1.0

    def _npos(nid):
        x, y = positions[nid]
        return (x - min(xs)) / xr, (y - min(ys)) / yr

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.1, 1.1); ax.axis('off')
    ax.set_title(title, fontsize=fontsize + 4, fontweight='bold', pad=14)

    def _draw(node: TreeNode) -> None:
        nx, ny = _npos(node.node_id)

        if node.is_leaf:
            if task == 'multiclass':
                k = int(np.argmax(node.tau))
                color = cmap_leaf(k)
                lbl   = f'{c_names[k]}\nT={node.n_treated} C={node.n_control}'
            else:
                tau_f = float(node.tau) if node.tau is not None else 0.0
                color = cmap_leaf(norm_tau(tau_f))
                lbl   = f'τ̂={tau_f:+.3f}\nT={node.n_treated} C={node.n_control}'
            r, g, b, _ = color
            txt_col = 'white' if 0.299*r + 0.587*g + 0.114*b < 0.5 else 'black'
            ell = mpatches.Ellipse((nx, ny), .08, .06, facecolor=color,
                                   edgecolor='black', lw=0.8, zorder=3)
            ax.add_patch(ell)
            ax.text(nx, ny, lbl, ha='center', va='center',
                    fontsize=fontsize - 1, color=txt_col,
                    fontweight='bold', multialignment='center', zorder=4)
        else:
            fn = feat_names[node.feature] if node.feature < len(feat_names) else f'X{node.feature}'
            lbl = f'{fn}\n≤ {node.threshold:.3f}'
            ax.text(nx, ny, lbl, ha='center', va='center', fontsize=fontsize,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#EEE8F5',
                              ec='#7B2D8B', lw=1.0), zorder=3,
                    multialignment='center')
            for child, edge_lbl in ((node.left, '≤'), (node.right, '>')):
                cx, cy = _npos(child.node_id)
                ax.annotate('', xy=(cx, cy + .03), xytext=(nx, ny - .03),
                            arrowprops=dict(arrowstyle='->', color='#666',
                                            lw=1.0), zorder=2)
                ax.text((nx+cx)/2, (ny+cy)/2 + .01, edge_lbl,
                        ha='center', va='bottom',
                        fontsize=fontsize - 1, color='#444')
            _draw(node.left); _draw(node.right)

    _draw(root)

    # colourbar / legend
    if task == 'multiclass':
        handles = [mpatches.Patch(color=cmap_leaf(k), label=c_names[k])
                   for k in range(K)]
        ax.legend(handles=handles, title='Predicted class',
                  loc='lower right', fontsize=fontsize)
    else:
        sm = ScalarMappable(cmap=cmap_leaf, norm=norm_tau); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=.025, pad=.01,
                     label='Estimated CATE  τ̂')

    info = (f'Depth: {tree_layer.get_depth()}  |  '
            f'Leaves: {tree_layer.get_n_leaves()}  |  '
            f'Task: {task}  |  Criterion: {tree_layer.criterion_name}')
    ax.text(0.5, -0.04, info, ha='center', fontsize=fontsize,
            color='gray', transform=ax.transAxes)
    plt.tight_layout(); plt.show()


# =============================================================================
# 4. ASCII Tree
# =============================================================================

def print_tree(
    model,
    feature_names: Optional[list] = None,
    class_names:   Optional[list] = None,
    max_depth:     Optional[int]  = None,
) -> None:
    """
    Print an ASCII representation of the fitted tree.

    Args:
        model         : Fitted ``CausalTreeModel``.
        feature_names : Optional feature names.
        class_names   : Optional class names (multiclass).
        max_depth     : Maximum depth to print.
    """
    root  = model._tree.root_
    p     = model._tree.n_features_in_ or 0
    task  = model.task
    names = feature_names or [f'X{i}' for i in range(p)]
    K     = model._tree.n_classes
    cnames = class_names or ([f'C{k}' for k in range(K)] if K else [])

    def _print(node: TreeNode, prefix: str, is_left: bool) -> None:
        if max_depth is not None and node.depth > max_depth:
            return
        conn = '├── ' if is_left else '└── '
        if node.is_leaf:
            if task == 'multiclass':
                k   = int(np.argmax(node.tau))
                tau_str = f'class={cnames[k]}  τ̂={np.array2string(node.tau, precision=3)}'
            else:
                tau_str = f'τ̂={float(node.tau):+.4f}'
            print(f'{prefix}{conn}[LEAF] {tau_str}  T={node.n_treated} C={node.n_control}')
        else:
            fn = names[node.feature] if node.feature < len(names) else f'X{node.feature}'
            print(f'{prefix}{conn}{fn} ≤ {node.threshold:.4f}  (score={node.impurity:.4f})')
            cp = prefix + ('│   ' if is_left else '    ')
            _print(node.left,  cp, True)
            _print(node.right, cp, False)

    print(f'\nCausal Tree  [{task}]  depth={model.get_depth()}  leaves={model.get_n_leaves()}')
    print('─' * 64)
    if root is not None:
        if root.is_leaf:
            tau_str = (f'class={cnames[int(np.argmax(root.tau))]}' if task == 'multiclass'
                       else f'τ̂={float(root.tau):+.4f}')
            print(f'[ROOT/LEAF] {tau_str}  T={root.n_treated} C={root.n_control}')
        else:
            fn = names[root.feature] if root.feature < len(names) else f'X{root.feature}'
            print(f'[ROOT] {fn} ≤ {root.threshold:.4f}  (score={root.impurity:.4f})')
            _print(root.left,  '', True)
            _print(root.right, '', False)
    print('─' * 64)


# =============================================================================
# 5. CATE Distribution (regression / binary)
# =============================================================================

def plot_cate_distribution(
    tau_hat:  np.ndarray,
    title:    str = 'Distribution of Estimated CATE',
    true_ate: Optional[float] = None,
    bins:     int = 40,
    task:     str = 'regression',
) -> None:
    """
    Histogram of estimated CATE values.

    Args:
        tau_hat  : CATE estimates of shape ``(n,)``.
        title    : Figure title.
        true_ate : If provided, marks the true ATE as a vertical line.
        bins     : Histogram bins (default: 40).
        task     : ``'regression'`` or ``'binary'``.
    """
    xlabel = 'τ̂(x) — Estimated Risk Difference' if task == 'binary' else 'τ̂(x) — Estimated CATE'
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(tau_hat, bins=bins, color='mediumorchid', edgecolor='white',
            alpha=0.85, label='τ̂(x)')
    ax.axvline(float(tau_hat.mean()), color='darkviolet', lw=2, ls='--',
               label=f'Mean τ̂ = {tau_hat.mean():.3f}')
    if true_ate is not None:
        ax.axvline(true_ate, color='tomato', lw=2,
                   label=f'True ATE = {true_ate:.3f}')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# =============================================================================
# 6. CATE vs Feature
# =============================================================================

def plot_treatment_effect_by_feature(
    X:            np.ndarray,
    tau_hat:      np.ndarray,
    feature_idx:  int,
    feature_name: str = 'X',
    w:            Optional[np.ndarray] = None,
    title:        str = 'Treatment Effect by Feature',
    n_bins:       int = 20,
    task:         str = 'regression',
) -> None:
    """
    Scatter + binned-mean of τ̂ vs a single covariate.

    Args:
        X            : Covariate matrix ``(n, p)``.
        tau_hat      : CATE estimates ``(n,)``.
        feature_idx  : Column to plot on the x-axis.
        feature_name : Display name for the feature.
        w            : Treatment indicator for colour-coding (optional).
        title        : Figure title.
        n_bins       : Number of bins for the smoothed mean (default: 20).
        task         : Task type (affects y-axis label).
    """
    x_feat = X[:, feature_idx]
    ylabel = ('Risk Difference τ̂' if task == 'binary'
              else ('Predicted Class τ̂' if task == 'multiclass' else 'CATE τ̂'))
    fig, ax = plt.subplots(figsize=(10, 5))
    if w is not None:
        colors = np.where(w == 1, '#2E86AB', '#E84855')
        ax.scatter(x_feat, tau_hat, c=colors, alpha=0.4, s=18, zorder=2)
        ax.legend(handles=[
            mpatches.Patch(color='#2E86AB', label='Treated'),
            mpatches.Patch(color='#E84855', label='Control'),
        ], loc='upper right')
    else:
        ax.scatter(x_feat, tau_hat, color='mediumorchid', alpha=0.4, s=18, zorder=2)
    bins  = np.quantile(x_feat, np.linspace(0, 1, n_bins + 1))
    xm, ym = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x_feat >= lo) & (x_feat <= hi)
        if mask.sum() > 0:
            xm.append(x_feat[mask].mean()); ym.append(tau_hat[mask].mean())
    ax.plot(xm, ym, color='darkviolet', lw=2.5, label='Binned mean τ̂', zorder=3)
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.set_xlabel(feature_name, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


# =============================================================================
# 7. Decision Boundary / CATE Heatmap
# =============================================================================

def plot_decision_boundary(
    model,
    X:          np.ndarray,
    w:          np.ndarray,
    feat_x:     int = 0,
    feat_y:     int = 1,
    feat_names: Optional[list] = None,
    class_names: Optional[list] = None,
    title:      str = 'CATE Heatmap',
    grid_n:     int = 80,
) -> None:
    """
    2-D heatmap of the estimated CATE / predicted class for two features.

    All other features are held at their median values.  For multiclass,
    the left panel shows the predicted class and the right panel the
    policy boundary.

    Args:
        model       : Fitted ``CausalTreeModel``.
        X           : Covariate matrix ``(n, p)``.
        w           : Treatment indicator for scatter overlay.
        feat_x      : Column for the horizontal axis.
        feat_y      : Column for the vertical axis.
        feat_names  : Feature names list.
        class_names : Class names (multiclass).
        title       : Figure title.
        grid_n      : Grid resolution (default: 80).
    """
    task   = model.task
    names  = feat_names or [f'X{i}' for i in range(X.shape[1])]
    K      = model._tree.n_classes
    cnames = class_names or ([f'C{k}' for k in range(K)] if K else [])

    xr = np.linspace(X[:, feat_x].min(), X[:, feat_x].max(), grid_n)
    yr = np.linspace(X[:, feat_y].min(), X[:, feat_y].max(), grid_n)
    xx, yy = np.meshgrid(xr, yr)

    medians = np.median(X, axis=0)
    X_grid = np.tile(medians, (grid_n * grid_n, 1))
    X_grid[:, feat_x] = xx.ravel(); X_grid[:, feat_y] = yy.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    def _scatter(ax):
        ax.scatter(X[w==1, feat_x], X[w==1, feat_y],
                   c='navy', s=12, alpha=0.5, label='Treated', zorder=3)
        ax.scatter(X[w==0, feat_x], X[w==0, feat_y],
                   c='firebrick', s=12, alpha=0.5, marker='^',
                   label='Control', zorder=3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel(names[feat_x]); ax.set_ylabel(names[feat_y])

    if task == 'multiclass':
        preds = model.predict(X_grid).reshape(grid_n, grid_n)
        cmap_mc = plt.cm.get_cmap('tab10', K)
        axes[0].contourf(xx, yy, preds, levels=np.arange(-0.5, K+0.5, 1),
                         cmap=cmap_mc, alpha=0.7)
        axes[0].set_title('Predicted Class (τ̂ argmax)')
        _scatter(axes[0])
        handles = [mpatches.Patch(color=cmap_mc(k), label=cnames[k]) for k in range(K)]
        axes[0].legend(handles=handles, fontsize=8)

        axes[1].contourf(xx, yy, preds, levels=np.arange(-0.5, K+0.5, 1),
                         cmap=cmap_mc, alpha=0.4)
        for k in range(K):
            axes[1].contour(xx, yy, (preds == k).astype(float),
                            levels=[0.5], colors='black', linewidths=1.0)
        axes[1].set_title('Policy Boundary (treat → predicted best class)')
        _scatter(axes[1])
    else:
        tau_grid = model.predict(X_grid).reshape(grid_n, grid_n)
        vmax = float(np.abs(tau_grid).max())
        im = axes[0].contourf(xx, yy, tau_grid, levels=30, cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, alpha=0.85)
        fig.colorbar(im, ax=axes[0], label='τ̂(x)')
        axes[0].set_title('CATE Surface τ̂(x)')
        _scatter(axes[0])

        policy = (tau_grid > 0).astype(float)
        axes[1].contourf(xx, yy, policy, levels=1, cmap='RdYlGn', alpha=0.5)
        axes[1].contour(xx, yy, policy, levels=[0.5], colors='black', lw=1.5)
        axes[1].set_title('Optimal Binary Policy  (τ̂ > 0 → treat)')
        _scatter(axes[1])
        p_g = mpatches.Patch(color='green', alpha=0.5, label='Treat (τ̂>0)')
        p_r = mpatches.Patch(color='red',   alpha=0.5, label='Control (τ̂≤0)')
        axes[1].legend(handles=[p_g, p_r], fontsize=8)

    for ax in axes:
        ax.set_xlabel(names[feat_x]); ax.set_ylabel(names[feat_y])
    plt.tight_layout(); plt.show()


# =============================================================================
# 8. Per-leaf effects (regression / binary)
# =============================================================================

def plot_leaf_effects(
    model,
    X:            np.ndarray,
    y:            np.ndarray,
    w:            np.ndarray,
    n_bootstrap:  int   = 400,
    alpha:        float = 0.05,
    title:        str   = 'Per-Leaf Treatment Effect Estimates',
) -> None:
    """
    Dot-plot of per-leaf τ̂ with bootstrap confidence intervals.

    Args:
        model       : Fitted ``CausalTreeModel`` (regression or binary).
        X           : Covariate matrix ``(n, p)``.
        y           : Outcome vector ``(n,)``.
        w           : Treatment indicator ``(n,)``.
        n_bootstrap : Bootstrap replicates (default: 400).
        alpha       : Significance level (default: 0.05).
        title       : Figure title.
    """
    tree_layer = model._tree
    leaves     = tree_layer.get_leaf_nodes()
    if not leaves:
        print("No leaves — is the tree fitted?"); return

    tau_all = tree_layer.predict(X)
    rng     = np.random.default_rng(0)
    taus, lo_list, hi_list, ns = [], [], [], []

    for lf in leaves:
        lf_tau = float(lf.tau) if not isinstance(lf.tau, np.ndarray) else float(lf.tau[0])
        mask   = tau_all == lf_tau
        y_lf, w_lf, n_lf = y[mask], w[mask], int(mask.sum())

        if n_lf < 2 or (w_lf==1).sum()==0 or (w_lf==0).sum()==0:
            taus.append(lf_tau); lo_list.append(lf_tau); hi_list.append(lf_tau)
            ns.append(n_lf); continue

        boots = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_lf, n_lf)
            yb, wb = y_lf[idx], w_lf[idx]
            if (wb==1).sum()==0 or (wb==0).sum()==0: continue
            boots.append(float(yb[wb==1].mean() - yb[wb==0].mean()))

        if len(boots) > 10:
            lo_list.append(float(np.percentile(boots, 100*alpha/2)))
            hi_list.append(float(np.percentile(boots, 100*(1-alpha/2))))
        else:
            lo_list.append(lf_tau); hi_list.append(lf_tau)
        taus.append(lf_tau); ns.append(n_lf)

    order = np.argsort(taus)
    taus    = [taus[i]    for i in order]
    lo_list = [lo_list[i] for i in order]
    hi_list = [hi_list[i] for i in order]
    ns      = [ns[i]      for i in order]

    n_l  = len(taus)
    norm = Normalize(vmin=min(ns), vmax=max(ns))
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(10, max(4, n_l * 0.42)))

    for i, (tau, lo, hi, n) in enumerate(zip(taus, lo_list, hi_list, ns)):
        col = cmap(norm(n))
        ax.plot([lo, hi], [i, i], color=col, lw=2.0, alpha=0.8)
        ax.plot(tau, i, 'o', color=col, ms=8, zorder=3)

    ax.axvline(0, color='gray', ls='--', lw=1.2)
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f'Leaf {i+1}' for i in range(n_l)], fontsize=9)
    ax.set_xlabel('Estimated CATE  τ̂', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Leaf sample size', shrink=0.7)
    plt.tight_layout(); plt.show()


# =============================================================================
# 9. Multiclass per-class leaf heatmap
# =============================================================================

def plot_multiclass_effects(
    model,
    class_names: Optional[list] = None,
    title:       str = 'Per-Leaf Per-Class Treatment Effect  τ̂_k',
) -> None:
    """
    Heatmap of per-class τ̂_k estimates across all leaves.

    Rows = leaves (sorted by most-favoured class),
    Columns = classes k = 0, …, K−1.

    Args:
        model       : Fitted ``CausalTreeModel`` with ``task='multiclass'``.
        class_names : Class name list.
        title       : Figure title.
    """
    if model.task != 'multiclass':
        raise ValueError("plot_multiclass_effects requires task='multiclass'.")
    leaves  = model.get_leaf_nodes()
    K       = model._tree.n_classes
    cnames  = class_names or [f'C{k}' for k in range(K)]
    n_l     = len(leaves)

    # matrix (n_leaves, K)
    mat = np.array([lf.tau for lf in leaves])  # each row is the tau vector
    # sort by argmax class
    order = np.argsort(np.argmax(mat, axis=1))
    mat   = mat[order]

    fig, ax = plt.subplots(figsize=(max(6, K * 1.2), max(4, n_l * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(K)); ax.set_xticklabels(cnames, fontsize=10)
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f'Leaf {i+1}' for i in range(n_l)], fontsize=9)
    for i in range(n_l):
        for j in range(K):
            ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                    fontsize=8, color='black' if abs(mat[i,j]) < 0.3 else 'white')
    fig.colorbar(im, ax=ax, label='τ̂_k  (P(Y=k|T) − P(Y=k|C))')
    ax.set_xlabel('Class k', fontsize=12)
    ax.set_ylabel('Leaf', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()


# =============================================================================
# 10. Training / sweep history
# =============================================================================

def plot_training_history(
    history:     list[tuple],
    metric_name: str = 'tau_risk',
    xlabel:      str = 'Parameter value',
) -> None:
    """
    Line plot of a metric recorded over a hyperparameter sweep.

    Args:
        history     : List of ``(param_value, metric_value)`` tuples.
        metric_name : Label for the y-axis.
        xlabel      : Label for the x-axis.
    """
    xs, ys = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, color='darkviolet', marker='o', ms=5, lw=1.8,
             label=metric_name.upper())
    best_x = xs[int(np.argmin(ys))]; best_y = min(ys)
    plt.axvline(best_x, color='tomato', ls='--', lw=1.2,
                label=f'Best = {best_x}  ({best_y:.4f})')
    plt.xlabel(xlabel); plt.ylabel(metric_name.upper())
    plt.title('Hyperparameter Sweep', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
