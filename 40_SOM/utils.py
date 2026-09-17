# utils.py
"""
Utility functions for Self-Organizing Map (SOM) workflows.

Contents
--------
1.  ``export_to_json``          — Serialise a trained SOMModel to JSON.
2.  ``train_model``             — Convenience wrapper around SOMModel.fit().
3.  ``plot_training_history``   — Quantization error over epochs.
4.  ``plot_umatrix``            — Unified Distance Matrix with hit overlay.
5.  ``plot_component_planes``   — Per-feature weight slice visualisation.
6.  ``plot_winner_map``         — BMU activation frequency (hit map).
7.  ``plot_som_scatter``        — 2-D SOM embedding of labelled data.
8.  ``plot_weight_trajectories``— Weight vector movement during training.
9.  ``plot_calibration``        — Quantization and topographic error vs. epoch.
10. ``evaluate_metrics``        — Full metric report after training.
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

def export_to_json(model, filepath: str) -> None:
    """
    Serialise a trained ``SOMModel`` to a JSON file.

    The produced JSON contains three top-level keys:

    - ``architecture`` : Map dimensions, topology, kernel, distance metric,
                         schedule names, and hyperparameters.
    - ``training``     : Number of epochs trained and final quantization error.
    - ``weights``      : The weight matrix as a nested list of shape
                         ``[n_rows][n_cols][input_dim]``.

    Args:
        model    : A trained ``SOMModel`` instance.
        filepath : Destination file path.  Parent directories are created
                   automatically.
    """
    arch = {
        'n_rows':          model.n_rows,
        'n_cols':          model.n_cols,
        'input_dim':       model.input_dim,
        'topology':        model.topology,
        'kernel':          model.kernel_name,
        'distance':        model.distance_name,
        'lr_schedule':     model.lr_schedule_name,
        'sigma_schedule':  model.sigma_schedule_name,
        'lr_0':            model.lr_0,
        'lr_min':          model.lr_min,
        'sigma_0':         model.sigma_0,
        'sigma_min':       model.sigma_min,
        'init':            model.init,
    }
    training = {
        'n_epochs': model.n_epochs_,
        'final_qe': (
            model.quantization_errors_[-1][1]
            if model.quantization_errors_ else None
        ),
    }
    data = {
        'architecture': arch,
        'training':     training,
        'weights':      model.weights.tolist(),
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Model exported -> {filepath}")


# =============================================================================
# 2.  Training loop wrapper
# =============================================================================

def train_model(
    model,
    X:          np.ndarray,
    n_epochs:   int  = 100,
    mode:       str  = 'online',
    shuffle:    bool = True,
    log_every:  int  = 10,
) -> list:
    """
    Convenience wrapper around ``SOMModel.fit()``.

    Args:
        model     : ``SOMModel`` instance.
        X         : Training data of shape ``(N, d)``.
        n_epochs  : Number of training epochs (default: ``100``).
        mode      : ``'online'`` or ``'batch'`` (default: ``'online'``).
        shuffle   : Whether to shuffle data each epoch (default: ``True``).
        log_every : Logging frequency in epochs (default: ``10``).

    Returns:
        List of ``(epoch, quantization_error)`` tuples.
    """
    model.fit(X, n_epochs=n_epochs, mode=mode, shuffle=shuffle,
              verbose=True, log_every=log_every)
    return model.quantization_errors_


# =============================================================================
# 3.  Training history
# =============================================================================

def plot_training_history(
    history: list,
    title:   str = 'SOM Training — Quantization Error',
) -> None:
    """
    Plot the quantization error over training epochs.

    Args:
        history : List of ``(epoch, qe)`` tuples from ``train_model``.
        title   : Figure title.
    """
    epochs, qe = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, qe, color='darkviolet', marker='o',
             markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Quantization Error (mean L2)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  U-matrix
# =============================================================================

def plot_umatrix(
    model,
    X:      Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title:  str = 'U-Matrix — Unified Distance Matrix',
    cmap:   str = 'bone_r',
) -> None:
    """
    Visualise the Unified Distance Matrix (U-matrix) of the trained SOM.

    The U-matrix displays the mean distance between each neuron and its
    neighbors.  High values (dark in ``bone_r``) correspond to cluster
    boundaries; low values correspond to cluster interiors.

    When *X* is supplied, the Best Matching Unit of each training sample
    is overlaid as a scatter point (colored by *labels* if provided).

    Args:
        model  : Trained ``SOMModel`` instance.
        X      : Optional input data of shape ``(N, d)`` for BMU overlay.
        labels : Optional integer class labels of shape ``(N,)`` for coloring.
        title  : Figure title.
        cmap   : Colormap for the U-matrix heatmap (default: ``'bone_r'``).
    """
    U = model.u_matrix()   # (n_rows, n_cols)

    n_panels = 2 if (X is not None) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Left: U-matrix heatmap ----
    ax = axes[0]
    im = ax.imshow(U, cmap=cmap, interpolation='nearest', origin='upper')
    ax.set_title('U-Matrix')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    fig.colorbar(im, ax=ax, label='Mean neighbor distance')

    # ---- Right: U-matrix with BMU scatter overlay ----
    if X is not None:
        ax2 = axes[1]
        ax2.imshow(U, cmap=cmap, interpolation='nearest', origin='upper',
                   alpha=0.7)

        bmus = model.predict(X)
        jitter = np.random.RandomState(0).uniform(-0.3, 0.3, (len(bmus), 2))

        if labels is not None:
            unique_labels = np.unique(labels)
            palette = cm.get_cmap('tab10', len(unique_labels))
            for k, lbl in enumerate(unique_labels):
                mask = labels == lbl
                ax2.scatter(
                    bmus[mask, 1] + jitter[mask, 1],
                    bmus[mask, 0] + jitter[mask, 0],
                    c=[palette(k)], s=20, alpha=0.7,
                    label=f'Class {lbl}', edgecolors='none',
                )
            ax2.legend(fontsize=8, loc='upper right')
        else:
            ax2.scatter(
                bmus[:, 1] + jitter[:, 1],
                bmus[:, 0] + jitter[:, 0],
                c='tomato', s=18, alpha=0.6, edgecolors='none',
            )

        ax2.set_title('U-Matrix + BMU Activations')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Component planes
# =============================================================================

def plot_component_planes(
    model,
    feature_names: Optional[list] = None,
    cmap:          str = 'viridis',
    max_features:  int = 12,
    title:         str = 'SOM Component Planes',
) -> None:
    """
    Visualise the component planes of the SOM weight matrix.

    Each panel shows the 2-D weight map for one input feature.
    High values indicate that the neurons in that region of the map
    were strongly activated by that feature during training.

    Args:
        model         : Trained ``SOMModel`` instance.
        feature_names : Optional list of feature name strings.
        cmap          : Colormap (default: ``'viridis'``).
        max_features  : Maximum number of features to display (default: 12).
        title         : Figure title.
    """
    planes = model.component_planes()   # (d, n_rows, n_cols)
    d = min(planes.shape[0], max_features)
    names = feature_names or [f'Feature {k}' for k in range(d)]

    cols = min(d, 4)
    rows = (d + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten() if d > 1 else [axes]

    for k in range(d):
        ax = axes_flat[k]
        im = ax.imshow(planes[k], cmap=cmap, interpolation='nearest',
                       origin='upper')
        ax.set_title(names[k], fontsize=10)
        ax.set_xlabel('Col')
        ax.set_ylabel('Row')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(d, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  Winner / hit map
# =============================================================================

def plot_winner_map(
    model,
    X:      np.ndarray,
    labels: Optional[np.ndarray] = None,
    title:  str = 'SOM Hit Map (BMU Activation Frequency)',
) -> None:
    """
    Plot the BMU activation frequency map with optional class pie charts.

    Each cell in the grid is colored by how many training samples mapped
    to it.  When *labels* is supplied, a small pie chart inside each active
    cell shows the class distribution of samples that activated it.

    Args:
        model  : Trained ``SOMModel`` instance.
        X      : Input data of shape ``(N, d)``.
        labels : Optional integer class labels of shape ``(N,)``.
        title  : Figure title.
    """
    hmap = model.winner_map(X)   # (n_rows, n_cols)
    bmus = model.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Left: hit frequency heatmap ----
    ax = axes[0]
    im = ax.imshow(hmap, cmap='YlOrRd', interpolation='nearest', origin='upper')
    ax.set_title('Hit Frequency')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    fig.colorbar(im, ax=ax, label='Number of activations')

    for r in range(model.n_rows):
        for c in range(model.n_cols):
            if hmap[r, c] > 0:
                ax.text(c, r, str(hmap[r, c]),
                        ha='center', va='center',
                        fontsize=7, color='black')

    # ---- Right: per-cell class distribution (if labels given) ----
    ax2 = axes[1]
    ax2.set_xlim(-0.5, model.n_cols - 0.5)
    ax2.set_ylim(-0.5, model.n_rows - 0.5)
    ax2.set_aspect('equal')
    ax2.set_title('Class Distribution per BMU Cell')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.2)

    if labels is not None:
        unique = np.unique(labels)
        palette = cm.get_cmap('tab10', len(unique))
        lbl_colors = {lbl: palette(k) for k, lbl in enumerate(unique)}

        for r in range(model.n_rows):
            for c in range(model.n_cols):
                mask = (bmus[:, 0] == r) & (bmus[:, 1] == c)
                if mask.sum() == 0:
                    continue
                cell_lbls = labels[mask]
                counts    = [int((cell_lbls == lbl).sum()) for lbl in unique]
                colors    = [lbl_colors[lbl] for lbl in unique]
                if sum(counts) > 0:
                    ax2.pie(
                        counts, colors=colors, radius=0.4,
                        center=(c, r), frame=True,
                    )

        patches = [mpatches.Patch(color=lbl_colors[lbl],
                                  label=f'Class {lbl}') for lbl in unique]
        ax2.legend(handles=patches, loc='upper right', fontsize=8)
    else:
        for r in range(model.n_rows):
            for c in range(model.n_cols):
                freq = hmap[r, c] / max(hmap.max(), 1)
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=cm.YlOrRd(freq), alpha=0.6,
                )
                ax2.add_patch(rect)
                if hmap[r, c] > 0:
                    ax2.text(c, r, str(hmap[r, c]),
                             ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 7.  SOM scatter (2-D embedding)
# =============================================================================

def plot_som_scatter(
    model,
    X:      np.ndarray,
    labels: Optional[np.ndarray] = None,
    title:  str = 'SOM 2-D Embedding',
) -> None:
    """
    Plot the 2-D SOM embedding of the dataset.

    Each sample is projected to the (row, col) position of its BMU,
    revealing the topological structure that the map has learned.
    Samples that are close in SOM space were mapped to nearby neurons,
    indicating similarity in input space.

    A dual-panel figure is produced:

    - Left  : Scatter plot in SOM grid space (row vs. col), colored by label.
    - Right : U-matrix background with sample scatter overlay.

    Args:
        model  : Trained ``SOMModel`` instance.
        X      : Input data of shape ``(N, d)``.
        labels : Optional integer class labels of shape ``(N,)``.
        title  : Figure title.
    """
    bmus  = model.predict(X)                        # (N, 2)
    U     = model.u_matrix()                        # (n_rows, n_cols)
    jitter = np.random.RandomState(42).uniform(-0.35, 0.35, bmus.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if labels is not None:
        unique = np.unique(labels)
        palette = cm.get_cmap('tab10', len(unique))
        color_map = {lbl: palette(k) for k, lbl in enumerate(unique)}
        colors = np.array([color_map[lbl] for lbl in labels])
    else:
        colors = 'steelblue'

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 1:
            ax.imshow(U, cmap='bone_r', interpolation='nearest',
                      origin='upper', alpha=0.6,
                      extent=[-0.5, model.n_cols - 0.5,
                               model.n_rows - 0.5, -0.5])
        ax.scatter(
            bmus[:, 1] + jitter[:, 1],
            bmus[:, 0] + jitter[:, 0],
            c=colors, s=22, alpha=0.7, edgecolors='none',
        )
        ax.set_xlim(-0.5, model.n_cols - 0.5)
        ax.set_ylim(model.n_rows - 0.5, -0.5)
        ax.set_xlabel('SOM Column')
        ax.set_ylabel('SOM Row')
        ax.set_title(
            'Grid Scatter' if ax_idx == 0 else 'U-Matrix + Scatter'
        )
        ax.grid(True, alpha=0.25)

    if labels is not None:
        patches = [mpatches.Patch(color=color_map[lbl],
                                  label=f'Class {lbl}')
                   for lbl in unique]
        axes[0].legend(handles=patches, fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 8.  Weight trajectories
# =============================================================================

def plot_weight_trajectories(
    weight_snapshots: list,
    neuron_indices:   list,
    feature_pair:     tuple = (0, 1),
    title:            str = 'SOM Weight Vector Trajectories',
) -> None:
    """
    Visualise how selected neuron weight vectors move during training.

    Args:
        weight_snapshots : List of weight arrays ``(n_rows, n_cols, d)``
                           captured at different epochs.
        neuron_indices   : List of ``(row, col)`` tuples identifying the
                           neurons to trace.
        feature_pair     : Tuple ``(feat_x, feat_y)`` selecting two feature
                           dimensions for the 2-D trajectory plot (default:
                           ``(0, 1)``).
        title            : Figure title.
    """
    fx, fy = feature_pair
    palette = cm.get_cmap('plasma', len(neuron_indices))

    plt.figure(figsize=(8, 6))
    for k, (r, c) in enumerate(neuron_indices):
        traj = np.array([snap[r, c] for snap in weight_snapshots])
        color = palette(k)
        plt.plot(traj[:, fx], traj[:, fy], '-', color=color,
                 linewidth=1.4, alpha=0.8)
        plt.scatter(traj[0,  fx], traj[0,  fy], color=color,
                    marker='o', s=60, zorder=5, label=f'Neuron ({r},{c}) start')
        plt.scatter(traj[-1, fx], traj[-1, fy], color=color,
                    marker='*', s=100, zorder=5)

    plt.xlabel(f'Feature {fx}')
    plt.ylabel(f'Feature {fy}')
    plt.title(title)
    plt.legend(fontsize=7, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 9.  Calibration: QE and TE vs. epoch
# =============================================================================

def plot_calibration(
    qe_history: list,
    te_history: Optional[list] = None,
    title:      str = 'SOM Training Convergence',
) -> None:
    """
    Plot quantization error and (optionally) topographic error over epochs.

    Args:
        qe_history : List of ``(epoch, qe)`` tuples.
        te_history : Optional list of ``(epoch, te)`` tuples.
        title      : Figure title.
    """
    n_panels = 2 if te_history is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight='bold')

    epochs_qe, qe_vals = zip(*qe_history)
    axes[0].plot(epochs_qe, qe_vals, color='darkviolet',
                 marker='o', markersize=3, linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Quantization Error')
    axes[0].set_title('Quantization Error vs. Epoch')
    axes[0].grid(True, alpha=0.3)

    if te_history is not None:
        epochs_te, te_vals = zip(*te_history)
        axes[1].plot(epochs_te, te_vals, color='tomato',
                     marker='o', markersize=3, linewidth=1.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Topographic Error')
        axes[1].set_title('Topographic Error vs. Epoch')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 10.  Full metric evaluation
# =============================================================================

def evaluate_metrics(
    model,
    X:      np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute a full battery of SOM quality metrics.

    Metrics computed
    ----------------
    - **Quantization Error (QE)** : Mean L2 distance between inputs and BMUs.
    - **Reconstruction Error (RE)**: Mean squared reconstruction error.
    - **Topographic Error (TE)**  : Fraction of topologically violated pairs.
    - **BMU Entropy (H)**         : Shannon entropy of the activation distribution.
    - **KL Divergence (KL)**      : KL(uniform || p_bmu).
    - **Dead Neurons (%)**        : Fraction of neurons never activated.
    - **Silhouette Score**        : Mean silhouette (if labels provided).
    - **Weight Smoothness**       : Mean squared weight-neighbor difference.

    Args:
        model  : Trained ``SOMModel`` instance.
        X      : Input data of shape ``(N, d)``.
        labels : Optional integer labels for silhouette computation.

    Returns:
        Dictionary mapping metric names to scalar float values.
    """
    from losses import (
        quantization_error, reconstruction_error,
        topographic_error, bmu_entropy, kl_divergence,
    )
    from vi import weight_smoothness_loss, coverage_loss

    bmus = model.predict(X)                         # (N, 2)
    w    = model.weights

    metrics: dict = {}
    metrics['quantization_error']   = quantization_error(X, bmus, w)
    metrics['reconstruction_error'] = reconstruction_error(X, bmus, w)
    metrics['topographic_error']    = topographic_error(X, w, model.topology)
    metrics['bmu_entropy']          = bmu_entropy(bmus, model.n_rows, model.n_cols)
    metrics['kl_divergence']        = kl_divergence(bmus, model.n_rows, model.n_cols)
    metrics['dead_neurons_pct']     = coverage_loss(bmus, model.n_rows, model.n_cols) * 100
    metrics['weight_smoothness']    = weight_smoothness_loss(w)

    if labels is not None:
        from losses import silhouette_score as _sil
        flat_bmus = bmus[:, 0] * model.n_cols + bmus[:, 1]
        metrics['silhouette'] = _sil(X, flat_bmus)

    return metrics
