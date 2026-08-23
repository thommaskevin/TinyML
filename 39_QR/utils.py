# utils.py
"""
Utility functions for Quantile Regression (QR) workflows.

Contents
--------
1. ``export_to_json``            — Serialise a trained QRModel to JSON.
2. ``train_model``               — Generic training loop with history logging.
3. ``plot_training_history``     — Loss curve over epochs.
4. ``plot_quantile_bands``       — Quantile fan / prediction-interval plot.
5. ``plot_quantile_spread``      — Interval width as a function of x.
6. ``plot_calibration``          — Reliability diagram: empirical vs. nominal
                                   coverage across all quantile levels.
7. ``plot_residuals``            — Pinball residual distribution per quantile.
8. ``plot_feature_importance``   — Permutation-based feature importance.
9. ``plot_quantile_crossing``    — Fraction of crossing violations per epoch.
10. ``evaluate_metrics``         — Compute pinball loss, PICP, MPIW, Winkler.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =============================================================================
# 1.  Export
# =============================================================================

def export_to_json(model, filepath: str) -> None:
    """
    Serialise a trained ``QRModel`` to a JSON file.

    The produced JSON contains three top-level keys:

    - ``hidden_layers``  : Architecture metadata (sizes, activations, dropout).
    - ``quantile_head``  : Head configuration (in_features, quantile levels, K).
    - ``parameters``     : Mapping of parameter names to nested lists.

    Args:
        model    : A trained ``QRModel`` instance.
        filepath : Destination file path.  Parent directories are created
                   automatically if they do not already exist.
    """
    arch_hidden: list = []
    params: dict      = {}

    def _w(t):
        return t.detach().cpu().tolist()

    # --- Hidden backbone layers ---
    for i, layer in enumerate(model.backbone):
        info = {
            'index':        i,
            'type':         'Dense',
            'in_features':  layer.in_features,
            'out_features': layer.out_features,
            'activation':   layer.activation_name,
            'dropout':      float(layer.dropout.p) if layer.dropout else 0.0,
        }
        params[f'hidden_{i}_weight'] = _w(layer.linear.weight)
        params[f'hidden_{i}_bias']   = (
            _w(layer.linear.bias) if layer.linear.bias is not None else None
        )
        arch_hidden.append(info)

    # --- Quantile head ---
    head = model.head
    head_info = {
        'in_features': head.in_features,
        'quantiles':   head.quantiles,
        'K':           head.K,
    }
    params['head_weight'] = _w(head.linear.weight)
    params['head_bias']   = (
        _w(head.linear.bias) if head.linear.bias is not None else None
    )

    data = {
        'hidden_layers': arch_hidden,
        'quantile_head': head_info,
        'parameters':    params,
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Model exported → {filepath}")


# =============================================================================
# 2.  Training loop
# =============================================================================

def train_model(
    model,
    X_train:     torch.Tensor,
    y_train:     torch.Tensor,
    optimizer,
    loss_name:   str  = 'multi_pinball',
    epochs:      int  = 500,
    print_every: int  = 100,
    **loss_kwargs,
) -> list:
    """
    Generic training loop for ``QRModel``.

    Args:
        model       : ``QRModel`` instance.
        X_train     : Input tensor of shape ``(N, in_features)``.
        y_train     : Target tensor of shape ``(N,)`` or ``(N, 1)``.
        optimizer   : A PyTorch optimiser (e.g. ``torch.optim.Adam``).
        loss_name   : Loss function name (see ``losses.py``).
                      Default: ``'multi_pinball'``.
        epochs      : Number of training epochs (default: ``500``).
        print_every : Log frequency in epochs (default: ``100``).
        **loss_kwargs: Extra keyword arguments forwarded to the loss function
                       (e.g. ``quantiles=[0.1, 0.5, 0.9]``).

    Returns:
        A list of ``(epoch, loss_value)`` tuples recorded at each logged
        epoch and at epoch 1.
    """
    from losses import compute_loss

    model.train()
    history: list = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(X_train)
        loss   = compute_loss(loss_name, output, y_train, **loss_kwargs)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            loss_val = loss.item()
            print(
                f"  Epoch {epoch:>5}/{epochs}"
                f"  |  {loss_name.upper()} = {loss_val:.6f}"
            )
            history.append((epoch, loss_val))

    return history


# =============================================================================
# 3.  Training history plot
# =============================================================================

def plot_training_history(
    history:   list,
    loss_name: str = 'loss',
    title:     str = 'Training Loss',
) -> None:
    """
    Plot the training loss curve.

    Args:
        history   : List of ``(epoch, loss_value)`` tuples from
                    ``train_model``.
        loss_name : Y-axis label.
        title     : Figure title.
    """
    epochs, losses = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, color='darkviolet', marker='o', markersize=3,
             linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel(loss_name.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  Quantile fan / prediction-interval plot
# =============================================================================

def plot_quantile_bands(
    model,
    X_train:   torch.Tensor,
    y_train:   torch.Tensor,
    X_test:    torch.Tensor,
    y_test:    torch.Tensor,
    quantiles: list,
    title:     str = 'Quantile Regression — Prediction Bands',
    feature_idx: int = 0,
) -> None:
    """
    Plot quantile prediction bands (fan chart) alongside the training data.

    The function evaluates the model over a dense grid of input values and
    renders one shaded band for each pair of symmetric quantiles (e.g.
    0.1 / 0.9, 0.2 / 0.8) and the median as a solid line.

    Args:
        model       : Trained ``QRModel`` instance.
        X_train     : Training input tensor of shape ``(N_tr, p)``.
        y_train     : Training target tensor of shape ``(N_tr,)``.
        X_test      : Test input tensor of shape ``(N_te, p)``.
        y_test      : Test target tensor of shape ``(N_te,)``.
        quantiles   : List of K quantile levels (must match the model's
                      ``self.quantiles``), e.g. ``[0.1, 0.2, 0.5, 0.8, 0.9]``.
        title       : Figure title.
        feature_idx : Which input feature to use as the x-axis when
                      ``p > 1`` (default: ``0``).
    """
    model.eval()

    x_np = X_test[:, feature_idx].numpy()
    x_min, x_max = x_np.min() - 0.5, x_np.max() + 0.5
    x_grid_np = np.linspace(x_min, x_max, 400).astype(np.float32)

    # Build a grid tensor of the same dimension as X_train
    p = X_train.shape[1]
    X_grid = torch.zeros(400, p)
    X_grid[:, feature_idx] = torch.from_numpy(x_grid_np)

    with torch.no_grad():
        q_hat = model(X_grid).numpy()    # (400, K)

    # Pair up symmetric quantiles around the median
    K = len(quantiles)
    med_idx = K // 2 if K % 2 == 1 else None

    # Color palette: darker bands = narrower intervals
    n_pairs  = K // 2
    palette  = cm.get_cmap('BuPu', n_pairs + 2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # ---- Left panel: quantile fan chart ----
    ax = axes[0]

    for i in range(n_pairs):
        lo_col = q_hat[:, i]
        hi_col = q_hat[:, K - 1 - i]
        label  = f'Q{quantiles[i]:.2f} – Q{quantiles[K-1-i]:.2f}'
        color  = palette(i + 1)
        ax.fill_between(x_grid_np, lo_col, hi_col,
                        alpha=0.35, color=color, label=label)

    if med_idx is not None:
        ax.plot(x_grid_np, q_hat[:, med_idx],
                color='darkviolet', linewidth=2.0,
                label=f'Median (Q{quantiles[med_idx]:.2f})')

    ax.scatter(
        X_train[:, feature_idx].numpy(),
        y_train.numpy(),
        c='steelblue', s=15, alpha=0.6, zorder=5, label='Train data',
    )
    ax.scatter(
        X_test[:, feature_idx].numpy(),
        y_test.numpy(),
        c='tomato', s=15, alpha=0.7, zorder=5, label='Test data',
    )
    ax.set_xlabel(f'Feature {feature_idx}')
    ax.set_ylabel('y')
    ax.set_title('Quantile Fan Chart')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Right panel: all individual quantile lines ----
    ax2 = axes[1]
    cmap_lines = cm.get_cmap('plasma', K)
    for k, q in enumerate(quantiles):
        ax2.plot(x_grid_np, q_hat[:, k],
                 color=cmap_lines(k), linewidth=1.5,
                 label=f'Q{q:.2f}', alpha=0.85)

    ax2.scatter(
        X_test[:, feature_idx].numpy(),
        y_test.numpy(),
        c='black', s=10, alpha=0.4, zorder=5, label='Test data',
    )
    ax2.set_xlabel(f'Feature {feature_idx}')
    ax2.set_ylabel('y')
    ax2.set_title('Individual Quantile Lines')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Interval width (spread) plot
# =============================================================================

def plot_quantile_spread(
    model,
    X_test:    torch.Tensor,
    quantiles: list,
    lower_q:   float = 0.1,
    upper_q:   float = 0.9,
    title:     str   = 'Prediction Interval Width',
    feature_idx: int = 0,
) -> None:
    """
    Plot the prediction interval width (upper quantile − lower quantile)
    as a function of the input feature.

    Wider intervals indicate regions of higher aleatoric uncertainty in the
    conditional distribution of y | x.

    Args:
        model       : Trained ``QRModel`` instance.
        X_test      : Test input tensor of shape ``(N, p)``.
        quantiles   : List of quantile levels matching the model.
        lower_q     : Lower quantile level (default: ``0.1``).
        upper_q     : Upper quantile level (default: ``0.9``).
        title       : Figure title.
        feature_idx : Feature index for the x-axis (default: ``0``).
    """
    model.eval()
    lo_idx = quantiles.index(lower_q)
    hi_idx = quantiles.index(upper_q)

    with torch.no_grad():
        q_hat = model(X_test).numpy()   # (N, K)

    x_vals = X_test[:, feature_idx].numpy()
    width  = q_hat[:, hi_idx] - q_hat[:, lo_idx]

    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    w_sorted = width[sort_idx]

    plt.figure(figsize=(9, 4))
    plt.fill_between(x_sorted, 0, w_sorted,
                     alpha=0.4, color='mediumorchid', label='PI width')
    plt.plot(x_sorted, w_sorted, color='darkviolet', linewidth=1.5)
    plt.axhline(w_sorted.mean(), color='tomato', linestyle='--',
                linewidth=1.0, label=f'Mean width = {w_sorted.mean():.3f}')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel(f'Width  (Q{upper_q} − Q{lower_q})')
    plt.title(f'{title}  [{int((upper_q - lower_q) * 100)}% PI]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  Calibration / reliability diagram
# =============================================================================

def plot_calibration(
    model,
    X_test:    torch.Tensor,
    y_test:    torch.Tensor,
    quantiles: list,
    title:     str = 'Quantile Calibration (Reliability Diagram)',
) -> None:
    """
    Reliability diagram: compare nominal vs. empirical quantile coverage.

    A perfectly calibrated model has empirical coverage equal to the nominal
    quantile level for every τ.  The plot shows how far the model deviates
    from the 45-degree ideal calibration line.

    Args:
        model     : Trained ``QRModel`` instance.
        X_test    : Test input tensor of shape ``(N, p)``.
        y_test    : Test target tensor of shape ``(N,)``.
        quantiles : List of quantile levels matching the model.
        title     : Figure title.
    """
    model.eval()
    with torch.no_grad():
        q_hat = model(X_test).numpy()    # (N, K)

    y_np = y_test.numpy().squeeze()
    empirical = []

    for k in range(len(quantiles)):
        coverage = float((y_np <= q_hat[:, k]).mean())
        empirical.append(coverage)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Left: calibration curve ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfect calibration')
    ax.plot(quantiles, empirical, 'o-', color='darkviolet', linewidth=2.0,
            markersize=6, label='Empirical coverage')
    ax.fill_between(quantiles, quantiles, empirical,
                    alpha=0.2, color='tomato',
                    label='Calibration gap')
    ax.set_xlabel('Nominal quantile level τ')
    ax.set_ylabel('Empirical coverage P(Y ≤ Q̂_τ)')
    ax.set_title('Reliability Diagram')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Right: calibration error bar chart ----
    ax2 = axes[1]
    errors = [e - q for e, q in zip(empirical, quantiles)]
    colors = ['steelblue' if e >= 0 else 'tomato' for e in errors]
    ax2.bar(
        [f'Q{q:.2f}' for q in quantiles],
        errors,
        color=colors, alpha=0.75, edgecolor='white',
    )
    ax2.axhline(0, color='black', linewidth=1.0)
    ax2.set_xlabel('Quantile level')
    ax2.set_ylabel('Calibration error (empirical − nominal)')
    ax2.set_title('Calibration Error per Quantile')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 7.  Pinball residual distribution
# =============================================================================

def plot_residuals(
    model,
    X_test:    torch.Tensor,
    y_test:    torch.Tensor,
    quantiles: list,
    title:     str = 'Pinball Residual Distributions',
) -> None:
    """
    Plot the signed residual (y − ŷ_τ) distribution for each quantile.

    For a well-calibrated quantile τ, the fraction of positive residuals
    should equal τ (i.e. τ × 100% of samples lie above the predicted quantile).

    Args:
        model     : Trained ``QRModel`` instance.
        X_test    : Test input tensor of shape ``(N, p)``.
        y_test    : Test target tensor of shape ``(N,)``.
        quantiles : List of quantile levels matching the model.
        title     : Figure title.
    """
    model.eval()
    with torch.no_grad():
        q_hat = model(X_test).numpy()   # (N, K)

    y_np = y_test.numpy().squeeze()
    K    = len(quantiles)
    cols = min(K, 4)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten() if K > 1 else [axes]

    for k, q in enumerate(quantiles):
        ax       = axes_flat[k]
        residual = y_np - q_hat[:, k]
        frac_pos = (residual > 0).mean()
        ax.hist(residual, bins=40, color='mediumorchid', alpha=0.75,
                edgecolor='white')
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_title(f'Q{q:.2f}  |  P(y>ŷ) = {frac_pos:.3f}',
                     fontsize=10)
        ax.set_xlabel('Residual (y − ŷ_τ)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.25)

    # Hide any unused axes
    for idx in range(K, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 8.  Feature importance (permutation-based)
# =============================================================================

def plot_feature_importance(
    model,
    X_test:       torch.Tensor,
    y_test:       torch.Tensor,
    quantiles:    list,
    feature_names: Optional[list] = None,
    n_repeats:    int  = 10,
    target_q_idx: int  = 0,
    title:        str  = 'Permutation Feature Importance',
) -> np.ndarray:
    """
    Estimate permutation-based feature importance for a target quantile.

    For each feature, the function randomly permutes its values *n_repeats*
    times and measures the increase in pinball loss relative to the baseline.
    Features that cause a larger loss increase when permuted are more important.

    Args:
        model         : Trained ``QRModel`` instance.
        X_test        : Test input tensor of shape ``(N, p)``.
        y_test        : Test target tensor of shape ``(N,)``.
        quantiles     : List of quantile levels matching the model.
        feature_names : Optional list of p feature name strings.
        n_repeats     : Number of permutation repeats per feature.
        target_q_idx  : Column index (quantile index) used to compute the
                        baseline pinball loss (default: ``0``).
        title         : Figure title.

    Returns:
        Array of shape ``(p,)`` with mean importance scores.
    """
    from losses import pinball_loss

    model.eval()
    p = X_test.shape[1]
    feature_names = feature_names or [f'Feature {i}' for i in range(p)]
    tau = quantiles[target_q_idx]

    with torch.no_grad():
        baseline_pred = model(X_test)[:, target_q_idx]
        baseline_loss = float(pinball_loss(baseline_pred, y_test, quantile=tau))

    rng = np.random.RandomState(42)
    importances = np.zeros((p, n_repeats))

    for j in range(p):
        for r in range(n_repeats):
            X_perm = X_test.clone()
            perm   = rng.permutation(len(X_perm))
            X_perm[:, j] = X_perm[perm, j]
            with torch.no_grad():
                perm_pred = model(X_perm)[:, target_q_idx]
                perm_loss = float(
                    pinball_loss(perm_pred, y_test, quantile=tau)
                )
            importances[j, r] = perm_loss - baseline_loss

    mean_imp = importances.mean(axis=1)
    std_imp  = importances.std(axis=1)
    sort_idx = np.argsort(mean_imp)[::-1]

    fig, ax = plt.subplots(figsize=(max(6, p * 0.7 + 2), 5))
    bars = ax.bar(
        range(p),
        mean_imp[sort_idx],
        yerr=std_imp[sort_idx],
        capsize=4,
        color='mediumorchid', alpha=0.75, edgecolor='white',
        error_kw={'elinewidth': 1.5, 'ecolor': 'darkviolet'},
    )
    ax.set_xticks(range(p))
    ax.set_xticklabels(
        [feature_names[i] for i in sort_idx],
        rotation=40, ha='right',
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel(f'Mean pinball loss increase (Q{tau:.2f})')
    ax.set_title(f'{title}  [quantile = Q{tau:.2f}]')
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.show()

    return mean_imp


# =============================================================================
# 9.  Quantile crossing diagnostic
# =============================================================================

def plot_quantile_crossing(
    model,
    X_test:    torch.Tensor,
    quantiles: list,
    title:     str = 'Quantile Crossing Diagnostic',
) -> None:
    """
    Visualise the fraction and magnitude of any quantile crossing violations
    in the model's predictions on the test set.

    A crossing occurs when ŷ_{τ_k} > ŷ_{τ_{k+1}} for a given sample, which
    violates the monotonicity of the conditional quantile function.

    Args:
        model     : Trained ``QRModel`` instance.
        X_test    : Test input tensor of shape ``(N, p)``.
        quantiles : List of K quantile levels.
        title     : Figure title.
    """
    model.eval()
    with torch.no_grad():
        q_hat = model(X_test).numpy()   # (N, K)

    K   = len(quantiles)
    N   = q_hat.shape[0]
    pairs          = []
    crossing_frac  = []
    crossing_mag   = []

    for k in range(K - 1):
        gap         = q_hat[:, k + 1] - q_hat[:, k]   # should be ≥ 0
        violations  = gap < 0
        pairs.append(f'Q{quantiles[k]:.2f}→Q{quantiles[k+1]:.2f}')
        crossing_frac.append(float(violations.mean()) * 100)
        crossing_mag.append(float(np.abs(gap[violations]).mean())
                            if violations.any() else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Left: crossing fraction ----
    ax = axes[0]
    bars = ax.bar(pairs, crossing_frac,
                  color=['tomato' if v > 0 else 'steelblue' for v in crossing_frac],
                  alpha=0.75, edgecolor='white')
    ax.set_ylabel('Crossing fraction (%)')
    ax.set_title('Fraction of Test Samples with Crossing')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, crossing_frac):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    # ---- Right: mean violation magnitude ----
    ax2 = axes[1]
    ax2.bar(pairs, crossing_mag,
            color=['tomato' if v > 0 else 'steelblue' for v in crossing_mag],
            alpha=0.75, edgecolor='white')
    ax2.set_ylabel('Mean violation magnitude |ŷ_k − ŷ_{k+1}|')
    ax2.set_title('Mean Crossing Magnitude (when it occurs)')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 10.  Evaluation metrics
# =============================================================================

def evaluate_metrics(
    model,
    X_test:    torch.Tensor,
    y_test:    torch.Tensor,
    quantiles: list,
    lower_q:   float = 0.1,
    upper_q:   float = 0.9,
) -> dict:
    """
    Compute a standard battery of quantile regression evaluation metrics.

    Metrics computed
    ----------------
    - **Pinball loss** (per quantile and average)
    - **PICP** (Prediction Interval Coverage Probability): fraction of
      test samples within the (lower_q, upper_q) interval.
    - **MPIW** (Mean Prediction Interval Width): average width of the
      (lower_q, upper_q) interval.
    - **Winkler score**: Winkler interval score for the (lower_q, upper_q) PI.
    - **Median MAE**: mean absolute error of the median prediction (if
      0.5 is in *quantiles*).

    Args:
        model     : Trained ``QRModel`` instance.
        X_test    : Test input tensor of shape ``(N, p)``.
        y_test    : Test target tensor of shape ``(N,)``.
        quantiles : List of quantile levels matching the model.
        lower_q   : Lower bound quantile for interval metrics.
        upper_q   : Upper bound quantile for interval metrics.

    Returns:
        Dictionary of metric names to scalar float values.
    """
    from losses import pinball_loss

    model.eval()
    with torch.no_grad():
        q_hat = model(X_test).numpy()

    y_np = y_test.numpy().squeeze()
    N    = len(y_np)
    metrics: dict = {}

    # Per-quantile pinball loss
    total_pb = 0.0
    for k, q in enumerate(quantiles):
        pred = torch.from_numpy(q_hat[:, k])
        pb   = float(pinball_loss(pred, y_test, quantile=q))
        metrics[f'pinball_Q{q:.2f}'] = pb
        total_pb += pb
    metrics['pinball_avg'] = total_pb / len(quantiles)

    # PICP / MPIW / Winkler — only computed when both bound quantiles are
    # present in the model's quantile list.  For single-quantile models
    # (e.g. median-only) these metrics are skipped gracefully.
    has_lower = lower_q in quantiles
    has_upper = upper_q in quantiles

    if has_lower and has_upper:
        lo_col  = q_hat[:, quantiles.index(lower_q)]
        hi_col  = q_hat[:, quantiles.index(upper_q)]
        covered = (y_np >= lo_col) & (y_np <= hi_col)
        metrics['PICP'] = float(covered.mean())
        metrics['MPIW'] = float((hi_col - lo_col).mean())

        alpha   = 1.0 - (upper_q - lower_q)
        alpha   = max(alpha, 1e-8)          # guard against zero division
        below   = np.maximum(lo_col - y_np, 0.0)
        above   = np.maximum(y_np  - hi_col, 0.0)
        winkler = (hi_col - lo_col) + (2.0 / alpha) * (below + above)
        metrics['Winkler'] = float(winkler.mean())
    else:
        missing = []
        if not has_lower:
            missing.append(f'lower_q={lower_q}')
        if not has_upper:
            missing.append(f'upper_q={upper_q}')
        print(
            f"  [evaluate_metrics] Skipping PICP / MPIW / Winkler: "
            f"{', '.join(missing)} not in model quantiles {quantiles}. "
            f"Pass matching lower_q / upper_q to enable interval metrics."
        )

    # Median MAE (if 0.5 in quantiles)
    if 0.5 in quantiles:
        med_col = q_hat[:, quantiles.index(0.5)]
        metrics['median_MAE'] = float(np.abs(y_np - med_col).mean())

    return metrics