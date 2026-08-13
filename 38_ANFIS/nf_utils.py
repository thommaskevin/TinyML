# nf_utils.py
"""
Utility functions for Adaptive Neuro-Fuzzy Inference System (ANFIS) workflows.

Contents
--------
1. ``export_to_json``              — Serialise a trained ANFISModel to JSON.
2. ``plot_membership_functions``   — Visualise learned MFs per input.
3. ``plot_firing_strengths``       — Bar chart of rule activations.
4. ``plot_regression_surface``     — 2-D regression surface + data.
5. ``plot_decision_boundary``      — Decision boundary for classification.
6. ``train_model``                 — Generic training loop.
7. ``plot_training_history``       — Loss curve.
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from nf_layers import FuzzifyLayer, DenseLayer


# =============================================================================
# 1.  Export
# =============================================================================

def export_to_json(model, filepath: str) -> None:
    """
    Serialise a trained ``ANFISModel`` to a JSON file.

    The produced JSON contains:

    - ``anfis_config``   : Architecture metadata (n_inputs, n_terms, mf_type).
    - ``dense_layers``   : Architecture metadata for optional dense head.
    - ``parameters``     : Mapping of parameter names to nested lists.

    Args:
        model    : A trained ``ANFISModel`` instance.
        filepath : Destination file path.
    """
    params: dict = {}

    # --- Fuzzify layer ---
    fuzz = model.fuzzify
    if fuzz.mf_type == 'gaussian':
        params['fuzzify_center'] = fuzz.center.detach().cpu().tolist()
        params['fuzzify_sigma']  = fuzz.sigma.detach().cpu().tolist()
    elif fuzz.mf_type == 'triangular':
        params['fuzzify_a'] = fuzz.a.detach().cpu().tolist()
        params['fuzzify_b'] = fuzz.b.detach().cpu().tolist()
        params['fuzzify_c'] = fuzz.c.detach().cpu().tolist()
    elif fuzz.mf_type == 'bell':
        params['fuzzify_a'] = fuzz.a.detach().cpu().tolist()
        params['fuzzify_b'] = fuzz.b.detach().cpu().tolist()
        params['fuzzify_c'] = fuzz.c.detach().cpu().tolist()
    elif fuzz.mf_type == 'sigmoid':
        params['fuzzify_a'] = fuzz.a.detach().cpu().tolist()
        params['fuzzify_c'] = fuzz.c.detach().cpu().tolist()

    # --- Consequent layer ---
    params['consequent_coeff'] = model.consequent.coeff.detach().cpu().tolist()

    # --- Dense head ---
    for j, dense in enumerate(model.dense_head):
        params[f'dense_{j}_weight'] = dense.linear.weight.detach().cpu().tolist()
        if dense.linear.bias is not None:
            params[f'dense_{j}_bias'] = dense.linear.bias.detach().cpu().tolist()

    data = {
        'anfis_config': {
            'n_inputs':  model.n_inputs,
            'n_terms':   model.n_terms,
            'mf_type':   model.mf_type,
            'n_rules':   model.n_rules,
        },
        'dense_layers': [
            {
                'index':        j,
                'in_features':  d.linear.in_features,
                'out_features': d.linear.out_features,
                'activation':   d.activation_name,
            }
            for j, d in enumerate(model.dense_head)
        ],
        'parameters': params,
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"ANFIS model exported → {filepath}")


# =============================================================================
# 2.  Membership function plots
# =============================================================================

def plot_membership_functions(
    model,
    x_range: tuple[float, float] = (-3.0, 3.0),
    n_points: int = 400,
    title: str = 'Learned Membership Functions',
) -> None:
    """
    Plot the learned membership functions for every input variable.

    Args:
        model    : Trained ``ANFISModel``.
        x_range  : (min, max) for the x-axis.
        n_points : Number of evaluation points.
        title    : Figure title.
    """
    fuzz = model.fuzzify
    n_in = fuzz.n_inputs
    n_te = fuzz.n_terms

    x = torch.linspace(x_range[0], x_range[1], n_points)

    fig, axes = plt.subplots(n_in, 1, figsize=(10, 2.5 * n_in), sharex=True)
    if n_in == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    with torch.no_grad():
        for j, ax in enumerate(axes):
            # Evaluate MFs for input dimension j
            xj = x.unsqueeze(1)  # (n_points, 1)
            # Manually compute MFs for this dimension
            if fuzz.mf_type == 'gaussian':
                c = fuzz.center[j].unsqueeze(0)  # (1, n_terms)
                s = fuzz.sigma[j].unsqueeze(0)
                mu = torch.exp(-0.5 * ((xj - c) / s.clamp(min=1e-6)) ** 2)
            elif fuzz.mf_type == 'sigmoid':
                a = fuzz.a[j].unsqueeze(0)
                c = fuzz.c[j].unsqueeze(0)
                mu = torch.sigmoid(a * (xj - c))
            else:
                # Fallback: use forward with dummy other dims
                dummy = torch.zeros(n_points, n_in)
                dummy[:, j] = x
                mu_full = fuzz(dummy)  # (n_points, n_in, n_terms)
                mu = mu_full[:, j, :]  # (n_points, n_terms)

            mu_np = mu.numpy()
            for k in range(n_te):
                ax.plot(x.numpy(), mu_np[:, k], linewidth=2,
                        label=f'Term {k+1}')

            ax.set_ylabel(f'Input x{j+1}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Input value')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3.  Firing strengths
# =============================================================================

def plot_firing_strengths(
    model,
    X: torch.Tensor,
    title: str = 'Rule Firing Strengths',
    top_k: int = 10,
) -> None:
    """
    Bar chart of average firing strength per rule.

    Args:
        model : Trained ``ANFISModel``.
        X     : Input tensor of shape ``(N, n_inputs)``.
        title : Figure title.
        top_k : Show only the top-k most active rules.
    """
    model.eval()
    with torch.no_grad():
        w = model.get_firing_strengths(X)  # (N, n_rules)
        avg_w = w.mean(dim=0).numpy()

    n_rules = len(avg_w)
    top_k = min(top_k, n_rules)
    top_idx = np.argsort(avg_w)[-top_k:]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), avg_w[top_idx], color='steelblue')
    plt.xticks(range(top_k), [f'R{int(i)}' for i in top_idx], rotation=45)
    plt.ylabel('Average firing strength')
    plt.title(f'{title} (top {top_k} of {n_rules})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  Regression surface (2-D inputs only)
# =============================================================================

def plot_regression_surface(
    model,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    title:     str,
    n_grid:    int = 200,
) -> None:
    """
    Plot the learned regression surface for 2-D input ANFIS.

    Args:
        model   : Trained ``ANFISModel`` with ``n_inputs == 2``.
        X_train : Training inputs of shape ``(N, 2)``.
        y_train : Training targets of shape ``(N,)`` or ``(N, 1)``.
        title   : Figure title.
        n_grid  : Grid resolution.
    """
    if model.n_inputs != 2:
        raise ValueError("plot_regression_surface requires n_inputs == 2.")

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(xx, yy, Z, levels=30, cmap='viridis', alpha=0.9)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.squeeze(),
               edgecolors='k', cmap='viridis', s=40, zorder=5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label='ŷ')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Decision boundary (2-D inputs only)
# =============================================================================

def plot_decision_boundary(
    model,
    X:         np.ndarray,
    y:         np.ndarray,
    title:     str,
    task:      str = 'binary',
    n_grid:    int = 200,
) -> None:
    """
    Plot the decision boundary for a classification ANFIS (2-D inputs).

    Args:
        model : Trained ``ANFISModel`` with ``n_inputs == 2``.
        X     : Input feature matrix of shape ``(N, 2)``.
        y     : Integer class labels of shape ``(N,)``.
        title : Figure title.
        task  : ``'binary'`` or ``'multiclass'``.
        n_grid: Grid resolution.
    """
    if model.n_inputs != 2:
        raise ValueError("plot_decision_boundary requires n_inputs == 2.")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    model.eval()
    with torch.no_grad():
        out = model(grid)
        if task == 'binary':
            Z = torch.sigmoid(out).numpy().reshape(xx.shape)
        else:
            Z = torch.argmax(out, dim=1).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    if task == 'binary':
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdPu', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdPu')
    else:
        contour = ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    fig.colorbar(contour, ax=ax)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6.  Training loop
# =============================================================================

def train_model(
    model,
    X_train:     torch.Tensor,
    y_train:     torch.Tensor,
    optimizer,
    loss_name:   str = 'mse',
    epochs:      int = 500,
    print_every: int = 100,
    **loss_kwargs,
) -> list[tuple[int, float]]:
    """
    Generic training loop for ``ANFISModel``.

    Args:
        model       : ``ANFISModel`` instance.
        X_train     : Input tensor of shape ``(N, n_inputs)``.
        y_train     : Target tensor.
        optimizer   : A PyTorch optimiser.
        loss_name   : Name of the loss function (see ``losses.py``).
        epochs      : Number of training epochs.
        print_every : Print loss every *N* epochs.
        **loss_kwargs: Extra keyword arguments forwarded to the loss function.

    Returns:
        A list of ``(epoch, loss_value)`` tuples.
    """
    from losses import compute_loss

    model.train()
    history: list[tuple[int, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(X_train)
        loss   = compute_loss(loss_name, output, y_train, **loss_kwargs)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            loss_val = loss.item()
            print(f"  Epoch {epoch:>5}/{epochs}  |  {loss_name.upper()} = {loss_val:.6f}")
            history.append((epoch, loss_val))

    return history


# =============================================================================
# 7.  Training history plot
# =============================================================================

def plot_training_history(
    history:   list[tuple[int, float]],
    loss_name: str = 'loss',
) -> None:
    """
    Plot the training loss curve.

    Args:
        history   : List of ``(epoch, loss_value)`` tuples.
        loss_name : Label for the y-axis.
    """
    epochs, losses = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, color='steelblue', marker='o', markersize=3,
             linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel(loss_name.upper())
    plt.title('Training Loss — ANFIS')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
