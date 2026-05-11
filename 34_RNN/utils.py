# utils.py
"""
Utility functions for Elman RNN workflows.

Contents
--------
1. ``export_to_json``           — Serialise a trained RNNModel to JSON.
2. ``plot_regression_uncertainty`` — Predictive mean and uncertainty bands.
3. ``plot_decision_boundary``   — Decision boundary and epistemic uncertainty.
4. ``plot_sequence_prediction`` — True vs. predicted time series.
5. ``train_model``              — Generic training loop.
6. ``plot_training_history``    — Loss curve.
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from layers import RNNCell, DenseLayer


# =============================================================================
# 1.  Export
# =============================================================================

def export_to_json(model, filepath: str) -> None:
    """
    Serialise a trained ``RNNModel`` to a JSON file.

    The produced JSON contains three top-level keys:

    - ``recurrent_layers`` : Architecture metadata for every recurrent cell.
    - ``dense_layers``     : Architecture metadata for every dense layer.
    - ``parameters``       : Mapping of parameter names to nested lists
                             (weights and biases as plain Python lists).

    Args:
        model    : A trained ``RNNModel`` instance.
        filepath : Destination file path (parent directories are created
                   automatically if they do not already exist).
    """
    arch_rec: list = []
    arch_den: list = []
    params:   dict = {}

    # --- Recurrent cells ---
    for i, cell in enumerate(model.recurrent_cells):
        info = {
            'index':      i,
            'type':       'RNN',
            'input_size': cell.input_size,
            'hidden_size': cell.hidden_size,
            'activation': cell.activation_name,
        }
        params[f'rec_{i}_W_ih_weight'] = cell.W_ih.weight.detach().cpu().tolist()
        params[f'rec_{i}_W_ih_bias'] = (
            cell.W_ih.bias.detach().cpu().tolist()
            if cell.W_ih.bias is not None else None
        )
        params[f'rec_{i}_W_hh_weight'] = cell.W_hh.weight.detach().cpu().tolist()
        params[f'rec_{i}_W_hh_bias'] = (
            cell.W_hh.bias.detach().cpu().tolist()
            if cell.W_hh.bias is not None else None
        )
        arch_rec.append(info)

    # --- Dense layers ---
    for j, dense in enumerate(model.dense_head):
        info = {
            'index':       j,
            'in_features': dense.in_features,
            'out_features': dense.out_features,
            'activation':  dense.activation_name,
        }
        params[f'dense_{j}_weight'] = dense.linear.weight.detach().cpu().tolist()
        params[f'dense_{j}_bias'] = (
            dense.linear.bias.detach().cpu().tolist()
            if dense.linear.bias is not None else None
        )
        arch_den.append(info)

    data = {
        'recurrent_layers': arch_rec,
        'dense_layers':     arch_den,
        'parameters':       params,
    }

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Model exported → {filepath}")


# =============================================================================
# 2.  Regression uncertainty
# =============================================================================

def plot_regression_uncertainty(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    title:   str,
    n_samples: int = 100,
    seq_len:   int = 1,
) -> None:
    """
    Plot predictive mean and uncertainty bands for a regression RNN.

    The function samples the model *n_samples* times (enabling Monte-Carlo
    stochasticity if dropout or similar modules are present) and displays
    the mean prediction together with ±1 and ±2 standard-deviation bands.

    Args:
        model     : Trained ``RNNModel`` instance.
        X_train   : Training inputs  of shape ``(N, input_size)``.
        y_train   : Training targets of shape ``(N,)`` or ``(N, 1)``.
        X_test    : Test inputs  of shape ``(M, input_size)``.
        y_test    : Test targets of shape ``(M,)`` or ``(M, 1)``.
        title     : Figure title.
        n_samples : Number of stochastic forward passes (default: ``100``).
        seq_len   : Sequence length used to wrap the scalar input
                    (default: ``1``).
    """
    model.eval()

    x_min = float(X_test.min()) - 1.0
    x_max = float(X_test.max()) + 1.0
    X_plot_np = np.linspace(x_min, x_max, 400).reshape(-1, 1).astype(np.float32)
    # Shape: (N, seq_len, input_size=1)
    X_plot = torch.from_numpy(X_plot_np).unsqueeze(1).expand(-1, seq_len, -1)

    with torch.no_grad():
        preds = [model(X_plot).squeeze().numpy() for _ in range(n_samples)]

    preds      = np.array(preds)        # (n_samples, N)
    mean_preds = preds.mean(axis=0)
    std_preds  = preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Left panel: prediction + uncertainty bands
    ax = axes[0]
    x_line = X_plot_np.squeeze()
    ax.fill_between(x_line,
                    mean_preds - 2.0 * std_preds,
                    mean_preds + 2.0 * std_preds,
                    color='orange', alpha=0.25, label='±2 std')
    ax.fill_between(x_line,
                    mean_preds - std_preds,
                    mean_preds + std_preds,
                    color='orange', alpha=0.50, label='±1 std')
    ax.plot(x_line, mean_preds, 'r-', linewidth=2, label='Predictive mean')

    if X_test is not None and y_test is not None:
        ax.plot(
            X_test.squeeze().numpy(),
            y_test.squeeze().numpy(),
            'k--', alpha=0.5, label='True function',
        )
    if X_train is not None and y_train is not None:
        ax.scatter(
            X_train.squeeze().numpy(),
            y_train.squeeze().numpy(),
            c='steelblue', s=20, zorder=5, label='Training data',
        )

    ax.set_title('Prediction with Uncertainty Bands')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right panel: standard deviation (epistemic uncertainty)
    ax2 = axes[1]
    sc = ax2.scatter(x_line, std_preds, c=std_preds, cmap='plasma', s=6)
    ax2.set_title('Predictive Std Dev (Model Uncertainty)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('std')
    fig.colorbar(sc, ax=ax2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 3.  Decision boundary
# =============================================================================

def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title:    str,
    task:     str = 'binary',
    n_samples: int = 50,
    seq_len:   int = 1,
) -> None:
    """
    Plot the decision boundary and epistemic uncertainty for a classification RNN.

    Args:
        model     : Trained ``RNNModel`` instance.
        X         : Input feature matrix of shape ``(N, 2)`` (2-D features only).
        y         : Integer class labels of shape ``(N,)``.
        title     : Figure title.
        task      : ``'binary'`` or ``'multiclass'`` (default: ``'binary'``).
        n_samples : Number of stochastic forward passes (default: ``50``).
        seq_len   : Sequence length wrapper for each input point (default: ``1``).
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05),
    )
    grid     = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    grid_seq = grid.unsqueeze(1).expand(-1, seq_len, -1)  # (N, seq_len, 2)

    model.eval()
    with torch.no_grad():
        preds = []
        for _ in range(n_samples):
            out = model(grid_seq)
            if task == 'binary':
                probs = torch.sigmoid(out).numpy()
            else:
                probs = torch.softmax(out, dim=1).numpy()
            preds.append(probs)

    preds      = np.array(preds)   # (n_samples, N, C)
    mean_preds = preds.mean(axis=0)
    std_preds  = preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    ax = axes[0]
    if task == 'binary':
        Z_mean   = mean_preds[:, 0].reshape(xx.shape)
        contour  = ax.contourf(xx, yy, Z_mean, levels=20, cmap='RdBu', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu')
    else:
        Z_mean   = np.argmax(mean_preds, axis=1).reshape(xx.shape)
        contour  = ax.contourf(xx, yy, Z_mean, alpha=0.8, cmap='viridis')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
    ax.set_title('Decision Boundary (Predictive Mean)')
    fig.colorbar(contour, ax=ax)

    ax2  = axes[1]
    Z_std = (
        std_preds[:, 0] if task == 'binary' else std_preds.mean(axis=1)
    ).reshape(xx.shape)
    c2 = ax2.contourf(xx, yy, Z_std, levels=20, cmap='plasma', alpha=0.8)
    ax2.scatter(X[:, 0], X[:, 1], c='white', edgecolors='k', s=20, alpha=0.5)
    ax2.set_title('Model Uncertainty (Variance Across Runs)')
    fig.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 4.  Sequence prediction
# =============================================================================

def plot_sequence_prediction(
    model,
    X_seq:     torch.Tensor,
    y_true:    torch.Tensor,
    title:     str,
    n_display: int = 5,
) -> None:
    """
    Plot true versus predicted sequences for a sequence-to-sequence RNN.

    Args:
        model     : Trained ``RNNModel`` instance.
        X_seq     : Input tensor of shape ``(N, T, features)``.
        y_true    : Ground-truth tensor of shape ``(N, T)`` or ``(N,)``.
        title     : Figure title.
        n_display : Number of samples to display (default: ``5``).
    """
    model.eval()
    with torch.no_grad():
        y_pred = model.forward_sequence(X_seq[:n_display]).squeeze(-1).numpy()

    y_true_np = (
        y_true[:n_display].squeeze(-1).numpy()
        if y_true.dim() > 1
        else y_true[:n_display].numpy()
    )
    T = y_pred.shape[1]

    fig, axes = plt.subplots(n_display, 1, figsize=(12, 3 * n_display), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    if n_display == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(range(T), y_true_np[i], 'k--', label='True',      linewidth=1.5)
        ax.plot(range(T), y_pred[i],    'r-',  label='Predicted',  linewidth=1.5)
        ax.set_ylabel(f'Sample {i + 1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5.  Training helper
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
    Generic training loop for ``RNNModel``.

    Args:
        model       : ``RNNModel`` instance.
        X_train     : Input tensor of shape ``(N, T, features)``.
        y_train     : Target tensor.
        optimizer   : A PyTorch optimiser (e.g. ``torch.optim.Adam``).
        loss_name   : Name of the loss function (see ``losses.py``).
                      Default: ``'mse'``.
        epochs      : Number of training epochs (default: ``500``).
        print_every : Print loss every *N* epochs (default: ``100``).
        **loss_kwargs: Extra keyword arguments forwarded to the loss function
                       (e.g. ``delta=0.5`` for Huber).

    Returns:
        A list of ``(epoch, loss_value)`` tuples recorded at each printed
        epoch and at epoch 1.
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
            print(
                f"  Epoch {epoch:>5}/{epochs}"
                f"  |  {loss_name.upper()} = {loss_val:.6f}"
            )
            history.append((epoch, loss_val))

    return history


# =============================================================================
# 6.  Training history plot
# =============================================================================

def plot_training_history(
    history:   list[tuple[int, float]],
    loss_name: str = 'loss',
) -> None:
    """
    Plot the training loss curve.

    Args:
        history   : List of ``(epoch, loss_value)`` tuples as returned by
                    ``train_model``.
        loss_name : Label for the y-axis (default: ``'loss'``).
    """
    epochs, losses = zip(*history)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, 'b-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel(loss_name.upper())
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()