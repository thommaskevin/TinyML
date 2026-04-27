# utils.py
"""
Utility functions for Spiking Neural Network experiments.

Provides:
  - export_to_json  : serialise a trained SpikingModel to a JSON file
                      (mirrors the BNN export format).
  - plot_decision_boundary : 2-D classification decision surface with
                             uncertainty estimated via stochastic inference.
  - plot_regression_uncertainty : 1-D regression with predictive bands.
  - evaluate_accuracy : accuracy and confusion-matrix helper.
"""

import json
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from layers import SpikingLinear, SpikingConv2d, LeakyReadout
from model import SpikingModel


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_to_json(model: SpikingModel, filepath: str) -> None:
    """
    Export a trained SpikingModel to a JSON file.

    The JSON schema mirrors the BNN convention so that the downstream
    C++ generator can share the same loading logic:

    {
      "architecture": [ { "type": ..., "in_features": ..., ... }, ... ],
      "parameters":   { "layer_N_weight": [[...]], "layer_N_bias": [...] }
    }

    Parameters
    ----------
    model : SpikingModel
    filepath : str
        Destination path (created/overwritten).
    """
    arch = []
    params: dict = {}

    for i, layer in enumerate(model.layers):
        info: dict = {'type': layer.__class__.__name__}

        if isinstance(layer, SpikingLinear):
            info.update({
                'in_features':  layer.in_features,
                'out_features': layer.out_features,
                'beta':         layer.lif.beta,
                'threshold':    layer.lif.threshold,
                'reset_mode':   layer.lif.reset_mode,
            })
            params[f'layer_{i}_weight'] = layer.fc.weight.detach().cpu().tolist()
            if layer.fc.bias is not None:
                params[f'layer_{i}_bias'] = layer.fc.bias.detach().cpu().tolist()

        elif isinstance(layer, SpikingConv2d):
            info.update({
                'in_channels':  layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size':  layer.kernel_size,
                'stride':       layer.stride,
                'padding':      layer.padding,
                'beta':         layer.lif.beta,
                'threshold':    layer.lif.threshold,
                'reset_mode':   layer.lif.reset_mode,
            })
            params[f'layer_{i}_weight'] = layer.conv.weight.detach().cpu().tolist()
            if layer.conv.bias is not None:
                params[f'layer_{i}_bias'] = layer.conv.bias.detach().cpu().tolist()

        elif isinstance(layer, LeakyReadout):
            info.update({
                'in_features':  layer.in_features,
                'out_features': layer.out_features,
                'beta':         layer.beta,
            })
            params[f'layer_{i}_weight'] = layer.fc.weight.detach().cpu().tolist()
            if layer.fc.bias is not None:
                params[f'layer_{i}_bias'] = layer.fc.bias.detach().cpu().tolist()

        elif isinstance(layer, nn.Flatten):
            info['type'] = 'Flatten'

        arch.append(info)

    data = {
        'num_steps': model.num_steps,
        'encoding':  model.encoding,
        'architecture': arch,
        'parameters':   params,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[export_to_json] Model saved to '{filepath}'.")


# ---------------------------------------------------------------------------
# Stochastic inference (rate-coded uncertainty)
# ---------------------------------------------------------------------------

def _stochastic_predict(model: SpikingModel, x_tensor: torch.Tensor,
                        n_samples: int, task: str) -> tuple:
    """
    Run n_samples forward passes and return (mean, std) of predictions.

    For SNNs the stochasticity comes from the rate-coded Bernoulli
    encoding: each pass samples a different spike train from the same
    input intensity, yielding a natural estimate of predictive variance.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_tensor)       # (batch, out_features)
            if task == 'binary':
                p = torch.sigmoid(out).cpu().numpy()
            elif task == 'multiclass':
                p = torch.softmax(out, dim=-1).cpu().numpy()
            else:
                p = out.cpu().numpy()
            all_preds.append(p)

    arr = np.array(all_preds)           # (n_samples, batch, out_features)
    return arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------------------
# Visualisation — decision boundary
# ---------------------------------------------------------------------------

def plot_decision_boundary(model: SpikingModel, X: np.ndarray, y: np.ndarray,
                           title: str = 'SNN Decision Boundary',
                           task: str = 'binary',
                           n_samples: int = 50) -> None:
    """
    Plot the 2-D decision boundary and epistemic uncertainty.

    Uses stochastic (rate-coded) inference to estimate the predictive
    distribution. The left panel shows the mean prediction; the right
    panel shows the standard deviation as a proxy for uncertainty.

    Parameters
    ----------
    model : SpikingModel
    X : ndarray of shape (N, 2)
    y : ndarray of shape (N,)
    title : str
    task : str   — 'binary' or 'multiclass'
    n_samples : int
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    grid_np = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid = torch.from_numpy(grid_np)

    # Normalise to [0, 1] if using rate encoding
    if model.encoding == 'rate':
        mins = grid.min(0).values
        maxs = grid.max(0).values
        grid = (grid - mins) / (maxs - mins + 1e-8)

    mean_preds, std_preds = _stochastic_predict(model, grid, n_samples, task)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # --- Mean prediction ---
    ax = axes[0]
    if task == 'binary':
        Z_mean = mean_preds[:, 0].reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z_mean, levels=20,
                              cmap='RdBu', alpha=0.8)
    else:
        Z_mean = np.argmax(mean_preds, axis=1).reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z_mean, alpha=0.8, cmap='viridis')

    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
               cmap='RdBu' if task == 'binary' else 'viridis', s=20)
    ax.set_title('Decision Boundary (Predictive Mean)')
    fig.colorbar(contour, ax=ax)

    # --- Uncertainty ---
    ax = axes[1]
    if task == 'binary':
        Z_std = std_preds[:, 0].reshape(xx.shape)
    else:
        Z_std = std_preds.mean(axis=1).reshape(xx.shape)

    contour_std = ax.contourf(xx, yy, Z_std, levels=20,
                              cmap='plasma', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c='white', edgecolors='k', s=15, alpha=0.5)
    ax.set_title('Predictive Uncertainty (Firing-Rate Variance)')
    fig.colorbar(contour_std, ax=ax)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation — regression uncertainty
# ---------------------------------------------------------------------------

def plot_regression_uncertainty(model: SpikingModel,
                                X_train: torch.Tensor, y_train: torch.Tensor,
                                X_test: torch.Tensor, y_test: torch.Tensor,
                                title: str = 'SNN Regression',
                                n_samples: int = 100) -> None:
    """
    Plot predictive mean and ±1 / ±2 std bands for a 1-D regression task.

    Parameters
    ----------
    model : SpikingModel
    X_train, y_train, X_test, y_test : Tensors
    title : str
    n_samples : int
    """
    model.eval()
    X_plot = torch.linspace(
        X_test.min().item() - 1, X_test.max().item() + 1, 400
    ).view(-1, 1)

    # Normalise if rate encoding
    if model.encoding == 'rate':
        x_min_val = X_plot.min()
        x_max_val = X_plot.max()
        X_plot_enc = (X_plot - x_min_val) / (x_max_val - x_min_val + 1e-8)
    else:
        X_plot_enc = X_plot

    mean_preds, std_preds = _stochastic_predict(
        model, X_plot_enc, n_samples, 'regression'
    )
    mean_preds = mean_preds.squeeze()
    std_preds = std_preds.squeeze()
    x_np = X_plot.squeeze().numpy()

    plt.figure(figsize=(10, 6))
    plt.fill_between(x_np,
                     mean_preds - 2 * std_preds,
                     mean_preds + 2 * std_preds,
                     color='orange', alpha=0.3,
                     label='Predictive Uncertainty (±2 std)')
    plt.fill_between(x_np,
                     mean_preds - std_preds,
                     mean_preds + std_preds,
                     color='orange', alpha=0.5, label='±1 std')
    plt.plot(x_np, mean_preds, 'r-', linewidth=2, label='Predictive Mean')
    plt.plot(X_test.squeeze().numpy(), y_test.squeeze().numpy(),
             'k--', alpha=0.5, label='True Function')
    plt.scatter(X_train.squeeze().numpy(), y_train.squeeze().numpy(),
                c='blue', s=20, label='Training Data')

    plt.title(title, fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_accuracy(model: SpikingModel, X: torch.Tensor,
                      y: torch.Tensor, task: str = 'multiclass',
                      batch_size: int = 256) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    model : SpikingModel
    X : Tensor of shape (N, features)
    y : Tensor of shape (N,)
    task : 'multiclass' or 'binary'
    batch_size : int

    Returns
    -------
    accuracy : float  in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            out = model(xb)
            if task == 'multiclass':
                preds = out.argmax(dim=-1)
            else:
                preds = (out.squeeze(-1) > 0).long()
            correct += (preds == yb).sum().item()
            total += len(yb)

    return correct / total


def plot_confusion_matrix(model: SpikingModel, X: torch.Tensor,
                          y: torch.Tensor, class_names: Optional[list] = None,
                          title: str = 'Confusion Matrix') -> None:
    """
    Display a confusion matrix for multiclass classification.

    Parameters
    ----------
    model : SpikingModel
    X, y : Tensors
    class_names : list of str or None
    title : str
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        out = model(X)
        all_preds = out.argmax(dim=-1).cpu().numpy()

    y_np = y.cpu().numpy()
    cm = confusion_matrix(y_np, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.show()