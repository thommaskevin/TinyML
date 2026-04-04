# utils.py
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from layers import BayesianLinear, BayesianConv2d

def export_to_json(model, filepath):
    """
    Export a trained BayesianModel to a JSON file.
    
    The JSON contains:
        - architecture: list of layer descriptions
        - parameters: dictionary mapping parameter names to lists
    """
    arch = []
    params = {}
    for i, layer in enumerate(model.layers):
        layer_info = {'type': layer.__class__.__name__}
        
        if isinstance(layer, BayesianLinear):
            layer_info['in_features'] = layer.in_features
            layer_info['out_features'] = layer.out_features
            # Optionally store prior_var or other attributes
            params[f'layer_{i}_weight_mu'] = layer.weight_mu.detach().cpu().tolist()
            params[f'layer_{i}_weight_logvar'] = layer.weight_logvar.detach().cpu().tolist()
            params[f'layer_{i}_bias_mu'] = layer.bias_mu.detach().cpu().tolist()
            params[f'layer_{i}_bias_logvar'] = layer.bias_logvar.detach().cpu().tolist()
        
        elif isinstance(layer, BayesianConv2d):
            layer_info['in_channels'] = layer.in_channels
            layer_info['out_channels'] = layer.out_channels
            layer_info['kernel_size'] = layer.kernel_size
            layer_info['stride'] = layer.stride
            layer_info['padding'] = layer.padding
            params[f'layer_{i}_weight_mu'] = layer.weight_mu.detach().cpu().tolist()
            params[f'layer_{i}_weight_logvar'] = layer.weight_logvar.detach().cpu().tolist()
            params[f'layer_{i}_bias_mu'] = layer.bias_mu.detach().cpu().tolist()
            params[f'layer_{i}_bias_logvar'] = layer.bias_logvar.detach().cpu().tolist()
        
        elif isinstance(layer, torch.nn.ReLU):
            layer_info['type'] = 'ReLU'
        elif isinstance(layer, torch.nn.Tanh):
            layer_info['type'] = 'Tanh'
        elif isinstance(layer, torch.nn.Sigmoid):
            layer_info['type'] = 'Sigmoid'
        # Add other layer types as needed
        
        arch.append(layer_info)
    
    data = {'architecture': arch, 'parameters': params}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def elbo_loss(output, target, model, kl_weight=1.0, likelihood='regression'):
    if likelihood == 'regression':
        if output.shape != target.shape:
            target = target.view_as(output)
        nll = F.mse_loss(output, target, reduction='mean')
    elif likelihood == 'binary':
        if output.dim() > 1 and output.size(1) > 1:
            output = output.squeeze()
        target = target.view_as(output).float()
        nll = F.binary_cross_entropy_with_logits(output, target, reduction='mean')
    elif likelihood == 'classification':
        nll = F.cross_entropy(output, target, reduction='mean')
    else:
        raise ValueError("likelihood must be 'classification', 'binary' or 'regression'")

    kl = model.kl_loss() / output.size(0)
    return nll + kl_weight * kl


# =============================================================================
# 3. Visualization Functions
# =============================================================================
def plot_decision_boundary(model, X, y, title, task='binary', n_samples=50):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    model.eval()
    with torch.no_grad():
        preds = []
        for _ in range(n_samples):
            out = model(grid, sample=True)
            if task == 'binary':
                probs = torch.sigmoid(out).numpy()
            else:
                probs = torch.softmax(out, dim=1).numpy()
            preds.append(probs)
            
    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    ax = axes[0]
    if task == 'binary':
        Z_mean = mean_preds[:, 0].reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z_mean, levels=20, cmap="RdBu", alpha=0.8)
    else:
        Z_mean = np.argmax(mean_preds, axis=1).reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z_mean, alpha=0.8, cmap="viridis")
        
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="RdBu" if task=='binary' else "viridis")
    ax.set_title("Decision Boundary (Predictive Mean)")
    fig.colorbar(contour, ax=ax)

    ax = axes[1]
    if task == 'binary':
        Z_std = std_preds[:, 0].reshape(xx.shape)
    else:
        # Corrected: Using axis=1 for multiclass uncertainty calculation
        Z_std = std_preds.mean(axis=1).reshape(xx.shape)
        
    contour_std = ax.contourf(xx, yy, Z_std, levels=20, cmap="plasma", alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c='white', edgecolors='k', s=20, alpha=0.5)
    ax.set_title("Model Uncertainty (Epistemic Variance)")
    fig.colorbar(contour_std, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_regression_uncertainty(model, X_train, y_train, X_test, y_test, title, n_samples=100):
    model.eval()
    X_plot = torch.linspace(X_test.min().item() - 1, X_test.max().item() + 1, 400).view(-1, 1)
    
    with torch.no_grad():
        preds = []
        for _ in range(n_samples):
            out = model(X_plot, sample=True)
            preds.append(out.numpy())
            
    preds = np.array(preds).squeeze()
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(X_plot.squeeze().numpy(), 
                     mean_preds - 2 * std_preds, 
                     mean_preds + 2 * std_preds, 
                     color='orange', alpha=0.3, label='Epistemic Uncertainty (+/- 2 std)')
    plt.fill_between(X_plot.squeeze().numpy(), 
                     mean_preds - std_preds, 
                     mean_preds + std_preds, 
                     color='orange', alpha=0.5, label='+/- 1 std')
    
    plt.plot(X_plot.squeeze().numpy(), mean_preds, 'r-', linewidth=2, label='Predictive Mean')
    plt.plot(X_test.squeeze().numpy(), y_test.squeeze().numpy(), 'k--', alpha=0.5, label='True Function')
    plt.scatter(X_train.squeeze().numpy(), y_train.squeeze().numpy(), c='blue', s=20, label='Training Data')
    
    plt.title(title, fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()