# nf_example.py
"""
Complete usage example for the ANFIS (Neuro-Fuzzy) framework.

This script demonstrates:
1. Creating a synthetic regression dataset
2. Building and training an ANFIS model
3. Visualising membership functions and regression surface
4. Exporting to JSON for Arduino code generation
5. Generating Arduino C++ code
"""

import numpy as np
import torch
import torch.nn as nn

from nf_model import ANFISModel
from nf_utils import (
    train_model,
    plot_training_history,
    plot_membership_functions,
    plot_regression_surface,
    export_to_json,
)
from nf_cpp_generator import generate_ino


# ---------------------------------------------------------------------------
# 1. Synthetic dataset: 2-D nonlinear function
# ---------------------------------------------------------------------------
np.random.seed(42)
n_samples = 300

X = np.random.uniform(-2, 2, size=(n_samples, 2)).astype(np.float32)
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(n_samples)
y = y.astype(np.float32).reshape(-1, 1)

X_train = torch.from_numpy(X[:250])
y_train = torch.from_numpy(y[:250])
X_test  = torch.from_numpy(X[250:])
y_test  = torch.from_numpy(y[250:])

print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
print(f"Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")

# ---------------------------------------------------------------------------
# 2. Build ANFIS model
# ---------------------------------------------------------------------------
model = ANFISModel(
    n_inputs=2,
    n_terms=5,          # 5 linguistic terms per input → 5² = 25 rules
    mf_type='gaussian',
    dense_layers=None,  # No post-processing head (scalar output)
)

print(f"\nANFIS architecture:")
print(f"  Inputs : {model.n_inputs}")
print(f"  Terms  : {model.n_terms}")
print(f"  Rules  : {model.n_rules}")
print(f"  MF type: {model.mf_type}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params}")

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

history = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    optimizer=optimizer,
    loss_name='mse',
    epochs=800,
    print_every=100,
)

plot_training_history(history, loss_name='mse')

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()
    print(f"\nTest MSE: {test_mse:.6f}")

# ---------------------------------------------------------------------------
# 5. Visualise
# ---------------------------------------------------------------------------
plot_membership_functions(model, x_range=(-2.5, 2.5))
plot_regression_surface(model, X_train.numpy(), y_train.numpy(),
                        title='ANFIS: sin(x₁)·cos(x₂)')

# ---------------------------------------------------------------------------
# 6. Export to JSON
# ---------------------------------------------------------------------------
json_path = 'anfis_model.json'
export_to_json(model, json_path)

# ---------------------------------------------------------------------------
# 7. Generate Arduino code
# ---------------------------------------------------------------------------
generate_ino(
    json_path=json_path,
    output_dir='arduino_anfis',
    board='avr',
    use_flash=True,
    task='regression',
)

# ---------------------------------------------------------------------------
# 8. Rule explanation (interpretability)
# ---------------------------------------------------------------------------
sample = X_test[0]
print(f"\nSample input: {sample.numpy()}")
explanations = model.get_rule_explanation(sample, top_k=3)
print("Top-3 active rules:")
for exp in explanations:
    print(f"  Rule {exp['rule_id']:2d}: {exp['antecedent']}")
    print(f"           strength={exp['strength']:.4f}, consequent={exp['consequent']:.4f}")
