# nf_layers.py
"""
Layer definitions for the Adaptive Neuro-Fuzzy Inference System (ANFIS).

Contents
--------
- Membership functions (Gaussian, Triangular, Bell, Sigmoid)
- FuzzifyLayer    : fuzzification with learnable parameters
- RuleLayer       : T-norm (product) aggregation of antecedents
- NormalizeLayer  : normalisation of firing strengths
- ConsequentLayer : linear consequent for each rule
- DefuzzifyLayer  : weighted sum defuzzification
- DenseLayer      : optional post-processing dense head

Background
----------
ANFIS maps crisp inputs to a crisp output through five layers:

1. Fuzzification   : μ(x)  — membership degrees
2. Rule firing     : w_i   — product of membership degrees per rule
3. Normalisation   : w̄_i   — w_i / Σ w_j
4. Consequent      : f_i   — linear combination of inputs per rule
5. Defuzzification : ŷ     — Σ w̄_i · f_i

All parameters (centres, widths, consequent coefficients) are
learnable via gradient descent.

References
----------
Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference
    System. *IEEE Transactions on Systems, Man, and Cybernetics*,
    23(3), 665–685.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Membership Functions
# =============================================================================

def gaussian_mf(x, center, sigma):
    """Gaussian membership function: exp(-0.5 * ((x-c)/σ)²)."""
    return torch.exp(-0.5 * ((x - center) / sigma.clamp(min=1e-6)) ** 2)


def triangular_mf(x, a, b, c):
    """Triangular membership function (piecewise-linear)."""
    left  = torch.clamp((x - a) / (b - a + 1e-6), 0.0, 1.0)
    right = torch.clamp((c - x) / (c - b + 1e-6), 0.0, 1.0)
    return torch.min(left, right)


def bell_mf(x, a, b, c):
    """Generalised bell membership function: 1 / (1 + |(x-c)/a|^(2b))."""
    return 1.0 / (1.0 + torch.abs((x - c) / a.clamp(min=1e-6)) ** (2 * b))


def sigmoid_mf(x, a, c):
    """Sigmoid membership function: 1 / (1 + exp(-a·(x-c)))."""
    return torch.sigmoid(a * (x - c))


# Registry
MF_FUNCTIONS = {
    'gaussian':  gaussian_mf,
    'triangular': triangular_mf,
    'bell':      bell_mf,
    'sigmoid':   sigmoid_mf,
}


# =============================================================================
# 1. Fuzzification Layer
# =============================================================================

class FuzzifyLayer(nn.Module):
    """
    Fuzzification layer: computes membership degrees for every input
    variable and every linguistic term.

    For each input feature j (j = 1 … n_in) and each term k
    (k = 1 … n_terms) we learn the parameters of a membership function.

    Output shape: (batch, n_in, n_terms)

    Args:
        n_inputs   : Number of crisp input features.
        n_terms    : Number of linguistic terms per input (e.g. 3 for
                     {low, medium, high}).
        mf_type    : Membership function type — ``'gaussian'``,
                     ``'triangular'``, ``'bell'``, or ``'sigmoid'``.
    """

    def __init__(self, n_inputs: int, n_terms: int, mf_type: str = 'gaussian'):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_terms  = n_terms
        self.mf_type  = mf_type.lower()

        if self.mf_type not in MF_FUNCTIONS:
            raise ValueError(f"Unknown mf_type '{mf_type}'.")

        # Learnable parameters — shape (n_inputs, n_terms)
        if self.mf_type == 'gaussian':
            self.center = nn.Parameter(torch.randn(n_inputs, n_terms))
            self.sigma  = nn.Parameter(torch.ones(n_inputs, n_terms))
        elif self.mf_type == 'triangular':
            self.a = nn.Parameter(torch.randn(n_inputs, n_terms))
            self.b = nn.Parameter(torch.randn(n_inputs, n_terms))
            self.c = nn.Parameter(torch.randn(n_inputs, n_terms))
        elif self.mf_type == 'bell':
            self.a = nn.Parameter(torch.ones(n_inputs, n_terms))
            self.b = nn.Parameter(torch.ones(n_inputs, n_terms))
            self.c = nn.Parameter(torch.randn(n_inputs, n_terms))
        elif self.mf_type == 'sigmoid':
            self.a = nn.Parameter(torch.ones(n_inputs, n_terms))
            self.c = nn.Parameter(torch.randn(n_inputs, n_terms))

        self._init_params()

    def _init_params(self):
        """Initialise centres spread uniformly over [-1, 1]."""
        with torch.no_grad():
            if hasattr(self, 'center'):
                # Spread centres uniformly
                for j in range(self.n_inputs):
                    self.center[j, :] = torch.linspace(-1, 1, self.n_terms)
            if hasattr(self, 'c'):
                for j in range(self.n_inputs):
                    self.c[j, :] = torch.linspace(-1, 1, self.n_terms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, n_inputs)

        Returns:
            mu : (batch, n_inputs, n_terms)
        """
        x = x.unsqueeze(-1)  # (batch, n_inputs, 1)

        if self.mf_type == 'gaussian':
            mu = gaussian_mf(x, self.center, self.sigma)
        elif self.mf_type == 'triangular':
            # Ensure ordering a < b < c via sorting
            abc = torch.stack([self.a, self.b, self.c], dim=-1)
            abc_sorted, _ = torch.sort(abc, dim=-1)
            a, b, c = abc_sorted[..., 0], abc_sorted[..., 1], abc_sorted[..., 2]
            mu = triangular_mf(x, a, b, c)
        elif self.mf_type == 'bell':
            mu = bell_mf(x, self.a, self.b, self.c)
        elif self.mf_type == 'sigmoid':
            mu = sigmoid_mf(x, self.a, self.c)

        return mu


# =============================================================================
# 2. Rule Layer (T-norm product)
# =============================================================================

class RuleLayer(nn.Module):
    """
    Computes the firing strength of every fuzzy rule.

    For a system with *n_inputs* variables and *n_terms* terms per
    variable, the total number of rules is ``n_rules = n_terms ** n_inputs``
    (grid / complete rule base).  Each rule is the product of one
    membership degree from each input dimension.

    Output shape: (batch, n_rules)

    Args:
        n_inputs : Number of input features.
        n_terms  : Number of linguistic terms per input.
    """

    def __init__(self, n_inputs: int, n_terms: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_terms  = n_terms
        self.n_rules  = n_terms ** n_inputs

        # Pre-compute the index mapping: for each rule r, which term
        # index is used for each input dimension.
        # shape: (n_rules, n_inputs)
        indices = []
        for r in range(self.n_rules):
            idx = []
            tmp = r
            for _ in range(n_inputs):
                idx.append(tmp % n_terms)
                tmp //= n_terms
            indices.append(idx)
        self.register_buffer('rule_indices', torch.tensor(indices, dtype=torch.long))

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu : (batch, n_inputs, n_terms)

        Returns:
            w : (batch, n_rules)
        """
        batch = mu.size(0)
        # Gather the membership degrees used by each rule
        # mu_selected : (batch, n_rules, n_inputs)
        mu_selected = torch.stack([
            mu[:, range(self.n_inputs), self.rule_indices[r]]
            for r in range(self.n_rules)
        ], dim=1)

        # Product T-norm
        w = torch.prod(mu_selected, dim=2)  # (batch, n_rules)
        return w


# =============================================================================
# 3. Normalisation Layer
# =============================================================================

class NormalizeLayer(nn.Module):
    """
    Normalises firing strengths so they sum to 1 per sample.

    w̄_i = w_i / Σ_j w_j
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w : (batch, n_rules)

        Returns:
            w_bar : (batch, n_rules)
        """
        sum_w = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return w / sum_w


# =============================================================================
# 4. Consequent Layer (linear per rule)
# =============================================================================

class ConsequentLayer(nn.Module):
    """
    Computes the linear consequent f_i for each rule.

    f_i(x) = p_{i0} + p_{i1}·x_1 + … + p_{in}·x_n

    Args:
        n_inputs : Number of crisp input features.
        n_rules  : Number of fuzzy rules.
    """

    def __init__(self, n_inputs: int, n_rules: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules  = n_rules
        # Coefficients: (n_rules, n_inputs + 1)  [+1 for bias p0]
        self.coeff = nn.Parameter(torch.randn(n_rules, n_inputs + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, n_inputs)

        Returns:
            f : (batch, n_rules)
        """
        # Add bias column
        x_aug = torch.cat([torch.ones(x.size(0), 1, device=x.device), x], dim=1)
        # (batch, n_rules) = (batch, n_inputs+1) @ (n_inputs+1, n_rules).T
        f = x_aug @ self.coeff.t()
        return f


# =============================================================================
# 5. Defuzzification Layer (weighted average)
# =============================================================================

class DefuzzifyLayer(nn.Module):
    """
    Weighted average defuzzification:

        ŷ = Σ_i (w̄_i · f_i)

    Output shape: (batch, 1) for scalar regression.
    """

    def forward(self, w_bar: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_bar : (batch, n_rules)
            f     : (batch, n_rules)

        Returns:
            y : (batch, 1)
        """
        y = (w_bar * f).sum(dim=1, keepdim=True)
        return y


# =============================================================================
# 6. Optional Dense Head (post-defuzzification)
# =============================================================================

class DenseLayer(nn.Module):
    """
    Fully-connected layer with optional activation.

    Args:
        in_features  : Number of input features.
        out_features : Number of output features.
        activation   : One of ``'linear'``, ``'relu'``, ``'tanh'``,
                       ``'sigmoid'``, ``'softmax'``.
        bias         : Whether to include a bias term.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   str  = 'linear',
        bias:         bool = True,
    ) -> None:
        super().__init__()
        self.in_features     = in_features
        self.out_features    = out_features
        self.activation_name = activation.lower()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = self._get_activation()

    def _get_activation(self):
        a = self.activation_name
        if a == 'relu':      return nn.ReLU()
        if a == 'tanh':      return nn.Tanh()
        if a == 'sigmoid':   return nn.Sigmoid()
        if a == 'softmax':   return nn.Softmax(dim=-1)
        if a == 'leaky_relu':return nn.LeakyReLU()
        if a == 'gelu':      return nn.GELU()
        if a == 'swish':     return nn.SiLU()
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
