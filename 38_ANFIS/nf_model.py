# nf_model.py
"""
Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

Architecture
------------
The network follows the canonical five-layer ANFIS structure:

1. Fuzzification   — membership degrees μ(x)     (learnable MF params)
2. Rule firing     — T-norm product w            (combinatorial rule base)
3. Normalisation   — w̄ = w / Σw                  (softmax-like)
4. Consequent      — linear f_i(x) per rule      (learnable coefficients)
5. Defuzzification — ŷ = Σ w̄_i · f_i             (weighted average)

An optional dense head can be appended after defuzzification for
arbitrary output shapes (multiclass, multi-output regression, etc.).

Input / output shapes
---------------------
- Input  : ``(batch, n_inputs)``  (crisp features)
- Output : ``(batch, out_features)``  (after optional dense head)

Configuration format
--------------------
``anfis_config`` — dict with keys:

.. code-block:: python

    {
        'n_inputs':   2,        # number of crisp input features
        'n_terms':    3,        # linguistic terms per input (e.g. low/med/high)
        'mf_type':    'gaussian',
        'n_outputs':  1,        # scalar output before dense head
    }

``dense_layers`` — list of dicts (optional post-processing), same format
as LNNModel.

References
----------
Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference
    System. *IEEE Transactions on Systems, Man, and Cybernetics*,
    23(3), 665–685.
"""

import torch
import torch.nn as nn

from nf_layers import (
    FuzzifyLayer,
    RuleLayer,
    NormalizeLayer,
    ConsequentLayer,
    DefuzzifyLayer,
    DenseLayer,
)


class ANFISModel(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS).

    The forward pass implements the standard first-order Sugeno fuzzy
    inference with a product T-norm and weighted-average defuzzification.
    All parameters (membership-function centres/widths and consequent
    linear coefficients) are differentiable and trained by back-propagation.
    """

    def __init__(
        self,
        n_inputs:       int,
        n_terms:        int,
        mf_type:        str = 'gaussian',
        dense_layers:   list[dict] | None = None,
    ) -> None:
        super().__init__()

        self.n_inputs   = n_inputs
        self.n_terms    = n_terms
        self.mf_type    = mf_type
        self.n_rules    = n_terms ** n_inputs

        # ----- ANFIS core layers -----
        self.fuzzify    = FuzzifyLayer(n_inputs, n_terms, mf_type)
        self.rules      = RuleLayer(n_inputs, n_terms)
        self.normalize  = NormalizeLayer()
        self.consequent = ConsequentLayer(n_inputs, self.n_rules)
        self.defuzzify  = DefuzzifyLayer()

        # ----- Optional dense head -----
        self.dense_head = nn.ModuleList()
        prev_size = 1  # defuzzify outputs a scalar per sample

        if dense_layers:
            for cfg in dense_layers:
                layer = DenseLayer(
                    in_features=prev_size,
                    out_features=cfg['out_features'],
                    activation=cfg.get('activation', 'linear'),
                    bias=cfg.get('bias', True),
                )
                self.dense_head.append(layer)
                prev_size = cfg['out_features']

        self.dense_configs = dense_layers or []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full ANFIS + optional dense head.

        Args:
            x : Input tensor of shape ``(batch, n_inputs)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        # Layer 1: Fuzzification
        mu = self.fuzzify(x)           # (batch, n_inputs, n_terms)

        # Layer 2: Rule firing (product T-norm)
        w = self.rules(mu)             # (batch, n_rules)

        # Layer 3: Normalisation
        w_bar = self.normalize(w)      # (batch, n_rules)

        # Layer 4: Consequent
        f = self.consequent(x)         # (batch, n_rules)

        # Layer 5: Defuzzification
        y = self.defuzzify(w_bar, f)   # (batch, 1)

        # Optional dense head
        for layer in self.dense_head:
            y = layer(y)

        return y

    # ------------------------------------------------------------------
    def get_firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the raw (unnormalised) firing strength of every rule.

        Useful for interpretability / rule extraction.

        Args:
            x : (batch, n_inputs)

        Returns:
            w : (batch, n_rules)
        """
        with torch.no_grad():
            mu = self.fuzzify(x)
            w  = self.rules(mu)
        return w

    # ------------------------------------------------------------------
    def get_rule_explanation(self, x: torch.Tensor, top_k: int = 3) -> list[dict]:
        """
        Return human-readable explanations for the top-k most active rules.

        Args:
            x     : (batch, n_inputs) or (n_inputs,)
            top_k : Number of top rules to return.

        Returns:
            List of dicts with keys ``'rule_id'``, ``'strength'``,
            ``'antecedent'``, ``'consequent'``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            mu = self.fuzzify(x)           # (1, n_inputs, n_terms)
            w  = self.rules(mu).squeeze(0) # (n_rules,)
            wb = self.normalize(w.unsqueeze(0)).squeeze(0)
            f  = self.consequent(x).squeeze(0)

        # Build term names
        term_names = ['low', 'medium', 'high']
        if self.n_terms > 3:
            term_names = [f't{i}' for i in range(self.n_terms)]

        explanations = []
        top_indices = torch.topk(wb, min(top_k, self.n_rules)).indices

        for idx in top_indices:
            idx = int(idx)
            # Decode rule index into term indices per input
            terms = []
            tmp = idx
            for inp in range(self.n_inputs):
                t = tmp % self.n_terms
                terms.append(term_names[min(t, len(term_names)-1)])
                tmp //= self.n_terms

            antecedent = ' AND '.join(
                f"x{j+1} is {terms[j]}" for j in range(self.n_inputs)
            )

            explanations.append({
                'rule_id':     idx,
                'strength':    float(wb[idx]),
                'antecedent':  antecedent,
                'consequent':  float(f[idx]),
            })

        return explanations
