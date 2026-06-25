# cpp_generator.py
"""
Arduino C++ code generator for Causal Tree (CT) models.

Design notes
------------
A fitted Causal Tree reduces to a sequence of binary comparisons (feature ≤
threshold) that route each sample to a leaf, followed by a lookup of the
pre-computed CATE estimate stored at that leaf.  This is structurally
simpler than the LTC generator: no matrix multiplications, no ODE
integration.  The generated C++ is a single nested if-else tree, which
compiles to extremely compact machine code on AVR and ARM Cortex-M targets.

Supported tasks
---------------
- ``'regression'``  : returns a float τ̂ (CATE estimate).
- ``'binary'``      : returns a float τ̂ (risk difference); caller compares to 0.
- ``'multiclass'``  : returns an int (predicted class index = argmax τ̂_k).

Bug history and fixes
---------------------
FIX 1 — float literal precision
    All float weights are written with 8 decimal places and the ``f``
    suffix to avoid C implicit double promotion.

FIX 2 — multiclass leaf storage
    Per-class τ̂_k arrays stored as PROGMEM-compatible const float arrays
    with one entry per class; argmax is computed at inference time to avoid
    storing class-index assumptions in flash.

FIX 3 — deep nesting guard
    A depth cap (DEFAULT_MAX_DEPTH = 20) prevents runaway if-else chains
    that would overflow the AVR stack.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy reference forward pass (mirrors generated C exactly)
# ---------------------------------------------------------------------------

def _np_predict(node_dict: dict, x: np.ndarray, task: str):
    """
    Pure-NumPy forward pass that mirrors the generated C if-else tree.

    Args:
        node_dict : Dict representation of a ``TreeNode`` (from JSON).
        x         : 1-D float64 array of shape ``(n_features,)``.
        task      : ``'regression'``, ``'binary'``, or ``'multiclass'``.

    Returns:
        - regression / binary : float
        - multiclass          : int (argmax of tau array)
    """
    node = node_dict
    while not node['is_leaf']:
        feat = node['feature']
        thr  = node['threshold']
        node = node['left'] if x[feat] <= thr else node['right']

    tau = node['tau']
    if task == 'multiclass':
        return int(np.argmax(tau))
    return float(tau)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoCTGenerator:
    """
    Generates a two-file Arduino library (``CTModel.h`` + ``sketch.ino``)
    from a ``CausalTreeModel`` JSON export.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Directory where the generated files are written.
        board      : Target board: ``'avr'`` (Uno / Mega / Nano) or
                     ``'esp32'`` (ESP32 / ESP32-S3).  On AVR, large
                     arrays of leaf values are stored in PROGMEM.
        use_flash  : Store leaf tau arrays in PROGMEM (AVR only).
        task       : ``'regression'``, ``'binary'``, or ``'multiclass'``.
    """

    DEFAULT_MAX_DEPTH = 20   # guard against excessive nesting

    def __init__(
        self,
        json_path:  str,
        output_dir: str,
        board:      str  = 'avr',
        use_flash:  bool = True,
        task:       str  = 'regression',
    ):
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board.lower()
        self.use_flash  = use_flash and (self.board == 'avr')
        self.task       = task.lower()
        self._data      = None

    # ------------------------------------------------------------------
    def _load(self):
        with open(self.json_path, 'r') as f:
            self._data = json.load(f)

    @property
    def _tree(self):
        return self._data['tree']

    @property
    def _n_classes(self):
        return self._data['hyperparams'].get('n_classes') or 2

    @property
    def _n_features(self):
        """Infer number of features from the first split encountered."""
        def _walk(node):
            if node['is_leaf']:
                return 0
            return max(node['feature'] + 1, _walk(node['left']), _walk(node['right']))
        return _walk(self._tree)

    # ------------------------------------------------------------------
    def generate(self, sample_x: np.ndarray | None = None):
        """
        Generate the Arduino library files.

        Args:
            sample_x : Optional 1-D float64 array of shape ``(n_features,)``
                       used to produce verification values in the sketch.
                       If ``None``, a zero vector is used.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        n_feat = self._n_features
        if sample_x is None:
            np.random.seed(0)
            sample_x = np.random.randn(n_feat).astype(np.float64)

        expected = _np_predict(self._tree, sample_x, self.task)

        header_path = os.path.join(self.output_dir, 'CTModel.h')
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())

        sketch_path = os.path.join(self.output_dir, 'sketch.ino')
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_x, expected))

        print(f"Generated in  : {self.output_dir}  (board: {self.board}, task: {self.task})")
        print(f"n_features    : {n_feat}")
        print(f"Expected output (Python): {expected}")

    # ------------------------------------------------------------------
    def _header(self) -> str:
        lines = [
            '// CTModel.h  —  Auto-generated Causal Tree inference engine',
            '// Do NOT edit manually.',
            '',
            '#pragma once',
            '#include <math.h>',
            '',
        ]
        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']

        if self.task == 'multiclass':
            lines += [
                f'static const int CT_N_CLASSES = {self._n_classes};',
                '',
            ]

        lines += [
            'class CTModel {',
            'public:',
        ]

        if self.task == 'multiclass':
            lines += ['    int predict(float* x);']
        else:
            lines += ['    float predict(float* x);']

        lines += ['};', '']

        # Build the if-else tree body
        body_lines = []
        self._emit_node(self._tree, body_lines, indent=4, depth=0)

        # Method implementation
        if self.task == 'multiclass':
            lines += [
                'int CTModel::predict(float* x) {',
            ]
        else:
            lines += [
                'float CTModel::predict(float* x) {',
            ]
        lines += body_lines
        lines += ['}', '']

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _emit_node(self, node: dict, lines: list, indent: int, depth: int):
        """Recursively emit the if-else tree."""
        pad = ' ' * indent

        if node['is_leaf'] or depth >= self.DEFAULT_MAX_DEPTH:
            tau = node['tau']
            if self.task == 'multiclass':
                # emit array and argmax
                tau_list = tau if isinstance(tau, list) else [float(tau)]
                K = len(tau_list)
                vals = ', '.join(f'{float(v):.8f}f' for v in tau_list)
                lines.append(f'{pad}{{')
                if self.use_flash:
                    lines.append(f'{pad}    static const float tau[{K}] PROGMEM = {{{vals}}};')
                    lines.append(f'{pad}    int best = 0;')
                    lines.append(f'{pad}    for (int k = 1; k < {K}; k++)')
                    lines.append(f'{pad}        if (pgm_read_float(&tau[k]) > pgm_read_float(&tau[best])) best = k;')
                else:
                    lines.append(f'{pad}    const float tau[{K}] = {{{vals}}};')
                    lines.append(f'{pad}    int best = 0;')
                    lines.append(f'{pad}    for (int k = 1; k < {K}; k++)')
                    lines.append(f'{pad}        if (tau[k] > tau[best]) best = k;')
                lines.append(f'{pad}    return best;')
                lines.append(f'{pad}}}')
            else:
                tau_f = float(tau) if not isinstance(tau, list) else float(tau[0])
                lines.append(f'{pad}return {tau_f:.8f}f;')
            return

        feat = node['feature']
        thr  = float(node['threshold'])
        lines.append(f'{pad}if (x[{feat}] <= {thr:.8f}f) {{')
        self._emit_node(node['left'],  lines, indent + 4, depth + 1)
        lines.append(f'{pad}}} else {{')
        self._emit_node(node['right'], lines, indent + 4, depth + 1)
        lines.append(f'{pad}}}')

    # ------------------------------------------------------------------
    def _sketch(self, sample_x: np.ndarray, expected) -> str:
        n_feat   = len(sample_x)
        vals_str = ', '.join(f'{float(v):.8f}f' for v in sample_x)

        if self.task == 'multiclass':
            exp_comment = f'Expected class   : {int(expected)}'
        elif self.task == 'binary':
            exp_comment = (
                f'Expected τ̂      : {float(expected):.8f}\n'
                f' * Expected decision : {"TREAT" if float(expected) > 0 else "DO NOT TREAT"}'
            )
        else:
            exp_comment = f'Expected τ̂      : {float(expected):.8f}'

        lines = [
            '/*',
            ' * CTModel — Arduino verification sketch',
            ' * Generated automatically — do not edit.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * Input n_features  : {n_feat}',
            ' * Input values       : see float array below',
            ' *',
            f' * {exp_comment}',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm the printed value matches the expected value',
            ' * above to at least 5 decimal places.',
            ' */',
            '',
            '#include "CTModel.h"',
            '',
            'CTModel model;',
            '',
            'void setup() {',
            '    Serial.begin(115200);',
            '    while (!Serial);',
            '',
            f'    const int N_FEATURES = {n_feat};',
            f'    float x[N_FEATURES] = {{{vals_str}}};',
            '',
        ]

        if self.task == 'multiclass':
            lines += [
                f'    // Expected class : {int(expected)}',
                '    int output = model.predict(x);',
                '    Serial.print("Predicted class  : "); Serial.println(output);',
            ]
        elif self.task == 'binary':
            lines += [
                f'    // Expected τ̂     : {float(expected):.8f}',
                f'    // Expected decision: {"TREAT" if float(expected) > 0 else "DO NOT TREAT"}',
                '    float output = model.predict(x);',
                '    Serial.print("Predicted tau    : "); Serial.println(output, 8);',
                '    Serial.print("Decision         : "); Serial.println(output > 0.0f ? "TREAT" : "DO NOT TREAT");',
            ]
        else:
            lines += [
                f'    // Expected τ̂     : {float(expected):.8f}',
                '    float output = model.predict(x);',
                '    Serial.print("Predicted tau    : "); Serial.println(output, 8);',
            ]

        lines += [
            '}',
            '',
            'void loop() {',
            '    // Nothing to do here',
            '}',
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public convenience function (mirrors generate_ino from LNN)
# ---------------------------------------------------------------------------

def generate_ino(
    json_path:  str,
    output_dir: str,
    board:      str  = 'avr',
    use_flash:  bool = True,
    task:       str  = 'regression',
):
    """
    Generate Arduino Causal Tree library files from a JSON model export.

    Args:
        json_path  : Path to the JSON produced by ``export_to_json``.
        output_dir : Destination directory for ``CTModel.h`` and ``sketch.ino``.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : Store leaf arrays in PROGMEM (AVR only).
        task       : ``'regression'``, ``'binary'``, or ``'multiclass'``.

    Example::

        generate_ino(
            'json_model/regression_ct.json',
            'arduino_code/regression_ino',
            board='esp32',
            task='regression',
        )
    """
    gen = ArduinoCTGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate()
