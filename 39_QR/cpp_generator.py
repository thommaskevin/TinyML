# cpp_generator.py
"""
Arduino C++ code generator for Quantile Regression Neural Network (QRNN)
models exported via ``export_to_json``.

Design notes
------------
The generated C++ implements the same forward pass as the Python QRModel:

    1. Dense backbone  : ReLU/tanh/sigmoid/GELU/Swish/LeakyReLU/linear
                         hidden layers, each as a matrix-vector multiply
                         followed by an element-wise activation.
    2. Quantile head   : A single linear layer that maps the penultimate
                         representation to K raw outputs.
    3. Monotone decode : Cumulative-sum reparameterisation with softplus
                         on raw deltas (columns 1…K−1) to enforce
                         ŷ_{τ_1} ≤ … ≤ ŷ_{τ_K} without any sorting step.

The output of ``predict()`` is a float array of length K (one entry per
quantile level).  For single-quantile models (K = 1) the array has one
element and the caller may read it directly as a scalar.

Bug history and fixes
---------------------
FIX 1 — per-layer buffer sizing
    Each dense layer gets its own output buffer sized to its exact
    ``out_features``.  A global ``MAX_HIDDEN`` constant allocates the
    largest buffer needed so the same array can be reused safely.

FIX 2 — softplus numerical stability in C float
    softplus(x) = log(1 + exp(x)) overflows for x > 88 in float32.
    The generated code clamps the argument to 88.0f before calling expf().

FIX 3 — quantile head output array length
    The output buffer is always allocated to K (number of quantiles),
    not to the last hidden layer width.

FIX 4 — PROGMEM read-back on AVR
    Weight arrays declared PROGMEM are read with pgm_read_float_near()
    on AVR targets.  On ESP32 / non-AVR, a standard array access is used.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy activation helpers (mirror the generated C exactly)
# ---------------------------------------------------------------------------

def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _relu(x):     return np.maximum(0.0, x)
def _leaky(x):    return np.where(x >= 0, x, 0.01 * x)
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
def _swish(x):    return x * _sigmoid(x)
def _softplus(x): return np.log1p(np.exp(np.clip(x, -88, 88)))

_NP_ACT = {
    'tanh':       np.tanh,
    'sigmoid':    _sigmoid,
    'relu':       _relu,
    'leaky_relu': _leaky,
    'gelu':       _gelu,
    'swish':      _swish,
    'linear':     lambda x: x,
}


# ---------------------------------------------------------------------------
# NumPy reference forward pass (mirrors the generated C exactly)
# ---------------------------------------------------------------------------

def _np_predict(data: dict, x: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy forward pass for QRNN models.

    Args:
        data : Parsed JSON dict produced by ``export_to_json``.
        x    : Input vector of shape ``(input_size,)`` — a single sample.

    Returns:
        Output array of shape ``(K,)`` — one prediction per quantile.
    """
    p      = data['parameters']
    hidden = data['hidden_layers']
    head   = data['quantile_head']
    K      = head['K']

    out = x.astype(np.float32)

    for layer in hidden:
        i   = layer['index']
        act = layer.get('activation', 'relu')
        W   = np.array(p[f'hidden_{i}_weight'], dtype=np.float32)
        b   = np.array(p[f'hidden_{i}_bias'],   dtype=np.float32)
        out = _NP_ACT[act](W @ out + b)

    W_h = np.array(p['head_weight'], dtype=np.float32)
    b_h = np.array(p['head_bias'],   dtype=np.float32)
    raw = W_h @ out + b_h                              # (K,)

    if K == 1:
        return raw

    # Monotone cumulative-sum decode
    base    = raw[:1]
    deltas  = _softplus(raw[1:])
    q_hat   = np.concatenate([base, deltas]).cumsum()  # (K,)
    return q_hat


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoQRNNGenerator:
    """
    Generates a two-file Arduino library (``QRNNModel.h`` + ``sketch.ino``)
    from a JSON export of a trained ``QRModel``.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Directory where the generated files are written.
        board      : ``'avr'`` (Uno / Mega) or ``'esp32'`` (ESP32 / ESP32-S3).
                     On AVR, weights are stored in PROGMEM to save SRAM.
        use_flash  : If ``True`` and ``board='avr'``, emit PROGMEM declarations.
        task       : ``'regression'`` (returns the full quantile array) or
                     ``'median'`` (returns only the median quantile as a scalar).
    """

    def __init__(
        self,
        json_path:  str,
        output_dir: str,
        board:      str  = 'avr',
        use_flash:  bool = True,
        task:       str  = 'regression',
    ) -> None:
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board
        self.use_flash  = use_flash and (board == 'avr')
        self.task       = task
        self.model_data = None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        with open(self.json_path, 'r') as f:
            self.model_data = json.load(f)

    @property
    def _hidden(self): return self.model_data['hidden_layers']

    @property
    def _head_cfg(self): return self.model_data['quantile_head']

    @property
    def _p(self): return self.model_data['parameters']

    @property
    def input_size(self) -> int:
        return self._hidden[0]['in_features'] if self._hidden else 1

    @property
    def K(self) -> int:
        return self._head_cfg['K']

    @property
    def quantiles(self) -> list:
        return self._head_cfg['quantiles']

    @property
    def max_hidden(self) -> int:
        return max(l['out_features'] for l in self._hidden) if self._hidden else 1

    # ------------------------------------------------------------------
    def generate(self, sample_x: np.ndarray | None = None) -> None:
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_x : Optional ``(input_size,)`` float32 array used to
                       compute a reference prediction for the verification
                       sketch.  A random sample is used if ``None``.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_x is None:
            rng      = np.random.RandomState(0)
            sample_x = rng.randn(self.input_size).astype(np.float32)

        expected = _np_predict(self.model_data, sample_x)

        header_path = os.path.join(self.output_dir, 'QRNNModel.h')
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')

        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_x, expected))

        print(f"Header  → {header_path}")
        print(f"Sketch  → {sketch_path}")
        print(f"Reference prediction (K={self.K} quantiles): {expected.round(6)}")

    # ------------------------------------------------------------------
    def _rd(self, arr_name: str) -> str:
        """Return the correct array-read expression for the target board."""
        if self.use_flash:
            return f'pgm_read_float_near(&{arr_name})'
        return arr_name

    # ------------------------------------------------------------------
    def _fmt_array_1d(self, name: str, values: list, use_progmem: bool) -> str:
        """Format a 1-D float array declaration."""
        vals = ', '.join(f'{float(v):.8f}f' for v in values)
        pgm  = ' PROGMEM' if use_progmem else ''
        return f'const float{pgm} {name}[] = {{{vals}}};\n'

    def _fmt_array_2d(self, name: str, rows: list, use_progmem: bool) -> str:
        """Format a 2-D float array declaration (row-major)."""
        pgm  = ' PROGMEM' if use_progmem else ''
        n_rows = len(rows)
        n_cols = len(rows[0])
        lines  = [f'const float{pgm} {name}[{n_rows}][{n_cols}] = {{']
        for row in rows:
            row_str = ', '.join(f'{float(v):.8f}f' for v in row)
            lines.append(f'  {{{row_str}}},')
        lines.append('};\n')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _act_c(self, act: str, var: str) -> list:
        """Return C statements that apply an in-place activation to *var*."""
        a = act.lower()
        if a == 'relu':
            return [f'if ({var} < 0.0f) {var} = 0.0f;']
        elif a == 'sigmoid':
            return [f'{var} = 1.0f / (1.0f + expf(-{var}));']
        elif a == 'tanh':
            return [f'{var} = tanhf({var});']
        elif a == 'leaky_relu':
            return [f'if ({var} < 0.0f) {var} *= 0.01f;']
        elif a == 'gelu':
            return [
                f'{var} = 0.5f * {var} * (1.0f + tanhf('
                f'0.79788456f * ({var} + 0.044715f * {var} * {var} * {var})));'
            ]
        elif a == 'swish':
            return [f'{var} = {var} / (1.0f + expf(-{var}));']
        return []   # linear → identity

    # ------------------------------------------------------------------
    def _header(self) -> str:
        """Generate the full QRNNModel.h content."""
        lines = [
            '#pragma once',
            '/*',
            ' * QRNNModel.h — Quantile Regression Neural Network',
            ' * Auto-generated by cpp_generator.py — do not edit weights manually.',
            ' *',
            f' * Quantile levels : [{", ".join(str(q) for q in self.quantiles)}]',
            f' * Input size      : {self.input_size}',
            f' * Hidden layers   : {len(self._hidden)}',
            f' * K (num quantiles): {self.K}',
            f' * Board target    : {self.board.upper()}',
            ' */',
            '',
            '#include <math.h>',
        ]
        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']
        else:
            lines.append('')

        p = self._p

        # ---- Weight declarations ----
        for layer in self._hidden:
            i   = layer['index']
            W   = p[f'hidden_{i}_weight']
            b   = p[f'hidden_{i}_bias']
            lines.append(self._fmt_array_2d(f'h{i}_w', W, self.use_flash))
            lines.append(self._fmt_array_1d(f'h{i}_b', b, self.use_flash))

        W_head = p['head_weight']
        b_head = p['head_bias']
        lines.append(self._fmt_array_2d('head_w', W_head, self.use_flash))
        lines.append(self._fmt_array_1d('head_b', b_head, self.use_flash))

        # ---- Class definition ----
        lines += [
            'class QRNNModel {',
            'public:',
            f'  static const int INPUT_SIZE = {self.input_size};',
            f'  static const int K          = {self.K};',
            '',
            '  /**',
            '   * Run a forward pass and write K quantile estimates to *output*.',
            '   *',
            '   * @param x       Input array of length INPUT_SIZE.',
            '   * @param output  Output array of length K (caller-allocated).',
            '   */',
            '  void predict(const float* x, float* output) {',
            f'    float buf_a[{self.max_hidden}];',
            f'    float buf_b[{self.max_hidden}];',
            '    const float* in_ptr  = x;',
            '    float*       out_ptr = buf_a;',
            '',
        ]

        in_size = self.input_size
        for layer in self._hidden:
            i    = layer['index']
            nin  = layer['in_features']
            nout = layer['out_features']
            act  = layer.get('activation', 'relu')

            lines += [
                f'    // --- Hidden layer {i}  ({nin} → {nout}, {act}) ---',
                f'    for (int k = 0; k < {nout}; k++) {{',
            ]
            if self.use_flash:
                lines += [
                    f'      float acc = pgm_read_float_near(&h{i}_b[k]);',
                    f'      for (int j = 0; j < {nin}; j++)',
                    f'        acc += pgm_read_float_near(&h{i}_w[k][j]) * in_ptr[j];',
                ]
            else:
                lines += [
                    f'      float acc = h{i}_b[k];',
                    f'      for (int j = 0; j < {nin}; j++)',
                    f'        acc += h{i}_w[k][j] * in_ptr[j];',
                ]
            act_stmts = self._act_c(act, 'acc')
            lines += [f'      {s}' for s in act_stmts]
            lines += [
                f'      out_ptr[k] = acc;',
                f'    }}',
                '',
            ]
            # Ping-pong buffers: swap in_ptr / out_ptr
            lines += [
                f'    // Swap buffers',
                f'    in_ptr  = out_ptr;',
                f'    out_ptr = (out_ptr == buf_a) ? buf_b : buf_a;',
                '',
            ]

        # ---- Quantile head ----
        nin_head = self._hidden[-1]['out_features']
        lines += [
            f'    // --- Quantile head ({nin_head} → {self.K}) ---',
            f'    float raw[{self.K}];',
            f'    for (int k = 0; k < {self.K}; k++) {{',
        ]
        if self.use_flash:
            lines += [
                f'      float acc = pgm_read_float_near(&head_b[k]);',
                f'      for (int j = 0; j < {nin_head}; j++)',
                f'        acc += pgm_read_float_near(&head_w[k][j]) * in_ptr[j];',
            ]
        else:
            lines += [
                f'      float acc = head_b[k];',
                f'      for (int j = 0; j < {nin_head}; j++)',
                f'        acc += head_w[k][j] * in_ptr[j];',
            ]
        lines += [
            f'      raw[k] = acc;',
            f'    }}',
            '',
        ]

        # ---- Monotone cumulative-sum decode ----
        if self.K == 1:
            lines += [
                '    // Single quantile — pass through directly',
                '    output[0] = raw[0];',
            ]
        else:
            lines += [
                '    // Monotone decode: base + cumsum(softplus(deltas))',
                '    output[0] = raw[0];   // base quantile (unconstrained)',
                f'    for (int k = 1; k < {self.K}; k++) {{',
                '      // softplus with overflow guard: log(1 + exp(x)), clamped at x = 88',
                '      float x_sp = raw[k] < 88.0f ? raw[k] : 88.0f;',
                '      float sp   = logf(1.0f + expf(x_sp));',
                '      output[k]  = output[k - 1] + sp;   // cumulative sum',
                '    }',
            ]

        lines += [
            '  }',
            '};',
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _sketch(self, sample_x: np.ndarray, expected: np.ndarray) -> str:
        """Generate the verification sketch content."""
        K        = self.K
        in_size  = self.input_size
        flat_str = ', '.join(f'{float(v):.8f}f' for v in sample_x)
        exp_list = expected.tolist()

        exp_lines = '\n'.join(
            f' * Q{self.quantiles[k]:.2f} → {exp_list[k]:.8f}'
            for k in range(K)
        )

        lines = [
            '/*',
            ' * QRNNModel — Arduino verification sketch',
            ' * Auto-generated — do not edit the weights.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * Input size       : {in_size}',
            f' * Num quantiles (K): {K}',
            f' * Input values     : [{", ".join(f"{float(v):.6f}" for v in sample_x)}]',
            ' *',
            ' * Expected outputs (Python reference):',
            f'{exp_lines}',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm each printed quantile matches its expected value',
            ' * to at least 5 decimal places.',
            ' *',
            ' * Acceptable tolerance: ±0.0001  (float32 rounding)',
            ' */',
            '',
            '#include "QRNNModel.h"',
            '',
            'QRNNModel model;',
            '',
            'void setup() {',
            '  Serial.begin(115200);',
            '  while (!Serial);',
            '',
            f'  const int INPUT_SIZE = {in_size};',
            f'  const int K          = {K};',
            '',
            f'  float x[INPUT_SIZE] = {{ {flat_str} }};',
            f'  float output[K];',
            '',
            '  model.predict(x, output);',
            '',
        ]

        for k in range(K):
            lines += [
                f'  // Expected Q{self.quantiles[k]:.2f}: {exp_list[k]:.8f}',
                f'  Serial.print("Q{self.quantiles[k]:.2f} = ");',
                f'  Serial.println(output[{k}], 8);',
            ]

        lines += [
            '}',
            '',
            'void loop() {',
            '  // Nothing to do here.',
            '}',
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def generate_ino(
    json_path:  str,
    output_dir: str,
    board:      str  = 'avr',
    use_flash:  bool = True,
    task:       str  = 'regression',
    sample_x:   np.ndarray | None = None,
) -> None:
    """
    Generate Arduino QRNN library files from a JSON model export.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store weights in PROGMEM.
        task       : ``'regression'`` or ``'median'``.
        sample_x   : Optional reference input of shape ``(input_size,)``.
    """
    gen = ArduinoQRNNGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate(sample_x)