# cpp_generator.py
"""
Arduino C++ code generator for Liquid Time-Constant (LTC / LNN) models.

Design notes
------------
The generated C++ implements the same fixed-step Euler ODE integration that
the Python LTCCell uses.  The key difference from a vanilla RNN generator is
that each LTC cell requires four extra parameter arrays (W_tau, W_tauh, A,
and optionally b_tau) and a two-pass inner loop:

    Pass 1 — compute τ(x, h) via sigmoid + softplus
    Pass 2 — compute f(W_ih·x + W_hh·h) and Euler step

Bug history and fixes
---------------------
FIX 1 — per-layer hidden buffers (same fix as the RNN generator)
    Stacked layers each get their own named arrays h0[], h0_new[], etc.
    so that layer i does not accidentally read layer i-1's freshly updated
    state as its own previous hidden state.

FIX 2 — dense_in / dense_out sizing
    Allocated to the maximum in_features / out_features across all dense
    layers, not just the final output size.

FIX 3 — softplus numerical stability in C float
    softplus(x) = log(1 + exp(x)) can overflow for x > 88 in float32.
    The generated code clamps x to 88.0f before exponentiation.
    
FIX 4 — Pointer reset and state management
    Added `x_t = x_t_buf` inside the time loop to prevent input drifting.
    Added `reset_states()` to clear hidden state memory between inferences.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _relu(x):     return np.maximum(0.0, x)
def _leaky(x):    return np.where(x >= 0, x, 0.01 * x)
def _gelu(x):     return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
def _swish(x):    return x * _sigmoid(x)
def _softmax(x):  e = np.exp(x - x.max()); return e / e.sum()
def _softplus(x): return np.log1p(np.exp(np.clip(x, -88, 88)))

_NP_ACT = {
    'tanh':       np.tanh,
    'sigmoid':    _sigmoid,
    'relu':       _relu,
    'softmax':    _softmax,
    'leaky_relu': _leaky,
    'gelu':       _gelu,
    'swish':      _swish,
    'linear':     lambda x: x,
}


# ---------------------------------------------------------------------------
# NumPy reference forward pass (mirrors generated C exactly)
# ---------------------------------------------------------------------------

def _np_predict(data, x_seq):
    """
    Pure-NumPy forward pass for LTC models.

    x_seq : (seq_len, input_size)  float32 numpy array
    Returns a numpy array of shape (output_size,).
    """
    p          = data['parameters']
    rec_layers = data['recurrent_layers']
    den_layers = data['dense_layers']

    # Per-layer hidden state initialisation
    h_state = {}
    for layer in rec_layers:
        i  = layer['index']
        hs = layer['hidden_size']
        h_state[i] = np.zeros(hs, dtype=np.float32)

    # Recurrent pass
    for t in range(len(x_seq)):
        x_t = x_seq[t].astype(np.float32)

        for layer in rec_layers:
            i           = layer['index']
            hs          = layer['hidden_size']
            ins         = layer['input_size']
            act_name    = layer.get('activation', 'tanh')
            ode_unfolds = layer.get('ode_unfolds', 6)
            dt          = float(layer.get('dt', 1.0))
            tau_min     = float(layer.get('tau_min', 0.1))
            delta       = dt / ode_unfolds

            W_ih  = np.array(p[f'rec_{i}_W_ih_weight'],  dtype=np.float32)
            b_ih  = np.array(p[f'rec_{i}_W_ih_bias'],    dtype=np.float32)
            W_hh  = np.array(p[f'rec_{i}_W_hh_weight'],  dtype=np.float32)
            b_hh  = np.array(p[f'rec_{i}_W_hh_bias'],    dtype=np.float32)
            W_tau = np.array(p[f'rec_{i}_W_tau_weight'], dtype=np.float32)
            b_tau = np.array(p[f'rec_{i}_W_tau_bias'],   dtype=np.float32)
            W_tauh= np.array(p[f'rec_{i}_W_tauh_weight'],dtype=np.float32)
            A     = np.array(p[f'rec_{i}_A'],             dtype=np.float32)

            h = h_state[i].copy()

            for _ in range(ode_unfolds):
                # Backbone activation
                f_val = _NP_ACT[act_name](W_ih @ x_t + b_ih + W_hh @ h + b_hh)
                # Time constant
                gate  = _sigmoid(W_tau @ x_t + b_tau + W_tauh @ h)
                tau   = tau_min + _softplus(A * gate)
                # Euler step
                dh    = (-h + f_val) / tau
                h     = h + delta * dh

            h_state[i] = h
            x_t        = h

    # Dense head
    out = x_t.copy()
    for layer in den_layers:
        j   = layer['index']
        act = layer.get('activation', 'linear')
        W   = np.array(p[f'dense_{j}_weight'], dtype=np.float32)
        b   = np.array(p[f'dense_{j}_bias'],   dtype=np.float32)
        out = _NP_ACT[act](W @ out + b)

    return out


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoLNNGenerator:
    """
    Generates a two-file Arduino library (``LNNModel.h`` + ``sketch.ino``)
    from an exported LNNModel JSON file.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Directory where the generated files are written.
        board      : Target board family: ``'avr'`` (Uno / Mega) or
                     ``'esp32'`` (ESP32 / ESP32-S3).  On AVR, weights are
                     stored in PROGMEM to save SRAM.
        use_flash  : If ``True`` and ``board='avr'``, emit PROGMEM declarations.
        task       : ``'regression'``, ``'binary'``, or ``'multiclass'``.
    """

    def __init__(self, json_path, output_dir, board='avr',
                 use_flash=True, task='regression'):
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board
        self.use_flash  = use_flash and (board == 'avr')
        self.task       = task
        self.model_data = None

    def _load(self):
        with open(self.json_path, 'r') as f:
            self.model_data = json.load(f)

    @property
    def _rec(self): return self.model_data['recurrent_layers']
    @property
    def _den(self): return self.model_data['dense_layers']
    @property
    def _p(self):   return self.model_data['parameters']

    @property
    def input_size(self):
        return self._rec[0]['input_size'] if self._rec else 1

    @property
    def output_size(self):
        return self._den[-1]['out_features'] if self._den else 1

    @property
    def max_hidden(self):
        return max(l['hidden_size'] for l in self._rec) if self._rec else 1

    @property
    def max_dense_dim(self):
        if not self._den:
            return 1
        return max(
            max(l['in_features'], l['out_features'])
            for l in self._den
        )

    # ------------------------------------------------------------------
    def generate(self, sample_seq: np.ndarray | None = None):
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_seq : Optional ``(seq_len, input_size)`` float32 array used
                         to generate verification values in the sketch.
                         If ``None``, a random sample is drawn automatically.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        seq_len = 4
        if sample_seq is None:
            np.random.seed(0)
            sample_seq = np.random.randn(seq_len, self.input_size).astype(np.float32)

        expected = _np_predict(self.model_data, sample_seq)

        # Write header
        header_path = os.path.join(self.output_dir, 'LNNModel.h')
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())

        # Write sketch
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_seq, expected))

        print(f"Generated in : {self.output_dir}  (board: {self.board})")
        print(f"Verification  seq_len={len(sample_seq)}, input_size={self.input_size}")
        print(f"Expected output (Python) : {expected.tolist()}")

    # ------------------------------------------------------------------
    def _rd(self, arr_name):
        """Return a read expression honouring PROGMEM vs SRAM."""
        if self.use_flash:
            return f'pgm_read_float(&{arr_name})'
        return arr_name

    # ------------------------------------------------------------------
    def _fmt(self, values):
        """Format a flat or nested list of floats as C float literals."""
        if isinstance(values[0], list):
            rows = [
                '    {' + ', '.join(f'{v:.8f}f' for v in row) + '}'
                for row in values
            ]
            return '{\n' + ',\n'.join(rows) + '\n}'
        return '{' + ', '.join(f'{v:.8f}f' for v in values) + '}'

    # ------------------------------------------------------------------
    def _progmem(self, typ, name, values):
        """Declare a PROGMEM or plain const array."""
        suffix = ' PROGMEM' if self.use_flash else ''
        return f'static const {typ} {name}[]{suffix} = {self._fmt(values)};'

    # ------------------------------------------------------------------
    def _progmem2d(self, typ, name, rows, cols, values):
        suffix = ' PROGMEM' if self.use_flash else ''
        return f'static const {typ} {name}[{rows}][{cols}]{suffix} = {self._fmt(values)};'

    # ------------------------------------------------------------------
    def _header(self):
        p   = self._p
        lines = [
            '// LNNModel.h  —  Auto-generated Liquid Time-Constant Network',
            '// Do NOT edit weights manually.',
            '',
            '#pragma once',
            '#include <math.h>',
            '',
        ]

        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']

        # ---- Weight constants ----
        for layer in self._rec:
            i           = layer['index']
            hs          = layer['hidden_size']
            ins         = layer['input_size']
            T           = 'float'

            lines += [f'// LTC layer {i}  input={ins}  hidden={hs}']
            lines.append(self._progmem2d(T, f'rec{i}_W_ih_w', hs, ins, p[f'rec_{i}_W_ih_weight']))
            lines.append(self._progmem(  T, f'rec{i}_W_ih_b',          p[f'rec_{i}_W_ih_bias']))
            lines.append(self._progmem2d(T, f'rec{i}_W_hh_w', hs, hs,  p[f'rec_{i}_W_hh_weight']))
            lines.append(self._progmem(  T, f'rec{i}_W_hh_b',          p[f'rec_{i}_W_hh_bias']))
            lines.append(self._progmem2d(T, f'rec{i}_W_tau_w', hs, ins, p[f'rec_{i}_W_tau_weight']))
            lines.append(self._progmem(  T, f'rec{i}_W_tau_b',          p[f'rec_{i}_W_tau_bias']))
            lines.append(self._progmem2d(T, f'rec{i}_W_tauh_w', hs, hs, p[f'rec_{i}_W_tauh_weight']))
            lines.append(self._progmem(  T, f'rec{i}_A',                p[f'rec_{i}_A']))
            lines.append('')

        for layer in self._den:
            j   = layer['index']
            out = layer['out_features']
            inp = layer['in_features']
            T   = 'float'
            lines += [f'// Dense layer {j}  in={inp}  out={out}']
            lines.append(self._progmem2d(T, f'den{j}_w', out, inp, p[f'dense_{j}_weight']))
            if p[f'dense_{j}_bias'] is not None:
                lines.append(self._progmem(T, f'den{j}_b', p[f'dense_{j}_bias']))
            lines.append('')

        # ---- Class definition ----
        lines += [
            'class LNNModel {',
            'public:',
            f'    float predict(float* input, int seq_len);',
            '',
            '    // Reset hidden states (Call this before predicting a new sequence!)',
            '    void reset_states() {',
        ]
        
        for layer in self._rec:
            i = layer['index']
            hs = layer['hidden_size']
            lines.append(f'        for(int k=0; k<{hs}; k++) h{i}[k] = 0.0f;')
            
        lines += [
            '    }',
            '',
            'private:',
        ]

        # Per-layer hidden state buffers
        for layer in self._rec:
            i  = layer['index']
            hs = layer['hidden_size']
            lines.append(f'    float h{i}[{hs}] = {{0}};')
            lines.append(f'    float h{i}_new[{hs}];')

        lines += [
            '};',
            '',
        ]

        # ---- predict() implementation ----
        lines += [
            'float LNNModel::predict(float* input, int seq_len) {',
            f'    const int INPUT_SIZE = {self.input_size};',
            f'    float dense_in[{self.max_dense_dim}];',
            f'    float dense_out[{self.max_dense_dim}];',
            '    float x_t_buf[INPUT_SIZE];',
            '    float* x_t = x_t_buf;',
            '',
            '    for (int t = 0; t < seq_len; t++) {',
            '        x_t = x_t_buf; // <--- CORREÇÃO CRÍTICA: Reseta o ponteiro no início do time step',
            '        for (int k = 0; k < INPUT_SIZE; k++)',
            '            x_t_buf[k] = input[t * INPUT_SIZE + k];',
            '',
        ]

        rd = self._rd

        for layer in self._rec:
            i           = layer['index']
            hs          = layer['hidden_size']
            ins         = layer['input_size']
            act         = layer.get('activation', 'tanh')
            ode_unfolds = layer.get('ode_unfolds', 6)
            dt          = float(layer.get('dt', 1.0))
            tau_min     = float(layer.get('tau_min', 0.1))
            delta       = dt / ode_unfolds

            lines += [
                f'        // ---- LTC layer {i}  ode_unfolds={ode_unfolds}  delta={delta:.6f} ----',
                f'        for (int u = 0; u < {ode_unfolds}; u++) {{',
                f'            // -- backbone f(W_ih·x + W_hh·h + b) --',
                f'            float f_val[{hs}];',
                f'            for (int k = 0; k < {hs}; k++) {{',
                f'                float acc = {rd(f"rec{i}_W_ih_b[k]")};',
                f'                for (int j = 0; j < {ins}; j++)',
                f'                    acc += {rd(f"rec{i}_W_ih_w[k][j]")} * x_t[j];',
                f'                float hacc = {rd(f"rec{i}_W_hh_b[k]")};',
                f'                for (int j = 0; j < {hs}; j++)',
                f'                    hacc += {rd(f"rec{i}_W_hh_w[k][j]")} * h{i}[j];',
                f'                float pre = acc + hacc;',
            ]
            lines += ['                ' + l for l in self._act_c(act, 'pre')]
            lines += [
                f'                f_val[k] = pre;',
                f'            }}',
                f'            // -- time constant τ = tau_min + softplus(A * sigmoid(W_tau·x + W_tauh·h)) --',
                f'            for (int k = 0; k < {hs}; k++) {{',
                f'                float g = {rd(f"rec{i}_W_tau_b[k]")};',
                f'                for (int j = 0; j < {ins}; j++)',
                f'                    g += {rd(f"rec{i}_W_tau_w[k][j]")} * x_t[j];',
                f'                for (int j = 0; j < {hs}; j++)',
                f'                    g += {rd(f"rec{i}_W_tauh_w[k][j]")} * h{i}[j];',
                f'                float gate = 1.0f / (1.0f + expf(-g));',
                f'                float ag   = {rd(f"rec{i}_A[k]")} * gate;',
                f'                // softplus with overflow guard',
                f'                float sp   = ag < 88.0f ? logf(1.0f + expf(ag)) : ag;',
                f'                float tau  = {tau_min:.6f}f + sp;',
                f'                // Euler step: h += delta * (-h + f) / tau',
                f'                h{i}_new[k] = h{i}[k] + {delta:.8f}f * (-h{i}[k] + f_val[k]) / tau;',
                f'            }}',
                f'            for (int k = 0; k < {hs}; k++) h{i}[k] = h{i}_new[k];',
                f'        }}',
                f'        x_t = h{i};',
                '',
            ]

        lines += [
            '    } // end time-step loop',
            '',
            '    // ---- Dense head ----',
        ]

        last_rec = self._rec[-1]['index']
        prev_src = f'h{last_rec}'
        n_dense  = len(self._den)

        for li, layer in enumerate(self._den):
            j   = layer['index']
            out = layer['out_features']
            inp = layer['in_features']
            act = layer.get('activation', 'linear')
            lines += [
                f'    // Dense {j}  in={inp}  out={out}  act={act}',
                f'    for (int k = 0; k < {out}; k++) {{',
                f'        float acc = {rd(f"den{j}_b[k]")};',
                f'        for (int j = 0; j < {inp}; j++)',
                f'            acc += {rd(f"den{j}_w[k][j]")} * {prev_src}[j];',
            ]
            lines += ['        ' + l for l in self._act_c(act, 'acc')]
            lines += [
                f'        dense_out[k] = acc;',
                f'    }}',
            ]
            if li < n_dense - 1:
                next_inp = self._den[li + 1]['in_features']
                lines += [f'    for (int k = 0; k < {next_inp}; k++) dense_in[k] = dense_out[k];']
                prev_src = 'dense_in'
            lines.append('')

        if self.task == 'multiclass':
            lines += [
                f'    int best = 0;',
                f'    for (int k = 1; k < {self.output_size}; k++)',
                f'        if (dense_out[k] > dense_out[best]) best = k;',
                f'    return (float)best;',
            ]
        else:
            lines.append('    return dense_out[0];')

        lines.append('}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _act_c(self, act: str, var: str):
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
            return [f'{var} = 0.5f * {var} * (1.0f + tanhf(0.79788456f * ({var} + 0.044715f * {var} * {var} * {var})));']
        elif a == 'swish':
            return [f'{var} = {var} / (1.0f + expf(-{var}));']
        return []   # linear / softmax → identity

    # ------------------------------------------------------------------
    def _sketch(self, sample_seq: np.ndarray, expected: np.ndarray):
        seq_len  = len(sample_seq)
        flat     = sample_seq.flatten()
        flat_str = ', '.join(f'{float(v):.8f}f' for v in flat)
        exp_list = expected.tolist() if hasattr(expected, 'tolist') else [float(expected)]

        input_rows = [
            f' * t={t}: [{", ".join(f"{float(v):.8f}" for v in row)}]'
            for t, row in enumerate(sample_seq)
        ]

        if self.task == 'multiclass':
            exp_comment = f'Expected class    : {int(exp_list[0])}'
        elif self.task == 'binary':
            exp_comment = (f'Expected logit    : {exp_list[0]:.8f}\n'
                           f' * Expected class    : {int(exp_list[0] > 0)}')
        else:
            exp_comment = f'Expected value    : {exp_list[0]:.8f}'

        lines = [
            '/*',
            ' * LNN (LTC) Model -- Arduino verification sketch',
            ' * Generated automatically -- do not edit the weights.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * Input  seq_len    : {seq_len}',
            f' * Input  input_size : {self.input_size}',
            ' * Input values (same order as the flat array below):',
        ] + input_rows + [
            ' *',
            f' * {exp_comment}',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm the printed value matches the expected value',
            ' * above to at least 5 decimal places.',
            ' *',
            ' * Acceptable tolerance: +/-0.0001  (float32 + Euler rounding)',
            ' */',
            '',
            '#include "LNNModel.h"',
            '',
            'LNNModel model;',
            '',
            'void setup() {',
            '    Serial.begin(115200);',
            '    while (!Serial);',
            '',
            f'    const int SEQ_LEN    = {seq_len};',
            f'    const int INPUT_SIZE = {self.input_size};',
            '',
            '    // Verification input (auto-generated by Python exporter)',
            f'    float input[SEQ_LEN * INPUT_SIZE] = {{',
            f'        {flat_str}',
            f'    }};',
            '',
            '    // IMPRESCINDÍVEL: Resetar o estado antes de passar a sequência!',
            '    model.reset_states();',
            '    float output = model.predict(input, SEQ_LEN);',
            '',
        ]

        if self.task == 'binary':
            lines += [
                f'    // Expected logit  : {exp_list[0]:.8f}',
                f'    // Expected class  : {int(exp_list[0] > 0)}',
                '    Serial.print("Predicted logit  : "); Serial.println(output, 8);',
                '    Serial.print("Predicted class  : "); Serial.println(output > 0.0f ? 1 : 0);',
            ]
        elif self.task == 'multiclass':
            lines += [
                f'    // Expected class  : {int(exp_list[0])}',
                '    Serial.print("Predicted class  : "); Serial.println((int)output);',
            ]
        else:
            lines += [
                f'    // Expected value  : {exp_list[0]:.8f}',
                '    Serial.print("Predicted value  : "); Serial.println(output, 8);',
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
# Public convenience function
# ---------------------------------------------------------------------------

def generate_ino(json_path, output_dir, board='avr',
                 use_flash=True, task='regression'):
    """
    Generate Arduino LNN library files from a JSON model export.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store weights in PROGMEM.
        task       : ``'regression'``, ``'binary'``, or ``'multiclass'``.
    """
    gen = ArduinoLNNGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate()