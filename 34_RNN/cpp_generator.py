# cpp_generator.py
"""
Arduino C++ code generator for RNN / LSTM / GRU models.

Bug history and fixes
---------------------
FIX 1  (previous version) — in-place h_buf overwrite during W_hh matmul
    A single h_buf[k] was being written while the same loop still needed to
    read h_buf[j] for j > k.  Fixed by accumulating into h_new[] and
    committing after the full pass.

FIX 2  (THIS VERSION, critical) — stacked layers share one h_buf / c_buf
    When two or more recurrent layers are stacked, every layer was reading
    and writing the SAME h_buf and c_buf arrays.  After layer 0 committed
    its output, layer 1's W_hh multiply incorrectly read layer 0's new
    hidden state instead of layer 1's own previous hidden state.  The cell
    state c_buf of layer 0 was also overwritten before it was needed on the
    next time step.

    Fix: each recurrent layer gets its own named buffers:
        h0[], h0_new[], c0[], c0_new[]
        h1[], h1_new[], c1[], c1_new[]   etc.

FIX 3  (THIS VERSION) — sketch.ino used zeros + arbitrary seq_len
    The sketch now embeds a real random sample, runs the identical NumPy
    forward pass, and prints the Python-expected output as a comment so
    the user can verify the Arduino Serial Monitor matches exactly.

FIX 4  (THIS VERSION) — dense_in / dense_out buffer overflow
    dense_in and dense_out were allocated with the size of the final output
    (e.g. 1) but used as scratch space for intermediate dense layers that
    could be much wider (e.g. 32).  This caused memory corruption and
    incorrect predictions.  They are now sized to the maximum number of
    features (in_features or out_features) across all dense layers.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy reference forward pass (must mirror the generated C exactly)
# ---------------------------------------------------------------------------

def _sigmoid(x):    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _relu(x):       return np.maximum(0.0, x)
def _leaky(x):      return np.where(x >= 0, x, 0.01 * x)
def _gelu(x):       return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
def _swish(x):      return x * _sigmoid(x)
def _softmax(x):    e = np.exp(x - x.max()); return e / e.sum()

_NP_ACT = {
    'tanh':        np.tanh,
    'sigmoid':     _sigmoid,
    'relu':        _relu,
    'softmax':     _softmax,
    'leaky_relu':  _leaky,
    'gelu':        _gelu,
    'swish':       _swish,
    'linear':      lambda x: x,
}


def _np_predict(data, x_seq):
    """
    Pure-NumPy forward pass that exactly mirrors the corrected C code.

    Each recurrent layer keeps its OWN hidden / cell state vectors —
    matching the per-layer buffers generated in the C code.

    x_seq : (seq_len, input_size) float32 numpy array
    Returns a numpy array of shape (output_size,).
    """
    p          = data['parameters']
    rec_layers = data['recurrent_layers']
    den_layers = data['dense_layers']

    # Per-layer state initialisation
    h_state = {}   # layer_index -> h  (all types)
    c_state = {}   # layer_index -> c  (LSTM only)
    for layer in rec_layers:
        i  = layer['index']
        hs = layer['hidden_size']
        h_state[i] = np.zeros(hs, dtype=np.float32)
        c_state[i]  = np.zeros(hs, dtype=np.float32)

    # Recurrent pass
    for t in range(len(x_seq)):
        x_t = x_seq[t].astype(np.float32)

        for layer in rec_layers:
            i   = layer['index']
            typ = layer['type']
            hs  = layer['hidden_size']
            h   = h_state[i]
            c   = c_state[i]

            if typ == 'RNN':
                act  = layer.get('activation', 'tanh')
                W_ih = np.array(p[f'rec_{i}_W_ih_weight'], dtype=np.float32)
                b_ih = np.array(p[f'rec_{i}_W_ih_bias'],   dtype=np.float32)
                W_hh = np.array(p[f'rec_{i}_W_hh_weight'], dtype=np.float32)
                b_hh = np.array(p[f'rec_{i}_W_hh_bias'],   dtype=np.float32)
                pre        = W_ih @ x_t + b_ih + W_hh @ h + b_hh
                h_state[i] = _NP_ACT[act](pre)
                x_t        = h_state[i]

            elif typ == 'LSTM':
                W_ih  = np.array(p[f'rec_{i}_W_ih_weight'], dtype=np.float32)
                b_ih  = np.array(p[f'rec_{i}_W_ih_bias'],   dtype=np.float32)
                W_hh  = np.array(p[f'rec_{i}_W_hh_weight'], dtype=np.float32)
                b_hh  = np.array(p[f'rec_{i}_W_hh_bias'],   dtype=np.float32)
                gates  = W_ih @ x_t + b_ih + W_hh @ h + b_hh
                ig = _sigmoid(gates[0*hs : 1*hs])
                fg = _sigmoid(gates[1*hs : 2*hs])
                gg = np.tanh (gates[2*hs : 3*hs])
                og = _sigmoid(gates[3*hs : 4*hs])
                c_new      = fg * c + ig * gg
                h_new      = og * np.tanh(c_new)
                h_state[i] = h_new
                c_state[i]  = c_new
                x_t        = h_new

            elif typ == 'GRU':
                def _gw(n): return np.array(p[f'rec_{i}_{n}_weight'], dtype=np.float32)
                def _gb(n): return np.array(p[f'rec_{i}_{n}_bias'],   dtype=np.float32)
                r = _sigmoid(_gw('W_ir') @ x_t + _gb('W_ir') + _gw('W_hr') @ h + _gb('W_hr'))
                z = _sigmoid(_gw('W_iz') @ x_t + _gb('W_iz') + _gw('W_hz') @ h + _gb('W_hz'))
                n = np.tanh (_gw('W_in') @ x_t + _gb('W_in') + r * (_gw('W_hn') @ h + _gb('W_hn')))
                h_state[i] = (1.0 - z) * n + z * h
                x_t        = h_state[i]

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
# Generator
# ---------------------------------------------------------------------------

class ArduinoRNNGenerator:
    def __init__(self, json_path, output_dir, board='avr',
                 use_flash=True, task='regression'):
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board
        self.use_flash  = use_flash
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
        """Maximum in_features or out_features across all dense layers."""
        if not self._den:
            return 1
        return max(
            max(l['in_features'], l['out_features'])
            for l in self._den
        )

    def _flash(self):
        return ' PROGMEM' if (self.board == 'avr' and self.use_flash) else ''

    def _rd(self, expr):
        return f'pgm_read_float(&{expr})' if (self.board == 'avr' and self.use_flash) else expr

    # ------------------------------------------------------------------ entry
    def generate(self):
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        np.random.seed(0)
        seq_len    = 4
        sample_seq = np.random.randn(seq_len, self.input_size).astype(np.float32)
        expected   = _np_predict(self.model_data, sample_seq)

        with open(os.path.join(self.output_dir, 'RNNModel.h'),   'w') as f:
            f.write(self._header())
        with open(os.path.join(self.output_dir, 'RNNModel.cpp'), 'w') as f:
            f.write(self._implementation())
        with open(os.path.join(self.output_dir, 'sketch.ino'),   'w') as f:
            f.write(self._sketch(sample_seq, expected))

        print(f"Generated in : {self.output_dir}  (board: {self.board})")
        print(f"Verification  seq_len={seq_len}, input_size={self.input_size}")
        print(f"Expected output (Python) : {expected.tolist()}")

    # ==================================================================
    # Header
    # ==================================================================
    def _header(self):
        fa  = self._flash()
        mh  = max(self.max_hidden, 1)
        g4  = max(self.max_hidden * 4, 1)
        md  = self.max_dense_dim   # <-- FIX: allocate for largest dense layer

        lines = [
            '#ifndef RNNMODEL_H',
            '#define RNNMODEL_H',
            '',
            '#include <stdint.h>',
            '#include <avr/pgmspace.h>' if self.board == 'avr' else '#include <Arduino.h>',
            '',
            'class RNNModel {',
            'public:',
            '    RNNModel();',
            '    float predict(const float* input, int seq_len);',
            '',
            'private:',
            f'    // Shared scratch buffers (sized for the largest layer)',
            f'    static float gate_buf[{g4}];',
            f'    static float dense_in[{md}];',
            f'    static float dense_out[{md}];',
            '',
            '    // Per-layer hidden / cell state buffers',
            '    // (FIX: each layer must have its own h and c to preserve',
            '    //  the recurrent state across time steps independently)',
        ]

        for layer in self._rec:
            i   = layer['index']
            hs  = layer['hidden_size']
            typ = layer['type']
            lines.append(f'    static float h{i}[{hs}];')
            lines.append(f'    static float h{i}_new[{hs}];')
            if typ == 'LSTM':
                lines.append(f'    static float c{i}[{hs}];')
                lines.append(f'    static float c{i}_new[{hs}];')
            elif typ == 'GRU':
                lines.append(f'    static float r{i}[{hs}];')
                lines.append(f'    static float z{i}[{hs}];')

        lines.append('')

        # Weight declarations
        for layer in self._rec:
            i   = layer['index']
            typ = layer['type']
            hs  = layer['hidden_size']
            ins = layer['input_size']
            if typ == 'RNN':
                lines += [
                    f'    static const float{fa} rec{i}_W_ih_w[{hs}][{ins}];',
                    f'    static const float{fa} rec{i}_W_ih_b[{hs}];',
                    f'    static const float{fa} rec{i}_W_hh_w[{hs}][{hs}];',
                    f'    static const float{fa} rec{i}_W_hh_b[{hs}];',
                ]
            elif typ == 'LSTM':
                lines += [
                    f'    static const float{fa} rec{i}_W_ih_w[{4*hs}][{ins}];',
                    f'    static const float{fa} rec{i}_W_ih_b[{4*hs}];',
                    f'    static const float{fa} rec{i}_W_hh_w[{4*hs}][{hs}];',
                    f'    static const float{fa} rec{i}_W_hh_b[{4*hs}];',
                ]
            elif typ == 'GRU':
                for gate in ['W_ir','W_hr','W_iz','W_hz','W_in','W_hn']:
                    dim_in = ins if gate[2] == 'i' else hs
                    lines.append(f'    static const float{fa} rec{i}_{gate}_w[{hs}][{dim_in}];')
                    lines.append(f'    static const float{fa} rec{i}_{gate}_b[{hs}];')

        for layer in self._den:
            j   = layer['index']
            out = layer['out_features']
            inp = layer['in_features']
            lines += [
                f'    static const float{fa} den{j}_w[{out}][{inp}];',
                f'    static const float{fa} den{j}_b[{out}];',
            ]

        lines += ['};', '', '#endif']
        return '\n'.join(lines)

    # ==================================================================
    # Weight array helpers
    # ==================================================================
    def _arr2d(self, name, data, fa):
        rows, cols = len(data), len(data[0])
        lines = [f'const float RNNModel::{name}[{rows}][{cols}]{fa} = {{']
        for row in data:
            lines.append('    {' + ', '.join(f'{float(v):.8f}f' for v in row) + '},')
        lines.append('};\n')
        return '\n'.join(lines)

    def _arr1d(self, name, data, fa, n=None):
        n    = n or len(data)
        body = ', '.join(f'{float(v):.8f}f' for v in data)
        return f'const float RNNModel::{name}[{n}]{fa} = {{{body}}};\n'

    # ==================================================================
    # Implementation (.cpp)
    # ==================================================================
    def _implementation(self):
        fa  = self._flash()
        g4  = max(self.max_hidden * 4, 1)
        md  = self.max_dense_dim   # <-- FIX

        p   = self._p

        lines = [
            '#include "RNNModel.h"',
            '#include <math.h>',
            '#include <avr/pgmspace.h>' if self.board == 'avr' else '',
            '',
            '// Shared scratch',
            f'float RNNModel::gate_buf[{g4}]  = {{0}};',
            f'float RNNModel::dense_in[{md}]  = {{0}};',
            f'float RNNModel::dense_out[{md}] = {{0}};',
            '',
            '// Per-layer hidden / cell state buffers',
        ]

        for layer in self._rec:
            i   = layer['index']
            hs  = layer['hidden_size']
            typ = layer['type']
            lines.append(f'float RNNModel::h{i}[{hs}]     = {{0}};')
            lines.append(f'float RNNModel::h{i}_new[{hs}]  = {{0}};')
            if typ == 'LSTM':
                lines.append(f'float RNNModel::c{i}[{hs}]     = {{0}};')
                lines.append(f'float RNNModel::c{i}_new[{hs}]  = {{0}};')
            elif typ == 'GRU':
                lines.append(f'float RNNModel::r{i}[{hs}]     = {{0}};')
                lines.append(f'float RNNModel::z{i}[{hs}]     = {{0}};')
        lines.append('')

        # Weight data
        for layer in self._rec:
            i   = layer['index']
            typ = layer['type']
            if typ == 'RNN':
                lines.append(self._arr2d(f'rec{i}_W_ih_w', p[f'rec_{i}_W_ih_weight'], fa))
                lines.append(self._arr1d(f'rec{i}_W_ih_b', p[f'rec_{i}_W_ih_bias'],   fa))
                lines.append(self._arr2d(f'rec{i}_W_hh_w', p[f'rec_{i}_W_hh_weight'], fa))
                lines.append(self._arr1d(f'rec{i}_W_hh_b', p[f'rec_{i}_W_hh_bias'],   fa))
            elif typ == 'LSTM':
                lines.append(self._arr2d(f'rec{i}_W_ih_w', p[f'rec_{i}_W_ih_weight'], fa))
                lines.append(self._arr1d(f'rec{i}_W_ih_b', p[f'rec_{i}_W_ih_bias'],   fa))
                lines.append(self._arr2d(f'rec{i}_W_hh_w', p[f'rec_{i}_W_hh_weight'], fa))
                lines.append(self._arr1d(f'rec{i}_W_hh_b', p[f'rec_{i}_W_hh_bias'],   fa))
            elif typ == 'GRU':
                for gate in ['W_ir','W_hr','W_iz','W_hz','W_in','W_hn']:
                    lines.append(self._arr2d(f'rec{i}_{gate}_w', p[f'rec_{i}_{gate}_weight'], fa))
                    lines.append(self._arr1d(f'rec{i}_{gate}_b', p[f'rec_{i}_{gate}_bias'],   fa))

        for layer in self._den:
            j = layer['index']
            lines.append(self._arr2d(f'den{j}_w', p[f'dense_{j}_weight'], fa))
            lines.append(self._arr1d(f'den{j}_b', p[f'dense_{j}_bias'],   fa))

        lines += ['', 'RNNModel::RNNModel() {}', '', self._predict_method()]
        return '\n'.join(lines)

    # ==================================================================
    # predict() — one named buffer per layer
    # ==================================================================
    def _predict_method(self):
        rd = self._rd

        # Reset lines: one per layer
        reset_lines = []
        for layer in self._rec:
            i  = layer['index']
            hs = layer['hidden_size']
            reset_lines.append(
                f'    for (int i = 0; i < {hs}; i++) {{ h{i}[i] = 0.0f;'
                + (f' c{i}[i] = 0.0f;' if layer['type'] == 'LSTM' else '')
                + ' }'
            )

        lines = [
            'float RNNModel::predict(const float* input, int seq_len) {',
            '',
            '    // Reset per-layer hidden (and cell) states',
        ] + reset_lines + [
            '',
            '    // ================================================================',
            '    // Recurrent pass',
            '    // ================================================================',
            '    for (int t = 0; t < seq_len; t++) {',
            f'        const float* x_t = input + t * {self.input_size};',
            '',
        ]

        for layer in self._rec:
            i   = layer['index']
            typ = layer['type']
            hs  = layer['hidden_size']
            ins = layer['input_size']

            # ---- RNN -------------------------------------------------------
            if typ == 'RNN':
                act = layer.get('activation', 'tanh')
                lines += [
                    f'        // ---- RNN layer {i}  in={ins} hidden={hs} act={act} ----',
                    f'        for (int k = 0; k < {hs}; k++) {{',
                    f'            float acc = {rd(f"rec{i}_W_ih_b[k]")};',
                    f'            for (int j = 0; j < {ins}; j++)',
                    f'                acc += {rd(f"rec{i}_W_ih_w[k][j]")} * x_t[j];',
                    f'            for (int j = 0; j < {hs}; j++)',
                    f'                acc += {rd(f"rec{i}_W_hh_w[k][j]")} * h{i}[j];',
                    f'            acc += {rd(f"rec{i}_W_hh_b[k]")};',
                ]
                lines += ['            ' + l for l in self._act_c(act, 'acc')]
                lines += [
                    f'            h{i}_new[k] = acc;',
                    f'        }}',
                    f'        for (int k = 0; k < {hs}; k++) h{i}[k] = h{i}_new[k];',
                    f'        x_t = h{i};',
                    '',
                ]

            # ---- LSTM -------------------------------------------------------
            elif typ == 'LSTM':
                lines += [
                    f'        // ---- LSTM layer {i}  in={ins} hidden={hs} ----',
                    f'        // Step 1: all 4*H gate pre-activations into gate_buf.',
                    f'        for (int k = 0; k < {4*hs}; k++) {{',
                    f'            float acc = {rd(f"rec{i}_W_ih_b[k]")};',
                    f'            for (int j = 0; j < {ins}; j++)',
                    f'                acc += {rd(f"rec{i}_W_ih_w[k][j]")} * x_t[j];',
                    f'            for (int j = 0; j < {hs}; j++)',
                    f'                acc += {rd(f"rec{i}_W_hh_w[k][j]")} * h{i}[j];',
                    f'            acc += {rd(f"rec{i}_W_hh_b[k]")};',
                    f'            gate_buf[k] = acc;',
                    f'        }}',
                    f'        // Step 2: gate activations → h{i}_new / c{i}_new.',
                    f'        for (int k = 0; k < {hs}; k++) {{',
                    f'            float ig = 1.0f / (1.0f + expf(-gate_buf[{0*hs} + k]));',
                    f'            float fg = 1.0f / (1.0f + expf(-gate_buf[{1*hs} + k]));',
                    f'            float gg = tanhf(gate_buf[{2*hs} + k]);',
                    f'            float og = 1.0f / (1.0f + expf(-gate_buf[{3*hs} + k]));',
                    f'            c{i}_new[k] = fg * c{i}[k] + ig * gg;',
                    f'            h{i}_new[k] = og * tanhf(c{i}_new[k]);',
                    f'        }}',
                    f'        // Commit layer {i} state.',
                    f'        for (int k = 0; k < {hs}; k++) {{ h{i}[k] = h{i}_new[k]; c{i}[k] = c{i}_new[k]; }}',
                    f'        x_t = h{i};',
                    '',
                ]

            # ---- GRU -------------------------------------------------------
            elif typ == 'GRU':
                lines += [
                    f'        // ---- GRU layer {i}  in={ins} hidden={hs} ----',
                    f'        // Step 1: r and z gates from old h{i}.',
                    f'        for (int k = 0; k < {hs}; k++) {{',
                    f'            float acc_r = {rd(f"rec{i}_W_ir_b[k]")};',
                    f'            float acc_z = {rd(f"rec{i}_W_iz_b[k]")};',
                    f'            for (int j = 0; j < {ins}; j++) {{',
                    f'                acc_r += {rd(f"rec{i}_W_ir_w[k][j]")} * x_t[j];',
                    f'                acc_z += {rd(f"rec{i}_W_iz_w[k][j]")} * x_t[j];',
                    f'            }}',
                    f'            for (int j = 0; j < {hs}; j++) {{',
                    f'                acc_r += {rd(f"rec{i}_W_hr_w[k][j]")} * h{i}[j];',
                    f'                acc_z += {rd(f"rec{i}_W_hz_w[k][j]")} * h{i}[j];',
                    f'            }}',
                    f'            r{i}[k] = 1.0f / (1.0f + expf(-acc_r));',
                    f'            z{i}[k] = 1.0f / (1.0f + expf(-acc_z));',
                    f'        }}',
                    f'        // Step 2: candidate + new hidden into h{i}_new.',
                    f'        for (int k = 0; k < {hs}; k++) {{',
                    f'            float acc_n = {rd(f"rec{i}_W_in_b[k]")};',
                    f'            for (int j = 0; j < {ins}; j++)',
                    f'                acc_n += {rd(f"rec{i}_W_in_w[k][j]")} * x_t[j];',
                    f'            float rh = {rd(f"rec{i}_W_hn_b[k]")};',
                    f'            for (int j = 0; j < {hs}; j++)',
                    f'                rh += {rd(f"rec{i}_W_hn_w[k][j]")} * h{i}[j];',
                    f'            float n_k = tanhf(acc_n + r{i}[k] * rh);',
                    f'            h{i}_new[k] = (1.0f - z{i}[k]) * n_k + z{i}[k] * h{i}[k];',
                    f'        }}',
                    f'        // Commit layer {i} state.',
                    f'        for (int k = 0; k < {hs}; k++) h{i}[k] = h{i}_new[k];',
                    f'        x_t = h{i};',
                    '',
                ]

        lines += [
            '    } // end time-step loop',
            '',
            '    // ================================================================',
            '    // Dense head',
            '    // ================================================================',
        ]

        # Last recurrent layer index for the first dense input
        last_rec = self._rec[-1]['index']
        prev_src = f'h{last_rec}'
        n_dense  = len(self._den)

        for li, layer in enumerate(self._den):
            j   = layer['index']
            out = layer['out_features']
            inp = layer['in_features']
            act = layer.get('activation', 'linear')
            lines += [
                f'    // Dense layer {j}  in={inp} out={out} act={act}',
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

    # ==================================================================
    # C activation snippet
    # ==================================================================
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

    # ==================================================================
    # Sketch with real verification values
    # ==================================================================
    def _sketch(self, sample_seq: np.ndarray, expected: np.ndarray):
        seq_len  = len(sample_seq)
        flat     = sample_seq.flatten()
        flat_str = ', '.join(f'{float(v):.8f}f' for v in flat)
        exp_list = expected.tolist() if hasattr(expected, 'tolist') else [float(expected)]

        input_rows = [
            f' *   t={t}: [{", ".join(f"{float(v):.8f}" for v in row)}]'
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
            ' * RNN Model -- Arduino verification sketch',
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
            ' * Acceptable tolerance: +/-0.00002  (float32 rounding)',
            ' */',
            '',
            '#include "RNNModel.h"',
            '',
            'RNNModel model;',
            '',
            'void setup() {',
            '    Serial.begin(115200);',
            '    while (!Serial);  // Wait for Serial on native-USB boards',
            '',
            f'    const int SEQ_LEN    = {seq_len};',
            f'    const int INPUT_SIZE = {self.input_size};',
            '',
            '    // Verification input (auto-generated by Python exporter)',
            f'    float input[SEQ_LEN * INPUT_SIZE] = {{',
            f'        {flat_str}',
            f'    }};',
            '',
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
    gen = ArduinoRNNGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate()