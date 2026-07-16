"""
Arduino C++ code generator for GAN Generator models.

Generates two files: ``GANModel.h`` and ``sketch.ino``.

The Generator forward pass — a stack of fully-connected blocks with
optional batch normalization — is straightforward to translate to C:

    for each block:
        linear(x)
        optional BN: (x - running_mean) / sqrt(running_var + eps) * gamma + beta
        activation

Notes on batch normalization in inference
-----------------------------------------
During training, BatchNorm1d tracks running statistics (running_mean and
running_var) via exponential moving averages.  At inference time (eval mode),
these stored statistics are used instead of batch statistics.  The generator
must be called in eval mode (model.eval()) before export so that the running
statistics are correctly populated.

The generated C code reads:
    x_norm = (x - running_mean) / sqrt(running_var + eps) * gamma + beta

which is the affine-transformed normalization used at inference.

ConditionalBatchNorm is NOT supported in the C generator (it requires
embedding table lookups).  For embedded deployment, use unconditional GANs.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _relu(x):     return np.maximum(0.0, x)
def _leaky(x):    return np.where(x >= 0, x, 0.2 * x)
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
def _swish(x):    return x * _sigmoid(x)
def _softmax(x):  e = np.exp(x - x.max()); return e / e.sum()

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

def _np_generate(data: dict, z: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy Generator forward pass.

    z    : (latent_dim,) float32 numpy array — a single latent vector.
    Returns a (output_dim,) float32 numpy array.
    """
    p      = data['parameters']
    layers = data['layers']
    arch   = data['architecture']

    x = z.astype(np.float32)

    for layer in layers:
        i   = layer['index']
        act = layer.get('activation', 'linear')
        W   = np.array(p[f'gen_{i}_weight'], dtype=np.float32)
        b   = p[f'gen_{i}_bias']
        b   = np.array(b, dtype=np.float32) if b is not None else np.zeros(W.shape[0], dtype=np.float32)

        x = W @ x + b   # linear

        if layer.get('use_bn', False) and not layer.get('conditional', False):
            gamma = np.array(p[f'gen_{i}_bn_weight'],      dtype=np.float32)
            beta  = np.array(p[f'gen_{i}_bn_bias'],        dtype=np.float32)
            rmean = np.array(p[f'gen_{i}_bn_running_mean'], dtype=np.float32)
            rvar  = np.array(p[f'gen_{i}_bn_running_var'],  dtype=np.float32)
            eps   = 1e-5
            x = gamma * (x - rmean) / np.sqrt(rvar + eps) + beta

        x = _NP_ACT[act](x)

    return x


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoGANGenerator:
    """
    Generates a two-file Arduino library (``GANModel.h`` + ``sketch.ino``)
    from an exported Generator JSON file.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``,
                     augmented with BN running statistics via
                     ``add_bn_stats_to_json``.
        output_dir : Directory where the generated files are written.
        board      : Target board family: ``'avr'`` (Uno / Mega) or
                     ``'esp32'`` (ESP32 / ESP32-S3).
        use_flash  : If ``True`` and ``board='avr'``, emit PROGMEM for weights.
    """

    def __init__(
        self,
        json_path:  str,
        output_dir: str,
        board:      str  = 'esp32',
        use_flash:  bool = False,
    ) -> None:
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board
        self.use_flash  = use_flash and (board == 'avr')
        self.data       = None

    def _load(self) -> None:
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

    @property
    def _layers(self): return self.data['layers']
    @property
    def _p(self):      return self.data['parameters']
    @property
    def _arch(self):   return self.data['architecture']

    @property
    def latent_dim(self): return self._arch['latent_dim']
    @property
    def output_dim(self): return self._arch['output_dim']
    @property
    def max_width(self):
        return max(
            max(l['in_features'], l['out_features'])
            for l in self._layers
        )

    # ------------------------------------------------------------------
    def generate(self, sample_z: np.ndarray = None) -> None:
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_z : Optional ``(latent_dim,)`` float32 array used to
                       generate a verification vector.  If ``None``, a random
                       sample is drawn with seed 0.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_z is None:
            np.random.seed(0)
            sample_z = np.random.randn(self.latent_dim).astype(np.float32)

        expected = _np_generate(self.data, sample_z)

        header_path = os.path.join(self.output_dir, 'GANModel.h')
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')

        # CORREÇÃO: adicionado encoding='utf-8' para evitar erro de codificação
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_z, expected))

        print(f"Generated in : {self.output_dir}  (board: {self.board})")
        print(f"Latent dim   : {self.latent_dim}  Output dim: {self.output_dim}")
        print(f"Expected output (Python) : {expected[:6].tolist()} ...")

    # ------------------------------------------------------------------
    def _pm(self): return ' PROGMEM' if self.use_flash else ''

    def _rd(self, expr: str) -> str:
        return f'pgm_read_float(&{expr})' if self.use_flash else expr

    def _fmt1d(self, values) -> str:
        return '{' + ', '.join(f'{float(v):.8f}f' for v in values) + '}'

    def _fmt2d(self, values) -> str:
        rows = ['    {' + ', '.join(f'{float(v):.8f}f' for v in row) + '}'
                for row in values]
        return '{\n' + ',\n'.join(rows) + '\n}'

    # ------------------------------------------------------------------
    def _header(self) -> str:
        pm  = self._pm()
        p   = self._p
        mw  = self.max_width
        lines = [
            # CORREÇÃO: travessão substituído por hífen para evitar caracteres não ASCII
            '// GANModel.h  -  Auto-generated GAN Generator',
            '// Do NOT edit weights manually.',
            '',
            '#pragma once',
            '#include <math.h>',
            '',
        ]
        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']

        # Weight constants
        for layer in self._layers:
            i   = layer['index']
            inp = layer['in_features']
            out = layer['out_features']
            act = layer.get('activation', 'linear')
            lines += [f'// Generator block {i}  in={inp}  out={out}  act={act}']

            W = p[f'gen_{i}_weight']
            b = p[f'gen_{i}_bias']
            lines.append(
                f'static const float gen{i}_W[{out}][{inp}]{pm} = {self._fmt2d(W)};'
            )
            b_vals = b if b is not None else [[0.0]] * out
            lines.append(
                f'static const float gen{i}_b[{out}]{pm} = {self._fmt1d(b_vals if b is not None else [0.0]*out)};'
            )

            if layer.get('use_bn', False) and not layer.get('conditional', False):
                gm  = p.get(f'gen_{i}_bn_weight',       [1.0] * out)
                bt  = p.get(f'gen_{i}_bn_bias',         [0.0] * out)
                rm  = p.get(f'gen_{i}_bn_running_mean', [0.0] * out)
                rv  = p.get(f'gen_{i}_bn_running_var',  [1.0] * out)
                lines.append(f'static const float gen{i}_bn_gamma[{out}]{pm} = {self._fmt1d(gm)};')
                lines.append(f'static const float gen{i}_bn_beta[{out}]{pm}  = {self._fmt1d(bt)};')
                lines.append(f'static const float gen{i}_bn_rmean[{out}]{pm} = {self._fmt1d(rm)};')
                lines.append(f'static const float gen{i}_bn_rvar[{out}]{pm}  = {self._fmt1d(rv)};')
            lines.append('')

        # Class definition
        lines += [
            'class GANModel {',
            'public:',
            f'    void generate(const float* z, float* out);',
            '',
            'private:',
            f'    float buf_a[{mw}];',
            f'    float buf_b[{mw}];',
            '};',
            '',
        ]

        # generate() implementation
        rd  = self._rd
        lines += [
            'void GANModel::generate(const float* z, float* out) {',
            f'    // Copy z into working buffer',
            f'    for (int k = 0; k < {self.latent_dim}; k++) buf_a[k] = z[k];',
            f'    float* src = buf_a;',
            f'    float* dst = buf_b;',
            '',
        ]

        for li, layer in enumerate(self._layers):
            i   = layer['index']
            inp = layer['in_features']
            out = layer['out_features']
            act = layer.get('activation', 'linear')
            use_bn = layer.get('use_bn', False) and not layer.get('conditional', False)
            is_last = (li == len(self._layers) - 1)

            if is_last:
                dst_name = 'out'
            else:
                dst_name = 'dst'

            lines += [
                f'    // Block {i}: linear {inp}→{out}  act={act}  bn={use_bn}',
                f'    for (int k = 0; k < {out}; k++) {{',
                f'        float acc = {rd(f"gen{i}_b[k]")};',
                f'        for (int j = 0; j < {inp}; j++)',
                f'            acc += {rd(f"gen{i}_W[k][j]")} * src[j];',
            ]

            if use_bn:
                lines += [
                    f'        float rmean = {rd(f"gen{i}_bn_rmean[k]")};',
                    f'        float rvar  = {rd(f"gen{i}_bn_rvar[k]")};',
                    f'        float gamma = {rd(f"gen{i}_bn_gamma[k]")};',
                    f'        float beta  = {rd(f"gen{i}_bn_beta[k]")};',
                    f'        acc = gamma * (acc - rmean) / sqrtf(rvar + 1e-5f) + beta;',
                ]

            lines += ['        ' + ln for ln in self._act_c(act, 'acc')]
            lines += [
                f'        {dst_name}[k] = acc;',
                f'    }}',
            ]

            if not is_last:
                lines += [
                    f'    {{ float* tmp = src; src = dst; dst = tmp; }}',
                    '',
                ]

        lines += ['}']
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _act_c(self, act: str, var: str):
        a = act.lower()
        if a == 'relu':
            return [f'if ({var} < 0.0f) {var} = 0.0f;']
        elif a == 'leaky_relu':
            return [f'if ({var} < 0.0f) {var} *= 0.2f;']
        elif a == 'sigmoid':
            return [f'{var} = 1.0f / (1.0f + expf(-{var}));']
        elif a == 'tanh':
            return [f'{var} = tanhf({var});']
        elif a == 'gelu':
            return [f'{var} = 0.5f * {var} * (1.0f + tanhf(0.79788456f * ({var} + 0.044715f * {var} * {var} * {var})));']
        elif a == 'swish':
            return [f'{var} = {var} / (1.0f + expf(-{var}));']
        return []   # linear / softmax → identity

    # ------------------------------------------------------------------
    def _sketch(self, sample_z: np.ndarray, expected: np.ndarray) -> str:
        z_str   = ', '.join(f'{float(v):.8f}f' for v in sample_z)
        exp_str = ', '.join(f'{float(v):.6f}' for v in expected[:8])
        if len(expected) > 8:
            exp_str += ', ...'

        lines = [
            '/*',
            ' * GAN Generator -- Arduino verification sketch',
            ' * Generated automatically -- do not edit the weights.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * Latent dim  : {self.latent_dim}',
            f' * Output dim  : {self.output_dim}',
            ' *',
            f' * Input z (first 8 shown): [{", ".join(f"{float(v):.6f}" for v in sample_z[:8])}]',
            f' * Expected output (first 8): [{exp_str}]',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm the printed values match to at least 4 decimal places.',
            ' */',
            '',
            '#include "GANModel.h"',
            '',
            'GANModel model;',
            '',
            'void setup() {',
            '    Serial.begin(115200);',
            '    while (!Serial);',
            '',
            f'    const int LATENT_DIM = {self.latent_dim};',
            f'    const int OUTPUT_DIM = {self.output_dim};',
            '',
            '    // Verification latent vector (auto-generated by Python exporter)',
            f'    float z[LATENT_DIM] = {{',
            f'        {z_str}',
            f'    }};',
            '',
            '    float out[OUTPUT_DIM];',
            '    model.generate(z, out);',
            '',
            '    Serial.println("Generated output (first 8 dims):");',
            '    int n_print = OUTPUT_DIM < 8 ? OUTPUT_DIM : 8;',
            '    for (int i = 0; i < n_print; i++) {',
            '        Serial.print("  out["); Serial.print(i);',
            '        Serial.print("] = "); Serial.println(out[i], 6);',
            '    }',
            '}',
            '',
            'void loop() {',
            '    // Nothing to do here',
            '}',
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Helper: attach BN running stats to the JSON (call after model.eval())
# ---------------------------------------------------------------------------

def add_bn_stats_to_json(generator, json_path: str) -> None:
    """
    Append BatchNorm running statistics (running_mean, running_var) to an
    existing Generator JSON export.

    Must be called **after** ``model.eval()`` to ensure the running statistics
    are populated from training.

    Args:
        generator : The trained ``Generator`` instance (in eval mode).
        json_path : Path to the JSON file produced by ``export_to_json``.
                    The file is updated in place.
    """
    import torch
    with open(json_path, 'r') as f:
        data = json.load(f)

    p = data['parameters']

    for i, block in enumerate(generator.blocks):
        if block.norm is not None and not block.conditional:
            bn = block.norm
            p[f'gen_{i}_bn_running_mean'] = bn.running_mean.cpu().tolist()
            p[f'gen_{i}_bn_running_var']  = bn.running_var.cpu().tolist()

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"BN running stats appended → {json_path}")


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def generate_ino(
    json_path:  str,
    output_dir: str,
    board:      str  = 'esp32',
    use_flash:  bool = False,
) -> None:
    """
    Generate Arduino GAN Generator library files from a JSON model export.

    Args:
        json_path  : Path to the JSON file (with BN stats added).
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store weights in PROGMEM.
    """
    gen = ArduinoGANGenerator(json_path, output_dir, board, use_flash)
    gen.generate()