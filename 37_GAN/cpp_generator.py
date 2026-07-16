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

ConditionalBatchNorm is FULLY SUPPORTED now.
For each conditional block, the C code computes:
    gamma = gamma_fc(embed(label))
    beta  = beta_fc(embed(label))
and applies BN with these sample-specific affine parameters.
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

def _np_generate(data: dict, z: np.ndarray, label: int = 0) -> np.ndarray:
    """
    Pure-NumPy Generator forward pass (supports conditional generation).

    z    : (latent_dim,) float32 numpy array — a single latent vector.
    label: integer class label (only used if conditional).

    Returns a (output_dim,) float32 numpy array.
    """
    p      = data['parameters']
    layers = data['layers']
    arch   = data['architecture']
    cond   = arch.get('conditional', False)
    embed_dim = arch.get('embed_dim', 0)

    x = z.astype(np.float32)

    # Conditional embedding + concatenation
    if cond:
        embed_weight = np.array(p['label_embed_weight'], dtype=np.float32)
        emb = embed_weight[label]  # (embed_dim,)
        x = np.concatenate([x, emb])  # (latent_dim + embed_dim,)

    for layer in layers:
        i   = layer['index']
        act = layer.get('activation', 'linear')
        W   = np.array(p[f'gen_{i}_weight'], dtype=np.float32)
        b   = p[f'gen_{i}_bias']
        b   = np.array(b, dtype=np.float32) if b is not None else np.zeros(W.shape[0], dtype=np.float32)

        x = W @ x + b   # linear

        if layer.get('use_bn', False):
            gamma = None
            beta = None
            rmean = np.array(p[f'gen_{i}_bn_running_mean'], dtype=np.float32)
            rvar  = np.array(p[f'gen_{i}_bn_running_var'],  dtype=np.float32)
            eps   = 1e-5

            if layer.get('conditional', False):
                # ConditionalBatchNorm: gamma and beta are computed from the label embedding
                gamma_w = np.array(p[f'gen_{i}_bn_gamma_w'], dtype=np.float32)
                gamma_b = np.array(p[f'gen_{i}_bn_gamma_b'], dtype=np.float32)
                beta_w  = np.array(p[f'gen_{i}_bn_beta_w'],  dtype=np.float32)
                beta_b  = np.array(p[f'gen_{i}_bn_beta_b'],  dtype=np.float32)
                if 'label_embed_weight' in p:
                    emb_vec = np.array(p['label_embed_weight'], dtype=np.float32)[label]
                else:
                    raise RuntimeError("Conditional block but no label_embed_weight in JSON")
                gamma = gamma_w @ emb_vec + gamma_b
                beta  = beta_w  @ emb_vec + beta_b
            else:
                gamma = np.array(p[f'gen_{i}_bn_weight'], dtype=np.float32)
                beta  = np.array(p[f'gen_{i}_bn_bias'],   dtype=np.float32)

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

    Supports both unconditional and conditional (cGAN) generators.
    For conditional, the C++ generate() method receives an extra `label` argument.

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

        arch = self.data['architecture']
        self.num_classes = arch.get('num_classes', 0)
        self.embed_dim   = arch.get('embed_dim', 0)
        self.conditional = self.num_classes > 0

        if self.conditional:
            p = self.data['parameters']
            if 'label_embed_weight' not in p:
                raise ValueError(
                    "JSON is marked as conditional but 'label_embed_weight' is missing. "
                    "Make sure you exported with export_to_json() after training."
                )
            self.label_embed_weight = np.array(p['label_embed_weight'], dtype=np.float32)

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
        widths = []
        for l in self._layers:
            widths.append(l['in_features'])
            widths.append(l['out_features'])
        return max(widths) if widths else 0

    # ------------------------------------------------------------------
    def generate(self, sample_z: np.ndarray = None, sample_label: int = 0) -> None:
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_z    : Optional ``(latent_dim,)`` float32 array used to
                          generate a verification vector.  If ``None``, a random
                          sample is drawn with seed 0.
            sample_label: Label used for verification (only matters for cGAN).
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_z is None:
            np.random.seed(0)
            sample_z = np.random.randn(self.latent_dim).astype(np.float32)

        if self.conditional:
            expected = _np_generate(self.data, sample_z, sample_label)
        else:
            expected = _np_generate(self.data, sample_z)

        header_path = os.path.join(self.output_dir, 'GANModel.h')
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')

        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_z, expected, sample_label))

        print(f"Generated in : {self.output_dir}  (board: {self.board})")
        print(f"Latent dim   : {self.latent_dim}  Output dim: {self.output_dim}")
        print(f"Conditional  : {self.conditional}  Classes: {self.num_classes}")
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
            '// GANModel.h  -  Auto-generated GAN Generator',
            '// Do NOT edit weights manually.',
            '',
            '#pragma once',
            '#include <math.h>',
            '',
        ]
        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']

        # Conditional embedding table
        if self.conditional:
            lines += [
                f'// Conditional GAN: label embedding table ({self.num_classes} classes, {self.embed_dim} dims)',
                f'static const float label_embed[{self.num_classes}][{self.embed_dim}]{pm} = {self._fmt2d(self.label_embed_weight)};',
                '',
            ]

        # Weight constants
        for layer in self._layers:
            i   = layer['index']
            inp = layer['in_features']
            out = layer['out_features']
            act = layer.get('activation', 'linear')
            cond = layer.get('conditional', False)
            use_bn = layer.get('use_bn', False)

            lines += [f'// Generator block {i}  in={inp}  out={out}  act={act}  bn={use_bn}  cond={cond}']

            W = p[f'gen_{i}_weight']
            b = p[f'gen_{i}_bias']
            lines.append(
                f'static const float gen{i}_W[{out}][{inp}]{pm} = {self._fmt2d(W)};'
            )
            b_vals = b if b is not None else [0.0] * out
            lines.append(
                f'static const float gen{i}_b[{out}]{pm} = {self._fmt1d(b_vals if b is not None else [0.0]*out)};'
            )

            if use_bn:
                # running stats are always exported (by add_bn_stats_to_json)
                rm = p.get(f'gen_{i}_bn_running_mean', [0.0] * out)
                rv = p.get(f'gen_{i}_bn_running_var',  [1.0] * out)
                lines.append(f'static const float gen{i}_bn_rmean[{out}]{pm} = {self._fmt1d(rm)};')
                lines.append(f'static const float gen{i}_bn_rvar[{out}]{pm}  = {self._fmt1d(rv)};')

                if cond:
                    # ConditionalBatchNorm: gamma and beta are computed via linear layers
                    gm_w = p[f'gen_{i}_bn_gamma_w']
                    gm_b = p[f'gen_{i}_bn_gamma_b']
                    bt_w = p[f'gen_{i}_bn_beta_w']
                    bt_b = p[f'gen_{i}_bn_beta_b']
                    lines.append(f'static const float gen{i}_bn_gamma_w[{out}][{self.embed_dim}]{pm} = {self._fmt2d(gm_w)};')
                    lines.append(f'static const float gen{i}_bn_gamma_b[{out}]{pm} = {self._fmt1d(gm_b)};')
                    lines.append(f'static const float gen{i}_bn_beta_w[{out}][{self.embed_dim}]{pm}  = {self._fmt2d(bt_w)};')
                    lines.append(f'static const float gen{i}_bn_beta_b[{out}]{pm}  = {self._fmt1d(bt_b)};')
                else:
                    # Standard BatchNorm: fixed gamma/beta
                    gm = p.get(f'gen_{i}_bn_weight', [1.0] * out)
                    bt = p.get(f'gen_{i}_bn_bias',   [0.0] * out)
                    lines.append(f'static const float gen{i}_bn_gamma[{out}]{pm} = {self._fmt1d(gm)};')
                    lines.append(f'static const float gen{i}_bn_beta[{out}]{pm}  = {self._fmt1d(bt)};')

            lines.append('')

        # Class definition
        lines += [
            'class GANModel {',
            'public:',
        ]
        if self.conditional:
            lines.append('    void generate(const float* z, int label, float* out);')
        else:
            lines.append('    void generate(const float* z, float* out);')
        lines += [
            '',
            'private:',
            f'    float buf_a[{mw}];',
            f'    float buf_b[{mw}];',
        ]
        if self.conditional:
            lines.append(f'    float emb_buf[{self.embed_dim}];')
        lines += [
            '};',
            '',
        ]

        # ---- generate() implementation ----
        # *** FIX: the signature must match the declaration above ***
        rd  = self._rd
        if self.conditional:
            lines.append('void GANModel::generate(const float* z, int label, float* out) {')
        else:
            lines.append('void GANModel::generate(const float* z, float* out) {')

        # Body of the function
        if self.conditional:
            lines += [
                f'    // Conditional generation: embed the label and concatenate to z',
                f'    if (label < 0 || label >= {self.num_classes}) label = 0;',
                f'    for (int k = 0; k < {self.embed_dim}; k++) {{',
                f'        emb_buf[k] = label_embed[label][k];',
                f'    }}',
                f'    // Concatenate z and embedding into buf_a',
                f'    for (int k = 0; k < {self.latent_dim}; k++) buf_a[k] = z[k];',
                f'    for (int k = 0; k < {self.embed_dim}; k++) buf_a[{self.latent_dim} + k] = emb_buf[k];',
                f'    float* src = buf_a;',
            ]
        else:
            lines += [
                f'    // Copy z into working buffer',
                f'    for (int k = 0; k < {self.latent_dim}; k++) buf_a[k] = z[k];',
                f'    float* src = buf_a;',
            ]
        lines += [
            f'    float* dst = buf_b;',
            '',
        ]

        for li, layer in enumerate(self._layers):
            i   = layer['index']
            inp = layer['in_features']
            out = layer['out_features']
            act = layer.get('activation', 'linear')
            use_bn = layer.get('use_bn', False)
            cond = layer.get('conditional', False)
            is_last = (li == len(self._layers) - 1)

            if is_last:
                dst_name = 'out'
            else:
                dst_name = 'dst'

            lines += [
                f'    // Block {i}: linear {inp}→{out}  act={act}  bn={use_bn}  cond={cond}',
                f'    for (int k = 0; k < {out}; k++) {{',
                f'        float acc = {rd(f"gen{i}_b[k]")};',
                f'        for (int j = 0; j < {inp}; j++)',
                f'            acc += {rd(f"gen{i}_W[k][j]")} * src[j];',
            ]

            if use_bn:
                lines += [
                    f'        float rmean = {rd(f"gen{i}_bn_rmean[k]")};',
                    f'        float rvar  = {rd(f"gen{i}_bn_rvar[k]")};',
                ]
                if cond:
                    lines += [
                        f'        // Conditional BN: compute gamma and beta from label embedding',
                        f'        float gamma = {rd(f"gen{i}_bn_gamma_b[k]")};',
                        f'        for (int j = 0; j < {self.embed_dim}; j++) {{',
                        f'            gamma += {rd(f"gen{i}_bn_gamma_w[k][j]")} * emb_buf[j];',
                        f'        }}',
                        f'        float beta = {rd(f"gen{i}_bn_beta_b[k]")};',
                        f'        for (int j = 0; j < {self.embed_dim}; j++) {{',
                        f'            beta += {rd(f"gen{i}_bn_beta_w[k][j]")} * emb_buf[j];',
                        f'        }}',
                        f'        acc = gamma * (acc - rmean) / sqrtf(rvar + 1e-5f) + beta;',
                    ]
                else:
                    lines += [
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
    def _sketch(self, sample_z: np.ndarray, expected: np.ndarray, sample_label: int = 0) -> str:
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
            f' * Conditional : {self.conditional}',
        ]
        if self.conditional:
            lines.append(f' * Classes     : {self.num_classes}  (using label={sample_label} for verification)')
        lines += [
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
        ]
        if self.conditional:
            lines.append(f'    model.generate(z, {sample_label}, out);')
        else:
            lines.append('    model.generate(z, out);')
        lines += [
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

    This function now supports both unconditional and conditional BatchNorm.

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
        if block.norm is not None:
            bn = block.norm
            # ConditionalBatchNorm wraps a plain BatchNorm1d in .bn
            if hasattr(bn, 'bn'):
                bn_inner = bn.bn
            else:
                bn_inner = bn
            p[f'gen_{i}_bn_running_mean'] = bn_inner.running_mean.cpu().tolist()
            p[f'gen_{i}_bn_running_var']  = bn_inner.running_var.cpu().tolist()

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

    Now fully supports conditional GANs (cGANs) with ConditionalBatchNorm.

    Args:
        json_path  : Path to the JSON file (with BN stats added).
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store weights in PROGMEM.
    """
    gen = ArduinoGANGenerator(json_path, output_dir, board, use_flash)
    gen.generate()