# cpp_generator.py
"""
Arduino C++ code generator for Self-Organizing Map (SOM) models exported
via ``export_to_json``.

Design notes
------------
SOM inference on a microcontroller reduces to a single operation: finding
the Best Matching Unit (BMU) for an input vector x by computing the
Euclidean distance from x to every prototype weight vector w_k and
returning the index of the minimum.

    bmu = argmin_{k in 0..K-1}  sum_{j=0}^{d-1}  (x[j] - w[k][j])^2

The generated C++ implements this search in a tight for-loop over K neurons
and d features.  On AVR boards (Uno / Mega) the weight matrix is stored in
PROGMEM to conserve the 2 KB SRAM; on ESP32 and similar MCUs a normal
float array in flash is used.

The output of ``predict()`` is the flat BMU index k = bmu_row * n_cols + bmu_col.
Helper functions ``bmu_row()`` and ``bmu_col()`` decode it back to 2-D
coordinates.  An optional ``quantize()`` function reconstructs the input by
returning the BMU weight vector.

Bug history and fixes
---------------------
FIX 1 — PROGMEM weight access on AVR
    Weight values in PROGMEM must be read with pgm_read_float_near().
    A plain array dereference reads from SRAM (which holds garbage after
    the flash copy is not performed for PROGMEM).

FIX 2 — Accumulated distance must use squared differences
    Early versions computed sum(|x-w|) (L1) instead of sum((x-w)^2) (L2^2).
    The BMU is the same for both metrics on convex clusters, but L2^2 is
    faster (no sqrtf needed) and matches the training objective.

FIX 3 — UTF-8 file encoding on Windows
    All output files are written with encoding='utf-8' to avoid
    UnicodeEncodeError on cp1252 Windows systems.

FIX 4 — n_cols used for index decoding
    bmu_row = bmu_index / n_cols (integer division)
    bmu_col = bmu_index % n_cols
    An earlier version mistakenly used n_rows for both divisions.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy reference forward pass (mirrors generated C exactly)
# ---------------------------------------------------------------------------

def _np_predict(data: dict, x: np.ndarray) -> int:
    """
    Pure-NumPy BMU search for SOM models.

    Args:
        data : Parsed JSON dict produced by ``export_to_json``.
        x    : Input vector of shape ``(input_dim,)`` — a single sample.

    Returns:
        Flat BMU index k = bmu_row * n_cols + bmu_col.
    """
    w_list = data['weights']                       # [n_rows][n_cols][d]
    n_rows = data['architecture']['n_rows']
    n_cols = data['architecture']['n_cols']
    K      = n_rows * n_cols

    w_flat = np.array(w_list, dtype=np.float32).reshape(K, -1)
    diff   = w_flat - x.astype(np.float32)
    dists  = (diff ** 2).sum(axis=1)
    return int(dists.argmin())


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoSOMGenerator:
    """
    Generates a two-file Arduino library (``SOMModel.h`` + ``sketch.ino``)
    from a JSON export of a trained ``SOMModel``.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Directory where the generated files are written.
        board      : ``'avr'`` (Uno / Mega) or ``'esp32'`` (ESP32 / ESP32-S3).
        use_flash  : If ``True`` and ``board='avr'``, store weights in PROGMEM.
        task       : ``'cluster'`` (return BMU index) or
                     ``'quantize'`` (return BMU weight vector).
    """

    def __init__(
        self,
        json_path:  str,
        output_dir: str,
        board:      str  = 'avr',
        use_flash:  bool = True,
        task:       str  = 'cluster',
    ) -> None:
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board.lower()
        self.use_flash  = use_flash and (self.board == 'avr')
        self.task       = task.lower()
        self.model_data = None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)

    @property
    def _arch(self):  return self.model_data['architecture']
    @property
    def n_rows(self): return self._arch['n_rows']
    @property
    def n_cols(self): return self._arch['n_cols']
    @property
    def K(self):      return self.n_rows * self.n_cols
    @property
    def d(self):      return self._arch['input_dim']

    # ------------------------------------------------------------------
    def generate(self, sample_x: np.ndarray | None = None) -> None:
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_x : Optional ``(input_dim,)`` float32 array used to
                       compute a reference BMU for the verification sketch.
                       A zero vector is used if ``None``.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_x is None:
            sample_x = np.zeros(self.d, dtype=np.float32)

        expected_bmu = _np_predict(self.model_data, sample_x)
        exp_row = expected_bmu // self.n_cols
        exp_col = expected_bmu %  self.n_cols

        header_path = os.path.join(self.output_dir, 'SOMModel.h')
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')

        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_x, expected_bmu, exp_row, exp_col))

        print(f"Header  -> {header_path}")
        print(f"Sketch  -> {sketch_path}")
        print(f"Reference BMU: flat={expected_bmu}  row={exp_row}  col={exp_col}")

    # ------------------------------------------------------------------
    def _fmt_weight_array(self) -> str:
        """Format the full weight matrix as a C float array (flat, row-major)."""
        w_list = self.model_data['weights']
        w_flat = np.array(w_list, dtype=np.float32).flatten()
        vals   = ', '.join(f'{float(v):.8f}f' for v in w_flat)
        pgm    = ' PROGMEM' if self.use_flash else ''
        return (
            f'// SOM weights: shape [{self.K}][{self.d}]  '
            f'(flat row-major: neuron k -> w[k*{self.d} .. k*{self.d}+{self.d-1}])\n'
            f'const float{pgm} som_weights[{self.K * self.d}] = {{\n'
            f'  {vals}\n'
            f'}};\n'
        )

    # ------------------------------------------------------------------
    def _rd(self, expr: str) -> str:
        """Return the correct weight read expression for the target board."""
        if self.use_flash:
            return f'pgm_read_float_near(&{expr})'
        return expr

    # ------------------------------------------------------------------
    def _header(self) -> str:
        """Generate the full SOMModel.h content."""
        lines = [
            '#pragma once',
            '/*',
            ' * SOMModel.h -- Self-Organizing Map',
            ' * Auto-generated by cpp_generator.py -- do not edit weights manually.',
            ' *',
            f' * Map size    : {self.n_rows} x {self.n_cols} = {self.K} neurons',
            f' * Input dim   : {self.d}',
            f' * Topology    : {self._arch.get("topology", "rectangular")}',
            f' * Kernel      : {self._arch.get("kernel", "gaussian")}',
            f' * Board       : {self.board.upper()}',
            ' */',
            '',
            '#include <math.h>',
        ]
        if self.use_flash:
            lines += ['#include <avr/pgmspace.h>', '']
        else:
            lines.append('')

        lines.append(self._fmt_weight_array())
        lines += [
            'class SOMModel {',
            'public:',
            f'  static const int N_NEURONS  = {self.K};',
            f'  static const int INPUT_DIM  = {self.d};',
            f'  static const int N_ROWS     = {self.n_rows};',
            f'  static const int N_COLS     = {self.n_cols};',
            '',
            '  /**',
            '   * Find the Best Matching Unit (BMU) for input vector x.',
            '   *',
            '   * @param x  Input array of length INPUT_DIM.',
            '   * @return   Flat BMU index in [0, N_NEURONS).  Decode with',
            '   *           bmu_row() and bmu_col().',
            '   */',
            '  int predict(const float* x) {',
            '    int   best_k    = 0;',
            '    float best_dist = 3.4028235e+38f;   // FLT_MAX',
            '',
            f'    for (int k = 0; k < {self.K}; k++) {{',
            '      float dist = 0.0f;',
            f'      for (int j = 0; j < {self.d}; j++) {{',
            f'        float diff = x[j] - {self._rd(f"som_weights[k * {self.d} + j]")};',
            '        dist += diff * diff;',
            '        // Early exit: prune if already worse than best',
            '        if (dist >= best_dist) break;',
            '      }',
            '      if (dist < best_dist) {',
            '        best_dist = dist;',
            '        best_k    = k;',
            '      }',
            '    }',
            '    return best_k;',
            '  }',
            '',
            '  /** Decode flat BMU index to row coordinate. */',
            '  int bmu_row(int bmu_index) { return bmu_index / N_COLS; }',
            '',
            '  /** Decode flat BMU index to column coordinate. */',
            '  int bmu_col(int bmu_index) { return bmu_index % N_COLS; }',
            '',
            '  /**',
            '   * Quantize: copy the BMU weight vector into output[].',
            '   *',
            '   * @param bmu_index  Flat BMU index returned by predict().',
            '   * @param output     Caller-allocated array of length INPUT_DIM.',
            '   */',
            '  void quantize(int bmu_index, float* output) {',
            f'    int base = bmu_index * {self.d};',
            f'    for (int j = 0; j < {self.d}; j++)',
            f'      output[j] = {self._rd("som_weights[base + j]")};',
            '  }',
            '',
            '  /**',
            '   * Quantization error: Euclidean distance from x to its BMU weight.',
            '   *',
            '   * @param x          Input array of length INPUT_DIM.',
            '   * @param bmu_index  Flat BMU index returned by predict().',
            '   * @return           L2 distance (not squared).',
            '   */',
            '  float quantization_error(const float* x, int bmu_index) {',
            '    float proto[INPUT_DIM];',
            '    quantize(bmu_index, proto);',
            '    float sq = 0.0f;',
            f'    for (int j = 0; j < {self.d}; j++) {{',
            '      float d = x[j] - proto[j];',
            '      sq += d * d;',
            '    }',
            '    return sqrtf(sq);',
            '  }',
            '};',
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _sketch(
        self,
        sample_x:    np.ndarray,
        expected_bmu: int,
        exp_row:     int,
        exp_col:     int,
    ) -> str:
        """Generate the verification sketch content."""
        flat_str = ', '.join(f'{float(v):.8f}f' for v in sample_x)
        lines = [
            '/*',
            ' * SOMModel -- Arduino verification sketch',
            ' * Auto-generated -- do not edit the weights.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * Input dim      : {self.d}',
            f' * Map size       : {self.n_rows} x {self.n_cols}',
            f' * Input values   : [{", ".join(f"{float(v):.6f}" for v in sample_x)}]',
            ' *',
            f' * Expected BMU   : flat={expected_bmu}  '
            f'row={exp_row}  col={exp_col}',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm the printed BMU matches the expected values above.',
            ' *',
            ' * Acceptable tolerance: exact integer match for BMU index.',
            ' */',
            '',
            '#include "SOMModel.h"',
            '',
            'SOMModel model;',
            '',
            'void setup() {',
            '  Serial.begin(115200);',
            '  while (!Serial);',
            '',
            f'  const int INPUT_DIM = {self.d};',
            f'  float x[INPUT_DIM] = {{ {flat_str} }};',
            '',
            '  int bmu = model.predict(x);',
            '',
            f'  // Expected: flat={expected_bmu}  row={exp_row}  col={exp_col}',
            '  Serial.print("BMU flat index : "); Serial.println(bmu);',
            '  Serial.print("BMU row        : "); Serial.println(model.bmu_row(bmu));',
            '  Serial.print("BMU col        : "); Serial.println(model.bmu_col(bmu));',
            '  Serial.print("Quant. error   : ");',
            '  Serial.println(model.quantization_error(x, bmu), 6);',
        ]
        if self.task == 'quantize':
            lines += [
                '',
                f'  float proto[INPUT_DIM];',
                '  model.quantize(bmu, proto);',
                '  Serial.println("BMU weight vector:");',
                f'  for (int j = 0; j < INPUT_DIM; j++) {{',
                '    Serial.print("  w["); Serial.print(j);',
                '    Serial.print("] = "); Serial.println(proto[j], 8);',
                '  }',
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
    task:       str  = 'cluster',
    sample_x:   np.ndarray | None = None,
) -> None:
    """
    Generate Arduino SOM library files from a JSON model export.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store weights in PROGMEM.
        task       : ``'cluster'`` (return BMU index) or
                     ``'quantize'`` (return BMU weight vector).
        sample_x   : Optional reference input of shape ``(input_dim,)``.
    """
    gen = ArduinoSOMGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate(sample_x)
