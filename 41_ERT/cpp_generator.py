# cpp_generator.py
"""
Arduino C++ code generator for Extra Trees (Extremely Randomized Trees)
models exported via ``export_to_json``.

Design notes
------------
Extra Trees inference on a microcontroller reduces to traversing B binary
decision trees and aggregating their leaf predictions.  Each tree traversal
is an O(depth) sequence of threshold comparisons:

    node = root
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    prediction = node.value

The generated C++ encodes each tree as two parallel arrays of structs
(``InternalNode`` and ``LeafNode``) to avoid dynamic memory allocation:

  - ``InternalNode`` : {feature_idx, threshold, left_child, right_child}
                       left_child / right_child are indices into the
                       InternalNode or LeafNode array; a negative index
                       signals a leaf (absolute value is the leaf index).
  - ``LeafNode``     : For classification: {class_probs[K]}
                       For regression:    {value}

Traversal uses an iterative stack-free approach (tail recursion via index
chasing) that requires zero additional memory beyond the two flat arrays.

On AVR boards, both node arrays are stored in PROGMEM to conserve the 2 KB
SRAM.  On ESP32, normal flash-allocated const arrays are used.

Bug history and fixes
---------------------
FIX 1 -- 32-bit index overflow on AVR
    Left/right child indices stored as int8_t overflow for trees with
    more than 127 nodes.  Changed to int16_t (supports up to 32767 nodes).

FIX 2 -- Probability accumulation for classification
    Early versions summed integer class counts rather than float probabilities,
    causing large ensembles to overflow int8 or int16 accumulators.
    Changed to float32 accumulators summed across trees.

FIX 3 -- UTF-8 encoding for output files on Windows
    All output files opened with encoding='utf-8' to prevent
    UnicodeEncodeError on cp1252 Windows systems.

FIX 4 -- n_classes from metadata, not from leaf value length
    Leaf value arrays can be shorter than n_classes when some classes
    are absent from a leaf's training subset.  n_classes is now taken
    directly from the JSON metadata field rather than inferred from
    the first leaf encountered.
"""

import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# NumPy reference forward pass (mirrors generated C exactly)
# ---------------------------------------------------------------------------

def _np_predict_one(tree_dict: dict, x: np.ndarray, task: str, n_classes: int):
    """
    Traverse a single serialised tree for input vector x.

    Args:
        tree_dict : Serialised tree root dict (from export_to_json).
        x         : Input vector of shape ``(p,)``.
        task      : ``'classification'`` or ``'regression'``.
        n_classes : Number of output classes (classification).

    Returns:
        For classification: np.ndarray of shape ``(n_classes,)`` (class probs).
        For regression:     float scalar.
    """
    node = tree_dict
    while node is not None and not node['is_leaf']:
        if x[node['feature_idx']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    if node is None:
        return np.zeros(n_classes) if task == 'classification' else 0.0
    val = node['value']
    if task == 'classification':
        probs = np.zeros(n_classes, dtype=np.float32)
        for k, v in enumerate(val):
            if k < n_classes:
                probs[k] = float(v)
        return probs
    return float(val[0])


def _np_predict(data: dict, x: np.ndarray):
    """
    Full Extra Trees reference prediction for a single sample.

    Args:
        data : Parsed JSON dict from ``export_to_json``.
        x    : Input vector of shape ``(p,)``.

    Returns:
        Predicted class index (classification) or float mean (regression).
    """
    task      = data['architecture']['task']
    n_classes = data['metadata']['n_classes']
    trees     = data['trees']

    if task == 'classification':
        accum = np.zeros(n_classes, dtype=np.float64)
        for t in trees:
            accum += _np_predict_one(t, x, task, n_classes)
        accum /= max(len(trees), 1)
        return int(accum.argmax())
    else:
        total = sum(
            _np_predict_one(t, x, task, n_classes) for t in trees
        )
        return total / max(len(trees), 1)


# ---------------------------------------------------------------------------
# Tree flattening helpers
# ---------------------------------------------------------------------------

def _flatten_tree(root_dict: dict, task: str, n_classes: int):
    """
    Convert a nested tree dict into two flat arrays suitable for C generation.

    Returns
    -------
    internal_nodes : list of dicts {feature_idx, threshold, left, right}
                     left/right are indices; negative = -(leaf_idx+1).
    leaf_nodes     : list of lists (class probs for classification; [value] for
                     regression).
    """
    internal_nodes = []
    leaf_nodes     = []

    def _visit(node):
        if node is None:
            # Should not happen in a valid tree
            lidx = len(leaf_nodes)
            leaf_nodes.append([0.0] * max(n_classes, 1))
            return -(lidx + 1)

        if node['is_leaf']:
            lidx = len(leaf_nodes)
            val  = node['value'] or [0.0]
            if task == 'classification':
                probs = [0.0] * n_classes
                for k, v in enumerate(val):
                    if k < n_classes:
                        probs[k] = float(v)
                leaf_nodes.append(probs)
            else:
                leaf_nodes.append([float(val[0])])
            return -(lidx + 1)

        # Internal node
        nidx = len(internal_nodes)
        internal_nodes.append({
            'feature_idx': node['feature_idx'],
            'threshold':   node['threshold'],
            'left':        None,
            'right':       None,
        })
        internal_nodes[nidx]['left']  = _visit(node['left'])
        internal_nodes[nidx]['right'] = _visit(node['right'])
        return nidx

    _visit(root_dict)
    return internal_nodes, leaf_nodes


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ArduinoETGenerator:
    """
    Generates a two-file Arduino library (``ETModel.h`` + ``sketch.ino``)
    from a JSON export of a trained ``ExtraTreesModel``.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Directory where the generated files are written.
        board      : ``'avr'`` (Uno / Mega) or ``'esp32'`` (ESP32 / ESP32-S3).
        use_flash  : If ``True`` and ``board='avr'``, store node arrays in
                     PROGMEM.
        task       : ``'classification'`` or ``'regression'``.
    """

    def __init__(
        self,
        json_path:  str,
        output_dir: str,
        board:      str  = 'avr',
        use_flash:  bool = True,
        task:       str  = 'classification',
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
    def _arch(self): return self.model_data['architecture']
    @property
    def _meta(self): return self.model_data['metadata']
    @property
    def n_trees(self): return len(self.model_data['trees'])
    @property
    def n_features(self): return self._meta['n_features']
    @property
    def n_classes(self): return self._meta['n_classes']

    # ------------------------------------------------------------------
    def generate(self, sample_x: np.ndarray | None = None) -> None:
        """
        Generate the Arduino library files and write them to ``output_dir``.

        Args:
            sample_x : Optional ``(n_features,)`` float32 array used to
                       compute a reference prediction for the verification
                       sketch.  A zero vector is used if ``None``.
        """
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_x is None:
            sample_x = np.zeros(self.n_features, dtype=np.float32)

        expected = _np_predict(self.model_data, sample_x)

        header_path = os.path.join(self.output_dir, 'ETModel.h')
        sketch_path = os.path.join(self.output_dir, 'sketch.ino')

        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(self._header())
        with open(sketch_path, 'w', encoding='utf-8') as f:
            f.write(self._sketch(sample_x, expected))

        print(f"Header  -> {header_path}")
        print(f"Sketch  -> {sketch_path}")
        if self.task == 'classification':
            print(f"Reference prediction: class {expected}")
        else:
            print(f"Reference prediction: {float(expected):.6f}")

    # ------------------------------------------------------------------
    def _progmem(self, decl: str) -> str:
        return decl.replace('const ', 'const PROGMEM ') if self.use_flash else decl

    def _rd_float(self, expr: str) -> str:
        return f'pgm_read_float_near(&{expr})' if self.use_flash else expr

    def _rd_int16(self, expr: str) -> str:
        return f'(int16_t)pgm_read_word(&{expr})' if self.use_flash else expr

    # ------------------------------------------------------------------
    def _header(self) -> str:
        lines = [
            '#pragma once',
            '/*',
            ' * ETModel.h -- Extra Trees (Extremely Randomized Trees)',
            ' * Auto-generated by cpp_generator.py -- do not edit manually.',
            ' *',
            f' * n_estimators : {self.n_trees}',
            f' * n_features   : {self.n_features}',
            f' * n_classes    : {self.n_classes}  (regression: 1)',
            f' * task         : {self.task}',
            f' * board        : {self.board.upper()}',
            ' */',
            '',
            '#include <math.h>',
            '#include <stdint.h>',
        ]
        if self.use_flash:
            pass
            #lines += ['#include <avr/pgmspace.h>', '']
        else:
            lines.append('')

        trees     = self.model_data['trees']
        n_classes = self.n_classes

        # Emit per-tree node arrays
        for b, tree_dict in enumerate(trees):
            int_nodes, leaf_nodes = _flatten_tree(
                tree_dict, self.task, n_classes
            )

            # Internal node struct array
            if int_nodes:
                lines.append(
                    self._progmem(
                        f'const int16_t et{b}_int[{len(int_nodes)}][4] = {{'
                    )
                )
                for nd in int_nodes:
                    lines.append(
                        f'  {{{nd["feature_idx"]}, '
                        f'{nd["left"]}, {nd["right"]}, 0}},  '
                        f'/* threshold stored separately */'
                    )
                lines.append('};\n')

                lines.append(
                    self._progmem(
                        f'const float et{b}_thr[{len(int_nodes)}] = {{'
                    )
                )
                thresh_vals = ', '.join(
                    f'{nd["threshold"]:.8f}f' for nd in int_nodes
                )
                lines.append(f'  {thresh_vals}')
                lines.append('};\n')

            # Leaf node array
            n_leaf_vals = n_classes if self.task == 'classification' else 1
            lines.append(
                self._progmem(
                    f'const float et{b}_leaf[{len(leaf_nodes)}][{n_leaf_vals}] = {{'
                )
            )
            for lf in leaf_nodes:
                vals = ', '.join(f'{float(v):.8f}f' for v in lf)
                lines.append(f'  {{{vals}}},')
            lines.append('};\n')

        # Class: ETModel
        K = max(n_classes, 1)
        lines += [
            'class ETModel {',
            'public:',
            f'  static const int N_FEATURES = {self.n_features};',
            f'  static const int N_CLASSES  = {K};',
            f'  static const int N_TREES    = {self.n_trees};',
            '',
        ]

        # _traverse helper — navigates one tree
        lines += [
            '  /* Traverse one tree; returns leaf index (>=0). */',
            '  int _traverse(const int16_t* int_arr, const float* thr_arr,',
            '                const float* x, int root_internal_nodes) {',
            '    int16_t cur = 0;',
            '    while (cur >= 0) {',
            f'      int16_t feat  = {self._rd_int16("int_arr[cur * 4]")};',
            f'      int16_t left  = {self._rd_int16("int_arr[cur * 4 + 1]")};',
            f'      int16_t right = {self._rd_int16("int_arr[cur * 4 + 2]")};',
            f'      float   thr   = {self._rd_float("thr_arr[cur]")};',
            '      cur = (x[feat] <= thr) ? left : right;',
            '    }',
            '    return (int)(-(cur + 1));   /* leaf index */',
            '  }',
            '',
        ]

        if self.task == 'classification':
            lines += [
                '  /**',
                '   * Predict the class index for input vector x.',
                '   * @param x  Input array of length N_FEATURES.',
                '   * @return   Predicted class index in [0, N_CLASSES).',
                '   */',
                '  int predict(const float* x) {',
                f'    float accum[{K}] = {{0}};',
            ]
            for b, tree_dict in enumerate(trees):
                int_nodes, leaf_nodes = _flatten_tree(
                    tree_dict, self.task, n_classes
                )
                n_int = len(int_nodes)
                if n_int == 0:
                    # Single-leaf tree
                    lines.append(
                        f'    for (int k = 0; k < {K}; k++) '
                        f'accum[k] += {self._rd_float(f"et{b}_leaf[0][k]")};'
                    )
                else:
                    lines += [
                        f'    {{',
                        f'      int lidx = _traverse((const int16_t*)et{b}_int, et{b}_thr, x, {n_int});',
                        f'      for (int k = 0; k < {K}; k++)',
                        f'        accum[k] += {self._rd_float(f"et{b}_leaf[lidx][k]")};',
                        f'    }}',
                    ]
            lines += [
                f'    int best = 0;',
                f'    for (int k = 1; k < {K}; k++)',
                f'      if (accum[k] > accum[best]) best = k;',
                '    return best;',
                '  }',
                '',
            ]
        else:
            lines += [
                '  /**',
                '   * Predict the regression value for input vector x.',
                '   * @param x  Input array of length N_FEATURES.',
                '   * @return   Predicted value (ensemble mean).',
                '   */',
                '  float predict(const float* x) {',
                '    float total = 0.0f;',
            ]
            for b, tree_dict in enumerate(trees):
                int_nodes, leaf_nodes = _flatten_tree(
                    tree_dict, self.task, n_classes
                )
                n_int = len(int_nodes)
                if n_int == 0:
                    lines.append(
                        f'    total += {self._rd_float(f"et{b}_leaf[0][0]")};'
                    )
                else:
                    lines += [
                        f'    {{',
                        f'      int lidx = _traverse((const int16_t*)et{b}_int, et{b}_thr, x, {n_int});',
                        f'      total += {self._rd_float(f"et{b}_leaf[lidx][0]")};',
                        f'    }}',
                    ]
            lines += [
                f'    return total / {self.n_trees}.0f;',
                '  }',
                '',
            ]

        lines.append('};')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _sketch(self, sample_x: np.ndarray, expected) -> str:
        flat_str = ', '.join(f'{float(v):.8f}f' for v in sample_x)
        if self.task == 'classification':
            exp_comment = f'Expected class: {int(expected)}'
            result_line = (
                '  Serial.print("Predicted class : "); '
                'Serial.println(model.predict(x));'
            )
        else:
            exp_comment = f'Expected value: {float(expected):.8f}'
            result_line = (
                '  Serial.print("Predicted value : "); '
                'Serial.println(model.predict(x), 8);'
            )
        lines = [
            '/*',
            ' * ETModel -- Arduino verification sketch',
            ' * Auto-generated -- do not edit the weights.',
            ' *',
            ' * VERIFICATION GUIDE',
            ' * -------------------',
            f' * n_features : {self.n_features}',
            f' * n_trees    : {self.n_trees}',
            f' * task       : {self.task}',
            f' * {exp_comment}',
            ' *',
            ' * Upload this sketch, open Serial Monitor at 115200 baud,',
            ' * and confirm the printed prediction matches the expected',
            ' * value above.',
            ' */',
            '',
            '#include "ETModel.h"',
            '',
            'ETModel model;',
            '',
            'void setup() {',
            '  Serial.begin(115200);',
            '  while (!Serial);',
            '',
            f'  float x[{self.n_features}] = {{ {flat_str} }};',
            '',
            f'  // {exp_comment}',
            f'  {result_line}',
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
    task:       str  = 'classification',
    sample_x:   np.ndarray | None = None,
) -> None:
    """
    Generate Arduino Extra Trees library files from a JSON model export.

    Args:
        json_path  : Path to the JSON file produced by ``export_to_json``.
        output_dir : Destination directory for the generated files.
        board      : ``'avr'`` or ``'esp32'``.
        use_flash  : If ``True`` and AVR board, store node arrays in PROGMEM.
        task       : ``'classification'`` or ``'regression'``.
        sample_x   : Optional reference input of shape ``(n_features,)``.
    """
    gen = ArduinoETGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate(sample_x)
