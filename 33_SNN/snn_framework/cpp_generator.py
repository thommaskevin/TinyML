# cpp_generator.py
"""
Arduino / ESP32 C++ code generator for trained Spiking Neural Networks.

This module mirrors the BNN ArduinoGenerator in structure and public API.
It reads the JSON produced by ``utils.export_to_json`` and emits three files:

  SNNModel.h    — class declaration with PROGMEM weight arrays.
  SNNModel.cpp  — LIF forward-pass implementation.
  sketch.ino    — minimal usage example.

Supported layer types
---------------------
  SpikingLinear  — LIF neurons with a dense synaptic weight matrix.
  LeakyReadout   — Non-spiking integrator (output layer).

The generated C++ uses only fixed-point ``float`` arithmetic and the
standard ``<math.h>`` library, making it compatible with AVR (Uno/Nano)
and ESP32 targets.
"""

import json
import os
import numpy as np


class SNNArduinoGenerator:
    """
    Generate embedded C++ inference code for a Spiking Neural Network.

    Parameters
    ----------
    json_path : str
        Path to the JSON model file produced by ``export_to_json``.
    output_dir : str
        Directory in which to write the generated files.
    board : str
        Target board family: 'avr' (ATmega) or 'esp32'.
    use_flash : bool
        If True (AVR only), store weight arrays in PROGMEM to save RAM.
    task : str
        'regression', 'binary', or 'multiclass'. Controls the sketch output
        and the argmax / threshold logic in the predict function.
    """

    def __init__(self, json_path: str, output_dir: str,
                 board: str = 'avr', use_flash: bool = True,
                 task: str = 'regression'):
        self.json_path = json_path
        self.output_dir = output_dir
        self.board = board
        self.use_flash = use_flash
        self.task = task

        self.model_data: dict = {}
        self.num_steps: int = 25
        self.architecture: list = []
        self.params: dict = {}
        self.layer_infos: list = []

        self.input_size: int = 0
        self.max_buffer_size: int = 0
        self.num_classes: int = 1

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_json(self) -> None:
        with open(self.json_path, 'r') as f:
            self.model_data = json.load(f)

        self.num_steps = self.model_data.get('num_steps', 25)
        self.architecture = self.model_data['architecture']
        self.params = self.model_data['parameters']
        self._extract_layer_info()

    def _extract_layer_info(self) -> None:
        current_size = None
        for idx, layer in enumerate(self.architecture):
            info = {'index': idx, 'type': layer['type']}

            if layer['type'] == 'SpikingLinear':
                in_f = layer['in_features']
                out_f = layer['out_features']
                info.update({
                    'in_features':  in_f,
                    'out_features': out_f,
                    'beta':         layer.get('beta', 0.9),
                    'threshold':    layer.get('threshold', 1.0),
                    'reset_mode':   layer.get('reset_mode', 'zero'),
                    'weight':       self.params[f'layer_{idx}_weight'],
                    'bias':         self.params.get(f'layer_{idx}_bias'),
                    'output_size':  out_f,
                })
                current_size = out_f
                if idx == 0:
                    self.input_size = in_f

            elif layer['type'] == 'LeakyReadout':
                in_f = layer['in_features']
                out_f = layer['out_features']
                info.update({
                    'in_features':  in_f,
                    'out_features': out_f,
                    'beta':         layer.get('beta', 0.9),
                    'weight':       self.params[f'layer_{idx}_weight'],
                    'bias':         self.params.get(f'layer_{idx}_bias'),
                    'output_size':  out_f,
                })
                current_size = out_f
                self.num_classes = out_f

            self.layer_infos.append(info)

        sizes = [
            info.get('output_size', 0)
            for info in self.layer_infos
            if 'output_size' in info
        ]
        self.max_buffer_size = max(sizes) if sizes else 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self) -> None:
        self.load_json()
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'SNNModel.h'), 'w') as f:
            f.write(self._generate_header())
        with open(os.path.join(self.output_dir, 'SNNModel.cpp'), 'w') as f:
            f.write(self._generate_implementation())
        with open(os.path.join(self.output_dir, 'sketch.ino'), 'w') as f:
            f.write(self._generate_sketch())

        print(f"[SNNArduinoGenerator] Files written to '{self.output_dir}' "
              f"(board: {self.board}, steps: {self.num_steps}).")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _generate_header(self) -> str:
        fa = " PROGMEM" if (self.board == 'avr' and self.use_flash) else ""
        lines = [
            "#ifndef SNNMODEL_H",
            "#define SNNMODEL_H",
            "",
            "#include <stdint.h>",
        ]
        if self.board == 'avr':
            lines.append("#include <avr/pgmspace.h>")
        else:
            lines.append("#include <Arduino.h>")

        lines += [
            "",
            f"#define SNN_NUM_STEPS {self.num_steps}",
            f"#define SNN_INPUT_SIZE {self.input_size}",
            f"#define SNN_OUTPUT_SIZE {self.num_classes}",
            "",
            "class SNNModel {",
            "public:",
            "    SNNModel();",
            "    float predict(const float* input);",
            "",
            "private:",
            f"    static float spike_buf_a[{self.max_buffer_size}];",
            f"    static float spike_buf_b[{self.max_buffer_size}];",
        ]

        # Per-layer membrane potential arrays (one float per neuron)
        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx = info['index']
                out_f = info['out_features']
                lines.append(f"    static float u{idx}[{out_f}];   "
                             f"// membrane potential, layer {idx}")

        lines.append("")

        # Weight / bias declarations
        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx = info['index']
                out_f = info['out_features']
                in_f = info['in_features']
                lines.append(
                    f"    static const float{fa} w{idx}[{out_f}][{in_f}];"
                )
                if info.get('bias') is not None:
                    lines.append(
                        f"    static const float{fa} b{idx}[{out_f}];"
                    )

        lines += ["};", "", "#endif"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _generate_implementation(self) -> str:
        fa = " PROGMEM" if (self.board == 'avr' and self.use_flash) else ""
        lines = [
            '#include "SNNModel.h"',
            '#include <math.h>',
        ]
        if self.board == 'avr':
            lines.append('#include <avr/pgmspace.h>')

        lines += [
            "",
            f"float SNNModel::spike_buf_a[{self.max_buffer_size}] = {{0}};",
            f"float SNNModel::spike_buf_b[{self.max_buffer_size}] = {{0}};",
        ]

        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx = info['index']
                out_f = info['out_features']
                lines.append(
                    f"float SNNModel::u{idx}[{out_f}] = {{0}};"
                )

        lines.append("")

        # Weight / bias definitions
        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx = info['index']
                out_f = info['out_features']
                in_f = info['in_features']
                w = info['weight']

                lines.append(
                    f"const float SNNModel::w{idx}[{out_f}][{in_f}]{fa} = {{"
                )
                for row in w:
                    lines.append(
                        "    {" + ", ".join(repr(float(v)) for v in row) + "},"
                    )
                lines.append("};\n")

                if info.get('bias') is not None:
                    b = info['bias']
                    lines.append(
                        f"const float SNNModel::b{idx}[{out_f}]{fa} = {{"
                    )
                    lines.append(
                        "    " + ", ".join(repr(float(v)) for v in b) + "};"
                    )
                    lines.append("")

        lines += [
            "",
            "SNNModel::SNNModel() {",
            "    // Initialise membrane potentials to zero",
        ]
        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx = info['index']
                out_f = info['out_features']
                lines.append(
                    f"    for (int i = 0; i < {out_f}; i++) u{idx}[i] = 0.0f;"
                )
        lines += ["}", ""]

        lines.append(self._generate_predict_method())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # predict() method
    # ------------------------------------------------------------------

    def _generate_predict_method(self) -> str:
        # Build membrane reset lines for every stateful layer
        reset_lines = ["    // Reset membrane potentials (mirrors Python reset_states())"]
        for info in self.layer_infos:
            if info['type'] in ('SpikingLinear', 'LeakyReadout'):
                idx   = info['index']
                out_f = info['out_features']
                reset_lines.append(
                    f"    for (int i = 0; i < {out_f}; i++) u{idx}[i] = 0.0f;"
                )
        reset_lines.append("")

        lines = [
            "float SNNModel::predict(const float* input) {",
        ] + reset_lines + [
            f"    // Accumulate readout over {self.num_steps} time steps",
            f"    float readout_acc[{self.num_classes}] = {{0}};",
            "",
            f"    for (int t = 0; t < SNN_NUM_STEPS; t++) {{",
            "",
            "        // Copy input to spike buffer a",
            f"        for (int i = 0; i < {self.input_size}; i++)"
            "  spike_buf_a[i] = input[i];",
            "",
            "        float* cur_in  = spike_buf_a;",
            "        float* cur_out = spike_buf_b;",
            "        float* tmp;",
            f"        int    cur_size = {self.input_size};",
            "",
        ]

        for info in self.layer_infos:
            idx = info['index']
            t = info['type']

            if t == 'SpikingLinear':
                out_f = info['out_features']
                in_f = info['in_features']
                beta = info['beta']
                thr = info['threshold']
                reset_mode = info['reset_mode']

                lines.append(f"        // SpikingLinear layer {idx}")
                lines.append(f"        for (int i = 0; i < {out_f}; i++) {{")
                lines.append(f"            float I = 0.0f;")
                lines.append(f"            for (int j = 0; j < {in_f}; j++) {{")

                if self.board == 'avr' and self.use_flash:
                    lines.append(
                        f"                I += pgm_read_float(&w{idx}[i][j]) * cur_in[j];"
                    )
                else:
                    lines.append(f"                I += w{idx}[i][j] * cur_in[j];")

                lines.append(f"            }}")

                if info.get('bias') is not None:
                    if self.board == 'avr' and self.use_flash:
                        lines.append(
                            f"            I += pgm_read_float(&b{idx}[i]);"
                        )
                    else:
                        lines.append(f"            I += b{idx}[i];")

                # LIF membrane update
                if reset_mode == 'zero':
                    lines += [
                        f"            float spike_prev = (u{idx}[i] >= {thr}f) ? 1.0f : 0.0f;",
                        f"            u{idx}[i] = {beta}f * u{idx}[i] * (1.0f - spike_prev) + I;",
                    ]
                else:  # subtraction
                    lines += [
                        f"            float spike_prev = (u{idx}[i] >= {thr}f) ? 1.0f : 0.0f;",
                        f"            u{idx}[i] = {beta}f * (u{idx}[i] - spike_prev * {thr}f) + I;",
                    ]

                lines += [
                    f"            cur_out[i] = (u{idx}[i] >= {thr}f) ? 1.0f : 0.0f;",
                    f"        }}",
                    f"        tmp = cur_in; cur_in = cur_out; cur_out = tmp;",
                    f"        cur_size = {out_f};",
                    "",
                ]

            elif t == 'LeakyReadout':
                out_f = info['out_features']
                in_f = info['in_features']
                beta = info['beta']

                lines.append(f"        // LeakyReadout layer {idx}")
                lines.append(f"        for (int i = 0; i < {out_f}; i++) {{")
                lines.append(f"            float I = 0.0f;")
                lines.append(f"            for (int j = 0; j < {in_f}; j++) {{")

                if self.board == 'avr' and self.use_flash:
                    lines.append(
                        f"                I += pgm_read_float(&w{idx}[i][j]) * cur_in[j];"
                    )
                else:
                    lines.append(f"                I += w{idx}[i][j] * cur_in[j];")

                lines.append(f"            }}")

                if info.get('bias') is not None:
                    if self.board == 'avr' and self.use_flash:
                        lines.append(
                            f"            I += pgm_read_float(&b{idx}[i]);"
                        )
                    else:
                        lines.append(f"            I += b{idx}[i];")

                lines += [
                    f"            u{idx}[i] = {beta}f * u{idx}[i] + I;",
                    f"            cur_out[i] = u{idx}[i];",
                    f"            readout_acc[i] += cur_out[i];  "
                    f"// accumulate for time-averaging",
                    f"        }}",
                    "",
                ]

        # Close time-step loop
        lines.append("    }  // end time-step loop")
        lines.append("")

        # Time-average readout
        lines.append(f"    for (int i = 0; i < {self.num_classes}; i++)"
                     f"  readout_acc[i] /= SNN_NUM_STEPS;")
        lines.append("")

        # Final output logic
        if self.task == 'multiclass':
            lines += [
                "    int best_idx = 0;",
                "    float best_val = readout_acc[0];",
                f"    for (int i = 1; i < {self.num_classes}; i++) {{",
                "        if (readout_acc[i] > best_val) {",
                "            best_val = readout_acc[i];",
                "            best_idx = i;",
                "        }",
                "    }",
                "    return (float)best_idx;",
            ]
        elif self.task == 'binary':
            lines.append("    return readout_acc[0];  "
                         "// threshold at 0 for binary decision")
        else:
            lines.append("    return readout_acc[0];")

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sketch
    # ------------------------------------------------------------------

    def _generate_sketch(self) -> str:
        example_input = "{" + ", ".join(["0.5"] * self.input_size) + "}"
        lines = [
            "// SNN inference example sketch",
            '#include "SNNModel.h"',
            "",
            "SNNModel model;",
            "",
            "void setup() {",
            "  Serial.begin(115200);",
            f"  float input[SNN_INPUT_SIZE] = {example_input};",
            "  float output = model.predict(input);",
        ]

        if self.task == 'binary':
            lines += [
                '  Serial.print("Predicted logit: ");',
                "  Serial.println(output, 6);",
                '  Serial.print("Predicted class (0 or 1): ");',
                "  Serial.println(output > 0.0f ? 1 : 0);",
            ]
        elif self.task == 'multiclass':
            lines += [
                '  Serial.print("Predicted class: ");',
                "  Serial.println((int)output);",
            ]
        else:
            lines += [
                '  Serial.print("Predicted value: ");',
                "  Serial.println(output, 6);",
            ]

        lines += [
            "}",
            "",
            "void loop() {",
            "  // Nothing to do here",
            "}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrapper (mirrors generate_ino from the BNN codebase)
# ---------------------------------------------------------------------------

def generate_ino(json_path: str, output_dir: str,
                 board: str = 'avr', use_flash: bool = True,
                 task: str = 'regression') -> None:
    """
    Generate Arduino/ESP32 C++ files from a trained SNN JSON model.

    Parameters
    ----------
    json_path : str
    output_dir : str
    board : str   — 'avr' or 'esp32'
    use_flash : bool
    task : str    — 'regression', 'binary', or 'multiclass'
    """
    gen = SNNArduinoGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate()