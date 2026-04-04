# cpp_generator.py
import json
import os
import numpy as np

class ArduinoGenerator:
    def __init__(self, json_path, output_dir, board='avr', quantize=False,
                 use_flash=True, mode='deterministic', task='regression'):
        self.json_path = json_path
        self.output_dir = output_dir
        self.board = board
        self.quantize = quantize
        self.use_flash = use_flash
        self.mode = mode
        self.task = task
        self.model_data = None
        self.architecture = []
        self.params = {}
        self.layer_infos = []
        self.max_buffer_size = 0
        self.input_size = 0
        self.num_classes = 1

    def load_json(self):
        with open(self.json_path, 'r') as f:
            self.model_data = json.load(f)
        self.architecture = self.model_data['architecture']
        self.params = self.model_data['parameters']
        self._extract_layer_info()

    def _extract_layer_info(self):
        current_size = None
        for idx, layer in enumerate(self.architecture):
            info = {'index': idx, 'type': layer['type']}
            if layer['type'] == 'BayesianLinear':
                in_f = layer['in_features']
                out_f = layer['out_features']
                info['in_features'] = in_f
                info['out_features'] = out_f
                info['weight_mu'] = self.params[f'layer_{idx}_weight_mu']
                info['weight_logvar'] = self.params.get(f'layer_{idx}_weight_logvar')
                info['bias_mu'] = self.params.get(f'layer_{idx}_bias_mu')
                info['bias_logvar'] = self.params.get(f'layer_{idx}_bias_logvar')
                info['output_size'] = out_f
                current_size = out_f
                if idx == 0:
                    self.input_size = in_f
                if idx == len(self.architecture)-1:
                    self.num_classes = out_f
            elif layer['type'] in ['ReLU', 'Tanh', 'Sigmoid']:
                info['activation'] = layer['type']
                info['output_size'] = current_size
            self.layer_infos.append(info)
        sizes = [info.get('output_size', 0) for info in self.layer_infos if 'output_size' in info]
        self.max_buffer_size = max(sizes) if sizes else 0

    def generate(self):
        self.load_json()
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'BNNModel.h'), 'w') as f:
            f.write(self._generate_header())
        with open(os.path.join(self.output_dir, 'BNNModel.cpp'), 'w') as f:
            f.write(self._generate_implementation())
        with open(os.path.join(self.output_dir, 'sketch.ino'), 'w') as f:
            f.write(self._generate_sketch())
        print(f"Arduino files generated in: {self.output_dir} for board: {self.board}")

    def _generate_header(self):
        data_type = "float"
        lines = []
        lines.append("#ifndef BNNMODEL_H")
        lines.append("#define BNNMODEL_H")
        lines.append("")
        lines.append("#include <stdint.h>")
        if self.board == 'avr':
            lines.append("#include <avr/pgmspace.h>")
        elif self.board == 'esp32':
            lines.append("#include <Arduino.h>")
        lines.append("")
        lines.append("class BNNModel {")
        lines.append("public:")
        lines.append("    BNNModel();")
        lines.append(f"    {data_type} predict(const {data_type}* input);")
        lines.append("")
        lines.append("private:")
        # CORREÇÃO: usar max_buffer_size em vez de input_size para evitar overflow
        lines.append(f"    static {data_type} input_buffer[{self.max_buffer_size}];")
        if self.max_buffer_size > 0:
            lines.append(f"    static {data_type} hidden_buffer[{self.max_buffer_size}];")
        lines.append("")
        storage = "static const"
        flash_attr = " PROGMEM" if (self.board == 'avr' and self.use_flash) else ""
        for info in self.layer_infos:
            if info['type'] == 'BayesianLinear':
                idx = info['index']
                out_f = info['out_features']
                in_f = info['in_features']
                lines.append(f"    {storage} float{flash_attr} w{idx}_mu[{out_f}][{in_f}];")
                if info['bias_mu'] is not None:
                    lines.append(f"    {storage} float{flash_attr} b{idx}_mu[{out_f}];")
        lines.append("};")
        lines.append("")
        lines.append("#endif")
        return "\n".join(lines)

    def _generate_implementation(self):
        data_type = "float"
        flash_attr = " PROGMEM" if (self.board == 'avr' and self.use_flash) else ""
        lines = []
        lines.append('#include "BNNModel.h"')
        lines.append('#include <math.h>')
        if self.board == 'avr':
            lines.append('#include <avr/pgmspace.h>')
        lines.append("")

        # CORREÇÃO: input_buffer com tamanho max_buffer_size
        lines.append(f"{data_type} BNNModel::input_buffer[{self.max_buffer_size}] = {{0}};")
        if self.max_buffer_size > 0:
            lines.append(f"{data_type} BNNModel::hidden_buffer[{self.max_buffer_size}] = {{0}};")
        lines.append("")

        for info in self.layer_infos:
            if info['type'] == 'BayesianLinear':
                idx = info['index']
                out_f = info['out_features']
                in_f = info['in_features']
                w = info['weight_mu']
                lines.append(f"const float BNNModel::w{idx}_mu[{out_f}][{in_f}] {flash_attr} = {{")
                for i in range(out_f):
                    row = w[i]
                    lines.append("    {" + ", ".join(repr(x) for x in row) + "},")
                lines.append("};\n")

                if info['bias_mu'] is not None:
                    b = info['bias_mu']
                    lines.append(f"const float BNNModel::b{idx}_mu[{out_f}] {flash_attr} = {{")
                    lines.append("    " + ", ".join(repr(x) for x in b) + "};\n")
        lines.append("")

        lines.append("BNNModel::BNNModel() {}")
        lines.append("")
        lines.append(self._generate_predict_method())
        return "\n".join(lines)

    def _generate_predict_method(self):
        data_type = "float"
        lines = []
        lines.append(f"{data_type} BNNModel::predict(const {data_type}* input) {{")
        lines.append(f"    {data_type}* temp;")
        lines.append(f"    for (int i = 0; i < {self.input_size}; i++) input_buffer[i] = input[i];")
        lines.append("")
        lines.append(f"    {data_type}* current_in = input_buffer;")
        lines.append(f"    {data_type}* current_out = hidden_buffer;")
        lines.append(f"    int current_size = {self.input_size};")
        lines.append("")

        for info in self.layer_infos:
            if info['type'] == 'BayesianLinear':
                idx = info['index']
                out_f = info['out_features']
                in_f = info['in_features']

                lines.append(f"    // Linear layer {idx}")
                lines.append(f"    for (int i = 0; i < {out_f}; i++) {{")
                lines.append(f"        float acc = 0.0f;")
                lines.append(f"        for (int j = 0; j < {in_f}; j++) {{")
                if self.board == 'avr' and self.use_flash:
                    lines.append(f"            acc += pgm_read_float(&w{idx}_mu[i][j]) * current_in[j];")
                else:
                    lines.append(f"            acc += w{idx}_mu[i][j] * current_in[j];")
                lines.append(f"        }}")
                if info['bias_mu'] is not None:
                    if self.board == 'avr' and self.use_flash:
                        lines.append(f"        acc += pgm_read_float(&b{idx}_mu[i]);")
                    else:
                        lines.append(f"        acc += b{idx}_mu[i];")
                lines.append(f"        current_out[i] = acc;")
                lines.append(f"    }}")
                
                lines.append(f"    temp = current_in;")
                lines.append(f"    current_in = current_out;")
                lines.append(f"    current_out = temp;")
                lines.append(f"    current_size = {out_f};")
                lines.append("")
            elif info['type'] in ['ReLU', 'Tanh', 'Sigmoid']:
                act = info['type'].lower()
                lines.append(f"    // Activation {act}")
                if act == 'relu':
                    lines.append(f"    for (int i = 0; i < current_size; i++) {{")
                    lines.append(f"        if (current_in[i] < 0) current_in[i] = 0;")
                    lines.append(f"    }}")
                elif act == 'tanh':
                    lines.append(f"    for (int i = 0; i < current_size; i++) current_in[i] = tanh(current_in[i]);")
                elif act == 'sigmoid':
                    lines.append(f"    for (int i = 0; i < current_size; i++) current_in[i] = 1.0f / (1.0f + exp(-current_in[i]));")
                lines.append("")

        if self.task == 'multiclass':
            lines.append(f"    int best_idx = 0;")
            lines.append(f"    float best_val = current_in[0];")
            lines.append(f"    for (int i = 1; i < current_size; i++) {{")
            lines.append(f"        if (current_in[i] > best_val) {{")
            lines.append(f"            best_val = current_in[i];")
            lines.append(f"            best_idx = i;")
            lines.append(f"        }}")
            lines.append(f"    }}")
            lines.append(f"    return (float)best_idx;")
        else:
            lines.append(f"    return current_in[0];")
        lines.append("}")
        return "\n".join(lines)

    def _generate_sketch(self):
        data_type = "float"
        lines = []
        lines.append("// BNN model example sketch")
        lines.append("#include \"BNNModel.h\"")
        lines.append("")
        lines.append("BNNModel model;")
        lines.append("")
        lines.append("void setup() {")
        lines.append("  Serial.begin(115200);")
        lines.append(f"  {data_type} input[{self.input_size}] = {self._generate_example_input()};")
        lines.append(f"  {data_type} output = model.predict(input);")
        if self.task == 'binary':
            lines.append("  Serial.print(\"Predicted logit: \");")
            lines.append("  Serial.println(output, 6);")
            lines.append("  Serial.print(\"Predicted class (0 or 1): \");")
            lines.append("  Serial.println(output > 0 ? 1 : 0);")
        elif self.task == 'multiclass':
            lines.append("  Serial.print(\"Predicted class: \");")
            lines.append("  Serial.println((int)output);")
        else: 
            lines.append("  Serial.print(\"Predicted value: \");")
            lines.append("  Serial.println(output, 6);")
        lines.append("}")
        lines.append("")
        lines.append("void loop() {")
        lines.append("  // Nothing to do here")
        lines.append("}")
        return "\n".join(lines)

    def _generate_example_input(self):
        return "{" + ", ".join(["0.0"] * self.input_size) + "}"


def generate_ino(json_path, output_dir, board='avr', quantize=False,
                 use_flash=True, mode='deterministic', task='regression'):
    generator = ArduinoGenerator(json_path, output_dir, board, quantize, use_flash, mode, task)
    generator.generate()