"""
Arduino C++ code generator for Adaptive Neuro-Fuzzy Inference System (ANFIS).

The generated C++ implements the five-layer ANFIS forward pass:
    1. Fuzzification   : compute mu_jk(x_j) for every input j and term k
    2. Rule firing     : w_i = PROD_j mu_j,term_i(j)   (product T-norm)
    3. Normalisation   : wbar_i = w_i / SUM_j w_j
    4. Consequent      : f_i = p_i0 + SUM_j p_ij * x_j
    5. Defuzzification : yhat = SUM_i wbar_i * f_i

For TinyML, n_inputs <= 4 and n_terms <= 5 are recommended.
"""

import json
import os
import numpy as np


def _np_predict(data, x):
    """Pure-NumPy forward pass for ANFIS models."""
    p   = data["parameters"]
    cfg = data["anfis_config"]
    den = data.get("dense_layers", [])

    n_in  = cfg["n_inputs"]
    n_te  = cfg["n_terms"]
    n_r   = cfg["n_rules"]
    mf    = cfg["mf_type"]

    x = x.astype(np.float32)

    # Layer 1: Fuzzification
    mu = np.zeros((n_in, n_te), dtype=np.float32)
    if mf == "gaussian":
        center = np.array(p["fuzzify_center"], dtype=np.float32)
        sigma  = np.array(p["fuzzify_sigma"],  dtype=np.float32)
        for j in range(n_in):
            for k in range(n_te):
                mu[j, k] = np.exp(-0.5 * ((x[j] - center[j, k]) / max(sigma[j, k], 1e-6)) ** 2)
    elif mf == "sigmoid":
        a = np.array(p["fuzzify_a"], dtype=np.float32)
        c = np.array(p["fuzzify_c"], dtype=np.float32)
        for j in range(n_in):
            for k in range(n_te):
                mu[j, k] = 1.0 / (1.0 + np.exp(-a[j, k] * (x[j] - c[j, k])))
    elif mf == "bell":
        a = np.array(p["fuzzify_a"], dtype=np.float32)
        b = np.array(p["fuzzify_b"], dtype=np.float32)
        c = np.array(p["fuzzify_c"], dtype=np.float32)
        for j in range(n_in):
            for k in range(n_te):
                mu[j, k] = 1.0 / (1.0 + np.abs((x[j] - c[j, k]) / max(a[j, k], 1e-6)) ** (2 * b[j, k]))
    else:
        raise ValueError("MF type '{}' not yet supported in C++ generator.".format(mf))

    # Layer 2: Rule firing (product T-norm)
    w = np.zeros(n_r, dtype=np.float32)
    for r in range(n_r):
        tmp = r
        prod = 1.0
        for j in range(n_in):
            t = tmp % n_te
            prod *= mu[j, t]
            tmp //= n_te
        w[r] = prod

    # Layer 3: Normalisation
    sum_w = w.sum()
    w_bar = w / max(sum_w, 1e-8)

    # Layer 4: Consequent
    coeff = np.array(p["consequent_coeff"], dtype=np.float32)
    f = np.zeros(n_r, dtype=np.float32)
    for r in range(n_r):
        f[r] = coeff[r, 0]
        for j in range(n_in):
            f[r] += coeff[r, j + 1] * x[j]

    # Layer 5: Defuzzification
    y = np.dot(w_bar, f)

    # Optional dense head
    out = np.array([y], dtype=np.float32)
    for layer in den:
        j   = layer["index"]
        W   = np.array(p["dense_{}_weight".format(j)], dtype=np.float32)
        b   = np.array(p["dense_{}_bias".format(j)],   dtype=np.float32)
        out = W @ out + b
        act = layer.get("activation", "linear")
        if act == "relu":
            out = np.maximum(0.0, out)
        elif act == "sigmoid":
            out = 1.0 / (1.0 + np.exp(-out))
        elif act == "tanh":
            out = np.tanh(out)

    return out


class ArduinoANFISGenerator:
    def __init__(self, json_path, output_dir, board="avr",
                 use_flash=True, task="regression"):
        self.json_path  = json_path
        self.output_dir = output_dir
        self.board      = board
        self.use_flash  = use_flash and (board == "avr")
        self.task       = task
        self.model_data = None

    def _load(self):
        with open(self.json_path, "r") as f:
            self.model_data = json.load(f)

    @property
    def _cfg(self): return self.model_data["anfis_config"]
    @property
    def _den(self): return self.model_data.get("dense_layers", [])
    @property
    def _p(self):   return self.model_data["parameters"]

    @property
    def n_inputs(self):  return self._cfg["n_inputs"]
    @property
    def n_terms(self):   return self._cfg["n_terms"]
    @property
    def n_rules(self):   return self._cfg["n_rules"]
    @property
    def mf_type(self):   return self._cfg["mf_type"]

    @property
    def output_size(self):
        if not self._den:
            return 1
        return self._den[-1]["out_features"]

    def generate(self, sample_input=None):
        self._load()
        os.makedirs(self.output_dir, exist_ok=True)

        if sample_input is None:
            np.random.seed(0)
            sample_input = np.random.randn(self.n_inputs).astype(np.float32)

        expected = _np_predict(self.model_data, sample_input)

        header_path = os.path.join(self.output_dir, "ANFISModel.h")
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(self._header())

        sketch_path = os.path.join(self.output_dir, "sketch.ino")
        with open(sketch_path, "w", encoding="utf-8") as f:
            f.write(self._sketch(sample_input, expected))

        print("Generated in : {}  (board: {})".format(self.output_dir, self.board))
        print("Verification  n_inputs={}, n_terms={}, n_rules={}".format(
            self.n_inputs, self.n_terms, self.n_rules))
        print("Expected output (Python) : {}".format(expected.tolist()))

    def _rd(self, arr_name):
        if self.use_flash:
            return "pgm_read_float(&{})".format(arr_name)
        return arr_name

    def _fmt(self, values):
        if isinstance(values[0], list):
            rows = []
            for row in values:
                row_str = ", ".join("{:.8f}f".format(v) for v in row)
                rows.append("    {" + row_str + "}")
            return "{\n" + ",\n".join(rows) + "\n}"
        return "{" + ", ".join("{:.8f}f".format(v) for v in values) + "}"

    def _progmem(self, typ, name, values):
        # CORREÇÃO: adicionado [] para declarar arrays unidimensionais
        suffix = " PROGMEM" if self.use_flash else ""
        return "static const {} {}[]{} = {};".format(typ, name, suffix, self._fmt(values))

    def _progmem2d(self, typ, name, rows, cols, values):
        suffix = " PROGMEM" if self.use_flash else ""
        return "static const {} {}[{}][{}]{} = {};".format(
            typ, name, rows, cols, suffix, self._fmt(values))

    def _header(self):
        p   = self._p
        cfg = self._cfg
        n_in  = cfg["n_inputs"]
        n_te  = cfg["n_terms"]
        n_r   = cfg["n_rules"]
        mf    = cfg["mf_type"]

        lines = [
            "// ANFISModel.h  —  Auto-generated Adaptive Neuro-Fuzzy Inference System",
            "// Do NOT edit weights manually.",
            "",
            "#pragma once",
            "#include <math.h>",
            "",
        ]

        if self.use_flash:
            lines += ["#include <avr/pgmspace.h>", ""]

        lines.append("// ANFIS config: n_inputs={}, n_terms={}, n_rules={}, mf={}".format(
            n_in, n_te, n_r, mf))
        lines.append("")

        if mf == "gaussian":
            lines.append(self._progmem2d("float", "fuzzify_center", n_in, n_te, p["fuzzify_center"]))
            lines.append(self._progmem2d("float", "fuzzify_sigma",  n_in, n_te, p["fuzzify_sigma"]))
        elif mf == "sigmoid":
            lines.append(self._progmem2d("float", "fuzzify_a", n_in, n_te, p["fuzzify_a"]))
            lines.append(self._progmem2d("float", "fuzzify_c", n_in, n_te, p["fuzzify_c"]))
        elif mf == "bell":
            lines.append(self._progmem2d("float", "fuzzify_a", n_in, n_te, p["fuzzify_a"]))
            lines.append(self._progmem2d("float", "fuzzify_b", n_in, n_te, p["fuzzify_b"]))
            lines.append(self._progmem2d("float", "fuzzify_c", n_in, n_te, p["fuzzify_c"]))

        lines.append("")
        lines.append(self._progmem2d("float", "consequent_coeff", n_r, n_in + 1, p["consequent_coeff"]))
        lines.append("")

        for layer in self._den:
            j   = layer["index"]
            out = layer["out_features"]
            inp = layer["in_features"]
            lines += ["// Dense layer {}  in={}  out={}".format(j, inp, out)]
            lines.append(self._progmem2d("float", "den{}_w".format(j), out, inp, p["dense_{}_weight".format(j)]))
            if p.get("dense_{}_bias".format(j)) is not None:
                lines.append(self._progmem("float", "den{}_b".format(j), p["dense_{}_bias".format(j)]))
            lines.append("")

        lines += [
            "class ANFISModel {",
            "public:",
            "    float predict(float* input);",
            "",
            "private:",
            "    float mu[{}][{}];".format(n_in, n_te),
            "    float w[{}];".format(n_r),
            "    float w_bar[{}];".format(n_r),
            "    float f_val[{}];".format(n_r),
            "    float dense_out[16];",
            "};",
            "",
        ]

        lines += [
            "float ANFISModel::predict(float* input) {",
            "    const int N_IN  = {};".format(n_in),
            "    const int N_TE  = {};".format(n_te),
            "    const int N_R   = {};".format(n_r),
            "",
            "    // ---- Layer 1: Fuzzification ----",
        ]

        rd = self._rd

        if mf == "gaussian":
            lines += [
                "    for (int j = 0; j < N_IN; j++) {",
                "        for (int k = 0; k < N_TE; k++) {",
                "            float c = " + rd("fuzzify_center[j][k]") + ";",
                "            float s = " + rd("fuzzify_sigma[j][k]") + ";",
                "            float diff = input[j] - c;",
                "            mu[j][k] = expf(-0.5f * (diff * diff) / (s * s + 1e-6f));",
                "        }",
                "    }",
            ]
        elif mf == "sigmoid":
            lines += [
                "    for (int j = 0; j < N_IN; j++) {",
                "        for (int k = 0; k < N_TE; k++) {",
                "            float a = " + rd("fuzzify_a[j][k]") + ";",
                "            float c = " + rd("fuzzify_c[j][k]") + ";",
                "            mu[j][k] = 1.0f / (1.0f + expf(-a * (input[j] - c)));",
                "        }",
                "    }",
            ]
        elif mf == "bell":
            lines += [
                "    for (int j = 0; j < N_IN; j++) {",
                "        for (int k = 0; k < N_TE; k++) {",
                "            float a = " + rd("fuzzify_a[j][k]") + ";",
                "            float b = " + rd("fuzzify_b[j][k]") + ";",
                "            float c = " + rd("fuzzify_c[j][k]") + ";",
                "            float num = fabsf((input[j] - c) / (a + 1e-6f));",
                "            mu[j][k] = 1.0f / (1.0f + powf(num, 2.0f * b));",
                "        }",
                "    }",
            ]

        lines += [
            "",
            "    // ---- Layer 2: Rule firing (product T-norm) ----",
            "    for (int r = 0; r < N_R; r++) {",
            "        float prod = 1.0f;",
            "        int tmp = r;",
            "        for (int j = 0; j < N_IN; j++) {",
            "            int t = tmp % N_TE;",
            "            prod *= mu[j][t];",
            "            tmp /= N_TE;",
            "        }",
            "        w[r] = prod;",
            "    }",
            "",
            "    // ---- Layer 3: Normalisation ----",
            "    float sum_w = 0.0f;",
            "    for (int r = 0; r < N_R; r++) sum_w += w[r];",
            "    sum_w = sum_w < 1e-8f ? 1e-8f : sum_w;",
            "    for (int r = 0; r < N_R; r++) w_bar[r] = w[r] / sum_w;",
            "",
            "    // ---- Layer 4: Consequent ----",
            "    for (int r = 0; r < N_R; r++) {",
            "        float acc = " + rd("consequent_coeff[r][0]") + ";",
            "        for (int j = 0; j < N_IN; j++)",
            "            acc += " + rd("consequent_coeff[r][j+1]") + " * input[j];",
            "        f_val[r] = acc;",
            "    }",
            "",
            "    // ---- Layer 5: Defuzzification ----",
            "    float y = 0.0f;",
            "    for (int r = 0; r < N_R; r++)",
            "        y += w_bar[r] * f_val[r];",
            "",
        ]

        if self._den:
            lines.append("    dense_out[0] = y;")
            prev_src = "dense_out"
            n_dense = len(self._den)

            for li, layer in enumerate(self._den):
                j   = layer["index"]
                out = layer["out_features"]
                inp = layer["in_features"]
                act = layer.get("activation", "linear")
                lines += [
                    "    // Dense {}".format(j),
                    "    float d_out{}[{}];".format(j, out),
                    "    for (int k = 0; k < {}; k++) {{".format(out),
                    "        float acc = {};".format(rd("den{}_b[k]".format(j))),
                    "        for (int m = 0; m < {}; m++)".format(inp),
                    "            acc += {} * {}[m];".format(rd("den{}_w[k][m]".format(j)), prev_src),
                ]
                lines += ["        " + l for l in self._act_c(act, "acc")]
                lines += [
                    "        d_out{}[k] = acc;".format(j),
                    "    }",
                ]
                if li < n_dense - 1:
                    next_inp = self._den[li + 1]["in_features"]
                    lines.append("    for (int k = 0; k < {}; k++) dense_out[k] = d_out{}[k];".format(next_inp, j))
                    prev_src = "dense_out"
                else:
                    prev_src = "d_out{}".format(j)
                lines.append("")

            # CORREÇÃO: retorno correto do último vetor denso
            lines.append("    return {}[0];".format(prev_src))
        else:
            lines.append("    return y;")

        lines.append("}")
        return "\n".join(lines)

    def _act_c(self, act, var):
        a = act.lower()
        if a == "relu":
            return ["if (" + var + " < 0.0f) " + var + " = 0.0f;"]
        elif a == "sigmoid":
            return [var + " = 1.0f / (1.0f + expf(-" + var + "));"]
        elif a == "tanh":
            return [var + " = tanhf(" + var + ");"]
        elif a == "leaky_relu":
            return ["if (" + var + " < 0.0f) " + var + " *= 0.01f;"]
        elif a == "gelu":
            return [var + " = 0.5f * " + var + " * (1.0f + tanhf(0.79788456f * (" + var + " + 0.044715f * " + var + " * " + var + " * " + var + ")));"]
        elif a == "swish":
            return [var + " = " + var + " / (1.0f + expf(-" + var + "));"]
        return []

    def _sketch(self, sample_input, expected):
        flat_str = ", ".join("{:.8f}f".format(float(v)) for v in sample_input)
        exp_list = expected.tolist() if hasattr(expected, "tolist") else [float(expected)]

        input_vals = ", ".join("{:.8f}".format(float(v)) for v in sample_input)

        lines = [
            "/*",
            " * ANFIS Model -- Arduino verification sketch",
            " * Generated automatically -- do not edit the weights.",
            " *",
            " * Input  size     : " + str(self.n_inputs),
            " * Terms per input : " + str(self.n_terms),
            " * Total rules     : " + str(self.n_rules),
            " * MF type         : " + str(self.mf_type),
            " *",
            " * Input values    : [" + input_vals + "]",
            " * Expected output : " + "{:.8f}".format(exp_list[0]),
            " *",
            " * Upload this sketch, open Serial Monitor at 115200 baud,",
            " * and confirm the printed value matches the expected value",
            " * above to at least 4 decimal places.",
            " */",
            "",
            '#include "ANFISModel.h"',
            "",
            "ANFISModel model;",
            "",
            "void setup() {",
            "    Serial.begin(115200);",
            "    while (!Serial);",
            "",
            "    const int N_INPUTS = " + str(self.n_inputs) + ";",
            "",
            "    float input[N_INPUTS] = {",
            "        " + flat_str,
            "    };",
            "",
            "    float output = model.predict(input);",
            "",
            "    // Expected value  : " + "{:.8f}".format(exp_list[0]),
            '    Serial.print("Predicted value  : "); Serial.println(output, 8);',
            "}",
            "",
            "void loop() {",
            "    // Nothing to do here",
            "}",
        ]
        return "\n".join(lines)


def generate_ino(json_path, output_dir, board="avr",
                 use_flash=True, task="regression"):
    gen = ArduinoANFISGenerator(json_path, output_dir, board, use_flash, task)
    gen.generate()