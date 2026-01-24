import json
import numpy as np
import re
import os
import base64
import zlib
import pickle
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

# ==========================================
# Settings and Data Structures
# ==========================================

@dataclass
class GeneratorConfig:
    """Settings for C++ code generation"""
    target_device: str = "arduino_uno"
    use_fixed_point: bool = False
    precision: int = 6
    
    @property
    def data_type(self) -> str:
        return "int32_t" if self.use_fixed_point else "float"
    
    @property
    def scale_factor(self) -> int:
        return 10 ** self.precision if self.use_fixed_point else 1

@dataclass
class ModelMetadata:
    """Metadata extracted from the model JSON"""
    input_dim: int
    hidden_dim: int
    output_dim: int
    rbf_type: str
    
    # Model Weights
    centers: np.ndarray
    widths: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    
    # Normalization Parameters (Mean and Scale/StdDev)
    has_normalization: bool
    input_mean: Optional[np.ndarray] = None
    input_scale: Optional[np.ndarray] = None # sqrt(variance)
    output_mean: Optional[np.ndarray] = None
    output_scale: Optional[np.ndarray] = None # sqrt(variance)

# ==========================================
# Main Class
# ==========================================

class RBFArduinoCodeGenerator:
    """Optimized C++ code generator for RBF inference on microcontrollers."""

    TARGET_PROFILES = {
        "arduino_uno":  GeneratorConfig("arduino_uno", use_fixed_point=True, precision=4),
        "arduino_nano": GeneratorConfig("arduino_nano", use_fixed_point=True, precision=4),
        "esp32":        GeneratorConfig("esp32", use_fixed_point=False, precision=6),
        "generic":      GeneratorConfig("generic", use_fixed_point=False, precision=6)
    }

    def __init__(self, model_json_path: Union[str, Path]):
        self.json_path = Path(model_json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.json_path}")
        
        self.model_data: Dict[str, Any] = {}
        self.metadata: Optional[ModelMetadata] = None
        self._load_model()

    def _load_model(self):
        """Loads and validates the model JSON, handling nested layers and normalization."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)

        params = self.model_data.get('learned_parameters', {})
        arch = self.model_data.get('architecture', {})
        comps = self.model_data.get('components', {})

        if not params:
            raise ValueError("JSON missing 'learned_parameters'.")

        # 1. Locate RBF and Dense Layers dynamically
        rbf_layer = {}
        dense_layer = {}
        
        # Search for layers containing specific keys
        for layer_name, layer_data in params.items():
            if 'centers' in layer_data and 'widths' in layer_data:
                rbf_layer = layer_data
            elif 'weights' in layer_data:
                dense_layer = layer_data

        if not rbf_layer:
            raise ValueError("Could not locate RBF layer (looking for 'centers' and 'widths').")
        
        # 2. Extract Weights
        centers = self._extract_array(rbf_layer, 'centers')
        widths = self._extract_array(rbf_layer, 'widths')
        weights = self._extract_array(dense_layer, 'weights')
        bias = self._extract_array(dense_layer, 'bias')

        # 3. Extract Dimensions from shapes
        input_dim = centers.shape[1] if len(centers.shape) > 1 else 1
        hidden_dim = centers.shape[0]
        output_dim = weights.shape[1] if len(weights.shape) > 1 else 1

        # Handle bias default if missing
        if bias is None:
            bias = np.zeros(output_dim)

        # 4. Extract Normalization Components
        input_mean = self._extract_array(comps.get('normalizers', {}), 'input_scaler') # Mean
        input_var = self._extract_array(comps.get('normalizers', {}), 'input_var')     # Variance
        output_mean = self._extract_array(comps.get('normalizers', {}), 'output_scaler')
        output_var = self._extract_array(comps.get('normalizers', {}), 'output_var')

        has_norm = (input_mean is not None and input_var is not None)
        
        # Calculate Scale (Standard Deviation) from Variance for C++ usage
        input_scale = np.sqrt(input_var) if input_var is not None else None
        output_scale = np.sqrt(output_var) if output_var is not None else None

        self.metadata = ModelMetadata(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rbf_type='gaussian', # Defaulting to gaussian as per JSON
            centers=centers,
            widths=widths,
            weights=weights,
            bias=bias,
            has_normalization=has_norm,
            input_mean=input_mean,
            input_scale=input_scale,
            output_mean=output_mean,
            output_scale=output_scale
        )

    def _extract_array(self, source_dict: Dict, key: str) -> Optional[np.ndarray]:
        """Extracts numpy array from JSON object (handles list or dict/data format)."""
        obj = source_dict.get(key)
        if obj is None:
            return None
        
        # Case A: Dictionary with 'data' key (Standard in this JSON)
        if isinstance(obj, dict) and 'data' in obj:
            return np.array(obj['data'], dtype=np.float64)
            
        # Case B: Direct list
        if isinstance(obj, list):
            return np.array(obj, dtype=np.float64)
            
        return None

    def _generate_class_name(self) -> str:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', self.json_path.stem)
        parts = [p.capitalize() for p in clean_name.split('_') if p]
        return "".join(parts) + "RBF"

    def generate(self, output_dir: Union[str, Path], target_device: str = "arduino_uno"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        config = self.TARGET_PROFILES.get(target_device, self.TARGET_PROFILES["generic"])
        class_name = self._generate_class_name()

        files = {
            f"{class_name}.h": self._create_header(class_name, config),
            f"{class_name}.cpp": self._create_source(class_name, config),
            f"example_{class_name}.ino": self._create_example(class_name, config),
        }

        for filename, content in files.items():
            with open(out_path / filename, 'w', encoding='utf-8') as f:
                f.write(content)

        print(f"✅ Generated {class_name} for {target_device} in {out_path}")
        return files

    # ==========================================
    # C++ Templates
    # ==========================================

    def _create_header(self, class_name: str, config: GeneratorConfig) -> str:
        md = self.metadata
        return f"""#ifndef {class_name.upper()}_H
#define {class_name.upper()}_H

#include <Arduino.h>
#include <math.h>

// Dimensions
#define RBF_INPUT_DIM {md.input_dim}
#define RBF_HIDDEN_DIM {md.hidden_dim}
#define RBF_OUTPUT_DIM {md.output_dim}
#define RBF_SCALE_FACTOR {config.scale_factor}
#define RBF_FIXED_POINT {1 if config.use_fixed_point else 0}
#define RBF_USE_NORMALIZATION {1 if md.has_normalization else 0}

typedef {config.data_type} rbf_float;

class {class_name} {{
public:
    {class_name}();
    void predict(const rbf_float* input, rbf_float* output);
    void printInfo();

private:
    rbf_float computeKernel(rbf_float dist_sq, rbf_float width);
    rbf_float euclideanDistanceSq(const rbf_float* input, int center_idx);

    // Buffers
    rbf_float hidden_activations[RBF_HIDDEN_DIM];
    
    // Internal Normalization Helpers
    void normalizeInput(const rbf_float* input, rbf_float* norm_input);
    void denormalizeOutput(rbf_float* output);
}};

#endif
"""

    def _create_source(self, class_name: str, config: GeneratorConfig) -> str:
        md = self.metadata
        
        # Format Arrays
        centers = self._fmt(md.centers, config)
        widths = self._fmt(md.widths, config)
        weights = self._fmt(md.weights, config)
        bias = self._fmt(md.bias, config)
        
        # Normalization Arrays (Empty if unused)
        in_mean = self._fmt(md.input_mean, config) if md.has_normalization else "{}"
        in_scale = self._fmt(md.input_scale, config) if md.has_normalization else "{}"
        out_mean = self._fmt(md.output_mean, config) if md.has_normalization else "{}"
        out_scale = self._fmt(md.output_scale, config) if md.has_normalization else "{}"

        return f"""#include "{class_name}.h"

// ==========================================
// Flash/PROGMEM Macros
// ==========================================
#if defined(ESP32) || defined(ESP8266)
    #define RBF_CONST const
    #define RBF_READ(x) (*(x)) // Fixed: Dereference pointer
#elif defined(__AVR__)
    #include <avr/pgmspace.h>
    #define RBF_CONST const PROGMEM
    #if RBF_FIXED_POINT
        #define RBF_READ(addr) (rbf_float)pgm_read_dword(addr)
    #else
        #define RBF_READ(addr) (rbf_float)pgm_read_float(addr)
    #endif
#else
    #define RBF_CONST const
    #define RBF_READ(x) (*(x)) // Fixed: Dereference pointer
#endif

// ==========================================
// Model Parameters
// ==========================================
static RBF_CONST rbf_float RBF_CENTERS[] = {centers};
static RBF_CONST rbf_float RBF_WIDTHS[] = {widths};
static RBF_CONST rbf_float RBF_WEIGHTS[] = {weights};
static RBF_CONST rbf_float RBF_BIAS[] = {bias};

#if RBF_USE_NORMALIZATION
static RBF_CONST rbf_float NORM_IN_MEAN[] = {in_mean};
static RBF_CONST rbf_float NORM_IN_SCALE[] = {in_scale}; // StdDev
static RBF_CONST rbf_float NORM_OUT_MEAN[] = {out_mean};
static RBF_CONST rbf_float NORM_OUT_SCALE[] = {out_scale}; // StdDev
#endif

// ==========================================
// Implementation
// ==========================================

{class_name}::{class_name}() {{
    for(int i=0; i<RBF_HIDDEN_DIM; i++) hidden_activations[i] = 0;
}}

void {class_name}::predict(const rbf_float* input, rbf_float* output) {{
    // 1. Prepare Input (Normalize if needed)
    rbf_float processed_input[RBF_INPUT_DIM];
    #if RBF_USE_NORMALIZATION
        normalizeInput(input, processed_input);
        const rbf_float* in_ptr = processed_input;
    #else
        const rbf_float* in_ptr = input;
    #endif

    // 2. Hidden Layer (RBF)
    for (int i = 0; i < RBF_HIDDEN_DIM; i++) {{
        rbf_float dist_sq = euclideanDistanceSq(in_ptr, i);
        rbf_float width = RBF_READ(&RBF_WIDTHS[i]);
        hidden_activations[i] = computeKernel(dist_sq, width);
    }}

    // 3. Output Layer (Linear)
    for (int j = 0; j < RBF_OUTPUT_DIM; j++) {{
        #if RBF_FIXED_POINT
            int64_t sum = 0;
        #else
            float sum = 0.0f;
        #endif

        for (int i = 0; i < RBF_HIDDEN_DIM; i++) {{
            rbf_float weight = RBF_READ(&RBF_WEIGHTS[i * RBF_OUTPUT_DIM + j]);
            #if RBF_FIXED_POINT
                sum += (int64_t)hidden_activations[i] * weight;
            #else
                sum += hidden_activations[i] * weight;
            #endif
        }}

        #if RBF_FIXED_POINT
            sum = sum / RBF_SCALE_FACTOR;
            output[j] = (rbf_float)sum + RBF_READ(&RBF_BIAS[j]);
        #else
            output[j] = sum + RBF_READ(&RBF_BIAS[j]);
        #endif
    }}
    
    // 4. Denormalize Output
    #if RBF_USE_NORMALIZATION
        denormalizeOutput(output);
    #endif
}}

void {class_name}::normalizeInput(const rbf_float* input, rbf_float* norm_input) {{
    for(int i=0; i<RBF_INPUT_DIM; i++) {{
        rbf_float mean = RBF_READ(&NORM_IN_MEAN[i]);
        rbf_float scale = RBF_READ(&NORM_IN_SCALE[i]);
        #if RBF_FIXED_POINT
             // (Input - Mean) * SCALE_FACTOR / StdDev
             int64_t diff = (int64_t)input[i] - mean;
             norm_input[i] = (rbf_float)((diff * RBF_SCALE_FACTOR) / scale);
        #else
             norm_input[i] = (input[i] - mean) / scale;
        #endif
    }}
}}

void {class_name}::denormalizeOutput(rbf_float* output) {{
    for(int i=0; i<RBF_OUTPUT_DIM; i++) {{
        rbf_float mean = RBF_READ(&NORM_OUT_MEAN[i]);
        rbf_float scale = RBF_READ(&NORM_OUT_SCALE[i]);
        #if RBF_FIXED_POINT
             // (Output * StdDev) / SCALE_FACTOR + Mean
             int64_t val = (int64_t)output[i] * scale;
             output[i] = (rbf_float)(val / RBF_SCALE_FACTOR) + mean;
        #else
             output[i] = (output[i] * scale) + mean;
        #endif
    }}
}}

rbf_float {class_name}::euclideanDistanceSq(const rbf_float* input, int center_idx) {{
    #if RBF_FIXED_POINT
        int64_t sum = 0;
    #else
        float sum = 0.0f;
    #endif
    int base_idx = center_idx * RBF_INPUT_DIM;
    for (int k = 0; k < RBF_INPUT_DIM; k++) {{
        rbf_float c = RBF_READ(&RBF_CENTERS[base_idx + k]);
        rbf_float diff = input[k] - c;
        #if RBF_FIXED_POINT
            sum += ((int64_t)diff * diff) / RBF_SCALE_FACTOR;
        #else
            sum += diff * diff;
        #endif
    }}
    return (rbf_float)sum;
}}

rbf_float {class_name}::computeKernel(rbf_float dist_sq, rbf_float width) {{
    #if RBF_FIXED_POINT
        if (width == 0) return 0;
        double d = (double)dist_sq / RBF_SCALE_FACTOR;
        double w = (double)width / RBF_SCALE_FACTOR;
        return (rbf_float)(exp(-d / (2.0 * w * w)) * RBF_SCALE_FACTOR);
    #else
        return exp(-dist_sq / (2.0f * width * width));
    #endif
}}

void {class_name}::printInfo() {{
    Serial.println(F("Model Loaded: {class_name}"));
    Serial.print(F("Normalization: ")); 
    Serial.println(RBF_USE_NORMALIZATION ? F("YES") : F("NO"));
}}
"""

    def _create_example(self, class_name: str, config: GeneratorConfig) -> str:
        return f"""#include "{class_name}.h"

{class_name} model;
rbf_float inputs[RBF_INPUT_DIM];
rbf_float outputs[RBF_OUTPUT_DIM];

void setup() {{
    Serial.begin(115200);
    while(!Serial);
    model.printInfo();
    Serial.println(F("Send 't' to test."));
}}

void loop() {{
    if(Serial.read() == 't') {{
        // Generate random input
        for(int i=0; i<RBF_INPUT_DIM; i++) {{
            #if RBF_FIXED_POINT
                inputs[i] = random(-100, 100) * (RBF_SCALE_FACTOR / 100); 
            #else
                inputs[i] = random(-200, 200) / 100.0f; // Range -2.0 to 2.0
            #endif
            Serial.print(F("In: ")); Serial.println(inputs[i]);
        }}

        unsigned long t0 = micros();
        model.predict(inputs, outputs);
        unsigned long dt = micros() - t0;

        for(int i=0; i<RBF_OUTPUT_DIM; i++) {{
            Serial.print(F("Out: ")); Serial.println(outputs[i]);
        }}
        Serial.print(F("Time (us): ")); Serial.println(dt);
    }}
}}
"""

    def _fmt(self, array: np.ndarray, config: GeneratorConfig) -> str:
        if array is None: return "{}"
        data = array.flatten()
        if config.use_fixed_point:
            scaled = (data * config.scale_factor).astype(np.int32)
            vals = [str(x) for x in scaled]
        else:
            vals = [f"{x:.6f}" for x in data]
        return "{" + ", ".join(vals) + "}"

# Usage
if __name__ == "__main__":
    # Create a dummy file if testing locally, or assume path exists
    try:
        gen = RBFArduinoCodeGenerator("rbf_regression_model.json")
        gen.generate("output_arduino", "arduino_uno") # or esp32
    except Exception as e:
        print(f"Error: {e}")