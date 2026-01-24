#include "RbfRegressionModelRBF.h"

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
static RBF_CONST rbf_float RBF_CENTERS[] = {-0.438009, 1.009264, -1.342599, 0.260462, 1.452298, -0.867641, -0.126775, 0.499166, -1.144709, 1.677563, -0.615472, 1.158399, -1.506207, 0.832844, 0.137758};
static RBF_CONST rbf_float RBF_WIDTHS[] = {0.177462, 0.149135, 0.163608, 0.122704, 0.225265, 0.252169, 0.264533, 0.238704, 0.197890, 0.225265, 0.177462, 0.149135, 0.163608, 0.176420, 0.122704};
static RBF_CONST rbf_float RBF_WEIGHTS[] = {1.428320, 0.690991, 0.858143, 0.320764, 1.176468, 2.506329, 3.048524, 2.754395, -0.035568, 2.822957, 0.499863, 1.081307, 0.005268, 0.803481, 0.424257};
static RBF_CONST rbf_float RBF_BIAS[] = {-2.728589};

#if RBF_USE_NORMALIZATION
static RBF_CONST rbf_float NORM_IN_MEAN[] = {2.354645};
static RBF_CONST rbf_float NORM_IN_SCALE[] = {1.470028}; // StdDev
static RBF_CONST rbf_float NORM_OUT_MEAN[] = {0.966347};
static RBF_CONST rbf_float NORM_OUT_SCALE[] = {0.374724}; // StdDev
#endif

// ==========================================
// Implementation
// ==========================================

RbfRegressionModelRBF::RbfRegressionModelRBF() {
    for(int i=0; i<RBF_HIDDEN_DIM; i++) hidden_activations[i] = 0;
}

void RbfRegressionModelRBF::predict(const rbf_float* input, rbf_float* output) {
    // 1. Prepare Input (Normalize if needed)
    rbf_float processed_input[RBF_INPUT_DIM];
    #if RBF_USE_NORMALIZATION
        normalizeInput(input, processed_input);
        const rbf_float* in_ptr = processed_input;
    #else
        const rbf_float* in_ptr = input;
    #endif

    // 2. Hidden Layer (RBF)
    for (int i = 0; i < RBF_HIDDEN_DIM; i++) {
        rbf_float dist_sq = euclideanDistanceSq(in_ptr, i);
        rbf_float width = RBF_READ(&RBF_WIDTHS[i]);
        hidden_activations[i] = computeKernel(dist_sq, width);
    }

    // 3. Output Layer (Linear)
    for (int j = 0; j < RBF_OUTPUT_DIM; j++) {
        #if RBF_FIXED_POINT
            int64_t sum = 0;
        #else
            float sum = 0.0f;
        #endif

        for (int i = 0; i < RBF_HIDDEN_DIM; i++) {
            rbf_float weight = RBF_READ(&RBF_WEIGHTS[i * RBF_OUTPUT_DIM + j]);
            #if RBF_FIXED_POINT
                sum += (int64_t)hidden_activations[i] * weight;
            #else
                sum += hidden_activations[i] * weight;
            #endif
        }

        #if RBF_FIXED_POINT
            sum = sum / RBF_SCALE_FACTOR;
            output[j] = (rbf_float)sum + RBF_READ(&RBF_BIAS[j]);
        #else
            output[j] = sum + RBF_READ(&RBF_BIAS[j]);
        #endif
    }
    
    // 4. Denormalize Output
    #if RBF_USE_NORMALIZATION
        denormalizeOutput(output);
    #endif
}

void RbfRegressionModelRBF::normalizeInput(const rbf_float* input, rbf_float* norm_input) {
    for(int i=0; i<RBF_INPUT_DIM; i++) {
        rbf_float mean = RBF_READ(&NORM_IN_MEAN[i]);
        rbf_float scale = RBF_READ(&NORM_IN_SCALE[i]);
        #if RBF_FIXED_POINT
             // (Input - Mean) * SCALE_FACTOR / StdDev
             int64_t diff = (int64_t)input[i] - mean;
             norm_input[i] = (rbf_float)((diff * RBF_SCALE_FACTOR) / scale);
        #else
             norm_input[i] = (input[i] - mean) / scale;
        #endif
    }
}

void RbfRegressionModelRBF::denormalizeOutput(rbf_float* output) {
    for(int i=0; i<RBF_OUTPUT_DIM; i++) {
        rbf_float mean = RBF_READ(&NORM_OUT_MEAN[i]);
        rbf_float scale = RBF_READ(&NORM_OUT_SCALE[i]);
        #if RBF_FIXED_POINT
             // (Output * StdDev) / SCALE_FACTOR + Mean
             int64_t val = (int64_t)output[i] * scale;
             output[i] = (rbf_float)(val / RBF_SCALE_FACTOR) + mean;
        #else
             output[i] = (output[i] * scale) + mean;
        #endif
    }
}

rbf_float RbfRegressionModelRBF::euclideanDistanceSq(const rbf_float* input, int center_idx) {
    #if RBF_FIXED_POINT
        int64_t sum = 0;
    #else
        float sum = 0.0f;
    #endif
    int base_idx = center_idx * RBF_INPUT_DIM;
    for (int k = 0; k < RBF_INPUT_DIM; k++) {
        rbf_float c = RBF_READ(&RBF_CENTERS[base_idx + k]);
        rbf_float diff = input[k] - c;
        #if RBF_FIXED_POINT
            sum += ((int64_t)diff * diff) / RBF_SCALE_FACTOR;
        #else
            sum += diff * diff;
        #endif
    }
    return (rbf_float)sum;
}

rbf_float RbfRegressionModelRBF::computeKernel(rbf_float dist_sq, rbf_float width) {
    #if RBF_FIXED_POINT
        if (width == 0) return 0;
        double d = (double)dist_sq / RBF_SCALE_FACTOR;
        double w = (double)width / RBF_SCALE_FACTOR;
        return (rbf_float)(exp(-d / (2.0 * w * w)) * RBF_SCALE_FACTOR);
    #else
        return exp(-dist_sq / (2.0f * width * width));
    #endif
}

void RbfRegressionModelRBF::printInfo() {
    Serial.println(F("Model Loaded: RbfRegressionModelRBF"));
    Serial.print(F("Normalization: ")); 
    Serial.println(RBF_USE_NORMALIZATION ? F("YES") : F("NO"));
}
