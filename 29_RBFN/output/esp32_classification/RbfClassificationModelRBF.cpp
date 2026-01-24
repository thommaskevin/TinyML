#include "RbfClassificationModelRBF.h"

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
static RBF_CONST rbf_float RBF_CENTERS[] = {-0.735341, 1.188906, 0.991174, -0.812566, -0.430437, -0.191808, 0.583616, 0.105353, -1.719727, 0.167918, 0.425211, -1.212194, -0.034037, 1.162686, 1.641548, 0.501328, 1.236369, -2.105555, 2.096552, -0.256061, -0.547546, 1.579470, -0.708058, 0.518347, -0.208452, -1.115473, 0.825820, -1.394475, -1.506298, 0.921418, 1.436294, -1.086269, -1.376527, 0.511331, 0.318398, 0.555809, 0.505364, -0.347419, 1.785228, -0.856472, 1.848283, 0.201123, -0.022145, -1.540883, -1.567088, -0.229855, 0.320195, 1.340152, -0.036177, -0.740022, -1.093381, 1.105928, 1.266782, -1.628399, -0.274214, 1.737522, 1.463086, -0.310668, 0.447671, -1.755994};
static RBF_CONST rbf_float RBF_WIDTHS[] = {0.183765, 0.261269, 0.337631, 0.229742, 0.213027, 0.220065, 0.198100, 0.182252, 0.239062, 0.260122, 0.157869, 0.334253, 0.206545, 0.220065, 0.215065, 0.208903, 0.215065, 0.261367, 0.229742, 0.208903, 0.182252, 0.232209, 0.213027, 0.198100, 0.206545, 0.183765, 0.239062, 0.157869, 0.316890, 0.258360};
static RBF_CONST rbf_float RBF_WEIGHTS[] = {0.629576, -0.629576, -1.205516, 1.205516, -1.769818, 1.769818, 0.399524, -0.399524, 0.453808, -0.453808, -1.150586, 1.150586, 0.576052, -0.576052, -1.853481, 1.853481, -1.219684, 1.219684, -1.362474, 1.362474, 0.672928, -0.672928, -0.514378, 0.514378, -1.342557, 1.342557, -1.014934, 1.014934, 0.566971, -0.566971, -1.133588, 1.133588, 0.694611, -0.694611, 0.561046, -0.561046, 1.038002, -1.038002, -1.361392, 1.361392, -1.074637, 1.074637, -1.205167, 1.205167, 0.618635, -0.618635, 0.628509, -0.628509, -1.159493, 1.159493, 0.531215, -0.531215, -1.320249, 1.320249, 0.608010, -0.608010, -1.457977, 1.457977, -1.188332, 1.188332};
static RBF_CONST rbf_float RBF_BIAS[] = {0.380431, -0.380431};

#if RBF_USE_NORMALIZATION
static RBF_CONST rbf_float NORM_IN_MEAN[] = {0.425883, 0.283259};
static RBF_CONST rbf_float NORM_IN_SCALE[] = {0.891352, 0.498127}; // StdDev
static RBF_CONST rbf_float NORM_OUT_MEAN[] = {0.533333, 0.466667};
static RBF_CONST rbf_float NORM_OUT_SCALE[] = {0.498888, 0.498888}; // StdDev
#endif

// ==========================================
// Implementation
// ==========================================

RbfClassificationModelRBF::RbfClassificationModelRBF() {
    for(int i=0; i<RBF_HIDDEN_DIM; i++) hidden_activations[i] = 0;
}

void RbfClassificationModelRBF::predict(const rbf_float* input, rbf_float* output) {
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

void RbfClassificationModelRBF::normalizeInput(const rbf_float* input, rbf_float* norm_input) {
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

void RbfClassificationModelRBF::denormalizeOutput(rbf_float* output) {
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

rbf_float RbfClassificationModelRBF::euclideanDistanceSq(const rbf_float* input, int center_idx) {
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

rbf_float RbfClassificationModelRBF::computeKernel(rbf_float dist_sq, rbf_float width) {
    #if RBF_FIXED_POINT
        if (width == 0) return 0;
        double d = (double)dist_sq / RBF_SCALE_FACTOR;
        double w = (double)width / RBF_SCALE_FACTOR;
        return (rbf_float)(exp(-d / (2.0 * w * w)) * RBF_SCALE_FACTOR);
    #else
        return exp(-dist_sq / (2.0f * width * width));
    #endif
}

void RbfClassificationModelRBF::printInfo() {
    Serial.println(F("Model Loaded: RbfClassificationModelRBF"));
    Serial.print(F("Normalization: ")); 
    Serial.println(RBF_USE_NORMALIZATION ? F("YES") : F("NO"));
}
