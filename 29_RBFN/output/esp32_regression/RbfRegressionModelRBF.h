#ifndef RBFREGRESSIONMODELRBF_H
#define RBFREGRESSIONMODELRBF_H

#include <Arduino.h>
#include <math.h>

// Dimensions
#define RBF_INPUT_DIM 1
#define RBF_HIDDEN_DIM 15
#define RBF_OUTPUT_DIM 1
#define RBF_SCALE_FACTOR 1
#define RBF_FIXED_POINT 0
#define RBF_USE_NORMALIZATION 1

typedef float rbf_float;

class RbfRegressionModelRBF {
public:
    RbfRegressionModelRBF();
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
};

#endif
