// ANFISModel.h  —  Auto-generated Adaptive Neuro-Fuzzy Inference System
// Do NOT edit weights manually.

#pragma once
#include <math.h>

// ANFIS config: n_inputs=2, n_terms=4, n_rules=16, mf=gaussian

static const float fuzzify_center[2][4] = {
    {-1.90458262f, 0.48567772f, -0.40124092f, 2.93471670f},
    {-0.49982843f, -0.02446537f, 0.80958366f, 1.79818833f}
};
static const float fuzzify_sigma[2][4] = {
    {-0.01154431f, 0.17881469f, 0.10733767f, 0.80545264f},
    {1.52924335f, 0.39546517f, -0.03689066f, -0.09560809f}
};

static const float consequent_coeff[16][3] = {
    {-0.33766767f, 1.84512067f, 0.25740299f},
    {-2.82021976f, -0.24593744f, -5.34622622f},
    {0.88750619f, 0.00192730f, -4.09184837f},
    {-0.26608869f, 4.38534451f, -1.95053351f},
    {-0.18437420f, 2.50194955f, 1.28276694f},
    {-4.34604502f, -1.43119001f, 1.43825471f},
    {5.63508892f, -3.78007913f, 0.31290805f},
    {0.70044315f, 2.33001137f, 0.76734638f},
    {0.00909731f, 0.40670970f, 0.57209069f},
    {-0.66868460f, -0.61633086f, -0.82241446f},
    {2.87673473f, -1.72554648f, 0.31097779f},
    {-1.88955903f, 0.64279836f, -0.30386567f},
    {0.35347995f, -1.29593515f, -0.54727119f},
    {-0.97191650f, 0.54107070f, 0.48564619f},
    {-0.22713228f, 1.30463731f, -0.46651495f},
    {-0.70085698f, 0.31745932f, -1.15453660f}
};

class ANFISModel {
public:
    float predict(float* input);

private:
    float mu[2][4];
    float w[16];
    float w_bar[16];
    float f_val[16];
    float dense_out[16];
};

float ANFISModel::predict(float* input) {
    const int N_IN  = 2;
    const int N_TE  = 4;
    const int N_R   = 16;

    // ---- Layer 1: Fuzzification ----
    for (int j = 0; j < N_IN; j++) {
        for (int k = 0; k < N_TE; k++) {
            float c = fuzzify_center[j][k];
            float s = fuzzify_sigma[j][k];
            float diff = input[j] - c;
            mu[j][k] = expf(-0.5f * (diff * diff) / (s * s + 1e-6f));
        }
    }

    // ---- Layer 2: Rule firing (product T-norm) ----
    for (int r = 0; r < N_R; r++) {
        float prod = 1.0f;
        int tmp = r;
        for (int j = 0; j < N_IN; j++) {
            int t = tmp % N_TE;
            prod *= mu[j][t];
            tmp /= N_TE;
        }
        w[r] = prod;
    }

    // ---- Layer 3: Normalisation ----
    float sum_w = 0.0f;
    for (int r = 0; r < N_R; r++) sum_w += w[r];
    sum_w = sum_w < 1e-8f ? 1e-8f : sum_w;
    for (int r = 0; r < N_R; r++) w_bar[r] = w[r] / sum_w;

    // ---- Layer 4: Consequent ----
    for (int r = 0; r < N_R; r++) {
        float acc = consequent_coeff[r][0];
        for (int j = 0; j < N_IN; j++)
            acc += consequent_coeff[r][j+1] * input[j];
        f_val[r] = acc;
    }

    // ---- Layer 5: Defuzzification ----
    float y = 0.0f;
    for (int r = 0; r < N_R; r++)
        y += w_bar[r] * f_val[r];

    return y;
}