// ANFISModel.h  —  Auto-generated Adaptive Neuro-Fuzzy Inference System
// Do NOT edit weights manually.

#pragma once
#include <math.h>

// ANFIS config: n_inputs=2, n_terms=5, n_rules=25, mf=gaussian

static const float fuzzify_center[2][5] = {
    {-0.82569337f, -0.85014582f, -0.05016730f, 0.96144217f, 0.90497911f},
    {-1.89375174f, 0.10202774f, -0.09820934f, 0.31851342f, 1.68984163f}
};
static const float fuzzify_sigma[2][5] = {
    {0.80928290f, 1.21220410f, 0.92797011f, 0.94802588f, 1.29947793f},
    {0.99898499f, 1.45651066f, 1.11148512f, 0.98162270f, 0.89303803f}
};

static const float consequent_coeff[25][3] = {
    {1.84671545f, -0.53112745f, -0.89067775f},
    {-0.00224687f, 0.17397633f, 0.14500022f},
    {0.71682125f, 0.05459082f, 0.12493400f},
    {-1.88191617f, -0.88640273f, 0.64719820f},
    {-1.91578567f, 1.40478420f, -0.25981849f},
    {-0.21561888f, 1.21335495f, 0.46085155f},
    {-0.69869512f, 0.01113754f, -1.89814794f},
    {-0.68152863f, 0.50661850f, 1.16395068f},
    {2.10705686f, -0.40888494f, -1.83221531f},
    {1.31410813f, -0.89787984f, 1.27872109f},
    {-1.80365574f, -0.30759501f, -1.36955667f},
    {-1.35473740f, -1.04894304f, 0.55728614f},
    {-1.71697772f, 0.93254238f, -0.91767591f},
    {2.58085322f, 0.73388594f, -0.10815912f},
    {2.08710504f, -0.60825431f, 0.60712886f},
    {-1.41083789f, 2.06349850f, 0.04429647f},
    {-0.19287024f, -0.18814169f, 2.55739355f},
    {-0.28816274f, 3.01094890f, -0.69956648f},
    {2.62843466f, 0.60837013f, -0.48422140f},
    {-0.59889859f, -0.35711530f, 0.21164964f},
    {1.07012129f, -1.60719836f, 0.19058421f},
    {-0.99719572f, 0.73603565f, 1.40489483f},
    {-0.03417151f, -1.38013303f, -0.50525862f},
    {-0.39329746f, 0.11612441f, -2.09366345f},
    {-0.66978216f, -0.30588043f, 1.03231180f}
};

class ANFISModel {
public:
    float predict(float* input);

private:
    float mu[2][5];
    float w[25];
    float w_bar[25];
    float f_val[25];
    float dense_out[16];
};

float ANFISModel::predict(float* input) {
    const int N_IN  = 2;
    const int N_TE  = 5;
    const int N_R   = 25;

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