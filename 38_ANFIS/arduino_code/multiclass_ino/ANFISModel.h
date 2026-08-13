// ANFISModel.h  —  Auto-generated Adaptive Neuro-Fuzzy Inference System
// Do NOT edit weights manually.

#pragma once
#include <math.h>

// ANFIS config: n_inputs=2, n_terms=4, n_rules=16, mf=gaussian

static const float fuzzify_center[2][4] = {
    {-1.49741304f, 0.09696731f, 0.36313954f, 1.33315969f},
    {-1.00605440f, -0.06670769f, 1.03693521f, 0.23174255f}
};
static const float fuzzify_sigma[2][4] = {
    {0.34605464f, 0.75623173f, 0.53790176f, 0.37674785f},
    {0.66185117f, 0.28352353f, 0.90286642f, 0.26040837f}
};

static const float consequent_coeff[16][3] = {
    {0.70862168f, -1.16456485f, -1.74035871f},
    {0.63294393f, -0.71321428f, -1.53894985f},
    {-0.10199923f, -0.75801086f, 0.49823812f},
    {0.64699066f, -1.50672102f, 0.86583704f},
    {2.21052909f, -0.08978898f, -0.86904311f},
    {0.04215474f, 1.58350074f, -0.70801902f},
    {-1.31627226f, -0.34261250f, -0.45376107f},
    {-0.93899184f, 0.07920256f, 0.23116419f},
    {-0.19506857f, -0.90317351f, -1.09338939f},
    {-1.01010597f, 1.63677728f, -2.08620644f},
    {-2.03922844f, 0.28801775f, -1.15192342f},
    {1.52646017f, 0.39003813f, -0.86685812f},
    {0.36495727f, 1.13842142f, -1.31790721f},
    {-0.11527890f, 1.24806380f, 0.12141243f},
    {-1.01225603f, 0.59734190f, -1.70601857f},
    {-1.65882480f, -0.48882130f, 0.44018579f}
};

// Dense layer 0  in=1  out=16
static const float den0_w[16][1] = {
    {0.94363821f},
    {-0.98397726f},
    {0.37383738f},
    {-1.28405511f},
    {0.25628901f},
    {1.35142279f},
    {0.71237528f},
    {-0.06508902f},
    {0.10036910f},
    {-1.16609693f},
    {1.09455895f},
    {-1.13157475f},
    {-1.56443501f},
    {0.35157418f},
    {-1.35295022f},
    {0.34365165f}
};
static const float den0_b[] = {-0.09396424f, -0.37203583f, 1.53299630f, -0.53514755f, 1.86288488f, 0.10451856f, -0.04387191f, -0.20268813f, -0.76695061f, -0.42399478f, -0.71098167f, 1.48592496f, 1.21878433f, 0.29394200f, -0.63866735f, 0.99227941f};

// Dense layer 1  in=16  out=3
static const float den1_w[3][16] = {
    {-0.62642103f, 0.41313544f, -0.74404192f, 0.80708957f, -0.68594313f, -1.06341863f, -0.42731038f, -0.10721479f, -0.10405187f, 0.52223307f, -0.40733200f, 0.50852078f, 0.51243085f, -0.31493971f, 1.90113020f, -0.45545888f},
    {-0.64405137f, -0.82801539f, 0.76567560f, -0.72152364f, 0.80828303f, -1.73949635f, -0.81911296f, 0.14230710f, 0.12743753f, -0.81156451f, -0.71645039f, 1.06577933f, 0.90698212f, 0.11113621f, -1.78531539f, 0.74010777f},
    {0.52791864f, -0.60908896f, 0.08483016f, -0.96275800f, 0.34078488f, 1.32292366f, 0.45467106f, -0.22600116f, -0.10897078f, -0.26665717f, 0.61551666f, -0.86895955f, -0.75449312f, 0.27895933f, -0.50934964f, 0.22989997f}
};
static const float den1_b[] = {-0.78075564f, 0.99141276f, 0.23710458f};

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

    dense_out[0] = y;
    // Dense 0
    float d_out0[16];
    for (int k = 0; k < 16; k++) {
        float acc = den0_b[k];
        for (int m = 0; m < 1; m++)
            acc += den0_w[k][m] * dense_out[m];
        if (acc < 0.0f) acc = 0.0f;
        d_out0[k] = acc;
    }
    for (int k = 0; k < 16; k++) dense_out[k] = d_out0[k];

    // Dense 1
    float d_out1[3];
    for (int k = 0; k < 3; k++) {
        float acc = den1_b[k];
        for (int m = 0; m < 16; m++)
            acc += den1_w[k][m] * dense_out[m];
        d_out1[k] = acc;
    }

    return d_out1[0];
}