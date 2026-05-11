#ifndef RNNMODEL_H
#define RNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

class RNNModel {
public:
    RNNModel();
    float predict(const float* input, int seq_len);

private:
    // Shared scratch buffers (sized for the largest layer)
    static float gate_buf[256];
    static float dense_in[64];
    static float dense_out[64];

    // Per-layer hidden / cell state buffers
    // (FIX: each layer must have its own h and c to preserve
    //  the recurrent state across time steps independently)
    static float h0[64];
    static float h0_new[64];
    static float h1[64];
    static float h1_new[64];

    static const float rec0_W_ih_w[64][1];
    static const float rec0_W_ih_b[64];
    static const float rec0_W_hh_w[64][64];
    static const float rec0_W_hh_b[64];
    static const float rec1_W_ih_w[64][64];
    static const float rec1_W_ih_b[64];
    static const float rec1_W_hh_w[64][64];
    static const float rec1_W_hh_b[64];
    static const float den0_w[32][64];
    static const float den0_b[32];
    static const float den1_w[1][32];
    static const float den1_b[1];
};

#endif