#ifndef SNNMODEL_H
#define SNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

#define SNN_NUM_STEPS 25
#define SNN_INPUT_SIZE 1
#define SNN_OUTPUT_SIZE 1

class SNNModel {
public:
    SNNModel();
    float predict(const float* input);

private:
    static float spike_buf_a[64];
    static float spike_buf_b[64];
    static float u0[64];   // membrane potential, layer 0
    static float u1[64];   // membrane potential, layer 1
    static float u2[1];   // membrane potential, layer 2

    static const float w0[64][1];
    static const float b0[64];
    static const float w1[64][64];
    static const float b1[64];
    static const float w2[1][64];
    static const float b2[1];
};

#endif