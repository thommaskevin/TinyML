#ifndef SNNMODEL_H
#define SNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

#define SNN_NUM_STEPS 25
#define SNN_INPUT_SIZE 2
#define SNN_OUTPUT_SIZE 1

class SNNModel {
public:
    SNNModel();
    float predict(const float* input);

private:
    static float spike_buf_a[20];
    static float spike_buf_b[20];
    static float u0[20];   // membrane potential, layer 0
    static float u1[20];   // membrane potential, layer 1
    static float u2[1];   // membrane potential, layer 2

    static const float w0[20][2];
    static const float b0[20];
    static const float w1[20][20];
    static const float b1[20];
    static const float w2[1][20];
    static const float b2[1];
};

#endif