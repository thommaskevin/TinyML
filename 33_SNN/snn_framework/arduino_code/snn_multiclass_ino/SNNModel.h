#ifndef SNNMODEL_H
#define SNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

#define SNN_NUM_STEPS 25
#define SNN_INPUT_SIZE 2
#define SNN_OUTPUT_SIZE 3

class SNNModel {
public:
    SNNModel();
    float predict(const float* input);

private:
    static float spike_buf_a[30];
    static float spike_buf_b[30];
    static float u0[30];   // membrane potential, layer 0
    static float u1[3];   // membrane potential, layer 1

    static const float w0[30][2];
    static const float b0[30];
    static const float w1[3][30];
    static const float b1[3];
};

#endif