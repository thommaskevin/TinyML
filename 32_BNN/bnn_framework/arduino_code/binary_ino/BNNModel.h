#ifndef BNNMODEL_H
#define BNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

class BNNModel {
public:
    BNNModel();
    float predict(const float* input);

private:
    static float input_buffer[20];
    static float hidden_buffer[20];

    static const float w0_mu[20][2];
    static const float b0_mu[20];
    static const float w2_mu[20][20];
    static const float b2_mu[20];
    static const float w4_mu[1][20];
    static const float b4_mu[1];
};

#endif