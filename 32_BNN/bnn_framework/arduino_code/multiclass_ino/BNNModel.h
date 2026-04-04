#ifndef BNNMODEL_H
#define BNNMODEL_H

#include <stdint.h>
#include <Arduino.h>

class BNNModel {
public:
    BNNModel();
    float predict(const float* input);

private:
    static float input_buffer[30];
    static float hidden_buffer[30];

    static const float w0_mu[30][2];
    static const float b0_mu[30];
    static const float w2_mu[3][30];
    static const float b2_mu[3];
};

#endif