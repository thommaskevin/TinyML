// CTModel.h  —  Auto-generated Causal Tree inference engine
// Do NOT edit manually.

#pragma once
#include <math.h>

class CTModel {
public:
    float predict(float* x);
};

float CTModel::predict(float* x) {
    if (x[0] <= -0.13278285f) {
        return 0.26748833f;
    } else {
        return 1.82389114f;
    }
}
