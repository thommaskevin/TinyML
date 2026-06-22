// CTModel.h  —  Auto-generated Causal Tree inference engine
// Do NOT edit manually.

#pragma once
#include <math.h>

class CTModel {
public:
    float predict(float* x);
};

float CTModel::predict(float* x) {
    if (x[1] <= -0.74304290f) {
        return 0.19905586f;
    } else {
        return 0.13518519f;
    }
}
