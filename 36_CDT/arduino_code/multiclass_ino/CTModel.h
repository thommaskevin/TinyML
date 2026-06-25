// CTModel.h  —  Auto-generated Causal Tree inference engine
// Do NOT edit manually.

#pragma once
#include <math.h>

static const int CT_N_CLASSES = 3;

class CTModel {
public:
    int predict(float* x);
};

int CTModel::predict(float* x) {
    if (x[3] <= -0.35107985f) {
        if (x[4] <= 0.87765208f) {
            if (x[4] <= -0.71772499f) {
                {
                    const float tau[3] = {-0.20769231f, -0.31538462f, 0.52307692f};
                    int best = 0;
                    for (int k = 1; k < 3; k++)
                        if (tau[k] > tau[best]) best = k;
                    return best;
                }
            } else {
                {
                    const float tau[3] = {-0.14841270f, -0.00873016f, 0.15714286f};
                    int best = 0;
                    for (int k = 1; k < 3; k++)
                        if (tau[k] > tau[best]) best = k;
                    return best;
                }
            }
        } else {
            {
                const float tau[3] = {0.00000000f, 0.00000000f, 0.00000000f};
                int best = 0;
                for (int k = 1; k < 3; k++)
                    if (tau[k] > tau[best]) best = k;
                return best;
            }
        }
    } else {
        if (x[0] <= 0.09385948f) {
            {
                const float tau[3] = {-0.12217924f, -0.01353965f, 0.13571889f};
                int best = 0;
                for (int k = 1; k < 3; k++)
                    if (tau[k] > tau[best]) best = k;
                return best;
            }
        } else {
            if (x[1] <= -0.88298781f) {
                {
                    const float tau[3] = {-0.28571429f, 0.00000000f, 0.28571429f};
                    int best = 0;
                    for (int k = 1; k < 3; k++)
                        if (tau[k] > tau[best]) best = k;
                    return best;
                }
            } else {
                {
                    const float tau[3] = {-0.17794118f, -0.05588235f, 0.23382353f};
                    int best = 0;
                    for (int k = 1; k < 3; k++)
                        if (tau[k] > tau[best]) best = k;
                    return best;
                }
            }
        }
    }
}
