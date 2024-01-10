#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float kernels[3] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   3.4  , 1.9  , 0.2 );
                        kernels[1] = compute_kernel(x,   3.3  , 1.7  , 0.5 );
                        kernels[2] = compute_kernel(x,   2.4  , 3.3  , 1.0 );
                        float decision = -0.833910342285;
                        decision = decision - ( + kernels[0] * -0.31945543931  + kernels[1] * -0.240101867421 );
                        decision = decision - ( + kernels[2] * 0.559557306731 );

                        return decision > 0 ? 0 : 1;
                    }

                protected:
                    /**
                    * Compute kernel between feature vector and support vector.
                    * Kernel type: linear
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, 3);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 3; i++) {
                            kernel += x[i] * va_arg(w, double);
                        }

                        return kernel;
                    }
                };
            }
        }
    }