#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SEFR {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        return dot(x,   0.089210262828  , -0.097438891661  , 0.48245740461  , 0.68784598788 ) <= 2.0193337291074043 ? 0 : 1;
                    }

                protected:
                    /**
                    * Compute dot product
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, 4);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 4; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }

                        return dot;
                    }
                };
            }
        }
    }