#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class LinearRegression {
                public:
                    /**
                    * Predict class for features vector
                    */
                    float predict(float *x) {
                        return dot(x, 8.519119048170, -63.742223819482, 120.195642585753, 48.382158099066, -247.051828054808, 139.424879828758, 59.514918286320, 67.272041311423, 336.697813626620, -48.176667846134) + 37.12282895581433;
                    }

                protected:
                    /**
                    * Compute dot product
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, 10);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 10; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }

                        return dot;
                    }
                };
            }
        }
    }