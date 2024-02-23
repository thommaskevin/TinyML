#pragma once
namespace Eloquent {
    namespace ML {
        namespace Port {
            class PCA {
                public:
                    /**
                    * Apply dimensionality reduction
                    * @warn Will override the source vector if no dest provided!
                    */
                    void transform(float *x, float *dest = NULL) {
                        static float u[2] = { 0 };
                        u[0] = dot(x,   -0.010208828727  , 0.009522664636  , 0.07801034572  , 0.004103495259  , -0.003935853874  , -0.004232033634  , 0.21088426957  , -0.00282519103  , 0.974263356619 );
                        u[1] = dot(x,   -0.015215508318  , 0.016476395215  , 0.128355383147  , 0.008750586985  , -0.049637455695  , -0.049172415055  , 0.964175225747  , -0.0112017847  , -0.219782142624 );
                        memcpy(dest != NULL ? dest : x, u, sizeof(float) * 2);
                    }

                protected:
                    /**
                    * Compute dot product with varargs
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, 9);
                        static float mean[] = {  10.998048780488 , 7.26 , 14.477073170732 , 1.976829268293 , 49.616341463415 , 14.605853658537 , 44.325609756098 , 4.76243902439 , 58.316585365854  };
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 9; i++) {
                            dot += (x[i] - mean[i]) * va_arg(w, double);
                        }

                        return dot;
                    }
                };
            }
        }
    }