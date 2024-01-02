#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class GaussianNB {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float votes[3] = { 0.0f };
                        float theta[4] = { 0 };
                        float sigma[4] = { 0 };
                        theta[0] = 4.964516129032; theta[1] = 3.377419354839; theta[2] = 1.464516129032; theta[3] = 0.248387096774;
                        sigma[0] = 0.111966704288; sigma[1] = 0.136586891592; sigma[2] = 0.033257026868; sigma[3] = 0.011529659543;
                        votes[0] = 0.295238095238 - gauss(x, theta, sigma);
                        theta[0] = 5.862162162162; theta[1] = 2.724324324324; theta[2] = 4.210810810811; theta[3] = 1.302702702703;
                        sigma[0] = 0.275325057719; sigma[1] = 0.087246168019; sigma[2] = 0.239342588764; sigma[3] = 0.041344049684;
                        votes[1] = 0.352380952381 - gauss(x, theta, sigma);
                        theta[0] = 6.559459459459; theta[1] = 2.986486486486; theta[2] = 5.545945945946; theta[3] = 2.005405405405;
                        sigma[0] = 0.422410521562; sigma[1] = 0.096303874374; sigma[2] = 0.288429513527; sigma[3] = 0.085916730473;
                        votes[2] = 0.352380952381 - gauss(x, theta, sigma);
                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 3; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                protected:
                    /**
                    * Compute gaussian value
                    */
                    float gauss(float *x, float *theta, float *sigma) {
                        float gauss = 0.0f;

                        for (uint16_t i = 0; i < 4; i++) {
                            gauss += log(sigma[i]);
                            gauss += abs(x[i] - theta[i]) / sigma[i];
                        }

                        return gauss;
                    }
                };
            }
        }
    }