#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[2] <= 2.449999988079071) {
                            return 0;
                        }

                        else {
                            if (x[3] <= 1.6500000357627869) {
                                if (x[2] <= 4.950000047683716) {
                                    return 1;
                                }

                                else {
                                    if (x[3] <= 1.550000011920929) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 4.8500001430511475) {
                                    if (x[1] <= 3.100000023841858) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    return 2;
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }