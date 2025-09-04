#ifndef KAN_MODEL_H
#define KAN_MODEL_H

#include <cmath>

float predict(float x_1, float x_2, float x_3, float x_4, float x_5, float x_6, float x_7)
{
    float result = (-0.55425982936965 * x_1 + 0.110666238518483 * x_2 - 0.0450471091341996 * x_3 - 0.330026542054321 * x_4 - 0.0931681484426096 * x_5 + 0.228486458628294 * x_6 + 0.0367771005214196 * x_7 - 0.0771479301391296 * std::pow(0.215272135206579 - x_7, 2) + 0.0879234933576154 * std::pow(-x_3 - 0.0931694283586403, 2) + 0.0538988046415818 * std::pow(-x_7 - 0.181158237036153, 2) + 0.848163410733748);
    return result;
}

#endif // KAN_MODEL_H
