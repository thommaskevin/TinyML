#include <math.h>
double score(double *input)
{
    return exp(1.7347124654302846 + input[0] * 0.011406244946132144 + input[1] * 0.01010646886054758 + input[2] * 0.0028201461971878914);
}
