namespace Eloquent {
namespace ML {
namespace Port {
class PoissonRegressor {
public: 

float predict(float *x) { 
float predict = 0;
float coefficients[8] = {0.09042590463658687, -0.09696434781840536, 0.3653828589531427, 0.2711244659564742, 0.10702971852706969, -0.2758301947563339, 0.2689055595556031, 0.458699680210232};

    float z = 3.517012193074436;
    for (int i = 0; i < 8; i++)
     {
       z += coefficients[i] * x[i];
     }

predict =  exp(z);  
return predict; 
} 
};
}
}
}
