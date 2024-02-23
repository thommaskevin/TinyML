#include "LinearRegressor.h"

Eloquent::ML::Port::LinearRegression LinearRegressor;

void setup()
{
  Serial.begin(115200);
}

void loop()
{
  float X_1[] = {0.02717829,  0.05068012,  0.01750591, -0.03321323, -0.00707277, 0.04597154, -0.06549067,  0.07120998, -0.09643495, -0.05906719};
  int result_1 = LinearRegressor.predict(X_1);
  Serial.print("Result of predict with input X1 (real value = 13):");
  Serial.println(result_1);
  delay(2000);

  float X_2[] = {-0.07816532, -0.04464164, -0.0730303 , -0.05731319, -0.08412613,-0.07427747, -0.02499266, -0.03949338, -0.01811369, -0.08391984};
  int result_2 = LinearRegressor.predict(X_2);
  Serial.print("Result of predict with input X2 (real value = 40):");
  Serial.println(result_2);
  delay(2000);
}
