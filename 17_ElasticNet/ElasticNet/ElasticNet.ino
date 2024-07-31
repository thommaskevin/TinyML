#include "ElasticNet.h"

void setup()
{
  Serial.begin(115200);
}

void loop()
{
  double X_1[] = {4.6, 8, 304};
  float result_1 = score(X_1);
  Serial.print("Result of predict with input X1 (real value = 15.4):");
  Serial.println(result_1);
  delay(2000);

  double X_2[] = {1.5, 4, 216};
  float result_2 = score(X_2);
  Serial.print("Result of predict with input X2 (real value = 11.3):");
  Serial.println(result_2);
  delay(2000);
}