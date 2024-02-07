#include "LogisticRegressor.h"

Eloquent::ML::Port::LogisticRegressor LogisticRegressor;

void setup()
{
  Serial.begin(115200);
}

void loop()
{
  float X_1[] = {6., 2.7, 5.1, 1.6};
  int result_1 = LogisticRegressor.predict(X_1);
  Serial.print("Result of predict with input X1 (real value = 1):");
  Serial.println(result_1);
  delay(2000);

  float X_2[] = {4.8, 3.1, 1.6, 0.2};
  int result_2 = LogisticRegressor.predict(X_2);
  Serial.print("Result of predict with input X2 (real value = 0):");
  Serial.println(result_2);
  delay(2000);
}
