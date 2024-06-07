#include "XGBRegressor.h"


void setup() {
  Serial.begin(115200);
}

void loop() {
  double X_1[] = { 2.71782911e-02,  5.06801187e-02,  1.75059115e-02,
                  -3.32135761e-02, -7.07277125e-03,  4.59715403e-02,
                  -6.54906725e-02,  7.12099798e-02, -9.64332229e-02,
                  -5.90671943e-02};
  double result_1 = score(X_1);
  Serial.print("Result of predict with input X1 (real value = 69):");
  Serial.println(String(result_1, 7));
  delay(2000);
                                                                                                                                       
}


