// Arduino sketch
#include "GaussianNB.h"

Eloquent::ML::Port::GaussianNB classifier;

void setup() {
  Serial.begin(115200);

}

void loop() {
  float X_1[] = {5.1, 3.5, 1.4, 0.2};
  int result_1 = classifier.predict(X_1);
  Serial.print("Result of predict with input X1:");
  Serial.println(result_1);
  delay(2000);

  float X_2[] = {6.2, 2.2, 4.5, 1.5};
  int result_2 = classifier.predict(X_2);
  Serial.print("Result of predict with input X2:");
  Serial.println(result_2); 
  delay(2000);

  float X_3[] = {6.1, 3.0, 4.9, 1.8};
  int result_3 = classifier.predict(X_3);
  Serial.print("Result of predict with input X3:");
  Serial.println(result_3);
  delay(2000);
}
