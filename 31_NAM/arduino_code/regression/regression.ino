#include "nam_regression.h"

void setup() {
  Serial.begin(115200);
  float features[2] = {0.22793516, 0.98411032};
  float pred = predict(features);
  Serial.print("Predicts: ");
  Serial.print(pred, 7);              // <-- fixed
  Serial.print(" | Real: 1.06570879");
  Serial.print(" | Python Predict: 1.2661812");
}

void loop() {
}