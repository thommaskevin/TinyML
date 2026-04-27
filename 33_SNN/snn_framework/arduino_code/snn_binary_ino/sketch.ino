// SNN inference example sketch
#include "SNNModel.h"

SNNModel model;

void setup() {
  Serial.begin(115200);
  float input[SNN_INPUT_SIZE] = {0.5, 0.5};
  float output = model.predict(input);
  Serial.print("Predicted logit: ");
  Serial.println(output, 6);
  Serial.print("Predicted class (0 or 1): ");
  Serial.println(output > 0.0f ? 1 : 0);
}

void loop() {
  // Nothing to do here
}