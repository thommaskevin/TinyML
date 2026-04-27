// SNN inference example sketch
#include "SNNModel.h"

SNNModel model;

void setup() {
  Serial.begin(115200);
  float input[SNN_INPUT_SIZE] = {0.4112, 0.8610};
  float output = model.predict(input);
  Serial.print("Predicted class: ");
  Serial.println((int)output);
}

void loop() {
  // Nothing to do here
}