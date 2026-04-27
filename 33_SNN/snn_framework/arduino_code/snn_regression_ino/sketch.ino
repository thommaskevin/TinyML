// SNN inference example sketch
#include "SNNModel.h"

SNNModel model;

void setup() {
  Serial.begin(115200);
  float input[SNN_INPUT_SIZE] = {0.5};
  float output = model.predict(input);
  Serial.print("Predicted value: ");
  Serial.println(output, 6);
}

void loop() {
  // Nothing to do here
}