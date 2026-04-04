// BNN model example sketch
#include "BNNModel.h"

BNNModel model;

void setup() {
  Serial.begin(115200);
  float input[2] = {-2.62485019,  9.52601409};
  float output = model.predict(input);
  Serial.print("Predicted class: ");
  Serial.println((int)output);
  Serial.print("  | Real class: 0 ");
}

void loop() {
  // Nothing to do here
}