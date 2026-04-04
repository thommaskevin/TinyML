// BNN model example sketch
#include "BNNModel.h"

BNNModel model;

void setup() {
  Serial.begin(115200);
  float input[1] = {-4.};
  float output = model.predict(input);
  Serial.print("Predicted value: ");
  Serial.print(output, 6);
  Serial.println("  | Real class: 0.9495");
}

void loop() {
  // Nothing to do here
}