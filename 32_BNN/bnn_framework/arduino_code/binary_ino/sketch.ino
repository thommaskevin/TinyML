// BNN model example sketch
#include "BNNModel.h"

BNNModel model;

void setup() {
  Serial.begin(115200);
  float input[2] = { 1.5184023,  -0.55922781};
  float output = model.predict(input);
  Serial.print("Predicted logit: ");
  Serial.println(output, 6);
  Serial.print("Predicted class (0 or 1): ");
  Serial.print(output > 0 ? 1 : 0);
  Serial.println("  | Real class (0 or 1): 1 ");
}

void loop() {
  // Nothing to do here
}