/*
 * ETModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * n_features : 4
 * n_trees    : 100
 * task       : classification
 * Expected class: 0
 *
 * Upload this sketch, open Serial Monitor at 115200 baud,
 * and confirm the printed prediction matches the expected
 * value above.
 */

#include "ETModel.h"

ETModel model;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  float x[4] = { 4.40000010f, 3.20000005f, 1.29999995f, 0.20000000f };

  // Expected class: 0
    Serial.print("Predicted class : "); Serial.println(model.predict(x));
}

void loop() {
  // Nothing to do here.
}