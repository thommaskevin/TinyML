/*
 * ETModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * n_features : 4
 * n_trees    : 200
 * task       : regression
 * Expected value: 0.12678907
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

  float x[4] = { 0.35570350f, -0.47728628f, 0.46637958f, -0.09439251f };

  // Expected value: 0.12678907
    Serial.print("Predicted value : "); Serial.println(model.predict(x), 8);
}

void loop() {
  // Nothing to do here.
}