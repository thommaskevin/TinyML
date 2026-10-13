/*
 * ETModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * n_features : 13
 * n_trees    : 200
 * task       : classification
 * Expected class: 2
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

  float x[13] = { 0.20923167f, 0.22769377f, 0.01273209f, 0.15166079f, 1.42241180f, -1.03368390f, -1.35462189f, 1.35488844f, -0.22934584f, 1.83492303f, -1.56825173f, -1.40069890f, 0.29649782f };

  // Expected class: 2
    Serial.print("Predicted class : "); Serial.println(model.predict(x));
}

void loop() {
  // Nothing to do here.
}