/*
 * ETModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * n_features : 30
 * n_trees    : 200
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

  float x[30] = { 0.75340682f, -0.11394449f, 0.71386600f, 0.65815634f, -0.54870558f, -0.23729117f, -0.05749318f, 0.43439269f, 0.29711878f, -1.05719376f, 0.69956577f, 0.31596178f, 0.62519437f, 0.59411043f, -0.30600539f, -0.04348982f, -0.18679726f, 0.68839556f, 0.04452071f, -0.11419554f, 0.78293973f, 0.10141534f, 0.69814396f, 0.66698176f, -0.68246317f, -0.26950088f, -0.19376470f, 0.49933764f, -0.14682283f, -0.64647079f };

  // Expected class: 0
    Serial.print("Predicted class : "); Serial.println(model.predict(x));
}

void loop() {
  // Nothing to do here.
}