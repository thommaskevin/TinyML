/*
 * QRNNModel — Arduino verification sketch
 * Auto-generated — do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input size       : 1
 * Num quantiles (K): 1
 * Input values     : [1.764052]
 *
 * Expected outputs (Python reference):
 * Q0.50 → 0.91831297
 *
 * Upload this sketch, open Serial Monitor at 115200 baud,
 * and confirm each printed quantile matches its expected value
 * to at least 5 decimal places.
 *
 * Acceptable tolerance: ±0.0001  (float32 rounding)
 */

#include "QRNNModel.h"

QRNNModel model;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  const int INPUT_SIZE = 1;
  const int K          = 1;

  float x[INPUT_SIZE] = { 1.76405239f };
  float output[K];

  model.predict(x, output);

  // Expected Q0.50: 0.91831297
  Serial.print("Q0.50 = ");
  Serial.println(output[0], 8);
}

void loop() {
  // Nothing to do here.
}