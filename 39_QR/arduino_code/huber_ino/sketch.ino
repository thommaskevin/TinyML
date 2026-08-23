/*
 * QRNNModel — Arduino verification sketch
 * Auto-generated — do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input size       : 1
 * Num quantiles (K): 3
 * Input values     : [1.764052]
 *
 * Expected outputs (Python reference):
 * Q0.05 → -0.38100770
 * Q0.50 → 1.95364094
 * Q0.95 → 3.92858553
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
  const int K          = 3;

  float x[INPUT_SIZE] = { 1.76405239f };
  float output[K];

  model.predict(x, output);

  // Expected Q0.05: -0.38100770
  Serial.print("Q0.05 = ");
  Serial.println(output[0], 8);
  // Expected Q0.50: 1.95364094
  Serial.print("Q0.50 = ");
  Serial.println(output[1], 8);
  // Expected Q0.95: 3.92858553
  Serial.print("Q0.95 = ");
  Serial.println(output[2], 8);
}

void loop() {
  // Nothing to do here.
}