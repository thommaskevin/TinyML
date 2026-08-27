/*
 * QRNNModel — Arduino verification sketch
 * Auto-generated — do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input size       : 1
 * Num quantiles (K): 5
 * Input values     : [1.764052]
 *
 * Expected outputs (Python reference):
 * Q0.10 → -0.66467577
 * Q0.25 → 0.06903267
 * Q0.50 → 0.98674732
 * Q0.75 → 1.78638732
 * Q0.90 → 2.80685854
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
  const int K          = 5;

  float x[INPUT_SIZE] = { 1.76405239f };
  float output[K];

  model.predict(x, output);

  // Expected Q0.10: -0.66467577
  Serial.print("Q0.10 = ");
  Serial.println(output[0], 8);
  // Expected Q0.25: 0.06903267
  Serial.print("Q0.25 = ");
  Serial.println(output[1], 8);
  // Expected Q0.50: 0.98674732
  Serial.print("Q0.50 = ");
  Serial.println(output[2], 8);
  // Expected Q0.75: 1.78638732
  Serial.print("Q0.75 = ");
  Serial.println(output[3], 8);
  // Expected Q0.90: 2.80685854
  Serial.print("Q0.90 = ");
  Serial.println(output[4], 8);
}

void loop() {
  // Nothing to do here.
}