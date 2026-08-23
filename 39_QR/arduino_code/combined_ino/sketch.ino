/*
 * QRNNModel — Arduino verification sketch
 * Auto-generated — do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input size       : 5
 * Num quantiles (K): 7
 * Input values     : [1.764052, 0.400157, 0.978738, 2.240893, 1.867558]
 *
 * Expected outputs (Python reference):
 * Q0.05 → 1.67010224
 * Q0.10 → 1.67010224
 * Q0.25 → 2.05318475
 * Q0.50 → 2.05318570
 * Q0.75 → 2.87453127
 * Q0.90 → 2.87468672
 * Q0.95 → 3.01162505
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

  const int INPUT_SIZE = 5;
  const int K          = 7;

  float x[INPUT_SIZE] = { 1.76405239f, 0.40015721f, 0.97873801f, 2.24089313f, 1.86755800f };
  float output[K];

  model.predict(x, output);

  // Expected Q0.05: 1.67010224
  Serial.print("Q0.05 = ");
  Serial.println(output[0], 8);
  // Expected Q0.10: 1.67010224
  Serial.print("Q0.10 = ");
  Serial.println(output[1], 8);
  // Expected Q0.25: 2.05318475
  Serial.print("Q0.25 = ");
  Serial.println(output[2], 8);
  // Expected Q0.50: 2.05318570
  Serial.print("Q0.50 = ");
  Serial.println(output[3], 8);
  // Expected Q0.75: 2.87453127
  Serial.print("Q0.75 = ");
  Serial.println(output[4], 8);
  // Expected Q0.90: 2.87468672
  Serial.print("Q0.90 = ");
  Serial.println(output[5], 8);
  // Expected Q0.95: 3.01162505
  Serial.print("Q0.95 = ");
  Serial.println(output[6], 8);
}

void loop() {
  // Nothing to do here.
}