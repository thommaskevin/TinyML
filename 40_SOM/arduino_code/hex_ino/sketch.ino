/*
 * SOMModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input dim      : 6
 * Map size       : 12 x 12
 * Input values   : [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
 *
 * Expected BMU   : flat=0  row=0  col=0
 *
 * Upload this sketch, open Serial Monitor at 115200 baud,
 * and confirm the printed BMU matches the expected values above.
 *
 * Acceptable tolerance: exact integer match for BMU index.
 */

#include "SOMModel.h"

SOMModel model;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  const int INPUT_DIM = 6;
  float x[INPUT_DIM] = { 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f };

  int bmu = model.predict(x);

  // Expected: flat=0  row=0  col=0
  Serial.print("BMU flat index : "); Serial.println(bmu);
  Serial.print("BMU row        : "); Serial.println(model.bmu_row(bmu));
  Serial.print("BMU col        : "); Serial.println(model.bmu_col(bmu));
  Serial.print("Quant. error   : ");
  Serial.println(model.quantization_error(x, bmu), 6);
}

void loop() {
  // Nothing to do here.
}