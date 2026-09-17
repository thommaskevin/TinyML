/*
 * SOMModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input dim      : 4
 * Map size       : 10 x 10
 * Input values   : [0.222222, 0.625000, 0.067797, 0.041667]
 *
 * Expected BMU   : flat=39  row=3  col=9
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

  const int INPUT_DIM = 4;
  float x[INPUT_DIM] = { 0.22222222f, 0.62500000f, 0.06779661f, 0.04166667f };

  int bmu = model.predict(x);

  // Expected: flat=39  row=3  col=9
  Serial.print("BMU flat index : "); Serial.println(bmu);
  Serial.print("BMU row        : "); Serial.println(model.bmu_row(bmu));
  Serial.print("BMU col        : "); Serial.println(model.bmu_col(bmu));
  Serial.print("Quant. error   : ");
  Serial.println(model.quantization_error(x, bmu), 6);
}

void loop() {
  // Nothing to do here.
}