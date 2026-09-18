/*
 * SOMModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input dim      : 2
 * Map size       : 8 x 8
 * Input values   : [0.189751, 0.535791]
 *
 * Expected BMU   : flat=9  row=1  col=1
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

  const int INPUT_DIM = 2;
  float x[INPUT_DIM] = { 0.18975078f, 0.53579074f };

  int bmu = model.predict(x);

  // Expected: flat=9  row=1  col=1
  Serial.print("BMU flat index : "); Serial.println(bmu);
  Serial.print("BMU row        : "); Serial.println(model.bmu_row(bmu));
  Serial.print("BMU col        : "); Serial.println(model.bmu_col(bmu));
  Serial.print("Quant. error   : ");
  Serial.println(model.quantization_error(x, bmu), 6);

  float proto[INPUT_DIM];
  model.quantize(bmu, proto);
  Serial.println("BMU weight vector:");
  for (int j = 0; j < INPUT_DIM; j++) {
    Serial.print("  w["); Serial.print(j);
    Serial.print("] = "); Serial.println(proto[j], 8);
  }
}

void loop() {
  // Nothing to do here.
}