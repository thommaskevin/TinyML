/*
 * SOMModel -- Arduino verification sketch
 * Auto-generated -- do not edit the weights.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input dim      : 64
 * Map size       : 20 x 20
 * Input values   : [0.000000, 0.000000, 0.312500, 0.812500, 0.562500, 0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.812500, 0.937500, 0.625000, 0.937500, 0.312500, 0.000000, 0.000000, 0.187500, 0.937500, 0.125000, 0.000000, 0.687500, 0.500000, 0.000000, 0.000000, 0.266667, 0.750000, 0.000000, 0.000000, 0.500000, 0.533333, 0.000000, 0.000000, 0.357143, 0.500000, 0.000000, 0.000000, 0.562500, 0.571429, 0.000000, 0.000000, 0.250000, 0.687500, 0.000000, 0.062500, 0.750000, 0.437500, 0.000000, 0.000000, 0.125000, 0.875000, 0.312500, 0.625000, 0.750000, 0.000000, 0.000000, 0.000000, 0.000000, 0.375000, 0.812500, 0.625000, 0.000000, 0.000000, 0.000000]
 *
 * Expected BMU   : flat=182  row=9  col=2
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

  const int INPUT_DIM = 64;
  float x[INPUT_DIM] = { 0.00000000f, 0.00000000f, 0.31250000f, 0.81250000f, 0.56250000f, 0.06250000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.81250000f, 0.93750000f, 0.62500000f, 0.93750000f, 0.31250000f, 0.00000000f, 0.00000000f, 0.18750000f, 0.93750000f, 0.12500000f, 0.00000000f, 0.68750000f, 0.50000000f, 0.00000000f, 0.00000000f, 0.26666668f, 0.75000000f, 0.00000000f, 0.00000000f, 0.50000000f, 0.53333336f, 0.00000000f, 0.00000000f, 0.35714287f, 0.50000000f, 0.00000000f, 0.00000000f, 0.56250000f, 0.57142860f, 0.00000000f, 0.00000000f, 0.25000000f, 0.68750000f, 0.00000000f, 0.06250000f, 0.75000000f, 0.43750000f, 0.00000000f, 0.00000000f, 0.12500000f, 0.87500000f, 0.31250000f, 0.62500000f, 0.75000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.37500000f, 0.81250000f, 0.62500000f, 0.00000000f, 0.00000000f, 0.00000000f };

  int bmu = model.predict(x);

  // Expected: flat=182  row=9  col=2
  Serial.print("BMU flat index : "); Serial.println(bmu);
  Serial.print("BMU row        : "); Serial.println(model.bmu_row(bmu));
  Serial.print("BMU col        : "); Serial.println(model.bmu_col(bmu));
  Serial.print("Quant. error   : ");
  Serial.println(model.quantization_error(x, bmu), 6);
}

void loop() {
  // Nothing to do here.
}