/*
 * CTModel — Arduino verification sketch
 * Generated automatically — do not edit.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input n_features  : 1
 * Input values       : see float array below
 *
 * Expected τ̂      : 1.82389114
 *
 * Upload this sketch, open Serial Monitor at 115200 baud,
 * and confirm the printed value matches the expected value
 * above to at least 5 decimal places.
 */

#include "CTModel.h"

CTModel model;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    const int N_FEATURES = 1;
    float x[N_FEATURES] = {1.76405235f};

    // Expected τ̂     : 1.82389114
    float output = model.predict(x);
    Serial.print("Predicted tau    : "); Serial.println(output, 8);
}

void loop() {
    // Nothing to do here
}