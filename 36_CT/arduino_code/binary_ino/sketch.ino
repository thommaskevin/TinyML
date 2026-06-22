/*
 * CTModel — Arduino verification sketch
 * Generated automatically — do not edit.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input n_features  : 2
 * Input values       : see float array below
 *
 * Expected τ̂      : 0.13518519
 * Expected decision : TREAT
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

    const int N_FEATURES = 2;
    float x[N_FEATURES] = {1.76405235f, 0.40015721f};

    // Expected τ̂     : 0.13518519
    // Expected decision: TREAT
    float output = model.predict(x);
    Serial.print("Predicted tau    : "); Serial.println(output, 8);
    Serial.print("Decision         : "); Serial.println(output > 0.0f ? "TREAT" : "DO NOT TREAT");
}

void loop() {
    // Nothing to do here
}