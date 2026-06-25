/*
 * CTModel — Arduino verification sketch
 * Generated automatically — do not edit.
 *
 * VERIFICATION GUIDE
 * -------------------
 * Input n_features  : 5
 * Input values       : see float array below
 *
 * Expected class   : 2
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

    const int N_FEATURES = 5;
    float x[N_FEATURES] = {1.76405235f, 0.40015721f, 0.97873798f, 2.24089320f, 1.86755799f};

    // Expected class : 2
    int output = model.predict(x);
    Serial.print("Predicted class  : "); Serial.println(output);
}

void loop() {
    // Nothing to do here
}