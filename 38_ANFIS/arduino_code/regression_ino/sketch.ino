/*
 * ANFIS Model -- Arduino verification sketch
 * Generated automatically -- do not edit the weights.
 *
 * Input  size     : 2
 * Terms per input : 5
 * Total rules     : 25
 * MF type         : gaussian
 *
 * Input values    : [1.76405239, 0.40015721]
 * Expected output : 0.89409703
 *
 * Upload this sketch, open Serial Monitor at 115200 baud,
 * and confirm the printed value matches the expected value
 * above to at least 4 decimal places.
 */

#include "ANFISModel.h"

ANFISModel model;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    const int N_INPUTS = 2;

    float input[N_INPUTS] = {
        1.76405239f, 0.40015721f
    };

    float output = model.predict(input);

    // Expected value  : 0.89409703
    Serial.print("Predicted value  : "); Serial.println(output, 8);
}

void loop() {
    // Nothing to do here
}