#include "RbfRegressionModelRBF.h"

RbfRegressionModelRBF model;
rbf_float inputs[RBF_INPUT_DIM];
rbf_float outputs[RBF_OUTPUT_DIM];

void setup() {
    Serial.begin(115200);
    while(!Serial);
    model.printInfo();
    Serial.println(F("Send 't' to test."));
}

void loop() {
    if(Serial.read() == 't') {
        // Generate random input
        for(int i=0; i<RBF_INPUT_DIM; i++) {
            #if RBF_FIXED_POINT
                inputs[i] = random(-100, 100) * (RBF_SCALE_FACTOR / 100); 
            #else
                inputs[i] = random(-200, 200) / 100.0f; // Range -2.0 to 2.0
            #endif
            Serial.print(F("In: ")); Serial.println(inputs[i]);
        }

        unsigned long t0 = micros();
        model.predict(inputs, outputs);
        unsigned long dt = micros() - t0;

        for(int i=0; i<RBF_OUTPUT_DIM; i++) {
            Serial.print(F("Out: ")); Serial.println(outputs[i]);
        }
        Serial.print(F("Time (us): ")); Serial.println(dt);
    }
}
