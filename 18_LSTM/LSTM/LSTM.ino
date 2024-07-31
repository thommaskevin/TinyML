#include <Arduino.h>
// replace with your own model
#include "model.h"
// include the runtime specific for your board
// either tflm_esp32 or tflm_cortexm
#include <tflm_esp32.h>
// now you can include the eloquent tinyml wrapper
#include <eloquent_tinyml.h>

// this is trial-and-error process
// when developing a new model, start with a high value
// (e.g. 10000), then decrease until the model stops
// working as expected
#define ARENA_SIZE 30000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

float X_1[3] = {4.6, 8., 304.};
float X_2[3] = {2., 4., 216.};

void predictSample(float *input, float expectedOutput)
{

    while (!tf.begin(tfModel).isOk())
    {
        Serial.println(tf.exception.toString());
        delay(1000);
    }

    // classify class 0
    if (!tf.predict(input).isOk())
    {
        Serial.println(tf.exception.toString());
        return;
    }
    Serial.print("Expcted = ");
    Serial.print(expectedOutput);
    Serial.print(", predicted = ");
    Serial.println(tf.outputs[0]);
}

void setup()
{
    Serial.begin(115200);
    delay(3000);
    Serial.println("__TENSORFLOW LSTM__");

    // configure input/output
    // (not mandatory if you generated the .h model
    // using the eloquent_tensorflow Python package)
    tf.setNumInputs(TF_NUM_INPUTS);
    tf.setNumOutputs(TF_NUM_OUTPUTS);

    registerNetworkOps(tf);
}

void loop()
{
    /**
     * Run prediction
     */

    predictSample(X_1, 17.76);
    delay(2000);

    predictSample(X_2, 11.44);
    delay(2000);
}