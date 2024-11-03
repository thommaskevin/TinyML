#include "sample_image.h"

// #include "model_pico.h"
// #include "model_nano.h"
// #include "model_micro.h"
// #include "model_milli.h"
#include "model.h"

// PicoMobileNet net;
// NanoMobileNet net;
// MicroMobileNet net;
// MilliMobileNet net;
MobileNet net;

void setup()
{
    Serial.begin(9600);
}

void loop()
{
    size_t start = micros();
    net.predict(sample_image);

    Serial.print("Predicted output = ");
    Serial.print(net.output);
    Serial.println(" (Real value: 0) ");
    Serial.print("It took ");
    Serial.print(micros() - start);
    Serial.println(" us to inference.");
    delay(2000);
}