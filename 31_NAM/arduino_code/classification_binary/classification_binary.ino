#include "nam_binary.h"

void setup()
{
    Serial.begin(115200);
    delay(2000);

    Serial.println("Running NAM model...");

    // Features de entrada
    float features[2] = {0.07118865, 0.67296662};

    // Predição
    float pred = predict(features);

    Serial.print("Predict: ");
    Serial.print(pred, 7); // 7 casas decimais

    Serial.print(" | Real: ");
    Serial.print(1.0);

    Serial.print(" | Python Predict: ");
    Serial.println(0.9457689, 7);
}

void loop()
{
}