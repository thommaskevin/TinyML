#include "nam_multiclass.h"

void setup()
{
    Serial.begin(115200);
    delay(2000);

    Serial.println("Running NAM Multiclass model...");

    // Features de entrada
    float features[2] = {0.22793516f, 0.98411032f};

    // Vetor de saída (3 classes)
    float probs[3];

    // Predição
    predict(features, probs);

    Serial.println("Class probabilities:");

    Serial.print("Class 0: ");
    Serial.println(probs[0], 7);

    Serial.print("Class 1: ");
    Serial.println(probs[1], 7);

    Serial.print("Class 2: ");
    Serial.println(probs[2], 7);

    // Encontrar classe com maior probabilidade
    int predicted_class = 0;
    float max_prob = probs[0];

    for (int i = 1; i < 3; i++)
    {
        if (probs[i] > max_prob)
        {
            max_prob = probs[i];
            predicted_class = i;
        }
    }

    Serial.print("Predicted class: ");
    Serial.println(predicted_class);
}

void loop()
{
}