#include "KAN_MODEL.h"

float start_time = -1;
float end_time = -1;
float width_time = -1;

double x_1 = -0.0075;
double x_2 = 0.5547;
double x_3 = -0.0075;
double x_4 = 0.5547;
double x_5 = -0.0075;
double x_6 = 0.5547;
double x_7 = 0.5547;

void setup()
{
    Serial.begin(9600);
}

void loop()
{
    start_time = micros();
    double y_pred = predict(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
    end_time = micros();

    width_time = end_time - start_time;

    Serial.print("Predicted value (Python Predict: 0.6750, Real: 1.1605): ");
    Serial.println(y_pred);

    Serial.print("Execution time (microseconds): ");
    Serial.println(width_time);

    delay(2000);
}
