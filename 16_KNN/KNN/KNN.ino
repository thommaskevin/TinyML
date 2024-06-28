// Include the generated header file
#include "KNN.h"

KNeighborsClassifier knn;

float X_test[1][30] = {
    {12.47, 18.6, 81.09, 481.9, 0.09965, 0.1058, 0.08005, 0.03821, 0.1925, 0.06373,
     0.3961, 1.044, 2.497, 30.29, 0.006953, 0.01911, 0.02701, 0.01037, 0.01782,
     0.003586, 14.97, 24.64, 96.05, 677.9, 0.1426, 0.2378, 0.2671, 0.1015, 0.3014,
     0.0875}};

float X_test_2[1][30] = {
    {1.546e+01, 1.948e+01, 1.017e+02, 7.489e+02, 1.092e-01, 1.223e-01,
     1.466e-01, 8.087e-02, 1.931e-01, 5.796e-02, 4.743e-01, 7.859e-01,
     3.094e+00, 4.831e+01, 6.240e-03, 1.484e-02, 2.813e-02, 1.093e-02,
     1.397e-02, 2.461e-03, 1.926e+01, 2.600e+01, 1.249e+02, 1.156e+03,
     1.546e-01, 2.394e-01, 3.791e-01, 1.514e-01, 2.837e-01, 8.019e-02}};

void setup()
{
    Serial.begin(9600);
    // Initializing the KNN model
    Serial.println("KNN model initialized.");

    // Fitting the model with training data
    knn.fit(X_train, y_train, 10, 30);
    Serial.println("Model fitted with training data.");
}

void loop()
{
    // Example test with X_test
    int *predictions = knn.predict(X_test, 1);

    // Printing predictions
    Serial.print("Predicted class (Real = 0): ");
    Serial.println(predictions[0]);

    predictions = knn.predict(X_test_2, 1);
    Serial.print("Predicted class (Real = 1): ");
    Serial.println(predictions[0]);

    // Delay before next iteration (adjust as needed)
    delay(5000); // 5 seconds delay

    // Clean up memory
    delete[] predictions;
}
