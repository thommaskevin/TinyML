#include "lda_model.h"

// Variable to store sensor readings
// Ensure the size matches NUM_FEATURES defined in the .h file
float input_sensors[NUM_FEATURES];

void setup() {
  Serial.begin(9600);
  
  // Example: Simulating data for "Setosa" class
  // In a real project, replace these with analogRead() or sensor library calls
  input_sensors[0] = 5.1;
  input_sensors[1] = 3.5;
  input_sensors[2] = 1.4;
  input_sensors[3] = 0.2;
  
  Serial.println("System Ready. Starting inference...");
}

void loop() {
  // 1. Read Sensors (Update input_sensors array here)
  // ...
  
  // 2. Run Inference
  // Just call the function included from the header
  int predicted_class = predictLDA(input_sensors);
  
  // 3. Output Result
  Serial.print("Predicted Class ID: ");
  Serial.println(predicted_class);
  
  delay(1000);
}