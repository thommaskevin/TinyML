#include "KMeans.h"

Eloquent::ML::Port::KMeans k_means;


void setup() {
  Serial.begin(9600);
  

}

void loop() {
  // Input value
  float input[] = {15, 39}; 
  // Predict
  int cluster = k_means.predict(input);
  // Exiba o cluster encontrado
  Serial.print("Cluster member (real is 2): ");
  Serial.println(cluster);

  delay(2000);
}
