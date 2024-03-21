#include "GMM.h"

TKSF::ML::Port::GMM GMM;


void setup() {
  Serial.begin(9600);
  

}

void loop() {
  // Input value
  float input[] = {19, 39}; 
  // Predict
  int cluster = GMM.predict(input);
  Serial.print("Cluster member (real is 6): ");
  Serial.println(cluster);

  delay(2000);
}
