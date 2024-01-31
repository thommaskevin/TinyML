#include "PCA.h"

Eloquent::ML::Port::PCA pca;



void setup() {
  Serial.begin(115200);

}

void loop() {
  float X_1[9] = {11.04,  7.58, 14.83,  2.07, 49.81, 14.69, 43.75,  5.02, 63.19};
  float result_1[2];
  pca.transform(X_1, result_1);
  Serial.print("Result of predict with input X1:");
  for (int i = 0; i < 2; i++) {
    Serial.print(" ");
    Serial.print(result_1[i]);
  }
  Serial.println();  // Adiciona uma nova linha no final
  delay(2000);

  float X_2[9] = {10.76,  7.4 , 14.26,  1.86, 49.37, 14.05, 50.72,  4.92, 60.15};
  float result_2[2];
  pca.transform(X_2,  result_2); 
  Serial.print("Result of predict with input X2:");
  for (int i = 0; i < 2; i++) {
    Serial.print(" ");
    Serial.print(result_2[i]);
  }
  Serial.println();  // Adiciona uma nova linha no final
  delay(2000);

}
