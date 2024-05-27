#include "model.h"
#include <FS.h>
#include <SPIFFS.h>
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#define NUMBER_OF_INPUTS 3
#define NUMBER_OF_OUTPUTS 1

// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 8*1024

double start_time = -1;
double end_time = -1;
double width_time = -1;



Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

uint8_t *loadedModel;

void setup() {
    Serial.begin(115200);
    SPIFFS.begin(true);
    delay(3000);
    storeModel();
    loadModel();
    if (!ml.begin(loadedModel))
    {
    Serial.println("Cannot inialize model");
    Serial.println(ml.getErrorMessage());
    delay(60000);
    }
    delay(4000);
}

void loop() {      
    
     float input[3] = {0.51428571, 0.55555556, 0.41841004};

      //start_time = millis();
      start_time = micros();
      float predicted  = ml.predict(input);
      //end_time = millis();
      end_time = micros();
      
      width_time = end_time - start_time;

      Serial.print("Predict: ");
      Serial.println(predicted);
      Serial.print("Real: ");
      Serial.println(15.4);
      Serial.print("Processing time: ");
      Serial.println(width_time);
      Serial.println(" ");
      delay(500);
      
    }


void storeModel() {
  File file = SPIFFS.open("/sine.bin", "wb");
  file.write(model, model_len);
  file.close();
}


/**
 * Load model from SPIFFS
 */
void loadModel() {
  File file = SPIFFS.open("/sine.bin", "rb");
  size_t modelSize = file.size();

  Serial.print("Found model on filesystem of size ");
  Serial.print(modelSize);
  Serial.print(": it should be ");
  Serial.println(model_len);

  // allocate memory
  loadedModel = (uint8_t*) malloc(modelSize);

  // copy data from file
  for (size_t i = 0; i < modelSize; i++)
    loadedModel[i] = file.read();
  
  file.close();
}
