#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

//#include "model_pruned_L1.h"
//#include "model_pruned_interative.h"
//#include "model_original.h"
#include "model_random.h"


#define N_INPUTS 64 
#define N_OUTPUTS 10
// in future projects you may need to tweak this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 36*1024

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;


float start_time = -1;
float end_time = -1;
float width_time = -1;

float input[64] = {0.00000000000f, 0.12500000000f, 0.00000000000f, 0.50000000000f, 0.56250000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.81250000000f, 0.31250000000f, 0.87500000000f, 0.50000000000f, 0.43750000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.75000000000f, 0.31250000000f, 0.12500000000f, 0.00000000000f, 0.56250000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.43750000000f, 0.31250000000f, 0.00000000000f, 0.00000000000f, 0.18750000000f, 0.31250000000f, 0.00000000000f, 0.00000000000f, 0.18750000000f, 0.62500000000f, 0.00000000000f, 0.00000000000f, 0.12500000000f, 0.62500000000f, 0.00000000000f, 0.00000000000f, 0.06250000000f, 0.81250000000f, 0.00000000000f, 0.00000000000f, 0.06250000000f, 0.75000000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.31250000000f, 0.81250000000f, 0.31250000000f, 0.56250000000f, 0.81250000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.00000000000f, 0.56250000000f, 1.00000000000f, 1.00000000000f, 0.43750000000f, 0.00000000000f};

float y_pred[10] = {0};

void setup() {
    Serial.begin(9600);
    delay(4000);
    tf.begin(model);

    // check if model loaded fine
    if (!tf.isOk()) {
      Serial.print("ERROR: ");
      Serial.println(tf.getErrorMessage());

      while (true) delay(1000);
    }
}

void loop() {

      
      
        start_time = millis() ;
        //start_time = micros();
        tf.predict(input, y_pred);
        end_time = millis();
        //end_time = micros();
        for (int i = 0; i < 10; i++) {
            Serial.print(y_pred[i]);
            Serial.print(i == 9 ? '\n' : ',');
        }
      Serial.print("Predicted class is: ");
      Serial.println(tf.probaToClass(y_pred));
      // or you can skip the predict() method and call directly predictClass()
      Serial.print("Sanity check: ");
      Serial.println(tf.predictClass(input));
      Serial.print(" - Time (ms): ");
      width_time = end_time - start_time;
      Serial.println(width_time);
      delay(2000);

}