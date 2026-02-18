#include "gcn_model.h"

// Global array to store the model's predictions for all nodes.
// Allocated in global memory to avoid stack overflow issues.
int predictions[NUM_NODES];

void setup()
{
    // Initialize serial communication for debugging and reporting.
    Serial.begin(115200);
    delay(2000); // Waits for the serial connection to stabilize

    Serial.println("\n--- GNN Inference Report ---");
    // Displays the dimensions of the graph architecture.
    Serial.printf("Nodes: %d | Features: %d | Classes: %d\n\n", NUM_NODES, IN_FEATURES, OUT_CLASSES);

    // 1. Execute Inference
    // We measure the execution time (latency) of the forward pass.
    unsigned long start = micros();

    // Calls the C++ inference engine using the real feature matrix imported from Python.
    gnn_predict(GNN_REAL_X, predictions);

    unsigned long end = micros();

    // 2. Display Detailed Results
    int correct = 0;

    for (int i = 0; i < NUM_NODES; i++)
    {
        // A. Display Input Features (compacted)
        // Since there are 34 features using one-hot encoding (sparse vector),
        // we display only the index that is currently active (value approx. 1.0).
        int active_feature_index = -1;
        for (int f = 0; f < IN_FEATURES; f++)
        {
            // Check if the feature is active (floating-point comparison safety)
            if (GNN_REAL_X[i][f] > 0.9)
            {
                active_feature_index = f;
                break;
            }
        }

        // B. Compare Prediction vs. Ground Truth
        int real_label = GNN_REAL_Y[i];  // The actual class (from Python)
        int pred_label = predictions[i]; // The class predicted by the ESP32
        bool is_correct = (real_label == pred_label);

        if (is_correct)
            correct++;

        // C. Formatted Output
        // Prints the node ID, the input feature, the real class, and the predicted class.
        Serial.printf("Node[%02d] -> Input(Feature Index): %02d | Real: %d | Pred: %d | %s\n",
                      i, active_feature_index, real_label, pred_label, is_correct ? "OK" : "X");

        /* Optional: If you wish to view the full input vector (verbose), uncomment below:
           Serial.print("   Input Vector: [");
           for(int f=0; f<IN_FEATURES; f++) { Serial.printf("%.0f ", GNN_REAL_X[i][f]); }
           Serial.println("]");
        */
    }

    // 3. Final Summary
    // Calculates the overall accuracy percentage.
    float accuracy = (float)correct / NUM_NODES * 100.0;

    Serial.println("\n-----------------------------");
    // Converts microseconds to milliseconds for readability.
    Serial.printf("Inference Time: %.2f ms\n", (end - start) / 1000.0);
    Serial.printf("Final Accuracy: %.2f%%\n", accuracy);
    Serial.println("-----------------------------");
}

void loop()
{
    // The loop is kept empty as the inference is performed once during setup.
    delay(1000);
}