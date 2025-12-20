#ifndef LDA_MODEL_H
#define LDA_MODEL_H

// ==========================================
// Auto-generated LDA Model Parameters
// Exported from Python Sklearn
// ==========================================

#define NUM_FEATURES 4
#define NUM_CLASSES 3

// Weights Matrix (Coefficients): [Classes][Features]
const float lda_weights[3][4] = {
    { 6.314758, 12.139317, -16.946425, -20.770055 },
    { -1.531199, -4.376043, 4.695665, 3.062585 },
    { -4.783559, -7.763274, 12.250759, 17.707469 },
};

// Bias Vector (Intercepts): [Classes]
const float lda_bias[3] = { -15.477837, -2.021974, -33.537687 };

// ==========================================
// Inference Function (Dot Product + Argmax)
// ==========================================

/**
 * Predicts the class based on sensor features.
 * @param features: Array of floats containing sensor data (size NUM_FEATURES)
 * @return int: The index of the predicted class (0 to NUM_CLASSES-1)
 */
int predictLDA(float *features) {
    int best_class = -1;
    // Initialize with the smallest possible float
    float best_score = -3.4028235E+38; 

    for (int c = 0; c < NUM_CLASSES; c++) {
        // Step 1: Start with the class Bias
        float score = lda_bias[c];

        // Step 2: Dot Product (Feature * Weight)
        for (int f = 0; f < NUM_FEATURES; f++) {
            score += features[f] * lda_weights[c][f];
        }

        // Step 3: Argmax (Keep the highest score)
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

#endif // LDA_MODEL_H
