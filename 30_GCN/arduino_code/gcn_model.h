
#ifndef GNN_MODEL_H
#define GNN_MODEL_H

#define NUM_NODES 34
#define IN_FEATURES 34
#define HIDDEN_DIM 4
#define OUT_CLASSES 4

extern const float GNN_W1[HIDDEN_DIM][IN_FEATURES];
extern const float GNN_B1[HIDDEN_DIM];
extern const float GNN_W2[OUT_CLASSES][HIDDEN_DIM];
extern const float GNN_B2[OUT_CLASSES];
extern const float GNN_ADJ[NUM_NODES][NUM_NODES];

// Validation Data
extern const float GNN_REAL_X[NUM_NODES][IN_FEATURES];
extern const int GNN_REAL_Y[NUM_NODES];

void gnn_predict(const float input_features[NUM_NODES][IN_FEATURES], int output_classes[NUM_NODES]);

#endif
