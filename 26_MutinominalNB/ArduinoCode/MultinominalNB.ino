#include "MultinomialNB.h"

float start_time = -1;
float end_time = -1;
float width_time = -1;
int len_vocabulary = 280;

// Function to get the index of the word in the vocabulary
int getWordIndex(const char *word) {
  for (int i = 0; i < len_vocabulary; i++) {
    if (strcmp(word, vocabulary[i]) == 0) {
      return i;
    }
  }
  return -1;  // Word not found
}

void setup() {
  // Start serial communication at 9600 baud
  Serial.begin(9600);
  while (!Serial)
    ;  // Wait for connection to the serial port
  Serial.println("Enter a message for classification (type 'exit' to stop):");
}

void loop() {
  static String input = "";             // String to store user input
  static bool processingInput = false;  // Flag to control input processing

  // Check if input is available in the serial buffer
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      // Process input when a new line is detected
      processInput(input);
      input = "";               // Clear input for the next message
      processingInput = false;  // Reset flag
    } else {
      // Add character to input
      input += c;
      processingInput = true;
    }
  }
}

void processInput(String input) {
  // Print the original input message
  Serial.print("Input: ");
  Serial.println(input);

  // Initialize scores
  float spam_score = 0.0;
  float ham_score = 0.0;

  // Create a mutable copy of the input to use with strtok
  char input_copy[100];
  input.toCharArray(input_copy, sizeof(input_copy));

  start_time = micros();

  // Tokenize the input string
  char *token = strtok(input_copy, " ");
  while (token != NULL) {
    int index = getWordIndex(token);
    if (index != -1) {
      // If the word is in the vocabulary, update the scores
      spam_score += abs(log_probs_spam[index]);
      ham_score += abs(log_probs_ham[index]);
    }
    token = strtok(NULL, " ");  // Get the next token
  }
  end_time = micros();

  width_time = end_time - start_time;

  // Classify the input based on the scores
  if (spam_score > ham_score) {
    Serial.print("Classification: ");
    Serial.println("Spam");
    Serial.print("Inference time: ");
    Serial.print(width_time);
    Serial.println(" us ");
  } else {
    Serial.print("Classification: ");
    Serial.println("Ham");
    Serial.print("Inference time: ");
    Serial.print(width_time);
    Serial.println(" us ");
  }

  // Prompt for the next input
  Serial.println("Enter another message for classification (or type 'exit' to stop):");
}
