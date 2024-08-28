#include <EEPROM.h>     // To store the Q-Table in non-volatile memory
#define ALPHA 0.5       // Learning rate
#define GAMMA 0.9       // Discount factor
#define EPSILON 0.5     // Probability of choosing a random action
#define NUM_STATES 4095 // Number of possible states
#define NUM_ACTIONS 5   // Number of possible actions

float qTable[NUM_STATES][NUM_ACTIONS];
int currentState = 0;
int action = 0;
float reward = 0;

void initializeQTable()
{
    for (int i = 0; i < NUM_STATES; i++)
    {
        for (int j = 0; j < NUM_ACTIONS; j++)
        {
            qTable[i][j] = 0.0;
        }
    }
}

int chooseAction(int state)
{
    if (random(0, 100) < (EPSILON * 100))
    {
        return random(0, NUM_ACTIONS); // Choose a random action
    }
    else
    {
        int maxAction = 0;
        float maxValue = qTable[state][0];
        for (int i = 1; i < NUM_ACTIONS; i++)
        {
            if (qTable[state][i] > maxValue)
            {
                maxAction = i;
                maxValue = qTable[state][i];
            }
        }
        return maxAction; // Choose the best action
    }
}

void updateQTable(int state, int action, float reward, int nextState)
{
    float oldQValue = qTable[state][action];
    float maxNextQValue = qTable[nextState][0];
    for (int i = 1; i < NUM_ACTIONS; i++)
    {
        if (qTable[nextState][i] > maxNextQValue)
        {
            maxNextQValue = qTable[nextState][i];
        }
    }
    qTable[state][action] = oldQValue + ALPHA * (reward + GAMMA * maxNextQValue - oldQValue);
}

int readSensor()
{
    int sensorValue = analogRead(34); // Assuming the MQ-135 is connected to pin 34
    // Convert the read value to a discrete state
    return map(sensorValue, 0, 4095, 0, NUM_STATES - 1);
}

void setup()
{
    Serial.begin(115200);
    initializeQTable();
    pinMode(34, INPUT);
}

void loop()
{
    currentState = readSensor();
    action = chooseAction(currentState);

    // Perform the action (in this example, we won't do anything specific)
    delay(1000); // Wait one second between readings

    int nextState = readSensor();

    // Set the reward (in this example, we'll use a fictitious reward)
    reward = random(0, 10) / 10.0;

    updateQTable(currentState, action, reward, nextState);

    Serial.print("Current State: ");
    Serial.print(currentState);
    Serial.print(" | Action: ");
    Serial.print(action);
    Serial.print(" | Reward: ");
    Serial.print(reward);
    Serial.print(" | Next State: ");
    Serial.println(nextState);

    delay(1000);
}