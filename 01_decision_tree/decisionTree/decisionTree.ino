int predict(float *x) {
                        if (x[2] <= 2.449999988079071) {
                            return 0;
                        }

                        else {
                            if (x[3] <= 1.6500000357627869) {
                                if (x[2] <= 4.950000047683716) {
                                    return 1;
                                }

                                else {
                                    if (x[0] <= 6.150000095367432) {
                                        if (x[1] <= 2.450000047683716) {
                                            return 2;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }

                                    else {
                                        return 2;
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 4.8500001430511475) {
                                    if (x[1] <= 3.100000023841858) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    return 2;
                                }
                            }
                        }
                    };

void setup() {
  Serial.begin(115200);

}

void loop() {
  float X_1[] = {5.1, 3.5, 1.4, 0.2};
  int result_1 = predict(X_1);
  Serial.print("Result of predict with input X1:");
  Serial.println(result_1);
  delay(2000);

  float X_2[] = {6.2, 2.2, 4.5, 1.5};
  int result_2 = predict(X_2);
  Serial.print("Result of predict with input X2:");
  Serial.println(result_2); 
  delay(2000);

  float X_3[] = {6.1, 3.0, 4.9, 1.8};
  int result_3 = predict(X_3);
  Serial.print("Result of predict with input X3:");
  Serial.println(result_3);
  delay(2000);
}
