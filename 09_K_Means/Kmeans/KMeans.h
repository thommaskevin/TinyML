#include <math.h> 
 
 
namespace Eloquent {
namespace ML {
namespace Port {
class KMeans {
private: 

float centroids[7][2] = {{45.65789473684211, 44.42105263157895}, 
{29.249999999999996, 72.19444444444443}, 
{30.55555555555556, 13.05555555555555}, 
{24.033333333333328, 48.43333333333333}, 
{31.285714285714292, 90.39285714285712}, 
{51.035714285714285, 14.999999999999986}, 
{63.9090909090909, 50.63636363636363}};



public: 

int predict(float *x) { 
float euclidean_distance = 0;
 float euclidean_distance_old = 999999999;
 int cluster_member = -1;
 for (int index = 0; index < 7; ++index) { 
float error_square = 0; 
for (int index_value = 0; index_value < 2; ++index_value) { 
     error_square += pow(centroids[index][index_value] - x[index_value], 2);} 
euclidean_distance = sqrt(error_square); 
if (euclidean_distance < euclidean_distance_old) { 
     euclidean_distance_old = euclidean_distance;
     cluster_member = index;
     }
     }
 return cluster_member;
} 
};
}
}
}
