#include <math.h>
using namespace std;

const int num_components = 7;
const int num_features = 2;

namespace TKSF
{
  namespace ML
  {
    namespace Port
    {
      class GMM
      {
      private:
        float means[num_components][num_features] = {
            {56.60931459453662, 49.92251390822188},
            {29.264921745717615, 74.5990614961554},
            {43.54207887008299, 10.496799818644392},
            {41.078202034837176, 34.79082042618642},
            {31.101406340255828, 90.2290512000496},
            {31.37324943091291, 61.64977286632278},
            {22.872325540804805, 49.948720505624074},
        };

        float covariances[num_components][num_features][num_features] = {
            {{83.24599787684558, 2.890435963609233}, {2.890435963609235, 35.68617486791098}},
            {{29.310245867234695, -3.6811694981292513}, {-3.68116949812925, 11.065658605910743}},
            {{176.1500923010437, 13.01774646138936}, {13.017746461389358, 29.65985065852867}},
            {{75.42208070910617, -6.070541531090654}, {-6.0705415310906545, 97.8692020437219}},
            {{29.297433369873897, 4.575820469751143}, {4.575820469751143, 23.020446452070658}},
            {{33.43445449031923, -13.501586578314814}, {-13.501586578314813, 6.152388053936477}},
            {{16.965821015319975, -2.507134562074277}, {-2.507134562074276, 45.30126223322478}},
        };

        float coefficients[num_components] = {0.1984245445224706, 0.14337213895448095, 0.1649683239019345, 0.1911025085379841, 0.1413696101261269, 0.036998614584665225, 0.12376425937233766};

        float component_pdf(float x[num_features], float mean[num_features], float covariance[num_features][num_features])
        {
          float det = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
          float inv_cov[num_features][num_features] = {{covariance[1][1] / det, -covariance[0][1] / det}, {-covariance[1][0] / det, covariance[0][0] / det}};
          float exponent = -0.5 * (inv_cov[0][0] * (x[0] - mean[0]) * (x[0] - mean[0]) + 2 * inv_cov[0][1] * (x[0] - mean[0]) * (x[1] - mean[1]) + inv_cov[1][1] * (x[1] - mean[1]) * (x[1] - mean[1]));
          float coefficient = 1.0 / sqrt(2 * M_PI * det);
          return coefficient * exp(exponent);
        }

      public:
        int predict(float x[num_features])
        {
          float probabilities[num_components] = {0};
          for (int i = 0; i < num_components; ++i)
          {
            probabilities[i] = coefficients[i] * component_pdf(x, means[i], covariances[i]);
          }
          int maxIndex = 0;
          for (int i = 1; i < num_components; ++i)
          {
            if (probabilities[i] > probabilities[maxIndex])
            {
              maxIndex = i;
            }
          }
          return maxIndex;
        }
      };
    }
  }
}
