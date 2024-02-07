namespace Eloquent
{
  namespace ML
  {
    namespace Port
    {
      class LogisticRegressor
      {
      public:
        float predict(float *x)
        {
          float probability = 0;
          float coefficients[4] = {0.37895195105851737, -0.864988943034597, 2.2039107306380346, 0.9146609339089985};

          float z = -6.01143820166656;
          for (int i = 0; i < 4; i++)
          {
            z += coefficients[i] * x[i];
          }

          probability = 1 / (1 + exp(-1 * z));
          if (probability >= 0.5)
          {
            return 1;
          }
          else
          {
            return 0;
          }
        }
      };
    }
  }
}
