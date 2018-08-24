/*
 * Unit test for basic operations with Clock
 */

#include <iostream>

#include "Approximation.h"
#include "Clock.h"

int main (int argc, char** argv)
{
#ifndef DEBUG_INFO
  ALWAYS_ASSERT_MESSAGE ("Test requires debug info");
#endif /* !DEBUG_INFO */

  FPValue lambda = 0.04;

  FPValue stepAngle = PhysicsConst::Pi / 12;
  for (FPValue angle1 = FPValue (0); angle1 < 2 * PhysicsConst::Pi + stepAngle; angle1 += stepAngle)
  {
    for (FPValue angle2 = FPValue (0); angle2 < 2 * PhysicsConst::Pi + stepAngle; angle2 += stepAngle)
    {
      for (FPValue courantNum = FPValue (0.5); courantNum <= FPValue (1.0); courantNum += FPValue (0.25))
      {
        for (FPValue delta = FPValue (0.0002); delta <= FPValue (0.002); delta *= 2)
        {
          FPValue N_lambda = lambda / delta;
          ALWAYS_ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

          FPValue general_k = Approximation::approximateWaveNumberGeneral (delta, lambda, courantNum,
                                                                           N_lambda, angle1, angle2);
          FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda, angle1, angle2);

          // Accuracy is 1%
          FPValue accuracy = 2 * PhysicsConst::Pi / lambda * (1 / 100.0);
          ALWAYS_ASSERT (SQR (k - general_k) < accuracy);
        }
      }
    }
  }

  return 0;
} /* main */
