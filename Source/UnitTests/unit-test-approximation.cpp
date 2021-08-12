/*
 * Copyright (C) 2018 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * Unit test for basic operations with Clock
 */

#include <iostream>

#include "Approximation.h"
#include "Clock.h"

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

int main (int argc, char** argv)
{
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
