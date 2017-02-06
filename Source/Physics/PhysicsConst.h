#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include "FieldValue.h"
#include <cmath>

namespace PhysicsConst
{
  const CUDA_DEVICE FPValue SpeedOfLight = 2.99792458e+8;
  const CUDA_DEVICE FPValue Eps0 = 0.0000000000088541878176203892;
  const CUDA_DEVICE FPValue Mu0 = 0.0000012566370614359173;
  const CUDA_DEVICE FPValue accuracy = 0.001;
  const CUDA_DEVICE FPValue Pi = M_PI;
};

#endif /* PHYSICS_CONSTANTS_H */
