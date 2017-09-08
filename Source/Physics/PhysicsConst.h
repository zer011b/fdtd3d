#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include "FieldValue.h"
#include <cmath>

namespace PhysicsConst
{
  const CUDA_DEVICE FPValue Pi = M_PI;
  const CUDA_DEVICE FPValue SpeedOfLight = 299792458;
  const CUDA_DEVICE FPValue Mu0 = 4 * Pi * 0.0000001;
  const CUDA_DEVICE FPValue Eps0 = 1 / (Mu0 * SQR (SpeedOfLight));
  const CUDA_DEVICE FPValue accuracy = 0.001;
};

#endif /* PHYSICS_CONSTANTS_H */
