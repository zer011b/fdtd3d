#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include "FieldValue.h"

namespace PhysicsConst
{
  const CUDA_DEVICE FieldValue SpeedOfLight = 2.99792458e+8;
  const CUDA_DEVICE FieldValue Eps0 = 0.0000000000088541878176203892;
  const CUDA_DEVICE FieldValue Mu0 = 0.0000012566370614359173;
  const CUDA_DEVICE FieldValue accuracy = 0.001;
  const CUDA_DEVICE FieldValue Pi = 3.141592653589793238462643;
};

#endif /* PHYSICS_CONSTANTS_H */
