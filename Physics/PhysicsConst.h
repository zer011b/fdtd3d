#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include "FieldPoint.h"

#ifdef __CUDACC__
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

namespace PhysicsConst
{
  const __DEVICE__ FieldValue SpeedOfLight = 2.99792458e+8;
  const __DEVICE__ FieldValue Eps0 = 0.0000000000088541878176203892;
  const __DEVICE__ FieldValue Mu0 = 0.0000012566370614359173;
  const __DEVICE__ FieldValue accuracy = 0.001;
  const __DEVICE__ FieldValue Pi = 3.141592653589793238462643;
};

#endif /* PHYSICS_CONSTANTS_H */
