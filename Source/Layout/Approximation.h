#ifndef APPROXIMATION_H
#define APPROXIMATION_H

#include "FieldPoint.h"
#include "GridCoordinate3D.h"

#define APPROXIMATION_ACCURACY FPValue (0.0000001)

class Approximation
{
public:

  static CUDA_DEVICE CUDA_HOST FPValue getAccuracy ();

  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static CUDA_DEVICE CUDA_HOST FPValue getMaterial (const FieldPointValue *);

  static CUDA_DEVICE CUDA_HOST FPValue phaseVelocityIncidentWave (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateWaveNumber (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static CUDA_DEVICE CUDA_HOST FPValue approximateWaveNumberGeneral (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FieldValue approximateSphereFast (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue);

  static FieldValue approximateSphereAccurate (GridCoordinateFP1D, GridCoordinateFP1D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereAccurate (GridCoordinateFP2D, GridCoordinateFP2D, FPValue, FieldValue, FieldValue);
  static FieldValue approximateSphereAccurate (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue, FieldValue);
};

#endif /* APPROXIMATION_H */
