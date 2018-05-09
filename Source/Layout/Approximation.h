#ifndef APPROXIMATION_H
#define APPROXIMATION_H

#include "FieldPoint.h"
#include "GridCoordinate3D.h"

class Approximation
{
  static FPValue accuracy;

public:

  static FPValue getAccuracy ();

  static FPValue approximateMaterial (FPValue, FPValue);
  static FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue);
  static FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FPValue getMaterial (const FieldPointValue *);

  static FPValue phaseVelocityIncidentWave (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FieldValue approximateSphereFast (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue);
  static FieldValue approximateSphereAccurate (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue);
};

#endif /* APPROXIMATION_H */
