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

  static FPValue approximateMetaMaterial (FPValue, FPValue);

  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue,
                                     FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FPValue getMaterial (const FieldPointValue *);

  static FPValue phaseVelocityIncidentWave3D (FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
  static FPValue phaseVelocityIncidentWave2D (FPValue, FPValue, FPValue, FPValue, FPValue);

  static FieldValue approximateSphere (GridCoordinateFP3D, GridCoordinateFP3D, FPValue, FieldValue);
};

#endif /* APPROXIMATION_H */
