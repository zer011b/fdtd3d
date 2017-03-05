#ifndef APPROXIMATION_H
#define APPROXIMATION_H

#include "FieldPoint.h"

class Approximation
{
public:

  static FPValue approximateMaterial (FPValue, FPValue);
  static FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue);

  static FPValue approximateMetaMaterial (FPValue, FPValue);

  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue);
  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);

  static FPValue getMaterial (const FieldPointValue *);
};

#endif /* APPROXIMATION_H */
