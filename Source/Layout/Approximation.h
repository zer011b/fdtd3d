#ifndef APPROXIMATION_H
#define APPROXIMATION_H

#include "FieldValue.h"

class Approximation
{
public:

  static FPValue approximateMaterial (FPValue, FPValue);
  static FPValue approximateMaterial (FPValue, FPValue, FPValue, FPValue);

  static FPValue approximateMetaMaterial (FPValue, FPValue);

  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue);
  static void approximateDrudeModel (FPValue &, FPValue &, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue, FPValue);
};

#endif /* APPROXIMATION_H */
