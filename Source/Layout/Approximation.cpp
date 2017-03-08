#include "Approximation.h"
#include "Assert.h"

#include <cmath>

FPValue
Approximation::approximateMaterial (FPValue val1, FPValue val2)
{
  return (val1 + val2) / 2.0;
}

FPValue
Approximation::approximateMaterial (FPValue val1, FPValue val2, FPValue val3, FPValue val4)
{
  return (val1 + val2 + val3 + val4) / 4.0;
}

FPValue
Approximation::approximateMetaMaterial (FPValue val1, FPValue val2)
{
  return (val1 + val2) / sqrt (2.0);
}

void
Approximation::approximateDrudeModel (FPValue &omega,
                                      FPValue &gamma,
                                      FPValue permittivity1,
                                      FPValue permittivity2,
                                      FPValue permittivity3,
                                      FPValue permittivity4,
                                      FPValue omega1,
                                      FPValue omega2,
                                      FPValue omega3,
                                      FPValue omega4,
                                      FPValue gamma1,
                                      FPValue gamma2,
                                      FPValue gamma3,
                                      FPValue gamma4)
{
  /*
   * FIXME: incorrect!
   */
  FPValue dividerOmega = 0;
  FPValue dividerGamma = 0;

  ASSERT (permittivity1 == 1 && permittivity2 == 1 && permittivity3 == 1 && permittivity4 == 1);

  if (omega2 == omega3 == omega4 == 0 && gamma2 == gamma3 == gamma4 == 0
      || omega1 == omega3 == omega4 == 0 && gamma1 == gamma3 == gamma4 == 0
      || omega1 == omega2 == omega4 == 0 && gamma1 == gamma2 == gamma4 == 0
      || omega1 == omega2 == omega3 == 0 && gamma1 == gamma2 == gamma3 == 0)
  {
    dividerOmega = 2.0;
    dividerGamma = 1.0;
  }
  else if (omega1 == omega2 == 0 && omega3 == omega4 && gamma1 == gamma2 == 0 && gamma3 == gamma4
           || omega2 == omega4 == 0 && omega1 == omega3 && gamma2 == gamma4 == 0 && gamma1 == gamma3
           || omega3 == omega4 == 0 && omega1 == omega2 && gamma3 == gamma4 == 0 && gamma1 == gamma2
           || omega1 == omega3 == 0 && omega2 == omega4 && gamma1 == gamma3 == 0 && gamma2 == gamma4)
  {
    dividerOmega = sqrt (2.0);
    dividerGamma = 2.0;
  }
  else if (omega1 == 0 && omega2 == omega3 == omega4 && gamma1 == 0 && gamma2 == gamma3 == gamma4
           || omega2 == 0 && omega1 == omega3 == omega4 && gamma2 == 0 && gamma1 == gamma3 == gamma4
           || omega3 == 0 && omega1 == omega2 == omega4 && gamma3 == 0 && gamma1 == gamma2 == gamma4
           || omega4 == 0 && omega1 == omega2 == omega3 && gamma4 == 0 && gamma1 == gamma2 == gamma3)
  {
    dividerOmega = 2 / sqrt (3.0);
    dividerGamma = 3.0;
  }
  else
  {
    ASSERT (omega1 == omega2 == omega3 == omega4
            && gamma1 == gamma2 == gamma3 == gamma4);

    dividerOmega = 4.0;
    dividerOmega = 4.0;
  }

  ASSERT (dividerOmega != 0);
  ASSERT (dividerGamma != 0);

  omega = (omega1 + omega2 + omega3 + omega4) / dividerOmega;
  gamma = (gamma1 + gamma2 + gamma3 + gamma4) / dividerGamma;
}

void
Approximation::approximateDrudeModel (FPValue &omega,
                                      FPValue &gamma,
                                      FPValue permittivity1,
                                      FPValue permittivity2,
                                      FPValue omega1,
                                      FPValue omega2,
                                      FPValue gamma1,
                                      FPValue gamma2)
{
  /*
   * FIXME: incorrect!
   */
  FPValue dividerOmega = 0;
  FPValue dividerGamma = 0;

  ASSERT (permittivity1 == 1 && permittivity2 == 1);

  if (omega1 == 0 && gamma1 == 0
      || omega2 == 0 && gamma2 == 0)
  {
    dividerOmega = sqrtf (2.0);
    dividerGamma = 1.0;
  }
  else
  {
    ASSERT (omega1 == omega2 && gamma1 == gamma2);

    dividerOmega = 2.0;
    dividerGamma = 2.0;
  }

  ASSERT (dividerOmega != 0);
  ASSERT (dividerGamma != 0);

  omega = (omega1 + omega2) / dividerOmega;
  gamma = (gamma1 + gamma2) / dividerGamma;
}

FPValue
Approximation::getMaterial (const FieldPointValue *val)
{
#ifdef COMPLEX_FIELD_VALUES
  return val->getCurValue ().real ();
#else /* COMPLEX_FIELD_VALUES */
  return val->getCurValue ();
#endif /* !COMPLEX_FIELD_VALUES */
}
