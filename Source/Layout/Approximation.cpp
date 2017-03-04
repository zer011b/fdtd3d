#include "Approximation.h"

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
  FPValue dividerOmega = 4;
  FPValue dividerGamma = 4;

  omega = (omega1 + omega2 + omega3 + omega4) / dividerOmega;
  gamma = (gamma1 + gamma2 + gamma3 + gamma4) / dividerGamma;
}

void
Approximation::approximateDrudeModel (FPValue &omega,
                                      FPValue &gamma,
                                      FPValue omega1,
                                      FPValue omega2,
                                      FPValue gamma1,
                                      FPValue gamma2)
{
  /*
   * FIXME: incorrect!
   */
  FPValue dividerOmega = 2;
  FPValue dividerGamma = 2;

  /*
  FPValue dividerOmega = 0;
  FPValue dividerGamma = 0;

  if (omega1 == 0 || omega2 == 0)
  {
    dividerOmega = sqrtf (2.0);
    dividerGamma = 2.0;
  }
  else
  {
    if (omega1 != omega2 || gamma1 != gamma2)
    {
      ASSERT_MESSAGE ("Unimplemented metamaterials border condition");
    }

    dividerOmega = 2.0;
    dividerGamma = 2.0;
  }

  ASSERT (dividerOmega != 0);
  ASSERT (dividerGamma != 0);*/

  omega = (omega1 + omega2) / dividerOmega;
  gamma = (gamma1 + gamma2) / dividerGamma;
}
