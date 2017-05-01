#include "Approximation.h"
#include "Assert.h"
#include "PhysicsConst.h"

#include <cmath>

FPValue Approximation::accuracy = 0.0000001;

FPValue
Approximation::getAccuracy ()
{
  return accuracy;
}

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

FPValue
Approximation::phaseVelocityIncidentWave3D (FPValue delta,
                                            FPValue freeSpaceWaveLentgh,
                                            FPValue courantNum,
                                            FPValue N_lambda,
                                            FPValue incidentWaveAngle1,
                                            FPValue incidentWaveAngle2)
{
  if (incidentWaveAngle1 = PhysicsConst::Pi / 2
      && (incidentWaveAngle2 == 0
          || incidentWaveAngle2 == PhysicsConst::Pi / 2
          || incidentWaveAngle2 == PhysicsConst::Pi
          || incidentWaveAngle2 == 3 * PhysicsConst::Pi / 2))
  {
    /*
     * Special case of propagation along some axes
     */
    return PhysicsConst::SpeedOfLight * PhysicsConst::Pi /
           (N_lambda * asin (sin (PhysicsConst::Pi * courantNum / N_lambda) / courantNum));
  }

  if (incidentWaveAngle1 = PhysicsConst::Pi / 2
      && (incidentWaveAngle2 == PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 3 * PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 5 * PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 7 * PhysicsConst::Pi / 4))
  {
    /*
     * Special case of propagation at angle of Pi/4
     */
    return PhysicsConst::SpeedOfLight * PhysicsConst::Pi /
           (N_lambda * sqrt(2.0) * asin (sin (PhysicsConst::Pi * courantNum / N_lambda) / (courantNum * sqrt(2.0))));
  }


  FPValue k = 2 * PhysicsConst::Pi;
  FPValue k_prev = k + Approximation::getAccuracy ();

  FPValue normalized_delta = delta / freeSpaceWaveLentgh;

  FPValue A = normalized_delta * sin (incidentWaveAngle1) * cos (incidentWaveAngle2) / 2;
  FPValue B = normalized_delta * sin (incidentWaveAngle1) * sin (incidentWaveAngle2) / 2;
  FPValue C = normalized_delta * cos (incidentWaveAngle1) / 2;
  FPValue D = SQR (sin (PhysicsConst::Pi * courantNum / N_lambda)) / SQR (courantNum);

  while (SQR (k_prev - k) >= Approximation::getAccuracy ())
  {
    k_prev = k;

    FPValue diff1 = SQR (sin (A * k)) + SQR (sin (B * k)) + SQR (sin (C * k)) - D;
    FPValue diff2 = A * sin (2 * A * k) + B * sin (2 * B * k) + C * sin (2 * C * k);
    FPValue diff = diff1 / diff2;

    k -= diff;
  }

  return PhysicsConst::SpeedOfLight * 2 * PhysicsConst::Pi / k;
}

FPValue
Approximation::phaseVelocityIncidentWave2D (FPValue delta,
                                            FPValue freeSpaceWaveLentgh,
                                            FPValue courantNum,
                                            FPValue N_lambda,
                                            FPValue incidentWaveAngle2)
{
  return phaseVelocityIncidentWave3D (delta,
                                      freeSpaceWaveLentgh,
                                      courantNum,
                                      N_lambda,
                                      PhysicsConst::Pi / 2,
                                      incidentWaveAngle2);
}

FieldValue
Approximation::approximateSphere (GridCoordinateFP3D midPos,
                                  GridCoordinateFP3D center,
                                  FPValue radius,
                                  FieldValue eps)
{
  FPValue d = sqrt (SQR (midPos.getX () - center.getX ()) + SQR (midPos.getY () - center.getY ()) + SQR (midPos.getZ () - center.getZ ()));

  FPValue diff = d - radius;

#ifdef COMPLEX_FIELD_VALUES
  FieldValue eps_vacuum (1, 0);
#else /* COMPLEX_FIELD_VALUES */
  FieldValue eps_vacuum (1);
#endif /* !COMPLEX_FIELD_VALUES */

  if (diff < -0.5)
  {
    return eps;
  }
  else if (diff > 0.5)
  {
    return eps_vacuum;
  }

  FPValue proportion = 0.5 - diff;

  return proportion * eps + (1 - proportion) * eps_vacuum;

  // FPValue d1 = sqrt (SQR (pos1.getX () - center.getX ()) + SQR (pos1.getY () - center.getY ()) + SQR (pos1.getZ () - center.getZ ()));
  // FPValue d2 = sqrt (SQR (pos2.getX () - center.getX ()) + SQR (pos2.getY () - center.getY ()) + SQR (pos2.getZ () - center.getZ ()));
  // FPValue d3 = sqrt (SQR (pos3.getX () - center.getX ()) + SQR (pos3.getY () - center.getY ()) + SQR (pos3.getZ () - center.getZ ()));
  // FPValue d4 = sqrt (SQR (pos4.getX () - center.getX ()) + SQR (pos4.getY () - center.getY ()) + SQR (pos4.getZ () - center.getZ ()));
  //
  // FPValue d5 = sqrt (SQR (pos5.getX () - center.getX ()) + SQR (pos5.getY () - center.getY ()) + SQR (pos5.getZ () - center.getZ ()));
  // FPValue d6 = sqrt (SQR (pos6.getX () - center.getX ()) + SQR (pos6.getY () - center.getY ()) + SQR (pos6.getZ () - center.getZ ()));
  // FPValue d7 = sqrt (SQR (pos7.getX () - center.getX ()) + SQR (pos7.getY () - center.getY ()) + SQR (pos7.getZ () - center.getZ ()));
  // FPValue d8 = sqrt (SQR (pos8.getX () - center.getX ()) + SQR (pos8.getY () - center.getY ()) + SQR (pos8.getZ () - center.getZ ()));

  // bool inR1 = d1 < radius;
  // bool inR2 = d2 < radius;
  // bool inR3 = d3 < radius;
  // bool inR4 = d4 < radius;
  //
  // bool inR5 = d5 < radius;
  // bool inR6 = d6 < radius;
  // bool inR7 = d7 < radius;
  // bool inR8 = d8 < radius;

  // total number of cases 1+1+8+12+
  //
  // if (inR1 && inR2 && inR3 && inR4
  //     && inR5 && inR6 && inR7 && inR8)
  // {
  //   return eps;
  // }
  //
  // if (inR1 && inR2 && inR3 && inR4)
}
