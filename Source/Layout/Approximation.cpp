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
  return (approximateMaterial (val1, val2) + approximateMaterial (val3, val4)) / 2.0;
}

FPValue
Approximation::approximateMaterial (FPValue val1, FPValue val2, FPValue val3, FPValue val4,
                                    FPValue val5, FPValue val6, FPValue val7, FPValue val8)
{
  return (approximateMaterial (val1, val2, val3, val4) + approximateMaterial (val5, val6, val7, val8)) / 2.0;
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
   * TODO: incorrect!
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
   * TODO: incorrect!
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
                                      FPValue permittivity3,
                                      FPValue permittivity4,
                                      FPValue permittivity5,
                                      FPValue permittivity6,
                                      FPValue permittivity7,
                                      FPValue permittivity8,
                                      FPValue omega1,
                                      FPValue omega2,
                                      FPValue omega3,
                                      FPValue omega4,
                                      FPValue omega5,
                                      FPValue omega6,
                                      FPValue omega7,
                                      FPValue omega8,
                                      FPValue gamma1,
                                      FPValue gamma2,
                                      FPValue gamma3,
                                      FPValue gamma4,
                                      FPValue gamma5,
                                      FPValue gamma6,
                                      FPValue gamma7,
                                      FPValue gamma8)
{
  UNREACHABLE;
  /*
   * TODO: incorrect!
   */
  // FPValue dividerOmega = 0;
  // FPValue dividerGamma = 0;
  //
  // ASSERT (permittivity1 == 1 && permittivity2 == 1 && permittivity3 == 1 && permittivity4 == 1
  //         && permittivity5 == 1 && permittivity6 == 1 && permittivity7 == 1 && permittivity8 == 1);
  //
  // if (omega2 == omega3 == omega4 == omega5 == omega6 == omega7 == omega8 == 0 && gamma2 == gamma3 == gamma4 == gamma5 == gamma6 == gamma7 == gamma8 == 0
  //     || omega1 == omega3 == omega4 == omega5 == omega6 == omega7 == omega8 == 0 && gamma1 == gamma3 == gamma4 == gamma5 == gamma6 == gamma7 == gamma8 == 0
  //     || omega1 == omega2 == omega4 == omega5 == omega6 == omega7 == omega8 == 0 && gamma1 == gamma2 == gamma4 == gamma5 == gamma6 == gamma7 == gamma8 == 0
  //     || omega1 == omega2 == omega3 == omega5 == omega6 == omega7 == omega8 == 0 && gamma1 == gamma2 == gamma3 == gamma5 == gamma6 == gamma7 == gamma8 == 0
  //     || omega1 == omega2 == omega3 == omega4 == omega6 == omega7 == omega8 == 0 && gamma1 == gamma2 == gamma3 == gamma4 == gamma6 == gamma7 == gamma8 == 0
  //     || omega1 == omega2 == omega3 == omega4 == omega5 == omega7 == omega8 == 0 && gamma1 == gamma2 == gamma3 == gamma4 == gamma5 == gamma7 == gamma8 == 0
  //     || omega1 == omega2 == omega3 == omega4 == omega5 == omega6 == omega8 == 0 && gamma1 == gamma2 == gamma3 == gamma4 == gamma5 == gamma6 == gamma8 == 0
  //     || omega1 == omega2 == omega3 == omega4 == omega5 == omega6 == omega7 == 0 && gamma1 == gamma2 == gamma3 == gamma4 == gamma5 == gamma6 == gamma7 == 0)
  // {
  //   dividerOmega = 2.0;
  //   dividerGamma = 1.0;
  // }
  // else if (omega1 == omega2 == 0 && omega3 == omega4 && gamma1 == gamma2 == 0 && gamma3 == gamma4
  //          || omega2 == omega4 == 0 && omega1 == omega3 && gamma2 == gamma4 == 0 && gamma1 == gamma3
  //          || omega3 == omega4 == 0 && omega1 == omega2 && gamma3 == gamma4 == 0 && gamma1 == gamma2
  //          || omega1 == omega3 == 0 && omega2 == omega4 && gamma1 == gamma3 == 0 && gamma2 == gamma4)
  // {
  //   dividerOmega = sqrt (2.0);
  //   dividerGamma = 2.0;
  // }
  // else if (omega1 == 0 && omega2 == omega3 == omega4 && gamma1 == 0 && gamma2 == gamma3 == gamma4
  //          || omega2 == 0 && omega1 == omega3 == omega4 && gamma2 == 0 && gamma1 == gamma3 == gamma4
  //          || omega3 == 0 && omega1 == omega2 == omega4 && gamma3 == 0 && gamma1 == gamma2 == gamma4
  //          || omega4 == 0 && omega1 == omega2 == omega3 && gamma4 == 0 && gamma1 == gamma2 == gamma3)
  // {
  //   dividerOmega = 2 / sqrt (3.0);
  //   dividerGamma = 3.0;
  // }
  // else
  // {
  //   ASSERT (omega1 == omega2 == omega3 == omega4
  //           && gamma1 == gamma2 == gamma3 == gamma4);
  //
  //   dividerOmega = 4.0;
  //   dividerOmega = 4.0;
  // }
  //
  // ASSERT (dividerOmega != 0);
  // ASSERT (dividerGamma != 0);
  //
  // omega = (omega1 + omega2 + omega3 + omega4) / dividerOmega;
  // gamma = (gamma1 + gamma2 + gamma3 + gamma4) / dividerGamma;
}

FPValue
Approximation::getMaterial (const FieldPointValue *val)
{
  return getRealOnlyFromFieldValue (val->getCurValue ());
}

FPValue
Approximation::phaseVelocityIncidentWave (FPValue delta,
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

FieldValue
Approximation::approximateSphereFast (GridCoordinateFP3D midPos,
                                      GridCoordinateFP3D center,
                                      FPValue radius,
                                      FieldValue eps)
{
  FPValue d = sqrt (SQR (midPos.get1 () - center.get1 ()) + SQR (midPos.get2 () - center.get2 ()) + SQR (midPos.get3 () - center.get3 ()));

  FPValue diff = d - radius;

  FieldValue eps_vacuum = getFieldValueRealOnly (1.0);

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
}

FieldValue
Approximation::approximateSphereAccurate (GridCoordinateFP3D midPos,
                                          GridCoordinateFP3D center,
                                          FPValue radius,
                                          FieldValue eps)
{
  GridCoordinateFP3D start (midPos.get1 () - 0.5, midPos.get2 () - 0.5, midPos.get3 () - 0.5
#ifdef DEBUG_INFO
                            , CoordinateType::X, CoordinateType::Y, CoordinateType::Z
#endif
                            );
  GridCoordinateFP3D end (midPos.get1 () + 0.5, midPos.get2 () + 0.5, midPos.get3 () + 0.5
#ifdef DEBUG_INFO
                          , CoordinateType::X, CoordinateType::Y, CoordinateType::Z
#endif
                          );

  int numSteps = 100;
  FPValue step = 1.0 / numSteps;
  FPValue elemS = step * step;
  FPValue volume = 0;
  for (int i = 0; i < numSteps; ++i)
  {
    for (int j = 0; j < numSteps; ++j)
    {
      GridCoordinateFP3D pos (start.get1 () + i * step, start.get2 () + j * step, 0.0
#ifdef DEBUG_INFO
                              , CoordinateType::X, CoordinateType::Y, CoordinateType::Z
#endif
                              );
      FPValue temp = SQR (radius) - SQR (pos.get1 () - center.get1 ()) - SQR (pos.get2 () - center.get2 ());

      if (temp < 0)
      {
        pos.set3 (0.0);
      }
      else
      {
        if (midPos.get3 () > center.get3 ())
        {
          pos.set3 (center.get3 () + sqrt (temp));
        }
        else
        {
          pos.set3 (center.get3 () - sqrt (temp));
        }

        if (pos.get3 () < start.get3 ())
        {
          pos.set3 (1.0);
        }
        else if (pos.get3 () > end.get3 ())
        {
          pos.set3 (0.0);
        }
        else
        {
          pos.set3 (pos.get3 () - start.get3 ());
        }
      }
      volume += pos.get3 () * elemS;
    }
  }

  ASSERT (volume <= 1.0);

  FieldValue eps_vacuum = getFieldValueRealOnly (1.0);

  return volume * eps + (1 - volume) * eps_vacuum;
}
