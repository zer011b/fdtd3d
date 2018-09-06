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

  // if (omega1 == 0 && gamma1 == 0
  //     || omega2 == 0 && gamma2 == 0)
  // {
  //   dividerOmega = sqrtf (2.0);
  //   dividerGamma = 1.0;
  // }
  // else
  // {
  //   ASSERT (omega1 == omega2 && gamma1 == gamma2);
  //
  //   dividerOmega = 2.0;
  //   dividerGamma = 2.0;
  // }

  dividerOmega = 2.0;
  dividerGamma = 2.0;

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

  // if (omega2 == 0 && omega3 == 0 && omega4 == 0 && gamma2 == 0 && gamma3 == 0 && gamma4 == 0
  //     || omega1 == 0 && omega3 == 0 && omega4 == 0 && gamma1 == 0 && gamma3 == 0 && gamma4 == 0
  //     || omega1 == 0 && omega2 == 0 && omega4 == 0 && gamma1 == 0 && gamma2 == 0 && gamma4 == 0
  //     || omega1 == 0 && omega2 == 0 && omega3 == 0 && gamma1 == 0 && gamma2 == 0 && gamma3 == 0)
  // {
  //   dividerOmega = 2.0;
  //   dividerGamma = 1.0;
  // }
  // else if (omega1 == 0 && omega2 == 0 && omega3 == omega4 && gamma1 == 0 && gamma2 == 0 && gamma3 == gamma4
  //          || omega2 == 0 && omega4 == 0 && omega1 == omega3 && gamma2 == 0 && gamma4 == 0 && gamma1 == gamma3
  //          || omega3 == 0 && omega4 == 0 && omega1 == omega2 && gamma3 == 0 && gamma4 == 0 && gamma1 == gamma2
  //          || omega1 == 0 && omega3 == 0 && omega2 == omega4 && gamma1 == 0 && gamma3 == 0 && gamma2 == gamma4)
  // {
  //   dividerOmega = sqrt (2.0);
  //   dividerGamma = 2.0;
  // }
  // else if (omega1 == 0 && omega2 == omega3 && omega3 == omega4 && gamma1 == 0 && gamma2 == gamma3 && gamma3 == gamma4
  //          || omega2 == 0 && omega1 == omega3 && omega3 == omega4 && gamma2 == 0 && gamma1 == gamma3 && gamma3 == gamma4
  //          || omega3 == 0 && omega1 == omega2 && omega2 == omega4 && gamma3 == 0 && gamma1 == gamma2 && gamma2 == gamma4
  //          || omega4 == 0 && omega1 == omega2 && omega2 == omega3 && gamma4 == 0 && gamma1 == gamma2 && gamma2 == gamma3)
  // {
  //   dividerOmega = 2 / sqrt (3.0);
  //   dividerGamma = 3.0;
  // }
  // else
  // {
  //   ASSERT (omega1 == omega2 && omega2 == omega3 && omega3 == omega4
  //           && gamma1 == gamma2 && gamma2 == gamma3 && gamma3 == gamma4);
  //
  //   dividerOmega = 4.0;
  //   dividerGamma = 4.0;
  // }

  dividerOmega = 4.0;
  dividerGamma = 4.0;

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
Approximation::approximateWaveNumber (FPValue delta,
                                      FPValue freeSpaceWaveLentgh,
                                      FPValue courantNum,
                                      FPValue N_lambda,
                                      FPValue incidentWaveAngle1,
                                      FPValue incidentWaveAngle2)
{
  ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

  if ((incidentWaveAngle1 == 0
       || incidentWaveAngle1 == PhysicsConst::Pi / 2
       || incidentWaveAngle1 == PhysicsConst::Pi
       || incidentWaveAngle1 == 3 * PhysicsConst::Pi / 2)
      && (incidentWaveAngle2 == 0
          || incidentWaveAngle2 == PhysicsConst::Pi / 2
          || incidentWaveAngle2 == PhysicsConst::Pi
          || incidentWaveAngle2 == 3 * PhysicsConst::Pi / 2))
  {

    /*
     * Special case of propagation along some axes
     */
    return FPValue (2) / delta
           * asin (sin (PhysicsConst::Pi * courantNum / N_lambda) / courantNum);
  }

  if ((incidentWaveAngle1 == 0
       || incidentWaveAngle1 == PhysicsConst::Pi / 2
       || incidentWaveAngle1 == PhysicsConst::Pi
       || incidentWaveAngle1 == 3 * PhysicsConst::Pi / 2)
      && (incidentWaveAngle2 == PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 3 * PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 5 * PhysicsConst::Pi / 4
          || incidentWaveAngle2 == 7 * PhysicsConst::Pi / 4)
      || (incidentWaveAngle1 == PhysicsConst::Pi / 4
          || incidentWaveAngle1 == 3 * PhysicsConst::Pi / 4
          || incidentWaveAngle1 == 5 * PhysicsConst::Pi / 4
          || incidentWaveAngle1 == 7 * PhysicsConst::Pi / 4)
          && (incidentWaveAngle2 == 0
              || incidentWaveAngle2 == PhysicsConst::Pi / 2
              || incidentWaveAngle2 == PhysicsConst::Pi
              || incidentWaveAngle2 == 3 * PhysicsConst::Pi / 2))
  {
    /*
     * Special case of propagation at angle of Pi/4
     */
    return FPValue (2) * sqrt (FPValue (2)) / delta
           * asin (sin (PhysicsConst::Pi * courantNum / N_lambda) / (courantNum * sqrt (FPValue (2))));
  }

  return approximateWaveNumberGeneral (delta, freeSpaceWaveLentgh, courantNum, N_lambda, incidentWaveAngle1, incidentWaveAngle2);
}

FPValue
Approximation::approximateWaveNumberGeneral (FPValue delta,
                                             FPValue freeSpaceWaveLentgh,
                                             FPValue courantNum,
                                             FPValue N_lambda,
                                             FPValue incidentWaveAngle1,
                                             FPValue incidentWaveAngle2)
{
  ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

  FPValue k = 2 * PhysicsConst::Pi;
  FPValue k_prev = k + 2 * sqrt (Approximation::getAccuracy ());

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

  return k / freeSpaceWaveLentgh;
}

FPValue
Approximation::phaseVelocityIncidentWave (FPValue delta,
                                          FPValue freeSpaceWaveLentgh,
                                          FPValue courantNum,
                                          FPValue N_lambda,
                                          FPValue incidentWaveAngle1,
                                          FPValue incidentWaveAngle2)
{
  ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());
  return PhysicsConst::SpeedOfLight * 2 * PhysicsConst::Pi / freeSpaceWaveLentgh
         / approximateWaveNumber (delta, freeSpaceWaveLentgh, courantNum, N_lambda, incidentWaveAngle1, incidentWaveAngle2);
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

#define SPHERE_VOL_ACC_CUTOFF (1.0001)

FieldValue
Approximation::approximateSphereAccurate (GridCoordinateFP1D midPos,
                                          GridCoordinateFP1D center,
                                          FPValue radius,
                                          FieldValue eps,
                                          FieldValue outsideEps)
{
  GridCoordinateFP1D start (midPos.get1 () - 0.5
#ifdef DEBUG_INFO
                            , midPos.getType1 ()
#endif
                            );
  GridCoordinateFP1D end (midPos.get1 () + 0.5
#ifdef DEBUG_INFO
                          , midPos.getType1 ()
#endif
                          );

  /*
   * If this check is removed, be sure to support center of sphere not in the middle of cell at all usages of midPos
   */
  ASSERT (center.get1 () - FPValue (0.5) == (grid_coord) (center.get1 () - FPValue (0.5)));

  FPValue volume = 0;
  FPValue left = center.get1 () - radius;
  FPValue right = center.get1 () + radius;

  if (end.get1 () <= left
      || start.get1 () >= right)
  {
    /*
     * not in sphere
     */
  }
  else if (start.get1 () >= left
           || end.get1 () <= right)
  {
    /*
     * fully in sphere
     */
    volume = FPValue (1);
  }
  else if (start.get1 () < left)
  {
    ASSERT (end.get1 () >= left);
    volume = end.get1 () - left;
  }
  else if (end.get1 () > right)
  {
    ASSERT (start.get1 () < right);
    volume = right - start.get1 ();
  }

  ASSERT (volume >= FPValue (0) && volume <= FPValue (SPHERE_VOL_ACC_CUTOFF));
  if (volume > FPValue (1))
  {
    volume = FPValue (1);
  }

  return volume * eps + (1 - volume) * outsideEps;
}

FieldValue
Approximation::approximateSphereAccurate (GridCoordinateFP2D midPos,
                                          GridCoordinateFP2D center,
                                          FPValue radius,
                                          FieldValue eps,
                                          FieldValue outsideEps)
{
  GridCoordinateFP2D start (midPos.get1 () - 0.5, midPos.get2 () - 0.5
  #ifdef DEBUG_INFO
                            , midPos.getType1 (), midPos.getType2 ()
  #endif
                            );
  GridCoordinateFP2D end (midPos.get1 () + 0.5, midPos.get2 () + 0.5
  #ifdef DEBUG_INFO
                          , midPos.getType1 (), midPos.getType2 ()
  #endif
                          );

  /*
   * If this check is removed, be sure to support center of sphere not in the middle of cell at all usages of midPos,
   * for example:
   *
   *   if (midPos.get2 () > center.get2 ())
   */
  ASSERT (center.get1 () - FPValue (0.5) == (grid_coord) (center.get1 () - FPValue (0.5)));
  ASSERT (center.get2 () - FPValue (0.5) == (grid_coord) (center.get2 () - FPValue (0.5)));

  /*
   * TODO: use better approximation
   */

  int numSteps = solverSettings.getSphereAccuracy ();
  FPValue step = 1.0 / numSteps;
  FPValue elemS = step;
  FPValue volume = 0;
  for (int i = 0; i < numSteps; ++i)
  {
    GridCoordinateFP2D pos (start.get1 () + i * step, 0.0
  #ifdef DEBUG_INFO
                            , midPos.getType1 (), midPos.getType2 ()
  #endif
                            );
    /*
     * temp = R^2 - (x-x0)^2
     * Compare temp and (y-y0)^2
     */
    FPValue temp = SQR (radius) - SQR (pos.get1 () - center.get1 ());

    /*
     * This check cuts out all the points, which are not in the sphere or on the surface of sphere
     */
    if (temp < 0)
    {
      pos.set2 (0.0);
    }
    else
    {
      bool isAbove = true;

      /*
       * Obtain the actual y-axis coordinate of this point (pos):
       * check whether point is below or above line y=0, in order to identify sign of |y-y0|
       */
      if (midPos.get2 () > center.get2 ())
      {
        isAbove = true;
        pos.set2 (center.get2 () + sqrt (temp));
      }
      else
      {
        isAbove = false;
        pos.set2 (center.get2 () - sqrt (temp));
      }

      /*
       * Check if pos point is in cube for which volume is calculated. Truncate it if not.
       */
      if (pos.get2 () < start.get2 ())
      {
        pos.set2 (start.get2 ());
      }
      else if (pos.get2 () > end.get2 ())
      {
        pos.set2 (end.get2 ());
      }

      /*
       * Get relative value of height
       */
      if (isAbove)
      {
        pos.set2 (pos.get2 () - start.get2 ());
      }
      else
      {
        pos.set2 (end.get2 () - pos.get2 ());
      }
    }

    volume += pos.get2 () * elemS;
  }

  ASSERT (volume >= FPValue (0) && volume <= FPValue (SPHERE_VOL_ACC_CUTOFF));
  if (volume > FPValue (1))
  {
    volume = FPValue (1);
  }

  return volume * eps + (1 - volume) * outsideEps;
}

FieldValue
Approximation::approximateSphereAccurate (GridCoordinateFP3D midPos,
                                          GridCoordinateFP3D center,
                                          FPValue radius,
                                          FieldValue eps,
                                          FieldValue outsideEps)
{
  GridCoordinateFP3D start (midPos.get1 () - 0.5, midPos.get2 () - 0.5, midPos.get3 () - 0.5
#ifdef DEBUG_INFO
                            , midPos.getType1 (), midPos.getType2 (), midPos.getType3 ()
#endif
                            );
  GridCoordinateFP3D end (midPos.get1 () + 0.5, midPos.get2 () + 0.5, midPos.get3 () + 0.5
#ifdef DEBUG_INFO
                          , midPos.getType1 (), midPos.getType2 (), midPos.getType3 ()
#endif
                          );

  /*
   * If this check is removed, be sure to support center of sphere not in the middle of cell at all usages of midPos,
   * for example:
   *
   *   if (midPos.get3 () > center.get3 ())
   */
  ASSERT (center.get1 () - FPValue (0.5) == (grid_coord) (center.get1 () - FPValue (0.5)));
  ASSERT (center.get2 () - FPValue (0.5) == (grid_coord) (center.get2 () - FPValue (0.5)));
  ASSERT (center.get3 () - FPValue (0.5) == (grid_coord) (center.get3 () - FPValue (0.5)));

  /*
   * TODO: use better approximation
   */

  int numSteps = solverSettings.getSphereAccuracy ();
  FPValue step = 1.0 / numSteps;
  FPValue elemS = step * step;
  FPValue volume = 0;
  for (int i = 0; i < numSteps; ++i)
  {
    for (int j = 0; j < numSteps; ++j)
    {
      GridCoordinateFP3D pos (start.get1 () + i * step, start.get2 () + j * step, 0.0
#ifdef DEBUG_INFO
                              , midPos.getType1 (), midPos.getType2 (), midPos.getType3 ()
#endif
                              );
      /*
       * temp = R^2 - (x-x0)^2 - (y-y0)^2
       * Compare temp and (z-z0)^2
       */
      FPValue temp = SQR (radius) - SQR (pos.get1 () - center.get1 ()) - SQR (pos.get2 () - center.get2 ());

      /*
       * This check cuts out all the points, which are not in the sphere or on the surface of sphere
       */
      if (temp < 0)
      {
        pos.set3 (0.0);
      }
      else
      {
        bool isAbove = true;

        /*
         * Obtain the actual z-axis coordinate of this point (pos):
         * check whether point is below or above plane z=0, in order to identify sign of |z-z0|
         */
        if (midPos.get3 () > center.get3 ())
        {
          isAbove = true;
          pos.set3 (center.get3 () + sqrt (temp));
        }
        else
        {
          isAbove = false;
          pos.set3 (center.get3 () - sqrt (temp));
        }

        /*
         * Check if pos point is in cube for which volume is calculated. Truncate it if not.
         */
        if (pos.get3 () < start.get3 ())
        {
          pos.set3 (start.get3 ());
        }
        else if (pos.get3 () > end.get3 ())
        {
          pos.set3 (end.get3 ());
        }

        /*
         * Get relative value of height
         */
        if (isAbove)
        {
          pos.set3 (pos.get3 () - start.get3 ());
        }
        else
        {
          pos.set3 (end.get3 () - pos.get3 ());
        }
      }

      volume += pos.get3 () * elemS;
    }
  }

  ASSERT (volume >= FPValue (0) && volume <= FPValue (SPHERE_VOL_ACC_CUTOFF));
  if (volume > FPValue (1))
  {
    volume = FPValue (1);
  }

  return volume * eps + (1 - volume) * outsideEps;
}
