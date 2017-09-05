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
   * FIXME: incorrect!
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
Approximation::approximateSphere_1 (GridCoordinateFP3D midPos,
                                    GridCoordinateFP3D center,
                                    FPValue radius,
                                    FieldValue eps)
{
  struct temp1
  {
    GridCoordinateFP3D first;
    GridCoordinateFP3D second;

    temp1 (GridCoordinateFP3D f, GridCoordinateFP3D s)
    : first (f), second (s)
    {
    }

    temp1 ()
    : first (0.0, 0.0, 0.0), second (0.0, 0.0, 0.0)
    {
    }
  };

  GridCoordinateFP3D points[8];
  points[0] = GridCoordinateFP3D (midPos.getX () - 0.5, midPos.getY () - 0.5, midPos.getZ () - 0.5);
  points[1] = GridCoordinateFP3D (midPos.getX () + 0.5, midPos.getY () - 0.5, midPos.getZ () - 0.5);
  points[2] = GridCoordinateFP3D (midPos.getX () + 0.5, midPos.getY () + 0.5, midPos.getZ () - 0.5);
  points[3] = GridCoordinateFP3D (midPos.getX () - 0.5, midPos.getY () + 0.5, midPos.getZ () - 0.5);

  points[4] = GridCoordinateFP3D (midPos.getX () - 0.5, midPos.getY () - 0.5, midPos.getZ () + 0.5);
  points[5] = GridCoordinateFP3D (midPos.getX () + 0.5, midPos.getY () - 0.5, midPos.getZ () + 0.5);
  points[6] = GridCoordinateFP3D (midPos.getX () + 0.5, midPos.getY () + 0.5, midPos.getZ () + 0.5);
  points[7] = GridCoordinateFP3D (midPos.getX () - 0.5, midPos.getY () + 0.5, midPos.getZ () + 0.5);

  temp1 edges[12];
  edges[0] = temp1 (points[0], points[1]);
  edges[1] = temp1 (points[1], points[2]);
  edges[2] = temp1 (points[3], points[2]);
  edges[3] = temp1 (points[0], points[3]);

  edges[4] = temp1 (points[4], points[5]);
  edges[5] = temp1 (points[5], points[6]);
  edges[6] = temp1 (points[7], points[6]);
  edges[7] = temp1 (points[4], points[7]);

  edges[8] = temp1 (points[0], points[4]);
  edges[9] = temp1 (points[1], points[5]);
  edges[10] = temp1 (points[2], points[6]);
  edges[11] = temp1 (points[3], points[7]);

  FPValue x0 = center.getX ();
  FPValue y0 = center.getY ();
  FPValue z0 = center.getZ ();

  GridCoordinateFP3D plane_points[3];

  uint32_t index = 0;

  for (uint32_t i = 0; i < 12 && index < 3; ++i)
  {
    FPValue func1 = SQR (edges[i].first.getX () - center.getX ())
                    + SQR (edges[i].first.getY () - center.getY ())
                    + SQR (edges[i].first.getZ () - center.getZ ())
                    - radius * radius;

    FPValue func2 = SQR (edges[i].second.getX () - center.getX ())
                    + SQR (edges[i].second.getY () - center.getY ())
                    + SQR (edges[i].second.getZ () - center.getZ ())
                    - radius * radius;

    if (func1 * func2 < 0)
    {
      // sphere crosses this plane

      FPValue x1 = edges[i].first.getX ();
      FPValue y1 = edges[i].first.getY ();
      FPValue z1 = edges[i].first.getZ ();

      FPValue x2 = edges[i].second.getX ();
      FPValue y2 = edges[i].second.getY ();
      FPValue z2 = edges[i].second.getZ ();

      FPValue a = x2 - x1;
      FPValue b = y2 - y1;
      FPValue c = z2 - z1;

      FPValue p = y1 - y0;
      FPValue q = z1 - z0;

      if (a == 0)
      {
        // printf ("%f %f %f\n", midPos.getX (), midPos.getY (), midPos.getZ ());
        // printf ("%f %f %f\n", center.getX (), center.getY (), center.getZ ());
        // printf ()
        // ASSERT(false);

        if (b == 0)
        {
          FPValue q = SQR (x1 - x0) + SQR (y1 - y0);
          FPValue p = z0 - z1;
          FPValue discr_div_4 = SQR (radius) - q;

          ASSERT (discr_div_4 >= 0);

          if (discr_div_4 == 0)
          {
            FPValue alpha = p;
            if (z1 + alpha >= z1 && z1 + alpha <= z2)
            {
              // printf ("$9\n");
              plane_points[index++] = GridCoordinateFP3D (x1, y1, z1 + alpha);
            }
          }
          else
          {
            FPValue alpha1 = p + sqrt (discr_div_4);
            FPValue alpha2 = p - sqrt (discr_div_4);

            if (z1 + alpha1 >= z1 && z1 + alpha1 <= z2)
            {
              // printf ("$8\n");
              plane_points[index++] = GridCoordinateFP3D (x1, y1, z1 + alpha1);
            }
            else if (z1 + alpha2 >= z1 && z1 + alpha2 <= z2)
            {
              // printf ("$7\n");
              plane_points[index++] = GridCoordinateFP3D (x1, y1, z1 + alpha2);
            }
          }
        }
        else
        {
          FPValue q = SQR (x1 - x0);
          FPValue k = c / b;

          FPValue p = - k * y1 + z1 - z0;
          FPValue l = 1 + SQR (k);
          FPValue h = k * p - y0;
          FPValue u = q + SQR (y0) + SQR (p) - SQR (radius);

          FPValue discr_div_4 = SQR (h) - l * u;

          ASSERT (discr_div_4 >= 0);

          if (discr_div_4 == 0)
          {
            FPValue y = - h / l;
            FPValue z = k * (y - y1) + z1;

            if (y >= y1 && y <= y2)
            {
              // printf ("$6\n");
              plane_points[index++] = GridCoordinateFP3D (x1, y, z);
            }
          }
          else
          {
            FPValue y = (- h + sqrt (discr_div_4)) / l;
            FPValue z = k * (y - y1) + z1;

            FPValue yy = (- h - sqrt (discr_div_4)) / l;
            FPValue zz = k * (yy - y1) + z1;

            if (y >= y1 && y <= y2)
            {
              // printf ("$5\n");
              plane_points[index++] = GridCoordinateFP3D (x1, y, z);
            }
            else if (yy >= y1 && yy <= y2)
            {
              // printf ("$4\n");
              plane_points[index++] = GridCoordinateFP3D (x1, yy, zz);
            }
          }
        }
      }
      else
      {
        FPValue k = b / a;
        FPValue l = c / a;

        FPValue d = - k * x1 + p;
        FPValue h = - l * x1 + q;

        FPValue discr_div_4 = SQR (radius) * (1 + SQR (k) + SQR (l)) - SQR (k * x0 + d) - SQR (l * x0 + h) - SQR (d * l - h * k);

        ASSERT (discr_div_4 >= 0);

        if (discr_div_4 == 0)
        {
          FPValue x = - (k * d + l * h - x0) / (1 + SQR (k) + SQR (l));
          FPValue y = k * (x - x1) + y1;
          FPValue z = l * (x - x1) + z1;

          if (x >= x1 && x <= x2)
          {
            // printf ("$3\n");
            plane_points[index++] = GridCoordinateFP3D (x, y, z);
          }
        }
        else
        {
          FPValue x = (- (k * d + l * h - x0) + sqrt(discr_div_4)) / (1 + SQR (k) + SQR (l));
          FPValue y = k * (x - x1) + y1;
          FPValue z = l * (x - x1) + z1;

          FPValue xx = (- (k * d + l * h - x0) - sqrt(discr_div_4)) / (1 + SQR (k) + SQR (l));
          FPValue yy = k * (xx - x1) + y1;
          FPValue zz = l * (xx - x1) + z1;

          if (x >= x1 && x <= x2)
          {
            // printf ("$1\n");
            plane_points[index++] = GridCoordinateFP3D (x, y, z);
          }
          else if (xx >= x1 && xx <= x2)
          {
            // printf ("$2\n");
            plane_points[index++] = GridCoordinateFP3D (xx, yy, zz);
          }
        }
      }
    }
  }

  // printf ("%u\n", index);
  // printf ("%f %f %f\n", plane_points[0].getX (), plane_points[0].getY (), plane_points[0].getZ ());
  // printf ("%f %f %f\n", midPos.getX (), midPos.getY (), midPos.getZ ());

  ASSERT (index == 3 || index == 0);

  FieldValue eps_vacuum = getFieldValueRealOnly (1.0);

  if (index == 3)
  {
    FPValue _x0 = plane_points[0].getX ();
    FPValue _y0 = plane_points[0].getY ();
    FPValue _z0 = plane_points[0].getZ ();

    FPValue _x1 = plane_points[1].getX ();
    FPValue _y1 = plane_points[1].getY ();
    FPValue _z1 = plane_points[1].getZ ();

    FPValue _x2 = plane_points[2].getX ();
    FPValue _y2 = plane_points[2].getY ();
    FPValue _z2 = plane_points[2].getZ ();

    FPValue A = (_y1 - _y0) * (_z2 - _z0) - (_y2 - _y0) * (_z1 - _z0);
    FPValue B = - (_x1 - _x0) * (_z2 - _z0) + (_x2 - _x0) * (_z1 - _z0);
    FPValue C = (_x1 - _x0) * (_y2 - _y0) - (_x2 - _x0) * (_y1 - _y0);
    FPValue D = - (_x0 * A + _y0 * B + _z0 * C);

    FPValue part_volume = 0;

    FPValue startx0 = points[0].getX ();
    FPValue starty0 = points[0].getY ();
    FPValue startz0 = points[0].getZ ();

    FPValue endy0 = points[7].getY ();
    FPValue endz0 = points[7].getZ ();

    if (C == 0)
    {
      if (B == 0)
      {
        part_volume = (_x0 - startx0) * 1 * 1;

        if (startx0 < x0)
        {
          part_volume = 1 - part_volume;
        }
      }
      else
      {
        FPValue A1 = - A / B;
        FPValue B1 = - D / B;

        FPValue N = 1000;
        FPValue step = 1.0 / N;
        for (uint32_t step_i = 0; step_i < N; ++step_i)
        {
          FPValue yval = A1 * (step_i * step + startx0) + B1;

          if (yval < starty0)
          {
            yval = starty0;
          }
          else if (yval > endy0)
          {
            yval = endy0;
          }

          part_volume += yval * step;
        }

        part_volume -= starty0 * 1;

        if (starty0 < y0)
        {
          part_volume = 1 - part_volume;
        }
      }
    }
    else
    {
      FPValue A1 = - A / C;
      FPValue B1 = - B / C;
      FPValue C1 = - D / C;

      //============================
      // Volume
      //============================

      //FPValue part_volume = C1 * 1 * 1 + A1 * (SQR (points[0].getX() + 1) - SQR (points[0].getX()))/2 * 1 + B1 * (SQR (points[0].getY() + 1) - SQR (points[0].getY()))/2 * 1;
      {
        FPValue N = 1000;
        FPValue step = 1.0 / N;
        for (uint32_t step_i = 0; step_i < N; ++step_i)
        {
          for (uint32_t step_j = 0; step_j < N; ++step_j)
          {
            FPValue zval = A1 * (step_i * step + startx0) + B1 * (step_j * step + starty0) + C1;

            if (zval < startz0)
            {
              zval = startz0;
            }
            else if (zval > endz0)
            {
              zval = endz0;
            }

            part_volume += zval * step * step;
          }
        }

        part_volume -= startz0 * 1 * 1;

        if (startz0 < z0)
        {
          part_volume = 1 - part_volume;
        }
      }
    }

    FPValue full_volume = 1;

    FPValue distance0 = sqrt (SQR (points[0].getX() - x0) + SQR (points[0].getY() - y0) + SQR (points[0].getZ() - z0));

    FPValue diff = part_volume / full_volume;

    // if (distance0 > radius)
    // {
    //   diff = 1 - diff;
    // }

    return diff * eps + (1 - diff) * eps_vacuum;
  }
  else
  {
    FPValue d = sqrt (SQR (midPos.getX () - center.getX ()) + SQR (midPos.getY () - center.getY ()) + SQR (midPos.getZ () - center.getZ ()));

    if (d < radius)
    {
      return eps;
    }
    else
    {
      ASSERT (d > radius);

      return eps_vacuum;
    }
  }
}
