#include "Approximation.h"
#include "CallBack.h"
#include "PhysicsConst.h"
#include "Settings.h"

#include <cmath>
#include <complex>

#ifndef COMPLEX_FIELD_VALUES

FieldValue CallBack::polinom1_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * SQR (coord.get1 ()) * t;
}

FieldValue CallBack::polinom1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom1_ez (coord, t);
}

FieldValue CallBack::polinom1_jz (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (-2 * coord.get1 () * t + SQR (coord.get1 ()));
}

FieldValue CallBack::polinom1_my (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * (-2 * coord.get1 () * t
    + SQR (coord.get1 ()) * PhysicsConst::Eps0 * PhysicsConst::Mu0);
}

FieldValue CallBack::polinom2_ex (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get2 ()) + SQR (coord.get3 ()));
}

FieldValue CallBack::polinom2_ey (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get1 ()) + SQR (coord.get3 ()));
}

FieldValue CallBack::polinom2_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get1 ()) + SQR (coord.get2 ()));
}

FieldValue CallBack::polinom2_hx (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ex (coord, t);
}

FieldValue CallBack::polinom2_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ey (coord, t);
}

FieldValue CallBack::polinom2_hz (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ez (coord, t);
}

FieldValue CallBack::polinom2_jx (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get3 () - coord.get2 ()) + SQR (coord.get2 ()) + SQR (coord.get3 ()));
}

FieldValue CallBack::polinom2_jy (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get1 () - coord.get3 ()) + SQR (coord.get1 ()) + SQR (coord.get3 ()));
}

FieldValue CallBack::polinom2_jz (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get2 () - coord.get1 ()) + SQR (coord.get1 ()) + SQR (coord.get2 ()));
}

FieldValue CallBack::polinom2_mx (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get2 () - coord.get3 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get2 ()) + SQR (coord.get3 ())));
}

FieldValue CallBack::polinom2_my (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get3 () - coord.get1 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get1 ()) + SQR (coord.get3 ())));
}

FieldValue CallBack::polinom2_mz (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get1 () - coord.get2 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get1 ()) + SQR (coord.get2 ())));
}

FieldValue CallBack::polinom3_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * SQR (coord.get1 () * t);
}

FieldValue CallBack::polinom3_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom3_ez (coord, t);
}

FieldValue CallBack::polinom3_jz (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * 2 * PhysicsConst::Eps0 * coord.get1 () * t * (coord.get1 () - t);
}

FieldValue CallBack::polinom3_my (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * 2 * coord.get1 () * t
    * (PhysicsConst::Eps0 * PhysicsConst::Mu0 * coord.get1 () - t);
}

FieldValue CallBack::sin1_ez (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::SpeedOfLight * sin (t - coord.get1 ());
}

FieldValue CallBack::sin1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return - PhysicsConst::SpeedOfLight * sin (t - coord.get1 ());
}

#endif /* !COMPLEX_FIELD_VALUES */

/**
 * Incident wave E1 = i * e^(i*k*(z-z0) - i*w*(t-t0))
 *
 * @return value of E1
 */
FieldValue CallBack::exp1_ex (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;
  FPValue z0 = solverSettings.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue t0 = 0.5 * gridTimeStep;

  FPValue arg = k * (coord.get3 () - z0 * delta) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return i * std::exp (i * arg);
#else /* COMPLEX_FIELD_VALUES */
  return - sin (arg);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp1_ex */

/**
 * Incident wave H1 = E1 * (k / (mu0 * w))
 *
 * @return value of H1
 */
FieldValue CallBack::exp1_hy (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return (k / (PhysicsConst::Mu0 * w)) * exp1_ex (coord, t);
} /* CallBack::exp1_hy */

/**
 * Reflected E2 = -1/3 * i * e^(- i*k*z + i*k*(2*z_border - z0) - i*w*(t-t0))
 *
 * @return value of E2
 */
FieldValue CallBack::exp2_ex (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;
  FPValue z0 = solverSettings.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue t0 = 0.5 * gridTimeStep;

  FPValue zb = 2 * (solverSettings.getEpsSphereCenterZ () - solverSettings.getEpsSphereRadius ()) - z0;
  FPValue arg = - k * (coord.get3 () - zb * solverSettings.getGridStep ()) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return - FPValue (1) / FPValue (3) * i * std::exp (i * arg);
#else /* COMPLEX_FIELD_VALUES */
  return FPValue (1) / FPValue (3) * sin (arg);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp2_ex */

/**
 * Incident wave H2 = E2 * (k / (mu0 * w))
 *
 * @return value of H2
 */
FieldValue CallBack::exp2_hy (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return (- k / (PhysicsConst::Mu0 * w)) * exp2_ex (coord, t);
} /* CallBack::exp2_hy */

/**
 * Passing E3 = 2/3 * i * e^(i*k_2*x - i*k_2*(z_border + z0)/2 - i*w*(t-t0))
 * k_2 = (w * 2) / c = 2*k
 *
 * @return value of E3
 */
FieldValue CallBack::exp3_ex (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k_2 = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                      solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                      solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);
  k_2 = 2 * k_2;

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;
  FPValue z0 = solverSettings.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue t0 = 0.5 * gridTimeStep;

  FPValue zb = (solverSettings.getEpsSphereCenterZ () - solverSettings.getEpsSphereRadius () + z0) / FPValue (2);
  FPValue arg = k_2 * (coord.get3 () - zb * solverSettings.getGridStep ()) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return FPValue (2) / FPValue (3) * i * std::exp (i * arg);
#else /* COMPLEX_FIELD_VALUES */
  return - FPValue (2) / FPValue (3) * sin (arg);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp3_ex */

/**
 * Incident wave H3 = E3 * (k / (mu0 * w))
 *
 * @return value of H3
 */
FieldValue CallBack::exp3_hy (GridCoordinateFP3D coord, /**< real floating point coordinate */
                              FPValue t) /**< real floating point time */
{
  FPValue lambda = solverSettings.getSourceWaveLength ();
  FPValue courantNum = solverSettings.getCourantNum ();
  FPValue delta = solverSettings.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k_2 = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                      solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                      solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);
  k_2 = 2 * k_2;

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return (k_2 / (PhysicsConst::Mu0 * w)) * exp3_ex (coord, t);
} /* CallBack::exp3_hy */
