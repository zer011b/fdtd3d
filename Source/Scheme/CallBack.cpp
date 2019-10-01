#include "Approximation.h"
#include "CallBack.h"
#include "PhysicsConst.h"
#include "Settings.h"

#include <cmath>

#ifndef COMPLEX_FIELD_VALUES

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom1_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * SQR (coord.get1 ()) * t;
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom1_ez (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom1_jz (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (-2 * coord.get1 () * t + SQR (coord.get1 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom1_my (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * (-2 * coord.get1 () * t
    + SQR (coord.get1 ()) * PhysicsConst::Eps0 * PhysicsConst::Mu0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_ex (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get2 ()) + SQR (coord.get3 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_ey (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get1 ()) + SQR (coord.get3 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * t * (SQR (coord.get1 ()) + SQR (coord.get2 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_hx (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ex (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ey (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_hz (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom2_ez (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_jx (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get3 () - coord.get2 ()) + SQR (coord.get2 ()) + SQR (coord.get3 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_jy (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get1 () - coord.get3 ()) + SQR (coord.get1 ()) + SQR (coord.get3 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_jz (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (2 * t * (coord.get2 () - coord.get1 ()) + SQR (coord.get1 ()) + SQR (coord.get2 ()));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_mx (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get2 () - coord.get3 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get2 ()) + SQR (coord.get3 ())));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_my (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get3 () - coord.get1 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get1 ()) + SQR (coord.get3 ())));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom2_mz (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * (2 * t * (coord.get1 () - coord.get2 ())
    + PhysicsConst::Eps0 * PhysicsConst::Mu0 * (SQR (coord.get1 ()) + SQR (coord.get2 ())));
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom3_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * SQR (coord.get1 () * t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom3_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom3_ez (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom3_jz (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * 2 * PhysicsConst::Eps0 * coord.get1 () * t * (coord.get1 () - t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::polinom3_my (GridCoordinateFP3D coord, FPValue t)
{
  return - SQR (PhysicsConst::SpeedOfLight) * 2 * coord.get1 () * t
    * (PhysicsConst::Eps0 * PhysicsConst::Mu0 * coord.get1 () - t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::sin1_ez (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::SpeedOfLight * sin (t - coord.get1 ());
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::sin1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return - PhysicsConst::SpeedOfLight * sin (t - coord.get1 ());
}

#endif /* !COMPLEX_FIELD_VALUES */

/*
 * ExHy
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ex_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  return - exp1_e (coord.get3 (), t, z0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hy_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  return - exp1_h (coord.get3 (), t, z0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ex_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp2_e (coord.get3 (), t, z0, zb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hy_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp2_h (coord.get3 (), t, z0, zb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ex_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp3_e (coord.get3 (), t, z0, zb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hy_exhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp3_h (coord.get3 (), t, z0, zb);
}

/*
 * ExHz
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ex_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  return exp1_e (coord.get2 (), t, y0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hz_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  return - exp1_h (coord.get2 (), t, y0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ex_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp2_e (coord.get2 (), t, y0, yb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hz_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp2_h (coord.get2 (), t, y0, yb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ex_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp3_e (coord.get2 (), t, y0, yb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hz_exhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp3_h (coord.get2 (), t, y0, yb);
}

/*
 * EyHx
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ey_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp1_ex_exhy (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hx_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  return exp1_h (coord.get3 (), t, z0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ey_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp2_ex_exhy (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hx_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp2_h (coord.get3 (), t, z0, zb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ey_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp3_ex_exhy (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hx_eyhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue z0 = SOLVER_SETTINGS.getTFSFSizeZLeft () - FPValue (2.5);
  FPValue zb = SOLVER_SETTINGS.getEpsSphereCenterZ () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp3_h (coord.get3 (), t, z0, zb);
}

/*
 * EyHz
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ey_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  return - exp1_e (coord.get1 (), t, x0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hz_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  return - exp1_h (coord.get1 (), t, x0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ey_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp2_e (coord.get1 (), t, x0, xb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hz_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp2_h (coord.get1 (), t, x0, xb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ey_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp3_e (coord.get1 (), t, x0, xb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hz_eyhz (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return - exp3_h (coord.get1 (), t, x0, xb);
}

/*
 * EzHx
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ez_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp1_ex_exhz (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hx_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  return exp1_h (coord.get2 (), t, y0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ez_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp2_ex_exhz (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hx_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp2_h (coord.get2 (), t, y0, yb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ez_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  return exp3_ex_exhz (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hx_ezhx (GridCoordinateFP3D coord, FPValue t)
{
  FPValue y0 = SOLVER_SETTINGS.getTFSFSizeYLeft () - FPValue (2.5);
  FPValue yb = SOLVER_SETTINGS.getEpsSphereCenterY () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp3_h (coord.get2 (), t, y0, yb);
}

/*
 * EzHy
 */

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_ez_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  return exp1_e (coord.get1 (), t, x0);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_hy_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  return exp1_hz_eyhz (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_ez_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp2_e (coord.get1 (), t, x0, xb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_hy_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  return exp2_hz_eyhz (coord, t);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_ez_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  FPValue x0 = SOLVER_SETTINGS.getTFSFSizeXLeft () - FPValue (2.5);
  FPValue xb = SOLVER_SETTINGS.getEpsSphereCenterX () - SOLVER_SETTINGS.getEpsSphereRadius ();
  return exp3_e (coord.get1 (), t, x0, xb);
}

CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_hy_ezhy (GridCoordinateFP3D coord, FPValue t)
{
  return exp3_hz_eyhz (coord, t);
}

/**
 * Incident wave (EInc) E1 = i * e^(i*k*(z-z0) - i*w*(t-t0))
 *
 * @return value of E1
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_e (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0) /**< start coordinate */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;

  FPValue t0 = 0.5 * gridTimeStep;

  FPValue arg = k * (coord - coord0 * delta) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return i * exponent (i * arg);
#else /* COMPLEX_FIELD_VALUES */
  return - sin (arg);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp1_e */

/**
 * Incident wave (HInc) H1 = E1 * (k / (mu0 * w))
 *
 * @return value of H1
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp1_h (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0) /**< start coordinate */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return exp1_e (coord, t, coord0) * (k / (PhysicsConst::Mu0 * w));
} /* CallBack::exp1_h */

/**
 * Reflected E2 = -1/3 * i * e^(- i*k*z + i*k*(2*z_border - z0) - i*w*(t-t0))
 *
 * @return value of E2
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_e (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0, /**< start coordinate */
                             FPValue coordb) /**< coordinate of border */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;

  FPValue t0 = 0.5 * gridTimeStep;
  FPValue diff = 2 * (coordb) - coord0;

  FPValue arg = - k * (coord - diff * delta) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return - i * exponent (i * arg) * FPValue (1) / FPValue (3);
#else /* COMPLEX_FIELD_VALUES */
  return sin (arg) * FPValue (1) / FPValue (3);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp2_e */

/**
 * Incident wave H2 = E2 * (k / (mu0 * w))
 *
 * @return value of H2
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp2_h (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0, /**< start coordinate */
                             FPValue coordb) /**< coordinate of border */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                    SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return exp2_e (coord, t, coord0, coordb) * (- k / (PhysicsConst::Mu0 * w));
} /* CallBack::exp2_h */

/**
 * Passing E3 = 2/3 * i * e^(i*k_2*x - i*k_2*(z_border + z0)/2 - i*w*(t-t0))
 * k_2 = (w * 2) / c = 2*k
 *
 * @return value of E3
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_e (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0, /**< start coordinate */
                             FPValue coordb) /**< coordinate of border */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k_2 = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                      SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                      SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);
  k_2 = 2 * k_2;

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;
  FPValue gridTimeStep = delta * courantNum / PhysicsConst::SpeedOfLight;

  FPValue t0 = 0.5 * gridTimeStep;
  FPValue diff = (coordb + coord0) / FPValue (2);

  FPValue arg = k_2 * (coord - diff * delta) - w * (t - t0);

#ifdef COMPLEX_FIELD_VALUES
  FieldValue i (0, 1);
  return i * exponent (i * arg) * FPValue (2) / FPValue (3);
#else /* COMPLEX_FIELD_VALUES */
  return - sin (arg) * FPValue (2) / FPValue (3);
#endif /* !COMPLEX_FIELD_VALUES */
} /* CallBack::exp3_e */

/**
 * Incident wave H3 = E3 * (k / (mu0 * w))
 *
 * @return value of H3
 */
CUDA_DEVICE CUDA_HOST FieldValue CallBack::exp3_h (FPValue coord, /**< real floating point coordinate */
                             FPValue t, /**< real floating point time */
                             FPValue coord0, /**< start coordinate */
                             FPValue coordb) /**< coordinate of border */
{
  FPValue lambda = SOLVER_SETTINGS.getSourceWaveLength ();
  FPValue courantNum = SOLVER_SETTINGS.getCourantNum ();
  FPValue delta = SOLVER_SETTINGS.getGridStep ();
  FPValue N_lambda = lambda / delta;

  FPValue k_2 = Approximation::approximateWaveNumber (delta, lambda, courantNum, N_lambda,
                                                      SOLVER_SETTINGS.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                                                      SOLVER_SETTINGS.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0);
  k_2 = 2 * k_2;

  FPValue f = PhysicsConst::SpeedOfLight / lambda;
  FPValue w = 2 * PhysicsConst::Pi * f;

  return exp3_e (coord, t, coord0, coordb) * (k_2 / (PhysicsConst::Mu0 * w));
} /* CallBack::exp3_h */
