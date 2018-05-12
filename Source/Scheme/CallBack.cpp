#include "CallBack.h"
#include "PhysicsConst.h"
#include "Settings.h"

#include <cmath>

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
  return SQR (PhysicsConst::SpeedOfLight)
         * sin (2 * PhysicsConst::Pi * (PhysicsConst::SpeedOfLight / solverSettings.getSourceWaveLength ()) * t
                - 2 * PhysicsConst::Pi / solverSettings.getSourceWaveLength () * coord.get1 ());
}

FieldValue CallBack::sin1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return -1 * SQR (PhysicsConst::SpeedOfLight) / (PhysicsConst::Mu0 * PhysicsConst::SpeedOfLight)
         * sin (2 * PhysicsConst::Pi * (PhysicsConst::SpeedOfLight / solverSettings.getSourceWaveLength ()) * t
                - 2 * PhysicsConst::Pi / solverSettings.getSourceWaveLength () * coord.get1 ());
}

#endif /* COMPLEX_FIELD_VALUES */
