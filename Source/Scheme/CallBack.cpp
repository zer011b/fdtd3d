#include "CallBack.h"
#include "PhysicsConst.h"

#ifndef COMPLEX_FIELD_VALUES

FieldValue CallBack::polinom1_ez (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * SQR (coord.getX ()) * t;
}

FieldValue CallBack::polinom1_hy (GridCoordinateFP3D coord, FPValue t)
{
  return PhysicsConst::Eps0 * polinom1_ez (coord, t);
}

FieldValue CallBack::polinom1_jz (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * PhysicsConst::Eps0 * (-2 * coord.getX () * t + SQR (coord.getX ()));
}

FieldValue CallBack::polinom1_my (GridCoordinateFP3D coord, FPValue t)
{
  return SQR (PhysicsConst::SpeedOfLight) * (-2 * coord.getX () * t
                                             + SQR (coord.getX ()) * PhysicsConst::Eps0 * PhysicsConst::Mu0);
}

#endif /* COMPLEX_FIELD_VALUES */
