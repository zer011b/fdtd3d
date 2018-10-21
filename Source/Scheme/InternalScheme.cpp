#include "InternalScheme.h"

template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == 0);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
InternalSchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (GridCoordinateFP3D realCoord, GridCoordinateFP3D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  FPValue z = realCoord.get3 () - zeroCoordFP.get3 ();

  FPValue d = x * sin (incAngle1) * cos (incAngle2) + y * sin (incAngle1) * sin (incAngle2) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return InternalSchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
