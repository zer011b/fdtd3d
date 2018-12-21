#include "InternalScheme.h"

#define _NAME(A,B) A ##B

#define SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET, STYPE, COORD, LAYOUT_TYPE, NAME, ARGSND, ARGS, NAME_HELPER) \
  template <> \
  RET \
  CLASS<static_cast<SchemeType_t> (SchemeType::STYPE), COORD, LAYOUT_TYPE>::NAME ARGSND \
  { \
    HELPER::NAME_HELPER ARGS; \
  }

#define SPECIALIZE_TEMPLATE_1D(CLASS, HELPER, RET1D, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_ExHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_ExHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EyHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EyHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EzHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EzHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_ExHy, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_ExHz, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EyHx, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EyHz, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EzHx, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET1D, Dim1_EzHy, GridCoordinate1DTemplate, H_CENTERED, NAME, ARGS1D, ARGS, NAME2)

#define SPECIALIZE_TEMPLATE_2D(CLASS, HELPER, RET2D, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEx, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEy, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TEz, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMx, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMy, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET2D, Dim2_TMz, GridCoordinate2DTemplate, H_CENTERED, NAME, ARGS2D, ARGS, NAME2)

#define SPECIALIZE_TEMPLATE_3D(CLASS, HELPER, RET3D, NAME, ARGS3D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET3D, Dim3, GridCoordinate3DTemplate, E_CENTERED, NAME, ARGS3D, ARGS, NAME2) \
  SPECIALIZE_TEMPLATE_FUNC(CLASS, HELPER, RET3D, Dim3, GridCoordinate3DTemplate, H_CENTERED, NAME, ARGS3D, ARGS, NAME2)

#define SPECIALIZE_TEMPLATE(CLASS, HELPER, RET1D, RET2D, RET3D, NAME, ARGS1D, ARGS2D, ARGS3D, ARGS) \
  SPECIALIZE_TEMPLATE_1D(CLASS, HELPER, RET1D, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_2D(CLASS, HELPER, RET2D, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_3D(CLASS, HELPER, RET3D, NAME, ARGS3D, ARGS, _NAME(NAME, 3D)) \

SPECIALIZE_TEMPLATE(InternalScheme, return InternalSchemeHelper,
                    bool, bool, bool,
                    doSkipBorderFunc,
                    (GridCoordinate1D pos, Grid<GridCoordinate1D> *grid),
                    (GridCoordinate2D pos, Grid<GridCoordinate2D> *grid),
                    (GridCoordinate3D pos, Grid<GridCoordinate3D> *grid),
                    (pos, grid))

#ifdef PARALLEL_GRID

SPECIALIZE_TEMPLATE_1D(InternalScheme, InternalSchemeHelper,
                       void,
                       allocateParallelGrids,
                       (),
                       (this),
                       allocateParallelGrids1D)

SPECIALIZE_TEMPLATE_2D(InternalScheme, InternalSchemeHelper,
                       void,
                       allocateParallelGrids,
                       (),
                       (this),
                       allocateParallelGrids2D)

SPECIALIZE_TEMPLATE_3D(InternalScheme, InternalSchemeHelper,
                       void,
                       allocateParallelGrids,
                       (),
                       (this),
                       allocateParallelGrids3D)

#endif /* PARALLEL_GRID */

#ifdef ENABLE_ASSERTS
#define IDEVICE
#define ISCHEME InternalScheme
#include "InternalScheme.cpp.inc"
#undef IDEVICE
#undef ISCHEME
#endif /* ENABLE_ASSERTS */

#ifdef MODE_EX_HY
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
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
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
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
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
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
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
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
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
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
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
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
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
#endif /* MODE_TEX */

#ifdef MODE_TEY
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
#endif /* MODE_TEY */

#ifdef MODE_TEZ
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
#endif /* MODE_TEZ */

#ifdef MODE_TMX
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
#endif /* MODE_TMX */

#ifdef MODE_TMY
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
#endif /* MODE_TMY */

#ifdef MODE_TMZ
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
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
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
#endif /* MODE_DIM3 */
