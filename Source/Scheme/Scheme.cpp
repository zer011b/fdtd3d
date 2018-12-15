#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "FieldValue.h"
#include "Settings.h"
#include "Scheme.h"
#include "Approximation.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#define _NAME(A,B) A ##B

#define SPECIALIZE_TEMPLATE_FUNC(RET, STYPE, COORD, LAYOUT_TYPE, NAME, ARGSND, ARGS, NAME_HELPER) \
  template <> \
  RET \
  Scheme<static_cast<SchemeType_t> (SchemeType::STYPE), COORD, LAYOUT_TYPE>::NAME ARGSND \
  { \
    return SchemeHelper::NAME_HELPER ARGS; \
  }

#define SPECIALIZE_TEMPLATE(RET1D, RET2D, RET3D, NAME, ARGS1D, ARGS2D, ARGS3D, ARGS) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_ExHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_ExHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EyHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EyHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EzHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EzHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET3D, Dim3, GridCoordinate3DTemplate, E_CENTERED, NAME, ARGS3D, ARGS, _NAME(NAME, 3D))

SPECIALIZE_TEMPLATE(GridCoordinate1D, GridCoordinate2D, GridCoordinate3D,
                    getStartCoordRes,
                    (OrthogonalAxis orthogonalAxis, GridCoordinate1D start, GridCoordinate1D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate2D start, GridCoordinate2D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate3D start, GridCoordinate3D size),
                    (orthogonalAxis, start, size))

SPECIALIZE_TEMPLATE(GridCoordinate1D, GridCoordinate2D, GridCoordinate3D,
                    getEndCoordRes,
                    (OrthogonalAxis orthogonalAxis, GridCoordinate1D end, GridCoordinate1D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate2D end, GridCoordinate2D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate3D end, GridCoordinate3D size),
                    (orthogonalAxis, end, size))

SPECIALIZE_TEMPLATE(NPair, NPair, NPair,
                    ntffN,
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEz, Grid<GridCoordinate1D> *curHx, Grid<GridCoordinate1D> *curHy, Grid<GridCoordinate1D> *curHz, GridCoordinate1D leftNTFF, GridCoordinate1D rightNTFF),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEz, Grid<GridCoordinate2D> *curHx, Grid<GridCoordinate2D> *curHy, Grid<GridCoordinate2D> *curHz, GridCoordinate2D leftNTFF, GridCoordinate2D rightNTFF),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEz, Grid<GridCoordinate3D> *curHx, Grid<GridCoordinate3D> *curHy, Grid<GridCoordinate3D> *curHz, GridCoordinate3D leftNTFF, GridCoordinate3D rightNTFF),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, intScheme->getGridStep (), intScheme->getSourceWaveLength (), intScheme->getHInc (), curEz, curHx, curHy, curHz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(NPair, NPair, NPair,
                    ntffL,
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEx, Grid<GridCoordinate1D> *curEy, Grid<GridCoordinate1D> *curEz, GridCoordinate1D leftNTFF, GridCoordinate1D rightNTFF),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEx, Grid<GridCoordinate2D> *curEy, Grid<GridCoordinate2D> *curEz, GridCoordinate2D leftNTFF, GridCoordinate2D rightNTFF),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEx, Grid<GridCoordinate3D> *curEy, Grid<GridCoordinate3D> *curEz, GridCoordinate3D leftNTFF, GridCoordinate3D rightNTFF),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, intScheme->getGridStep (), intScheme->getSourceWaveLength (), intScheme->getEInc (), curEx, curEy, curEz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(bool, bool, bool,
                    doSkipMakeScattered,
                    (GridCoordinateFP1D pos),
                    (GridCoordinateFP2D pos),
                    (GridCoordinateFP3D pos),
                    (pos, yeeLayout->getLeftBorderTFSF (), yeeLayout->getRightBorderTFSF ()))

SPECIALIZE_TEMPLATE(void, void, void,
                    initFullMaterialGrids,
                    (),
                    (),
                    (),
                    (intScheme->getEps (), totalEps,
                     intScheme->getMu (), totalMu,
                     intScheme->getOmegaPE (), totalOmegaPE,
                     intScheme->getOmegaPM (), totalOmegaPM,
                     intScheme->getGammaE (), totalGammaE,
                     intScheme->getGammaM (), totalGammaM))

SPECIALIZE_TEMPLATE(void, void, void,
                    initFullFieldGrids,
                    (),
                    (),
                    (),
                    (&totalInitialized,
                     intScheme->getDoNeedEx (), intScheme->getDoNeedEx () ? intScheme->getEx () : NULLPTR, &totalEx,
                     intScheme->getDoNeedEy (), intScheme->getDoNeedEy () ? intScheme->getEy () : NULLPTR, &totalEy,
                     intScheme->getDoNeedEz (), intScheme->getDoNeedEz () ? intScheme->getEz () : NULLPTR, &totalEz,
                     intScheme->getDoNeedHx (), intScheme->getDoNeedHx () ? intScheme->getHx () : NULLPTR, &totalHx,
                     intScheme->getDoNeedHy (), intScheme->getDoNeedHy () ? intScheme->getHy () : NULLPTR, &totalHy,
                     intScheme->getDoNeedHz (), intScheme->getDoNeedHz () ? intScheme->getHz () : NULLPTR, &totalHz))

#ifdef PARALLEL_GRID

SPECIALIZE_TEMPLATE(void, void, void,
                    initParallelBlocks,
                    (), (), (),
                    (this))

#endif /* PARALLEL_GRID */

/*
 * Specialization for Sigma
 */
#ifdef MODE_EX_HY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
};
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaX ());
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaY ());
  SchemeHelper::initSigmaZ<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>
    (yeeLayout, intScheme->getGridStep (), intScheme->getSigmaZ ());
}
#endif /* MODE_DIM3 */


#ifdef MODE_EX_HY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps1D<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps2D<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::performNSteps (time_step tStart, time_step N)
{
  SchemeHelper::performNSteps3D<static_cast<SchemeType_t> (SchemeType::Dim3), E_CENTERED> (this, tStart, N);
}
#endif /* MODE_DIM3 */
