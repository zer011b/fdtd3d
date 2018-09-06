#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "Kernels.h"
#include "Settings.h"
#include "Scheme.h"
#include "Approximation.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

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
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEz, Grid<GridCoordinate1D> *curHx, Grid<GridCoordinate1D> *curHy, Grid<GridCoordinate1D> *curHz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEz, Grid<GridCoordinate2D> *curHx, Grid<GridCoordinate2D> *curHy, Grid<GridCoordinate2D> *curHz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEz, Grid<GridCoordinate3D> *curHx, Grid<GridCoordinate3D> *curHy, Grid<GridCoordinate3D> *curHz),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy, curHz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(NPair, NPair, NPair,
                    ntffL,
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEx, Grid<GridCoordinate1D> *curEy, Grid<GridCoordinate1D> *curEz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEx, Grid<GridCoordinate2D> *curEy, Grid<GridCoordinate2D> *curEz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEx, Grid<GridCoordinate3D> *curEy, Grid<GridCoordinate3D> *curEz),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(bool, bool, bool,
                    doSkipBorderFunc,
                    (GridCoordinate1D pos, Grid<GridCoordinate1D> *grid),
                    (GridCoordinate2D pos, Grid<GridCoordinate2D> *grid),
                    (GridCoordinate3D pos, Grid<GridCoordinate3D> *grid),
                    (pos, grid))

SPECIALIZE_TEMPLATE(bool, bool, bool,
                    doSkipMakeScattered,
                    (GridCoordinateFP1D pos),
                    (GridCoordinateFP2D pos),
                    (GridCoordinateFP3D pos),
                    (pos, yeeLayout->getLeftBorderTFSF (), yeeLayout->getRightBorderTFSF ()))

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedEx = Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedEy = Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedEz = Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedHx = Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedHy = Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedHz = Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedSigmaX = Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedSigmaY = Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
const bool Scheme<Type, TCoord, layout_type>::doNeedSigmaZ = Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
                                                    || Type == static_cast<SchemeType_t> (SchemeType::Dim3);

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFExAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFEzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHxAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
  ASSERT (pos11.get3 () < pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () < pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHyAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () < pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () < pos12.get1 ());
  ASSERT (pos21.get1 () == pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () == pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::calculateTFSFHzAsserts
  (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
{
  ASSERT (pos11.get1 () == pos12.get1 ());
  ASSERT (pos21.get1 () < pos22.get1 ());
  ASSERT (pos11.get2 () < pos12.get2 ());
  ASSERT (pos21.get2 () == pos22.get2 ());
  ASSERT (pos11.get3 () == pos12.get3 ());
  ASSERT (pos21.get3 () == pos22.get3 ());
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Z;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Y;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Z;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Y;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::NONE;
  ct3 = CoordinateType::NONE;
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Y;
  ct2 = CoordinateType::Z;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::Z;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::Y;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::Y;
  ct2 = CoordinateType::Z;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::Z;
  ct3 = CoordinateType::NONE;
}
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::Y;
  ct3 = CoordinateType::NONE;
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::initCoordTypes ()
{
  ct1 = CoordinateType::X;
  ct2 = CoordinateType::Y;
  ct3 = CoordinateType::Z;
}

#ifdef PARALLEL_GRID
template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::allocateParallelGrids ()
{
#ifdef GRID_3D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> *pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> *) yeeLayout;

  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               solverSettings.getBufferSize (),
                                                                               ct1, ct2, ct3);

  SchemeHelper::allocateParallelGrids (pLayout, doNeedEx, doNeedEy, doNeedEz, doNeedHx, doNeedHy, doNeedHz,
                                       doNeedSigmaX, doNeedSigmaY, doNeedSigmaZ,
                                       bufSize, (ParallelGrid **)&Eps, (ParallelGrid **)&Mu,
                                       (ParallelGrid **)&Ex, (ParallelGrid **)&Ey, (ParallelGrid **)&Ez,
                                       (ParallelGrid **)&Hx, (ParallelGrid **)&Hy, (ParallelGrid **)&Hz,
                                       (ParallelGrid **)&Dx, (ParallelGrid **)&Dy, (ParallelGrid **)&Dz,
                                       (ParallelGrid **)&Bx, (ParallelGrid **)&By, (ParallelGrid **)&Bz,
                                       (ParallelGrid **)&D1x, (ParallelGrid **)&D1y, (ParallelGrid **)&D1z,
                                       (ParallelGrid **)&B1x, (ParallelGrid **)&B1y, (ParallelGrid **)&B1z,
                                       (ParallelGrid **)&SigmaX, (ParallelGrid **)&SigmaY, (ParallelGrid **)&SigmaZ,
                                       (ParallelGrid **)&ExAmplitude, (ParallelGrid **)&EyAmplitude, (ParallelGrid **)&EzAmplitude,
                                       (ParallelGrid **)&HxAmplitude, (ParallelGrid **)&HyAmplitude, (ParallelGrid **)&HzAmplitude,
                                       (ParallelGrid **)&OmegaPE, (ParallelGrid **)&GammaE,
                                       (ParallelGrid **)&OmegaPM, (ParallelGrid **)&GammaM);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                  "Recompile it with -DPARALLEL_GRID_DIMENSION=3.");
#endif
}
#endif

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                           bool parallelLayout,
                                           const TC& totSize,
                                           time_step tStep)
  : yeeLayout (layout)
  , isParallelLayout (parallelLayout)
  , Ex (NULLPTR)
  , Ey (NULLPTR)
  , Ez (NULLPTR)
  , Hx (NULLPTR)
  , Hy (NULLPTR)
  , Hz (NULLPTR)
  , Dx (NULLPTR)
  , Dy (NULLPTR)
  , Dz (NULLPTR)
  , Bx (NULLPTR)
  , By (NULLPTR)
  , Bz (NULLPTR)
  , D1x (NULLPTR)
  , D1y (NULLPTR)
  , D1z (NULLPTR)
  , B1x (NULLPTR)
  , B1y (NULLPTR)
  , B1z (NULLPTR)
  , ExAmplitude (NULLPTR)
  , EyAmplitude (NULLPTR)
  , EzAmplitude (NULLPTR)
  , HxAmplitude (NULLPTR)
  , HyAmplitude (NULLPTR)
  , HzAmplitude (NULLPTR)
  , Eps (NULLPTR)
  , Mu (NULLPTR)
  , OmegaPE (NULLPTR)
  , OmegaPM (NULLPTR)
  , GammaE (NULLPTR)
  , GammaM (NULLPTR)
  , SigmaX (NULLPTR)
  , SigmaY (NULLPTR)
  , SigmaZ (NULLPTR)
  , EInc (NULLPTR)
  , HInc (NULLPTR)
  , totalEx (NULLPTR)
  , totalEy (NULLPTR)
  , totalEz (NULLPTR)
  , totalHx (NULLPTR)
  , totalHy (NULLPTR)
  , totalHz (NULLPTR)
  , totalInitialized (false)
  , totalEps (NULLPTR)
  , totalMu (NULLPTR)
  , totalOmegaPE (NULLPTR)
  , totalOmegaPM (NULLPTR)
  , totalGammaE (NULLPTR)
  , totalGammaM (NULLPTR)
  , sourceWaveLength (0)
  , sourceWaveLengthNumerical (0)
  , sourceFrequency (0)
  , courantNum (0)
  , gridStep (0)
  , gridTimeStep (0)
  , totalStep (tStep)
  , ExBorder (NULLPTR)
  , ExInitial (NULLPTR)
  , EyBorder (NULLPTR)
  , EyInitial (NULLPTR)
  , EzBorder (NULLPTR)
  , EzInitial (NULLPTR)
  , HxBorder (NULLPTR)
  , HxInitial (NULLPTR)
  , HyBorder (NULLPTR)
  , HyInitial (NULLPTR)
  , HzBorder (NULLPTR)
  , HzInitial (NULLPTR)
  , Jx (NULLPTR)
  , Jy (NULLPTR)
  , Jz (NULLPTR)
  , Mx (NULLPTR)
  , My (NULLPTR)
  , Mz (NULLPTR)
  , ExExact (NULLPTR)
  , EyExact (NULLPTR)
  , EzExact (NULLPTR)
  , HxExact (NULLPTR)
  , HyExact (NULLPTR)
  , HzExact (NULLPTR)
  , useParallel (false)
{
  initCoordTypes ();

  if (solverSettings.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (solverSettings.getNTFFSizeX (), solverSettings.getNTFFSizeY (), solverSettings.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = layout->getEzSize () - leftNTFF + TC (1, 1, 1
#ifdef DEBUG_INFO
                                                      , ct1, ct2, ct3
#endif
                                                      );
  }

  if (solverSettings.getDoUseParallelGrid ())
  {
#ifndef PARALLEL_GRID
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.")
#endif

    ALWAYS_ASSERT (isParallelLayout);
#ifdef PARALLEL_GRID
    ALWAYS_ASSERT ((TCoord<grid_coord, false>::dimension == ParallelGridCoordinateTemplate<grid_coord, false>::dimension));
#endif

    useParallel = true;
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    allocateParallelGrids ();
#else /* PARALLEL_GRID */
    ALWAYS_ASSERT (false);
#endif /* !PARALLEL_GRID */
  }
  else
  {
    Eps = new Grid<TC> (layout->getEpsSize (), 0, "Eps");
    Mu = new Grid<TC> (layout->getEpsSize (), 0, "Mu");

    Ex = doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "Ex") : NULLPTR;
    Ey = doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "Ey") : NULLPTR;
    Ez = doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "Ez") : NULLPTR;
    Hx = doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "Hx") : NULLPTR;
    Hy = doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "Hy") : NULLPTR;
    Hz = doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "Hz") : NULLPTR;

    if (solverSettings.getDoUsePML ())
    {
      Dx = doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "Dx") : NULLPTR;
      Dy = doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "Dy") : NULLPTR;
      Dz = doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "Dz") : NULLPTR;
      Bx = doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "Bx") : NULLPTR;
      By = doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "By") : NULLPTR;
      Bz = doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "Bz") : NULLPTR;

      if (solverSettings.getDoUseMetamaterials ())
      {
        D1x = doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "D1x") : NULLPTR;
        D1y = doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "D1y") : NULLPTR;
        D1z = doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "D1z") : NULLPTR;
        B1x = doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "B1x") : NULLPTR;
        B1y = doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "B1y") : NULLPTR;
        B1z = doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "B1z") : NULLPTR;
      }

      SigmaX = doNeedSigmaX ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaX") : NULLPTR;
      SigmaY = doNeedSigmaY ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaY") : NULLPTR;
      SigmaZ = doNeedSigmaZ ? new Grid<TC> (layout->getEpsSize (), 0, "SigmaZ") : NULLPTR;
    }

    if (solverSettings.getDoUseAmplitudeMode ())
    {
      ExAmplitude = doNeedEx ? new Grid<TC> (layout->getExSize (), 0, "ExAmp") : NULLPTR;
      EyAmplitude = doNeedEy ? new Grid<TC> (layout->getEySize (), 0, "EyAmp") : NULLPTR;
      EzAmplitude = doNeedEz ? new Grid<TC> (layout->getEzSize (), 0, "EzAmp") : NULLPTR;
      HxAmplitude = doNeedHx ? new Grid<TC> (layout->getHxSize (), 0, "HxAmp") : NULLPTR;
      HyAmplitude = doNeedHy ? new Grid<TC> (layout->getHySize (), 0, "HyAmp") : NULLPTR;
      HzAmplitude = doNeedHz ? new Grid<TC> (layout->getHzSize (), 0, "HzAmp") : NULLPTR;
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      OmegaPE = new Grid<TC> (layout->getEpsSize (), 0, "OmegaPE");
      GammaE = new Grid<TC> (layout->getEpsSize (), 0, "GammaE");
      OmegaPM = new Grid<TC> (layout->getEpsSize (), 0, "OmegaPM");
      GammaM = new Grid<TC> (layout->getEpsSize (), 0, "GammaM");
    }

    totalEps = Eps;
    totalMu = Mu;
    totalOmegaPE = OmegaPE;
    totalOmegaPM = OmegaPM;
    totalGammaE = GammaE;
    totalGammaM = GammaM;
  }

  if (solverSettings.getDoUseTFSF ())
  {
    EInc = new Grid<GridCoordinate1D> (GridCoordinate1D (500*(totSize.get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                              ), 0, "EInc");
    HInc = new Grid<GridCoordinate1D> (GridCoordinate1D (500*(totSize.get1 ())
#ifdef DEBUG_INFO
                                                              , CoordinateType::X
#endif
                                                              ), 0, "HInc");
  }

  ASSERT (!solverSettings.getDoUseTFSF ()
          || (solverSettings.getDoUseTFSF ()
              && (yeeLayout->getLeftBorderTFSF () != TC (0, 0, 0, ct1, ct2, ct3)
                  || yeeLayout->getRightBorderTFSF () != yeeLayout->getSize ())));

  ASSERT (!solverSettings.getDoUsePML ()
          || (solverSettings.getDoUsePML () && (yeeLayout->getSizePML () != TC (0, 0, 0, ct1, ct2, ct3))));

  ASSERT (!solverSettings.getDoUseAmplitudeMode ()
          || solverSettings.getDoUseAmplitudeMode () && solverSettings.getNumAmplitudeSteps () != 0);

#ifdef COMPLEX_FIELD_VALUES
  ASSERT (!solverSettings.getDoUseAmplitudeMode ());
#endif /* COMPLEX_FIELD_VALUES */

  if (solverSettings.getDoSaveAsBMP ())
  {
    PaletteType palette = PaletteType::PALETTE_GRAY;
    OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

    if (solverSettings.getDoUsePaletteGray ())
    {
      palette = PaletteType::PALETTE_GRAY;
    }
    else if (solverSettings.getDoUsePaletteRGB ())
    {
      palette = PaletteType::PALETTE_BLUE_GREEN_RED;
    }

    if (solverSettings.getDoUseOrthAxisX ())
    {
      orthogonalAxis = OrthogonalAxis::X;
    }
    else if (solverSettings.getDoUseOrthAxisY ())
    {
      orthogonalAxis = OrthogonalAxis::Y;
    }
    else if (solverSettings.getDoUseOrthAxisZ ())
    {
      orthogonalAxis = OrthogonalAxis::Z;
    }

    dumper[FILE_TYPE_BMP] = new BMPDumper<TC> ();
    ((BMPDumper<TC> *) dumper[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);

    dumper1D[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate1D> ();
    ((BMPDumper<GridCoordinate1D> *) dumper1D[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  else
  {
    dumper[FILE_TYPE_BMP] = NULLPTR;
    dumper1D[FILE_TYPE_BMP] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsDAT ())
  {
    dumper[FILE_TYPE_DAT] = new DATDumper<TC> ();
    dumper1D[FILE_TYPE_DAT] = new DATDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_DAT] = NULLPTR;
    dumper1D[FILE_TYPE_DAT] = NULLPTR;
  }

  if (solverSettings.getDoSaveAsTXT ())
  {
    dumper[FILE_TYPE_TXT] = new TXTDumper<TC> ();
    dumper1D[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_TXT] = NULLPTR;
    dumper1D[FILE_TYPE_TXT] = NULLPTR;
  }

  if (!solverSettings.getEpsFileName ().empty ()
      || !solverSettings.getMuFileName ().empty ()
      || !solverSettings.getOmegaPEFileName ().empty ()
      || !solverSettings.getOmegaPMFileName ().empty ()
      || !solverSettings.getGammaEFileName ().empty ()
      || !solverSettings.getGammaMFileName ().empty ())
  {
    {
      loader[FILE_TYPE_BMP] = new BMPLoader<TC> ();

      PaletteType palette = PaletteType::PALETTE_GRAY;
      OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

      if (solverSettings.getDoUsePaletteGray ())
      {
        palette = PaletteType::PALETTE_GRAY;
      }
      else if (solverSettings.getDoUsePaletteRGB ())
      {
        palette = PaletteType::PALETTE_BLUE_GREEN_RED;
      }

      if (solverSettings.getDoUseOrthAxisX ())
      {
        orthogonalAxis = OrthogonalAxis::X;
      }
      else if (solverSettings.getDoUseOrthAxisY ())
      {
        orthogonalAxis = OrthogonalAxis::Y;
      }
      else if (solverSettings.getDoUseOrthAxisZ ())
      {
        orthogonalAxis = OrthogonalAxis::Z;
      }

      ((BMPLoader<TC> *) loader[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
    }
    {
      loader[FILE_TYPE_DAT] = new DATLoader<TC> ();
    }
    {
      loader[FILE_TYPE_TXT] = new TXTLoader<TC> ();
    }
  }
  else
  {
    loader[FILE_TYPE_BMP] = NULLPTR;
    loader[FILE_TYPE_DAT] = NULLPTR;
    loader[FILE_TYPE_TXT] = NULLPTR;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::~Scheme ()
{
  delete Eps;
  delete Mu;

  delete Ex;
  delete Ey;
  delete Ez;

  delete Hx;
  delete Hy;
  delete Hz;

  if (solverSettings.getDoUsePML ())
  {
    delete Dx;
    delete Dy;
    delete Dz;

    delete Bx;
    delete By;
    delete Bz;

    if (solverSettings.getDoUseMetamaterials ())
    {
      delete D1x;
      delete D1y;
      delete D1z;

      delete B1x;
      delete B1y;
      delete B1z;
    }

    delete SigmaX;
    delete SigmaY;
    delete SigmaZ;
  }

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    delete ExAmplitude;
    delete EyAmplitude;
    delete EzAmplitude;
    delete HxAmplitude;
    delete HyAmplitude;
    delete HzAmplitude;
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
    delete OmegaPE;
    delete OmegaPM;
    delete GammaE;
    delete GammaM;
  }

  if (solverSettings.getDoUseTFSF ())
  {
    delete EInc;
    delete HInc;
  }

  if (totalInitialized)
  {
    delete totalEx;
    delete totalEy;
    delete totalEz;

    delete totalHx;
    delete totalHy;
    delete totalHz;
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    delete totalEps;
    delete totalMu;

    delete totalOmegaPE;
    delete totalOmegaPM;
    delete totalGammaE;
    delete totalGammaM;
#else /* PARALLEL_GRID */
    UNREACHABLE;
#endif /* !PARALLEL_GRID */
  }

  delete dumper[FILE_TYPE_BMP];
  delete dumper[FILE_TYPE_DAT];
  delete dumper[FILE_TYPE_TXT];

  delete loader[FILE_TYPE_BMP];
  delete loader[FILE_TYPE_DAT];
  delete loader[FILE_TYPE_TXT];

  delete dumper1D[FILE_TYPE_BMP];
  delete dumper1D[FILE_TYPE_DAT];
  delete dumper1D[FILE_TYPE_TXT];
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t EnumVal>
void
Scheme<Type, TCoord, layout_type>::performPointSourceCalc (time_step t)
{
  Grid<TC> *grid = NULLPTR;

  switch (EnumVal)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      ASSERT (doNeedEx);
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      ASSERT (doNeedEy);
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      ASSERT (doNeedEz);
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      ASSERT (doNeedHx);
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      ASSERT (doNeedHy);
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      ASSERT (doNeedHz);
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (grid != NULLPTR);

  TC pos = TC::initAxesCoordinate (solverSettings.getPointSourcePositionX (),
                                   solverSettings.getPointSourcePositionY (),
                                   solverSettings.getPointSourcePositionZ (),
                                   ct1, ct2, ct3);

  FieldPointValue* pointVal = grid->getFieldPointValueOrNullByAbsolutePos (pos);

  if (pointVal)
  {
#ifdef COMPLEX_FIELD_VALUES
    pointVal->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                       cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
    pointVal->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
void Scheme<Type, TCoord, layout_type>::calculateTFSF (TC posAbs,
                                                       FieldValue &valOpposite11,
                                                       FieldValue &valOpposite12,
                                                       FieldValue &valOpposite21,
                                                       FieldValue &valOpposite22,
                                                       TC pos11,
                                                       TC pos12,
                                                       TC pos21,
                                                       TC pos22)
{
  bool doNeedUpdate11;
  bool doNeedUpdate12;
  bool doNeedUpdate21;
  bool doNeedUpdate22;

  bool isRevertVals;

  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFExAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEx);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::UP);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateExBorder (posAbs, LayoutDirection::FRONT);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFEyAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEy);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::FRONT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateEyBorder (posAbs, LayoutDirection::RIGHT);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFEzAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedEz);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::RIGHT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateEzBorder (posAbs, LayoutDirection::UP);

      isRevertVals = true;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHxAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHx);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::FRONT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHxBorder (posAbs, LayoutDirection::UP);

      isRevertVals = false;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHyAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHy);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::RIGHT);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::BACK);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHyBorder (posAbs, LayoutDirection::FRONT);

      isRevertVals = false;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
#ifdef ENABLE_ASSERTS
      calculateTFSFHzAsserts (pos11, pos12, pos21, pos22);
#endif
      ASSERT (doNeedHz);

      doNeedUpdate11 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::DOWN);
      doNeedUpdate12 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::UP);

      doNeedUpdate21 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::LEFT);
      doNeedUpdate22 = yeeLayout->doNeedTFSFUpdateHzBorder (posAbs, LayoutDirection::RIGHT);

      isRevertVals = false;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  TC auxPos1;
  TC auxPos2;
  FieldValue diff1;
  FieldValue diff2;

  if (isRevertVals)
  {
    if (doNeedUpdate11)
    {
      auxPos1 = pos12;
    }
    else if (doNeedUpdate12)
    {
      auxPos1 = pos11;
    }

    if (doNeedUpdate21)
    {
      auxPos2 = pos22;
    }
    else if (doNeedUpdate22)
    {
      auxPos2 = pos21;
    }
  }
  else
  {
    if (doNeedUpdate11)
    {
      auxPos1 = pos11;
    }
    else if (doNeedUpdate12)
    {
      auxPos1 = pos12;
    }

    if (doNeedUpdate21)
    {
      auxPos2 = pos21;
    }
    else if (doNeedUpdate22)
    {
      auxPos2 = pos22;
    }
  }

  if (doNeedUpdate11 || doNeedUpdate12)
  {
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        if (doNeedHz)
        {
          TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        if (doNeedHx)
        {
          TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        if (doNeedHy)
        {
          TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos1));
          diff1 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        if (doNeedEy)
        {
          TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos1));
          diff1 = FPValue (-1.0) * yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        if (doNeedEz)
        {
          TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos1));
          diff1 = FPValue (-1.0) * yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        if (doNeedEx)
        {
          TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos1));
          diff1 = FPValue (-1.0) * yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  if (doNeedUpdate21 || doNeedUpdate22)
  {
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        if (doNeedHy)
        {
          TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        if (doNeedHz)
        {
          TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        if (doNeedHx)
        {
          TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos2));
          diff2 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        if (doNeedEz)
        {
          TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos2));
          diff2 = FPValue (-1.0) * yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        if (doNeedEx)
        {
          TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos2));
          diff2 = FPValue (-1.0) * yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        if (doNeedEy)
        {
          TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos2));
          diff2 = FPValue (-1.0) * yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));
        }

        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }

  if (isRevertVals)
  {
    if (doNeedUpdate11)
    {
      valOpposite12 -= diff1;
    }
    else if (doNeedUpdate12)
    {
      valOpposite11 -= diff1;
    }

    if (doNeedUpdate21)
    {
      valOpposite22 -= diff2;
    }
    else if (doNeedUpdate22)
    {
      valOpposite21 -= diff2;
    }
  }
  else
  {
    if (doNeedUpdate11)
    {
      valOpposite11 -= diff1;
    }
    else if (doNeedUpdate12)
    {
      valOpposite12 -= diff1;
    }

    if (doNeedUpdate21)
    {
      valOpposite21 -= diff2;
    }
    else if (doNeedUpdate22)
    {
      valOpposite22 -= diff2;
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepIterationExact (time_step t,
                                                                     TC pos,
                                                                     Grid<TC> *grid,
                                                                     SourceCallBack exactFunc,
                                                                     FPValue &normRe,
                                                                     FPValue &normIm,
                                                                     FPValue &normMod,
                                                                     FPValue &maxRe,
                                                                     FPValue &maxIm,
                                                                     FPValue &maxMod)
{
  TC posAbs = grid->getTotalPosition (pos);

  TCFP realCoord;
  FPValue timestep;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      realCoord = yeeLayout->getExCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      realCoord = yeeLayout->getEyCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      realCoord = yeeLayout->getEzCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      realCoord = yeeLayout->getHxCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      realCoord = yeeLayout->getHyCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      realCoord = yeeLayout->getHzCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  FieldValue numerical = grid->getFieldPointValue (pos)->getCurValue ();
  FieldValue exact = exactFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);

#ifdef COMPLEX_FIELD_VALUES
  FPValue modExact = sqrt (SQR (exact.real ()) + SQR (exact.imag ()));
  FPValue modNumerical = sqrt (SQR (numerical.real ()) + SQR (numerical.imag ()));

  //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName ().c_str (), exact.real (), numerical.real ());

  normRe += SQR (exact.real () - numerical.real ());
  normIm += SQR (exact.imag () - numerical.imag ());
  normMod += SQR (modExact - modNumerical);

  FPValue exactAbs = fabs (exact.real ());
  if (maxRe < exactAbs)
  {
    maxRe = exactAbs;
  }

  exactAbs = fabs (exact.imag ());
  if (maxIm < exactAbs)
  {
    maxIm = exactAbs;
  }

  exactAbs = modExact;
  if (maxMod < exactAbs)
  {
    maxMod = exactAbs;
  }
#else
  normRe += SQR (exact - numerical);

  //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName ().c_str (), exact, numerical);

  FPValue exactAbs = fabs (exact);
  if (maxRe < exactAbs)
  {
    maxRe = exactAbs;
  }
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepIterationBorder (time_step t,
                                                                      TC pos,
                                                                      Grid<TC> *grid,
                                                                      SourceCallBack borderFunc)
{
  TC posAbs = grid->getTotalPosition (pos);

  if (doSkipBorderFunc (posAbs, grid))
  {
    return;
  }

  TCFP realCoord;
  FPValue timestep;
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      realCoord = yeeLayout->getExCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      realCoord = yeeLayout->getEyCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      realCoord = yeeLayout->getEzCoordFP (posAbs);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      realCoord = yeeLayout->getHxCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      realCoord = yeeLayout->getHyCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      realCoord = yeeLayout->getHzCoordFP (posAbs);
      timestep = t + 1.0;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  grid->getFieldPointValue (pos)->setCurValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep));
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool useMetamaterials>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepIterationPML (time_step t,
                                                                   TC pos,
                                                                   Grid<TC> *grid,
                                                                   Grid<TC> *gridPML1,
                                                                   Grid<TC> *gridPML2,
                                                                   GridType gridType,
                                                                   GridType gridPMLType1,
                                                                   Grid<TC> *materialGrid1,
                                                                   GridType materialGridType1,
                                                                   Grid<TC> *materialGrid4,
                                                                   GridType materialGridType4,
                                                                   Grid<TC> *materialGrid5,
                                                                   GridType materialGridType5,
                                                                   FPValue materialModifier)
{
  FPValue eps0 = PhysicsConst::Eps0;

  TC posAbs = gridPML2->getTotalPosition (pos);

  FieldPointValue *valField = gridPML2->getFieldPointValue (pos);

  FieldPointValue *valField1;

  if (useMetamaterials)
  {
    valField1 = gridPML1->getFieldPointValue (pos);
  }
  else
  {
    valField1 = grid->getFieldPointValue (pos);
  }

  FPValue material1 = materialGrid1 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1) : 0;
  FPValue material4 = materialGrid4 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4) : 0;
  FPValue material5 = materialGrid5 ? yeeLayout->getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5) : 0;

  FPValue modifier = material1 * materialModifier;
  if (useMetamaterials)
  {
    modifier = 1;
  }

  FPValue k_mod1 = 1;
  FPValue k_mod2 = 1;

  FPValue Ca = (2 * eps0 * k_mod2 - material5 * gridTimeStep) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
  FPValue Cb = ((2 * eps0 * k_mod1 + material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);
  FPValue Cc = ((2 * eps0 * k_mod1 - material4 * gridTimeStep) / (modifier)) / (2 * eps0 * k_mod2 + material5 * gridTimeStep);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue val = calcFieldFromDOrB (valField->getPrevValue (),
                                      valField1->getCurValue (),
                                      valField1->getPrevValue (),
                                      Ca,
                                      Cb,
                                      Cc);
#else
  ALWAYS_ASSERT (0);
#endif

  valField->setCurValue (val);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepIterationPMLMetamaterials (time_step t,
                                                                               TC pos,
                                                                               Grid<TC> *grid,
                                                                               Grid<TC> *gridPML,
                                                                               GridType gridType,
                                                                               Grid<TC> *materialGrid1,
                                                                               GridType materialGridType1,
                                                                               Grid<TC> *materialGrid2,
                                                                               GridType materialGridType2,
                                                                               Grid<TC> *materialGrid3,
                                                                               GridType materialGridType3,
                                                                               FPValue materialModifier)
{
  TC posAbs = grid->getTotalPosition (pos);
  FieldPointValue *valField = grid->getFieldPointValue (pos);
  FieldPointValue *valField1 = gridPML->getFieldPointValue (pos);

  FPValue material1;
  FPValue material2;

  FPValue material = yeeLayout->getMetaMaterial (posAbs, gridType,
                                                 materialGrid1, materialGridType1,
                                                 materialGrid2, materialGridType2,
                                                 materialGrid3, materialGridType3,
                                                 material1, material2);

  /*
   * TODO: precalculate coefficients
   */
  FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
  FPValue a1 = (4 + 2*gridTimeStep*material2) / A;
  FPValue a2 = -8 / A;
  FPValue a3 = (4 - 2*gridTimeStep*material2) / A;
  FPValue a4 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
  FPValue a5 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;

#if defined (TWO_TIME_STEPS)
  FieldValue val = calcFieldDrude (valField->getCurValue (),
                                   valField->getPrevValue (),
                                   valField->getPrevPrevValue (),
                                   valField1->getPrevValue (),
                                   valField1->getPrevPrevValue (),
                                   a1,
                                   a2,
                                   a3,
                                   a4,
                                   a5);
  valField1->setCurValue (val);
#else
  ALWAYS_ASSERT (0);
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepIteration (time_step t,
                                                               TC pos,
                                                               Grid<TC> *grid,
                                                               GridType gridType,
                                                               Grid<TC> *materialGrid,
                                                               GridType materialGridType,
                                                               Grid<TC> *oppositeGrid1,
                                                               Grid<TC> *oppositeGrid2,
                                                               SourceCallBack rightSideFunc,
                                                               FPValue materialModifier)
{
  FPValue eps0 = PhysicsConst::Eps0;

  // TODO: add getTotalPositionDiff here, which will be called before loop
  TC posAbs = grid->getTotalPosition (pos);
  // TODO: [possible] move 1D gridValues to 3D gridValues array
  FieldPointValue *valField = grid->getFieldPointValue (pos);

  FPValue material = materialGrid ? yeeLayout->getMaterial (posAbs, gridType, materialGrid, materialGridType) : 0;

  TC pos11;
  TC pos12;
  TC pos21;
  TC pos22;

  TCFP coordFP;
  FPValue timestep;

  FPValue k_mod;
  FPValue Ca;
  FPValue Cb;

  // TODO: add circuitElementDiff here, which will be called before loop
  // TODO: add coordFPDiff here, which will be called before loop
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      pos11 = yeeLayout->getExCircuitElement (pos, LayoutDirection::DOWN);
      pos12 = yeeLayout->getExCircuitElement (pos, LayoutDirection::UP);
      pos21 = yeeLayout->getExCircuitElement (pos, LayoutDirection::BACK);
      pos22 = yeeLayout->getExCircuitElement (pos, LayoutDirection::FRONT);

      // TODO: do not invoke in case no right side
      coordFP = yeeLayout->getExCoordFP (posAbs);
      timestep = t;

      FPValue k_y = 1;
      k_mod = k_y;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      pos11 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::BACK);
      pos12 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::FRONT);
      pos21 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::LEFT);
      pos22 = yeeLayout->getEyCircuitElement (pos, LayoutDirection::RIGHT);

      coordFP = yeeLayout->getEyCoordFP (posAbs);
      timestep = t;

      FPValue k_z = 1;
      k_mod = k_z;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      pos11 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::LEFT);
      pos12 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::RIGHT);
      pos21 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::DOWN);
      pos22 = yeeLayout->getEzCircuitElement (pos, LayoutDirection::UP);

      coordFP = yeeLayout->getEzCoordFP (posAbs);
      timestep = t;

      FPValue k_x = 1;
      k_mod = k_x;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      pos11 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::BACK);
      pos12 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::FRONT);
      pos21 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::DOWN);
      pos22 = yeeLayout->getHxCircuitElement (pos, LayoutDirection::UP);

      coordFP = yeeLayout->getHxCoordFP (posAbs);
      timestep = t + 0.5;

      FPValue k_y = 1;
      k_mod = k_y;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      pos11 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::LEFT);
      pos12 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::RIGHT);
      pos21 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::BACK);
      pos22 = yeeLayout->getHyCircuitElement (pos, LayoutDirection::FRONT);

      coordFP = yeeLayout->getHyCoordFP (posAbs);
      timestep = t + 0.5;

      FPValue k_z = 1;
      k_mod = k_z;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      pos11 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::DOWN);
      pos12 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::UP);
      pos21 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::LEFT);
      pos22 = yeeLayout->getHzCircuitElement (pos, LayoutDirection::RIGHT);

      coordFP = yeeLayout->getHzCoordFP (posAbs);
      timestep = t + 0.5;

      FPValue k_x = 1;
      k_mod = k_x;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (usePML)
  {
    Ca = (2 * eps0 * k_mod - material * gridTimeStep) / (2 * eps0 * k_mod + material * gridTimeStep);
    Cb = (2 * eps0 * gridTimeStep / gridStep) / (2 * eps0 * k_mod + material * gridTimeStep);
  }
  else
  {
    Ca = 1.0;
    Cb = gridTimeStep / (material * materialModifier * gridStep);
  }

  // TODO: separate previous grid and current
  FieldValue prev11 = FIELDVALUE (0, 0);
  FieldValue prev12 = FIELDVALUE (0, 0);
  FieldValue prev21 = FIELDVALUE (0, 0);
  FieldValue prev22 = FIELDVALUE (0, 0);

  if (oppositeGrid1)
  {
    FieldPointValue *val11 = oppositeGrid1->getFieldPointValue (pos11);
    FieldPointValue *val12 = oppositeGrid1->getFieldPointValue (pos12);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev11 = val11->getPrevValue ();
    prev12 = val12->getPrevValue ();
#else
    ALWAYS_ASSERT (0);
#endif
  }

  if (oppositeGrid2)
  {
    FieldPointValue *val21 = oppositeGrid2->getFieldPointValue (pos21);
    FieldPointValue *val22 = oppositeGrid2->getFieldPointValue (pos22);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    prev21 = val21->getPrevValue ();
    prev22 = val22->getPrevValue ();
#else
    ALWAYS_ASSERT (0);
#endif
  }

  if (solverSettings.getDoUseTFSF ())
  {
    calculateTFSF<grid_type> (posAbs, prev11, prev12, prev21, prev22, pos11, pos12, pos21, pos22);
  }

  FieldValue prevRightSide = 0;
  if (rightSideFunc != NULLPTR)
  {
    prevRightSide = rightSideFunc (expandTo3D (coordFP * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);
  }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  // TODO: precalculate Ca,Cb
  FieldValue val = calcField (valField->getPrevValue (),
                              prev12,
                              prev11,
                              prev22,
                              prev21,
                              prevRightSide,
                              Ca,
                              Cb,
                              gridStep);
#else
  ALWAYS_ASSERT (0);
#endif

  valField->setCurValue (val);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStepInit (Grid<TC> **grid, GridType *gridType, Grid<TC> **materialGrid, GridType *materialGridType, Grid<TC> **materialGrid1, GridType *materialGridType1,
Grid<TC> **materialGrid2, GridType *materialGridType2, Grid<TC> **materialGrid3, GridType *materialGridType3, Grid<TC> **materialGrid4, GridType *materialGridType4,
Grid<TC> **materialGrid5, GridType *materialGridType5, Grid<TC> **oppositeGrid1, Grid<TC> **oppositeGrid2, Grid<TC> **gridPML1, GridType *gridPMLType1, Grid<TC> **gridPML2, GridType *gridPMLType2,
SourceCallBack *rightSideFunc, SourceCallBack *borderFunc, SourceCallBack *exactFunc, FPValue *materialModifier)
{
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      ASSERT (doNeedEx);
      *grid = Ex;
      *gridType = GridType::EX;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hz;
      *oppositeGrid2 = Hy;

      *rightSideFunc = Jx;
      *borderFunc = ExBorder;
      *exactFunc = ExExact;

      if (usePML)
      {
        *grid = Dx;
        *gridType = GridType::DX;

        *gridPML1 = D1x;
        *gridPMLType1 = GridType::DX;

        *gridPML2 = Ex;
        *gridPMLType2 = GridType::EX;

        *materialGrid = SigmaY;
        *materialGridType = GridType::SIGMAY;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaX;
        *materialGridType4 = GridType::SIGMAX;

        *materialGrid5 = SigmaZ;
        *materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      ASSERT (doNeedEy);
      *grid = Ey;
      *gridType = GridType::EY;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hx;
      *oppositeGrid2 = Hz;

      *rightSideFunc = Jy;
      *borderFunc = EyBorder;
      *exactFunc = EyExact;

      if (usePML)
      {
        *grid = Dy;
        *gridType = GridType::DY;

        *gridPML1 = D1y;
        *gridPMLType1 = GridType::DY;

        *gridPML2 = Ey;
        *gridPMLType2 = GridType::EY;

        *materialGrid = SigmaZ;
        *materialGridType = GridType::SIGMAZ;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaY;
        *materialGridType4 = GridType::SIGMAY;

        *materialGrid5 = SigmaX;
        *materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      ASSERT (doNeedEz);
      *grid = Ez;
      *gridType = GridType::EZ;

      *materialGrid = Eps;
      *materialGridType = GridType::EPS;
      *materialModifier = PhysicsConst::Eps0;

      *oppositeGrid1 = Hy;
      *oppositeGrid2 = Hx;

      *rightSideFunc = Jz;
      *borderFunc = EzBorder;
      *exactFunc = EzExact;

      if (usePML)
      {
        *grid = Dz;
        *gridType = GridType::DZ;

        *gridPML1 = D1z;
        *gridPMLType1 = GridType::DZ;

        *gridPML2 = Ez;
        *gridPMLType2 = GridType::EZ;

        *materialGrid = SigmaX;
        *materialGridType = GridType::SIGMAX;

        *materialGrid1 = Eps;
        *materialGridType1 = GridType::EPS;

        *materialGrid4 = SigmaZ;
        *materialGridType4 = GridType::SIGMAZ;

        *materialGrid5 = SigmaY;
        *materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPE;
          *materialGridType2 = GridType::OMEGAPE;

          *materialGrid3 = GammaE;
          *materialGridType3 = GridType::GAMMAE;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      ASSERT (doNeedHx);
      *grid = Hx;
      *gridType = GridType::HX;

      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ey;
      *oppositeGrid2 = Ez;

      *rightSideFunc = Mx;
      *borderFunc = HxBorder;
      *exactFunc = HxExact;

      if (usePML)
      {
        *grid = Bx;
        *gridType = GridType::BX;

        *gridPML1 = B1x;
        *gridPMLType1 = GridType::BX;

        *gridPML2 = Hx;
        *gridPMLType2 = GridType::HX;

        *materialGrid = SigmaY;
        *materialGridType = GridType::SIGMAY;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaX;
        *materialGridType4 = GridType::SIGMAX;

        *materialGrid5 = SigmaZ;
        *materialGridType5 = GridType::SIGMAZ;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      ASSERT (doNeedHy);
      *grid = Hy;
      *gridType = GridType::HY;

      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ez;
      *oppositeGrid2 = Ex;

      *rightSideFunc = My;
      *borderFunc = HyBorder;
      *exactFunc = HyExact;

      if (usePML)
      {
        *grid = By;
        *gridType = GridType::BY;

        *gridPML1 = B1y;
        *gridPMLType1 = GridType::BY;

        *gridPML2 = Hy;
        *gridPMLType2 = GridType::HY;

        *materialGrid = SigmaZ;
        *materialGridType = GridType::SIGMAZ;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaY;
        *materialGridType4 = GridType::SIGMAY;

        *materialGrid5 = SigmaX;
        *materialGridType5 = GridType::SIGMAX;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      ASSERT (doNeedHz);
      *grid = Hz;
      *gridType = GridType::HZ;
      *materialGrid = Mu;
      *materialGridType = GridType::MU;
      *materialModifier = PhysicsConst::Mu0;

      *oppositeGrid1 = Ex;
      *oppositeGrid2 = Ey;

      *rightSideFunc = Mz;
      *borderFunc = HzBorder;
      *exactFunc = HzExact;

      if (usePML)
      {
        *grid = Bz;
        *gridType = GridType::BZ;

        *gridPML1 = B1z;
        *gridPMLType1 = GridType::BZ;

        *gridPML2 = Hz;
        *gridPMLType2 = GridType::HZ;

        *materialGrid = SigmaX;
        *materialGridType = GridType::SIGMAX;

        *materialGrid1 = Mu;
        *materialGridType1 = GridType::MU;

        *materialGrid4 = SigmaZ;
        *materialGridType4 = GridType::SIGMAZ;

        *materialGrid5 = SigmaY;
        *materialGridType5 = GridType::SIGMAY;

        if (useMetamaterials)
        {
          *materialGrid2 = OmegaPM;
          *materialGridType2 = GridType::OMEGAPM;

          *materialGrid3 = GammaM;
          *materialGridType3 = GridType::GAMMAM;
        }
      }
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
void
Scheme<Type, TCoord, layout_type>::calculateFieldStep (time_step t, TC start, TC end)
{
  // TODO: add metamaterials without pml
  if (!usePML && useMetamaterials)
  {
    UNREACHABLE;
  }

  FPValue eps0 = PhysicsConst::Eps0;

  Grid<TC> *grid = NULLPTR;
  GridType gridType = GridType::NONE;

  Grid<TC> *materialGrid = NULLPTR;
  GridType materialGridType = GridType::NONE;

  Grid<TC> *materialGrid1 = NULLPTR;
  GridType materialGridType1 = GridType::NONE;

  Grid<TC> *materialGrid2 = NULLPTR;
  GridType materialGridType2 = GridType::NONE;

  Grid<TC> *materialGrid3 = NULLPTR;
  GridType materialGridType3 = GridType::NONE;

  Grid<TC> *materialGrid4 = NULLPTR;
  GridType materialGridType4 = GridType::NONE;

  Grid<TC> *materialGrid5 = NULLPTR;
  GridType materialGridType5 = GridType::NONE;

  Grid<TC> *oppositeGrid1 = NULLPTR;
  Grid<TC> *oppositeGrid2 = NULLPTR;

  Grid<TC> *gridPML1 = NULLPTR;
  GridType gridPMLType1 = GridType::NONE;

  Grid<TC> *gridPML2 = NULLPTR;
  GridType gridPMLType2 = GridType::NONE;

  SourceCallBack rightSideFunc = NULLPTR;
  SourceCallBack borderFunc = NULLPTR;
  SourceCallBack exactFunc = NULLPTR;

  /*
   * TODO: remove this, multiply on this at initialization
   */
  FPValue materialModifier;
  calculateFieldStepInit<grid_type, usePML, useMetamaterials> (&grid, &gridType,
    &materialGrid, &materialGridType, &materialGrid1, &materialGridType1, &materialGrid2, &materialGridType2,
    &materialGrid3, &materialGridType3, &materialGrid4, &materialGridType4, &materialGrid5, &materialGridType5,
    &oppositeGrid1, &oppositeGrid2, &gridPML1, &gridPMLType1, &gridPML2, &gridPMLType2,
    &rightSideFunc, &borderFunc, &exactFunc, &materialModifier);

  GridCoordinate3D start3D;
  GridCoordinate3D end3D;

  expandTo3DStartEnd (start, end, start3D, end3D, ct1, ct2, ct3);

  // TODO: remove this check for each iteration
  if (t > 0)
  {
    for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
      {
        // TODO: check that this is optimized out in case 2D mode
        for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIteration<grid_type, usePML> (t, pos, grid, gridType, materialGrid, materialGridType,
                                                          oppositeGrid1, oppositeGrid2, rightSideFunc, materialModifier);
        }
      }
    }

    if (usePML)
    {
      if (useMetamaterials)
      {
#ifdef TWO_TIME_STEPS
        for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
          {
            // TODO: check that this loop is optimized out
            for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
            {
              TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
              calculateFieldStepIterationPMLMetamaterials (t, pos, grid, gridPML1, gridType,
                materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
                materialModifier);
            }
          }
        }
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of two steps in time. Recompile it with -DTIME_STEPS=2.");
#endif
      }

      for (grid_coord i = start3D.get1 (); i < end3D.get1 (); ++i)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord j = start3D.get2 (); j < end3D.get2 (); ++j)
        {
          // TODO: check that this loop is optimized out
          for (grid_coord k = start3D.get3 (); k < end3D.get3 (); ++k)
          {
            TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
            calculateFieldStepIterationPML<useMetamaterials> (t, pos, grid, gridPML1, gridPML2, gridType, gridPMLType1,
              materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
              materialModifier);
          }
        }
      }
    }
  }

  if (borderFunc != NULLPTR)
  {
    GridCoordinate3D startBorder;
    GridCoordinate3D endBorder;

    expandTo3DStartEnd (TC::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3),
                        grid->getSize (),
                        startBorder,
                        endBorder,
                        ct1, ct2, ct3);

    for (grid_coord i = startBorder.get1 (); i < endBorder.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startBorder.get2 (); j < endBorder.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startBorder.get3 (); k < endBorder.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIterationBorder<grid_type> (t, pos, grid, borderFunc);
        }
      }
    }
  }

  if (exactFunc != NULLPTR)
  {
    FPValue normRe = 0.0;
    FPValue normIm = 0.0;
    FPValue normMod = 0.0;

    FPValue maxRe = 0.0;
    FPValue maxIm = 0.0;
    FPValue maxMod = 0.0;

    GridCoordinate3D startNorm = start3D;
    GridCoordinate3D endNorm = end3D;

    if (solverSettings.getExactSolutionCompareStartX () != 0)
    {
      startNorm.set1 (solverSettings.getExactSolutionCompareStartX ());
    }
    if (solverSettings.getExactSolutionCompareStartY () != 0)
    {
      startNorm.set2 (solverSettings.getExactSolutionCompareStartY ());
    }
    if (solverSettings.getExactSolutionCompareStartZ () != 0)
    {
      startNorm.set3 (solverSettings.getExactSolutionCompareStartZ ());
    }

    if (solverSettings.getExactSolutionCompareEndX () != 0)
    {
      endNorm.set1 (solverSettings.getExactSolutionCompareEndX ());
    }
    if (solverSettings.getExactSolutionCompareEndY () != 0)
    {
      endNorm.set2 (solverSettings.getExactSolutionCompareEndY ());
    }
    if (solverSettings.getExactSolutionCompareEndZ () != 0)
    {
      endNorm.set3 (solverSettings.getExactSolutionCompareEndZ ());
    }

    Grid<TC> *normGrid = grid;
    if (usePML)
    {
      grid = gridPML2;
    }

    for (grid_coord i = startNorm.get1 (); i < endNorm.get1 (); ++i)
    {
      // TODO: check that this loop is optimized out
      for (grid_coord j = startNorm.get2 (); j < endNorm.get2 (); ++j)
      {
        // TODO: check that this loop is optimized out
        for (grid_coord k = startNorm.get3 (); k < endNorm.get3 (); ++k)
        {
          TC pos = TC::initAxesCoordinate (i, j, k, ct1, ct2, ct3);
          calculateFieldStepIterationExact<grid_type> (t, pos, grid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);
        }
      }
    }

#ifdef COMPLEX_FIELD_VALUES
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());
    normIm = sqrt (normIm / grid->getSize ().calculateTotalCoord ());
    normMod = sqrt (normMod / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " , " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% , " FP_MOD_ACC " %% ), module = " FP_MOD_ACC " = ( " FP_MOD_ACC " %% )\n",
      grid->getName ().c_str (), t, normRe, normIm, normRe * 100.0 / maxRe, normIm * 100.0 / maxIm, normMod, normMod * 100.0 / maxMod);
#else
    normRe = sqrt (normRe / grid->getSize ().calculateTotalCoord ());

    /*
     * NOTE: do not change this! test suite depdends on the order of values in output
     */
    printf ("-> DIFF NORM %s. Timestep %u. Value = ( " FP_MOD_ACC " ) = ( " FP_MOD_ACC " %% )\n",
      grid->getName ().c_str (), t, normRe, normRe * 100.0 / maxRe);
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performPlaneWaveESteps (time_step t)
{
  grid_coord size = EInc->getSize ().get1 ();

  ASSERT (size > 0);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Eps0 * gridStep);

  for (grid_coord i = 1; i < size; ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , CoordinateType::X
#endif
                          );

    FieldPointValue *valE = EInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i - 1
#ifdef DEBUG_INFO
                              , CoordinateType::X
#endif
                              );
    GridCoordinate1D posRight (i
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif
                               );

    FieldPointValue *valH1 = HInc->getFieldPointValue (posLeft);
    FieldPointValue *valH2 = HInc->getFieldPointValue (posRight);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue val = valE->getPrevValue () + modifier * (valH1->getPrevValue () - valH2->getPrevValue ());
#else
    ALWAYS_ASSERT (0);
#endif

    valE->setCurValue (val);
  }

  GridCoordinate1D pos (0
#ifdef DEBUG_INFO
                        , CoordinateType::X
#endif
                        );
  FieldPointValue *valE = EInc->getFieldPointValue (pos);

  FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;

#ifdef COMPLEX_FIELD_VALUES
  valE->setCurValue (FieldValue (sin (arg), cos (arg)));
#else /* COMPLEX_FIELD_VALUES */
  valE->setCurValue (sin (arg));
#endif /* !COMPLEX_FIELD_VALUES */

#ifdef ENABLE_ASSERTS
  GridCoordinate1D posEnd (size - 1, CoordinateType::X);
  ALWAYS_ASSERT (EInc->getFieldPointValue (posEnd)->getCurValue () == getFieldValueRealOnly (0.0));
#endif

  EInc->nextTimeStep ();
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performPlaneWaveHSteps (time_step t)
{
  grid_coord size = HInc->getSize ().get1 ();

  ASSERT (size > 1);

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Mu0 * gridStep);

  for (grid_coord i = 0; i < size - 1; ++i)
  {
    GridCoordinate1D pos (i
#ifdef DEBUG_INFO
                          , CoordinateType::X
#endif
                          );

    FieldPointValue *valH = HInc->getFieldPointValue (pos);

    GridCoordinate1D posLeft (i
#ifdef DEBUG_INFO
                              , CoordinateType::X
#endif
                              );
    GridCoordinate1D posRight (i + 1
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif
                               );

    FieldPointValue *valE1 = EInc->getFieldPointValue (posLeft);
    FieldPointValue *valE2 = EInc->getFieldPointValue (posRight);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue val = valH->getPrevValue () + modifier * (valE1->getPrevValue () - valE2->getPrevValue ());
#else
    ALWAYS_ASSERT (0);
#endif

    valH->setCurValue (val);
  }

#ifdef ENABLE_ASSERTS
  GridCoordinate1D pos (size - 2, CoordinateType::X);
  ALWAYS_ASSERT (HInc->getFieldPointValue (pos)->getCurValue () == getFieldValueRealOnly (0.0));
#endif

  HInc->nextTimeStep ();
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FieldValue
Scheme<Type, TCoord, layout_type>::approximateIncidentWaveE (TCFP pos)
{
  YeeGridLayout<Type, TCoord, layout_type> *layout = Scheme<Type, TCoord, layout_type>::yeeLayout;
  return SchemeHelper::approximateIncidentWaveE<Type, TCoord> (pos, layout->getZeroIncCoordFP (), EInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
}
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FieldValue
Scheme<Type, TCoord, layout_type>::approximateIncidentWaveH (TCFP pos)
{
  YeeGridLayout<Type, TCoord, layout_type> *layout = Scheme<Type, TCoord, layout_type>::yeeLayout;
  return SchemeHelper::approximateIncidentWaveH<Type, TCoord> (pos, layout->getZeroIncCoordFP (), HInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
}

template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == 0);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue z = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == 0 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = z - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate> (GridCoordinateFP1D realCoord, GridCoordinateFP1D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2 && incAngle2 == 0);

  FPValue d = x - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue y = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == PhysicsConst::Pi / 2);

  FPValue d = y * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue z = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle2 == 0);

  FPValue d = x * sin (incAngle1) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate> (GridCoordinateFP2D realCoord, GridCoordinateFP2D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  ASSERT (incAngle1 == PhysicsConst::Pi / 2);

  FPValue d = x * cos (incAngle2) + y * sin (incAngle2) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}
template <>
FieldValue
SchemeHelper::approximateIncidentWave<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (GridCoordinateFP3D realCoord, GridCoordinateFP3D zeroCoordFP,
                                                   FPValue dDiff, Grid<GridCoordinate1D> *FieldInc, FPValue incAngle1, FPValue incAngle2)
{
  FPValue x = realCoord.get1 () - zeroCoordFP.get1 ();
  FPValue y = realCoord.get2 () - zeroCoordFP.get2 ();
  FPValue z = realCoord.get3 () - zeroCoordFP.get3 ();

  FPValue d = x * sin (incAngle1) * cos (incAngle2) + y * sin (incAngle1) * sin (incAngle2) + z * cos (incAngle1) - dDiff;
  ASSERT (d > 0);

  return SchemeHelper::approximateIncidentWaveHelper (d, FieldInc);
}

/*
 * Specialization for Sigma
 */
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
};

template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
};

template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaY);
  SchemeHelper::initSigmaZ<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, SigmaZ);
};

template <SchemeType_t Type, template <typename, bool> class TCoord>
FieldValue
SchemeHelper::approximateIncidentWaveE (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, Grid<GridCoordinate1D> *EInc, FPValue incAngle1, FPValue incAngle2)
{
  return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.0, EInc, incAngle1, incAngle2);
}

template <SchemeType_t Type, template <typename, bool> class TCoord>
FieldValue
SchemeHelper::approximateIncidentWaveH (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, Grid<GridCoordinate1D> *HInc, FPValue incAngle1, FPValue incAngle2)
{
  return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.5, HInc, incAngle1, incAngle2);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performNSteps (time_step startStep, time_step numberTimeSteps)
{
  time_step diffT = solverSettings.getRebalanceStep ();

  int processId = 0;

  time_step stepLimit = startStep + numberTimeSteps;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (processId == 0)
  {
    DPRINTF (LOG_LEVEL_STAGES, "Performing computations for [%u,%u] time steps.\n", startStep, stepLimit);
  }

  for (time_step t = startStep; t < stepLimit; ++t)
  {
    if (processId == 0)
    {
      DPRINTF (LOG_LEVEL_STAGES, "Calculating time step %u...\n", t);
    }

    TC ExStart = doNeedEx ? Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC ExEnd = doNeedEx ? Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC EyStart = doNeedEy ? Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC EyEnd = doNeedEy ? Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC EzStart = doNeedEz ? Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC EzEnd = doNeedEz ? Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HxStart = doNeedHx ? Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HxEnd = doNeedHx ? Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HyStart = doNeedHy ? Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HyEnd = doNeedHy ? Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HzStart = doNeedHz ? Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HzEnd = doNeedHz ? Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    if (useParallel && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (solverSettings.getDoUseTFSF ())
    {
      performPlaneWaveESteps (t);
    }

    if (doNeedEx)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
    }
    if (doNeedEy)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
    }
    if (doNeedEz)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
    }

    if (useParallel && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (doNeedEx)
    {
      Ex->nextTimeStep ();
    }
    if (doNeedEy)
    {
      Ey->nextTimeStep ();
    }
    if (doNeedEz)
    {
      Ez->nextTimeStep ();
    }

    if (solverSettings.getDoUsePML ())
    {
      if (doNeedEx)
      {
        Dx->nextTimeStep ();
      }
      if (doNeedEy)
      {
        Dy->nextTimeStep ();
      }
      if (doNeedEz)
      {
        Dz->nextTimeStep ();
      }
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (doNeedEx)
      {
        D1x->nextTimeStep ();
      }
      if (doNeedEy)
      {
        D1y->nextTimeStep ();
      }
      if (doNeedEz)
      {
        D1z->nextTimeStep ();
      }
    }

    if (useParallel && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (solverSettings.getDoUseTFSF ())
    {
      performPlaneWaveHSteps (t);
    }

    if (doNeedHx)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
    }
    if (doNeedHy)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
    }
    if (doNeedHz)
    {
      performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
    }

    if (useParallel && solverSettings.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (doNeedHx)
    {
      Hx->nextTimeStep ();
    }
    if (doNeedHy)
    {
      Hy->nextTimeStep ();
    }
    if (doNeedHz)
    {
      Hz->nextTimeStep ();
    }

    if (solverSettings.getDoUsePML ())
    {
      if (doNeedHx)
      {
        Bx->nextTimeStep ();
      }
      if (doNeedHy)
      {
        By->nextTimeStep ();
      }
      if (doNeedHz)
      {
        Bz->nextTimeStep ();
      }
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (doNeedHx)
      {
        B1x->nextTimeStep ();
      }
      if (doNeedHy)
      {
        B1y->nextTimeStep ();
      }
      if (doNeedHz)
      {
        B1z->nextTimeStep ();
      }
    }

    if (solverSettings.getDoSaveIntermediateRes ()
        && t % solverSettings.getIntermediateSaveStep () == 0)
    {
      gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldIntermediate ());
      saveGrids (t);
    }

    if (solverSettings.getDoUseNTFF ()
        && t > 0 && t % solverSettings.getIntermediateNTFFStep () == 0)
    {
      saveNTFF (solverSettings.getDoCalcReverseNTFF (), t);
    }

    additionalUpdateOfGrids (t, diffT);
  }

  if (solverSettings.getDoSaveRes ())
  {
    gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldRes ());
    saveGrids (stepLimit);
  }
}
//
// template <SchemeType_t Type, template <typename, bool> class TCoord, typename Layout>
// void
// Scheme<Type, TCoord, Layout>::performAmplitudeSteps (time_step startStep)
// {
// #ifdef COMPLEX_FIELD_VALUES
//   UNREACHABLE;
// #else /* COMPLEX_FIELD_VALUES */
//
//   ASSERT_MESSAGE ("Temporary unsupported");
//
//   int processId = 0;
//
//   if (solverSettings.getDoUseParallelGrid ())
//   {
// #ifdef PARALLEL_GRID
//     processId = ParallelGrid::getParallelCore ()->getProcessId ();
// #else /* PARALLEL_GRID */
//     ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
// #endif /* !PARALLEL_GRID */
//   }
//
//   int is_stable_state = 0;
//
//   GridCoordinate3D EzSize = Ez->getSize ();
//
//   time_step t = startStep;
//
//   while (is_stable_state == 0 && t < solverSettings.getNumAmplitudeSteps ())
//   {
//     FPValue maxAccuracy = -1;
//
//     //is_stable_state = 1;
//
//     GridCoordinate3D ExStart = Ex->getComputationStart (yeeLayout->getExStartDiff ());
//     GridCoordinate3D ExEnd = Ex->getComputationEnd (yeeLayout->getExEndDiff ());
//
//     GridCoordinate3D EyStart = Ey->getComputationStart (yeeLayout->getEyStartDiff ());
//     GridCoordinate3D EyEnd = Ey->getComputationEnd (yeeLayout->getEyEndDiff ());
//
//     GridCoordinate3D EzStart = Ez->getComputationStart (yeeLayout->getEzStartDiff ());
//     GridCoordinate3D EzEnd = Ez->getComputationEnd (yeeLayout->getEzEndDiff ());
//
//     GridCoordinate3D HxStart = Hx->getComputationStart (yeeLayout->getHxStartDiff ());
//     GridCoordinate3D HxEnd = Hx->getComputationEnd (yeeLayout->getHxEndDiff ());
//
//     GridCoordinate3D HyStart = Hy->getComputationStart (yeeLayout->getHyStartDiff ());
//     GridCoordinate3D HyEnd = Hy->getComputationEnd (yeeLayout->getHyEndDiff ());
//
//     GridCoordinate3D HzStart = Hz->getComputationStart (yeeLayout->getHzStartDiff ());
//     GridCoordinate3D HzEnd = Hz->getComputationEnd (yeeLayout->getHzEndDiff ());
//
//     if (solverSettings.getDoUseTFSF ())
//     {
//       performPlaneWaveESteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
//
//     for (int i = ExStart.get1 (); i < ExEnd.get1 (); ++i)
//     {
//       for (int j = ExStart.get2 (); j < ExEnd.get2 (); ++j)
//       {
//         for (int k = ExStart.get3 (); k < ExEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isExInPML (Ex->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Ex->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = ExAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EyStart.get1 (); i < EyEnd.get1 (); ++i)
//     {
//       for (int j = EyStart.get2 (); j < EyEnd.get2 (); ++j)
//       {
//         for (int k = EyStart.get3 (); k < EyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEyInPML (Ey->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Ey->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = EyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EzStart.get1 (); i < EzEnd.get1 (); ++i)
//     {
//       for (int j = EzStart.get2 (); j < EzEnd.get2 (); ++j)
//       {
//         for (int k = EzStart.get3 (); k < EzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEzInPML (Ez->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Ez->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = EzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     Ex->nextTimeStep ();
//     Ey->nextTimeStep ();
//     Ez->nextTimeStep ();
//
//     if (solverSettings.getDoUsePML ())
//     {
//       Dx->nextTimeStep ();
//       Dy->nextTimeStep ();
//       Dz->nextTimeStep ();
//     }
//
//     if (solverSettings.getDoUseTFSF ())
//     {
//       performPlaneWaveHSteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
//
//     for (int i = HxStart.get1 (); i < HxEnd.get1 (); ++i)
//     {
//       for (int j = HxStart.get2 (); j < HxEnd.get2 (); ++j)
//       {
//         for (int k = HxStart.get3 (); k < HxEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHxInPML (Hx->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Hx->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = HxAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HyStart.get1 (); i < HyEnd.get1 (); ++i)
//     {
//       for (int j = HyStart.get2 (); j < HyEnd.get2 (); ++j)
//       {
//         for (int k = HyStart.get3 (); k < HyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHyInPML (Hy->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Hy->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = HyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HzStart.get1 (); i < HzEnd.get1 (); ++i)
//     {
//       for (int j = HzStart.get2 (); j < HzEnd.get2 (); ++j)
//       {
//         for (int k = HzStart.get3 (); k < HzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHzInPML (Hz->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = Hz->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = HzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     Hx->nextTimeStep ();
//     Hy->nextTimeStep ();
//     Hz->nextTimeStep ();
//
//     if (solverSettings.getDoUsePML ())
//     {
//       Bx->nextTimeStep ();
//       By->nextTimeStep ();
//       Bz->nextTimeStep ();
//     }
//
//     ++t;
//
//     if (maxAccuracy < 0)
//     {
//       is_stable_state = 0;
//     }
//
//     DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " FP_MOD ". \n", t, maxAccuracy);
//   }
//
//   if (is_stable_state == 0)
//   {
//     ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.");
//   }
//
// #endif /* !COMPLEX_FIELD_VALUES */
// }

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
int
Scheme<Type, TCoord, layout_type>::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

  int is_stable_state = 1;

  FPValue valAmp = amplitudeValue->getCurValue ();

  val = val >= 0 ? val : -val;

  if (val >= valAmp)
  {
    FPValue accuracy = val - valAmp;
    if (valAmp != 0)
    {
      accuracy /= valAmp;
    }
    else if (val != 0)
    {
      accuracy /= val;
    }

    if (accuracy > PhysicsConst::accuracy)
    {
      is_stable_state = 0;

      amplitudeValue->setCurValue (val);
    }

    if (accuracy > *maxAccuracy)
    {
      *maxAccuracy = accuracy;
    }
  }

  return is_stable_state;
#endif /* !COMPLEX_FIELD_VALUES */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performSteps ()
{
#if defined (CUDA_ENABLED)

  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (solverSettings.getDoUsePML ()
      || solverSettings.getDoUseTFSF ()
      || solverSettings.getDoUseAmplitudeMode ()
      || solverSettings.getDoUseMetamaterials ())
  {
    ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");
  }

  CudaExitStatus status;

  cudaExecute3DSteps (&status, yeeLayout, gridTimeStep, gridStep, Ex, Ey, Ez, Hx, Hy, Hz, Eps, Mu, totalStep, processId);

  ASSERT (status == CUDA_OK);

  if (solverSettings.getDoSaveRes ())
  {
    gatherFieldsTotal (solverSettings.getDoSaveScatteredFieldRes ());
    saveGrids (totalStep);
  }

#else /* CUDA_ENABLED */

  if (solverSettings.getDoUseMetamaterials () && !solverSettings.getDoUsePML ())
  {
    ASSERT_MESSAGE ("Metamaterials without pml are not implemented");
  }

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    if (solverSettings.getDoUseAmplitudeMode ())
    {
      ASSERT_MESSAGE ("Parallel amplitude mode is not implemented");
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  performNSteps (0, totalStep);

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    UNREACHABLE;
    //performAmplitudeSteps (totalStep);
  }

#endif /* !CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initScheme (FPValue dx, FPValue sourceWaveLen)
{
  sourceWaveLength = sourceWaveLen;
  sourceFrequency = PhysicsConst::SpeedOfLight / sourceWaveLength;

  gridStep = dx;
  courantNum = solverSettings.getCourantNum ();
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  ALWAYS_ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

  FPValue phaseVelocity0 = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, PhysicsConst::Pi / 2, 0);
  FPValue phaseVelocity = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ());
  FPValue k = 2 * PhysicsConst::Pi * PhysicsConst::SpeedOfLight / sourceWaveLength / phaseVelocity0;

  relPhaseVelocity = phaseVelocity0 / phaseVelocity;
  sourceWaveLengthNumerical = 2 * PhysicsConst::Pi / k;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "initScheme: "
                                      "\n\tphase velocity relation -> %f "
                                      "\n\tphase velosity 0 -> %f "
                                      "\n\tphase velocity -> %f "
                                      "\n\tanalytical wave number -> %.20f "
                                      "\n\tnumerical wave number -> %.20f"
                                      "\n\tanalytical wave length -> %.20f"
                                      "\n\tnumerical wave length -> %.20f"
                                      "\n\tnumerical grid step -> %.20f"
                                      "\n\tnumerical time step -> %.20f"
                                      "\n\twave length -> %.20f"
                                      "\n",
           relPhaseVelocity, phaseVelocity0, phaseVelocity, 2*PhysicsConst::Pi/sourceWaveLength, k,
           sourceWaveLength, sourceWaveLengthNumerical, gridStep, gridTimeStep, sourceFrequency);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initCallBacks ()
{
#ifndef COMPLEX_FIELD_VALUES
  if (solverSettings.getDoUsePolinom1BorderCondition ())
  {
    EzBorder = CallBack::polinom1_ez;
    HyBorder = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoUsePolinom2BorderCondition ())
  {
    ExBorder = CallBack::polinom2_ex;
    EyBorder = CallBack::polinom2_ey;
    EzBorder = CallBack::polinom2_ez;

    HxBorder = CallBack::polinom2_hx;
    HyBorder = CallBack::polinom2_hy;
    HzBorder = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoUsePolinom3BorderCondition ())
  {
    EzBorder = CallBack::polinom3_ez;
    HyBorder = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoUseSin1BorderCondition ())
  {
    EzBorder = CallBack::sin1_ez;
    HyBorder = CallBack::sin1_hy;
  }

  if (solverSettings.getDoUsePolinom1StartValues ())
  {
    EzInitial = CallBack::polinom1_ez;
    HyInitial = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoUsePolinom2StartValues ())
  {
    ExInitial = CallBack::polinom2_ex;
    EyInitial = CallBack::polinom2_ey;
    EzInitial = CallBack::polinom2_ez;

    HxInitial = CallBack::polinom2_hx;
    HyInitial = CallBack::polinom2_hy;
    HzInitial = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoUsePolinom3StartValues ())
  {
    EzInitial = CallBack::polinom3_ez;
    HyInitial = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoUseSin1StartValues ())
  {
    EzInitial = CallBack::sin1_ez;
    HyInitial = CallBack::sin1_hy;
  }

  if (solverSettings.getDoUsePolinom1RightSide ())
  {
    Jz = CallBack::polinom1_jz;
    My = CallBack::polinom1_my;
  }
  else if (solverSettings.getDoUsePolinom2RightSide ())
  {
    Jx = CallBack::polinom2_jx;
    Jy = CallBack::polinom2_jy;
    Jz = CallBack::polinom2_jz;

    Mx = CallBack::polinom2_mx;
    My = CallBack::polinom2_my;
    Mz = CallBack::polinom2_mz;
  }
  else if (solverSettings.getDoUsePolinom3RightSide ())
  {
    Jz = CallBack::polinom3_jz;
    My = CallBack::polinom3_my;
  }

  if (solverSettings.getDoCalculatePolinom1DiffNorm ())
  {
    EzExact = CallBack::polinom1_ez;
    HyExact = CallBack::polinom1_hy;
  }
  else if (solverSettings.getDoCalculatePolinom2DiffNorm ())
  {
    ExExact = CallBack::polinom2_ex;
    EyExact = CallBack::polinom2_ey;
    EzExact = CallBack::polinom2_ez;

    HxExact = CallBack::polinom2_hx;
    HyExact = CallBack::polinom2_hy;
    HzExact = CallBack::polinom2_hz;
  }
  else if (solverSettings.getDoCalculatePolinom3DiffNorm ())
  {
    EzExact = CallBack::polinom3_ez;
    HyExact = CallBack::polinom3_hy;
  }
  else if (solverSettings.getDoCalculateSin1DiffNorm ())
  {
    EzExact = CallBack::sin1_ez;
    HyExact = CallBack::sin1_hy;
  }
#endif

  if (solverSettings.getDoCalculateExp1ExHyDiffNorm ())
  {
    ExExact = CallBack::exp1_ex_exhy;
    HyExact = CallBack::exp1_hy_exhy;
  }
  else if (solverSettings.getDoCalculateExp2ExHyDiffNorm ())
  {
    ExExact = CallBack::exp2_ex_exhy;
    HyExact = CallBack::exp2_hy_exhy;
  }
  else if (solverSettings.getDoCalculateExp3ExHyDiffNorm ())
  {
    ExExact = CallBack::exp3_ex_exhy;
    HyExact = CallBack::exp3_hy_exhy;
  }

  if (solverSettings.getDoCalculateExp1ExHzDiffNorm ())
  {
    ExExact = CallBack::exp1_ex_exhz;
    HzExact = CallBack::exp1_hz_exhz;
  }
  else if (solverSettings.getDoCalculateExp2ExHzDiffNorm ())
  {
    ExExact = CallBack::exp2_ex_exhz;
    HzExact = CallBack::exp2_hz_exhz;
  }
  else if (solverSettings.getDoCalculateExp3ExHzDiffNorm ())
  {
    ExExact = CallBack::exp3_ex_exhz;
    HzExact = CallBack::exp3_hz_exhz;
  }

  if (solverSettings.getDoCalculateExp1EyHxDiffNorm ())
  {
    EyExact = CallBack::exp1_ey_eyhx;
    HxExact = CallBack::exp1_hx_eyhx;
  }
  else if (solverSettings.getDoCalculateExp2EyHxDiffNorm ())
  {
    EyExact = CallBack::exp2_ey_eyhx;
    HxExact = CallBack::exp2_hx_eyhx;
  }
  else if (solverSettings.getDoCalculateExp3EyHxDiffNorm ())
  {
    EyExact = CallBack::exp3_ey_eyhx;
    HxExact = CallBack::exp3_hx_eyhx;
  }

  if (solverSettings.getDoCalculateExp1EyHzDiffNorm ())
  {
    EyExact = CallBack::exp1_ey_eyhz;
    HzExact = CallBack::exp1_hz_eyhz;
  }
  else if (solverSettings.getDoCalculateExp2EyHzDiffNorm ())
  {
    EyExact = CallBack::exp2_ey_eyhz;
    HzExact = CallBack::exp2_hz_eyhz;
  }
  else if (solverSettings.getDoCalculateExp3EyHzDiffNorm ())
  {
    EyExact = CallBack::exp3_ey_eyhz;
    HzExact = CallBack::exp3_hz_eyhz;
  }

  if (solverSettings.getDoCalculateExp1EzHxDiffNorm ())
  {
    EzExact = CallBack::exp1_ez_ezhx;
    HxExact = CallBack::exp1_hx_ezhx;
  }
  else if (solverSettings.getDoCalculateExp2EzHxDiffNorm ())
  {
    EzExact = CallBack::exp2_ez_ezhx;
    HxExact = CallBack::exp2_hx_ezhx;
  }
  else if (solverSettings.getDoCalculateExp3EzHxDiffNorm ())
  {
    EzExact = CallBack::exp3_ez_ezhx;
    HxExact = CallBack::exp3_hx_ezhx;
  }

  if (solverSettings.getDoCalculateExp1EzHyDiffNorm ())
  {
    EzExact = CallBack::exp1_ez_ezhy;
    HyExact = CallBack::exp1_hy_ezhy;
  }
  else if (solverSettings.getDoCalculateExp2EzHyDiffNorm ())
  {
    EzExact = CallBack::exp2_ez_ezhy;
    HyExact = CallBack::exp2_hy_ezhy;
  }
  else if (solverSettings.getDoCalculateExp3EzHyDiffNorm ())
  {
    EzExact = CallBack::exp3_ez_ezhy;
    HyExact = CallBack::exp3_hy_ezhy;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initMaterialFromFile (GridType gridType, Grid<TC> *grid, Grid<TC> *totalGrid)
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  std::string filename;

  switch (gridType)
  {
    case GridType::EPS:
    {
      filename = solverSettings.getEpsFileName ();
      break;
    }
    case GridType::MU:
    {
      filename = solverSettings.getMuFileName ();
      break;
    }
    case GridType::OMEGAPE:
    {
      filename = solverSettings.getOmegaPEFileName ();
      break;
    }
    case GridType::OMEGAPM:
    {
      filename = solverSettings.getOmegaPMFileName ();
      break;
    }
    case GridType::GAMMAE:
    {
      filename = solverSettings.getGammaEFileName ();
      break;
    }
    case GridType::GAMMAM:
    {
      filename = solverSettings.getGammaMFileName ();
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (filename.empty ())
  {
    return;
  }

  FileType type = GridFileManager::getFileType (filename);
  loader[type]->initManual (0, CURRENT, processId, filename, "", "");
  loader[type]->loadGrid (totalGrid);

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = grid->calculatePositionFromIndex (i);
      TC posAbs = grid->getTotalPosition (pos);

      FieldPointValue *val = grid->getFieldPointValue (pos);
      *val = *totalGrid->getFieldPointValue (posAbs);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGridWithInitialVals (GridType gridType, Grid<TC> *grid, FPValue timestep)
{
  SourceCallBack cb = NULLPTR;

  switch (gridType)
  {
    case GridType::EX:
    {
      cb = ExInitial;
      break;
    }
    case GridType::EY:
    {
      cb = EyInitial;
      break;
    }
    case GridType::EZ:
    {
      cb = EzInitial;
      break;
    }
    case GridType::HX:
    {
      cb = HxInitial;
      break;
    }
    case GridType::HY:
    {
      cb = HyInitial;
      break;
    }
    case GridType::HZ:
    {
      cb = HzInitial;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (cb == NULLPTR)
  {
    return;
  }

  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    TC pos = grid->calculatePositionFromIndex (i);
    TC posAbs = grid->getTotalPosition (pos);
    TCFP realCoord;

    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    grid->getFieldPointValue (pos)->setCurValue (cb (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep));
  }
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids3D (Eps, totalEps,
                                         Mu, totalMu,
                                         OmegaPE, totalOmegaPE,
                                         OmegaPM, totalOmegaPM,
                                         GammaE, totalGammaE,
                                         GammaM, totalGammaM);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGrids ()
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    if (!solverSettings.getEpsFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalEps->initialize ();
    }
    if (!solverSettings.getMuFileName ().empty () || solverSettings.getDoSaveMaterials ())
    {
      totalMu->initialize ();
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (!solverSettings.getOmegaPEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPE->initialize ();
      }
      if (!solverSettings.getOmegaPMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalOmegaPM->initialize ();
      }
      if (!solverSettings.getGammaEFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaE->initialize ();
      }
      if (!solverSettings.getGammaMFileName ().empty () || solverSettings.getDoSaveMaterials ())
      {
        totalGammaM->initialize ();
      }
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  Eps->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::EPS, Eps, totalEps);

  if (solverSettings.getEpsSphere () != 1)
  {
    for (grid_coord i = 0; i < Eps->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = Eps->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getEpsCoordFP (Eps->getTotalPosition (pos));
      FieldPointValue *val = Eps->getFieldPointValue (pos);

      FieldValue epsVal = getFieldValueRealOnly (solverSettings.getEpsSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(solverSettings.getEpsSphereCenterX (),
                                             solverSettings.getEpsSphereCenterY (),
                                             solverSettings.getEpsSphereCenterZ (),
                                             ct1, ct2, ct3);
      val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                  center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                        , ct1, ct2, ct3
#endif
                                                                                                        ),
                                                                  solverSettings.getEpsSphereRadius () * modifier,
                                                                  epsVal,
                                                                  getFieldValueRealOnly (1.0)));
    }
  }
  if (solverSettings.getUseEpsAllNorm ())
  {
    for (grid_coord i = 0; i < Eps->getSize ().calculateTotalCoord (); ++i)
    {
      FieldPointValue *val = Eps->getFieldPointValue (i);
      val->setCurValue (getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Eps0));
    }
  }

  Mu->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::MU, Mu, totalMu);

  if (solverSettings.getMuSphere () != 1)
  {
    for (grid_coord i = 0; i < Mu->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = Mu->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getMuCoordFP (Mu->getTotalPosition (pos));
      FieldPointValue *val = Mu->getFieldPointValue (pos);

      FieldValue muVal = getFieldValueRealOnly (solverSettings.getMuSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(solverSettings.getMuSphereCenterX (),
                                             solverSettings.getMuSphereCenterY (),
                                             solverSettings.getMuSphereCenterZ (),
                                             ct1, ct2, ct3);
      val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                  center * modifier + TCFP (0.5, 0.5, 0.5
  #ifdef DEBUG_INFO
                                                                                                        , ct1, ct2, ct3
  #endif
                                                                                                        ),
                                                                  solverSettings.getMuSphereRadius () * modifier,
                                                                  muVal,
                                                                  getFieldValueRealOnly (1.0)));
    }
  }
  if (solverSettings.getUseMuAllNorm ())
  {
    for (grid_coord i = 0; i < Mu->getSize ().calculateTotalCoord (); ++i)
    {
      FieldPointValue *val = Mu->getFieldPointValue (i);
      val->setCurValue (getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Mu0));
    }
  }

  if (solverSettings.getDoUseMetamaterials ())
  {
    OmegaPE->initialize ();
    initMaterialFromFile (GridType::OMEGAPE, OmegaPE, totalOmegaPE);

    if (solverSettings.getOmegaPESphere () != 0)
    {
      for (grid_coord i = 0; i < OmegaPE->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = OmegaPE->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (OmegaPE->getTotalPosition (pos));
        FieldPointValue *val = OmegaPE->getFieldPointValue (pos);

        FieldValue omegapeVal = getFieldValueRealOnly (solverSettings.getOmegaPESphere () * 2 * PhysicsConst::Pi * sourceFrequency);

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (solverSettings.getOmegaPESphereCenterX (),
                                                solverSettings.getOmegaPESphereCenterY (),
                                                solverSettings.getOmegaPESphereCenterZ (),
                                                ct1, ct2, ct3);
        val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    solverSettings.getOmegaPESphereRadius () * modifier,
                                                                    omegapeVal,
                                                                    getFieldValueRealOnly (0.0)));
      }
    }

    OmegaPM->initialize ();
    initMaterialFromFile (GridType::OMEGAPM, OmegaPM, totalOmegaPM);

    if (solverSettings.getOmegaPMSphere () != 0)
    {
      for (grid_coord i = 0; i < OmegaPM->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = OmegaPM->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (OmegaPM->getTotalPosition (pos));
        FieldPointValue *val = OmegaPM->getFieldPointValue (pos);

        FieldValue omegapmVal = getFieldValueRealOnly (solverSettings.getOmegaPMSphere () * 2 * PhysicsConst::Pi * sourceFrequency);

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (solverSettings.getOmegaPMSphereCenterX (),
                                                solverSettings.getOmegaPMSphereCenterY (),
                                                solverSettings.getOmegaPMSphereCenterZ (),
                                                ct1, ct2, ct3);
        val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    solverSettings.getOmegaPMSphereRadius () * modifier,
                                                                    omegapmVal,
                                                                    getFieldValueRealOnly (0.0)));
      }
    }

    GammaE->initialize ();
    initMaterialFromFile (GridType::GAMMAE, GammaE, totalGammaE);

    GammaM->initialize ();
    initMaterialFromFile (GridType::GAMMAM, GammaM, totalGammaM);
  }

  if (solverSettings.getDoUsePML ())
  {
    initSigmas ();
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (solverSettings.getDoSaveMaterials ())
    {
      if (useParallel)
      {
        initFullMaterialGrids ();
      }

      if (processId == 0)
      {
        TC startEps, startMu, startOmegaPE, startOmegaPM, startGammaE, startGammaM;
        TC endEps, endMu, endOmegaPE, endOmegaPM, endGammaE, endGammaM;

        if (solverSettings.getDoUseManualStartEndDumpCoord ())
        {
          TC start = TC::initAxesCoordinate (solverSettings.getSaveStartCoordX (),
                                            solverSettings.getSaveStartCoordY (),
                                            solverSettings.getSaveStartCoordZ (),
                                            ct1, ct2, ct3);
          TC end = TC::initAxesCoordinate (solverSettings.getSaveEndCoordX (),
                                          solverSettings.getSaveEndCoordY (),
                                          solverSettings.getSaveEndCoordZ (),
                                          ct1, ct2, ct3);
          startEps = startMu = startOmegaPE = startOmegaPM = startGammaE = startGammaM = start;
          endEps = endMu = endOmegaPE = endOmegaPM = endGammaE = endGammaM = end;
        }
        else
        {
          startEps = getStartCoord (GridType::EPS, totalEps->getSize ());
          endEps = getEndCoord (GridType::EPS, totalEps->getSize ());

          startMu = getStartCoord (GridType::MU, totalMu->getSize ());
          endMu = getEndCoord (GridType::MU, totalMu->getSize ());

          if (solverSettings.getDoUseMetamaterials ())
          {
            startOmegaPE = getStartCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());
            endOmegaPE = getEndCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());

            startOmegaPM = getStartCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());
            endOmegaPM = getEndCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());

            startGammaE = getStartCoord (GridType::GAMMAE, totalGammaE->getSize ());
            endGammaE = getEndCoord (GridType::GAMMAE, totalGammaE->getSize ());

            startGammaM = getStartCoord (GridType::GAMMAM, totalGammaM->getSize ());
            endGammaM = getEndCoord (GridType::GAMMAM, totalGammaM->getSize ());
          }
        }

        dumper[type]->init (0, CURRENT, processId, "Eps");
        dumper[type]->dumpGrid (totalEps,
                                startEps,
                                endEps);

        dumper[type]->init (0, CURRENT, processId, "Mu");
        dumper[type]->dumpGrid (totalMu,
                                startMu,
                                endMu);

        if (solverSettings.getDoUseMetamaterials ())
        {
          dumper[type]->init (0, CURRENT, processId, "OmegaPE");
          dumper[type]->dumpGrid (totalOmegaPE,
                                  startOmegaPE,
                                  endOmegaPE);

          dumper[type]->init (0, CURRENT, processId, "OmegaPM");
          dumper[type]->dumpGrid (totalOmegaPM,
                                  startOmegaPM,
                                  endOmegaPM);

          dumper[type]->init (0, CURRENT, processId, "GammaE");
          dumper[type]->dumpGrid (totalGammaE,
                                  startGammaE,
                                  endGammaE);

          dumper[type]->init (0, CURRENT, processId, "GammaM");
          dumper[type]->dumpGrid (totalGammaM,
                                  startGammaM,
                                  endGammaM);
        }
        //
        // if (solverSettings.getDoUsePML ())
        // {
        //   dumper[type]->init (0, CURRENT, processId, "SigmaX");
        //   dumper[type]->dumpGrid (SigmaX,
        //                           GridCoordinate3D (0, 0, SigmaX->getSize ().get3 () / 2),
        //                           GridCoordinate3D (SigmaX->getSize ().get1 (), SigmaX->getSize ().get2 (), SigmaX->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "SigmaY");
        //   dumper[type]->dumpGrid (SigmaY,
        //                           GridCoordinate3D (0, 0, SigmaY->getSize ().get3 () / 2),
        //                           GridCoordinate3D (SigmaY->getSize ().get1 (), SigmaY->getSize ().get2 (), SigmaY->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "SigmaZ");
        //   dumper[type]->dumpGrid (SigmaZ,
        //                           GridCoordinate3D (0, 0, SigmaZ->getSize ().get3 () / 2),
        //                           GridCoordinate3D (SigmaZ->getSize ().get1 (), SigmaZ->getSize ().get2 (), SigmaZ->getSize ().get3 () / 2 + 1));
        // }
      }
    }
  }

  if (doNeedEx)
  {
    Ex->initialize ();
    initGridWithInitialVals (GridType::EX, Ex, 0.5 * gridTimeStep);
  }
  if (doNeedEy)
  {
    Ey->initialize ();
    initGridWithInitialVals (GridType::EY, Ey, 0.5 * gridTimeStep);
  }
  if (doNeedEz)
  {
    Ez->initialize ();
    initGridWithInitialVals (GridType::EZ, Ez, 0.5 * gridTimeStep);
  }

  if (doNeedHx)
  {
    Hx->initialize ();
    initGridWithInitialVals (GridType::HX, Hx, gridTimeStep);
  }
  if (doNeedHy)
  {
    Hy->initialize ();
    initGridWithInitialVals (GridType::HY, Hy, gridTimeStep);
  }
  if (doNeedHz)
  {
    Hz->initialize ();
    initGridWithInitialVals (GridType::HZ, Hz, gridTimeStep);
  }

  if (solverSettings.getDoUsePML ())
  {
    if (doNeedEx)
    {
      Dx->initialize ();
    }
    if (doNeedEy)
    {
      Dy->initialize ();
    }
    if (doNeedEz)
    {
      Dz->initialize ();
    }

    if (doNeedHx)
    {
      Bx->initialize ();
    }
    if (doNeedHy)
    {
      By->initialize ();
    }
    if (doNeedHz)
    {
      Bz->initialize ();
    }

    if (solverSettings.getDoUseMetamaterials ())
    {
      if (doNeedEx)
      {
        D1x->initialize ();
      }
      if (doNeedEy)
      {
        D1y->initialize ();
      }
      if (doNeedEz)
      {
        D1z->initialize ();
      }

      if (doNeedHx)
      {
        B1x->initialize ();
      }
      if (doNeedHy)
      {
        B1y->initialize ();
      }
      if (doNeedHz)
      {
        B1z->initialize ();
      }
    }
  }

  if (solverSettings.getDoUseAmplitudeMode ())
  {
    if (doNeedEx)
    {
      ExAmplitude->initialize ();
    }
    if (doNeedEy)
    {
      EyAmplitude->initialize ();
    }
    if (doNeedEz)
    {
      EzAmplitude->initialize ();
    }

    if (doNeedHx)
    {
      HxAmplitude->initialize ();
    }
    if (doNeedHy)
    {
      HyAmplitude->initialize ();
    }
    if (doNeedHz)
    {
      HzAmplitude->initialize ();
    }
  }

  if (solverSettings.getDoUseTFSF ())
  {
    EInc->initialize ();
    HInc->initialize ();
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());

    ((ParallelGrid *) Eps)->share ();
    ((ParallelGrid *) Mu)->share ();

    if (solverSettings.getDoUsePML ())
    {
      if (doNeedSigmaX)
      {
        ((ParallelGrid *) SigmaX)->share ();
      }
      if (doNeedSigmaY)
      {
        ((ParallelGrid *) SigmaY)->share ();
      }
      if (doNeedSigmaZ)
      {
        ((ParallelGrid *) SigmaZ)->share ();
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

NPair
SchemeHelper::ntffN3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHz1 = curHz->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHz2 = curHz->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHy1 = curHy->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHy2 = curHy->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHy1 != NULL && valHy2 != NULL);

      FieldValue Hz1 = valHz1->getCurValue ();
      FieldValue Hz2 = valHz2->getCurValue ();
      FieldValue Hy1 = valHy1->getCurValue ();
      FieldValue Hy2 = valHy2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hz2 -= yeeLayout->getHzFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy1 -= yeeLayout->getHyFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy2 -= yeeLayout->getHyFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHz1 = curHz->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHz2 = curHz->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHx1 = curHx->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHx2 = curHx->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHz1 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
          || valHz2 == NULL || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHz1 != NULL && valHz2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hz1 = valHz1->getCurValue ();
      FieldValue Hz2 = valHz2->getCurValue ();
      FieldValue Hx1 = valHx1->getCurValue ();
      FieldValue Hx2 = valHx2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hz1 -= yeeLayout->getHzFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hz2 -= yeeLayout->getHzFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx1 -= yeeLayout->getHxFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx2 -= yeeLayout->getHxFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1) * ((Hz1 + Hz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *HInc,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
      pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1);
      GridCoordinate3D pos21 = convertCoord (pos2);
      GridCoordinate3D pos31 = convertCoord (pos3);
      GridCoordinate3D pos41 = convertCoord (pos4);

      FieldPointValue *valHy1 = curHy->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valHy2 = curHy->getFieldPointValueOrNullByAbsolutePos (pos21);

      FieldPointValue *valHx1 = curHx->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valHx2 = curHx->getFieldPointValueOrNullByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
      if (valHy1 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
          || valHy2 == NULL || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
          || valHx1 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
          || valHx2 == NULL || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
      {
        continue;
      }
#endif

      ASSERT (valHy1 != NULL && valHy2 != NULL && valHx1 != NULL && valHx2 != NULL);

      FieldValue Hy1 = valHy1->getCurValue ();
      FieldValue Hy2 = valHy2->getCurValue ();
      FieldValue Hx1 = valHx1->getCurValue ();
      FieldValue Hx2 = valHx2->getCurValue ();

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Hy1 -= yeeLayout->getHyFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hy2 -= yeeLayout->getHyFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx1 -= yeeLayout->getHxFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Hx2 -= yeeLayout->getHxFromIncidentH (SchemeHelper::approximateIncidentWaveH<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), HInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1) * (-(Hy1 + Hy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1) * ((Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (x0, coordY - 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (x0, coordY + 0.5, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (x0, coordY, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (x0, coordY, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
      pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0.5, 0, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      FieldPointValue *valEy11 = curEy->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEy12 = curEy->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEy21 = curEy->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEy22 = curEy->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEz11 = curEz->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEz12 = curEz->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEz21 = curEz->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEz22 = curEz->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ey1 = (valEy11->getCurValue () + valEy12->getCurValue ()) / FPValue(2.0);
      FieldValue Ey2 = (valEy21->getCurValue () + valEy22->getCurValue ()) / FPValue(2.0);
      FieldValue Ez1 = (valEz11->getCurValue () + valEz12->getCurValue ()) / FPValue(2.0);
      FieldValue Ez2 = (valEz21->getCurValue () + valEz22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ey1 -= yeeLayout->getEyFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey2 -= yeeLayout->getEyFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez1 -= yeeLayout->getEzFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez2 -= yeeLayout->getEzFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                                  + (Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, y0, coordZ
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, y0, coordZ - 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, y0, coordZ + 0.5
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0, 0.5, 0
#ifdef DEBUG_INFO
                                             , ct1, ct2, ct3
#endif
                                             ));

      FieldPointValue *valEx11 = curEx->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEx12 = curEx->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEx21 = curEx->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEx22 = curEx->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEz11 = curEz->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEz12 = curEz->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEz21 = curEz->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEz22 = curEz->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEz11 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
          || valEz12 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
          || valEz21 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
          || valEz22 == NULL || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEz11 != NULL && valEz12 != NULL && valEz21 != NULL && valEz22 != NULL);

      FieldValue Ex1 = (valEx11->getCurValue () + valEx12->getCurValue ()) / FPValue(2.0);
      FieldValue Ex2 = (valEx21->getCurValue () + valEx22->getCurValue ()) / FPValue(2.0);
      FieldValue Ez1 = (valEz11->getCurValue () + valEz12->getCurValue ()) / FPValue(2.0);
      FieldValue Ez2 = (valEz21->getCurValue () + valEz22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ex2 -= yeeLayout->getExFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez1 -= yeeLayout->getEzFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ez2 -= yeeLayout->getEzFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent;

      sum_phi += SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1) * ((Ez1 + Ez2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffL3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YL3D_Dim3 *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate1D> *EInc,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue diffx0 = curEz->getTotalSize ().get1 () / 2;
  FPValue diffy0 = curEz->getTotalSize ().get2 () / 2;
  FPValue diffz0 = curEz->getTotalSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );
  GridCoordinateFP3D coordEnd (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0
#ifdef DEBUG_INFO
                                 , ct1, ct2, ct3
#endif
                                 );

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      GridCoordinateFP3D pos1 (coordX - 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos2 (coordX + 0.5, coordY, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos3 (coordX, coordY - 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );
      GridCoordinateFP3D pos4 (coordX, coordY + 0.5, z0
#ifdef DEBUG_INFO
                               , ct1, ct2, ct3
#endif
                               );

      pos1 = pos1 - yeeLayout->getMinExCoordFP ();
      pos2 = pos2 - yeeLayout->getMinExCoordFP ();

      pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
      pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

      GridCoordinate3D pos11 = convertCoord (pos1-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos12 = convertCoord (pos1+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos21 = convertCoord (pos2-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos22 = convertCoord (pos2+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));

      GridCoordinate3D pos31 = convertCoord (pos3-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos32 = convertCoord (pos3+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos41 = convertCoord (pos4-GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));
      GridCoordinate3D pos42 = convertCoord (pos4+GridCoordinateFP3D(0, 0, 0.5
#ifdef DEBUG_INFO
                                                                     , ct1, ct2, ct3
#endif
                                                                     ));

      FieldPointValue *valEx11 = curEx->getFieldPointValueOrNullByAbsolutePos (pos11);
      FieldPointValue *valEx12 = curEx->getFieldPointValueOrNullByAbsolutePos (pos12);
      FieldPointValue *valEx21 = curEx->getFieldPointValueOrNullByAbsolutePos (pos21);
      FieldPointValue *valEx22 = curEx->getFieldPointValueOrNullByAbsolutePos (pos22);

      FieldPointValue *valEy11 = curEy->getFieldPointValueOrNullByAbsolutePos (pos31);
      FieldPointValue *valEy12 = curEy->getFieldPointValueOrNullByAbsolutePos (pos32);
      FieldPointValue *valEy21 = curEy->getFieldPointValueOrNullByAbsolutePos (pos41);
      FieldPointValue *valEy22 = curEy->getFieldPointValueOrNullByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
      if (valEx11 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
          || valEx12 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
          || valEx21 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
          || valEx22 == NULL || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
          || valEy11 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
          || valEy12 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos32)
          || valEy21 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41)
          || valEy22 == NULL || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos42))
      {
        continue;
      }
#endif

      ASSERT (valEx11 != NULL && valEx12 != NULL && valEx21 != NULL && valEx22 != NULL
              && valEy11 != NULL && valEy12 != NULL && valEy21 != NULL && valEy22 != NULL);

      FieldValue Ex1 = (valEx11->getCurValue () + valEx12->getCurValue ()) / FPValue(2.0);
      FieldValue Ex2 = (valEx21->getCurValue () + valEx22->getCurValue ()) / FPValue(2.0);
      FieldValue Ey1 = (valEy11->getCurValue () + valEy12->getCurValue ()) / FPValue(2.0);
      FieldValue Ey2 = (valEy21->getCurValue () + valEy22->getCurValue ()) / FPValue(2.0);

      if (solverSettings.getDoCalcScatteredNTFF ())
      {
        Ex1 -= yeeLayout->getExFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos1, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ex2 -= yeeLayout->getExFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos2, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey1 -= yeeLayout->getEyFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos3, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
        Ey2 -= yeeLayout->getEyFromIncidentE (SchemeHelper::approximateIncidentWaveE<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate> (pos4, yeeLayout->getZeroIncCoordFP (), EInc, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ()));
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1) * (-(Ey1 + Ey2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent;

      sum_phi += SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1) * ((Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (anglePhi))
                                                + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent;
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

NPair
SchemeHelper::ntffN3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YL3D_Dim3 *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate1D> *HInc,
                       Grid<GridCoordinate3D> *curEz,
                       Grid<GridCoordinate3D> *curHx,
                       Grid<GridCoordinate3D> *curHy,
                       Grid<GridCoordinate3D> *curHz)
{
  NPair nx = SchemeHelper::ntffN3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHy, curHz)
                     + ntffN3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHy, curHz);
  NPair ny = SchemeHelper::ntffN3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHz)
                     + SchemeHelper::ntffN3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHz);
  NPair nz = SchemeHelper::ntffN3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy)
                     + SchemeHelper::ntffN3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy);

  return nx + ny + nz;
}

NPair
SchemeHelper::ntffL3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YL3D_Dim3 *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate1D> *EInc,
                       Grid<GridCoordinate3D> *curEx,
                       Grid<GridCoordinate3D> *curEy,
                       Grid<GridCoordinate3D> *curEz)
{
  NPair lx = SchemeHelper::ntffL3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEy, curEz)
                     + SchemeHelper::ntffL3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEy, curEz);
  NPair ly = SchemeHelper::ntffL3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEz)
                     + SchemeHelper::ntffL3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEz);
  NPair lz = SchemeHelper::ntffL3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz)
                     + SchemeHelper::ntffL3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz);

  return lx + ly + lz;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *curEx, Grid<TC> *curEy, Grid<TC> *curEz,
                       Grid<TC> *curHx, Grid<TC> *curHy, Grid<TC> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue k = 2 * PhysicsConst::Pi / sourceWaveLength; // TODO: check numerical here

  NPair N = ntffN (angleTeta, anglePhi, curEz, curHx, curHy, curHz);
  NPair L = ntffL (angleTeta, anglePhi, curEx, curEy, curEz);

  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();

    FieldValue tmpArray[4];
    FieldValue tmpArrayRes[4];
    const int count = 4;

    tmpArray[0] = N.nTeta;
    tmpArray[1] = N.nPhi;
    tmpArray[2] = L.nTeta;
    tmpArray[3] = L.nPhi;

    // gather all sum_teta and sum_phi on 0 node
    MPI_Reduce (tmpArray, tmpArrayRes, count, MPI_FPVALUE, MPI_SUM, 0, ParallelGrid::getParallelCore ()->getCommunicator ());

    if (processId == 0)
    {
      N.nTeta = FieldValue (tmpArrayRes[0]);
      N.nPhi = FieldValue (tmpArrayRes[1]);

      L.nTeta = FieldValue (tmpArrayRes[2]);
      L.nPhi = FieldValue (tmpArrayRes[3]);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  if (processId == 0)
  {
    FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);

    FieldValue first = -L.nPhi + n0 * N.nTeta;
    FieldValue second = -L.nTeta - n0 * N.nPhi;

    FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
    FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());

    return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
  }
  else
  {
    return 0.0;
  }
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
{
  return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::makeGridScattered (Grid<TC> *grid, GridType gridType)
{
  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    FieldPointValue *val = grid->getFieldPointValue (i);

    TC pos = grid->calculatePositionFromIndex (i);
    TC posAbs = grid->getTotalPosition (pos);

    TCFP realCoord;
    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doSkipMakeScattered (realCoord))
    {
      continue;
    }

    FieldValue iVal;
    if (gridType == GridType::EX
        || gridType == GridType::EY
        || gridType == GridType::EZ)
    {
      iVal = approximateIncidentWaveE (realCoord);
    }
    else if (gridType == GridType::HX
             || gridType == GridType::HY
             || gridType == GridType::HZ)
    {
      iVal = approximateIncidentWaveH (realCoord);
    }
    else
    {
      UNREACHABLE;
    }

    FieldValue incVal;
    switch (gridType)
    {
      case GridType::EX:
      {
        incVal = yeeLayout->getExFromIncidentE (iVal);
        break;
      }
      case GridType::EY:
      {
        incVal = yeeLayout->getEyFromIncidentE (iVal);
        break;
      }
      case GridType::EZ:
      {
        incVal = yeeLayout->getEzFromIncidentE (iVal);
        break;
      }
      case GridType::HX:
      {
        incVal = yeeLayout->getHxFromIncidentH (iVal);
        break;
      }
      case GridType::HY:
      {
        incVal = yeeLayout->getHyFromIncidentH (iVal);
        break;
      }
      case GridType::HZ:
      {
        incVal = yeeLayout->getHzFromIncidentH (iVal);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    val->setCurValue (val->getCurValue () - incVal);
  }
}


template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids3D (&totalInitialized, doNeedEx, &Ex, &totalEx, doNeedEy, &Ey, &totalEy, doNeedEz, &Ez, &totalEz,
                                      doNeedHx, &Hx, &totalHx, doNeedHy, &Hy, &totalHy, doNeedHz, &Hz, &totalHz);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::gatherFieldsTotal (bool scattered)
{
  if (useParallel)
  {
    initFullFieldGrids ();
  }
  else
  {
    if (totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = *Ex;
      }
      if (doNeedEy)
      {
        *totalEy = *Ey;
      }
      if (doNeedEz)
      {
        *totalEz = *Ez;
      }

      if (doNeedHx)
      {
        *totalHx = *Hx;
      }
      if (doNeedHy)
      {
        *totalHy = *Hy;
      }
      if (doNeedHz)
      {
        *totalHz = *Hz;
      }
    }
    else
    {
      if (scattered)
      {
        if (doNeedEx)
        {
          totalEx = new Grid<TC> (yeeLayout->getExSize (), 0, "Ex");
          *totalEx = *Ex;
        }
        if (doNeedEy)
        {
          totalEy = new Grid<TC> (yeeLayout->getEySize (), 0, "Ey");
          *totalEy = *Ey;
        }
        if (doNeedEz)
        {
          totalEz = new Grid<TC> (yeeLayout->getEzSize (), 0, "Ez");
          *totalEz = *Ez;
        }

        if (doNeedHx)
        {
          totalHx = new Grid<TC> (yeeLayout->getHxSize (), 0, "Hx");
          *totalHx = *Hx;
        }
        if (doNeedHy)
        {
          totalHy = new Grid<TC> (yeeLayout->getHySize (), 0, "Hy");
          *totalHy = *Hy;
        }
        if (doNeedHz)
        {
          totalHz = new Grid<TC> (yeeLayout->getHzSize (), 0, "Hz");
          *totalHz = *Hz;
        }

        totalInitialized = true;
      }
      else
      {
        if (doNeedEx)
        {
          totalEx = Ex;
        }
        if (doNeedEy)
        {
          totalEy = Ey;
        }
        if (doNeedEz)
        {
          totalEz = Ez;
        }

        if (doNeedHx)
        {
          totalHx = Hx;
        }
        if (doNeedHy)
        {
          totalHy = Hy;
        }
        if (doNeedHz)
        {
          totalHz = Hz;
        }
      }
    }
  }

  if (scattered)
  {
    if (doNeedEx)
    {
      makeGridScattered (totalEx, GridType::EX);
    }
    if (doNeedEy)
    {
      makeGridScattered (totalEy, GridType::EY);
    }
    if (doNeedEz)
    {
      makeGridScattered (totalEz, GridType::EZ);
    }

    if (doNeedHx)
    {
      makeGridScattered (totalHx, GridType::HX);
    }
    if (doNeedHy)
    {
      makeGridScattered (totalHy, GridType::HY);
    }
    if (doNeedHz)
    {
      makeGridScattered (totalHz, GridType::HZ);
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveGrids (time_step t)
{
  int processId = 0;

  TC startEx;
  TC endEx;
  TC startEy;
  TC endEy;
  TC startEz;
  TC endEz;
  TC startHx;
  TC endHx;
  TC startHy;
  TC endHy;
  TC startHz;
  TC endHz;

  if (solverSettings.getDoUseManualStartEndDumpCoord ())
  {
    TC start = TC::initAxesCoordinate (solverSettings.getSaveStartCoordX (),
                                       solverSettings.getSaveStartCoordY (),
                                       solverSettings.getSaveStartCoordZ (),
                                       ct1, ct2, ct3);
    TC end = TC::initAxesCoordinate (solverSettings.getSaveEndCoordX (),
                                     solverSettings.getSaveEndCoordY (),
                                     solverSettings.getSaveEndCoordZ (),
                                     ct1, ct2, ct3);

    startEx = startEy = startEz = startHx = startHy = startHz = start;
    endEx = endEy = endEz = endHx = endHy = endHz = end;
  }
  else
  {
    startEx = doNeedEx ? getStartCoord (GridType::EX, Ex->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEx = doNeedEx ? getEndCoord (GridType::EX, Ex->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startEy = doNeedEy ? getStartCoord (GridType::EY, Ey->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEy = doNeedEy ? getEndCoord (GridType::EY, Ey->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startEz = doNeedEz ? getStartCoord (GridType::EZ, Ez->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEz = doNeedEz ? getEndCoord (GridType::EZ, Ez->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHx = doNeedHx ? getStartCoord (GridType::HX, Hx->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHx = doNeedHx ? getEndCoord (GridType::HX, Hx->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHy = doNeedHy ? getStartCoord (GridType::HY, Hy->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHy = doNeedHy ? getEndCoord (GridType::HY, Hy->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHz = doNeedHz ? getStartCoord (GridType::HZ, Hz->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHz = doNeedHz ? getEndCoord (GridType::HZ, Hz->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (doNeedEx)
    {
      dumper[type]->init (t, CURRENT, processId, totalEx->getName ().c_str ());
      dumper[type]->dumpGrid (totalEx, startEx, endEx);
    }

    if (doNeedEy)
    {
      dumper[type]->init (t, CURRENT, processId, totalEy->getName ().c_str ());
      dumper[type]->dumpGrid (totalEy, startEy, endEy);
    }

    if (doNeedEz)
    {
      dumper[type]->init (t, CURRENT, processId, totalEz->getName ().c_str ());
      dumper[type]->dumpGrid (totalEz, startEz, endEz);
    }

    if (doNeedHx)
    {
      dumper[type]->init (t, CURRENT, processId, totalHx->getName ().c_str ());
      dumper[type]->dumpGrid (totalHx, startHx, endHx);
    }

    if (doNeedHy)
    {
      dumper[type]->init (t, CURRENT, processId, totalHy->getName ().c_str ());
      dumper[type]->dumpGrid (totalHy, startHy, endHy);
    }

    if (doNeedHz)
    {
      dumper[type]->init (t, CURRENT, processId, totalHz->getName ().c_str ());
      dumper[type]->dumpGrid (totalHz, startHz, endHz);
    }

    if (solverSettings.getDoSaveTFSFEInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->init (t, CURRENT, processId, "EInc");
      dumper1D[type]->dumpGrid (EInc, GridCoordinate1D (0
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        ),
                                EInc->getSize ());
    }

    if (solverSettings.getDoSaveTFSFHInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->init (t, CURRENT, processId, "HInc");
      dumper1D[type]->dumpGrid (HInc, GridCoordinate1D (0
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        ),
                                HInc->getSize ());
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveNTFF (bool isReverse, time_step t)
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  std::ofstream outfile;
  std::ostream *outs;
  const char *strName;
  FPValue start;
  FPValue end;
  FPValue step;

  if (isReverse)
  {
    strName = "Reverse diagram";
    start = yeeLayout->getIncidentWaveAngle2 ();
    end = yeeLayout->getIncidentWaveAngle2 ();
    step = 1.0;
  }
  else
  {
    strName = "Forward diagram";
    start = 0.0;
    end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
    step = PhysicsConst::Pi * solverSettings.getAngleStepNTFF () / 180;
  }

  if (processId == 0)
  {
    if (solverSettings.getDoSaveNTFFToStdout ())
    {
      outs = &std::cout;
    }
    else
    {
      outfile.open (solverSettings.getFileNameNTFF ().c_str ());
      outs = &outfile;
    }
    (*outs) << strName << std::endl << std::endl;
  }

  for (FPValue angle = start; angle <= end; angle += step)
  {
    FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
                                 angle,
                                 Ex,
                                 Ey,
                                 Ez,
                                 Hx,
                                 Hy,
                                 Hz) / Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);

    if (processId == 0)
    {
      (*outs) << "timestep = "
              << t
              << ", incident wave angle=("
              << yeeLayout->getIncidentWaveAngle1 () << ","
              << yeeLayout->getIncidentWaveAngle2 () << ","
              << yeeLayout->getIncidentWaveAngle3 () << ","
              << "), angle NTFF = "
              << angle
              << ", NTFF value = "
              << val
              << std::endl;
    }
  }

  if (processId == 0)
  {
    if (!solverSettings.getDoSaveNTFFToStdout ())
    {
      outfile.close ();
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::additionalUpdateOfGrids (time_step t, time_step &diffT)
{
  if (useParallel && solverSettings.getDoUseDynamicGrid ())
  {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
    //if (false && t % solverSettings.getRebalanceStep () == 0)
    if (t % diffT == 0 && t > 0)
    {
      if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
      {
        DPRINTF (LOG_LEVEL_STAGES, "Try rebalance on step %u, steps elapsed after previous %u\n", t, diffT);
      }

      ASSERT (isParallelLayout);

      ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;

      if (parallelYeeLayout->Rebalance (diffT))
      {
        DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalancing for process %d!\n", ParallelGrid::getParallelCore ()->getProcessId ());

        ((ParallelGrid *) Eps)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        ((ParallelGrid *) Mu)->Resize (parallelYeeLayout->getMuSizeForCurNode ());

        if (doNeedEx)
        {
          ((ParallelGrid *) Ex)->Resize (parallelYeeLayout->getExSizeForCurNode ());
        }
        if (doNeedEy)
        {
          ((ParallelGrid *) Ey)->Resize (parallelYeeLayout->getEySizeForCurNode ());
        }
        if (doNeedEz)
        {
          ((ParallelGrid *) Ez)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
        }

        if (doNeedHx)
        {
          ((ParallelGrid *) Hx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
        }
        if (doNeedHy)
        {
          ((ParallelGrid *) Hy)->Resize (parallelYeeLayout->getHySizeForCurNode ());
        }
        if (doNeedHz)
        {
          ((ParallelGrid *) Hz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
        }

        if (solverSettings.getDoUsePML ())
        {
          if (doNeedEx)
          {
            ((ParallelGrid *) Dx)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          }
          if (doNeedEy)
          {
            ((ParallelGrid *) Dy)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          }
          if (doNeedEz)
          {
            ((ParallelGrid *) Dz)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
          }

          if (doNeedHx)
          {
            ((ParallelGrid *) Bx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          }
          if (doNeedHy)
          {
            ((ParallelGrid *) By)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          }
          if (doNeedHz)
          {
            ((ParallelGrid *) Bz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
          }

          if (solverSettings.getDoUseMetamaterials ())
          {
            if (doNeedEx)
            {
              ((ParallelGrid *) D1x)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            }
            if (doNeedEy)
            {
              ((ParallelGrid *) D1y)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            }
            if (doNeedEz)
            {
              ((ParallelGrid *) D1z)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            }

            if (doNeedHx)
            {
              ((ParallelGrid *) B1x)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            }
            if (doNeedHy)
            {
              ((ParallelGrid *) B1y)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            }
            if (doNeedHz)
            {
              ((ParallelGrid *) B1z)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            }
          }

          if (doNeedSigmaX)
          {
            ((ParallelGrid *) SigmaX)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
          if (doNeedSigmaY)
          {
            ((ParallelGrid *) SigmaY)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
          if (doNeedSigmaZ)
          {
            ((ParallelGrid *) SigmaZ)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
        }

        if (solverSettings.getDoUseAmplitudeMode ())
        {
          if (doNeedEx)
          {
            ((ParallelGrid *) ExAmplitude)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          }
          if (doNeedEy)
          {
            ((ParallelGrid *) EyAmplitude)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          }
          if (doNeedEz)
          {
            ((ParallelGrid *) EzAmplitude)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
          }

          if (doNeedHx)
          {
            ((ParallelGrid *) HxAmplitude)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          }
          if (doNeedHy)
          {
            ((ParallelGrid *) HyAmplitude)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          }
          if (doNeedHz)
          {
            ((ParallelGrid *) HzAmplitude)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
          }
        }

        if (solverSettings.getDoUseMetamaterials ())
        {
          ((ParallelGrid *) OmegaPE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) GammaE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) OmegaPM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) GammaM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        }

        //diffT += 1;
        //diffT *= 2;
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                    "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getStartCoord (GridType gridType, TC size)
{
  TC start (0, 0, 0
#ifdef DEBUG_INFO
            , ct1, ct2, ct3
#endif
            );

  if (solverSettings.getDoSaveWithoutPML ()
      && solverSettings.getDoUsePML ())
  {
    TCFP leftBorder = convertCoord (yeeLayout->getLeftBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    start = convertCoord (expandTo3D (leftBorder - min, ct1, ct2, ct3)) + GridCoordinate3D (1, 1, 1
#ifdef DEBUG_INFO
                                                                                            , ct1, ct2, ct3
#endif
                                                                                            );
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (solverSettings.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (solverSettings.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (solverSettings.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getStartCoordRes (orthogonalAxis, start, size);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getEndCoord (GridType gridType, TC size)
{
  TC end = size;
  if (solverSettings.getDoSaveWithoutPML ()
      && solverSettings.getDoUsePML ())
  {
    TCFP rightBorder = convertCoord (yeeLayout->getRightBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    end = convertCoord (expandTo3D (rightBorder - min, ct1, ct2, ct3));
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (solverSettings.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (solverSettings.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (solverSettings.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getEndCoordRes (orthogonalAxis, end, size);
}

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED >;

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED >;

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED >;
