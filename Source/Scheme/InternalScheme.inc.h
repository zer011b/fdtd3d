#ifdef GPU_INTERNAL_SCHEME

namespace InternalSchemeKernelHelpers
{
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void calculateFieldStepIterationKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          time_step t,
                                          TCoord<grid_coord, false> diff11,
                                          TCoord<grid_coord, false> diff12,
                                          TCoord<grid_coord, false> diff21,
                                          TCoord<grid_coord, false> diff22,
                                          IGRID< TCoord<grid_coord, true> > *grid,
                                          IGRID< TCoord<grid_coord, true> > *oppositeGrid1,
                                          IGRID< TCoord<grid_coord, true> > *oppositeGrid2,
                                          SourceCallBack rightSideFunc,
                                          IGRID< TCoord<grid_coord, true> > *Ca,
                                          IGRID< TCoord<grid_coord, true> > *Cb,
                                          CoordinateType ct1,
                                          CoordinateType ct2,
                                          CoordinateType ct3,
                                          bool usePML,
                                          GridType gridType,
                                          IGRID<TC> *materialGrid,
                                          GridType materialGridType,
                                          FPValue materialModifier,
                                          bool usePrecomputedGrids)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    // TODO: add getTotalPositionDiff here, which will be called before loop
    TCoord<grid_coord, true> posAbs = grid->getTotalPosition (pos);

    TCoord<FPValue, true> coordFP;

    if (rightSideFunc != NULLPTR)
    {
      switch (grid_type)
      {
        case (static_cast<uint8_t> (GridType::EX)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getExCoordFP (posAbs);
          break;
        }
        case (static_cast<uint8_t> (GridType::EY)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getEyCoordFP (posAbs);
          break;
        }
        case (static_cast<uint8_t> (GridType::EZ)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getEzCoordFP (posAbs);
          break;
        }
        case (static_cast<uint8_t> (GridType::HX)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getHxCoordFP (posAbs);
          break;
        }
        case (static_cast<uint8_t> (GridType::HY)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getHyCoordFP (posAbs);
          break;
        }
        case (static_cast<uint8_t> (GridType::HZ)):
        {
          coordFP = gpuScheme->getYeeLayout ()->getHzCoordFP (posAbs);
          break;
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }

    if (usePrecomputedGrids)
    {
      gpuScheme->calculateFieldStepIteration<grid_type, true> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                         grid, coordFP,
                                                         oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                         usePML,
                                                         gridType, materialGrid, materialGridType,
                                                         materialModifier);
    }
    else
    {
      gpuScheme->calculateFieldStepIteration<grid_type, false> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                         grid, coordFP,
                                                         oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                         usePML,
                                                         gridType, materialGrid, materialGridType,
                                                         materialModifier);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void calculateFieldStepIterationPMLMetamaterialsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          time_step t,
                                         TC pos,
                                         IGRID<TC> *grid,
                                         IGRID<TC> *gridPML,
                                         IGRID<TC> *CB0,
                                         IGRID<TC> *CB1,
                                         IGRID<TC> *CB2,
                                         IGRID<TC> *CA1,
                                         IGRID<TC> *CA2,
                                         GridType gridType,
                                         IGRID<TC> *materialGrid1,
                                         GridType materialGridType1,
                                         IGRID<TC> *materialGrid2,
                                         GridType materialGridType2,
                                         IGRID<TC> *materialGrid3,
                                         GridType materialGridType3,
                                         FPValue materialModifier)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    gpuScheme->calculateFieldStepIterationPMLMetamaterials (t, pos, grid, gridPML, CB0, CB1, CB2, CA1, CA2,
      gridType,
      materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
      materialModifier);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, bool useMetamaterials>
  __global__
  void calculateFieldStepIterationPMLKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          time_step t,
                                           TC pos,
                                           IGRID<TC> *grid,
                                           IGRID<TC> *gridPML1,
                                           IGRID<TC> *gridPML2,
                                           IGRID<TC> *Ca,
                                           IGRID<TC> *Cb,
                                           IGRID<TC> *Cc,
                                           GridType gridPMLType1,
                                           IGRID<TC> *materialGrid1,
                                           GridType materialGridType1,
                                           IGRID<TC> *materialGrid4,
                                           GridType materialGridType4,
                                           IGRID<TC> *materialGrid5,
                                           GridType materialGridType5,
                                           FPValue materialModifier)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    gpuScheme->calculateFieldStepIterationPML<useMetamaterials> (t, pos, grid, gridPML1, gridPML2, Ca, Cb, Cc,
      gridPMLType1, materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
      materialModifier);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void performPlaneWaveEStepsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                     time_step t, GridCoordinate1D Start, GridCoordinate1D CoordPerKernel)
  {
    GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                 CoordinateType::X);
    posStart = posStart * CoordPerKernel + Start;
    GridCoordinate1D posEnd = posStart + CoordPerKernel;
    gpuScheme->performPlaneWaveESteps (t, posStart, posEnd);
  }
    template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void performPlaneWaveHStepsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                     time_step t, GridCoordinate1D Start, GridCoordinate1D CoordPerKernel)
  {
    GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                 CoordinateType::X);
    posStart = posStart * CoordPerKernel + Start;
    GridCoordinate1D posEnd = posStart + CoordPerKernel;
    gpuScheme->performPlaneWaveHSteps (t, posStart, posEnd);
  }

#define SHIFT_IN_TIME_KERNEL(NAME) \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void shiftInTimeKernel ## NAME (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme) \
  { \
    ASSERT (blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0); \
    gpuScheme-> get ## NAME () -> shiftInTime (); \
  }

  SHIFT_IN_TIME_KERNEL(Ex)
  SHIFT_IN_TIME_KERNEL(Ey)
  SHIFT_IN_TIME_KERNEL(Ez)
  SHIFT_IN_TIME_KERNEL(Hx)
  SHIFT_IN_TIME_KERNEL(Hy)
  SHIFT_IN_TIME_KERNEL(Hz)
  SHIFT_IN_TIME_KERNEL(Dx)
  SHIFT_IN_TIME_KERNEL(Dy)
  SHIFT_IN_TIME_KERNEL(Dz)
  SHIFT_IN_TIME_KERNEL(Bx)
  SHIFT_IN_TIME_KERNEL(By)
  SHIFT_IN_TIME_KERNEL(Bz)
  SHIFT_IN_TIME_KERNEL(D1x)
  SHIFT_IN_TIME_KERNEL(D1y)
  SHIFT_IN_TIME_KERNEL(D1z)
  SHIFT_IN_TIME_KERNEL(B1x)
  SHIFT_IN_TIME_KERNEL(B1y)
  SHIFT_IN_TIME_KERNEL(B1z)

#undef SHIFT_IN_TIME_KERNEL

#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL(NAME) \
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type> \
  __global__ \
  void shiftInTimePlaneWaveKernel ## NAME (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme) \
  { \
    ASSERT (blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0); \
    gpuScheme-> get ## NAME () -> shiftInTime (); \
  }

  SHIFT_IN_TIME_PLANE_WAVE_KERNEL(EInc)
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL(HInc)

#undef SHIFT_IN_TIME_PLANE_WAVE_KERNEL
};

#endif /* GPU_INTERNAL_SCHEME */

class INTERNAL_SCHEME_HELPER
{
public:

#ifdef GPU_INTERNAL_SCHEME

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  allocateGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme,
                        InternalScheme<Type, TCoord, layout_type> *cpuScheme,
                        TCoord<grid_coord, true> blockSize, TCoord<grid_coord, true> bufSize);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  freeGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void allocateGridsOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void freeGridsOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  copyGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                    TCoord<grid_coord, true> start, TCoord<grid_coord, true> end);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  copyGridsToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme,
                  InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  copyGridsBackToCPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                      time_step N,
                      bool finalCopy);

#else /* GPU_INTERNAL_SCHEME */

#if defined (PARALLEL_GRID)
  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static void allocateParallelGrids (InternalScheme<Type, ParallelGridCoordinateTemplate, layout_type> *intScheme);
#endif /* PARALLEL_GRID */

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST
  static void allocateGrids (InternalScheme<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST
  static void allocateGridsInc (InternalScheme<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout);

#endif /* !GPU_INTERNAL_SCHEME */

  ICUDA_DEVICE
  static FieldValue approximateIncidentWaveHelper (FPValue d, IGRID<GridCoordinate1D> *FieldInc)
  {
    FPValue coordD1 = (FPValue) ((grid_coord) d);
    FPValue coordD2 = coordD1 + 1;
    FPValue proportionD2 = d - coordD1;
    FPValue proportionD1 = 1 - proportionD2;

    grid_coord coord1 = (grid_coord) coordD1;
    grid_coord coord2 = (grid_coord) coordD2;

    FieldValue val1 = *FieldInc->getFieldValue (coord1, 1);
    FieldValue val2 = *FieldInc->getFieldValue (coord2, 1);

    return val1 * proportionD1 + val2 * proportionD2;
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE
  static
  FieldValue approximateIncidentWave (TCoord<FPValue, true>, TCoord<FPValue, true>, FPValue, IGRID<GridCoordinate1D> *, FPValue, FPValue);

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *EInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.0, EInc, incAngle1, incAngle2);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *HInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.5, HInc, incAngle1, incAngle2);
  }

  ICUDA_DEVICE
  static bool doSkipBorderFunc1D (GridCoordinate1D pos, IGRID<GridCoordinate1D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1;
  }
  ICUDA_DEVICE
  static bool doSkipBorderFunc2D (GridCoordinate2D pos, IGRID<GridCoordinate2D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1;
  }
  ICUDA_DEVICE
  static bool doSkipBorderFunc3D (GridCoordinate3D pos, IGRID<GridCoordinate3D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1
           && pos.get3 () != 0 && pos.get3 () != grid->getTotalSize ().get3 () - 1;
  }
};

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class INTERNAL_SCHEME_BASE
{
  friend class INTERNAL_SCHEME_HELPER;

// #ifdef CUDA_ENABLED
// #ifdef GPU_INTERNAL_SCHEME
//
//   template <SchemeType_t Type1, template <typename, bool> class TCoord1, LayoutType layout_type1>
//   friend class InternalScheme;
//   friend class InternalSchemeHelper;
//
// #else /* GPU_INTERNAL_SCHEME */
//
//   template <SchemeType_t Type1, template <typename, bool> class TCoord1, LayoutType layout_type1>
//   friend class InternalSchemeGPU;
//   friend class InternalSchemeHelperGPU;
//
// #endif /* !GPU_INTERNAL_SCHEME */
// #endif

protected:

  /**
   * Different types of template coordinates
   */
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  /**
   * Flag whether scheme is initialized
   */
  bool isInitialized;

  bool useParallel;

  /**
   * Yee grid layout, which is being used for computations
   */
  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout;

  /**
   * Coordinate types (some might be CoordinateType::NONE)
   */
  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;

#define GRID_NAME(x) \
  IGRID<TC> *x;
#include "Grids.inc.h"
#undef GRID_NAME

  /**
   * Auxiliary TF/SF 1D grids
   */
  IGRID<GridCoordinate1D> *EInc;
  IGRID<GridCoordinate1D> *HInc;

  /**
   * Wave length analytical
   */
  FPValue sourceWaveLength;

  /**
   * Wave length numerical
   */
  FPValue sourceWaveLengthNumerical;

  /**
   * Wave frequency
   */
  FPValue sourceFrequency;

  /**
   * Wave relative phase velocity
   */
  FPValue relPhaseVelocity;

  /**
   * Courant number
   */
  FPValue courantNum;

  /**
   * dx (step in space)
   */
  FPValue gridStep;

  /**
   * dt (step in time)
   */
  FPValue gridTimeStep;

  TC leftNTFF;
  TC rightNTFF;

  SourceCallBack ExBorder;
  SourceCallBack ExInitial;

  SourceCallBack EyBorder;
  SourceCallBack EyInitial;

  SourceCallBack EzBorder;
  SourceCallBack EzInitial;

  SourceCallBack HxBorder;
  SourceCallBack HxInitial;

  SourceCallBack HyBorder;
  SourceCallBack HyInitial;

  SourceCallBack HzBorder;
  SourceCallBack HzInitial;

  SourceCallBack Jx;
  SourceCallBack Jy;
  SourceCallBack Jz;
  SourceCallBack Mx;
  SourceCallBack My;
  SourceCallBack Mz;

  SourceCallBack ExExact;
  SourceCallBack EyExact;
  SourceCallBack EzExact;
  SourceCallBack HxExact;
  SourceCallBack HyExact;
  SourceCallBack HzExact;

  /*
   * TODO: maybe add separate for Dx, etc.
   */
  const bool doNeedEx;
  const bool doNeedEy;
  const bool doNeedEz;
  const bool doNeedHx;
  const bool doNeedHy;
  const bool doNeedHz;

  const bool doNeedSigmaX;
  const bool doNeedSigmaY;
  const bool doNeedSigmaZ;

protected:

#ifdef GPU_INTERNAL_SCHEME

  ICUDA_HOST
  void allocateGridsFromCPU (InternalScheme<Type, TCoord, layout_type> *cpuScheme, TC blockSize, TC bufSize)
  {
    InternalSchemeHelperGPU::allocateGridsFromCPU<Type, TCoord, layout_type> (this, cpuScheme, blockSize, bufSize);
  }
  ICUDA_HOST
  void allocateGridsOnGPU ()
  {
    InternalSchemeHelperGPU::allocateGridsOnGPU<Type, TCoord, layout_type> (this);
  }
  ICUDA_HOST
  void freeGridsFromCPU ()
  {
    InternalSchemeHelperGPU::freeGridsFromCPU<Type, TCoord, layout_type> (this);
  }
  ICUDA_HOST
  void freeGridsOnGPU ()
  {
    InternalSchemeHelperGPU::freeGridsOnGPU<Type, TCoord, layout_type> (this);
  }

  ICUDA_HOST
  void copyGridsFromCPU (TC start, TC end)
  {
    InternalSchemeHelperGPU::copyGridsFromCPU<Type, TCoord, layout_type> (this, start, end);
  }
  ICUDA_HOST
  void copyGridsToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
  {
    InternalSchemeHelperGPU::copyGridsToGPU<Type, TCoord, layout_type> (this, gpuScheme);
  }
  ICUDA_HOST
  void copyGridsBackToCPU (time_step N, bool finalCopy)
  {
    InternalSchemeHelperGPU::copyGridsBackToCPU<Type, TCoord, layout_type> (this, N, finalCopy);
  }

#else /* GPU_INTERNAL_SCHEME */

#if defined (PARALLEL_GRID)

  ICUDA_HOST void allocateParallelGrids ()
  {
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=X, where X is required dimension");
  }

#endif

  ICUDA_HOST
  void allocateGrids ()
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = InternalScheme<Type, TCoord, layout_type>::yeeLayout;
    InternalSchemeHelper::allocateGrids<Type, TCoord, layout_type> (this, layout);
  }
  ICUDA_HOST
  void allocateGridsInc ()
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = InternalScheme<Type, TCoord, layout_type>::yeeLayout;
    InternalSchemeHelper::allocateGridsInc<Type, TCoord, layout_type> (this, layout);
  }

#endif /* !GPU_INTERNAL_SCHEME */

  ICUDA_HOST void initCoordTypes ();

  ICUDA_DEVICE ICUDA_HOST bool doSkipBorderFunc (TC, IGRID<TC> *);

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFExAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFEyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFEzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFHxAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFHyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSFHzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
#endif /* ENABLE_ASSERTS */

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &, TC, TC, TC, TC);

  template<uint8_t EnumVal>
  ICUDA_DEVICE
  void performPointSourceCalc (time_step);

  ICUDA_DEVICE
  FieldValue calcField (const FieldValue & prev, const FieldValue & oppositeField12, const FieldValue & oppositeField11,
                        const FieldValue & oppositeField22, const FieldValue & oppositeField21, const FieldValue & prevRightSide,
                        const FieldValue & Ca, const FieldValue & Cb, const FPValue & delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return prev * Ca + tmp * Cb;
  }

  ICUDA_DEVICE
  FieldValue calcFieldDrude (const FieldValue & curDOrB, const FieldValue & prevDOrB, const FieldValue & prevPrevDOrB,
                             const FieldValue & prevEOrH, const FieldValue & prevPrevEOrH,
                             const FieldValue & b0, const FieldValue & b1, const FieldValue & b2, const FieldValue & a1, const FieldValue & a2)
  {
    return curDOrB * b0 + prevDOrB * b1 + prevPrevDOrB * b2 - prevEOrH * a1 - prevPrevEOrH * a2;
  }

  ICUDA_DEVICE
  FieldValue calcFieldFromDOrB (const FieldValue & prevEOrH, const FieldValue & curDOrB, const FieldValue & prevDOrB,
                                const FieldValue & Ca, const FieldValue & Cb, const FieldValue & Cc)
  {
    return prevEOrH * Ca + curDOrB * Cb - prevDOrB * Cc;
  }

public:

  ICUDA_HOST
  INTERNAL_SCHEME_BASE ();

  ICUDA_HOST
  ~INTERNAL_SCHEME_BASE ();

  template <uint8_t grid_type, bool usePrecomputedGrids>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIteration (time_step, TC, TC, TCS, TCS, TCS, TCS, IGRID<TC> *, TCFP,
                                    IGRID<TC> *, IGRID<TC> *, SourceCallBack, IGRID<TC> *, IGRID<TC> *, bool,
                                    GridType, IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_HOST
  void calculateFieldStepInit (IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *,
    IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **);

  template <uint8_t grid_type>
  ICUDA_HOST
  void calculateFieldStepInitDiff (TCS *, TCS *, TCS *, TCS *);

  ICUDA_DEVICE
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, IGRID<TC> *, IGRID<TC> *,
       IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <bool useMetamaterials>
  ICUDA_DEVICE
  void calculateFieldStepIterationPML (time_step, TC, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *,
       IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

#ifndef GPU_INTERNAL_SCHEME
  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateFieldStepIterationBorder (time_step, TC, IGRID<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateFieldStepIterationExact (time_step, TC, IGRID<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);
#endif /* !GPU_INTERNAL_SCHEME */

#ifdef GPU_INTERNAL_SCHEME
  ICUDA_HOST void initFromCPU (InternalScheme<Type, TCoord, layout_type> *cpuScheme, TC, TC);
  ICUDA_HOST void initOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);

  ICUDA_HOST void uninitFromCPU ();
  ICUDA_HOST void uninitOnGPU ();

  ICUDA_HOST void copyFromCPU (TCoord<grid_coord, true>, TCoord<grid_coord, true>);
  ICUDA_HOST void copyToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);
  ICUDA_HOST void copyBackToCPU (time_step N, bool finalCopy);
#else
  ICUDA_HOST void init (YeeGridLayout<Type, TCoord, layout_type> *layout, bool parallelLayout);
  ICUDA_HOST void initBlocks (time_step);
#endif

  ICUDA_HOST
  void
  initScheme (FPValue, FPValue);

#ifdef GPU_INTERNAL_SCHEME

#define SETUP_BLOCKS_AND_THREADS \
  ASSERT (diff3D.get1 () % SOLVER_SETTINGS.getNumCudaThreadsX () == 0); \
  ASSERT (diff3D.get2 () % SOLVER_SETTINGS.getNumCudaThreadsY () == 0); \
  ASSERT (diff3D.get3 () % SOLVER_SETTINGS.getNumCudaThreadsZ () == 0); \
  dim3 blocks (diff3D.get1 () == 0 ? 1 : diff3D.get1 () / SOLVER_SETTINGS.getNumCudaThreadsX (), \
               diff3D.get2 () == 0 ? 1 : diff3D.get2 () / SOLVER_SETTINGS.getNumCudaThreadsY (), \
               diff3D.get3 () == 0 ? 1 : diff3D.get3 () / SOLVER_SETTINGS.getNumCudaThreadsZ ()); \
  dim3 threads (diff3D.get1 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsX (), \
                diff3D.get2 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsY (), \
                diff3D.get3 () == 0 ? 1 : SOLVER_SETTINGS.getNumCudaThreadsZ ()); \

  template <uint8_t grid_type>
  CUDA_HOST
  void calculateFieldStepIterationKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
                                                TCS diff11,
                                                TCS diff12,
                                                TCS diff21,
                                                TCS diff22,
                                                IGRID<TC> *grid,
                                                IGRID<TC> *oppositeGrid1,
                                                IGRID<TC> *oppositeGrid2,
                                                SourceCallBack rightSideFunc,
                                                IGRID<TC> *Ca,
                                                IGRID<TC> *Cb,
                                                bool usePML,
                                                GridType gridType,
                                                IGRID<TC> *materialGrid,
                                                GridType materialGridType,
                                                FPValue materialModifier,
                                                bool usePrecomputedGrids)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationKernel<Type, TCoord, layout_type, grid_type> <<< blocks, threads >>>
      (d_gpuScheme, start3D, t, diff11, diff12, diff21, diff22, grid, oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb, ct1, ct2, ct3,
       usePML, gridType, materialGrid, materialGridType, materialModifier, usePrecomputedGrids);
    cudaCheckError ();
  }

  CUDA_HOST
  void calculateFieldStepIterationPMLMetamaterialsKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
                                               TC pos,
                                               IGRID<TC> *grid,
                                               IGRID<TC> *gridPML,
                                               IGRID<TC> *CB0,
                                               IGRID<TC> *CB1,
                                               IGRID<TC> *CB2,
                                               IGRID<TC> *CA1,
                                               IGRID<TC> *CA2,
                                               GridType gridType,
                                               IGRID<TC> *materialGrid1,
                                               GridType materialGridType1,
                                               IGRID<TC> *materialGrid2,
                                               GridType materialGridType2,
                                               IGRID<TC> *materialGrid3,
                                               GridType materialGridType3,
                                               FPValue materialModifier)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationPMLMetamaterialsKernel<Type, TCoord, layout_type> <<< blocks, threads >>>
      (d_gpuScheme, start3D, t, pos, grid, gridPML, CB0, CB1, CB2, CA1, CA2, gridType,
       materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
       materialModifier);
    cudaCheckError ();
  }

  template <bool useMetamaterials>
  CUDA_HOST
  void calculateFieldStepIterationPMLKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
                                                 TC pos,
                                                 IGRID<TC> *grid,
                                                 IGRID<TC> *gridPML1,
                                                 IGRID<TC> *gridPML2,
                                                 IGRID<TC> *Ca,
                                                 IGRID<TC> *Cb,
                                                 IGRID<TC> *Cc,
                                                 GridType gridPMLType1,
                                                 IGRID<TC> *materialGrid1,
                                                 GridType materialGridType1,
                                                 IGRID<TC> *materialGrid4,
                                                 GridType materialGridType4,
                                                 IGRID<TC> *materialGrid5,
                                                 GridType materialGridType5,
                                                 FPValue materialModifier)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationPMLKernel<Type, TCoord, layout_type, useMetamaterials> <<< blocks, threads >>>
      (d_gpuScheme, start3D, t, pos, grid, gridPML1, gridPML2, Ca, Cb, Cc, gridPMLType1,
        materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
        materialModifier);
    cudaCheckError ();
  }

#define SHIFT_IN_TIME_KERNEL_LAUNCH(NAME) \
  CUDA_HOST \
  void shiftInTimeKernelLaunch ## NAME (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuSchemeOnGPU) \
  { \
    dim3 blocks (1, 1, 1); \
    dim3 threads (1, 1, 1); \
    InternalSchemeKernelHelpers::shiftInTimeKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU); \
    cudaCheckError (); \
  }

  SHIFT_IN_TIME_KERNEL_LAUNCH(Ex)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Ey)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Ez)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hx)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hy)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Hz)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dx)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dy)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Dz)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Bx)
  SHIFT_IN_TIME_KERNEL_LAUNCH(By)
  SHIFT_IN_TIME_KERNEL_LAUNCH(Bz)
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1x)
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1y)
  SHIFT_IN_TIME_KERNEL_LAUNCH(D1z)
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1x)
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1y)
  SHIFT_IN_TIME_KERNEL_LAUNCH(B1z)

#undef SHIFT_IN_TIME_KERNEL_LAUNCH

#define SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(NAME) \
  CUDA_HOST \
  void shiftInTimePlaneWaveKernelLaunch ## NAME (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuSchemeOnGPU) \
  { \
    dim3 blocks (1, 1, 1); \
    dim3 threads (1, 1, 1); \
    InternalSchemeKernelHelpers::shiftInTimePlaneWaveKernel ## NAME <Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU); \
    cudaCheckError (); \
  }

  SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(EInc)
  SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH(HInc)

#undef SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH

  CUDA_HOST
  void performPlaneWaveEStepsKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuSchemeOnGPU,
                                           time_step t, GridCoordinate1D Start, GridCoordinate1D End)
  {
    GridCoordinate1D diff = End - Start;
    int thrds = SOLVER_SETTINGS.getNumCudaThreadsX ()
                  * SOLVER_SETTINGS.getNumCudaThreadsY ()
                  * SOLVER_SETTINGS.getNumCudaThreadsZ ();
    ASSERT (diff.get1 () % thrds == 0);
    dim3 blocks (diff.get1 () / thrds, 1, 1);
    dim3 threads (thrds, 1, 1);
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X);
    InternalSchemeKernelHelpers::performPlaneWaveEStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU, t, Start, one);
    cudaCheckError ();
  }
  CUDA_HOST
  void performPlaneWaveHStepsKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuSchemeOnGPU,
                                           time_step t, GridCoordinate1D Start, GridCoordinate1D End)
  {
    GridCoordinate1D diff = End - Start;
    int thrds = SOLVER_SETTINGS.getNumCudaThreadsX ()
                  * SOLVER_SETTINGS.getNumCudaThreadsY ()
                  * SOLVER_SETTINGS.getNumCudaThreadsZ ();
    ASSERT (diff.get1 () % thrds == 0);
    dim3 blocks (diff.get1 () / thrds, 1, 1);
    dim3 threads (thrds, 1, 1);
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X);
    InternalSchemeKernelHelpers::performPlaneWaveHStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU, t, Start, one);
    cudaCheckError ();
  }

#endif /* GPU_INTERNAL_SCHEME */

  ICUDA_DEVICE
  void performPlaneWaveESteps (time_step, GridCoordinate1D start, GridCoordinate1D end);
  ICUDA_DEVICE
  void performPlaneWaveHSteps (time_step, GridCoordinate1D start, GridCoordinate1D end);

  ICUDA_DEVICE
  FieldValue approximateIncidentWaveE (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveE<Type, TCoord> (pos, layout->getZeroIncCoordFP (), EInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }
  ICUDA_DEVICE
  FieldValue approximateIncidentWaveH (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveH<Type, TCoord> (pos, layout->getZeroIncCoordFP (), HInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }

  ICUDA_DEVICE
  FPValue getMaterial (const TC &, GridType, IGRID<TC> *, GridType);
  ICUDA_DEVICE
  FPValue getMetaMaterial (const TC &, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType,
                           FPValue &, FPValue &);

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEx () const { return doNeedEx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEy () const { return doNeedEy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEz () const { return doNeedEz; }

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHx () const { return doNeedHx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHy () const { return doNeedHy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHz () const { return doNeedHz; }

  ICUDA_DEVICE ICUDA_HOST
  CoordinateType getType1 ()
  {
    return ct1;
  }
  ICUDA_DEVICE ICUDA_HOST
  CoordinateType getType2 ()
  {
    return ct2;
  }
  ICUDA_DEVICE ICUDA_HOST
  CoordinateType getType3 ()
  {
    return ct3;
  }

#define GRID_NAME(x) \
  IGRID<TC> * get ## x () \
  { \
    ASSERT (x); \
    return x; \
  }
#include "Grids.inc.h"
#undef GRID_NAME

  ICUDA_DEVICE ICUDA_HOST
  IGRID<GridCoordinate1D> * getEInc ()
  {
    ASSERT (EInc);
    return EInc;
  }
  ICUDA_DEVICE ICUDA_HOST
  IGRID<GridCoordinate1D> * getHInc ()
  {
    ASSERT (HInc);
    return HInc;
  }

  ICUDA_DEVICE ICUDA_HOST
  FPValue getGridTimeStep ()
  {
    return gridTimeStep;
  }

  ICUDA_DEVICE ICUDA_HOST
  YeeGridLayout<Type, TCoord, layout_type> * getYeeLayout ()
  {
    ASSERT (yeeLayout);
    return yeeLayout;
  }
};

#include "InternalScheme.template.inc.h"

#undef INTERNAL_SCHEME_BASE
#undef INTERNAL_SCHEME_HELPER
#undef IGRID
#undef ICUDA_HOST
#undef ICUDA_DEVICE

#undef INTERNAL_SCHEME_1D
#undef INTERNAL_SCHEME_2D
#undef INTERNAL_SCHEME_3D

#undef INTERNAL_SCHEME_1D_EX_HY
#undef INTERNAL_SCHEME_1D_EX_HZ
#undef INTERNAL_SCHEME_1D_EY_HX
#undef INTERNAL_SCHEME_1D_EY_HZ
#undef INTERNAL_SCHEME_1D_EZ_HX
#undef INTERNAL_SCHEME_1D_EZ_HY
#undef INTERNAL_SCHEME_2D_TEX
#undef INTERNAL_SCHEME_2D_TEY
#undef INTERNAL_SCHEME_2D_TEZ
#undef INTERNAL_SCHEME_2D_TMX
#undef INTERNAL_SCHEME_2D_TMY
#undef INTERNAL_SCHEME_2D_TMZ
#undef INTERNAL_SCHEME_3D_3D

#undef SETUP_BLOCKS_AND_THREADS
