#ifdef GPU_INTERNAL_SCHEME

namespace InternalSchemeKernelHelpers
{
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type, bool usePML, bool useMetamaterials>
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
                                          CoordinateType ct3)
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

    gpuScheme->calculateFieldStepIteration<grid_type, usePML> (t, pos, posAbs, diff11, diff12, diff21, diff22,
                                                               grid, coordFP,
                                                               oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb);
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
  copyGridsBackToCPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                      time_step N,
                      bool finalCopy);

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  ICUDA_HOST static void
  copyGridsToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme,
                  InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme);

#else /* GPU_INTERNAL_SCHEME */

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
  ICUDA_DEVICE ICUDA_HOST
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *EInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.0, EInc, incAngle1, incAngle2);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  ICUDA_DEVICE ICUDA_HOST
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true> realCoord, TCoord<FPValue, true> zeroCoord, IGRID<GridCoordinate1D> *HInc, FPValue incAngle1, FPValue incAngle2)
  {
    return approximateIncidentWave<Type, TCoord> (realCoord, zeroCoord, 0.5, HInc, incAngle1, incAngle2);
  }

#ifndef GPU_INTERNAL_SCHEME
#if defined (PARALLEL_GRID)

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static
  void allocateParallelGrids (InternalScheme<Type, ParallelGridCoordinateTemplate, layout_type> *intScheme);

#endif /* PARALLEL_GRID */

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static
  void performNSteps1D (INTERNAL_SCHEME_BASE<Type, GridCoordinate1DTemplate, layout_type> *intScheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < intScheme->blockCount.get1 (); ++c1)
    {
      GridCoordinate1D blockIdx = GRID_COORDINATE_1D (c1, intScheme->ct1);

      // TODO: save block to prev blocks storage
      intScheme->performNStepsForBlock (tStart, N, blockIdx);
    }

    intScheme->share ();
    intScheme->rebalance ();
  }

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static
  void performNSteps2D (INTERNAL_SCHEME_BASE<Type, GridCoordinate2DTemplate, layout_type> *intScheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < intScheme->blockCount.get1 (); ++c1)
    {
      for (grid_coord c2 = 0; c2 < intScheme->blockCount.get2 (); ++c2)
      {
        GridCoordinate2D blockIdx = GRID_COORDINATE_2D (c1, c2, intScheme->ct1, intScheme->ct2);

        // TODO: save block to prev blocks storage
        intScheme->performNStepsForBlock (tStart, N, blockIdx);
      }
    }

    intScheme->share ();
    intScheme->rebalance ();
  }

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static
  void performNSteps3D (INTERNAL_SCHEME_BASE<Type, GridCoordinate3DTemplate, layout_type> *intScheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < intScheme->blockCount.get1 (); ++c1)
    {
      for (grid_coord c2 = 0; c2 < intScheme->blockCount.get2 (); ++c2)
      {
        for (grid_coord c3 = 0; c3 < intScheme->blockCount.get3 (); ++c3)
        {
          GridCoordinate3D blockIdx = GRID_COORDINATE_3D (c1, c2, c3, intScheme->ct1, intScheme->ct2, intScheme->ct3);

          // TODO: save block to prev blocks storage
          intScheme->performNStepsForBlock (tStart, N, blockIdx);
        }
      }
    }

    intScheme->share ();
    intScheme->rebalance ();
  }
#endif /* !GPU_INTERNAL_SCHEME */

  ICUDA_DEVICE ICUDA_HOST
  static bool doSkipBorderFunc1D (GridCoordinate1D pos, IGRID<GridCoordinate1D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1;
  }
  ICUDA_DEVICE ICUDA_HOST
  static bool doSkipBorderFunc2D (GridCoordinate2D pos, IGRID<GridCoordinate2D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1;
  }
  ICUDA_DEVICE ICUDA_HOST
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

#ifdef CUDA_ENABLED
#ifdef GPU_INTERNAL_SCHEME

  template <SchemeType_t Type1, template <typename, bool> class TCoord1, LayoutType layout_type1>
  friend class InternalScheme;
  friend class InternalSchemeHelper;

#else /* GPU_INTERNAL_SCHEME */

  template <SchemeType_t Type1, template <typename, bool> class TCoord1, LayoutType layout_type1>
  friend class InternalSchemeGPU;
  friend class InternalSchemeHelperGPU;

#endif /* !GPU_INTERNAL_SCHEME */
#endif

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

  /**
   * Field grids
   */
  IGRID<TC> *Ex;
  IGRID<TC> *Ey;
  IGRID<TC> *Ez;
  IGRID<TC> *Hx;
  IGRID<TC> *Hy;
  IGRID<TC> *Hz;

  IGRID<TC> *Dx;
  IGRID<TC> *Dy;
  IGRID<TC> *Dz;
  IGRID<TC> *Bx;
  IGRID<TC> *By;
  IGRID<TC> *Bz;

  /**
   * Auxiliary field grids
   */
  IGRID<TC> *D1x;
  IGRID<TC> *D1y;
  IGRID<TC> *D1z;
  IGRID<TC> *B1x;
  IGRID<TC> *B1y;
  IGRID<TC> *B1z;

  /**
   * Amplitude field grids
   */
  IGRID<TC> *ExAmplitude;
  IGRID<TC> *EyAmplitude;
  IGRID<TC> *EzAmplitude;
  IGRID<TC> *HxAmplitude;
  IGRID<TC> *HyAmplitude;
  IGRID<TC> *HzAmplitude;

  /**
   * Material grids
   */
  IGRID<TC> *Eps;
  IGRID<TC> *Mu;

  /**
   * Sigmas
   */
  IGRID<TC> *SigmaX;
  IGRID<TC> *SigmaY;
  IGRID<TC> *SigmaZ;

  /**
   * Metamaterial grids
   */
  IGRID<TC> *OmegaPE;
  IGRID<TC> *GammaE;

  IGRID<TC> *OmegaPM;
  IGRID<TC> *GammaM;

  /**
   * Helper grids
   */
  IGRID<TC> *CaEx;
  IGRID<TC> *CbEx;
  IGRID<TC> *CaEy;
  IGRID<TC> *CbEy;
  IGRID<TC> *CaEz;
  IGRID<TC> *CbEz;
  IGRID<TC> *DaHx;
  IGRID<TC> *DbHx;
  IGRID<TC> *DaHy;
  IGRID<TC> *DbHy;
  IGRID<TC> *DaHz;
  IGRID<TC> *DbHz;

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

  TC blockCount;
  TC blockSize;

#ifndef GPU_INTERNAL_SCHEME
  time_step totalTimeSteps;
  time_step NTimeSteps;

  bool useParallel;

  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntScheme;
  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntSchemeOnGPU;
  InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuIntSchemeOnGPU;

#endif /* !GPU_INTERNAL_SCHEME */

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
  ICUDA_DEVICE ICUDA_HOST
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &, TC, TC, TC, TC);

#ifndef GPU_INTERNAL_SCHEME
  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_HOST
  void calculateFieldStep (time_step, TC, TC);
#endif /* !GPU_INTERNAL_SCHEME */

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_HOST
  void calculateFieldStepInit (IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *,
    IGRID<TC> **, IGRID<TC> **);

  template <uint8_t grid_type>
  ICUDA_HOST
  void calculateFieldStepInitDiff (TCS *, TCS *, TCS *, TCS *);

  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <bool useMetamaterials>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationPML (time_step, TC, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationBorder (time_step, TC, IGRID<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIterationExact (time_step, TC, IGRID<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);

  template<uint8_t EnumVal>
  ICUDA_DEVICE ICUDA_HOST
  void performPointSourceCalc (time_step);

  ICUDA_DEVICE ICUDA_HOST
  FieldValue calcField (const FieldValue & prev, const FieldValue & oppositeField12, const FieldValue & oppositeField11,
                        const FieldValue & oppositeField22, const FieldValue & oppositeField21, const FieldValue & prevRightSide,
                        const FieldValue & Ca, const FieldValue & Cb, const FPValue & delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return prev * Ca + tmp * Cb;
  }

  ICUDA_DEVICE ICUDA_HOST
  FieldValue calcFieldDrude (const FieldValue & curDOrB, const FieldValue & prevDOrB, const FieldValue & prevPrevDOrB,
                             const FieldValue & prevEOrH, const FieldValue & prevPrevEOrH,
                             const FieldValue & b0, const FieldValue & b1, const FieldValue & b2, const FieldValue & a1, const FieldValue & a2)
  {
    return curDOrB * b0 + prevDOrB * b1 + prevPrevDOrB * b2 - prevEOrH * a1 - prevPrevEOrH * a2;
  }

  ICUDA_DEVICE ICUDA_HOST
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

  template <uint8_t grid_type, bool usePML>
  ICUDA_DEVICE ICUDA_HOST
  void calculateFieldStepIteration (time_step, TC, TC, TCS, TCS, TCS, TCS, IGRID<TC> *, TCFP,
                                    IGRID<TC> *, IGRID<TC> *, SourceCallBack, IGRID<TC> *, IGRID<TC> *);

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

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
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
                                                IGRID<TC> *Cb)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationKernel<Type, TCoord, layout_type, grid_type, usePML, useMetamaterials> <<< blocks, threads >>>
      (d_gpuScheme, start3D, t, diff11, diff12, diff21, diff22, grid, oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb, ct1, ct2, ct3);
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

#ifndef GPU_INTERNAL_SCHEME

  /**
   * Perform all time steps for scheme
   */
  ICUDA_HOST
  void performSteps ()
  {
    for (time_step t = 0; t < totalTimeSteps; t += NTimeSteps)
    {
      /*
       * Each NTimeSteps sharing will be performed.
       *
       * For sequential solver, NTimeSteps == totalTimeSteps
       * For parallel/cuda solver, NTimeSteps == min (bufSize, cudaBufSize)
       */
      performNSteps (t, NTimeSteps);
    }
  }

  ICUDA_HOST
  void performNSteps (time_step tStart, time_step N);

  ICUDA_HOST
  void performNStepsForBlock (time_step tStart, time_step N, TC blockIdx);

  ICUDA_HOST
  void share ();

  ICUDA_HOST
  void rebalance ();

  /**
   * Perform computations of single time step for specific field and for specified chunk.
   *
   * NOTE: For GPU InternalScheme this method is not defined, because it is supposed to be ran on CPU only,
   *       and call kernels deeper in call tree.
   *
   * NOTE: Start and End coordinates should correctly consider buffers in parallel grid,
   *       which means, that computations are not performed for incorrect grid points.
   */
  template <uint8_t grid_type>
  ICUDA_HOST
  void performFieldSteps (time_step t, /**< time step to compute */
                          TC Start, /**< start coordinate of chunk to compute */
                          TC End) /**< end coordinate of chunk to compute */
  {
    /*
     * TODO: remove check performed on each iteration
     */
    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, true, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, true, false> (t, Start, End);
      }
    }
    else
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, false, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, false, false> (t, Start, End);
      }
    }

    bool doUsePointSource;
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEz ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHz ();
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doUsePointSource)
    {
      performPointSourceCalc<grid_type> (t);
    }
  }
#endif

  ICUDA_DEVICE
  void performPlaneWaveESteps (time_step, GridCoordinate1D start, GridCoordinate1D end);
  ICUDA_DEVICE
  void performPlaneWaveHSteps (time_step, GridCoordinate1D start, GridCoordinate1D end);

  ICUDA_DEVICE ICUDA_HOST
  FieldValue approximateIncidentWaveE (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveE<Type, TCoord> (pos, layout->getZeroIncCoordFP (), EInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }
  ICUDA_DEVICE ICUDA_HOST
  FieldValue approximateIncidentWaveH (TCFP pos)
  {
    YeeGridLayout<Type, TCoord, layout_type> *layout = INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::yeeLayout;
    return INTERNAL_SCHEME_HELPER::approximateIncidentWaveH<Type, TCoord> (pos, layout->getZeroIncCoordFP (), HInc, layout->getIncidentWaveAngle1 (), layout->getIncidentWaveAngle2 ());
  }

#ifndef GPU_INTERNAL_SCHEME
  ICUDA_HOST
  FPValue getMaterial (const TC &, GridType, IGRID<TC> *, GridType);
  ICUDA_HOST
  FPValue getMetaMaterial (const TC &, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType, IGRID<TC> *, GridType,
                           FPValue &, FPValue &);
#endif

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEx () const { return doNeedEx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEy () const { return doNeedEy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedEz () const { return doNeedEz; }

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHx () const { return doNeedHx; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHy () const { return doNeedHy; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedHz () const { return doNeedHz; }

  CoordinateType getType1 ()
  {
    return ct1;
  }
  CoordinateType getType2 ()
  {
    return ct2;
  }
  CoordinateType getType3 ()
  {
    return ct3;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEx ()
  {
    ASSERT (Ex);
    ASSERT (doNeedEx);
    return Ex;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEy ()
  {
    ASSERT (Ey);
    ASSERT (doNeedEy);
    return Ey;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEz ()
  {
    ASSERT (Ez);
    ASSERT (doNeedEz);
    return Ez;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHx ()
  {
    ASSERT (Hx);
    ASSERT (doNeedHx);
    return Hx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHy ()
  {
    ASSERT (Hy);
    ASSERT (doNeedHy);
    return Hy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getHz ()
  {
    ASSERT (Hz);
    ASSERT (doNeedHz);
    return Hz;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getEps ()
  {
    ASSERT (Eps);
    return Eps;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getMu ()
  {
    ASSERT (Mu);
    return Mu;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaX ()
  {
    ASSERT (SigmaX);
    ASSERT (doNeedSigmaX);
    return SigmaX;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaY ()
  {
    ASSERT (SigmaY);
    ASSERT (doNeedSigmaY);
    return SigmaY;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getSigmaZ ()
  {
    ASSERT (SigmaZ);
    ASSERT (doNeedSigmaZ);
    return SigmaZ;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getOmegaPE ()
  {
    ASSERT (OmegaPE);
    return OmegaPE;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getOmegaPM ()
  {
    ASSERT (OmegaPM);
    return OmegaPM;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getGammaE ()
  {
    ASSERT (GammaE);
    return GammaE;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getGammaM ()
  {
    ASSERT (GammaM);
    return GammaM;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDx ()
  {
    ASSERT (Dx);
    ASSERT (doNeedEx);
    return Dx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDy ()
  {
    ASSERT (Dy);
    ASSERT (doNeedEy);
    return Dy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDz ()
  {
    ASSERT (Dz);
    ASSERT (doNeedEz);
    return Dz;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBx ()
  {
    ASSERT (Bx);
    ASSERT (doNeedHx);
    return Bx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBy ()
  {
    ASSERT (By);
    ASSERT (doNeedHy);
    return By;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getBz ()
  {
    ASSERT (Bz);
    ASSERT (doNeedHz);
    return Bz;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1x ()
  {
    ASSERT (D1x);
    ASSERT (doNeedEx);
    return D1x;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1y ()
  {
    ASSERT (D1y);
    ASSERT (doNeedEy);
    return D1y;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getD1z ()
  {
    ASSERT (D1z);
    ASSERT (doNeedEz);
    return D1z;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1x ()
  {
    ASSERT (B1x);
    ASSERT (doNeedHx);
    return B1x;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1y ()
  {
    ASSERT (B1y);
    ASSERT (doNeedHy);
    return B1y;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getB1z ()
  {
    ASSERT (B1z);
    ASSERT (doNeedHz);
    return B1z;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCaEx ()
  {
    ASSERT (CaEx);
    return CaEx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCbEx ()
  {
    ASSERT (CbEx);
    return CbEx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCaEy ()
  {
    ASSERT (CaEy);
    return CaEy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCbEy ()
  {
    ASSERT (CbEy);
    return CbEy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCaEz ()
  {
    ASSERT (CaEz);
    return CaEz;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getCbEz ()
  {
    ASSERT (CbEz);
    return CbEz;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDaHx ()
  {
    ASSERT (DaHx);
    return DaHx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDbHx ()
  {
    ASSERT (DbHx);
    return DbHx;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDaHy ()
  {
    ASSERT (DaHy);
    return DaHy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDbHy ()
  {
    ASSERT (DbHy);
    return DbHy;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDaHz ()
  {
    ASSERT (DaHz);
    return DaHz;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<TC> * getDbHz ()
  {
    ASSERT (DbHz);
    return DbHz;
  }

  ICUDA_DEVICE ICUDA_HOST IGRID<GridCoordinate1D> * getEInc ()
  {
    ASSERT (EInc);
    return EInc;
  }
  ICUDA_DEVICE ICUDA_HOST IGRID<GridCoordinate1D> * getHInc ()
  {
    ASSERT (HInc);
    return HInc;
  }

  ICUDA_DEVICE ICUDA_HOST
  FPValue getTimeStep ()
  {
    return gridTimeStep;
  }

  ICUDA_DEVICE ICUDA_HOST
  YeeGridLayout<Type, TCoord, layout_type> * getYeeLayout ()
  {
    ASSERT (yeeLayout);
    return yeeLayout;
  }

// #ifndef GPU_INTERNAL_SCHEME
//
//   InternalSchemeGPU<Type, TCoord, layout_type> *getGPUInternalScheme ()
//   {
//     ASSERT (gpuIntScheme != NULLPTR);
//     return gpuIntScheme;
//   }
//
//   void setGPUInternalScheme (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
//   {
//     ASSERT (gpuScheme != NULLPTR);
//     gpuIntScheme = gpuScheme;
//   }
//
//   InternalSchemeGPU<Type, TCoord, layout_type> *getGPUInternalSchemeOnGPU ()
//   {
//     ASSERT (gpuIntSchemeOnGPU != NULLPTR);
//     return gpuIntSchemeOnGPU;
//   }
//
//   void setGPUInternalSchemeOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuSchemeOnGPU)
//   {
//     ASSERT (gpuSchemeOnGPU != NULLPTR);
//     gpuIntSchemeOnGPU = gpuSchemeOnGPU;
//   }
//
//   InternalSchemeGPU<Type, TCoord, layout_type> *getGPUInternalSchemeOnGPUFull ()
//   {
//     ASSERT (d_gpuIntSchemeOnGPU != NULLPTR);
//     return d_gpuIntSchemeOnGPU;
//   }
//
//   void setGPUInternalSchemeOnGPUFull (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme)
//   {
//     ASSERT (d_gpuScheme != NULLPTR);
//     d_gpuIntSchemeOnGPU = d_gpuScheme;
//   }
//
// #endif /* !GPU_INTERNAL_SCHEME */
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

#undef SHIFT_IN_TIME_KERNEL
#undef SHIFT_IN_TIME_PLANE_WAVE_KERNEL
#undef SETUP_BLOCKS_AND_THREADS
#undef SHIFT_IN_TIME_KERNEL_LAUNCH
#undef SHIFT_IN_TIME_PLANE_WAVE_KERNEL_LAUNCH
