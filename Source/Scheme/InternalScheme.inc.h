#ifdef GPU_INTERNAL_SCHEME

namespace InternalSchemeKernelHelpers
{
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void performPointSourceCalcKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                     time_step t)
  {
    ASSERT (blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
    gpuScheme->performPointSourceCalc<grid_type> (t);
  }

  /**
   * GPU kernel.
   *
   * Perform calculateFieldStepIteration for specific thread in block. Thread corresponds 1 to 1 to grid point.
   */
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void calculateFieldStepIterationKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme, /**< GPU internal scheme */
                                          GridCoordinate3D start3D, /**< start coordinate of block, for which computations are performed */
                                          GridCoordinate3D end3D, /**< end coordinate of block, for which computations are performed */
                                          FPValue timestep, /**< time step to compute */
                                          TCoord<grid_coord, false> diff11, /**< offset in layout */
                                          TCoord<grid_coord, false> diff12, /**< offset in layout */
                                          TCoord<grid_coord, false> diff21, /**< offset in layout */
                                          TCoord<grid_coord, false> diff22, /**< offset in layout */
                                          IGRID< TCoord<grid_coord, true> > *grid, /**< core grid to perform computations for */
                                          IGRID< TCoord<grid_coord, true> > *oppositeGrid1, /**< grid from circuit */
                                          IGRID< TCoord<grid_coord, true> > *oppositeGrid2, /**< grid from circuit */
                                          SourceCallBack rightSideFunc, /**< right side function to be called */
                                          IGRID< TCoord<grid_coord, true> > *Ca, /**< grid with precomputed values */
                                          IGRID< TCoord<grid_coord, true> > *Cb, /**< grid with precomputed values */
                                          CoordinateType ct1, /**< coordinate type for specified scheme */
                                          CoordinateType ct2, /**< coordinate type for specified scheme */
                                          CoordinateType ct3, /**< coordinate type for specified scheme */
                                          bool usePML, /**< flag whether to use PML */
                                          GridType gridType, /**< type of core grid */
                                          IGRID< TCoord<grid_coord, true> > *materialGrid, /**< material grid */
                                          GridType materialGridType, /**< type of material */
                                          FPValue materialModifier, /**< additional multiplier for material */
                                          bool usePrecomputedGrids) /**< flag whether to use precomputed values */
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                           (blockIdx.y * blockDim.y) + threadIdx.y,
                                                           (blockIdx.z * blockDim.z) + threadIdx.z,
                                                           CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    if (!(pos3D < end3D))
    {
      // skip kernels, which do not correspond to actual grid points
      return;
    }
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
      gpuScheme->calculateFieldStepIteration<grid_type, true> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                         grid, coordFP,
                                                         oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                         usePML,
                                                         gridType, materialGrid, materialGridType,
                                                         materialModifier);
    }
    else
    {
      gpuScheme->calculateFieldStepIteration<grid_type, false> (timestep, pos, posAbs, diff11, diff12, diff21, diff22,
                                                         grid, coordFP,
                                                         oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb,
                                                         usePML,
                                                         gridType, materialGrid, materialGridType,
                                                         materialModifier);
    }
  }

  /**
   * GPU kernel.
   *
   * Perform calculateFieldStepIteration for specific thread in block. Thread corresponds 1 to 1 to grid point.
   */
  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void calculateFieldStepIterationCurrentKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme, /**< GPU internal scheme */
                                                 FieldValue current, /**< value of current (J,M) */
                                                 IGRID< TCoord<grid_coord, true> > *grid, /**< core grid to perform computations for */
                                                 IGRID< TCoord<grid_coord, true> > *Ca, /**< grid with precomputed values */
                                                 IGRID< TCoord<grid_coord, true> > *Cb, /**< grid with precomputed values */
                                                 bool usePML, /**< flag whether to use PML */
                                                 GridType gridType, /**< type of core grid */
                                                 IGRID< TCoord<grid_coord, true> > *materialGrid, /**< material grid */
                                                 GridType materialGridType, /**< type of material */
                                                 FPValue materialModifier, /**< additional multiplier for material */
                                                 bool usePrecomputedGrids) /**< flag whether to use precomputed values */
  {
    ASSERT (blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);

    if (usePrecomputedGrids)
    {
      gpuScheme->calculateFieldStepIterationCurrent<grid_type, true> (current,
                                                                      grid, Ca, Cb,
                                                                      usePML,
                                                                      gridType, materialGrid, materialGridType,
                                                                      materialModifier);
    }
    else
    {
      gpuScheme->calculateFieldStepIterationCurrent<grid_type, false> (current,
                                                                      grid, Ca, Cb,
                                                                      usePML,
                                                                      gridType, materialGrid, materialGridType,
                                                                      materialModifier);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void calculateFieldStepIterationPMLMetamaterialsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          GridCoordinate3D end3D,
                                          time_step t,
                                         IGRID< TCoord<grid_coord, true> > *grid,
                                         IGRID< TCoord<grid_coord, true> > *gridPML,
                                         IGRID< TCoord<grid_coord, true> > *CB0,
                                         IGRID< TCoord<grid_coord, true> > *CB1,
                                         IGRID< TCoord<grid_coord, true> > *CB2,
                                         IGRID< TCoord<grid_coord, true> > *CA1,
                                         IGRID< TCoord<grid_coord, true> > *CA2,
                                         CoordinateType ct1,
                                         CoordinateType ct2,
                                         CoordinateType ct3,
                                         GridType gridType,
                                         IGRID< TCoord<grid_coord, true> > *materialGrid1,
                                         GridType materialGridType1,
                                         IGRID< TCoord<grid_coord, true> > *materialGrid2,
                                         GridType materialGridType2,
                                         IGRID< TCoord<grid_coord, true> > *materialGrid3,
                                         GridType materialGridType3,
                                         FPValue materialModifier,
                                         bool usePrecomputedGrids)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    if (!(pos3D < end3D))
    {
      // skip kernels, which do not correspond to actual grid points
      return;
    }
    TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    if (usePrecomputedGrids)
    {
      gpuScheme->calculateFieldStepIterationPMLMetamaterials<true> (t, pos, grid, gridPML, CB0, CB1, CB2, CA1, CA2,
        gridType,
        materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
        materialModifier);
    }
    else
    {
      gpuScheme->calculateFieldStepIterationPMLMetamaterials<false> (t, pos, grid, gridPML, CB0, CB1, CB2, CA1, CA2,
        gridType,
        materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
        materialModifier);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, bool useMetamaterials>
  __global__
  void calculateFieldStepIterationPMLKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          GridCoordinate3D end3D,
                                          time_step t,
                                           IGRID< TCoord<grid_coord, true> > *grid,
                                           IGRID< TCoord<grid_coord, true> > *gridPML1,
                                           IGRID< TCoord<grid_coord, true> > *gridPML2,
                                           IGRID< TCoord<grid_coord, true> > *Ca,
                                           IGRID< TCoord<grid_coord, true> > *Cb,
                                           IGRID< TCoord<grid_coord, true> > *Cc,
                                           CoordinateType ct1,
                                           CoordinateType ct2,
                                           CoordinateType ct3,
                                           GridType gridPMLType1,
                                           IGRID< TCoord<grid_coord, true> > *materialGrid1,
                                           GridType materialGridType1,
                                           IGRID< TCoord<grid_coord, true> > *materialGrid4,
                                           GridType materialGridType4,
                                           IGRID< TCoord<grid_coord, true> > *materialGrid5,
                                           GridType materialGridType5,
                                           FPValue materialModifier,
                                           bool usePrecomputedGrids)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    if (!(pos3D < end3D))
    {
      // skip kernels, which do not correspond to actual grid points
      return;
    }
    TCoord<grid_coord, true> pos = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    if (usePrecomputedGrids)
    {
      gpuScheme->calculateFieldStepIterationPML<useMetamaterials, true> (t, pos, grid, gridPML1, gridPML2, Ca, Cb, Cc,
        gridPMLType1, materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
        materialModifier);
    }
    else
    {
      gpuScheme->calculateFieldStepIterationPML<useMetamaterials, false> (t, pos, grid, gridPML1, gridPML2, Ca, Cb, Cc,
        gridPMLType1, materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
        materialModifier);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void calculateFieldStepIterationBorderKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          GridCoordinate3D end3D,
                                          time_step t,
                                          IGRID< TCoord<grid_coord, true> > *grid,
                                          SourceCallBack borderFunc,
                                          CoordinateType ct1,
                                          CoordinateType ct2,
                                          CoordinateType ct3)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    if (!(pos3D < end3D))
    {
      // skip kernels, which do not correspond to actual grid points
      return;
    }
    TCoord<grid_coord, true> posAbs = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    gpuScheme->calculateFieldStepIterationBorder<grid_type> (t, posAbs, grid, borderFunc);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type, uint8_t grid_type>
  __global__
  void calculateFieldStepIterationExactKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                          GridCoordinate3D start3D,
                                          GridCoordinate3D end3D,
                                          time_step t,
                                          IGRID< TCoord<grid_coord, true> > *grid,
                                          SourceCallBack exactFunc,
                                          CoordinateType ct1,
                                          CoordinateType ct2,
                                          CoordinateType ct3)
  {
    GridCoordinate3D pos3D = start3D + GRID_COORDINATE_3D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                                         (blockIdx.y * blockDim.y) + threadIdx.y,
                                                         (blockIdx.z * blockDim.z) + threadIdx.z,
                                                          CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    if (!(pos3D < end3D))
    {
      // skip kernels, which do not correspond to actual grid points
      return;
    }
    TCoord<grid_coord, true> posAbs = TCoord<grid_coord, true>::initAxesCoordinate (pos3D.get1 (), pos3D.get2 (), pos3D.get3 (), ct1, ct2, ct3);

    FPValue normRe = 0;
    FPValue normIm = 0;
    FPValue normMod = 0;
    FPValue maxRe = 0;
    FPValue maxIm = 0;
    FPValue maxMod = 0;

    gpuScheme->calculateFieldStepIterationExact<grid_type> (t, posAbs, grid, exactFunc, normRe, normIm, normMod, maxRe, maxIm, maxMod);

    atomicAdd (&gpuScheme->getd_norm ()[0], normRe);
    atomicAdd (&gpuScheme->getd_norm ()[1], normRe);
    atomicAdd (&gpuScheme->getd_norm ()[2], normIm);
    atomicAdd (&gpuScheme->getd_norm ()[3], maxRe);
    atomicAdd (&gpuScheme->getd_norm ()[4], maxIm);
    atomicAdd (&gpuScheme->getd_norm ()[5], maxMod);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void performPlaneWaveEStepsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                     time_step t, GridCoordinate1D Start, GridCoordinate1D End, GridCoordinate1D CoordPerKernel)
  {
    GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                 CoordinateType::X);
    posStart = posStart * CoordPerKernel + Start;
    GridCoordinate1D posEnd = posStart + CoordPerKernel;
    if (posStart >= End)
    {
      return;
    }
    if (posEnd >= End)
    {
      posEnd = End;
    }
    gpuScheme->performPlaneWaveESteps (t, posStart, posEnd);
  }
    template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  __global__
  void performPlaneWaveHStepsKernel (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                     time_step t, GridCoordinate1D Start, GridCoordinate1D End, GridCoordinate1D CoordPerKernel)
  {
    GridCoordinate1D posStart = GRID_COORDINATE_1D ((blockIdx.x * blockDim.x) + threadIdx.x,
                                 CoordinateType::X);
    posStart = posStart * CoordPerKernel + Start;
    GridCoordinate1D posEnd = posStart + CoordPerKernel;
    if (posStart >= End)
    {
      return;
    }
    if (posEnd >= End)
    {
      posEnd = End;
    }
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
  static void allocateParallelGrids1D (InternalScheme<Type, GridCoordinate1DTemplate, layout_type> *intScheme);

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static void allocateParallelGrids2D (InternalScheme<Type, GridCoordinate2DTemplate, layout_type> *intScheme);

  template <SchemeType_t Type, LayoutType layout_type>
  ICUDA_HOST
  static void allocateParallelGrids3D (InternalScheme<Type, GridCoordinate3DTemplate, layout_type> *intScheme);
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
private:

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

private:

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

#define CALLBACK_NAME(x) \
  SourceCallBack x;
#include "Callbacks.inc.h"
#undef CALLBACK_NAME

#ifdef GPU_INTERNAL_SCHEME
  FPValue *d_norm;
#endif /* GPU_INTERNAL_SCHEME */

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

private:

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

  ICUDA_HOST void allocateParallelGrids ();

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

  ICUDA_HOST void initCoordTypes ()
  {
    ct1 = get_ct1 ();
    ct2 = get_ct2 ();
    ct3 = get_ct3 ();
  }

  ICUDA_DEVICE bool doSkipBorderFunc (TC, IGRID<TC> *);

#ifdef ENABLE_ASSERTS
  ICUDA_DEVICE void calculateTFSFExAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
  ICUDA_DEVICE void calculateTFSFEyAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
  ICUDA_DEVICE void calculateTFSFEzAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
  ICUDA_DEVICE void calculateTFSFHxAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
  ICUDA_DEVICE void calculateTFSFHyAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
  ICUDA_DEVICE void calculateTFSFHzAsserts (TC pos11, TC pos12, TC pos21, TC pos22);
#endif /* ENABLE_ASSERTS */

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &, TC, TC, TC, TC);

  ICUDA_DEVICE
  FieldValue calcField (const FieldValue & prev, const FieldValue & oppositeField12, const FieldValue & oppositeField11,
                        const FieldValue & oppositeField22, const FieldValue & oppositeField21, const FieldValue & prevRightSide,
                        const FieldValue & Ca, const FieldValue & Cb, const FPValue & delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 - prevRightSide * delta;
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

  ICUDA_DEVICE
  FieldValue calcCurrent (const FieldValue & current, const FieldValue & Cb, const FPValue & delta)
  {
    return current * delta * Cb;
  }

  ICUDA_HOST
  INTERNAL_SCHEME_BASE ();

  ICUDA_HOST
  ~INTERNAL_SCHEME_BASE ();

  template <uint8_t grid_type, bool usePrecomputedGrids>
  ICUDA_DEVICE
  void calculateFieldStepIteration (FPValue, TC, TC, TCS, TCS, TCS, TCS, IGRID<TC> *, TCFP,
                                    IGRID<TC> *, IGRID<TC> *, SourceCallBack, IGRID<TC> *, IGRID<TC> *, bool,
                                    GridType, IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type, bool usePrecomputedGrids>
  ICUDA_DEVICE
  void calculateFieldStepIterationCurrent (FieldValue, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *,
                                           bool, GridType, IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  ICUDA_HOST
  void calculateFieldStepInit (IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, GridType *, IGRID<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *,
    IGRID<TC> **, IGRID<TC> **,
    IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **, IGRID<TC> **);

#ifndef GPU_INTERNAL_SCHEME
  template <uint8_t grid_type>
  ICUDA_HOST
  void calculateFieldStepInitDiff (TCS *, TCS *, TCS *, TCS *);
#endif

  template <bool usePrecomputedGrids>
  ICUDA_DEVICE
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, IGRID<TC> *, IGRID<TC> *,
       IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <bool useMetamaterials, bool usePrecomputedGrids>
  ICUDA_DEVICE
  void calculateFieldStepIterationPML (time_step, TC, IGRID<TC> *, IGRID<TC> *, IGRID<TC> *,
       IGRID<TC> *, IGRID<TC> *, IGRID<TC> *, GridType,
       IGRID<TC> *, GridType,  IGRID<TC> *, GridType,  IGRID<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateFieldStepIterationBorder (time_step, TC, IGRID<TC> *, SourceCallBack);

  template <uint8_t grid_type>
  ICUDA_DEVICE
  void calculateFieldStepIterationExact (time_step, TC, IGRID<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);

  template<uint8_t EnumVal>
  ICUDA_DEVICE
  void performPointSourceCalc (time_step);

  template <bool usePrecomputedGrids>
  ICUDA_DEVICE
  void computeCaCb (FieldValue &, FieldValue &, TC, TC,
                    IGRID<TC> *, IGRID<TC> *, bool,
                    GridType, IGRID<TC> *, GridType, FPValue);

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
#endif

  ICUDA_HOST
  void
  initScheme (FPValue, FPValue);

#ifdef GPU_INTERNAL_SCHEME

  template <uint8_t grid_type>
  CUDA_HOST
  void performPointSourceCalcKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuSchemeOnGPU,
                                           time_step t)
  {
    dim3 blocks (1, 1, 1);
    dim3 threads (1, 1, 1);
    InternalSchemeKernelHelpers::performPointSourceCalcKernel <Type, TCoord, layout_type, grid_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU, t);
    cudaCheckError ();
  }

#define SETUP_BLOCKS_AND_THREADS \
  if (diff3D.get1 () % SOLVER_SETTINGS.getNumCudaThreadsX () != 0) { diff3D.set1 ((diff3D.get1 () / SOLVER_SETTINGS.getNumCudaThreadsX () + 1) * SOLVER_SETTINGS.getNumCudaThreadsX ()); } \
  if (diff3D.get2 () % SOLVER_SETTINGS.getNumCudaThreadsY () != 0) { diff3D.set2 ((diff3D.get2 () / SOLVER_SETTINGS.getNumCudaThreadsY () + 1) * SOLVER_SETTINGS.getNumCudaThreadsY ()); } \
  if (diff3D.get3 () % SOLVER_SETTINGS.getNumCudaThreadsZ () != 0) { diff3D.set3 ((diff3D.get3 () / SOLVER_SETTINGS.getNumCudaThreadsZ () + 1) * SOLVER_SETTINGS.getNumCudaThreadsZ ()); } \
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
                                                FPValue timestep,
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
      (d_gpuScheme, start3D, end3D, timestep, diff11, diff12, diff21, diff22, grid, oppositeGrid1, oppositeGrid2, rightSideFunc, Ca, Cb, ct1, ct2, ct3,
       usePML, gridType, materialGrid, materialGridType, materialModifier, usePrecomputedGrids);
    cudaCheckError ();
  }

  template <uint8_t grid_type>
  CUDA_HOST
  void calculateFieldStepIterationCurrentKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                       FieldValue current,
                                                       IGRID<TC> *grid,
                                                       IGRID<TC> *Ca,
                                                       IGRID<TC> *Cb,
                                                       bool usePML,
                                                       GridType gridType,
                                                       IGRID<TC> *materialGrid,
                                                       GridType materialGridType,
                                                       FPValue materialModifier,
                                                       bool usePrecomputedGrids)
  {
    dim3 blocks (1, 1, 1);
    dim3 threads (1, 1, 1);
    InternalSchemeKernelHelpers::calculateFieldStepIterationCurrentKernel<Type, TCoord, layout_type, grid_type> <<< blocks, threads >>>
      (d_gpuScheme, current, grid, Ca, Cb,
       usePML, gridType, materialGrid, materialGridType, materialModifier, usePrecomputedGrids);
    cudaCheckError ();
  }

  CUDA_HOST
  void calculateFieldStepIterationPMLMetamaterialsKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
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
                                               FPValue materialModifier,
                                               bool usePrecomputedGrids)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationPMLMetamaterialsKernel<Type, TCoord, layout_type> <<< blocks, threads >>>
      (d_gpuScheme, start3D, end3D, t, grid, gridPML, CB0, CB1, CB2, CA1, CA2, ct1, ct2, ct3, gridType,
       materialGrid1, materialGridType1, materialGrid2, materialGridType2, materialGrid3, materialGridType3,
       materialModifier, usePrecomputedGrids);
    cudaCheckError ();
  }

  template <bool useMetamaterials>
  CUDA_HOST
  void calculateFieldStepIterationPMLKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
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
                                                 FPValue materialModifier,
                                                 bool usePrecomputedGrids)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationPMLKernel<Type, TCoord, layout_type, useMetamaterials> <<< blocks, threads >>>
      (d_gpuScheme, start3D, end3D, t, grid, gridPML1, gridPML2, Ca, Cb, Cc, ct1, ct2, ct3, gridPMLType1,
        materialGrid1, materialGridType1, materialGrid4, materialGridType4, materialGrid5, materialGridType5,
        materialModifier, usePrecomputedGrids);
    cudaCheckError ();
  }

  template <uint8_t grid_type>
  CUDA_HOST
  void calculateFieldStepIterationBorderKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
                                                IGRID< TCoord<grid_coord, true> > *grid,
                                                SourceCallBack borderFunc)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationBorderKernel<Type, TCoord, layout_type, grid_type> <<< blocks, threads >>>
      (d_gpuScheme, start3D, end3D, t, grid, borderFunc, ct1, ct2, ct3);
    cudaCheckError ();
  }

  template <uint8_t grid_type>
  CUDA_HOST
  void calculateFieldStepIterationExactKernelLaunch (InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuScheme,
                                                     InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                                GridCoordinate3D start3D,
                                                GridCoordinate3D end3D,
                                                time_step t,
                                                IGRID< TCoord<grid_coord, true> > *grid,
                                                SourceCallBack exactFunc,
                                                FPValue & normRe,
                                                FPValue & normIm,
                                                FPValue & normMod,
                                                FPValue & maxRe,
                                                FPValue & maxIm,
                                                FPValue & maxMod)
  {
    GridCoordinate3D diff3D = end3D - start3D;
    cudaCheckErrorCmd (cudaMemset (gpuScheme->d_norm, 0, 6 * sizeof(FPValue)));
    SETUP_BLOCKS_AND_THREADS;
    InternalSchemeKernelHelpers::calculateFieldStepIterationExactKernel<Type, TCoord, layout_type, grid_type> <<< blocks, threads >>>
      (d_gpuScheme, start3D, end3D, t, grid, exactFunc, ct1, ct2, ct3);
    cudaCheckError ();
    FPValue buf[6];
    cudaCheckErrorCmd (cudaMemcpy (buf, gpuScheme->d_norm, 6 * sizeof(FPValue), cudaMemcpyDeviceToHost));
    normRe = buf[0];
    normIm = buf[1];
    normMod = buf[2];
    maxRe = buf[3];
    maxIm = buf[4];
    maxMod = buf[5];
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
    if (diff.get1 () % thrds != 0) { diff.set1 ((diff.get1 () / thrds + 1) * thrds); }
    ASSERT (diff.get1 () % thrds == 0);
    dim3 blocks (diff.get1 () / thrds, 1, 1);
    dim3 threads (thrds, 1, 1);
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X);
    InternalSchemeKernelHelpers::performPlaneWaveEStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU, t, Start, End, one);
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
    if (diff.get1 () % thrds != 0) { diff.set1 ((diff.get1 () / thrds + 1) * thrds); }
    ASSERT (diff.get1 () % thrds == 0);
    dim3 blocks (diff.get1 () / thrds, 1, 1);
    dim3 threads (thrds, 1, 1);
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X);
    InternalSchemeKernelHelpers::performPlaneWaveHStepsKernel<Type, TCoord, layout_type> <<< blocks, threads >>> (d_gpuSchemeOnGPU, t, Start, End, one);
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

  ICUDA_DEVICE ICUDA_HOST bool getDoNeedSigmaX () const { return doNeedSigmaX; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedSigmaY () const { return doNeedSigmaY; }
  ICUDA_DEVICE ICUDA_HOST bool getDoNeedSigmaZ () const { return doNeedSigmaZ; }

  ICUDA_HOST CoordinateType get_ct1 ()
  {
    switch (static_cast<uint8_t> (Type))
    {
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHy)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHx)):
      {
        return CoordinateType::Z;
      }
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMx)):
      {
        return CoordinateType::Y;
      }
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEz)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMz)):
      case (static_cast<uint8_t> (SchemeType::Dim3)):
      {
        return CoordinateType::X;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return CoordinateType::NONE;
  }

  ICUDA_HOST CoordinateType get_ct2 ()
  {
    switch (static_cast<uint8_t> (Type))
    {
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHy)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHx)):
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHx)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHy)):
      {
        return CoordinateType::NONE;
      }
      case (static_cast<uint8_t> (SchemeType::Dim2_TEx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMy)):
      {
        return CoordinateType::Z;
      }
      case (static_cast<uint8_t> (SchemeType::Dim2_TEz)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMz)):
      case (static_cast<uint8_t> (SchemeType::Dim3)):
      {
        return CoordinateType::Y;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return CoordinateType::NONE;
  }

  ICUDA_HOST CoordinateType get_ct3 ()
  {
    switch (static_cast<uint8_t> (Type))
    {
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHy)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHx)):
      case (static_cast<uint8_t> (SchemeType::Dim1_ExHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHx)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EyHz)):
      case (static_cast<uint8_t> (SchemeType::Dim1_EzHy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMx)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMy)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TEz)):
      case (static_cast<uint8_t> (SchemeType::Dim2_TMz)):
      {
        return CoordinateType::NONE;
      }
      case (static_cast<uint8_t> (SchemeType::Dim3)):
      {
        return CoordinateType::Z;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    return CoordinateType::NONE;
  }

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
  ICUDA_DEVICE ICUDA_HOST \
  IGRID<TC> * get ## x () \
  { \
    ASSERT (x); \
    return x; \
  } \
  ICUDA_DEVICE ICUDA_HOST \
  bool has ## x () \
  { \
    return x != NULLPTR; \
  }
#include "Grids.inc.h"
#undef GRID_NAME

#define CALLBACK_NAME(x) \
  ICUDA_HOST \
  SourceCallBack getCallback ## x () \
  { \
    return x; \
  } \
  ICUDA_HOST \
  void setCallback ## x (SourceCallBack new_cb) \
  { \
    ASSERT (new_cb != NULLPTR); \
    x = new_cb; \
  }
#include "Callbacks.inc.h"
#undef CALLBACK_NAME

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

  ICUDA_HOST
  FPValue getGridStep ()
  {
    return gridStep;
  }

  ICUDA_HOST
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

  ICUDA_HOST
  FPValue getSourceFrequency ()
  {
    return sourceFrequency;
  }

  ICUDA_HOST
  FPValue getSourceWaveLength ()
  {
    return sourceWaveLength;
  }

#ifdef GPU_INTERNAL_SCHEME
  ICUDA_DEVICE
  FPValue * getd_norm ()
  {
    ASSERT (d_norm != NULLPTR);
    return d_norm;
  }
#endif /* GPU_INTERNAL_SCHEME */
};

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateTFSF (TC posAbs,
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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = true;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = false;
      }
      else
      {
        UNREACHABLE;
      }

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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = true;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = false;
      }
      else
      {
        UNREACHABLE;
      }

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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = true;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = false;
      }
      else
      {
        UNREACHABLE;
      }

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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = false;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = true;
      }
      else
      {
        UNREACHABLE;
      }

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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = false;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = true;
      }
      else
      {
        UNREACHABLE;
      }

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

      if (layout_type == E_CENTERED)
      {
        isRevertVals = false;
      }
      else if (layout_type == H_CENTERED)
      {
        isRevertVals = true;
      }
      else
      {
        UNREACHABLE;
      }

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
        TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff1 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff1 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff1 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff1 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff1 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos1));
        diff1 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff1 *= FPValue (-1.0);
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
        TCFP realCoord = yeeLayout->getHyCoordFP (Hy->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getHyFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff2 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        TCFP realCoord = yeeLayout->getHzCoordFP (Hz->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getHzFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff2 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        TCFP realCoord = yeeLayout->getHxCoordFP (Hx->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getHxFromIncidentH (approximateIncidentWaveH (realCoord));

        if (layout_type == H_CENTERED)
        {
          diff2 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        TCFP realCoord = yeeLayout->getEzCoordFP (Ez->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getEzFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff2 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        TCFP realCoord = yeeLayout->getExCoordFP (Ex->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getExFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff2 *= FPValue (-1.0);
        }

        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        TCFP realCoord = yeeLayout->getEyCoordFP (Ey->getTotalPosition (auxPos2));
        diff2 = yeeLayout->getEyFromIncidentE (approximateIncidentWaveE (realCoord));

        if (layout_type == E_CENTERED)
        {
          diff2 *= FPValue (-1.0);
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

#ifndef GPU_INTERNAL_SCHEME
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepInitDiff (TCS *diff11, TCS *diff12, TCS *diff21, TCS *diff22)
{
  switch (grid_type)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      *diff11 = yeeLayout->getExCircuitElementDiff (LayoutDirection::DOWN);
      *diff12 = yeeLayout->getExCircuitElementDiff (LayoutDirection::UP);
      *diff21 = yeeLayout->getExCircuitElementDiff (LayoutDirection::BACK);
      *diff22 = yeeLayout->getExCircuitElementDiff (LayoutDirection::FRONT);
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      *diff11 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::BACK);
      *diff12 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::FRONT);
      *diff21 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::LEFT);
      *diff22 = yeeLayout->getEyCircuitElementDiff (LayoutDirection::RIGHT);
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      *diff11 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::LEFT);
      *diff12 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::RIGHT);
      *diff21 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::DOWN);
      *diff22 = yeeLayout->getEzCircuitElementDiff (LayoutDirection::UP);
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      *diff11 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::BACK);
      *diff12 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::FRONT);
      *diff21 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::DOWN);
      *diff22 = yeeLayout->getHxCircuitElementDiff (LayoutDirection::UP);
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      *diff11 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::LEFT);
      *diff12 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::RIGHT);
      *diff21 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::BACK);
      *diff22 = yeeLayout->getHyCircuitElementDiff (LayoutDirection::FRONT);
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      *diff11 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::DOWN);
      *diff12 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::UP);
      *diff21 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::LEFT);
      *diff22 = yeeLayout->getHzCircuitElementDiff (LayoutDirection::RIGHT);
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }
}
#endif

/**
 * Initialize grids used in further computations
 *
 * TODO: force inline this
 */
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePML, bool useMetamaterials>
ICUDA_HOST
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepInit (IGRID<TC> **grid, GridType *gridType, IGRID<TC> **materialGrid, GridType *materialGridType, IGRID<TC> **materialGrid1, GridType *materialGridType1,
IGRID<TC> **materialGrid2, GridType *materialGridType2, IGRID<TC> **materialGrid3, GridType *materialGridType3, IGRID<TC> **materialGrid4, GridType *materialGridType4,
IGRID<TC> **materialGrid5, GridType *materialGridType5, IGRID<TC> **oppositeGrid1, IGRID<TC> **oppositeGrid2, IGRID<TC> **gridPML1, GridType *gridPMLType1, IGRID<TC> **gridPML2, GridType *gridPMLType2,
SourceCallBack *rightSideFunc, SourceCallBack *borderFunc, SourceCallBack *exactFunc, FPValue *materialModifier,
  IGRID<TC> **Ca, IGRID<TC> **Cb, IGRID<TC> **CB0, IGRID<TC> **CB1, IGRID<TC> **CB2, IGRID<TC> **CA1, IGRID<TC> **CA2, IGRID<TC> **CaPML, IGRID<TC> **CbPML, IGRID<TC> **CcPML)
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

      *Ca = CaEx;
      *Cb = CbEx;

      *CB0 = CB0Ex;
      *CB1 = CB1Ex;
      *CB2 = CB2Ex;

      *CaPML = CaPMLEx;
      *CbPML = CbPMLEx;
      *CcPML = CcPMLEx;

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

      *Ca = CaEy;
      *Cb = CbEy;

      *CB0 = CB0Ey;
      *CB1 = CB1Ey;
      *CB2 = CB2Ey;

      *CaPML = CaPMLEy;
      *CbPML = CbPMLEy;
      *CcPML = CcPMLEy;

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

      *Ca = CaEz;
      *Cb = CbEz;

      *CB0 = CB0Ez;
      *CB1 = CB1Ez;
      *CB2 = CB2Ez;

      *CaPML = CaPMLEz;
      *CbPML = CbPMLEz;
      *CcPML = CcPMLEz;

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

      *Ca = DaHx;
      *Cb = DbHx;

      *CB0 = DB0Hx;
      *CB1 = DB1Hx;
      *CB2 = DB2Hx;

      *CaPML = DaPMLHx;
      *CbPML = DbPMLHx;
      *CcPML = DcPMLHx;

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

      *Ca = DaHy;
      *Cb = DbHy;

      *CB0 = DB0Hy;
      *CB1 = DB1Hy;
      *CB2 = DB2Hy;

      *CaPML = DaPMLHy;
      *CbPML = DbPMLHy;
      *CcPML = DcPMLHy;

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

      *Ca = DaHz;
      *Cb = DbHz;

      *CB0 = DB0Hz;
      *CB1 = DB1Hz;
      *CB2 = DB2Hz;

      *CaPML = DaPMLHz;
      *CbPML = DbPMLHz;
      *CcPML = DcPMLHz;

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
template<bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::computeCaCb (FieldValue &valCa,
                                                              FieldValue &valCb,
                                                              TC pos,
                                                              TC posAbs,
                                                              IGRID<TC> *Ca,
                                                              IGRID<TC> *Cb,
                                                              bool usePML,
                                                              GridType gridType,
                                                              IGRID<TC> *materialGrid,
                                                              GridType materialGridType,
                                                              FPValue materialModifier)
{
  if (usePrecomputedGrids)
  {
    ASSERT (Ca != NULLPTR);
    ASSERT (Cb != NULLPTR);

    valCa = *Ca->getFieldValue (pos, 0);
    valCb = *Cb->getFieldValue (pos, 0);
  }
  else
  {
    ASSERT (Ca == NULLPTR);
    ASSERT (Cb == NULLPTR);
    ASSERT (materialGrid != NULLPTR || SOLVER_SETTINGS.getDoUsePML ());

    FPValue material = materialGrid ? getMaterial (posAbs, gridType, materialGrid, materialGridType) : 0;
    FPValue ca = FPValue (0);
    FPValue cb = FPValue (0);

    FPValue k_mod = FPValue (1);

    if (usePML)
    {
      FPValue eps0 = PhysicsConst::Eps0;
      FPValue dd = (2 * eps0 * k_mod + material * gridTimeStep);
      ca = (2 * eps0 * k_mod - material * gridTimeStep) / dd;
      cb = (2 * eps0 * gridTimeStep / gridStep) / dd;
    }
    else
    {
      ca = 1.0;
      cb = gridTimeStep / (material * materialModifier * gridStep);
    }

    valCa = FIELDVALUE (ca, 0);
    valCb = FIELDVALUE (cb, 0);
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIteration (FPValue timestep,
                                                                             TC pos,
                                                                             TC posAbs,
                                                                             TCS diff11,
                                                                             TCS diff12,
                                                                             TCS diff21,
                                                                             TCS diff22,
                                                                             IGRID<TC> *grid,
                                                                             TCFP coordFP,
                                                                             IGRID<TC> *oppositeGrid1,
                                                                             IGRID<TC> *oppositeGrid2,
                                                                             SourceCallBack rightSideFunc,
                                                                             IGRID<TC> *Ca,
                                                                             IGRID<TC> *Cb,
                                                                             bool usePML,
                                                                             GridType gridType,
                                                                             IGRID<TC> *materialGrid,
                                                                             GridType materialGridType,
                                                                             FPValue materialModifier)
{
  ASSERT (grid != NULLPTR);
  grid_coord coord = grid->calculateIndexFromPosition (pos);
  FieldValue val = *grid->getFieldValue (coord, 1);

  FieldValue valCa = FIELDVALUE (0, 0);
  FieldValue valCb = FIELDVALUE (0, 0);

  computeCaCb<usePrecomputedGrids> (valCa, valCb, pos, posAbs, Ca, Cb, usePML, gridType, materialGrid, materialGridType, materialModifier);

  ASSERT (valCa != FIELDVALUE (0, 0));
  ASSERT (valCb != FIELDVALUE (0, 0));

  FieldValue prev11 = FIELDVALUE (0, 0);
  FieldValue prev12 = FIELDVALUE (0, 0);
  FieldValue prev21 = FIELDVALUE (0, 0);
  FieldValue prev22 = FIELDVALUE (0, 0);

  FieldValue prevRightSide = FIELDVALUE (0, 0);

  if (oppositeGrid1)
  {
    prev11 = *oppositeGrid1->getFieldValue (pos + diff11, 1);
    prev12 = *oppositeGrid1->getFieldValue (pos + diff12, 1);
  }

  if (oppositeGrid2)
  {
    prev21 = *oppositeGrid2->getFieldValue (pos + diff21, 1);
    prev22 = *oppositeGrid2->getFieldValue (pos + diff22, 1);
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    calculateTFSF<grid_type> (posAbs, prev11, prev12, prev21, prev22, pos + diff11, pos + diff12, pos + diff21, pos + diff22);
  }

  if (rightSideFunc != NULLPTR)
  {
    prevRightSide = rightSideFunc (expandTo3D (coordFP * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);
  }

  FieldValue valNew = calcField (val, prev12, prev11, prev22, prev21, prevRightSide, valCa, valCb, gridStep);
  grid->setFieldValue (valNew, coord, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t grid_type, bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationCurrent (FieldValue current,
                                                                             IGRID<TC> *grid,
                                                                             IGRID<TC> *Ca,
                                                                             IGRID<TC> *Cb,
                                                                             bool usePML,
                                                                             GridType gridType,
                                                                             IGRID<TC> *materialGrid,
                                                                             GridType materialGridType,
                                                                             FPValue materialModifier)
{
  TC pos = TC::initAxesCoordinate (SOLVER_SETTINGS.getCurrentSourcePositionX (),
                                   SOLVER_SETTINGS.getCurrentSourcePositionY (),
                                   SOLVER_SETTINGS.getCurrentSourcePositionZ (),
                                   ct1, ct2, ct3);
  TC posAbs = grid->getTotalPosition (pos);

  FieldValue* pointVal = grid->getFieldValueOrNullByAbsolutePos (pos, 0);

  if (pointVal)
  {
    FieldValue valCa = FIELDVALUE (0, 0);
    FieldValue valCb = FIELDVALUE (0, 0);

    computeCaCb<usePrecomputedGrids> (valCa, valCb, pos, posAbs, Ca, Cb, usePML, gridType, materialGrid, materialGridType, materialModifier);

    *pointVal -= calcCurrent (current, valCb, getGridStep ());
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPMLMetamaterials (time_step t,
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
  ASSERT (grid != NULLPTR);
  ASSERT (gridPML != NULLPTR);
  grid_coord coord = grid->calculateIndexFromPosition (pos);

  FieldValue cur = *grid->getFieldValue (coord, 0);
  FieldValue prev = *grid->getFieldValue (coord, 1);
  FieldValue prevPrev = *grid->getFieldValue (coord, 2);

  FieldValue prevPML = *gridPML->getFieldValue (coord, 1);
  FieldValue prevPrevPML = *gridPML->getFieldValue (coord, 2);

  FieldValue valb0 = FIELDVALUE (0, 0);
  FieldValue valb1 = FIELDVALUE (0, 0);
  FieldValue valb2 = FIELDVALUE (0, 0);
  FieldValue vala1 = FIELDVALUE (0, 0);
  FieldValue vala2 = FIELDVALUE (0, 0);

  if (usePrecomputedGrids)
  {
    ASSERT (CB0 != NULLPTR);
    ASSERT (CB1 != NULLPTR);
    ASSERT (CB2 != NULLPTR);
    ASSERT (CA1 != NULLPTR);
    ASSERT (CA2 != NULLPTR);

    valb0 = *CB0->getFieldValue (coord, 0);
    valb1 = *CB1->getFieldValue (coord, 0);
    valb2 = *CB2->getFieldValue (coord, 0);
    vala1 = *CA1->getFieldValue (coord, 0);
    vala2 = *CA2->getFieldValue (coord, 0);
  }
  else
  {
    ASSERT (CB0 == NULLPTR);
    ASSERT (CB1 == NULLPTR);
    ASSERT (CB2 == NULLPTR);
    ASSERT (CA1 == NULLPTR);
    ASSERT (CA2 == NULLPTR);

    ASSERT (materialGrid1 != NULLPTR);
    ASSERT (materialGrid2 != NULLPTR);
    ASSERT (materialGrid3 != NULLPTR);

    TC posAbs = grid->getTotalPosition (pos);

    FPValue material1;
    FPValue material2;

    FPValue material = getMetaMaterial (posAbs, gridType,
                                        materialGrid1, materialGridType1,
                                        materialGrid2, materialGridType2,
                                        materialGrid3, materialGridType3,
                                        material1, material2);

    FPValue A = 4*materialModifier*material + 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1);
    FPValue b0 = (4 + 2*gridTimeStep*material2) / A;
    FPValue b1 = -8 / A;
    FPValue b2 = (4 - 2*gridTimeStep*material2) / A;
    FPValue a1 = (2*materialModifier*SQR(gridTimeStep*material1) - 8*materialModifier*material) / A;
    FPValue a2 = (4*materialModifier*material - 2*gridTimeStep*materialModifier*material*material2 + materialModifier*SQR(gridTimeStep*material1)) / A;

    valb0 = FIELDVALUE (b0, 0);
    valb1 = FIELDVALUE (b1, 0);
    valb2 = FIELDVALUE (b2, 0);
    vala1 = FIELDVALUE (a1, 0);
    vala2 = FIELDVALUE (a2, 0);
  }

  ASSERT (valb0 != FIELDVALUE (0, 0));
  ASSERT (valb1 != FIELDVALUE (0, 0));
  ASSERT (valb2 != FIELDVALUE (0, 0));
  ASSERT (vala1 != FIELDVALUE (0, 0));
  ASSERT (vala2 != FIELDVALUE (0, 0));

  FieldValue valNew = calcFieldDrude (cur, prev, prevPrev, prevPML, prevPrevPML, valb0, valb1, valb2, vala1, vala2);
  gridPML->setFieldValue (valNew, coord, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <bool useMetamaterials, bool usePrecomputedGrids>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationPML (time_step t,
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
  ASSERT (gridPML2 != NULLPTR);
  grid_coord coord = gridPML2->calculateIndexFromPosition (pos);

  FieldValue prevEorH = *gridPML2->getFieldValue (coord, 1);
  FieldValue curDorB = FIELDVALUE (0, 0);
  FieldValue prevDorB = FIELDVALUE (0, 0);

  FieldValue valCa = FIELDVALUE (0, 0);
  FieldValue valCb = FIELDVALUE (0, 0);
  FieldValue valCc = FIELDVALUE (0, 0);

  if (useMetamaterials)
  {
    ASSERT (gridPML1 != NULLPTR);
    curDorB = *gridPML1->getFieldValue (coord, 0);
    prevDorB = *gridPML1->getFieldValue (coord, 1);
  }
  else
  {
    ASSERT (grid != NULLPTR);
    curDorB = *grid->getFieldValue (coord, 0);
    prevDorB = *grid->getFieldValue (coord, 1);
  }

  if (usePrecomputedGrids)
  {
    ASSERT (Ca != NULLPTR);
    ASSERT (Cb != NULLPTR);
    ASSERT (Cc != NULLPTR);

    valCa = *Ca->getFieldValue (coord, 0);
    valCb = *Cb->getFieldValue (coord, 0);
    valCc = *Cc->getFieldValue (coord, 0);
  }
  else
  {
    FPValue eps0 = PhysicsConst::Eps0;
    TC posAbs = gridPML2->getTotalPosition (pos);

    FPValue material1 = materialGrid1 ? getMaterial (posAbs, gridPMLType1, materialGrid1, materialGridType1) : 0;
    FPValue material4 = materialGrid4 ? getMaterial (posAbs, gridPMLType1, materialGrid4, materialGridType4) : 0;
    FPValue material5 = materialGrid5 ? getMaterial (posAbs, gridPMLType1, materialGrid5, materialGridType5) : 0;

    FPValue modifier = material1 * materialModifier;
    if (useMetamaterials)
    {
      modifier = 1;
    }

    FPValue k_mod1 = 1;
    FPValue k_mod2 = 1;

    FPValue dd = (2 * eps0 * k_mod2 + material5 * gridTimeStep);

    FPValue ca = (2 * eps0 * k_mod2 - material5 * gridTimeStep) / dd;
    FPValue cb = ((2 * eps0 * k_mod1 + material4 * gridTimeStep) / (modifier)) / dd;
    FPValue cc = ((2 * eps0 * k_mod1 - material4 * gridTimeStep) / (modifier)) / dd;

    valCa = FIELDVALUE (ca, 0);
    valCb = FIELDVALUE (cb, 0);
    valCc = FIELDVALUE (cc, 0);
  }

  ASSERT (valCa != FIELDVALUE (0, 0));
  ASSERT (valCb != FIELDVALUE (0, 0));
  ASSERT (valCc != FIELDVALUE (0, 0));

  FieldValue valNew = calcFieldFromDOrB (prevEorH, curDorB, prevDorB, valCa, valCb, valCc);
  gridPML2->setFieldValue (valNew, coord, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationBorder (time_step t,
                                                                      TC posAbs,
                                                                      IGRID<TC> *grid,
                                                                      SourceCallBack borderFunc)
{
  if (doSkipBorderFunc (posAbs, grid)
      || grid->getFieldValueOrNullByAbsolutePos (posAbs, 0) == NULL)
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

  TC pos = grid->getRelativePosition (posAbs);
  grid->setFieldValue (borderFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep), pos, 0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template <uint8_t grid_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::calculateFieldStepIterationExact (time_step t,
                                                                     TC posAbs,
                                                                     IGRID<TC> *grid,
                                                                     SourceCallBack exactFunc,
                                                                     FPValue &normRe,
                                                                     FPValue &normIm,
                                                                     FPValue &normMod,
                                                                     FPValue &maxRe,
                                                                     FPValue &maxIm,
                                                                     FPValue &maxMod)
{
  if (grid->getFieldValueOrNullByAbsolutePos (posAbs, 0) == NULL
      || grid->isBufferLeftPosition (posAbs)
      || grid->isBufferRightPosition (posAbs))
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

  FieldValue numerical = *grid->getFieldValueByAbsolutePos (posAbs, 0);
  FieldValue exact = exactFunc (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep * gridTimeStep);

#ifdef COMPLEX_FIELD_VALUES
  FPValue modExact = sqrt (SQR (exact.real ()) + SQR (exact.imag ()));
  FPValue modNumerical = sqrt (SQR (numerical.real ()) + SQR (numerical.imag ()));

  // printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact.real (), numerical.real ());

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

  //printf ("EXACT %u %s %.20f %.20f\n", t, grid->getName (), exact, numerical);

  FPValue exactAbs = fabs (exact);
  if (maxRe < exactAbs)
  {
    maxRe = exactAbs;
  }
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
template<uint8_t EnumVal>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPointSourceCalc (time_step t)
{
  IGRID<TC> *grid = NULLPTR;

  FPValue timestep;
  switch (EnumVal)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      ASSERT (doNeedEx);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      ASSERT (doNeedEy);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      ASSERT (doNeedEz);
      timestep = t + 0.5;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      ASSERT (doNeedHx);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      ASSERT (doNeedHy);
      timestep = t + 1.0;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      ASSERT (doNeedHz);
      timestep = t + 1.0;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (grid != NULLPTR);

  TC pos = TC::initAxesCoordinate (SOLVER_SETTINGS.getPointSourcePositionX (),
                                   SOLVER_SETTINGS.getPointSourcePositionY (),
                                   SOLVER_SETTINGS.getPointSourcePositionZ (),
                                   ct1, ct2, ct3);

  FieldValue* pointVal = grid->getFieldValueOrNullByAbsolutePos (pos, 0);

  if (pointVal)
  {
#ifdef COMPLEX_FIELD_VALUES
    *pointVal = FieldValue (sin (gridTimeStep * timestep * 2 * PhysicsConst::Pi * sourceFrequency),
                            cos (gridTimeStep * timestep * 2 * PhysicsConst::Pi * sourceFrequency));
#else /* COMPLEX_FIELD_VALUES */
    *pointVal = sin (gridTimeStep * timestep * 2 * PhysicsConst::Pi * sourceFrequency);
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
FPValue
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::getMaterial (const TC &posAbs,
                            GridType typeOfField,
                            IGRID<TC> *gridMaterial,
                            GridType typeOfMaterial)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  yeeLayout->template initMaterialCoordinates<false> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                          absPos31, absPos32, absPos41, absPos42);

  ASSERT (typeOfMaterial == GridType::EPS
          || typeOfMaterial == GridType::MU
          || typeOfMaterial == GridType::SIGMAX
          || typeOfMaterial == GridType::SIGMAY
          || typeOfMaterial == GridType::SIGMAZ);

  if (yeeLayout->getIsDoubleMaterialPrecision ())
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getRealOnlyFromFieldValue (yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos31, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos32, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos41, 0),
                                                  *gridMaterial->getFieldValueByAbsolutePos (absPos42, 0)));
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    if (layout_type == E_CENTERED)
    {
      switch (typeOfField)
      {
        case GridType::EX:
        case GridType::DX:
        case GridType::EY:
        case GridType::DY:
        case GridType::EZ:
        case GridType::DZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0)));
        }
        case GridType::HX:
        case GridType::BX:
        case GridType::HY:
        case GridType::BY:
        case GridType::HZ:
        case GridType::BZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0)));
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }
    else if (layout_type == H_CENTERED)
    {
      switch (typeOfField)
      {
        case GridType::EX:
        case GridType::DX:
        case GridType::EY:
        case GridType::DY:
        case GridType::EZ:
        case GridType::DZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0)));
        }
        case GridType::HX:
        case GridType::BX:
        case GridType::HY:
        case GridType::BY:
        case GridType::HZ:
        case GridType::BZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                    *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0)));
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }
    else
    {
      UNREACHABLE;
    }
  }

  UNREACHABLE;
  return FPValue (0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
FPValue
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::getMetaMaterial (const TC &posAbs,
                                GridType typeOfField,
                                IGRID<TC> *gridMaterial,
                                GridType typeOfMaterial,
                                IGRID<TC> *gridMaterialOmega,
                                GridType typeOfMaterialOmega,
                                IGRID<TC> *gridMaterialGamma,
                                GridType typeOfMaterialGamma,
                                FPValue &omega,
                                FPValue &gamma)
{
  TC absPos11;
  TC absPos12;
  TC absPos21;
  TC absPos22;

  TC absPos31;
  TC absPos32;
  TC absPos41;
  TC absPos42;

  yeeLayout->template initMaterialCoordinates<true> (posAbs, typeOfField, absPos11, absPos12, absPos21, absPos22,
                         absPos31, absPos32, absPos41, absPos42);

  ASSERT ((typeOfMaterialOmega == GridType::OMEGAPE && typeOfMaterialGamma == GridType::GAMMAE)
          || (typeOfMaterialOmega == GridType::OMEGAPM && typeOfMaterialGamma == GridType::GAMMAM));

  if (yeeLayout->getIsDoubleMaterialPrecision ())
  {
    switch (typeOfField)
    {
      case GridType::EX:
      case GridType::DX:
      case GridType::EY:
      case GridType::DY:
      case GridType::HX:
      case GridType::BX:
      case GridType::HY:
      case GridType::BY:
      case GridType::HZ:
      case GridType::BZ:
      case GridType::EZ:
      case GridType::DZ:
      {
        return getRealOnlyFromFieldValue (yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterial->getFieldValueByAbsolutePos (absPos42, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterialOmega->getFieldValueByAbsolutePos (absPos42, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos21, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos22, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos31, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos32, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos41, 0),
                                                      *gridMaterialGamma->getFieldValueByAbsolutePos (absPos42, 0),
                                                      omega, gamma));
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    if (layout_type == E_CENTERED)
    {
      switch (typeOfField)
      {
        case GridType::EX:
        case GridType::DX:
        case GridType::EY:
        case GridType::DY:
        case GridType::EZ:
        case GridType::DZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                        omega, gamma));
        }
        case GridType::HX:
        case GridType::BX:
        case GridType::HY:
        case GridType::BY:
        case GridType::HZ:
        case GridType::BZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos22, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos22, 0),
                                                        omega, gamma));
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }
    else if (layout_type == H_CENTERED)
    {
      switch (typeOfField)
      {
        case GridType::EX:
        case GridType::DX:
        case GridType::EY:
        case GridType::DY:
        case GridType::EZ:
        case GridType::DZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos22, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos22, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos21, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos22, 0),
                                                        omega, gamma));
        }
        case GridType::HX:
        case GridType::BX:
        case GridType::HY:
        case GridType::BY:
        case GridType::HZ:
        case GridType::BZ:
        {
          return getRealOnlyFromFieldValue (yeeLayout->getApproximateMetaMaterial (*gridMaterial->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterial->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialOmega->getFieldValueByAbsolutePos (absPos12, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos11, 0),
                                                        *gridMaterialGamma->getFieldValueByAbsolutePos (absPos12, 0),
                                                        omega, gamma));
        }
        default:
        {
          UNREACHABLE;
        }
      }
    }
    else
    {
      UNREACHABLE;
    }
  }

  UNREACHABLE;
  return FPValue (0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_HOST
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::INTERNAL_SCHEME_BASE ()
  : isInitialized (false)
  , useParallel (false)
  , yeeLayout (NULLPTR)

#define GRID_NAME(x) \
  , x (NULLPTR)
#include "Grids.inc.h"
#undef GRID_NAME

  , EInc (NULLPTR)
  , HInc (NULLPTR)
  , sourceWaveLength (0)
  , sourceWaveLengthNumerical (0)
  , sourceFrequency (0)
  , courantNum (0)
  , gridStep (0)
  , gridTimeStep (0)
#define CALLBACK_NAME(x) \
  , x (NULLPTR)
#include "Callbacks.inc.h"
#undef CALLBACK_NAME

#ifdef GPU_INTERNAL_SCHEME
  , d_norm (NULLPTR)
#endif /* GPU_INTERNAL_SCHEME */
  , doNeedEx (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedEy (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedEz (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHx (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHy (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedHz (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
              || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz) || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaX (Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaY (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHz) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMz)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
  , doNeedSigmaZ (Type == static_cast<SchemeType_t> (SchemeType::Dim1_ExHy) || Type == static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TEy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMx) || Type == static_cast<SchemeType_t> (SchemeType::Dim2_TMy)
                  || Type == static_cast<SchemeType_t> (SchemeType::Dim3))
{
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPlaneWaveESteps (time_step t, GridCoordinate1D start, GridCoordinate1D end)
{
  ASSERT (end.get1 () > start.get1 ());
  ASSERT (end.get1 () <= EInc->getSize ().get1 ());

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Eps0 * gridStep);

  grid_coord cstart = start.get1 ();
  grid_coord cend = end.get1 ();

  bool setSource = false;
  if (cstart == 0)
  {
    setSource = true;
    cstart = 1;
  }

  for (grid_coord i = cstart; i < cend; ++i)
  {
    FieldValue valE = *EInc->getFieldValue (i, 1);
    FieldValue valH1 = *HInc->getFieldValue (i - 1, 1);
    FieldValue valH2 = *HInc->getFieldValue (i, 1);

    FieldValue val = valE + (valH1 - valH2) * modifier;
    EInc->setFieldValue (val, i, 0);
  }

  if (setSource)
  {
    FPValue arg = gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency;

#ifdef COMPLEX_FIELD_VALUES
    EInc->setFieldValue (FieldValue (sin (arg), cos (arg)), 0, 0);
#else /* COMPLEX_FIELD_VALUES */
    EInc->setFieldValue (sin (arg), 0, 0);
#endif /* !COMPLEX_FIELD_VALUES */

    //printf ("EInc[0] %f \n", valE->getCurValue ());
  }

#ifdef ENABLE_ASSERTS
  ALWAYS_ASSERT (*EInc->getFieldValue (EInc->getSize ().get1 () - 1, 0) == getFieldValueRealOnly (0.0));
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
ICUDA_DEVICE
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::performPlaneWaveHSteps (time_step t, GridCoordinate1D start, GridCoordinate1D end)
{
  ASSERT (end.get1 () > start.get1 ());
  ASSERT (end.get1 () <= HInc->getSize ().get1 ());

  FPValue modifier = gridTimeStep / (relPhaseVelocity * PhysicsConst::Mu0 * gridStep);

  grid_coord cstart = start.get1 ();
  grid_coord cend = end.get1 ();

  if (cend == HInc->getSize ().get1 ())
  {
    cend--;
  }

  for (grid_coord i = cstart; i < cend; ++i)
  {
    FieldValue valH = *HInc->getFieldValue (i, 1);
    FieldValue valE1 = *EInc->getFieldValue (i, 1);
    FieldValue valE2 = *EInc->getFieldValue (i + 1, 1);

    FieldValue val = valH + (valE1 - valE2) * modifier;
    HInc->setFieldValue (val, i, 0);
  }

#ifdef ENABLE_ASSERTS
  ALWAYS_ASSERT (*HInc->getFieldValue (HInc->getSize ().get1 () - 2, 0) == getFieldValueRealOnly (0.0));
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
INTERNAL_SCHEME_BASE<Type, TCoord, layout_type>::initScheme (FPValue dx, FPValue sourceWaveLen)
{
  sourceWaveLength = sourceWaveLen;
  sourceFrequency = PhysicsConst::SpeedOfLight / sourceWaveLength;

  gridStep = dx;
  courantNum = SOLVER_SETTINGS.getCourantNum ();
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  //ALWAYS_ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

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
                                      "\n\tfrequency -> %.20f"
                                      "\n",
           relPhaseVelocity, phaseVelocity0, phaseVelocity, 2*PhysicsConst::Pi/sourceWaveLength, k,
           sourceWaveLength, sourceWaveLengthNumerical, gridStep, gridTimeStep, sourceFrequency);
}

#undef INTERNAL_SCHEME_BASE
#undef INTERNAL_SCHEME_HELPER
#undef IGRID
#undef ICUDA_HOST
#undef ICUDA_DEVICE

#undef SETUP_BLOCKS_AND_THREADS
