#include <iostream>

#include <ctime>
#include <sys/time.h>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#include <mpi.h>
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "Settings.h"
#include "Scheme.h"

#include "PhysicsConst.h"

int cudaThreadsX = 8;
int cudaThreadsY = 8;
int cudaThreadsZ = 8;

#ifdef CUDA_ENABLED
static void cudaInfo ()
{
  int cudaDevicesCount;

  cudaCheckErrorCmd (cudaGetDeviceCount (&cudaDevicesCount));

  for (int i = 0; i < cudaDevicesCount; i++)
  {
    cudaDeviceProp prop;
    cudaCheckErrorCmd (cudaGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Concurrent kernels number: %d\n", prop.concurrentKernels);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total global mem: %llu\n", (long long unsigned) prop.totalGlobalMem);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
#endif /* CUDA_ENABLED */

#ifdef PARALLEL_GRID
template <SchemeType_t Type, LayoutType layout_type>
void initParallel (ParallelYeeGridLayout<Type, layout_type> **yeeLayout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   ParallelGridCoordinate overallSize,
                   ParallelGridCoordinate pmlSize,
                   ParallelGridCoordinate tfsfSizeLeft,
                   ParallelGridCoordinate tfsfSizeRight,
                   int argc, char** argv)
{
  ParallelGridCoordinate topology (solverSettings.getTopologySizeX (),
                                   solverSettings.getTopologySizeY (),
                                   solverSettings.getTopologySizeZ ()
#ifdef DEBUG_INFO
                                   , CoordinateType::X, CoordinateType::Y, CoordinateType::Z
#endif /* DEBUG_INFO */
                                   );

  *parallelGridCore = new ParallelGridCore (*rank, *numProcs, overallSize,
                                            solverSettings.getDoUseManualVirtualTopology (),
                                            topology);
  ParallelGrid::initializeParallelCore (*parallelGridCore);

  if (*rank >= (*parallelGridCore)->getTotalProcCount ())
  {
    *skipProcess = true;
  }

  if (!*skipProcess)
  {
    DPRINTF (LOG_LEVEL_STAGES, "Start process %d of %d (using %d)\n", *rank, *numProcs, (*parallelGridCore)->getTotalProcCount ());

    *yeeLayout = new ParallelYeeGridLayout<Type, layout_type> (
                 overallSize,
                 pmlSize,
                 tfsfSizeLeft,
                 tfsfSizeRight,
                 solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getIncidentWaveAngle3 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getDoUseDoubleMaterialPrecision ());
    (*(ParallelYeeGridLayout<Type, layout_type> **) yeeLayout)->Initialize (*parallelGridCore);
  }
}

#ifdef MODE_EX_HY
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSizeLeft,
                   GridCoordinate1D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                 skipProcess, overallSize, pmlSize,
                                                                                 tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSizeLeft,
                   GridCoordinate2D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                                skipProcess, overallSize, pmlSize,
                                                                                tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate3D overallSize,
                   GridCoordinate3D pmlSize,
                   GridCoordinate3D tfsfSizeLeft,
                   GridCoordinate3D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_3D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                            skipProcess, overallSize, pmlSize,
                                                                            tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, H_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate3D overallSize,
                   GridCoordinate3D pmlSize,
                   GridCoordinate3D tfsfSizeLeft,
                   GridCoordinate3D tfsfSizeRight,
                   int argc, char** argv)
{
#ifdef GRID_3D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), H_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), H_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim3)), H_CENTERED> (pLayout, parallelGridCore, rank, numProcs,
                                                                            skipProcess, overallSize, pmlSize,
                                                                            tfsfSizeLeft, tfsfSizeRight, argc, argv);
#endif
}
#endif /* MODE_DIM3 */

#endif

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void initSettings (TCoord<grid_coord, true> &, TCoord<grid_coord, true> &, TCoord<grid_coord, true> &, TCoord<grid_coord, true> &);

#ifdef MODE_EX_HY
static
void
initSettings_EX_HY (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeZ (), CoordinateType::Z);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeZ (), CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeZLeft (), CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeZRight (), CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EX_HY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EX_HY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EX_HY */

#ifdef MODE_EX_HZ
static
void
initSettings_EX_HZ (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeY (), CoordinateType::Y);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeY (), CoordinateType::Y);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeYLeft (), CoordinateType::Y);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeYRight (), CoordinateType::Y);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EX_HZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EX_HZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EX_HZ */

#ifdef MODE_EY_HX
static
void
initSettings_EY_HX (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeZ (), CoordinateType::Z);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeZ (), CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeZLeft (), CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeZRight (), CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EY_HX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EY_HX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EY_HX */

#ifdef MODE_EY_HZ
static
void
initSettings_EY_HZ (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeX (), CoordinateType::X);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeX (), CoordinateType::X);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeXLeft (), CoordinateType::X);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeXRight (), CoordinateType::X);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EY_HZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EY_HZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EY_HZ */

#ifdef MODE_EZ_HX
static
void
initSettings_EZ_HX (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeY (), CoordinateType::Y);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeY (), CoordinateType::Y);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeYLeft (), CoordinateType::Y);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeYRight (), CoordinateType::Y);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EZ_HX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EZ_HX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EZ_HX */

#ifdef MODE_EZ_HY
static
void
initSettings_EZ_HY (GridCoordinate1D &overallSize, GridCoordinate1D &pmlSize, GridCoordinate1D &tfsfSizeLeft, GridCoordinate1D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_1D (solverSettings.getSizeX (), CoordinateType::X);
  pmlSize = GRID_COORDINATE_1D (solverSettings.getPMLSizeX (), CoordinateType::X);
  tfsfSizeLeft = GRID_COORDINATE_1D (solverSettings.getTFSFSizeXLeft (), CoordinateType::X);
  tfsfSizeRight = GRID_COORDINATE_1D (solverSettings.getTFSFSizeXRight (), CoordinateType::X);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EZ_HY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, H_CENTERED>
  (GridCoordinate1D &overallSize,
   GridCoordinate1D &pmlSize,
   GridCoordinate1D &tfsfSizeLeft,
   GridCoordinate1D &tfsfSizeRight)
{
  initSettings_EZ_HY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_EZ_HY */

#ifdef MODE_TEX
static
void
initSettings_TEX (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeY (), solverSettings.getSizeZ (), CoordinateType::Y, CoordinateType::Z);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeY (), solverSettings.getPMLSizeZ (), CoordinateType::Y, CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeYLeft (), solverSettings.getTFSFSizeZLeft (), CoordinateType::Y, CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeYRight (), solverSettings.getTFSFSizeZRight (), CoordinateType::Y, CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TEX */

#ifdef MODE_TEY
static
void
initSettings_TEY (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeX (), solverSettings.getSizeZ (), CoordinateType::X, CoordinateType::Z);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeZ (), CoordinateType::X, CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXLeft (), solverSettings.getTFSFSizeZLeft (), CoordinateType::X, CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXRight (), solverSettings.getTFSFSizeZRight (), CoordinateType::X, CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TEY */

#ifdef MODE_TEZ
static
void
initSettings_TEZ (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeX (), solverSettings.getSizeY (), CoordinateType::X, CoordinateType::Y);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeY (), CoordinateType::X, CoordinateType::Y);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXLeft (), solverSettings.getTFSFSizeYLeft (), CoordinateType::X, CoordinateType::Y);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXRight (), solverSettings.getTFSFSizeYRight (), CoordinateType::X, CoordinateType::Y);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TEZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TEZ */

#ifdef MODE_TMX
static
void
initSettings_TMX (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeY (), solverSettings.getSizeZ (), CoordinateType::Y, CoordinateType::Z);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeY (), solverSettings.getPMLSizeZ (), CoordinateType::Y, CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeYLeft (), solverSettings.getTFSFSizeZLeft (), CoordinateType::Y, CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeYRight (), solverSettings.getTFSFSizeZRight (), CoordinateType::Y, CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMX (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TMX */

#ifdef MODE_TMY
static
void
initSettings_TMY (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeX (), solverSettings.getSizeZ (), CoordinateType::X, CoordinateType::Z);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeZ (), CoordinateType::X, CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXLeft (), solverSettings.getTFSFSizeZLeft (), CoordinateType::X, CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXRight (), solverSettings.getTFSFSizeZRight (), CoordinateType::X, CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMY (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TMY */

#ifdef MODE_TMZ
static
void
initSettings_TMZ (GridCoordinate2D &overallSize, GridCoordinate2D &pmlSize, GridCoordinate2D &tfsfSizeLeft, GridCoordinate2D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_2D (solverSettings.getSizeX (), solverSettings.getSizeY (), CoordinateType::X, CoordinateType::Y);
  pmlSize = GRID_COORDINATE_2D (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeY (), CoordinateType::X, CoordinateType::Y);
  tfsfSizeLeft = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXLeft (), solverSettings.getTFSFSizeYLeft (), CoordinateType::X, CoordinateType::Y);
  tfsfSizeRight = GRID_COORDINATE_2D (solverSettings.getTFSFSizeXRight (), solverSettings.getTFSFSizeYRight (), CoordinateType::X, CoordinateType::Y);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, H_CENTERED>
  (GridCoordinate2D &overallSize,
   GridCoordinate2D &pmlSize,
   GridCoordinate2D &tfsfSizeLeft,
   GridCoordinate2D &tfsfSizeRight)
{
  initSettings_TMZ (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_TMZ */

#ifdef MODE_DIM3
static
void
initSettings_DIM3 (GridCoordinate3D &overallSize, GridCoordinate3D &pmlSize, GridCoordinate3D &tfsfSizeLeft, GridCoordinate3D &tfsfSizeRight)
{
  overallSize = GRID_COORDINATE_3D (solverSettings.getSizeX (), solverSettings.getSizeY (), solverSettings.getSizeZ (),
                                    CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  pmlSize = GRID_COORDINATE_3D (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeY (), solverSettings.getPMLSizeZ (),
                                CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  tfsfSizeLeft = GRID_COORDINATE_3D (solverSettings.getTFSFSizeXLeft (), solverSettings.getTFSFSizeYLeft (), solverSettings.getTFSFSizeZLeft (),
                                     CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
  tfsfSizeRight = GRID_COORDINATE_3D (solverSettings.getTFSFSizeXRight (), solverSettings.getTFSFSizeYRight (), solverSettings.getTFSFSizeZRight (),
                                      CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>
  (GridCoordinate3D &overallSize,
   GridCoordinate3D &pmlSize,
   GridCoordinate3D &tfsfSizeLeft,
   GridCoordinate3D &tfsfSizeRight)
{
  initSettings_DIM3 (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, H_CENTERED>
  (GridCoordinate3D &overallSize,
   GridCoordinate3D &pmlSize,
   GridCoordinate3D &tfsfSizeLeft,
   GridCoordinate3D &tfsfSizeRight)
{
  initSettings_DIM3 (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);
}
#endif /* MODE_DIM3 */

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
int runMode (int argc, char** argv)
{
  int rank = 0;
  int numProcs = 1;

  struct timeval tv1, tv2;

#ifdef PARALLEL_GRID
  ParallelGridCore *parallelGridCore = NULLPTR;
  bool skipProcess = false;
#endif

  TCoord<grid_coord, true> overallSize;
  TCoord<grid_coord, true> pmlSize;
  TCoord<grid_coord, true> tfsfSizeLeft;
  TCoord<grid_coord, true> tfsfSizeRight;
  initSettings<Type, TCoord, layout_type> (overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight);

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout = NULLPTR;

  bool isParallel = false;

#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

#ifdef MPI_CLOCK
  DPRINTF (LOG_LEVEL_1, "MPI_Wtime resolution %.10f (seconds)\n", MPI_Wtick ());
#endif /* MPI_CLOCK */

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  isParallel = numProcs > 1;

  if (isParallel)
  {
    ALWAYS_ASSERT ((TCoord<grid_coord, false>::dimension == ParallelGridCoordinateTemplate<grid_coord, false>::dimension));

    initParallel (&yeeLayout, &parallelGridCore, &rank, &numProcs, &skipProcess, overallSize, pmlSize, tfsfSizeLeft, tfsfSizeRight, argc, argv);
  }
  else
#endif
  {
    yeeLayout = new YeeGridLayout<Type, TCoord, layout_type> (
                overallSize,
                pmlSize,
                tfsfSizeLeft,
                tfsfSizeRight,
                solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0,
                solverSettings.getIncidentWaveAngle3 () * PhysicsConst::Pi / 180.0,
                solverSettings.getDoUseDoubleMaterialPrecision ());
  }

#if defined (PARALLEL_GRID)
  if (!skipProcess)
#endif
  {
#ifdef CUDA_ENABLED
    if (SOLVER_SETTINGS.getDoUseCuda ())
    {
      cudaInfo ();

      // Parse GPU indexes
      std::vector<int> idxGPU;
      std::string idxGPUStr = SOLVER_SETTINGS.getCudaGPUs ();
      std::string delimiter = ",";

      size_t pos = 0;
      size_t pos_old = 0;
      std::string token;
      do
      {
        pos = idxGPUStr.find(delimiter, pos_old);
        token = idxGPUStr.substr(pos_old, pos);
        idxGPU.push_back (STOI (token.c_str ()));
        pos_old = pos + delimiter.length ();
      }
      while (pos != std::string::npos);

      ALWAYS_ASSERT (idxGPU.size () == numProcs);
      ALWAYS_ASSERT (rank < idxGPU.size ());
      solverSettings.setIndexOfGPUForCurrentNode (idxGPU[rank]);

      if (idxGPU[rank] != NO_GPU)
      {
        cudaCheckErrorCmd (cudaSetDevice(idxGPU[rank]));
      }

      if (isParallel)
      {
#if defined (PARALLEL_GRID)
        if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinateTemplate<grid_coord, false>::dimension)
        {
          UNREACHABLE;
        }
#else
        UNREACHABLE;
#endif
      }

      cudaThreadsX = solverSettings.getNumCudaThreadsX ();
      cudaThreadsY = solverSettings.getNumCudaThreadsY ();
      cudaThreadsZ = solverSettings.getNumCudaThreadsZ ();
    }
#endif

    solverSettings.Initialize ();

    Scheme<Type, TCoord, layout_type > scheme (yeeLayout,
                                               isParallel,
                                               overallSize,
                                               solverSettings.getNumTimeSteps ());
    scheme.initScheme (solverSettings.getGridStep (), /* dx */
                       solverSettings.getSourceWaveLength (), /* source wave length */
                       solverSettings.getNumTimeSteps ());

    gettimeofday(&tv1, NULL);
    scheme.performSteps ();
    gettimeofday(&tv2, NULL);
  }

  delete yeeLayout;

  if (isParallel)
  {
#if defined (PARALLEL_GRID)
    if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinateTemplate<grid_coord, false>::dimension)
    {
      UNREACHABLE
    }
    else
    {
      delete parallelGridCore;

      MPI_Barrier (MPI_COMM_WORLD);
      MPI_Finalize ();
    }
#else
    UNREACHABLE;
#endif
  }

  if (rank == 0)
  {
    printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));

    printf ("Dimension: %dD\n", solverSettings.getDimension ());
    printf ("Grid size: ");
    overallSize.print ();
    printf ("Number of time steps: %d\n", solverSettings.getNumTimeSteps ());

    printf ("\n");

#ifdef FLOAT_VALUES
    printf ("Value type: float\n");
#endif
#ifdef DOUBLE_VALUES
    printf ("Value type: double\n");
#endif
#ifdef LONG_DOUBLE_VALUES
    printf ("Value type: long double\n");
#endif

    printf ("\n-------- Details --------\n");
    printf ("Parallel grid: %d\n", isParallel);

    if (isParallel)
    {
#if defined (PARALLEL_GRID)
      printf ("Number of processes: %d\n", numProcs);

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
      printf ("Parallel grid scheme: X\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
      printf ("Parallel grid scheme: Y\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
      printf ("Parallel grid scheme: Z\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
      printf ("Parallel grid scheme: XY\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
      printf ("Parallel grid scheme: YZ\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
      printf ("Parallel grid scheme: XZ\n");
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
      printf ("Parallel grid scheme: XYZ\n");
#endif

      printf ("Buffer size: %d\n", solverSettings.getBufferSize ());
#else
      UNREACHABLE;
#endif
    }
  }

  return EXIT_OK;
}

int main (int argc, char** argv)
{
  solverSettings.SetupFromCmd (argc, argv);

  int exit_code = EXIT_OK;

  if (solverSettings.getDimension () == 1)
  {
    switch (solverSettings.getSchemeType ())
    {
#ifdef MODE_EX_HY
      case SchemeType::Dim1_ExHy:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EX_HY */
#ifdef MODE_EX_HZ
      case SchemeType::Dim1_ExHz:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EX_HZ */
#ifdef MODE_EY_HX
      case SchemeType::Dim1_EyHx:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EY_HX */
#ifdef MODE_EY_HZ
      case SchemeType::Dim1_EyHz:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EY_HZ */
#ifdef MODE_EZ_HX
      case SchemeType::Dim1_EzHx:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EZ_HX */
#ifdef MODE_EZ_HY
      case SchemeType::Dim1_EzHy:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_EZ_HY */
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else if (solverSettings.getDimension () == 2)
  {
    switch (solverSettings.getSchemeType ())
    {
#ifdef MODE_TEX
      case SchemeType::Dim2_TEx:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TEX */
#ifdef MODE_TEY
      case SchemeType::Dim2_TEy:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TEY */
#ifdef MODE_TEZ
      case SchemeType::Dim2_TEz:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TEZ */
#ifdef MODE_TMX
      case SchemeType::Dim2_TMx:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TMX */
#ifdef MODE_TMY
      case SchemeType::Dim2_TMy:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TMY */
#ifdef MODE_TMZ
      case SchemeType::Dim2_TMz:
      {
        if (solverSettings.getLayoutType () == E_CENTERED)
        {
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        }
        else
        {
          ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
          exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, H_CENTERED> (argc, argv);
        }
        break;
      }
#endif /* MODE_TMZ */
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    ASSERT (solverSettings.getDimension () == 3);

#ifdef MODE_DIM3
    if (solverSettings.getSchemeType () == SchemeType::Dim3)
    {
      if (solverSettings.getLayoutType () == E_CENTERED)
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (argc, argv);
      }
      else
      {
        ALWAYS_ASSERT (solverSettings.getLayoutType () == H_CENTERED);
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, H_CENTERED> (argc, argv);
      }
    }
    else
#endif /* MODE_DIM3 */
    {
      UNREACHABLE;
    }
  }

  solverSettings.Uninitialize ();

  return exit_code;
}
