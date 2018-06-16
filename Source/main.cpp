#include <iostream>

#include <ctime>
#include <sys/time.h>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#include <mpi.h>
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#ifdef CUDA_ENABLED
#include "CudaInterface.h"
#endif

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

#ifdef PARALLEL_GRID
template <SchemeType_t Type, LayoutType layout_type>
void initParallel (ParallelYeeGridLayout<Type, layout_type> **yeeLayout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   ParallelGridCoordinate overallSize,
                   ParallelGridCoordinate pmlSize,
                   ParallelGridCoordinate tfsfSize,
                   int argc, char** argv)
{
  ParallelGridCoordinate topology (solverSettings.getTopologySizeX (),
                                   solverSettings.getTopologySizeY (),
                                   solverSettings.getTopologySizeZ ()
#ifdef DEBUG_INFO
                                   , CoordinateType::X, CoordinateType::Y, CoordinateType::Z
#endif /* DEBUG_INFO */
                                   );

  MPI_Init(&argc, &argv);

#ifdef MPI_CLOCK
  DPRINTF (LOG_LEVEL_1, "MPI_Wtime resolution %.10f (seconds)\n", MPI_Wtick ());
#endif /* MPI_CLOCK */

  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, numProcs);

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
                 tfsfSize,
                 solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getIncidentWaveAngle3 () * PhysicsConst::Pi / 180.0,
                 solverSettings.getDoUseDoubleMaterialPrecision ());
    (*(ParallelYeeGridLayout<Type, layout_type> **) yeeLayout)->Initialize (*parallelGridCore);
  }
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate1D overallSize,
                   GridCoordinate1D pmlSize,
                   GridCoordinate1D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_1D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate2D overallSize,
                   GridCoordinate2D pmlSize,
                   GridCoordinate2D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_2D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}

void initParallel (YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED> **layout,
                   ParallelGridCore **parallelGridCore,
                   int *rank,
                   int *numProcs,
                   bool *skipProcess,
                   GridCoordinate3D overallSize,
                   GridCoordinate3D pmlSize,
                   GridCoordinate3D tfsfSize,
                   int argc, char** argv)
{
#ifdef GRID_3D
  ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> **pLayout =
    (ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> **) layout;
  initParallel<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> (pLayout, parallelGridCore, rank, numProcs, skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
#endif
}
#endif

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void initSettings (TCoord<grid_coord, true> &, TCoord<grid_coord, true> &, TCoord<grid_coord, true> &);

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeY ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Y
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeY ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Y
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeY ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Y
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeX ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeX ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeY ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Y
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeY ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Y
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeY ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Y
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>
  (GridCoordinate1DTemplate<grid_coord, true> &overallSize,
   GridCoordinate1DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate1DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getSizeX ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
#endif
                                                            );
  pmlSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getPMLSizeX ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        );
  tfsfSize = GridCoordinate1DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeY (),
                                                            solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Y
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeY (),
                                                        solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Y
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeY (),
                                                         solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Y
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeX (),
                                                            solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeX (),
                                                        solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX (),
                                                         solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeX (),
                                                            solverSettings.getSizeY ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
                                                            , CoordinateType::Y
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeX (),
                                                        solverSettings.getPMLSizeY ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
                                                        , CoordinateType::Y
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX (),
                                                         solverSettings.getTFSFSizeY ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
                                                         , CoordinateType::Y
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeY (),
                                                            solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::Y
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeY (),
                                                        solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::Y
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeY (),
                                                         solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::Y
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeX (),
                                                            solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeX (),
                                                        solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX (),
                                                         solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
                                                         , CoordinateType::Z
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>
  (GridCoordinate2DTemplate<grid_coord, true> &overallSize,
   GridCoordinate2DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate2DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getSizeX (),
                                                            solverSettings.getSizeY ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
                                                            , CoordinateType::Y
#endif
                                                            );
  pmlSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getPMLSizeX (),
                                                        solverSettings.getPMLSizeY ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
                                                        , CoordinateType::Y
#endif
                                                        );
  tfsfSize = GridCoordinate2DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX (),
                                                         solverSettings.getTFSFSizeY ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
                                                         , CoordinateType::Y
#endif
                                                         );
}

template <>
void
initSettings<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>
  (GridCoordinate3DTemplate<grid_coord, true> &overallSize,
   GridCoordinate3DTemplate<grid_coord, true> &pmlSize,
   GridCoordinate3DTemplate<grid_coord, true> &tfsfSize)
{
  overallSize = GridCoordinate3DTemplate<grid_coord, true> (solverSettings.getSizeX (),
                                                            solverSettings.getSizeY (),
                                                            solverSettings.getSizeZ ()
#ifdef DEBUG_INFO
                                                            , CoordinateType::X
                                                            , CoordinateType::Y
                                                            , CoordinateType::Z
#endif
                                                            );
  pmlSize = GridCoordinate3DTemplate<grid_coord, true> (solverSettings.getPMLSizeX (),
                                                        solverSettings.getPMLSizeY (),
                                                        solverSettings.getPMLSizeZ ()
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
                                                        , CoordinateType::Y
                                                        , CoordinateType::Z
#endif
                                                        );
  tfsfSize = GridCoordinate3DTemplate<grid_coord, true> (solverSettings.getTFSFSizeX (),
                                                         solverSettings.getTFSFSizeY (),
                                                         solverSettings.getTFSFSizeZ ()
#ifdef DEBUG_INFO
                                                         , CoordinateType::X
                                                         , CoordinateType::Y
                                                         , CoordinateType::Z
#endif
                                                         );
}

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
  TCoord<grid_coord, true> tfsfSize;
  initSettings<Type, TCoord, layout_type> (overallSize, pmlSize, tfsfSize);

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout = NULLPTR;

  bool isParallel = false;

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinateTemplate<grid_coord, false>::dimension)
    {
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                      "Recompile it with -DPARALLEL_GRID=ON.");
    }
    else
    {
      isParallel = true;
      initParallel (&yeeLayout, &parallelGridCore, &rank, &numProcs, &skipProcess, overallSize, pmlSize, tfsfSize, argc, argv);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
  else
  {
    yeeLayout = new YeeGridLayout<Type, TCoord, layout_type> (
                overallSize,
                pmlSize,
                tfsfSize,
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
    cudaInfo ();

    if (solverSettings.getDoUseParallelGrid ())
    {
#if defined (PARALLEL_GRID)
      if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinateTemplate<grid_coord, false>::dimension)
      {
        UNREACHABLE;
      }
      else
      {
        cudaInit (rank % solverSettings.getNumCudaGPUs ());
      }
#else
      UNREACHABLE;
#endif
    }
    else
    {
      cudaInit (solverSettings.getNumCudaGPUs ());
    }

    cudaThreadsX = solverSettings.getNumCudaThreadsX ();
    cudaThreadsY = solverSettings.getNumCudaThreadsY ();
    cudaThreadsZ = solverSettings.getNumCudaThreadsZ ();
#endif

    Scheme<Type, TCoord, layout_type > scheme (yeeLayout,
                                               isParallel,
                                               overallSize,
                                               solverSettings.getNumTimeSteps ());
    scheme.initScheme (solverSettings.getGridStep (), /* dx */
                       solverSettings.getSourceWaveLength ()); /* source wave length */
    scheme.initCallBacks ();
    scheme.initGrids ();

    gettimeofday(&tv1, NULL);
    scheme.performSteps ();
    gettimeofday(&tv2, NULL);
  }

  delete yeeLayout;

  if (solverSettings.getDoUseParallelGrid ())
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

#ifdef TWO_TIME_STEPS
    printf ("Number of time steps: 2\n");
#endif
#ifdef ONE_TIME_STEP
    printf ("Number of time steps: 1\n");
#endif

    printf ("\n-------- Details --------\n");
    printf ("Parallel grid: %d\n", solverSettings.getDoUseParallelGrid ());

    if (solverSettings.getDoUseParallelGrid ())
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
      case SchemeType::Dim1_ExHy:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim1_ExHz:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim1_EyHx:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim1_EyHz:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim1_EzHx:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim1_EzHy:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED> (argc, argv);
        break;
      }
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
      case SchemeType::Dim2_TEx:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim2_TEy:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim2_TEz:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim2_TMx:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim2_TMy:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      case SchemeType::Dim2_TMz:
      {
        exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (argc, argv);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }
  }
  else
  {
    ASSERT (solverSettings.getDimension () == 3);

    exit_code = runMode<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (argc, argv);
  }

  return exit_code;
}
