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

template <SchemeType Type, template <typename, bool> class TCoord, uint8_t layout_type>
int runMode ()
{
  int rank = 0;
  int numProcs = 1;

  struct timeval tv1, tv2;

#ifdef PARALLEL_GRID
  ParallelGridCore *parallelGridCore = NULLPTR;
  bool skipProcess = false;
#endif

  TCoord<grid_coord, true> overallSize (solverSettings.getSizeX (), solverSettings.getSizeY (), solverSettings.getSizeZ ());
  TCoord<grid_coord, true> pmlSize (solverSettings.getPMLSizeX (), solverSettings.getPMLSizeY (), solverSettings.getPMLSizeZ ());
  TCoord<grid_coord, true> tfsfSize (solverSettings.getTFSFSizeX (), solverSettings.getTFSFSizeY (), solverSettings.getTFSFSizeZ ());

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout = NULLPTR;

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinate::dimension)
    {
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                      "Recompile it with -DPARALLEL_GRID=ON.");
    }
    else
    {
      TCoord<grid_coord, true> topology (solverSettings.getTopologySizeX (),
                                         solverSettings.getTopologySizeY (),
                                         solverSettings.getTopologySizeZ ());

      MPI_Init(&argc, &argv);

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

      parallelGridCore = new ParallelGridCore (rank, numProcs, overallSize,
                                               solverSettings.getDoUseManualVirtualTopology (),
                                               topology);
      ParallelGrid::initializeParallelCore (parallelGridCore);

      if (rank >= parallelGridCore->getTotalProcCount ())
      {
        skipProcess = true;
      }

      if (!skipProcess)
      {
        DPRINTF (LOG_LEVEL_STAGES, "Start process %d of %d (using %d)\n", rank, numProcs, parallelGridCore->getTotalProcCount ());

        yeeLayout = new ParallelYeeGridLayout<layout_type> (
                    overallSize,
                    pmlSize,
                    tfsfSize,
                    solverSettings.getIncidentWaveAngle1 () * PhysicsConst::Pi / 180.0,
                    solverSettings.getIncidentWaveAngle2 () * PhysicsConst::Pi / 180.0,
                    solverSettings.getIncidentWaveAngle3 () * PhysicsConst::Pi / 180.0,
                    solverSettings.getDoUseDoubleMaterialPrecision ());
        ((ParallelYeeGridLayout *) yeeLayout)->Initialize (parallelGridCore);
      }
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
      if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinate::dimension)
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

    if (solverSettings.getDoUseParallelGrid ())
    {
#ifdef PARALLEL_GRID
      if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinate::dimension)
      {
        UNREACHABLE
      }
      else
      {
        Scheme<Type, TCoord, ParallelYeeGridLayout<layout_type> > scheme ((ParallelYeeGridLayout *) yeeLayout,
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
#else
      UNREACHABLE;
#endif
    }
    else
    {
      Scheme<Type, TCoord, YeeGridLayout<Type, TCoord, layout_type> > scheme (yeeLayout,
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
  }

  delete yeeLayout;

  if (solverSettings.getDoUseParallelGrid ())
  {
#if defined (PARALLEL_GRID)
    if (TCoord<grid_coord, false>::dimension != ParallelGridCoordinate::dimension)
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
        exit_code = runMode<SchemeType::Dim1_ExHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim1_ExHz:
      {
        exit_code = runMode<SchemeType::Dim1_ExHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim1_EyHx:
      {
        exit_code = runMode<SchemeType::Dim1_EyHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim1_EyHz:
      {
        exit_code = runMode<SchemeType::Dim1_EyHz, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim1_EzHx:
      {
        exit_code = runMode<SchemeType::Dim1_EzHx, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim1_EzHy:
      {
        exit_code = runMode<SchemeType::Dim1_EzHy, GridCoordinate1DTemplate, LayoutType::E_CENTERED> ();
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
        exit_code = runMode<SchemeType::Dim2_TEx, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim2_TEy:
      {
        exit_code = runMode<SchemeType::Dim2_TEy, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim2_TEz:
      {
        exit_code = runMode<SchemeType::Dim2_TEz, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim2_TMx:
      {
        exit_code = runMode<SchemeType::Dim2_TMx, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim2_TMy:
      {
        exit_code = runMode<SchemeType::Dim2_TMy, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
        break;
      }
      case SchemeType::Dim2_TMz:
      {
        exit_code = runMode<SchemeType::Dim2_TMz, GridCoordinate2DTemplate, LayoutType::E_CENTERED> ();
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

    exit_code = runMode<SchemeType::Dim3, GridCoordinate3DTemplate, LayoutType::E_CENTERED> ();
  }

  return exit_code;
}
