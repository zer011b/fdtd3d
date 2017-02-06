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

#ifdef CXX11_ENABLED
#else
#include "cstdlib"
#endif

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "SchemeTMz.h"
#include "SchemeTEz.h"
#include "Scheme3D.h"

#include "PhysicsConst.h"

int cudaThreadsX = 8;
int cudaThreadsY = 8;
int cudaThreadsZ = 8;

int main (int argc, char** argv)
{
  int totalTimeSteps = 100;

#ifdef GRID_2D
  int gridSizeX = 100;
  int gridSizeY = 100;
#endif
#ifdef GRID_3D
  int gridSizeX = 100;
  int gridSizeY = 100;
  int gridSizeZ = 100;
#endif

  int bufSize = 10;
  int dumpRes = 0;

  int numCudaGPUs = 1;

  int dimension;
  bool is_parallel_grid;

#ifdef GRID_2D
#ifdef CUDA_ENABLED
  if (argc != 9)
#else
  if (argc != 6)
#endif
#endif
#ifdef GRID_3D
#ifdef CUDA_ENABLED
  if (argc != 11)
#else
  if (argc != 7)
#endif
#endif
  {
    return 1;
  }
  else
  {
#ifdef CXX11_ENABLED
#ifdef GRID_2D
    totalTimeSteps = std::stoi (argv[1]);
    gridSizeX = std::stoi (argv[2]);
    gridSizeY = std::stoi (argv[3]);
    bufSize = std::stoi (argv[4]);
    dumpRes = std::stoi (argv[5]);
#ifdef CUDA_ENABLED
    numCudaGPUs = std::stoi (argv[6]);
    cudaThreadsX = std::stoi (argv[7]);
    cudaThreadsY = std::stoi (argv[8]);
#endif
#endif
#ifdef GRID_3D
    totalTimeSteps = std::stoi (argv[1]);
    gridSizeX = std::stoi (argv[2]);
    gridSizeY = std::stoi (argv[3]);
    gridSizeZ = std::stoi (argv[4]);
    bufSize = std::stoi (argv[5]);
    dumpRes = std::stoi (argv[6]);
#ifdef CUDA_ENABLED
    numCudaGPUs = std::stoi (argv[7]);
    cudaThreadsX = std::stoi (argv[8]);
    cudaThreadsY = std::stoi (argv[9]);
    cudaThreadsZ = std::stoi (argv[10]);
#endif
#endif
#else
#ifdef GRID_2D
    totalTimeSteps = atoi (argv[1]);
    gridSizeX = atoi (argv[2]);
    gridSizeY = atoi (argv[3]);
    bufSize = atoi (argv[4]);
    dumpRes = atoi (argv[5]);
#ifdef CUDA_ENABLED
    numCudaGPUs = atoi (argv[6]);
    cudaThreadsX = atoi (argv[7]);
    cudaThreadsY = atoi (argv[8]);
#endif
#endif
#ifdef GRID_3D
    totalTimeSteps = atoi (argv[1]);
    gridSizeX = atoi (argv[2]);
    gridSizeY = atoi (argv[3]);
    gridSizeZ = atoi (argv[4]);
    bufSize = atoi (argv[5]);
    dumpRes = atoi (argv[6]);
#ifdef CUDA_ENABLED
    numCudaGPUs = atoi (argv[7]);
    cudaThreadsX = atoi (argv[8]);
    cudaThreadsY = atoi (argv[9]);
    cudaThreadsZ = atoi (argv[10]);
#endif
#endif
#endif
  }

  const clock_t begin_time = clock();
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);

#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#if PRINT_MESSAGE
  printf ("Start process %d of %d\n", rank, numProcs);
#endif /* PRINT_MESSAGE */

  ParallelGridCore parallelGridCore (rank, numProcs);
  ParallelGrid::initializeParallelCore (&parallelGridCore);

  is_parallel_grid = true;
#else /* PARALLEL_GRID */
  is_parallel_grid = false;
#endif /* !PARALLEL_GRID */

#ifdef CUDA_ENABLED
  cudaInfo ();

#if defined (PARALLEL_GRID)
  cudaInit (rank % numCudaGPUs);
#else
  cudaInit (numCudaGPUs);
#endif
#endif

#ifdef GRID_2D
  dimension = 2;
#endif
#ifdef GRID_3D
  dimension = 3;
#endif

#if defined (PARALLEL_GRID)
#ifdef GRID_2D
  ParallelGridCoordinate overallSize (gridSizeX, gridSizeY);
  ParallelGridCoordinate bufferSize (bufSize);

  SchemeTMz scheme (overallSize, bufferSize, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate2D (20, 20), false, GridCoordinate2D (30, 30), 0, false);
  //SchemeTEz scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate2D (20, 20), true, GridCoordinate2D (30, 30), 0);
#endif
#ifdef GRID_3D
  ParallelGridCoordinate overallSize (gridSizeX, gridSizeY, gridSizeZ);
  ParallelGridCoordinate bufferSize (bufSize);

  Scheme3D scheme (overallSize, bufferSize, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate3D (10, 10, 10), false, GridCoordinate3D (13, 13, 13), PhysicsConst::Pi / 2, 0, 0, true);
#endif
#else
#ifdef GRID_2D
  GridCoordinate2D overallSize (gridSizeX, gridSizeY);

  //SchemeTMz scheme (overallSize, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate2D (20, 20), false, GridCoordinate2D (30, 30), /*PhysicsConst::Pi / 4*/0, true);
  SchemeTEz scheme (overallSize, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate2D (20, 20), true, GridCoordinate2D (30, 30), /*PhysicsConst::Pi / 4*/0);
#endif
#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSizeX, gridSizeY, gridSizeZ);

  Scheme3D scheme (overallSize, totalTimeSteps, false, 2 * totalTimeSteps, true, GridCoordinate3D (10, 10, 10), false, GridCoordinate3D (13, 13, 13), PhysicsConst::Pi / 2, 0, 0, true);
#endif
#endif

  scheme.initScheme (0.0005, /* dx */
                     30000000000); /* source frequency */

  scheme.initGrids ();

  scheme.performSteps (dumpRes);

  gettimeofday(&tv2, NULL);

#if defined (PARALLEL_GRID)
#if PRINT_MESSAGE
  printf ("Main process %d.\n", rank);
#endif

  MPI_Finalize();
#endif

#if defined (PARALLEL_GRID)
  if (rank == 0)
  {
#endif /* PARALLEL_GRID */

    const clock_t end_time = clock();
    printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));

    printf ("Dimension: %d\n", dimension);
#ifdef GRID_2D
    printf ("Grid size: %dx%d\n", gridSizeX, gridSizeY);
#endif
#ifdef GRID_3D
    printf ("Grid size: %dx%dx%d\n", gridSizeX, gridSizeY, gridSizeZ);
#endif
    printf ("Number of time steps: %d\n", totalTimeSteps);

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
    printf ("Parallel grid: %d\n", is_parallel_grid);

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

    printf ("Buffer size: %d\n", bufSize);
#endif

    printf ("\n-------- Execution Time --------\n");

    FPValue execution_time = (FPValue) (((FPValue) (end_time - begin_time)) /  CLOCKS_PER_SEC);

    printf ("Execution time (by clock()): %f seconds.\n", execution_time);

#if defined (PARALLEL_GRID)
  }
#endif /* PARALLEL_GRID */

  return 0;
}
